# goal/agentic-speed-and-hardware — work charter

Branch: `goal/agentic-speed-and-hardware`, based on `main` @ `9d44a31`
(2026-09-16). Worktree: `/home/mik/ggrun-project/ggrun-perf`.
Created 2026-09-17. **Nothing is committed on this branch yet.**

This file states what the branch is for, how "done" is judged, and where the
evidence lives — so a different model can review the work without re-deriving
the context.

## The goal (from the standing `/goal` prompt)

> ggrun's job is to **CHOOSE THE BEST CONFIGURATION** for the selected model on
> the given machine; the case that matters most is the **big MoE that needs CPU,
> RAM and every GPU at once**, which is ggrun's core feature. Judge "best" by
> **measured agent-work speed**, never by `fraction_of_vram` or SM occupancy,
> both of which are measurably anti-correlated with speed here.

The user restated it on 2026-09-17 as two coupled aims:

1. **Serve the user-selected model as fast as possible for agentic work** on the
   given hardware.
2. **Use the given hardware to the fullest** — use all available compute.

**These two conflict on this rig, and the conflict is measured, not theoretical.**
Note the earlier `/goal` phrasing ("reach maximum hardware usage ... which should
result in fastest possible serving") was **retired as false on this rig**: six
matched arms showed raising VRAM spend is same-or-slower. What survives is that
aim 2 is a real user requirement, while aim 1 is the tie-breaker — so a change
must not buy speed by leaving hardware idle *without saying so*.

## What "best" is judged by (invariants, from the change contract)

`docs/core-engine-change-contract.md` is binding. The load-bearing ones here:

- **5 — the objective is real agent work.** Optimize cache-backed turn time and
  requested-concurrency workflow makespan, preserving prefill *and* decode.
  Aggregate tok/s alone is never sufficient.
- **6 — keep phases separate.** Cold prefill, cached append, decode, and mixed
  foreground-decode-plus-prefill keep separate evidence; a phase regression
  cannot be hidden by a good aggregate.
- **3 — explicit user choices are constraints.** Never silently reduce useful
  per-agent context or quality to win throughput.
- **7 — prediction chooses at most a finalist.** Only repeated identical live
  A/B can promote. A baseline-only measurement is not a completed decision.
- **10 — lifecycle is part of correctness.** Never disturb the user's live
  server; warn before optimizer-driven long reloads.

Method rules the project paid for: state the VRAM of *both* arms before calling
anything a hardware-usage test; `/health` is the authority for "serving", not a
log string; a predicted improvement is not promotion evidence; record background
load with every absolute number.

## The measured starting position

From the live run on `:8081` (Qwen3.8-Flash-Next UD-Q3_K_XL, 86 GB, 262,144 ctx,
`--n-cpu-moe 29`), read 2026-09-17:

| quantity | value |
|---|---|
| decode | 10.89 tok/s (91.81 ms/token) |
| uncached prefill | 105.7 tok/s |
| cache hit | 97.44% |
| tokens per `llama_decode` | 1.0032 (batch = 1) |
| CPU in the 13 `call` threads | 83.80% of 240,477 CPU-s |
| `cuda*` host threads | 0.03% |
| GPUs during work | 0% SM / 0% mem-ctrl, P8, 210 MHz |
| VRAM held | 74.05 / 80.85 / 89.35% |
| **queue share of turn time** | **41.4%** (`sum(queue_ms)` 27,333 s of `sum(total_ms)` 65,944 s) |

Three things follow, and they are the reason this branch exists:

1. **Nothing is saturated** — CPU ~45%, GPUs 0%, VRAM held but not exercised.
   Throughput is nevertheless capped. The ledger's reading is memory latency on
   the host expert path, which is an **inference**, not a counter (see `BOTTLENECK-1`).
2. **The objective is a hand-weighted bandwidth model, not absent.** The packer
   (`go/pkg/placement/placement.go:3312-3334`) is a greedy first-fit maximising
   resident expert layers under VRAM (`roomMBPer`) and models no time at all.
   But choosing between complete configurations *does* use a speed scalar, in the
   optimizer package — `optimizer.go:731`:

   ```
   AgentCost = (0.58*PrefillCost + 0.42*DecodeCost) * waves * contention
   ```

   Every term is a bytes/bandwidth service time (backbone bytes over VRAM
   bandwidth, expert bytes over VRAM or host DRAM bandwidth, activation round
   trips over link bandwidth). So the objective exists but is a **static,
   hand-weighted bandwidth estimate with a 58/42 prefill/decode split and a
   linear contention factor of `1 + 0.07*(lanes-1)`** — no measured
   tokens/second, no cache-prefix reuse, no queue/TTFT, no thread count, and no
   feedback from the agent-work measurements the project already collects. That
   is the concrete thing to fix.
3. **The plan is 4 expert layers worse than a plan ggrun itself computed.** Every
   `.place` record for this model at the live coordinates (ctx 262144 / ub 256 /
   par 1 / kv q8_0) agrees: `d4f0fa99dca4` (2026-09-15) stores
   `CACHED_NCPUMOE=25` with 23 GPU expert layers; `8e0cf3887cc2`, `38e30480c79c`,
   `c8b97d05c89b`, `88a847b97a70`, `9d6f1d5897ff` cluster at 25–27. **The live run
   is on 29 with 19 GPU layers.** The KV-first reservation is honoured and *not*
   the cause (self-KV 3,264 MiB is on the GPUs at the charged amounts,
   `PROBED_CONTEXT_MB_HOST=0`), and the place-cache key did not change, so this
   is not a schema migration. The gap is unexplained and is the concrete defect
   this branch exists to close.

   *Correction to an earlier reading of this branch:* the broader "25–27 cluster"
   spans **different coordinates** (ctx 196,608–1,048,576, ub 64–512, par 1–4) and
   is not evidence about the live plan. Only the same-coordinate records above are
   comparable. At ctx 1,048,576 the same model legitimately carries
   `NCPUMOE` 47–48 with 0–1 GPU layers.

   ### The gap is explained, and the cause is verified

   > **CORRECTION (same day, by running the probe).** The coefficient theory below
   > was **falsified**. It is preserved because the falsification is the useful
   > result. See "What the probe actually found" immediately after it.

   `loadMeasuredComputeCoefficient` (`placement.go:5086-5155`) builds the compute
   coefficient from **every** `.probe` whose text merely *contains the model
   basename*, ignoring the probe's own evidence class, its `ctx`, its `ubatch` and
   its `parallel`. It parses only `# ctx=`, `ubatch=` and
   `PROBED_COMPUTE_BUF_MB_CUDA*` — it never reads `PROBED_COMPUTE_BUF_EVIDENCE`.

   Reproduced independently for this model (148 samples):

   | | value | meaning |
   |---|---:|---|
   | median over all probes | **197.0927** | what the code uses |
   | median over `live-allocated` only | **148.3826** | the measurements |
   | samples | 148 = **124 `oracle-planned` + 24 `live-allocated`** | 84% are predictions |

   The charge formula (`placement.go:5241`) is
   `floor + (est − floor) · ctx/ref`, **not** `max(est·ctx/ref, floor)`. With
   `est = ubatch · hidden · layers · C / 1e6`:

   | C | est | charge |
   |---|---:|---:|
   | 197.09 (code) | 6,200 | **2,318 MiB** |
   | 148.38 (live) | 4,667 | 1,934 MiB |
   | 0 (pre-`d46e139`, unscaled 42) | 1,321 | 1,321 MiB |

   The backend's own `sched_reserve` allocated **1,159.25 / 1,036.11 / 828.00 MiB**.
   So the planner charges **2,318 MiB against a measured 1,159** — an overcharge of
   1,159 MiB, ≈ one whole expert layer (1,062.5 MiB) per split-owner card, ≈ the
   4-layer gap.

   **Timeline (verified):** the live binary was built `2026-09-16 08:44:47`;
   `d46e139` ("launch, detect, preflight: five fixes") is `08:44:20` — 27 s before.
   The live server started `09:13:31`, i.e. from that binary. The comparable
   cached plan `d4f0fa99dca4` (`NCPUMOE=25`) is `2026-09-15 08:41:33`, **before**
   the change. So the live run is the first launch on the new coefficient path,
   and the gap is a regression from it, not a property of the packer.

   Two independent defects live in the same function: **(a)** the evidence class
   is ignored, so 124 predictions outvote 24 measurements; **(b)** the 1,024 MiB
   floor is added *after* scaling rather than discounted by `(1 − ctx/ref)`
   (`computeFloorMB · (1 − 768/1024) = 768 MiB` of pure overcharge).

   **This is a plan-input defect, not an objective defect.** It is in a protected
   path, so it needs the full contract procedure and `scripts/verify-core-engine.sh`.

   ### What the probe actually found — the coefficient theory is FALSIFIED

   The numbers above are all real and reproduce exactly. **They are also
   irrelevant to the live plan.** Running the read-only probe
   (`ggrun dry-run … --ctx-size 262144 --ubatch-size 256 --cache-type-k q8_0
   --cache-type-v q8_0 --parallel 1`, with `GGRUN_TRACE_PLACEMENT=1`) shows the
   coefficient never reaches the packer:

   ```
   [trace] gpu0 probeHit=false ctx=262144 ub=256 kvq=q8_0 par=1 free=2779 sysOH=1845 compute=0 growth=0 fixed=1845
   ```

   `compute=0` is structural, not a lookup failure. `optionsFromRequest`
   (`main.go:2691`) sets **`RequireMeasuredBuffers: true`**, and
   `placement.go:3003-3007` then does:

   ```go
   if opts.RequireMeasuredBuffers {
       computeBufMB = 0   // the coefficient-derived estimate is DISCARDED
       expertOnlyComputeMB = 0
   }
   ```

   Default `opts := placement.Options{}` leaves `RequireMeasuredBuffers` **false**,
   which is why the coefficient path exists and its unit test passes — but every
   real launch and dry-run sets it true. **So the 197.09-vs-148.38 evidence-class
   defect is a genuine latent bug with no effect on the live plan.** Two real
   defects were found and neither explains the gap. Confirmed on the empty-key
   path too (`probeHit=false` with default options).

   **The live gap has a different, mundane cause: the probe cache was invalidated
   by measurement.** `gpuSignatureHash` (`placement.go:6352-6368`) embeds
   `bw%d` — the measured bandwidth — so when the 2026-09-16 bandwidth work landed,
   the GPU signature changed `f10a43e9645f` → `21cfcac0c1d0` and the entire older
   probe corpus stopped keying. This model's probes split **145 under the old
   signature, 8 under the new**. The live server started `09:13:31`, two minutes
   before the first new-signature probe was written (`09:15:02`), so **it launched
   with no usable probe and took the conservative cold-start path** — which packs
   fewer expert layers onto the GPUs. That is why it used 29 instead of the 25 a
   probe-informed plan produced the day before.

   Current probes for this model at the live coordinates under the new signature
   are already healthy: `d9af2da9171c` (ctx 262144 / ub 256 / kv gpu / par 0) has
   `PROBED_COMPUTE_BUF_MB=1159` `EVIDENCE=live-allocated`.

   **Consequence for the plan:** the branch's first target is not the coefficient
   and not the packer objective. It is that **a measurement change silently
   invalidates the probe corpus, and a launch during that window gets a worse
   plan with no warning.** The real question is why the bandwidth measurement —
   which is a *speed* input — belongs in the placement *identity* key at all.

   Also observed: `ggrun dry-run` fails today with
   `Model does not fit … no GPU has free VRAM after CUDA/compute overhead`, because
   the live server holds the VRAM. The probe still printed the full ledger and
   `-ot`, which is why it was decisive — a guard message, not a launch
   (`main.go:494` assembles it; no process was started).

## Where the evidence lives

- **`docs/core-standard-launch-todos.md`** — the measurement ledger. Its current
  head is on branch `fix/residency-ratchet` (102 ahead of main), worktree
  `/tmp/ggrun-reserve`. `main`'s copy is truncated and ~4,060 lines behind.
  **Read the branch copy, not main's.** Entries to read first: `BOTTLENECK`
  (2026-09-17), `LIVEAGENT`/`LIVEAGENT-4`, `VRAMAUDIT`, `SLOTCLEAN`, `SEATARMS`/
  `SEATCLEAN`, `CTXSTEP`, `HOSTBOUND`, `BOTTLENECK-CONFIRM` (added 2026-09-17 by
  this branch's author).
- **`docs/claude-local-agentic-goal-handoff.md`** — the goal/milestone record.
  Note: the standing `/goal` prompt names `claude-local-agentic-handoff.md`,
  which never existed; a local-only symlink now resolves it.
- **User constraint: development data does NOT go to the public repo.** Code goes
  to `origin/main`; the ledger and handoff stay local. Do not push docs.

## Open milestones, and their blockers

| item | state | blocker |
|---|---|---|
| `M3-BENEFIT` — matched companion on/off pair | open | needs exclusive GPU access |
| `M4` — different-model validation across resident/boundary/over-capacity | open | needs GPU |
| `M4` — the workflow-speed question | open | `LIVEAGENT` shows 96.2% cached prefix, which no calibration screen reproduces |
| `M5` — exact-commit Linux **GPU** run; GLM regression | open | needs GPU |
| `BOTTLENECK-1` — confirm the latency inference with a counter | open | needs `kernel.perf_event_paranoid=0` (root) |
| `BOTTLENECK-2` — why only ~6–7.6 of 14 threads are ever busy | open | cheap to test; distinguishes stalls from serialization |
| `SPECOFF` — `ggrun spec-test` has never run, so `--spec auto` has no profile | open | this checkpoint has **no draft head** (0 of 1,224 tensors), so it needs a separate draft model |
| `CTXSPEND-2` / `-3` — context sizing; `AGENT_CTX_*` demand read by no Go code | open | one-lever sweep |

## Rules for working on this branch

- `main` is the always-stable path. **Nothing merges to main from here without a
  deliberate decision**; this branch is a staging area for review.
- Protected paths (read-only unless the task explicitly requires it):
  `go/pkg/placement/`, `go/pkg/benchmark/`, `calibrate.go`, launch/admission in
  `main.go`/`preflight*.go`/`memory_probe_cmd.go`, `claude_progress*.go`,
  `go/pkg/detect/`, `go/pkg/recovery/`, `go/pkg/server/`.
- Any core change needs an invariant-focused test **and**
  `scripts/verify-core-engine.sh` (uncached).
- **Exactly one ggrun binary**: `/home/mik/go/bin/ggrun` via
  `go install -trimpath ./cmd/ggrun`. Never a second artifact or `.bak`.
- Never restart, kill, or reload the user's live server on `:8081`, and never
  start a rival model server while it holds VRAM (invariant 10). Check
  `curl :8081/health` before any measurement harness.

## Reviewer notes

For the reviewing model: the honest state is that **aim 2 (use all compute) and
aim 1 (fastest agentic work) are in measured tension here**, and the unresolved
question is whether today's plan is the best of the two or merely a fit-driven
compromise. The first useful review question is not "is the code correct" but
**"is the packer's objective the right one, and is the live plan optimal under
it?"** — because if the objective is wrong, every downstream tuning is settled in
the wrong direction.
