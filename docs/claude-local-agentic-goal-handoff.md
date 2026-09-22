# ggrun: easy, useful local agentic work

> **Current handoff — 2026-09-20:** read
> [/home/mik/ggrun-project/ggrun/docs/claude-local-agentic-goal-handoff.md](/home/mik/ggrun-project/ggrun/docs/claude-local-agentic-goal-handoff.md)
> first, including `LIVE-20260920` in that checkout's core TODO ledger. It contains
> the latest phase-specific live findings, reasoning corrections and next tasks.
> The material below is historical; keep new status in the canonical handoff.
> This pointer and development evidence remain local, not for publication.

User-aligned handoff, 2026-09-14. This is a development plan, not a claim that
the acceptance checks below have passed. Refresh branch and CI state before
acting; preserve the existing dirty production checkout.


> **Filename note.** The standing `/goal` prompt names
> `docs/claude-local-agentic-handoff.md`, which has never existed. The real
> document is this one, `claude-local-agentic-goal-handoff.md`. A symlink at the
> named path points here so the prompt resolves; it is a local convenience and
> is not pushed. Fix the prompt rather than duplicating this file — two copies
> would drift within a session.

## Current worker review — 2026-09-21

Read this section first. It supersedes conflicting status, diagnoses and
priorities below. Canonical handoff: this file in the main checkout, including
when working in `ggrun-perf` or `/tmp/ggrun-reserve`.

**The worker made useful progress, but the branch is not ready to merge.**
Reviewed `goal/agentic-speed-and-hardware` at `c7507c9`; main remains `9d44a31`.
The latest worker report is in Claude session
`c8e6f6b3-4d12-4b0b-a6b9-9ba129e6c949`, final response at
2026-09-21 18:13:54 UTC. No remote CI run for the worker branch was present at
review. Editing a workflow does not establish a passing run.

**Testing authorization:** the user explicitly freed the hardware on September
21 for ggrun development and live testing. The older instruction to wait for a
free testing window is superseded. Launch, stop and reload owned test models as
needed; check current ownership before affecting any subsequently started work.
Keep development evidence local. No production code or running model was changed
in this Codex review.

### Verified regressions and remaining correctness gaps

1. **Automatic placement can reuse a stale plan after a material link change.**
   `97404a3` removes bandwidth from the plan signature on the premise that the
   resulting tensor split already captures it. In the actual auto path,
   `placement.go:1773` computes the lookup key before strategy selection sets
   `s.TensorSplit`; a hit can return at line 1925 before replanning. The new
   tests supply a split manually and miss this lifecycle. An offline fixture
   using actual `Compute` and `SavePlacementCache` changes one measured link
   from 6269 to 1200 MB/s while preserving GPU identity and topology. Current
   code reuses split `[0.59, 0.27, 0.14]` with 17 CPU expert layers; fresh planning
   selects `[0.69, 0.31, 0]` with 16. With the previous `e593c4b` implementation,
   the same fixture misses the cache and agrees with fresh planning. This proves
   a plan-validity regression, not a speedup or memory-safety failure.

   Preserve stable allocation-evidence keys. Make material planning inputs
   participate in plan validation through the real automatic lifecycle, without
   restoring a flapping raw-bandwidth bucket. Test small measurement noise,
   meaningful changes, device order/topology, and save/reload behavior. Do not
   remove GPU indices from device-indexed evidence without explicit remapping.

2. **The new terminal test fails in a real terminal.**
   `TestTerminalAvailableFollowsStdinNotADevicePath` passes headlessly but fails
   under a PTY for pipe, null-device and regular-file stdin. Its assertions reject
   the controlling-console fallback that the implementation deliberately supports.
   Fix the test contract and cover both a detached process without a terminal and
   a usable controlling console. Source-string checks and cross-compilation do
   not establish native Windows console behavior or complete issue #63.

3. **The root containment change needs accurate errors and ownership handling.**
   `scope_unix.go:166` recommends `--ram-budget 0` to disable containment, but
   `backendMemoryMaxMB` treats zero as automatic budgeting and still applies a
   limit. Remove this ineffective advice; preserve required memory protection.
   The code also claims to resolve the systemd instance once per `Process`, while
   cleanup, statistics and property operations call `resolveScopeMode()` again
   and `Process` stores only a unit name. Retain the owning instance throughout
   the lifecycle, including failure cleanup. Directory/environment presence is
   not proof of a reachable manager; the non-root, missing-session path still
   selects `--user`. Cover usable root/system and non-root/user instances,
   unavailable/stale session state, startup failure, cleanup and memory evidence.
   The worker's actual root test was blocked by sudo authentication; do not
   represent forced-mode argv tests as end-to-end issue #64 acceptance.

4. **The new Windows job does not close the runtime coverage gap.**
   Existing CI already includes Windows cross-build/vet and native install/release
   smoke jobs. The added Ubuntu cross-build of all packages is somewhat broader,
   but the old `/dev/tty` code also compiled successfully. Correct the workflow's
   coverage claims and add behavior-oriented coverage for the missing console
   case. Run relevant Windows CI and Linux root lifecycle validation on the
   candidate; do not equate adding a job with passing it.

### Independent checks and evidence limits

- At `c7507c9`, Codex reran `scripts/verify-core-engine.sh`: all six packages and
  vet passed uncached. Headless `go test -count=1 ./pkg/tui` also passed.
- The targeted terminal test under a PTY failed with three assertions, as above.
- The offline planner fixture reproduces the regression against both current
  code and the old implementation. Source, outputs and replay instructions:
  `.cache/codex-review/worker-review-20260921/`. See `WORKER-REVIEW-20260921` in
  the main checkout's `docs/core-standard-launch-todos.md` for exact commands.
- The worker correctly retracted its claimed 9.53 -> 10.34 tok/s improvement:
  the prompts differed (9139 versus 9181 tokens), so it was not a matched test.
  Sequential runs in different processes can be valid comparisons when workload,
  state and other inputs are controlled; process identity alone is not the issue.
- Its fresh load wrote allocation evidence. A clean relaunch consuming that
  evidence, the 14-versus-28-thread comparison and Linux GPU E2E remain incomplete.
  Decode phase profiling and conversation-cache investigation remain pending.
- The 13.3-of-14-cores sample does not rule out memory stalls: on-core cache/DRAM
  stalls still accrue CPU time. It does not prove extra SMT threads improve
  useful work. The 15-CPU-expert-layer run also differs from September 20's
  29-layer run; their utilization samples are not a matched comparison.

### Continue in this order

Fix the verified cache regression, terminal test and containment gaps in bounded,
reviewable changes, following the protected-core contract. Then complete exact
fresh-load/clean-relaunch acceptance and Linux GPU E2E on the candidate commit,
plus relevant Windows runtime checks. Report unavailable coverage precisely.

Only then use phase-aligned profiling to select a bounded performance challenger
and trace the costly conversation-cache rebuilds. Compare the same real agent
work, useful context, quality and required review work with repeated matched
runs. Preserve the earlier parallel-1 evidence. Model/hardware inventories test
planning properties; they cannot establish universal performance. Success remains
correct, responsive local agent work on the available hardware, especially large
MoE models using host RAM and multiple GPUs.

## Historical review and live findings — 2026-09-20

Read this section first. It supersedes conflicting diagnoses and next-step
priorities below; older entries remain as history. This is the canonical local
handoff, including when working in `ggrun-perf` or `/tmp/ggrun-reserve`.

**Goal:** make correct local agent work easy, responsive and reliable on the
hardware the user owns, especially large MoE models needing host memory and
multiple GPUs. Measure useful completed work and foreground responsiveness at
the requested context/quality. Hardware activity is diagnostic evidence, not a
separate objective against which agent speed is merely a tie-breaker. A failed
attempt to improve speed by filling VRAM does not prove all ways of using more
compute are exhausted.

### Source state and verification

**Updated 2026-09-21.** The bullets below this note are the 2026-09-20 state and
are kept for history; this is where the branch actually is now.

- Main is `9d44a31` and unchanged. Claude's branch
  `goal/agentic-speed-and-hardware` is at **`a3c71c1`, 14 local commits, not
  merged or published.** Uncached core gate **6/6 green** on every commit;
  Windows and darwin cross-builds clean. `pkg/tui` and `pkg/download` pass — the
  gate does not cover them, so they are run separately.
- **Issues #63 and #64 are fixed.** #63 (`0cd1d6b`, `3906c61`): the TUI probed the
  Unix-only `/dev/tty`, so it was unreachable on Windows; now platform-split and
  mirroring bubbletea's own input selection, keeping `/dev/tty` on Unix and
  `CONIN$` on Windows. #64 (`c7507c9`): every scope command hardcoded
  `systemd-run --user`, which a root session cannot reach; the instance is now
  resolved once and threaded through all eight sites, so root gets a **system
  scope and keeps containment** rather than losing it.
- **The bandwidth-key boundary the 2026-09-20 note asks about is closed**
  (`97404a3`): the class was removed from the plan key entirely, because
  `tensorSplit` already carries what a bandwidth change moves. Measured:
  `12,149` and `12,151` classified differently while `orderGPUsByBandwidth`
  returned the identical order `[1 0 2]`.
- **A second orphaning mechanism was found and fixed** (`a3c71c1`): the identity
  hash also carried `g.Index`, a slot ordinal assigned by bus-id sort, so a
  reseat re-orphaned evidence.
- **Cache identity is verified end to end, not just asserted.** Load 1 wrote
  `live-allocated` evidence under the stable identity; load 2 reported
  `probeHit=true` on all three GPUs at the exact coordinate, reading load 1's
  `compute=1398` rather than the cold estimate `1751`. **Load time fell 210 s to
  60 s.**
- **Decode path: the earlier DRAM-latency framing is retired.** Decode measures
  **9.1-10.9 tok/s and is nearly flat** (91.6 ms/token at ctx 4 vs ~108 ms at ctx
  10,768). The ~1,900 ms/token outliers are a **full reprocess leaking into the
  decode measurement**, not slow decode. A checkpoint is reusable only when the
  incoming prompt's prefix extends a stored position; the review's task 928457
  (25,539 tokens) failed against checkpoints at 89k+, while 928839 (105,905)
  reused 105,849 from the same slot. So the reprocess is **correct behaviour**,
  not an eviction bug. Rate: 10 of 101 recorded tasks.
- **A matched A/B came back negative and is recorded as such** (`CKPTAB`):
  `--ctx-checkpoints` 16 vs 32 produced **bit-identical** cached-token counts.
  The cap is not the constraint. ggrun's 16 is also crash-driven, not
  conservative — a 32-checkpoint save crashed DeepSeek-V4 mid-request on
  2026-07-08.
- **Simulated small machines work** (`SIMBOX`): across three boxes and three
  model sizes, context scales with the box (51k / 96k / 230k), refusals name the
  resource and the number, and a 90 GB model on a 24 GB box correctly takes the
  CPU-offload path. **Planning correctness only** — a synthetic inventory does not
  emulate another machine's CPU, RAM speed or PCIe, so no performance claim may
  be based on it.
- **Linux GPU E2E: verified here, both paths.** `scripts/verify-gpu-install.py
  --relaunch` passed **both stages** on this hardware against the already-installed
  launcher, and now also against a **freshly installed** one — the gap the first
  run left. The install flow was run as CI runs it (`cat setup.sh | bash` into an
  empty `LLM_APP_HOME`, `INSTALL EXIT=0`), then served and relaunched:
  `passed: true` both stages, `CUDA1` allocated at 4,747 MiB each time,
  generation, streaming, cancel/reconnect in ~0.16 s, `port_released: true`,
  `forced_cleanup: false`. Closssing that gap found a real difference: the fresh
  install ships the **ik_llama** backend, which correctly refuses `nanbeige`
  ("No installed backend supports the nanbeige architecture") — a case the
  installed-launcher run could not see, because production has a different
  backend. A supported model was used instead.
  **Outstanding: only the CI *job*.** It is gated
  `github.ref == 'refs/heads/main'` and cannot be dispatched from a branch; that
  gate is deliberate, and merging to `main` is the user's decision.
- **Also found while installing:** a piped `setup.sh` without
  `LLM_SETUP_NONINTERACTIVE=1` prompts for a quantization and dies on EOF
  (`download_any_gguf.py:723`), plus `setup-home.sh:339: warn: command not
  found`. Non-interactive install is the documented entry point, so this is worth
  a follow-up; it is not this branch's change.
- **Relevant Windows CI: added, not yet executed.** The `windows-cross-build` job
  (`GOOS=windows go build ./...` and `go vet ./...` over every package) is in
  `ci.yml` and was verified to pass locally, but no CI run of it has been
  observed.
- **No performance has been promoted.** Every measurement is a correctness
  acceptance or a description of current state. Invariants 5, 6 and 7 are
  untouched, and the objective function (`optimizer.go:731`) has not been changed.
- Published release remains v3.2.9. Old PR #62 is still open.
- **A second senior review of the fixed branch returned HOLD**, and its findings
  were addressed in `9be1ec2` (two serious defects: the plan key losing its
  bandwidth signal, and `--ctx-checkpoints` dropped on the verified-config reuse
  path), `1819505` (the shipped charter's retracted premise, a vacuous test) and
  `81eea3e` (the dead legacy migration). A third pass over the current HEAD has
  not been run. The first review found real defects in this work, so assuming the
  fixes are clean without a second pass would repeat the mistake it caught.

### Source state and verification — 2026-09-20 (historical)

- Main is `9d44a31`. Its CI and Linux/Windows install jobs pass; its GPU jobs
  were skipped. The last verified Linux GPU pass is the older `0a66834`.
- Claude's latest branch is `goal/agentic-speed-and-hardware` at `e593c4b`, six
  local commits, not yet merged or published. `78b51b6` separates allocation
  evidence identity from noisy bandwidth measurements; `e593c4b` adds dry-run
  inventories. Codex independently reran the uncached core gate on September 20:
  six packages and `go vet` passed. Live performance of these changes is unproven.
- Before merging, cover generic bandwidth-key boundaries: 12,149 -> 12,151 MB/s
  changes the new 100 MB/s class despite only 2 MB/s of noise. Stable allocation
  keys are a useful correction; the plan-key stability claim is still too broad.
- Published release remains v3.2.9. Old PR #62 is open with a failing Python job
  and overlaps fixes already on main; reconcile it rather than blindly merge it.

### What the existing live run actually shows

Read `LIVE-20260920` in the main checkout's
`docs/core-standard-launch-todos.md` for evidence, windows and limitations.
No model was started, stopped, reconfigured or sent a test inference request.

- The running Flash-Next still uses one slot, 262,144 context, 14 CPU threads
  and 29 CPU expert layers, with a Qwen3.5-4B companion. It predates the new fix.
- **Sustained decode:** about 10.3-10.5 tok/s on recent agent work. A 22.8 s
  decode sample used 7.7 CPU cores; GPU activity averaged 11.8/23.7/9.3%.
  No major faults or disk reads occurred in either sampled phase. This does not
  identify DRAM latency as the cause: host expert operations, GPU operations,
  synchronization and long-context attention still need a phase-level breakdown.
- **Cache rebuild:** task 928457 could not restore a compatible checkpoint and
  reprocessed all 25,539 tokens, taking 210.5 s before its decode phase. During
  prefill, the 4070 received a median 10.12 GB/s over PCIe (peak 12.48 GB/s),
  near the independently cached H2D measurement in bursts. Decode traffic was
  much lower. A phase-specific transfer constraint is plausible; one blanket
  hardware-bottleneck label is not.
- **Queueing:** overlapping requests waited 244.5, 441.2 and 235.7 seconds.
  At the 12:18 UTC snapshot, queue time was 16.8% of summed recent request
  durations. Earlier near-zero queue readings omitted requests still in flight.
  Prompt tokens were nevertheless 98.3% cache-reused across that recent window:
  a high aggregate hit rate can hide costly individual rebuilds.
- The allocation-evidence cache fixed by `78b51b6` and this prompt/state cache
  are different systems. The former fix does not establish a fix for the latter.

### Next work and generalization boundary

1. Finish the cache-key review, including boundary noise, topology/driver/backend
   changes and allocation-versus-plan validity. In the user-approved live window,
   verify a fresh load writes reusable evidence and a clean relaunch consumes it.
   A particular expert-layer count is not the acceptance criterion.
2. Trace the observed cache rebuild and queue case through checkpoint selection,
   eviction and conversation changes. Determine which recomputation was required
   for model correctness before proposing retention or scheduling changes.
   Preserve useful per-agent context; do not widen slots automatically from this
   observation. Earlier parallel-1 wins remain evidence to account for.
3. Profile the measured decode path before choosing a performance challenger.
   Separate host expert compute/gather, attention, transfers and synchronization.
   CPU utilization alone does not measure DRAM stalls; speculation is not proven
   to be the only remaining lever. Evaluate the companion in the same resource
   budget, retaining required review work in both comparison arms.
4. Use generic model metadata, backend capabilities and measured topology to
   select legal candidates. Cover a large offloaded MoE, a resident dense model,
   and differing attention/state-cache behavior; exercise CPU-only, single-GPU,
   mixed-GPU and constrained-memory planning. Restricted hardware and inventories
   test planning/admission, not the performance of an emulated PCIe/CPU system.
5. Promote only after exact admission, matched real agent workflows, phase
   regression checks, a clean relaunch and GPU CI on the exact candidate commit.
   Unknown counters stay unknown. Unsupported backend features need bounded
   fallbacks and clear errors, not a promise that every model can run everywhere.

The method and correctness contracts should generalize. This rig's winning
thread count, split, slot count, cache policy or speculative method need not.
Keep evidence local; do not push these development notes. Preserve the user's
live server until the agreed testing window.

## Status 2026-09-16 — the goal, restated by the user, and where it stands

Two corrections landed on 2026-09-16 and both change how this file should be
read. Everything above the next `##` supersedes the 2026-09-15 framing.

### The goal, in the user's words

> "the goal of ggrun should be to choose the best if it already does that is
> good right?"

> "vram resident models are not the problem but big moe models that need the
> whole system, that is ggrun core feature and where we win"

> "ggrun should utilize the hardware to the fullest with big moe models, that is
> the key"

So: **"maximum hardware usage" as a standalone target is dropped** — it was a
proxy, and on two model classes it is anti-correlated with speed. What replaces
it is sharper, not weaker: **ggrun must choose the best configuration, and the
case that matters is the big MoE that needs CPU, RAM and every GPU at once.**
The VRAM-resident case is not the target and should not absorb more effort.

### Correction 1: six matched arms, not eight

The eight-arm table previously here **over-counted**. Adding a VRAM column — the
only way to tell a hardware-usage test from a mislabelled one — showed three of
the eight did not raise VRAM at all:

| # | arm vs ggrun default | model | VRAM base -> arm | result |
|---|---|---|---|---|
| 1 | more resident weights (`VRAMFILL`) | GLM | 0.7023 -> 0.7627 (+2,988 MiB) | tie, 6.83 -> 6.83 tok/s |
| 2 | hot-expert cache K=32, par 1 | GLM | 0.408 -> 0.703 | tie (inside 3.6% noise floor) |
| 3 | hot-expert cache K=32, par 2 | GLM | 0.389 -> 0.683 | **-7.9%** |
| 4 | hot-expert cache 8 slots/layer | GLM | 32,954 -> 36,498 MiB | **-57% decode** |
| 5 | `--n-cpu-moe` 42 -> 39 | GLM | 0.408 -> 0.600 | **-15.5%** |
| 6 | `--parallel 2`, 3 GPUs engaged | Qwen 27B | 29,950 -> 44,062 MiB | **-15% aggregate** |

Excluded, and not to be cited as hardware-usage evidence again:

- `--parallel 2` 4 lanes on GLM (-40%): VRAM went 0.408 -> **0.389**, down. That
  is a concurrency result, not a VRAM result.
- context 664,576 -> 131,072 (no change): tests *less* hardware.
- ctx fit -> 65,536 (-23%): arm B used **4,564 MiB less** than baseline.
  Retracted; the user caught it.

The conclusion is unchanged — on these two model classes ggrun's automatic plan
won or tied every matched arm — but its support is narrower than was claimed,
and is now stated accurately. See `VRAMAUDIT` in the TODO ledger.

### Correction 2: the CPU was never tested at all

All six arms were **VRAM placement** levers. GLM runs 42 of its expert layers on
the **CPU**. Not one CPU lever was ever measured — not thread count, not
affinity, not SMT, not NUMA. For a model whose work happens on the host, the
hardware-utilisation question was asked only of the GPUs. See `CPUPIN`.

Worse, ggrun's affinity is chosen **blind**. `placement.go:1507` takes
`caps.CPU.Cores` and emits `--cpu-range 0-13 --cpu-strict 1` without ever
checking whether those cores are free. On the measurement rig a background
process held cpu3 outright plus the SMT siblings of cores 0, 1, 6 and 8 —
**5 of the 14 pinned cores** — and `--cpu-strict` forbade migration to the 14
idle siblings alongside. Killing it moved loadavg from 5.14 to 0.52.

**Every absolute tok/s figure recorded on 2026-09-14/15 was taken under that
contention.** Relative A/B results survive (both arms shared the load); absolute
numbers are depressed, and a thread-scaling effect could have been masked
entirely.

### What is now measured rather than inferred

`ggrun detect --bandwidth` had **never run on this machine** — the command
failed on every install (`VRAMAUDIT-4`, since fixed by embedding the probe).
Every prior "host/PCIe bandwidth is the bottleneck" claim in this project was an
inference from spec sheets. Measured:

| | measured | theoretical | efficiency |
|---|---:|---:|---:|
| CUDA0 4070 H2D | 12,190 MB/s | 15,760 | 77% |
| CUDA1 3090 Ti H2D | 12,318 MB/s | 15,760 | 78% |
| CUDA2 3060 H2D (x8) | 6,270 MB/s | 7,880 | 80% |
| aggregate PCIe | **30,778 MB/s** | 39,400 | 78% |
| host memory (one-way memcpy) | **26,298 MB/s** | — | — |

The measurement splits what this project treated as one resource: aggregate
PCIe (30.8 GB/s) and host DRAM are **separate** ceilings of the same order.
Future claims must say which. Note the memcpy figure is one-way, so real DRAM
traffic is ~52.6 GB/s and read-only streaming sits somewhere in 40-55 GB/s.

### Honest position

- On GLM-class and resident-class models, **no configuration has been found that
  beats ggrun's automatic choice**, across six matched arms. That is evidence it
  chooses well; it is not proof it chooses optimally.
- The **CPU dimension is unexplored**, and on a big MoE that is where the work
  happens. `CPUPIN-1/2/3` are the open questions and the most promising
  remaining lead.
- Reliability work remains open independently: `SPENDSWEEP-2`,
  `RESERVEFIX-REGRESSION`, `PROBESTALL-4`. None is measured in tokens per second.


## Session evidence 2026-09-16 — measured on a quiet machine and a live server

Detail lives in `docs/core-standard-launch-todos.md` under the tags named here.
This section is the coordination summary; do not duplicate the ledger.

### Shipped to origin/main (commit d46e139, 23 files, +2,051)

| tag | defect | verification |
|---|---|---|
| `ARCHVETO` | a static ik-only arch table vetoed a **registered backend that carries the architecture**, so MiniMax-M3 (152 GB) could not launch at all | fixed; probe now overrides the table, unreadable binaries keep the conservative failure; 10 tests; **live: SERVED, ctx 358,400** |
| `VRAMAUDIT-4` | `ggrun detect --bandwidth` failed on **every install** — the script was not shipped | embedded content-addressed; 16 tests; works from a bare `go install` |
| memory-probe ledger | recovery ledger was passed `nil`, so the probe never learned from its own rejections | wired; per-round progress reporting |
| preflight descent | context priced but unable to cover a deficit caused a stall | descends; invariant tests |
| `PLANOVER-7` | GLM (an MoE) was given the **dense** compute coefficient — 2,026 MiB predicted vs 13,140 measured | uses the measured coefficient; probe 6 attempts -> 3 |

Uncached `scripts/verify-core-engine.sh` green on all six protected packages.
**Development data deliberately NOT pushed** (user constraint): the ledger,
this handoff and the theory note stay on the local `fix/residency-ratchet`.

### The measurement that reframes the rest — `LIVEAGENT`

2.5 hours of **real agentic work**, read read-only off a user-owned live server
(Qwen3.8-Flash-Next, 86 GB):

| | live agentic | same-day synthetic |
|---|---|---|
| decode | **11.86 tok/s** | 21.82 tok/s — **overstated 84%** |
| prompt cache hit rate | **96.2%** | disabled (`cache_prompt:false`) |
| peak context reached | **128,256** | 128 |
| `spec_decode_*` counters | **0** | n/a |

**Consequences for this document.** Every tok/s figure previously recorded here
is a 128-token completion with caching off. It is not a bound on agent-work
speed. Agentic work is dominated by cached prefix — 96% of prompt tokens never
reach the model — so cached-turn latency and KV behaviour outrank raw prefill,
exactly as invariant 5 states and this file has not been honouring.

**Context sizing is settled by three independent sources**: the user's 131,072
floor, `AGENT_CTX_MAX = 143,718` over 12,038 recorded samples, and a live peak of
**128,256** against **262,144 allocated**. The floor is right; allocation runs
about 2x what work uses. See `CTXSPEND`.


### Milestone 5 (release loop) — the standing blocker is fixed and CI is green

CI had failed **10 consecutive runs** since 2026-09-15 15:45, on `python-tests` /
`tests/test_release_relocation.py`:

    Error: scripts/setup-home.sh missing; the release entry points cannot run without it.

**Diagnosis.** `package-release.sh` fails closed on `scripts/setup-home.sh` and
`install.sh`, deliberately — its own comment records that an archive shipped
without them had a documented command that died with
`setup-home.sh: No such file or directory`, and that `install-e2e` never caught
it because CI runs the installer from the repo checkout while a real user runs
it from the archive. **The guard is correct.** The test fixture predated it and
packaged without the required inputs, so the test failed on a missing input
rather than the runpath fault it exists to catch.

**Verified before changing anything.** The exact CI error was reproduced in a
hand-built fixture (exit 1), the two files added, and the error disappeared —
leaving only a missing `patchelf`, which is local tooling CI has and which the
test's own `skipUnless` gates.

**Result.** Commit `9d44a31`, one file, +7 lines. Local python suite 41 passed /
1 skipped / 0 failed. CI run 35099584313: **all 16 jobs green, 0 failures**,
including `python-tests` and `core-engine-contract`.

Remaining on milestone 5, unchanged: an exact-commit Linux **GPU** run and a GLM
regression are still outstanding; Windows GPU remains untested. Only the
CPU/install half of the loop is now demonstrably closed.


### Milestone 3 (worker/reviewer) — contract VERIFIED, benefit still UNPROVEN

Previously recorded as "source audit complete; integration benefit unverified"
and carried as untouched. Splitting it properly: the milestone has two halves
and they are in very different states.

**Half 1 — the required-behaviour contract: verified.** `go/pkg/claudeauto`,
78 tests, 0 failures (`go test ./pkg/claudeauto/ -count=1`).

The central design point, which the milestone table demands and the code honours:
**ggrun never issues a permission verdict itself.** On reviewer overflow or
reviewer failure it re-routes the review to the *main model* (self-classify)
rather than skipping it, and every such path emits an explicit stderr notice —
so a required review is neither silently skipped nor implicitly granted.

| required behaviour (milestone table) | evidence |
|---|---|
| malformed output is not approval | `TestReviewerUnusableHTTP200FallsBackToMainModel` — 6 adversarial shapes: `{}`, thinking-only, tool-only, **prose-only `"This looks safe."`**, **ambiguous `<block>no</block><block>yes</block>`**, tool-only streaming. All must fall back, none may approve. |
| preserve review semantics on fallback | `TestReviewerFailureFallsBackToMainModel`, `TestUnreachableReviewerFallsBackToMainModel`, `TestReviewerClientErrorFallsBackToMainModel` |
| stop-sequence contract not misread | `TestReviewerAcceptsStopStrippedJSONVerdict`, `...StreamingVerdict`, `TestStopStrippedReviewerVerdictStaysUnambiguous` |
| context overflow handled | `TestRouterSelfClassifiesWhenPromptExceedsReviewerContext`, `TestUtilityOverflowStillGoesToMainWithoutCompanion`, `TestEstimatedPromptTokensErrsHigh` (estimator errs high, never under-counts) |
| foreground protected from background starvation | `TestForegroundTurnDoesNotWaitBehindAWholeFanOut`, `TestSchedulerRunsSafetyBeforeWarmBulk`, `TestSchedulerAgesOutAStarvedWaiter` |
| cancellation without duplicate side effects | `TestSchedulerCancelledWaiterReleasesItsPlace` |
| companion failure does not block main serving | `TestUtilityTrafficFallsBackWhenReviewerFails`, `TestUtilityFallsThroughToSeatedReviewerWithoutCompanion` |
| concurrency bounds hold | `TestSchedulerConcurrentLoadNeverExceedsTheLimit` |
| rejected verdict is accounted | `TestRejectedReviewerVerdictIsRecorded` |

**Half 2 — the benefit: not measured, and not measurable right now.**
Acceptance requires a *matched* comparison — companion-on versus a baseline that
still performs all required reviews on the main model — scoring main latency,
review wait, worker success, rework/retries and total time to a correct result.
That needs the GPUs, which were held throughout by a user-owned live agentic
session (see `LIVEAGENT`). Running it would have violated invariant 10.

Explicitly unchanged: **more processes or more GPU occupancy are not success**,
and extra review calls must not be invented to keep a companion busy.

- [ ] **M3-BENEFIT — the matched pair.** Same workflow, companion-on vs
  main-only-with-all-reviews. Score the five workflow measures above, not
  decode rate. Blocked only on exclusive GPU access.


### Milestone 4 (optimizer) — both PR #61 review findings are RESOLVED

Acceptance was *"Resolve the two review findings; different-model validation;
answer actual workflow-speed question before more tuning."* The first criterion
is now **met**, verified by running the regressions rather than reading the code.

**Finding 1 — post-load mmap failures counted as pre-load refusals: FIXED.**
`argvTimeAdmissionClasses` (main.go:4447) now holds only four classes;
`exactAdmissionMMap` and `exactAdmissionCUDAOOM` are **absent**, so a failure
raised after `startLaunchProcess` succeeds is charged as expensive. The review
also required that a future typed class must not be assumed cheap —
`TestEveryAdmissionClassIsClassified` (calibrate_budget_test.go:48) fails when a
new class appears.

**Finding 2 — changed admission policy retained the old calibration schema: FIXED.**
`CalibrationSchemaVersion` is **25** (was 24) and `PlacementPolicy` is part of
the scope key (calibrate.go:587). The two upgrade regressions the review
specified both exist and pass:

| review demand | test |
|---|---|
| an old decision cannot suppress the newly legal search | `TestDecisionFromAnOlderAdmissionPolicyIsNotReused` |
| a fresh decision is reused on the next identical launch | `TestFreshDecisionIsReusedOnAnIdenticalLaunch` |
| the policy change is versioned | `TestAdmissionPolicyChangeAdvancedTheSchemaVersion` |

All green (`go test ./pkg/placement/ -run 'Upgrade|Schema|Stale'`,
`./cmd/ggrun/ -run 'AdmissionClass|MMap|LoadedWeights|Budget'`).

**Remaining on milestone 4, both GPU-blocked:**

- [ ] different-model validation across the resident / boundary / over-capacity
      classes on the final fixed candidate;
- [ ] the workflow-speed question. The strongest exact statement remains
      *ubatch-512 completed one calibration screen in 32.37 s vs 48.72 s
      (-34% wall) while decode fell 16.6 -> 10.3 tok/s*. That does **not**
      establish which plan completes representative agent work faster, and
      `LIVEAGENT` now shows why it cannot: agent traffic is 96.2% cached prefix,
      which no calibration screen reproduces. The phase guard stays as-is;
      changing its 5% policy is an explicit objective change requiring
      representative-workflow and foreground evidence.

Standing caution from the review, still binding: a request carrying only a
ubatch override may also move expert placement and topology, so any comparison
must diff the **final full argv**, not the override.


### DeepSeek V4.1 — checked before buying hardware — 2026-09-16

User asked about running V4.1 for the ngram case and offered to raise RAM to
256 GB. Three gates, and RAM is not the binding one. Detail in `DEEPSEEK41-0`.

| gate | state |
|---|---|
| backend supports `deepseek41` | **missing** — 0 of 5 installed forks; no reviewed recipe. Mainline PR #28696 adds *conversion only*, no runtime. `ik_llama.cpp` PR #2455 is furthest along, and this rig already has an ik_llama backend. |
| RAM >= ~256 GB | 246.3 GB model needs ~198 GB resident; 212 GB leaves ~14 GB headroom |
| coefficient gate | `placement.go` matches the literal `deepseek4` in 146 places; `deepseek41` misses all of them, so the first launch gets the dense coefficient — the `PLANOVER-7` defect again |

**Recommendation recorded: do not buy RAM yet.** The backend gate binds first,
and separate work on *bounded SSD expert caching* (`halo-box/strix-llama.cpp`
#48/#49) would remove the resident-RAM requirement altogether — which is exactly
this project's "useful work from affordable hardware" objective. Wait to see
which lands.

**Generalisable defect found on the way** (`ARCHVETO-2`): arch handling compares
**exact literals**. Two near-miss strings changed planner behaviour in one
session — `glm5next` silently took the dense compute coefficient (`PLANOVER-7`,
6.5x low on 60% of the plan), and `deepseek41` will do the same against
`deepseek4`. Prefer family/prefix matching with an explicit exception list over
146 exact comparisons.

Positive note: ggrun's unsupported-architecture path is **correct and helpful** —
it names the architecture, the backend that lacks it, and offers to search open
PRs for a fork. That is the `ARCHVETO` failure class handled properly.

### Open, ranked by measured or predicted magnitude

| rank | item | status |
|---|---|---|
| 1 | `SPECOFF` — GLM ships a 35 MB MTP head that is loaded and **discarded**; all spec counters zero in production | the only lever that cuts **bytes per accepted token** rather than moving them faster. `--spec auto` is correctly gated on a verified profile and **no profile has ever been generated**. `ggrun spec-test` is both experiment and fix. |
| 2 | `CPUPIN` — affinity chosen by index, never checked for load | **+14% decode and +77% context measured** from stopping one contending process. `CPUPIN-3`: rank cores by measured idle. |
| 3 | `LIVEAGENT-2` — the 3090 Ti holds 11 of 19 GPU expert layers and 63% of the tensor split, and runs at **5%** while the 4070 pegs at 91% | observed, **not** diagnosed. `--main-gpu` unset is a candidate. Do not read utilisation as throughput (`HOTEXPERTS`). |
| 4 | `CTXSPEND-2` — context vs residency, one lever, agent metric | `-1` was **inconclusive**: four levers moved at once including KV to host, and arm A lost its samples to a harness defect. |
| 5 | `VRAMIDLE` — 8-17 GiB idle on every launch, ceiling 83% | reserve explains ~1.2 GiB; granularity does not explain the rest. |
| 6 | `CTXREPEATS` — Qwen 27B batch-shape trade, 44% of steady-state turn time | the one original symptom that survived re-checking. Untouched. |

### Corrections issued this session

- **Six matched arms, not eight.** Three cited arms did not raise VRAM at all.
- **"The planner never spends freed VRAM" is wrong.** Given context as a free
  variable ggrun re-plans hard: `--n-cpu-moe` 42 -> 34, ~24 GiB of experts moved
  onto the cards, KV relocated to host.
- **"ggrun underestimates PCIe by 2x" retracted** the same turn. X299 is PCIe 3.0;
  ggrun reads the host-limited ceiling correctly.
- **The global `CLAUDE.md` hardware block was wrong** on CPU, RAM and GPU order.
- **CI was already red** for 8 consecutive runs before this session's push
  (`python-tests` / `test_release_relocation.py`); not caused here, not fixed here.

### Method rules added after paying for them

1. A harness must distinguish **absent** from **slow** from **zero**, in the file
   a reader will look at. One served arm recorded zero samples and no error.
2. **Never `pkill -f llama-server` before checking port 8081.** A harness came one
   command from destroying a live 2.5-hour session. All harnesses now refuse.
3. State the **VRAM of both arms** before calling anything a hardware-usage test.
4. **One lever per arm.** Recorded twice, violated twice.


## The core objective — non-negotiable, partly met, one dimension unmeasured

**Run the selected model as fast as the given hardware allows.** The model is the
user's choice. ggrun's job is to extract the most from that model on that
machine, never to be faster by serving something else or something smaller.

**The automatic default path must reach it unaided.** `ggrun <model>` with no
flags is the configuration that has to handle every case. A result that needs a
hand-passed `-ctx`, `--parallel`, `--n-cpu-moe`, or any other tuning knob is a
*diagnostic*, not a fix: it proves a faster configuration exists and that the
planner failed to find it. Flags stay available as overrides and as measurement
instruments, but no milestone here may be closed by an invocation the user would
have to know to type. Where this document records a win from an explicit flag,
that win is open work until the automatic path produces it.

This milestone is not complete while a selected model is measurably slower than
its own hardware permits. Status as of 2026-09-16, with each original symptom
re-checked against measurement rather than restated:

| model | symptom as originally written | status after measurement |
|---|---|---|
| GLM-5.3-Flash | **24-30% of VRAM unused** (`fraction_of_vram` 0.70-0.76) while 42-43 of 48 expert layers sit in host RAM | **RESOLVED — not a defect.** Filling it was tested twice with VRAM genuinely raised (`VRAMFILL` +2,988 MiB; `HOTEXPERTS` 0.408 -> 0.703) and decode did not improve in either. A residency gain pays in proportion to the *offloaded bulk* it moves; 3 GiB of a ~120 GB host-resident expert set is 2.5%, inside noise. `fraction_of_vram` is a diagnostic, never a promotion target. |
| Qwen3.8-27B | automatic context buys the largest fitting window by shrinking batch shape to 2048/512 instead of 8192/1024 | **STILL OPEN** (`CTXREPEATS`, 44% of steady-state turn time). This one is a real objective-vs-fit divergence and is untouched. |
| every launch measured | one GPU near saturation while another idles | **RE-READ.** SM occupancy was shown to be a bad signal: `HOTEXPERTS` drove CUDA1 from 5.1% to 17.7% SM *while throughput fell*, because the occupancy was upload traffic, not useful compute. Busy is not productive. Engaging the idle device directly was also measured and lost 15% aggregate on the 27B. Do not treat an idle GPU as evidence of a defect without a matched throughput arm. |

**The dimension that was never measured is the CPU**, and on a big MoE — the
case the user has identified as ggrun's core feature — that is where the work
happens. See `CPUPIN`.

### The defect underneath all three

**ggrun optimises for fit, not for outcome.** Context fit takes the largest
window that fits; placement takes the most weights that fit; admission asks only
whether it fits. Those are capacity objectives. They coincide with speed often
enough to look correct, and they diverge measurably.

Filling VRAM with KV nothing reads is not *using* the hardware, it is occupying
it. Equally, leaving 12 GiB idle while experts page from host RAM is not
conservatism, it is unspent capacity.

The right shape is:

- **Requirement**: per-agent context at least what the workload needs — the floor
  `CTXFLOOR` found, where 16,384 silently truncated a six-turn session.
- **Constraint**: it must fit, fail-closed. Unchanged.
- **Objective**: fastest correct agent work on the selected model. **Currently
  absent from the planner.**

ggrun already *measures* the objective — `correct_tasks_per_minute`, workload
makespan, the phase guards. It does not let the planner *choose* on it, because
the coordinates that move it (context, slots, topology, threads, expert
residency) are not in the candidate space. The baseline won every calibration run
in this session across four model/mode combinations, which is what a search with
nothing to offer looks like.

### Optimise the logic, not this machine

The mechanism generalises; the numbers do not. "Do not trade batch shape for
context nobody requested" is model-independent. "Cap context at 32k" is a fact
about one model on one rig and must not be encoded. The change contract says the
same thing, and every fix under this objective is held to it.

### Acceptance

This objective is met when, for a selected model on given hardware, ggrun can
show that no candidate configuration it can construct serves real agent work
faster — and when the levers above are reachable by the search rather than only
by a human passing flags by hand.

## The actual product goal

Make capable local agentic work easy to start, responsive, and reliable on the
hardware the user owns, including inexpensive consumer machines. Getting useful
work from affordable hardware and extracting the most from an existing machine
are central product requirements. Do not assume a new high-end GPU, homogeneous
devices, or a model that fits entirely in VRAM.

The user should select a model or receive useful hardware-aware recommendations,
launch through the ordinary TUI or CLI, and do
real work without becoming a llama.cpp configuration expert.

The requested model must serve correctly with useful context and working tool
turns. The main model, optional worker/reviewer, caches, memory, and concurrency
form one serving system. Use the available hardware to improve the completed
workflow. Do not substitute VRAM fill, SM utilization, expert residency, or an
isolated token rate for that outcome.

The user's ambition remains the best performance the hardware can provide.
Engineering evidence can establish the best validated configuration among the
alternatives explored, explain remaining bottlenecks, and identify the next
opportunity. It cannot certify a universal 100% optimum from one machine.

Product acceptance includes:

- A straightforward install, model selection/download, launch, and connection
  to a supported agent client. Startup progress and failures are understandable.
- Generic configuration from actual model metadata, backend capabilities, and
  available hardware. Respect explicit user choices. Try bounded legal recovery
  before a precise, actionable unsupported/insufficient-resource error.
- Useful per-agent context, reliable tool calls and streaming, working prefix
  reuse, foreground responsiveness, cancellation, and clean restart.
- Optional local workers and reviewers that improve the complete workflow,
  preserve required review behavior, and degrade gracefully when unavailable.
- Hardware-aware model recommendations that distinguish capacity, supported
  features, expected performance, and task capability. Do not silently replace
  the chosen model with a smaller one to obtain a faster benchmark.
- Practical configurations for supported CPU-only, single-GPU and mixed-GPU
  systems, including limited VRAM and host-memory offload. Make capacity/speed
  trade-offs understandable; do not require a hardware purchase to bypass a
  software defect or present a barely responsive fit as a good agent experience.
- Tested Linux installation and real GPU serving, Windows installation/release
  coverage, versioned evidence, and instructions matching the shipped artifact.
  Windows GPU coverage must be reported accurately when unavailable. macOS is
  lower priority for this work.

## Direction review — 2026-09-15

Review time: 2026-09-15T08:25:20+00:00. Source reviewed through local `ce3bc0a`. PR #61 is
merged as `0a66834`; Linux CPU, Windows CPU and Linux real-GPU install/serving
run 34898139305 passed on that exact commit. PR #62 was still open at the
previous `81c9f96` head when checked; the latest local KV-pinning change had not
been covered by those remote checks. Refresh identity before concluding CI is
current. Earlier PR #61 mmap-accounting and schema findings were addressed in
its merged result; they are not outstanding #61 blockers.

**Direction: keep the focus on the actual Claude Code path, useful per-agent
context, and main/companion workflow performance.** The slot/KV interaction is
a concrete relevant problem. Preserving baseline KV while varying automatic
slot count is a sensible candidate-generation fix. The reported two extra GPU
expert layers are a planner result, not a measured speedup. The cheaper
parse-request-to-candidate reproduction should remain as regression coverage;
it is more useful than repeated large loads to locate a filter.

### Correct the reviewer evidence before selecting a default

The saved `scratchpad/review-lane.py` and `review-ab.sh` do not support the broad
`REVIEWLANE` conclusion yet:

- The driver has two serial loops running alongside each other: at most one
  review and one foreground request in flight, not twelve concurrent requests.
- It discards successful response bodies. Eight HTTP successes do not establish
  eight correct review decisions or useful foreground answers. Prompts are
  tiny synthetic read-file questions; delegated worker tasks are absent.
- The wrapper waits for backend health and a router address, then starts traffic
  before final launch/profile/client acceptance. It launches `--claude-code`
  with stdin from `/dev/null`. The main-only log ends with the client error
  "Input must be provided either through stdin or as a prompt argument when
  using --print". The companion arm's log ends with an Anthropic-router canary
  failure. Startup and teardown therefore contaminate the comparison.
- Main-only metrics contain immediate 502s with zero queue/response time after
  earlier requests completed. This does not prove overload or that
  self-classification intrinsically fails; trace launcher/backend lifetime and
  the error cause first. The gateway log also admits only one active main
  request despite four backend slots, so four-slot queueing cannot be assumed.
- Both arms retain roughly 172-178 second foreground maxima. Comparing only
  16.58 versus 10.28 second medians hides the unresolved long stall.
- Cleanup selects every `bin/llama-server` process on the machine. Replace it
  with owned PID/process-group cleanup; this is especially important with other
  work ongoing on the shared box.

The raw 0.099-second reviewer HTTP responses suggest the helper can isolate
cheap requests, but do not justify a universal 2B recommendation. First run an
accepted, stable local client/router session with unchanged required-review
semantics, verify actual outputs against expected decisions, include useful
worker tasks, and account for startup/lifecycle separately from serving time.
Measure the requested real workflow at a representative context and report
correct results, total completion time, tail stalls and resource trade-offs.
Keep the experiment bounded. Main-only remains a supported baseline to repair
and compare, not a path to write off because this harness produced 502s.

### Address two PR #62 generality issues

1. Slot candidate generation and eligibility changed, while
   `CalibrationSchemaVersion` is still 25, the version introduced by #61.
   Old baseline-won/admission-only decisions can still suppress this newly legal
   search at the same baseline scope. Version or migrate the changed policy and
   retain an upgrade/reuse regression. Clearing caches manually is not the
   product fix.
2. `holdExpertResidency` receives raw `outcome.Device` but builds a penalty map
   for `ReplanAfterOOM`, which indexes physical `caps.GPUs[i].Index`. The nearby
   existing recovery path uses `physicalGPUIndex(..., visibleToPhysical)`;
   the new helper calls omit that translation. On selected/reordered GPUs this
   can penalize the wrong card or match none. Convert at the boundary and test a
   non-identity visible-to-physical mapping. The current 0/1/2 rig can hide it.

The speculative residency ratchet has not demonstrated that its repacking
branch fixes an observed launch. Its tests cover bookkeeping and fallback,
not a successful repack that restores the intended per-device relief. Require
that case and a full-ledger regression before merging it together with the
slot change; use a separate small change if that keeps the proof clear.

Focused existing slot/residency tests passed in this review. No core code,
running model or CI configuration was changed; no new performance run was made.
Priority: repair the comparison lifecycle and the two code issues, verify final
slot candidates through the real resolver on another model class, then select
main-only versus companion and slot policy from correct completed workflows.
Avoid further topology/thread/hot-cache exploration until these answers are
reliable. Keep long-context and real-worker acceptance open.

## Independent review and current direction — 2026-09-14T19:50:58+00:00

This review supersedes stale PR states and completion claims below. Preserve the
product goal and Claude's detailed progress record. Reviewed PR #58's merged
result `49463f4` and PR #61 at `91be53c`; the latter was open, mergeable, with
22 successful checks and two skipped GPU jobs at inspection. Refresh before
acting. No model was launched, stopped or reconfigured by this review.

### The user's latest steering

The latest two substantive messages in Claude's session were:

- 19:44 UTC: "but which of the runs is overall faster in agetnic work"
- 19:45 UTC, absorbed mid-turn: after seeing a change work, test a different
  model from the one used to develop it.

These add two acceptance requirements: answer with matched complete-workflow
results, and verify generality beyond the reproducing model. A benchmark guard
working correctly and a fix working on Flash-Next alone are intermediate proof.

### Latest user steering — 2026-09-14 19:54 UTC

The user asks whether the runs exercise the actual Claude Code mode, notes that
main-model safety classification can be acceptable, and asks whether the second
model's worker role makes the combined setup better. These messages were partly
sent mid-turn; they are product requirements, not benchmark conclusions.

- Verify the tested entry point and effective mode. Record whether the run uses
  the real Claude Code client/router, which model handles foreground, review
  and utility requests, and the actual companion reservation/allocation. A
  backend-only test does not establish that this integration works.
- Main-only serving is a valid comparison: the chosen model handles reasoning,
  required safety classification and the same utility workload. Preserve review
  semantics and client permissions; a separate reviewer is not mandatory merely
  because one is available.
- Compare that baseline with the same workload using a companion for both
  required reviews and explicitly delegated worker/utility tasks. Charge its
  weights, context and contention against the main model's usable resources.
  Count successful worker tasks and review outcomes, queueing, main-model
  responsiveness, retries/rework and total time to a correct result. An idle
  companion or a healthy reviewer endpoint is not evidence of benefit.
- Keep main-model context/quality and the requested work equivalent. If either
  full configuration cannot meet those constraints, record that capacity result
  rather than quietly reducing the main model's context. Add a review-only arm
  only if it is needed to explain the combined result; avoid a broad sweep.
- Select the configuration on the complete local-agent outcome. Retain the
  existing main fallback when a companion cannot do useful work. Do not disable
  required review to manufacture a speedup or assume that two models must win.

### Two concrete PR #61 issues to resolve before merge

1. **Post-load mmap failures are incorrectly counted as pre-load refusals.**
   At `91be53c`, `exactAdmissionLoadedWeights` in `go/cmd/ggrun/main.go` returns
   false for every typed class except CUDA OOM. But `exactAdmissionMMap` is
   emitted after `startLaunchProcess` succeeds and
   `validateObservedMMapPageability` fails; the process is then stopped. This
   can exclude a real model load from the expensive-failure budget and print an
   incorrect "before any model load" message. The new budget test explicitly
   puts mmap in the cheap list, so its passing result does not catch this bug.
   Correct the classification using actual lifecycle evidence. At minimum count
   this post-load path and unknown typed classes as expensive; do not infer
   that a future failure class is cheap simply because it is typed. Add a
   regression tied to the post-start path and verify cheap admission refusals
   still leave the bounded fallback search available.
2. **The changed admission policy retains the old calibration schema.**
   `CalibrationSchemaVersion` remains 24 and the policy is absent from the
   scope key. `calibrationPlan` and cached-winner application can reuse an old
   `default` decision before the new ladder runs. Admission-only suppression
   also keys the finalist, which need not change when its fallbacks change.
   Manually moving cache files for experiments hides this upgrade problem.
   Version or explicitly migrate performance/admission decisions for the new
   selection semantics. Preserve independent fit proof where valid. Add an
   upgrade regression: an old decision cannot suppress the newly legal search;
   a fresh decision is reused on the next identical launch.

These are source-review findings, not new observed serving failures. The
existing focused recovery, ladder and budget tests passed uncached in this
review; they do not cover these two requirements adequately. No core source was
modified and no new full core gate or performance experiment was run here.

### Generality and the speed question

At the reviewed state, the strongest exact statement is: **ubatch-512 completed
one calibration screen in 32.37 s versus 48.72 s for default** (about 34% less
wall time). Decode throughput in that screen fell from 16.6 to 10.3 tok/s.
That is why the current phase guard retained default. It does not establish
which plan completes representative agent work faster. The separate tool-task
comparison for ubatch-512 had not completed; its attempted wrapper was killed
by a broad `pkill -f` command. That attempt is neither a model failure nor a
performance result. Use owned PID/process-group cleanup with identity checks;
do not repeat broad process-name matching.

Run the same functional workflow with default and the exactly admitted
ubatch-512 plan, preserving model/quant, context, KV quality, client demand,
cache state and companion policy. A request containing only an ubatch override
may also change expert placement and topology: compare the final full argv and
ledger, and describe a whole-plan comparison when those differ. Include a
representative sustained-output case and actual project context as well as the
short repair screen. Keep results for distinct harnesses separate. Retain the
current phase guard while evaluating; any proposed change to its 5% policy is
an explicit objective/policy change requiring representative-workflow and
foreground evidence, not a workaround to promote this one aggregate winner.

Claude is checking the resident Qwen3.8-27B case and plans to recheck the GLM
memory-constrained case before merging #61. Keep that order, on the final fixed
candidate. Exercise actual generated candidates across these different classes;
synthetic tests containing only names and identical strategies do not establish
semantic diversity. The current parser groups `moe`, `topology`, `single` and
`split` separately although they can all change placement. Prefer typed
coordinate metadata or actual configuration differences for diversity and
exact-configuration deduplication; add rename-invariance coverage. A smaller
ubatch may fit after a larger one fails, so diversity remains a preference
inside a bounded admission ladder, not proof that its other rungs are useless.
Do not claim that these three models on one rig validate every public system.

### Milestone status and next work

| Milestone | Reviewed state | Next acceptance |
| --- | --- | --- |
| 1: launch/recovery | #58 merged; expert-relief rollback fixed; deficit-sized ceiling step reverted | Recheck Qwen and GLM on the final #61 candidate; keep the exact ledger regressions |
| 2: local agent experience | Tool smoke, cancellation/reconnect and roughly 6.4k-prefix observation added | Real client, substantial repository/multi-turn context; connect prefix checks to appropriate GPU validation; fix cached-summary display separately |
| 3: worker/reviewer | Existing routing/fallback audited | End-to-end required-review and worker correctness plus matched companion-on/main-only comparison; benefit remains unproven |
| 4: optimizer | A complete measured screen and baseline restoration demonstrated | Resolve the two review findings; different-model validation; answer actual workflow-speed question before more tuning |
| 5: release | CPU/install checks green; GPU skipped on the reviewed PR | Exact-commit Linux GPU run, GLM regression, final artifact/install proof; Windows GPU remains untested |

The installed-serving cancellation check demonstrates successful follow-up
completion after disconnect. Its `recovery_s` includes that request's work and
is not an isolated scheduler slot-release measurement. The prefix check records
a ratio but does not validate answer correctness or fail when cache evidence is
missing; treat it as observation until the intended cache contract has an
appropriate capability-aware assertion. Configuring 262k context and sending
6k-8k prompts is not 262k-context acceptance.

High configured-worker CPU usage and uneven GPU activity identify hypotheses.
They do not by themselves prove that more threads or a different topology will
help; an unsaturated average PCIe counter also does not rule out transfer or
synchronization stalls. Finish the workflow/generality evidence before opening
another thread/affinity optimization. Likewise, a worker/reviewer code audit and
request timing fields do not establish review accuracy, worker task success or
reduced rework: retain task-level outcomes in the comparison.

## Reasoning corrections to keep

1. The Qwen3.8-Flash-Next recovery fix is valuable because it turns a rejected
   launch into usable serving. Its three-task result (7.49 correct tasks/min,
   about 15 seconds median) is functional and preliminary performance evidence.
   It does not prove general coding quality or that all core recovery is fixed.
2. Comparisons between different models do not isolate expert residency.
   Architecture, active parameters, quantization, generated output, context and
   attention behavior also differ. A resident-layer fraction is not the same as
   the fraction of routed work served from a hot cache.
3. The GLM smaller-context/CPU-KV run completed 3/3 tasks at approximately 0.91
   correct tasks/min versus 1.03 for the earlier baseline. Multiple settings
   changed. This neither isolates KV cost nor disproves expert caching.
4. Keep KV near the device doing attention where practical. In the inspected
   GLM backend, disabling KV offload also moves the corresponding attention
   computation to CPU. Freeing VRAM this way has a computation cost.
5. CPU-resident experts ordinarily execute using host memory; do not describe
   all their weights as streaming across PCIe every token. Separate activation
   transfers from actual expert-cache weight uploads and synchronization.
6. Two baseline runs do not establish a statistical noise floor. Three small
   tasks cannot establish an 8% general improvement or loss. Changing both
   client lanes and server slots measures a whole setup, not the slot effect.
7. The product contract already prioritizes correct agent work over utilization.
   Trace the actual decision code before concluding its objective is VRAM fill.
   A field with no readers does not prove an entire capability is absent;
   inspect current main and existing candidate/ledger paths before adding one.

## Execution order

### 1. Consolidate launch correctness

PR #58 merged as `49463f4`; #59 and #60 also merged. Its deficit-sized ceiling
step was reverted; preserving proven expert relief across measured recomputation
resolved the observed launch cycle. Continue from that result. Review PR #61
against the current-direction section above without folding unrelated staged
work from the production checkout into it.

Before editing protected paths, read `AGENTS.md`,
`docs/core-engine-change-contract.md`, `docs/optimizer-theory.md`, and the
relevant current `docs/core-standard-launch-todos.md` entries.

Review the recovery invariant, not just the successful Qwen example:

- Preserve the full effective configuration and explicit constraints through
  planning, oracle admission, measured recomputation, recovery and restore.
- Prefer candidates with credible relief on the failing resource. A total KV
  saving across three GPUs is not relief of the same size on the failing GPU.
  CPU KV relief cannot simply be credited against a GPU deficit.
- Specifically review `contextReclaimTokens` and
  `oracleContextDropCoversDeficit` in PR #58: the inspected implementation uses
  an aggregate KV estimate, and unknown geometry receives priority. Treat this
  as a generality concern requiring tests, not proof of another observed OOM.
- A total VRAM deficit may be relieved by a different allocation on that same
  device. Do not insist that a compute failure can only be fixed by changing
  compute memory. Compare complete per-device and host ledgers.
- Rank relief as an estimate, not admission authority or a universal hard veto.
  Retain legal fallback candidates when geometry is unknown or no single lever
  closes the deficit. Re-admit every changed argv exactly.
- Bound attempts, total startup time and expensive reloads. Detect repeated
  configurations and negligible progress. Increasing a retry count alone does
  not resolve selection of an ineffective recovery action.

Add invariant-focused regressions for the observed Qwen/GLM failures plus
uneven GPU KV, CPU KV, unknown/recurrent geometry, explicit context/ubatch,
multiple competing device deficits and no-strong-candidate fallback. Scope any
refactoring to the demonstrated problem. Run the uncached core gate, then
verify clean default launch, tool turns, cache reuse and restart on the fixed
Qwen case and the GLM regression case. Record the exact installed identity.

Deliverable: the generic launch/recovery contract survives these cases on the
merged build; documentation distinguishes completed proof from remaining work.

### 2. Verify the ordinary local agent experience

Follow the actual user journey: clean temporary installation, normal TUI or CLI
model selection, automatic configuration, supported agent-client connection,
and a repository task that reads, edits, runs tests and resumes after tools.
Exercise a substantial reusable project prefix and cancellation/reconnect.
Use equivalent inputs to check TUI/CLI resolver agreement.

Show effective per-agent context and any capacity limitation clearly. Do not
invent a universally sufficient 128k/32k floor or silently reduce context to
win a short benchmark. Preserve current user intent; an explicit context or
maximum request is a constraint. A separate automatic context policy needs
long-context evidence before replacing current behavior.

Repair the highest-impact failures encountered in this path before expanding
optimization scope. Healthy `/health` or text generation alone is not completion.

### 3. Include the worker/reviewer in the same plan

Audit and extend the existing `go/pkg/claudeauto` routing, scheduler, metrics and
fallback code; do not create a second unrelated orchestration system.

The inspected implementation already recognizes an explicit `local-fast` tier
and a separate Claude Auto permission-review protocol. That reviewer is not
automatically a general code-review agent. Keep these contracts distinct:

| Role | Intended benefit | Required behavior |
| --- | --- | --- |
| Main model | Reasoning and user-facing work | Preserve chosen model, context and foreground progress |
| Utility/worker | Suitable delegated tasks, summaries and other cheap-tier work | Honor explicit client tier/task selection; validate output/tool contract; retain a usable fallback |
| Permission reviewer | Required agent permission decisions | Correct protocol and timely answers; malformed output is not approval; preserve review semantics on fallback |
| General code reviewer, if added | Find errors and reduce rework | Separate task contract and evidence of useful findings; tests remain independent correctness evidence |

Budget companion weights, KV, compute buffers, CPU demand and runtime overhead
together with the main model. Account for the experts/context it displaces and
the shared host bandwidth it consumes. Place it where the full workflow wins;
do not reserve a particular GPU index on every machine.

Protect required review and foreground work from background queue starvation.
Handle context overflow, companion crash, timeout, malformed output,
cancellation and retry without duplicate side effects or repeated retry loops.
Keep optional-companion failure from blocking otherwise usable main serving.
Do not silently skip a required review or route local work to a cloud service.

Compare the same workflow with a separate companion against an equivalent
baseline that still performs all required reviews on the main model. Measure
main latency, review wait, worker success, rework/retries and total time to a
correct result. A slower main decode can still be acceptable if the complete
workflow is reliably faster and foreground bounds hold. More processes or more
GPU occupancy alone are not success. Do not require arbitrary extra review
calls merely to keep the companion busy.

### 4. Spend remaining capacity on demonstrated bottlenecks

Use the resolved baseline and its allocation evidence to generate complete,
legal alternatives. Reuse the existing candidate controller. Automatic launch
must remain bounded; avoid a new exhaustive benchmark or repeated giant-model
reload loop.

Preserve context/quality while comparing expert packing, batch/ubatch, topology
and useful slots. For slot tests, keep client demand, task inputs and guaranteed
per-agent context equivalent. Charge the extra KV and displaced experts. Extra
slots may improve throughput even with some displacement; neither displacement
nor model-size/VRAM ratio is an automatic rejection rule.

Keep hot experts as a concrete hypothesis:

- Verify cache support and the effective backend graph, not just emitted flags.
- Measure per-layer routed hit coverage, miss behavior, CPU expert time, GPU
  expert time, cache uploads/evictions and synchronization where available.
  Label missing counters unknown. Collect only evidence needed to decide the
  next candidate; do not start an unrelated telemetry project.
- Check output/correctness, prefill bypass, multi-token/speculative paths and
  multi-device ownership for the exact supported backend/model layout.
- Compare cache off/on at fixed context, KV placement/quality, workload and
  concurrency. Also account for the opportunity cost of replacing ordinary
  resident expert layers with cache capacity in the same total memory budget.
- GPU KV is the initial comparison where feasible. CPU KV is a separately
  labeled whole-plan experiment. A failed whole-plan test does not reject all
  cache implementations or prove that no other placement can help.
- Promote only a repeated, material agent-workflow gain with phase and
  correctness guards. Otherwise keep the functioning baseline and persist the
  scoped negative/inconclusive result to avoid repeatedly testing it.

### 5. Close the release loop

For core changes run `scripts/verify-core-engine.sh` uncached plus the repository
checks relevant to the final changes. Reuse the existing workload harness and
record its limits; the three simple repair tasks are a smoke test, not the
entire agentic acceptance suite.

Use validation depth appropriate to the decision:

- **During recovery debugging:** keep the small three-repair smoke suite. It
  checks basic tool flow quickly. Reproduce the failure and verify the fixed
  model plus regression models on the exact candidate. Do not interrupt an
  unresolved launch failure with a broad performance sweep.
- **Before changing automatic performance policy:** run realistic multi-turn
  repository work with a substantial reusable project prefix, appended tool
  output, branch/replay, and objectively checked results. Record actual prompt
  tokens; configuring a 128k or 500k context does not exercise that much
  attention or runtime memory. Choose a representative normal session size and
  a separate long-context case based on intended use, rather than testing every
  possible size. Include sustained generation as well as short tool responses.
- **Before claiming the full local-agent experience works:** exercise the real
  client/router, required reviewer traffic and explicitly delegated worker work.
  Check responsiveness under fan-out, companion failure/fallback, cancellation,
  context pressure and clean restart. A direct call to the model backend does
  not validate these integration paths.

Reuse the existing phase benchmark for cold prefill, cached append, decode and
mixed traffic. Its inspected default corpus is approximately 8k tokens and its
adaptive screen can be shorter; record the actual size rather than calling it
long-context acceptance. It complements the functional tasks. Add a bounded
long-context canary where needed instead of building another benchmark system.
For hot-expert decisions, use multiple task types and distinguish warmup from
steady state so repeated tiny fixtures do not define the routing locality.
Keep A/B demand and cache conditions matched, use repeated measurements and
report dispersion. Treat small differences as inconclusive when the sample
cannot resolve them. Run deeper validation at a stable candidate, not after
every small edit.

Validate Linux CPU/real GPU and Windows installation/release paths on the exact
candidate commit. Include a resident model, a moderately CPU-offloaded MoE and
the GLM tight-fit regression using this machine when available. Exercise
long-context/cache behavior and companion fallback. Mark unavailable hardware
coverage as untested. Public generic claims need an expanded hardware/model
matrix; this box alone cannot certify them.

Record model/quant/shards, backend and ggrun identities, effective argv, resource
ledger, client settings, workload hash, cache/warmup state, repeats, correctness,
phase timings, task latencies and clean relaunch. Use matched repeat evidence
before promotion and stop inside a defined experiment budget when inconclusive.
Do not count main-branch CI as proof that an older release contains the fixes.

Preserve the canonical PATH installation convention and unrelated live servers.
Keep durable live evidence in the existing core TODO/theory ledger. Do not merge
unrelated branches simply to reduce branch count. Ship truthful, concise install
and feature documentation with the validated release.

## Progress record

Updated 2026-09-15. Detailed measurements live in
`docs/core-standard-launch-todos.md`; this section carries state only.

### The goal's premise is falsified on this model class — decision needed

The standing goal reads "reach maximum hardware usage ... which should result in
fastest possible serving". On GLM-5.3-Flash (137.4 GiB on 48 GiB of VRAM) those
two clauses point in **opposite** directions, and that is now measured seven
independent ways:

| measurement | more hardware used | result |
|---|---|---|
| `HOSTBOUND` fill sweep | 0.408 -> 0.763 -> 0.600 | 1.074 -> 1.03 -> 0.907 tasks/min |
| `VRAMFILL` reserve spend | +2,988 MiB resident | decode 6.83 -> 6.83, no change |
| `HOTEXPERTS` cache K=32 | 0.408 -> 0.703 | tie at 1 slot, -7.9% at 2 |
| hot cache 8 slots/layer (2026-09-15) | +2,868 MiB on CUDA1 | **-57% decode** |
| `--parallel 2`, 4 lanes | more slots | **-40%** |
| `--n-cpu-moe` 42 -> 39 | +3 expert layers on GPU | **-12 to -16%** |
| arm B, context traded for residency | *less* VRAM, 28,782 vs 32,954 | -23% (unfair arm, retracted) |

**The fastest arrangement measured on this model is the one using the least
VRAM.** `fraction_of_vram` is anti-correlated with speed here, because the single
contended resource is host/PCIe bandwidth feeding 39-42 CPU-resident expert
layers per token, and every lever either spends that resource to save it or adds
synchronisation to a path already waiting on the CPU.

So "maximum hardware usage" is not a route to "fastest serving" for a model 3x
over VRAM capacity — it is a route away from it. README already states the
correct target: *"the fastest **stable** plan for the requested workload, not
maximum VRAM fill."*

### Tested on the other model class too — the same result

The obvious objection to the table above is that GLM is 3x over VRAM, so of
course hardware use cannot buy speed. `Qwen3.8-27B-UD-Q4_K_XL` (16.7 GiB) fits
entirely in 48 GiB with room to spare, which is the case where both clauses
should agree. Measured 2026-09-15, same harness, same token counts, two
concurrent clients, one variable:

| arm | VRAM used | GPUs engaged | per-stream | aggregate |
|---|---:|---:|---:|---:|
| ggrun auto (`--parallel 1`) | 29,950 MiB | **2 of 3** (CUDA2 at 115 MiB) | 35.5 tok/s | **24.8 tok/s** |
| forced `--parallel 2` | 44,062 MiB | 3 of 3 | 21.2 tok/s | **21.0 tok/s** |

**47% more VRAM, all three GPUs engaged, 15% slower.** Per-stream decode fell
from 35.5 to 21.2 tok/s. Single-stream baseline was 35.56 tok/s at 0.18% spread,
and two concurrent clients on one slot finished the same 1,536 tokens in 61.9s
against 63.9s for one client — they serialise, and serialising still beat
splitting across three devices.

ggrun's automatic plan left an entire 12 GiB GPU idle **and that was the faster
choice.** The mechanism is the same one `HOSTBOUND` identified: adding a device
adds cross-device synchronisation to a path that was not waiting on capacity.

This does not prove one slot is right at every concurrency — two clients is the
realistic Claude Code shape (main plus an occasional subagent), and a 4- or
8-lane workload may trade differently. An older note recording "a second slot was
~29% faster" on this model predates the `HARNESSKILL` correction and disagrees
with this measurement; this one is clean and the older one should not be relied
on.

**So the premise fails on both model classes tested**, host-bound and resident
alike, and in both cases ggrun's automatic choice was at or near the best
measured. Eight measurements now point the same way.

**This needs the user's decision, not another tuning attempt.** Either:

1. **Keep the speed goal, drop the hardware-usage clause.** Then for GLM the
   honest answer is that ggrun is at or near the achievable maximum on this
   hardware, and `HOSTBOUND`'s recommendation stands: a smaller quantisation or
   more VRAM. The remaining engineering (`PLANOVER-7`, `PROBESTALL`) buys
   reliability and planning speed, not tokens per second.
2. **Keep both clauses.** Then the goal is unreachable as written on this model
   class, and it should say so rather than remain permanently open.
3. **Re-target to a model near the capacity boundary**, where residency gains
   move a large fraction of the offloaded bulk. That is the case `VRAMFILL`
   identifies as untested and where both clauses would agree.

Until that is decided, "the core objective is not met" is true but misleading: no
measured lever on this model would move it, and six attempts to find one have
each made things slower or left them unchanged.

### Session close 2026-09-15 — state of the core objective

**The core objective remains NOT MET.** The planner still optimises for fit. No
change this session moved a model closer to the speed its hardware allows, and
nothing here should be read as progress on that.

What the session did produce:

| item | state |
|---|---|
| `memory-probe` passed `nil` where launch passes its recovery ledger | **fixed** (`PROBECONVERGE`), gate green |
| probe search exhaustion was undiagnosable | **fixed** (`PROBECREEP-2`), per-round report, 11 tests |
| why the search stalls | **root cause found** (`PROBESTALL`), not yet fixed |
| `memory-probe -ctx fit` converges | **warm only**; cold still exhausts. `SPENDSWEEP-2` open |
| hot experts on GLM | **measured properly**: 57% decode loss at a budget that fits; reference budget cannot fit at all (`HOTSTARVED`) |
| preflight accepts unservable contexts | open, boundary 617,472-670,720 unresolved |
| launch-path non-convergence | open, cause still unexplained |

**The single most useful thing found.** `PROBESTALL`: the context search steps
usefully only in rounds where the deficit can be priced into tokens. Where
`contextReclaimTokens` yields nothing, `automaticContextCeiling` silently
degrades to a one-granule step — 1,024 tokens against a 9.3 GiB shortfall, on the
order of 500 rounds to converge. `automaticContextCeiling` is shared with the
launch path, and the `RESERVEFIX-REGRESSION` boundary sits inside the range this
search crawls through, so `PROBESTALL-3` and the preflight boundary are plausibly
one defect seen from two sides. **Start there.**

### Method warnings earned this session

Five claims were retracted after measurement contradicted them. The pattern is
worth more than the individual corrections:

1. **A ledger is not an outcome.** `RESERVEFIX` was reported as verified from the
   preflight ledger; the launch aborted at 96% of warm-up. Check serving.
2. **Arms must match in resource spend.** An A/B that lowered context "to buy
   expert residency" used 4.6 GiB *less* VRAM than its baseline, so it compared
   two machine sizes, not two allocations.
3. **A null change reads as a tie.** `--moe-expert-cache` silently self-disabled
   for want of VRAM; the arms were identical and would have been reported as "no
   effect". Verify the treatment was applied before trusting the measurement.
4. **Unit tests do not establish convergence.** `PROBECONVERGE` was committed as
   "cause found and fixed" on green tests; the live check failed.
5. **Search this ledger before designing an experiment.** `VRAMFILL` and
   `HOTEXPERTS` already answered the "spend idle VRAM on experts" question,
   fairly, the previous day. Re-running it worse cost GPU time and a retraction.

### Housekeeping

All 2026-09-15 commits are on `fix/residency-ratchet` in a worktree at
`/tmp/ggrun-reserve`. **`/tmp` does not survive a reboot** — relocate under
`~/ggrun-project/` or push the branch before relying on this evidence.

### Milestone 1 — observed launch blockers resolved; wider coverage remains

PR #59 and #60 merged; #58 merged as `49463f4`. The chronology below records
its investigation and final scope rather than current unmerged work.

**Both target models now launch.** The blocker was not the lever selector at
all; see "Resolution" below. Full evidence is in the core ledger under
`EXPERTPIN`.

| model | result | evidence |
|---|---|---|
| Qwen3.8-Flash-Next UD-Q3_K_XL | LOADED, then 3/3 oracle at 7.67 correct tasks/min | `EXPERTPIN` |
| GLM-5.3-Flash UD-Q3_K_XL | LOADED, context fit 498,688 tokens, 1 slot | `EXPERTPIN` |

The apparent tension recorded below was real but was a symptom. Two models
needed opposite things from the same context step:

| model | needs | with a deficit-sized context step | without it |
|---|---|---|---|
| Qwen3.8-Flash-Next UD-Q3_K_XL | a larger step | **launches** | fails, `did not reach a fixed point` |
| GLM-5.3-Flash UD-Q3_K_XL | the gentle descent | **fails**, no lever survives the leap | launches, converges to 500,736 |

Verified by bisect, not inference: `87275f1` (the branch before the oracle and
per-device work) already fails GLM, while merged main `67cbe7f` launches it. So
the regression came from the deficit-sized step itself — the commit added to fix
Qwen — and it was never re-checked against the working GLM case.

The step has been **reverted**. The ceiling is again one token below the
rejected context. The deficit-to-token conversion is retained for judging
whether a proposed drop covers the shortfall; it no longer forces one.

Every root cause proposed for that tension before the per-device ledger was read
turned out to be wrong; they are kept below only so they are not re-derived.

### Landed in #58 after scope reduction

- `placement.DeviceKVCacheMB` apportions KV by the layer split, so a deficit on
  one GPU is no longer judged against the aggregate across all devices, and
  host-resident KV yields zero GPU relief.
- A context drop must cover the deficit to win outright, retained as a
  preference with a legal fallback rather than a veto.
- The re-plan budget charges churn, not convergence.
- Regressions: per-device apportionment, host KV not credited to a GPU deficit,
  explicit context never recorded while its ubatch still is, competing deficits
  judged per device, unknown geometry leaving a fallback, and context bowing out
  when it cannot cover the deficit above a usable floor.
- Uncached `scripts/verify-core-engine.sh` green on all six packages.

### Verified on candidate builds

| build | case | result |
|---|---|---|
| `v3.2.9-dev.dad77d9` | Qwen3.8-Flash-Next | 3/3 oracle, 7.76 correct tasks/min, 14.6s median |
| `v3.2.9-dev.revert` | GLM-5.3-Flash | preflight converges to 500,736, model loads |
| `v3.2.9-dev.revert` | Qwen3.8-Flash-Next | **still blocked** |
| `v3.2.9-dev.lever2` | Qwen3.8-Flash-Next | still blocked; compute lever unreachable |
| `v3.2.9-dev.relief` | Qwen3.8-Flash-Next | still blocked; relief predicate too permissive |

### Corrections applied to earlier claims in the core ledger

1. "Resident expert fraction predicts agentic speed" compared three different
   architectures, quantisations and active-parameter counts. Confounded.
2. "Every token crosses PCIe for 42 of 43 expert layers" is wrong: CPU-resident
   experts compute on the host; activations transfer, not all weights.
3. "Measured 3.6% noise floor" rested on two runs of three tasks. Not a floor.
4. "DeviceSlackMB has no readers, so nothing consumes slack" inferred an absent
   capability from one unread field without tracing candidate/ledger paths.

### Resolution — the measured re-plan handed back proven expert relief

The per-device ledger from `/tmp/q5.log`, which this file had been asking for
under "compare complete per-device and host ledgers", shows it in three rows:

| round | CUDA2 need / limit | outcome |
|---|---:|---|
| 1 | 12,010 / 11,909 MiB | does not fit |
| after expert derate | 10,937 / 11,909 MiB | **fits** — 1,073 MiB freed |
| 2 (backend-measured re-plan) | 12,003 / 11,909 MiB | does not fit again |

The lever set was never the problem. The expert derate worked on the first
attempt and freed over a gigabyte. The backend-measured recompute then replanned
from the original request and put the displaced layer straight back. Because
that recompute yields a **different** argv from the one already rejected,
`recomputeDecision`'s identity ledger could not catch it, and the launch cycled
until the replan budget ran out.

`boundByProvenLimits` already pins context and ubatch across a recompute, but
`placement.Options` takes no expert-residency input, so `n-cpu-moe` cannot be
pinned on the way in. It is now guarded on the way out: `launchMemoryRecovery`
records the largest `NCPUMoE` an exact preflight has proven, and
`undoesProvenExpertRelief` vetoes a recompute that lowers it. The ratchet only
moves toward more CPU residency.

Regression: `TestMeasuredReplanCannotUndoProvenExpertRelief` covers the ratchet,
the no-accepted-plan case, re-proposing the accepted plan, moving further
experts to the CPU, and nil. Uncached `scripts/verify-core-engine.sh` green on
all six core packages.

The guard costs nothing at serve time. Qwen3.8-Flash-Next at the stable
candidate returns 3/3 oracle-passed, 7.67 correct tasks/min, 14.8s median,
against 7.49 and 7.76 on the same suite before the fix.

### Seven attempts, six reverted — what was eliminated

Recorded so none of it is repeated. All six reverted changes were reasoned from
inference about which branch executes; the one that worked came from reading the
ledger.

1. **Widen the context step to the measured deficit.** Fixes Qwen, breaks GLM
   (662,528 -> 236,544 in one round; the guards refuse the plans at that depth).
2. **Bound that step to a quarter of the window.** Converges close to main's
   500,736 but still fails GLM.
3. **Allow compute memory as a last-resort lever for oracle-planned deficits.**
   The branch is never reached; the earlier expert lever reports success first.
4. **Rank levers by `candidateRelievesFailedDevice`.** The predicate returns
   true for the weak derate, so the ranking never fires.
5. **Rank by `predictedDeviceReliefMB`,** a measured per-device relief estimate.
   Builds and gates cleanly; did not change the observed trace.
6. **Offer one bounded ubatch rung when the allocation is unmeasured.** Same.

Three diagnoses recorded as root causes were each overturned by
`preflight_recovery_qwen_test.go`, which drives the selector with the exact
failing Qwen argv and outcome:

- "The synthesised `AllocMB` disqualifies the ubatch lever" — the lever is
  available and steps 256 -> 64.
- "`n-cpu-moe` is stuck at 22" — it steps 22 -> 23, CUDA2 expert layers 4 -> 3.
- "The pins are partial sub-pins and are mis-priced" — the OT pattern includes
  `down`, so they are whole-layer.

That test exists so the next attempt starts from observation. It cost minutes
and would have saved most of the six reverted attempts.

Two design items from those attempts remain genuinely open, now decoupled from
the launch blocker:

1. `outcome.AllocMB` is synthesised from the deficit for oracle-planned
   shortfalls while `outcome.AllocMBMeasured` records the distinction and is not
   consulted at the sizing sites. A lever should never be sized by a fabricated
   allocation, even though that was not what blocked Qwen.
2. Relief is a boolean derived from argv differences rather than a measured
   quantity compared against the deficit.

### Milestone 2 — verify the ordinary local agent experience: partly verified

Evidence in the core ledger under `AGENTPATH`, all on `v3.2.9-dev.expertpin`.

| item | state |
|---|---|
| Clean temporary install, download, launch, generate, cleanup | covered by the install-e2e job on Linux and Windows |
| Readiness, generation, streaming, clean shutdown, port release | passed |
| **Cancellation mid-stream then reconnect** | **new check**; slot back in 0.607 s, no restart |
| **Reusable project prefix** | **new check**; 8.0% of a 6,433-token prefix re-evaluated |
| Bounded repository task with tool turns | passed, 2 lanes |
| TUI/CLI resolver agreement | **settled by construction plus three new regressions** |
| Effective per-agent context shown | formatted in code but not printed on the calibrated path |
| Automatic context policy left alone | unchanged; no 128k/32k floor invented |

TUI/CLI agreement needed no comparison harness. `cmdGUI` turns selections into
argv with `tuiLaunchArgs` and calls the same `cmdLaunch` the command line calls,
so the only way they can disagree is a selection that does not survive the argv
round trip. `tui_cli_agreement_test.go` now holds the round trip, the
preference-versus-instruction distinction that keeps the recovery ladder able to
move an untouched row, and a reflection guard that fails when a field is added
and never wired. The guard found `FlashAttn` dead: hardcoded true, never
emitted, never read.

The two new serving checks are in `scripts/verify-installed-serving.py`.
Cancellation runs everywhere; prefix reuse is behind `--prefix-reuse` because it
needs a context larger than the 2,048 the install CI jobs use.

Remaining for this milestone:

- The calibrated launch path returns before `printOptimizationSummary` when it
  reuses a cached decision, so `ctx N total / M per agent` is never printed. The
  numbers are still visible in the `[placement] context fit` line. Left
  unpatched because it is display-only code inside a protected path; it wants
  its own small change with the core gate rather than a drive-by edit.
- `--prefix-reuse` is not plumbed through `verify-gpu-install.py`, so the GPU CI
  job cannot request it.
- Not yet exercised: a long multi-turn session against a substantial repository,
  as opposed to the bounded three-task suite.

### Milestone 3 — worker/reviewer: source audit complete; integration benefit unverified

Read-only audit of `go/pkg/claudeauto` against the four role contracts in this
file. The required behaviours are already implemented; no second orchestration
system is needed, and none was added.

| required behaviour | where | state |
|---|---|---|
| Malformed reviewer output is not approval | `validReviewerVerdict` | holds — exactly one `<block>yes/no</block>`, or the stop-stripped form; thinking-only, tool-only, prose-mixed and *both* verdicts all fail |
| Reviewer failure does not take the review down | `installReviewerFallbackHooks`, `deferredWriter` | holds — transport errors and non-2xx are withheld, nothing reaches the client, the main model self-classifies |
| A required review is never silently skipped | `Router.ServeHTTP` classifier branch | holds — every fallback path routes to the main model and says so on stderr |
| Context overflow is handled | reviewer context window check | holds — an oversized review prompt goes to the main model |
| Foreground and safety work do not starve | `scheduler.go` lanes | holds — `LaneSafety` outranks interactive, which outranks bulk; fair share bounds the wait by *active conversations*, not queue depth |
| Companion budgeted with the main model | `launchRequest.ReviewerReservation` | attached to every `Compute`, including recovery re-plans |
| Cheap-tier work does not queue behind the main model | utility route | holds — `tryReviewerUtility` has the same withhold-and-retry safety without the verdict contract |

Two details worth keeping:

- A rejected reviewer answer is recorded as `reviewer-rejected/invalid-verdict`
  rather than `unusable-response`, and the answer itself is printed. Without
  that split a template mismatch is indistinguishable from an absent reviewer,
  and every review silently leaks to the main model while looking healthy.
- `conversationKey` once returned one key for every request, which made the
  scheduler's affinity ordering inert. It was replaced by fair share after a
  production run measured a foreground turn waiting a median of 35.4 minutes
  behind a fan-out, against roughly 109 s of actual compute per turn.

**The one thing missing is the comparison this file asks for**: the same
workflow with a separate companion against a baseline that still performs every
required review on the main model, measuring main latency, review wait, worker
success, rework and total time to a correct result. The instrumentation is
already there — `RequestRecord` carries `Route`, `QueueMS`, `TTFBMS`,
`TotalMS`, `Aborted` and cache-read usage, which is exactly that set — so this
is a measurement to run, not code to write. It needs a long two-arm session and
has not been run.

### Milestone 4 — one optimizer comparison completed; policy/generalization review open

First standard launch of Qwen3.8-Flash-Next with the candidate controller
enabled. Detail in the core ledger under `CALIBSPEND`.

The plan it settled on serves **262,144 tokens with 28 of 48 expert layers
resident at 89.5% VRAM**, and returns 7.63 correct tasks/min — inside the
7.49-7.76 band measured at 18,912 tokens. Fourteen times the context at the
same throughput on this suite is the result worth keeping, and it is exactly
what this file means by preserving useful context. The suite's tasks are short,
so it shows no detected short-task slowdown from the larger configured
capacity in this sample; long-context performance remains untested.

**The run exposed a defect in candidate generation.** All three candidates were
larger ubatch values; all three failed admission by 1,885 to 7,026 MiB; the
failure budget was exhausted and the search stopped. Placement had already
printed nine times that ubatch 512 yields no usable whole-layer MoE plan for
this model. Calibration does not consult that. The cost is not three wasted
reloads but that expert packing, topology and slot count are **never reached**
on this model class, which is why the baseline always wins on this shape.

Two bottlenecks are now named with measurements rather than inferred:

- prefill shows uneven activity, CUDA0 at 80% SM against CUDA2 at 7% on a
  0.27/0.59/0.14 split; topology is a hypothesis, not an isolated cause;
- decode is in the CPU-expert path with PCIe active but **not proven saturated**
  (52/34 MiB/s), so DRAM and synchronization stay live candidates.

Both changes are implemented and a comparison completed. The independent
review above identifies two remaining correctness gaps. Detail in `CALIBBUDGET`
and `CALIBLADDER`.

- **Budget accounting** (`CALIBBUDGET`): a refusal that reads no weights no
  longer charges the reload failure budget. Correct on its own terms, but the
  live trace showed it was **not** what blocked topology — the budget was
  reached exactly as the last of three challengers finished, so nothing had ever
  been skipped because of it.
- **Candidate selection** (`CALIBLADDER`): that was the blocker. Three challenger
  slots all went to one lever family. `selectAutomaticCalibrationAdmissionPlan`
  already stated the fix in a comment and implemented it only for `parallel-`;
  it now applies to every lever, with the family read from the generator's own
  naming.

The search then reached and measured a real challenger for the first time on
this model, and **the phase guard refused it**: `ubatch-512` was 1.505x better
on workload makespan (48.72 s to 32.37 s) with prefill up 57%, but decode fell
38% against a 5% allowance. `default` won. Invariant 6 holding on live data is
worth more than the speedup would have been — scoring the aggregate alone would
have selected a plan whose measured decode throughput was 38% lower in this
screen. That does not establish the outcome for every agent workflow.

The bottleneck moved with it. At ubatch 512 decode and mixed show high use of the
configured host workers (87% and 90%, measured), and the optimizer names the
next lever itself: physical-core count, affinity, and separate batch/decode
thread settings. No candidate moves those today. PCIe during cached append rose
roughly 78x and is still not proven saturated, so the current counters do not establish bus saturation or exclude
transfer/synchronization costs.

**Generality check — and it found a third defect** (`CALIBGENERAL`). Both fixes
were developed and verified on Flash-Next alone, which proves the bug is gone
there and nothing about whether the fix travels. Run against Qwen3.8-27B, a
fully resident model, the ladder broke: that frontier produces **27 candidates**
against Flash-Next's 5, and its finalist is a **compound name**,
`batch-1024-ubatch-512`. Grouping families by the leading token called it
`batch` and treated it as unrelated to `ubatch-512`, so a refused ubatch rung
would be followed by a candidate carrying the same ubatch. Families are now
parsed as key-value runs and collide on coordinate overlap.

The lesson is the one the contract already states. Verifying on the model that
exposed the bug is not evidence the fix generalises; every candidate-selection
change from here needs at least two residency classes before it is believed.

**The topology imbalance is the strongest unexploited signal, and it is not one
model's quirk**: Flash-Next prefill 80/7 and 79/8, its append 71/2, the 27B's
append 79/5, its challenger's prefill 98/0. Two models, two residency classes,
four launches — one card near saturation while another sits under 10%. No
candidate family moves it.

Also resolved from milestone 2: the `ctx N total / M per agent` line **does**
print on a fresh calibration (`[optimize] roomy-resident, ctx 262144 total /
262144 per agent, ...`). The gap is specific to the cached-decision path, as
recorded.

### Milestone 5 — close the release loop: partly closed, with coverage stated

What is validated, and on which commit. Nothing here counts main-branch CI as
proof that an older release contains these fixes.

| path | state | where |
|---|---|---|
| Uncached `scripts/verify-core-engine.sh`, six packages | green at every commit in #58 and #61 | local |
| Linux CPU install, download, generate, cancel, shutdown, port release | green | `install-e2e` linux |
| Windows install, reinstall preserving config, generate | green | `install-e2e` windows |
| Linux real-GPU serving | **green on the merged candidate** `0a66834` (run 34898139305: linux, windows, gpu all success) | `install-e2e` gpu |
| Windows GPU | **deferred by the user, 2026-09-15** — runner offline, `GGRUN_GPU_RUNNER_WINDOWS` false. To be tested another time; not blocking this milestone. | — |
| macOS | **untested**, lower priority for this work | — |

Model coverage on this machine, all on the installed single binary
(`/home/mik/go/bin/ggrun`, PATH symlink `~/.local/bin/ggrun`):

| class | model | state |
|---|---|---|
| resident | Qwen3.8-27B-UD-Q4_K_XL | served 262,144 tokens; cancellation 0.607 s; prefix reuse 8.0%; 3/3 agent tasks |
| moderately CPU-offloaded MoE | Qwen3.8-Flash-Next-UD-Q3_K_XL (1.75x over VRAM) | launches since `EXPERTPIN`; 262,144 tokens at 28/48 resident experts, 89.5% VRAM, 7.63 correct tasks/min |
| tight fit | GLM-5.3-Flash-UD-Q3_K_XL (2.86x over VRAM) | **re-run on the post-calibration binary**: 555,008 tokens, up from 498,688; ladder admitted `kv-alternate` between two ubatch rungs; no false promotion |

Recorded screen size, as the milestone asks rather than calling it long-context
acceptance: the bounded screen uses 23,296 bytes (~7,701 tokens) per lane, with
`reuse >=6503 tokens/lane` and a 48.48 s slowest workflow. The three-repair
suite is a smoke test; it does not exercise the 262k context the plan now
serves.

`docs/release-validation.md` now documents the two new serving checks and says
plainly which one is off in CI and why.

Remaining before this milestone is closed:

- ~~Dispatch the GPU job on the candidate commit~~ — done after #61 merged.
  Run 34898139305 on `0a66834`: `linux`, `windows` and the real-GPU `gpu` job
  all success; `gpu-windows` skipped, its runner being offline.
- ~~Re-run the GLM tight-fit regression~~ — done, 555,008 tokens, recorded above.
- Windows GPU and macOS stay marked untested. This box cannot certify generic
  public claims; that needs an expanded hardware and model matrix.

### Claude Code mode and the companion seat — measured, with one gap left

Full detail in `SEATCOST`, `CLAUDEMODE` and `SEATARMS`.

**Every measurement before this point used plain serving**, so no companion was
seated and none of it described the configuration an agent user runs. Claude
Code mode plans four slots at the same per-agent context, not one.

| | plain | `--claude-code` |
|---|---:|---:|
| plan | 262,144, 1 slot | ~1,046,528 total, 4 slots |
| per agent | 262,144 | 261,888 |
| resident experts | 28 of 48 | 19 of 48 |
| correct tasks/min | **7.63** | **2.42** |
| tasks completed | 3/3 | 2/3 |

**The four-slot plan, not the companion, is what costs the throughput.** All
three seats land together:

| seat | per-agent context | correct tasks/min |
|---|---:|---:|
| `off` self-classify | 261,888 | 2.44 |
| `qwen2b` review-only | 262,144 | 2.49 |
| `qwen` worker+reviewer | **211,968** | 2.27 |

Single runs, no established noise floor, so the ordering is not resolvable and
must not be read as one. The durable result is the capacity one: the 4B seat
costs 19% of per-agent context, the 2B seat costs none. **This table measures
the seat's cost with its review lane idle; see `REVIEWLANE` below for what
happens once reviews are actually issued.**

**The benefit side is now measured, and it reverses the reading** (`REVIEWLANE`).
Driving the lane — 8 classifier requests concurrent with 4 foreground turns —
separates the arms decisively:

| | `off` self-classify | `qwen2b` seated |
|---|---:|---:|
| routes served | `main: 13` | `reviewer: 8`, `main: 4` |
| reviews completed | **3 of 8** | **8 of 8** |
| review median | 14.17 s | **0.099 s** |
| foreground median | 16.58 s | **10.28 s** |
| errors | **6 x HTTP 502** | **0** |

With no seat every request lands on `main`, reviews queue behind foreground work
on the same four slots, and six of twelve fail outright. With the 2B seated the
reviews are served in ~100 ms and the foreground turns get faster as well.

**Provisional hypothesis, superseded by the 2026-09-15 direction review:** a
review-only companion may help. The saved comparison has startup/teardown
failures and does not validate review decisions, so it does not establish a
default policy or that self-classification collapses under normal agent work.

Still open: the 4B worker seat has never been driven with delegated utility
work, so its 19% context cost over the 2B has no measured benefit.

### Corrections to earlier entries in this record

1. "The failure budget blocked topology exploration" — it did not. The budget
   was reached exactly as the last of three challengers finished, so nothing was
   ever skipped by it. Candidate *selection* was the blocker.
2. "Three wasted reloads" — those candidates never loaded the model. Preflight
   refused each before a weight was read.
3. "No companion seat can launch" — all three launch. The non-convergence is
   intermittent and depends on starting residency, not on the configuration.
4. `SEATCOST`'s seat prices came from dry-run estimates (35 resident); the live
   plans landed at 19. The dry-run ranks the seats but does not size them.
5. The residency ratchet in PR #62 **has never executed**. It is gated and
   tested; it is not demonstrated to fix anything.

### RETRACTION — every Claude Code agent-suite number above is contaminated

Read this before acting on any throughput figure in this record. Detail in
`HARNESSKILL` and `PTYFIX`.

`--claude-code` starts the backend and then opens the Claude Code client. Driven
from a script with no TTY the client refuses to start, ggrun exits, and its
shutdown handler stops the backend **mid-suite**. Task-level evidence from the
four-slot arm: two tasks whose **oracle passed** came back as "Remote end closed
connection", and the third got "Connection refused". Those runs timed a
teardown, not a configuration.

Retracted:

- **"Claude Code mode costs two thirds of the throughput"** (2.42 against 7.63).
  The plain-serving side is sound; the Claude Code side is not.
- The three-seat comparison, 2.44 / 2.49 / 2.27. Treat as invalid, not merely
  inseparable.
- Slot-width throughput.

Still standing, because it is read from the launch plan rather than from
completed tasks:

- resident expert layers by slot width, **25 / 21 / 9** for 1 / 2 / 4 slots, and
  four slots also getting *less* per-agent context (207,360 against 261,632);
- `SEATCOST`'s seat prices and every `n-cpu-moe` trace;
- `REVIEWLANE`, which completed in seconds with zero errors and whose route
  counts come from the router's own metrics. **The recommendation to seat the
  review-only companion survives.**

**The measurement path is fixed.** Driving the launcher under a pty
(`script -qec`) keeps the client alive; verified alive 90 s past ready with the
client error absent, and the first clean run through Claude Code mode returned
3 of 3 tasks at 2.688 correct tasks/min. The retracted comparisons are now
runnable and should be re-run before any of those claims return.

### Blocked on disk, not on knowledge

The root filesystem is at 100% — 158 MiB free of 456 GiB. Measurements taken
under that pressure are untrustworthy: `ENOSPC` during a launch surfaces as
failures that resemble unrelated defects, which is the trap this session already
fell into twice. Everything below is runnable the moment there is headroom.

`~/2tb-disk` is a separate 1.9 TiB volume with **574 GiB free**; moving part of
`~/ggrun-project` (289 GiB, mostly models and `.src` build trees) there is
probably the cheapest fix, but it is a storage-layout decision for the user.

### Platform scope, settled 2026-09-15

**Linux is the main goal.** The user has scoped the platform matrix: Linux is
the target, **macOS is deprecated** and retired, and **Windows GPU is deferred**
to a later session. Neither of the latter blocks milestone 5.

Read this before planning work: effort belongs on the Linux path — CPU and real
GPU, install through serving — not on widening the matrix.

That closes the release-validation matrix for this work: Linux CPU and Linux
real-GPU are green on the merged candidate, Windows install/reinstall/generate
is green, Windows GPU is deferred by decision, and macOS is out of scope. A
future session should not re-open these as outstanding work.

### Re-measured after the disk was freed — what replaced the retraction

The pty fix made Claude Code mode measurable; the retracted comparisons were
then re-run properly. Every arm below completed 3 of 3, where the retracted runs
managed 0 to 2. Detail in `SLOTCLEAN`, `SEATCLEAN`, `GLMSEAT`, `MINICPM`.

**Slot widths** (Flash-Next, reviewer seated, per-agent context matched at
262,144):

| slots | resident experts | correct tasks/min | median |
|---:|---:|---:|---:|
| 1 | 23 of 48 | 4.02 | 30.8 s |
| 2 | 17 | 3.02 | 32.9 s |
| 4 | **1** | 3.41 | **26.5 s** |

**The four-slot penalty was the teardown, not the plan.** Four slots runs 18%
below one slot, not the 3x originally reported, and has the best median latency.
The ordering is non-monotone, so no slot width is promoted — but the catastrophic
penalty claim is positively contradicted, which makes PR #62's slot work an
optimizer completeness fix rather than a fix for a known performance bug.

This also **qualifies RESIDENCYFRACTION**: 23x fewer resident experts costs under
a fifth of throughput at matched per-agent context. Residency dominates across
models far more than within one.

**Seats** (same model and suite): 2.84 / 2.81 / 2.76 correct tasks/min for
`off` / `qwen2b` / `qwen` — a 2.8% spread, indistinguishable, and not a ranking.
The durable difference is capacity: the 4B seat costs 21% of per-agent context,
the 2B costs none. `REVIEWLANE` still answers the other question, and the
recommendation to seat `qwen2b` stands.

**Hardest real configuration works.** GLM-5.3-Flash (2.86x over VRAM) with a
reviewer seated plans, loads and serves: 809,984 tokens across 4 slots,
converging in one derate round.

**MiniCPM5-2B rejected.** 0 of 4 valid verdicts, twice, including with the
router's stop sequence. Fast but always prefaces with prose containing both
tags. Not wired; no `ModelSpec` added.

**Residency ratchet**: mechanism verified by test
(`re-packed n-cpu-moe 23 -> 25`); the live oscillation has not recurred in nine
launches across three configurations and two models. Correct guard, rare
trigger, and instrumented since `RATCHETOBS` to report its attempts.

### Milestone 5 coverage, final for this session

| path | state |
|---|---|
| Uncached core gate, six packages | green at every commit |
| Linux CPU install / download / generate / cancel / shutdown | green |
| Windows install / reinstall / generate | green |
| **Linux real GPU on the merged candidate** | **green** — run 34898139305 on `0a66834` |
| Windows GPU | **untested** — runner offline, `GGRUN_GPU_RUNNER_WINDOWS` false |
| macOS | **deprecated, 2026-09-15** — the user has retired this platform from the matrix. Do not spend effort here. |
| Resident / offloaded-MoE / tight-fit models | all three launched and served |

Windows GPU and macOS cannot be closed from this machine. They are reported
untested rather than inferred from the Linux result.

### Known limitations of current evidence

- Single runs per configuration; no matched repeats.
- Performance evidence is from one machine with three NVIDIA devices. CPU-only
  install/cancellation checks exist; equivalent CPU-only or single-GPU
  performance/generalization proof is still absent.
- The three-repair suite is a smoke test, not agentic acceptance, and was used
  only at stable candidates per this file's validation-depth guidance.

## How to continue from this file

Treat the product goal as durable and the dated observations as a snapshot.
Refresh the repository, active work, installed binary and CI before selecting
an unfinished milestone. Continue existing good work rather than restarting it.
Keep changes small enough to review and use the acceptance checks above to
establish completion.

Maintain a concise progress record here: completed milestones with commit and
evidence references, the next concrete deliverable, and remaining limitations.
Keep detailed live measurements in the existing core TODO/theory ledger and
link them here. Update stale PR references instead of repeating old findings as
current defects. Report unsupported or unavailable coverage explicitly.

## Short goal prompt

Updated 2026-09-20. Use the current review above, not an older utilization-based
diagnosis, to choose the next bounded task. Existing comparisons show that
increasing VRAM occupancy alone is insufficient; they do not prove a universal
inverse relationship between useful hardware activity and agent speed.

```text
/goal Read the current review in /home/mik/ggrun-project/ggrun/docs/claude-local-agentic-goal-handoff.md. Make default auto deliver the best validated correct agent-work performance for the chosen model and available hardware, especially large offloaded MoE models. Preserve useful context, quality and reliable serving. Follow the next bounded task and core change contract; distinguish measurements from hypotheses and validate across model/hardware classes before promotion. Keep the handoff current and development evidence local. Do not disturb the live server before the agreed testing window.
```

### Method rules this project paid for

1. **State the VRAM of both arms before calling anything a hardware-usage
   test.** An arm that used 4,564 MiB *less* than baseline was reported as
   "spending VRAM differently" and had to be retracted. Three of eight cited
   arms turned out not to raise VRAM at all.
2. **Use `/health` for backend readiness**, not a late log string. Establish
   usable agent serving separately with completed requests, tool turns and
   lifecycle checks; a healthy endpoint alone is not agent-work acceptance.
3. **Check which side of a boundary a number comes from.** sysfs
   `max_link_speed` is the *card's* ceiling; `pcie.link.gen.hostmax` is the
   *board's*. Reading the first produced a bogus "ggrun underestimates PCIe by
   2x" defect claim, retracted the same turn by one `nvidia-smi` call.
4. **Search this ledger before designing an experiment.** One re-run of an
   already-answered question cost GPU hours and produced only a retraction.
5. **A predicted improvement is not promotion evidence**, and neither is a
   fix committed before its live check. `PROBECONVERGE` was committed as
   "cause found and fixed" and then failed live.
6. **Record the machine's background load with every absolute number.** Every
   figure from 2026-09-14/15 was taken with a process holding 5 of the 14
   pinned cores; loadavg fell 5.14 -> 0.52 when it was stopped.
