# Core standard-launch optimizer TODOs

> Active core tracker for ordinary `ggrun` launches from either the TUI or
> direct CLI. This is the GGUF/llama.cpp lane; it does not depend on FreeToken,
> AI Tune, or multi-model orchestration.

Status: core hardening verified and installed, 2026-08-29; live hardware
acceptance remains active.

The theory, evidence schema, implementation map, known risks, and exact resume
sequence are preserved in [optimizer-theory.md](optimizer-theory.md). Treat it
as the handoff document for this tracker.

The current parallel-2 Qwen3.8 mixed-workload measurements are preserved in
[live-evidence-2026-08-29-qwen38-p2.md](live-evidence-2026-08-29-qwen38-p2.md).
They show that allocated slots and simultaneously active compute lanes must be
separate scheduler decisions.

## Implementation snapshot

The preserved 2026-08-28 checkpoint contains the first core implementation plus a
correction that restores the fit-first baseline, removes topology-name
priority, and separates backbone, GPU-expert, CPU-expert, and activation costs.
It also binds complete guarded peaks to an exact placement identity, preserves
unlabelled cgroup/device bytes, and versions qwen4exp tensor accounting so a
large host-only PLE table is neither charged to GPU backbone nor required when
the GGUF legitimately omits it. Phase-tagged GPU/process evidence and an
optional non-perturbing queue schema now feed a conservative typed bottleneck
diagnosis, which may prioritize one complete finalist but cannot bypass exact
admission or the live A/B gate. That checkpoint passed 971 focused tests and
1,435 full/race tests, with build, vet, formatting, shell/Python suites,
ShellCheck, and three supported cross-builds clean. The 2026-08-29 hardening
described below passed the protected core-engine gate plus the full normal and
race-enabled Go suites. Its canonical binary is installed at
`/home/mik/go/bin/ggrun` (also reached by `/home/mik/.local/bin/ggrun`). The
install handoff verifies both paths against the just-built artifact rather than
persisting a self-invalidating hash in this VCS-stamped source tree. An
already-running controller retains its older mapped executable until an
intentional relaunch. Checkboxes remain open under the tracking rule at the end
of this document until claims that depend on real hardware have preserved live
evidence.

| Work | Implementation state | Remaining acceptance |
|---|---|---|
| KVFIT-1 through KVFIT-7 | Implemented in the shared placement-backed automatic-context resolver, with exact adjacent-boundary and scope evidence | Commit plus live long-context/canary proof |
| KVFIT-8 | Synthetic coverage exists for selected/current-free GPUs, companion reservations, CPU-only, dense/MoE/recurrent shapes, explicit quality/max, and odd KV geometry | Finish the named public matrix and no-fit-oracle fixtures |
| KVFIT-9 | Boundary evidence is persisted and verified-config reuse is scoped | Capture load, cache, agent, and adjacent-rejection evidence on real hardware |
| PERF-1, PERF-3, PERF-5, PERF-9, UX-3 | Implemented: standard launch owns bounded search; candidates are complete placements; scope includes policy/capabilities; batch pairs preserve explicit intent | Commit and multi-hardware proof |
| PERF-2, PERF-6, PERF-7 | Implemented as baseline plus one calculated finalist, with a measured prefill pilot, identical budget-scaled cold+append scenarios, two samples, concurrent generation, mixed foreground traffic, lifecycle gates, delayed promotion, and a reusable baseline-won result | Add longer branch/replay and long-context hardware acceptance; quantify noise on public hardware |
| PERF-10 | Automatic legal slot neighbors use complete re-placement and useful per-agent context. Agent-parallel declares at least two runnable turns; automatic challengers wider than declared demand are dominated and skipped, while explicit maintenance orders the p1/p2/p4/p8 curve. Phase-aware router admission now separates allocated slots from active compute: long cold host-expert prefills serialize, while bounded small/cache-hot requests may overlap only after the first generated SSE delta. Reviewer stop-sequence-contract handling and enforceable Workflow-timeout fixes remove the observed false fallback/deadline paths; ambiguous Workflow source inputs fail early, materialized scripts are verified, and queue/service cancellations plus 60s/600s signatures are recorded. | Relaunch the new binary and complete controlled p1/p2 decode+decode, cold-prefill+decode, and cache-hot A/Bs plus capability-specific unified/partitioned KV A/B; tune the cold/append boundary only from matched public-hardware evidence. |
| PERF-4, PERF-8, PERF-12, UX-2 | Not complete | Implement measured headroom continuation, broader optional knobs, and bounded control UX |
| PERF-11 | Partially implemented in the working tree: MoE topology candidates include each feasible sole-backbone owner as a performance-only full recompute; ranking prices the serial backbone and routed GPU/CPU experts instead of prioritizing owner names | Finish exact-argv guard tests, run the intentional roomy hardware comparison, then add only capability-proven row/peer candidates |
| UX-1 | Implemented for launch, dry-run, dry-run JSON, TUI config screen, and `ggrun status` | Support-expert status remains the NanoBeige controller; launch inspect is `ggrun status` |
| ROOMY-1, ROOMY-2, ROOMY-3, ROOMY-4, ROOMY-6 | Implemented in source: exact residual slack classifies roomy; tight live-tests only the proven shape; topology ranking prefers a fitting fastest single GPU; batch/ubatch/slots are full recomputes; winner/baseline-won/boundary persist | Commit plus live roomy dense/MoE/recurrent proof |
| ROOMY-5 | Implemented in source: PCI-keyed SM plus NVIDIA PCIe RX/TX and Linux process-tree CPU/RSS/I/O sampling span each separate agent phase. Link saturation is claimed only against a known measured/detected ceiling; low traffic leaves DDR/synchronization unresolved. Only imbalance between ordinary-layer owners is actionable, and telemetry cannot select a predicted-slower topology. | Capture a matched live DeepSeek-class baseline/finalist comparison and verify phase transfer samples against the external `nvidia-smi dmon` trace; add peer counters only if they change finalist selection. |
| Exact-argv admission and long-load UX | Implemented in the working tree: a recomputed argv must receive its own allocation evidence, guarded peaks carry a placement identity, known challenger rewrite/recovery paths fail closed, lateral MoE split churn retains the exact proven placement, and 64+ GiB models warn before loading | Commit, then repeat the live MoE case after the current server is intentionally stopped |
| MMAP-1, MMAP-2, MMAP-4 | Implemented: production/preflight/recovery/daemon share capability-aware reclaim policy; unknown/anonymous loaders fail closed; mmap remains last-resort and consent-gated | Commit plus live resident/mmap/anonymous cases |
| MMAP-3 | Host ledger now separates exact reclaimable expert bytes from non-reclaimable runtime, KV, embeddings, and checkpoint reserve | Audit remaining backend-reported buffers/page tables/companions against live cgroup data |
| MMAP-5 through MMAP-7 | Not complete | Requires the storage/workload and real too-large-model acceptance window |
| HOT-1 through HOT-6 | Planned, backend-gated resident-MoE performance lane. A 2026-08-28 draft llama.cpp implementation is measured on Qwen3.8-Flash-Next but is not upstream-ready | Isolated fork audit, exact VRAM accounting, correctness and same-workload A/B on this three-GPU Q3 launch; then architecture matrix |
| PIN/P3 | Deliberately not started | Blocked on appropriate hardware/model artifacts as specified below |

### 2026-08-30 phase-aware hardening update

- Calibration evidence is schema 19. Admission-only suppression now carries a
  non-empty typed failure class and exact reason; an unexplained unavailable
  estimate cannot become durable negative evidence.
- Host-offloaded multi-slot Claude serving uses first-generated-delta phase boundaries.
  Cold prompts above the conservative 8k-token estimate run alone; bounded
  small/cache-hot work can use another physical slot after active prefill ends.
- Router evidence separates queue/service cancellation and the observed ~60 s
  and ~600 s deadline families. Workflow calls that cannot be rewritten with
  the maximum safe `stallMs` fail before launching work instead of silently
  running under the private deadline.
- Workflow inputs with multiple script sources now fail before tool precedence
  can select the wrong program; patched script files are read back before use.
  The hook advises file references for large protocols rather than duplicating
  them into model prompts.
- Request JSONL keeps transport `ttfb_ms` and adds `decode_start_ms`; streaming
  prefill/decode summaries use the first generated delta rather than the early
  Anthropic SSE envelope. Claude Code's stage-1 request intentionally stops on
  `</block>` and accepts an optional closing tag; the reviewer buffers and
  accepts only those exact, unambiguous stop-stripped yes/no verdicts unchanged,
  with a distinct `reviewer/stop-stripped-verdict` route label.
- NVIDIA phase samples include PCIe RX/TX. A measured saturation claim requires
  at least 65% of a known link ceiling; otherwise DDR, PCIe, and synchronization
  remain a composite diagnosis.
- Source and protected-core acceptance are complete. The next intentional
  relaunch must preserve matched cold+decode, append+decode, and decode+decode
  p1/p2 evidence before the threshold or policy is generalized further.

### 2026-08-29 hardening update

- Automatic calibration evidence is schema 18. Only typed deterministic exact
  admission failures may suppress the identical calculated retry; timeouts,
  incomplete benchmarks, and generic health failures remain retryable.
- Memory recovery is deficit-aware. It first targets the failed device, skips
  disproven ubatch rungs, and may reduce an automatic context only after the
  non-context levers are exhausted. Every context change re-enters the complete
  placement engine and rebuilds the full argv; explicit context stays fixed.
- Recurrent checkpoint memory is reserved per checkpoint and per slot even
  without SWA geometry. The first canary's backend-reported checkpoint size is
  charged before cgroup tightening, and a host cgroup OOM invalidates placement,
  calibration, lifecycle, and verified-config evidence as a typed failure.
- Reviewer profiles now carry model-appropriate contexts and exact measurement
  keys. Cheap-tier and safety calls use the seated review-only model rather than
  queueing behind main-model work.
- Parallel policy now separates capacity from demand. The agent-parallel
  workload represents at least two runnable turns even when the inherited
  server default is p1. Static context/memory/wave math removes impossible or
  dominated widths; p2 still needs one identical-workload A/B before a
  host-offloaded live router admits two requests. A measured/cached winner is
  scoped by model, backend, hardware, context, workload, and all coupled knobs.
- Hot-expert research now has two separate products: a dynamic VRAM expert
  cache for the common resident CPU-expert decode bottleneck, and selective
  mmap/mlock pinning for models beyond usable RAM. The former is described in
  HOT below; neither is silently enabled from a theory-only estimate.

## Live evidence: stable serving versus proven optimization

### Live Qwen3.8-Flash-Next qwen4exp cache review, 2026-08-27 22:47–23:25 UTC

The qwen4exp launch that failed placement on the afternoon of 2026-08-27 is now
the live server (after the host-resident PLE parser fix): pid 901203, started
~20:17 UTC, up 3h+; ggrun controller pid 888290 on `127.0.0.1:33645`; scope log
`.logs/ggrun-claude-server-v2-8081-670015b98e88d0be37009e66.log`. Serving argv
matches the afternoon dry-run: ctx 262144, one slot, `-b 2048 -ub 256`, K/V
`q5_1`, split `0.29,0.61,0.10`, 22 GPU expert layers (0–11 CUDA1, 12–17 CUDA0,
18–21 CUDA2) / 26 CPU, no mmap, CRAM 13824, 16 checkpoints at min spacing 512,
and `--kv-offload`. This already-running argv predates the new physical-core
affinity path; subsequent CPU-expert launches emit only exactly advertised
range/strict flags and only for a contiguous Linux CPU set allowed to the
process. KV buffers at load: 2304 + 864 MiB (q5_1, 24 layers x
262144 cells total across two shards of 12). A separate reviewer server
(Qwen3.5-2B Q4_K_M, `.bin/llama-server-cuda`, `:36539`, CUDA0) shares the host.

Cache review findings (server metrics + slot snapshot + log counters):

- **Prompt cache reuse is the headline: 91.3% cumulative.** At 23:21 the
  server had processed 246,134 uncached prompt tokens and reused 2,253,240
  cached tokens. A second sample at 23:24 shows 258,996 / 2,713,300 — a
  ~3-minute window of 12,862 prompt tokens at ~138 tok/s uncached ingest while
  reusing 460,060 cached tokens.
- **Current turn (task 79447): 94.4% cache hit.** Slot snapshot: 65,193 prompt
  tokens, 61,528 cached, only 1,906 processed. The turn is a long-context
  Claude Code agent turn (~138k) with per-step appends hitting
  `memory_seq_rm [n, end)`.
- **Context checkpoints are working, not free.** Log counts 111 created /
  68 erased-invalidated checkpoint events; each checkpoint is ~112.571 MiB, so
  16 slots cost ~1.8 GiB inside the CRAM budget. The `--kv-offload` +
  checkpoint pattern (PR #15293 machinery) restores mid-turn branches without
  full re-prefill; invalidated checkpoints are erased on divergence, which is
  the expected strict-append behavior.
- **Prompt cache (CRAM 13824 MiB) holds prior prompts** with their own
  checkpoint sets (~342 MiB for small prompts, ~524 MiB at 6.1k tokens).
  Multi-turn agent work keeps prior conversations restorable.
- **Decode improved as context got shorter: tg_3s ~11.9 tok/s** on the live
  task (vs 8.9 tok/s at ~140k context in the 22:47 sample), consistent with
  the CPU-expert/DRAM-bandwidth diagnosis — decode cost scales with live KV
  length, not just fixed expert work. Prefill sample: 5,720 tokens at 170.4
  tok/s; short-turn prefill 37–50 tok/s (small n dominates latency floor).
- **Aggregate counters**: 65,536 predicted tokens in 6,707 s (9.77 tok/s mean
  over the whole 3h window), 246,134 prompt tokens in 1,972 s (124.8 tok/s
  mean). GPU SM 10/19/8% (4070/3090 Ti/3060), 3090 Ti at 159 W; llama ~544%
  CPU (~14 threads of 14 pinned), load ~7.9, host 134 GiB available.
- The ~1 MiB `/metrics` spec_decode counters remain zero (no draft model).

Cache config verdict: the q5_1 K/V + kv-offload + 16-checkpoint + CRAM stack
is behaving as designed under Claude Code agent load — 9 in 10 prompt tokens
never re-hit the CPU/GPU prefill path. The remaining wall-clock cost is the
same 26-CPU-expert DRAM bottleneck recorded at 22:47; cache machinery is not
the bottleneck.

Follow-up shipped from this review: the Claude Code status line showed prefill
tok/s but no decode rate, because the monitor took decode only from the
`/metrics` gauge (a scheduler task that times out on the monitor's 3 s budget
exactly while decode owns the scheduler) and had no decode log parser. The
monitor now parses the backend's per-task `tg_3s` windowed-rate log line,
prefers it over the whole-run `/metrics` average, fetches `/metrics`
with one short cancellable attempt and an independent two-minute backoff, and
merges both rates in the passive path (CHANGELOG "Unreleased"; tests cover
decode capture, cancellation, backoff, and exact queue accounting). Installed
PATH `ggrun` sha256 prefix `e3250c7cfa4cedd3` (2026-08-28); the live
controller picks it up on its next launch.

### Live reviewer-lane error, 2026-08-27 23:3x UTC diagnosis

The live reviewer server itself (Qwen3.5-2B, `:36539`) is healthy: zero 5xx,
zero `send_error`, five tasks served. The user-visible error — Claude Code
permission-classifier timeouts blocking tool calls ("Stage 2 classifier
error", "auto mode cannot determine the safety of Bash") — comes from the
classifier lane having no dedicated slot:

- With a review-only reviewer (`ServesWorkers=false`), the utility lane falls
  through to the main model (`claudeauto.go` utilityEnabled = hasCompanion =
  false), so ~200 KB non-stream classifier calls queue behind the single slot
  (`--parallel 1`) while it serves 28-minute foreground streams: 45–80 s
  latency against an ~80 s client patience. Nine client aborts recorded in
  `.logs/ggrun-claude-requests-33645.jsonl`.
- The two mechanisms meant to keep classification on the reviewer both leak it
  back to main: (a) the overflow guard estimates tokens as bytes/3, which
  under-counts code-dense/escaped bodies — 18 historical 400 overflows in
  `ggrun-claude-reviewer-41197.log` (65,675–415,683 tokens vs 65,536 ctx);
  (b) the strict `<block>yes|no</block>` verdict matcher rejected all four
  real reviews today (only the engineered startup canary passed), so those
  went to main too (no reviewer row in the requests jsonl).
- ultra-zen fixed the same failure class on 2026-08-27 (commit `b414ea0`,
  "dedicated small-fast tier for the permission classifier"): route the
  cheap tier to its own model instead of the session model. The ggrun port is
  smaller — the 2B reviewer is already seated — and is: route utility/haiku
  requests to the reviewer backend when a separate reviewer exists and the
  request fits its context, plus loosening the bytes/3 estimate and fixing the
  verdict-template mismatch (prime suspect: the 2B running the
  `qwen3.8-27b.jinja` template file). `SetCompanion("local", …)` must stay
  review-only so Workflow worker sub-agents are not degraded to the 2B.

**Fix shipped 2026-08-28** (binary `4ae4df310c73f324`, symlinked at PATH):
direct probes of the live reviewer reproduced the verdict contract cleanly in
every shape tried (non-stream, stream, with tools: `<block>yes|no</block>` in
~50 ms at 0 temp), so the 20:21 rejections remain intermittent — likely
template/reasoning state on those specific prompts; the new
`reviewer-rejected/invalid-verdict` metrics rows will expose it on the next
run instead of guessing. Implemented now:

1. Utility lane → seated reviewer: cheap-tier (`local-fast`) requests route to
   the reviewer's own backend whenever a separate reviewer is seated and the
   prompt fits `claudeReviewerContextTokens`; overflow still falls back to
   main with the notice-once contract. The classifier lane keeps its existing
   reviewer-first behavior.
2. `estimatedPromptTokens` now errs high (bytes/2): the bytes/3 divisor
   demonstrably under-counted (65,675-token reviews scored under the 65,536
   window and 400'd 18 times historically).
3. Rejected verdicts are recorded (`reviewer-rejected/invalid-verdict` vs
   `reviewer-rejected/unusable-response`) so a verdict-format mismatch is
   visible in the request log instead of looking like reviewer downtime.

Evidence that the reviewer lane works when used: on 2026-08-28 09:12/09:15
two real 131 KB reviews went `route:reviewer`, 200, verdict accepted, ~1 s —
while same-conversation attempts routed to main aborted at 63–80 s. Full
`go test ./...` green; `TestUtilityLaneDisabledWithoutACompanion` and the
fallback tests updated to the new routing/rejection contract. The running
controller keeps its old binary until the next launch; Workflow-fan-out and
classifier timeouts should disappear from the next live session, and the
request log should show classifier traffic on `route:reviewer`.

**2026-08-29 follow-up:** the currently running reviewer and router are healthy,
but the controller was mapped from the older binary. Its request log still
shows utility/classifier calls on `route:main`, including repeated 60–70 s
client aborts behind the p1 foreground slot. That is old routing behavior, not
reviewer downtime. The source correction requires a newly installed binary and
an intentional controller relaunch; the active server must not be interrupted
merely to update this evidence.

### Live DeepSeek-V4-Flash Q3 XL, 2026-08-27 19:16–19:33 UTC

This is the Codex P1 inventory sample, not P1 acceptance.

TUI request 18:39:52: `ctx=fit`, explicit parallel 2, inherited bf16, Claude
Code, reviewer `qwen2b`. PATH ggrun from 18:00 (`81f2f37991a5f1dc`) started
`llama-server-cuda` at 19:16:12. Health OK after 12m27s. Scope
`100afe682825fc24426a7afe`. Log
`.logs/ggrun-claude-server-v2-8081-100afe682825fc24426a7afe.log`. Live probe
row `5404ea79f032.probe` (19:28:40) binds compute 2177/591/591 MiB to placement
hash `31edf8edc701b1eb5813b471231529d87beb7d4d8ac3a7baf128a8e1b1ad7fa1`.

Serving argv on `:8081`: ctx 987136 (two 493568-token slots), `-b 128 -ub 64`,
bf16 KV, split `0.26,0.64,0.10`, GPU experts 0–3 on CUDA1 / 4 on CUDA0 / 5 on
CUDA2, `--n-cpu-moe 37`, no mmap, CRAM 15360, 16 checkpoints. Reviewer 2B on
CUDA0 `:43071`. Backend model buffers: CUDA0 4308.71, CUDA1 14977.55, CUDA2
3559.56, host 99416.56 MiB. nvidia-smi while serving: 8491 / 20554 / 7157 MiB
used; CUDA0 100% SM, CUDA1 1%, CUDA2 0%. RSS ≈ 101.7 GiB. Host still had ~103
GiB available — roomy, not tight.

Eighteen backend timing pairs: 4786 prompt tokens at **16.35 tok/s**, 402 decode
tokens at **3.73 tok/s**. A 64-token decode concurrent with a 663-token prefill
ran **1.93 tok/s**. Solo 64-token decode was ~6.0 tok/s.

At 19:33:16 the controller tore down 8081 and launched a memguard probe on
`:45867` with `-b 8192 -ub 8192`, split `0.30,0.60,0.10`, `--n-cpu-moe 39`.
That probe was still loading at 19:35:30. It is not a promoted configuration.
No DeepSeek calibration JSON was written.

P1 still needs two identical agent-screen samples, exact-argv admission of one
finalist, and a persisted winner or baseline-won under schema 17.

### Earlier Qwen3.8 Flash Next run: safe baseline, optimization unresolved

That qwen4exp launch used 262,144 total context, one explicitly requested
slot, Q8 K/V, batch/ubatch `2048/256`, 21 GPU expert layers and 27 CPU-expert
layers, no mmap, a `0.29/0.61/0.10` layer split, 13,312 MiB CRAM, and 16
checkpoints. A separate 4B reviewer occupied about 4.25 GiB on the RTX 3060;
that card's apparent lack of free VRAM was therefore not all main-model state.

The server was healthy. At the latest sample it had processed 222,917 uncached
prompt tokens in 1,898.06 seconds (117.44 tok/s aggregate) and generated 28,234
tokens in 2,542.08 seconds (11.11 tok/s aggregate), with a maximum observed
sequence of 110,587 tokens. One completed 104k-context turn prefetched 12,129
tokens at 101.44 tok/s and decoded 17,626 tokens at 10.61 tok/s; the following
long-context turn was decoding around 9.9 tok/s. During its prompt phase the RTX
4070 reached about 79% SM while the 3090 Ti and 3060 were much lighter. During
decode all three fluctuate at low-to-moderate utilization. This is
phase-dependent sparse/CPU-offload behavior, not proof that parallel 2 or an
even split is faster.

The schema-15 optimizer record is not an optimality proof. It classified the
baseline as tight from an estimated ledger with GPU0 at -92 MiB, named
`ubatch-2048` as the sole finalist with low confidence, could not admit that
candidate, and restored the baseline. The guarded probe actually contained
exact aggregate peaks, but the backend labelled model bytes as
`unaccounted`; the old distribution matcher therefore reported zero exact
candidates. The source fix records a hash of every allocation-affecting
placement coordinate and trusts a matching guarded aggregate even when the
backend cannot label its model rows. It also carries the cgroup peak into the
same-launch host ledger and prevents a later KV-only observation from erasing
that proof. Calibration schema 17 retires the old settled claim and requires a
measured challenger outcome before performance evidence is reusable.

That Qwen answer remains: **stable and reasonable, but not yet the fastest
validated setting**. Parallel 1 was explicit, so the optimizer correctly did
not test parallel 2. The process is no longer live; the DeepSeek snapshot
above is the current hardware window.

### Earlier ~217 GB host DeepSeek-class run: roomy/performance evidence, 2026-08-26

That earlier 146 GiB-class Q3 XL launch is **not** representative tight-fit
evidence. The same checkpoint fits this server comfortably: its guarded probe
peaked at about 121.3 GiB of a 196.7 GiB host limit, and the live host still has
about 106 GiB available. Recovery to a known-safe argv does not change that
classification; recovery history and current residual capacity are separate
facts.

The serving plan has 1,048,576 total context, two 524,288-token slots, BF16 GPU
KV, `128/64`, seven GPU expert layers, 36 CPU-expert layers, and no mmap. Its
layer split is `0.23/0.67/0.09`: the 3090 Ti holds most bytes, but storage is not
the performance objective.

The first production argv changed an exact-probed `0.27/0.65/0.09` tensor split
to an unprobed `0.26/0.64/0.10` split. CUDA2 then failed a 617.26 MiB graph
allocation after 4m26s; health reported the failed attempt at 4m36s. The source
fix now binds allocation evidence to the exact argv, re-verifies genuinely
denser replans, and retains the proven split when only a lateral split changes.

The cache canary processed 6,347 cold tokens at 24.16 tok/s and restored 6,343
tokens on a strict append. The backend then explicitly forced full processing
for the older branch because recurrent/SWA state was unavailable, restoring
zero tokens; ggrun correctly left the profile degraded instead of promoting it.
The first real Claude request is a cold 76,433-token prompt at about 20.4 tok/s.
That is roughly 62–65 minutes of prefill. A live utilization sample makes the
immediate bottleneck more specific: the server holds about 95 GiB RSS but has
no disk reads or major faults, consumes only about one CPU core, saturates the
RTX 4070 at 94–100% SM, and leaves the 3090 Ti mostly at 0–6% and the 3060
mostly idle. Repeated live snapshots still show the RTX 4070 at 95–100% SM while
the 3090 Ti and RTX 3060 wait. The recovered `ubatch=64` may be conservative,
but the directly observed defect is serial/imbalanced layer ownership. The
source frontier now includes a fully recomputed sole-backbone-owner hypothesis
for every feasible GPU and scores sparse MoE topology by ordinary-layer role,
not just stored bytes. The running installed binary predates that change, so
this observation identifies the roomy performance miss; it does not validate
the fix.

The bottleneck is phase-dependent rather than one universal utilization
number. A later compute snapshot on 2026-08-27 showed the main process using
about 8.5 CPU cores while GPU SM was roughly 17% / 16% / 0%. With 36 CPU MoE
layers, that is consistent with a CPU-expert-limited phase. The calculated
frontier already prices CPU-expert bandwidth and prefers a feasible denser GPU
expert pack; the repeated live workflow still decides whether that, a larger
microbatch, more useful slots, or an owner topology actually wins. The current
source records phase-tagged process CPU/RSS/I/O, but not direct DRAM/PCIe/peer
traffic or a non-perturbing live queue counter. Those remain part of ROOMY-5
rather than being implied by GPU SM.

`parallel=2` is therefore not yet proven faster. It provides a second slot and
halves guaranteed context to 524,288 tokens per agent, but the observed request
uses one slot while the other is idle. A matched p1/p2 workload comparison must
wait for an intentional hardware window; do not reload the live model merely to
manufacture that number.

### Historical 128 GB host: fit/stability evidence, 2026-06-22

The earlier reference run is the representative constrained-fit case. On the
3090 Ti 24 GB + RTX 3060 12 GB + RTX 4070 12 GB + 128 GB RAM rig,
DeepSeek-V4-Flash UD-IQ4_XS (~128 GiB) ran at 1M context and parallel 4 with the
3090 Ti owning the dense path and the two smaller/slow-link cards storing three
complete expert layers each. It completed a 60,020-token request plus three
concurrent requests without OOM, restart, truncation, or health failure;
measured prefill was 29.05 tok/s for the main request and decode was 5.88 tok/s.
That preserved run validates the stable fit path under pressure. It must not be
relabelled as evidence that the current roomy host is optimized. Full details
remain in [launch-performance.md](launch-performance.md#deepseek-v4-long-context-service-stress).

## Product contract

The standard launch has two jobs:

1. **Fit:** run the requested model stably with useful agent context and the
   user's quality constraints on the hardware actually available.
2. **Go fast:** when it fits, use measured spare capacity to minimize real
   agent-workflow wall time instead of leaving safe performance unused.

These are phases of one core engine, not optional `ai-tune` or `--calibrate`
product modes. TUI and CLI must resolve to the same decision and evidence.

“Mid-size” is not a model-file threshold. The core must classify the *resolved
launch* from exact evidence:

- **Roomy resident:** the requested context, KV quality, companions, and model
  are resident with enough measured VRAM/RAM headroom to test at least one
  materially larger batch/ubatch or a faster legal placement. This is the next
  primary optimization path. ggrun should actively spend the headroom to
  minimize agent-workflow time.
- **Tight resident:** the launch is resident but one or more devices or host
  ledgers are near their verified boundary. Preserve the known-good shape and
  admit only monotonic, contained experiments; unused capacity on another GPU
  does not by itself authorize a riskier split.
- **Non-resident:** normal resident placement cannot satisfy the constraints;
  enter the mmap/offload recovery ladder visibly and optimize only inside the
  residency tier that actually survives.

Model parameter count, quant name, or summed nameplate VRAM must never choose
the class. The same checkpoint can be roomy on one machine and tight on
another, and companions or an explicit context can change the answer.

On a cold model/backend/hardware identity, “best” means the best safe estimate
available without delaying launch indefinitely. After bounded measurement, it
means the fastest **measured and fully validated** explored candidate. ggrun
must not claim a theoretical global optimum it has not measured.

The objective is lexicographic:

1. preserve correctness, explicit choices, minimum per-agent context, and
   required KV quality;
2. reject OOMs, hangs, cache corruption, unstable relaunches, and unacceptable
   foreground regressions;
3. minimize representative agent-turn time and workflow makespan;
4. use lower memory, disk I/O, and power as tie-breakers inside noise.

Raw prefill or decode speed is diagnostic, not the objective. Queue time, TTFT,
prefix reuse, tool-turn latency, foreground progress, and correct completed
tasks belong in the score.

## Residency and optimization ladder

```text
constraints + capabilities
          |
          v
stable placement baseline
          |
  exact allocation + residual headroom
          |
   +------+------+----------------+
   |             |                |
roomy resident  tight resident   no resident fit
   |             |                |
   v             v                v
spend headroom  preserve proven  ordinary file-backed mmap, only
on speed        shape; bounded   where CPU experts are reclaimable
                experiments
   |             |                |
   +------+------+----------------+
          |
validate and cache the winner
          |
          v
selective mmap + per-expert mlock only for a supported MoE
that still exceeds VRAM + usable RAM
```

The final box is experimental. It is not the backend's global `--mlock`:
locking an oversized whole mapping defeats reclaim and prevents fit. The idea
locks only workload-hot expert ranges and leaves the cold tail mmap-backed.

## What is already in place

- [x] TUI and direct CLI converge on the same launch/placement engine.
- [x] Hardware, GGUF, backend capabilities, companion reservations, and user
  flags feed placement.
- [x] Stable single/multi-GPU dense, MoE offload, dense CPU offload, and
  CPU-only strategies exist.
- [x] Exact allocation preflight, placement-bound guarded peaks, bounded OOM
  re-planning, runtime-growth evidence, functional/cache canaries, and
  verified-config reuse exist.
- [x] KV measurements reject known parallel and `swa-full` poisoning cases.
- [x] The first `ctx=fit` defects were fixed: Claude mode no longer turns fit
  into model maximum, and RAM is no longer counted as unbounded GPU KV space.
- [x] Ordinary mmap planning distinguishes resident CPU-expert memory from a
  reclaimable working set, has a consent gate, and has a reclaim-band design.
- [ ] The current core has not yet proven the fastest standard launch across
  ggrun's public hardware/model/backend matrix.
- [ ] KV fit and mmap recovery still need the end-to-end gates below.

## P0 — finish KV fit

`ctx=fit` must be a placement result, not aggregate-memory arithmetic bolted on
before placement.

- [ ] **KVFIT-1 — define fit.** Choose the largest useful automatic context in
  the fastest feasible residency tier under the selected model, KV quality,
  slots, backend, and user-granted devices. Never count RAM as GPU KV capacity
  or silently lower an explicit number/`ctx=max`.
- [ ] **KVFIT-2 — use effective inventory.** Size against current free memory
  on selected devices after backend/CUDA overhead, compute buffers,
  reviewer/draft companions, runtime growth, and other claims—not summed
  nameplate VRAM across every detected GPU.
- [ ] **KVFIT-3 — make it joint.** Recompute context, KV type/placement,
  `swa-full`, checkpoints, batch/ubatch, slots, speculation, tensor split, and
  expert placement as one candidate. No late overlay may invalidate the ledger.
- [ ] **KVFIT-4 — find the boundary.** Within each legal KV/residency class,
  retain the largest context granule exact admission accepts plus evidence for
  the adjacent rejected granule. Do not snap to powers of two.
- [ ] **KVFIT-5 — preserve semantics.** Forbidden KV types do not exist.
  Quality changes require explicit policy. Preserve user devices, parallel,
  and backend unless a visible recovery rule permits a change.
- [ ] **KVFIT-6 — unify entry points.** TUI, CLI, Claude defaults, dry-run,
  memory-probe, recovery, daemon/reload, and verified restore use one resolver
  and display one effective answer.
- [ ] **KVFIT-7 — scope evidence.** Key exact shards, backend build/capabilities,
  hardware policy, context, slots, KV/SWA, batch pair, speculation, companions,
  and planner schema. Reject evidence that cannot be normalized safely.
- [ ] **KVFIT-8 — regression matrix.** Cover resident and context-reduced dense,
  heterogeneous multi-GPU, CPU-only, MoE expert/KV competition, iSWA/recurrent,
  odd KV head dimensions, explicit max, occupied devices, and no fit oracle.
- [ ] **KVFIT-9 — live proof.** Record that the chosen context passes load and
  agent/cache canaries and that the next granule fails admission or crosses a
  visibly slower residency tier.

Exit: TUI, CLI, dry-run, and support agree; the chosen launch survives the
long-context gate; automatic fit never pushes a resident KV cache to host
silently.

## P1 — make resident standard launch converge on fastest

### Roomy resident fast path — next implementation priority

This is the common model-fits-easily case and the cleanest place to finish the
second product promise. It comes before more specialization for extreme
tight-fit models.

- [ ] **ROOMY-1 — evidence-based admission.** After exact allocation, compute
  residual VRAM/RAM and identify which batch/ubatch, KV, slot, and placement
  changes have real headroom. Do not use a fixed model-size or free-VRAM cutoff.
- [ ] **ROOMY-2 — fastest baseline.** Compare single fastest-device residency
  against multi-device placement when both fit; do not pay peer/scheduling
  overhead merely to fill every GPU. Keep context and quality fixed.
- [ ] **ROOMY-3 — spend graph headroom.** Raise ubatch and logical batch through
  complete placement recomputes until exact admission reaches the boundary;
  live-test only the best calculated finalist against the stable baseline.
- [ ] **ROOMY-4 — workload-owned slots.** Search 1/2/4 slots only when parallel
  was not explicit, preserve useful context per agent, and rank end-to-end
  workflow makespan plus foreground latency—not aggregate tok/s alone.
- [ ] **ROOMY-5 — observe device balance.** Sample per-device utilization,
  service time, CPU use, peer/host traffic, and queueing during the bounded
  workload. A full card that does no work is storage, not a successful speed
  plan. Feed that evidence into the next topology candidate.
- [ ] **ROOMY-6 — finite promotion.** Persist winner, baseline-won, rejected
  finalist, and the explored headroom boundary under the exact profile scope so
  later launches start immediately and do not repeat settled work.

Exit: on representative comfortably resident dense, MoE, and recurrent models,
the standard launch either promotes a faster validated configuration or records
that the stable baseline won. No explicit choice changes, and no run searches
again without a scope change, reset, or maintenance request.

### Controller and evidence

- [ ] **PERF-1 — core ownership.** Build a standard-launch optimizer shared by
  TUI/CLI. Reuse benchmark/calibration primitives without requiring legacy
  `--ai-tune`, `--tune`, or explicit `--calibrate`.
- [ ] **PERF-2 — agent workload.** Version a privacy-safe workload with cold
  long prefill, cached append, branch/replay, short tool turns, concurrent
  decode, and foreground traffic during fan-out.
- [ ] **PERF-3 — performance profile.** Key exact model/quant/shards, backend
  binary/build/capabilities, OS/driver, CPU/RAM/GPU/topology, selected devices,
  context/KV/SWA, slots/KV mode, batch pair, placement, speculation, workload,
  and explicit-intent bits.
- [ ] **PERF-4 — measured headroom.** Derive residual VRAM, RAM, compute, queue,
  foreground, and I/O headroom. Free VRAM permits a test; it does not prove a
  faster result.
- [ ] **PERF-5 — feasible candidates only.** Every candidate passes full
  placement and exact allocation/lifecycle admission. Do not patch argv after
  accounting.
- [ ] **PERF-6 — bounded search.** Screen neighbors of the stable baseline,
  fully run finalists, repeat the winner, and stop inside noise or budget.
- [ ] **PERF-7 — conservative promotion.** Require correctness, cache reuse,
  agent turns, foreground progress, long-context stability, clean relaunch, and
  exact identity. OOM/canary failure revokes the winner.
- [ ] **PERF-8 — reuse and continue.** Start immediately from a valid winner.
  Otherwise start the safe estimate and bound launch-time work; continue deeper
  search only in an explicit maintenance/idle window.

### Candidate dimensions

| Dimension | Gain | Cost/coupling | Treatment |
|---|---|---|---|
| logical batch `-b` | more prompt work per scheduler pass | long prefill may starve decode | agent fairness workload |
| microbatch `-ub` | occupancy and prompt speed | graph VRAM, MoE placement, checkpoints | full placement + preflight |
| `--parallel` | less queueing, more aggregate work | latency, KV/sequence state, per-agent context | measure makespan + foreground |
| unified/partitioned KV | dynamic slots or prefix sharing | independent prefixes may lose; semantics vary | capability-detect and A/B |
| device/layer/expert placement | faster resident compute | buffer entry fees and host traffic | exact device ledger/topology |
| CPU threads/affinity | faster host experts | contention/oversubscription | mixed-load measurement |
| flash/backend kernels | memory or attention speed | shape/correctness support | capability + canary gate |
| speculation | fewer target steps | draft memory and acceptance overhead | optional measured candidate |
| mmap/offload | makes an impossible model run | faults, disk bandwidth, tail latency | recovery, not resident tuning |

- [ ] **PERF-9 — coupled batch search.** Search legal `(batch, ubatch)` pairs,
  keep `ubatch <= batch`, recompute placement/checkpoints, protect explicit
  sides, and score prompt speed plus decode fairness.
- [ ] **PERF-10 — parallel/KV search.** Search workload-appropriate slots with
  real unified/partitioned semantics. Report total and guaranteed per-agent
  context.
- [ ] **PERF-11 — topology search.** Compare legal single/multi-device plans
  using measured bandwidth and compute buffers. No reference-rig GPU names or
  fixed VRAM thresholds in policy.
- [ ] **PERF-12 — optional knobs last.** Threads, affinity, flash, fork kernels,
  and speculation enter only after memory-shaping core is stable.

### User-visible behavior

- [ ] **UX-1 — explain it.** TUI, launch, dry-run, status, and support show
  baseline/winner state, context per agent, batch pair, slots/KV mode,
  placement/residency, evidence age, rejected finalists, and bottleneck.
- [ ] **UX-2 — bounded control.** Provide off/estimate/converge, finite budget,
  re-evaluation, and reset without making legacy commands the normal path.
- [ ] **UX-3 — explicit intent wins.** Model, quant, context, KV, devices,
  backend, batch, ubatch, parallel, mmap, and speculation remain constraints.
- [ ] **UX-4 — no routine consent noise.** Measurement inside an authorized
  residency tier does not repeatedly ask. Disk-backed residency, live probes,
  new devices, or quality relaxation stay explicit/fail-closed.

Exit: the promoted standard config beats the stable baseline outside noise on
agent wall time, preserves foreground/cache gates, survives relaunch, and is
reused without re-searching.

## HOT — dynamic VRAM expert cache for resident CPU-offloaded MoEs

This is a performance lane for the common case where the model and every expert
fit in VRAM+RAM, but some routed experts execute from host memory. It is not a
fit fallback and it is not static tensor placement. CPU decode repeatedly
streams routed expert weights through host DRAM; a backend-capable cache keeps
temporally reused expert slices in spare VRAM and computes cache hits on GPU.

The current reference is draft
[llama.cpp PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861),
opened 2026-08-28. It is unusually relevant because it measured
Qwen3.8-Flash-Next itself: a static top-32 list learned on half of a 54k-record
mixed corpus covered only about 10% of the other half, while per-layer temporal
LRU-64 simulated about 67% hits. Its decode-only cache measured 18.4 to 24.2
tok/s (+31%) with 48 slots per host-expert layer and about 4.1 GiB VRAM. Those
numbers establish a candidate, not a ggrun default: the PR is draft, currently
wires only a separate gate/up SiLU layout, and bypasses multi-token decode and
prefill.

- [ ] **HOT-1 — isolated capability.** Audit and pin a reviewed upstream commit
  in a separate backend fork. Detect the exact cache flags, CUDA/model layout,
  decode shape, and disabled-path identity. Unsupported models/backends produce
  no candidate.
- [ ] **HOT-2 — exact cache ledger.** Derive bytes per expert slot and per
  host-expert layer from GGUF/backend evidence. Reserve KV, graphs, checkpoints,
  prompt cache, companions, allocator growth, and device headroom before
  calculating any slot count; never copy the reference value 48 into policy.
- [ ] **HOT-3 — temporal evidence.** Record per-layer cache hits, misses,
  uploads, evictions, warmup, and drift separately for prefill/decode and p1/p2.
  Do not persist or promote a global static hot list merely because one corpus
  was skewed.
- [ ] **HOT-4 — one bounded A/B.** Compare cache-off with one calculated slot
  budget on identical cold-prefill, cached append, decode, mixed foreground,
  and workflow makespan. Require coherent output, no missing/double-counted
  expert contribution, clean relaunch, and material decode plus end-to-end gain
  without a prefill/cache regression.
- [ ] **HOT-5 — self-disable.** Cache allocation failure, unsupported graph,
  low hit rate, upload/synchronization regression, OOM, multi-token/speculative
  incompatibility, or correctness drift falls back to the exact stock argv.
  Failed evidence is scoped and finite.
- [ ] **HOT-6 — public generalization.** Validate this three-GPU
  Qwen3.8-Flash-Next Q3 case first, then separate gate/up, fused gate-up,
  heterogeneous CUDA, single GPU, p2, and at least one non-Qwen architecture.
  Keep the feature opt-in until the matrix proves safe automatic eligibility.

Exit: on an eligible resident CPU-expert MoE, ggrun either promotes a
profile-scoped cache that improves real agent workflow time outside noise or
records cache-off as the winner. Prefill remains on its separately optimized
batch/placement path until a backend exposes a proven prefill mechanism.

## P2 — harden ordinary mmap as final generic fit fallback

This is the existing file-backed CPU-expert path when non-reclaimable working
state fits but resident CPU expert bytes do not. It follows every normal
resident plan and precedes selective pinning.

- [ ] **MMAP-1 — re-audit current main.** Production, preflight, daemon, and
  recovery must share `memory.high` reclaim and whole-host `memory.max` rules.
- [ ] **MMAP-2 — prove pageability.** Gate on observed/backend capability that
  CPU experts remain file-backed, not only a backend name. Anonymous CUDA-host
  expert buffers are ineligible.
- [ ] **MMAP-3 — exact host ledger.** Count dense/shared/router tensors, CPU KV,
  checkpoints, graphs, pinned buffers, page tables, companions, page cache,
  and reclaimable experts exactly once.
- [ ] **MMAP-4 — last resort only.** Prefer a stable resident placement. First
  disk-backed transition is visible; unattended use needs explicit policy.
- [ ] **MMAP-5 — real workload tax.** Measure narrow/spread, cold/warm cache,
  long prefill, decode, and concurrent agents; record backend major faults,
  disk I/O/latency, reclaim, TTFT, and workflow wall time.
- [ ] **MMAP-6 — storage viability.** Detect measured storage behavior and warn
  or refuse a cold path that cannot meet the declared profile. No fixed mmap
  penalty assumption.
- [ ] **MMAP-7 — live acceptance.** Re-run the too-large-for-resident-RAM case
  that exposed the reclaim bug and an anonymous-expert refusal case.

Exit: accepted mmap survives spread traffic without cgroup OOM; anonymous
experts are refused; a resident winner is never silently replaced by mmap.

## P3 — selective mmap + mlock for models beyond RAM + VRAM

This imports the development-lab disk-MoE memory idea without making its
planner/worker routing a core dependency. It applies only to expert-major MoE
layouts whose loader exposes file-backed expert ranges. Dense, row-interleaved,
and anonymous CUDA-host layouts are ineligible.

It is blocked on appropriate hardware/model artifacts. Do not implement or
benchmark the large-model experiment on the 128 GB daily driver just to close a
checkbox.

### Phase 0 — cheap stop/go evidence

- [ ] **PIN-0.1 — capable route.** Prove exact backend/model/quant support,
  expert mapping, and pageability; fail closed for anonymous expert buffers.
- [ ] **PIN-0.2 — contained profile.** One short-context slot, small ubatch,
  capped reasoning/output, mmap + reclaim band, and no global `--mlock`.
- [ ] **PIN-0.3 — stop/go corpus.** Run ten privacy-safe agent planning briefs;
  record cold/warm TTFT, decode, disk reads, major faults, reclaim, correctness,
  warmup, and an expert-spread variant.
- [ ] **PIN-0.4 — stop if compute loses.** Below 0.5 warm tok/s, stop. Treat
  0.5–0.8 as research-only; build pinning only with a credible >=0.8 tok/s path.

### Phase 1 — calibrated pin-set

- [ ] **PIN-1.1 — per-expert ranges.** Extend GGUF metadata to
  `(layer, expert_id) -> [offset,length]`, validate shards/layout, and reject
  row-interleaving.
- [ ] **PIN-1.2 — activation evidence.** Trace bounded `{layer, expert_id}` data
  over a privacy-safe 64–256-turn agent corpus, keyed by model/quant/backend/
  workload/schema.
- [ ] **PIN-1.3 — residency ledger.** Reserve OS/runtime, dense/shared/router/
  embedding, KV, checkpoints, scratch, companions, VRAM weights, and an unlocked
  fault window before filling RAM with hot ranges.
- [ ] **PIN-1.4 — range control.** Mmap the original GGUF; prefault and `mlock`
  hot ranges; `madvise(MADV_COLD)` the tail; cleanly unlock on rollback/exit.
  Do not split or prune the model in version one.
- [ ] **PIN-1.5 — domain shift.** Record hit/miss, disk, TTFT, and turn wall
  time. Rebuild only on sustained drift; unseen experts remain on disk.
- [ ] **PIN-1.6 — same-corpus A/B.** Beat ordinary mmap outside noise. A miss
  rate >=25% rejects the pin profile.

### Phase 2 — product gate

- [ ] **PIN-2.1 — routing stays optional.** Prove single-model residency first.
  A planner+worker consumer needs separate routing/quality validation.
- [ ] **PIN-2.2 — no short-bakeoff auto selection.** Require a full hardware,
  model, and workload profile plus explicit authorization.
- [ ] **PIN-2.3 — TUI eligibility.** Require warm 2k-brief TTFT <15 s, warm
  decode >=1.0 tok/s, stable reclaim, and an end-to-end agent win.
- [ ] **PIN-2.4 — fail closed.** Bad evidence, unsupported layout/backend,
  inadequate fault-window RAM, failed `mlock`, disk stall, OOM, or miss storm
  falls back to ordinary mmap or a resident smaller model—never expert deletion
  or silent thrashing.

Exit: a model exceeding usable RAM+VRAM serves a bounded declared agent role
with every expert reachable, stable reclaim, and measured benefit. Until then
this is blocked/experimental and outside “automatic best.”

## P4 — public hardware and release proof

- [ ] **PORT-1 — capability policy.** Generate candidates from detected CUDA,
  ROCm/HIP, Vulkan, Metal, and CPU/backend semantics; unsupported knobs vanish.
- [ ] **PORT-2 — portable fixtures.** Use synthetic profiles and recorded
  evidence from multiple hardware shapes. Reference-rig data tests rules; it
  does not become a rule.
- [ ] **PORT-3 — hardware matrix.** Validate resident dense, heterogeneous
  multi-GPU, offloaded MoE, recurrent/SWA, parallel-agent, and mmap cases; add
  AMD/Metal/Windows evidence when hardware exists.
- [ ] **PORT-4 — fault injection.** Cover stale profiles, flag changes, busy
  devices, partial shards, no oracle, crash/OOM, disk stalls, and interrupted
  relaunch.
- [ ] **PORT-5 — release contract.** Document cold estimate vs converged winner,
  per-agent context, inspection/reset, and all last-resort warnings.

## Tracking rules

- Close a box only with implementation commit, focused regression test, and
  preserved hardware evidence for fit/performance claims.
- Convert reference-rig observations into capability/measurement rules before
  making defaults.
- A successful load alone is neither stability nor performance acceptance.
- Never weaken quality, useful context, foreground response, cache reuse, or an
  explicit setting silently to win.
- Replicas/worker pools stay in the workflow-capacity plan. FreeToken stays in
  its separate lane.
- AI Tune is legacy/optional. Manual calibration can remain diagnostic, but
  standard launch owns the final fit and performance decision.

## SMALLCTX — three defects found by one honest e2e run — 2026-09-10

The `install-e2e` Windows job installed cleanly, ran the backend, and then
failed to serve. Chasing that single failure surfaced three separate defects,
all in the same shape: a small or unusual model is not a broken model, and
ggrun refused to serve one it could have served.

Evidence: real llama-server, `ggml-org/models` `tinyllamas/stories260K.gguf`
(1.1 MB, `n_embd` 64, `n_head` 8, `n_head_kv` 4, `n_ctx_train` 2048, 512-entry
SPM vocabulary). Confirmed on the CI runner and reproduced locally on CPU.

**1. The e2e fixture could never have loaded.** Both jobs built their model
with `tests/build_synthetic_gguf.py`, which writes headers only. A real backend
rejects it outright:

```
error loading model hyperparameters: key not found in model:
  llama.attention.layer_norm_rms_epsilon
loaded meta data with 9 key-value pairs and 0 tensors
```

Only the fake test backend ever "loaded" it. The generator is correct for the
GGUF parser tests it was written for; it is not a serving fixture. Both jobs now
fetch stories260K, and both ask for a completion after `/health`, because a
server that answers `/health` has not necessarily loaded anything a user can
talk to.

**2. The KV block-size guard was blind, because the head count was derived from
the wrong key.** `parse_gguf.py` never read `attention.head_count`; ggrun
back-derived heads as `embd / key_length`. A model that states `head_count` and
omits `key_length` therefore had *both* at zero, `kvTypeFitsHeadDim` saw no
constraint, and ggrun emitted `--cache-type-k q8_0` for an 8-wide head:

```
K cache type q8_0 with block size 32 does not divide n_embd_head_k=8
server process exited during startup: exit status 0xc0000005
```

The guard itself was already right. Absent is not zero (invariant 9): llama.cpp
does not treat a missing `key_length` as unconstrained, it derives the width
from `n_embd / n_head` and enforces the rule against the derived value.
`parse_gguf.py` now reads `attention.head_count`; `kvHeadDims` derives the width
the same way llama.cpp does; the plan for this model is now `f16`.

**3. The cache canary sized its prompt in words, and paid in tokens.** Three
420-word segments are ~1,700 tokens on an ordinary tokenizer and 15,873 tokens
on a 512-entry vocabulary. Against a 2,048-token context the backend answered
HTTP 400 and ggrun rejected a server that was serving correctly:

```
request (15873 tokens) exceeds the available context size (2048 tokens)
Error verifying server profile: functional canary failed
```

The canary now receives the per-slot context, measures the tokenizer's actual
expansion through the backend's own `/tokenize`, and sizes its segments to fit.
When the context cannot hold two 512-token checkpoints it verifies the
completion endpoint and reports prefix reuse as *unproven* rather than failed --
a degraded profile, not a rejected launch. An unmeasurable tokenizer falls back
to one token per byte, the worst plausible expansion, never a low guess.

Verified end to end after all three: ggrun launches stories260K on CPU, serves,
and generates. It lands in `StateDegraded` for an honest reason ("output varied
across replay (expected on very small models)"), which is the correct
destination for a 260K-parameter model.

### Open

- [ ] The release workflow never ran the *packaged* backend outside its build
      tree, which is why #28 shipped: the smoke test ran
      `/tmp/llama.cpp/build/bin/llama-server`, where its libraries sit beside
      it. Now fixed for the cpu/vulkan bundles by extracting the tarball and
      running it with the build tree hidden. **The CUDA bundle still is not
      covered**: `package-cuda` exports `LD_LIBRARY_PATH` to the bundle's own
      `bin` before running it, which masks a missing RUNPATH exactly the way
      the build tree did. Decide whether that is legitimate (ggrun's `libhub`
      does set `LD_LIBRARY_PATH` for the backend it launches) or whether the
      CUDA bundle should be patchelf'd like the others. Do not change it blind.
- [ ] `--ctx-size` small enough to break the canary is reachable on ordinary
      models too, not just tiny ones. Worth a matched run with a normal model
      pinned to a small context to confirm the new sizing holds there.

## MACOS — the packaged backend installs but ggrun cannot find it — 2026-09-10

`release-install-macos-smoke` has been red on main for weeks and is not
affected by the 2026-09-10 packaging work. The install itself reports success:

```
⚠ llama-server installed but needs a GPU runtime on this machine
    (kept; llama.cpp will still be installed)
✓ Installed llama-server from ggrun-macos-arm64-metal.tar.gz
```

and then serving fails:

```
Error: selected backend "llama" was not found under APP_HOME "…/app" or the
registered backend paths; install/build it or choose backend auto
```

So the metal bundle is installed, the installer promises to install llama.cpp
as well, `--cpu` selects the `llama` backend, and nothing under APP_HOME
answers to that name. Either the promised llama.cpp install does not happen, or
macOS backend discovery does not see what was installed.

- [ ] Reproduce on real macOS hardware. The runner log cannot distinguish
      "never installed" from "installed where discovery does not look".
- [ ] Decide whether `--cpu` on a metal-only install should select the metal
      backend rather than failing, and whether that warning should be an error
      at install time instead of a pass.

## MACOS — resolved 2026-09-10: the tag names the dialect, not the fork

`release-install-macos-smoke` had been red on main for weeks. It was not a test
artifact and not the packaging bug: **every macOS install was unusable out of
the box.**

`scripts/setup-home.sh` records the backend for a metal install as:

```sh
elif [[ "$backend_config" == "cpu" || "$backend_config" == "metal" ]]; then
    backend_config="llama"
```

and `detectBackend` tags every darwin build `metal`, deliberately, so placement
does not emit CUDA or Vulkan device-routing flags for it. `backendMatches`
required an exact tag match, so a `llama` request could never resolve on macOS
and launch died with:

```
selected backend "llama" was not found under APP_HOME "…" or the registered
backend paths; install/build it or choose backend auto
```

The tag conflates two different things: which fork the binary came from
(llama.cpp versus ik_llama.cpp) and which device dialect placement must speak
(cuda, vulkan, metal). A `llama` request now accepts `llama`, `vulkan` or
`metal`, because all three are mainline llama.cpp. `ik_llama` is a genuinely
different fork and stays distinct, and an explicit `metal` or `vulkan` request
stays narrow.

Fixed in `backendMatches` rather than in the installer on purpose. The
installer only writes config on a first install ("upgrades must retain user
choices"), so an installer-side fix would have left every Mac already in the
field broken.

The same exact-tag rule also meant a Linux user whose config said `llama` could
not use an installed Vulkan build. That is fixed by the same change.

### Still open

- [ ] The install probe's classifier returns `needs_gpu` as its catch-all for
      output it does not recognise (`classify_probe_output`, install.sh). A
      binary that started, printed something and exited cleanly is by
      definition not missing a GPU runtime. The `runs` branch requires the word
      "version" in the output, so any backend that words its `--version`
      differently is filed as needing a GPU it does not need. Harmless today
      because `needs_gpu` keeps the backend, but it is the least accurate
      reading available and it made the macOS install log actively misleading
      while this bug was being chased.

## MACOS — deprecated, best-effort, never a release gate — 2026-09-10

macOS is in a working state as of today: the packaging fix bundles its dylibs
with `@loader_path`, and `backendMatches` resolves a `llama` request against a
`metal`-tagged backend, so `release-install-macos-smoke` is green on main.

It is nonetheless **deprecated**. Nobody working on ggrun has a Mac, so every
macOS defect this session was diagnosed from CI logs and artifacts alone, which
is slow and cannot distinguish some failure modes at all. The decision is to
keep it working where that is cheap and never to let it gate anything:

- `release-install-macos-smoke` is `continue-on-error: true`. The job still runs
  and its result is still worth reading; it does not block a merge.
- The macOS entry in the release `package` matrix is `optional: true` and the
  job is `continue-on-error`. `publish` merges whatever artifacts exist, so a
  macOS failure degrades to "no macOS bundle in this release", not "no release".
- README says so plainly, so a Mac user is not misled about the support level.

Delete rather than deprecate only if it starts costing real time again. Right
now it costs nothing and it works.

## Generic configuration review — 2026-09-13

Reviewed main `f777280`, independently of the staged production checkout and
unmerged hot-experts branch. Goal: a useful, correct automatic launch under
explicit constraints, with bounded evidence-driven fallback before refusal.

| Boundary | Finding | Next action |
|---|---|---|
| TUI / CLI | `tuiLaunchArgs` feeds `cmdLaunch`; both use placement and the launch recovery controller. | Preserve this shared path; extend fixtures across entry points. |
| Oracle / serving argv | `preflightArgs` dropped KV-offload, full-SWA, unified-KV, host-allocation switches and metadata overrides; equals-form overrides and negative layer counts were also lost. | Fixed transport for these flags. Unsupported oracle options return an error and select the existing contained probe. Retired v7 keyed probe measurements. |
| Recovery | `launchMemoryRecovery` rejects repeated argv; `recoverPreflightOOM` recomputes GPU failures and can reduce automatic context. Explicit constraints and exact challenger admission remain separate. | Trace the full GLM failed allocation through this controller; do not infer its cause from the oracle-filter defect. |
| Host admission | A cgroup OOM in `runGuardedAllocationPreflight` returns an ordinary error; the caller fails closed before GPU recovery selection. | Add typed host-failure evidence and bounded complete re-placement, preserving explicit resident/mmap and context constraints. |
| CPU-only | `preflightPlacement` returns immediately when no GPUs are present. | Review host admission and recovery independently; GPU tests do not establish CPU-only fit coverage. |
| Unknown capabilities | Missing containment may continue on an estimate; some memory-shaping backend extensions remain outside the oracle filter. | Define capability coverage explicitly and avoid presenting partial oracle coverage as complete allocation proof. |

Scope of this fix: contract invariants 2 (complete configurations), 4 (memory
admission), 8 (model/hardware independence) and 9 (evidence versioning). Tests
cover policy preservation, last-wins overrides, malformed arguments, unsupported
oracle options and old-cache rejection. No topology, quality, context or
performance-selection policy changed. Key invalidation also retires old keyed
live measurements conservatively; separately recorded runtime-growth evidence
remains available.

This is controller correctness evidence, not a claim that GLM now loads or that
all models/hardware fit. No production process was restarted. Full GLM serving,
long-context stability and matched agent-throughput acceptance remain open.

Validation: `scripts/verify-core-engine.sh` passed uncached after the final
change (six core package suites, formatting and vet); `git diff --check` passed.

### Context and reliability follow-up — 2026-09-14

User priority: failure to serve, or serving with insufficient context to complete
agent work, takes precedence over throughput experiments and helper expansion.
Preserve explicit parallelism; do not reinterpret a context repair as evidence
that wider concurrency is faster.

Confirmed a context-unit defect: automatic fit treated the model's native
per-sequence context as a total across all slots. The Claude option builder
also imposed that same native total cap. A roomy synthetic 131072-native model
with two slots received 131072 total rather than 262144. The backend's
`llama-context.cpp` compares per-sequence context with training context; its
server caps each slot separately.

Correction: derive a native total cap for each candidate slot count, retain the
existing total workload ceiling and complete memory-fit search, preserve
explicit numeric/max requests, and bound multiplication. Reduced-slot
candidates cannot inherit the wider candidate's native allowance. Planner
identity 6 retires verified configs produced by the earlier semantics.

Regression coverage includes 1/2/4 slots, total policy limits, reduced-slot
caps, integer overflow, the Claude option-to-placement path, and existing
explicit-context and occupied-memory tests. The uncached core gate passed
(six package suites, formatting and vet). This does not establish live memory
admission or throughput at the larger contexts.

Historical log inspection, 2026-09-14 (no live server found):
`/home/mik/ggrun-project/ggrun/.logs/ggrun-claude-server-v2-8081-aaa6009f8acf7e7ce82a0f42.log`
contains Qwen loads at total 262144, parallel 4, per-slot 65536, partitioned KV,
resident loading. The retained full-GLM probe failure
`.cache/memory-probes/failed-e8120838fcc66f29922823364dec37b8.log`
shows mmap, total 172032, parallel 1 and a failed 2946038912-byte CUDA0 compute
allocation. These are historical log excerpts, not new serving acceptance or
exact current process identities. The per-slot cap correction cannot explain
that single-slot GLM reduction; its compute/admission recovery remains open.

The private lab's `qwen38-p1-p2-selection-2026-08-31.md` retains p1 as the serial
baseline and notes unmatched traffic and queue-pressure limitations. No new
concurrency winner is inferred here. Helper routing already exists through
`claudeauto` utility/reviewer paths; expansion is deferred behind admission,
useful-context and failure-recovery correctness.

### User launch observation — 2026-09-14T09:11:48.856630+00:00

After the user launched the installed development binary, process inspection
found no running ggrun/llama-server process. The newest reviewer log,
`/home/mik/ggrun-project/ggrun/.logs/ggrun-claude-reviewer-45243.log`
(2026-09-14 09:11 UTC), records Qwen3.5-4B, one slot, 131072-token
context, listening on loopback port 45243 and health OK after 3 seconds.
There is no new main-model serving log; the previous log is from September 13.
The main model, exact launch argv and reason for exit are not established.
Awaiting terminal output rather than attributing old context errors to this
launch. Snapshot: `/tmp/ggrun-user-launch-observation.json`. No process was
stopped or restarted by this inspection. This is not serving acceptance.

### GLM Flash preflight recovery dead end — 2026-09-14

User terminal output identifies GLM-5.3-Flash Q3 XL (137.4 GiB), one slot,
592896 total context, q8_0 GPU KV, resident loading, batch/ubatch 2048/128,
all 43 expert layers on CPU. Reviewer Qwen3.5-4B reached health and measured
5650 MiB against its 6144 MiB reservation. Main backend was
`.src/fork-glm-5-3-flash/build-cuda/bin/llama-server` on port 8081.
The no-allocation oracle reported CUDA0 13707/11873 MiB (1834 MiB deficit).
Complete re-placement found 515072 tokens, but recovery discarded it because
its context differed; the remaining context-derate path only handles measured
compute-allocation failures. No main-model weight load followed.

Fix: accept a fully rebuilt smaller automatic-context candidate from an oracle
rejection for the next bounded preflight, keeping slots, KV policy, residency
and non-increasing ubatch. This does not mark it admitted or bypass the outer
exact-candidate/repeated-argv/retry gates. Explicit context remains immutable.
A real-Compute synthetic regression reproduced the same refusal at 69632 ->
55296 tokens before the fix. Larger GLM contexts remain subject to admission;
this repair is recovery correctness, not performance or serving proof.

## UTIL — the GPU e2e job never measured hardware use — 2026-09-14

The goal is the fastest agentic serving the hardware can give, so how much of
the hardware a launch actually consumes is the number to optimise. The GPU
install job proved a model *served*; it recorded which devices received weight
allocations and nothing about how much memory the launch used. A placement that
serves correctly while leaving two thirds of the VRAM idle passed exactly like
one that used the machine well.

`verify-installed-serving.py` now samples `nvidia-smi` before the launch and
again once the launcher is ready, and records the difference:

- `utilization.devices[CUDA<n>]` — `before_mib`, `after_mib`, `launch_mib`,
  `total_mib` per device.
- `utilization.launch_mib` / `capacity_mib` / `fraction_of_vram` — what this
  launch put on the GPUs against what the machine has.
- `served.context` / `served.slots` — from `/props`, what the backend actually
  served, not what was planned.

Two deliberate choices. The baseline subtraction keeps another tenant's memory
out of our number. Absent `nvidia-smi` records `null`, not `0` — absent is not
zero. The sample is taken before the `--min-weight-devices` assertion can abort
the run, because an underusing placement is the thing being hunted and its
evidence has to survive the failure.

`install-e2e.yml` gains a `gpu_model_path` dispatch input. Models large enough
to stress placement cannot be re-downloaded per run, so a path on the runner is
how that shape gets covered; the default still exercises `ggrun download`,
which is the documented flow and has to keep being tested. `gpu_context: 0`
already means automatic, which is the path that selects context and slots.

### Baseline, this rig, 2026-09-14

`ggrun v3.2.9-dev.c8924d9`, Qwen3.5-4B-Q4_K_M, automatic context and slots:

| | |
|---|---|
| weight devices | `CUDA1` only |
| launch VRAM | 8,937 MiB of 49,134 MiB — **0.1819** |
| CUDA0 / CUDA2 | 162 / 112 MiB, no weights |
| served | 262,144 tokens, 1 slot |

Lifecycle passed: generation, streaming, clean shutdown, port released. This is
a correct launch that uses under a fifth of the machine and leaves two cards
holding scratch. It is the reference point for the resident-fast-path work in
P1; a candidate that raises `fraction_of_vram` without losing the lifecycle
gates is the shape of an improvement.

### Open

- No large-model figure yet. GLM-5.3-Flash UD-Q3_K_XL (137.4 GiB) is the
  interesting case and currently fails before loading weights; see the context
  recovery work on `harden/preflight-memory-shape`.
- `fraction_of_vram` is not yet asserted. It is recorded and compared by hand
  until enough runs exist to say what a floor should be per model shape.
- The `gpu` job is gated on `github.ref == 'refs/heads/main'`, so this change
  has to land on main before a dispatch can exercise it.
## CTXRATCHET — memory recovery climbed back into a rejected context — 2026-09-14

GLM-5.3-Flash UD-Q3_K_XL (137.4 GiB), automatic context, `--parallel 1`, on the
three-card rig. Codex had just fixed recovery discarding a fully recomputed
smaller plan; with that fix the launch got further and then failed a second way:

```
[launch] preflight context-replanned after CUDA0 allocation 0 MiB (deficit 1810 MiB, ctx=385024, ...)
[launch] preflight: placement fits (CUDA0 7397/11873, CUDA1 21745/24112, CUDA2 5552/6251)
[placement] context fit: 589824 tokens, 1 slot(s), KV q8_0 on gpu
[launch] backend-measured memory re-plan 5/5 changed the exact argv; verifying the new placement before production
[launch] preflight: placement does not fit (CUDA0 13671/11873 ...)
```

Recovery lowered context from 592,896 to 385,024 and preflight **accepted** it.
The backend-measured re-plan then recomputed from `placementOpts()` — the
original automatic request — and proposed 589,824, back inside the range the
same launch had already disproved. That burned the re-plan budget and the
launch failed without ever loading weights.

`launchMemoryRecovery` already refuses to resurrect a rejected argv, but it
keys on the exact argv identity. The climbed-back plan has a *different* argv
at a disproved context, so no identity ever matches. Context needed its own
ledger entry.

The fix mirrors the derating discipline one function away in
`recoverPreflightOOM`, where a retry at ubatch 256 is explicitly never
recomputed back to 512:

- `launchMemoryRecovery.rejectContext` records the smallest **automatic**
  context this launch has proven does not fit. An explicit context is a user
  constraint (invariant 3) and is never recorded.
- `automaticContextCeiling` returns one token below it. `placement.Compute`
  floors `AutoContextMax` to its granule, so the rejected context is excluded
  without the launch package knowing the granule.
- `boundByRejectedContext` is the single place that applies the ceiling, so
  production and tests exercise the same rule rather than a re-description.

The ceiling only ratchets down. A later, larger rejection cannot raise it.

Invariants: this strengthens 4 (memory safety is fail-closed — a disproved
context stays disproved) and 10 (an ordinary launch does not become an
unbounded series of long reloads). It moves no coordinate the user pinned.

### What is proven and what is not

Proven: `TestRejectedAutomaticContextCapsTheBackendMeasuredRecompute` fails
without the fix (`ceiling 0 does not exclude the rejected context 69632`) and
passes with it; `TestContextCeilingOnlyRatchetsDown` replays the 592,896 →
385,024 → 589,824 sequence above. Uncached `scripts/verify-core-engine.sh`
passes on all six packages.

Not proven: that GLM-5.3-Flash now reaches healthy serving. The fix removes one
dead end; the launch may still need further recovery below 385,024, and only a
live load says so. No hardware-utilisation figure exists for a model of this
size yet — see the UTIL entry above for the measurement that will produce one.

### Live runs, GLM-5.3-Flash UD-Q3_K_XL, 2026-09-14

Three matched runs on the three-card rig, automatic context and slots, through
`scripts/verify-installed-serving.py` so each one leaves evidence.

`v3.2.9-dev.e049951` — rejected-context ceiling only. The ceiling engaged
(`model/policy context cap reached`) but the launch still failed:

| step | ctx | preflight |
|---|---:|---|
| initial | 670,720 | does not fit |
| recovery | 563,200 | does not fit |
| recovery | 555,008 | **fits** |
| measured re-plan 3/5 | 562,176 | does not fit |
| recovery | 500,736 | **fits** |
| measured re-plan 5/5 | 561,152 | does not fit — budget gone |

Both climbs sat *below* the smallest rejected context (563,200) and *above* a
plan preflight had just accepted. Bounding by rejection alone cannot catch
that: the accepted plan is the proof, and the re-plan was spending it.

`v3.2.9-dev.f7fdc1a` — accepted context bounds the re-plan. The climb stopped;
the re-plan re-proposed exactly the accepted 500,736. The launch still failed,
for a third reason of the same family: at that pinned context the re-plan
recomputed ubatch from the original automatic request to 256, where the
accepted plan had been derated to 128, and overshot every device —
`CUDA0 15880/11873, CUDA1 26826/24112, CUDA2 15227/11909`.

`v3.2.9-dev.2d1d0c1` — the accepted ubatch is pinned too.

The common defect across all three: **the measured re-plan recomputes from the
original automatic request and silently discards what this launch has already
proven.** `recoverPreflightOOM` had the discipline for ubatch and nothing else
had it for context. The ledger now carries both, applied in one place
(`boundByProvenLimits`), and only ratchets down.

### Open

- Whether GLM-5.3-Flash reaches healthy serving is still unproven; the third
  run is the first that can. Each fix removed a dead end without establishing
  that the remaining path converges.
- `maxPreflightReplans` is 5. Every one of these failures spent the budget on
  re-plans that undid prior work rather than on genuinely new shapes. If a
  budget increase is ever proposed, it is a symptom, not a fix.
- No `fraction_of_vram` figure for a model this size yet.

### Resolved — GLM-5.3-Flash serves, 2026-09-14

`v3.2.9-dev.2d1d0c1`, same rig, automatic context and slots, via
`verify-installed-serving.py`. The re-plan reached a fixed point instead of
undoing itself:

```
[launch] preflight: placement fits (CUDA0 11222/11873, CUDA1 23629/24112, CUDA2 10183/11909)
[launch] memory plan stable at oracle-planned evidence
```

| | |
|---|---|
| weight devices | `CUDA0`, `CUDA1`, `CUDA2` — all three |
| launch VRAM | 7,424 + 17,531 + 9,550 = **34,505 MiB of 49,134** |
| `fraction_of_vram` | **0.7023** |
| served | 500,736 tokens, 1 slot |
| lifecycle | generation, streaming, no forced cleanup, port released |

A 137.4 GiB model serving half a million tokens of context on 48 GB of VRAM,
using 70% of it, with a clean shutdown. Against the Qwen3.5-4B baseline in the
UTIL entry above (0.1819) this is the same launcher making very different use
of the same machine, which is the comparison the utilisation figure exists to
support.

What this does **not** establish: decode throughput. The harness proves the
lifecycle and how much hardware the placement claimed, not tokens per second.
Prompt processing during the canary ran at 20 tok/s on a 6,356-token prompt;
that is one cold observation, not a performance result, and P1's fast-path work
still needs matched agent-workload evidence. Comparing this against the
MiniMax-M3 row in the README requires a matched benchmark run, not this.

## UTIL — first GPU CI run that measures the hardware — 2026-09-14

[Run 34833268807](https://github.com/raketenkater/ggrun/actions/runs/34833268807),
`gpu` job on the self-hosted Linux runner, main at `7bab725`. Fresh install from
`setup.sh` into `runner.temp`, Qwen3.8-27B-UD-Q4_K_XL (17 GB, `qwen35`, dense)
from a path on the runner, `gpu_context=0` so context and slots are automatic.

Backend: `ik_llama-server-cuda`. Both the serve and relaunch stages passed.

| | serve | relaunch |
|---|---:|---:|
| CUDA0 | 2,974 MiB | 2,902 MiB |
| CUDA1 | 23,839 MiB | 23,699 MiB |
| CUDA2 | 210 MiB | 210 MiB |
| total | 27,023 / 49,134 | 26,811 / 49,134 |
| `fraction_of_vram` | **0.5500** | **0.5457** |
| served | 262,144 / 1 slot | 262,144 / 1 slot |

Weights landed on CUDA0 and CUDA1. **CUDA2 held 210 MiB and no weights** — a
whole 12 GB card idle while the plan used 55% of the machine. The launch is
correct and the lifecycle is clean; it simply does not use the third card.

This is the first figure of its kind produced by CI rather than by hand, and it
is the number P1 has to move. Three points now exist on this rig:

| model | `fraction_of_vram` | devices with weights | served |
|---|---:|---|---|
| Qwen3.5-4B-Q4_K_M | 0.1819 | CUDA1 | 262,144 / 1 |
| Qwen3.8-27B-UD-Q4_K_XL | 0.5500 | CUDA0, CUDA1 | 262,144 / 1 |
| GLM-5.3-Flash UD-Q3_K_XL | 0.7023 | all three | 500,736 / 1 |

Only the largest model uses all three cards. Utilisation tracks model size, not
a placement decision to spend the hardware — which is exactly the gap.

### A fork architecture cannot be a fresh-install CI target

The first attempt at this run used GLM-5.3-Flash and failed in 2 minutes:

```
No installed backend can load architecture glm5next for glm-5-3-flash.
... The .bin/llama-server-vulkan backend does not support it.
```

The job installs a clean backend; GLM-5.3-Flash needs the `glm-5-3-flash` fork
that exists only in a developer's own install. ggrun refused correctly rather
than loading a backend that cannot serve the architecture, and the Vulkan
mention is correct fallback reporting, not a backend-selection defect — the
passing run above selected CUDA on the same machine.

So GPU CI must use a mainline-supported architecture. Local evidence for
fork-architecture models stays local, and the CTXRATCHET entry's GLM figures
are not reproducible by this job.

### Open

- CUDA2 idle at 17 GB model size. No assertion on `fraction_of_vram` yet; it is
  recorded and compared by hand until enough shapes exist to set a floor.
- Decode throughput is still unmeasured here. The run logged a 765.2 tok/s
  prefill pilot; that is not a decode result and not an agent-workload result.

## RESERVE — the runtime graph reserve could only ratchet up — 2026-09-14

Why the hardware was not being used, measured rather than argued.

GLM-5.3-Flash, three-card rig. ggrun's preflight ledger against what the launch
actually allocated, sampled once the launcher reported ready:

| device | fit | overhead | runtime reserved | budgeted | actual | over-reserved |
|---|---:|---:|---:|---:|---:|---:|
| CUDA0 | 6,941 | 274 | 4,007 | 11,222 | 7,425 | 3,797 |
| CUDA1 | 16,778 | 445 | 6,406 | 23,629 | 17,532 | 6,097 |
| CUDA2 | 9,133 | 191 | 859 | 10,183 | 9,551 | 632 |
| total | | | 11,272 | 45,034 | 34,508 | **10,526** |

ggrun was not choosing to leave 14.6 GB idle. It believed the cards were 94%
full, and that reserve pushed 43 expert layers to the CPU on a model whose own
diagnosis reads `bottleneck CPU expert bandwidth`, with 122,844 MiB of weights
in host RAM.

### Root cause

`RecordRuntimeGraphGrowthFromOOM` has two production callers, `main.go:5245`
and `main.go:6572`. Both are failure paths. `RecordRuntimeGraphGrowth` — the
success-path recorder — had **no production caller at all**; only
`placement_test.go:1937` called it. No key could acquire a growth record by
working.

An automatic context lands on a different ctx most launches, so the exact key
is usually cold and falls back to `RelatedModelRuntimeGraphGrowth`. That carry
matches on model basename plus GPU signature and deliberately relaxes backend,
ctx and ubatch, then takes the **maximum** across every match. So one large
growth becomes the permanent floor for every later plan.

Here the 4,007 MiB came from this same model on **2026-09-03**, at ctx 790,528
/ ubatch 64, on the `glm-5-3-flash-hot-experts` build — applied eleven days
later to ctx 500,736 / ubatch 128 on the plain build. It is not a foreign
model's record, and an earlier draft of this entry was wrong to say so; it is
this model's own record from an unrelated configuration, which the carry cannot
distinguish.

### The fix

`RecordPostLaunchRuntimeGraphGrowth` files what a launch that reached healthy
serving actually needed, for its own signature. Growth is
`used - baseline - system overhead - (model + KV + compute)`, where the buffer
figures come from the backend's own log via `parseBuffersFromLog` rather than
from the strategy — deriving them from the plan would fold planning error into
the measurement. A device missing any input yields no entry: an unexplained
remainder is unknown, not growth.

It cannot loosen a real OOM bound. `recordRuntimeGraphGrowth` keeps the larger
of two measurements, so an observed allocation always wins; this only fills a
key that knows nothing or replaces a guess. If a recorded value ever proves too
small the backend OOMs, recovery derates, and the OOM recorder ratchets it back
up — the direction that already worked.

### Live evidence, same rig, automatic context and slots

| run | reserve | `fraction_of_vram` | host model buffer |
|---|---|---:|---:|
| before | carried 4007 / 6406 / 859 | 0.7023 | 122,844.81 MiB |
| A, learns | recorded 708 / 2303 / 474 | **0.7627** | 119,856.81 MiB |
| B, spends | used the record | **0.7626** | 119,856.81 MiB |

All three served 500,736 tokens on all three devices and passed generation,
streaming, clean shutdown and port release. 2,988 MiB of expert weights moved
off host RAM onto the cards, and the result is stable across two launches.

Note the recorded growth (3,485 MiB total) is far above the ~746 MiB idle
estimate in the table at the top. The recorder samples at the post-health point
with the graph live, which is the conservative figure, and the right one for a
memory reserve.

### Open

- The carry still takes the maximum across every configuration of a model. A
  record from a much larger context or a different backend build can still
  dominate a cold key. Narrowing it to comparable configurations is the next
  step; recording success growth reduces how often the carry is reached at all
  but does not change how it chooses.
- `OptimizationBoundary.DeviceSlackMB` is written and never read.
- `CalibrationCandidates` varies batch, ubatch, parallel and topology policy,
  but never expert residency, so no candidate proposes spending proven slack on
  moving experts off the CPU.
- Decode throughput is still unmeasured. This entry proves residency and
  lifecycle, not tokens per second.

## VRAMFILL — higher VRAM use did not make GLM faster — 2026-09-14

The reserve fix moved 2,988 MiB of expert weights off host RAM onto the cards
and raised `fraction_of_vram` from 0.7023 to 0.7627. Decode throughput did not
move. Every `eval time` sample of at least 8 tokens from the three matched
runs:

| run | samples | median tok/s | mean tok/s |
|---|---:|---:|---:|
| before the reserve fix | 13 | 6.83 | 6.87 |
| A, records the reserve | 20 | 6.77 | 6.23 |
| B, spends the reserve | 6 | 6.83 | 6.85 |

The medians agree to two decimals. This is not a small win hidden in noise; it
is no win.

The arithmetic explains it. GLM-5.3-Flash UD-Q3_K_XL keeps about 120 GB of
experts in host RAM on this 48 GB rig. Moving 2,988 MiB is **2.5% of the
offloaded weight**, against a bottleneck the optimizer itself labels
`CPU expert bandwidth`. No placement decision available inside 48 GB of VRAM
changes the other 117 GB.

### What this means for the objective

`fraction_of_vram` measures how much of the machine a plan claimed. It is not
a proxy for speed, and it must not become a promotion criterion. README already
states the product goal — "the fastest **stable** plan for the requested
workload, not maximum VRAM fill" — and this is the measurement that backs it.

The utilisation figure remains worth recording. It is how the phantom 11,272
MiB reserve was found, and it is the right diagnostic for "is anything that
could be resident sitting on the CPU". It is the wrong thing to maximise.

### Where residency gains should pay off

Not on a model 3x past VRAM capacity. The reserve fix should matter where a
few GB moves a large fraction of the offloaded weight — a MoE close to the
capacity boundary, where 3 GB is a third of what is on the CPU rather than a
fortieth. That case is untested here and is the one worth measuring next.

### Still unproven

Agent-workload makespan. These are 32-token generations from the serving
check, which measure decode rate, not cache-backed turn time or
requested-concurrency throughput. Invariant 5 asks for real agent work, and
`verify-installed-serving.py --agent-lanes` exists for it; no run here used it.

## AGENTBASE — first agent-workload baseline — 2026-09-14

Every throughput figure recorded before this entry was decode rate from
32-token generations. That is not what invariant 5 asks for. This is the first
measurement of bounded tool-using agent work on this rig.

`ggrun v3.2.9-dev.62bebe4`, Qwen3.8-27B-UD-Q4_K_XL, `--calibrate off` so the
20-minute optimizer comparison is not part of what is being timed. Driven by
`scripts/verify-agent-workload.py`, 2 lanes, 3 repeats, 1 warmup, max 8 turns.
Each task is a real repair with an oracle: read the source, write a fix, run
tests, and the harness checks the result against known cases.

| task | median | range | turns |
|---|---:|---|---:|
| ceiling | 10.89s | 10.82–13.18 | 4 |
| clamp | 14.47s | 14.02–14.75 | 5 |
| interval | 6.92s | 6.63–8.82 | 3 |
| all | **10.89s** | sum 100.50s over 9 tasks | 4 |

**9/9 oracle-passed.** Not "the model produced text" — the repaired functions
returned the right answers.

| | |
|---|---|
| VRAM | 6,873 + 22,988 + 115 = 29,976 of 49,134 (0.610) |
| served | 262,144 tokens, **1 slot** |

### The slot count is the first thing to question

`--parallel` was automatic and chose **1**, so the two agent lanes serialised
through a single slot. The makespan above is therefore a queued makespan, not
two lanes running concurrently. Earlier evidence that `parallel 1` is fastest
was collected on single-stream decode, where it is the right answer; it does
not follow that one slot is right when the workload is several agents at once.
That is a specific, cheap experiment: rerun this suite at `--parallel 2` and
compare makespan, not tok/s.

### What this baseline is for

Hot experts and worker/reviewer routing are the two levers proposed for
agentic speed, and neither had a number to beat. This is that number. A
candidate must improve median task time or total makespan here, at 9/9 oracle
passes, to count as faster — an aggregate tok/s gain does not.

Note the model choice. This suite on GLM-5.3-Flash at ~6.8 tok/s would take
hours and only re-confirm it is CPU-bandwidth-bound. A fully GPU-resident 27B
is what agent work would actually run against, so it is the honest baseline.

### Still unproven

- Concurrency: only 1 slot was exercised, see above.
- Cache-backed turn time is not isolated here; the suite measures whole-task
  wall time, which folds prefill, cache reuse and decode together.
- No hot-experts or reviewer-routed comparison exists yet.

## SLOTS — concurrency, not VRAM fill, is what made agent work faster — 2026-09-14

`--parallel 2` could not launch Qwen3.8-27B at all. Recovery derated context to
301,056 tokens and its own next candidate returned to 524,288:

```
[launch] preflight context-derate after CUDA0 allocation 2188 MiB (deficit 841 MiB, ctx=301056, ...)
[placement] context fit: 524288 tokens, 2 slot(s)
Error starting server: memory preflight did not converge after 5 re-plans
```

Same family as CTXRATCHET, on a site that entry missed. `boundByProvenLimits`
guarded the backend-measured recompute; `recoverPreflightOOM` derives its own
candidate from the original automatic request and never saw the ledger. Fixed
by threading the ledger into recovery and bounding both option derivations.

### The A/B, same suite, same model, same binary

`scripts/verify-agent-workload.py`, 2 lanes, 3 repeats, oracle-checked repairs.

| | parallel 1 | parallel 2 | parallel 2 repeat |
|---|---:|---:|---:|
| makespan | 51.47s | 34.94s | 39.97s |
| correct tasks/min | 10.49 | **13.74** | **13.51** |
| median task latency | 10.89s | 8.37s | 8.38s |
| max task latency | 14.75s | 9.54s | 9.54s |
| oracle-passed | 9/9 | 8/9 | 9/9 |
| slots x served ctx | 1 x 262,144 | 2 x 262,144 | 2 x 262,144 |

About +29% correct tasks per minute at equal correctness. `correct_tasks_per_minute`
already weights correctness, so the 8/9 run is not credited for the task it got
wrong. Median latency reproduces to 0.01s and max latency to 0.00s across the
two two-slot runs.

The single 8/9 was sampling variance, not a concurrency effect: the repeat under
identical settings passed 9/9. One failure out of eighteen tasks is not evidence
of a correctness cost, and it was checked rather than assumed.

### This qualifies the earlier "parallel 1 is fastest" result

That result stands for single-stream decode, and the 39 decode samples in
VRAMFILL agree with it. It does not transfer to several agents at once, where
one slot makes the lanes queue. Slot count must be chosen against the workload
shape, not inherited from a decode benchmark.

Note what did **not** produce this gain. Raising `fraction_of_vram` from 0.7023
to 0.7627 moved decode not at all. Unblocking a second slot moved agentic
makespan by a third. The lever was scheduling, not memory.

### Open

- Automatic slot selection still chose 1 for this workload. It has no signal
  that the client intends concurrent agents; the launcher cannot infer lane
  count from a serving request, so this remains an explicit `--parallel`
  decision until something carries that intent.
- Untested above 2 slots, and untested on a CPU-offloaded MoE, where added
  slots divide context and may interact with expert bandwidth very differently.
- Worker/reviewer routing is still unimplemented. It depends on concurrent
  serving working, which it now does.

## BOTHOBJECTIVES — slots maximise hardware use and speed together — 2026-09-14

VRAMFILL showed that raising model-weight residency did not make GLM faster.
SLOTS showed that a second serving slot made agent work about 29% faster. Those
two entries read as a conflict between "use more of the hardware" and "serve
faster". Measured together on the same suite, they are not in conflict at all.

Qwen3.8-27B-UD-Q4_K_XL, `scripts/verify-agent-workload.py`, 2 lanes, 3 repeats,
SM occupancy sampled once a second for the duration of the agent lanes.

| | parallel 1 | parallel 2 |
|---|---:|---:|
| CUDA0 | 6,873 MiB | 11,613 MiB |
| CUDA1 | 22,988 MiB | 22,732 MiB |
| CUDA2 | **115 MiB** | **9,759 MiB** |
| total | 29,976 of 49,134 (**0.610**) | 44,104 of 49,134 (**0.898**) |
| correct tasks/min | 10.49 | **13.27 - 13.74** |
| oracle-passed | 9/9 | 9/9 (and one 8/9, variance) |

A second slot raised utilisation from 0.610 to 0.898 *and* cut makespan by
roughly a third. It also brought CUDA2 from 115 MiB of scratch to 9,759 MiB of
real work.

### Why residency and slots behave differently

A slot's cost is **context**, not weights. Adding one doubles the KV cache,
which consumes VRAM that model weights could not usefully claim, and it gives
the scheduler a second stream to overlap. Adding resident expert layers
consumes the same VRAM but only shifts a fraction of a bandwidth-bound
bottleneck, which is exactly what VRAMFILL measured: 3 GB of a 120 GB CPU
expert set moved, decode unchanged at 6.83 tok/s.

So `fraction_of_vram` is a fine diagnostic and a bad objective. What to
maximise is useful concurrent work, and VRAM utilisation rises as a
*consequence* when that is done through slots.

### SM occupancy during agent work

| gpu | mean | median | p90 | max | samples busy >5% |
|---|---:|---:|---:|---:|---:|
| CUDA0 | 26.5% | 29% | 31% | 52% | 94% |
| CUDA1 | 37.8% | 39% | 43% | 54% | 98% |
| CUDA2 | 29.9% | 31% | 35% | 41% | 98% |

Two things follow. There is **no idle-card problem** during agent serving: all
three are engaged in 94-98% of samples. And compute is **not** the limit
either, at 27-38% mean SM — decode is memory-bandwidth bound, which is the
expected shape.

This also corrects a figure quoted earlier in this work. The
`GPU 0 at 93% SM while GPU 1 is at 1%` line from the GLM logs is a **prefill**
observation on a CPU-offloaded MoE. It does not describe agent serving on a
resident model and should not be cited as evidence of wasted hardware here.

### Open

- Untested above 2 slots. Each slot divides context, so there is a point where
  per-agent context becomes too small to be useful; the curve has one measured
  point and needs more.
- Untested on a CPU-offloaded MoE, where added slots divide context and
  interact with expert bandwidth. GLM is the case to try and it is slow enough
  that the suite needs a smaller repeat count.
- Automatic slot selection still chooses 1 and has no signal about intended
  lane count. Until a serving request can carry that, `--parallel` is the
  user's decision, and for agent work the measured answer on this rig is 2.

## HOTEXPERTS — measured against agent work, and it does not pay — 2026-09-14

The hot-expert cache has been the standing candidate for the CPU-offloaded MoE
bottleneck, on the strength of +6% at K=32 from a synthetic benchmark driven by
hand. It had never been measured against agent work. It has now, and the
recommendation is **do not integrate**.

Patched backends already exist on this machine for GLM-5.3-Flash and
Qwen3.8-Flash-Next, advertising `--moe-expert-cache` and
`--moe-expert-cache-inserts`, so the backend was never the blocker. Driving
`llama-server` directly avoids the 32-commit rebase entirely and answers the
question first.

GLM-5.3-Flash UD-Q3_K_XL, ctx 131,072, `--n-cpu-moe 42`,
`--moe-expert-cache-inserts 2`, `scripts/verify-agent-workload.py`, 1 repeat.

| condition | cache | correct tasks/min | oracle | VRAM | CUDA1 SM |
|---|---|---:|---:|---:|---:|
| parallel 1, 2 lanes | off | 1.074 | 3/3 | 0.408 | 5.8% |
| parallel 1, 2 lanes | off (repeat) | 1.037 | 3/3 | 0.408 | 6.0% |
| parallel 1, 2 lanes | K=32 | 1.028 | 3/3 | **0.703** | 24.2% |
| parallel 2, 4 lanes | off | **0.634** | 2/3 | 0.389 | 5.1% |
| parallel 2, 4 lanes | K=32 | **0.584** | 2/3 | **0.683** | 17.7% |

### The noise floor, measured rather than assumed

Two runs intended as different configurations turned out identical - lowering
`--n-cpu-moe` from 42 to 37 changed nothing, because the `-ot` string ends in
`exps=CPU` as a catch-all that overrides it. Identical VRAM (20,060 vs 20,070)
and identical SM confirm it. They scored 1.074 and 1.037: **3.6% spread on a
null change.** At 3 tasks per run, nothing under about 7% is a result.

That makes the parallel-1 cache gap (4.3%) a tie, and the load gap (7.9%) a
real but modest loss.

### Load makes it worse, not better

The hypothesis was that concurrent agents share an expert working set, so hit
rate should climb with load. The opposite happened. The mechanism is coherent:
the cache spends host/PCIe bandwidth on inserts to save host/PCIe bandwidth on
misses, and that path is the single contended resource here. More concurrency
means more pressure on it, so cache maintenance costs more.

The SM rise is the tell. CUDA1 goes 5.1% -> 17.7% with the cache on while
throughput falls. That occupancy is upload work, not useful compute. Busy is
not productive, and SM occupancy must not be used as a promotion signal.

### Concurrency itself also loses on this model

Independently of the cache, `--parallel 2` with 4 lanes scores 0.634 against
~1.05 at one slot: **about 40% slower**. Two lanes contend for one host-RAM
expert stream rather than overlapping. This is the mirror image of the resident
27B, where a second slot was ~29% faster, and it confirms ggrun's automatic
choice of 1 slot for this model was correct.

### Recommendation

- **Do not integrate** hot experts into the core engine. Four comparisons,
  never a win, -7.9% under load, and it costs 14.4 GB that would otherwise hold
  KV. On this rig that VRAM is worth more as agent context than as expert cache.
- **Pass the flag through** for anyone who wants to experiment. Unknown flags
  already forward to `llama-server`, so `--moe-expert-cache` needs no ggrun
  change at all.

### What this does not establish

One model, one quantisation, one rig. K=32 with 2 inserts is a single operating
point; a larger K, or fewer inserts, may trade differently, and a model whose
expert working set is small enough to fit a high hit rate could behave
differently again. What is established is that the +6% synthetic figure does
not survive an agent workload, and that the integration case cannot rest on it.

## HOSTBOUND — no placement lever improves GLM on this rig — 2026-09-14

Five levers tested against the same agent suite on GLM-5.3-Flash UD-Q3_K_XL
(137.4 GiB on 48 GiB of VRAM, ~120 GB of experts in host RAM). Every one either
did nothing or hurt.

| lever | result |
|---|---|
| more resident weights (VRAM 0.7023 -> 0.7627) | decode unchanged, 6.83 -> 6.83 tok/s |
| hot-expert cache K=32 | tie at 1 slot, **-7.9%** at 2 slots |
| `--parallel 2`, 4 lanes | **-40%** (0.634 vs ~1.05) |
| `--n-cpu-moe` 42 -> 39 | **-12 to -16%** |
| context 664,576 -> 131,072 | no change, within noise |

### Using more of the machine made it slower, monotonically

| config | ctx | n-cpu-moe | VRAM | correct tasks/min |
|---|---:|---:|---:|---:|
| backend direct | 131,072 | 42 | **0.408** | **1.074** |
| ggrun auto | 664,576 | 42 | 0.763 | 1.03 |
| ggrun, more resident | 131,072 | 39 | 0.600 | 0.907 |

All 3/3 oracle-passed, so this is not a correctness artefact. The fastest
arrangement used the least VRAM.

### Why

The single contended resource is host/PCIe bandwidth feeding 39-42 CPU-resident
expert layers per token. Every lever tried either spends that resource to save
it — the cache's inserts, a second lane's parallel demand — or adds
cross-device synchronisation to a path already waiting on the CPU, which is
what moving three expert layers onto GPUs did. Shifting a fortieth of the
weight cannot pay for the coordination it adds.

### What this means for the objective

`fraction_of_vram` is a diagnostic, not a target, and on this model class it is
actively anti-correlated with speed. A policy that maximises hardware use would
have chosen the slowest configuration measured here. The product goal in README
— "the fastest **stable** plan for the requested workload, not maximum VRAM
fill" — is the correct one, and this is the measurement that backs it on the
workload ggrun exists for.

The honest recommendation for a model 3x over VRAM capacity is a smaller
quantisation or more VRAM. There is no placement decision on this hardware that
makes it fast, and more tuning attempts are not warranted without new evidence.

### What is not established

One model, one quantisation, one rig, single samples against a measured 3.6%
noise floor. The `-12 to -16%` residency result is outside that floor but rests
on one run per arm. A MoE that is only slightly over VRAM capacity — where
resident experts are a large fraction of the offloaded set rather than a
fortieth — is the case where residency should pay, and it is untested.
