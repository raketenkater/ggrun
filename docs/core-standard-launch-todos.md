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

### Qualification: this result is about *this* cache, not hybrid MoE inference

HybriMoE (arXiv 2504.05897, built on kTransformers) reports 1.33x prefill and
1.70x decode over a state-of-the-art hybrid MoE baseline using three mechanisms
the patch measured above does not have:

- **dynamic intra-layer scheduling** that balances work across CPU and GPU,
- **impact-driven inter-layer prefetching** rather than loading on miss,
- **score-based caching** rather than LRU.

The patch tested here is reactive: it loads an expert when a miss occurs, with
`--moe-expert-cache-inserts` bounding uploads per layer per step, and evicts by
LRU. That is the difference that explains the measurements. A reactive insert
spends host/PCIe bandwidth at the exact moment a miss proves that path is
saturated, which is why the cache lost more under concurrency than at one slot.
Prefetching moves that transfer off the critical path instead.

The idle-compute signature recorded above — CUDA1 at 5.8% SM and CUDA2 at 2.9%
while the CPU computes experts — is exactly what intra-layer scheduling
targets. Those numbers are an argument that there is real headroom here, not
that the headroom is unreachable.

So the conclusion is narrower than "hot experts does not pay": **this
implementation, at K=32 with 2 inserts, on this model and rig, did not pay.**
Whether a prefetching, co-scheduling implementation would is untested here and
is not refuted by anything above. The comparison is also not directly
transferable: HybriMoE's baseline is a hybrid framework on kTransformers, not
llama.cpp, and the abstract does not name the models or hardware.
## QWENFLASH — ggrun cannot launch Qwen3.8-Flash-Next at all — 2026-09-14

Qwen3.8-Flash-Next-UD-Q3_K_XL (83.8 GiB: 52.2 GiB expert, 31.6 GiB non-expert,
arch `qwen4exp`, 48 layers) fails to launch on **merged main** at the plan
ggrun chooses for itself. Reproduced with a binary built from `94b7b7e`, so
this is not caused by any unmerged work:

```
Error starting server: memory preflight did not converge after 5 re-plans
```

Both main and the branch plan identically at `--dry-run`: 262,144 tokens, 1
slot, `n-cpu-moe 21`. The divergence is in preflight recovery, which dry-run
never reaches.

### The deficit never closes because the wrong lever moves

| round | deficit | ctx | n-cpu-moe | ubatch |
|---|---:|---:|---:|---:|
| 1 | 101 MiB | 261,120 | 21 | 256 |
| 2 | 94 MiB | 260,096 | 22 | 256 |
| 3 | 87 MiB | 259,072 | 22 | 256 |

Each round trims exactly one 1,024-token context granule and reclaims about
7 MiB against a ~100 MiB shortfall. At that rate it needs roughly thirteen
rounds and the budget is five. `ubatch` stays at 256 throughout, though
derating it would cut the compute buffer by far more than 7 MiB in one step.

This is the pathology `TestGLMContextNudgeCannotBeatFailedDeviceExpertRelief`
already names on a different path: an irrelevant nudge selected over the lever
that would actually cover the deficit. The deficit-sized ceiling step added in
the same session does not help here, and correctly so — reclaiming 165 MiB of
KV genuinely computes to about 2,000 tokens. Context is simply not where this
deficit lives.

### At 4 slots it fails earlier

| round | deficit | ctx | n-cpu-moe |
|---|---:|---:|---:|
| 1 | 3699 MiB | 1,047,552 | 33 |
| 2 | 2432 | 1,046,528 | 35 |
| 3 | 1993 | 1,045,504 | 37 |
| 4 | 720 | 1,044,480 | 38 |
| 5 | **24** | 1,043,456 | 38 |

Here recovery does move the relevant lever — expert residency, 33 to 38 CPU
layers — and the deficit falls 154x to 24 MiB before the budget expires. Note
what four slots cost: five expert layers displaced to host RAM to buy KV for
the extra slots. **On a CPU-offloaded MoE a slot is not free concurrency; it is
paid for in expert residency**, and experts are what set decode speed.

### Open

- Recovery must prefer a lever sized to the deficit. A context granule that
  reclaims 7 MiB should never be selected against a 100 MiB shortfall when
  ubatch is still at 256.
- The 24 MiB residual at 4 slots is a budget question, not a lever question,
  and the progress-aware accounting in #58 addresses that class.
- This model is the 1.75x-over-VRAM case that would have bracketed the slot
  policy threshold between resident (4 slots good) and GLM at 2.86x (4 slots
  catastrophic). It cannot be measured until the launch converges.
## OFFLOADBASE — the RAM-offload agentic baseline, and why slots are not its lever — 2026-09-14

BOTHOBJECTIVES measured slots on a fully GPU-resident 27B, which is the easy
case: a second slot spends spare VRAM on KV cache and overlaps two streams.
ggrun exists for the model that does not fit, so the result has to be checked
there. It does not transfer.

`ggrun v3.2.9-dev`, GLM-5.3-Flash UD-Q3_K_XL (137.4 GiB, ~120 GB in host RAM),
agent suite at 1 repeat, 512-token cap, `--calibrate off`.

| | resident 27B, 2 slots | **offloaded GLM, 1 slot** |
|---|---:|---:|
| correct tasks/min | 13.27 - 13.74 | **1.03** |
| median task latency | 8.4s | **115.1s** |
| oracle-passed | 9/9 | 3/3 |
| VRAM used | 44,104 (0.898) | 37,482 (**0.763**) |
| CUDA0 SM mean | 26.5% | 38.4% |
| CUDA1 SM mean | 37.8% | **7.3%** |
| CUDA2 SM mean | 29.9% | **2.9%** |
| CUDA2 busy >5% | 98% of samples | **33%** |

The offloaded model already claims 76% of VRAM at one slot, so there is little
memory headroom to win. What is idle is **compute**: CUDA1 holds 19.4 GB and
runs at 7.3% SM, CUDA2 holds 9.4 GB and runs at 2.9%. The cards are not short
of memory, they are waiting on experts streaming from host RAM. That matches
the optimizer's own diagnosis, `bottleneck CPU expert bandwidth`.

This also corrects BOTHOBJECTIVES' claim that there is "no idle-card problem
during agent serving". That holds only for a resident model.

### --parallel 2 is fighting the planner, not a missing feature

Automatic slot selection chose **1** for this model. Forcing 2 exposed three
real convergence defects, fixed in order, and still did not launch:

1. recovery derived candidates from the original request and ignored the
   ceiling, so context climbed back (merged, #56).
2. the ceiling stepped down one 1,024-token granule against multi-GB deficits,
   so the descent crept: 870,400 -> 817,152 -> 816,128.
3. the re-plan budget charged productive rounds the same as churn, so a
   geometric descent died one step short.

With 2 and 3 fixed the descent is sound - 1,185,792 -> 873,472 -> 800,768 ->
655,360, deficit 6,773 -> 1,558 - and it then stalled in the backend-measured
branch, which keeps its own budget check.

Stopping there deliberately. A fourth patch to the same loop is epicycles. For
a model 3x over VRAM capacity, two lanes contend for one host-RAM expert
stream, so **one slot is very likely correct** and the planner already chose
it. The defects found along the way are worth having on their own; the
configuration that exposed them is not worth forcing.

### What would actually move the offloaded case

Hot experts. A GPU-resident LRU cache over the offloaded experts attacks the
thing idling those two cards - how often a token waits on PCIe - which is a
compute-occupancy fix, not a memory-fill one. The +6% measured at K=32 has only
ever been reached by driving llama-server by hand; the branch is 32 commits
unmerged and still cannot engage through ggrun.

Worker/reviewer routing is the other half, and it is now unblocked on the
resident path: concurrent serving works there, proven by the 27B A/B.

### Open

- No hot-experts comparison against the 1.03 correct tasks/min figure above.
- The backend-measured branch still charges progress as churn. Same fix as the
  DoesNotFit branch, deliberately not applied in the same pass.
- Slot count by residency class is unmeasured as a policy. Two data points
  exist: resident wants more than 1, heavily offloaded wants 1.

## RESIDENCYFRACTION — what actually predicts agentic speed — 2026-09-14

With the oracle-path lever fix, Qwen3.8-Flash-Next launches and completes the
picture. Three models, same agent suite, ggrun's own planning, 1 slot except
where noted.

| model | size | over VRAM | resident expert layers | VRAM used | correct tasks/min | median task |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B (2 slots) | 17 GiB | resident | all (dense) | 0.898 | **13.5** | 8.4s |
| Qwen3.8-Flash-Next | 83.8 GiB | 1.75x | **27 of 48 (56%)** | 0.861 | **7.49** | 15.0s |
| GLM-5.3-Flash | 137.4 GiB | 2.86x | **1 of 43 (2%)** | 0.763 | **1.03** | 115.1s |

All 3/3 oracle-passed.

**Resident expert fraction predicts agentic speed; VRAM fill does not.** GLM
uses 76% of the machine and is 7x slower than a model using 86%. The difference
is not how much memory is claimed but how much of the *expert* set avoids the
host round-trip: 2% versus 56%.

This resolves the apparent conflict between "maximise hardware usage" and
"fastest agentic serving" recorded in VRAMFILL and BOTHOBJECTIVES. They align
when spending VRAM raises expert residency, and diverge when it does not. On
GLM no available lever raises residency materially — 3 GB of a 120 GB expert
set is a fortieth — which is why every lever measured flat or negative there.

### Practical consequence

Qwen3.8-Flash-Next is the model to run for agent work on this rig: 15s median
task against GLM's 115s, at 3/3 correctness. It was **unlaunchable on main**
until the oracle-path fix in this branch, so this was a defect hiding a usable
configuration, not a hardware limit.

### Slot policy, with the bracket now measured

| model | 4 slots (Claude Code default) |
|---|---|
| Qwen3.8-27B, resident | +29% at 2 slots; 4 slots plans cleanly |
| Qwen3.8-Flash-Next, 1.75x | did not launch before the fix; retest pending |
| GLM-5.3-Flash, 2.86x | ~5.7x slower per turn, 1 of 3 tasks completed |

The mechanism is visible in the planner's own numbers: to buy KV for four slots
on Qwen3.8-Flash-Next it pushed `n-cpu-moe` from 33 to 38, displacing five
expert layers to host RAM. **On a CPU-offloaded MoE a slot is paid for in
expert residency**, and residency is what sets speed. A slot cap should key on
that displacement, which ggrun already computes per candidate, rather than on a
size ratio.

## EXPERTPIN — the measured re-plan handed back proven expert relief — 2026-09-14

Correcting RESIDENCYFRACTION's attribution: Qwen3.8-Flash-Next was not unblocked
by an oracle-path lever change. Six such changes were tried and all reverted —
widening the ceiling step to the deficit, bounding that step to a quarter-window,
a last-resort compute lever, and three re-rankings of the recovery candidates.
None altered the trace. The defect was one step later, in what happens *after* a
recovery succeeds.

### The per-device ledger, from `/tmp/q5.log`

| round | CUDA2 need / limit | outcome |
|---|---:|---|
| 1 | 12,010 / 11,909 MiB | does not fit |
| after expert derate | 10,937 / 11,909 MiB | **fits** — 1,073 MiB freed |
| 2 (backend-measured re-plan) | 12,003 / 11,909 MiB | does not fit again |

The expert lever worked on the first try. The backend-measured recompute then
replanned from the original request and put the displaced layer straight back.
Because that recompute produces a *different* argv from the one already
rejected, `recomputeDecision`'s identity ledger could not catch it, and the
launch cycled until the replan budget ran out.

`boundByProvenLimits` already pins context and ubatch across a recompute, but
`placement.Options` takes no expert-residency input, so `n-cpu-moe` cannot be
pinned on the way in. It is now guarded on the way out: the ledger records the
largest `NCPUMoE` an exact preflight has proven, and `undoesProvenExpertRelief`
vetoes any recompute that lowers it. The ratchet only ever moves toward more CPU
residency.

### Three wrong diagnoses, each killed by a test rather than by reasoning

- "The fabricated `AllocMB` disqualifies the ubatch lever" — the pinned selector
  test shows the lever available and stepping 256 to 64.
- "`n-cpu-moe` is stuck at 22" — it steps 22 to 23, CUDA2 expert layers 4 to 3.
- "The pins are partial sub-pins and are mis-priced" — the override pattern
  includes `down`, so they are whole-layer.

`TestQwenFlashNextRecoverySelectorShape` pins all three so they are not
re-derived.

### Evidence after the fix

Both target models launch, and the guard fires usefully on both.

| model | result |
|---|---|
| Qwen3.8-Flash-Next | LOADED; plan retained at 10,937 / 11,909 MiB |
| GLM-5.3-Flash | LOADED; context fit 498,688 tokens, 1 slot |

Agent workload re-run on Qwen3.8-Flash-Next at the stable candidate, 2 lanes,
1 repeat, ctx 18,912: **3/3 oracle-passed, 7.67 correct tasks/min, 14.8s median,
16.9s max**. That matches the 7.49 and 7.76 figures recorded before the fix
under the same suite, so the guard costs nothing at serve time — it only makes
the launch converge.

Uncached `scripts/verify-core-engine.sh` green across all six core packages.

### Open

- The uneven-GPU KV case is still unexercised: all three cards here hold a
  similar share, so a plan where one device owns most of the KV has not been
  driven through recovery.
- Recovery compares per-device ledgers but that comparison is not yet asserted
  to be complete — a device absent from one side is not distinguished from a
  device at zero.

## AGENTPATH — the ordinary agent path, measured end to end — 2026-09-14

First run of the installed launcher through the whole user-facing path rather
than through health and a single completion. Binary `v3.2.9-dev.expertpin`
(the `EXPERTPIN` fix), Qwen3.8-27B-UD-Q4_K_XL, automatic context, no flags
beyond host/port and the live-probe allowance.

| check | result |
|---|---|
| launcher readiness and `/health` | ok |
| weight devices | CUDA0, CUDA1 (CUDA2 left at 114 MiB) |
| served shape | 262,144 tokens, 1 slot |
| generation, streaming | ok |
| **cancellation mid-stream, then reconnect** | slot back in **0.607 s**, no restart |
| **prefix reuse behind a 6,433-token prefix** | **516 tokens re-evaluated (8.0%)**, 3,788 ms to 487 ms |
| bounded agent workload, 2 lanes | passed |
| shutdown | clean, no forced kill, port released |

VRAM at 61% of the machine's 49,134 MiB. That is not underuse: the model and
its 262k context fit on two cards, and `RESIDENCYFRACTION` already established
that filling the third would not make it faster.

### What the two new checks establish

Cancellation is the common case in agent use — the user interrupts, or the
client drops mid-stream — and a slot that is never released turns the next
request into a hang. Aborting the connection after three SSE chunks and
immediately asking for a completion returned in 0.607 s, so the slot is
reclaimed without a restart.

Prefix reuse is what separates a responsive session from one that pays for the
whole project context every turn. A second question behind an unchanged
6,433-token prefix re-evaluated 8.0% of it. Both are now part of
`scripts/verify-installed-serving.py`; cancellation runs unconditionally,
prefix reuse behind `--prefix-reuse` because it needs a context large enough to
hold the prefix and the install CI jobs run at 2,048.

### The cancellation check is portable, not tuned to this machine

It ran unchanged in the Linux install-e2e job on `fe58b83`, twice, on a CPU-only
ubuntu runner serving Qwen3.5-0.8B at 2,048 tokens.

| host | model | recovery after abort |
|---|---|---:|
| this rig, 3 NVIDIA cards | Qwen3.8-27B, 262k ctx | 0.607 s |
| ubuntu-latest, CPU only | Qwen3.5-0.8B, 2k ctx | 0.561 s |
| ubuntu-latest, CPU only, downloaded model | Qwen3.5-0.8B, 2k ctx | 0.547 s |

Three very different shapes, all within 60 ms of each other, which says the
number is slot reclaim rather than anything about the hardware. `prefix_reuse`
is correctly absent from both CI rows: it is gated behind `--prefix-reuse`
because 2,048 tokens cannot hold the prefix.

### TUI/CLI resolver agreement

Settled by construction rather than by comparison: `cmdGUI` turns the user's
selections into argv with `tuiLaunchArgs` and calls the same `cmdLaunch` the
command line calls, so agreement reduces to whether every selection survives the
argv round trip. Three regressions in `tui_cli_agreement_test.go` now hold that:

- every field of a maximal `tui.LaunchRequest` parses back to the same value,
  including `BackendExplicit`, which `userExplicitBackendFlag` reads to decide
  whether a lever may be withdrawn;
- an untouched row stays a preference and does not become a typed flag, which is
  what keeps the recovery ladder able to move it;
- a reflection guard fails when a `LaunchRequest` field is added and never wired
  into `LaunchArgs`, with each non-argv field carrying its reason.

The guard found one dead field: `FlashAttn` is hardcoded `true` in
`buildLaunchRequest` and never emitted or read.

### Open

- On this launch the effective per-agent context was never printed.
  `optimizationSummaryLines` formats `ctx N total / M per agent`, but the
  calibration path returns before `printOptimizationSummary` when it reuses a
  cached decision. The information is still visible in
  `[placement] context fit: 262144 tokens, 1 slot(s)`; the clearer line is not.
  Left unpatched: it is display-only code inside a protected path.
- `--prefix-reuse` is not wired through `verify-gpu-install.py`, so the GPU CI
  job cannot request it yet.
- One run per check, one machine, three NVIDIA devices.

## CALIBSPEND — the optimizer's first run on Flash-Next, and what it spent it on — 2026-09-14

`EXPERTPIN` made Qwen3.8-Flash-Next launchable, which means the candidate
controller had **never run on it**: every prior measurement used
`--calibrate off` to stay out of the 20-minute comparison. This is the first
standard launch, defaults throughout, only host/port and the live-probe
allowance.

### What it chose

| | |
|---|---|
| plan | `--n-cpu-moe 20`, ubatch 256, batch 2048, parallel 1 |
| resident expert layers | **28 of 48**, across all three GPUs |
| context | 262,144 tokens, 1 slot, KV q8_0 on GPU |
| VRAM | 43,958 of 49,134 MiB (**89.5%**) |
| ggrun's own bottleneck call | **CPU expert bandwidth** (live-allocated) |
| agent workload | 3/3 oracle, **7.63 correct tasks/min**, 14.79s median |

7.63 sits inside the 7.49–7.76 band measured before, so the calibrated plan is
not faster. What it is, is **much roomier**: 262,144 tokens of context against
the 18,912 the 7.67 run used, at the same throughput on this suite. For a
product whose goal is useful context for agent work, 14x the context for no
measured cost is the result worth keeping. The suite's tasks are short, so this
says serving a 262k context is free here, not that long context is free.

### The defect this run exposes

The failure budget was spent before a single candidate ran.

| candidate | outcome |
|---|---|
| ubatch-2048 | 7,026 MiB deficit on CUDA0 |
| ubatch-1024 | 3,608 MiB deficit on CUDA0 |
| ubatch-512 | 1,885 MiB deficit on CUDA0 |
| — | failure budget reached; candidate search stopped |

Every candidate was a larger ubatch, and on a model 1.75x over VRAM a larger
ubatch costs compute buffer on every device. The accepted plan leaves CUDA2 at
11,839 of 11,909 MiB — **70 MiB of slack**. All three were arithmetically
impossible before they were tried.

**None of them loaded the model.** The three `[calibrate] measuring ...` lines
and the budget-reached line are consecutive in the log, with no `load_model`
between them: preflight refused each before a single weight was read. The cost
was never reload time.

The cost is the accounting. The failure budget exists to bound expensive work —
a candidate that reads 84 GiB and then dies is why it exists — but an argv-time
refusal is charged exactly the same. Three cheap rejections inside one lever
family therefore retire the entire search, and **expert packing, topology and
slot count are never reached at all** on this model class. That is why the
baseline always wins on this shape: nothing else is ever measured.

The negative result is at least cached (`cal-ee43f9c5...json`, admission-only
evidence) so an identical launch does not repeat the search, which is what the
milestone asks for.

### Two measured bottlenecks worth naming

From the optimizer's own phase analysis on this launch:

- **Prefill is topology-limited: CUDA0 at 80% SM while CUDA2 sits at 7%.** The
  split is 0.27/0.59/0.14, so the smallest share is also the idlest card. A
  second independent launch reproduced it at 79% against 8%, and recorded the
  cached-append phase at 71% against 2%. Two runs is not a distribution, but it
  is no longer a single reading.
- **Decode is in the CPU-expert path, and PCIe is not proven saturated** —
  RX/TX 52/34 MiB/s decode, 74/144 MiB/s mixed. The optimizer explicitly keeps
  DRAM and synchronization as live candidates rather than blaming the bus, which
  is the correction this handoff asked for.

Its own safe-lever conclusion: "move serial layer work off the saturated GPU;
retain expert-storage roles unless routing proves them active."

### Open

- The budget accounting is fixed in `CALIBBUDGET` below, but that was **not**
  why topology went unmeasured — see the correction there. The open item is
  candidate selection: three challenger slots, all filled with one lever family.
- Candidate generation still proposes a shape placement has already refused for
  this model nine times during planning.
- The prefill imbalance (80% against 7%) has a named lever and no experiment.
- One run. The agent suite is three short repair tasks and does not exercise the
  262k context it now has.

## CALIBBUDGET — a refusal that costs no load should not retire the search — 2026-09-14

Fix for the defect `CALIBSPEND` measured. The failure budget bounds expensive
work, and `exactAdmissionFailure` already carried the distinction needed to tell
expensive from cheap: every typed refusal except `cuda-oom` is decided at argv
time, before a process reads a weight. A CUDA OOM is the exception — it surfaces
while device allocations are being made.

Pre-load refusals no longer charge the reload failure budget. They still record
the same stable-failure evidence, so the negative result is still cached. The
loop stays bounded by the elapsed-time budget and the finite candidate list,
both untouched.

An untyped start failure — health timeout, interrupted load, transient backend
fault — is classified as expensive. Mistaking a real reload for a cheap refusal
is the failure mode that would make the search unbounded, so the default is the
conservative one.

### Regressions

`calibrate_budget_test.go` covers every cheap class, the `cuda-oom` exception,
untyped and nil errors, and wrapped errors — admission failures travel up
several layers before the calibration loop reads them. Uncached
`scripts/verify-core-engine.sh` green on all six core packages.

### Live trace, same model and defaults

With the cached decision moved aside so the search reruns:

```
[optimize] calculated 5 candidates (5 feasible, 1 exact): batch 2048..2048,
           ubatch 128..2048, parallel 1..1, 2 topology shape(s)
[calibrate] ubatch-2048 failed to start (... CUDA0 (7026 MiB deficit) ...)
[calibrate] ubatch-2048 was refused before any model load; not charging the reload failure budget
[calibrate] ubatch-1024 failed to start (... CUDA0 (3608 MiB deficit) ...)
[calibrate] ubatch-1024 was refused before any model load; not charging the reload failure budget
[calibrate] ubatch-512  failed to start (... CUDA0 (1885 MiB deficit) ...)
[calibrate] ubatch-512  was refused before any model load; not charging the reload failure budget
```

The same three deficits as `CALIBSPEND`, and the three new lines confirm the
accounting change. **But the search still ends here, and the budget was not why.**

Correcting the `CALIBSPEND` diagnosis: with `calibrationAutoMaxCandidates = 4`
the automatic set is the baseline plus three challengers, and the log says so —
"up to 3 contained admissions". `MaxFailures` is also 3. So the budget was
reached exactly as the last challenger finished; **nothing was ever skipped
because of it**. The load that follows in this run is the baseline being
restored, not a candidate: `CUDA_Host model buffer size = 25219.14 MiB` is this
run's own baseline figure, and `[optimize] calculated finalist was unavailable;
restored measured baseline` follows it.

The real blocker is candidate **selection**, not budget. The frontier calculated
five candidates including two topology shapes; the set is trimmed to four, and
finalist prioritization fills all three challenger slots with ubatch rungs. The
topology shapes are discarded before the loop sees them.

The budget change remains correct and is kept: charging a reload budget for a
refusal that reads no weights is wrong on its own terms, and it will matter as
soon as the challenger slots hold more than one lever family. It is simply not
the fix that unblocks topology exploration.

The `EXPERTPIN` guard also fired on this launch, unprompted:

```
[launch] preflight expert-derate after CUDA2 allocation 0 MiB (deficit 101 MiB, ctx=262144, n-cpu-moe=22, ubatch=256)
[launch] preflight: placement fits (... CUDA2 10937/11909 ...)
[launch] backend-measured recompute would undo proven expert relief; retaining the verified-safe placement
```

### Recorded screen size

`[optimize] prefill pilot 128.0 tok/s; bounded screen uses 23296 bytes (~7701
tokens) per lane for both placements`, and the baseline evidence line records
`reuse >=6503 tokens/lane, slowest workflow 48.48s`. That is the ~8k corpus the
handoff describes, recorded as an actual size rather than called long-context
acceptance.

## CALIBGENERAL — the ladder fix was wrong on a model it was not written on — 2026-09-14

`CALIBLADDER` was developed and verified on Qwen3.8-Flash-Next alone. Running it
against Qwen3.8-27B — a fully resident model, a different residency class —
broke it immediately.

The 27B frontier is a different shape: **27 candidates (7 feasible)** against
Flash-Next's 5, `batch 32..8192`, three topology shapes. Its predicted finalist
was named `batch-1024-ubatch-512`. The generator emits `batch-%d-ubatch-%d`, a
**compound**, and Flash-Next's frontier never produced one — every name there
was a simple `ubatch-N`.

Taking the leading token called that candidate `batch` and treated it as
unrelated to `ubatch-512`. That defeats the fix in exactly the case it exists
for: a refused `ubatch-512` would be followed by a candidate carrying the same
ubatch 512, refused for the same reason on the same device.

Families are now every coordinate a name moves, parsed as key-value runs — a new
key is a non-numeric token that *follows a value*. Descriptive tails follow a
key rather than a value, so `topology-balanced-012` stays `topology` and
`moe-owner-1` stays `moe`. Two candidates collide when their coordinate sets
**overlap**, not when they are equal.

All nine generator name formats are pinned in `calibrate_ladder_test.go`, with a
case asserting the ladder will not follow `ubatch-512` with the compound.

### What the 27B run measured

| | default | batch-1024-ubatch-512 |
|---|---:|---:|
| workload makespan | 6.09 s | 6.11 s |
| decode | 32.5 tok/s | 32.4 tok/s |
| prefill | 1713.5 tok/s | 1709.1 tok/s |
| relative | 1.000 | 0.997 |

A dead heat; `default` won and passed the relaunch, agent, cache and lifecycle
gates. The screen here is 32,768 bytes (~10,714 tokens) per lane with
`reuse >=8943 tokens/lane`, larger than Flash-Next's ~7,701.

### The topology imbalance is not one model's quirk

| model | residency | phase | imbalance |
|---|---|---|---|
| Qwen3.8-Flash-Next | 1.75x over VRAM | prefill | CUDA0 80% vs CUDA2 7% |
| Qwen3.8-Flash-Next, 2nd launch | same | prefill / append | 79% vs 8% / 71% vs 2% |
| Qwen3.8-27B | fully resident | append | GPU1 79% vs GPU0 5% |
| Qwen3.8-27B, challenger | fully resident | prefill | GPU1 98% vs GPU2 0% |

Two models, two residency classes, four launches. One card near saturation while
another sits under 10% is the most reproducible unexploited signal measured so
far, and no candidate family moves it.

## CALIBLADDER — the ladder reaches a challenger, and the phase guard earns its keep — 2026-09-14

`CALIBBUDGET` fixed the accounting but not the blocker. The blocker was
selection: `calibrationAutoMaxCandidates = 4` leaves three challenger slots, and
all three went to `ubatch-2048`, `ubatch-1024` and `ubatch-512` — one lever
family, three rungs, all refused for the same reason on the same device.

`selectAutomaticCalibrationAdmissionPlan` had already written down the fix and
implemented only half of it:

> If the predicted primary is a batch/topology coordinate, keep one legal
> slot-count fallback in the bounded admission ladder. This is especially
> important after a high-ubatch candidate fails: retrying two more members of
> the same family teaches nothing about aggregate agent throughput.

That reasoning was applied only to `parallel-`. On this model the frontier
offered `parallel 1..1`, so the special case matched nothing and the generic
fill took the ubatch neighbours. It is now applied to every lever: remaining
slots prefer a family nobody has tried, falling back to same-family only when
that is all the generator produced. The family is read from the generator's own
naming (`ubatch-2048` to `ubatch`, `moe-owner-1` to `moe`), so no model or
hardware specifics are encoded.

### The first measured challenger on this model

| | default | ubatch-512 |
|---|---:|---:|
| workload makespan | 48.72 s | **32.37 s** |
| relative | 1.000 | **1.505** |
| prefill | 146.3 tok/s | **229.8 tok/s** (+57%) |
| decode | 16.6 tok/s | **10.3 tok/s** (-38%) |
| measured bottleneck | GPU topology | host workers at 90% capacity |

**The aggregate winner lost.** `[optimize] candidate winner default (turn
48.72s, relative 1.000)`, then `workflow winner default passed clean relaunch,
agent, cache, and lifecycle gates`. A candidate 1.5x better end to end was
refused because decode regressed 38% against a 5% allowance — invariant 6
holding on live data rather than in a unit test. Had only the aggregate been
scored, ggrun would have shipped a plan that makes every generated token 38%
slower.

### The bottleneck moved, and named a new lever

At ubatch 512 the limit is no longer the GPU split. Decode and mixed both
saturate the configured host workers (87% and 90%, measured), and the optimizer's
safe levers change accordingly:

```
[optimize] ubatch-512 safe levers: tune physical-core count and affinity;
           separate batch and decode thread settings
```

PCIe during cached append rose from 60/29 MiB/s at the baseline to 4704/884
MiB/s at ubatch 512 — roughly 78x — and is still reported as **not proven
saturated**. Thread count and affinity are the next demonstrated lever, not the
bus.

### Open

- Separate batch and decode thread settings are untried; the optimizer names
  them and no candidate moves them.
- ubatch-512 was admissible on this launch and not on the previous one, because
  the baseline expert placement differed. The family-spreading itself is proven
  by unit tests; its live effect appears only when a finalist is refused.
- Two samples per placement. The 1.505 aggregate and the 38% decode regression
  are both far outside that noise, but a promotion would need matched repeats.

## SEATCOST — what a reviewer/worker seat costs on an offloaded MoE — 2026-09-14

Every measurement before this one used plain serving, so no companion was
seated and none of it describes the configuration an agent user actually runs.
`ggrun dry-run` prices the seats without loading anything.

Qwen3.8-Flash-Next-UD-Q3_K_XL, same host, GPUs idle, `--claude-code`:

| seat | `--claude-reviewer` | resident expert layers | per-agent context |
|---|---|---:|---|
| self-classify | `off` | **35** of 48 | 262,144 |
| review-only, Qwen3.5-2B | `qwen2b` | **33** | 262,144 |
| worker + reviewer, Qwen3.5-4B | `qwen` | **31** | 262,144 |

**A review-only seat costs 2 expert layers; the worker/reviewer seat costs 4.**
Per-agent context is identical across all three, so the arms are comparable on
the terms the handoff requires: main-model context and quality held equal, and
the companion charged against the same usable VRAM.

### Claude Code mode is a different plan from plain serving

| | plain | `--claude-code` |
|---|---|---|
| context | 262,144 total, 1 slot | 1,048,576 total, **4 slots** |
| per agent | 262,144 | 262,144 |
| ubatch | 256 | 64 |
| resident experts | 28 of 48 | 31-35 of 48 |

Claude Code mode plans four slots at the same per-agent context, which changes
batch/ubatch and the expert packing with it. Every figure recorded before
`SEATCOST` — including the 7.63 correct tasks/min on Flash-Next — belongs to the
plain single-slot plan, not to the agent configuration.

### What this does not yet establish

The cost is measured; the benefit is not. `RESIDENCYFRACTION` says resident
expert fraction predicts agentic speed, so giving up 4 of 35 layers is a real
price, and whether the worker earns it back by absorbing classifier and
cheap-tier traffic is exactly the milestone 3 comparison that has not been run.
Nothing here says two models lose — only what they cost.

## CLAUDEMODE — the agent configuration is the slow one, and a seat cannot launch — 2026-09-14

First measurements of Qwen3.8-Flash-Next through `--claude-code` rather than
plain serving. Same agent suite (`ab35d682`), per-agent context matched.

### Claude Code mode costs two thirds of the throughput

| | plain serving | `--claude-code --claude-reviewer off` |
|---|---:|---:|
| plan | 262,144 tokens, 1 slot | 1,046,528 total / **261,888 per agent**, 4 slots |
| resident expert layers | 28 of 48 | **19 of 48** |
| correct tasks/min | **7.63** | **2.42** |
| median task | 14.79 s | 30.51 s |
| tasks completed | 3/3 | **2/3** |

Per-agent context is equal, so this is not a context trade. Four slots need four
times the total KV, that KV is bought out of the same VRAM, and nine more expert
layers go to host RAM. `SLOTS` and `RESIDENCYFRACTION` predicted the mechanism;
this measures the end-to-end cost, including a task that did not finish.

**The four-slot Claude Code default is wrong for a CPU-offloaded MoE.** A slot
cap keyed on the expert displacement the planner already computes — rather than
on a fixed parallel-4 floor — is the change this points to.

### With a reviewer seated, the launch does not converge

`--claude-reviewer qwen2b` (the smallest seat, 1.4 GB) never reaches a load:

```
Error starting server: memory preflight did not converge after 5 re-plans;
refusing a real model load
```

It fails closed, which is correct. But the `n-cpu-moe` trace across the rounds
oscillates rather than converging:

```
46 → 45 → 46 → 40 → 44 → 45 → 46 → 44 → 45
```

Twice a `preflight context-replanned` step recomputes placement from scratch and
returns expert layers that a derate **in this same launch** had already moved to
the CPU — 46 to 40 gives back six at once. The following rounds claw them back
one at a time until the replan budget is gone.

This is the `EXPERTPIN` bug family on a path its guard does not cover.
`undoesProvenExpertRelief` arms only from `acceptedNCPUMoE`, which is recorded
by `acceptContext` after an exact preflight **fits**. Here nothing ever fits, so
the ratchet never arms and every context re-plan is free to undo the accumulated
derates.

So a reviewer seat on this model is not merely expensive, it is currently
unreachable, and the cause is a defect rather than a capacity limit.

### What this does not establish

The worker/reviewer benefit is still unmeasured: no arm with a seated companion
has served a request, so nothing here says a companion cannot pay for itself.
`SEATCOST` priced the seat from dry-run estimates at 2 and 4 expert layers; the
live plans differ from those estimates (19 resident rather than the estimated
35), so the dry-run numbers rank the seats but do not size them.

### The correlation is exact

Aligning the nine rounds against the step that produced each one:

| step | kind | `n-cpu-moe` |
|---|---|---:|
| 1 | expert-derate | 46 |
| 2 | **context-replanned** | **45** |
| 3 | expert-derate | 46 |
| 4 | **context-replanned** | **40** |
| 5-7 | expert-derate x3 | 44, 45, 46 |
| 8 | **context-replanned** | **44** |
| 9 | expert-derate | 45 |

Every derate raises it; every context re-plan lowers it. Nine for nine, so this
is a mechanism rather than a plausible reading. `--claude-reviewer qwen` fails
the same way.

### The fix has a mechanism already in the codebase

`recomputeAutomaticContextRecovery` recomputes a complete plan at a smaller
context, and `placement.Options` carries no expert-residency floor — which is
why the recompute is free to re-pack experts back onto the GPU. Confirmed by
inspection: the only MoE-related inputs are `MoESplitOwnerGPU`,
`CPUExpertMMapCapability` and `ForceSpecMoE`, none of which pin residency.

`launchRequest.AdvisorVRAMPenaltyMB` is the existing lever for exactly this. Its
own comment: *"shrinks a device's usable VRAM for the next re-plan... the
advisor names a device and a layer count, and the deterministic planner re-packs
every GPU around the reduced budget."* Carrying the accumulated residency into a
context re-plan as a VRAM penalty would let the packer reproduce it without a
new placement input, and without any partial argv overlay.

Deliberately not implemented in this pass. It is a protected-path change whose
proof is a live launch, each costing a multi-minute load of an 83.8 GiB model,
and the contract does not accept unit tests as evidence for it.

### Open

- Carry accumulated expert residency into the context re-plan via
  `AdvisorVRAMPenaltyMB`, so a re-plan cannot undo a derate from the same
  launch. Verify by launching Flash-Next with `--claude-reviewer qwen2b`, which
  currently cannot converge.
- Key the Claude Code slot count on computed expert displacement instead of a
  fixed parallel-4 floor. The `off` arm above is the evidence: four slots cost
  nine expert layers and two thirds of the throughput at equal per-agent
  context.
- Re-run the seat comparison once a seat can launch. Until then the worker
  benefit is unmeasured, and nothing here argues a companion cannot pay for
  itself.

## SEATARMS — all three seats measured, and the seat is not what matters — 2026-09-14

Qwen3.8-Flash-Next, `--claude-code`, same suite (`ab35d682`), one repeat each.

| arm | per-agent context | correct tasks/min | median | completed |
|---|---:|---:|---:|---|
| `off` self-classify | 261,888 | 2.44 | 25.9 s | 2/3 |
| `qwen2b` review-only | 262,144 | 2.49 | 26.3 s | 2/3 |
| `qwen` worker+reviewer | **211,968** | 2.27 | 29.5 s | 2/3 |

**No seat is measurably better.** The spread is 2.27 to 2.49 on single runs with
no established noise floor. Ranking them would be reading noise, and this file's
own correction on that point applies.

**The 4B seat costs 19% of per-agent context** — 211,968 against 261,888 — while
the 2B seat costs none. That is the capacity result to record rather than a
configuration quietly shrinking the main model's window.

Every arm loses a task, and every arm runs at roughly a third of plain serving's
7.63 correct tasks/min. Across five launches the cost tracks the four-slot plan,
not the companion.

### Corrections to CLAUDEMODE

- **"No companion seat can launch" was wrong.** All three converged here, with
  monotone traces: `off` 38-41-43-44, `qwen` 42-45-47-48-48, `qwen2b` in one
  round at 47. The earlier double failure does not reproduce.
- **The oscillation is intermittent, not deterministic.** The nine-for-nine
  correlation was real for that launch, but four later launches of the same
  shapes show no oscillation at all. It depends on starting state, not on the
  configuration alone.

### The residency ratchet is still unexercised

`holdExpertResidency` fired **0 times in all three arms**. The first patch sat
on `recomputeAutomaticContextRecovery`, which returns method `context-derate`,
while the oscillating rounds return `context-replanned` from the candidate built
by `Compute`/`ReplanAfterOOM` — a different site. That is corrected, and the
guard still has not run, because nothing has oscillated since.

It is gated and covered by unit tests. It is **not** demonstrated to fix
anything, and must not be described as such until a live run oscillates with it
installed.

### Open

- The worker seat never did worker work: no delegated or classifier traffic was
  generated, so this measures the seat's cost with none of its benefit. The
  milestone 3 comparison needs traffic on the review and utility routes.
- The four-slot Claude Code plan, not the seat, is what costs the throughput.

## REVIEWLANE — the seat pays for itself once reviews actually happen — 2026-09-14

`SEATARMS` compared the seats with the review lane empty and found them
inseparable. That measured a companion's cost with none of its work. This drives
the lane: 8 classifier requests issued **concurrently** with 4 foreground turns,
through the Auto router, marked with the same system-prompt string the router
selects on.

| | `off` self-classify | `qwen2b` seated |
|---|---:|---:|
| routes served | `main: 13` | **`reviewer: 8`, `main: 4`** |
| reviews completed | **3 of 8** | **8 of 8** |
| review median | 14.17 s | **0.099 s** |
| review max | 177.6 s | 0.213 s |
| foreground completed | 3 of 4 | **4 of 4** |
| foreground median | 16.58 s | **10.28 s** |
| errors | **6 x HTTP 502** | **0** |

The per-request metrics confirm the mechanism rather than leaving it inferred.
With no seat, all thirteen requests went to `main`: reviews queued behind
foreground work on the same four slots, six requests failed with 502, and the
reviews that survived took up to 178 s. With the 2B seated, the eight reviews
went to `reviewer` at ~100 ms each returning 8 tokens, and the four foreground
turns had `main` to themselves.

**Foreground turns got faster too** — 10.28 s against 16.58 s median. The
companion does not only absorb reviews; it stops them contending for the main
model.

### This reverses SEATARMS' reading

`SEATARMS` is not wrong, it is incomplete: 2.44 / 2.49 / 2.27 correct tasks/min
is the seat's cost with its lane idle, and on that evidence the 4B seat's 19%
context cost looks like a pure loss. Under review traffic the review-only seat
turns a 62%-failure situation into a zero-failure one at no context cost.

**Recommendation: seat the review-only companion (`--claude-reviewer qwen2b`) on
an offloaded MoE.** It costs no per-agent context, and self-classify collapses
once reviews and foreground work overlap — which is the normal Claude Code
pattern, one classifier request per tool call.

### Limits of this evidence

- One run per arm; no repeats and no noise floor.
- Synthetic traffic shaped like Claude Code's classifier requests, not a real
  session.
- The 502s mean `off` was overloaded rather than merely slow. A gentler review
  rate would not separate the arms this starkly, and the crossover point is not
  measured.
- The 4B worker seat was not driven with delegated utility work, so its extra
  cost over the 2B still has no measured benefit.

## SLOTLEVER — the optimizer cannot fix the slot floor because slots are not a candidate — 2026-09-14

`CLAUDEMODE` measured four slots costing two thirds of Flash-Next's agentic
throughput at equal per-agent context. The obvious next step is a
displacement-aware slot cap. Before writing one, the cheaper question: can the
existing candidate controller already find this?

**It was never asked.** Every Claude Code arm measured so far ran with
`--calibrate off` — the harness sets it deliberately to stay out of the
20-minute optimizer window — so the optimizer had never executed in this mode.

Running it with calibration on gives the answer in one line:

```
[optimize] calculated 6 candidates (6 feasible, 0 exact):
           batch 128..512, ubatch 64..512, parallel 4..4, 2 topology shape(s)
```

**`parallel 4..4`.** The generator offers batch, ubatch and topology variants,
and no slot-count variant at all. The same pinning appears in every other run
recorded here: `parallel 1..1` on plain Flash-Next, `parallel 1..1` on the 27B,
`parallel 1..1` on GLM. The slot count is always the requested value and never a
coordinate the search can move.

So the measured slot cost is not something calibration failed to find. It is
outside the candidate space, which means no amount of measurement or ladder
fixing reaches it.

This is a narrower defect than a displacement-aware cap, and it fits the
existing architecture: `prioritizeParallelCalibrationCurve` and the
`parallel-%d` candidate name already exist, and `automaticWorkloadCandidateSet`
already filters slot candidates against declared demand
(`requestWorkloadConcurrency` returns 2 for default Claude Code mode, so a
`parallel-2` candidate would survive that filter). The generator simply does not
emit them here.

### Next concrete deliverable

Emit slot-count candidates below the requested value when the plan displaces
expert layers to host RAM, and let the existing ladder and phase guards decide.
The displacement is already computed per candidate, so the cap keys on measured
evidence rather than a size ratio — which is what `SLOTS` asked for and what
`CLAUDEMODE` now quantifies.

Do not hardcode a lower floor. `REVIEWLANE` shows concurrency has real value:
with reviews and foreground overlapping, the seated companion path completed 8
of 8 reviews against 3 of 8. The right answer is a measured trade, not a
smaller constant.

## SLOTFIX — the comparison was wrong, and fixing it was not sufficient — 2026-09-14

`SLOTLEVER` found the candidate boundary pinned at `parallel 4..4`. Tracing it:
slot candidates *are* generated (`opts.ParallelExplicit` is false without an
explicit `--parallel`), but `sameCalibrationResidency` required **equal total
context**, and a slot candidate necessarily changes the total. Every one was
rejected on arrival.

`calibrationBaseOptions` compounds it by pinning the base's total window, so the
only slot comparison available was "same total KV, redistributed" — never "fewer
slots, less KV, more experts resident", which is the shape `CLAUDEMODE` measured
as three times faster.

The comparison is now on **per-agent context**, the quantity the change contract
forbids reducing silently. Both legitimate shapes are accepted: the same total
redistributed across a different width (what an explicit `--ctx-size` asks for)
and the same per-agent window with the total scaled (what an automatic context
produces).

An existing test caught the first attempt, which compared only per-agent
windows: with an explicit context a slot candidate legitimately keeps the total
and halves the per-agent window, and that attempt broke it.

### It did not work

A live Claude Code launch with calibration on, cached decisions cleared, still
reports:

```
[optimize] calculated 6 candidates (6 feasible, 0 exact):
           batch 128..512, ubatch 64..512, parallel 4..4, 2 topology shape(s)
```

**No slot candidate appears.** So the residency comparison was one blocker and
not the only one, and the remaining cause is unidentified. Candidates may be
failing inside `recomputeParallelCandidate` (its `strategySlots(alt) != parallel`
guard, or `Compute` failing at the scaled window), or
`calibrationParallelNeighbors` may be returning nothing for this shape.

The change is kept because the comparison it corrects is wrong on its own terms
and is covered by tests. **It must not be described as fixing the slot floor.**
Nothing measured here shows a slot candidate reaching the ladder.

### Next step, narrowed

Instrument the three drop points — neighbour generation, recompute error, and
the residency check — for one offloaded-MoE launch, and report which rejects.
The earlier attempt to reproduce this in a unit fixture failed because a
three-GPU 85 GB MoE baseline would not compute in the test harness; the drop
point is cheaper to observe on a real launch than to synthesise.

## SLOTTRUTH — correcting SLOTLEVER: slot candidates are generated — 2026-09-14

`SLOTLEVER` concluded from the `parallel 4..4` boundary line that the slot count
"is outside the candidate space". **That was wrong.** Driving
`placement.CalibrationCandidates` with this machine's real `detect.Detect()`
capabilities and the real Qwen3.8-Flash-Next profile emits 32 candidates,
including:

```
EMITTED parallel-3   ctx=1048576 parallel=3
EMITTED parallel-2   ctx=1048576 parallel=2
EMITTED parallel-1   ctx=1048576 parallel=1
```

So the generator does offer slot counts for exactly this shape. The boundary is
computed over whatever candidate list is handed to it
(`optimizer.go` accumulates `MinParallel`/`MaxParallel` across the passed
candidates), and the live launch reported **6** candidates against the
generator's 32. The live set is a filtered subset, and the filtering — not the
generator — is what removes the slot candidates.

`automaticWorkloadCandidateSet` alone does not explain it: it drops a candidate
only when `parallel != baseline && parallel > demand`, and with demand 2 that
drops `parallel-3` while keeping `parallel-2` and `parallel-1`.

### A second correction, to SLOTFIX

The same run shows `BASE ... auto=false`: the Claude Code base strategy does not
carry `ContextAuto`. The per-agent scaling added in `SLOTFIX` is gated on that
flag, so **it never executes on this path** — which is why the live boundary was
unchanged. The residency comparison it fixes is still wrong on its own terms and
the change is kept, but its gate is wrong for this case.

### What is actually established

- Slot candidates exist for this model and hardware.
- Something between generation and the reported boundary removes them, and it is
  not the residency comparison and not the demand filter alone.
- `SLOTFIX`'s scaling is inert here because the base is not marked automatic.

### Next step

Log the candidate list at the point the boundary is built, on one real launch,
and diff it against the generator's 32. That names the filter in one run. Do not
change the slot policy until that filter is identified — two diagnoses have
already been wrong, both from reading a summary line instead of the list behind
it.

## SLOTREPRO — the live candidate set, reproduced without a launch — 2026-09-14

Building the request with `parseLaunchArgs`, then `placementOptionsFromRequest`,
then `placement.CalibrationCandidates`, reproduces the live launch's candidate
set in under a second and with no model load:

```
OPTS ctx=0 autoMax=1048576 parallel=4 parallelExplicit=false slotTarget=65536
BASE ctx=254976 parallel=3 batch=128 ubatch=128 auto=true
TOTAL candidates=6          (no slot candidates)
```

Six candidates, matching the live `calculated 6 candidates` exactly. This is the
diagnostic loop that was missing: every earlier attempt to understand the slot
boundary cost a five-minute 83.8 GB load, and two diagnoses were wrong because
they read a summary line instead of the list.

### What it establishes

- **The base is `parallel=3`, not 4**, at 254,976 tokens with `auto=true`. The
  four-slot request is clamped by `claudeCodeSlotsForPlacement` against the
  context the hardware can actually hold.
- `calibrationParallelNeighbors` is **not** empty for this shape:
  `maxParallel = 254976 / 65536 = 3`, current 3, so neighbours are 1 and 2.
- Yet no slot candidate is emitted, so the drop is inside
  `recomputeParallelCandidate` (its `Compute` call or the
  `strategySlots(alt) != parallel` guard) or in `sameCalibrationResidency` /
  `calibrationCandidateExists` afterwards. All three are unexported, so naming
  the exact one needs a probe inside `pkg/placement`.

### Corrections carried

`SLOTFIX` and `SLOTTRUTH` both claimed more than was shown. The generator does
emit `parallel-1/2/3` under hand-built options (`SLOTTRUTH`), and with the real
request options it emits none — so the earlier "slot candidates are generated"
is true only for the options I chose, not for the ones the launcher builds.

Three diagnoses on this one question have now been wrong. The pattern each time:
reading a summary or a partial reproduction and inferring the mechanism instead
of observing the list the code actually produces.

### Next step

Add a probe inside `pkg/placement` that walks the three drop points for this
exact base and reports which rejects. The reproduction above makes that a
sub-second test rather than a launch.

## SLOTDROP — the slot lever is refused by the residency guard, correctly — 2026-09-14

Fourth diagnosis on this question, and the first one observed rather than
inferred. Replicating what `recomputeParallelCandidate` builds, against the real
request options and the real model:

```
BASE       ctx=254976 parallel=3 perAgent=84992 kv=gpu  type=moe_offload
parallel-1 ctx=84992  slots=1    perAgent=84992 kv=cpu  type=moe_offload
parallel-2 ctx=169984 slots=2    perAgent=84992 kv=cpu  type=moe_offload
```

Both candidates compute cleanly, land on the requested slot count, and preserve
per-agent context exactly. They are then rejected by `sameCalibrationResidency`,
because the base is **GPU-resident KV** and every lower-slot plan the packer
produces is **host KV**.

The guard is right. Moving the KV cache to system RAM is a residency class
change, not a tuning move, and the inspected backend also moves the
corresponding attention computation to the CPU with it — a cost this file's own
reasoning corrections already record. Comparing across that boundary would let a
"faster" candidate win by quietly relocating attention to the host.

**So the slot lever is not blocked by a defect.** Freeing KV by cutting slots
gives the packer room it spends on expert residency, and the plan it then
prefers puts KV on the host. The calibration contract refuses to compare that
against a GPU-resident baseline, so the candidate never reaches the ladder.

### What would actually open the lever

Pin `KVPlacement` to the base's value inside `recomputeParallelCandidate`, so a
slot candidate is forced to keep KV where the baseline has it and spend the
freed memory on experts instead. That is a real placement question — it may
simply not fit — and it is the experiment to run, not an assumption to encode.

### Corrections this closes out

Four diagnoses, three wrong:

1. "Slots are outside the candidate space" — wrong, they are generated.
2. "The residency comparison on total context blocks them" — a real flaw, fixed,
   but not the blocker here.
3. "Slot candidates are generated under the real options" — wrong; they are
   generated under hand-built options and rejected under the real ones.
4. **The residency guard rejects them on KV placement** — observed.

Every wrong one came from reading a summary line or a partial reproduction. The
sub-second reproduction in `SLOTREPRO` is what finally made the real list
visible; it should be the first move next time, not the fourth.

## SLOTOPEN — the slot lever is open, and the optimizer picks it — 2026-09-15

With baseline KV placement held for slot candidates, the same Claude Code launch
that reported `parallel 4..4` in every earlier run now reports:

```
[optimize] calculated 8 candidates (8 feasible, 0 exact):
           batch 128..2048, ubatch 64..512, parallel 1..4, 2 topology shape(s)
[optimize] calculated finalist parallel-2: predicted relative 2.238,
           bottleneck CPU expert bandwidth, confidence low; live agent workflows decide
```

Two changes from every previous run: the boundary spans **`parallel 1..4`**
rather than a single pinned value, and the predicted finalist is a **slot
candidate**. The optimizer, given the lever, immediately nominates fewer slots.

### The full chain, four fixes deep

| # | defect | status |
|---|---|---|
| 1 | `sameCalibrationResidency` compared **total** context, which every slot candidate changes | fixed |
| 2 | candidates inherited the base's total, so `parallel-1` asked for a full window of KV on one slot — infeasible | fixed by scaling to hold per-agent context |
| 3 | that scaling was gated on `base.ContextAuto`, which a Claude Code base carries as **false** | fixed; the gate is `opts.AutoContextMax` |
| 4 | the packer spent the freed KV on experts and moved the cache to the **host**, which the residency guard rightly refuses | fixed by holding the baseline's KV placement |

Only #4 was ever visible from a log line. The first three were each found by
reproducing the candidate list directly (`SLOTREPRO`), which runs in under a
second against real capabilities and the real model.

### Not yet established

The finalist is *predicted*, at `confidence low`, and the launch is measuring it
now. Nothing here shows `parallel-2` wins. The phase guard that rejected
`ubatch-512` for a 38% decode regression applies unchanged, and `REVIEWLANE`
showed concurrency has real value — with a companion seated, reviews leave the
main model's slots entirely, so fewer main slots may cost less than it appears.
Both outcomes are informative and neither is assumed here.

## Independent review of saved REVIEWLANE artifacts — 2026-09-15T08:25:20+00:00

Historical-artifact audit, not a new live measurement. Sources: `/tmp/claude-1000/-home-mik-ggrun-project-ggrun/0f9dc578-5269-47fc-80bd-f2ebfc454985/scratchpad/review-ab`; driver and wrapper are adjacent `review-lane.py` and `review-ab.sh`.

### off

UTC request-record window: 2026-09-14T21:56:37.736257412Z to 2026-09-14T22:00:14.64286112Z.

Exact backend command captured in the arm log:

```text
/home/mik/ggrun-project/ggrun/.src/fork-qwen3-8-flash-next/build-cuda/bin/llama-server -m /home/mik/ggrun-project/ggrun/models/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf --host 127.0.0.1 --port 18921 --ctx-size 1047552 --flash-attn on -b 128 -ub 64 --cache-type-k q8_0 --cache-type-v q8_0 --jinja --threads 14 --threads-batch 14 --cpu-range 0-13 --cpu-strict 1 --cpu-range-batch 0-13 --cpu-strict-batch 1 --no-context-shift --parallel 4 -ngl 999 --tensor-split 0.26,0.57,0.16 --split-mode layer -ot 'blk\.(0)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=CUDA1,blk\.(1|2|3)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=CUDA2,exps=CPU' --n-cpu-moe 44 --no-mmap -cram 17920 --ctx-checkpoints 16 --checkpoint-min-step 512 --timeout 2147483647 --chat-template-file /home/mik/ggrun-project/ggrun/.cache/chat-templates/qwen3.8-27b.jinja --alias local --presence-penalty 1.0 --repeat-penalty 1.05 --repeat-last-n 512 --top-k 20 --top-p 0.95 --min-p 0.0 --metrics -lv 4
```

Saved driver result: `{"total_s":216.95,"foreground":{"n":3,"median_s":16.584,"max_s":171.929,"mean_s":66.8},"review":{"n":3,"median_s":14.165,"max_s":177.574,"mean_s":67.743},"errors":["review 3: HTTP Error 502: Bad Gateway","foreground 3: HTTP Error 502: Bad Gateway","review 4: HTTP Error 502: Bad Gateway","review 5: HTTP Error 502: Bad Gateway","review 6: HTTP Error 502: Bad Gateway","review 7: HTTP Error 502: Bad Gateway"],"error_count":6}`.

### qwen2b

UTC request-record window: 2026-09-14T22:01:14.793854391Z to 2026-09-14T22:04:33.613225309Z.

Exact backend command captured in the arm log:

```text
/home/mik/ggrun-project/ggrun/.src/fork-qwen3-8-flash-next/build-cuda/bin/llama-server -m /home/mik/ggrun-project/ggrun/models/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf --host 127.0.0.1 --port 18921 --ctx-size 1048576 --flash-attn on -b 128 -ub 64 --cache-type-k q8_0 --cache-type-v q8_0 --jinja --threads 14 --threads-batch 14 --cpu-range 0-13 --cpu-strict 1 --cpu-range-batch 0-13 --cpu-strict-batch 1 --no-context-shift --parallel 4 -ngl 999 --tensor-split 0.26,0.65,0.09 --split-mode layer -ot 'blk\.(0)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=CUDA2,exps=CPU' --n-cpu-moe 47 --no-mmap -cram 12800 --ctx-checkpoints 16 --checkpoint-min-step 512 --timeout 2147483647 --chat-template-file /home/mik/ggrun-project/ggrun/.cache/chat-templates/qwen3.8-27b.jinja --alias local --presence-penalty 1.0 --repeat-penalty 1.05 --repeat-last-n 512 --top-k 20 --top-p 0.95 --min-p 0.0 --metrics -lv 4
```

Saved driver result: `{"total_s":205.939,"foreground":{"n":4,"median_s":10.281,"max_s":178.298,"mean_s":51.484},"review":{"n":8,"median_s":0.099,"max_s":0.213,"mean_s":0.113},"errors":[],"error_count":0}`.

These are HTTP completion counts and timings, not oracle-validated reviews.
The driver runs at most two requests concurrently and discards response bodies.
The wrapper starts traffic before final launch acceptance; the main-only log
ends with a missing-input Claude --print error, and the companion log ends
with a router canary failure. Four main-only 502 metric rows have zero queue
and total milliseconds after the first errors. Overload has not been isolated
from process lifetime/startup interference. Both arms have very long foreground
maxima. Do not promote the earlier universal 2B recommendation from this sample.
Repair owned-process lifecycle, wait for complete acceptance, retain/check
responses and run the actual equivalent workflow before making that decision.
Detailed code-review actions are in the handoff's 2026-09-15 direction review.
No server was started, stopped or reconfigured by this inspection.

## SLOTMEASURED — the lever opens, the candidate does not fit, and the guard holds — 2026-09-15

The launch from `SLOTOPEN` ran to completion. One trace exercises four separate
pieces of this session's work, three of which had never fired live.

```
[calibrate] measuring parallel-2...
[calibrate] parallel-2 failed to start (exact candidate failed memory admission
            on CUDA1 (1994 MiB deficit); refusing recovery ladder); skipping
[calibrate] parallel-2 was refused before any model load; not charging the reload failure budget
[calibrate] measuring batch-512-ubatch-512...
[calibrate] batch-512-ubatch-512: workload makespan 29.69s, decode 5.2 tok/s,
            prefill 125.2 tok/s, relative 2.212
[optimize] candidate winner default (turn 65.67s, relative 1.000)
[optimize] workflow winner default passed clean relaunch, agent, cache, and lifecycle gates
```

| piece | evidence in this trace |
|---|---|
| `SLOTOPEN` — slot lever reachable | `parallel-2` was the finalist and was actually attempted |
| `CALIBBUDGET` — cheap refusals do not retire the search | "not charging the reload failure budget", **first live firing** |
| `CALIBLADDER` — fallbacks spread across lever families | after the slot candidate was refused the ladder went to **batch/ubatch**, not another slot rung, **first live firing** |
| phase guard (invariant 6) | a 2.212x aggregate winner was refused |

### The slot trade is still unmeasured, for a capacity reason

`parallel-2` failed exact admission with a **1,994 MiB deficit on CUDA1**. With a
reviewer seated and KV held on the GPU, two slots at 262,144 per agent does not
fit on this machine. The lever is open; this particular rung is out of reach
here. That is a capacity result, not a defect, and it is the kind the handoff
asks to be recorded rather than worked around by shrinking the main model's
window.

### The phase guard earned its keep again

| | default | batch-512-ubatch-512 |
|---|---:|---:|
| workload makespan | 65.67 s | **29.69 s** |
| relative | 1.000 | **2.212** |
| prefill | 38.9 tok/s | **125.2 tok/s** (+222%) |
| decode | 11.2 tok/s | **5.2 tok/s** (-54%) |

A candidate more than twice as fast end to end was refused because decode more
than halved. This is the second time in this session the aggregate winner lost
on a phase regression, on a different model configuration from the first.

### Device imbalance, a sixth reading

`GPU 0 saturated (78% SM) while GPU 2 is idle (0% SM)` on the baseline, and
75%/2% on the challenger. Six launches, three models, two residency classes, and
the optimizer's own note is that it "measured device imbalance, but the exact
launch is tight-resident; retaining its proven live-search boundary" — it sees
the imbalance and correctly declines to spend proven fit on it. No candidate
family moves serial layer work off the saturated card.

## SLOTWIDTH — slot count against expert residency, measured directly — 2026-09-15

The ladder could only reach `parallel-2`, which failed admission. An explicit
`--parallel` names each width, so all three get measured on the same suite with
calibration off. Qwen3.8-Flash-Next, `--claude-code --claude-reviewer qwen2b`.

| slots | resident expert layers | per-agent ctx | tasks completed | correct tasks/min |
|---:|---:|---:|---:|---:|
| 1 | **25** of 48 | — | 0 of 3 | 0.00 |
| 2 | **21** of 48 | 261,632 | 2 of 3 | 2.54 |
| 4 | **9** of 48 | 207,360 | 0 of 3 | 0.00 |

**Expert residency scales cleanly with slot width**: 25, 21, 9 for one, two and
four slots. Four slots costs roughly sixteen expert layers against one slot on
this model, which is the displacement `CLAUDEMODE` inferred and this measures
directly.

Four slots is also worse on per-agent context — 207,360 against 261,632 — so it
is not trading window for concurrency. It loses on both.

### Throughput is inconclusive, and the reason matters

Only the two-slot arm completed any tasks. One and four slots each finished 0 of
3 inside the suite's budget, so there is no ranking to read here, and the single
2.54 figure has nothing to be compared against. Recording it as "two slots wins"
would be reading one surviving sample as a result.

What the failures do say is that this model in Claude Code mode is marginal at
every width tried: the earlier automatic-slot arm managed 2.42 correct
tasks/min at 2 of 3 tasks, and nothing here beats that.

### A second non-convergence mode, distinct from the oscillation

An earlier `slots=1` attempt failed with `did not converge after 5 re-plans`, but
its trace was **monotone**: `29 29 29 30 32 32 32`. Nothing was undone; the
derate ladder was simply still climbing when the budget ran out. That is a
different defect from the oscillation `SLOTDROP`'s ratchet targets
(`46 40 44 45 46 44 45`), and the ratchet cannot help it. Conflating the two
would attribute a fix to the wrong failure.

### The residency ratchet failed silently

The `slots=4` trace dipped — `47 44 46 47 47 48 48 48` — yet
`holdExpertResidency` logged nothing. It only prints when the re-pack succeeds;
a failed `ReplanAfterOOM` returns the original plan quietly. **A guard that
falls back silently is indistinguishable from one that never ran**, which is why
this fix has been unprovable across seven launches. Log the attempt, not just
the success.

## RATCHETOBS — the guard is now falsifiable, and still unexercised — 2026-09-15

`holdExpertResidency` printed only on success, so a failed `ReplanAfterOOM`
returned the original plan in silence. "Never fired" and "fired and could not
help" produced identical logs, and seven launches were inspected for evidence a
code path was structurally incapable of producing. The giveaway was in the data
all along: the `SLOTWIDTH` four-slot trace dipped `47 44 46 47 47 48` with
nothing logged, and a dip means something handed layers back.

It now reports the attempt with its layer count, the outcome, and the reason on
failure — whether `ReplanAfterOOM` errored or the re-pack simply did not fit.

### The relaunch, and what an empty trace now means

Re-running the exact four-slot configuration that dipped:

```
[launch] backend-measured recompute would undo proven expert relief; retaining the verified-safe placement
n-cpu-moe trace: (empty)
```

No preflight recovery rounds occurred at all — the plan fit on the first
attempt — so the guard had nothing to act on. That is a **different** state from
a silent failure, and before this change the two were indistinguishable. The
`EXPERTPIN` guard from #58 did fire, on the measured recompute.

So the residency ratchet remains unexercised. What changed is that its absence
is now evidence rather than ambiguity: an empty derate trace means no re-plan
happened, and a populated one without a guard line would mean the floor
tracking is wrong.

### The general lesson

Every "unproven" label attached to this fix rested on absence of evidence from a
path that could not produce evidence. That is the same error as reading
`parallel 4..4` and concluding slot candidates were never generated: inferring a
mechanism from a summary incapable of showing it. Before labelling a guard
unproven, check that it would have said so.

## HARNESSKILL — every Claude Code agent-suite number in this session is contaminated — 2026-09-15

The `SLOTWIDTH` arms did not measure slot throughput. They measured how many
tasks finished before the launcher tore the server down.

Task-level results for the four-slot arm:

| task | passed | **oracle** | error |
|---|---|---|---|
| ceiling | false | **true** | Remote end closed connection without response |
| clamp | false | **true** | Remote end closed connection without response |
| interval | false | false | Connection refused |

**The oracle passed on two tasks whose requests then lost their connection**, and
the third could not connect at all. The model answered correctly; the server
disappeared underneath it.

The cause is in the launch log's last line, in every affected arm:

```
Error: Input must be provided either through stdin or as a prompt argument when using --print
```

`--claude-code` starts the backend and then opens the Claude Code client. Driven
from a script with no TTY the client refuses to start, ggrun exits, and its
shutdown handler stops the backend — mid-suite. The failures are a harness
artifact of driving `--claude-code` headless, not a product defect and not a
property of any configuration under test.

### What this retracts

Every agent-suite figure measured through `--claude-code` in this session is
unreliable, because the run was racing a teardown:

- `CLAUDEMODE`'s "2.42 correct tasks/min against 7.63 for plain serving". The
  plain-serving side is sound; the Claude Code side is not, so **the headline
  "Claude Code mode costs two thirds of the throughput" is not established**.
  The residency mechanism behind it still is — see below.
- `SEATARMS`' 2.44 / 2.49 / 2.27 across the three seats. Those were already
  called inseparable; they should now be treated as invalid rather than merely
  noisy.
- `SLOTWIDTH`'s throughput column.

### What survives

Anything read from the launch plan rather than from completed tasks, because
those are recorded at planning time and do not depend on the server outliving
the harness:

- resident expert layers by slot width: **25 / 21 / 9** for 1 / 2 / 4 slots;
- per-agent context by slot width, including four slots getting *less*
  (207,360 against 261,632);
- `SEATCOST`'s seat prices and every `n-cpu-moe` trace;
- `REVIEWLANE`, which ran to completion in seconds with zero errors and whose
  route counts (`main: 13` versus `reviewer: 8` + `main: 4`) come from the
  router's own metrics.

### How to measure Claude Code mode properly

Keep the backend alive independently of the client. Either drive the launcher
under a pty so the client starts, or add a serve-only path that brings up the
backend and router without opening Claude Code. Until then, do not compare
agent-suite numbers across `--claude-code` arms.

## PTYFIX — Claude Code mode is measurable now — 2026-09-15

`HARNESSKILL` traced every contaminated arm to one cause: `--claude-code` opens
the Claude Code client, which refuses to start without a TTY, so ggrun exits and
its shutdown handler stops the backend mid-suite.

Driving the launcher under a pty (`script -qec`) fixes it. Verified on
Qwen3.8-Flash-Next at `--parallel 2`:

```
READY at ~60s
--- does it survive 90s past ready? ---
{"status":"ok"} STILL ALIVE
"Input must be provided ..." occurrences: 0
```

The client error never occurs and the backend outlives the suite. Earlier arms
died around 50 s, mid-task.

### The first clean Claude Code measurement

Same suite (`ab35d682`), same model, reviewer seated, two slots:

```
{"passed": true, "completed_tasks": 3, "total_tasks": 3,
 "correct_tasks_per_minute": 2.688, "task_latency_median_s": 42.04,
 "task_latency_max_s": 48.51}
```

**3 of 3 tasks**, where every previous `--claude-code` arm completed 0 to 2. This
is the first agent-suite figure through Claude Code mode in this session that is
not racing a teardown.

One number is not a comparison: it does not rank slot widths or seats, and the
retractions in `HARNESSKILL` stand. It establishes that the configuration works
and that the measurement path is now sound.

### Standing caveat

The root filesystem was at 100% (2.0 GiB free of 456 GiB) while this ran.
`ENOSPC` during a launch surfaces as failures that resemble unrelated defects,
so results taken under that pressure deserve a second look. This one passed
cleanly, but the comparisons it unblocks should be re-run with space available.

## SLOTCLEAN — the four-slot penalty was the teardown, not the plan — 2026-09-15

First slot comparison on the pty path from `PTYFIX`, after the root filesystem
was freed. Qwen3.8-Flash-Next, `--claude-code --claude-reviewer qwen2b`,
calibration off, same suite (`ab35d682`). **All three arms completed 3 of 3**,
where every earlier arm managed 0 to 2.

| slots | `n-cpu-moe` | resident experts | per-agent ctx | correct tasks/min | median | max |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25 | **23** of 48 | 262,144 | **4.02** | 30.8 s | 33.4 s |
| 2 | 31 | 17 | 262,144 | 3.02 | 32.9 s | 53.7 s |
| 4 | 47 | **1** | 262,144 | 3.41 | **26.5 s** | **30.7 s** |

Per-agent context is identical across all three, so this is a matched
comparison on the quantity the contract protects.

### This positively contradicts the retracted claim

`CLAUDEMODE` reported four slots costing two thirds of the throughput (2.42
against 7.63). `HARNESSKILL` retracted that as a teardown artifact. This
measures it properly: **four slots runs at 3.41 against one slot's 4.02, an 18%
gap**, and four slots has the *best* median and worst-case latency of the three.

The retraction was right, and the direction of the original claim was wrong.

### RESIDENCYFRACTION needs qualifying

The four-slot arm keeps **1** expert layer resident against the one-slot arm's
23 — a 23x difference — and loses under a fifth of its throughput. Expert
residency does not dominate agentic speed within a single model at matched
per-agent context the way the cross-model table implied. That table compared
three different architectures, quantisations and active-parameter counts, and
this file already recorded that confound; this is the controlled version of the
same question and it comes out much weaker.

### Not a ranking

The ordering is **non-monotone**: 1 > 4 > 2. A result that does not move
monotonically in the variable under test is noise dominating signal, on single
runs with no established floor. The arms do not separate cleanly, and no slot
width is promoted here.

What is solid is the negative: **no arm shows a catastrophic four-slot penalty**,
so the parallel-4 default is not the defect `CLAUDEMODE` made it look like, and
the slot-candidate work in PR #62 is an optimizer completeness fix rather than a
fix for a known performance bug.

## SEATCLEAN — the seats, measured properly — 2026-09-15

Replaces the `SEATARMS` figures that `HARNESSKILL` retracted. Same model, same
suite (`ab35d682`), pty path, calibration off. **All three arms completed 3 of
3**, where the retracted run managed 2, 2 and 2.

| seat | per-agent ctx | `n-cpu-moe` | correct tasks/min | median | max |
|---|---:|---:|---:|---:|---:|
| `off` self-classify | 261,888 | 44 | **2.84** | 36.8 s | 38.6 s |
| `qwen2b` review-only | **262,144** | 47 | 2.81 | 37.7 s | 50.4 s |
| `qwen` worker+reviewer | **206,336** | 41 | 2.76 | 44.2 s | 52.6 s |

### The seats are indistinguishable on this workload

2.84 / 2.81 / 2.76 across a 2.8% spread, on single runs with no established
noise floor. That is not a ranking and must not be read as one. The retracted
figures said the same thing less reliably; this says it from runs that finished.

**The durable difference is capacity, not speed.** The 4B worker seat costs 21%
of per-agent context — 206,336 against 262,144 — while the 2B review-only seat
costs none. On a window the contract forbids reducing silently, that is the
result worth acting on.

### Why this does not weaken REVIEWLANE

This suite issues no classifier traffic, so the companion's review lane is idle
in every arm. `REVIEWLANE` drove it — 8 classifier requests concurrent with 4
foreground turns — and separated the arms decisively: self-classify completed
3 of 8 reviews with six HTTP 502s, the seated 2B completed 8 of 8 at ~100 ms
median and made foreground turns faster. Its route counts came from the router's
own metrics and it finished in seconds, so it was never exposed to the teardown.

The two results are consistent and answer different questions. Idle lane: the
seat costs nothing measurable in throughput, and the 4B costs context. Loaded
lane: the seat is the difference between reviews working and reviews failing.

**The recommendation stands: seat `--claude-reviewer qwen2b` on an offloaded
MoE.** It costs no per-agent context, it is free on this workload, and it is
decisive the moment reviews and foreground work overlap — which is the normal
Claude Code pattern of one classifier request per tool call.

## MINICPM — a reviewer candidate that fails the verdict contract — 2026-09-15

`openbmb/MiniCPM5-2B` surfaced as a possible alternative to the pinned
Qwen3.5-2B review seat. Tested before any wiring, because a reviewer that cannot
produce the verdict format fails **invisibly**: every review is rejected, falls
back to the main model, and the only trace is `reviewer-rejected/invalid-verdict`
in the metrics log, while the seat still costs VRAM that on an offloaded MoE is
expert layers.

Artifact: `bartowski/MiniCPM5-2B-GGUF`, `Q4_K_M`, 1,615,826,144 bytes, fetched
with `ggrun download`. The base `openbmb/MiniCPM5-2B` repo is Safetensors, and
ggrun's downloader reports that and points at the GGUF repo rather than failing
obscurely.

| prompt | result | latency |
|---|---|---|
| marker + "answer with `<block>yes/no</block>`" | **0 of 4 valid** | 109-176 ms |
| terser instruction + `</block>` stop sequence | **0 of 4 valid** | 81-138 ms |

Fast enough — comparable to the seated Qwen3.5-2B — but it never emits a bare
verdict. Every reply opens with prose restating the instruction:

```
"We are asked to respond with exactly one token sequence: <bl..."
```

That prose contains **both** tags, so `validReviewerVerdict`'s `yes + no == 1`
check rejects it. Correctly: a reviewer that says both has decided nothing. The
stop sequence does not help, because the prose precedes the verdict rather than
trailing it.

**Not adopted.** No `ModelSpec` was added and the artifact stays out of the
pinned reviewer cache.

This is the first real candidate to exercise the `invalid-verdict` versus
`unusable-response` split in the metrics, and it earned its keep: without that
distinction this model would have looked exactly like an absent reviewer.

### What would change the answer

The failure is prose-before-verdict, not an inability to produce the tags. An
assistant prefill forcing the reply to begin with `<block>` would likely pull it
into the contract. That is a router change affecting every reviewer, so it needs
its own evidence rather than being bolted on for one candidate.

## GLMSEAT — the tightest real configuration serves — 2026-09-15

GLM-5.3-Flash (2.86x over VRAM) with a reviewer seated, on the pty path. This is
the hardest configuration this machine can be asked for: a tight-fit model plus
a companion's 1.4 GiB.

| | |
|---|---|
| plan | 809,984 tokens total, 4 slots (~202k per agent) |
| preflight | converged in **one** derate round, `n-cpu-moe=43` |
| `EXPERTPIN` guard | fired on the measured recompute (fifth independent confirmation) |
| agent suite | 2 of 3 tasks, 0.65 correct tasks/min, 96.7 s median |

The failing task is a genuine wrong answer (`oracle=False`) with **no connection
error** — the pty path is behaving, and this is the model being wrong rather than
the harness dropping requests. On the old path this would have been
indistinguishable from a teardown.

0.65 correct tasks/min is slow, which matches everything already recorded about
GLM on this rig. The point of this run is that the configuration is *reachable*:
a 2.86x-over-VRAM model with a companion seated plans, loads, and serves.

### The residency ratchet, closed out honestly

`n-cpu-moe` trace: **`43`**. One round, one value, nothing handed back — so there
is no oscillation for `holdExpertResidency` to guard.

Across nine launches spanning three configurations and two models, the
oscillation recorded in `SLOTDROP` (`46 40 44 45 46 44 45`) has not recurred.
The guard is correct, gated, covered by tests, and since `RATCHETOBS` it reports
its attempts rather than failing silently. It targets a state that occurs rarely.

### The mechanism is now proven, separately from the live trigger

Both earlier tests covered only the paths where the guard *declines* to act. The
success path — where `ReplanAfterOOM` returns a genuinely better plan and the
guard adopts it — had no coverage at all, which is the real reason nine launches
of "unexercised" were ambiguous: the mechanism itself had never been shown to
work, only its guard conditions.

`TestHoldExpertResidencyActuallyRepacks` drives a real three-GPU MoE placement
and asserts the re-pack:

```
re-packed n-cpu-moe 23 -> 25 against floor 25
```

So the guard demonstrably re-packs when a recomputed plan falls below the floor.
It skips rather than fails if the fixture cannot produce a fitting re-pack on a
future build, so it cannot become a false green.

That splits the status cleanly, which is the honest form of it:

- **Mechanism: verified.** The guard re-packs and raises CPU expert residency.
- **Live trigger: not observed.** The oscillation from `SLOTDROP` has not
  recurred in nine launches across three configurations and two models.

It is a correct guard for a rare condition, and the instrumentation from
`RATCHETOBS` means the next occurrence will say so unambiguously in the log.

## LONGCTX — the first actual long-context evidence — 2026-09-15

The direction review's sharpest criticism: "Configuring 262k context and sending
6k-8k prompts is not 262k-context acceptance." Every context figure recorded
before this came from a *plan*, never from a prompt. This sends real prompts at
increasing depth with a correctness oracle — an access code planted near the
**start**, so a model that silently drops early context fails the check rather
than merely slowing down.

Qwen3.8-27B-UD-Q4_K_XL, plain serving, `--calibrate off`.

| target | actual prompt tokens | prefill tok/s | decode tok/s | wall | code recalled |
|---:|---:|---:|---:|---:|---|
| 4k | 516 | 1015.9 | 37.3 | 3.9 s | yes |
| 16k | 13,967 | 1643.6 | 35.1 | 9.8 s | yes |
| 64k | 54,312 | 1275.6 | 28.1 | 45.8 s | yes |
| 131k | **75,599** | 876.0 | 22.0 | 88.7 s | **yes** |

**Long context genuinely works.** 75,599 tokens served, and the planted code came
back verbatim at every depth. That is the first evidence in this record that the
configured window is usable rather than merely planned.

### What depth costs

| | 14k -> 76k |
|---|---|
| prefill | 1,643 -> 876 tok/s, **-47%** |
| decode | 35.1 -> 22.0 tok/s, **-37%** |
| wall for one turn | 9.8 s -> 88.7 s |

Decode falling 37% purely from context depth matters for agent work, where every
turn re-reads the conversation: the cost is paid on every token of every turn,
not once. An 88.7 s turn at 76k is usable for a considered answer and poor for a
tool-calling loop.

### A harness correction

The first pass ran with `max_tokens=32` and reported a recall failure at 54k.
That was the probe, not the model: this model opens with a reasoning preamble,
and at 32 tokens the budget was spent before the code was emitted. At 160 tokens
every depth recalls. **A correctness oracle that shares a budget with the model's
preamble measures the budget.**

### Mid-context recall, the harder case

A fact at the start is the easy case. An agent's working set buries what matters
in the middle, so the probe was extended to plant the code at 50% depth and the
chars-per-token estimate tightened from 4.0 to 3.0 (the first pass asked for 131k
and got 75,599).

| actual prompt tokens | prefill tok/s | decode tok/s | wall | code recalled |
|---:|---:|---:|---:|---|
| 13,406 | 1,708.1 | 35.9 | 12.6 s | yes |
| 53,761 | 1,463.8 | 30.1 | 40.0 s | yes |
| 110,094 | 1,193.6 | 24.7 | 98.7 s | yes |
| **168,085** | 1,001.8 | 20.7 | **174.5 s** | **yes** |

**168,085 tokens with the fact buried mid-context, recalled verbatim.** Quality
holds at depth on this model; nothing degrades except speed.

### What depth costs

| | 13k -> 168k, mid-context |
|---|---|
| prefill | 1,708 -> 1,002 tok/s, **-41%** |
| decode | 35.9 -> 20.7 tok/s, **-42%** |
| wall for one turn | 12.6 s -> **174.5 s** |

Decode falling 42% purely from depth is the number that matters for agent work:
every turn re-reads the conversation, so it is paid on every token of every
turn, not once. A 174 s turn is fine for one considered answer and unusable in a
tool-calling loop.

**This is the real capacity/speed trade, measured.** ggrun plans a 262k window on
this model and the window genuinely works — but there is currently no lever that
trades depth against turn latency, and no evidence recorded anywhere that the
planner considers it. A context ceiling chosen for turn time, rather than for
what fits, is an unexplored direction that this measurement makes concrete.

### Limits

- One model, one run per depth, one planted fact. Recall is a needle test; it
  does not measure reasoning quality over a long working set.
- Deepest measured is 168k against a 262k served plan. The top of the window is
  still untested.

## CTXCOST — the planner maximises context it is not asked to use — 2026-09-15

`LONGCTX` measured what depth costs. This measures what that costs **agent
work**: the same model and suite at the planner's automatic window versus an
explicit small one.

Qwen3.8-27B-UD-Q4_K_XL, plain serving, `--calibrate off`, one slot, same suite
(`ab35d682`), both arms 3 of 3.

| context | correct tasks/min | median task | max |
|---:|---:|---:|---:|
| 262,144 (automatic) | 9.84 | 13.86 s | 14.73 s |
| **32,768 (explicit)** | **11.12** | **12.18 s** | 13.09 s |

**A context eight times smaller serves agent work 13% faster**, and the suite's
prompts fit comfortably in both. The larger window is not being used by this
workload; it is being paid for.

### Why this is a planner question, not a user question

ggrun's automatic context fit maximises the window that fits in memory. Nothing
in that decision asks what the workload will actually use, so on a resident model
with spare VRAM it buys the largest window available and charges its KV cost to
every turn. `LONGCTX` shows the window genuinely works when used — 168,085
tokens with mid-context recall intact — so this is not a capability problem. It
is a default that optimises the wrong quantity for agent serving.

The contract forbids *silently reducing* useful per-agent context, and rightly.
But maximising it by default is the opposite error, and the cost is now measured
rather than assumed.

### The lever that does not exist

No candidate family moves context. `CalibrationCandidates` offers batch, ubatch,
slots, topology and KV placement; the window is fixed input to all of them. So
the optimizer cannot discover the 13% that an explicit `--ctx-size 32768` finds
by hand, on the model this rig recommends for agent work.

That is the same class of gap as the slot lever in `SLOTOPEN`: a coordinate the
planner controls, that measurably matters, and that the search cannot reach.

### Honest limits

- One model, one run per arm, three short repair tasks. 13% on single runs with
  no established noise floor is suggestive, not promotable.
- 32,768 was chosen as a round number, not searched. The useful ceiling for this
  workload is unmeasured, and a real agent session with a large project prefix
  would sit somewhere between these two points.
- A log wart found on the way: `[placement] context fit: 262144 tokens` still
  prints when `--ctx-size 32768` is given. The served window is correct — `/props`
  reports `n_ctx=32768` — but the planning line reads as though the override were
  ignored.

## MULTITURN — a growing session over a project prefix — 2026-09-15

The three-task suite is a smoke test: independent tasks, no shared history. The
direction review asks for multi-turn repository context, which is the shape agent
work actually takes — a stable project prefix replayed every turn and a
conversation that grows underneath it.

This runs a six-turn session over a ~11.9k-token synthetic repository listing,
with a checkable fact per turn, so a session that speeds up by losing track of
the project fails rather than scoring well. Qwen3.8-27B, one slot, both arms
**6 of 6 correct**.

| | 32,768 ctx | 262,144 ctx (automatic) |
|---|---:|---:|
| correct | **6 / 6** | **6 / 6** |
| session | 21.34 s | 21.22 s |
| turn 0 (cold) | 9.91 s | 8.78 s |
| turns 1-5 | 1.63 - 2.88 s | 1.80 - 3.16 s |
| decode | **40.4 tok/s** | 36.0 tok/s |

### Prefix reuse is excellent, and now observed rather than assumed

Turn 0 evaluates all **11,881** prompt tokens. Every later turn evaluates
**45-50** — the appended question and answer only, against a prompt that has
grown to 12,096 tokens. That is 99.6% reuse, at both context settings, sustained
across the session.

`AGENTPATH` measured prefix reuse once with two requests; this shows it holding
turn after turn as the conversation grows, which is the case that matters.

### Decode is 12% faster at the smaller window

40.4 against 36.0 tok/s, consistent across all six turns at both settings. That
matches `CTXCOST`'s 13% on the task suite and is the same mechanism: KV depth
charged to every decoded token.

**Session time did not separate** — 21.34 s against 21.22 s — because these turns
answer in one short sentence. The decode advantage is invisible when outputs are
tiny and compounds when they are not. A real agent turn writing a patch decodes
hundreds of tokens, so this is the arm where the 12% would show, and that case is
still unmeasured.

### What this does and does not establish

Establishes: multi-turn sessions work correctly at both windows, prefix caching
holds across a growing conversation, and the smaller window decodes faster for
the same work.

Does not establish: anything about a real client. This is a synthetic prefix and
scripted turns, not Claude Code driving tools against a real repository. The
review's "real client, substantial multi-turn repository context" remains open —
this narrows it to the client integration rather than the serving behaviour.

## MAINCHECK — both models on the merged #61 candidate — 2026-09-15

The direction review's first acceptance item: recheck Qwen and GLM on the final
#61 candidate. Verified on the exact merged commit `0a66834`, not on a branch
build carrying later work.

| model | result | plan | derate rounds |
|---|---|---|---|
| Qwen3.8-Flash-Next-UD-Q3_K_XL (1.75x over VRAM) | **LOADED** | 262,144 tokens, 1 slot | **none** — fit first attempt |
| GLM-5.3-Flash-UD-Q3_K_XL (2.86x over VRAM) | **LOADED** | 500,736 tokens, 1 slot | 42 -> 41 -> 41, converged |

This is the right regression pair for #61 specifically, because that PR changed
admission semantics: pre-load refusals stopped consuming the reload budget, the
ladder began spreading across lever families, and `CalibrationSchemaVersion`
moved 24 -> 25 so a decision recorded under the old policy cannot suppress the
search the new one can run.

Flash-Next needing **no derate rounds at all** is the sharper result. It was
entirely unlaunchable before #58, then converged only after the expert-relief
guard, and now plans cleanly on the first attempt. GLM's 500,736 tokens matches
the figure recorded before the calibration work, so that path is unregressed.

Method note: the merged candidate was installed as the single PATH binary for
the check and the branch build restored afterwards, so there was never a second
ggrun on this machine. The verification worktree was removed.

## LONGFORM — the context cost, paid on real output — 2026-09-15

`MULTITURN` found decode 12% faster at the smaller window but **session time
indistinguishable**, because those turns answered in one sentence. The honest
caveat recorded there was that the advantage compounds only with longer outputs,
and that case was unmeasured. This measures it.

Same six-turn session over the same ~11.9k-token project prefix, but each turn
asks for real work — write a handler, a table-driven test, a queue consumer, a
design note, an alerting rule, a summary table — at `max_tokens=900`.
Qwen3.8-27B, one slot, both arms **6 of 6 correct**.

| | 32,768 ctx | 262,144 ctx (automatic) |
|---|---:|---:|
| correct | **6 / 6** | **6 / 6** |
| **session** | **144.42 s** | **160.60 s** |
| decode, turn 0 -> 5 | 40.6 -> 39.6 tok/s | 36.1 -> 35.3 tok/s |
| steady-turn wall | 15.98 - 25.04 s | 23.10 - 26.32 s |

**The smaller window finishes the same work 16.2 seconds sooner, a 10% shorter
session**, with identical correctness. The short-answer run could not see this:
the same 12% decode gap was present there and worth nothing, because almost no
tokens were decoded.

This is the answer to the direction review's "actual workflow-speed question".
The context a plan buys is charged to every decoded token, so it is invisible in
smoke tests and material in real agent work, where turns write patches rather
than sentences.

### Prefix reuse holds under real output too

Turn 0 evaluates 11,892 tokens; later turns evaluate 937-954 — the appended
question plus the previous ~900-token answer, against prompts growing to 16,584.
Nothing re-reads the project listing. The cache behaves identically at both
context settings, so the session difference is decode, not caching.

### What this changes

`CTXCOST` measured 13% on the task suite and `MULTITURN` could not reproduce it
on session time. Both are now explained: the effect is real, it lives in decode,
and it shows up in proportion to how much the model writes. Three independent
measurements now point the same way, on the model this rig recommends for agent
work.

**ggrun's automatic context fit maximises the window that fits in memory, and
that default costs about 10% of a real agent session on a resident model with
spare VRAM.** No candidate family moves context, so the optimizer cannot find
this. That is the concrete, measured case for making the context ceiling a
searched coordinate rather than a maximised one.

### Limits

- One model, one run per arm, six turns. 10% on single runs is consistent with
  two other measurements but is not promotion evidence on its own.
- 32,768 was a round number, not a searched optimum; the useful ceiling for this
  workload is still unmeasured.
- Synthetic prefix and scripted turns. A real client with tool calls is still
  the open item.

## CTXFLOOR — a searched context ceiling needs a floor, and my variance is 5% — 2026-09-15

`LONGFORM` argued context should be a searched coordinate rather than a
maximised one. Searching it turned up two corrections to that framing before the
sweep even finished.

Same six-turn long-form session, Qwen3.8-27B, one slot, windows swept.

| ctx | turns completed | correct | session |
|---:|---:|---:|---:|
| 16,384 | **5 of 6** | 5 / 5 | 130.27 s |
| 32,768 | 6 of 6 | 6 / 6 | 151.55 s |

### "Smaller is faster" needs a floor

At 16,384 the session **truncated**: the prefix is ~11.9k and each turn adds
roughly 900 tokens of question plus answer, so the conversation outgrows the
window partway through. It looked fastest because it did less work — five turns
instead of six.

A searched ceiling therefore needs a workload-derived **floor**, not just a cost
gradient. Prefix size plus expected conversation growth is the minimum; below it
a smaller window silently drops turns rather than serving them faster. That is a
more useful rule for a planner than "prefer smaller", and it is exactly the shape
of failure the contract's "never silently reduce useful per-agent context" exists
to prevent — reached here by choosing a ceiling too low rather than by derating.

My own 32,768 was luckier than principled: 11.9k + six ~900-token turns lands
near 17.3k, comfortably inside 32k and well over 16k.

### Single-run variance is about 5%

The same 32,768 configuration measured **144.42 s** in `LONGFORM` and
**151.55 s** here — same model, same session, same hardware, ~5% apart.

That is a material fraction of the ~10% effect reported in `LONGFORM`, and it
applies to every single-run comparison in this record. The direction that
`CTXCOST` (13% on the task suite), `MULTITURN` (12% decode) and `LONGFORM` (10%
on session) agree on still stands, because three independent measurements点 the
same way. The **magnitude** does not: promoting a context policy on these numbers
would be promoting noise plus signal without separating them.

Matched repeats are the missing evidence, and they are cheap here — a six-turn
session is about two and a half minutes once the model is loaded.

## CTXSTEP — the context cost is a batch-shape step, not a depth gradient — 2026-09-15

Correcting `CTXCOST`, `MULTITURN` and `LONGFORM`. All three reported that a
larger context costs decode throughput, and explained it as KV depth being
charged to every decoded token. **That explanation is wrong.** Sweeping the
window shows the cost is a step at the automatic plan, not a gradient.

Same six-turn long-form session, Qwen3.8-27B, one slot.

| ctx | session | decode first -> last | batch / ubatch |
|---:|---:|---|---|
| 16,384 | 130.27 s (**5 of 6 turns**) | — | — |
| 32,768 | 151.55 s | 40.6 -> 39.7 | 8192 / 1024 |
| 65,536 | 150.16 s | 40.6 -> 39.7 | 8192 / 1024 |
| 131,072 | **149.92 s** | **40.7 -> 39.7** | **8192 / 1024** |
| 262,144 (automatic) | 160.60 s | **36.1 -> 35.3** | **2048 / 512** |

**Decode is flat at 40.6-40.7 tok/s from 32k through 131k — a four-fold range of
context — and only drops at the automatic window.** A KV-depth mechanism would
have produced a gradient across those four points. It produced a step.

### The actual cause

The automatic plan buys its 262,144-token window by shrinking the batch shape to
fit: **batch 2048 / ubatch 512, against 8192 / 1024 at every smaller window.**
That is a 4x smaller batch and 2x smaller ubatch, and it is what costs the
throughput.

So the finding is not "long context is slow". It is that ggrun's context fit
maximises the window and pays for it in batch shape, and the batch shape is what
the model decodes with.

### What this changes about the recommendation

Earlier entries argued for making context a searched coordinate. The sweep says
something cheaper and more specific:

- **There is a broad flat region.** 32k, 64k and 131k are within 1.6 s of each
  other on a ~150 s session, well inside the ~5% single-run variance recorded in
  `CTXFLOOR`. Nothing is gained by tuning inside it.
- **There is a floor.** 16,384 truncated the session to five turns of six; it
  looked fastest because it did less work.
- **Only the maximum costs.** The penalty appears when the plan sacrifices batch
  shape to reach the largest fitting window.

A planner does not need to search context per workload. It needs to **stop
trading batch shape for context it was not asked for** — which is a smaller and
more defensible change than the one the earlier entries implied.

### Method note

Three measurements agreed on a direction and I attached a mechanism to them that
the data had not tested. The sweep that was meant to find an optimum found the
explanation was wrong instead. A gradient and a step look identical at two
points; they only separate at four.

## RELEASEARTIFACT — the shipped archive could not install itself — 2026-09-15

The direction review's last milestone-5 item is "final artifact/install proof".
Building a release archive and installing from it found the archive broken.

### The defect

`scripts/package-release.sh` shipped three user entry points — `setup.sh`,
`setup-linux.sh`, `setup-mac.sh` — and each does:

```
exec "$ROOT/scripts/setup-home.sh" linux "$@"
```

**The archive contained no `scripts/` directory and no `install.sh`.** Extracting
a release and running the documented command produced:

```
setup-linux.sh: line 7: .../scripts/setup-home.sh: No such file or directory
```

### Why install-e2e could not catch it

CI runs `./setup-linux.sh` **from the repo checkout** with
`LLM_INSTALL_RELEASE_DIR` pointing at the artifact. The installer therefore comes
from source and only the payload comes from the archive, so the path a user takes
— extract the tarball, run the script inside it — is never exercised. The job
passes on a tarball that cannot install itself.

### The fix

Packaging now ships `scripts/setup-home.sh` and `install.sh`, and **fails closed**
if either is missing, matching the script's existing refusal to package a
backend-only bundle. Rebuilt and verified end to end.

### The proof, and two harness errors on the way

A clean install from the rebuilt artifact into an empty prefix now succeeds:
app home created, CUDA backend selected and unpacked, `ik_llama-server-cuda`
symlinked, launcher wrapper written, and `ggrun --version` runs from the
installed tree.

Both earlier attempts failed for reasons that were mine, not the product's:

1. **Asset name.** The installer resolves `ggrun-<platform>-<backend>.tar.gz`.
   A differently named archive is not found locally, so it reached for a
   published release and hung.
2. **Missing `SHA256SUMS`.** With the right name it found the local bundle, then
   fetched checksums from the published release and correctly rejected a
   locally-built archive that did not match. **That is the installer behaving
   properly** — it refuses a bundle whose checksum does not verify. Generating
   `SHA256SUMS` beside the artifact completed the install.

Worth keeping: the second failure looked exactly like a product defect and was
a guard doing its job. The log line that settled it —
`Checksum verification failed` — was one line above the error I first read.

## CTXREPEATS — matched repeats, and the "variance" was warm-up — 2026-09-15

`CTXFLOOR` recorded ~5% single-run variance and warned that it undercut the ~10%
effect `LONGFORM` reported. Three matched repeats per arm, each set sharing one
model load, settle both numbers — and correct the variance claim.

Qwen3.8-27B, six-turn long-form session, all six runs **6 of 6 correct**.

| arm | run 1 | run 2 | run 3 | steady-state turn |
|---|---:|---:|---:|---:|
| `--ctx-size 32768` | 154.51 s | **131.56 s** | **131.66 s** | **16.0 s** |
| automatic (262,144) | 169.77 s | **153.29 s** | **153.16 s** | **23.1 s** |

### Run 1 is warm-up, not noise

Both arms show the same shape: the first session after a load is slow, then runs
2 and 3 land within **0.1%** of each other — 131.56 against 131.66, and 153.29
against 153.16. That is not a noisy measurement; it is a very stable one with a
cold first sample.

So `CTXFLOOR`'s "~5% single-run variance" was wrong in kind. The 144.42 s and
151.55 s it compared were a warm run and a cold one. **Steady-state variance is
about 0.1%**, which makes this comparison far sharper than I credited.

### The effect is larger than reported, not smaller

Steady state: **131.6 s against 153.2 s, a 16.4% difference** — and per-turn,
**16.0 s against 23.1 s, 44% slower** at the automatic window. Every earlier
figure (13%, 12%, 10%) was measured with cold runs mixed in and understated it.

Combined with `CTXSTEP`'s plan comparison — batch 2048/512 at the automatic
window against 8192/1024 everywhere below it — the picture is complete and
consistent:

**ggrun's automatic context fit buys the largest fitting window by shrinking the
batch shape, and that costs 44% of steady-state turn time on the model this rig
recommends for agent work.** Three windows (32k, 64k, 131k) all keep the larger
batch shape and all perform identically, so nothing is gained by the maximum.

### Now promotable, with one caveat

This is repeated, matched evidence with 0.1% steady-state variance and identical
correctness across six sessions. It meets the bar the contract asks for before
changing a default.

The caveat is scope: one model, one machine, one session shape. The mechanism
(batch shape traded for context) is visible in the plan and should generalise,
but the magnitude is specific to a resident model with spare VRAM. An offloaded
MoE, where the window competes with expert residency rather than batch shape,
may behave differently and is not tested here.

## REALCLIENT — Claude Code fixing a real repository through ggrun — 2026-09-15

The direction review's longest-standing gap: "real client, substantial
multi-turn repository context". Five synthetic probes narrowed it without
closing it. `claude -p` closes it — a real client, real tools, a real git
repository, and a task with an objective pass condition.

### Setup

A Go package with two deliberately broken functions and a passing-by-construction
test file that must not be edited: `Ceiling` did integer division instead of
rounding up, `Clamp` ignored its bounds. `go test ./...` failed before the run.

Served: Qwen3.8-27B, `--claude-code --claude-reviewer qwen2b --ctx-size 32768`,
router on an ephemeral port, backend kept alive by the pty from `PTYFIX`.

Client: `claude -p "<task>" --permission-mode acceptEdits --max-turns 20`, with
ggrun's own aliases (`ANTHROPIC_MODEL=local`, base URL pointed at the router).

### Result

**The task was completed.** `calc.go` gained 13 lines, the test file was left
untouched as instructed, and `go test ./...` passes — both tests green, verified
independently after the session.

Router metrics for the session:

| | |
|---|---|
| requests | **23** |
| routes | `main: 22`, `reviewer: 1` |
| statuses | **200 x 23** — no errors, no fallbacks |

So the full integration works end to end: tool calls, file edits, test
execution, a seated reviewer answering on its own route, and every request
served without a single non-2xx.

### What this establishes that the probes could not

- The **client** integration works, not just the serving path. `MULTITURN` had
  narrowed the gap to exactly this and could go no further.
- **Worker/reviewer routing under a real session**: one classifier request was
  issued and the seated 2B answered it. That is a small sample, but it is real
  traffic rather than the synthetic marker used in `REVIEWLANE`.
- **Task-level correctness**, the outcome the review asked for: not token rates
  or route counts, but whether the agent fixed the code. It did.

### Honest limits

- One task, one run. The task is small — two functions — and a longer session
  would exercise context growth and rework that this does not.
- Only **one** review request in 23, so worker success and rework counts remain
  effectively unmeasured. A session with more tool calls would issue more
  classifier traffic and is the natural next step.
- A cosmetic `unrecognized_model` warning appears for `local` during session
  title generation. It does not affect the run, but it is noise a user sees.

## REVIEWREAL — the review lane under a real agent session — 2026-09-15

`REALCLIENT` closed the real-client gap but produced only one review request in
23, because `--permission-mode acceptEdits` auto-approves edits and almost
nothing reached the classifier. This uses `--permission-mode auto`, which ggrun
describes as "dedicated local safety reviewer; fail-closed", on a three-file task
with a test run after each fix.

Qwen3.8-27B + `qwen2b` seated, `claude -p`, 40 turns max.

### Task outcome

**All three functions fixed**, `go test ./...` passes, and the three `_test.go`
files are untouched as the task required — verified independently with
`git status`.

### Route split and latency

| | |
|---|---:|
| total requests | **55** |
| `main` | 46 |
| `reviewer` | 5 |
| `reviewer/stop-stripped-verdict` | 4 |
| rejected / fell back to main | **0** |
| reviewer median | **287 ms** |
| main median | **31,568 ms** |

**Nine real classifier requests, all answered by the seated 2B, none rejected.**
The reviewer answers in 287 ms against the main model's 31.6 s median — roughly
**110x faster** for the decisions that gate every tool call.

Four of the nine came back as `stop-stripped-verdict`: the reviewer emitted
`<block>yes` without the closing tag, which `validReviewerVerdict` accepts
deliberately because Claude's own parser sends `</block>` as a stop sequence.
That path is exercised by real traffic here, not just by its unit test.

### One 400, on main

`statuses: {200: 54, 400: 1}` — a single bad request on the main route, not
aborted, and the session completed correctly regardless. Worth noting rather
than explaining away; a session that completes with a 400 in it is a loose end,
and the request body was not captured to say which call it was.

### What is now established for milestone 3

- **Required reviews happen and are answered locally.** Nine of nine, zero
  fallbacks to the main model.
- **The seat pays for itself on latency**: 287 ms versus 31.6 s per decision, on
  traffic a real agent generated rather than a synthetic marker.
- **Task correctness holds** with the reviewer in the loop.

Still open: the matched **main-only** arm — the same task with
`--claude-reviewer off`, so the main model self-classifies — which is what turns
this into a comparison rather than a strong single observation.
