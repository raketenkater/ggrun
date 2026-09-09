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
| PERF-2, PERF-6, PERF-7 | Implemented as baseline plus one calculated finalist per launch, with a measured prefill pilot, identical budget-scaled cold+append scenarios, two samples, concurrent generation, mixed foreground traffic, lifecycle gates, delayed promotion, and a reusable baseline-won result. Stable coordinate-local failures now accumulate and advance the bounded frontier instead of freezing it. | Add longer branch/replay and long-context hardware acceptance; quantify noise on public hardware |
| PERF-10 | Automatic legal slot neighbors use complete re-placement and useful per-agent context. Agent-parallel declares at least two runnable turns; automatic challengers wider than declared demand are dominated and skipped, while explicit maintenance orders the p1/p2/p4/p8 curve. Phase-aware router admission now separates allocated slots from active compute: long cold host-expert prefills serialize, while bounded small/cache-hot requests may overlap only after the first generated SSE delta. Reviewer stop-sequence-contract handling and enforceable Workflow-timeout fixes remove the observed false fallback/deadline paths; ambiguous Workflow source inputs fail early, materialized scripts are verified, and queue/service cancellations plus 60s/600s signatures are recorded. | Relaunch the new binary and complete controlled p1/p2 decode+decode, cold-prefill+decode, and cache-hot A/Bs plus capability-specific unified/partitioned KV A/B; tune the cold/append boundary only from matched public-hardware evidence. |
| PERF-4, PERF-8, PERF-12, UX-2 | Not complete | Implement measured headroom continuation, broader optional knobs, and bounded control UX |
| PERF-11 | Partially implemented in the working tree: MoE topology candidates include each feasible sole-backbone owner as a performance-only full recompute; ranking prices the serial backbone and routed GPU/CPU experts instead of prioritizing owner names | Finish exact-argv guard tests, run the intentional roomy hardware comparison, then add only capability-proven row/peer candidates |
| UX-1 | Implemented for launch, dry-run, dry-run JSON, TUI config screen, and `ggrun status` | Support-expert status remains the NanoBeige controller; launch inspect is `ggrun status` |
| ROOMY-1, ROOMY-2, ROOMY-3, ROOMY-4, ROOMY-6 | Implemented in source: exact residual slack classifies roomy; tight launch remains the fit baseline but measured imbalance may authorize one same-workload topology that does not increase CPU expert offload; batch/ubatch/slots are full recomputes; winner/baseline-won/boundary and bounded rejection history persist | Commit plus live roomy dense/MoE/recurrent proof |
| ROOMY-5 | Implemented in source: PCI-keyed SM plus NVIDIA PCIe RX/TX and Linux process-tree CPU/RSS/I/O sampling span each separate agent phase. Link saturation is claimed only against a known measured/detected ceiling; low traffic leaves DDR/synchronization unresolved. Only imbalance between ordinary-layer owners is actionable. Measured imbalance may select one predicted-equal/slower topology for contained A/B because the static prior cannot veto its own correction; live workflow evidence still decides promotion. | Capture a matched live Qwen3.8 baseline/finalist comparison and verify phase transfer samples against the external `nvidia-smi dmon` trace; add peer counters only if they change finalist selection. |
| Exact-argv admission and long-load UX | Implemented in the working tree: a recomputed argv must receive its own allocation evidence, guarded peaks carry a placement identity, known challenger rewrite/recovery paths fail closed, lateral MoE split churn retains the exact proven placement, and 64+ GiB models warn before loading | Commit, then repeat the live MoE case after the current server is intentionally stopped |
| MMAP-1, MMAP-2, MMAP-4 | Implemented: production/preflight/recovery/daemon share capability-aware reclaim policy; unknown/anonymous loaders fail closed; mmap remains last-resort and consent-gated | Commit plus live resident/mmap/anonymous cases |
| MMAP-3 | Host ledger now separates exact reclaimable expert bytes from non-reclaimable runtime, KV, embeddings, and checkpoint reserve | Audit remaining backend-reported buffers/page tables/companions against live cgroup data |
| MMAP-5 through MMAP-7 | Not complete | Requires the storage/workload and real too-large-model acceptance window |
| HOT-1, HOT-2 | Implemented in source: immutable capability-only recipe, exact flag probing, static eligibility, cache-scoped allocation evidence, and per-device GGUF slot ledger | Build the pinned CUDA backend and preserve a real allocation/activation trace |
| HOT-3 | Partially implemented: aggregate decode steps/hits/misses/hit rate are required after a deterministic candidate-only canary and retained with calibration/verified evidence | Add backend per-layer uploads/evictions/warmup telemetry; collect drift evidence |
| HOT-4, HOT-5 | Integrated into the ordinary bounded optimizer: one calculated challenger, identical agent A/B, separate material decode gain, lifecycle gates, exact-argv admission, typed negative evidence, and cache-free fallback/invalidation | Run correctness and same-workload A/B on the three-GPU Q3 launch; repeat clean reuse and forced failure |
| HOT-6 | Not complete; automatic eligibility remains narrow (resident, p1, layer-split, separate gate/up, non-speculative MoE) | Public architecture/topology matrix before broadening support or claiming a default |
| PIN/P3 | Deliberately not started | Blocked on appropriate hardware/model artifacts as specified below |

### 2026-09-05 measurement reuse scope hardening

Related growth and compute-scaling predictions now require the same artifact,
backend/features, physical hardware signature, and slot count. The existing
hashed probe identity is checked using the record's context/microbatch/KV
coordinates, preserving compatible measurements while rejecting same-basename
replacements and foreign backend observations. Placement-plan version 8 and
calibration schema 25 invalidate derived decisions; raw observations and exact
fit records remain. Regression tests and the uncached core gate pass. Live
fit/performance acceptance is still open; see the corresponding section in
`optimizer-theory.md`.

### 2026-09-05 metadata-preserving cache writes

Compute and growth writers now preserve unrelated source/role labels under the
file lock, and retained compute values keep their original GPU role. Probe
schema 9 excludes potentially mislabelled old exact rows; scoped older growth
remains conservative fallback evidence. Placement-plan version 9 and calibration
schema 26 invalidate derived decisions, while validated fit records retain
their schema. Regression tests and the uncached core gate pass; live acceptance
remains open. Details are in optimizer-theory.md.

### 2026-09-01 Qwen3.8 three-GPU saturation finding

- The live cache-free baseline served at about 118 prompt tok/s and 15 decode
  tok/s. Phase evidence recorded GPU 0 at 73% SM while GPU 2 reached only 3%; an
  external decode sample showed the 3090 Ti at roughly 21–26% SM, the 4070 at
  7–15%, and the 3060 at 4–10%, with host traffic concentrated on the 3090 Ti.
- This is not proof that every GPU should show 100%. Layer split is a serial
  pipeline for one sequence; cards used mainly for resident expert storage may
  wait. The objective remains cold prefill, cached append, decode, foreground
  latency, and complete agent makespan—not cosmetic aggregate utilization.
- The old controller detected the imbalance but refused a topology experiment
  because the allocation was labelled `tight-resident`. Fit and phase diagnosis
  are now separated: an exact tight baseline may measure one topology that does
  not increase CPU expert offload, and the candidate must still pass contained
  exact admission and the normal live regression gates.
- `hot-experts=auto` did not activate in this run. The model exposes 512 experts,
  routes 10 per token, and leaves 31 complete routed-expert layers on host. The
  exact residual ledger appears capable of a low-20s-slot candidate, but the
  estimator ranked an oversized ubatch first; that candidate failed CUDA0
  admission. Scoped negative evidence now advances rather than freezing the
  frontier, and hot-expert exclusion reasons are retained in optimizer status.
- 2026-09-01 Qwen3.8-Flash-Next q4_0 p1 relaunch generated a leftover cache-on
  candidate (no leftover-VRAM exclusion; 3090 slack 531 MiB) but spent the one
  live A/B on `kv-alternate`, which fail-closed CUDA0 1158 MiB. Schema 22 now
  keeps a feasible `hot-experts-*` candidate as the measured auto finalist so
  KV/topology estimates cannot consume that slot. Schema 23 applies leftover or
  demoted cache-on from exact packed evidence onto the launch strategy itself
  so `--moe-expert-cache` is present when auto can plan it; packed stays the
  fail-closed fallback. Live cache-on vs packed remains unproven until that
  relaunch.
- 2026-09-02 (branch `hot-experts-turboquant`) reverses the schema-23
  first-serve choice: a live GLM-5.3-Flash Q3 serve stayed ~10.5 h on the
  demoted all-CPU-expert topology because a failed cache-on challenger's
  fallback stripped only the cache flags, and because `auto` served the
  cache-on layout as candidate 0 so the packed-vs-cache-on A/B never ran.
  `auto` now serves the packed cache-free layout as the fail-closed default
  and hands cache-on to calibration as the challenger; a demoting challenger
  captures its exact packed pre-demotion `Strategy` and the fallback restores
  that (or recomputes cache-free and fails closed). `CalibrationSchemaVersion`
  24; `HotExpertCacheEvidenceSchemaVersion` gate drops a stale verified
  record's cache while keeping its fit proof. Full record:
  `docs/optimizer-theory.md` "2026-09-02 hot-expert fallback + promotion-gate
  correction". D2 cache-graph-reserve/slot-ladder and prefill microbatch work
  deferred — the `ub=64` number came from the degraded run.
- 2026-09-02 follow-up: the first `auto` launch of the restored packed layout
  loaded, passed health, then OOMed on the warmup decode inside
  `cudaGraphInstantiate` (no runtime graph-growth evidence for this never-run
  model; the no-alloc oracle can't see graph-exec memory). The
  `verifyAndActivateLaunch`-canary seam now routes a post-`model loaded` CUDA
  OOM through `RecordRuntimeGraphGrowthFromOOM` + re-plan + re-verify (bounded
  2), with a `CUDA_SCALE_LAUNCH_QUEUES=1x` / `GGML_CUDA_GRAPHS=0` env rung ahead
  of moving layers for a graph-capture abort. Convergence for this GLM key and
  the `=1x` throughput cost are unproven until a live run. Full record:
  `docs/optimizer-theory.md` "2026-09-02 verify-canary CUDA-OOM recovery".
- Full convergence across winning topology, hot cache, batch, and slot settings
  is still open. A named winner must become the next safe baseline and continue
  bounded coordinate descent in a later launch/idle window until a complete
  baseline-won decision closes the scope; do not claim fullest hardware use
  before this exists and passes live multi-model acceptance.

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

### Failed TUI Claude p4 start, 2026-09-01 15:33–15:35 UTC (inventory)

User launched from the TUI after the 14:16 canonical binary
`/home/mik/go/bin/ggrun` sha256
`400896a2bdda39623df48c4e2baba544fae3e042bede5b46168f25688f368230`.
Saved request: `.cache/latest-tui-launch.json` at 15:33:59Z. Handoff was
`Backend=auto`, `ParallelSet=false`, `CtxFlag=fit`, Claude Code on, reviewer
`qwen2b`. KV was **not** fully automatic: `KVQuality=q4_0` with
`KVQualitySet=true`, `KVPlacement=gpu`. SWA full was true with
`SWAFullSet=false`. Disk config still has `LLM_CTX_SIZE=131072`,
`LLM_KV_QUALITY=bf16`, `LLM_SWA_FULL=true`.

The core did **not** replay the 14:12 verified p1/q8_0/`n-cpu-moe 18` config.
Observed live argv before death: 262144 total context, **parallel 4** (65536
per slot), q4_0 KV, `-b 128 -ub 64`, `n-cpu-moe=19`, same hot-experts fork.
Reviewer Qwen3.5-2B came up on `:44175` in 2s
(`.logs/ggrun-claude-reviewer-44175.log`) and is now gone with the controller.

Failure artifact:
`.cache/memory-probes/failed-a052bb96276efe01f81a408414209d30.log` (15:35).
Companion probe cache `.cache/0c3f1082a782.probe` (15:34:37Z) records
`parallel=4`, `ubatch=64`, `kv_quality=q4_0`, and
`PROBED_FREE_VRAM="0:11873 1:24112 2:8067"`. At load, CUDA2 (3060) had only
7957 MiB free. Model buffers were 9620 / 20569 / 7182 MiB; KV+indexer+RS+compute
on CUDA2 added another ~0.8 GiB after that. Warmup `llama_decode` aborted:

```text
ggml-cuda.cu:107: CUDA error
CUDA error: out of memory
current device: 2
cuMemCreate(&handle, reserve_size, &prop, 0)
Aborted (core dumped)
```

No `:8081` scope log was written for this start. At 15:37 UTC no ggrun or
llama-server remains; nvidia-smi shows ~1 MiB used on all three GPUs.

What this does **not** prove: that p4 is a legal automatic width for this
model, that q4 KV was chosen by the optimizer rather than the saved/TUI
quality row, or that the 14:12 p1 result is invalid. It does prove a
Claude-mode TUI launch with automatic parallel and explicit q4 KV reached
n_seq_max=4, packed CUDA2 past the VMM pool, and failed closed on device 2
during warmup rather than serving.

A second TUI launch at 15:42:24Z used the same request (q4_0 set, parallel
unset, Claude + qwen2b). Reviewer `:36093` became healthy in 2s. The main
load rewrote the same failed-probe path at 15:43 with identical shape
(`n_seq_max=4`, 128/64, CUDA2 model 7182 MiB) and the same device-2
`cuMemCreate` OOM during warmup. Controller pid 3497321 exited; GPUs idle.
No `:8081` scope log. The 14:12 p1 plan was still not reused.

### TUI Claude p1/q8 serving, 2026-09-01 15:44–15:46 UTC (inventory)

Third TUI attempt after the two p4 OOMs. Request
`.cache/latest-tui-launch.json` 15:45:15Z changed two constraints:
`KVQuality=auto` (`KVQualitySet=true`) and **`Parallel=1` with
`ParallelSet=true`**. Backend still `auto`, `CtxFlag=fit`, Claude + qwen2b.
Binary still `400896a2…368230`. Controller pid 3502040 started 15:44:33.

Contained loads on `:42099` then `:37295`, then serving pid 3505909 on
`:8081`. Health ok after 22s. Scope
`.logs/ggrun-claude-server-v2-8081-d68ded82b43cbb9e084cc767.log`. Reviewer
Qwen3.5-2B healthy on `:46309` (`--device CUDA0`).

Exact serving argv: ctx 262144, **`--parallel 1`**, `-b 2048 -ub 256`, **K/V
q8_0**, `--kv-offload`, `--fit off`, `--n-cpu-moe 21`, split
`0.29,0.62,0.10`, expert layers 0–14 CUDA1 / 15–21 CUDA0 / 22–26 CUDA2 /
rest CPU, CRAM 13312, 16 checkpoints. **No `--moe-expert-cache`.** Slot idle
at 15:46:24; nvidia-smi 8831 / 19726 / 10065 MiB used, 0% SM.

This is not a replay of the 14:12 verified `n-cpu-moe 18` plan (this run
keeps 21 CPU experts). `ggrun status` still showed the stale 11:35
admission-only 15/118 tok/s decision at sample time. Hot-expert cache still
unproven. The p4/q4 OOM path was avoided because parallel was pinned to 1
and KV quality was auto (resolved q8_0), not because p4 became legal.

Follow-up 15:50–15:51 UTC: after a short 8081 canary (task 232: 19-token
prefill 56.57 tok/s, 64-token decode 21.19 tok/s; six cached prompts), the
controller stopped `:8081`, ran a contained reload of the **same** argv on
localhost, then clean-relaunched pid 3517482 on `:8081`. Health returned ok.
Reviewer `:46309` stayed up. No CUDA error. Same p1/q8/`n-cpu-moe 21` shape;
still no `--moe-expert-cache`. During the new 6924-token canary prefill,
nvidia-smi showed 4070 SM 95% / 3090 5% / 3060 0%. This is the ordinary
lifecycle/canary path, not a crash.

Follow-up 16:10 UTC (inventory): same pid 3517482 still serving, health ok,
no `--moe-expert-cache`. `ggrun status` now has a **this-launch** decision
at 2026-09-01T15:52:18Z, schema 20, roomy-resident, winner `default`,
validation still `admission-only-v1`. Baseline 17.77 decode / 152.54 prompt
/ 18.97 mixed tok/s, 94.0 s turn. Finalist `ubatch-2048` unavailable/memory
(CUDA0 6617 MiB deficit); probes
`.cache/memory-probes/failed-bfc857a62bd7a4503cf1febfdb069244.log` (ubatch
2048, cudaMalloc 6616 MiB device 0) and
`failed-7b377248c15da43ea6baa943fcf441d2.log` /
`failed-8aa6ab8083f9a1d1ba9c632b65e529a3.log` (ubatch 1024, 3308 MiB device
0) match that fail-closed admission. Hot-experts excluded three times:
`residual per-device VRAM cannot hold the minimum 10 useful slots` (device
slack 697/186/572 MiB). Explored topologies include owner-0/1/2; none
became the served argv. Parallel stayed 1–1 (TUI `ParallelSet=true`). Live
Claude task 4367: 93099 prompt tokens, 90954 cached, 1271 processed at
120.28 tok/s, 1072 generated at 12.14 tok/s (`tg_3s` ~12.1). Cumulative
metrics 95854 uncached prompt / 552910 cached. nvidia-smi dmon during
decode: 4070 12–19% SM / 11.2 GB, 3090 29–31% / 24.0 GB, 3060 9–10% /
11.4 GB. This proves exact ubatch refusal and typed hot-expert skip; it
does not prove a cache-on run or a topology win.

Follow-up 17:24 UTC after the Grok working-tree rebuild: the same explicit-p1
TUI scope remained healthy and recorded schema 22 evidence at 18.25 decode,
153.20 prompt, 19.59 mixed tok/s, and a 94.78 s two-lane turn. The calculated
`ubatch-2048` finalist again failed exact admission, now with a 7109 MiB CUDA0
deficit, so the measured default was retained. This is negative feasibility
evidence, not proof that the default is fastest. Review found that the new
priority hot-expert candidate derived extra VRAM slack by demoting GPU expert
layers while leaving the cache-free allocation marked exact and without moving
the bytes into the host ledger. The ledger now conserves those bytes, labels the
changed placement derived/non-exact, and requires contained backend admission
to restore exact authority. The full Go and race suites pass after correction.

### Full-auto TUI/core-path acceptance, 2026-09-01 14:02–14:15 UTC

Canonical test binary was `/home/mik/go/bin/ggrun` (also resolved through
`/home/mik/.local/bin/ggrun`), sha256
`43b37899049ea2c981bba5ea97d8173e0c31a2955fe3f66ef07862a895291085`
for the two live trials. The post-replay-fix canonical binary installed at
14:15 UTC has sha256
`400896a2bdda39623df48c4e2baba544fae3e042bede5b46168f25688f368230`.
The TUI regression gate proves that an automatic backend remains `auto` in its
`LaunchRequest`/argv while the UI may preview the installed architecture route
and hot-expert overlay; serialized `PARALLEL=1` remains policy-auto instead of
becoming an explicit `--parallel 1`. Explicit per-model values and the
unsupported-route continue-once action remain authoritative.

The controlled command used the active three-GPU hardware and deliberately
neutralized old saved preferences without pinning backend, parallel, topology,
batch, ubatch, or hot-expert slots:

```text
LLM_SERVER_NO_UPDATE_CHECK=1 LLM_CTX_SIZE=fit LLM_KV_PLACEMENT=auto \
LLM_KV_QUALITY=auto LLM_SWA_FULL=false LLM_BACKEND=auto \
/home/mik/go/bin/ggrun launch \
/home/mik/ggrun-project/ggrun/models/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf \
--worker-benchmark --support-expert off --allow-live-memory-probe
```

The core automatically routed `qwen4exp` to
`qwen3-8-flash-next-hot-experts`, selected ctx 262144, GPU q8_0 KV, batch
2048/ubatch 256, and p1. The cold estimate started at `--n-cpu-moe 17`. Exact
preflight found CUDA1 compute allocation 1032 MiB short by 5 MiB, then CUDA0
allocation 1160 MiB short by 64 MiB. Four rejected measured configurations
were retained; monotonic expert derating converged at `--n-cpu-moe 18` and the
final exact placement loaded in about 21 seconds. The early lateral repacks did
not reduce total attempts and remain an optimizer-sequencing target; this run
does not justify a guessed static margin.

Persisted evidence:

- `.cache/system_f10a43e9645f.cache`, system-probe schema 3: CUDA overhead
  274/445/191 MiB and host overhead 572 MiB.
- `.cache/memory-probes/probe-0b2d3d1a5c93bcc636cdf40167bcb0d5.json`, exact
  guarded allocation and argv for n-cpu-moe 18.
- `.cache/profiles/profile-18386e04f623bfc8a3873f8bf7fef7a1.json`, active
  lifecycle with allocation, health, functional, cache, performance, and
  active gates passed.
- `.cache/verified-configs/verified-75d8452c6b87.json` and
  `.cache/0a591f573712.place`, direct-start config and exact placement.

The final backend allocation was model buffers CUDA0/1/2 =
8547.97/19336.13/9488.03 MiB; the two KV regions plus recurrent state summed
to 1156.64/2686.59/757.35 MiB; compute was 1159.25/1031.30/1024.09 MiB. Host
model buffers were 27465.95 MiB CPU + 20969.14 MiB CUDA-host, with 937.31 MiB
host compute. This is the first live proof that recurrent/indexer KV regions
must be summed rather than averaged.

First-run cache canary measured 6888-token cold prefill at 176.87 tok/s;
the 64-token throughput probe measured 77.34 prompt tok/s and 23.24 decode
tok/s. Worker cases passed 4/4 at 84.80 aggregate prompt tok/s and 23.87 decode
tok/s. An identical second command reported verified direct reuse and measured
176.06 tok/s cold prefill, 76.70 prompt tok/s, 24.19 decode tok/s, and 4/4
worker cases at 84.33/23.64 tok/s. Compared with the earlier bf16/n-cpu-moe31
baseline (~118 prompt, ~15 decode), this is a large end-to-end gain, but it is
a combined KV-quality plus expert-residency change and does not isolate either
factor. No `--moe-expert-cache` flags were active, so it is not evidence of a
hot-expert-cache speedup.

The repeat also exposed semantic drift in verified replay: its reconstructed
argv omitted `--kv-offload` and `--fit off`. Backend defaults happened to keep
GPU KV and its fit pass aborted without changing explicit placement, so the
measured result remained equivalent, but exact replay must not depend on those
defaults. Source now re-derives fit/KV dialect capabilities from the current
exact backend on every verified hit; a regression asserts both flags. After
that correction, 1,025 focused normal/race tests, 1,529 full tests, and `go
vet ./...` pass. The post-fix canonical binary's argv-only automatic dry run
emitted both `--kv-offload` and `--fit off`; it did not load the 84 GiB model a
third time. The already-live allocation/canary result plus exact-argv unit gate
cover the correction, while a later ordinary relaunch can close redundant live
replay acceptance without spending another load solely for this boolean-field
change.

### Live Qwen3.8-Flash-Next Q3 restart, 2026-09-01 11:34–11:46 UTC (inventory)

Restart after the 11:24 canonical binary
`/home/mik/go/bin/ggrun` sha256
`afa3f4d8679613f998c276ab5b500b6ee3072f40bde32c6e7cfc8e3950e4b7b8`.
Controller pid 3076621 started 11:28:21 UTC from that path (cwd was
`/home/mik/v0-leaderboard`; this does not change the serving argv). Main
server pid 3090292 started 11:34:40 UTC, health ok, scope
`.logs/ggrun-claude-server-v2-8081-b2f892035b9ade136376d20e.log`. Backend
binary is
`.src/fork-qwen3-8-flash-next-hot-experts/build-cuda/bin/llama-server`
(`qwen3-8-flash-next-hot-experts@llama-server-4736964f7fc4b06766ff64a0`).
Reviewer Qwen3.5-2B Q4_K_M on `:43587` (`--device CUDA0`) is healthy.

Exact serving argv (0.0.0.0:8081): ctx 262144, `--parallel 1`, `-b 2048 -ub
256`, K/V bf16, `--flash-attn on`, `--kv-offload`, `--no-mmap`, `--n-cpu-moe
31`, split `0.29,0.61,0.10`, expert `-ot` layers 0–8 CUDA1 / 9–12 CUDA0 /
13–16 CUDA2 / remaining `exps=CPU`, CRAM 12288, 16 checkpoints min spacing
512. **No `--moe-expert-cache` / `--moe-expert-cache-inserts`.** The backend
`--help` exposes both flags; they are not on this process.

CUDA_DEVICE_ORDER=PCI_BUS_ID as seen by this build (do not reuse the older
3090-first mapping):

- CUDA0 = RTX 4070 12GB (nvidia-smi bus `17:00.0`)
- CUDA1 = RTX 3090 Ti 24GB (bus `65:00.0`)
- CUDA2 = RTX 3060 12GB (bus `B3:00.0`)

CPU at load: Intel i9-10940X, 28 threads, host ~212 GiB free in the backend
report.

`ggrun status` at 11:45 UTC (schema 20, scope
`839ef2d103b531c08d1d7d4216e6e4d6dab255b72ff9a333a4682f82368fe951`):
tight-resident, winner `default`, validation `admission-only-v1`, measured
2026-09-01T11:35:06Z. Baseline 14.996 decode tok/s, 118.195 prompt tok/s,
16.177 mixed tok/s, 99.076 s agent-turn mean (2 samples, 2 lanes). Bottleneck
string: GPU 0 saturated (74% SM) while GPU 2 idle (4% SM). Finalists
`ubatch-1024` and `ubatch-2048` both `unavailable` / `memory` on CUDA0 (3285
MiB and 6569 MiB deficits); recovery ladder refused. Explored topologies
include `moe_offload:owner-1:gpu-1`; none became the served argv. Exclusion:
`hot-experts: exact cache-free allocation evidence is unavailable`. Device
slack estimate 980/1355/729 MiB.

Live 11:45–11:46 UTC sample (do not treat as a completed A/B):

- Slot 0 task 557: cold 130,302-token Claude prompt, `n_keep=0`, cache empty.
  Prefill progress 20,480 / 130,302 (~16%) at 117.36 tok/s after 174.51 s
  (chunk timings 126.02 → 117.36 tok/s from 2,048 through 20,480 tokens).
- `/metrics` still zero prompt/predicted totals; `n_decode_total` 8;
  `requests_processing` 1; `n_tokens_max` 16384.
- nvidia-smi dmon 3s: GPU0 4070 SM 87–95% / 9007 MiB / ~60 W / rxpci up to
  ~11.7 GB/s; GPU1 3090 Ti SM 5–10% / 20274 MiB / 118–128 W; GPU2 3060 SM
  0–32% / 10637 MiB / ~40–46 W.

What this does **not** prove: a topology win, a hot-expert cache-on run, or
that the 11:24 optimizer changes selected a new finalist. This 8081 argv is
the previous default placement. Prefill ~118 tok/s matches the stored
baseline; decode is not in this sample yet. Hot-expert capability is present
in the fork binary and absent from the served command.

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
  Otherwise start the safe estimate and bound launch-time work. A promoted
  coordinate becomes the next safe baseline; continue topology → residency/hot
  experts → batch/ubatch → useful slots in later launches or an explicit idle
  window until a baseline-won comparison proves convergence. Prevent cycles by
  retaining scoped measured edges and accumulated typed rejections.

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

- [x] **HOT-1 — isolated capability.** Audit and pin a reviewed upstream commit
  in a separate backend fork. Detect the exact cache flags, CUDA/model layout,
  decode shape, and disabled-path identity. Unsupported models/backends produce
  no candidate. Implemented as the non-routed `hot-experts` recipe pinned to
  `csantiago78/llama.cpp@bccbacdb8945680f1cfc7e6bffd1e59014705750`;
  install/update/rollback all re-probe both required flags. The same reviewed
  commit is embedded as a source-feature overlay: the TUI can compose it onto
  an exact pinned llama.cpp-derived architecture fork in a separate checkout,
  retain base-patch order, validate the composite, and reuse it by exact base
  identity. Patch conflicts and binary-only/divergent forks fail closed rather
  than attempting an unsafe automatic merge.
- [x] **HOT-2 — exact cache ledger.** Derive bytes per expert slot and per
  host-expert layer from GGUF/backend evidence. Reserve KV, graphs, checkpoints,
  prompt cache, companions, allocator growth, and device headroom before
  calculating any slot count; never copy the reference value 48 into policy.
  The slot ceiling is the tightest physical router GPU after an exact
  cache-free allocation, includes the backend's dummy slice/device table and
  alignment guard, and is capped by the model's expert count.
- [ ] **HOT-3 — temporal evidence.** Record per-layer cache hits, misses,
  uploads, evictions, warmup, and drift separately for prefill/decode and p1/p2.
  Do not persist or promote a global static hot list merely because one corpus
  was skewed. Aggregate steps/hits/misses/hit rate are now mandatory after a
  256-token deterministic decode canary and persist with the finalist; the
  reference fork does not yet expose the remaining per-layer/upload/eviction
  counters, and p2 is therefore still ineligible.
  - [ ] Add a second reviewed ggrun overlay patch after the pinned upstream
    cache commit which emits bounded structured heat epochs: per layer/expert
    access count, hit/miss, upload/eviction, and reuse-distance/window data.
    Never log prompts or token contents.
  - [ ] Merge epochs atomically under an exact model fingerprint plus workload
    class. Keep recent windows and a decayed EWMA instead of one immortal
    cumulative count, while retaining total observations and run count for
    confidence. Backend build, quant/layout, and expert-count mismatches fail
    closed; hardware placement remains a consumer, not part of expert identity.
  - [ ] Feed the heat profile into candidate generation only. It may seed LRU
    warmup and rank selective expert-range mmap/mlock/pinning candidates, but
    each candidate must still pass exact byte/page accounting and the ordinary
    agent-workload A/B. Static top-N frequency is not promotion evidence.
- [ ] **HOT-4 — one bounded A/B.** Compare cache-off with one calculated slot
  budget on identical cold-prefill, cached append, decode, mixed foreground,
  and workflow makespan. Require coherent output, no missing/double-counted
  expert contribution, clean relaunch, and material decode plus end-to-end gain
  without a prefill/cache regression. The source path now enforces this through
  the standard two-sample agent screen, a 3% workflow and separate 3% decode
  margin, 5% phase-regression ceiling, and normal functional/cache/relaunch
  gates; live hardware acceptance remains open.
- [ ] **HOT-5 — self-disable.** Cache allocation failure, unsupported graph,
  low hit rate, upload/synchronization regression, OOM, multi-token/speculative
  incompatibility, or correctness drift falls back to the exact stock argv.
  Failed evidence is scoped and finite. Source now rejects p2/spec/mmap/fused
  and unproven layouts, validates the backend's exact startup allocation, and
  records completed-canary telemetry failures as typed negative evidence.
  Runtime activation or functional/cache lifecycle failure erases both the
  verified config and its original cache-free calibration decision, restores
  the identical cache-free strategy, and records a scoped negative only after
  that baseline itself reaches `StateActive`. Clean-relaunch recovery returns
  its actual strategy/argv, so a stripped cache cannot be mislabeled as the
  measured cache-on winner.
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

The bounded per-layer/per-expert heat epochs from HOT-3 are shared evidence for
the later selective mmap/mlock lane. They can identify expert tensor slices
worth retaining in RAM or VRAM, but page alignment, tensor contiguity, transfer
bandwidth, temporal reuse, and measured tail latency remain mandatory: an
all-time popularity ranking alone must never decide residency.

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

## PACKED — hot-expert sizing ignores unmeasured runtime growth

Measured live 2026-09-09, immediately after the crossover fix landed.

With the target-driven sizing in place, `auto` plans the configuration the sweep
proved best:

```
before   --n-cpu-moe 39  --moe-expert-cache 8    (measured 0.967 relative)
after    --n-cpu-moe 42  --moe-expert-cache 32   (measured 7.72 tok/s, +6.0%)
```

Planning is fixed. **Admission is not.** The launch aborted with exit 134:

```
sched_reserve: CUDA0 4855.21  CUDA1 5114.50  CUDA2 4855.00 MiB compute
MoE expert cache enabled: 41 layers x 32 slots, 14344.7 MiB
common_init_: warming up the model with an empty run
CUDA error: out of memory   current device: 1, ggml_cuda_kernel_can_use_pdl
```

Everything allocated. The failure is in **warmup**, and the ledger says why:

```
CUDA0 10334/11873 MiB (fit=10060 overhead=274 runtime=0)
CUDA1 24008/24112 MiB (fit=23563 overhead=445 runtime=0)   <- 104 MiB margin
CUDA2 10836/11909 MiB (fit=10645 overhead=191 runtime=0)
```

`runtime=0`: no runtime graph growth was reserved, because growth evidence is
keyed per plan and this plan had never run. So the slot arithmetic spent every
byte of slack and warmup had nowhere to grow.

### The inconsistency

`ComputeExpertSeats` already refuses to spend an unmeasured margin -- "an
unmeasured margin buys no seats. The alternative, a default margin, is exactly
the static fudge reserve invariant 8 forbids". The hot-expert slot arithmetic
does not apply that rule: it packs to `SlackMB` whatever the growth evidence
says. Two paths in the same package, opposite policies on the same question.

Note this is not the cache being too large in principle: a manual llama-server
run at the same 32 slots, with all 42 layers cached and no pinned layer, served
fine and measured +6.0%. The difference is that ggrun's plan retained one pinned
expert layer (blk 7 -> CUDA2), used a slightly different tensor-split, and then
packed the remainder to 104 MiB.

TODO:

- [ ] Subtract a runtime-growth allowance from the slack the slot arithmetic
      spends, using `RelatedModelRuntimeGraphGrowth` (the same looser lookup
      residency.go trusts: matching GPU signature and slot count, exact evidence
      only). Where a device has no growth evidence, fail closed on that device
      rather than spending its slack -- consistent with the seat detector.
- [ ] Decide what a first launch of a new plan should do. It has no growth
      evidence by construction, so a strict rule makes every new plan
      unlaunchable. Options: size conservatively for the first launch and
      re-size once growth is recorded, or carry growth across plans that differ
      only in slot count. The second is closer to what the evidence supports:
      growth is a property of the graph shape, and slot count changes it little.
- [ ] Investigate why `--n-cpu-moe 42` still pins blk 7 to CUDA2. The cache then
      covers 41 layers rather than 42, and the pinned layer's ~3.1 GB sits on a
      device the cache also wants.

## CROSSOVER — ggrun's automatic slot target is below the useful range

Measured 2026-09-08/09. This is the defect that made hot experts look useless.

`hotExpertMinUsefulSlots` returns `model.ExpertUsedCount` -- 8 on GLM 5.3 Flash
-- and the demotion loop stops at the first layout that clears it. So automatic
sizing lands at or just above 8, and every automatic measurement ever taken
sampled the losing side of a crossing nobody had located.

### The curve (matched workload, 8 generations x 800 tokens)

```
config              cache MiB   hit rate   decode   prefill   vs no cache
no cache                    0          -     7.28     11.70            -
K=14 (auto's pick)     6364.7      25.8%     6.53     11.42       -10.3%
K=20                   9346.3      32.8%     7.56     11.64        +3.8%
K=24                  11126.6      36.6%     7.53     11.43        +3.4%
K=32                  14687.1      43.3%     7.72     11.40        +6.0%
K=35                  16022.2          -      OOM         -            -
```

Crossover is between 14 and 20, near 17-18. Ceiling on this rig is K=34.
**The useful window is roughly K=20..34, and auto targets 8.**

### ggrun's own calibration reproduced it live

Production auto run, 2026-09-09, ggrun's own agent workload:

```
default         makespan 58.05s   decode 5.5 tok/s   prefill 28.4 tok/s
hot-experts-14  failed admission on CUDA1 by 17 MiB; skipped
hot-experts-8   makespan 60.04s   decode 4.8 tok/s   prefill 28.2   relative 0.967
```

It tested 8 and 14, found 8 slower by 3.3% makespan, and rejected the feature.
Both verdicts are individually correct and the conclusion is wrong: the
configuration that wins by 6% was never a candidate. Note also that 14 missed
admission by **17 MiB** -- the useful range is not merely unchosen here, it is
barely out of reach at the residency auto keeps.

### What has to change

- [ ] Derive the automatic target from the measured ceiling, not from
      ExpertUsedCount. The ceiling formula is exact (see the research doc):
      model + KV + compute + CUDA context overhead + cache(K+1) per device,
      minimum across devices. On this rig that is 34, and a target near
      0.6-0.9 of it lands in the useful window.
- [ ] Demote resident expert layers toward that target. Auto currently stops
      demoting the moment it clears 8. Freeing all expert layers is what makes
      K=32 reachable, and resident expert layers measured flat on BOTH decode
      (9/7/4 layers) and prefill (11.40-11.70 across every config), so they are
      the cheapest VRAM on the rig to give up.
- [ ] Revisit the displacement gate added in 9e2a890. Its premise -- that
      displacing resident layers for a cache loses -- was measured at K=14 and
      is false at K=32. The rule should be "displace only toward a slot count
      above the measured crossover", not "do not displace".
- [ ] Calibrate at least one candidate inside the useful window. A sweep that
      only samples below the crossover cannot discover it.
- [ ] Leave `--moe-expert-cache-inserts` at 2. Raising it to 4 improves hit rate
      43.3% -> 46.1% and LOWERS decode 7.72 -> 7.38: upload cost is the binding
      term, not miss rate.

## BLIND — hot-expert telemetry was never observable

Opened 2026-09-08, verified live the same day.

Every hot-expert measurement taken this session was blind, including the matched
A/B that concluded the cache costs 12% decode. The backend emits its hit/miss
telemetry as `LLAMA_LOG_DEBUG`:

```
moe-cache: steps=N hits=N misses=N hit-rate=X%      (every 512 decode steps)
```

In this fork's ladder `LOG_LEVEL_DEBUG` is **5** and `LOG_LEVEL_TRACE` is **4**
-- debug sits ABOVE trace, inverted from most logging systems -- and
`common/log.cpp` drops debug when the threshold is below 5. ggrun requests
`-lv 4` (`backendTraceVerbosity`, main.go). **Off by exactly one level.**

Measured: the A/B cache arm's log contains 1240 info lines and **zero** debug
lines, so zero telemetry lines, despite the cache allocating 6,364.7 MiB.
Re-running the same launch with `LLAMA_ARG_LOG_VERBOSITY=5` produced 2650 debug
lines in the first 45 seconds. The diagnosis is verified, not inferred.

Two consequences:

1. **No hot-expert conclusion this session is supported by mechanism
   evidence.** "The cache costs 12% decode" was measured, but whether the cache
   was ever hit is unknown, so whether that cost bought anything is unknown.
   A 6.2 GB allocation that is never hit and a 6.2 GB allocation with a 60% hit
   rate look identical in the logs we captured.

2. **`ValidateHotExpertCacheTelemetry` can never pass.** It requires observed
   telemetry with at least one hit and >= 512 steps. That telemetry is
   unobtainable at the verbosity ggrun requests, so the hot-expert verification
   path is dead by construction -- which plausibly explains why the feature
   never promoted despite repeated attempts.

TODO:

- [ ] Raise the verbosity ggrun requests when hot experts is enabled, or promote
      the telemetry line to `LLAMA_LOG_INFO` in the patch we own. The second is
      cheaper at runtime (one line per 512 decode steps rather than all of
      llama.cpp's debug output) and makes the contract satisfiable regardless of
      what verbosity a user sets. Decide after the telemetry has been seen to
      work.
- [ ] Re-run the hot-expert comparison with telemetry visible and record the
      hit rate beside the throughput. Until then the feature is unevaluated,
      not disproven.
- [ ] Add a launch-time check that a feature ggrun validates through log output
      is actually emitted at the verbosity ggrun requests. This class of bug --
      a contract that depends on output the caller silences -- is invisible to
      every unit test.

## FIRST — cold start on a machine with no evidence

Opened 2026-09-08. Everything in this file assumes ggrun has measured the
machine. On a new install it has not, and every gate here is deliberately
fail-closed, so a first run gets the most conservative plan ggrun can build.

The risk is almost certainly not correctness -- fail-closed means a cold launch
is safe by construction -- it is **bootstrap cost and first impression**. A user
whose first launch is slow, and whose second and third are also slow because
each gate needs its own completed serve, concludes the product does not work.
On a 138 GB model at ~6 minutes per load that is a long way to walk before
ggrun looks good.

This is worse after today's changes, and that has to be said plainly: the
displacement gate, the phase weight and the split-mode probe each add evidence a
cold machine does not have. They are right individually and they compound.

### What a cold machine lacks, and what each costs to obtain

| Evidence | Consumer | Cost to obtain |
|---|---|---|
| MeasuredAllocation (Exact) | ledger, hot-expert sizing | 1 completed launch |
| RuntimeGraphGrowth per GPU | occupancy margin, seats | 1 healthy serve |
| SystemCUDAOverhead | ledger | system probe (cheap) |
| ComputeBufByGPU | occupancy, entry cost | fit oracle (cheap) |
| AgentContextDemand | context cap | 200 request samples |
| AgentPhaseTiming | batch/objective weight | 8 request samples |
| HotExpertDisplacementProof | hot-expert auto | 2 launches (matched pair) |
| SplitModeSupport | mode eligibility | 2 fit-oracle probes (~2 s each) |
| ObservedPerformance | reporting, ranking | 1 serve |

Open question this table exists to answer: **how many full launches does a cold
machine actually need before the plan stops improving?** Nobody has counted. If
the answer is one or two, there is no problem to solve. If it is five, the
bootstrap needs to be consolidated so a single serve records everything a serve
can record.

### How to test it

Three layers, cheapest first. Only the first belongs in the core gate.

1. **Cold-cache unit suite (gate-able, no GPU).** Every evidence consumer must
   return a safe, launchable answer against an empty `CacheDir`: no panic, no
   error that blocks a launch, no seat offered, no cache funded, no batch
   promoted. This is the layer that would catch a future gate that accidentally
   makes a cold machine unlaunchable, and it costs milliseconds.

2. **Cold dry-run (cheap, one host, no serve).** `ggrun dry-run <model>` with
   `CACHE_DIR` pointed at an empty directory must print a complete argv. This is
   the honest first-run smoke test: it exercises real detection, the real fit
   oracle and the real planner without a 6-minute load, and it is the only layer
   that catches a cold-start failure arising from the interaction of components
   rather than from any one of them. Add it as a script, not a Go test, so it
   can run against any model the developer has.

3. **Bootstrap convergence run (live, periodic, not in the gate).** Wipe the
   cache, launch N times with a fixed workload, and record after each launch
   which evidence now exists and what the plan changed to. The output is a
   convergence curve and a number: launches-to-stable-plan. Run it when a gate
   is added or changed, since that is exactly when the number moves.

Layer 2 is the one to build first. It is the closest thing to what a new user
actually does, and it is cheap enough to run on every change.

### Measured 2026-09-08: cold start over-provisions context

`scripts/first-run-smoke.sh` (below) found this on its first run. A cold cache
does produce a complete, launchable plan -- first run is not broken -- but it is
measurably worse than a warm one on the same model and hardware:

```
cold   --ctx-size 1048576   --n-cpu-moe 42
warm   --ctx-size  287744   --n-cpu-moe 40
```

A 3.6x larger context, and because that KV occupies VRAM the plan would
otherwise spend on residency, **two more expert layers are pushed onto the
host**. The first run is therefore slower than the same rig will be later, in
the phase where a new user forms their opinion of the product.

Cause: `AgentContextDemand` needs `minAgentContextSamples` (200) request
samples before it can size context to the observed workload, so a cold machine
falls back to the model/policy ceiling. The ceiling is the safe choice in the
absence of evidence -- it never truncates an agent -- but "safe" and "what a
new user should get" are not the same, and 1M tokens of KV is a very expensive
default for a workload whose measured demand is 287k.

Worth noting the fallback is asymmetric: guessing the context too small
truncates real work, guessing too large only costs speed. That asymmetry is why
the current behaviour is defensible; it is not why it is right.

TODO:

- [ ] Decide what a cold machine should assume about context. Options: a
      smaller evidence floor than 200 samples, a first-run default below the
      model ceiling, or sizing from the first session's observed requests
      mid-run. All three are guesses about an unmeasured workload, so this
      needs a decision on which guess is least harmful, recorded as such.
- [ ] Count launches-to-stable-plan on a wiped cache. Until this number exists
      the size of the problem is unknown, and every fix below is speculative.
- [ ] Cold-cache unit suite asserting a launchable plan with empty evidence.
- [ ] `scripts/first-run-smoke.sh`: empty CACHE_DIR, dry-run, assert a complete
      argv and a zero exit.
- [ ] Consolidate what one serve records. If growth, compute buffers, phase
      timings and observed performance can all be captured from a single
      healthy serve, a cold machine needs one launch rather than four.
- [ ] Decide what a cold `auto` should do about hot experts. Today it declines
      (no displacement proof), which is the safe answer and also means the
      feature never engages unless a user asks for it by name. That may be
      correct; it should be a decision, not a side effect.
- [ ] Make the first launch say what it is doing. "Measuring this machine;
      later launches will be faster" turns a slow first run from a defect into
      an explanation.

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


### 2026-09-06 HOT-2 / HOT-5 admission follow-up

Cache composition now drops cache-free exactness until cache-on allocation is
observed. Startup and contained-probe CUDA OOMs without a byte count become
typed exact-argv rejections; no fabricated reserve, ubatch change, or disabled
CUDA graph is introduced. Calibration schema 27. Core regression tests and vet
pass. The observed cache-on warmup failure still needs a newly admitted viable
configuration and matched agent A/B before HOT-4/HOT-5 live acceptance closes.
