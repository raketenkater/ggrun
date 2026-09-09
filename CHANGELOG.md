# Changelog

## Unreleased

- Automatic challenger admission now recognizes CUDA warmup OOMs even when
  the driver omits an allocation size. These produce a typed rejection rather
  than an inconclusive result. Cache-free allocation plus calculated expert
  cache bytes no longer counts as exact proof for the changed cache-on graph;
  real admission remains required. Calibration evidence is schema 27.

- Last-run throughput now separates completed prompt and decode timing rows.
  Fast decode is retained regardless of model size, and cumulative prefill
  progress no longer counts as multiple completed prompts. Old ambiguous
  throughput summaries are ignored until remeasured; placement evidence is
  unchanged.

- Probe-cache updates preserve each retained measurement's GPU role and growth
  source. Updating compute buffers no longer turns OOM evidence into healthy
  serving evidence, and recording growth no longer erases compute roles.
  Older probe metadata is treated conservatively until remeasured.

- Related memory observations now match the model artifact, backend build and
  features, hardware, and slot count before informing a new placement. This
  prevents another backend's measurements or a same-name model replacement
  from changing the current launch's memory reserve. Compatible observations
  remain reusable without another model load.

- **ggrun now always launches: a terminal safe-floor tier replaces `os.Exit(1)`.**
  The optimizer was fail-closed at every stage with no guaranteed-serviceable
  floor, so a *prediction* miss became a *terminal* failure — three separate
  ones on the same rig in one day. When every ordinary path and the advisor
  retry are exhausted, `placement.ComputeSafeFloor` descends a bounded ladder —
  single split owner on the roomiest GPU, hot experts and speculation off,
  microbatch 64, then evict GPU-seated companions, then KV to host, then the
  minimum context, then CPU-only — and serves the first tier that resolves. Each
  rung is a complete placement from the ordinary `Compute` path, so exact-argv
  admission still gates it; the CPU-only rung is mandatory because it is the
  only one that does not depend on companions or on the driver-side CUDA-graph
  executable no oracle predicts.
  Contract invariant 3 is preserved: the floor moves only coordinates the user
  left automatic, and refuses — naming the new `--safe-floor` opt-in — when it
  would have to surrender a pinned one. `--parallel` is never surrendered by any
  rung, because runtime graph growth (the one measurement a floor launch
  uniquely produces) only transfers to another key at a matching slot count.
  A floor placement is never promoted, never cached as a winner, and never
  suppresses the optimizer. Every activation prints the original failure in full
  plus each surrendered coordinate, and is recorded in
  `safe-floor-activations.json`; a second activation for the same launch scope
  is reported as a defect rather than as normal operation.
- **A per-GPU compute-buffer measurement is now tagged with the role it was
  taken in, and never charged to a different role.** A device that carried only
  pinned expert tensors measures a compute buffer orders of magnitude smaller
  than a split owner's. That reading was being replayed verbatim onto a later
  plan that made the same device a split owner, so the ledger under-reserved it:
  live 2026-09-02, CUDA2 carried a 192 MiB expert-only reading, placement put
  4184 MiB of weights and 800 MiB of KV on the card, and llama.cpp then billed
  the real 4570 MiB and failed the load in `graph_reserve` — five contained
  probes, 24 minutes, then the launch gave up. The probe cache now stores
  `PROBED_COMPUTE_BUF_EXPERT_ONLY_CUDA<n>` beside each reading, and a
  split-owner plan falls back to the aggregate or the cold estimate rather than
  charging an expert-only value. The role is recorded, not inferred from
  magnitude: a secondary split owner legitimately measures far below the primary
  (DeepSeek-V4 recorded 599 against a 17970 primary), so no ratio separates the
  two cases. Readings written before the flag existed stay trusted and heal on
  the next measured run.
- **`hot-experts=on` can no longer make a launch shape permanently unlaunchable.**
  Sizing the cache requires an allocation-measured cache-free ledger for the
  exact key (model, context, ubatch, KV quality/placement, backend, slots), and
  only a completed launch of that key records one — so `on` refused every shape
  that had never run: the cache needed evidence, the evidence needed a launch,
  and the launch was refused for want of the cache. Changing KV quality or
  context was enough to trigger it (`Error computing placement: hot experts
  required but no cache-on placement was admitted: exact cache-free allocation
  evidence is unavailable`). That one condition is now the typed
  `ErrHotExpertBaselineUnmeasured`: `on` serves the cache-free baseline for a
  single launch to measure it, says so in the optimization summary, and engages
  the cache on the next launch of the same shape. Every other reason — an
  unsupported layout, mmap, speculative decode, a backend without the exact flag
  pair — still fails closed, and `auto` is unchanged.
- **The TUI no longer offers a resume for a session whose model is gone.**
  `loadResumableSession` used `claudesession.Latest`, which returns the newest
  record whatever its state, and never consulted the package's own
  `ModelPathExists`. A six-week-old record pointing at a since-deleted GGUF was
  still shown as `[R] Resume`, and taking it failed the launch instantly on a
  path that no longer exists. It now requires `LatestRecoverable` (something to
  actually reopen) plus a model file still on disk; a stale record is simply not
  offered and the ordinary launch is unaffected.
- **A CUDA OOM inside the lifecycle-verification canary now recovers instead of
  exiting.** The first real decode (e.g. `cudaGraphInstantiate` for the decode
  graph) can OOM after the server already loaded and passed health -- a seam the
  startup ladder (load only) and the serving-loop runtime ladder (post-verify
  only) both missed, so it exited 1 with no retry. The verify path now reads the
  crashed server's log, and on a post-`model loaded` CUDA OOM it fully reaps the
  process, records the per-device graph-growth deficit, drops any staged
  calibration decision for the crashed placement, and re-plans + re-verifies,
  bounded to two attempts. For a graph-capture abort specifically
  (`ggml_cuda_graph_evaluate_and_capture` / `cudaGraphInstantiate`), the first
  rung is the targeted lever: cancel the multi-GPU `CUDA_SCALE_LAUNCH_QUEUES=4x`
  child-env multiplier (level 2 also sets `GGML_CUDA_GRAPHS=0`) and relaunch the
  exact placement, since graph-exec memory scales with graph node/split count,
  not ubatch or weight residency. Preflight now discloses when a GPU-backed
  placement has no measured runtime graph-growth evidence rather than implying
  its "fits" is a guarantee.
- **A failed hot-expert cache can no longer strand the degraded all-CPU-expert
  topology, and `auto` no longer serves the cache on prediction alone.** When
  the priority challenger demotes GPU expert layers to make cache room, it now
  captures the exact packed pre-demotion placement. A cache-on admission or
  lifecycle failure restores that captured baseline (or, on a verified-config
  replay with no snapshot, recomputes a cache-free placement and fails closed)
  instead of stripping only the cache flags and keeping the inflated
  `--n-cpu-moe` / trimmed `-ot`. `auto` serves the packed cache-free layout as
  the fail-closed default and hands the cache-on placement to calibration as
  the decode challenger, so the packed-vs-cache-on agent A/B actually runs
  (candidate 0 is the packed layout) rather than the cache being promoted to
  the first serve. Calibration schema 24 retires decisions recorded under the
  pre-24 default. A verified cache-on record whose `hot_expert_evidence_schema`
  predates the current admission semantics replays without its cache while its
  cache-free fit still applies. `on` and a numeric slot count are unchanged:
  they serve the cache or fail closed.
- **The TUI can provision hot experts for a model's actual architecture fork.**
  `Hot experts = auto` now offers a one-time isolated composite build on the
  first confirmed MoE launch, then reuses that exact base+feature backend from
  the standard launch resolver. The reviewed patch is applied after any base
  recipe patches and must pass architecture, CUDA, server, and exact flag
  conformance. Source conflicts, binary-only forks, and divergent fork families
  fail closed without modifying the working base backend; repeated installs use
  staged update/rollback rather than rebuilding over the active binary.
- **Hot experts `auto` is a first-class optimizer coordinate.** The packed
  GPU-expert layout remains the cache-free baseline. If leftover VRAM cannot
  hold a useful cache (at least as many slots as experts used per token), the
  first challenger demotes GPU expert layers until it can. The live A/B still
  decides. Automatic launch now keeps that cache-on candidate as the measured
  finalist: the relative cost model does not price cache hits, so a cheaper-
  looking KV or topology neighbor cannot consume the single live A/B slot.
  Auto also applies leftover or demoted cache-on from exact cache-free
  evidence so `--moe-expert-cache` is on the first serve when it can be
  planned; packed GPU experts remain the fail-closed fallback. A completed
  cache-on measurement is the requested coordinate unless it regresses a
  phase. Calibration schema 23 invalidates older default-won records that
  measured the wrong challenger. TUI Settings and the model config screen expose
  `Hot experts` as `auto`/`on`/`off`: `auto` may restore the cache-free baseline,
  while `on` requires an optimizer-sized cache-on launch. A numeric slot count
  remains a CLI exact request. Settings now group the standard auto
  launch path above expert overrides, with `[r]` resetting context/KV/hot
  experts/backend to auto.
- **Standard launch can measure an experimental GPU cache for host-resident
  MoE experts.** The capability-only `hot-experts` recipe is pinned to an
  immutable reviewed fork revision and is never architecture-routed. Automatic
  mode derives one per-layer slot count only from exact residual per-device
  VRAM, validates the backend's allocation acknowledgement, then requires a
  deterministic decode canary with consistent non-zero hit telemetry plus
  material pure-decode and end-to-end agent gains. p2, speculative, mmap,
  fused, row-split, and unaccounted layouts fail closed. Activation or runtime
  evidence failure invalidates the cache-on records and restores the exact
  cache-free placement; public hardware/model acceptance remains experimental.
- **Agent transport hardening now fails early and records truthful phases.**
  Workflow calls with competing `name`, `script`, or `scriptPath` sources are
  rejected instead of silently following tool precedence; materialized scripts
  are read back before approval, and large protocols are kept by file reference.
  Request telemetry retains HTTP TTFB but measures prefill/decode from the first
  generated SSE delta. Claude Code stage-1 verdicts whose configured
  `</block>` stop sequence is correctly stripped from response content are
  accepted unchanged and recorded distinctly; prose,
  conflicts, and incomplete decisions still take the safe main-model fallback.
- **Host-offloaded agent serving now schedules active phases separately from
  allocated slots.** Multi-slot MoEs keep their requested context capacity,
  but long cold prefills serialize across the shared CPU-expert/PCIe path;
  bounded small or cache-hot appends may enter an idle slot only after the
  active request emits its first generated SSE delta. Router status exposes prefill/decode
  occupancy, and cancellation telemetry identifies queue versus service plus
  recurring 60-second and 600-second client-deadline signatures.
- **Reviewer verdicts honor Claude Code's stop-stripped stage-1 contract.**
  Claude Code deliberately configures `</block>` as a stop sequence and accepts
  the closing tag as optional; llama.cpp therefore returns `<block>yes` or
  `<block>no` while counting the sampled delimiter tokens. ggrun previously
  rejected those healthy answers and retried them on the slow main model. The
  reviewer route now keeps a 32-token tokenizer-safe budget and accepts only
  an exact, unambiguous stop-stripped verdict without rewriting it. Opaque
  named Workflow calls whose agent timeout cannot be rewritten are rejected
  before work begins with an inline/scriptPath retry instruction instead of
  silently restoring the private long-running-agent deadline.
- **Phase diagnostics now retain measured PCIe traffic.** NVIDIA sampling maps
  per-GPU RX/TX MiB/s into cold-prefill, append, decode, and mixed evidence.
  PCIe is called saturated only when measured traffic crosses 65% of a known
  link ceiling; otherwise DDR, synchronization, and transfer remain explicitly
  unresolved. Calibration schema 19 also records the exact class and reason of
  an unavailable finalist and refuses to suppress a retry from an unexplained
  failure.

- **Agent-parallel optimization now measures the concurrency it intends to
  serve.** The inherited single-slot server default no longer collapses the
  `agent-parallel` workload to one runnable turn. Automatic launch models at
  least two independent agent lanes, removes slot-width challengers above the
  declared demand, and still measures p1 versus p2 before changing a
  host-offloaded model's live admission. A validated winner admits only the
  number of lanes represented by its workload evidence; cold and
  admission-only launches remain serialized, while explicit `--parallel` and
  `--claude-max-active` continue to win.
- **Automatic optimization no longer reloads a known-unavailable finalist on
  every identical launch.** A bounded challenger ladder that fails only typed
  exact-admission checks is cached as scoped admission evidence after the
  restored baseline passes its lifecycle gates. That evidence suppresses the
  same destructive retry without claiming that the baseline is fastest;
  timeouts and incomplete benchmarks remain retryable.
- **Recurrent context checkpoints are charged before host cgroups are
  tightened.** Hybrid/recurrent launches reserve checkpoint RAM per checkpoint
  and per slot even when GGUF metadata exposes no sliding-window geometry. The
  first functional canary's exact backend-reported checkpoint size is persisted,
  applied to the live host footprint, and carried by verified configs. A later
  cgroup OOM is now a typed runtime failure that revokes unsafe placement,
  calibration, lifecycle, and verified-config evidence instead of appearing as
  an unexplained server exit.
- **Cheap-tier and classifier calls use the seated reviewer instead of
  queueing behind main-model work.** With a review-only reviewer (no worker
  companion) the router previously sent `local-fast` traffic — Claude Code's
  permission classifier and haiku-tier background calls — to the main model,
  where 45–80 s behind a single slot's long foreground streams timed out tool
  calls and blocked Workflow fan-out agents. The cheap tier now routes to the
  seated reviewer whenever the prompt fits its context, keeping the same
  visible-notice and overflow-fallback contracts as the classifier lane.
- **The routed prompt token estimate errs high.** `estimatedPromptTokens`
  divided body bytes by 3 and could score code-dense or escaped-JSON prompts
  below the reviewer's real window: reviews of 65,675 tokens arrived past a
  65,536-token estimate and overflowed the reviewer with a 400. A 2-byte
  divisor replaces the 3-byte one; a conservative false positive merely falls
  back to the main model before the reviewer is overrun.
- **Rejected reviewer verdicts are recorded.** A reviewer answer whose text
  fails the `<block>` verdict contract now writes a
  `reviewer-rejected/invalid-verdict` metrics row next to the main-model retry,
  instead of leaving a template or format mismatch indistinguishable from
  reviewer unavailability in the request log.
- **Standard launch now diagnoses performance per agent phase.** The bounded
  workload retains separate cold-prefill, cache-backed-append, decode, and
  mixed-traffic GPU samples plus Linux process-tree CPU, RSS, and disk-I/O
  deltas. A typed conservative classifier distinguishes capacity, GPU
  compute/memory/topology, host execution, the still-ambiguous CPU-expert path,
  storage faults, scheduler starvation, and pipeline underfill. It can
  prioritize one matching complete finalist, but exact admission and the
  repeated phase-regression-gated live A/B remain the only promotion authority.
- **Core optimizer edits now have an explicit review boundary.** Root agent
  instructions and Claude memory import a written change contract, critical
  optimizer paths have CODEOWNERS, and a dedicated CI check runs the uncached
  `scripts/verify-core-engine.sh` formatting, focused-test, and vet gate. The
  contract preserves fit-before-speed,
  complete configurations, explicit user constraints, phase safety, versioned
  evidence, and bounded lifecycle behavior.
- **Claude Code status line now shows decode tok/s, not only prefill.** The
  progress monitor took its decode rate only from the `/metrics` gauge, which
  submits a task into llama-server's scheduler and therefore timed out on the
  monitor's 3 s budget exactly while a long decode owned the scheduler — so
  the status line showed prefill-only during generation. The monitor now
  parses the backend's periodic `tg_3s` windowed-rate log line (the same
  per-task stream prefill already used), prefers it over the whole-run
  `/metrics` average, gives that scheduler-backed endpoint one short
  cancellable attempt with an independent two-minute backoff, and merges
  prefill + decode rates in the passive log path.
- **CPU-expert launches pin llama.cpp to physical cores for prefill and
  decode.** Agent turns pay the same DRAM-bound expert GEMM on cold ingest and
  on token generation. For CPU-expert paths, ggrun now passes `--cpu-range` /
  `--cpu-strict` and only exactly advertised batch counterparts, using a
  contiguous set of online physical cores allowed to the Linux process. It
  omits affinity when topology/capability proof is incomplete, re-derives it on
  verified-config reuse, and scopes performance evidence to the resulting host
  policy. A host script documents the remaining Linux knobs that need root.
- **Model start now reports overall elapsed time and ETA.** The health-wait
  spinner already showed a phase line on a TTY; Claude Code and launch logs
  only recorded "health check OK after 12m". ggrun now prints the model size
  at start, keeps a live status line on a TTY, and writes a newline progress
  snapshot about every 10 seconds (elapsed, remaining estimate from the
  observed read rate, bytes read, phase) until the server is healthy.
- **qwen4exp PLE accounting is versioned and no longer charged as GPU VRAM.**
  A launch that cannot fit with "no GPU has free VRAM after CUDA/compute
  overhead" was packing the ~27 GiB `per_layer_token_embd` table onto every
  card because `~/.local/bin/parse_gguf.py` predated host-resident accounting.
  ggrun now prefers helpers exposing tensor-accounting schema 2, records PLE
  tensor presence separately from its byte count, and fails closed only when a
  stale parser cannot make that distinction or reports an unaccounted present
  tensor. Valid qwen4exp GGUFs with no optional PLE tensor remain supported.
- **Guarded allocation evidence now survives unlabelled backend buffers without
  leaking across placements.** Complete CUDA peaks and the host cgroup peak are
  persisted with a hash of every allocation-affecting strategy coordinate.
  Matching aggregates are exact even when a backend reports model weights only
  as `unaccounted`; a changed split, batch pair, KV/mmap/checkpoint policy, or
  speculation shape cannot reuse them. A later KV-only launch observation no
  longer erases the complete guarded breakdown. Calibration schema 17 also
  retires baseline-only decisions that were previously marked performance
  tuned without a measured challenger outcome.
- **Unsupported architectures search open llama.cpp PRs and Hugging Face GGUF
  cards for a supporting fork.** When no installed backend can load a GGUF
  architecture and no reviewed recipe names it, ggrun queries the official
  llama.cpp pull-request index and the publisher's Hugging Face model card
  (from local `quantized_by` / name metadata, then a Hub architecture search).
  Only official `ggml-org/llama.cpp/pull/N` links are followed. The fetched PR
  must return the same official number/URL, mention the requested architecture
  in its current title/body, expose a GitHub HTTPS head, and pin a hexadecimal
  40-character commit; detail fetches are bounded. Publisher-cited PRs and
  titles that add the architecture rank ahead of later fix PRs that merely
  mention it. An open unmerged github.com head is offered as an isolated
  `.src/fork-*` clone before falling back to a mainline backend update. The
  lookup is architecture generic; a network miss fails closed to the existing
  update prompt.
- **Standard launch now keeps MoE fit placement separate from performance
  hypotheses.** The stable planner still resolves the fit-first baseline; a
  sole `--tensor-split` owner is generated only as one fully recomputed roomy
  challenger. The frontier ranks candidates by phase-aware backbone, routed
  GPU-expert, CPU-expert, and activation-transfer cost rather than by an
  `moe-owner-*` name. Optimizer challengers fail closed on the first exact
  admission miss instead of walking the fit recovery ladder.
- **Roomy MoE optimization now distinguishes compute ownership from storage.**
  The bounded frontier fully recomputes every feasible GPU as a sole
  ordinary-layer/KV/output owner while the remaining cards store complete
  routed-expert layers. PCI-keyed SM sampling spans the complete agent trial;
  sustained imbalance between ordinary-layer owners can help select one
  predicted-faster topology. An idle expert-storage card is not itself a
  balance defect, and utilization cannot promote a costlier owner. Exact
  allocation admission and the normal live workload/lifecycle gates remain
  promotion authority. Recovery to a safe argv no longer suppresses roomy
  performance measurement, while exact-ledger tight launches retain their
  proven-shape filter.
- **The automatic agent screen now fits its own time budget and scores cold
  work.** A short uncached prefill pilot selects one rounded per-lane prompt
  geometry for both baseline and finalist; slow CPU-expert MoEs no longer time
  out trying to run the fixed two-lane ~8k-token prompt inside a 20-minute
  controller, while fast models retain the full screen and explicit maintenance
  remains full-length. Promotion now minimizes the repeated cold-ingest plus
  cache-backed-append scenario instead of ignoring cold prefill in the score.
  Prompt bytes are persisted, mismatched geometry fails closed, and workload
  profile/calibration schemas invalidate older append-only decisions.
- **Automatic core optimization now measures one finalist, not every idea.**
  Standard TUI/CLI launch calculates the complete safe candidate set, selects
  one highest-confidence neighbor (preferring exact allocation evidence), and
  benchmarks only it against the stable baseline. A failed finalist or baseline
  win becomes a reusable scoped decision after normal canaries; explicit `--calibrate on`
  retains the wider maintenance sweep. Tuned batch/ubatch survives every
  backend-measured placement recompute. Exact allocator evidence can no longer
  authorize a changed argv: denser MoE replans are contained again and lateral
  tensor-split churn retains the proven placement. Very large models now print
  a long-load warning before the first backend or reviewer start.
- **Parallel-agent batch and microbatch are now measured together.** Claude
  hybrid/recurrent serving starts from a valid `128/128` fairness baseline
  instead of emitting `128/512` and relying on llama.cpp's silent clamp.
  Candidate generation covers a bounded `256/128` through `512/512` ladder;
  automatic launch screens one calculated finalist and maintenance mode can
  traverse more. The workload uses two repeated cold-ingest-plus-cache-append
  scenarios, aggregate decode, and foreground decode under competing prefill.
  End-to-end workflow wall time—not a weighted tok/s score—is authoritative.
  Every batch pair is recomputed through the coupled
  placement ledger, and an exact/explicit ubatch can no longer be silently
  derated into another configuration. Recovered or runtime-OOM shapes never
  become screened winners. Screened decisions are reusable only with explicit
  `--calibrate on`. Ordinary `auto` launch now owns a bounded candidate search
  and promotes its exact winner only after the repeated agent screen, cache
  branch/replay, functional/gateway, lifecycle, and clean-relaunch gates.
  Workload and calibration schema versions retire the old evidence, explicit batch intent
  is preserved, and tune overlays enforce `ubatch <= batch`. AI Tune cannot
  mutate batch/ubatch in any workload, including through old cached/community
  overlays, because it does not own placement recomputation; parallel-agent
  Tune uses the same direct turn-time objective for its remaining safe flags.
  Full verified-config replay now preserves hybrid/MoE runtime semantics, and
  its schema plus planner scope are advanced so an older direct-start record
  cannot bypass the recurrent fairness or context-shift policy.
- **Hardware bandwidth can be measured and bound to the exact machine.**
  `ggrun detect --bandwidth` measures host memory and pinned CUDA transfer
  paths, persists only a complete CPU/RAM/GPU/PCI fingerprint, and feeds valid
  observations into MoE placement and recommendation speed estimates. Stale,
  corrupt, partial, or mismatched profiles fail closed to derived topology.
  (`8da0520`)
- **Dense multi-GPU placement uses the fastest VRAM first.** When every device
  has known memory bandwidth, fully resident layer splits fill the fastest
  usable VRAM before adding slower cards, account for combined weights and KV,
  and leave unnecessary GPUs available. Degenerate zero-capacity and
  all-zero-tensor-split cases now fall back safely instead of emitting invalid
  backend arguments. (`695ed42`, `1509bcb`, `10ba58f`)
- **Automatic context fit now spends VRAM deliberately.** `--ctx fit` sizes
  against available VRAM, keeps KV resident where possible, rounds only to
  1024-token granularity rather than discarding up to half the proven window,
  and weighs a larger multi-GPU context against its predicted dense decode
  slowdown. Explicit oversized contexts remain user-owned but receive a
  warning derived from the planner's actual answer. Verified configs are
  versioned so plans from older placement logic are retired. (`3e537d3`,
  `70098f2`, `e74cd05`, `964c6f1`, `7d29692`)
- **Offloaded MoE calibration measures larger microbatches.** Explicit
  `--calibrate on` can challenge the default with bounded 1024/2048
  ubatch candidates, subjects each exact argv to allocation and lifecycle
  checks, and refuses to cache a candidate that needed memory recovery into a
  different shape. Runtime OOM invalidation removes every calibration decision
  for the affected model. (`9c7156d`)
- **Launch safety and diagnostics are fail-closed.** Unknown commands no longer
  become accidental model launches; model errors are reported before waiting
  on an old server; `benchmark` validates its model, flags, port, and running
  server; contained-probe failures retain their evidence path and bounded
  recovery guidance; and KV presets promote to f16 when a model's K/V head
  dimensions cannot represent block-quantized cache types. Exact incompatible
  KV requests fail clearly. (`a9cc4b3`)
- **`--gpus` cannot silently broaden or narrow its hardware scope.** A
  selection naming a missing device is rejected with the detected device list
  instead of dropping indices—or falling back to every GPU when none remain.
  (`8767514`)
- **Claude's local reviewer/worker path is more reliable and explicit.** A
  seated reviewer that crashes, rejects a request, or returns an unusable
  response is buffered and visibly retried on the admitted main route. The
  auto-installer uses the pinned artifact that matches its VRAM reservation and
  supports slow/resumable downloads. Help, TUI labels, dry-run output, and flag
  validation now agree on worker/reviewer roles and the `off` behavior;
  the Qwen3.5-2B review-only profile cannot receive ordinary `local-fast`
  worker traffic; arbitrary reviewer-model overrides also default to that
  review-only capability. Launch verification now exercises the actual XML
  safety-verdict route, and runtime fallback rejects non-2xx, empty, thinking-
  only, tool-only, ambiguous, or malformed reviewer responses before Claude
  sees them. Non-interactive TUI/config commands fail early with actionable
  guidance.
  (`44f99a0`, `e1a95d4`, `7b70ed8`)
- **Claude request telemetry separates work from cancellation.** Requests
  cancelled before their first response byte no longer inflate streamed decode
  time or depress reported decode throughput. Router summaries now distinguish
  successful, failed, and aborted requests and retain aborted queue/service
  totals for capacity diagnosis.
- **Source-checkout updates no longer assume `~/ggrun`.** Update and backend
  maintenance resolve the active checkout, preserve install state, and work
  when the repository lives elsewhere. (`971e196`)
- **The model recommendation catalog was refreshed.** Generated catalog data
  and its recommendation fixtures reflect the latest reproducible snapshot.
  (`88f6f81`)
- **Experimental FreeToken adapter.** An explicitly experimental command can
  launch a FreeToken-compatible sidecar with health checks, bounded request
  handling, and process cleanup. It remains a separate optional development
  lane and is not part of the GGUF placement or release gate. (`eb19f46`)

## v3.2.8 — 2026-08-20

- **An Auto review that cannot reach a separate reviewer goes to the main model.**
  A classifier request used to route unconditionally to the reviewer, so when no
  separate reviewer was seated it silently ran on the main model while being
  recorded as a "reviewer" turn and skipping main admission control, and an
  oversized prompt was sent into the reviewer's context regardless. Both cases
  now route to the main model explicitly (self-classify), each with a visible
  notice, and are recorded as `route=main` so they take and release a main
  admission slot — they do run on that backend. The overflow threshold is the
  same constant that builds the reviewer's `--ctx-size`, so the routing window
  cannot drift from the launch.
- **Explicit "off" reviewer option (no separate Auto reviewer).** Whether a
  separate reviewer was seated was purely a VRAM decision — on a box with
  headroom it always launched, with no way to decline it. `--claude-reviewer off`
  and a matching "off" entry in the TUI reviewer/worker selector now seat no
  reviewer at all: no companion reservation, the proactive VRAM gate returns
  early instead of prompting (picking "off" IS the self-review consent), and no
  reviewer process starts. The router then sees no separate reviewer and
  self-classifies with the existing visible notice, so the choice stays explicit
  rather than silent.
- **Installer no longer aborts mid-upgrade on a fork-named llama-server.** With
  any backend path matching `fork-`/`fork_` detected, `report_system_installs`
  ended on a false loop test whose status became the function's return value,
  killing the whole install under `set -e` before any backend work — the failure
  behind "install failed: Linux cuda". `warn_probe_detail` had the identical bug
  and is fixed the same way, so "backend did not start, fall back" no longer
  turns into a hard failure on the CUDA path. Failure reports now also name the
  enclosing function so `BASH_COMMAND`/call-site lines cannot disagree.
- **Config-screen lag, self-review, resume-across-model, decode-in-status.**
  The model-config screen cached two expensive file reads so renders are O(1)
  when config is unchanged; the separate reviewer is only dropped for a VRAM
  model with explicit consent (never silent self-review); "resume latest"
  launches the current model even when the recorded one differs; the status line
  shows both prefill and decode tok/s when both are known.
- **Install keeps the CUDA bundle and still installs llama.cpp.** A downloaded
  CUDA `llama-server` is no longer deleted when `--version` fails on a missing
  CUDA runtime library (NCCL, cublas, cudart). Setup then installs llama.cpp
  Vulkan/CPU instead of compiling ik_llama.cpp. The real `--version` error is
  printed. Source compile runs only when no CUDA ELF landed.
- **ggrun uses a backend that can start.** A CUDA binary that cannot load
  `libnccl.so.2` no longer becomes `LLAMA_SERVER` or wins auto-select over a
  working Vulkan/CPU llama.cpp.
- **CUDA install fetches NCCL when the bundle needs it.** `libnccl.so.2` is
  not the NVIDIA driver and is not in a typical CUDA toolkit. Setup downloads
  NVIDIA's public NCCL redist into the isolated backend box so the existing
  CUDA tarball can start.

## v3.2.2 — 2026-08-17

- **Linux CUDA bundle on latest.** Publishes `ggrun-linux-x86_64-cuda.tar.gz`
  so one-command Linux setup downloads ik_llama.cpp instead of compiling it.
  v3.2.0 and v3.2.1 never attached assets (release preflight died).

## v3.2.1 — 2026-08-17

- Tag only; release preflight failed before package-cuda.

## v3.2.0 — 2026-08-17

- **Linux CUDA release bundle.** The tag publishes `ggrun-linux-x86_64-cuda.tar.gz`
  (pinned ik_llama.cpp) when the CUDA job succeeds. `install.sh` / `install.ps1`
  are attached and listed in `SHA256SUMS`. Self-update refuses an installer that
  is not in that file.
- **Reviewed-recipe selection.** An already-registered MiniMax-M3 (or other
  recipe) backend no longer exempts a *different* selected binary that does not
  implement the architecture.
- **Install docs** list the v3.1.0 assets (CPU/Vulkan/Metal/Windows CPU) and
  treat Linux CUDA as a tagged asset, not a guaranteed historical file.

- **Local Claude Code research no longer dies with `local is temporarily unavailable`.**
  Claude's Auto permission mode depends on a separate supported classifier path and can
  reject Workflow, MCP, WebFetch, and Bash calls before execution on a custom local
  endpoint. ggrun now launches local sessions in non-bypass `acceptEdits` mode, keeps
  the exact DuckDuckGo search/fetch tools pre-approved, and supports an explicit
  `GGRUN_CLAUDE_PERMISSION_MODE=auto|inherit` override.
- **Claude Code shows live local inference progress.** `--claude-code` now adds a
  session-only status line showing queued requests, the active slot, prompt progress,
  token counts, prompt/generation tok/s, and parallel activity. It reads llama-server's
  slots/metrics plus prompt-progress logs without storing prompt content, preserves an
  existing custom Claude status line, and can be disabled with
  `GGRUN_CLAUDE_PROGRESS=off`.
- **Large-MoE placement is topology-aware and load-tested at 1M context.** Slow-link
  GPUs are used as whole-expert storage while dense ownership stays on the fast GPU;
  routing tensors follow their experts, every owner receives measured graph reserve,
  expert placement respects live free VRAM/OOM penalties, and generic huge-KV MoEs can
  move KV to CPU. The DeepSeek-V4 reference plan completed a 60,020-token request plus
  three concurrent workers without OOM or restart. Server and Claude Code request
  timeouts now default to four hours so valid long-context fan-out is not cancelled.

- **Windows GPU without a toolchain.** `install.ps1 -Backend cuda` now downloads
  prebuilt llama.cpp CUDA binaries (server + cudart) from ggml-org by default; the
  from-source build (CUDA Toolkit + MSVC + CMake) remains only as a fallback. Third-party
  installs (the CUDA download, Python via winget) ask for consent first — `-AssumeYes` /
  `LLM_INSTALL_NONINTERACTIVE=1` keep CI and piped installs non-interactive.
- **Windows: CPU-only bundle no longer crash-loops on NVIDIA systems.** The
  default CPU bundle used to emit CUDA offload flags its own binary couldn't
  honor (`unknown buffer type`), which the launcher hid as `failure: unknown:`
  and retried forever. ggrun now probes the active backend's real device support
  before placement: a CPU-only build on a GPU machine runs CPU-clean with a note
  on how to get GPU (`install.ps1 -Backend cuda`). Deterministic load failures
  now fail fast with the backend's real error instead of an infinite restart loop.
- **Windows: model-download dependencies install into the right interpreter.**
  The installer and runtime resolved Python in opposite orders, so
  `huggingface_hub` could land in a different interpreter than the one ggrun ran
  — a `ModuleNotFoundError` despite a "successful" install. Both now resolve
  `python3 → python → py` and validate the interpreter; the downloader prints an
  actionable message (naming the exact interpreter) instead of a raw traceback.
- **Recommender only suggests models the bundled backend can run.** Frontier
  catalog entries whose architecture no shipped backend can load (e.g. DeepSeek
  V4's `deepseek4`) are no longer recommended — they stay in the catalog for
  users with a custom backend build. Best-overall now picks each model's best
  *practical* quant (blended quality, speed, and fit) rather than the
  highest-quality quant that merely fits, so e.g. a 27B surfaces at a fast
  single-GPU Q5 instead of a RAM-spilling BF16.

## v3.1.0 — 2026-06-20

- **Renamed to `ggrun`** (formerly `llm-server`; pronounced "g-run", from "gguf
  run"). Existing `~/llm-server` app homes auto-migrate to `~/ggrun` on the next
  `setup.sh`, and the old `llm-server` command keeps working via a symlink.
  GitHub redirects a renamed repository's old URLs as long as the old name
  isn't reused.
- **PCIe-bandwidth-weighted MoE tensor-split.** On heterogeneous-PCIe rigs (e.g.
  a card stuck at x1) the split now concentrates layer ownership on the
  fastest-link GPU, so CPU-expert streaming isn't bottlenecked — up to 3.4x
  prefill on MiniMax-M3 with no decode regression. Symmetric rigs fall back to
  the previous free-VRAM-proportional split.
- **Model recommendation ranking.** The download picker ranks by effective
  intelligence (AA index × quantization quality retained) × predicted speed ×
  fit, in three categories (best for your machine / smartest that fits / fastest
  capable), preferring Unsloth dynamic quants; the catalog auto-refreshes.
- **Clear backend/architecture errors.** Launching an ik_llama-only architecture
  (e.g. `minimax-m3`) on a mainline llama.cpp backend now fails fast with the fix
  instead of a cryptic load crash.
- **Launch UX.** An animated startup status replaces the raw backend log
  spam while a model loads; the TUI config screen is grouped into Context /
  Tuning / Run mode / Actions sections.
- **Exact-ledger multi-GPU MoE placement.** Large MoE models now load reliably
  on heterogeneous multi-GPU rigs instead of over-committing the smallest card.
  The launcher emits `--tensor-split` *and* `-ot` from an exact per-GPU VRAM
  ledger — measured CUDA-context + compute-buffer overhead plus GGUF-exact
  non-expert, KV, and expert sizes — fills expert layers in GPU-bandwidth
  order, honors leading dense blocks, and drops GPUs that can't carry their
  share. No percentage headroom; every term is measured or read from the GGUF.
  A `cudaMalloc` out-of-memory during load now triggers an adaptive retry that
  derates the offending GPU and caches the corrected placement.
- **Interactive GUI overhaul.** The whole TUI is navigable with arrow keys and
  Enter (letter hotkeys still work). New Settings screen lists every config
  option with its current value — enums cycle, booleans toggle, no typing —
  and saves to the config file; backend selection is an arrow-select; the
  advanced launch screen is fully navigable. Fixes an unreachable keep-alive
  toggle.
- **Reliable shutdown.** Ctrl+C now exits promptly; the keep-alive recovery
  loop no longer treats a requested shutdown as a crash and restarts the server
  being stopped. The GUI launch path gains a second-Ctrl+C / timeout force-quit.
- **Config is the single source of truth.** The installer no longer exports
  per-setting environment variables (model dir, backend, cache, logs) that
  silently shadowed the config file, so CLI and GUI edits actually take effect.
  Only `LLM_APP_HOME` and `PATH` are exported.
- **One launcher binary.** Installs a single `llm-server` (no duplicate
  `llm-server-go` copy, no `llm-server-gui` wrappers) — `llm-server` with no
  arguments opens the GUI. The `llm-server-bash` v2 migration shim is retained.
- **Dockerfiles** for CPU, CUDA, and Vulkan, plus an Open WebUI compose file (build locally; GHCR publishing is not wired yet).

- **Community tune pool.** When a model has no local AI-Tune cache, the
  launcher now checks a shared pool (one HTTPS GET keyed by
  model+size+GPU-set+backend, mirroring the local cache file naming) and
  applies a community-measured config after sanitization. Only benign
  performance flags survive (batch/threads/flash-attn/spec settings); model
  paths, ports, devices, placement flags, and output-quality flags (KV-cache
  quantization, parallel slots) can never be injected.
  Hits cache for 7 days, misses for 24 h, lookups time out in 3 s — fully
  offline-safe. Disable with `LLM_COMMUNITY_TUNES=off`; point elsewhere with
  `LLM_COMMUNITY_TUNES_URL`. After a successful `--ai-tune`, the launcher
  prints how to contribute the result back.

- **Apple Silicon: Metal now actually engages.** Hardware detection knew only
  nvidia-smi/rocm-smi/vulkaninfo, so Macs reported zero GPUs and launched with
  `-ngl 0` (CPU-only inference despite the Metal bundle). Detection now
  synthesizes a unified-memory GPU (75% of `hw.memsize`, Metal's default
  working-set limit), macOS backends are tagged `metal`, and CUDA/Vulkan
  device-routing flags are no longer emitted for them. Pending validation on
  real Apple hardware.
- **Native Windows** — PowerShell installer (`install.ps1`), Windows process
  and signal handling, CUDA backend via official llama.cpp prebuilts or a local
  build.
- **Benchmarked against llama.cpp `--fit`** — on a 3090 Ti + 4070 + 3060 rig at
  32k context, v3 default placement beat upstream auto-fit on every model;
  driving the same master binary the win held, so it is placement, not just
  backend choice. See `docs/launch-performance.md`.
- **Rewrote the README intro and repo metadata.** Added a benchmark table
  against raw llama.cpp `--fit` and Ollama, a methodology note for it, and
  the release-asset install path.

## v3.0.0 — 2026-06-11

llm-server v3 is a ground-up Go rewrite of the Bash launcher. The Go binary is
now the primary `llm-server` command; the Bash implementation ships as
`llm-server-bash` for migration only.

- **Go launcher** — single static binary with `launch`, `dry-run`, `tune`,
  `download`, `daemon`, `benchmark`, `detect`, `probe`, `config`, `update`, and
  a Bubble Tea terminal UI (`llm-server-gui`).
- **Measured v3 performance** — see the README benchmark table; v3 IK CUDA
  AI-tune reached +50% decode throughput vs raw llama-server defaults on a
  4B dense model, and stable 32k-context launches where raw defaults OOM.
- **AI Tune v2** — deterministic candidate plan plus optional LLM-proposed
  flags, candidate validation against backend help/VRAM headroom, 1% noise
  floor before replacing the baseline, vision-aware cache files, and resumable
  progress saves.
- **Model recommendations** — weekly-refreshed catalog (Artificial Analysis
  intelligence ratings + Hugging Face GGUF quant search) matched against
  detected hardware in the GUI download picker.
- **Speculative decoding policy** — `--spec auto` only enables validated paths
  (MTP, EAGLE-3, compatible draft GGUF); ngram modes stay explicit.
- **Fixes in this cycle**
  - `--gpus` now actually restricts placement and sets
    `CUDA_VISIBLE_DEVICES`/`GGML_VK_VISIBLE_DEVICES` (it was silently ignored).
  - `launch`/`gui` refuse to start when the port is already in use — the health
    check previously hit the existing server and reported a dead child as live.
  - Startup failures are detected immediately instead of polling the health
    endpoint until the full model-size-scaled timeout (up to 15 min for MoE).
  - Crash classification reads this launcher's own log (not the newest file in
    /tmp) and matches OOM on word boundaries (a model named "Bloom" no longer
    classifies as out-of-memory).
  - `--ai-tune` reuses a completed tune cache and says so; `--retune` forces a
    fresh run (previously accepted but ignored).
  - `firstPositional` knows all value-taking flags (`--parallel 2 <repo>
    --download` no longer tries to download "2").
  - `--update` reports git pull failures instead of claiming success.
  - Version string is single-sourced in `pkg/update` and stamped by release
    builds via `-ldflags -X`.
