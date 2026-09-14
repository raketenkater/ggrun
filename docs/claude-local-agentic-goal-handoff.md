# ggrun: easy, useful local agentic work

User-aligned handoff, 2026-09-14. This is a development plan, not a claim that
the acceptance checks below have passed. Refresh branch and CI state before
acting; preserve the existing dirty production checkout.

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

At the last inspection, PR #58 was open and conflicting at `27812a3`; #59 and
#60 were documentation PRs with checks still running. Refresh this state.
Review and reconcile the fixes on current main, without folding in unrelated
staged work from the production checkout.

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

Updated 2026-09-14. Detailed measurements live in
`docs/core-standard-launch-todos.md`; this section carries state only.

### Milestone 1 — consolidate launch correctness: resolved

Refreshed PR state, superseding the snapshot above: #59 and #60 merged. #58
rebased onto current main and mergeable.

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

### Milestone 3 — worker/reviewer in the same plan: audited, one deliverable left

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

### Milestone 4 — spend remaining capacity on demonstrated bottlenecks: started

First standard launch of Qwen3.8-Flash-Next with the candidate controller
enabled. Detail in the core ledger under `CALIBSPEND`.

The plan it settled on serves **262,144 tokens with 28 of 48 expert layers
resident at 89.5% VRAM**, and returns 7.63 correct tasks/min — inside the
7.49-7.76 band measured at 18,912 tokens. Fourteen times the context at the
same throughput on this suite is the result worth keeping, and it is exactly
what this file means by preserving useful context. The suite's tasks are short,
so it shows serving that context is free, not that long context is free.

**The run exposed a defect in candidate generation.** All three candidates were
larger ubatch values; all three failed admission by 1,885 to 7,026 MiB; the
failure budget was exhausted and the search stopped. Placement had already
printed nine times that ubatch 512 yields no usable whole-layer MoE plan for
this model. Calibration does not consult that. The cost is not three wasted
reloads but that expert packing, topology and slot count are **never reached**
on this model class, which is why the baseline always wins on this shape.

Two bottlenecks are now named with measurements rather than inferred:

- prefill is topology-limited, CUDA0 at 80% SM against CUDA2 at 7% on a
  0.27/0.59/0.14 split;
- decode is in the CPU-expert path with PCIe active but **not proven saturated**
  (52/34 MiB/s), so DRAM and synchronization stay live candidates.

Next for this milestone: stop infeasible candidates from consuming the failure
budget, so the levers that matter on an offloaded MoE get measured at all. The
prefill imbalance has a named lever and no experiment yet.

Also resolved from milestone 2: the `ctx N total / M per agent` line **does**
print on a fresh calibration (`[optimize] roomy-resident, ctx 262144 total /
262144 per agent, ...`). The gap is specific to the cached-decision path, as
recorded.

### Known limitations of current evidence

- Single runs per configuration; no matched repeats.
- One machine, three NVIDIA devices; no CPU-only or single-GPU coverage.
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

```text
/goal Read /home/mik/ggrun-project/ggrun/docs/claude-local-agentic-goal-handoff.md and work through its unfinished milestones. Make ggrun deliver easy, reliable local agentic work on affordable hardware, extracting the best practical performance while preserving useful context and correctness. Refresh current state, implement and verify each milestone, and keep the handoff updated with evidence and remaining work.
```
