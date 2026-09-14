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

Updated 2026-09-14. Detailed measurements live in
`docs/core-standard-launch-todos.md`; this section carries state only.

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
| Windows GPU | **untested** — runner offline, `GGRUN_GPU_RUNNER_WINDOWS` false | — |
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
costs 19% of per-agent context, the 2B seat costs none.

**The benefit side is still unmeasured.** No arm generated review or delegated
worker traffic, so this is the seat's cost with its lane empty. A driver that
issues classifier-marked requests concurrently with foreground turns is written
(`scratchpad/review-lane.py`, `review-ab.sh`) and was interrupted before it
produced results. It needs one uninterrupted GPU window.

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

### Milestone 5 coverage, final for this session

| path | state |
|---|---|
| Uncached core gate, six packages | green at every commit |
| Linux CPU install / download / generate / cancel / shutdown | green |
| Windows install / reinstall / generate | green |
| **Linux real GPU on the merged candidate** | **green** — run 34898139305 on `0a66834` |
| Windows GPU | **untested** — runner offline, `GGRUN_GPU_RUNNER_WINDOWS` false |
| macOS | **untested**, lower priority |
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

```text
/goal Read /home/mik/ggrun-project/ggrun/docs/claude-local-agentic-goal-handoff.md and work through its unfinished milestones. Make ggrun deliver easy, reliable local agentic work on affordable hardware, extracting the best practical performance while preserving useful context and correctness. Refresh current state, implement and verify each milestone, and keep the handoff updated with evidence and remaining work.
```
