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

### Milestone 1 — consolidate launch correctness: blocked, scope reduced

Refreshed PR state, superseding the snapshot above: #59 and #60 merged. #58
rebased onto current main and mergeable.

**The central finding is a direct tension this file's review helped expose.**
Two models need opposite things from the same recovery selector:

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

Root cause of the tension, which a bigger step only masked: for an
oracle-planned deficit `DerateCUDAOOMArgsForDeficit` never considers ubatch,
because that is gated on `IsComputeBuffer`. Qwen sits at ubatch 256 untouched
while expert derating stalls at `n-cpu-moe` 21 -> 22, so context is pressed into
relief it cannot deliver. **The fix is lever selection with credible relief on
the failing resource, not a larger context step.**

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

### Next concrete deliverable, with four hypotheses already eliminated

Make recovery choose a lever with credible relief on the failing device. Four
attempts were made and reverted; recording them so they are not repeated:

1. **Widen the context step to the measured deficit.** Fixes Qwen, breaks GLM
   (662,528 -> 236,544 in one round; the plans at that depth are refused by the
   guards). Reverted.
2. **Bound that step to a quarter of the window.** Converges close to main's
   500,736 but still fails GLM. Reverted.
3. **Allow compute memory as a last-resort lever for oracle-planned deficits.**
   The branch is never reached: the earlier expert lever reports success first.
   Reverted.
4. **Rank levers by `candidateRelievesFailedDevice` before accepting one.** That
   predicate returns **true** for the weak derate, so the ranking never fires.
   Reverted.

What (4) established is the sharpest available diagnosis: the lever set is not
the problem and `candidateRelievesFailedDevice` is too permissive. On
Qwen3.8-Flash-Next it credits a change as relieving CUDA2 while `n-cpu-moe`
stays at 22 and the measured deficit falls only 101 -> 94 -> 87 MiB, about
7 MiB per round, which is the size of the 1,024-token context nudge rather than
of any expert movement. Recovery believes it is relieving the failing device and
is not.

The work is therefore to make relief a **measured quantity** compared against
the deficit, not a boolean derived from argv differences: predict MiB freed on
the failing device for each candidate lever, rank by that, and accept the weak
candidate only as an explicit fallback. That requires reading the per-device
ledger rather than comparing flags, which is why it is a design change and not a
fifth point patch.

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
