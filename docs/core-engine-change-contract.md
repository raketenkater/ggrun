# Core engine change contract

This contract protects ggrun's standard launch optimizer from plausible-looking
changes that violate its actual product goal. It applies to both TUI and direct
CLI launch; calibration is an internal measurement mechanism, not a separate
product lane.

## Non-negotiable invariants

1. **One standard launch path, two ordered stages.** First prove the requested
   model fits stably with useful agent context. Only then spend proven headroom
   on performance. Tight and non-resident launches may not inherit roomy-lane
   experiments.
2. **Complete configurations only.** Context, KV, batch, ubatch, slots, threads,
   mmap/mlock, offload, topology, CPU threads/affinity, memory limits, and
   backend capability flags are one coupled configuration. Never promote a
   partial argv overlay.
3. **Explicit user choices are constraints.** Automatic search may move only
   coordinates the user left automatic. Never silently reduce useful per-agent
   context or quality to win throughput.
4. **Memory safety is fail-closed.** Estimated fit only ranks/advises. Exact argv
   admission, process-scoped failure containment, observed allocations, and
   clean resource release are the authorities. Mmap/offload is a declared
   last-resort capacity path, not a default speed trick.
5. **The objective is real agent work.** Optimize cache-backed turn time and
   requested-concurrency workflow makespan while preserving both prefill and
   decode. Aggregate tok/s alone is never sufficient.
6. **Keep phases separate.** Cold prefill, cached append, decode, and mixed
   foreground-decode-plus-prefill must retain separate evidence. A candidate
   with a material phase regression cannot win on an aggregate score.
7. **Prediction chooses at most a finalist.** Cost models and bottleneck
   classifiers may select one bounded challenger; only repeated, identical live
   A/B evidence can promote it. A baseline-only measurement is not a completed
   performance decision.
8. **Public hardware, conservative claims.** Missing counters and unknown
   backend capabilities stay unknown. Do not encode one server's GPU indexes,
   VRAM sizes, CPU numbering/affinity, PCIe layout, model family, or fork
   behavior as a universal rule.
   **Absent is not zero.** A reserve or margin read from a map, a probe, or a
   log must distinguish "measured, and the value is zero" from "never
   measured", because for a reserve the safe reading of unknown is *large* and
   zero is the least conservative value available. Carry the distinction in the
   type (`RuntimeMeasured`, `MarginMeasured`) rather than in each caller's
   memory: a single-value map read of a missing key yields zero silently, and
   on 2026-09-09 that packed a device to 24008/24112 MiB and aborted the launch
   in warmup.
11. **Required and discretionary allocations answer to different rules.**
   Model weights, KV and the compute buffer are *required*: they may proceed on
   incomplete evidence, because refusing would make the first launch of any new
   plan impossible and no evidence could ever be gathered. The expert cache, an
   extra resident layer, and any other optional spend are *discretionary*: they
   are optional by definition, so the cost of being wrong is an OOM a plan
   without them would have survived. `ComputeExpertSeats` applies this to expert
   seats ("an unmeasured margin buys no seats"). Where a discretionary spender
   does not yet apply it, that is a recorded open decision, not licence.
9. **Evidence is versioned.** When eligibility, workload, scoring, or resource
   semantics change, invalidate or explicitly migrate persisted optimization
   evidence. Old fit proof may remain valid while old performance proof does
   not.
10. **Lifecycle is part of correctness.** Never overlap large model processes,
    disturb an unrelated live server, or turn ordinary launch into an
    unbounded series of long reloads. Warn before the first optimizer-driven
    long reload and restore the measured baseline after failed admissions.

## Required change procedure

Before editing:

- inspect the complete dirty diff and identify ownership of existing changes;
- read `docs/optimizer-theory.md` and the relevant TODO/evidence section;
- state which invariant and hardware/model classes the change affects;
- prefer a new typed component over scattering heuristics through launch code.

While editing:

- keep measurement, diagnosis, finalist selection, admission, promotion, and
  persistence as separate steps;
- attach confidence and evidence to every bottleneck claim;
- preserve backward compatibility or bump the relevant evidence schema;
- add failure-path and missing-evidence tests, not only happy-path tests.

Before handing off:

```sh
scripts/verify-core-engine.sh
```

For a performance-affecting change, also record what remains unproven until a
matched live run. Unit tests establish controller correctness; they do not
establish that a model is faster.
