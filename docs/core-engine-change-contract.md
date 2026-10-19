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
   An arch-proven sparse lookup table that the exact backend reads lazily
   (DeepSeek-V4.1 Engram) is a separate file-backed tier, not execution-weight
   paging: it never enters GPU or resident host ledgers, gets an explicit
   bounded lookup-cache allowance, must be confirmed by live mapping
   observation, and never authorizes paging execution weights.
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
