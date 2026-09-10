# Architecture

ggrun is a Go launcher around upstream GGUF serving backends. The product
surface is the `ggrun` command; helper scripts exist only for packaging,
GGUF metadata extraction, and downloading.

## Main Components

- `go/cmd/ggrun`: CLI entrypoint and compatibility argument parsing.
- `go/pkg/config`: config loading, environment handling, and app-home support.
- `go/pkg/detect`: CPU, RAM, CUDA, and Vulkan detection.
- `go/pkg/gguf`: GGUF metadata parsing through the bundled parser helper.
- `go/pkg/placement`: backend flag selection, KV planning, MoE placement,
  speculative decoding, batch-pair candidates, and vision projector lookup.
- `go/pkg/benchmark`: serial throughput/correctness probes plus the parallel
  agent benchmark (budget-scaled cold prefill, cached append, aggregate decode,
  and mixed foreground progress).
- `go/pkg/tune`: AI Tune benchmarking, candidate validation, and cache handling.
- `go/pkg/update`: self-update, backend update, rollback, and startup update
  checks for interactive users.
- `go/pkg/tui`: terminal UI launched by running `ggrun` with no arguments (or `ggrun gui`).

## Tool Layout

Runtime helper scripts live under `tools/`:

```text
tools/gguf/parse_gguf.py
tools/download/download_any_gguf.py
tools/models/model_index.py
```

Installers copy these helpers beside the installed Go binary so release bundles
and source installs can run without relying on repo-relative paths.

## Legacy Bash

The old Bash launcher is not the current product line. Existing installs can be
preserved as `llm-server-bash` during migration, and benchmark scripts can accept
an external Bash v2 path for before/after numbers. The repository root remains
Go-first.

## Backend Contract

ggrun selects and composes registered backend builds, selects flags, starts the
backend, validates health, runs benchmarks, and records cache metadata. Optional
source patches belong to explicit backend recipes with recorded provenance;
composing a feature does not establish model support or performance. Unknown
launcher flags are forwarded to the backend so upstream options remain usable.

`n_ubatch <= n_batch` is a launcher invariant. Placement normalizes restored
state before memory accounting, the Claude scheduler normalizes its conservative
baseline, and tune overlays normalize again before emission. This avoids
llama.cpp silently clamping a command whose displayed configuration and
effective allocation disagree.

Legacy AI Tune and the core placement optimizer are intentionally separate.
Tune explores backend performance flags; for a parallel-agent workload it ranks
them by repeated cache-backed agent turn time. AI Tune always protects batch
and ubatch, including cached and community overlays. Only the core optimizer
may change those coupled inputs, because it recomputes the complete
memory/expert placement. The ordinary `auto` controller calculates all bounded
neighbors, measures a short baseline prefill to bound one identical prompt
geometry, then runs repeated cold-ingest-plus-cache-append scenarios for the
stable baseline and one finalist. Explicit `--calibrate on` is the wider,
screening-only maintenance sweep. Exact allocation evidence never transfers to
a changed argv; a denser recompute is contained again, while a lateral MoE split
cannot displace an already-proven placement. Promotion still requires the
shared branch/replay, functional/gateway, lifecycle, and clean-relaunch gates.
Generic, agent-screen, and automatically promoted artifacts are separately
scoped by model, exact backend build, hardware, context, slots, effective batch
pair, explicit-intent bits, and a versioned workload profile.

A complete guarded allocator/cgroup peak has a second, stricter placement
identity covering tensor/expert ownership, batch and ubatch, KV/SWA policy,
slots, mmap/mlock, checkpoints/CRAM, graphs, projector, and speculation. This
allows a backend's exact aggregate to remain authoritative when it cannot label
individual model buffers, while preventing that aggregate from becoming proof
for a different argv shape. Sparse post-launch KV measurements merge into a
matching complete ledger instead of replacing it. GGUF tensor accounting has
its own parser schema: optional host-only tensors such as qwen4exp PLE report
presence separately from bytes, so a valid no-PLE file is distinguishable from
a stale parser.

Resident optimization has two distinct controller paths. A roomy launch has
exactly measured residual headroom and actively searches a faster resident
configuration—first avoiding unnecessary multi-GPU overhead, then spending
graph headroom on batch/ubatch and workload-appropriate automatic slots. A
tight launch stays anchored to its allocation-proven placement and admits only
bounded monotonic experiments. These are launch classifications, not model-size
classes: context, KV, companions, available devices, and current claims can move
the same GGUF between them.

For a roomy sparse MoE, resident bytes and compute ownership are deliberately
separate signals. The frontier may fully recompute each feasible GPU as the sole
ordinary-layer/KV/output owner while the other cards remain complete-expert
storage. This is a performance-only challenger and never replaces the stable
fit baseline by policy. Sustained SM imbalance between ordinary-layer owners can
help select a predicted-faster topology as the single live finalist; an idle
expert-storage card is expected sparse behavior, not sufficient evidence of a
defect. The finalist still passes exact allocation admission and the same
agent-workload/lifecycle gates before promotion. A recovered argv is only proof
of a safe allocation; it is not evidence that a roomy host is tight.

## Development lanes

The active engineering sequence is the
[ggrun core development roadmap](development-roadmap.md). It covers the
GGUF/llama.cpp product and its next release independently.

The accepted next core milestone is the
[standard-launch optimizer](core-standard-launch-todos.md). It finishes KV fit
and makes the shared TUI/direct-CLI path converge from a safe estimate to the
fastest validated configuration for real agent work. Ordinary mmap remains the
final generic fit fallback. Selective per-expert mmap/mlock pinning is a later,
hardware-gated last resort for supported MoEs beyond usable RAM plus VRAM; it
is not the backend's global `--mlock` mode.

The optimizer's phase model, llama.cpp control semantics, data schema, current
evidence, known gaps, and resume checklist live in the durable
[optimizer theory and handoff](optimizer-theory.md).

The evidence-driven [Workflow capacity planner](workflow-capacity-plan.md)
follows that single-backend core milestone. It evaluates primary slots,
same-model replicas, and smaller worker pools only on devices the user
explicitly grants, while reserving the primary model for foreground work.

The optional FreeToken experiment is recorded in a
[separate parked lane](freetoken-development-lane.md). Work in that lane cannot
change or block the core planner, installer, test matrix, or release gate.

The [specialist model fabric](specialist-model-fabric-plan.md) is retained only
as a parked exploratory idea. It is not an accepted architecture or an active
roadmap item and cannot change the current core release sequence.

## Adding features safely

Keep new product behavior in its owning package; the command layer translates
user intent and coordinates it. Shared file replacement lives in
`go/internal/atomicfile`; configuration uses it to preserve the previous file
until encoding, sync and close succeed. It does not serialize read-modify-write
transactions or promise filesystem-independent power-loss durability.

Daemon HTTP document limits and JSON errors live in `go/pkg/daemon/http.go`.
Decode a complete bounded request before taking the lifecycle lock. Feature
handlers must validate before planning, stopping a process, or persisting state.
Keep model startup timeouts separate from short HTTP request-read timeouts.

Protected placement/admission behavior remains governed by the
[core change contract](core-engine-change-contract.md). The current work sequence
and release evidence requirements are in [production readiness](production-readiness.md).
