# Hardening and remaining integration review — 2026-09-13

Reviewed main: `93f60bdb6fb8b9aceb60f6b5a0f34929ce02a5cc`.

## Linux GPU and restart evidence

[Manual install E2E 34753549254](https://github.com/raketenkater/ggrun/actions/runs/34753549254)
passed Linux CPU, Windows CPU and Linux GPU on `pentestserver-gpu`. Windows GPU
was skipped. This run tests the merged main, rather than relying on older PR CI.

The immediate same-backend restart failure was a checker defect: ik's
cpp-httplib enables SO_REUSEPORT on Linux, whereas the checker tried to bind with
SO_REUSEADDR. The incompatible probe refused TIME_WAIT sockets even though the
backend could reuse them. The checker now refuses an active listener using a
bounded connection probe, fails closed on unexpected socket errors, and leaves
actual binding to the launched backend. Readiness, generation, streaming and
clean shutdown remain required. It never enables SO_REUSEPORT to share a live
listener. Real same-port initial launch and immediate relaunch passed on main
with the published v3.2.9 CUDA backend (`/tmp/ggrun-main-restart-socket-proof`).
The earlier cross-backend switch failure is separate; this does not prove all
backends can inherit each other's TIME_WAIT sockets.

Linux GPU CI now repeats serving on the same port and retains both results.

## PR 50 and issue 49

The issue correctly identifies a real capacity-detection problem. The proposed
head `5e708aa062a8be4a91aef4f4de8aced4cebfb503` passes its existing uncached core
gate, but is not ready to merge:

1. **Backend-independent fabricated capacity.** `parseVulkanGPUs` grants every
   named AMD integrated GPU 80% of total RAM without knowing the selected
   backend, current free memory, driver allocation limit or actual heap budget.
   The issue's 89,889,816,576-byte Vulkan heap is 85,725 MiB, below the proposed
   102,460 MiB capacity. Registering a HIP backend does not constrain this code
   to HIP launches. `VRAMUsedMB` remains zero.
2. **Shared-pool accounting is absent.** Dense CPU offload sums GPU free memory
   and host free RAM in `placement.go`. On an APU these can be the same physical
   pages. Synthesizing the GPU budget does not make them independent pools.
3. **Unrelated path names change device routing.** `detectBackend` searches
   the entire directory for the substring `hip`. A plain llama-server in a
   directory named `chip-project` is falsely classified as ROCm. A local
   regression test calling the real detection function reproduced the failure.
   The PR has no routing tests covering this case or actual device discovery.

Keep PR 50 open. Separate backend device dialect discovery from shared-memory
capacity; derive names from backend/device evidence, preserve an honest Vulkan
heap ceiling, and account for the shared host/GPU pool once. Add busy-memory,
missing-probe, multiple-device and path-name regression cases. AMD hardware
acceptance is still needed; this NVIDIA host cannot supply it.

## Remaining hot-experts branch

Reviewed branch head: `3a466b9b83f1cc3ca23022f1c00ea0d07e5fb5bf`.
This is a divergent integration tree, not a small feature patch: its tree differs
from main in 145 files, including older CI, installer and catalog snapshots.
Do not merge those snapshots over the newly validated release base. Root's
separate staged work is also preserved and must not be confused with this head.

Reconcile in these boundaries:

- Keep current main's CI, installers, docs, matched-workload runner and finite
  score checks. The old branch still extrapolates workload time by waves in
  `calibrate.go`; its schema 35 does not make that measurement comparable.
  Integrating the main measurement semantics needs a new evidence schema.
- Retain reviewed backend composition, pinned overlay provenance, bounded
  telemetry and in-flight deduplication as a separately reviewed capability
  layer. A backend's name or cache flag does not prove model graph engagement.
- Resolve reserve accounting before enabling automatic cache spending.
  `hotExpertCacheMaxSlots` explicitly spends SlackMB without checking
  RuntimeMeasured and records a warmup OOM from that path. Exact model-buffer
  allocation is not proof of runtime-growth headroom.
- Preserve the corrected promotion gate in the branch: it now requires workflow
  improvement and additional decode improvement when cache slots increase.
  The early cache-on success bypass reported against an older staged snapshot
  is not present in this committed function. Main's nonfinite-score rejection
  and identical-work checks must also survive integration.
- The overlay graph guard is still single-token, separate gate/up/down, SiLU,
  no pre-FFN weighting and no LoRA. Four server slots do not establish cache
  engagement during multi-token continuous batches. Validate exact supported
  layouts, cache-off identity and phase/correctness regressions before promotion.

No hot-expert policy is promoted by this hardening patch. Review and bounded
fixture correctness are not substitutes for cache-on/off live performance proof.

## Current live run (observational snapshot)

A separate user run occupies this machine: Qwen3.8-Flash-Next Q3_K_XL, total
context 262144, four slots (65536 each), plus a 2B reviewer. The main backend is
hot-experts-capable but its actual argv has no --moe-expert-cache; source defaults
the slot count to zero. It is serving cache-off with 24 CPU-expert layers.

At the snapshot all four slots were idle and health was OK. Aggregate GPU memory
use was approximately 10.0/21.2/10.9 GiB, including the reviewer. The router log
contained 185 requests, 176 HTTP 200 and nine HTTP 400, with successful-request
median queue 2 ms and p95 3 ms. Backend errors show prompts exceeding the 65536
slot context, often by only 50–51 tokens. Fix rendered-prompt boundary accounting
before interpreting these failures as insufficient concurrency. These are
lifetime observations, not a controlled benchmark.

A separate 27B comparison attempt was refused during contained preflight because
this live workload had occupied the GPUs. Its failure is retained, and the live
process was not stopped or reconfigured. The bounded repair client can exercise
the existing endpoint without launching another model; shared-run measurements
cannot establish an isolated configuration speedup.

The direct-backend repair run completed 9/9 oracle-verified tasks in 85.48 s
(two client lanes, three repetitions plus one excluded warmup), or 6.32 correct
fixture repairs/minute. Median task latency was 17.82 s and maximum 25.70 s.
All tasks used read, write and post-edit test calls. Evidence is retained at
`/tmp/ggrun-live-agent-repair-2lanes`. This exercised the existing four-slot
cache-off server; it did not go through the Claude router/reviewer and does not
prove that cache-on would improve it. A matched isolated server-configuration
A/B remains pending an idle hardware window.
