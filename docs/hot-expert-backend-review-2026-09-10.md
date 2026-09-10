# Hot-expert backend review — 2026-09-10

There is no evidence that ggrun's current hot-expert backend is the best choice
for completed agent workflows. Keep it as a bounded implementation for supported
layouts, while treating backend choice as part of the complete serving plan.
Do not switch production backends on published decode numbers alone.

## What we actually ship

`go/pkg/backends/backends.go` composes the selected model backend with
`features/hot-experts/bccbacdb8945`, telemetry and in-flight upload deduplication.
The bundled patch's graph guard requires one token, separate gate/up/down,
SILU, no pre-FFN weighting and no LoRA. Composition does not expand that coverage.
A CLI flag advertising a cache therefore does not prove that a particular model
or continuously batched request will execute it.

The source is [llama.cpp PR 27861](https://github.com/ggml-org/llama.cpp/pull/27861),
still a draft at review time. Its design computes cache hits on the GPU and
misses on the CPU, with throttled asynchronous uploads. The author reports
18.4 → 24.2 decode tokens/s for one Qwen configuration. That is evidence for a
specific mechanism, not concurrent agent throughput or general model coverage.
Its single-token restriction makes batching and speculative verification a
particularly important gap for ggrun's objective.

## Alternatives and decisions

| Candidate | Primary evidence | Engineering assessment for ggrun |
| --- | --- | --- |
| Current hybrid LRU overlay | Above PR and checked-in patch | Retain the integration and harden admission. Useful reference; insufficient as a universal cache implementation. |
| Persistent expert slot pool | [RFC 28248](https://github.com/ggml-org/llama.cpp/discussions/28248) describes a single GPU execution chain, uploading misses before use. It covers fused and separate tensors, and small multi-token batches when the working set fits. The author says commit `78598a8` disables the cache with multiple GPUs. | Most relevant alternative cache design to inspect next. Its synchronous misses trade CPU work for PCIe transfers. It needs device-local ownership, synchronization and multi-GPU validation before replacing our current path. |
| IK llama.cpp | [Project README](https://github.com/ikawrakow/ik_llama.cpp) documents the fork and warns about incoherent output for some graph-split configurations with partial offload. | A useful whole-backend challenger because ggrun already has an IK dialect. Neither compatibility nor a hot-cache advantage can be assumed. Start with a supported topology and preserve quality. |
| KTransformers / KT-Kernel with SGLang | [Official KT-Kernel documentation](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md) documents CPU/GPU heterogeneous serving, AMX/AVX kernels, NUMA configuration and GPU expert placement. Its llamafile path accepts CPU GGUF weights; the documented example also needs original GPU weights. | Strategic candidate if CPU expert execution remains the dominant cost. This requires a serving adapter and model/weight support checks, not swapping a llama-server binary. AMX results cannot be assumed on an AVX2 host. |
| Broader heatmap/transfer patch | [PR 26824](https://github.com/ggml-org/llama.cpp/pull/26824), successor to 26563, was closed on August 10. | Useful prior art. Closure does not disprove its performance, but the larger unmerged change is not currently a lower-risk default foundation. |

These assessments are engineering inferences from source and documentation,
not a new benchmark result. This review does not establish a universal winner.

## Next engineering steps

1. Complete reserve accounting across reviewed overlay provenance. Parent
   measurements may add a reserve floor; they must never supply overlay fit
   proof or erase a larger local observation. Missing reserve evidence remains
   unknown. The remaining discretionary-spend gate in
   `hotExpertCacheMaxSlots` is a separate open issue: importing a floor does not
   make launches with no usable growth evidence safe.
2. Represent cache coverage explicitly: tensor layout, activation, batch width,
   speculative path and device ownership. Use this in automatic eligibility;
   unsupported graphs must not silently look cache-enabled.
3. Review the persistent-pool implementation for fused tensors and small batches,
   especially mapping publication, eviction and per-device stream ownership.
   Keep asynchronous hybrid misses as a competing design where PCIe is costly.
4. Integrate a different backend only after its model support and serving API
   preserve tool calls, prefix reuse, context and quality. Compare complete
   configurations, including cache-off on the same build.

Later performance acceptance must use repeated identical agent workflows at the
requested concurrency, with correct completed work per time as the objective.
Keep cold prefill, cached append, decode and mixed-phase regression guards;
record expert-cache engagement and transfers as explanations. Match model,
quantization, useful context, prompt reuse and sampling. Require exact admission
and a clean relaunch. No serving benchmark or model restart was run for this
review.
