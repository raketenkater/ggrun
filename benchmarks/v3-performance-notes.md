# v3 Go-core benchmark notes

Hardware: RTX 3090 Ti 24GB + RTX 3060 12GB + RTX 4070 12GB. Context: 32768. CUDA/IK backend: `/home/mik/ik_llama.cpp/build/bin/llama-server` build 4485. Vulkan backend: `/home/mik/llama.cpp/build-vulkan/bin/llama-server` build b9123/927dada6c.

## Qwen3.5 4B Q4_K_M

CUDA/IK baseline 3-round harness: `benchmarks/qwen35-4b-ik-baseline-r3/summary.md`

| Target | Decode tok/s | Notes |
|---|---:|---|
| raw avg | 11.80 | Minimal llama-server launch |
| Bash avg | 156.83 | Existing launcher |
| Go avg | 157.20 | go-core placement parity |

AI-tune IK cache: `/home/mik/.cache/llm-server/tune_Qwen3.5-4B-Q4_K_M.gguf_2740937888_hw5a96d3b6_ik.json`

| Candidate | Decode tok/s | Result |
|---|---:|---|
| baseline | 153.19 | kept |
| `--parallel 2` | 154.30 | +0.73%, below 1% noise floor |
| `--threads 16` | 153.88 | below threshold |

Vulkan one-round harness: `benchmarks/qwen35-4b-vulkan-baseline-r1-fixed/summary.md`; fresh current-binary run: `benchmarks/live-20260528-4b-vulkan-baseline/summary.md`.

| Target | Decode tok/s | Notes |
|---|---:|---|
| raw | 54.20 | Direct Vulkan llama-server in fresh live run |
| Bash | failed | passes `CUDA0`, invalid for Vulkan |
| Go | 156.68 | emits `Vulkan0`, works |

AI-tune Vulkan cache: `/home/mik/.cache/llm-server/tune_Qwen3.5-4B-Q4_K_M.gguf_2740937888_hw5a96d3b6_vulkan.json`. Fresh 2-round retune: baseline 158.18 tok/s; `-b 16384` measured 155.22; `--threads 16` measured 157.29. Baseline wins and the cached config now auto-applies on future Go launches. A post-cache benchmark measured 157.33 tok/s with no override flags.

## Qwen3.6 27B Q5_K_M

CUDA/IK fixed one-round harness: `benchmarks/qwen36-27b-ik-baseline-r1-fixed/summary.md`

| Target | Decode tok/s | Notes |
|---|---:|---|
| raw | 1.86 | Raw path with jinja + ctx, no placement/tuned flags |
| Bash | 37.90 | Existing launcher |
| Go | 37.90 | go-core parity |

AI-tune IK cache: `/home/mik/.cache/llm-server/tune_Qwen3.6-27B-Q5_K_M.gguf_19509790944_hw5a96d3b6_ik.json`. Baseline was 37.86 tok/s; `-b 4096` candidates measured 37.69 and 37.46, so baseline wins.

Vulkan one-round harness: `benchmarks/qwen36-27b-vulkan-baseline-r1/summary.md`

| Target | Decode tok/s | Notes |
|---|---:|---|
| raw | 20.25 | Direct Vulkan llama-server |
| Bash | failed | passes `CUDA0`, invalid for Vulkan |
| Go | 37.66 | emits `Vulkan0`, works |

AI-tune Vulkan cache: `/home/mik/.cache/llm-server/tune_Qwen3.6-27B-Q5_K_M.gguf_19509790944_hw5a96d3b6_vulkan.json`. Baseline was 37.68 tok/s; candidates measured 37.60 and 37.63, so baseline wins. Fresh current-binary post-cache benchmark measured 37.70 tok/s and auto-selected the cache with no override flags.

## MiniMax-M2.7 UD-Q3_K_XL MoE

Go CUDA/IK baseline: `benchmarks/minimax-m2.7-ud-q3-ik-go-r1/summary.md`

| Target | Decode tok/s | Prompt tok/s | Notes |
|---|---:|---:|---|
| Go IK | 11.38 | 26.62 | 3-GPU expert offload, CPU experts, 56.6 GiB pinned host memory |

Go Vulkan baseline: `benchmarks/minimax-m2.7-ud-q3-vulkan-go-r1/summary.md`

| Target | Decode tok/s | Prompt tok/s | Notes |
|---|---:|---:|---|
| Go Vulkan | 2.44 | 7.98 | Works, but not competitive for this MoE on this box |

AI-tune IK one-round: `benchmarks/minimax-m2.7-ud-q3-ik-ai-tune-r1/summary.md`. Baseline was 11.40 tok/s. The first fallback candidate tried q8 KV and crashed with CUDA OOM, so baseline wins. The fallback logic now skips q8 KV upgrades for MoE/offload layouts and will test batch/ubatch first. Current Go dry-run auto-selects the total-shard-size IK tune cache (`101939873056` bytes), not the stale first-shard artifact.

## Serving throughput

Short 4B Vulkan throughput runs show why v3 needs a serving benchmark in addition to single-request decode:

| Mode | Total ctx | Per-slot ctx | Parallel | Concurrency | Completion tok/s | Avg latency |
|---|---:|---:|---:|---:|---:|---:|
| fixed total ctx | 32768 | 32768 | 1 | 4 | 131.29 | 2.42s |
| fixed total ctx | 32768 | 8192 | 4 | 4 | 250.79 | 1.53s |
| fixed total ctx | 32768 | 4096 | 8 | 8 | 147.52 | 5.20s |
| fixed per-slot ctx | 131072 | 32768 | 4 | 4 | 267.61 | 1.43s |

Important correction: llama.cpp divides `--ctx-size` across `--parallel` slots. The first `parallel=4`/`parallel=8` runs were fixed-total-context tests, not fixed-per-request-context tests. The harness now supports `--ctx-per-slot` to keep per-slot context constant. On this 4B Vulkan workload, `parallel=4` is still the current sweet spot with fair 32768-token slots.

## AI-tune correctness

AI-tune is operating correctly after the current fixes: it measures the baseline, launches separate candidate servers, persists public `tune_*.json` files, applies only safe override flags, auto-selects matching cached configs on normal Go launch/dry-run, and uses a 1% improvement floor so noise-level wins are not shipped. The internal cache is now scoped by model, hardware, backend, and vision mode, and split GGUF tune filenames now use total shard size instead of only the first shard.

## Read

The big v3 story is not that AI-tune magically boosts dense single-request decode. On 4B and 27B, the placement defaults are already strong, and tuning mostly finds noise or regressions. The real v3 performance gain is Go making the launch path typed, reproducible, cross-platform, and backend-aware: Go matches Bash on CUDA/IK, fixes Vulkan device naming, and exposes reliable benchmark artifacts.

The next largest performance path is MoE and throughput tuning: sweep `--parallel`, batch/ubatch, `--n-cpu-moe`, pinned memory, mmap/no-mmap, and expert/layer placement under memory headroom constraints. Speculative decoding/MTP should be added behind backend capability detection and measured with acceptance rate plus quality checks, not enabled blindly. For a public v3 product/post, report dense CUDA parity, Vulkan enablement, MoE CUDA versus Vulkan reality, and a clear roadmap toward GGUF serving throughput against vLLM-style expectations.
