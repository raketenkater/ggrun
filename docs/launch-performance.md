# Launch Performance Tables

These are curated local measurements for release posts. Raw run directories are
not committed; benchmark scripts write new artifacts under `.benchmarks/`.

Hardware: RTX 3090 Ti 24GB, RTX 3060 12GB, RTX 4070 12GB. Context: 32768.
CUDA/IK backend: ik_llama.cpp build 4485. Vulkan backend: llama.cpp Vulkan build
b9123/927dada6c.

Dense comparison runs used the `long` prompt profile with 256 generated tokens.

## Dense Models

| Model | Backend | Raw llama-server tok/s | llm-server v3 tok/s | Result |
|---|---|---:|---:|---|
| Qwen3.5 4B Q4_K_M | IK/CUDA | 11.89 | 159.66 | 13.4x faster launch config |
| Qwen3.5 4B Q4_K_M | Vulkan | 47.04 | 155.21 | 3.3x faster; Vulkan device path fixed |
| Qwen3.6 27B Q5_K_M | IK/CUDA | 1.86 | 38.11 | 20.5x faster launch config |
| Qwen3.6 27B Q5_K_M | Vulkan | 17.33 | 37.54 | 2.2x faster; Vulkan device path fixed |

Historical Bash v2 comparison, when measured:

| Model | Backend | Bash v2 tok/s | llm-server v3 tok/s | Result |
|---|---|---:|---:|---|
| Qwen3.5 4B Q4_K_M | IK/CUDA | 156.35 | 159.66 | parity, slightly ahead |
| Qwen3.5 4B Q4_K_M | Vulkan | failed | 155.21 | v3 fixes Vulkan launch |
| Qwen3.6 27B Q5_K_M | IK/CUDA | 38.02 | 38.11 | parity |
| Qwen3.6 27B Q5_K_M | Vulkan | failed | 37.54 | v3 fixes Vulkan launch |

## AI Tune

On these dense local runs, AI Tune did not find a stable improvement over the
v3 baseline. That is a good release message: v3's default placement is already
strong, and AI Tune refuses noise-level wins.

| Model | Backend | Baseline tok/s | Best candidate | Result |
|---|---|---:|---|---|
| Qwen3.5 4B Q4_K_M | IK/CUDA | 153.19 | `--parallel 2` at 154.30 tok/s | +0.73%, below threshold |
| Qwen3.5 4B Q4_K_M | Vulkan | 158.18 | `-b 16384` at 155.22 tok/s | baseline wins |
| Qwen3.6 27B Q5_K_M | IK/CUDA | 37.86 | `-b 4096` at 37.69/37.46 tok/s | baseline wins |
| Qwen3.6 27B Q5_K_M | Vulkan | 37.68 | candidates at 37.60/37.63 tok/s | baseline wins |

## MoE

MiniMax-M2.7 UD-Q3_K_XL is a 94.9 GiB MoE model. The measured IK/CUDA path used
3-GPU expert offload, CPU experts, and about 56.6 GiB pinned host memory.

| Model | Backend | Decode tok/s | Prompt tok/s | Notes |
|---|---|---:|---:|---|
| MiniMax-M2.7 UD-Q3_K_XL | IK/CUDA | 11.38 | 26.62 | 3-GPU MoE placement, CPU expert fallback |
| MiniMax-M2.7 UD-Q3_K_XL | Vulkan | 2.44 | 7.98 | Works, but not competitive on this hardware |

Fresh MoE AI Tune sweep at `--ctx-size 32768`:

| Candidate | Flags | Decode tok/s | Prompt tok/s | Result |
|---|---|---:|---:|---|
| baseline | current IK MoE placement | 11.40 | 26.49 | best |
| larger-batch | `-b 4096` | 11.34 | 26.57 | slower |
| smaller-moe-batch | `-b 1024` | 11.37 | 26.18 | slower |
| larger-ubatch | `-ub 1024` | n/a | n/a | CUDA OOM/hung backend |

## Speculative Decoding

Qwen3.6 27B Q5_K_M with ik_llama.cpp, 32k context, one slot, and 512 generated
tokens per request:

| Mode | Profile | Median generation tok/s | Draft accepted | Result |
|---|---|---:|---:|---|
| `--spec off` | structured/spec | 40.31 | 0/0 | baseline |
| `--spec auto` | structured/spec | 278.06 | 2367/2367 | major win |
| `--spec off` | code | 38.42 | 0/0 | baseline |
| `--spec auto` | code | 22.71 | 1161/3208 | slower |

Speculative decoding is therefore a measured, workload-gated feature. It should
not be advertised as always faster.

## Launch Read

The strongest v3 performance claim is not that AI Tune magically improves every
model. The strongest claim is that v3 makes the GGUF server launch path typed,
reproducible, cross-platform, and backend-aware:

- v3 matches Bash v2 on CUDA/IK.
- v3 fixes Vulkan launch paths where Bash v2 failed.
- v3 provides measured MoE placement instead of fragile manual flags.
- AI Tune is conservative and rejects noise, unsafe memory expansion, and
  crashing MoE candidates.
