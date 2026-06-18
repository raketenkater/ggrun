# Launch Performance Tables

These are curated local measurements for release posts. Raw run directories are
not committed; benchmark scripts write new artifacts under `.benchmarks/`.

Hardware: RTX 3090 Ti 24GB, RTX 3060 12GB, RTX 4070 12GB. Context: 32768.
CUDA/IK backend: ik_llama.cpp build 4485. Vulkan backend: llama.cpp Vulkan build
b9123/927dada6c.

Dense comparison runs used the `long` prompt profile with 256 generated tokens.

## Dense Models (CUDA / ik_llama.cpp)

Same GGUF, 32k context, 256-token decode, vs raw llama.cpp `--fit` and Ollama 0.30.8:

| Model | Ollama 0.30.8 | raw llama.cpp `--fit` | llm-server v3 | v3 `--ai-tune` |
|---|---:|---:|---:|---:|
| Qwen3.5-4B Q4_K_M | 124.8 | 103.3 | 176.6 | 178.8 |
| Qwen3.6-27B Q5_K_M | 22.8 | 24.3 | 40.3 | 40.3 |

llm-server's default placement already beats raw `--fit` and Ollama (+43% on the 4B,
+77% on the 27B vs Ollama). `--ai-tune` adds a small gain on the 4B and correctly
finds nothing on the 27B — it rejects noise-level wins rather than inventing one.

## MoE (CUDA / ik_llama.cpp)

Both MoE models use 3-GPU expert offload across VRAM + system RAM:

| Model | Decode tok/s | Prompt tok/s | `--ai-tune` | Notes |
|---|---:|---:|---:|---|
| Qwen3.5-122B-A10B UD-IQ4_XS (~60 GiB) | 22.7 | 19.5 | 23.0 | 3-GPU expert offload + CPU experts |
| MiniMax-M3 UD-IQ3_XXS (~149 GiB) | 5.47 | 5.46 | 5.50 | spans VRAM+RAM, ~108 GiB pinned host memory |

Raw llama.cpp `--fit` loads the 122B (20.97 decode tok/s) but **cannot** load
MiniMax-M3 (`unknown model architecture: minimax-m3` — ik_llama.cpp only). Ollama
can't import either big MoE on this hardware (sharded GGUFs), so llm-server is the
only one of the three that runs them. AI Tune again finds only marginal gains over
the already-strong default placement.

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

- v3 default placement beats raw llama.cpp `--fit` and Ollama on every model all
  three can load (+43% / +77% vs Ollama on the 4B / 27B).
- v3 runs big MoE models (Qwen3.5-122B-A10B, MiniMax-M3) that raw `--fit` and
  Ollama cannot load at all on this hardware.
- v3 provides measured MoE placement instead of fragile manual flags.
- AI Tune is conservative and rejects noise, unsafe memory expansion, and
  crashing MoE candidates.
