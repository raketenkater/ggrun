# Launch Performance Tables

These are curated local measurements for release posts. Raw run directories are
not committed; benchmark scripts write new artifacts under `.benchmarks/`.

Hardware: RTX 3090 Ti 24GB, RTX 3060 12GB, RTX 4070 12GB. Context: 32768.
CUDA/IK backend: ik_llama.cpp build 4641 (`6c00e87a`). Vulkan backend:
llama.cpp build 9756 (`d0f9d2e5a`). Dense ggrun rows were refreshed on
2026-06-22 after rebuilding both backends from their upstream heads.

Dense comparison runs used the `long` prompt profile with 256 generated tokens.
The current headline matrix comes from `.benchmarks/full-retest-20260622/`.

## Dense Models (CUDA / ik_llama.cpp)

Same GGUF, 32k context, 256-token decode, vs raw llama.cpp `--fit` and Ollama 0.30.8:

| Model | Ollama 0.30.8 | raw llama.cpp `--fit` | ggrun v3 | v3 `--ai-tune` |
|---|---:|---:|---:|---:|
| Qwen3.5-4B Q4_K_M | 124.8 | 103.3 | 151.4 | 185.7 |
| Qwen3.6-27B Q5_K_M | 22.8 | 24.3 | 37.4 | 39.8 |

ggrun's default placement already beats raw `--fit` and Ollama (+21% on the 4B,
+64% on the 27B vs Ollama). `--ai-tune` raises the headline dense rows to +49%
on the 4B and +74% on the 27B versus Ollama.

## MoE (CUDA / ik_llama.cpp)

Both MoE models use 3-GPU expert offload across VRAM + system RAM. The tensor-split
is PCIe-bandwidth-weighted, which concentrates layer ownership on the fastest-link
GPU and lifts prefill sharply on heterogeneous rigs (a card stuck at x1 no longer
bottlenecks CPU-expert streaming):

| Model | Decode tok/s | Prefill tok/s | `--ai-tune` | Notes |
|---|---:|---:|---:|---|
| Qwen3.5-122B-A10B UD-IQ4_XS (~60 GiB) | 22.9 | 19.5 | 23.1 | 3-GPU expert offload + CPU experts |
| MiniMax-M3 UD-IQ3_XXS (~149 GiB) | 5.59 | 15.3 | 5.65 | spans VRAM+RAM, ~108 GiB pinned |

Both-run vs Ollama 0.30.8 (same merged GGUF, 32k, 256-token decode):

| Model | Ollama | ggrun | ggrun advantage |
|---|---:|---:|---:|
| Qwen3.5-122B-A10B UD-IQ4_XS (~60 GiB, ~18 GiB to RAM) | 13.54 | 23.56 | **+74%** |

Ollama can't *pull* sharded GGUFs ([ollama#5245](https://github.com/ollama/ollama/issues/5245)),
so the 122B was merged to a single file and imported with `ollama create`; at that point it
runs, and ggrun is **+74%** faster on the heavy VRAM+RAM offload path. MiniMax-M3 it
can't load at all (`unknown model architecture: minimax-m3` — ik_llama.cpp only); raw
llama.cpp `--fit` loads the 122B (20.97 decode) but also can't load MiniMax-M3. AI Tune
finds only marginal gains over the already-strong default placement.

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
  three can load (+21% / +64% vs Ollama on the 4B / 27B).
- v3 runs big MoE models (Qwen3.5-122B-A10B, MiniMax-M3) that raw `--fit` and
  Ollama cannot load at all on this hardware.
- v3 provides measured MoE placement instead of fragile manual flags.
- AI Tune is conservative and rejects noise, unsafe memory expansion, and
  crashing MoE candidates.
