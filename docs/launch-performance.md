# Benchmarks

ggrun's default placement vs raw llama.cpp `--fit` (its built-in auto-placement) on the
same GGUFs, measured with `scripts/bench-v3-comparison.sh`. These are ggrun's defaults —
no `--ai-tune` (see the note at the end).

Hardware: RTX 3090 Ti 24GB, RTX 3060 12GB, RTX 4070 12GB, 128GB RAM. Context: 32768,
256-token decode. CUDA backend: ik_llama.cpp build 4641 (`6c00e87a`). Numbers are from
the 2026-06-22 retest on this rig.

> **Read these as one data point, not a universal speedup.** This rig is deliberately
> awkward: the RTX 3060 sits on a PCIe gen1 x1 link. A large part of ggrun's multi-GPU
> and MoE advantage here comes from PCIe-bandwidth-weighted placement that concentrates
> work on the fast-link GPUs and routes around that x1 bottleneck — on a machine where
> every GPU has full PCIe bandwidth, the gap to raw `--fit` is smaller. The single-GPU 4B
> gain comes from the ik_llama.cpp backend and flag defaults, not PCIe.

## Dense models (CUDA / ik_llama.cpp)

| Model | Ollama 0.30.8 | raw llama.cpp `--fit` | ggrun |
|---|---:|---:|---:|
| Qwen3.5-4B Q4_K_M | 124.8 | 103.3 | 151.4 |
| Qwen3.6-27B Q5_K_M | 22.8 | 24.3 | 37.4 |

ggrun's default placement beats raw `--fit` on both (+47% on the 4B, +54% on the 27B) on
this rig.

## MoE (CUDA / ik_llama.cpp)

Both MoE models use 3-GPU expert offload across VRAM + system RAM, with a
PCIe-bandwidth-weighted tensor-split that concentrates layer ownership on the
fastest-link GPU.

| Model | ggrun decode tok/s | ggrun prefill tok/s | Notes |
|---|---:|---:|---|
| Qwen3.5-122B-A10B UD-IQ4_XS (~60 GiB) | 22.9 | 19.5 | 3-GPU expert offload + CPU experts |
| MiniMax-M3 UD-IQ3_XXS (~149 GiB) | 5.59 | 15.3 | spans VRAM+RAM, ~108 GiB pinned |

Raw llama.cpp `--fit` loads the 122B at 20.97 decode (ggrun: 22.9, +9% here); it cannot
load MiniMax-M3 at all (`unknown model architecture: minimax-m3` — ik_llama.cpp only).
The point of these rows is less the exact tok/s than that the models **run**: ggrun loads
big MoE models across VRAM + system RAM that don't fit in VRAM alone.

## Speculative decoding

Qwen3.6 27B Q5_K_M with ik_llama.cpp, 32k context, one slot, 512 generated tokens:

| Mode | Profile | Median generation tok/s | Draft accepted | Result |
|---|---|---:|---:|---|
| `--spec off` | structured/spec | 40.31 | 0/0 | baseline |
| `--spec auto` | structured/spec | 278.06 | 2367/2367 | faster (best case) |
| `--spec off` | code | 38.42 | 0/0 | baseline |
| `--spec auto` | code | 22.71 | 1161/3208 | slower |

Speculative decoding is a measured, workload-gated feature: the 278 tok/s row is one
best-case structured prompt with full draft acceptance, and the same setting is slower on
code output. It should not be advertised as always faster.

## A note on AI Tune

The tables above are ggrun's **default** placement — no tuning. `--ai-tune` benchmarks a
few flag variants and caches the fastest, but on this rig the gains are marginal and
model-specific (the default often wins the confirmation re-measure), so it's an optional
extra, not part of the headline numbers.

## Launch Read

- ggrun's default placement beats raw llama.cpp `--fit` on every dense model tested here.
- ggrun runs big MoE models (Qwen3.5-122B-A10B, MiniMax-M3) across VRAM + RAM that raw
  `--fit` can't fit or can't load at all.
- Placement is computed from the GGUF and real VRAM, not guessed — no manual flags.
