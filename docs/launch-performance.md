# Benchmarks

## Agent workload results — 2026-10-13 (v3.2.11)

What a fresh installation recommends for four hardware profiles, and what
happened when each recommendation was downloaded and launched with the
defaults and used through Claude Code. Each first launch ran three small
repository repair tasks (`calc`, `textstats`, `inventory`; correctness checked
by each repository's unit tests, never by the model); each relaunch ran one
longer task (`ledger`, 20-35 turns). Times are wall seconds per task.

- Installation: `setup-linux.sh` default mode into an empty home, from
  bundles packaged as `release.yml` packages them (ik_llama 1fddd12 + #2356
  CUDA backend built with the release flags; mainline d2462f8 Vulkan);
  recommendations from the embedded catalog of 2026-10-13. Downloads are not included in "ready"; one recommended model
  (GLM-5.3-Flash UD-IQ3_XXS, 112 GB) was downloaded through `ggrun download`
  in 3 h 6 min.
- Profiles: `--gpus N --ram-budget X` (or `--cpu`) on the same machine, with
  the backend's GPU visibility and memory cgroup checked on every launch.
  Limits do not emulate a smaller CPU, slower memory or other PCIe links.
- Machine: i9-10940X (14 cores), 212 GB RAM, RTX 4070 12 GB + RTX 3090 Ti
  24 GB + RTX 3060 12 GB on PCIe Gen 3 (3060 at x8), driver 580.178.04.
- Prefill is the first long prompt of the session (~9k-21k tokens); decode is
  the median over requests that generated at least 32 tokens.

Tasks: calc / textstats / inventory on first launch; ledger on relaunch.

| Profile | Category | Model (quant) | First launch: ready; tasks (s) | Relaunch: ready; long task (s) | Context | Prefill / decode tok/s |
|---|---|---|---|---|---:|---|
| Full host | Best overall | Qwen3.8-27B UD-Q5_K_S | 121 s; 116 / 82 / 98 | 14 s; 497 | 262,144 | 1,391 / 30.7 |
| Full host | Smartest | GLM-5.3 UD-IQ2_M (222 GB) | not run: the model disk failed during download | — | — | — |
| Full host | Fastest | Qwen3.6-35B-A3B UD-IQ4_XS | 315 s; 38 / 38 / 43 | 13 s; 108 | 262,144 | 3,645 / 80.2 |
| 12 GB GPU + 32 GiB | Best overall, Smartest | Qwen3.8-27B UD-IQ2_XXS | 256 s; 231 / 137 / 85 | 50 s; 470 | 76,800 | 1,119 / 31.2 |
| 12 GB GPU + 32 GiB | Fastest | Gemma 4 26B A4B IQ4_XS | 225 s; 161 / 153 / 168 | 8 s; 206 | 74,752 | 265 / 30.0 |
| 24 GB GPU + 128 GiB | Best overall | Qwen3.8-27B UD-Q5_K_S | 175 s; 64 / 79 / 76 | 6 s; 256 | 122,880 | 1,008 / 34.7 |
| 24 GB GPU + 128 GiB | Smartest | GLM-5.3-Flash UD-IQ3_XXS (112 GB) | backend built automatically (mainline CUDA), then 337 / 272 / 414 | 1,108 s; 992 | 628,736 | — |
| 24 GB GPU + 128 GiB | Fastest | Qwen3.6-35B-A3B UD-IQ4_XS | 106 s; 23 / 20 / 1,338 | 9 s; 99 | 262,144 | 3,185 / 92.5 |
| CPU + 16 GiB | Best overall, Fastest | Qwen3.6-35B-A3B UD-IQ2_XXS | 206 s; 323 / 147 / **timeout** (3,600) | 15 s; 586 | 108,544 | 127 / 13.1 |
| CPU + 16 GiB | Smartest | Qwen3.8-27B UD-IQ2_XXS | 468 s; 2,613 / 1,450 / 1,206 | 9 s; **timeout** (5,400) | 131,072 | 16 / 3.1 |

Notes:

- A timeout is a failure, not a pass. CPU-only Qwen3.8-27B is correct but
  takes 20-44 minutes per small task and timed out on the long task; on the
  CPU profile Qwen3.6-35B-A3B timed out once after it repeated a tool call
  for an hour (the server answered every request).
- Claude Code mode (`--claude-code`, local reviewer on the third GPU) passed
  the same tasks on Qwen3.8-27B: three tasks at once in 478 s, the long task
  in 593 s. Four slots on this model family produced corrupted output on the
  bundled ik_llama build, so Claude Code mode now uses one slot for it.
- GLM-5.3-Flash needs a backend newer than the bundled ones: its first launch
  built current mainline llama.cpp with CUDA (about 15 minutes here) before
  loading. The full-host Smartest pick (GLM-5.3 UD-IQ2_M, 222 GB) was not
  measured: the model disk failed during its download.
- Single runs per cell; expect run-to-run variation, especially for agent
  task times. No other runtime was measured in this round, so no comparison
  is claimed here.

## Historical: 2026-06-22 decode comparison

ggrun's default placement vs raw llama.cpp `--fit` (its built-in auto-placement) on the
same GGUFs, measured with `scripts/bench-v3-comparison.sh`. These are ggrun's defaults —
no `--ai-tune` (see the note at the end).

June 22, 2026 retest. Hardware then: RTX 3090 Ti 24GB, RTX 3060 12GB, RTX 4070 12GB, 128GB RAM. Context: 32768,
256-token decode. CUDA backend: ik_llama.cpp build 4641 (`6c00e87a`). Numbers are from
the 2026-06-22 retest on this rig.

> **Read these as one data point, not a universal speedup.** This rig is deliberately
> awkward: the RTX 3060 sits on a PCIe gen1 x1 link. A large part of ggrun's multi-GPU
> and MoE advantage here comes from PCIe-bandwidth-weighted placement that concentrates
> work on the fast-link GPUs and routes around that x1 bottleneck — on a machine where
> every GPU has full PCIe bandwidth, the gap to raw `--fit` is smaller. The single-GPU 4B
> gain comes from the ik_llama.cpp backend and flag defaults, not PCIe.

### Dense models (CUDA / ik_llama.cpp)

| Model | Ollama 0.30.8 (June 2026) | raw llama.cpp `--fit` | ggrun |
|---|---:|---:|---:|
| Qwen3.5-4B Q4_K_M (backend difference) | 124.8 | 103.3 | 151.4 |
| Qwen3.6-27B Q5_K_M | 22.8 | 24.3 | 37.4 |

ggrun was faster than raw `--fit` on both on this rig: +54% on the 27B, and +47% on the
4B, which is a backend difference (ik_llama.cpp vs llama.cpp on one GPU), not placement.

### MoE (CUDA / ik_llama.cpp)

The MoE models use expert offload across VRAM + system RAM. On asymmetric PCIe
topologies, dense-layer ownership stays on the fast link while cards at one-third
or less of that bandwidth become whole-expert storage.

| Model | ggrun decode tok/s | ggrun prefill tok/s | Notes |
|---|---:|---:|---|
| Qwen3.5-122B-A10B UD-IQ4_XS (~60 GiB) | 22.9 | 19.5 | 3-GPU expert offload + CPU experts |
| MiniMax-M3 UD-IQ3_XXS (~149 GiB) | 5.59 | 15.3 | spans VRAM+RAM, ~108 GiB pinned |
| DeepSeek-V4-Flash UD-IQ4_XS (~128 GiB) | 5.88 | 32.24 | ctx 1M, parallel 4; x1/x4 GPUs store 3 complete expert layers each |

Raw llama.cpp `--fit` loaded the 122B at 20.97 decode (ggrun: 22.9, +9% here). On
2026-06-22 mainline llama.cpp could not load MiniMax-M3 (`unknown model architecture:
minimax-m3`); only ik_llama.cpp ran it then.
The point of these rows is less the exact tok/s than that the models **run**: ggrun loads
big MoE models across VRAM + system RAM that don't fit in VRAM alone.

#### DeepSeek-V4 long-context service stress

The same 1M-context, parallel-4 placement completed a four-request stress run with one
60,020-token prompt and three concurrent prompts of 4,115, 4,243, and 4,371 tokens. Every
request returned 16 non-empty output tokens; wall time was 2,377.10s. The main prompt
processed at 29.05 tok/s, crossed the 8k/16k checkpoint boundaries, and completed without
an OOM, restart, truncation, or health failure. Post-run VRAM was 23,962 / 9,329 / 9,363
MiB on the 3090 Ti / 3060 / 4070. Reproduce against a running server with
`scripts/loadtest-moe.py`; its default shape is this exact 60k + 3-worker test.

### Speculative decoding

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

### A note on AI Tune

The tables above are ggrun's **default** placement — no tuning. `--ai-tune` benchmarks a
few flag variants and caches the fastest, but on this rig the gains are marginal and
model-specific (the default often wins the confirmation re-measure), so it's an optional
extra, not part of the headline numbers.

### Launch Read (2026-06-22)

- ggrun was faster than raw llama.cpp `--fit` on both dense models tested; the 4B gain
  was a backend difference (ik_llama.cpp vs llama.cpp), not placement.
- ggrun ran big MoE models (Qwen3.5-122B-A10B, MiniMax-M3) across VRAM + RAM; raw `--fit`
  loaded the 122B but could not load MiniMax-M3 then.
- Placement was computed from the GGUF and real VRAM, with no manual flags.
