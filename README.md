# ggrun

*`ggrun` = "gguf run". Formerly **llm-server**.*

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/raketenkater/ggrun)](https://github.com/raketenkater/ggrun/releases/latest)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#requirements)
[![Backends](https://img.shields.io/badge/backends-ik__llama.cpp%20%7C%20llama.cpp-orange)](#backends)

**Stop hand-writing `--tensor-split`, `-ot`, and KV-cache flags.** Point ggrun
at a GGUF and it measures your GPUs, RAM, and PCIe topology, picks the backend
(llama.cpp or the faster ik_llama.cpp fork), computes multi-GPU and MoE expert
placement, and serves an OpenAI-compatible API — one command from file to endpoint.

```bash
ggrun recommend                           # rank the best models for YOUR hardware
ggrun unsloth/Qwen3.6-27B-GGUF --download # HF repo → hardware-matched quant → served
ggrun model.gguf                          # local GGUF → served
ggrun                                     # no args → interactive TUI
```

![demo](demo.gif)

*No flags: `ggrun` with no arguments opens a TUI that detects your GPUs, lists your models, computes hardware-matched launch settings, and ranks downloads by fit.*

**Just run `ggrun` with no arguments** to open the full arrow-key TUI — browse and
download models, adjust settings, and launch, without writing a single flag. Pass a model
path or flags for one-shot CLI use instead.

## Benchmarks

Same rig (RTX 3090 Ti 24GB + 4070 12GB + 3060 12GB, 125GB RAM), same GGUFs, 32k context,
decode tok/s (256-token generation), slowest backend on the left:

| Model (quant) | Ollama 0.30.8 | llama.cpp `--fit` | ggrun v3 | v3 `--ai-tune` | v3 vs Ollama |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B Q4_K_M | 124.8 | 103.3 | 150.4 | **154.2** | **+24%** |
| Qwen3.6-27B Q5_K_M | 22.8 | 24.3 | **37.5** | **37.5** | **+64%** |
| Qwen3.5-122B-A10B UD-IQ4_XS (MoE) | 13.5† | 21.0 | 23.6 | **23.6** | **+74%** |
| MiniMax-M3 UD-IQ3_XXS (MoE) | ✗ won't load | ✗ won't load | **5.24** | not rerun | Ollama can't load |

† Ollama can't import sharded GGUFs ([ollama#5245](https://github.com/ollama/ollama/issues/5245)),
so the 122B was merged to one file before importing; MiniMax-M3 it can't load at all
(`minimax-m3` is ik_llama-only). Where models load, ggrun is **24–74% faster than
Ollama — including +74% on the 122B MoE** at heavy VRAM+RAM offload (60 GB, ~18 GB spilled
to RAM). Driving the *same* llama.cpp master binary (no ik_llama), ggrun still beat raw
`--fit` — so the gain is the placement, not just the backend swap. Full methodology,
exact commands, and artifacts: [docs/performance.md](docs/performance.md). Numbers are
reproducible with [`scripts/bench-v3-comparison.sh`](scripts/bench-v3-comparison.sh) —
regressions against these tables are treated as bugs.

## Install

Linux / macOS — self-contained app home under `~/ggrun`:

```bash
curl -fsSL https://raw.githubusercontent.com/raketenkater/ggrun/main/setup.sh | bash
```

Windows (PowerShell); add `-Backend cuda` for native NVIDIA CUDA:

```powershell
iwr -useb https://raw.githubusercontent.com/raketenkater/ggrun/main/install.ps1 | iex
```

From a clone:

```bash
git clone https://github.com/raketenkater/ggrun.git && cd ggrun && ./setup.sh
```

Since v3, [prebuilt release bundles](https://github.com/raketenkater/ggrun/releases/latest)
(Linux CPU/Vulkan, macOS arm64 Metal, Windows x86_64 CPU) install without compiling,
verified against `SHA256SUMS`; Linux CUDA/ik_llama.cpp builds from source for your GPU.
Run `ggrun` with no arguments to open the TUI. Installer options and the app-home
layout are in [docs/install.md](docs/install.md).

## Quick start

```bash
ggrun ~/models/model.gguf                 # launch a local model
ggrun unsloth/Qwen3.6-27B-GGUF --download # download a fitting quant, then launch
ggrun model.gguf --ai-tune                # benchmark flag sets, cache the fastest
ggrun model.gguf --dry-run                # print the backend command without running
ggrun model.gguf --benchmark              # load, measure tok/s, exit
```

Common flags: `--backend ik_llama|llama|vulkan`, `--gpus 0,1`, `--ctx-size`,
`--kv-quality`, `--kv-placement`, `--vram-headroom 2G`, `--ram-headroom 8G`, `--vision`, `--spec auto`. Unknown flags pass straight
through to `llama-server`, so nothing upstream is out of reach. Full list:
[docs/usage.md](docs/usage.md).

> **Security:** ggrun serves on `0.0.0.0` (all interfaces) by default and the
> OpenAI-compatible API is **unauthenticated**, so on a shared network anyone on
> your LAN can reach the model. On untrusted networks bind to localhost with
> `--host 127.0.0.1` (or set `LLM_HOST` / the Host setting in the TUI).

## How it compares

**vs raw llama.cpp.** Upstream `--fit` auto-picks GPU layers, tensor-split, and context.
If that covers you, raw llama.cpp may be enough. ggrun goes further: it selects the
backend (ik_llama.cpp is meaningfully faster on CUDA), picks KV-cache type and batch sizes
from measured probes, benchmarks candidate flag sets (`--ai-tune`), finds/validates vision
projectors and speculative drafts, and recovers from crashes.

**vs Ollama.** Ollama wins on one-command simplicity and ecosystem on common hardware.
ggrun targets where Ollama's conservative heuristics leave performance behind:
mismatched multi-GPU rigs, MoE models split across VRAM/RAM, ik_llama.cpp speed, and full
flag access. One GPU and want zero config? Use Ollama.

**vs llama-swap.** llama-swap hot-swaps between model commands you write yourself;
ggrun computes those commands. They compose — point llama-swap at `ggrun dry-run`
output, or use `ggrun daemon` for single-model swapping.

| Capability | raw llama.cpp | ggrun |
|---|---:|---:|
| Multi-GPU / heterogeneous split | `--fit` (recent) | automatic, PCIe/bandwidth-weighted |
| MoE expert placement | `--fit` / manual `-ot` | exact per-GPU ledger, backend-aware |
| Backend selection (ik_llama / llama / Vulkan) | manual | automatic, dialect-aware |
| KV-cache type / batch sizing | manual | probe-measured |
| AI Tune (measured flag search) | no | yes, cached per model+hardware |
| Hardware-matched quant download | no | yes (HF search + intelligence ranking) |
| Vision projector / speculative decoding | manual | automatic, validated |
| Crash recovery / backend fallback | no | yes |

## Features

- One Go binary; Linux, macOS, and native Windows. CUDA / Vulkan / Metal / CPU.
- Exact-ledger multi-GPU + MoE expert placement (`--tensor-split` + `-ot` from measured
  VRAM and GGUF sizes), with adaptive retry on out-of-memory.
- **AI Tune** — benchmarks candidate flag sets and caches the fastest valid result per
  model + hardware; a community tune pool seeds first launches (`LLM_COMMUNITY_TUNES=off`).
- Hugging Face downloader with hardware-aware quant selection and a GUI recommendation
  picker ranked by intelligence-per-fit.
- Speculative decoding (MTP, EAGLE-3, validated draft GGUFs) and vision (`mmproj`) support.
- OpenAI-compatible server, arrow-key TUI, crash recovery with backend fallback.

## Backends

ik_llama.cpp (CUDA, source build) · llama.cpp (Vulkan, Metal, CPU) · native Windows CUDA
via `install.ps1 -Backend cuda`. The backend binary is pluggable via `LLAMA_SERVER`.

## Requirements

- **Linux:** `curl`, `git`, `python3`; `cmake`/compiler + NVIDIA CUDA toolkit for CUDA
  source builds; `vulkaninfo` for Vulkan detection.
- **macOS:** Apple Silicon; Xcode command-line tools for source builds.
- **Windows:** Windows 10/11 x86_64, PowerShell 5+, Python; CUDA Toolkit + VS C++ Build
  Tools for `-Backend cuda`.

## Documentation

[Install](docs/install.md) ·
[Usage](docs/usage.md) ·
[Architecture](docs/architecture.md) ·
[Performance](docs/performance.md) ·
[Launch benchmarks](docs/launch-performance.md) ·
[Speculative decoding](docs/speculative-decoding.md) ·
[Model recommendations](docs/model-recommendations.md) ·
[Changelog](CHANGELOG.md)

## License

MIT
