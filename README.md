# ggrun

ggrun is a small helper for [llama.cpp](https://github.com/ggml-org/llama.cpp).
You point it at a GGUF and it figures out the flags, the multi-GPU split, and the
MoE expert placement so you don't have to. It's good at two things: making
llama.cpp easier to run, and running big MoE models that wouldn't otherwise fit —
by spreading them across your GPUs and system RAM.

I started it as a script for my own mismatched 3-GPU box, where hand-writing
`-ngl`, `--tensor-split`, and `-ot` for every model and context size got old.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/raketenkater/ggrun)](https://github.com/raketenkater/ggrun/releases/latest)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#backends)

```bash
ggrun model.gguf                          # serve a local GGUF
ggrun unsloth/Qwen3.6-27B-GGUF --download # download a fitting quant, then serve
ggrun                                     # no arguments → interactive TUI
```

![demo](demo.gif)

## What it does

- **Works out the placement.** Reads your GPUs, RAM, and PCIe layout and computes
  `--tensor-split` + `-ot` from the GGUF's exact tensor sizes — and loads big MoE
  models across VRAM + system RAM, so ones that don't fit on the GPU alone still run.
- **Finds the vision projector.** Detects and validates the matching `mmproj`
  automatically, so multimodal models just work.
- **Starts already tuned.** A community tune pool seeds the first launch with a
  known-good flag set for your model + hardware; `--ai-tune` can search for a
  faster one and cache it.
- **Recommends and downloads what fits.** Ranks models for your hardware and pulls
  the GGUF at a quant sized to your VRAM, straight from Hugging Face.
- **Puts it all behind a menu.** Run `ggrun` with no arguments for a TUI that walks
  the whole detect → recommend → download → configure → launch loop, no flags.

Underneath, it picks the backend (llama.cpp, or the faster ik_llama.cpp on CUDA),
serves an OpenAI-compatible API on `127.0.0.1`, and recovers if a launch crashes.

## Benchmarks

ggrun's default placement vs raw llama.cpp `--fit` on the same GGUFs (RTX 3090 Ti +
4070 + 3060, 128GB RAM, 32k context, decode tok/s):

| Model | Ollama 0.30.8 | llama.cpp `--fit` | ggrun |
|---|---:|---:|---:|
| Qwen3.5-4B Q4_K_M | 124.8 | 103.3 | 151.4 |
| Qwen3.6-27B Q5_K_M | 22.8 | 24.3 | 37.4 |
| Qwen3.5-122B-A10B UD-IQ4_XS (MoE) | 13.5 | 20.97 | 22.9 |
| MiniMax-M3 UD-IQ3_XXS (MoE) | ✗ can't load | ✗ can't load | 5.59 |

One rig — and its 3060 is on a PCIe x1 link, which amplifies the multi-GPU/MoE gains.
Full method and caveats: [docs/launch-performance.md](docs/launch-performance.md). These
are ggrun's defaults, no `--ai-tune`.

## When you might not need it

If you have a single GPU and a model that fits in its VRAM, plain llama.cpp or
[Ollama](https://ollama.com) is simpler and works great. ggrun earns its keep on
mismatched multi-GPU rigs and on MoE models that have to spill into system RAM —
that's the situation it was built for.

## Install

Linux / macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/raketenkater/ggrun/main/setup.sh | bash
```

Windows (PowerShell):

```powershell
iwr -useb https://raw.githubusercontent.com/raketenkater/ggrun/main/install.ps1 | iex
```

Prebuilt bundles install without compiling; Linux CUDA (ik_llama.cpp) builds from
source for your GPU. Details in [docs/install.md](docs/install.md).

## Usage

```bash
ggrun model.gguf --dry-run     # print the llama-server command without running it
ggrun model.gguf --ai-tune     # benchmark a few flag sets, cache the fastest
ggrun model.gguf --benchmark   # load, measure tok/s, exit
ggrun model.gguf --claude-code # serve + launch Claude Code wired to this model
```

Unknown flags pass straight through to `llama-server`. Full list in
[docs/usage.md](docs/usage.md).

> **Security:** the OpenAI-compatible API is unauthenticated and binds to
> `127.0.0.1`. To reach it from other machines, set `--host 0.0.0.0` and put it
> behind a firewall.

## Backends

ik_llama.cpp (CUDA, source build) · llama.cpp (Vulkan, Metal, CPU) · native
Windows CUDA. The backend binary is pluggable via `LLAMA_SERVER`. AMD and Intel
GPUs run through Vulkan (no ROCm/HIP). macOS/Metal builds and detects unified
memory but isn't yet validated on Apple hardware.

## Documentation

[Install](docs/install.md) ·
[Usage](docs/usage.md) ·
[Architecture](docs/architecture.md) ·
[Benchmarks](docs/launch-performance.md) ·
[Model recommendations](docs/model-recommendations.md) ·
[Changelog](CHANGELOG.md)

## License

MIT
