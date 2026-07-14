# ggrun

ggrun is a small helper for [llama.cpp](https://github.com/ggml-org/llama.cpp).
You point it at a GGUF and it figures out the flags, the multi-GPU split, and the
MoE expert placement so you don't have to. It's good at two things: making
llama.cpp easier to run, and running big MoE models that wouldn't otherwise fit —
by spreading them across your GPUs and system RAM.

**Run Anthropics ultracode workflows now with local model(unlimited tokens) with --claude-code**


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
- **Recommends and downloads what fits.** Ranks models for your hardware and pulls
  the GGUF at a quant sized to your VRAM, straight from Hugging Face.
- **Puts it all behind a menu.** Run `ggrun` with no arguments for a TUI that walks
  the whole detect → recommend → download → configure → launch loop, no flags.

## Benchmarks

ggrun's default placement vs raw llama.cpp `--fit` on the same GGUFs (RTX 3090 Ti x16 (pcie speed) +
4070 x4 + 3060 x1, 128GB RAM, 32k context, decode tok/s):

| Model | Ollama 0.30.8 | llama.cpp `--fit` | ggrun |
|---|---:|---:|---:|
| Qwen3.5-4B Q4_K_M | 124.8 | 103.3 | 151.4 |
| Qwen3.6-27B Q5_K_M | 22.8 | 24.3 | 37.4 |
| Qwen3.5-122B-A10B UD-IQ4_XS (MoE) | 13.5 | 20.97 | 22.9 |
| MiniMax-M3 UD-IQ3_XXS (MoE) | ✗ can't load | ✗ can't load | 5.59 |


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
ggrun spec-test model.gguf --ctx 1048576 --parallel 4 # prove MTP ceilings before Auto
```


Unknown flags pass straight through to `llama-server`. Full list in
[docs/usage.md](docs/usage.md).


## Backends

ik_llama.cpp (CUDA, source build) · llama.cpp (Vulkan, Metal, CPU) · native
Windows CUDA. The backend binary is pluggable via `LLAMA_SERVER`. AMD and Intel
GPUs run through Vulkan (no ROCm/HIP). macOS/Metal builds and detects unified
memory.

For brand-new architectures, For example, `ggrun backend install hy3` installs
the pinned `hy_v3` backend. See [fork backends](docs/fork-backends.md).

## Documentation

[Install](docs/install.md) ·
[Usage](docs/usage.md) ·
[Architecture](docs/architecture.md) ·
[Fork backends](docs/fork-backends.md) ·
[Benchmarks](docs/launch-performance.md) ·
[Model recommendations](docs/model-recommendations.md) ·
[Changelog](CHANGELOG.md)

## License

MIT
