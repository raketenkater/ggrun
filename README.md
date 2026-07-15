# ggrun

ggrun is my launcher around [llama.cpp](https://github.com/ggml-org/llama.cpp)
and ik_llama.cpp. I started it because loading one GGUF is easy, but getting a
large model to use a weird mix of GPUs and system RAM without bad splits, OOMs,
or a pile of manual flags is not.

The project is mainly about three things:

1. running big MoE models that do not fit neatly into VRAM;
2. finding a fast configuration that also stays stable at the context and load I
   actually want to use;
3. making Claude Code's Ultracode workflows actually usable with a local model:
   parallel agents, tools, long contexts, research, and no cloud inference for
   the model calls.

ggrun is not an inference engine. It reads the GGUF and the machine, builds a
launch plan for the selected backend, checks that the plan fits, starts the
server, and keeps the generated command visible. Unknown flags still pass
through to `llama-server`.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/raketenkater/ggrun)](https://github.com/raketenkater/ggrun/releases/latest)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#backends)

## Quick start

Linux / macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/raketenkater/ggrun/main/setup.sh | bash
```

Windows (PowerShell):

```powershell
iwr -useb https://raw.githubusercontent.com/raketenkater/ggrun/main/install.ps1 | iex
```

Then run a local GGUF, download one from Hugging Face, or open the TUI:

```bash
ggrun model.gguf
ggrun unsloth/Qwen3.6-27B-GGUF --download
ggrun
```

![ggrun TUI demo](demo.gif)

## What it does

- Plans large-MoE placement from the GGUF, available VRAM and RAM, GPU
  bandwidth, and backend capabilities.
- Checks model, KV-cache, and safety-headroom memory before it starts a server.
- Supports dense and MoE models across single GPU, multi-GPU, CPU, and RAM
  offload configurations.
- Measures a bounded set of safe performance options with `--ai-tune` and
  preserves the winning configuration for the same setup.
- Keeps model downloads, recommendations, launches, and the generated command
  in one inspectable TUI flow.

## Local Claude Code and Ultracode workflows

```bash
ggrun model.gguf --claude-code
```

This starts the model, points Claude Code model aliases at the local
Anthropic-compatible endpoint, and launches the `claude` CLI when it is
installed. The point is to make Claude Code's Ultracode workflows usable with a
local model: parallel agents, tools, long contexts, research, and no cloud
inference for the model calls.

Context is shared between slots: `1M` total context with `--parallel 4` is about
`256k` per request. ggrun lowers the default parallelism when that split would
make the individual slots too small, and explicit values always win.

Claude Code itself still needs to be installed separately. ggrun replaces its
model endpoint and wires the local workflow; it does not make a model with weak
tool use behave like a strong coding model. The complete setup and overrides are
documented in [docs/usage.md](docs/usage.md#use-with-claude-code).

## Numbers from my weird rig

My reference machine is deliberately awkward: RTX 3090 Ti 24GB, RTX 3060 12GB,
RTX 4070 12GB, and 128GB RAM, with the smaller cards on slow PCIe links. These are
decode results from the dated, reproducible runs in
[docs/launch-performance.md](docs/launch-performance.md), not a promise that
every machine gets the same speedup.

Default placement, 32k context, without `--ai-tune`:

| Model | Ollama 0.30.8 | raw llama.cpp `--fit` | ggrun |
|---|---:|---:|---:|
| Qwen3.5-4B Q4_K_M | 124.8 | 103.3 | 151.4 tok/s |
| Qwen3.6-27B Q5_K_M | 22.8 | 24.3 | 37.4 tok/s |
| Qwen3.5-122B-A10B UD-IQ4_XS | 13.5 | 20.97 | 22.9 tok/s |
| MiniMax-M3 UD-IQ3_XXS | could not load | could not load | 5.59 tok/s |

The interesting result for me is not only the percentage: MiniMax-M3 spans VRAM
and RAM and actually runs. A separate DeepSeek-V4-Flash test at 1M context and
parallel 4 completed a 60,020-token request plus three concurrent requests at
5.88 decode tok/s without an OOM or restart. Full model, quant, backend, prefill,
memory, and load-test details are in the benchmark document.

The goal is the fastest **stable** plan for the requested workload, not maximum
VRAM fill or one lucky short benchmark. The default is a conservative, measured
placement heuristic, and `--ai-tune` explores a bounded set of flags for the
installed backend.

## Useful commands

```bash
ggrun model.gguf --dry-run       # print the backend command, do not launch
ggrun model.gguf --benchmark     # load, measure, and exit
ggrun model.gguf --ai-tune       # measure safe flag variants and cache the winner
ggrun model.gguf --claude-code   # launch a local Claude Code workflow
ggrun model.gguf --spec auto     # use only a validated speculative profile
ggrun spec-test model.gguf --ctx 1048576 --parallel 4
ggrun recommend                  # rank models for this machine and backend
ggrun models list                # show local GGUFs and grouped split models
ggrun models rm model.gguf       # safely remove a downloaded model
```

Placement and memory can be constrained explicitly:

```bash
ggrun model.gguf --gpus 0,1
ggrun model.gguf --ctx-size 32768
ggrun model.gguf --vram-headroom 2G
ggrun model.gguf --ram-headroom 8G
```

See [docs/usage.md](docs/usage.md) for all launcher options. Backend flags that
ggrun does not own are forwarded unchanged.

## Backends

- **Linux NVIDIA:** ik_llama.cpp CUDA is the most tested and fastest path on my
  machine. Matching releases ship a portable CUDA bundle; source build is the fallback.
- **Linux AMD / Intel:** mainline llama.cpp through Vulkan.
- **macOS:** mainline llama.cpp with Metal and unified-memory detection.
- **Windows:** CPU bundles and native NVIDIA CUDA support.
- **Custom binaries:** select one with `--server-bin` or `LLAMA_SERVER`.

## Documentation

[Install](docs/install.md) ·
[Getting started](docs/getting-started.md) ·
[Troubleshooting](docs/troubleshooting.md) ·
[Usage](docs/usage.md) ·
[Architecture](docs/architecture.md) ·
[Benchmarks](docs/launch-performance.md) ·
[Speculative decoding](docs/speculative-decoding.md) ·
[Model recommendations](docs/model-recommendations.md) ·
[Changelog](CHANGELOG.md)

## License

MIT
