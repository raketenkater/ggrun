# ggrun

ggrun (pronounced "g-run", from "gguf run") is my launcher around
[llama.cpp](https://github.com/ggml-org/llama.cpp) and ik_llama.cpp. It reads a
GGUF's tensor layout against the VRAM, RAM, and per-GPU bandwidth actually on
the machine, then places big MoE models across a mismatched multi-GPU setup so
the split fits and the server starts without an OOM or a pile of hand-tuned
flags. I started it because loading one GGUF is easy, but getting that split
right by hand was not.

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
[![CI](https://github.com/raketenkater/ggrun/actions/workflows/ci.yml/badge.svg)](https://github.com/raketenkater/ggrun/actions/workflows/ci.yml)

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
# (equivalently: ggrun download unsloth/Qwen3.6-27B-GGUF)
ggrun
```

On NVIDIA hardware, run this once (and again after changing GPUs or slots) to
replace topology estimates with measured host-memory and pinned PCIe transfer
bandwidth:

```bash
ggrun detect --bandwidth
```

The cached profile is accepted only on the exact CPU/RAM/GPU/PCI layout that
produced it. Normal `ggrun detect` and launches fall back to derived PCIe values
when the profile is absent, stale, incomplete, or corrupt.

![ggrun TUI demo](demo.gif)

## What it does

- Plans large-MoE placement from the GGUF, available VRAM and RAM, GPU
  bandwidth, and backend capabilities.
- Can measure each NVIDIA GPU's pinned host-to-device bandwidth and use the
  hardware-matched result for MoE placement and recommendation speed estimates.
- Checks model, KV-cache, and safety-headroom memory before it starts a server.
- Supports dense and MoE models across single GPU, multi-GPU, CPU, and RAM
  offload configurations.
- Measures a bounded set of safe performance options with `--ai-tune` and
  preserves the winning configuration for the same setup.
- Makes the ordinary TUI/direct launch converge from a stable placement estimate
  toward a faster measured whole configuration for its agent workload. The
  automatic path computes the complete safe neighbor set, then live-compares
  only the baseline and one highest-confidence finalist; the wider sweep stays
  an explicit maintenance operation. It measures repeated cache-backed turns
  plus mixed prefill/decode and promotes only after contained admission,
  branch/replay, lifecycle, and clean-relaunch gates. Exact evidence—including
  a measured result where the stable baseline won—is reused on the next launch.
- Prints an informational warning before the first load of a very large model,
  including the startup bound and the possibility of a bounded measured retry.
- Keeps model downloads, recommendations, launches, and the generated command
  in one inspectable TUI flow.
- Exposes llama.cpp's OpenAI-compatible `/v1` API (chat completions and
  completions) at the configured host:port, alongside the Anthropic endpoint
  used for Claude Code.

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

Agent loops resend a mostly-repeated prompt every turn — system prompt, tool
schemas, prior turns — so reprocessing all of it each time is the expensive
part, not generation. On backends that can shift a transformer context, Claude
mode turns on `--cache-reuse 256`, which reuses prompt chunks that moved after
a compaction or context shift, not just an exact prefix match: a compacted
4,506-token prefill went from 45.1 seconds to one processed token in 0.15
seconds in a production test. Hybrid/recurrent and multimodal models that
can't shift context that way (native DeepSeek V4, Laguna) get a rolling
context checkpoint per slot instead, kept when there's enough host RAM
headroom to hold it. Either path, a turn after the first one costs a fraction
of what it would cold. Mechanics and the opt-out flags are in
[docs/usage.md](docs/usage.md#prompt-caching-and-hybrid-models).

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
ggrun models browse              # browse curated downloads that fit this machine
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
- **macOS (deprecated, best-effort):** mainline llama.cpp with Metal and
  unified-memory detection. It builds, installs and serves today, and its CI job
  still runs so a regression stays visible. But nobody working on ggrun has a Mac
  to reproduce on, so macOS never holds a release and a release may ship without
  a macOS bundle. Use it; do not depend on it.
- **Windows:** CPU bundles and native NVIDIA CUDA support.
- **Custom binaries:** select one with `--server-bin` or `LLAMA_SERVER`.
- **FreeToken (experimental):** `ggrun freetoken <checkpoint> --gpu N` provides
  a strict single-NVIDIA-GPU process/API adapter without pretending FreeToken is
  a llama.cpp flag dialect. See [the adapter boundary](docs/freetoken.md).

## Documentation

[Install](docs/install.md) ·
[Getting started](docs/getting-started.md) ·
[Troubleshooting](docs/troubleshooting.md) ·
[Usage](docs/usage.md) ·
[Architecture](docs/architecture.md) ·
[Development roadmap](docs/development-roadmap.md) ·
[Fitting the hardware](docs/fitting-the-hardware.md) ·
[Benchmarks](docs/launch-performance.md) ·
[Speculative decoding](docs/speculative-decoding.md) ·
[Model recommendations](docs/model-recommendations.md) ·
[FreeToken adapter](docs/freetoken.md) ·
[Docker](docker/README.md) ·
[Release verification](docs/releases.md) ·
[Changelog](CHANGELOG.md) ·
[Contributing](CONTRIBUTING.md)

## License

MIT
