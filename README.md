# llm-server

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20WSL2-lightgrey)](#requirements)
[![Backends](https://img.shields.io/badge/backends-ik__llama.cpp%20%7C%20llama.cpp-orange)](#backends)

Go-first launcher for GGUF inference servers. llm-server detects the local
hardware, selects llama.cpp or ik_llama.cpp flags, and launches an
OpenAI-compatible server without hand-writing placement, KV-cache, batch, and
backend options for every model.

```bash
llm-server model.gguf
llm-server-gui
```

![demo](demo.gif)

## Install

Recommended self-contained setup:

```bash
curl -fsSL https://raw.githubusercontent.com/raketenkater/llm-server/main/setup.sh | bash
```

From a clone:

```bash
git clone https://github.com/raketenkater/llm-server.git
cd llm-server
./setup.sh
```

This creates an app home under `~/llm-server`:

```text
bin/      llm-server, llm-server-gui, bundled backend when available
models/   GGUF models and downloaded vision projectors
cache/    AI Tune and model index cache
logs/     setup and server logs
config/   local config loaded by the launcher
src/      backend source/build fallback
```

Use it with:

```bash
source ~/llm-server/env.sh
llm-server-gui
llm-server <repo/name> --download
```

Classic install to `~/.local/bin`:

```bash
curl -fsSL https://raw.githubusercontent.com/raketenkater/llm-server/main/install.sh | bash
```

Installer controls:

```bash
LLM_INSTALL_MODE=release ./install.sh      # require a prebuilt bundle
LLM_INSTALL_MODE=build ./install.sh        # force source build
LLM_INSTALL_BACKEND=skip ./install.sh      # install launcher/tools only
LLM_INSTALL_PY_DEPS=skip ./install.sh      # skip downloader Python deps
LLM_INSTALL_PREFIX=/usr/local/bin ./install.sh
```

Existing Bash installs are treated as legacy. The installer preserves them as
`llm-server-bash` when replacing the primary `llm-server` command with Go v3.

## Why Use It

| Capability | raw llama.cpp | llm-server |
|---|---:|---:|
| Multi-GPU placement | manual | automatic |
| Heterogeneous GPU split | manual | automatic |
| MoE expert placement | manual | automatic with fallback |
| Backend selection | manual | llama.cpp / ik_llama.cpp aware |
| AI Tune | no | measured flag search |
| Vision projector lookup | manual | automatic local/HF lookup |
| Speculative decoding | manual | validated, backend-aware modes |
| Release benchmarking | manual | scripts with JSON/Markdown artifacts |

Typical raw command:

```bash
llama-server   -m model.gguf   --ctx-size 32768   --tensor-split 24,12,12   --split-mode layer   --cache-type-k q4_0   --cache-type-v q4_0   --threads 8 --threads-batch 8   -b 8192 -ub 1024   --jinja --flash-attn on   --port 8081
```

With llm-server:

```bash
llm-server model.gguf
```

## Features

- Go v3 command line with Linux, macOS, WSL2, and cross-build support.
- Backend-aware launch flags for ik_llama.cpp and llama.cpp.
- Multi-GPU placement using VRAM, free memory, and PCIe weighting.
- MoE-aware expert placement with `-ot` / `--n-cpu-moe` fallback paths.
- AI Tune: benchmarks candidate flag sets and caches the fastest valid result.
- Model downloader for Hugging Face GGUF repos with hardware-aware quant choice.
- Vision projector lookup and validation for multimodal GGUF models.
- Speculative decoding modes for MTP, EAGLE-3, validated draft models, and
  explicit ngram modes.
- Startup update checks for interactive users, with rollback on failed updates.
- Terminal UI via `llm-server-gui` or `llm-server gui`.

## Quick Start

```bash
# Launch a local model
llm-server ~/models/model.gguf

# Download a GGUF from Hugging Face, then launch it
llm-server unsloth/Qwen3.5-27B-GGUF --download

# Run AI Tune once, then reuse the cached result
llm-server model.gguf --ai-tune

# Print the backend command without launching
llm-server model.gguf --dry-run

# Run the terminal UI
llm-server-gui
```

## Usage

```bash
# Backends
llm-server --backend ik_llama model.gguf
llm-server --backend llama model.gguf
llm-server --backend vulkan model.gguf

# Placement and memory
llm-server model.gguf --gpus 0,1
llm-server model.gguf --ram-budget 90G
llm-server model.gguf --ctx-size 32768
llm-server model.gguf --kv-quality mid
llm-server model.gguf --kv-placement gpu

# Vision
llm-server model.gguf --vision
llm-server model.gguf --mmproj /path/to/mmproj.gguf

# Tuning and cached configs
llm-server model.gguf --ai-tune
llm-server model.gguf --ai-tune --retune
llm-server --show-configs
llm-server model.gguf --tune-cache ~/.cache/llm-server/tune.json

# Speculative decoding
llm-server model.gguf --spec auto
llm-server model.gguf --spec mtp
llm-server model.gguf --spec eagle3
llm-server model.gguf --spec draft
llm-server model.gguf --spec ngram-mod

# Maintenance
llm-server --update
llm-server model.gguf --benchmark
```

Unknown flags are passed through to `llama-server`, so upstream options remain
available without wrapper changes.

## AI Tune

`--ai-tune` starts from the launcher heuristic, benchmarks it, tests candidate
flag sets, and stores the best successful result in the local cache. The served
model can propose candidate flags, but the launcher validates them against
backend help, memory headroom, crash behavior, and benchmark results before a
cache entry is reused.

Release claims should be tied to benchmark artifacts. See
[docs/performance.md](docs/performance.md) for the benchmark format and the
current speculative-decoding findings.

## Speculative Decoding

`--spec auto` only enables a real validated path:

1. MTP when the target GGUF has NextN/MTP metadata and the backend supports it.
2. EAGLE-3 when a matching speculator is available and the backend advertises it.
3. A compatible draft GGUF found locally or through Hugging Face search.
4. Off when no validated path exists.

Ngram modes are explicit because they are workload-sensitive. Recent local tests
showed a large gain on structured/repetitive output and a regression on code
continuation with low draft acceptance. See
[docs/speculative-decoding.md](docs/speculative-decoding.md).

## Benchmarks

Compare raw backend launch against Go v3:

```bash
scripts/bench-v3-comparison.sh model.gguf   --server-bin /path/to/llama-server   --ctx-size 32768   --rounds 3
```

Optional historical comparison against an installed Bash v2 launcher:

```bash
scripts/bench-v3-comparison.sh model.gguf   --server-bin /path/to/llama-server   --bash-bin ~/.local/bin/llm-server-bash
```

The script writes JSON logs and a Markdown summary under `benchmarks/`. Generated
benchmark runs are ignored by Git; commit only curated summaries.

## Backends

- CUDA/NVIDIA: ik_llama.cpp source build by default, or a manually published CUDA
  release bundle.
- Vulkan: llama.cpp Vulkan build.
- Metal/macOS: llama.cpp Metal build or release bundle.
- CPU: llama.cpp CPU build or release bundle.

## Requirements

Linux:

- `curl`, `git`, `python3`
- `cmake`, `make`, and a compiler when building a backend from source
- NVIDIA driver and CUDA toolkit for CUDA source builds
- `vulkaninfo` for Vulkan auto-detection

macOS:

- Apple Silicon recommended
- Xcode command line tools when building from source
- Metal-capable llama.cpp build or release bundle

Windows:

- Use WSL2. Native Windows packages are not produced today.

## Documentation

- [Architecture](docs/architecture.md)
- [Performance](docs/performance.md)
- [Speculative decoding](docs/speculative-decoding.md)
- [Release checklist](docs/releasing.md)
- [Repository hygiene](docs/repo-hygiene.md)
- [Changelog](CHANGELOG.md)

## License

MIT
