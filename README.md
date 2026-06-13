# llm-server

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/raketenkater/llm-server)](https://github.com/raketenkater/llm-server/releases/latest)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#requirements)
[![Backends](https://img.shields.io/badge/backends-ik__llama.cpp%20%7C%20llama.cpp-orange)](#backends)

**Stop hand-writing `--tensor-split`, `-ot`, and KV-cache flags.** llm-server
is an auto-tuned launcher for GGUF models: it measures your GPUs, RAM, and PCIe
topology, picks the right backend (llama.cpp or the faster ik_llama.cpp fork),
computes multi-GPU and MoE expert placement, and serves an OpenAI-compatible
API — one command from GGUF file to running endpoint.

```bash
llm-server model.gguf                          # local GGUF → served
llm-server unsloth/Qwen3.6-27B-GGUF --download # HF repo → hardware-matched quant → served
llm-server model.gguf --ai-tune                # benchmark flag sets, cache the fastest
```

![demo](demo.gif)

Built for machines Ollama serves poorly: mismatched multi-GPU rigs
(24GB + 12GB + 12GB), big MoE models split across VRAM and RAM, and anyone who
wants llama.cpp's full flag surface with measured — not guessed — defaults.

## Install

Recommended self-contained setup on Linux/macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/raketenkater/llm-server/main/setup.sh | bash
```

Native Windows setup from PowerShell:

```powershell
iwr -useb https://raw.githubusercontent.com/raketenkater/llm-server/main/install.ps1 | iex
```

Native Windows NVIDIA CUDA setup:

```powershell
iwr -useb https://raw.githubusercontent.com/raketenkater/llm-server/main/install.ps1 -OutFile install.ps1
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Backend cuda
```

Since v3.0.0, [prebuilt release bundles](https://github.com/raketenkater/llm-server/releases/latest)
(Linux CPU/Vulkan, macOS arm64 Metal, Windows x86_64 CPU) are downloaded and verified against the
published `SHA256SUMS` — no compile needed. Linux CUDA/ik_llama.cpp installs build from source
for your exact GPU architecture. Windows NVIDIA CUDA installs use a native llama.cpp CUDA backend,
either from an optional `llm-server-windows-x86_64-cuda.zip` release asset or by building it locally.

From a clone:

```bash
git clone https://github.com/raketenkater/llm-server.git
cd llm-server
./setup.sh
```

This creates a clean app home under `~/llm-server`:

```text
llm-server       CLI launcher
llm-server-gui   terminal GUI launcher
models/          GGUF models and downloaded vision projectors
.bin/            Go binary, tools, and bundled backend when available
.config/         local config loaded by the launcher
.cache/          AI Tune and model index cache
.logs/           setup and server logs
.src/            backend source/build fallback
```

Use it with:

```bash
~/llm-server/llm-server-gui
~/llm-server/llm-server <repo/name> --download
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

Typical raw command for a heterogeneous 3-GPU box:

```bash
llama-server   -m model.gguf   --ctx-size 32768   --tensor-split 24,12,12   --split-mode layer   --cache-type-k q4_0   --cache-type-v q4_0   --threads 8 --threads-batch 8   -b 8192 -ub 1024   --jinja --flash-attn on   --port 8081
```

With llm-server:

```bash
llm-server model.gguf
```

### How it compares

**vs raw llama.cpp.** Upstream recently gained `--fit` (auto GPU layers,
tensor-split, context targeting ~85-90% VRAM, and some MoE tensor
overrides) — if that is all you need,
raw llama.cpp may be enough. llm-server goes further: it selects the backend
(ik_llama.cpp is meaningfully faster on CUDA for many models, but its flag
dialect differs), chooses KV-cache quantization and batch sizes from measured
probes, finds and validates vision projectors and speculative-decoding drafts,
benchmarks candidate flag sets against each other (`--ai-tune`), and restarts
or falls back to mainline on crashes. Unknown flags pass straight through, so
nothing upstream is ever out of reach.

**vs Ollama.** Ollama optimizes for one-command simplicity on common hardware
and has a far larger ecosystem. llm-server targets the machines where Ollama's
conservative heuristics leave performance behind: mismatched multi-GPU rigs,
MoE models split across VRAM/RAM, ik_llama.cpp speed, and full llama.cpp flag
access. If you have one GPU and want zero configuration, use Ollama.

**vs llama-swap.** llama-swap is a proxy that hot-swaps between configured
model commands; you still write each model's flags yourself. llm-server
computes those flags. They compose well: point llama-swap entries at
`llm-server dry-run` output, or use `llm-server daemon` (control API with
`/reload`) for single-model swapping.

| Capability | raw llama.cpp | llm-server |
|---|---:|---:|
| Multi-GPU placement | `--fit` (recent) | automatic, PCIe/bandwidth-weighted |
| Heterogeneous GPU split | `--fit` (recent) | automatic |
| MoE expert placement | `--fit`/manual `-ot` (recent) | automatic, backend-aware, with `--n-cpu-moe` fallback |
| Backend selection (ik_llama vs mainline vs Vulkan) | manual | automatic, dialect-aware |
| KV-cache type / batch sizing | manual | probe-measured |
| AI Tune (measured flag search) | no | yes, cached per model+hardware |
| Hardware-matched quant download | no | yes (HF search + intelligence ranking) |
| Vision projector lookup | manual | automatic local/HF lookup |
| Speculative decoding | manual | validated, backend-aware modes |
| Crash recovery / backend fallback | no | yes |

## Features

- Go v3 command line with Linux, macOS, native Windows, and cross-build support.
- Backend-aware launch flags for ik_llama.cpp and llama.cpp.
- Multi-GPU placement using VRAM, free memory, and PCIe weighting.
- MoE-aware expert placement with `-ot` / `--n-cpu-moe` fallback paths.
- AI Tune: benchmarks candidate flag sets and caches the fastest valid result.
- Community tune pool: first launches reuse configs measured by others on the
  same GPU set, sanitized to safe performance flags (`LLM_COMMUNITY_TUNES=off` to opt out).
- Model downloader that searches Hugging Face GGUF repos and picks a hardware-aware quant.
- GUI recommended-download fast path ranked by intelligence signal after hardware fit.
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
llm-server unsloth/Qwen3.6-27B-GGUF --download

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

The script writes JSON logs and a Markdown summary under `.benchmarks/`. Generated
benchmark runs are ignored by Git; commit only curated summaries.

### v3 release measurements

**Methodology:** "raw" rows run the same backend binary with its own default
flags at the same context size on the same hardware — the gains come from
placement, KV/batch selection, and tuned flags, not from comparing different
backends. Every number is reproducible with
[`scripts/bench-v3-comparison.sh`](scripts/bench-v3-comparison.sh) (JSON +
Markdown artifacts); the exact commands for each row are in
[docs/performance.md](docs/performance.md). If your numbers differ, open an
issue with the artifact — regressions against these tables are treated as bugs.

Measured on 2026-06-10 on Linux with RTX 3090 Ti 24GB, RTX 3060 12GB,
RTX 4070 12GB, 128GB RAM, and i7-10700K. Context was 32k for every row.
Dense-model rows use the long prompt profile with 512 generated tokens and the
median of three rounds. MoE rows use 256 generated tokens because the 95GB split
model has multi-minute startup/repack time.

| Model | Backend / mode | Decode tok/s | Result |
|---|---:|---:|---|
| Qwen3.5 4B Q4_K_M | raw IK CUDA backend | 122.66 | baseline |
| Qwen3.5 4B Q4_K_M | v3 IK CUDA default | 156.21 | +27% vs raw |
| Qwen3.5 4B Q4_K_M | v3 IK CUDA AI-tune | 183.85 | +50% vs raw |
| Qwen3.5 4B Q4_K_M | llama.cpp CPU | 11.26 | CPU fallback |
| Qwen3.5 4B Q4_K_M | llama.cpp Vulkan default | 158.42 | cross-platform GPU path |
| Qwen3.5 4B Q4_K_M | llama.cpp Vulkan AI-tune | 169.61 | `-ub 512`, +7% vs Vulkan default |
| Qwen3.6 27B Q5_K_M | raw IK CUDA backend | failed | OOM on this 24GB primary GPU setup |
| Qwen3.6 27B Q5_K_M | v3 IK CUDA default | 37.68 | stable 32k context |
| Qwen3.6 27B Q5_K_M | v3 IK CUDA AI-tune | 37.69 | baseline correctly kept |
| Qwen3.6 27B Q5_K_M | llama.cpp Vulkan default | 37.72 | stable 32k context |
| Qwen3.6 27B Q5_K_M | llama.cpp Vulkan AI-tune | 37.67 | baseline correctly kept |
| Qwen3.6 27B Q5_K_M | IK CUDA speculative auto | 11.16 | not a release default; draft acceptance ~18% |
| MiniMax M2.7 UD-Q3_K_XL | v3 IK CUDA MoE default | 11.27 | 230B/A10B MoE, 95GB split GGUF |
| MiniMax M2.7 UD-Q3_K_XL | MoE AI-tune `-b 1024` | 11.29 | below 1% tune threshold |
| MiniMax M2.7 UD-Q3_K_XL | MoE AI-tune `-b 1536` | 11.30 | below 1% tune threshold |

AI-tune uses a 1% noise floor before replacing the default config. That is why
small MoE differences are reported but not selected. The MoE path currently
prioritizes stability, full context, and safe expert placement over chasing tiny
single-run gains. The next major MoE tuning target is persistent in-process
candidate testing, because relaunching a 95GB split model dominates tune time.

## Backends

- CUDA/NVIDIA: ik_llama.cpp source build by default, or a manually published CUDA
  release bundle.
- Vulkan: llama.cpp Vulkan build.
- Metal/macOS: llama.cpp Metal build or release bundle.
- Windows: native x86_64 CPU release bundle; NVIDIA CUDA via `install.ps1 -Backend cuda` or a custom `LLAMA_SERVER`.
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

- Windows 10/11 x86_64
- PowerShell 5+
- Python available as `python3`, `python`, or `py` for GGUF parsing/downloader helpers
- Native CPU release bundle is produced automatically
- NVIDIA driver, CUDA Toolkit with `nvcc`, CMake, Git, and Visual Studio C++ Build Tools for `install.ps1 -Backend cuda`
- Windows Vulkan is not a supported llm-server target

## Documentation

- [Architecture](docs/architecture.md)
- [Performance](docs/performance.md)
- [Launch performance tables](docs/launch-performance.md)
- [Speculative decoding](docs/speculative-decoding.md)
- [Model recommendations](docs/model-recommendations.md)
- [Release checklist](docs/releasing.md)
- [Repository hygiene](docs/repo-hygiene.md)
- [Changelog](CHANGELOG.md)

## License

MIT
