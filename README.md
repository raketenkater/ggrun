# llm-server

[![Stars](https://img.shields.io/github/stars/raketenkater/llm-server?style=social)](https://github.com/raketenkater/llm-server/stargazers)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20WSL2-lightgrey)](#requirements)
[![Backends](https://img.shields.io/badge/backends-ik__llama.cpp%20%7C%20llama.cpp-orange)](#)
[![Issues](https://img.shields.io/github/issues/raketenkater/llm-server)](https://github.com/raketenkater/llm-server/issues)

**Smart launcher for [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and [llama.cpp](https://github.com/ggml-org/llama.cpp).** Auto-detects your hardware, figures out the optimal configuration, and launches the server — no manual flag tuning. Then lets the model **tune its own flags** for up to +54% tokens/sec.

```bash
llm-server model.gguf      # launch a model
llm-server-gui             # interactive TUI picker
```

![demo](demo.gif)

## Install

Recommended v3 self-contained setup. This installs the Go `llm-server` as the primary command, downloads or builds the matching llama.cpp backend, and writes a local app-home config:

```bash
curl -fsSL https://raw.githubusercontent.com/raketenkater/llm-server/main/setup.sh | bash
```

Or from a clone:

```bash
git clone https://github.com/raketenkater/llm-server.git
cd llm-server
./setup.sh         # auto-detects Linux / WSL2 / macOS
```

This creates `~/llm-server` with:

```text
~/llm-server/bin      # llm-server, llm-server-gui, bundled backend when available
~/llm-server/models   # GGUF models and downloaded mmproj files
~/llm-server/cache    # AI Tune caches
~/llm-server/logs     # setup/server logs
~/llm-server/config   # local config loaded automatically
~/llm-server/src      # backend source/build fallback
```

Use it with:

```bash
source ~/llm-server/env.sh
llm-server-gui
llm-server <repo/name> --download
```

Legacy one-line install to `~/.local/bin`:

```bash
curl -fsSL https://raw.githubusercontent.com/raketenkater/llm-server/main/install.sh | bash
```

Detects your GPU, installs the Go launcher to `~/.local/bin/llm-server`, keeps the legacy Bash launcher as `llm-server-bash` when present, tries a matching prebuilt release bundle, then falls back to building the right backend from source (ik_llama.cpp for CUDA, llama.cpp for Vulkan/Metal/CPU). CUDA currently uses source builds unless a CUDA bundle was manually attached to the release. See [Requirements](#requirements) for what gets installed.

Prefer the manual path? `git clone … && cd llm-server && ./install.sh` works identically.

Installer controls:

```bash
LLM_INSTALL_MODE=release ./install.sh      # require a prebuilt bundle
LLM_INSTALL_MODE=build ./install.sh        # force source build
LLM_INSTALL_BACKEND=skip ./install.sh      # install scripts only
LLM_INSTALL_PY_DEPS=skip ./install.sh      # skip downloader Python deps
LLM_INSTALL_MAIN=bash ./install.sh          # keep legacy Bash as primary
LLM_INSTALL_PREFIX=/usr/local/bin ./install.sh
```

## Contents

- [Quick start](#quick-start)
- [Why llm-server?](#why-llm-server)
- [Features](#features)
- [AI self-tuning](#ai-self-tuning---ai-tune)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [v3 performance roadmap](docs/v3-performance-roadmap.md)
- [Release checklist](docs/releasing.md)
- [Changelog](CHANGELOG.md)

## Quick start

```bash
# Basic — auto-detects everything
llm-server model.gguf

# Let the model tune its own flags (cached forever after)
llm-server model.gguf --ai-tune

# Grab a quant from HuggingFace, recommends a size for your VRAM+RAM
llm-server unsloth/Qwen3.5-27B-GGUF --download

# Interactive TUI picker
llm-server-gui
```

## Why llm-server?

| | raw llama.cpp | Ollama | LM Studio | **llm-server** |
|---|---|---|---|---|
| Multi-GPU auto-placement | ❌ manual flags | partial | ❌ | ✅ |
| MoE expert offload (`-ot`/`--n-cpu-moe`) | ❌ manual | ❌ | ❌ | ✅ auto |
| AI self-tuning | ❌ | ❌ | ❌ | ✅ |
| Heterogeneous GPUs (different VRAM/PCIe) | ❌ manual | ❌ | ❌ | ✅ weighted |
| CLI-first / scriptable | ✅ | partial | ❌ | ✅ |
| Uses upstream llama.cpp / ik_llama.cpp | ✅ | fork | fork | ✅ |

Running llama.cpp on multi-GPU means juggling dozens of flags. llm-server figures it all out:

<table>
<tr><th>Without llm-server</th><th>With llm-server</th></tr>
<tr>
<td>

```bash
llama-server \
  -m model.gguf \
  -ngl 81 \
  --ctx-size 32768 \
  --tensor-split 24,12,12 \
  --split-mode graph -mg 0 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -fa on \
  --threads 8 --threads-batch 16 \
  -b 4096 -ub 1024 \
  --jinja --run-time-repack \
  -khad -defrag-thold 0.1 \
  --port 8081
```

</td>
<td>

```bash
llm-server model.gguf
```

</td>
</tr>
</table>

## Features

- **AI self-tuning** — the model being served proposes its own flags, benchmarks, and iterates 8 rounds. Crashes get free retries. Winner cached. Up to **+54% tok/s** in real tests.
- **Smart multi-GPU placement** — any mix of NVIDIA cards (0-8+), heterogeneous VRAM sizes, PCIe-bandwidth weighted. `--gpus 0,1` restricts to specific GPUs for multi-instance workloads.
- **MoE expert auto-placement** — starts conservative, measures actual VRAM, optimizes, caches. Automatic `--n-cpu-moe` fallback if `-ot` placement fails.
- **Auto-update + rollback** — `--update` pulls and rebuilds both backends; if the new binary breaks, rolls back to the last working commit.
- **Auto-fallback** — if ik_llama can't load a model, strips ik-specific flags and retries on mainline llama.cpp mid-launch.
- **Built-in downloader** — `--download` with any HuggingFace repo; recommends the best quant for your VRAM+RAM budget.
- **Vision (multimodal)** — `--vision` auto-detects and downloads the matching `mmproj` from HuggingFace.
- **Speculative decoding** — `--spec auto|mtp|eagle3|draft|ngram|ngram-mod|ngram-k4v` with backend-aware flag dialects. Off by default; `auto` prefers MTP when the model has NextN/MTP layers, then EAGLE-3 or a validated draft model, including Hugging Face GGUF drafter search when local candidates are missing, and otherwise stays off. Ngram modes are explicit/profile-gated only.
- **Terminal GUI** — `llm-server-gui` for interactive model picking with option toggles.

## AI self-tuning (`--ai-tune`)

Most llm-server users leave 20-50% performance on the table because "good enough" defaults aren't optimal for their specific hardware + model combination. `--ai-tune` fixes this:

```bash
llm-server model.gguf --ai-tune
```

The model being served proposes its own optimal flags, benchmarks each one, learns from the results, and iterates — 8 rounds of self-optimization. Crashes get free retries. The winner is cached for instant use on every future launch.

```
Round 0/8: Benchmarking baseline (heuristic config)...
  Baseline: gen=23.17 tok/s  pp=119.87 tok/s
Round 1/8: Config: optimized_split_hadamard_v2
  Result: gen=22.05 tok/s (-4.8%)
Round 3/8: Config: optimized_tensor_split_q4
  ★ NEW BEST: gen=24.56 tok/s (+6.0%)
Round 7/8: Config: ultra_stable_3090ti_bias
  ★ NEW BEST: gen=24.77 tok/s (+6.9%)

AI Tune complete: ultra_stable_3090ti_bias wins!
  Baseline: 23.17 tok/s → Best: 24.77 tok/s (+6.9%)
```

**No external API, no internet required.** The model tunes itself using its own intelligence. Results are cached — run `--ai-tune` once, benefit forever.

| Model | raw `llama-server` | llm-server heuristic | llm-server `--ai-tune` |
|---|---|---|---|
| **Qwen3.5-122B** | 4.1 tok/s | 11.2 tok/s | **17.47 tok/s** (+326%) |
| **Qwen3.5-27B Q4_K_M** | 18.5 tok/s | 25.94 tok/s | **40.05 tok/s** (+116%) |
| **gemma-4-31B UD-Q4_K_XL** | 14.2 tok/s | 23.17 tok/s | **24.77 tok/s** (+74%) |

*RTX 3090 Ti + RTX 4070 + RTX 3060 (49GB total VRAM) + 128GB RAM. More hardware results in [this r/LocalLLaMA post](https://www.reddit.com/r/LocalLLaMA/comments/1sl85r5/the_llm_tunes_its_own_llamacpp_flags_54_toks_on/).*

## Usage

```bash
# AI tune + re-tune
llm-server model.gguf --ai-tune
llm-server model.gguf --ai-tune --retune

# Vision
llm-server model.gguf --vision

# Show all cached configs / detail for one model
llm-server --show-configs
llm-server --show-configs model.gguf

# Launch with a specific tuned config
llm-server model.gguf --tune-cache ~/.cache/llm-server/tune_model_hw_llama.json

# Force a specific backend
llm-server --backend llama model.gguf
llm-server --backend ik_llama model.gguf

# Unattended / always-on
llm-server model.gguf --keep-alive

# Multi-instance: big model on GPUs 0+1, small on GPU 2
llm-server big-model.gguf   --gpus 0,1 --port 8081 --ram-budget 90G
llm-server small-model.gguf --gpus 2   --port 8082 --ram-budget 30G

# Update backends (rolls back on failure)
llm-server --update

# Quick tok/s benchmark then exit
llm-server --benchmark model.gguf

# Speculative decoding
llm-server model.gguf --spec auto       # MTP/EAGLE/draft when real, otherwise off
llm-server model.gguf --spec ngram      # conservative ngram map-k
llm-server model.gguf --spec mtp        # use MTP/NextN when the model and backend support it
llm-server model.gguf --spec eagle3     # use a matching EAGLE-3 speculator when available
llm-server model.gguf --spec ngram-mod  # explicit repetition/code self-speculation when supported
llm-server model.gguf --spec mtp        # IK MTP, or mainline draft-mtp when advertised
```

Any flag llm-server doesn't recognize is forwarded to `llama-server`, so every upstream option works without wrapping.

## How it works

### AI self-tuning
1. Launches with heuristic config → benchmarks as baseline
2. Sends the running model its hardware profile, GGUF metadata, full `--help`, and baseline
3. Model proposes flag overrides as JSON → script applies, launches, benchmarks
4. Feeds results back → model proposes next config → repeat for 8 rounds
5. Crashes get free retries (up to 4). Large models (>70% RAM) get a confirmation prompt.
6. Server processes are marked as OOM-kill targets — a bad config kills the server, never your system.
7. Winner cached at `~/.cache/llm-server/` — used automatically on future launches.
8. All results persisted to `tune_history.jsonl` — the model learns across sessions.

### Smart downloader
`--download` calculates `Total = System VRAM + System RAM`, inspects the HuggingFace repo, and recommends the quantization level with the best speed/quality balance for your exact hardware.

Downloaded and locally discovered GGUFs are indexed in `<model-dir>/.llm-server/models.json`. The index records model path, size, architecture, quant, matching local vision projectors, HuggingFace source metadata, and available AI Tune configs so the GUI can show runnable models and configs without guessing from filenames only.

### Vision
With `--vision`, llm-server checks for an existing `mmproj`, verifies it matches the loaded model, and downloads the correct `mmproj-F16.gguf` from HuggingFace if missing. The correct repo is inferred from GGUF metadata (`general.basename` + `general.quantized_by`).

### Auto-update
`llm-server --update` or `llm-server update` pulls llm-server, ik_llama.cpp, and llama.cpp; backs up each working binary; rebuilds with the existing cmake config; smoke-tests; and rolls back to the previous commit if anything fails. On interactive Go startup, llm-server checks whether the local repos are behind upstream or a newer GitHub release exists and asks before updating; release-bundle installs fall back to the latest tagged installer. Choose `d` to dismiss the prompt for 7 days or set `LLM_SERVER_NO_UPDATE_CHECK=1` to disable the startup check.

### Auto-fallback
If ik_llama.cpp can't load a model (unsupported architecture), llm-server detects the load failure, switches to mainline llama.cpp, strips ik-specific flags, and retries — no manual intervention.

## Requirements

**Linux:**
- Prebuilt release bundle when available, otherwise [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for CUDA or [llama.cpp](https://github.com/ggml-org/llama.cpp) for Vulkan/CPU
- CUDA source builds need NVIDIA drivers, CUDA toolkit/nvcc, `cmake`, `make`, and a compiler toolchain
- `nvidia-smi` (for GPU detection)
- `python3`, `huggingface_hub`, `tqdm`, `curl`

**macOS (Apple Silicon):**
- Prebuilt Metal release bundle when available, otherwise [llama.cpp](https://github.com/ggml-org/llama.cpp) built with Metal (or `brew install llama.cpp`)
- `python3`, `huggingface_hub`, `tqdm`, `curl`
- `llm-server-mac` supports Metal auto-offload, downloader, vision/mmproj auto-detect/download, benchmarks, GUI launching, startup fallbacks, and macOS AI Tune for Metal-safe flags (batch, ubatch, KV type, threads, parallel, flash-attn). Placement tuning is Linux/WSL2-only because Apple Silicon exposes a single unified Metal GPU.

**Windows:**
- Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) (`wsl --install` in PowerShell)
- Inside WSL2, follow the Linux instructions above
- NVIDIA GPU passthrough works automatically in WSL2 with up-to-date drivers
- Native Windows is not supported by llm-server today.

The `install.sh` one-liner handles release bundle detection, dependency checks, Python downloader dependencies, and backend source builds automatically.

## License

MIT

---
<p align="right">
  <a href="https://www.buymeacoffee.com/raketenkater">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 32px !important;width: 116px !important;" >
  </a>
</p>
