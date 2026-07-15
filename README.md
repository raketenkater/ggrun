# ggrun

ggrun is my launcher around [llama.cpp](https://github.com/ggml-org/llama.cpp)
and ik_llama.cpp. I started it because loading one GGUF is easy, but getting a
large model to use a weird mix of GPUs and system RAM without bad splits, OOMs,
or a pile of manual flags is not.

The project is mainly about three things:

1. running big MoE models that do not fit neatly into VRAM;
2. finding a fast configuration that also stays stable at the context and load I
   actually want to use;
3. making the served model useful for local coding-agent workflows, including
   Claude Code.

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

## What works now

- **Large-MoE placement.** ggrun reads exact tensor sizes from the GGUF and uses
  the available VRAM, RAM, GPU bandwidth, and backend capabilities to compute
  `--tensor-split`, layer ownership, and whole-expert storage. On mixed rigs it
  can keep dense work on the fast GPU while slower cards store experts.
- **Memory checks before launch.** Model tensors, KV cache, compute buffers,
  speculative models, and safety headroom are included in the plan. Failed
  placements can be recovered and corrected instead of restarting forever.
- **Normal serving paths.** Dense single-GPU, dense multi-GPU, CPU-only, RAM
  offload, and large-MoE paths all use the same planner. CUDA, Vulkan, Metal,
  CPU, and native Windows CUDA backends are supported.
- **Measured tuning.** `--ai-tune` tests safe performance flags against the
  installed backend, confirms the result, and caches the fastest successful
  configuration. Context size, parallelism, and quality-affecting settings stay
  under user control.
- **Speculative decoding with a brake pedal.** ggrun can resolve embedded MTP,
  compatible companion models, DFlash, EAGLE-3, and draft GGUFs. Auto mode only
  uses a speculative path when a matching measured profile passed its checks;
  otherwise it stays off.
- **Model and vision setup.** The recommender filters by hardware and backend
  support, downloads a fitting quant from Hugging Face, and finds and validates
  the matching vision projector when one is needed.
- **A TUI for the whole loop.** Running `ggrun` without arguments opens the
  detect -> recommend -> download -> configure -> launch flow. The generated
  plan is still inspectable, so the TUI is not hiding what gets executed.

## Local Claude Code workflows

```bash
ggrun model.gguf --claude-code
```

This starts the model, points all Claude Code model aliases at the local
Anthropic-compatible endpoint, and launches the `claude` CLI when it is
installed. ggrun also sets up the parts that became necessary once I tried
running actual agent workflows instead of a short chat:

- four local model slots by default when the total context is large enough;
- per-slot compaction and prompt-cache reuse where the model architecture allows it;
- long-running request handling without a small cloud-style inference timeout;
- live request progress for queued, prompt-processing, and generation states;
- local web search and page fetching through a DuckDuckGo MCP when `uvx` is available;
- fail-closed Claude Auto permission reviews through a separate local Qwen3.5-2B
  reviewer, instead of spending the main model's context on every hidden review.

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
VRAM fill or one lucky short benchmark. Today the default is a conservative,
measured placement heuristic and `--ai-tune` explores a bounded set of flags.
Generic first-run calibration across every model and hardware combination is
still work in progress, so I do not claim that ggrun already proves the global
fastest configuration everywhere.

## Useful commands

```bash
ggrun model.gguf --dry-run       # print the backend command, do not launch
ggrun model.gguf --benchmark     # load, measure, and exit
ggrun model.gguf --ai-tune       # measure safe flag variants and cache the winner
ggrun model.gguf --claude-code   # launch a local Claude Code workflow
ggrun model.gguf --spec auto     # use only a validated speculative profile
ggrun spec-test model.gguf --ctx 1048576 --parallel 4
ggrun recommend                  # rank models for this machine and backend
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
  machine. The installer currently builds it for the local GPU architecture.
- **Linux AMD / Intel:** mainline llama.cpp through Vulkan.
- **macOS:** mainline llama.cpp with Metal and unified-memory detection.
- **Windows:** CPU bundles and native NVIDIA CUDA support.
- **Custom binaries:** select one with `--server-bin` or `LLAMA_SERVER`.

New architectures often land in a fork before upstream. ggrun can install a
pinned fork recipe or register any already-built `llama-server`, keep it isolated,
and route only matching GGUF architectures to it:

```bash
ggrun backend recipes
ggrun backend install hy3
ggrun backend register --tag new-model --path /path/to/llama-server \
  --route-arch new_model_arch
```

The HY3 recipe build and routing are verified, but real HY3 GGUF serving is still
experimental until its load, output, and parallel tests are complete. See
[docs/fork-backends.md](docs/fork-backends.md).

## Project status

This is an active personal project. The Linux CUDA path on mixed NVIDIA hardware
has received the deepest live testing because that is the machine I built it for.
The other serving paths have load/generation tests and CI coverage, but they do
not all have the same amount of performance data yet.

The remaining work is tracked in [TODO.md](TODO.md). The main open items are the
broader MTP performance matrix, real HY3 validation, a complete four-agent Claude
Code acceptance run, and generic first-run speed calibration.

## Documentation

[Install](docs/install.md) ·
[Usage](docs/usage.md) ·
[Architecture](docs/architecture.md) ·
[Fork backends](docs/fork-backends.md) ·
[Benchmarks](docs/launch-performance.md) ·
[Speculative decoding](docs/speculative-decoding.md) ·
[Model recommendations](docs/model-recommendations.md) ·
[Changelog](CHANGELOG.md)

## License

MIT
