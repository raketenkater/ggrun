# ggrun

A launcher for llama.cpp and ik_llama.cpp. It recommends a model for your machine, places it across mismatched GPUs and RAM, and checks that it fits before it serves.

ggrun is not an inference engine. It reads the GGUF and the hardware: CPU, RAM, GPUs and PCIe links. From those it builds a `llama-server` command for a backend that supports the model's architecture, and checks that command against the backend's own memory measurement before it starts the server. The generated command stays visible (`--dry-run`), and flags ggrun does not handle are passed to `llama-server` unchanged. Linux with NVIDIA is the tested path.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/raketenkater/ggrun)](https://github.com/raketenkater/ggrun/releases/latest)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)](#platforms)
[![CI](https://github.com/raketenkater/ggrun/actions/workflows/ci.yml/badge.svg)](https://github.com/raketenkater/ggrun/actions/workflows/ci.yml)
[![Install E2E](https://github.com/raketenkater/ggrun/actions/workflows/install-e2e.yml/badge.svg)](https://github.com/raketenkater/ggrun/actions/workflows/install-e2e.yml)

![ggrun: install, recommended model, download, launch and a Claude Code fix](demo.gif)

Recorded October 14, 2026 at `16c2ae1` (v3.2.11 plus fixes) on 3 GPUs. It shows a public install, the recommended model, the download, an automatic launch, and Claude Code fixing a failing test. Sped-up parts are captioned with their real duration.

## Install

Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/raketenkater/ggrun/main/setup.sh | bash
```

Windows (PowerShell):

```powershell
iwr -useb https://raw.githubusercontent.com/raketenkater/ggrun/main/install.ps1 | iex
```

With an NVIDIA GPU, the Windows installer picks CUDA automatically (upstream llama.cpp CUDA prebuilt, or a source build). `-Backend cpu` forces CPU. See [docs/install.md](docs/install.md).

The installer creates an app home under `~/ggrun` with the launcher, models, config, cache and logs.

Before installing anything, it looks for an existing `llama-server`, the NVIDIA driver and the CUDA toolkit. It reuses an existing backend unless it fails to load. Otherwise it installs prebuilts (ik_llama.cpp CUDA on NVIDIA, plus llama.cpp Vulkan; CPU if neither runs) and builds from source only as a fallback. It never installs the NVIDIA driver; it asks before installing Vulkan runtime packages or the CUDA toolkit.

Windows installs the latest release. Linux setup builds ggrun from current `main` (it fetches Go if needed).

Installer controls, the app-home layout and uninstall steps are in [docs/install.md](docs/install.md).

## Quick start

```bash
ggrun recommend                            # 1. models and quants that fit this machine
ggrun download unsloth/Qwen3.6-27B-GGUF    # 2. choose a quant (--yes takes the recommended one)
ggrun                                      # 3. terminal UI: pick the downloaded model, dry run or launch
```

The download pins one repository revision, checks free disk space first, skips draft and companion files, and hash-checks every file. When it finishes, it prints the `ggrun <file>` command for the downloaded model.

Other ways to launch:

- `ggrun <file>.gguf` plans, checks and serves on :8081 (`/v1` and `/v1/messages`).
- `ggrun <file>.gguf --claude-code` serves the model and starts the `claude` CLI against it. Claude Code is installed separately.

On NVIDIA, run this once, and again after changing GPUs or slots:

```bash
ggrun detect --bandwidth
```

It measures pinned host-to-device bandwidth on each GPU and saves the result. The saved profile is used only on the exact CPU/RAM/GPU/PCI layout that produced it. Without it, ggrun uses PCIe values derived from link generation and width.

## What it adds on top of llama-server

llama.cpp's `--fit` is on by default and fits a model to free memory well. If one GPU holds your model, `--fit` is probably enough. ggrun is for the cases around it:

- **Placement weighted by bandwidth on mismatched rigs.** One fast-link GPU owns the dense layers, and slower-link GPUs store whole expert layers. A per-device ledger from the GGUF counts the KV cache, SWA, recurrent state, compute buffers and an optional companion model.
- **Admission before serving.** The plan is checked against `llama-fit-params` or an allocation-only dry run. When neither is available, ggrun asks before running a contained full-load probe. It re-plans until the plan fits.
- **Host memory containment on Linux.** Each backend runs in its own cgroup v2 scope (through systemd). MemoryHigh and MemoryMax come from the RAM policy, so a breach kills the backend, not the machine. ggrun refuses resident plans whose host footprint plus reserve would exceed the whole-host ceiling. If a model fits only with weights paged from disk, ggrun stops and asks `Use mmap? [y/N]`; non-interactive runs need `--mmap`. Hosts without systemd, such as containers, can set `GGRUN_SCOPE_MODE=none`. The backend then runs uncontained, and only the pre-launch check applies.
- **Backend chosen by architecture.** ggrun uses an installed backend that registers the model's architecture. If none does, it offers a reviewed pinned fork recipe, a current mainline CUDA build, or an open llama.cpp PR for that architecture as an unreviewed fork.
- **Recommendations that launch.** Models are ranked by Artificial Analysis intelligence within the detected VRAM/RAM. Each row shows the best quant that fits, a speed estimate, and whether an installed backend can run the model now or needs a build first.
- **Claude Code settings for long agent sessions:**
  - the auto-compact window is set to the real context per slot;
  - prompt reuse after compaction (`--cache-reuse 256`, or a context checkpoint per slot on hybrid models);
  - `CLAUDE_CODE_ATTRIBUTION_HEADER=0` keeps the prompt prefix stable so the backend's cache is reused (the printed recipe sets the same variables);
  - context shift is off, so a full context errors instead of truncating;
  - agents queue at the gateway when the model is offloaded to CPU;
  - a live prefill/decode status line;
  - `ggrun claude resume` relaunches the recorded backend shape and reopens the session.

  The OpenAI-compatible `/v1` endpoint serves other agent harnesses as well.

### Claude Code on your laptop

Remote client mode serves the same model gateway without launching Claude Code
on the inference server. In the TUI choose **Claude Code → Remote client**, then
connect the separate [ggrun-client](https://github.com/raketenkater/ggrun-client)
from your laptop project. Tools, files and sessions stay on the laptop; model
serving and optional reviewer/worker routing stay on the server. Any supported
HTTP(S) connection can be used; Tailscale is optional. This feature is currently
being qualified; see [setup and compatibility](docs/remote-claude.md).

## Compared with other tools

| Tool | Use it when | What ggrun adds |
|---|---|---|
| llama.cpp `llama-server --fit` | The model fits one GPU, or GPUs on similar PCIe links | Placement weighted by bandwidth, admission check, cgroup containment, backend choice, recommendations. ggrun runs llama-server; it does not replace it. |
| ik_llama.cpp `--fit` | You run ik_llama by hand (its fit is opt-in and does not adjust context) | The same automation, plus switching between mainline, ik_llama and forks by architecture |
| Ollama, LM Studio | You want a GUI, AMD support, model hot-swap, the simplest setup | Placement of experts on each GPU, ik_llama.cpp and fork backends, contained admission. No current matched speed comparison exists, so none is claimed. |
| llama-swap | You serve several models behind one endpoint | Computing each model's command line (llama-swap expects you to write it) |
| llmfit | You want to see what fits and pull it into Ollama or llama.cpp | Builds a backend if needed, writes the placement, checks admission and serves |

## Measured results: October 13, 2026 (v3.2.11)

These come from one machine: an i9-10940X (14 cores) with 212 GiB DDR4 and an RTX 4070 12 GB, RTX 3090 Ti 24 GB and RTX 3060 12 GB on PCIe Gen 3 (the 3060 runs at x8).

The smaller profiles were imposed with `--gpus`/`--cpu` and `--ram-budget` on that host. They do not emulate a smaller CPU, slower memory or other PCIe links. Each cell is a single run.

Method (October 13, 2026, v3.2.11):

- A fresh install recommended a model for each profile, which was then launched with defaults.
- Claude Code fixed small repositories, and each repository's unit tests checked the result.
- Times are wall seconds per task. A timeout counts as a failure.
- Prefill is the first long prompt of the session (~9k-21k tokens). Decode is the median over requests that generated at least 32 tokens. Context is the served context ggrun chose.

| Profile | Model (quant) | Ready | Tasks (s) | Context | Prefill / decode tok/s |
|---|---|---:|---|---:|---|
| 3 GPUs, 212 GiB | Qwen3.6-35B-A3B UD-IQ4_XS | 315 s | 38 / 38 / 43 | 262,144 | 3,645 / 80.2 |
| 3 GPUs, 212 GiB | Qwen3.8-27B UD-Q5_K_S | 121 s | 116 / 82 / 98 | 262,144 | 1,391 / 30.7 |
| 3 GPUs, 212 GiB | GLM-5.3 UD-IQ2_M (222 GiB) | not run in this round: the model disk failed during download (run later, see below) | — | — | — |
| RTX 3090 Ti 24 GB + 128 GiB | Qwen3.8-27B UD-Q5_K_S | 175 s | 64 / 79 / 76 | 122,880 | 1,008 / 34.7 |
| RTX 3090 Ti 24 GB + 128 GiB | Qwen3.6-35B-A3B UD-IQ4_XS | 106 s | 23 / 20 / 1,338 | 262,144 | 3,185 / 92.5 |
| RTX 3090 Ti 24 GB + 128 GiB | GLM-5.3-Flash UD-IQ3_XXS (112 GB) | backend built (~15 min) | 337 / 272 / 414 | 628,736 | — |
| RTX 4070 12 GB + 32 GiB | Qwen3.8-27B UD-IQ2_XXS | 256 s | 231 / 137 / 85 | 76,800 | 1,119 / 31.2 |
| RTX 4070 12 GB + 32 GiB | Gemma 4 26B A4B IQ4_XS | 225 s | 161 / 153 / 168 | 74,752 | 265 / 30.0 |
| CPU, 16 GiB | Qwen3.6-35B-A3B UD-IQ2_XXS | 206 s | 323 / 147 / **timeout** | 108,544 | 127 / 13.1 |
| CPU, 16 GiB | Qwen3.8-27B UD-IQ2_XXS | 468 s | 2,613 / 1,450 / 1,206 | 131,072 | 16 / 3.1 |

No other runtime was measured in this round, so the table makes no comparison. Relaunch times, the long task and Claude Code mode are in [docs/launch-performance.md](docs/launch-performance.md#agent-workload-results--2026-10-13-v3211).

## Large models on one host

All rows used the full host above (3 GPUs, 212 GiB), Linux, models on an NTFS HDD and `--claude-code`. They ran on `main` after v3.2.12, which is not yet in a release. Each row is one run. Tasks are Claude Code repository fixes checked by the repositories' unchanged unit tests. Prefill and decode here are medians from the backend log over the session's requests.

| Model (quant, size) | Date, build | Ready | Tasks (s) | Context | Prefill / decode tok/s |
|---|---|---:|---|---:|---|
| GLM-5.3 UD-IQ2_M, 222 GiB, `--mmap` | 2026-10-28, v3.2.12-70-g4925fd7 | 3,372 s | 1,964 / 876 / 1,491, all correct | 185,344 (2 slots) | — |
| same, clean relaunch | 2026-10-28, v3.2.12-70-g4925fd7 | 3,204 s | 4,582 (11 tests), correct | 185,344 (2 slots) | 4.6 / 3.5 |
| MiniMax-M3.1-Flash UD-IQ3_XXS, 149 GiB | 2026-10-28, v3.2.12-70-g4925fd7, llama.cpp PR #31192 backend | 2,112 s | 1,961 / 2,223 / 2,621, all correct | 361,472 planned (4 slots) | 9.7 / 2.4 (not representative) |
| DeepSeek V4.1 Flash IQ2_XS, 366 GiB, paged lookup tier, cold | 2026-10-23, v3.2.12-23-ge546a94 | 3,003 s | first task timed out at 3,600 s; the rest skipped | 1,048,576 (4 slots) | see below |
| same, relaunch | 2026-10-23, v3.2.12-23-ge546a94 | 1,621 s | 6,521 (11 tests), correct | 1,048,576 (4 slots) | see below |

- **GLM-5.3 (mapped experts).** At 222 GiB the model is larger than the 212 GiB of RAM, so its experts stay mapped from disk. Before #99, on 2026-10-24, `memory.high` throttled the page cache that held those experts. The first task timed out at 3,600 s, with prefill at 0.13-0.44 tok/s and decode at 0.32 tok/s. Since #102, startup probes wait for idle slots, and a slow canary marks the profile degraded instead of rejecting the launch.
- **MiniMax-M3.1-Flash.** It needs llama.cpp PR #31192, which was unmerged on 2026-10-28. The PR is registered with `ggrun backend add` and selected with `--backend`. That backend put its KV cache on the CPU and left the RTX 3090 Ti unused, so its speed says little about the model.
- **DeepSeek V4.1 Flash (paged lookup tables).** It has n-gram lookup tables (Engram), 195 GiB of the 366 GiB. ggrun keeps those paged from disk while the execution weights stay resident.
  - Prefill of new text ran at ~3.2-4.0 tok/s, limited by random reads from the HDD. Repeated prompts ran at ~16 tok/s.
  - Decode was 1.6-8.2 tok/s on 8-token test replies. Agent-length decode was not measured.
  - In a manual prefetch experiment, a relaunch first read back the previous session's hot table pages (1,500 s). A 4-test task then passed in 1,285 s.
  - It needs a manually registered smalinin/llama.cpp build at commit `425badfe`. GGUFs converted by upstream llama.cpp are untested.

A smaller new model went through the same download-to-agent flow. Nanbeige4.3-33B Q8_0 (bartowski, 33.6 GB) was downloaded with `ggrun download` on 2026-10-27 (v3.2.12-26-g101af89). It was ready in 251 s and passed tasks in 174 / 132 / 700 s, all correct. The downloader picked the largest quant that fits VRAM, which left 34,816 tokens of context (1 slot, KV q4_0).

`--mmap` and the paged lookup tier are Linux only. Details: GLM-5.3 and DeepSeek in [docs/large-model-mmap-evidence-20261019.md](docs/large-model-mmap-evidence-20261019.md); MiniMax-M3.1-Flash and Nanbeige4.3 under "NEW MODELS" in [docs/core-standard-launch-todos.md](docs/core-standard-launch-todos.md).

## Platforms

| Platform | Status |
|---|---|
| Linux + NVIDIA | Most tested. ik_llama.cpp CUDA or mainline CUDA, cgroup containment (needs systemd), bandwidth profile. GPUs below compute capability 7.5 get Vulkan by default. |
| Linux AMD / Intel | Mainline llama.cpp through Vulkan. No ROCm. AMD APU unified memory is broken ([#49](https://github.com/raketenkater/ggrun/issues/49)). |
| CPU only | Works; the CPU rows above show what that means for agent tasks. |
| Windows | Installer and CPU serving are verified in CI. CUDA comes from upstream llama.cpp prebuilts or a source build, not a ggrun bundle. GPU serving is unverified. There is no memory containment (the backend is stopped when ggrun exits), and fork builds are disabled. |
| Docker | `docker/` has CPU, CUDA and Vulkan images. They set `GGRUN_SCOPE_MODE=none`: no cgroup containment, but pre-launch admission still applies; use `docker run --memory` as the hard limit. No image is built in CI, and the GPU images have not been verified in a container. |

## Limitations

- All live evidence comes from the one rig above, mostly single runs.
- Linux launches need systemd (a user session or root) unless `GGRUN_SCOPE_MODE=none` is set, which drops memory containment.
- Fresh measurements against `llama-server --fit` and Ollama have not been run. Upstream expert GPU cache options are not planned for or compared yet.
- `--calibrate auto` runs a live comparison with a 20-minute budget. Models that take ~20 minutes to load never get a measured improvement. On DeepSeek V4.1 the auto budget skipped its own best candidate (`-ub 128`). That candidate later measured +29% warm prefill (20.1-20.7 vs 15.6-16.0 tok/s, short samples, no task run).
- Large models from an HDD are slow to start (1,621-3,372 s to ready in the table above) and slow to prefill cold.
- The downloader picks the largest quant that fits VRAM without reserving context. Nanbeige4.3-33B Q8_0 got 34,816 tokens on 1 slot.
- MiniMax-M3.1-Flash needs an unmerged llama.cpp PR (#31192) registered as a custom backend. On that run the backend did not follow ggrun's plan (KV cache on CPU, RTX 3090 Ti unused), and admission did not catch it.
- AI Tune gains are marginal, and speculative decoding was slower on code in the published test. Both are opt-in.
- Claude Code is a separate install, and the local model's quality limits how well tool use works.

## Useful commands

```bash
ggrun model.gguf --dry-run                 # print the backend command, do not launch
ggrun dry-run model.gguf --emit-server-argv-json
ggrun memory-probe model.gguf --json       # per-device memory plan, no serving
ggrun status model.gguf --json             # cached optimizer decision
ggrun model.gguf --benchmark               # load, measure, exit
ggrun model.gguf --gpus 0,1 --ram-headroom 8G --vram-headroom 2G
ggrun download <repo/name> --quant <name> --yes
ggrun models list | browse | path | rm
ggrun backend list | add | register | update | rollback | remove
ggrun claude list | resume                 # recorded Claude Code sessions
ggrun update                               # ggrun and backends, with rollback
```

Every option is listed in [docs/usage.md](docs/usage.md).

## Documentation

[Install](docs/install.md) ·
[Getting started](docs/getting-started.md) ·
[Usage](docs/usage.md) ·
[Troubleshooting](docs/troubleshooting.md) ·
[Model recommendations](docs/model-recommendations.md) ·
[Fork backends](docs/fork-backends.md) ·
[Benchmarks](docs/launch-performance.md#agent-workload-results--2026-10-13-v3211) ·
[Large-model evidence](docs/large-model-mmap-evidence-20261019.md) ·
[Docker](docker/README.md) ·
[Architecture](docs/architecture.md) ·
[Changelog](CHANGELOG.md) ·
[Contributing](CONTRIBUTING.md)

The canonical repository is `github.com/raketenkater/ggrun`. Install only with the commands above.

## License

MIT
