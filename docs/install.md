# Install

## Recommended (self-contained app home)

Linux / macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/raketenkater/ggrun/main/setup.sh | bash
```

Native Windows from PowerShell:

```powershell
iwr -useb https://raw.githubusercontent.com/raketenkater/ggrun/main/install.ps1 | iex
```

Native Windows NVIDIA CUDA:

```powershell
iwr -useb https://raw.githubusercontent.com/raketenkater/ggrun/main/install.ps1 -OutFile install.ps1
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Backend cuda
```

From a clone:

```bash
git clone https://github.com/raketenkater/ggrun.git
cd ggrun
./setup.sh
```

## App home layout

`setup.sh` creates a clean app home under `~/ggrun`:

```text
ggrun       launcher (no arguments opens the terminal GUI)
models/          GGUF models and downloaded vision projectors
.bin/            Go binary, tools, and bundled backend when available
.config/         local config loaded by the launcher (single source of truth)
.cache/          AI Tune and model index cache
.logs/           setup and server logs
.src/            backend source/build fallback
```

Use it with:

```bash
~/ggrun/ggrun                       # interactive GUI
~/ggrun/ggrun <repo/name> --download
```

(`ggrun download <repo/name>` is the same action, the form used in getting-started.md.)

Only `LLM_APP_HOME` and `PATH` are exported by the environment; everything else (model
dir, backend, cache, logs, llama-server path) is read from `.config/config`, so CLI and
GUI edits take effect instead of being shadowed by environment variables.

New installs write `LLM_BACKEND="auto"`: each launch picks an installed backend
that loads the model's architecture and offers a build when none does. A named
backend (`llama`, `ik_llama`, `vulkan`) is treated as your explicit choice and
turns that off. Installs made before this change may still have
`LLM_BACKEND="llama"` (CPU and Windows) or a single backend name; updates keep
your config, so set it to `auto` yourself to get automatic selection.

## Release bundles

The [latest GitHub release](https://github.com/raketenkater/ggrun/releases/latest)
ships these archives, checksummed in `SHA256SUMS`:

| Platform | Assets |
|---|---|
| Linux x86_64 | CPU, Vulkan, CUDA |
| macOS arm64 | Metal |
| Windows x86_64 | CPU |

Linux `auto` first searches the machine for an existing `llama-server`
(ik_llama.cpp, llama.cpp, and forks) plus the NVIDIA driver and CUDA toolkit.
It reuses those, then installs the CUDA prebuilt and llama.cpp (Vulkan).
A downloaded CUDA binary is kept even if `--version` fails on a missing CUDA
runtime library. Source compile happens only when no CUDA ELF landed and
`nvcc` is already there. CPU is last if nothing else runs. The NVIDIA
proprietary driver is never auto-installed. Windows CUDA: `install.ps1 -Backend cuda`.

## Classic install to `~/.local/bin`

```bash
curl -fsSL https://raw.githubusercontent.com/raketenkater/ggrun/main/install.sh | bash
```

## Installer controls

```bash
LLM_INSTALL_MODE=release ./install.sh      # require a prebuilt bundle
LLM_INSTALL_MODE=build ./install.sh        # force source build
LLM_INSTALL_BACKEND=skip ./install.sh      # install launcher/tools only
LLM_INSTALL_PY_DEPS=skip ./install.sh      # skip downloader Python deps
LLM_INSTALL_PREFIX=/usr/local/bin ./install.sh
```

Existing Bash installs are treated as legacy. The installer preserves them as
`llm-server-bash` when replacing the primary `ggrun` command with the Go binary.
