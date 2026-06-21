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

Only `LLM_APP_HOME` and `PATH` are exported by the environment; everything else (model
dir, backend, cache, logs, llama-server path) is read from `.config/config`, so CLI and
GUI edits take effect instead of being shadowed by environment variables.

## Release bundles

Since v3, [prebuilt release bundles](https://github.com/raketenkater/ggrun/releases/latest)
(Linux CPU/Vulkan, macOS arm64 Metal, Windows x86_64 CPU) are downloaded and verified
against the published `SHA256SUMS` — no compile needed. Linux CUDA/ik_llama.cpp installs
build from source for your exact GPU architecture. Windows NVIDIA CUDA installs use a
native llama.cpp CUDA backend, either from an optional
`ggrun-windows-x86_64-cuda.zip` release asset or by building it locally.

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
