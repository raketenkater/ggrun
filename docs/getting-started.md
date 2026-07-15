# Get started in three minutes

ggrun is for serving local GGUF models with llama.cpp-compatible backends. The
easy path is the terminal UI; the CLI path below is useful when you already know
which model you want.

## 1. Install

Linux / macOS:

~~~bash
curl -fsSL https://raw.githubusercontent.com/raketenkater/ggrun/main/setup.sh | bash
~~~

Windows PowerShell:

~~~powershell
iwr -useb https://raw.githubusercontent.com/raketenkater/ggrun/main/install.ps1 | iex
~~~

The installer creates an app home containing the launcher, model directory,
configuration, cache, logs, and backend. See [install.md](install.md) for
platform-specific choices.

## 2. Let ggrun find a model that fits

~~~bash
ggrun
~~~

The TUI detects the machine and offers recommended downloads. Use the arrow keys,
press Enter to select a model, and launch it from the configuration screen.

From the CLI, see the same hardware-aware recommendations with:

~~~bash
ggrun recommend
~~~

## 3. Download and run

~~~bash
ggrun download <hugging-face-owner/repository>
ggrun models list
ggrun <model.gguf>
~~~

ggrun download chooses a quant using detected VRAM and RAM. ggrun models list
shows the downloaded GGUFs, including split GGUFs as one model. Use the name
printed by that command to launch a model.

If you already have a GGUF, pass its path directly:

~~~bash
ggrun /path/to/model.gguf
~~~

## Before a real launch

See the complete backend command without starting anything:

~~~bash
ggrun <model.gguf> --dry-run
~~~

Useful safe controls:

~~~bash
ggrun <model.gguf> --ctx-size 32768
ggrun <model.gguf> --vram-headroom 2G
ggrun <model.gguf> --ram-headroom 8G
ggrun <model.gguf> --gpus 0,1
~~~

Run ggrun config show to see the active settings and where each value came
from. Command-line flags override configuration and environment values.

## Claude Code and Ultracode workflows

With the claude CLI installed, ggrun can start Claude Code against the local
model:

~~~bash
ggrun <model.gguf> --claude-code
~~~

This enables the local Claude Code/Ultracode workflow setup: local model aliases,
parallel slots where the context supports them, request progress, local research
tools, and a separate local permission reviewer. See
[usage.md](usage.md#use-with-claude-code) for requirements and overrides.

## Need help?

Start with [troubleshooting.md](troubleshooting.md). For a bug report, include
the output of ggrun version, ggrun detect, and the relevant --dry-run command
without private paths or model files.
