# Troubleshooting

Start with these commands before changing launch flags:

~~~bash
ggrun version
ggrun detect
ggrun probe
ggrun config show
ggrun <model.gguf> --dry-run
~~~

They show the installed build, detected hardware, currently free memory,
configuration precedence, and the exact backend command ggrun intends to run.
Do not post private paths, model files, tokens, or prompts in a public issue.

## Installation

### The installer wants to build a backend

Release bundles are used when a matching backend is available. Otherwise the
installer may need a compiler and backend dependencies. Re-run the installer
with the backend you want, or install the launcher without a backend and point
LLAMA_SERVER at an existing llama-server binary.

See [install.md](install.md) for platform commands and installer controls.

### ggrun: command not found

Open a new terminal after installation. The installer prints the directory that
must be on PATH. You can also invoke the launcher from its app home directly.

## Models and downloads

### I cannot find a model I downloaded

~~~bash
ggrun models path
ggrun models list
ggrun config show
~~~

The first command prints the active model directory. If a GGUF is on another
disk, pass its full path to ggrun or update MODEL_DIR through ggrun config edit
or the TUI.

### Model download says Python is missing

Model search and download use the bundled Python downloader. Install Python 3,
reopen the terminal, and retry. Local serving still works with GGUF files you
already downloaded.

### I want to delete a downloaded model

~~~bash
ggrun models list
ggrun models rm <model.gguf>
~~~

The remove command lists the size and shard count, asks for confirmation, and
only removes files below the configured model directory. Add --yes only for
non-interactive use.

## GPU, memory, and model loading

### No GPU is detected or the wrong backend is used

Run ggrun detect. Then choose a backend explicitly when needed:

~~~bash
ggrun --backend ik_llama <model.gguf>
ggrun --backend vulkan <model.gguf>
ggrun --backend llama <model.gguf>
~~~

Use --backend only with a backend installed on the machine. A custom build can
be selected with --server-bin /path/to/llama-server.

### The model does not fit or the server reports an OOM

First print the plan:

~~~bash
ggrun <model.gguf> --dry-run
~~~

Lower context, reserve memory used by other applications, or limit the GPU set:

~~~bash
ggrun <model.gguf> --ctx-size 32768
ggrun <model.gguf> --vram-headroom 2G --ram-headroom 8G
ggrun <model.gguf> --gpus 0,1
~~~

For large MoE models, let ggrun plan the split before adding manual
--tensor-split or tensor-override flags. The launch log records recovery
attempts and the final plan.

### The model architecture is not supported

Update the selected backend first. Some new GGUF architectures arrive in a
llama.cpp fork before mainline. ggrun can register an existing compatible
backend or install a reviewed fork recipe; see
[fork-backends.md](fork-backends.md).

## Claude Code and Ultracode workflows

### Claude Code cannot reach the local model

Launch it through ggrun:

~~~bash
ggrun <model.gguf> --claude-code
~~~

If the claude command is not installed, ggrun prints the environment needed to
start it in another terminal. Keep that terminal and the model server running
while using the workflow.

### A long request looks stuck

Local inference can spend a long time processing a large prompt before emitting
the first token. Claude mode displays request progress where the terminal
supports it. Check the backend log path printed at launch and use
ggrun claude-status for the current local status.

## Logs and support

Set LLM_LOG_DIR in ggrun config edit to keep server logs in a known location.
When opening an issue, include:

- ggrun version and operating system;
- GPU/RAM details from ggrun detect;
- model filename, quant, backend, context, and parallelism;
- the --dry-run output and the relevant error lines.

Do not include model files, API tokens, private paths, or prompt contents.
