# Releasing llm-server

llm-server supports Linux, macOS, and Windows through WSL2. Native Windows
packages are intentionally not produced.

## Automated assets

Pushing a `v*` tag runs `.github/workflows/release.yml` and publishes:

- `llm-server-linux-x86_64-cpu.tar.gz`
- `llm-server-linux-x86_64-vulkan.tar.gz`
- `llm-server-macos-arm64-metal.tar.gz`
- `SHA256SUMS`

The installer looks for a matching release asset first, then falls back to a
source build. Checksums are published with each tagged release.

## CUDA assets

CUDA/ik_llama.cpp bundles are not built on generic GitHub-hosted runners. Build
them on a CUDA-capable Linux host with the target driver/toolkit compatibility
you want to support:

```bash
git clone https://github.com/ikawrakow/ik_llama.cpp.git ~/ik_llama.cpp
cmake -S ~/ik_llama.cpp -B ~/ik_llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build ~/ik_llama.cpp/build --config Release -j"$(nproc)" -t llama-server

scripts/package-release.sh \
  llm-server-linux-x86_64-cuda.tar.gz \
  ~/ik_llama.cpp/build/bin/llama-server \
  dist
```

Attach `dist/llm-server-linux-x86_64-cuda.tar.gz` to the GitHub release if you
want CUDA installs to use a prebuilt bundle. Without that asset, the installer
builds ik_llama.cpp from source.


## Public release gate

Before calling a tag stable, run the Go launcher against this matrix. A release
candidate can ship with fewer machines, but the public notes must say exactly
which rows have passed.

| Target | Required checks |
|---|---|
| Linux CPU-only | install, `detect`, dry-run, one short benchmark |
| Linux NVIDIA CUDA / ik_llama.cpp | install/build, `--spec mtp`, `--spec ngram`, `--ai-tune`, one benchmark |
| Linux Vulkan NVIDIA | install/build, `--spec auto`, `--spec ngram`, one benchmark |
| Linux Vulkan AMD or Intel | install/build, dry-run, one benchmark, no CUDA assumptions |
| macOS arm64 Metal | install/build, dry-run, one benchmark |
| WSL2 NVIDIA | install, backend detection, dry-run, one benchmark |
| No supported GPU | installer falls back cleanly to CPU bundle or source build |
| Missing backend tools | installer prints the missing package/tool and exits without partial config corruption |
| Go updater / latest release | `llm-server --update`, `llm-server update`, latest-release check, backend rebuild, smoke test, rollback path |

For speculative decoding, verify these commands before release notes claim
support:

```bash
llm-server model.gguf --dry-run --spec off
llm-server model.gguf --dry-run --spec auto
llm-server model.gguf --dry-run --spec mtp
llm-server model.gguf --dry-run --spec eagle3
llm-server model.gguf --dry-run --spec draft
llm-server model.gguf --dry-run --spec ngram
llm-server model.gguf --dry-run --spec ngram-mod
llm-server model.gguf --dry-run --spec ngram-k4v
llm-server model.gguf --dry-run --spec mtp
```

Expected policy:

- `off` emits no speculative flags.
- `auto` prefers MTP when the model has NextN/MTP layers, then EAGLE-3 or a
  validated draft model, and otherwise emits no speculative flags.
- `eagle3` only emits when the backend advertises EAGLE-3 and a matching
  speculator is available.
- `draft` only emits when a validated compatible draft model is available locally or from Hugging Face GGUF drafter search.
- `ngram` uses the broadly compatible ngram map-k dialect and is explicit only.
- `ngram-mod` and `ngram-k4v` only emit when the selected backend advertises the
  matching flags in `llama-server --help`; otherwise they fall back safely.
- `mtp` emits IK flags for ik_llama.cpp, emits mainline `draft-mtp` only when the
  backend advertises it, and otherwise skips with a warning.

## Performance evidence for release notes

Release notes should include raw numbers, not only percentages. For every tested
model/backend row, record:

- model path or HuggingFace repo, quant, and file size
- CPU, RAM, GPU model, VRAM, driver, backend commit/version
- backend mode: CUDA/IK, Vulkan/mainline, Metal, or CPU
- context size, batch, ubatch, KV types, parallel slots, spec mode
- prompt-processing tok/s, generation tok/s, accepted speculative tokens when
  available, and output sanity check result
- baseline raw llama-server command, llm-server heuristic command, and AI Tune
  winner
