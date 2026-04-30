# Releasing llm-server

llm-server supports Linux, macOS, and Windows through WSL2. Native Windows
packages are intentionally not produced.

## Automated assets

Pushing a `v*` tag runs `.github/workflows/release.yml` and publishes:

- `llm-server-linux-x86_64-cpu.tar.gz`
- `llm-server-macos-arm64-metal.tar.gz`

The installer looks for a matching release asset first, then falls back to a
source build.

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
