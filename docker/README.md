# Docker

Container images for ggrun. Each image bundles the Go launcher, the Python
helper tools, and a `llama-server` backend built from a pinned llama.cpp commit
(the same one the release bundles use). Build from the repo root so the build
context includes `go/` and `tools/`.

| Image | Backend | Host requirement |
|---|---|---|
| `Dockerfile.cpu` | CPU | none |
| `Dockerfile.cuda` | NVIDIA CUDA | [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| `Dockerfile.vulkan` | Vulkan (AMD / Intel / NVIDIA) | `/dev/dri` (AMD/Intel) or NVIDIA Container Toolkit |

## Build

```bash
# from the repository root
docker build -f docker/Dockerfile.cpu    -t ggrun:cpu    .
docker build -f docker/Dockerfile.cuda   -t ggrun:cuda   .
docker build -f docker/Dockerfile.vulkan -t ggrun:vulkan .
```

Build args (optional): `LLAMA_CPP_REF` pins the backend commit; the CUDA image
also takes `CUDA_IMAGE` (default `12.4.1`) and `CUDA_ARCHITECTURES`
(default `75;80;86;89;90`).

## Run

A model path (inside the container) or a Hugging Face repo is passed as the
command — the same arguments `ggrun` takes on the host.

```bash
# CPU, local model
docker run --rm -p 8081:8081 -v ~/ai_models:/models \
  ggrun:cpu /models/model.gguf

# NVIDIA CUDA
docker run --rm --gpus all -p 8081:8081 -v ~/ai_models:/models \
  ggrun:cuda /models/model.gguf

# Vulkan on AMD/Intel
docker run --rm --device /dev/dri -p 8081:8081 -v ~/ai_models:/models \
  ggrun:vulkan /models/model.gguf

# Download a quant from Hugging Face into the mounted models volume, then serve
docker run --rm --gpus all -p 8081:8081 -v ~/ai_models:/models \
  ggrun:cuda unsloth/Qwen3.6-27B-GGUF --download
```

The OpenAI-compatible API is then on `http://localhost:8081/v1`.

Volumes:
- `/models` — your GGUF files (and downloaded vision projectors / quants)
- `/cache` — AI Tune and probe cache (persist it to keep tuned configs)

## Full stack with Open WebUI

```bash
cd docker
MODEL=/models/your-model.gguf docker compose up
# Open WebUI → http://localhost:3000, wired to ggrun's API
```

## Notes

- GPU placement, KV/batch sizing, and AI Tune work the same as on the host —
  the launcher detects the GPUs exposed to the container.
- These images ship the **mainline** llama.cpp backend. The faster
  ik_llama.cpp CUDA path is a source build (see the top-level README); it is not
  bundled in the container by default.
- Set `LLM_COMMUNITY_TUNES=off` to disable the shared tune-config lookup.
