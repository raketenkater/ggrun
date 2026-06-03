# Architecture

llm-server is a Go launcher around upstream GGUF serving backends. The product
surface is the `llm-server` command; helper scripts exist only for packaging,
GGUF metadata extraction, and downloading.

## Main Components

- `go/cmd/llm-server`: CLI entrypoint and compatibility argument parsing.
- `go/pkg/config`: config loading, environment handling, and app-home support.
- `go/pkg/detect`: CPU, RAM, CUDA, and Vulkan detection.
- `go/pkg/gguf`: GGUF metadata parsing through the bundled parser helper.
- `go/pkg/placement`: backend flag selection, KV planning, MoE placement,
  speculative decoding, and vision projector lookup.
- `go/pkg/tune`: AI Tune benchmarking, candidate validation, and cache handling.
- `go/pkg/update`: self-update, backend update, rollback, and startup update
  checks for interactive users.
- `go/pkg/tui`: terminal UI launched by `llm-server gui` or `llm-server-gui`.

## Tool Layout

Runtime helper scripts live under `tools/`:

```text
tools/gguf/parse_gguf.py
tools/download/download_any_gguf.py
tools/models/model_index.py
```

Installers copy these helpers beside the installed Go binary so release bundles
and source installs can run without relying on repo-relative paths.

## Legacy Bash

The old Bash launcher is not the current product line. Existing installs can be
preserved as `llm-server-bash` during migration, and benchmark scripts can accept
an external Bash v2 path for before/after numbers. The repository root remains
Go-first.

## Backend Contract

llm-server does not fork llama.cpp behavior. It selects flags, starts the
backend, validates health, runs benchmarks, and records cache metadata. Unknown
launcher flags are forwarded to the backend so upstream options remain usable.
