# Performance

llm-server performance claims should be reproducible. Record the model, backend,
hardware, context, flags, and benchmark artifacts for every public number.

## Benchmark Scripts

Raw backend versus Go v3:

```bash
scripts/bench-v3-comparison.sh model.gguf   --server-bin /path/to/llama-server   --ctx-size 32768   --rounds 3   --max-tokens 512
```

Optional historical Bash v2 comparison:

```bash
scripts/bench-v3-comparison.sh model.gguf   --server-bin /path/to/llama-server   --bash-bin ~/.local/bin/llm-server-bash
```

Throughput serving benchmark:

```bash
scripts/bench-v3-throughput.sh model.gguf   --server-bin /path/to/llama-server   --parallel 4   --concurrency 8   --requests 24
```

Generated benchmark directories under `benchmarks/` are ignored. Commit only
curated summaries that include the exact command and hardware description.

## Required Fields

For each published result, include:

- model name, quant, file size, and source repo when applicable
- CPU, RAM, GPU model, VRAM, driver, and operating system
- backend name and commit/version
- context size, batch, ubatch, KV types, parallel slots, and spec mode
- prompt-processing tok/s and generation tok/s
- speculative draft tokens and accepted tokens when spec is enabled
- output sanity result
- raw backend command and llm-server command

## Current Local Findings

On the local Qwen3.6 27B Q5_K_M test with ik_llama.cpp, 32k context, one slot,
and 512 generated tokens per request:

| Mode | Profile | Median generation tok/s | Draft accepted | Result |
|---|---:|---:|---:|---|
| `--spec off` | structured/spec | 40.31 | 0/0 | baseline |
| `--spec auto` | structured/spec | 278.06 | 2367/2367 | faster |
| `--spec off` | code | 38.42 | 0/0 | baseline |
| `--spec auto` | code | 22.71 | 1161/3208 | slower |

Artifact directory from that run:

```text
benchmarks/spec-live-warm-20260603T094706Z/
```

The result is important for product policy: speculative decoding must be
measured or workload-gated. A compatible draft model can be a major win for
predictable structured continuations and a loss for code-like output with low
acceptance.

## AI Tune Policy

AI Tune candidates must preserve safety first:

- do not reduce context size to win a benchmark
- do not reuse memory-expanding cached flags when current VRAM headroom is lower
- do not change MoE context or speculative mode implicitly
- reject crashes, short completions, and invalid backend flags
- cache results by model, hardware, backend, and relevant runtime mode

## MoE Direction

MoE speed work should focus on memory movement and scheduling, not blind batch
increases. Useful measurements include:

- GPU split and expert placement
- CPU expert count and host memory pressure
- prompt-processing and decode tok/s separately
- effect of mmap, deferred experts, batch, ubatch, and continuous batching
- fixed per-slot context when testing `--parallel`

## KV Cache Direction

llm-server should not claim vLLM-style paged KV behavior until the backend
supports it. The launcher can still improve product behavior by measuring KV
budget, selecting `--parallel` conservatively, and reporting prefix/session stats
when llama.cpp exposes them.
