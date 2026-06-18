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

Generated benchmark directories under `.benchmarks/` are ignored. Commit only
curated summaries that include the exact command and hardware description.

For launch-ready 4B, 27B, and MoE tables, see
[launch-performance.md](launch-performance.md).

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
.benchmarks/spec-live-warm-20260603T094706Z/
```

The result is important for product policy: speculative decoding must be
measured or workload-gated. A compatible draft model can be a major win for
predictable structured continuations and a loss for code-like output with low
acceptance.

> **Current launch numbers** (newer binary; full 4B / 27B / 122B-A10B / MiniMax-M3
> matrix at 32k) live in [launch-performance.md](launch-performance.md) and the
> README. The table below is the dated 2026-06-12 `--fit` run that used
> MiniMax-M2.7 as the MoE model — kept as a methodology record, not the headline.

## Upstream llama.cpp --fit Comparison (2026-06-12)

Upstream llama.cpp master was built from `70b54e140c90` (`vendor : update cpp-httplib to 0.47.0 (#24395)`) with CUDA (`GGML_CUDA=ON`, CUDA 13.2.51). The benchmark host was the release rig: i7-10700K, 128 GiB RAM, RTX 3090 Ti 24 GiB, RTX 3060 12 GiB, RTX 4070 12 GiB, NVIDIA driver 580.159.03.

Method: `scripts/bench-v3-comparison.sh`, 32k requested context, `long` prompt profile, median of three new upstream fit rounds. Qwen rows generated 512 tokens; the MiniMax MoE row generated 256 tokens, matching the release MoE methodology. The upstream rows passed no explicit `--n-gpu-layers` or `--tensor-split`; `llama-server --help` reports `--fit` defaulting to `on` and `--n-gpu-layers` defaulting to `auto`. The script was run with `/bin/false` as the wrapper binary so the new artifact summaries intentionally show failed `v3-go` rows instead of rerunning cached v3 measurements.

Artifacts:

- upstream fit rows: `/home/mik/llm-server/.benchmarks/fit-master-20260612/`
- existing llm-server rows: `/home/mik/llm-server/.benchmarks/fresh-20260609/` and `/home/mik/llm-server/.benchmarks/v3-release-20260610/`

| Model | Mode | Fitted/requested context | Median decode tok/s | Median prompt tok/s | Placement/result |
|---|---|---:|---:|---:|---|
| Qwen3.5 4B Q4_K_M | llama.cpp master CUDA `--fit` | 32k / 32k | 102.57 | 940.13 | `llama-fit-params`: `-c 32768 -ngl -1` |
| Qwen3.5 4B Q4_K_M | llm-server v3 IK/CUDA default | 32k / 32k | 156.21 | 2212.63 | cached release row |
| Qwen3.5 4B Q4_K_M | llm-server v3 IK/CUDA AI-tune | 32k / 32k | 183.85 | 2209.94 | cached release row |
| Qwen3.6 27B Q5_K_M | llama.cpp master CUDA `--fit` | 32k / 32k | 24.13 | 269.56 | `llama-fit-params`: `-c 32768 -ngl -1` |
| Qwen3.6 27B Q5_K_M | llm-server v3 IK/CUDA default | 32k / 32k | 37.68 | 577.79 | cached release row |
| Qwen3.6 27B Q5_K_M | llm-server v3 IK/CUDA AI-tune | 32k / 32k | 37.69 | 575.85 | cached release row |
| MiniMax M2.7 UD-Q3_K_XL | llama.cpp master CUDA `--fit` | 32k / 32k | 9.69 | 5.09 | `-ngl 63 -ts 6,12,45` plus CPU expert overrides; log warns CPU overrides with mmap may be slower |
| MiniMax M2.7 UD-Q3_K_XL | llm-server v3 IK/CUDA default | 32k / 32k | 11.25 | 32.95 | cached release row; two clean artifact rounds |

Findings:

- Upstream `--fit` successfully avoided OOM for the 27B and MoE cases and preserved the requested 32k context in all three models.
- No upstream `--fit` row beat llm-server v3 default placement on this host, so this run did not produce placement-bug issues under the regressions policy.
- The largest gap is prompt processing on the MoE row: upstream fit selected many CPU expert overrides while keeping mmap enabled, and the server warned that this combination may be slower.

### Same-Binary Placement Rows

To isolate placement from backend version, llm-server was also run with the same llama.cpp master CUDA server binary used by the raw `--fit` rows: `/tmp/llama.cpp-fit-master/build-cuda/bin/llama-server` (`70b54e1`). These rows use the same 32k requested context, `long` prompt profile, and 512 generated tokens as the raw fit rows.

Artifacts:

- `/home/mik/llm-server/.benchmarks/fit-master-20260612/4b-llmserver-mainline-master/`
- `/home/mik/llm-server/.benchmarks/fit-master-20260612/27b-llmserver-mainline-master/`

| Model | Mode | Requested context | Median decode tok/s | Median prompt tok/s | Notes |
|---|---|---:|---:|---:|---|
| Qwen3.5 4B Q4_K_M | llama.cpp master CUDA `--fit` | 32k | 102.57 | 940.13 | no explicit placement flags |
| Qwen3.5 4B Q4_K_M | llm-server + same llama.cpp master binary | 32k | 175.98 | 1768.40 | `--backend llama --server-bin ...`, default placement |
| Qwen3.6 27B Q5_K_M | llama.cpp master CUDA `--fit` | 32k | 24.13 | 269.56 | no explicit placement flags |
| Qwen3.6 27B Q5_K_M | llm-server + same llama.cpp master binary | 32k | 39.01 | 502.63 | cached tune checked; baseline won, no override flags applied |

### Ollama Same-GGUF Rows

Ollama `0.17.1-rc1` was tested through local Modelfiles using the same GGUF paths and `PARAMETER num_ctx 32768`. Server startup reported `OLLAMA_FLASH_ATTENTION=false`, `OLLAMA_KV_CACHE_TYPE` unset, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_CONTEXT_LENGTH=0`, and CUDA devices visible for the RTX 3090 Ti, RTX 4070, and RTX 3060.

Artifacts and exact Modelfiles:

- `/home/mik/llm-server/.benchmarks/ollama-20260613/`

| Model | Ollama result | Throughput row |
|---|---|---:|
| Qwen3.5 4B Q4_K_M | Imported from the local GGUF, but `/api/generate` failed to load it: `unknown model architecture: qwen35` | not recorded |
| Qwen3.6 27B Q5_K_M | Import failed while copying the 19GB GGUF into Ollama blobs: `no space left on device` on `/home/mik/.ollama/models` | not recorded |
| MiniMax M2.7 UD-Q3_K_XL | Not imported: the filesystem had only 805 MiB free after the 27B import failure, while Ollama would need a blob-store copy of the ~96GB split GGUF | not recorded |

No registry Ollama models were substituted, because that would compare different quantizations/files rather than launcher behavior on the same GGUFs.

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
