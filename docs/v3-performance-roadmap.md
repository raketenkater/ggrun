# v3 Performance Roadmap

This document tracks the performance work that matters for the Go v3 line.
The core product promise remains: performance, stability, and output quality
for GGUF serving across consumer hardware.

## Current local LLM direction

- llama.cpp speculative decoding has expanded beyond simple draft models. Current
  docs list `draft-simple`, `draft-mtp`, `ngram-cache`, `ngram-simple`,
  `ngram-map-k`, `ngram-map-k4v`, and `ngram-mod`, with per-implementation
  flags and runtime statistics for accepted draft tokens:
  https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- vLLM now documents GGUF support, but still labels it highly experimental and
  under-optimized. That keeps llm-server's GGUF-first positioning valid:
  https://docs.vllm.ai/en/v0.18.1/features/quantization/gguf/
- The serving stacks chasing maximum throughput are focusing on EAGLE-style
  speculative decoding, prefix/KV caching, chunked prefill, CUDA graphs, and
  disaggregated prefill/decode. For example, vLLM's P-EAGLE work reports
  parallel speculative decoding gains over EAGLE-3:
  https://vllm.ai/blog/2026-03-13-p-eagle
- SGLang's current speculative decoding docs recommend EAGLE-3 for best
  speed/quality, EAGLE-2 for broader compatibility, and adaptive/spec-v2 paths
  where request acceptance varies:
  https://docs.sglang.ai/advanced_features/speculative_decoding.html
- MTP is now a real model architecture trend, not only a serving trick. The
  DeepSeek-V3 report describes multi-token prediction as a training objective
  that can also enable future-token prediction at inference time:
  https://arxiv.org/abs/2412.19437


## Scheduled v3 performance work

### P0: runtime safety before speed

- Done in Go launcher: cached AI-tune configs are now checked against current
  runtime VRAM headroom before memory-expanding overrides are applied. This
  protects flags such as larger `-b`, larger `-ub`, higher KV precision, higher
  `--parallel`, and speculative draft-depth changes from being reused when the
  GPU is already under memory pressure. CPU/thread-only tune overrides still
  apply.

### P1: workload-aware AI Tune profiles

Add one `--ai-tune` surface, but make the measured workload explicit in the
cache key and benchmark artifact. Planned profiles:

- `chat`: short interactive responses, optimize latency and no short-completion
  artifacts.
- `long`: normal long-form assistant output, optimize stable serving throughput.
- `code`: deterministic/code continuation, eligible for ngram speculation.
- `repeat-spec`: structured/repetitive continuation, used to measure the upper
  bound of self-speculation.
- `moe`: MoE-specific prompt and timeout profile, with lower-risk memory probes.

The tune cache key should include profile, backend, model size/shards, hardware
hash, vision mode, and spec mode. Old caches without a profile should be treated
as `long`/legacy and should not auto-enable spec.

### P2: measured speculative gating

Spec mode should be selected only when it wins for the active profile. A valid
spec result must have:

- no benchmark errors and no short completions,
- accepted draft tokens reported by the backend/API,
- generation tok/s improvement over baseline above the noise floor,
- output sanity checks passing for the profile.

Current measured policy from local data:

- `repeat-spec` / code continuation: allow `ngram-mod` when it clears the floor.
- normal `long` parallel serving: keep spec off unless a fresh local profile run
  proves otherwise.
- MoE: keep spec off by default; require explicit forced testing or a proven
  MoE profile win.

### P3: MoE speed path

The next MoE speed gains are likely in scheduling and memory movement, not in
blind batch increases. Work items:

- Benchmark MoE serving with `parallel=2/4` only when total context preserves
  the same per-slot context. Lowering effective context is not an optimization.
- Add MoE-specific AI-tune candidates for `--n-cpu-moe`, `--defer-experts`,
  pinned memory on/off, `--no-mmap`, smaller batch/ubatch, and expert/layer
  placement variants. Automatic MoE tune candidates must not change context or
  speculative decoding mode.
- Record host pinned-memory allocation time, GPU split, CPU expert count, and
  prompt/decode separately.
- Prefer candidates that reduce CPU/GPU expert traffic or improve concurrent
  serving; reject candidates that only improve prefill while hurting decode.

### P4: KV-cache and prefix-cache direction

Do not claim vLLM-style KV cache management until the backend supports it. vLLM
gets major throughput from a KV cache manager built around fixed-size blocks,
block tables, reference counts, a free queue, and hash-based prefix reuse.
llm-server can port the product behavior in stages:

1. Frontend prompt-prefix cache: hash tokenized prompts/templates and route
   repeated prefixes to long-lived slots when llama.cpp exposes reliable session
   or prefix-cache reuse.
2. Scheduler policy: choose `--parallel`, total context, and request admission
   from available KV budget instead of a static user number.
3. Backend integration: expose llama.cpp prefix-cache/session stats in benchmark
   artifacts and tune decisions.
4. Upstream/backend work: true paged KV needs attention kernels and backend KV
   allocation changes; this belongs in llama.cpp/ik_llama.cpp, with llm-server
   selecting and measuring it once available.

Primary references:

- vLLM prefix caching design: https://docs.vllm.ai/en/stable/design/prefix_caching/
- vLLM paged-attention kernel note: https://docs.vllm.ai/en/v0.21.0/design/paged_attention/
- PagedAttention paper: https://arxiv.org/abs/2309.06180

## What v3 should harness now

1. Backend-aware speculative decoding

   Keep `--spec off` as the default. Make `--spec auto` useful only when there
   is a real learned/spec-head path. The current priority is:

   - MTP when the model has NextN/MTP layers and the backend supports it
   - EAGLE-3 when a matching speculator is available and the backend advertises it
   - validated local/downloaded draft model found by same-folder lookup or Hugging Face GGUF drafter search
   - off; do not fall back to ngram in auto mode

2. Benchmark speculative acceptance, not just tok/s

   Speculative decoding can increase raw server work while lowering end-user
   latency only when enough draft tokens are accepted. Benchmark output should
   parse backend logs for:

   - generated draft tokens
   - accepted draft tokens
   - acceptance rate
   - final generation tok/s
   - output sanity result

3. Make AI Tune spec-aware

   AI Tune should treat spec mode as a controlled candidate family, not random
   JSON from the model. Candidate sets should include:

   - baseline no spec
   - MTP for MTP-capable models
   - EAGLE-3 when a matching speculator is available
   - validated draft model
   - explicit `ngram-map-k` / `ngram-mod` / `ngram-map-k4v` only for repeated/code workloads

4. MoE-specific path

   MoE should stay conservative by default. Enable speculative paths for MoE
   only when forced or when a benchmark proves a gain, because prior local tests
   showed large MoE configs can lose speed or trigger memory pressure.

5. Release-grade hardware data

   The next public post should report raw numbers across backends:

   - plain `llama-server`
   - llm-server heuristic
   - llm-server `--ai-tune`
   - `--spec off`
   - best measured speculative mode

   Each row needs model, quant, context, backend commit/version, GPU/driver,
   prompt-processing tok/s, generation tok/s, and accepted speculative tokens.
