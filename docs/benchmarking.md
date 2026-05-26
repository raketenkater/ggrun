# Benchmarking TODO

Follow-up plan for v3 optimization after the stable MoE/KV placement fix.

## Baseline

- [x] Record current stable MiniMax-M2.7 max-context q8 GPU-KV result:
  - `--ctx-size 196608 --kv-quality mid --kv-placement auto`
  - Expected placement: 8 GPU expert layers, 54 CPU expert layers
- [x] Keep the known 64k baseline for comparison:
  - `--ctx-size 65536`, q4 KV, `--parallel 1`
  - Last observed benchmark: 17.8 tok/s prompt processing, 10.4 tok/s generation

## Placement Comparison

- [x] Benchmark max-context q8 with CPU KV:
  - `--ctx-size 196608 --kv-quality mid --kv-placement cpu`
  - Observed placement: 25 GPU expert layers, 37 CPU expert layers
- [x] Compare GPU KV vs CPU KV for short/normal chat prompts.
- [x] Compare GPU KV vs CPU KV for long-context prefill workloads.
- [x] Decide default recommendation by workload:
  - GPU KV: better once long-context decode matters.
  - CPU KV: better for short/normal prompts because more experts fit on GPU.

## Results: MiniMax-M2.7 Q3_K_XL

All max-context runs used:

- Model: `/home/mik/ai_models/UD-Q3_K_XL/MiniMax-M2.7-UD-Q3_K_XL-00001-of-00004.gguf`
- Server: `/home/mik/llama.cpp/build/bin/llama-server`
- Context: `--ctx-size 196608`
- KV quality: `--kv-quality mid` -> q8_0 K/V
- Parallel: `--parallel 1`

Short benchmark:

- GPU KV: 8 GPU expert layers, 54 CPU expert layers; 12.3 tok/s prompt processing, 8.2 tok/s generation.
- CPU KV: 25 GPU expert layers, 37 CPU expert layers; 17.3 tok/s prompt processing, 8.8 tok/s generation.

Long prefill benchmark:

- Prompt shape: generated 1024-section technical document, 34,880 prompt tokens, `max_tokens=16`.
- GPU KV: 84.4 tok/s prompt processing, 6.88 tok/s generation, 415.3s wall time.
- CPU KV: 79.9 tok/s prompt processing, 0.46 tok/s generation, 471.2s wall time.

Current read:

- CPU KV is better for normal chat/short prompts because it allows many more expert layers on GPU.
- GPU KV is better once the active context is long enough that decode attention over KV dominates.

## Tuning

- [ ] Retune around the winning placement:
  - `--parallel`
  - `-b`
  - `-ub`
  - `--n-cpu-moe` recovery path if it beats explicit `-ot`
- [ ] Cache the best config per model/context/KV placement.
- [ ] Re-run stability checks after tuning.

## UX Defaults

- [ ] Make interactive context prompt default to `fit`.
- [ ] Keep `max` available but never silently choose it when it creates major KV/expert tradeoffs.
- [ ] Prompt for KV placement only when GPU KV is infeasible, over threshold, or significantly reduces GPU expert layers.
- [ ] Keep `LLM_KV_PLACEMENT` and CLI flags as authoritative overrides.
