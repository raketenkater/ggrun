# Usage

With no command, `ggrun` opens the interactive TUI. Otherwise it takes a model
(local path or Hugging Face repo) plus flags.

```bash
# Backends
ggrun --backend ik_llama model.gguf
ggrun --backend llama model.gguf
ggrun --backend vulkan model.gguf

# Placement and memory
ggrun model.gguf --gpus 0,1
ggrun model.gguf --ram-budget 90G
ggrun model.gguf --ctx-size 32768
ggrun model.gguf --kv-quality mid
ggrun model.gguf --kv-placement gpu

# Vision
ggrun model.gguf --vision
ggrun model.gguf --mmproj /path/to/mmproj.gguf

# Tuning and cached configs
ggrun model.gguf --ai-tune
ggrun model.gguf --ai-tune --retune
ggrun --show-configs
ggrun model.gguf --tune-cache ~/.cache/ggrun/tune.json

# Speculative decoding
ggrun model.gguf --spec auto
ggrun model.gguf --spec mtp
ggrun model.gguf --spec eagle3
ggrun model.gguf --spec draft
ggrun model.gguf --spec ngram-mod

# Maintenance
ggrun --update
ggrun model.gguf --benchmark
ggrun model.gguf --dry-run
```

Unknown flags are passed through to `llama-server`, so upstream options remain available
without wrapper changes.

## AI Tune

`--ai-tune` starts from the launcher heuristic, benchmarks it, tests candidate flag sets,
and stores the best successful result in the local cache. The served model can propose
candidate flags, but the launcher validates them against backend help, memory headroom,
crash behavior, and benchmark results before a cache entry is reused. A 1% noise floor
guards against replacing a good baseline with single-run noise.

See [performance.md](performance.md) for the benchmark format and artifacts.

## Speculative decoding

`--spec auto` only enables a validated path:

1. MTP when the target GGUF has NextN/MTP metadata and the backend supports it.
2. EAGLE-3 when a matching speculator is available and the backend advertises it.
3. A compatible draft GGUF found locally or through Hugging Face search.
4. Off when no validated path exists.

Ngram modes are explicit because they are workload-sensitive. See
[speculative-decoding.md](speculative-decoding.md).
