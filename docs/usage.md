# Usage

With no command, `llm-server` opens the interactive TUI. Otherwise it takes a model
(local path or Hugging Face repo) plus flags.

```bash
# Backends
llm-server --backend ik_llama model.gguf
llm-server --backend llama model.gguf
llm-server --backend vulkan model.gguf

# Placement and memory
llm-server model.gguf --gpus 0,1
llm-server model.gguf --ram-budget 90G
llm-server model.gguf --ctx-size 32768
llm-server model.gguf --kv-quality mid
llm-server model.gguf --kv-placement gpu

# Vision
llm-server model.gguf --vision
llm-server model.gguf --mmproj /path/to/mmproj.gguf

# Tuning and cached configs
llm-server model.gguf --ai-tune
llm-server model.gguf --ai-tune --retune
llm-server --show-configs
llm-server model.gguf --tune-cache ~/.cache/llm-server/tune.json

# Speculative decoding
llm-server model.gguf --spec auto
llm-server model.gguf --spec mtp
llm-server model.gguf --spec eagle3
llm-server model.gguf --spec draft
llm-server model.gguf --spec ngram-mod

# Maintenance
llm-server --update
llm-server model.gguf --benchmark
llm-server model.gguf --dry-run
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
