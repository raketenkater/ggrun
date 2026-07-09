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
ggrun model.gguf --vram-headroom 2G   # leave 2 GB of VRAM free for other apps
ggrun model.gguf --ram-headroom 8G    # leave 8 GB of system RAM free for other apps
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
and stores the best successful result in the local cache. Because it re-measures against
whatever llama.cpp / ik_llama.cpp build you currently have, it keeps your launch flags in
step with the backends as they change upstream, instead of you tracking new flags and
defaults by hand. The served model can propose candidate flags, but the launcher validates
them against backend help, memory headroom, crash behavior, and benchmark results before a
cache entry is reused. A 1% noise floor guards against replacing a good baseline with
single-run noise.

AI Tune only changes performance knobs (batch, microbatch, threads, flash attention,
mmap/mlock, defrag, speculative decoding). It never changes anything that affects output
quality — KV-cache quantization, context size, and `--parallel` are user-owned and left
exactly as you set them, including in cached and community-shared tunes.

See [launch-performance.md](launch-performance.md) for the benchmark tables and method.

## Speculative decoding

`--spec auto` only enables a validated path:

1. MTP when the target GGUF has NextN/MTP metadata and the backend supports it.
2. EAGLE-3 when a matching speculator is available and the backend advertises it.
3. A compatible draft GGUF found locally or through Hugging Face search.
4. Off when no validated path exists.

Ngram modes are explicit because they are workload-sensitive. See
[speculative-decoding.md](speculative-decoding.md).

## Use with Claude Code

ggrun serves llama.cpp's native Anthropic `/v1/messages` endpoint (`--jinja` on for
tool use), so Claude Code talks to a local model directly — no proxy.

```bash
ggrun model.gguf --claude-code   # serve, then launch Claude Code wired to it
```

If the `claude` CLI is on your PATH, ggrun starts the server and drops you straight
into Claude Code; on exit it stops the server. (In the TUI: open a model with Enter,
toggle **[x] Claude Code**, launch.) If `claude` isn't installed, ggrun prints the
env to run it yourself in another terminal:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8081 ANTHROPIC_AUTH_TOKEN=ggrun
export ANTHROPIC_MODEL=local ANTHROPIC_SMALL_FAST_MODEL=local
export ANTHROPIC_DEFAULT_HAIKU_MODEL=local ANTHROPIC_DEFAULT_SONNET_MODEL=local ANTHROPIC_DEFAULT_OPUS_MODEL=local
export API_TIMEOUT_MS=1800000              # let queued fan-out/subagent requests finish, not cancel
export API_FORCE_IDLE_TIMEOUT=0            # local PP can exceed Claude Code's stream-idle watchdog
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=90  # compact early to fit the real per-slot window (ggrun computes this)
claude --disallowedTools WebSearch
```

All five tiers point at `local` on purpose: Claude Code routes background work and
the command-safety (auto-permission) classifier through the haiku/sonnet/opus
aliases, so without the overrides those calls leave for `api.anthropic.com` and fail.

- **Thinking is on** — a normal launch never passes `--reasoning off` (benchmark-only).
- **Context fits the slot.** `--parallel` splits `--ctx-size` across sequence slots,
  so each request only sees `ctx ÷ parallel` (e.g. 262k at V4 train max `--ctx-size 1048576 --parallel 4`).
  Behind a custom base URL Claude Code assumes a 200k window and won't auto-compact in
  time, overflowing the slot (a hard fail with `--no-context-shift`). ggrun derives
  `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` from the real slot so compaction triggers early
  enough; subagents and workflow agents inherit it. A value you set yourself wins.
- **Wide fan-out** (subagents, workflows) queues behind the GPU; `API_TIMEOUT_MS` is
  raised so queued requests wait for a slot instead of cancelling. `API_FORCE_IDLE_TIMEOUT=0`
  disables Claude Code's separate stream-idle watchdog, which can fire while llama.cpp is
  still prompt-processing a very large request and has not streamed a first token yet.
- **Anti-loop sampling.** The Anthropic API has no repetition-penalty fields and the
  client only sends temperature, so ggrun sets server-side defaults in claude-code
  mode (`--presence-penalty 1.0 --repeat-penalty 1.05 --repeat-last-n 512 --top-k 20
  --top-p 0.95 --min-p 0`) — quantized thinking models loop endlessly without them.
  Pass any of these flags yourself (after `--`) and your value wins.
- **Web research:** the built-in WebSearch runs on Anthropic's servers and is hidden
  on a non-first-party endpoint, so ggrun disables it and auto-wires a no-key
  DuckDuckGo search MCP (`mcp__ddg-search__search`) when `uvx` is installed —
  `--claude-code` does this for you. Prefer another provider? Add it with
  `claude mcp add …` (it runs alongside `ddg-search`), or launch `claude` yourself
  from the printed recipe and drop/replace the `--mcp-config` line.

Quality depends on the local model: pick a tool-capable coder, and keep one
llama-server in mind for wide agent fan-out (it serializes). Best for single-agent,
scoped, or offline/private work.
