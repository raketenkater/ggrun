# Usage

With no command, `ggrun` opens the interactive TUI. Otherwise it takes a model
(local path or Hugging Face repo) plus flags.

The TUI starts without a network prompt, detects each GGUF architecture from
metadata, and uses the same placement planner as the CLI. Use `/` to search a
large model directory and `u` when you explicitly want to update ggrun and its
backends.

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
ggrun memory-probe model.gguf --json  # measure the selected backend, then stop
ggrun model.gguf --ctx-size 32768
ggrun model.gguf --kv-quality auto
ggrun model.gguf --kv-quality q5_1
ggrun model.gguf --kv-placement gpu

# First-launch placement calibration
ggrun model.gguf --calibrate auto   # default: measure alternatives once for small models
ggrun model.gguf --calibrate on     # force calibration even on a large MoE
ggrun model.gguf --calibrate off    # never calibrate; always serve the estimated placement

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
ggrun model.gguf --spec dflash
ggrun model.gguf --spec eagle3
ggrun model.gguf --spec draft
ggrun model.gguf --spec ngram-mod

# Maintenance
ggrun --update
ggrun model.gguf --benchmark
ggrun model.gguf --dry-run

# Model storage
ggrun models list
ggrun models browse              # hardware-matched curated downloads
ggrun models path
ggrun models rm model.gguf
ggrun models rm model.gguf --yes
```

Unknown flags are passed through to `llama-server`, so upstream options remain available
without wrapper changes.

## Contained memory probes

`ggrun memory-probe <model> --json` runs the same bounded memory fixed-point
used before serving, prints the final per-device model/context/compute and
unaccounted allocator bytes, and stops. A matching `llama-fit-params` is used
when the selected backend ships one. Other llama.cpp-style forks are measured
through ggrun's Linux CUDA allocation firewall inside a cgroup v2 scope.

If the backend does not advertise an allocation-only `--dry-run`, ggrun asks
before performing a contained full-load probe. Non-interactive use must pass
`--allow-live-memory-probe`. Incomplete guard coverage can be used only for that
explicitly approved run and is never written as reusable verified evidence.

The configured `ram_limit_percent` (95 by default), `--ram-budget`, and
`--ram-headroom` determine the backend cgroup's `MemoryHigh`/`MemoryMax` limit.
CUDA pinned host allocations are disabled during probes. A host-memory breach
kills the backend scope, not the rest of the server.

### KV cache types

`--kv-quality auto` is the default and lets ggrun choose a model-aware safe
cache type. For ordinary models it currently starts from the `mid`/`q8_0`
quality tier; for architectures with known correctness constraints, ggrun may
force a safer type such as `f16`.

`--kv-quality` also accepts friendly `low` (`q4_0`), `mid` (`q8_0`), and
`high` (`f16`) presets, or an exact supported llama.cpp type: `f32`, `f16`,
`bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, or `q5_1`.

Use an exact type when the memory/quality trade-off matters, for example
`--kv-quality q5_1`. ggrun uses that same type for its memory plan and emits it
for both K and V caches. The equivalent upstream spelling is accepted too:

```bash
ggrun model.gguf --cache-type-k q5_1 --cache-type-v q5_1
```

K and V must currently use the same type. ggrun rejects a mixed pair instead
of producing a placement plan with the wrong KV-memory estimate.

## First-launch placement calibration

ggrun's placement planner computes where a model's weights, experts, and KV
cache live from the GGUF and real measured VRAM — but on a multi-GPU host more
than one placement usually *fits*, and the estimate can only guess which is
*fastest* on your exact topology. First-launch calibration closes that gap:
on the first launch of a model + hardware + workload shape, ggrun measures the
real decode throughput of each alternative placement (for a MoE, KV-on-GPU vs
KV-on-CPU with the freed VRAM going to more GPU experts; for multi-GPU dense,
the inverted split), keeps the fastest, and caches the decision.

Later launches with the same scope apply the cached winner directly — no
re-measurement, no restart. Change the model, backend build, GPU set, context,
slot count, or workload profile and the scope key changes, so a decision is
never applied to a launch it didn't measure.

Calibration restarts the server once per candidate, so `auto` (the default)
only runs it for models small enough to restart cheaply (under ~40 GB). Force
it on a large MoE with `--calibrate on`, or disable it entirely with
`--calibrate off`. A candidate that fails to start or OOMs is skipped; if no
alternative beats the estimated default by a meaningful margin, the default
stands — calibration can never leave you on a slower placement than the one
the planner chose.

## Model storage

`ggrun models list` shows GGUF files under the configured model directory and
groups split GGUFs as one model. `ggrun models browse` opens the same curated,
hardware-aware download list as `ggrun recommend`. `ggrun models path` prints that directory.
Use `ggrun models rm <model.gguf>` to remove a listed model; it asks before
deleting and only operates inside the configured model directory. Add `--yes`
for scripts or set `LLM_ASSUME_YES=true` in a non-interactive environment.

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

1. Embedded NextN/MTP or a validated same-architecture MTP-only companion.
2. A validated target-specific DFlash companion.
3. EAGLE-3 when a matching speculator is available and the backend advertises it.
4. A compatible draft GGUF found locally or through Hugging Face search.
5. Off when no validated path exists.

Ngram modes are explicit because they are workload-sensitive. See
[speculative-decoding.md](speculative-decoding.md).

## Use with Claude Code

ggrun serves llama.cpp's native Anthropic `/v1/messages` endpoint (`--jinja` on for
tool use). In Auto mode a loopback-only ggrun router sends normal coding turns to
the selected model and hidden permission reviews to a small local reviewer.

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
export CLAUDE_CODE_EFFORT_LEVEL=xhigh       # agentic default; use max for one demanding session
export API_TIMEOUT_MS=2147483647            # maximum safe timer; no practical inference deadline
export CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS=2147483647
export CLAUDE_ENABLE_BYTE_WATCHDOG=0 CLAUDE_ENABLE_STREAM_WATCHDOG=0 API_FORCE_IDLE_TIMEOUT=0
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=262144 CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=75
claude --permission-mode auto --disallowedTools WebSearch
```

All five inference tiers point at `local` on purpose, so foreground and background
model calls cannot leave for `api.anthropic.com`.

- **Thinking is on** — a normal launch never passes `--reasoning off` (measurement-only:
  benchmark and the deterministic core `spec-test` matrix).
- **Context fits the slot.** `--parallel` splits `--ctx-size` across sequence slots,
  so each request only sees `ctx ÷ parallel`. Claude mode requests four main-model
  slots: a native 1M model gets about 256k per slot. ggrun automatically lowers the
  slot count if the selected total context would provide less than about 64k per
  slot, so the portable 128k-total fallback uses two slots. Explicit `--ctx-size`
  and `--parallel` values always win.
  Claude Code's assumed context has changed across releases and model aliases, which
  can make percentage-only overrides miss the real backend limit. ggrun exports the
  actual per-slot capacity as `CLAUDE_CODE_AUTO_COMPACT_WINDOW` and compacts at 75%,
  leaving room for a reply and tool output. Subagents and workflow agents inherit
  both values; values you set yourself win.
- **Wide fan-out** (subagents, workflows) runs up to four main-model requests at once
  and queues additional work behind those slots.
  ggrun sets the maximum safe Claude request/background-agent timers, disables both
  stream-idle watchdogs, and gives llama-server no practical socket deadline. Claude's
  Workflow tool has a separate 180-second `stallMs`; a session-only PreToolUse hook
  deterministically rewrites every `agent()` call to the maximum safe value before it
  runs. Startup, process-health, and shell-command guards remain active so a real crash
  or hung command is still visible.
- **Anti-loop sampling.** The Anthropic API has no repetition-penalty fields and the
  client only sends temperature, so ggrun sets server-side defaults in claude-code
  mode (`--presence-penalty 1.0 --repeat-penalty 1.05 --repeat-last-n 512 --top-k 40
  --top-p 0.95 --min-p 0.05`) — quantized thinking models loop endlessly without them.
  Pass any of these flags yourself (after `--`) and your value wins.
- **Compaction reuses moved prompt chunks.** On shiftable transformer contexts, when
  the backend supports it, Claude mode enables `--cache-reuse 256`. This complements
  ordinary common-prefix caching by
  shifting repeated system, tool and workflow chunks after old results are removed.
  A controlled production-cache test reduced a compacted 4,506-token prefill from
  45.1 seconds to one processed token in 0.15 seconds. Pass `--cache-reuse 0` or
  `--no-cache-prompt` explicitly to opt out. Hybrid/recurrent contexts such as native
  DeepSeek V4 cannot shift their state, so ggrun does not emit the unsupported flag;
  it instead keeps one rolling context checkpoint per slot when at least 512 MiB of
  host headroom per slot remains. This lets llama.cpp restore append-only agent turns
  without exposing the unsafe 32-checkpoint backend default.
- **Hybrid slot fairness.** Claude mode caps the logical prompt batch at 128 tokens on
  hybrid/recurrent models. Physical ubatch remains placement-derived. This prevents a
  long prefill batch from withholding decode work from the other active slot for more
  than a minute; explicit backend arguments still win.
- **Web research:** the built-in WebSearch runs on Anthropic's servers and is hidden
  on a non-first-party endpoint, so ggrun disables it and auto-wires a no-key
  DuckDuckGo MCP when `uvx` is installed. Its `search` and `fetch_content` tools
  are pre-authorized so agents can locate and read current sources without a
  permission prompt — `--claude-code` does this for you. Prefer another provider? Add it with
  `claude mcp add …` (it runs alongside `ddg-search`), or launch `claude` yourself
  from the printed recipe and drop/replace the `--mcp-config` line.
- **Auto works locally and remains fail-closed.** Claude Code sends hidden permission
  reviews to the same model ID as coding turns. ggrun detects those exact
  security-monitor requests and routes them to a pinned Qwen3.5-2B reviewer running
  locally; all other traffic stays on the selected coding model. The reviewer starts
  before placement, so its measured VRAM use is included when ggrun places the main
  model. This is Auto, not `bypassPermissions`. The first launch downloads and verifies
  the pinned ~1.3 GiB Q4_K_M GGUF and serves it with one independent 64k slot and Q8
  KV cache. GPU visibility
  is isolated to its selected physical device; if it does not fit any selected GPU,
  ggrun falls back to CPU. Override it with `GGRUN_CLAUDE_REVIEWER_MODEL=/path/model.gguf`,
  choose another permission mode with `GGRUN_CLAUDE_PERMISSION_MODE=acceptEdits`, or
  use `inherit` to preserve your global Claude setting. See Claude Code's
  [permission-mode requirements](https://code.claude.com/docs/en/permission-modes#eliminate-prompts-with-auto-mode).
- **Live local progress:** while a local request is queued, ingesting its prompt, or
  generating, ggrun adds a session-only Claude status line with the active slot,
  prompt progress bar, token counts, tok/s, active requests, and queue depth. It uses
  llama-server's structured slot/metrics endpoints and exact prompt-progress logs.
  If structured telemetry stalls during a long prefill, the backend health check
  and passive log lifecycle remain authoritative, the last request stays visible as
  `status delayed`, and endpoint polling backs off instead of creating cancellation
  pressure. Prompt contents are never stored. Existing custom Claude status lines are preserved,
  with progress shown in the terminal title instead. Set `GGRUN_CLAUDE_PROGRESS=off`
  to disable the display.

Quality depends on the local model: pick a tool-capable coder, and keep one
llama-server in mind for wide agent fan-out (it serializes). Best for single-agent,
scoped, or offline/private work.
