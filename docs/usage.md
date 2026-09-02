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
ggrun freetoken Qwen/Qwen3.6-35B-A3B --gpu 0  # experimental separate engine

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

# Core launch optimization
ggrun model.gguf --calibrate auto   # default: bounded search on a cold scope, direct reuse afterward
ggrun model.gguf --calibrate on     # wider explicit screen; its result stays explicit-only
ggrun model.gguf --calibrate off    # serve the stable planner estimate without performance search

# Experimental resident-MoE hot-expert cache
ggrun backend feature install hot-experts --base my-model-fork
ggrun model.gguf --hot-experts auto

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

## Experimental hot-expert cache

The reviewed `hot-experts` source feature is an isolated build of the draft
llama.cpp GPU LRU cache for routed experts that otherwise execute from host
memory. It is a performance experiment, not a new model architecture. From the
TUI Settings or model config screen, set `Hot experts` to `auto` for an
opportunistic measured candidate, or `on` to require an optimizer-sized cache-on
launch. Only `auto` may restore the stable cache-free baseline after rejection.
A positive integer still requests that exact count from the CLI. On the first
confirmed launch of an MoE, ggrun offers a cancel-first, one-time build for
the model's installed architecture fork. Later launches reuse the validated
composite automatically.

The same flow is available directly:

```bash
ggrun backend feature install hot-experts --base glm5next
ggrun model.gguf --hot-experts auto
```

For a stock/mainline-compatible model the TUI can install the standalone
`hot-experts` recipe. For a source-built fork it creates a helper-only composite
such as `glm5next-hot-experts`, preserving the exact base loader and route.
The active base checkout/binary is never modified. Patch conflict, compile
failure, missing CUDA capability, missing model architecture, or absent cache
flags refuses the composite and leaves the base available. A binary-only or
substantially divergent fork cannot be made compatible mechanically; it needs a
reviewed family-specific adapter. See [fork backend composition](fork-backends.md).

`auto` turns the cache on as a first-class coordinate. The packed GPU-expert
layout stays the cache-free baseline. After that exact baseline has allocation
evidence, the optimizer calculates one useful cache size — from leftover VRAM
if it is enough, otherwise by demoting GPU expert layers until a useful cache
fits — and compares that complete challenger with the same two-sample agent
workload. The cache wins only with material pure-decode and end-to-end
workflow gains, no material prefill/mixed regression, non-zero consistent
backend hit telemetry, and the normal functional, prompt-cache, lifecycle,
and clean-relaunch gates. A baseline win is remembered, so ggrun does not
reload the model to repeat the same unsuccessful experiment every launch.

Automatic eligibility is intentionally narrow: a resident CPU-offloaded MoE,
one server slot, layer split, complete separate gate/up/down expert tensors,
no speculative companion, and no mmap last-resort placement. The reference
backend has additional graph-time restrictions; runtime hit telemetry plus the
identical A/B is the authority when GGUF metadata cannot prove them statically.
Prefill continues on the normal batch/ubatch/placement path because this cache
is decode-only. The current reference cache also bypasses multi-token decode,
so ggrun evaluates native MTP/speculation and hot experts as separate candidates
rather than claiming both benefits at once.

`--hot-experts off` disables the lane. `--hot-experts on` requires cache-on but
lets the optimizer choose its size. A positive integer requests that exact
number of cache slots per eligible layer and fails closed unless an exact
cache-free allocation proves it fits; it cannot exceed the model's expert
count. The native `--moe-expert-cache` spelling before `--` is accepted as an
alias, but raw cache flags after `--` are rejected because they would bypass
the per-device VRAM ledger. Insertions remain fixed at two per decode step so
the first measured integration changes only one dimension.

This remains experimental until the public model/hardware matrix in
[the core optimizer tracker](core-standard-launch-todos.md#hot--dynamic-vram-expert-cache-for-resident-cpu-offloaded-moes)
is complete. On activation, telemetry, correctness, or memory failure, automatic
mode invalidates the cache-on evidence and restores the exact cache-free plan.
A deterministic failure is remembered only after that cache-free plan passes
the same lifecycle, preventing both false attribution and repeated long reloads.

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

## Core standard-launch optimization

ggrun's placement planner computes where a model's weights, experts, KV cache,
and runtime graph live from the GGUF and measured memory — but more than one
placement or scheduler shape can usually *fit*, and an estimate can only guess
which is fastest on your exact topology and workload. The default `auto` mode
therefore computes the complete bounded neighbor set but live-compares only the
stable baseline and one selected finalist. It prefers a finalist already
covered by the exact allocation signature (for example, a larger logical batch
at the same physical ubatch); otherwise it admits the nearest completely
recomputed neighbor. `--calibrate on` widens the live search for an explicit
maintenance experiment; its screening-only result is never consumed by the
no-flag path.

For parallel Claude/Ultracode on a hybrid or recurrent model, candidate
generation jointly covers valid `batch/ubatch` pairs: `128/128`, `256/128`,
`256/256`, `512/256`, and `512/512`. Automatic launch screens one calculated
finalist; explicit maintenance mode can traverse more of the ladder. It does
not use the generic short serial benchmark. Every pair is recomputed through
placement because ubatch changes graph VRAM and can move experts. Each live
candidate receives two distinct cache-backed append turns, concurrent cold
prefill, concurrent decode, and a mixed phase where a foreground decoder must
keep moving while the other slots ingest prompts. Automatic launch first runs a
small uncached prefill pilot, then chooses one per-lane prompt length that fits
two repeated baseline/finalist scenarios inside the finite budget; both
placements receive exactly the same bytes, and fast models retain the full
~8k-token screen. Explicit maintenance mode always uses the full prompt.
Lowest end-to-end cold-ingest plus cache-backed-append workflow wall time wins;
append-only latency, prefill/decode throughput, mixed foreground progress,
reused tokens, and the slowest confirmation sample remain visible diagnostics
and stability gates. A pair with no reported prefix reuse or mismatched prompt
geometry is invalid.

An `auto` winner is not reusable merely because its benchmark was faster. The
restarted exact argv must also pass the shared functional, append-cache,
older-branch/replay, workload-gateway, lifecycle, and clean-relaunch gates.
Only then are its decision, verified configuration, and eligible placement
cache promoted together. A failed cache write cannot strand a tuned-looking
verified config. Change the model, backend build/capabilities, GPU set, context,
slot count, batch pair, sampling/workload policy, companions, or explicit-intent
bits and the scope changes. A later runtime OOM revokes the scoped evidence.

Automatic optimization benchmarks at most two configurations: the stable
baseline and one finalist, with one failed challenger and a 20-minute search
budget. A failed finalist or a baseline win is recorded after the ordinary
launch gates, so the same scope does not repeat the experiment forever.
`--calibrate on` widens the maintenance sweep to nine candidates, two failures,
and 60 minutes. The calculated candidate set can include general batch/ubatch
neighbors, automatic 1/2/4 slot shapes, the parallel-agent recurrent ladder,
larger 1024/2048 MoE ubatches, a separately computed MoE KV placement, or an
inverted dense split where applicable. Roomy MoE candidates also include every
feasible sole-backbone owner while other cards remain complete-expert storage;
sustained per-device imbalance may redirect the one finalist to that computed
topology. `off` ignores optimizer decisions.

Exact allocation evidence authorizes only the exact argv that produced it. If
measured data yields a genuinely denser placement, ggrun subjects that new argv
to another contained admission pass; it retains an already-proven MoE placement
instead of changing only its tensor split without evidence. A candidate that
fails, OOMs, or reaches a different argv through recovery is not cached under
the requested name. The winner must still pass functional/cache/lifecycle and
clean-relaunch canaries, and a later runtime OOM invalidates it. If no
alternative wins beyond the 3% noise floor, the conservative default stands.

Models of 64 GiB or larger receive a warning before the first backend/reviewer
start. It is informational rather than another consent prompt and explains
that exact allocation measurement or bounded recovery may require more than
one long load.

## Model storage

`ggrun models list` shows GGUF files under the configured model directory and
groups split GGUFs as one model. `ggrun models browse` opens the same curated,
hardware-aware download list as `ggrun recommend`. `ggrun models path` prints that directory.
Use `ggrun models rm <model.gguf>` to remove a listed model; it asks before
deleting and only operates inside the configured model directory. Add `--yes`
for scripts or set `LLM_ASSUME_YES=true` in a non-interactive environment.

When a backend cannot report complete allocation evidence without loading the
model, ggrun asks before running one contained live memory probe. An interactive
approval is remembered as `LLM_ALLOW_LIVE_MEMORY_PROBE=true`, so later model
launches do not repeat that consent question. Set it to `false` in the config to
restore per-launch prompting; `--allow-live-memory-probe` remains a one-launch
override.

## AI Tune

`--ai-tune` starts from the launcher heuristic, benchmarks it, tests candidate flag sets,
and stores the best successful result in the local cache. Because it re-measures against
whatever llama.cpp / ik_llama.cpp build you currently have, it keeps your launch flags in
step with the backends as they change upstream, instead of you tracking new flags and
defaults by hand. The served model can propose candidate flags, but the launcher validates
them against backend help, memory headroom, crash behavior, and benchmark results before a
cache entry is reused. A 1% noise floor guards against replacing a good baseline with
single-run noise.

AI Tune changes safe performance knobs such as threads, flash attention,
mmap/mlock, defrag, and speculative decoding. It never changes batch or
microbatch: both require coupled placement recomputation and belong to the core
placement optimizer for every workload. It also never changes anything
that affects output quality — KV-cache quantization and context remain
constraints, and an explicit `--parallel` is left exactly as set. The core
standard-launch optimizer may compare automatic slot counts, but AI Tune and
cached/community tune overlays may not.

Parallel Claude launches use the placement-owned core optimizer above for
batch/ubatch experiments. Generic/community tune files are not
automatically applied to Claude mode because they do not encode slot scheduling
or mixed prefill/decode behavior. When `--ai-tune` is explicitly combined with
parallel Claude on a hybrid/recurrent model, it uses repeated cache-backed agent
turn time for the remaining safe flags and refuses all batch/ubatch proposals.
Those results are stored separately by slots, context, KV layout, and
placement; they cannot overwrite or auto-select as a generic serial tune. Tune
overlays protect the existing batch pair, with `ubatch <= batch` normalization
retained as defense in depth; explicit `--batch-size` and `--ubatch-size` remain
user-owned.

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
export ANTHROPIC_MODEL=local ANTHROPIC_SMALL_FAST_MODEL=local-fast
export ANTHROPIC_DEFAULT_HAIKU_MODEL=local-fast ANTHROPIC_DEFAULT_SONNET_MODEL=local ANTHROPIC_DEFAULT_OPUS_MODEL=local
export CLAUDE_CODE_EFFORT_LEVEL=xhigh       # agentic default; use max for one demanding session
export API_TIMEOUT_MS=2147483647            # maximum safe timer; no practical inference deadline
export CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS=2147483647
export CLAUDE_ENABLE_BYTE_WATCHDOG=0 CLAUDE_ENABLE_STREAM_WATCHDOG=0 API_FORCE_IDLE_TIMEOUT=0
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=262144 CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=75
claude --permission-mode auto --disallowedTools WebSearch
```

Foreground tiers point at `local`; Claude's cheap/Haiku tiers address the
router-only `local-fast` alias. With a worker-capable companion, that alias
uses it; with `qwen2b`, `off`, or no seated worker it is rewritten to the
main `local` model. Both aliases stay inside ggrun, so no inference call can
leave for `api.anthropic.com`.

### Resuming a session and its workflow

A long local workflow does not have to be restarted from zero when you stop the
backend. ggrun assigns each `--claude-code` launch its own Claude Code session ID
and records it with the exact backend shape that produced it:

```bash
ggrun claude list             # recorded sessions for this directory
ggrun claude resume           # relaunch the recorded shape, reopen the newest session
ggrun claude resume <id>      # or a specific one
```

In the TUI, the pre-launch screen shows a resumable session when one exists and
offers **[r] Resume that session and its workflow**.

Resume relaunches the recorded backend, reopens the conversation, and asks Claude
Code to continue the interrupted workflow run from its journal. Agents that had
finished replay from cache without a model call; agents still running when the
session stopped re-run. The summary printed before launch states how many agents
are actually recoverable.

Two deliberate restrictions:

- **The backend shape must match.** If the context, KV type, placement or batch
  settings changed since the session was recorded, ggrun refuses and names the
  differing setting. Reusing a conversation under different settings does not
  fail at runtime; it reinterprets state built under the old ones. Use
  `--claude-resume-force` only if you accept that.
- **`--fork-session` is refused with a resume.** Forking mints a new session ID,
  which moves the workflow journal path and silently discards every cached agent.

### Context sizing and concurrency

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
- **Wide fan-out** (subagents, workflows) retains up to four independent main-model
  slots, so each agent keeps its own reusable context. GPU-resident models can run
  those requests concurrently. For host-offloaded MoE or dense models, ggrun admits
  one generation at a time and queues the other agents at its loopback gateway:
  interleaving several CPU-offloaded decode graphs can reduce aggregate throughput by
  an order of magnitude. The status line includes both backend and gateway queue depth.
  ggrun sets the maximum safe Claude request/background-agent timers, disables both
  stream-idle watchdogs, and gives llama-server no practical socket deadline. Claude's
  Workflow tool has a separate 180-second `stallMs`; a session-only PreToolUse hook
  deterministically rewrites every `agent()` call to the maximum safe value before it
  runs. Startup, process-health, and shell-command guards remain active so a real crash
  or hung command is still visible.

### Sampling defaults

- **Anti-loop sampling.** The Anthropic API has no repetition-penalty fields and the
  client only sends temperature, so ggrun sets server-side defaults in claude-code
  mode (`--presence-penalty 1.0 --repeat-penalty 1.05 --repeat-last-n 512
  --top-k 20 --top-p 0.95 --min-p 0.0`) — quantized thinking models loop
  endlessly without them. DeepSeek V4 uses its separately validated
  `top-k 40` / `min-p 0.05` recipe and closes the reasoning budget. Pass any
  of these flags yourself (after `--`) and your value wins.

### Prompt caching and hybrid models

- **Compaction reuses moved prompt chunks.** On shiftable transformer contexts, when
  the backend supports it, Claude mode enables `--cache-reuse 256`. This complements
  ordinary common-prefix caching by
  shifting repeated system, tool and workflow chunks after old results are removed.
  A controlled production-cache test reduced a compacted 4,506-token prefill from
  45.1 seconds to one processed token in 0.15 seconds. Pass `--cache-reuse 0` or
  `--no-cache-prompt` explicitly to opt out. Hybrid/recurrent, multimodal, and
  multi-position-RoPE contexts such as native DeepSeek V4 and Laguna cannot shift
  their state, so ggrun does not emit the unsupported flag;
  it instead keeps one rolling context checkpoint per slot when at least 512 MiB of
  host headroom per slot remains. This lets llama.cpp restore append-only agent turns
  without exposing the unsafe 32-checkpoint backend default.
- **Hybrid slot fairness and optimization.** Claude mode starts parallel
  hybrid/recurrent models at a valid `batch=128, ubatch=128` baseline. This
  prevents a long prefill batch from withholding decode work from the other
  active slot for more than a minute. The standard-launch optimizer can measure
  the bounded batch pair and automatic-slot ladders; a winner becomes reusable
  only after its cache/workload/lifecycle gates pass. Explicit batch, ubatch,
  and parallel arguments win.

### Permissions and Auto mode

- **Web research:** the built-in WebSearch runs on Anthropic's servers and is hidden
  on a non-first-party endpoint, so ggrun disables it and auto-wires a no-key
  DuckDuckGo MCP when `uvx` is installed. Its `search` and `fetch_content` tools
  are pre-authorized so agents can locate and read current sources without a
  permission prompt — `--claude-code` does this for you. Prefer another provider? Add it with
  `claude mcp add …` (it runs alongside `ddg-search`), or launch `claude` yourself
  from the printed recipe and drop/replace the `--mcp-config` line.
- **Auto works locally and remains fail-closed.** ggrun detects Claude Code's
  exact security-monitor requests and routes them to a pinned local companion.
  The default `auto` profile is Qwen3.5-4B and also handles explicit
  `local-fast` cheap-tier work. `--claude-reviewer qwen2b` selects the smaller
  review-only Qwen3.5-2B; its presence does not authorize worker traffic, so
  `local-fast` falls back to the admitted main model. `nanbeige` is an
  explicit larger worker/reviewer choice, and `off` seats no companion and
  visibly self-reviews on the main model. The companion's reservation is part
  of main placement, and its measured VRAM replaces the conservative initial
  reservation. This is Auto, not `bypassPermissions`. The first launch
  downloads and verifies the selected pinned GGUF and serves one independent
  64k slot. GPU visibility is isolated to its selected physical device; an
  unplanned fallback may try CPU when no GPU seat succeeds. Override the
  artifact with `GGRUN_CLAUDE_REVIEWER_MODEL=/path/model.gguf`; custom artifacts
  are fail-closed to the review-only lane because ggrun has not verified their
  general worker capability. Each launch verifies both the Anthropic gateway and
  the actual XML safety-verdict route before activating the profile. A reviewer
  4xx/5xx, empty 200, thinking-only answer, tool call, or malformed verdict is
  withheld and retried through the admitted main route. Choose another
  permission mode with `GGRUN_CLAUDE_PERMISSION_MODE=acceptEdits`, or use
  `inherit` to preserve your global Claude setting. See Claude Code's
  [permission-mode requirements](https://code.claude.com/docs/en/permission-modes#eliminate-prompts-with-auto-mode).

### Progress display

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

Quality depends on the local model: pick a tool-capable coder. Main-model
fan-out is still bounded by its slots and ggrun's admission policy; a separate
worker/reviewer only accelerates requests explicitly addressed to its lane.
