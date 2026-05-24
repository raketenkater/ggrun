# Go Core Gap Analysis

This document tracks the gaps between the current Bash implementation and the
Go implementation on the `go-core` branch, plus the places where the Go version
should intentionally improve instead of copying Bash behavior exactly.

## Current State

- Branch: `go-core`
- Bash entrypoints still exist: `llm-server`, `llm-server-gui`, `llm-server-mac`.
- Go implementation exists under `go/` with packages for config, detection,
  placement, daemon, recovery, TUI, tuning, update, probe, download, and vision.
- The Go binary is not yet wired as the default installed `llm-server`.
- Local environment note: `go` was not on PATH during this review, so Go tests
  could not be run here.

## High-Level Verdict

The Go version is the better long-term architecture, but it is not yet a drop-in
replacement for the Bash launcher. It has stronger structure, testable packages,
daemon support, cleaner placement code, and a real TUI foundation. The remaining
work is mostly parity, install/entrypoint wiring, and hardening the behavior
that Bash already learned from production bugs.

## Merge / Integration Risk

- Text conflict risk is low. The Go branch mostly adds `go/` and lightly changes
  Bash files.
- Behavior risk is medium. Bash has mature edge-case handling and tests that the
  Go path does not yet fully mirror.
- Recommended integration style:
  - Keep Bash as a compatibility fallback until Go is validated on real systems.
  - Add an explicit build/install path for the Go binary.
  - Make the default entrypoint switch deliberate, not implicit.
  - Run Go unit tests plus Bash regression tests before publishing.

## Hard Parity Gaps

### Entrypoint and Install Wiring

Bash:
- `llm-server` is the primary CLI today.
- `llm-server-gui` is the primary terminal UI wrapper.
- Install scripts and docs are written around the Bash entrypoints.

Go:
- Main command exists at `go/cmd/llm-server/main.go`.
- No verified install path makes the Go binary the default `llm-server`.
- No wrapper decision is documented: Go primary with Bash fallback, or Bash
  primary with opt-in Go.

Needed:
- Add build target, install target, and release artifact plan.
- Decide binary name during transition, for example `llm-server-go` first, then
  promote to `llm-server`.
- Keep a `--use-bash` or fallback path until Go parity is proven.

### CLI Compatibility

Bash supports many existing flags and positional flows:
- `llm-server <model.gguf>`
- `llm-server <repo/name> --download`
- `--dry-run`, `--benchmark`, `--cpu`, `--server-bin`, `--lib-path`
- `--backend`, `--kv-quality`, `--kv-placement`, `--gpus`, `--ram-budget`
- `--vision`, `--mmproj auto`, `--mmproj <path>`
- `--ai-tune`, `--retune`, `--rounds`, `--tune-cache`
- `--show-configs`, `config show/edit/path/reset`

Go currently uses subcommands:
- `launch <model.gguf>`
- `dry-run <model.gguf>`
- `download <repo/name>`
- `tune <model.gguf>`
- `daemon`, `probe`, `detect`, `config`, `update`, `gui`

Gap:
- Existing scripts and user muscle memory expect the Bash positional CLI.
- Go needs a compatibility parser or a wrapper that maps old forms to new
  subcommands.

Needed:
- Support `llm-server model.gguf` as an alias for `launch`.
- Support `llm-server model.gguf --dry-run` as an alias for `dry-run`.
- Support old flag spelling in addition to Go short names.
- Add tests that compare Bash and Go command outputs for common invocations.

### Config Format Compatibility

Bash:
- Uses `LLM_*` keys in a sourceable config file.
- Loads `LLM_CONFIG`, `LLM_APP_HOME/config/config`, or
  `~/.config/llm-server/config`.
- Migrates `config.sh` into `config`.
- Environment variables override file values.

Go:
- Has a structured `config.Config`.
- Loads config and migrates legacy config.
- The code currently mixes unprefixed keys such as `PORT`, `CTX_SIZE` with
  `LLM_*` environment names.

Gap:
- Existing user config files use `LLM_PORT`, `LLM_CTX_SIZE`,
  `LLM_KV_PLACEMENT`, etc. Go must load those exact keys reliably.
- Config show/edit should generate the same canonical keys users already have.

Needed:
- Accept both legacy unprefixed keys and current `LLM_*` keys.
- Write only canonical `LLM_*` keys going forward.
- Add tests for env-over-file precedence and `config.sh` migration.

### Context Size Semantics

Bash on local `main` has learned behavior around:
- `LLM_CTX_SIZE=fit`
- `--ctx-size fit`
- `--ctx-size max`
- numeric context values
- non-interactive safety, especially suppressing prompts under
  `LLM_ASSUME_YES=1`

Go:
- Computes auto-fit context when `ContextSize <= 0`.
- TUI defaults to a fit mode.
- CLI config currently models context as an integer.

Gap:
- Go needs first-class context mode, not only an integer.
- `fit`, `auto`, and `max` need to survive config parsing, TUI selection, and
  CLI compatibility.

Needed:
- Add a context mode enum: `fit`, `max`, `manual`.
- Preserve exact user intent in config and launch args.
- Add tests for `--ctx-size fit`, `--ctx-size max`, numeric env/config values,
  and non-interactive dry runs.

### Vision / mmproj Safety

Bash:
- Detects local matching `mmproj`.
- Rejects mismatched explicit projectors.
- Rejects incomplete projectors.
- Has test coverage for local matching and mismatch failure.
- README describes downloading the correct projector based on GGUF metadata.

Go:
- Has `placement/vision.go` and `vision/vision.go`.
- Can find local generic `mmproj` files.
- Can call the downloader for `--mmproj-only`.

Gap:
- Go projector matching is currently more filename-driven.
- It needs the same metadata validation behavior as Bash for explicit projectors.
- It needs equivalent tests for mismatch and incomplete projector rejection.

Needed:
- Reuse GGUF metadata to verify projector compatibility.
- Add Go tests mirroring `tests/test_safety.sh` vision cases.
- Keep downloader behavior deterministic and non-interactive in dry-run/tests.

### AI Tune Parity

Bash:
- Implements self-tuning against a running server.
- Protects user-explicit flags from tuner overrides.
- Distinguishes perf-only vs perf+placement scope.
- Stores per-model/backend/vision tune cache files.
- Provides `--show-configs`.

Go:
- Has tune package, cache picker, and a tune command.
- TUI can list tuned configs.

Gap:
- Need prove Go tuning writes/reads the same cache schema.
- Need ensure user-locked flags are honored across all old CLI aliases.
- Need ensure backend-specific tune files never cross-apply incorrectly.

Needed:
- Add parity tests for tune cache parsing and selected tuned config application.
- Add command-output comparison for `--show-configs`.
- Keep tune cache schema backward-compatible.

### MoE Recovery and Cached Placement

Bash:
- Has extensive MoE placement logic.
- Uses measured probe/system cache.
- Has mmap fallback.
- Has `--n-cpu-moe` recovery when `-ot` placement fails.
- Writes Bash-compatible `.conf` placement caches.
- Validates cached placement against current RAM/VRAM state.

Go:
- Implements MoE placement and Bash-compatible cache load/save.
- Uses probe/system cache.
- Has recovery package and daemon reload recomputation.

Gap:
- Need real-world validation that Go recovery matches Bash for large MoE cases.
- Need ensure compatible fallback caches do not load unsafe stale settings.
- Need preserve conservative fallback behavior without reintroducing the bug
  where a worse conservative setting wins over a working cache.

Needed:
- Add tests for cache key stability and compatible cache selection.
- Add tests for stale cache rejection when RAM/VRAM constraints changed.
- Validate on large MoE after CUDA reboot.

### GPU Detection and Hardware Heuristics

Bash:
- Uses `nvidia-smi`.
- Handles compute capability, PCIe lane/gen, GPU sorting, and some fallback
  behavior.
- Existing issue #16 shows V100/SM70 needs more hardware-specific tuning.

Go:
- Detects GPU PCI bus ID, compute capability, VRAM, driver, and PCIe bandwidth.
- Has sysfs fallback for PCIe bandwidth.
- Placement code is easier to extend with typed hardware rules.

Gap:
- Go should add explicit older datacenter GPU heuristics instead of copying the
  current generic defaults.

Needed:
- Add SM70/V100 profile:
  - prefer smaller batch/ubatch candidates for throughput testing,
  - consider f16 KV when memory headroom exists,
  - avoid assuming large batch is always faster,
  - record hardware profile in tune keys.
- Add tests for compute capability driven defaults.

### Speculative Decoding

Bash:
- Currently has little direct speculative decoding integration.

Go:
- Has `placement/draft.go`.
- Can detect draft model candidates.
- Falls back to ngram speculative flags.
- Recent branch history shows changes around disabling auto spec when it has no
  GPU-model speedup, then using ngram defaults.

Gap:
- Need a policy decision: speculative decoding should be opt-in, auto, or
  hardware/model gated.
- Need benchmark proof that default ngram helps before enabling it by default.

Needed:
- Add a config/CLI option such as `--spec off|ngram|draft|auto`.
- Record spec mode in dry-run output and tune cache keys.
- Only enable auto-spec when a measured benchmark proves positive gain.

### Daemon / Reload Behavior

Bash:
- No first-class daemon equivalent.

Go:
- Has daemon package and `/reload`.
- `/reload` recomputes placement for new model paths when explicit server args
  are not provided.

Gap:
- This is a Go advantage, but it needs API contract tests.

Needed:
- Document `/reload` request schema.
- Add tests for model swap with and without explicit `server_args`.
- Add graceful shutdown and health timeout tests.

### TUI / GUI Feature Parity

Bash GUI:
- Has direct model launch path.
- Handles backend selection and saved backend defaults.
- Handles model directory changes.
- Handles tuned config picker.
- Handles context, KV placement, AI tune, benchmark, vision, keep-alive.
- Has shell regression tests.

Go TUI:
- Uses Bubble Tea and is a better foundation.
- Has model list, settings, download, tuned config picker, launch request.

Gap:
- Need verify every Bash GUI workflow exists in Go TUI.
- Need preserve direct non-interactive `--model` behavior or document replacement.
- Need automated TUI-level tests where possible.

Needed:
- Add command-level tests for TUI launch requests without requiring a real
  terminal.
- Ensure settings written by Go TUI are readable by Bash fallback and vice versa.

### macOS Path

Bash:
- Has `llm-server-mac`.
- Supports Metal-safe flags, macOS AI tune, vision, downloader, benchmarks, and
  startup fallback behavior.

Go:
- Detection has Darwin memory/CPU support.
- No confirmed feature parity with `llm-server-mac`.

Gap:
- Go Linux/CUDA work should not silently regress macOS users.

Needed:
- Decide whether Go replaces `llm-server-mac` or remains Linux-first for now.
- Add Darwin build/test plan.
- Keep Bash macOS launcher until Go macOS parity is proven.

### Tests and CI

Bash:
- Has regression scripts for estimator, GUI, safety, model index, MoE placement,
  and mac launcher.

Go:
- Has package tests in several areas.
- Could not be run locally during this review because `go` was unavailable.

Gap:
- Need CI that runs both Bash tests and Go tests.
- Need parity tests that compare computed args for synthetic GGUF fixtures.

Needed:
- Install Go in CI.
- Run `go test ./...`.
- Run shell regressions.
- Add golden dry-run fixtures shared between Bash and Go.

## Where Go Can Be Better Than Bash

### Typed Placement Engine

Go should become the source of truth for placement math. The Bash placement code
is powerful but too large and fragile. Go can make this better by:

- keeping model, hardware, cache, and strategy as typed structs,
- returning structured errors instead of parsing printed text,
- exposing placement as JSON for GUI/daemon/API consumers,
- unit testing every placement strategy without shell process setup.

### Structured Compatibility Layer

Instead of keeping Bash-style parsing forever, Go can use a compatibility layer:

- parse old CLI forms,
- normalize into structured launch options,
- emit warnings for deprecated flags,
- preserve old behavior in one place.

This lets the internal Go API stay clean while users keep existing commands.

### Safer Cache Model

Go should improve cache handling:

- version every cache schema,
- include hardware signature, backend tag, spec mode, vision flag, context mode,
  KV mode, and GPU filter in keys,
- validate stale caches with typed reasons,
- explain why a cache was accepted or rejected.

### Better Hardware Profiles

Go can add explicit profiles for hardware families:

- V100/SM70
- Ampere consumer GPUs
- Ada consumer GPUs
- datacenter multi-GPU PCIe systems
- NVLink systems

These profiles should shape candidate generation, not hard-force one config.
The tuner should still measure and override guesses.

### Measured Defaults

Go should prefer measured defaults over static defaults:

- first launch uses conservative safe estimates,
- successful launch writes probe/system cache,
- second launch uses measured compute/KV/CUDA overhead,
- AI tune uses those measurements to choose candidates.

### First-Class API Mode

The daemon can become a real management layer:

- load/reload models,
- dry-run placement,
- benchmark,
- expose active config,
- expose health and startup logs,
- eventually support local GUI/web frontends without shelling out.

## Suggested Integration Plan

1. Keep `go-core` branch focused on Go integration.
2. Add Go build/install wiring without replacing Bash by default.
3. Add CLI compatibility aliases for common Bash invocations.
4. Fix config key compatibility around `LLM_*`.
5. Add context mode support in Go config and CLI.
6. Add vision metadata validation parity.
7. Add parity tests for dry-run output using synthetic GGUFs.
8. Run `go test ./...` once Go is available.
9. Test on real CUDA hardware after reboot.
10. Promote Go binary to default only after parity tests and real GPU dry-runs
    pass.

## Immediate TODOs

- [ ] Install or expose Go on the development machine.
- [ ] Run `go test ./...` in `go/`.
- [ ] Add old CLI compatibility mode to Go.
- [ ] Make Go config read/write canonical `LLM_*` keys.
- [ ] Add `fit|max|manual` context mode to Go config and CLI.
- [ ] Add mmproj metadata validation in Go.
- [ ] Add Bash-vs-Go dry-run golden tests.
- [ ] Decide default speculative decoding policy.
- [ ] Validate V100/SM70 tuning behavior against issue #16.
- [ ] Decide when Go replaces Bash as the default entrypoint.
