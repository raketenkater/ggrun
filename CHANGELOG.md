# Changelog

## v3.1.0 — 2026-06-20

- **PCIe-bandwidth-weighted MoE tensor-split.** On heterogeneous-PCIe rigs (e.g.
  a card stuck at x1) the split now concentrates layer ownership on the
  fastest-link GPU, so CPU-expert streaming isn't bottlenecked — up to **3.4×
  prefill** on MiniMax-M3 with no decode regression. Symmetric rigs fall back to
  the previous free-VRAM-proportional split.
- **Smarter model recommendations.** The download picker ranks by effective
  intelligence (AA index × quantization quality retained) × predicted speed ×
  fit, in three categories (best for your machine / smartest that fits / fastest
  capable), preferring Unsloth dynamic quants; the catalog auto-refreshes.
- **Clear backend/architecture errors.** Launching an ik_llama-only architecture
  (e.g. `minimax-m3`) on a mainline llama.cpp backend now fails fast with the fix
  instead of a cryptic load crash.
- **Polished launch UX.** An animated startup status replaces the raw backend log
  spam while a model loads; the TUI config screen is grouped into Context /
  Tuning / Run mode / Actions sections.
- **Exact-ledger multi-GPU MoE placement.** Large MoE models now load reliably
  on heterogeneous multi-GPU rigs instead of over-committing the smallest card.
  The launcher emits `--tensor-split` *and* `-ot` from an exact per-GPU VRAM
  ledger — measured CUDA-context + compute-buffer overhead plus GGUF-exact
  non-expert, KV, and expert sizes — fills expert layers in GPU-bandwidth
  order, honors leading dense blocks, and drops GPUs that can't carry their
  share. No percentage headroom; every term is measured or read from the GGUF.
  A `cudaMalloc` out-of-memory during load now triggers an adaptive retry that
  derates the offending GPU and caches the corrected placement.
- **Interactive GUI overhaul.** The whole TUI is navigable with arrow keys and
  Enter (letter hotkeys still work). New Settings screen lists every config
  option with its current value — enums cycle, booleans toggle, no typing —
  and saves to the config file; backend selection is an arrow-select; the
  advanced launch screen is fully navigable. Fixes an unreachable keep-alive
  toggle.
- **Reliable shutdown.** Ctrl+C now exits promptly; the keep-alive recovery
  loop no longer treats a requested shutdown as a crash and restarts the server
  being stopped. The GUI launch path gains a second-Ctrl+C / timeout force-quit.
- **Config is the single source of truth.** The installer no longer exports
  per-setting environment variables (model dir, backend, cache, logs) that
  silently shadowed the config file, so CLI and GUI edits actually take effect.
  Only `LLM_APP_HOME` and `PATH` are exported.
- **One launcher binary.** Installs a single `llm-server` (no duplicate
  `llm-server-go` copy, no `llm-server-gui` wrappers) — `llm-server` with no
  arguments opens the GUI. The `llm-server-bash` v2 migration shim is retained.
- **Dockerfiles** for CPU, CUDA, and Vulkan, plus an Open WebUI compose file (build locally; GHCR publishing is not wired yet).

- **Community tune pool.** When a model has no local AI-Tune cache, the
  launcher now checks a shared pool (one HTTPS GET keyed by
  model+size+GPU-set+backend, mirroring the local cache file naming) and
  applies a community-measured config after sanitization. Only flags on the
  tune allow-list survive (batch/threads/KV-types/flash-attn/spec settings);
  model paths, ports, devices, and placement flags can never be injected.
  Hits cache for 7 days, misses for 24 h, lookups time out in 3 s — fully
  offline-safe. Disable with `LLM_COMMUNITY_TUNES=off`; point elsewhere with
  `LLM_COMMUNITY_TUNES_URL`. After a successful `--ai-tune`, the launcher
  prints how to contribute the result back.

- **Apple Silicon: Metal now actually engages.** Hardware detection knew only
  nvidia-smi/rocm-smi/vulkaninfo, so Macs reported zero GPUs and launched with
  `-ngl 0` (CPU-only inference despite the Metal bundle). Detection now
  synthesizes a unified-memory GPU (75% of `hw.memsize`, Metal's default
  working-set limit), macOS backends are tagged `metal`, and CUDA/Vulkan
  device-routing flags are no longer emitted for them. Pending validation on
  real Apple hardware.
- **Native Windows** — PowerShell installer (`install.ps1`), Windows process
  and signal handling, CUDA backend via official llama.cpp prebuilts or a local
  build.
- **Benchmarked against llama.cpp `--fit`** — on a 3090 Ti + 4070 + 3060 rig at
  32k context, v3 default placement beat upstream auto-fit on every model;
  driving the same master binary the win held, so it is placement, not just
  backend choice. See `docs/performance.md`.
- **Repositioned README and repo metadata** for discoverability: pain-first
  intro, honest comparison vs raw llama.cpp `--fit` / Ollama / llama-swap,
  benchmark methodology statement, release-asset install path.

## v3.0.0 — 2026-06-11

llm-server v3 is a ground-up Go rewrite of the Bash launcher. The Go binary is
now the primary `llm-server` command; the Bash implementation ships as
`llm-server-bash` for migration only.

- **Go launcher** — single static binary with `launch`, `dry-run`, `tune`,
  `download`, `daemon`, `benchmark`, `detect`, `probe`, `config`, `update`, and
  a Bubble Tea terminal UI (`llm-server-gui`).
- **Measured v3 performance** — see the README benchmark table; v3 IK CUDA
  AI-tune reached +50% decode throughput vs raw llama-server defaults on a
  4B dense model, and stable 32k-context launches where raw defaults OOM.
- **AI Tune v2** — deterministic candidate plan plus optional LLM-proposed
  flags, candidate validation against backend help/VRAM headroom, 1% noise
  floor before replacing the baseline, vision-aware cache files, and resumable
  progress saves.
- **Model recommendations** — weekly-refreshed catalog (Artificial Analysis
  intelligence ratings + Hugging Face GGUF quant search) matched against
  detected hardware in the GUI download picker.
- **Speculative decoding policy** — `--spec auto` only enables validated paths
  (MTP, EAGLE-3, compatible draft GGUF); ngram modes stay explicit.
- **Fixes in this cycle**
  - `--gpus` now actually restricts placement and sets
    `CUDA_VISIBLE_DEVICES`/`GGML_VK_VISIBLE_DEVICES` (it was silently ignored).
  - `launch`/`gui` refuse to start when the port is already in use — the health
    check previously hit the existing server and reported a dead child as live.
  - Startup failures are detected immediately instead of polling the health
    endpoint until the full model-size-scaled timeout (up to 15 min for MoE).
  - Crash classification reads this launcher's own log (not the newest file in
    /tmp) and matches OOM on word boundaries (a model named "Bloom" no longer
    classifies as out-of-memory).
  - `--ai-tune` reuses a completed tune cache and says so; `--retune` forces a
    fresh run (previously accepted but ignored).
  - `firstPositional` knows all value-taking flags (`--parallel 2 <repo>
    --download` no longer tries to download "2").
  - `--update` reports git pull failures instead of claiming success.
  - Version string is single-sourced in `pkg/update` and stamped by release
    builds via `-ldflags -X`.
