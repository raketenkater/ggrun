# Changelog

## Unreleased (v3.0.1 candidates)

- **Apple Silicon: Metal now actually engages.** Hardware detection knew only
  nvidia-smi/rocm-smi/vulkaninfo, so Macs reported zero GPUs and launched with
  `-ngl 0` (CPU-only inference despite the Metal bundle). Detection now
  synthesizes a unified-memory GPU (75% of `hw.memsize`, Metal's default
  working-set limit), macOS backends are tagged `metal`, and CUDA/Vulkan
  device-routing flags are no longer emitted for them. Pending validation on
  real Apple hardware.
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

## Unreleased

- **Pgid-scoped shutdown** — backends now launch under `setsid` and shut down via process-group signal, so spawned helper threads/children get cleaned up. Port-listener cleanup will refuse to `kill -9` foreign processes bound to the same port (warns instead) — the previous lsof-sweep could nuke unrelated services.
- **Unified `launch_backend()`** — `try_start`, `run_with_restart`, and `try_start_with_overrides` all spawn through one helper that handles setsid, log offsets, OOM-score, and `RUNNING_PID` registration. `try_start` now also attempts ik→mainline fallback once on health-check failure, matching `run_with_restart` semantics.
- **MoE `--n-cpu-moe` fallback** — when `-ot` expert placement fails (VRAM pressure, odd layouts), the launcher now sweeps `--n-cpu-moe` and caches the recovered config.
- **`--show-configs`** — compact table of all cached model configs, with per-model detail view showing the exact command, tokens/sec, tune rounds, and backend commit freshness.
- **Unknown flags forwarded to llama-server** — any flag `llm-server` doesn't recognize is passed through to the underlying server (after tune-cache flags, so user overrides win).

## Recent

- **Vision-aware tuning** — `--ai-tune` now maintains independent caches for vision-enabled runs. If `--vision` is used, the system auto-detects/downloads the necessary `mmproj` and uses a dedicated, vision-optimized configuration.
- **Stage 1 (heuristic) + Stage 2 (tuned) layering** — the launcher first calculates safe heuristic placement (expert placement for MoE, tensor splitting for dense) as a baseline. The AI tuner is then strictly layered on top, optimizing performance parameters while respecting these stability boundaries (unless `--unlimited` is requested).
- **Robust fallback** — if `ik_llama` encounters hardware or compatibility issues, the launcher now automatically strips incompatible ik-specific flags and retries with mainline `llama.cpp`.
- **Vision-aware VRAM budgeting** — accurate memory estimation for the vision encoder (`mmproj`) is now performed *before* launch to prevent OOM startup crashes.
- **`--gpus` and `--ram-budget`** — restrict an instance to specific GPUs and cap RAM usage, enabling multi-instance coexistence (e.g. big model on GPUs 0+1, small model on GPU 2).
- **Split-mode graph** — automatically enables `-sm graph` for both `ik_llama.cpp` and mainline for superior multi-GPU scaling.
- **Crash recovery** — auto-restarts with backoff on runtime crashes, detects CUDA errors and image decode loops. `--keep-alive` for unattended deployments that must never go offline.
