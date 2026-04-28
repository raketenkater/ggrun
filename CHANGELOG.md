# Changelog

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
