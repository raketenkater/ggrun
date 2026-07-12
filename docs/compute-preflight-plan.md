# Compute preflight: fastest stable placement without paying for failed loads

Status: stage 1 (no-alloc launch preflight) implemented 2026-07-07. Stages 2-3 planned.

## Why

ggrun's goal is to find the fastest stable whole-layer plan for big MoE serving.
Maximum VRAM fill is not the objective when it reduces prefill or parallel throughput.
Two failure
classes broke that promise (DeepSeek-V4-Flash 146GB, 3×GPU, 2026-07-06/07):

1. **Load-time CUDA OOM.** ggrun's Go-side placement math disagreed with the backend's
   actual allocation and the launch died at `cudaMalloc` — after paying up to 15 minutes
   of `--no-mmap` model load per attempt on HDD.
2. **Runtime CUDA OOM after a healthy load.** The server loaded, passed the health
   check, served small requests — then a real ~30k-token prompt needed an extra
   1000 MiB on the tightest GPU at the 8192-token prefill/checkpoint step and the
   server died. ggrun had cached that placement as "good" because it only ever
   validated startup.

Both must be caught *before* the expensive load, or at worst recovered automatically —
and never by guessed fixed margins (workspace rule: placement math derives from real or
measured values only; fixed-MB cushions break at scale).

## What llama.cpp already provides (source-verified)

- `common_get_device_memory_data()` (`common/fit.cpp`) loads model+context with
  `mparams.no_alloc = true`: tensors get dummy buffers, the scheduler builds the real
  startup graphs, and `llama_get_memory_breakdown()` reports planned per-device
  model/context/compute bytes. No VRAM is committed; runs in ~1s even for a 146GB model.
- CLI frontend: `llama-fit-params --fit-print on <args>` prints per-device MiB rows:
  `CUDA0 <model> <context> <compute>` (+ `Host` row). Built from `tools/fit-params/`
  (target `llama-fit-params`, not built by default).
- Limit: startup reserve covers the synthetic prompt-processing and token-generation
  graphs. Real requests go through `process_ubatch` → `ggml_backend_sched_alloc_graph`,
  which can re-reserve a *larger* graph later (llama-server creates prompt-cache
  checkpoints every 8192 tokens by default — the exact crash point observed). So the
  no-alloc preflight is necessary but not sufficient for runtime stability.

Measured accuracy (DeepSeek-V4 @ ctx 1M, q8_0 KV, parallel 4, mainline b9859):
preflight per-GPU totals matched post-load `nvidia-smi` within ~40-300 MiB of the
measured values (plus the separately-probed ~680 MB CUDA context overhead per GPU).

## Stage 1 — implemented

`go/cmd/ggrun/preflight.go` + gate in `startLaunchWithCUDAOOMRecovery`:

- `findFitParamsBin`: looks for `llama-fit-params` next to the resolved server binary
  (backend build dir), then `.bin/`, then PATH. If a custom backend does not ship a
  companion binary, the gate is silently skipped and behavior is unchanged.
- Before every launch attempt (including OOM re-plans), ggrun runs the fit-print with
  the memory-shaping subset of the real args (`-m/-c/-b/-ub/-ctk/-ctv/-np/-ngl/-ts/
  -sm/-ot/--n-cpu-moe/-fa/-mg`) and `CUDA_DEVICE_ORDER=PCI_BUS_ID` (same device
  numbering contract as the real launch — mainline's default enumeration is
  fastest-first, which put a 15.6GB buffer on a 12GB card when launched without it).
- Per CUDA device: `model+context+compute + measured CUDA overhead` vs free VRAM.
  A deficit feeds `ReplanAfterOOM` with the exact measured overshoot — same machinery
  as startup OOM recovery, but at ~1s instead of a full load. Capped at 3 preflight
  re-plans; infrastructure failures never block the launch.
- Placement caches (`.place`) are now written **only after a healthy load** (main.go
  success branch, recovery.handleHealthy), and overwritten on success after a derate.
  Previously OOM re-plans persisted never-loaded plans, which poisoned later launches.

## Stage 2 — runtime-shape canary (planned)

The +1000 MiB runtime growth is invisible to the startup reserve. Plan:

- After health check, before declaring a placement stable (and before `.place` is
  trusted for max-fill), send one synthetic long prompt sized to the real per-slot
  budget (crossing at least one 8192-token checkpoint boundary), sampling per-GPU
  `nvidia-smi memory.used` peaks.
- Cache `runtime_extra_mb_by_gpu = peak - post_load_baseline` in the model probe
  (keyed like compute-buffer probes: model+ctx+ubatch+kv+backend).
- Placement then reserves the *measured* runtime delta per GPU on subsequent packs —
  a measured value, not a guessed margin. First-launch (no probe yet) keeps stage-1
  behavior; the canary runs once and upgrades the cache.
- 2026-07-07 measurement (mainline, V4, 30k-token canary): see
  `.benchmarks`/session notes — used to validate whether current mainline exhibits
  the same runtime growth.

## Stage 3 — runtime OOM recovery (planned)

Startup OOM recovery exists; post-health crashes currently just kill the server.

- `recovery.Launcher` already restarts crashed servers; teach its failure parser that
  a post-health `cudaMalloc failed` (log contains a served request before the crash)
  is a placement error, not a transient: feed the failed alloc size + device into
  `ReplanAfterOOM` (measured penalty), invalidate the `.place` cache, relaunch once.
- This is the safety net for shapes the canary didn't exercise (bigger parallel
  fan-out, vision, speculative decoding).

## Non-goals

- No unverified per-arch memory formulas as final authority. Cold-cache estimates
  may use measured architecture fallbacks, but the backend preflight remains the
  oracle and replaces them with exact rows before load.
- No fixed safety margins. Every reserve must trace to a probe, a fit-print row, or a
  parsed backend log line.
