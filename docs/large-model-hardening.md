# Large-model Linux GPU hardening

Routine install E2E keeps the small model. A manual Linux GPU run can use a model
larger than one card and require evidence of weights on multiple devices. This
exercises placement and lifecycle on real hardware; it does not prove optimal
throughput, hot-expert correctness, or long agent-task quality.

After these workflow changes reach main, use the existing local runner:

```bash
scripts/gpu-ci-runner.sh run \
  -f gpu_model_repo=YOUR_ORG/YOUR_GGUF_REPO \
  -f gpu_model_quant=Q4_K_M \
  -f gpu_context=32768 \
  -f gpu_startup_timeout=1800 \
  -f gpu_request_timeout=300 \
  -f gpu_min_weight_devices=2
```

Choose a model whose total weights exceed the largest card and fit the machine's
VRAM/RAM budget. Check free RAM, VRAM and disk space first. The runner's default
3 GB free-VRAM check is only a small-model floor, not proof a large model fits.
The GPU job has a 60-minute total limit, including installation and download;
use the local path for an already-downloaded model or a longer investigation.
CPU and Windows smoke jobs retain their original model.

The dispatched job installs into a fresh app home, downloads into a dedicated
model directory, validates split GGUF completeness, runs the standard launcher,
generates text and checks shutdown/port release. Evidence includes input.json,
download.log, serve.log, reply.json and result.json. Model weights are excluded
from uploaded artifacts. A two-device requirement rejects CPU fallback and
single-device placement; KV, compute, output and abandoned launch allocations
do not count as model weights.

## Iterate locally without downloading again

Use an installed launcher and the same checker:

```bash
python3 scripts/verify-gpu-install.py \
  --launcher /path/to/installed/ggrun \
  --model /path/to/model-00001-of-00003.gguf \
  --output /tmp/ggrun-large-model-check \
  --ctx 32768 --timeout 1800 --request-timeout 300 \
  --min-weight-devices 2 --port 18855
```

Use `--ctx 0` to omit the context override and exercise automatic context selection.

This local mode tests serving through the installed launcher but does not test
the installer or download. Use a new output directory per run. It refuses an
occupied port and terminates only its own process group. Preserve the failure
artifacts before changing anything. Fix a reproduced defect, add a regression
test, then rerun with the same model, backend, context and constraints. Repeat a
successful launch to check cache/relaunch behavior. For agentic performance,
follow this with identical long-context, concurrent agent tasks; a short
completion is only functional evidence.

Release-bundle acceptance additionally requires installing the exact packaged
CUDA asset. A source launcher with an existing custom backend is useful for
isolated debugging, but is not evidence that a new CUDA release asset works.

## Windows Server: run later on the target hardware

From a checkout containing these scripts, in native PowerShell:

```powershell
.\scripts\test-windows-server.ps1 `
  -WorkDir D:\ggrun-checks\large-model-01 `
  -ModelPath D:\models\model-00001-of-00003.gguf `
  -Backend cuda -MinWeightDevices 2 -Context 32768
```

Or replace ModelPath with `-ModelRepo YOUR_ORG/YOUR_GGUF_REPO -Quant Q4_K_M`.
Use a directory that does not already exist. `-Release vX.Y.Z` pins the launcher
release; `-ReleaseDir D:\candidate` tests local checksummed candidate archives.
The CUDA backend source is whatever that installer selects; retain its version
log and do not infer a CUDA release bundle exists from a CPU launcher package.

The script installs a separate app home without changing PATH, checks launcher
and backend startup, records hardware, downloads and serves a small smoke model,
reinstalls without changing user configuration, and relaunches. If a large model
is specified it then serves and relaunches it with the required GPU count. Each
serving pass checks readiness, nonempty generated text and graceful cleanup.
Python must be available after installation; the script reports a missing Python
command as a failure. CUDA requires the server's NVIDIA driver.

Evidence is kept under WorkDir/evidence, including a final summary even on
failure. Downloads remain under their run directories; exclude downloaded-models
when sharing evidence. It does not delete the installation, edit the existing
production configuration, or terminate unrelated servers. Review gpu-before.txt
and gpu-after.txt for resource release on the actual server. Windows CI runs this
script against a real CPU candidate bundle, including paths with spaces; that
checks the script plumbing, not Windows GPU/offload behavior.
