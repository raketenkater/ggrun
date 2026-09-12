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
