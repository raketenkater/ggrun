# Release validation

The pipelines distinguish fast regression tests, candidate installation, and the published product.

- `ci.yml` runs unit/contract suites, cross-builds, and installer/package fixtures. ELF relocation tests compile a real program and private shared library, hide the build directory, then execute the archive copy.
- `install-e2e.yml` on PRs/pushes builds the current launcher and packages it with a checksum-verified v3.2.8 real backend. Linux and Windows install the candidate, use the installed wrapper, download through `ggrun download`, generate text, and verify shutdown and port release. This tests changes before publication, but does not replace testing a newly compiled backend.
- `release.yml` exercises newly produced Linux/macOS and Windows archives through install, real CPU generation, reinstall preserving configuration, and generation again before their package jobs succeed. CUDA retains its separate build path and requires GPU evidence separately.
- Published-release checks are available through manual dispatch (`published: true`), the weekly schedule, and the release event. A release created with GitHub's default workflow token may not trigger another workflow; use manual dispatch or the schedule in that case. No additional token permission is granted.
- The optional GPU job runs only through a manual dispatch on main with `GGRUN_GPU_RUNNER=true` and a trusted self-hosted runner labelled `gpu`. It uses a run-specific app directory and requires a positive GPU model allocation. Hosted CPU success does not prove GPU admission or throughput.

Serving evidence is retained as artifacts even on failure. The shared local check is:

```sh
python3 scripts/verify-installed-serving.py \
  --launcher /tmp/my-ggrun/ggrun \
  --model /tmp/my-ggrun/models/model.gguf \
  --output /tmp/my-ggrun-evidence --cpu
```

On Windows use the installed `ggrun.cmd` wrapper. The check uses a fixed 2048-token context and a short generation budget. It refuses an occupied port, bounds readiness, records the reply and log, and fails if forced cleanup is needed or the port remains occupied. This is installation/lifecycle validation; performance promotion still requires the protected core contract and matched agent-workload evidence.

Two checks cover the parts of the agent path that health and one completion miss:

- **Cancellation** runs everywhere. It abandons a stream after three chunks and then asks for a completion, so a slot that is never released shows up as a hang rather than passing silently. Recovery has measured 0.55-0.61 s across a CPU-only runner and a three-GPU box, which is slot reclaim rather than anything about the hardware.
- **Prefix reuse** is behind `--prefix-reuse` and is off in CI, because it needs a context larger than the 2048 the install jobs use. It asks two questions behind one long shared prefix and records how much of it the second had to re-evaluate. It reports the measurement rather than asserting a ratio: prompt caching can legitimately be off.

Add `--agent-lanes N` to run the bounded tool-using repair tasks against the same server. Those three tasks are a smoke test, not agentic acceptance.
