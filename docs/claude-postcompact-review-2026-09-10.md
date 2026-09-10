# Claude Code review: 10 September 2026

Reviewed the latest session `0f9dc578-5269-47fc-80bd-f2ebfc454985`, especially the work after its 10:57:59 UTC compaction, through the 16:28 update. Cross-checked code, commits, GitHub job failures, and an isolated Linux installation. Main was e06d6e8, then advanced to 341bd27 (PR #31); downloader/GPU work is on 914a0e0. Fixes from this review are on `linux-release-hardening`.

## Findings

1. **High: candidate installation was not a release gate.** Published-release E2E kept testing v3.2.8 while new changes were being prepared, and package checks could pass without the installed wrapper generating anything. Added candidate installation and real generation checks, plus reinstall/config preservation and shutdown checks. PR E2E uses the current launcher with a checksum-verified prior real backend; release jobs exercise their newly built backend. These are distinct evidence sources.
2. **High: CUDA packaging lacked its new dependency.** The release ran for over two hours before failing because the container lacked patchelf. Claude initially called this job independent of the packaging changes, then corrected that statement and added its dependencies in PR #31. The job must pass before this can be considered fixed in a release.
3. **High: macOS setup rejects Mach-O backends.** `usable_llama_server` only accepted ELF/PE, causing an installed macOS backend to be omitted from configuration. This explains a remaining red smoke test independently of the loader-path fixes. Added Mach-O recognition and fixture coverage; real macOS CI remains required.
4. **High: reinstall resets user configuration.** Both shell and Windows installers unconditionally rewrote `.config/config`. Changed them to create defaults only when absent. Linux reinstall preserved the exact config, model, and packaged launcher; Windows validation is wired into the release gate but has not run locally.
5. **Medium: explicit release installation rebuilds the launcher from source.** A checksum-verified binary was replaced and Go could be downloaded unnecessarily. Explicit release mode now retains its archive launcher; auto/build behavior is unchanged.
6. **Medium: Python dependency installation fails inside a virtual environment.** `pip --user` is invalid there. Installer now omits `--user` in a venv, including ensurepip. Reproduced the original failure and completed a fresh install with the fix.
7. **Medium: ELF relocation misses missing RUNPATH and incorrect ORIGIN subdirectories.** Claude's guard avoided corrupting static Go fixtures but skipped valid dynamic-library failure cases. Relocation now checks for dependencies actually shipped in the bundle. Real compiled ELF fixtures pass after their original build directory is moved away.
8. **High when enabled: self-hosted GPU execution was available to PR workflows.** Restricted this optional job to manual runs on main and gave it an isolated app directory. It now requires a positive device model allocation in the backend log, plus real generation and cleanup. This job has not been run on a GPU here.

## What the review supports

The explicit attention-head metadata and KV dimension fallback fix a real small-model failure. Replacing header-only GGUF fixtures with real models materially improves installation evidence. Relocated archive checks address a genuine distribution failure. Earlier atomic-file, daemon, and parent-accounting hardening remains in the development branch; it was not lost after compaction.

The adaptive canary still extrapolates from a tokenized text segment and reserves a fixed framing budget. It does not tokenize the complete rendered chat request. Treat this as an estimate, not proof that every template/context combination fits. Any further change belongs under the protected core contract and its invariant tests. No protected core behavior was changed in this review branch.

At review time, the main/tag CI still failed at `release-install-macos-smoke`; the published E2E failures were real v3.2.8 defects. The first v3.2.9 release attempt failed on CUDA packaging; the next failed in preflight. Therefore the earlier claim that only CUDA remained applied to package jobs, not project-wide readiness. Claude subsequently acknowledged the failures.

Evidence: [tag CI](https://github.com/raketenkater/ggrun/actions/runs/34501887694), [first release attempt](https://github.com/raketenkater/ggrun/actions/runs/34485857826), [next release attempt](https://github.com/raketenkater/ggrun/actions/runs/34501887702).

## Local validation

Artifacts are under `/tmp/ggrun-linux-e2e-et3tc1pk`. This was a new app installation on the existing Linux host, not a fresh OS image.

- Original installer failure: `install.log`; successful fresh install: `install-fixed.log`.
- Built-in download: `download.log`, Qwen3.5-0.8B Q4_0.
- Real CPU generation and graceful shutdown: `real-serving-cpu/result.json`, `reply.json`, `serve.log`.
- Reinstall preservation: `upgrade-result.json`; generation again: `upgrade-serving/result.json`.
- Installed launcher SHA256 matched its archive: `9967d1ceef5eb563aa124fdc5ce0453cd2e87e7eb07e0e35bed6be0dab2e6fd0`.
- Full uncached Go race suite and Linux/Windows vet passed before incorporating PR #31's workflow-only change.

An initial harness attempt using a CPU-only bundle on a GPU-visible host failed; the harness also initially used the wrong context flag. The successful runs explicitly use `--cpu` and `--ctx 2048`. They establish installed CPU serving and lifecycle correctness, not GPU performance, hot-expert promotion, or agent-workload throughput.
