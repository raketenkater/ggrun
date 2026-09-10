# Production readiness

Active direction, 2026-09-10: stabilize product boundaries around the existing
serving core so features can be added without duplicating launch policy.
This is an engineering checklist, not a declaration that all release gates pass.
The older development roadmap contains dated hardware snapshots; do not treat
its process IDs, installed builds or active-run descriptions as current state.

## Implemented in this hardening pass

- Configuration saves stage the complete output, check write/sync/close errors,
  then replace the destination. Existing permissions and valid symlink targets
  are preserved; new files are private. A failed encoding or promotion removes
  its temporary file and reports failure. A dangling symlink is refused.
- Daemon reloads require exactly one complete JSON document within the body
  limit before planning or starting a backend. Trailing values, malformed
  suffixes and excessive trailing whitespace are rejected. Unknown fields retain
  their existing compatibility behavior.
- Daemon errors are consistently JSON-encoded, including quoted paths and
  multi-line backend errors. Request-read timeouts bound slow clients without
  imposing a short response deadline on legitimate model startup.
- Contribution checks run from a consistent root directory. New shared state
  writes and control routes have explicit implementation boundaries.

## Next priorities

| Priority | Boundary | Completion evidence |
| --- | --- | --- |
| 1 | Daemon lifecycle | Status and shutdown remain responsive during a slow start; cancellation releases resources; failed reload preserves a recoverable prior configuration; no overlapping model processes. Review its current mutex-held startup before extending the control API. |
| 2 | Backend discovery/build/update | Every external probe is bounded; failed builds and installs preserve the previous executable and manifest; cancellation and interrupted promotion have regression coverage. Preserve the current installer work while reviewing this. |
| 3 | Configuration and feature contracts | Explicit validated inputs, provenance and compatibility for each new option; invalid requests cannot partially change runtime or disk state. Avoid parallel copies of placement policy. |
| 4 | User diagnostics | Stable error categories and redacted reports identify the phase, failed operation and recovery; no credential or prompt disclosure. |
| 5 | Release reproducibility | Clean install, upgrade and rollback in isolated app homes; packaged helpers and backend identity verified on supported platforms. |

## Feature acceptance boundary

A feature proposal names its owning package, inputs, capabilities, persistent
schema and failure behavior. CLI/TUI code should call that package rather than
reimplement it. Model/quant/context/user constraints remain inputs to the shared
launch path. Backend flag presence alone does not establish execution support.

Use `internal/atomicfile.Write` for whole-file replacement where its contract
fits. Callers supply transaction locking when merging concurrent changes. Do
not migrate protected evidence storage opportunistically: it has separate
schema and durability requirements. Existing files retain their permission
bits, but ACL/xattr preservation is outside this helper's contract.

Add failure-focused tests for corrupt/partial inputs, interrupted operations
and compatibility boundaries. Run package tests and the full Go/race/vet gate;
core changes additionally require `scripts/verify-core-engine.sh`. Runtime
performance acceptance still needs exact admission, identical agent workflows,
phase regression guards and clean relaunch. Unit tests and cross-builds cannot
establish live stability or platform runtime behavior.

This pass starts no model workload, publishes no release and changes no
placement policy. Core hot-expert evidence work remains tracked separately in
`core-standard-launch-todos.md`.

## Validation for this pass

Passed: full native Go build and vet, uncached `go test -race ./...`, Windows
amd64 vet/build and Darwin arm64 build. The final daemon read/lock changes also
passed targeted race tests, native/Windows vet and Darwin build. Failure tests
cover partial staging writes, failed promotion cleanup, permissions/symlinks,
invalid configuration preservation, trailing/oversized JSON, escaped backend
errors and status responsiveness while a reload body is blocked.

Cross-builds establish compile compatibility, not runtime behavior on those
platforms. Shell/Python release suites and live install/upgrade/model-serving
acceptance were not run in this pass.
