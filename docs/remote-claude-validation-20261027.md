# Remote Claude implementation validation — October 27

Base: `101af89`. Isolated branch `feat/remote-claude-client`; root checkout and
the other worker's seven branches were preserved. This is a development receipt,
not release acceptance or a performance result.

## Scope and invariants

All model/hardware classes use the existing normal launch/admission and Claude
router. Changes in main only parse remote options, own the gateway lifecycle,
skip local client ownership in headless mode and share client policy. Placement,
calibration, GPU selection and scheduler algorithms are unchanged. Linux and
Windows, CPU-only and GPU paths remain required; this host's selected model and
private network are validation cases only. Tailscale is not a dependency.

The remote gateway is independent from private backend ports. It reserves its
port before loading, authenticates even loading/recovery responses, publishes
only typed metadata, preserves the Messages body and streaming protocol, and
never exposes backend administration. Known fixed request headers bind a client
to the model identity and context it configured; a changed model or reduced
runtime capacity is rejected before inference.

Client files/tools/sessions live where the actual Claude child process runs.
Credentials stay outside project repositories; Unix owner-only files and Windows
user DPAPI are separate implementations. Client release workflows refuse a
development module replace and require native CI before publication.

## Local checks passed

- Uncached `scripts/verify-core-engine.sh`: protected package tests and vet.
- Full `go test -count=1 ./pkg/tui`: 79.859 seconds; includes existing behavior.
- Shared clientkit `go test -race -count=1 ./...`: profile fixture, aliases,
  context bounds, private credential creation/read/replacement and URL handling.
- Gateway `go test -race -count=1 ./pkg/remoteclaude`: authentication, protected
  routes, future fields, headers, tool/thinking/usage responses, unbuffered ping,
  active disconnect, queued cancellation through the real router, and stale
  model/context rejection before inference.
- Client `go test -race -count=1 ./...`: prefixed URL, credential/redirect/error
  handling, private profile, model/context-bound resume (including a grown
  session), settings conflicts, Workflow mutation without granting permission,
  real child executable/CWD/scoped environment and server survival after exit.
- Client `go vet ./...`; Windows amd64 client and server cross-builds.
- `git diff --check`.

The child executable in tests is a native fixture. It proves launcher ownership,
not that an actual Claude coding workflow has succeeded. The server's no-Claude
test exercises loading/recovery/port reservation without Claude or Tailscale on
PATH; a full live headless model launch is still required.

## Required before completion

- Review the feature against overlapping worker Claude-mode changes. Keep useful
  resume/exit fixes without reintroducing a second environment-policy table.
- Publish/pin the shared module and client source; remove the local development
  replace. Obtain actual Linux/Windows CI and released-installer receipts.
- Review before updating the owner's PATH binary, then real TUI headless Qwen
  Flash-Next acceptance, cache-backed follow-ups, cancellation, stop/relaunch.
  Confirm current RAM-health evidence and no competing model load first.
- Real laptop OS/device acceptance: local edits/tests, utility routing, long
  Workflow, disconnect/reconnect/resume and model persistence after client exit.
- The serving terminal currently prints readiness, connection instructions and
  Ctrl+C stop; the client status line shows cached active/queued counts. A richer
  server TUI serving panel and phase/rate display remain to be qualified.
- Windows runtime credential/terminal/hook behavior and Claude 2.1.307 live
  compatibility are not established by a cross-build. macOS is low priority and
  must remain explicitly cross-build-only until evidence exists.

No production binary, Tailscale Serve route, model download or live model process
was changed during these checks.
