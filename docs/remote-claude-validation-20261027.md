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

The unit-test child is a native fixture. Actual Claude/GPU acceptance is recorded
separately below; fixture coverage alone is not live compatibility evidence.

## Published CI and same-host live acceptance

Server draft PR [#101](https://github.com/raketenkater/ggrun/pull/101), commit
`e3b47b991081`: Linux/Windows native contracts, core, full Go/race/vet,
installer E2E and portable builds passed. The existing best-effort macOS smoke
fails because its synthetic model exceeds the runtime reservation on that runner;
macOS remains low priority, not qualified. GPU CI was skipped; GPU evidence below
comes from this server instead.

Client draft PR [#1](https://github.com/raketenkater/ggrun-client/pull/1), commit
`095262b`: both native platforms and all six cross-build checks passed. This
includes native Windows DPAPI, not a real Windows Claude terminal session.
The shared module resolves remotely at `e3b47b991081`; no development replace.

Live model: existing Qwen3.8 Flash Next Uncensored IQ3_M, approximately 83.4 GiB,
three NVIDIA GPUs. Existing RAM soft-offline mitigation was checked; no memory
stress, new model download or competing large model process was introduced.
The tested Linux candidate is now installed atomically on PATH, with the prior
binary retained at `.cache/remote-claude/ggrun-before-remote-feature`. Candidate
version injection used the wrong symbol, so it reports v3.2.0-go; use
`github.com/raketenkater/ggrun/pkg/update.currentVersion` in the qualified build.

Real Claude CLI 2.1.307, temporary project:

- Coding task: corrected calc.py, all four unchanged unittest cases passed;
  5 turns, 90.295 seconds, 28,473 cache-read tokens.
- Same-session resume: four tests passed without edits; 3 turns, 39.891 seconds,
  20,607 cache-read tokens. Different tasks, not a speed comparison.
- Resume notice originally polluted JSON stdout. Client `095262b` moves it to
  stderr and adds a regression using an actual native child executable.
- Actual gateway auth/route/model tests passed: protected unauthenticated routes
  return 401, backend administration 404, unsupported model 400.
- Generic HTTP reverse proxy with `/llm/` prefix connected and streamed.
- Disconnect check observed first content at 1.117 seconds, then both router
  admission idle and private backend `is_processing=false` within 0.204 seconds.
  The server remained ready. This is one cancellation receipt, not a latency SLA.
- Clean stop released model allocations (all GPUs at approximately 1 MiB) before
  TUI headless relaunch. TUI three-way toggle and saved launch path exercised.
- Relaunch changed actual per-slot context from 152,576 to 144,384; resume into
  the smaller capacity was correctly refused before inference. Seamless resume
  across reduced capacity is not promised.
- Optional owner Tailscale HTTPS on previously unused 8444 was configured;
  existing 443, 8006 and 8443 mappings were preserved and checked. Authenticated
  connect and real Claude requests worked from this server over that endpoint.
  This is same-host tailnet transport, not a remote laptop receipt.

## Worker report compatibility: discovered release blocker

A real foreground haiku-tier Agent call reached the utility worker, ran all four
tests and completed its handback. Reviewer and main routes returned 200, yet the
caller repeatedly missed the report and reached its turn limit. A prompt hint
made a second attempt finish with invented test findings and was rejected.

Root cause: Claude sends the report as `messages[].role=system`; the selected
Qwen Jinja template silently omits late system/developer messages. Neither HTTP
success nor a completed worker proves the caller received its findings.

The [official gateway protocol guide](https://code.claude.com/docs/en/llm-gateway-protocol)
documents a capability-rejection retry carrying that text in user messages.
The message bridge now rejects unsupported system roles before admission or
inference. Top-level system and ordinary user/assistant requests stay unchanged.
Claude 2.1.307 requires both `Unexpected role` and `input message role` in the
error to recognize the fallback; a first shorter error failed immediately.
The generic bridge role contract applies to both local and remote Claude modes,
independent of CPU/GPU/model choice. Native system-role support needs an explicit
backend/template capability contract, not one template assumed universal.

The corrected role rejection passed a real Claude 2.1.307 foreground Agent test:
one haiku-tier utility worker completed, the caller accurately reported all four
passing tests and their real cases, no files changed, and the test oracle SHA-256
was unchanged. Two turns, 99.230 seconds, 20,008 main cache-read tokens. The
uncached core gate and claudeauto/remoteclaude race suites passed afterward.
The initial fix receipt used a temporary facade. The complete rebuilt TUI
server at `0b82f8d` then passed the same test over the optional HTTPS endpoint:
two turns, 143.789 seconds, one completed foreground utility worker, all four
real test names accurately reported and unchanged test oracle. Published alpha
client v0.1.0-alpha.1 resumed that session in one turn (14.252 seconds), with
valid JSON stdout and the correct prior result. The original prompt policy is
retained; the failed hint is not shipped.

## Shutdown lifecycle follow-up

The next real stop exposed a false memory-release wait: the main backend had
exited but the 4B companion still held approximately 5.5 GiB, while the shutdown
check compared against the baseline from before both processes started. The
companion was only stopped after the 20-second wait, producing a false pending
release marker. All allocations did eventually release; this was bounded.

The serving shutdown now stops its owned router/companion before checking the
whole-launch memory baseline. Regression coverage requires companion cleanup
before sampling and preserves backend-memory waiting without any companion.
This affects CPU-only and GPU serving lifecycles, not placement or performance
selection. The full rebuilt server then stopped in 5.029 seconds: all GPUs returned to
1 MiB and no release-pending marker remained. Clean relaunch of the exact saved
request passed, preserving context 144384 and the separate worker. The published
client doctor and actual auth/private-route checks passed again over HTTPS.

## Published alpha installer

[v0.1.0-alpha.1](https://github.com/raketenkater/ggrun-client/releases/tag/v0.1.0-alpha.1)
was published by the tag pipeline after native tests and all portable builds.
[Release run](https://github.com/raketenkater/ggrun-client/actions/runs/40551932536)
also passed installation of the actual published, checksum-verified binaries on
Linux and Windows. The Linux installer was independently run into a temporary
directory on this server; that released binary passed the live resume above.
This is an alpha, not laptop qualification or a stable product release.

## Owner PATH candidate deployment

After core, race, native contracts and real same-host acceptance, the exact tested
candidate was atomically copied to the existing PATH target. The symlink remains
unchanged. `ggrun --version` reports `v3.2.12-remote-0b82f8d`; SHA-256 is
`07edd57c3a7bdd946dd9116bc736206cc3628626b4bb5f34c79bd8400638097f`.
The previous binary remains available for rollback. This is an owner development
candidate, not a stable release. The active server uses the same candidate bytes
and isolated test configuration; no owner serving preferences were overwritten.

## Required before completion

- Review overlapping worker resume fixes without duplicating shared policy.
- Actual laptop coding/workflow/disconnect/resume receipt, with files/tools on
  that laptop and server still serving afterward. Laptop OS remains pending.
- Richer server status UI and Windows real-terminal behavior remain unqualified.

No performance tuning or universal hardware/model support claim follows from
these transport and correctness checks.
