# Remote Claude Code through ggrun — implementation plan

Status: implementation in progress in the isolated `feat/remote-claude-client` worktree at `.cache/remote-claude/server`, based on `101af89`. The separate client draft lives at `.cache/remote-claude/ggrun-client`. The installed PATH binary and network configuration have not been changed. Publication and live qualification remain pending.

## Product outcome

Use the existing ggrun TUI on an inference server to select a model and enable **Remote Claude server**. ggrun applies its normal Claude-mode serving policy and stays running without starting a local Claude Code client. On a laptop, a separate `ggrun-client` program connects to a configured HTTP(S) gateway URL and starts the installed Claude Code CLI in the laptop's project directory.

The server owns model placement, admission, model processes, KV/prompt caches, optional reviewer/worker and inference scheduling. The laptop owns the terminal, project files, shell/tools, permission settings, MCP processes, workflow scripts, memories and Claude sessions. Workflows continue to use the normal Claude Code harness; no replacement coding agent or chat UI is required.

This is a public-product feature. Addresses, credentials, hardware and model choices are runtime configuration. Ordinary HTTP(S) is the connection contract. Tailscale, a LAN, an SSH tunnel and an existing HTTPS reverse proxy are deployment options. Tailscale is the owner's chosen validation setup, not the default, a mandatory dependency or something either program installs/configures automatically.

## Fastest path to a useful product — decisions fixed before implementation

The first release is a complete remote CLI workflow, not a replacement Claude frontend or a new network-management product. The server TUI and laptop command must be usable without manual reconstruction of environment variables. Avoid a second mandatory laptop proxy, an account service, SSH-based model management, remote filesystem mounting or new model downloads.

1. **Server experience:** choose model in TUI, choose Remote client mode, confirm listener/credential settings, launch, and receive the advertised connection URL plus one client command. Keep the current terminal/service lifecycle explicit. The TUI describes backend listening separately from the advertised client URL; a loopback backend URL must not be printed as reachable from another machine.
2. **Client experience:** install a released binary (Go is not required), save a named connection with URL and privately entered token, run doctor, then run `ggrun-client claude` from the laptop project. This launches the actual Claude Code terminal with its normal project tools. Give precise next actions for missing Claude, DNS/network failures, invalid credentials, a loading model, incompatible versions and insufficient context.
3. **Connection defaults:** gateway listens on loopback unless the user explicitly selects a reachable interface or configures a proxy/tunnel. Accept any valid supported HTTP(S) URL; prefer HTTPS for direct remote access and make cleartext HTTP a deliberate connection setting. Never infer that every user has Tailscale, a particular private IP, or this host's DNS name. Store the advertised URL only when supplied/confirmed; do not guess a remotely reachable address from `0.0.0.0`.
4. **Ownership:** laptop client exit stops its own helpers and cancels its requests; only an explicit server Stop operation or actual server failure ends serving. Connection loss is reported without pretending an interrupted turn completed. Resume uses laptop sessions and verifies the current server/model/context before continuing. No automatic replay of partially delivered tool-generating turns.
5. **Compatibility:** support the installed Claude CLI first through a declared, tested version range. Ordinary CLI use, foreground workflows and qualified child agents must work. Report unsupported features/capabilities rather than promising all behavior exposed by future Claude releases. Core server serving must not depend on Claude being installed.
6. **Scope discipline:** reuse the current serving path and client policy, keep user-selected model/context/permissions/reviewer settings, and run the existing core gate. Do not mix this feature with the worker's seven hardening branches or a new optimizer experiment. Coordinate the integration base, carry necessary reviewed fixes, and record exact revisions.

First acceptance is one real coding task from the laptop over the owner's chosen connection, with files and tests executed on the laptop and the model left loaded after client exit. Product tests additionally verify the generic URL path without Tailscale installed. More elaborate network discovery, remote model switching and persistent background-agent supervision can follow evidence from this complete first workflow.

## Current implementation audit

Reference: installed binary `v3.2.12-26-g101af89`, same commit as inspected `origin/main` on October 27. The root checkout is stale and contains extensive unrelated edits; implementation must start from a fresh isolated checkout of current main and coordinate with the worker's hardening changes.

| Existing behavior | Code | Remote requirement |
| --- | --- | --- |
| TUI Claude toggle, launch confirmation and argv serialization | `go/pkg/tui/tui.go`, `latest_launch.go` | Explicit local/remote selection, saved/restored correctly; dry-run and launched argv agree |
| Claude placement/cache/sampling/verification policy | `go/cmd/ggrun/main.go` | Keep the same shared launch pipeline; headless changes client ownership, not serving policy |
| Gateway, classifier/utility routing and reviewer fallback | `go/cmd/ggrun/claude_auto.go`, `go/pkg/claudeauto/` | Stable loopback listener and authenticated remote API around existing routing |
| Gateway currently listens at `127.0.0.1:0` | `claudeauto.StartRouter` | Configured fixed port for remote mode; retain existing local behavior |
| Non-message gateway paths currently proxy straight to backend | `claudeauto.StartRouter` | Remote surface needs explicit supported routes; backend administration routes must remain private |
| Local client launch and environment | `runClaudeCodeClient`, `claudeCodeEnv`, `printClaudeCodeRecipe` | Separate launch from serving; share a typed client policy and keep recipe/runtime parity |
| Model aliases `local` and `local-fast` | client env + router | Foreground, cheap-tier, subagent and classifier calls all remain on the intended local-model routes |
| Attribution-header suppression and cache/checkpoint settings | client env + launch policy | Carry these to the laptop and verify actual prefix reuse |
| Actual slot context and compaction limits | client env + resume helpers | Advertise resolved context per slot; do not infer from advertised maximum or requested total context |
| Slow-inference watchdog policy and Workflow hook | `claude_workflow.go` | Hook must run on laptop files/transcripts; version-specific behavior needs tests |
| Progress polling and status line | `claude_progress.go` | Current status command reads a local file by port; laptop needs authenticated remote status with bounded polling |
| Search replacement | `claudeCodeSearchMCPArgs` | Detect/install prerequisites on laptop, preserve user MCP configuration |
| Session recording/resume and exit handling | `claude_resume.go`, `runClaudeCodeClient` | Sessions stay on laptop; client exit/disconnect must not terminate server; signal behavior differs intentionally |

Do not copy server paths or a shell command returned by the server into laptop execution. The connection response is descriptive data; the installed client constructs its own known commands.

## TUI and CLI design

TUI setting: **Claude Code: Off / On this machine / Remote client**. The existing shortcut cycles these choices consistently wherever the toggle is available. Retain reviewer selection, context and scheduling settings in both Claude modes.

Remote mode shows the gateway port, advertised connection URL and credential configuration. The launch confirmation includes mode, model and resolved serving settings. After launch, a serving status view shows readiness, connection instructions, requests/queue and an explicit server Stop action. Leaving the laptop client is independent of this action. Background persistence should use the existing supported lifecycle or a documented user service; do not imply that merely closing the server terminal detaches its processes.

Proposed CLI spelling: `ggrun launch MODEL --claude-code --headless --gateway-port 8080`. Final flag naming should follow repository conventions. `--headless` here is specifically the Claude client ownership choice. Reject contradictory local-session resume flags in server mode with an explanation that laptop sessions resume via ggrun-client.

Old saved TUI launches with `ClaudeCode=true` restore **On this machine**. Add an explicit typed mode or validated additive field; ensure old false values restore Off, mode transitions do not leave stale flags, latest-launch serialization migrates safely, and local behavior stays compatible. No independent TUI planner.

## Server API and transport

- Keep llama-server and companion listeners on loopback. Authenticate the externally reachable gateway using an independently provisioned bearer credential, not today's dummy `ggrun` token. Protect inference, discovery, metadata and status; do not print credentials in logs, put them in URLs, or return them in discovery.
- Provide a small versioned connection document, e.g. `GET /ggrun/client`: protocol version, server/model identity, aliases, actual context per slot, supported capabilities, utility/reviewer availability and typed client-policy version. No arbitrary environment variables, executable commands, local filesystem paths or raw prompts in this document.
- Route `POST /v1/messages` through existing admission and reviewer handling. Preserve SSE event ordering, tool calls/results, thinking blocks, usage, cancellation, recognizable error bodies and relevant protocol headers. Do not buffer a complete answer before forwarding it.
- Support `/v1/messages/count_tokens` when the selected backend can implement it correctly; otherwise advertise its absence and let supported clients estimate. Do not fabricate exact counts.
- Advertise `/v1/models` only for models/aliases the gateway actually serves. Reject unsupported IDs clearly. Normal aliases must not silently reach an unrelated upstream.
- Expose bounded authenticated health/status. Progress should use cached server observations and must not compete with inference or expose backend administration APIs.
- Disconnect cancels that request and releases admission when the backend actually finishes/aborts; it must neither kill the model nor claim cancellation has completed while an old batch is still running. Reuse the worker's ongoing slow-canary/cancellation hardening rather than implementing a second workaround. Do not automatically replay partially delivered generations or tool actions.
- Generic deployment: configurable gateway listener plus an independently configured advertised HTTP(S) URL. Support localhost, private-network binding and existing reverse proxies/tunnels; no hardcoded network provider. Parse URLs correctly, including IPv6, path prefixes and trailing slashes. Claude appends `/v1/messages` to its base URL; do not accidentally duplicate or strip an API prefix. Validate endpoint metadata against the requested origin, and do not forward credentials across redirects to another origin.
- Optional owner deployment: Tailscale Serve HTTPS to the dedicated loopback gateway. Tailscale already serves ports 443, 8443 and 8006 on this host, including `/` and `/ac2` routes. Reserve an unused endpoint after checking the current configuration; never reset or replace existing Serve mappings. Tailnet grants/ACLs and the API credential govern access. Document this as one optional example, alongside generic direct/proxy/tunnel setup.

## Laptop repository and upstream relationship

Planned repository: `raketenkater/ggrun-client` (does not exist at planning time). Build a small Go CLI with released binaries and commands such as `connect`, `doctor`, `claude`, `claude --resume` and an internal status/hook command. The user installs Claude Code separately through its official distribution; this repository provides integration, not a redistributed Claude binary.

Both repositories link each other in their README and setup documentation. ggrun's TUI/CLI connection instructions point to the client installation guide. The client's README identifies ggrun as the serving engine and records supported gateway protocol/server versions. Publish a compatibility matrix and changelog for contract changes. Keep a schema and conformance fixtures in ggrun as the contract source; the client pins a reviewed contract revision. Use released dependencies where useful; avoid a submodule or a requirement to install the entire server on the laptop.

Refactor only the necessary client policy into a small reusable component in ggrun, with tests proving local launcher/printed recipe/exported metadata agree. The laptop client may consume a reviewed pinned package or a narrow wire contract; do not fork a second independent table of aliases/context/cache rules. OS-local hook and credential execution belongs to the client.

Client connection profiles store URL and identity with private credential storage, outside project repositories. Read tokens through a private prompt, standard input, a supported credential store or environment; avoid command arguments and committed project files. Redact secrets in diagnostics. Use an OS-appropriate private file only where a credential store is unavailable. Invocation-scoped environment/settings preserve normal Claude login, user hooks, custom status lines, MCPs and explicit permission choices. Diagnose managed-settings or credential conflicts rather than trying to bypass them. Ensure any spawned workflow/subagent process inherits the intended connection; supervisor-hosted background sessions need separate qualification because shell inheritance alone is insufficient.

At startup validate server protocol, readiness, identity, context and Claude client version; run from the laptop's working directory. Pin the supported Claude version range in CI. A resumed session checks model identity and capacity; a server/model change must not silently truncate useful context or route elsewhere. Cache loss is reported honestly after a server restart.

## Online research findings that change the design

1. [Official gateway compatibility guide](https://code.claude.com/docs/en/llm-gateway-protocol): Anthropic Messages is a supported gateway format; count_tokens is optional. Streaming must preserve the complete event sequence and pings, and request headers/body fields evolve. Contract tests must cover errors and new fields, not only a successful text reply.
2. [Official connection guide](https://code.claude.com/docs/en/llm-gateway-connect): bearer auth uses ANTHROPIC_AUTH_TOKEN. Different Claude surfaces load settings differently; gateway shell exports do not automatically configure the Desktop app or every supervisor process. Initial support is the CLI and its qualified descendants.
3. [Official model configuration](https://code.claude.com/docs/en/model-config): unrecognized gateway aliases may assume 200k context. Configure the actual served window using supported settings, retain compaction, and test against the chosen Claude version. The legacy SMALL_FAST variable is deprecated; current aliases and subagent defaults need explicit review. Avoid masquerading a local model as a Claude model merely to inherit capability flags.
4. [Official hooks](https://code.claude.com/docs/en/hooks) and [status line](https://code.claude.com/docs/en/statusline): these are client-side integration points. Use installed laptop helpers and respect existing settings. The Workflow timeout rewrite is an existing ggrun behavior, not a universal guarantee for future Claude versions.
5. [Official Tailscale Serve CLI](https://tailscale.com/docs/reference/tailscale-cli/serve): supports loopback reverse proxies and separate HTTPS ports. Preserve this host's existing routes and verify streaming over the actual proxy.
6. [Official gateway overview](https://code.claude.com/docs/en/llm-gateway): Anthropic does not support routing to non-Claude models through gateways. ggrun therefore owns compatibility validation and must avoid promising every Claude feature. Voice, cloud sessions and Remote Control are outside the first CLI milestone.

## Delivery and acceptance

1. Server/TUI feature PR: typed mode, compatibility-preserving serialization, stable gateway listener, auth and versioned metadata. Read the protected-core contract/theory/ledger before any launch changes; add invariant tests and run the uncached core gate. Rebase against reviewed worker changes before integration.
2. Client repository: connection profile, doctor, version/capability checks, known local helpers, scoped configuration and real-Claude launch/resume. Add Linux and Windows release/install tests and portable cross-builds; macOS live acceptance can remain explicitly untested initially.
3. Contract CI in both repositories: supported versions, invalid credentials, unsupported model, context metadata, tool/think/stream round trips, queued cancellation, upstream loss, settings preservation, alias routing and typed metadata rejection. Exercise an intentionally missing `claude` executable in server mode. Test generic direct and reverse-proxy URLs, including path prefixes, IPv6 parsing, unavailable ports and a deliberately absent Tailscale executable. Ensure an unauthenticated call cannot consume inference resources or read metadata/status. Keep tests independent of real model downloads where possible.
4. Live server acceptance through the owner's PATH ggrun, launched from the TUI in Remote client mode with the installed unrestricted Qwen Flash-Next: compare emitted argv against equivalent local Claude mode, observe gateway auth and generic streaming, verify cached follow-ups, cancellation and clean server stop/relaunch. Respect current GPU work and memory-health status. Add the owner's optional Tailscale path without changing the generic connection contract.
5. Real laptop acceptance: install the released client artifact on the owner's actual laptop OS, connect through its configured URL (Tailscale for the owner), launch Claude in a laptop project, read/edit/run tests locally, route reviewer/utility calls correctly, exercise a long enough workflow to check timeout behavior, disconnect/reconnect, resume and verify that the model remains loaded. The laptop OS has been requested and is pending; portable implementation and server tests need not wait for it. A local HTTP test or cross-build does not establish this laptop milestone; obtain the real laptop test receipt before declaring completion.
6. Return reviewed PRs, linked repo/docs, exact binary/model/Claude identities, test results and explicit remaining limitations. No new model download or broad optimizer retuning is required for this feature.

## Implementation checkpoint — October 27

The connection contract is provider-neutral HTTP(S). Loopback is the default
listener; HTTPS proxy, LAN, SSH tunnel and Tailscale are optional deployments.
Neither program installs, detects or configures Tailscale.

Implemented as uncommitted drafts:

- Server CLI and TUI three-mode selection, saved launches and stable gateway
  reservation before loading. Authenticated loading/recovery, health, metadata,
  model discovery, Messages streaming, cancellation and cached queue status.
- Shared `clientkit` for typed policy, Workflow rewriting, Unix private files
  and Windows user-bound credential protection. Local launcher and remote
  metadata use the same context/alias/timeout policy.
- Client connect, doctor, Claude child launch and verified resume. Profiles and
  tools remain client-local; conflicting saved connection settings are diagnosed.
  Resume currently supports the last connector-managed session per project and
  connection. Arbitrary older-session selection is outside this first milestone.
- Protocol schema/fixture, reciprocal docs, Linux/Windows contract CI and client
  release/install workflow drafts.

Remaining acceptance gates, in order:

1. Final uncached core and contract gates after edits. Review policy parity and
   overlapping Claude-mode worker changes before integration.
2. Review/commit the server feature; pin the client dependency to a remotely
   resolvable reviewed module version/commit. Remove its development replace.
3. Publish the separate client repo; obtain native Linux/Windows CI receipts
   and released-installer validation. No first release exists yet.
4. Review the candidate before updating the owner's PATH binary. Real TUI
   headless launch with installed Qwen Flash-Next, cached follow-ups, cancellation
   and clean stop/relaunch. Recheck active work and memory-health evidence first.
5. Actual laptop coding task: local edits/tests, utility routing, long workflow,
   connection loss and resume. Laptop OS remains pending. Fixture executables
   and cross-builds do not establish this milestone.

No feature-complete claim until these gates have receipts. Finish one complete
remote coding workflow before expanding the UI, model switching or networking.
