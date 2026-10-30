# Claude Code on another computer

Remote client mode keeps ggrun's model, cache, scheduler and optional reviewer
on the inference server, while Claude Code runs in the client computer's project.
File edits, shell commands, MCP processes and session transcripts stay on that
client computer. Claude does not need to be installed on the inference server.

This feature is being qualified. It requires the matching remote-enabled ggrun
build and [ggrun-client](https://github.com/raketenkater/ggrun-client). It is not
present in older released server binaries. Protocol v1 initially targets Claude
Code 2.1.307; passing contract tests does not establish compatibility with every
Claude feature or release.

## Start serving

In `ggrun tui`, select a model, cycle **Claude Code** to **Remote client**, and
set the gateway listening address, port and optional advertised URL. The launch
uses the same serving policy as local Claude mode. Old saved Claude launches
remain **On this machine**. Reviewer and profile selections are retained.

The equivalent CLI is:

```sh
ggrun launch model.gguf --claude-code --headless --gateway-port 8080
```

The default gateway listens at `127.0.0.1:8080`. The model backend stays on
loopback. Use an existing reverse proxy/tunnel, or deliberately choose a reachable
interface with `--gateway-host`. `--gateway-url` describes the client-facing base
URL; it does not configure your proxy, firewall or TLS certificate. Wildcard
listeners are not client destination URLs.

The stable gateway port is reserved before model loading. Loading and recovery
return authenticated HTTP 503, while readiness follows verified admission.
Missing credentials return 401 without reaching inference. Discovery, model
requests and cached queue status all require the same gateway credential.
Backend administration endpoints remain private.

ggrun creates a private credential file automatically. Display its value only
when pairing a client:

```sh
ggrun gateway-token
# If you selected a different file:
ggrun gateway-token --file /your/private/credential-file
```

Enter this token into the client's hidden prompt. Keep it out of URLs, shell
arguments, screenshots and committed configuration. Stop serving and remove the
old credential file to provision a new token; reconnect clients afterwards.

## Connect from the client computer

Install ggrun-client and Claude Code on that computer. From your project:

```sh
ggrun-client connect https://models.example.com --name server
ggrun-client doctor --name server
ggrun-client claude --connection server
# Later, from the same project:
ggrun-client claude --connection server --resume
```

Base URLs can include a proxy prefix, such as `https://models.example.com/llm`.
Do not append `/v1`. A remote cleartext HTTP URL requires an explicit client
`--allow-http`; loopback HTTP through a local tunnel works without it. Connection
checks do not follow redirects with credentials.

The client checks actual served context per slot, model/backend identity and
protocol compatibility. Foreground aliases and default subagents use `local`;
the cheap model tier uses `local-fast`, with the server owning fallback to the
main model. Session resume refuses a different model/backend identity or reduced
context. Server restart can lose prompt caches and require prefill again.

Normal CLI tools work on the client computer. Search MCP prerequisites such as
`uvx` belong there as well. Existing Claude settings are preserved. A saved
setting that overrides the gateway connection is diagnosed before a turn; use a
compatible Claude configuration rather than editing server settings. An explicit
custom `--settings` remains authoritative and must include the Workflow timeout
hook for deterministic long-workflow handling.

Exiting Claude or losing the client connection does not stop ggrun. An interrupted
generation is not automatically replayed. Stop the server with Ctrl+C in its
serving terminal. Closing that terminal is not a promise of background
persistence; use your normal supported service manager for persistent hosting.

While serving, the terminal reports readiness, active main-model requests and
queued requests. It reads in-memory router counters every two seconds and prints
only changes; status observation does not poll the model backend. The configured
main limit is an upper bound, not a guarantee that all requests run concurrently.
During shutdown, new authenticated requests receive 503 before the model stops.

## Network examples

- **HTTPS proxy:** proxy a dedicated hostname or path to the loopback gateway.
  Preserve Anthropic headers, SSE streaming and cancellation; allow long idle
  prefill periods and disable response buffering. Set the advertised URL to the
  public-facing base URL.
- **SSH tunnel:** `ssh -N -L 18080:127.0.0.1:8080 user@server`, then connect to
  `http://127.0.0.1:18080` on the client. The API credential is still required.
- **LAN:** bind the gateway to your selected private interface and connect using
  its reachable URL. Remote cleartext HTTP needs `--allow-http`.
- **Optional Tailscale:** use an existing private network address or Tailscale
  Serve HTTPS to the loopback gateway. Tailscale is not installed, detected or
  configured by either program. Preserve existing Serve routes and select an
  unused endpoint. [Tailscale Serve documentation](https://tailscale.com/docs/reference/tailscale-cli/serve)
  describes the deployment controls.

The initial scope is the real Claude CLI and qualified foreground workflows.
Desktop, Remote Control, cloud sessions and supervisor-hosted background agents
need separate validation. Exact `/v1/messages/count_tokens` is not currently
advertised; the gateway returns a recognizable unsupported response.

## Compatibility contract

[`clientkit/contract.schema.json`](../clientkit/contract.schema.json) and its
fixture define connection protocol v1 and client policy v1. The shared Go module
constructs aliases, compaction/context, attribution and timeout policy for both
local and remote clients. It accepts descriptive fields, never arbitrary shell
commands or environment settings from the server. Contract changes require
versioning and a reviewed client dependency update.

The implementation follows the official [gateway protocol](https://code.claude.com/docs/en/llm-gateway-protocol),
[connection setup](https://code.claude.com/docs/en/llm-gateway-connect),
[model configuration](https://code.claude.com/docs/en/model-config) and
[settings precedence](https://code.claude.com/docs/en/env-vars) documentation.
ggrun owns validation of its local-model compatibility.
