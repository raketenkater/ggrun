# Remote Claude completion audit — October 28

The goal remains incomplete. This audit follows the implementation plan rather
than treating green tests or same-host requests as real laptop acceptance.
Public platform targets are clients on Linux/macOS/Windows and headless servers
on Linux/macOS. Existing Windows serving paths remain regression-covered.

| Requirement | Current evidence | Result / remaining gate |
| --- | --- | --- |
| TUI Off / local / remote, old saved launches and exact argv | Native TUI tests; actual Qwen launch from Remote client mode; saved-request replay | Implemented. TUI does not have a separate planner. |
| Normal placement, admission, backend selection and reviewer/worker policy | Existing launch pipeline, uncached core gate, real Qwen/reviewer run | Implemented for the verified scope; not universal hardware proof. |
| Stable gateway reserved before loading; no server Claude executable required | Port/loading/auth tests with empty PATH; actual headless launch with Claude absent from PATH | Verified. |
| Private model/companion listeners and authenticated narrow gateway | Flag tests, invalid-auth/private-route tests and actual HTTPS API receipt | Verified. Administration endpoints remain unavailable remotely. |
| Typed model/context/policy metadata; no executable commands from server | clientkit schema/fixture validation; real served context 144384; fixed stale-context/model headers | Verified for protocol/policy v1. |
| Messages, future fields, thinking/tool blocks, errors and unbuffered SSE | Contract/race tests; actual coding, worker and streaming runs | Verified for qualified Claude 2.1.307 and selected backend; future releases require qualification. |
| Exact token count capability | Explicit 501 and client estimation; real CLI completed work | Supported as an explicit absent capability; no fabricated exact counts. |
| Generic URLs, redirects, IPv6 parsing and reverse-proxy prefixes | Unit/native contracts; actual `/llm/` proxy; authenticated optional HTTPS deployment | Verified at those scopes. IPv6 parsing is not an external IPv6 network receipt. |
| Disconnect cancels work without stopping serving | Queued/active cancellation tests; actual GPU backend idle in 0.204 s and CPU backend idle in 0.255 s; server remained ready | Verified on this Linux run. No automatic turn replay. |
| Client private profiles and credentials | Unix mode/owner tests; native Windows DPAPI tests; native Mac tests | Verified natively on Linux/Windows and Apple Silicon Mac. Intel Mac is cross-built. |
| Real child, client CWD, settings/hooks/MCP/permission preservation | Native child-fixture tests, scoped settings tests, real Linux Claude CWD and tool execution | Physical laptop filesystem/tools and real Windows terminal remain unverified. |
| Workflow timeout helper and eligible foreground agents | Shared rewriting/invariant tests; actual foreground haiku-tier worker report and reviewer routing | Worker report correctness verified. Long Workflow execution on the laptop still required. |
| Worker handback survives template limitations | Actual two-turn HTTPS handoff, four real test cases accurately reported, unchanged oracle; documented 400 capability retry | Verified for Claude 2.1.307. Rejected prompt workaround is not shipped. |
| Session resume and reduced-context/model refusal | Native tests; actual published-alpha resume; actual reduced-context refusal before inference | Same-host verified; laptop disconnect/reconnect/resume still required. |
| Cached follow-ups and genuine context capacity | Actual cache-read usage, canary evidence, resolved per-slot context metadata | Verified for this backend/configuration; restart can lose prompt cache. |
| Serving status, queue and explicit Stop action | New in-memory counter view, change-only rendering, cancellation/error tests, activation wiring and shutdown 503 tests | Verified on real CPU-only 4B: successful SSE, queued/idle display, stopping 503, clean exit. Final 7f352a8 was then deployed with owner authorization. |
| Server process ownership and persistence | Clean main/companion stop, all GPUs at 1 MiB, clean relaunch; October 28 user service active with lingering and Restart=no | Verified on Linux. Optional deployment, not an automatically installed product service. |
| Shared policy parity | clientkit Environment used by local launcher, printed recipe and remote profile; context/alias tests | Implemented. Integration with overlapping PR #106 must retain this single policy source. |
| Separate repo, linked docs, compatibility matrix and changelog | Public client repo/PR, README links, schema, versioned alpha and updated changelogs | Delivered as an alpha. Both feature PRs remain drafts. |
| Released client install without Go | Actual alpha release pipeline and checksum-verified installer CI on all three OSes | Verified on native CI; actual laptop installation receipt still required. |
| Linux/macOS server checks and preserved Windows coverage | `3617440` and `a230daa` completed Linux/Windows E2E, native contracts and Mac packaging/default CPU launch; manual GPU CI skipped | Verified CI scopes. Real GPU evidence is from this Linux host; Metal inference remains unqualified. |
| Optional Tailscale without product dependency | Generic URL code, no Tailscale binary tests, preserved existing Serve routes and actual HTTPS requests | Verified same-host deployment. A physical laptop crossing this connection still needs its receipt. |
| Exact build/model identity and safe owner deployment | PATH candidate `v3.2.12-remote-7f352a8`, recorded SHA-256, rollback binary, unchanged owner preferences | Final build deployed after checking idle inference; rollback binaries retained. |
| Integration-base coordination | Main still `101af89`; scoped review of current worker #106 stacked on #102, with environment/resume/permission overlap notes | Review notes recorded; no unrelated worker stack merged into this feature. |
| Actual laptop read/edit/test, long workflow, reconnect/resume and server survival | User notified that the service is ready; actual device result requested | **Missing. Required before marking the goal complete.** |

Detailed receipts: [validation](remote-claude-validation-20261027.md). The owner
connection instructions and private deployment receipts are in ignored
`.cache/remote-claude/`; they contain no published gateway token.

Next actions: verify the final deployed endpoint, finish alpha.2 released
installer receipts and obtain the actual laptop receipt. Do not replace any of these with another
same-host coding run or a cross-build and call it complete.
