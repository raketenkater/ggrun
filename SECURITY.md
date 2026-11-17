# Security

## Reporting a vulnerability

Please open a [GitHub Security Advisory](https://github.com/raketenkater/ggrun/security/advisories/new)
on this repo instead of a public issue. Include ggrun version, platform, and
steps to reproduce.

## Scope

In scope: the ggrun launcher itself — install scripts (`install.sh`,
`install.ps1`, `setup.sh`), the CLI/TUI, and backend-launch argument handling;
the remote Claude gateway and router (`ggrun --claude-code`, `go/pkg/remoteclaude`,
`go/pkg/claudeauto`), including its authentication; and the client,
[ggrun-client](https://github.com/raketenkater/ggrun-client), with the shared
`clientkit` module.

Out of scope: llama.cpp and ik_llama.cpp themselves. Report issues in those
backends to their own projects.

## Please don't

Don't open a public issue or PR for an undisclosed vulnerability.
