# Contributing

Thanks for looking at ggrun. This file covers what we expect from a change.

ggrun is Go-first. Changes should preserve the public product layout and
include tests that match the risk of the change.

The [core development roadmap](docs/development-roadmap.md) is the active
engineering sequence. External-engine experiments, including FreeToken, stay
in their own parked lanes and are not core release dependencies.

## Building locally

```bash
cd go && go build ./cmd/ggrun
```

For the full clone-and-run setup (app home, models dir, config), see the
"From a clone" steps in [docs/install.md](docs/install.md#recommended-self-contained-app-home).

## CI-built Linux binary

Every CI run uploads the stamped Linux amd64 binary (14-day retention) as
`ggrun-linux-amd64-<12-char commit sha>`. Fetch and verify it:

```bash
gh run download <run-id> --repo raketenkater/ggrun --name ggrun-linux-amd64-<sha> --dir out
(cd out && sha256sum -c ggrun.sha256 && chmod +x ggrun && ./ggrun version)
```

## Reporting bugs and proposing changes

Use the GitHub issue forms for reproducible bugs and focused feature requests.
For launch or performance reports, include ggrun version, redacted ggrun
detect output, model/quant, backend, context, parallelism, and the relevant
--dry-run output. Never attach model files, credentials, private paths, or
prompt contents.

Not sure about something? Ask in [Discussions](https://github.com/raketenkater/ggrun/discussions);
open an [issue](https://github.com/raketenkater/ggrun/issues) once it's an actual bug or a
concrete proposal.

Before opening a pull request, run what CI actually gates merges on — all of it, not just the
Go tests:

```bash
cd go && go build ./... && go vet ./... && test -z "$(gofmt -l .)" && go test -race ./...
GOOS=windows GOARCH=amd64 go vet ./...
shellcheck --severity=error $(git ls-files '*.sh')
bash -n install.sh scripts/*.sh setup.sh setup-linux.sh setup-mac.sh
python3 tests/test_parse_gguf.py
for f in tests/test_estimator.sh tests/test_safety.sh tests/test_setup_home.sh tests/test_model_index.sh tests/test_moe_placement.sh; do bash "$f"; done
python3 -m pytest tests/test_downloader_quant.py tests/test_update_recommendations.py
```

The install/release-packaging and cross-platform build jobs in `ci.yml` stay CI-only — not
practical to reproduce locally — but the suites above are exactly the ones that fail silently
if you only ran `go test ./...`.

For performance changes, include the benchmark command, hardware, model, backend,
context size, and generated artifact path. Do not commit generated benchmark run
directories or model files.

## Commit messages

Use a `scope: lowercase summary` subject (e.g. `tune: protect KV-cache flags from
AI-tune`), with an optional body explaining the why. Keep messages human and
specific — no `Update X` placeholders, and no AI co-author or attribution
trailers in the public history.

## AI-assisted contributions

AI-assisted contributions are welcome — the code and the reasoning behind it are what get
reviewed, not how they were produced. Same rule as commit messages, extended to the PR
description: no AI-attribution trailers or disclaimers, just the change and why. Read your
own diff before opening the PR; you're accountable for it either way.
