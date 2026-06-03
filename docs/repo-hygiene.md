# Repository Hygiene

The public repository should look like a maintained product, not a scratch
workspace. Keep the root small, keep generated artifacts out of Git, and write
public docs as user-facing documentation.

## Root Layout

Allowed root entries:

```text
.github/
docs/
examples/
go/
scripts/
tests/
tools/
CHANGELOG.md
LICENSE
README.md
install.sh
setup.sh
setup-linux.sh
setup-mac.sh
```

Large generated files, local benchmark runs, model files, and old implementation
copies do not belong in the root.

## Commit Rules

- Do not commit GGUF model files, draft models, mmproj downloads, cache files, or
  local benchmark run directories.
- Commit benchmark data only as curated summaries with command, hardware, model,
  backend, context, and artifact path.
- Avoid public TODO, gap-analysis, or migration-notebook documents.
- Keep legacy code under `legacy/` only when it has an active migration or test
  purpose.
- Prefer Go implementations for product behavior. Python helpers are acceptable
  for GGUF/tooling until replaced.
- Public performance claims must have a reproducible benchmark command.

## Documentation Tone

Use direct engineering language. Avoid internal planning phrases such as
"gap analysis", "future plans", "big rework", or unverified claims. Describe
what exists, how it is configured, and how it was measured.
