# Contributing

ggrun is Go-first. Changes should preserve the public product layout and
include tests that match the risk of the change.

## Reporting bugs and proposing changes

Use the GitHub issue forms for reproducible bugs and focused feature requests.
For launch or performance reports, include ggrun version, redacted ggrun
detect output, model/quant, backend, context, parallelism, and the relevant
--dry-run output. Never attach model files, credentials, private paths, or
prompt contents.

Before opening a pull request:

```bash
cd go && go test ./...
bash -n install.sh scripts/*.sh setup.sh setup-linux.sh setup-mac.sh
python3 tests/test_parse_gguf.py
```

For performance changes, include the benchmark command, hardware, model, backend,
context size, and generated artifact path. Do not commit generated benchmark run
directories or model files.

## Commit messages

Use a `scope: lowercase summary` subject (e.g. `tune: protect KV-cache flags from
AI-tune`), with an optional body explaining the why. Keep messages human and
specific — no `Update X` placeholders, and no AI co-author or attribution
trailers in the public history.
