# Contributing

llm-server is Go-first. Changes should preserve the public product layout and
include tests that match the risk of the change.

Before opening a pull request:

```bash
cd go && go test ./...
bash -n install.sh scripts/*.sh setup.sh setup-linux.sh setup-mac.sh
python3 tests/test_parse_gguf.py
```

For performance changes, include the benchmark command, hardware, model, backend,
context size, and generated artifact path. Do not commit generated benchmark run
directories or model files.

See `docs/repo-hygiene.md` for repository layout and public documentation rules.
