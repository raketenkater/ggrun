#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir/go"

unformatted="$(gofmt -l ./cmd/ggrun ./pkg/benchmark ./pkg/detect ./pkg/placement ./pkg/recovery ./pkg/server ./pkg/backends ./pkg/tui)"
if [[ -n "$unformatted" ]]; then
  echo "core engine has unformatted Go files:" >&2
  echo "$unformatted" >&2
  exit 1
fi

go test -count=1 ./cmd/ggrun ./pkg/benchmark ./pkg/detect ./pkg/placement ./pkg/recovery ./pkg/server ./pkg/backends ./pkg/tui

go vet ./cmd/ggrun ./pkg/benchmark ./pkg/detect ./pkg/placement ./pkg/recovery ./pkg/server ./pkg/backends ./pkg/tui
