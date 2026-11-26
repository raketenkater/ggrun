#!/usr/bin/env bash
# Fails when go/ holds a function that no binary or test reaches on any of
# Linux, Windows and macOS (deadcode sees one GOOS at a time, so only what all
# three report is dead). clientkit is reached from ggrun-client, not from this
# module; pkg/tune is removed with AI Tune (review W22).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../go"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
# Build it with the module's Go: a tool built by an older Go cannot load newer packages.
GOBIN="$tmp" GOTOOLCHAIN="$(go env GOVERSION)+auto" go install golang.org/x/tools/cmd/deadcode@v0.52.0
for os in linux windows darwin; do
  GOOS="$os" "$tmp/deadcode" -test ./... > "$tmp/$os.raw"
  { grep -v -E 'clientkit/|^pkg/tune/' "$tmp/$os.raw" || true; } | sed -E 's/:[0-9]+:[0-9]+:/:/' | sort -u > "$tmp/$os"
done
out="$(comm -12 "$tmp/linux" "$tmp/windows" | comm -12 - "$tmp/darwin")"
if [[ -n "$out" ]]; then
  printf '%s\n' "$out"
  echo "unreachable functions above: delete them, or call them" >&2
  exit 1
fi
echo "deadcode: nothing unreachable on linux, windows or darwin"
