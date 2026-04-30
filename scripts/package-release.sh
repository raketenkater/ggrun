#!/usr/bin/env bash
#
# Build a self-contained release archive for install.sh.
#
# Usage:
#   scripts/package-release.sh <asset-name> <llama-server-path> <output-dir>

set -euo pipefail

ASSET_NAME="${1:-}"
SERVER_BIN="${2:-}"
OUT_DIR="${3:-dist}"

if [[ -z "$ASSET_NAME" || -z "$SERVER_BIN" ]]; then
    echo "Usage: $0 <asset-name> <llama-server-path> <output-dir>" >&2
    exit 2
fi
if [[ ! -x "$SERVER_BIN" ]]; then
    echo "Error: llama-server binary not executable: $SERVER_BIN" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
WORK_DIR="$(mktemp -d -t llm-server-package.XXXXXX)"
PAYLOAD="$WORK_DIR/${ASSET_NAME%.tar.gz}"

cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

mkdir -p "$PAYLOAD/bin"

for f in setup.sh setup-linux.sh setup-mac.sh llm-server llm-server-mac llm-server-gui parse_gguf.py model_index.py download_any_gguf.py LICENSE README.md CHANGELOG.md; do
    [[ -f "$ROOT_DIR/$f" ]] || continue
    install -m 0644 "$ROOT_DIR/$f" "$PAYLOAD/$f"
done
chmod 0755 "$PAYLOAD/llm-server" "$PAYLOAD/llm-server-mac" "$PAYLOAD/llm-server-gui" \
    "$PAYLOAD/setup.sh" "$PAYLOAD/setup-linux.sh" "$PAYLOAD/setup-mac.sh" \
    "$PAYLOAD/parse_gguf.py" "$PAYLOAD/model_index.py" "$PAYLOAD/download_any_gguf.py" 2>/dev/null || true

install -m 0755 "$SERVER_BIN" "$PAYLOAD/bin/llama-server"

BIN_DIR="$(cd "$(dirname "$SERVER_BIN")" && pwd)"
while IFS= read -r lib; do
    install -m 0644 "$lib" "$PAYLOAD/bin/$(basename "$lib")"
done < <(find "$BIN_DIR" -maxdepth 1 -type f \( -name 'lib*.so*' -o -name 'lib*.dylib' -o -name '*.dll' \) 2>/dev/null | sort)

(
    cd "$WORK_DIR"
    tar -czf "$OUT_DIR/$ASSET_NAME" "$(basename "$PAYLOAD")"
)

echo "$OUT_DIR/$ASSET_NAME"
