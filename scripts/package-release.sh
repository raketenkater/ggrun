#!/usr/bin/env bash
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
WORK_DIR="$(mktemp -d -t ggrun-package.XXXXXX)"
PAYLOAD="$WORK_DIR/${ASSET_NAME%.tar.gz}"

cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

mkdir -p "$PAYLOAD/bin"

for f in LICENSE README.md CHANGELOG.md; do
    [[ -f "$ROOT_DIR/$f" ]] && install -m 0644 "$ROOT_DIR/$f" "$PAYLOAD/$f"
done
for f in setup.sh setup-linux.sh setup-mac.sh; do
    [[ -f "$ROOT_DIR/$f" ]] && install -m 0755 "$ROOT_DIR/$f" "$PAYLOAD/$f"
done
[[ -f "$ROOT_DIR/install.ps1" ]] && install -m 0644 "$ROOT_DIR/install.ps1" "$PAYLOAD/install.ps1"

install -m 0755 "$SERVER_BIN" "$PAYLOAD/bin/llama-server"

if [[ -x "$ROOT_DIR/go/ggrun" ]]; then
    install -m 0755 "$ROOT_DIR/go/ggrun" "$PAYLOAD/bin/ggrun"
fi
if [[ -f "$ROOT_DIR/legacy/bash/ggrun" ]]; then
    install -m 0755 "$ROOT_DIR/legacy/bash/ggrun" "$PAYLOAD/llm-server-bash"
fi

for spec in \
    "tools/gguf/parse_gguf.py:parse_gguf.py" \
    "tools/models/model_index.py:model_index.py" \
    "tools/download/download_any_gguf.py:download_any_gguf.py"; do
    src="${spec%%:*}"
    dst="${spec##*:}"
    [[ -f "$ROOT_DIR/$src" ]] && install -m 0755 "$ROOT_DIR/$src" "$PAYLOAD/bin/$dst"
done

BIN_DIR="$(cd "$(dirname "$SERVER_BIN")" && pwd)"
while IFS= read -r lib; do
    install -m 0644 "$lib" "$PAYLOAD/bin/$(basename "$lib")"
done < <(find "$BIN_DIR" -maxdepth 1 -type f \( -name 'lib*.so*' -o -name 'lib*.dylib' -o -name '*.dll' \) 2>/dev/null | sort)

# Shared-library IK builds keep runtime libraries outside build/bin. Copy the
# project libraries referenced by the server under the names requested by its
# dynamic dependencies so the archive remains relocatable.
if command -v ldd >/dev/null 2>&1; then
    while IFS='|' read -r soname lib; do
        [[ -n "$soname" && -f "$lib" ]] || continue
        install -m 0644 "$lib" "$PAYLOAD/bin/$soname"
    done < <(
        ldd "$SERVER_BIN" 2>/dev/null |
            awk '$1 ~ /^lib(ggml|llama|mtmd)/ && $2 == "=>" && $3 ~ "^/" { print $1 "|" $3 }'
    )
fi

(
    cd "$WORK_DIR"

    tar -czf "$OUT_DIR/$ASSET_NAME" "$(basename "$PAYLOAD")"
)

echo "$OUT_DIR/$ASSET_NAME"
