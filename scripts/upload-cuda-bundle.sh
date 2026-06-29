#!/usr/bin/env bash
# Publish the prebuilt Linux x86_64 CUDA bundle to a GitHub release and refresh
# SHA256SUMS, so `install.sh` serves a CUDA backend with no toolchain.
#
# Build the bundle first (on a CUDA host) with scripts/build-linux-cuda-bundle.sh,
# or reuse an existing dist/ggrun-linux-x86_64-cuda.tar.gz. Then run this.
#
# Usage:
#   scripts/upload-cuda-bundle.sh [tag]      # tag defaults to the latest release
set -euo pipefail

REPO="${GGRUN_REPO:-raketenkater/ggrun}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="$ROOT/dist/ggrun-linux-x86_64-cuda.tar.gz"
ASSET="ggrun-linux-x86_64-cuda.tar.gz"

command -v gh >/dev/null 2>&1 || { echo "Error: gh CLI not found." >&2; exit 1; }
[[ -f "$BUNDLE" ]] || { echo "Error: $BUNDLE not found — build it with scripts/build-linux-cuda-bundle.sh" >&2; exit 1; }

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
    TAG="$(gh release view -R "$REPO" --json tagName --jq .tagName)"
fi
echo "==> Uploading $ASSET to $REPO release $TAG"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Refresh SHA256SUMS: pull the current one (if any), drop a stale CUDA line, append ours.
gh release download "$TAG" -R "$REPO" -p SHA256SUMS -D "$tmp" 2>/dev/null || : > "$tmp/SHA256SUMS"
[[ -f "$tmp/SHA256SUMS" ]] || : > "$tmp/SHA256SUMS"
sed -i "/  $ASSET\$/d" "$tmp/SHA256SUMS" 2>/dev/null || true
( cd "$ROOT/dist" && sha256sum "$ASSET" ) >> "$tmp/SHA256SUMS"

gh release upload "$TAG" -R "$REPO" "$BUNDLE" --clobber
gh release upload "$TAG" -R "$REPO" "$tmp/SHA256SUMS" --clobber

echo "==> Done. Linux NVIDIA installs will now fetch the prebuilt CUDA backend (no compile)."
echo "    sha256: $(cd "$ROOT/dist" && sha256sum "$ASSET" | cut -d' ' -f1)"
