#!/usr/bin/env bash
# One-command setup entrypoint for ggrun.
#
# Works from a cloned repo:
#   ./setup.sh
#
# Or remotely:
#   curl -fsSL https://raw.githubusercontent.com/raketenkater/ggrun/main/setup.sh | bash

set -euo pipefail

REPO="raketenkater/ggrun"
REF="${LLM_SETUP_REF:-main}"
TMP_DIR=""

cleanup() {
    [[ -n "$TMP_DIR" ]] && rm -rf "$TMP_DIR"
}
trap cleanup EXIT

err() { printf 'Error: %s\n' "$*" >&2; }

local_root() {
    local src="${BASH_SOURCE[0]:-}"
    [[ -n "$src" && -f "$src" ]] || return 1
    local dir
    dir="$(cd "$(dirname "$src")" 2>/dev/null && pwd -P)" || return 1
    [[ -f "$dir/scripts/setup-home.sh" ]] || return 1
    echo "$dir"
}

download_root() {
    command -v curl >/dev/null 2>&1 || { err "curl is required. Install curl with your OS package manager, then rerun setup."; exit 1; }
    command -v tar >/dev/null 2>&1 || { err "tar is required. Install tar with your OS package manager, then rerun setup."; exit 1; }
    TMP_DIR="$(mktemp -d -t ggrun-setup.XXXXXX)"
    local archive="$TMP_DIR/source.tar.gz"
    local url="https://codeload.github.com/$REPO/tar.gz/$REF"
    if ! curl -fL --retry 3 --connect-timeout 20 "$url" -o "$archive"; then
        err "could not download ggrun source from $url"
        err "Check internet/proxy access to github.com, or clone the repository and run ./setup.sh locally."
        exit 1
    fi
    if ! tar -xzf "$archive" -C "$TMP_DIR"; then
        err "downloaded source archive is invalid or could not be unpacked: $archive"
        exit 1
    fi
    find "$TMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1
}

ROOT="$(local_root || download_root)"
[[ -n "$ROOT" && -f "$ROOT/scripts/setup-home.sh" ]] || { err "could not locate setup files"; exit 1; }

case "$(uname -s)" in
    Linux)
        LLM_SETUP_REF="$REF" exec "$ROOT/scripts/setup-home.sh" linux "$@"
        ;;
    Darwin)
        LLM_SETUP_REF="$REF" exec "$ROOT/scripts/setup-home.sh" mac "$@"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        err "Use install.ps1 for native Windows installs, or run this Bash setup on Linux/macOS."
        exit 1
        ;;
    *)
        err "unsupported OS: $(uname -s)"
        exit 1
        ;;
esac
