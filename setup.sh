#!/usr/bin/env bash
# One-command setup entrypoint for llm-server.
#
# Works from a cloned repo:
#   ./setup.sh
#
# Or remotely:
#   curl -fsSL https://raw.githubusercontent.com/raketenkater/llm-server/main/setup.sh | bash

set -euo pipefail

REPO="raketenkater/llm-server"
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
    command -v curl >/dev/null 2>&1 || { err "curl is required for remote setup"; exit 1; }
    command -v tar >/dev/null 2>&1 || { err "tar is required for remote setup"; exit 1; }
    TMP_DIR="$(mktemp -d -t llm-server-setup.XXXXXX)"
    local archive="$TMP_DIR/source.tar.gz"
    curl -fsSL "https://codeload.github.com/$REPO/tar.gz/$REF" -o "$archive"
    tar -xzf "$archive" -C "$TMP_DIR"
    find "$TMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1
}

ROOT="$(local_root || download_root)"
[[ -n "$ROOT" && -f "$ROOT/scripts/setup-home.sh" ]] || { err "could not locate setup files"; exit 1; }

case "$(uname -s)" in
    Linux)
        exec "$ROOT/scripts/setup-home.sh" linux "$@"
        ;;
    Darwin)
        exec "$ROOT/scripts/setup-home.sh" mac "$@"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        err "Native Windows is not supported. Use WSL2, then run this inside Ubuntu."
        exit 1
        ;;
    *)
        err "unsupported OS: $(uname -s)"
        exit 1
        ;;
esac
