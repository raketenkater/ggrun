#!/usr/bin/env bash
# Create a self-contained llm-server app home and install into it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLATFORM="${1:-}"
APP_NAME="${LLM_SETUP_APP_NAME:-llm-server}"
APP_HOME="${LLM_APP_HOME:-$HOME/$APP_NAME}"
INSTALL_MODE="${LLM_SETUP_MODE:-${LLM_INSTALL_MODE:-auto}}"
BACKEND="${LLM_SETUP_BACKEND:-${LLM_INSTALL_BACKEND:-auto}}"
PY_DEPS="${LLM_SETUP_PY_DEPS:-${LLM_INSTALL_PY_DEPS:-auto}}"
NONINTERACTIVE="${LLM_SETUP_NONINTERACTIVE:-${LLM_INSTALL_NONINTERACTIVE:-0}}"
LOG_TS="$(date +%Y%m%d-%H%M%S)"

say() { printf '%s\n' "$*"; }
err() { printf 'Error: %s\n' "$*" >&2; }

case "$PLATFORM" in
    linux|mac) ;;
    *) err "usage: scripts/setup-home.sh linux|mac"; exit 1 ;;
esac

OS="$(uname -s)"
case "$PLATFORM:$OS" in
    linux:Linux) ;;
    mac:Darwin) ;;
    linux:Darwin) err "setup-linux.sh is for Linux/WSL2. Use setup-mac.sh on macOS."; exit 1 ;;
    mac:Linux) err "setup-mac.sh is for macOS. Use setup-linux.sh on Linux/WSL2."; exit 1 ;;
    *) err "unsupported OS: $OS"; exit 1 ;;
esac

if [[ "$PLATFORM" == "mac" && "$BACKEND" == "auto" ]]; then
    BACKEND="metal"
fi

mkdir -p "$APP_HOME/bin" "$APP_HOME/models" "$APP_HOME/logs" "$APP_HOME/cache" "$APP_HOME/config" "$APP_HOME/src"
LOG_FILE="$APP_HOME/logs/setup-$LOG_TS.log"
exec > >(tee -a "$LOG_FILE") 2>&1

say "═══ $APP_NAME setup ($PLATFORM) ═══"
say "App home: $APP_HOME"
say "Logs:     $LOG_FILE"
say ""

LLM_INSTALL_PREFIX="$APP_HOME/bin" \
LLM_INSTALL_MODEL_DIR="$APP_HOME/models" \
LLM_INSTALL_BACKEND_ROOT="$APP_HOME/src" \
LLM_INSTALL_MODE="$INSTALL_MODE" \
LLM_INSTALL_BACKEND="$BACKEND" \
LLM_INSTALL_PY_DEPS="$PY_DEPS" \
LLM_INSTALL_NONINTERACTIVE="$NONINTERACTIVE" \
"$ROOT/install.sh"

cat >"$APP_HOME/config/config.sh" <<EOF
# $APP_NAME local config. This file is loaded automatically by launchers in $APP_HOME/bin.
export LLM_APP_HOME="$APP_HOME"
export LLM_MODEL_DIR="$APP_HOME/models"
export LLM_CACHE_DIR="$APP_HOME/cache"
export LLM_LOG_DIR="$APP_HOME/logs"
EOF

cat >"$APP_HOME/env.sh" <<EOF
# Source this to use $APP_NAME from any shell:
#   source "$APP_HOME/env.sh"
export LLM_APP_HOME="$APP_HOME"
export LLM_MODEL_DIR="$APP_HOME/models"
export LLM_CACHE_DIR="$APP_HOME/cache"
export LLM_LOG_DIR="$APP_HOME/logs"
export PATH="$APP_HOME/bin:\$PATH"
EOF

ln -sf "$APP_HOME/bin/llm-server" "$APP_HOME/run"
ln -sf "$APP_HOME/bin/llm-server-gui" "$APP_HOME/gui"

say ""
say "Done."
say "Use:"
say "  source \"$APP_HOME/env.sh\""
say "  llm-server-gui"
say "  llm-server <repo/name> --download"
say ""
say "Or without changing PATH:"
say "  \"$APP_HOME/gui\""
say "  \"$APP_HOME/run\" \"$APP_HOME/models/your-model.gguf\""
