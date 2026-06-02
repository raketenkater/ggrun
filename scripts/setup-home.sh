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
LLM_INSTALL_MAIN=go \
"$ROOT/install.sh"

backend_bin=""
if [[ -x "$APP_HOME/bin/llama-server" ]]; then
    backend_bin="$APP_HOME/bin/llama-server"
elif [[ -x "$APP_HOME/src/ik_llama.cpp/build/bin/llama-server" ]]; then
    backend_bin="$APP_HOME/src/ik_llama.cpp/build/bin/llama-server"
elif [[ -x "$APP_HOME/src/llama.cpp/build-vulkan/bin/llama-server" ]]; then
    backend_bin="$APP_HOME/src/llama.cpp/build-vulkan/bin/llama-server"
elif [[ -x "$APP_HOME/src/llama.cpp/build/bin/llama-server" ]]; then
    backend_bin="$APP_HOME/src/llama.cpp/build/bin/llama-server"
fi

backend_config="$BACKEND"
if [[ "$backend_config" == "auto" ]]; then
    if [[ "$backend_bin" == *ik_llama.cpp* ]]; then
        backend_config="ik_llama"
    elif [[ "$backend_bin" == *vulkan* ]]; then
        backend_config="vulkan"
    elif [[ "$PLATFORM" == "mac" ]]; then
        backend_config="llama"
    else
        backend_config="llama"
    fi
elif [[ "$backend_config" == "cuda" ]]; then
    backend_config="ik_llama"
elif [[ "$backend_config" == "cpu" || "$backend_config" == "metal" ]]; then
    backend_config="llama"
fi

cat >"$APP_HOME/config/config" <<EOF
# $APP_NAME Go config. Loaded when LLM_APP_HOME points at this app home.
LLM_APP_HOME="$APP_HOME"
LLM_MODEL_DIR="$APP_HOME/models"
LLM_CACHE_DIR="$APP_HOME/cache"
LLM_LOG_DIR="$APP_HOME/logs"
LLM_BACKEND="$backend_config"
EOF
if [[ -n "$backend_bin" ]]; then
    printf 'LLAMA_SERVER="%s"\n' "$backend_bin" >>"$APP_HOME/config/config"
fi

cat >"$APP_HOME/config/config.sh" <<EOF
# $APP_NAME shell config. Sourced by env.sh and wrapper commands.
export LLM_APP_HOME="$APP_HOME"
export LLM_MODEL_DIR="$APP_HOME/models"
export LLM_CACHE_DIR="$APP_HOME/cache"
export LLM_LOG_DIR="$APP_HOME/logs"
export LLM_BACKEND="$backend_config"
EOF
if [[ -n "$backend_bin" ]]; then
    printf 'export LLAMA_SERVER="%s"\n' "$backend_bin" >>"$APP_HOME/config/config.sh"
fi

cat >"$APP_HOME/env.sh" <<EOF
# Source this to use $APP_NAME from any shell:
#   source "$APP_HOME/env.sh"
source "$APP_HOME/config/config.sh"
export PATH="$APP_HOME/bin:\$PATH"
EOF

cat >"$APP_HOME/run" <<EOF
#!/usr/bin/env bash
source "$APP_HOME/env.sh"
exec "$APP_HOME/bin/llm-server" "\$@"
EOF
chmod 0755 "$APP_HOME/run"

cat >"$APP_HOME/gui" <<EOF
#!/usr/bin/env bash
source "$APP_HOME/env.sh"
exec "$APP_HOME/bin/llm-server-gui" "\$@"
EOF
chmod 0755 "$APP_HOME/gui"

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
