#!/usr/bin/env bash
# Create a self-contained ggrun app home and install into it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLATFORM="${1:-}"
APP_NAME="${LLM_SETUP_APP_NAME:-ggrun}"
APP_HOME="${LLM_APP_HOME:-$HOME/$APP_NAME}"
APP_BIN="$APP_HOME/.bin"
APP_MODELS="$APP_HOME/models"
APP_LOGS="$APP_HOME/.logs"
APP_CACHE="$APP_HOME/.cache"
APP_CONFIG="$APP_HOME/.config"
APP_SRC="$APP_HOME/.src"
APP_ENV="$APP_HOME/.env.sh"

# Migrate a pre-rename install: ggrun was formerly llm-server. If an old
# ~/llm-server app home exists and ~/ggrun does not, move it over so the user's
# models, config, cache, and tuned configs carry forward.
OLD_APP_HOME="$HOME/llm-server"
if [[ "$APP_HOME" == "$HOME/ggrun" && -d "$OLD_APP_HOME" && ! -e "$APP_HOME" ]]; then
    printf '==> Migrating existing install: %s -> %s (formerly llm-server)\n' "$OLD_APP_HOME" "$APP_HOME"
    mv "$OLD_APP_HOME" "$APP_HOME"
fi
INSTALL_MODE="${LLM_SETUP_MODE:-${LLM_INSTALL_MODE:-auto}}"
BACKEND="${LLM_SETUP_BACKEND:-${LLM_INSTALL_BACKEND:-auto}}"
INSTALL_REF="${LLM_SETUP_REF:-${LLM_INSTALL_REF:-main}}"
SOURCE_REPO_DIR=""
if [[ ! -d "$ROOT/.git" ]]; then
    SOURCE_REPO_DIR="$APP_SRC/ggrun"
fi
PY_DEPS="${LLM_SETUP_PY_DEPS:-${LLM_INSTALL_PY_DEPS:-auto}}"
DEPS="${LLM_SETUP_DEPS:-${LLM_INSTALL_DEPS:-auto}}"
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
    linux:Darwin) err "setup-linux.sh is for Linux. Use setup-mac.sh on macOS."; exit 1 ;;
    mac:Linux) err "setup-mac.sh is for macOS. Use setup-linux.sh on Linux."; exit 1 ;;
    *) err "unsupported OS: $OS"; exit 1 ;;
esac

if [[ "$PLATFORM" == "mac" && "$BACKEND" == "auto" ]]; then
    BACKEND="metal"
fi

mkdir -p "$APP_BIN" "$APP_MODELS" "$APP_LOGS" "$APP_CACHE" "$APP_CONFIG" "$APP_SRC"
LOG_FILE="$APP_LOGS/setup-$LOG_TS.log"
exec > >(tee -a "$LOG_FILE") 2>&1

say "═══ $APP_NAME setup ($PLATFORM) ═══"
say "App home: $APP_HOME"
say "Logs:     $LOG_FILE"
say ""

LLM_INSTALL_PREFIX="$APP_BIN" \
LLM_INSTALL_MODEL_DIR="$APP_MODELS" \
LLM_INSTALL_BACKEND_ROOT="$APP_SRC" \
LLM_INSTALL_REPO_DIR="$SOURCE_REPO_DIR" \
LLM_INSTALL_REF="$INSTALL_REF" \
LLM_INSTALL_MODE="$INSTALL_MODE" \
LLM_INSTALL_BACKEND="$BACKEND" \
LLM_INSTALL_PY_DEPS="$PY_DEPS" \
LLM_INSTALL_DEPS="$DEPS" \
LLM_INSTALL_NONINTERACTIVE="$NONINTERACTIVE" \
LLM_INSTALL_MAIN=go \
"$ROOT/install.sh"

if [[ ! -x "$APP_BIN/ggrun" ]]; then
    err "ggrun launcher was not installed. See log: $LOG_FILE"
    exit 1
fi


backend_bin=""
if [[ -x "$APP_BIN/llama-server-cuda" ]]; then
    backend_bin="$APP_BIN/llama-server-cuda"
elif [[ -x "$APP_BIN/ik_llama-server-cuda" ]]; then
    backend_bin="$APP_BIN/ik_llama-server-cuda"
elif [[ -x "$APP_BIN/llama-server-vulkan" ]]; then
    backend_bin="$APP_BIN/llama-server-vulkan"
elif [[ -x "$APP_BIN/llama-server" ]]; then
    backend_bin="$APP_BIN/llama-server"
elif [[ -x "$APP_SRC/llama.cpp/build-cuda/bin/llama-server" ]]; then
    backend_bin="$APP_SRC/llama.cpp/build-cuda/bin/llama-server"
elif [[ -x "$APP_SRC/ik_llama.cpp/build/bin/llama-server" ]]; then
    backend_bin="$APP_SRC/ik_llama.cpp/build/bin/llama-server"
elif [[ -x "$APP_SRC/llama.cpp/build-vulkan/bin/llama-server" ]]; then
    backend_bin="$APP_SRC/llama.cpp/build-vulkan/bin/llama-server"
elif [[ -x "$APP_SRC/llama.cpp/build/bin/llama-server" ]]; then
    backend_bin="$APP_SRC/llama.cpp/build/bin/llama-server"
fi

backend_real="$backend_bin"
if [[ -n "$backend_bin" ]]; then
    backend_real="$(readlink -f "$backend_bin" 2>/dev/null || printf '%s' "$backend_bin")"
fi

backend_config="$BACKEND"
if [[ "$backend_config" == "auto" ]]; then
    if [[ "$backend_real" == *ik_llama.cpp* ]]; then
        backend_config="ik_llama"
    elif [[ "$backend_real" == *vulkan* || "$backend_real" == *build-vulkan* ]]; then
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

cat >"$APP_CONFIG/config" <<EOF
# $APP_NAME Go config. Loaded when LLM_APP_HOME points at this app home.
LLM_APP_HOME="$APP_HOME"
LLM_MODEL_DIR="$APP_MODELS"
LLM_CACHE_DIR="$APP_CACHE"
LLM_LOG_DIR="$APP_LOGS"
LLM_BACKEND="$backend_config"
EOF
if [[ -n "$backend_bin" ]]; then
    printf 'LLAMA_SERVER="%s"\n' "$backend_bin" >>"$APP_CONFIG/config"
fi

cat >"$APP_ENV" <<EOF
# Source this to use $APP_NAME from any shell:
#   source "$APP_ENV"
#
# Only LLM_APP_HOME and PATH are exported. $APP_NAME reads model dir, backend,
# cache, logs and the llama-server path from its config file
# ($APP_CONFIG/config), so CLI/GUI edits take effect instead of being shadowed
# by stale environment variables.
export LLM_APP_HOME="$APP_HOME"
export PATH="$APP_BIN:\$PATH"
EOF

cat >"$APP_HOME/ggrun" <<EOF
#!/usr/bin/env bash
source "$APP_ENV"
exec "$APP_BIN/ggrun" "\$@"
EOF
chmod 0755 "$APP_HOME/ggrun"

# Backward-compat: keep the old `llm-server` command working for existing users.
ln -sf ggrun "$APP_BIN/llm-server" 2>/dev/null || true
ln -sf "$APP_HOME/ggrun" "$APP_HOME/llm-server" 2>/dev/null || true

say ""
say "╔════════════════════════════════════════════════════════════╗"
say "║ ggrun is installed and ready                         ║"
say "╚════════════════════════════════════════════════════════════╝"
say "Backend:   ${backend_bin:-not installed}"
say "CLI:       $APP_HOME/ggrun"
say "GUI:       $APP_HOME/ggrun   (no arguments opens the GUI)"
say "Models:    $APP_MODELS"
say "Config:    $APP_CONFIG/config"
say "Logs:      $APP_LOGS"
say ""
say "Try now:"
say "  \"$APP_HOME/ggrun\"            # interactive GUI"
say "  \"$APP_HOME/ggrun\" detect"
say "  \"$APP_HOME/ggrun\" <repo/name> --download"
say "  \"$APP_HOME/ggrun\" \"$APP_MODELS/your-model.gguf\""
say ""
if [[ -n "$SOURCE_REPO_DIR" ]]; then
    say "Source:    $SOURCE_REPO_DIR"
fi
say "Internals: $APP_BIN, $APP_CACHE, $APP_SRC"
