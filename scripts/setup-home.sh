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

GUIDE_DOWNLOAD_MODEL=0
GUIDE_PATH=0

ask_tty() {
    local p="$1" d="${2:-n}" reply
    if [[ "$NONINTERACTIVE" == "1" || ! -r /dev/tty ]]; then
        [[ "$d" == "y" ]]
        return
    fi
    read -r -p "$p " reply </dev/tty || reply=""
    reply="${reply:-$d}"
    [[ "$reply" =~ ^[Yy] ]]
}

read_tty() {
    local prompt="$1" default="$2" reply
    if [[ "$NONINTERACTIVE" == "1" || ! -r /dev/tty ]]; then
        printf '%s\n' "$default"
        return
    fi
    if [[ -n "$default" ]]; then
        read -r -p "$prompt [$default]: " reply </dev/tty || reply=""
    else
        read -r -p "$prompt: " reply </dev/tty || reply=""
    fi
    printf '%s\n' "${reply:-$default}"
}

if [[ "$NONINTERACTIVE" != "1" && -r /dev/tty ]]; then
    say ""
    say "ggrun setup"
    say "This install will put a working ggrun and a local model server on this machine."
    say "Press Enter to keep a default."
    say ""
    APP_HOME="$(read_tty "Install directory" "$APP_HOME")"
    APP_BIN="$APP_HOME/.bin"
    APP_MODELS="$(read_tty "Model directory" "$APP_HOME/models")"
    APP_LOGS="$APP_HOME/.logs"
    APP_CACHE="$APP_HOME/.cache"
    APP_CONFIG="$APP_HOME/.config"
    APP_SRC="$APP_HOME/.src"
    APP_ENV="$APP_HOME/.env.sh"

    say ""
    say "Backend: auto uses llama.cpp / ik_llama.cpp / forks / CUDA already on"
    say "this machine, then installs what is missing (Vulkan loader, CUDA toolkit,"
    say "prebuilts, or a CUDA source build if nvcc is here)."
    if [[ -x "$ROOT/install.sh" ]]; then
        disc="$(mktemp)"
        if LLM_INSTALL_NONINTERACTIVE=1 "$ROOT/install.sh" --discover >"$disc" 2>/dev/null; then
            if [[ -s "$disc" ]]; then
                say ""
                say "Already on this machine:"
                while IFS='=' read -r k v; do
                    case "$k" in
                        nvidia_smi) say "  NVIDIA driver: $v" ;;
                        nvcc) say "  CUDA toolkit:  $v" ;;
                        ik_llama) say "  ik_llama.cpp:  $v" ;;
                        llama_vulkan) say "  llama.cpp Vulkan: $v" ;;
                        llama_cuda) say "  llama.cpp CUDA: $v" ;;
                        llama) say "  llama.cpp:     $v" ;;
                        fork) say "  fork:          $v" ;;
                    esac
                done <"$disc"
            fi
        fi
        rm -f "$disc"
    fi
    if [[ "$PLATFORM" == "linux" ]] && ! command -v nvidia-smi >/dev/null 2>&1 \
        && [[ ! -x /usr/bin/nvidia-smi && ! -e /dev/nvidia0 ]]; then
        say "No NVIDIA driver found. CUDA needs one. auto will use Vulkan or CPU if needed."
    fi
    reply="$(read_tty "Backend [auto/cuda/vulkan/cpu/metal/skip]" "$BACKEND")"
    case "$reply" in
        auto|cuda|vulkan|cpu|metal|skip) BACKEND="$reply" ;;
        *) say "Using $BACKEND" ;;
    esac
    if [[ "$BACKEND" == "cuda" ]] && ! command -v nvidia-smi >/dev/null 2>&1 \
        && [[ ! -x /usr/bin/nvidia-smi && ! -e /dev/nvidia0 ]]; then
        say "CUDA was requested but no NVIDIA driver was found. Install the driver, or pick auto."
        if ask_tty "Switch to auto (Vulkan/CPU if CUDA cannot be downloaded)? [Y/n]" y; then
            BACKEND="auto"
        fi
    fi
    if ask_tty "Download a model that fits this machine after install? [Y/n]" y; then
        GUIDE_DOWNLOAD_MODEL=1
    fi
    if ask_tty "Add ggrun to PATH in your shell rc? [Y/n]" y; then
        GUIDE_PATH=1
    fi
    export LLM_INSTALL_PROMPT=0
    export LLM_INSTALL_PY_DEPS=install
    say ""
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


usable_llama_server() {
    local p="$1" t hdr
    [[ -n "$p" && -e "$p" ]] || return 1
    t="$(readlink -f "$p" 2>/dev/null || printf '%s' "$p")"
    [[ -f "$t" && -x "$t" ]] || return 1
    case "$(basename "$t")" in
        llama-server|llama-server.exe|ik_llama-server|ik_llama-server-cuda|ik_llama-server-vulkan|llama-server-vulkan|llama-server-cuda|llama-server-vulkan.exe|llama-server-cuda.exe) ;;
        *) return 1 ;;
    esac
    case "$t" in
        *simulator*|*llama-eval*|*/examples/*) return 1 ;;
    esac
    hdr="$(head -c 4 "$t" 2>/dev/null || true)"
    [[ "$hdr" == $'\x7fELF' || "${hdr:0:2}" == "MZ" ]] && return 0
    case "$hdr" in
        $'\xfe\xed\xfa\xcf'|$'\xcf\xfa\xed\xfe'|$'\xfe\xed\xfa\xce'|$'\xce\xfa\xed\xfe'|$'\xca\xfe\xba\xbe'|$'\xbe\xba\xfe\xca'|$'\xca\xfe\xba\xbf'|$'\xbf\xba\xfe\xca') return 0 ;;
    esac
    return 1
}

# Prefer a backend that can actually start. A CUDA ELF that cannot load
# libnccl must not hide a working Vulkan/CPU llama-server.
backend_starts() {
    local p="$1" t dir
    usable_llama_server "$p" || return 1
    t="$(readlink -f "$p" 2>/dev/null || printf '%s' "$p")"
    dir="$(dirname "$t")"
    if command -v timeout >/dev/null 2>&1; then
        timeout 8 env LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$t" --version >/dev/null 2>&1
    else
        env LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$t" --version >/dev/null 2>&1
    fi
}

backend_candidates() {
    printf '%s\n' \
        "$APP_BIN/llama-server-cuda" \
        "$APP_BIN/ik_llama-server-cuda" \
        "$APP_BIN/llama-server-vulkan" \
        "$APP_BIN/llama-server" \
        "$APP_SRC/llama.cpp/build-cuda/bin/llama-server" \
        "$APP_SRC/ik_llama.cpp/build/bin/llama-server" \
        "$APP_SRC/llama.cpp/build-vulkan/bin/llama-server" \
        "$APP_SRC/llama.cpp/build/bin/llama-server"
}

backend_bin=""
while IFS= read -r cand; do
    if backend_starts "$cand"; then
        backend_bin="$cand"
        break
    fi
done < <(backend_candidates)
if [[ -z "$backend_bin" ]]; then
    while IFS= read -r cand; do
        if usable_llama_server "$cand"; then
            backend_bin="$cand"
            break
        fi
    done < <(backend_candidates)
fi

backend_real="$backend_bin"
if [[ -n "$backend_bin" ]]; then
    backend_real="$(readlink -f "$backend_bin" 2>/dev/null || printf '%s' "$backend_bin")"
fi

backend_config="$BACKEND"
if [[ "$backend_config" == "auto" ]]; then
    # Both servers may be present. Leave auto so ggrun can pick per model.
    if [[ -e "$APP_BIN/ik_llama-server-cuda" && -e "$APP_BIN/llama-server-vulkan" ]]; then
        backend_config="auto"
    elif [[ "$backend_real" == *ik_llama.cpp* || -e "$APP_BIN/ik_llama-server-cuda" ]]; then
        backend_config="ik_llama"
    elif [[ "$backend_real" == *vulkan* || -e "$APP_BIN/llama-server-vulkan" ]]; then
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
elif [[ "$backend_config" == "skip" ]]; then
    # "skip" controls installation only; it is not a runtime backend tag.
    backend_config="auto"
fi

# Upgrades must retain user choices; defaults belong to the first installation.
if [[ ! -e "$APP_CONFIG/config" ]]; then
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

if ! LLM_APP_HOME="$APP_HOME" "$APP_BIN/ggrun" version >/dev/null 2>&1; then
    err "Installed ggrun failed its version check. See log: $LOG_FILE"
    exit 1
fi
if ! LLM_APP_HOME="$APP_HOME" "$APP_BIN/ggrun" detect >/dev/null 2>&1; then
    err "Installed ggrun failed hardware detection. See log: $LOG_FILE"
    exit 1
fi
if [[ -n "$backend_bin" ]]; then
    backend_dir="$(dirname "$(readlink -f "$backend_bin" 2>/dev/null || printf '%s' "$backend_bin")")"
    if ! env LD_LIBRARY_PATH="$backend_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$backend_bin" --version >/dev/null 2>&1; then
        say "  ⚠ Backend at $backend_bin did not start (--version). GPU driver/ICD may be missing; ggrun still runs if a CPU backend is installed."
    fi
fi
ok_msg="CLI, hardware detection, and backend startup checks passed"
say "  ✓ $ok_msg"

if (( GUIDE_DOWNLOAD_MODEL )); then
    say ""
    say "── Model that fits this machine ──"
    LLM_APP_HOME="$APP_HOME" "$APP_BIN/ggrun" recommend -n 3 || true
    first_repo="$(LLM_APP_HOME="$APP_HOME" "$APP_BIN/ggrun" recommend --first 2>/dev/null | tail -n 1 || true)"
    repo="$(read_tty "Hugging Face repo to download (empty skips)" "$first_repo")"
    if [[ -n "$repo" ]]; then
        say "Downloading $repo …"
        if LLM_APP_HOME="$APP_HOME" "$APP_BIN/ggrun" download "$repo"; then
            say "  Model download finished"
        else
            warn "Download failed. You can retry: $APP_HOME/ggrun download $repo"
        fi
    fi
fi

if (( GUIDE_PATH )); then
    SHELL_RC="$HOME/.bashrc"
    [[ "$(uname -s)" == "Darwin" ]] && SHELL_RC="$HOME/.zshrc"
    line="source \"$APP_ENV\""
    if [[ -f "$SHELL_RC" ]] && grep -Fqs "$APP_ENV" "$SHELL_RC"; then
        say "bashrc already sources $APP_ENV (new terminals get ggrun)"
    else
        printf '\n# ggrun\n%s\n' "$line" >>"$SHELL_RC"
        say "Added $line to $SHELL_RC"
    fi
    mkdir -p "$HOME/.local/bin"
    ln -sfn "$APP_HOME/ggrun" "$HOME/.local/bin/ggrun"
    ln -sfn "$APP_HOME/ggrun" "$HOME/.local/bin/llm-server" 2>/dev/null || true
    hash -r 2>/dev/null || true
    say "Linked $HOME/.local/bin/ggrun -> $APP_HOME/ggrun"
fi

say ""
say "╔════════════════════════════════════════════════════════════╗"
say "║ ggrun is installed and ready                               ║"
say "╚════════════════════════════════════════════════════════════╝"
say "Backend:   ${backend_bin:-not installed}"
say "CLI:       $APP_HOME/ggrun"
say "GUI:       $APP_HOME/ggrun   (no arguments opens the GUI)"
say "Models:    $APP_MODELS"
say "Config:    $APP_CONFIG/config"
say "Logs:      $APP_LOGS"
say ""
say "Try now:"
if command -v ggrun >/dev/null 2>&1; then
    say "  ggrun            # interactive GUI"
    say "  ggrun detect"
else
    say "  This shell does not have ggrun on PATH yet. Run:"
    say "    source \"$APP_ENV\""
    say "  or:"
    say "    \"$APP_HOME/ggrun\""
fi
say "  \"$APP_HOME/ggrun\" detect"
say "  \"$APP_HOME/ggrun\" <repo/name> --download"
say "  \"$APP_HOME/ggrun\" \"$APP_MODELS/your-model.gguf\""
say ""
if [[ -n "$SOURCE_REPO_DIR" ]]; then
    say "Source:    $SOURCE_REPO_DIR"
fi
say "Internals: $APP_BIN, $APP_CACHE, $APP_SRC"
