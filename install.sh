#!/usr/bin/env bash
#
# install.sh — One-command installer for llm-server.
#
# Usage (remote):
#   curl -fsSL https://raw.githubusercontent.com/raketenkater/llm-server/main/install.sh | bash
# Usage (local):
#   ./install.sh                  # from a cloned repo
#
# Installs scripts to ~/.local/bin. Optionally clones + builds a llama.cpp
# backend (ik_llama.cpp for CUDA, llama.cpp for Vulkan/Metal/CPU).
#
# Flags (env vars):
#   LLM_INSTALL_BACKEND=auto|cuda|vulkan|metal|cpu|skip   default: auto
#   LLM_INSTALL_MODE=auto|release|build|scripts           default: auto
#   LLM_INSTALL_RELEASE=latest|vX.Y.Z                      default: latest
#   LLM_INSTALL_RELEASE_DIR=<dir>                          local bundle dir (tests/offline)
#   LLM_INSTALL_PREFIX=<dir>                              default: ~/.local/bin
#   LLM_INSTALL_MODEL_DIR=<dir>                           default: ~/ai_models
#   LLM_INSTALL_BACKEND_ROOT=<dir>                         default: ~
#   LLM_INSTALL_PY_DEPS=auto|install|skip                  default: auto
#   LLM_INSTALL_NONINTERACTIVE=1                          skip prompts

set -euo pipefail

REPO_URL="https://github.com/raketenkater/llm-server.git"
RAW_URL="https://raw.githubusercontent.com/raketenkater/llm-server/main"
GITHUB_REPO="raketenkater/llm-server"
INSTALL_DIR="${LLM_INSTALL_PREFIX:-$HOME/.local/bin}"
MODEL_DIR="${LLM_INSTALL_MODEL_DIR:-$HOME/ai_models}"
BACKEND_ROOT="${LLM_INSTALL_BACKEND_ROOT:-$HOME}"
BACKEND_CHOICE="${LLM_INSTALL_BACKEND:-auto}"
INSTALL_MODE="${LLM_INSTALL_MODE:-auto}"
INSTALL_RELEASE="${LLM_INSTALL_RELEASE:-latest}"
INSTALL_RELEASE_DIR="${LLM_INSTALL_RELEASE_DIR:-}"
PY_DEPS_MODE="${LLM_INSTALL_PY_DEPS:-auto}"
NONINTERACTIVE="${LLM_INSTALL_NONINTERACTIVE:-0}"
[[ ! -t 0 ]] && NONINTERACTIVE=1   # piped via curl | bash → no stdin

say()  { printf '%s\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m⚠\033[0m %s\n' "$*"; }
err()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; }
ask()  { # ask "prompt" default_yn
    local p="$1" d="${2:-n}" reply
    if (( NONINTERACTIVE )); then [[ "$d" == "y" ]]; return; fi
    read -r -p "$p " reply </dev/tty || reply=""
    reply="${reply:-$d}"
    [[ "$reply" =~ ^[Yy] ]]
}

case "$INSTALL_MODE" in
    auto|release|build|scripts) ;;
    *) err "unknown install mode: $INSTALL_MODE"; exit 1 ;;
esac
case "$PY_DEPS_MODE" in
    auto|install|skip) ;;
    *) err "unknown python dependency mode: $PY_DEPS_MODE"; exit 1 ;;
esac

say "═══ llm-server installer ═══"

# ── Stage 1: use local repo if present; clone only if source fallback needs it ──
SRC_DIR=""
if [[ -f "./llm-server" && -f "./llm-server-gui" ]]; then
    SRC_DIR="$(pwd)"
    ok "Using local repo at $SRC_DIR"
fi

ensure_source_repo() {
    [[ -n "$SRC_DIR" ]] && return 0
    command -v git >/dev/null || { err "git required to fetch repo"; exit 1; }
    SRC_DIR="$(mktemp -d -t llm-server-install.XXXXXX)"
    say "── Cloning $REPO_URL ──"
    git clone --depth=1 "$REPO_URL" "$SRC_DIR" >/dev/null 2>&1 && ok "Cloned to $SRC_DIR" \
        || { err "git clone failed"; exit 1; }
    trap 'rm -rf "$SRC_DIR"' EXIT
}

# ── Stage 2: detect platform + backend ──────────────────────────────────────
OS="$(uname -s)"
detect_backend() {
    if [[ "$OS" == "Darwin" ]]; then echo metal; return; fi
    if [[ "$OS" == MINGW* || "$OS" == MSYS* || "$OS" == CYGWIN* ]]; then
        err "Native Windows is not supported. Use WSL2, then run this installer inside Ubuntu."
        exit 1
    fi
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
        echo cuda; return
    fi
    if command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary 2>/dev/null | grep -qi "GPU\|deviceName"; then
        echo vulkan; return
    fi
    echo cpu
}
[[ "$BACKEND_CHOICE" == "auto" ]] && BACKEND_CHOICE="$(detect_backend)"
ok "Detected backend: $BACKEND_CHOICE"

# ── amd-smi check for Vulkan on Linux ──────────────────────────────────────
# amd-smi (Radeon Software / ROCm tools) provides GPU memory metrics needed
# for VRAM-aware model selection and autotune on AMD hardware.
if [[ "$BACKEND_CHOICE" == "vulkan" && "$OS" == "Linux" ]]; then
    if command -v amd-smi &>/dev/null; then
        ok "amd-smi found (AMD GPU memory monitoring available)"
    else
        err "amd-smi not found — required for Vulkan on AMD hardware."
        say  "Install it via your distribution's package manager:"
        say  "  Ubuntu/Debian: sudo apt install radeonsi-tools  (provides amd-smi)"
        say  "  Fedora:        sudo dnf install radeonsi-tools"
        say  "  Arch:          sudo pacman -S radeonsi-tools"
        say  ""
        say  "Or install ROCm tools: https://rocm.docs.amd.com/"
        exit 1
    fi
fi

platform_slug() {
    local arch slug_os slug_arch
    arch="$(uname -m)"
    case "$OS" in
        Linux)  slug_os="linux" ;;
        Darwin) slug_os="macos" ;;
        *) echo ""; return 1 ;;
    esac
    case "$arch" in
        x86_64|amd64) slug_arch="x86_64" ;;
        arm64|aarch64) slug_arch="arm64" ;;
        *) echo ""; return 1 ;;
    esac
    echo "${slug_os}-${slug_arch}"
}

release_asset_name() {
    local platform="$1" backend="$2"
    case "$backend" in
        cuda|vulkan|metal|cpu) printf 'llm-server-%s-%s.tar.gz\n' "$platform" "$backend" ;;
        *) return 1 ;;
    esac
}

release_api_url() {
    if [[ "$INSTALL_RELEASE" == "latest" ]]; then
        printf 'https://api.github.com/repos/%s/releases/latest\n' "$GITHUB_REPO"
    else
        printf 'https://api.github.com/repos/%s/releases/tags/%s\n' "$GITHUB_REPO" "$INSTALL_RELEASE"
    fi
}

find_release_asset_url() {
    local asset="$1" api
    if [[ -n "$INSTALL_RELEASE_DIR" && -f "$INSTALL_RELEASE_DIR/$asset" ]]; then
        printf 'file://%s/%s\n' "$(cd "$INSTALL_RELEASE_DIR" && pwd)" "$asset"
        return 0
    fi
    api="$(release_api_url)"
    curl -fsSL "$api" 2>/dev/null \
        | grep -Eo '"browser_download_url"[[:space:]]*:[[:space:]]*"[^"]+"' \
        | sed -E 's/^"browser_download_url"[[:space:]]*:[[:space:]]*"//; s/"$//' \
        | grep -F "/$asset" \
        | head -n 1
}

install_payload_file() {
    local src="$1" dst="$2" mode="${3:-0755}"
    [[ -f "$src" ]] || return 1
    install -m "$mode" "$src" "$dst"
}

install_release_bundle() {
    local platform asset url tmp archive payload_root found_backend=0
    [[ "$BACKEND_CHOICE" == "skip" ]] && return 1
    # CUDA/ik_llama artifacts need a CUDA-capable build host and driver/toolkit
    # compatibility checks. The default public release workflow intentionally
    # does not publish CUDA bundles from generic GitHub-hosted runners.
    [[ "$BACKEND_CHOICE" == "cuda" ]] && return 1
    command -v curl >/dev/null 2>&1 || return 1
    command -v tar >/dev/null 2>&1 || return 1
    platform="$(platform_slug)" || return 1
    asset="$(release_asset_name "$platform" "$BACKEND_CHOICE")" || return 1
    url="$(find_release_asset_url "$asset" || true)"
    [[ -n "$url" ]] || return 1

    say ""
    say "── Installing release bundle: $asset ──"
    tmp="$(mktemp -d -t llm-server-release.XXXXXX)"
    archive="$tmp/$asset"
    if ! curl -fL "$url" -o "$archive"; then
        rm -rf "$tmp"
        return 1
    fi
    mkdir -p "$tmp/payload"
    if ! tar -xzf "$archive" -C "$tmp/payload"; then
        rm -rf "$tmp"
        return 1
    fi
    payload_root="$(find "$tmp/payload" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    [[ -n "$payload_root" ]] || payload_root="$tmp/payload"

    for f in setup.sh setup-linux.sh setup-mac.sh llm-server llm-server-mac llm-server-gui parse_gguf.py model_index.py download_any_gguf.py; do
        if install_payload_file "$payload_root/$f" "$INSTALL_DIR/$f"; then
            ok "Installed $f"
        elif install_payload_file "$payload_root/bin/$f" "$INSTALL_DIR/$f"; then
            ok "Installed $f"
        fi
    done
    if [[ "$OS" == "Darwin" && -f "$INSTALL_DIR/llm-server-mac" ]]; then
        ln -sf "$INSTALL_DIR/llm-server-mac" "$INSTALL_DIR/llm-server"
        ok "Symlinked llm-server → llm-server-mac"
    fi

    for candidate in "$payload_root/llama-server" "$payload_root/bin/llama-server"; do
        if [[ -f "$candidate" ]]; then
            install -m 0755 "$candidate" "$INSTALL_DIR/llama-server"
            found_backend=1
            ok "Installed bundled llama-server"
            break
        fi
    done
    while IFS= read -r lib; do
        install -m 0644 "$lib" "$INSTALL_DIR/$(basename "$lib")"
        ok "Installed $(basename "$lib")"
    done < <(find "$payload_root" -maxdepth 2 -type f \( -name 'lib*.so*' -o -name 'lib*.dylib' -o -name '*.dll' \) 2>/dev/null | sort)

    rm -rf "$tmp"
    (( found_backend ))
}

# ── Stage 3: install scripts ────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR" "$MODEL_DIR"
RELEASE_INSTALLED=0

if [[ "$INSTALL_MODE" == "auto" || "$INSTALL_MODE" == "release" ]]; then
    if install_release_bundle; then
        RELEASE_INSTALLED=1
    elif [[ "$INSTALL_MODE" == "release" ]]; then
        err "No compatible release bundle found for $(platform_slug 2>/dev/null || echo unknown)-$BACKEND_CHOICE"
        [[ "$BACKEND_CHOICE" == "cuda" ]] && err "CUDA currently uses source builds unless you publish a manually built CUDA bundle."
        exit 1
    else
        if [[ "$BACKEND_CHOICE" == "cuda" ]]; then
            warn "CUDA backend selected; no generic prebuilt CUDA bundle is used. Falling back to ik_llama.cpp source build."
        else
            warn "No compatible release bundle found; falling back to local script install + source build."
        fi
    fi
fi

say ""
say "── Installing scripts to $INSTALL_DIR ──"

if (( RELEASE_INSTALLED )); then
    ok "Scripts installed from release bundle"
else
    ensure_source_repo
    if [[ "$OS" == "Darwin" ]]; then
        FILES=("setup.sh" "setup-mac.sh" "llm-server-mac" "llm-server-gui" "parse_gguf.py" "model_index.py" "download_any_gguf.py")
    else
        FILES=("setup.sh" "setup-linux.sh" "llm-server" "llm-server-gui" "parse_gguf.py" "model_index.py" "download_any_gguf.py")
    fi
    for f in "${FILES[@]}"; do
        if [[ -f "$SRC_DIR/$f" ]]; then
            install -m 0755 "$SRC_DIR/$f" "$INSTALL_DIR/$f"
            ok "Installed $f"
        else
            warn "$f not found in source; skipping"
        fi
    done
    if [[ "$OS" == "Darwin" && -f "$INSTALL_DIR/llm-server-mac" ]]; then
        ln -sf "$INSTALL_DIR/llm-server-mac" "$INSTALL_DIR/llm-server"
        ok "Symlinked llm-server → llm-server-mac"
    fi
    if [[ -f "$SRC_DIR/download_any_gguf.py" && ! -f "$MODEL_DIR/download_any_gguf.py" ]]; then
        install -m 0755 "$SRC_DIR/download_any_gguf.py" "$MODEL_DIR/download_any_gguf.py"
        ok "Installed downloader to $MODEL_DIR"
    fi
    if [[ -f "$SRC_DIR/model_index.py" && ! -f "$MODEL_DIR/model_index.py" ]]; then
        install -m 0755 "$SRC_DIR/model_index.py" "$MODEL_DIR/model_index.py"
        ok "Installed model indexer to $MODEL_DIR"
    fi
fi

# ── Stage 4: python deps (for downloader) ──────────────────────────────────
say ""
say "── Python dependencies ──"
if [[ "$PY_DEPS_MODE" == "skip" ]]; then
    warn "Skipped python dependency install. Downloader needs huggingface_hub + tqdm."
elif command -v python3 >/dev/null 2>&1; then
    if python3 -c "import huggingface_hub, tqdm" 2>/dev/null; then
        ok "huggingface_hub + tqdm already installed"
    else
        if [[ "$PY_DEPS_MODE" == "install" ]] || ask "Install huggingface_hub + tqdm via pip --user? [Y/n]" y; then
            python3 -m pip install --user --quiet huggingface_hub tqdm \
                && ok "Installed python deps" \
                || warn "pip install failed — downloader may not work"
        fi
    fi
else
    warn "python3 not found — downloader disabled"
fi

# ── Stage 5: optional backend build ─────────────────────────────────────────
case "$BACKEND_CHOICE" in
    cuda)
        BACKEND_REPO="https://github.com/ikawrakow/ik_llama.cpp.git"
        BACKEND_DIR="$BACKEND_ROOT/ik_llama.cpp"
        BACKEND_BUILD="$BACKEND_DIR/build"
        BACKEND_CMAKE=(-DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON)
        ;;
    vulkan)
        BACKEND_REPO="https://github.com/ggml-org/llama.cpp.git"
        BACKEND_DIR="$BACKEND_ROOT/llama.cpp"
        BACKEND_BUILD="$BACKEND_DIR/build-vulkan"
        BACKEND_CMAKE=(-DGGML_VULKAN=ON)
        ;;
    metal)
        BACKEND_REPO="https://github.com/ggml-org/llama.cpp.git"
        BACKEND_DIR="$BACKEND_ROOT/llama.cpp"
        BACKEND_BUILD="$BACKEND_DIR/build"
        BACKEND_CMAKE=(-DGGML_METAL=ON)
        ;;
    cpu)
        BACKEND_REPO="https://github.com/ggml-org/llama.cpp.git"
        BACKEND_DIR="$BACKEND_ROOT/llama.cpp"
        BACKEND_BUILD="$BACKEND_DIR/build"
        BACKEND_CMAKE=()
        ;;
    skip) BACKEND_REPO="" ;;
    *)    err "unknown backend: $BACKEND_CHOICE"; exit 1 ;;
esac

if [[ -n "$BACKEND_REPO" ]]; then
    say ""
    say "── Backend: $BACKEND_CHOICE ──"
    if (( RELEASE_INSTALLED )); then
        ok "Using bundled backend at $INSTALL_DIR/llama-server"
    elif [[ -x "$BACKEND_BUILD/bin/llama-server" ]]; then
        ok "Backend already built at $BACKEND_BUILD"
    elif [[ "$INSTALL_MODE" == "scripts" ]]; then
        warn "Scripts-only mode selected. Run with LLM_INSTALL_MODE=build to build a backend."
    elif [[ "$INSTALL_MODE" == "release" ]]; then
        warn "Release mode selected but no backend build requested."
    elif [[ "$INSTALL_MODE" == "build" ]] || ask "Clone + build $BACKEND_DIR now? (needs cmake, compiler toolchain, ~5-20min) [Y/n]" y; then
        missing=()
        for dep in git cmake make; do
            command -v "$dep" >/dev/null || missing+=("$dep")
        done
        if (( ${#missing[@]} )); then
            warn "Missing build dependencies: ${missing[*]}"
            warn "Install them, then rerun: LLM_INSTALL_MODE=build ./install.sh"
        elif [[ "$OS" == "Linux" && "$BACKEND_CHOICE" == "cuda" && ! -d /usr/local/cuda && ! -n "${CUDA_PATH:-}" ]]; then
            warn "CUDA toolkit not found. Install CUDA toolkit/nvcc, then rerun: LLM_INSTALL_MODE=build ./install.sh"
        else
            if [[ ! -d "$BACKEND_DIR/.git" ]]; then
                git clone "$BACKEND_REPO" "$BACKEND_DIR" || { err "clone failed"; exit 1; }
            fi
            cmake -S "$BACKEND_DIR" -B "$BACKEND_BUILD" -DCMAKE_BUILD_TYPE=Release "${BACKEND_CMAKE[@]}" \
                && cmake --build "$BACKEND_BUILD" --config Release -j"$(nproc 2>/dev/null || echo 4)" -t llama-server \
                && ok "Built llama-server at $BACKEND_BUILD/bin/llama-server" \
                || warn "Build failed — run 'llm-server --update' later or build manually"
        fi
    else
        warn "Skipped backend build. Run 'llm-server --update' later, or build $BACKEND_DIR yourself."
    fi
fi

# ── Stage 6: PATH hint ──────────────────────────────────────────────────────
say ""
if ! echo ":$PATH:" | grep -q ":$INSTALL_DIR:"; then
    SHELL_RC="$HOME/.bashrc"
    [[ "$OS" == "Darwin" ]] && SHELL_RC="$HOME/.zshrc"
    warn "$INSTALL_DIR is not in PATH"
    say  "  Add this line to $SHELL_RC:"
    say  "    export PATH=\"$INSTALL_DIR:\$PATH\""
fi

say ""
say "Done. Next:"
say "  llm-server ~/ai_models/your-model.gguf"
say "  llm-server-gui                    # pick a model interactively"
say "  llm-server <hf-repo> --download   # grab a gguf from HuggingFace"
