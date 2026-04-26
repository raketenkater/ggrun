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
#   LLM_INSTALL_PREFIX=<dir>                              default: ~/.local/bin
#   LLM_INSTALL_NONINTERACTIVE=1                          skip prompts

set -euo pipefail

REPO_URL="https://github.com/raketenkater/llm-server.git"
RAW_URL="https://raw.githubusercontent.com/raketenkater/llm-server/main"
INSTALL_DIR="${LLM_INSTALL_PREFIX:-$HOME/.local/bin}"
MODEL_DIR="$HOME/ai_models"
BACKEND_CHOICE="${LLM_INSTALL_BACKEND:-auto}"
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

say "═══ llm-server installer ═══"

# ── Stage 1: ensure we have the repo ────────────────────────────────────────
SRC_DIR=""
if [[ -f "./llm-server" && -f "./llm-server-gui" ]]; then
    SRC_DIR="$(pwd)"
    ok "Using local repo at $SRC_DIR"
else
    command -v git >/dev/null || { err "git required to fetch repo"; exit 1; }
    SRC_DIR="$(mktemp -d -t llm-server-install.XXXXXX)"
    say "── Cloning $REPO_URL ──"
    git clone --depth=1 "$REPO_URL" "$SRC_DIR" >/dev/null 2>&1 && ok "Cloned to $SRC_DIR" \
        || { err "git clone failed"; exit 1; }
    trap 'rm -rf "$SRC_DIR"' EXIT
fi

# ── Stage 2: detect platform + backend ──────────────────────────────────────
OS="$(uname -s)"
detect_backend() {
    if [[ "$OS" == "Darwin" ]]; then echo metal; return; fi
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

# ── Stage 3: install scripts ────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR" "$MODEL_DIR"
say ""
say "── Installing scripts to $INSTALL_DIR ──"

if [[ "$OS" == "Darwin" ]]; then
    FILES=("llm-server-mac" "llm-server-gui" "llm-opencode" "parse_gguf.py")
else
    FILES=("llm-server" "llm-server-gui" "llm-opencode" "parse_gguf.py")
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

# ── Stage 4: python deps (for downloader) ──────────────────────────────────
say ""
say "── Python dependencies ──"
if command -v python3 >/dev/null 2>&1; then
    if python3 -c "import huggingface_hub, tqdm" 2>/dev/null; then
        ok "huggingface_hub + tqdm already installed"
    else
        if ask "Install huggingface_hub + tqdm via pip --user? [Y/n]" y; then
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
        BACKEND_DIR="$HOME/ik_llama.cpp"
        BACKEND_BUILD="$BACKEND_DIR/build"
        BACKEND_CMAKE=(-DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON)
        ;;
    vulkan)
        BACKEND_REPO="https://github.com/ggml-org/llama.cpp.git"
        BACKEND_DIR="$HOME/llama.cpp"
        BACKEND_BUILD="$BACKEND_DIR/build-vulkan"
        BACKEND_CMAKE=(-DGGML_VULKAN=ON)
        ;;
    metal)
        BACKEND_REPO="https://github.com/ggml-org/llama.cpp.git"
        BACKEND_DIR="$HOME/llama.cpp"
        BACKEND_BUILD="$BACKEND_DIR/build"
        BACKEND_CMAKE=(-DGGML_METAL=ON)
        ;;
    cpu)
        BACKEND_REPO="https://github.com/ggml-org/llama.cpp.git"
        BACKEND_DIR="$HOME/llama.cpp"
        BACKEND_BUILD="$BACKEND_DIR/build"
        BACKEND_CMAKE=()
        ;;
    skip) BACKEND_REPO="" ;;
    *)    err "unknown backend: $BACKEND_CHOICE"; exit 1 ;;
esac

if [[ -n "$BACKEND_REPO" ]]; then
    say ""
    say "── Backend: $BACKEND_CHOICE ──"
    if [[ -x "$BACKEND_BUILD/bin/llama-server" ]]; then
        ok "Backend already built at $BACKEND_BUILD"
    elif ask "Clone + build $BACKEND_DIR now? (needs cmake, build-essential, ~5-20min) [y/N]" n; then
        for dep in git cmake make; do
            command -v "$dep" >/dev/null || { err "$dep required"; exit 1; }
        done
        if [[ ! -d "$BACKEND_DIR/.git" ]]; then
            git clone "$BACKEND_REPO" "$BACKEND_DIR" || { err "clone failed"; exit 1; }
        fi
        cmake -S "$BACKEND_DIR" -B "$BACKEND_BUILD" -DCMAKE_BUILD_TYPE=Release "${BACKEND_CMAKE[@]}" \
            && cmake --build "$BACKEND_BUILD" --config Release -j"$(nproc 2>/dev/null || echo 4)" -t llama-server \
            && ok "Built llama-server at $BACKEND_BUILD/bin/llama-server" \
            || warn "Build failed — run 'llm-server --update' later or build manually"
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
