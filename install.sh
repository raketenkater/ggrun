#!/usr/bin/env bash
#
# install.sh — One-command installer for ggrun.
#
# Usage (remote):
#   curl -fsSL https://raw.githubusercontent.com/raketenkater/ggrun/main/install.sh | bash
# Usage (local):
#   ./install.sh                  # from a cloned repo
#
# Installs the Go ggrun launcher and optionally installs or builds a
# llama.cpp backend (ik_llama.cpp for CUDA, llama.cpp for Vulkan/Metal/CPU).
# A small legacy Bash shim is installed as llm-server-bash only for migration.
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
#   LLM_INSTALL_DEPS=auto|install|skip                     default: auto
#   LLM_INSTALL_GO=auto|system|download|skip                default: auto
#   LLM_INSTALL_GO_VERSION=<version>                        default: go directive in go/go.mod
#   LLM_INSTALL_GO_ROOT=<dir>                               default: $LLM_INSTALL_BACKEND_ROOT/.llm-server-go
#   LLM_INSTALL_MAIN=go|bash                               default: go
#   LLM_INSTALL_NONINTERACTIVE=1                          skip prompts
#   LLM_INSTALL_PROMPT=0                                  never ask guided setup questions

set -euo pipefail

REPO_URL="https://github.com/raketenkater/ggrun.git"
GITHUB_REPO="raketenkater/ggrun"
SOURCE_REPO_DIR="${LLM_INSTALL_REPO_DIR:-}"
SOURCE_REF="${LLM_INSTALL_REF:-main}"
INSTALL_DIR="${LLM_INSTALL_PREFIX:-$HOME/.local/bin}"
MODEL_DIR="${LLM_INSTALL_MODEL_DIR:-$HOME/ai_models}"
BACKEND_ROOT="${LLM_INSTALL_BACKEND_ROOT:-$HOME}"
BACKEND_CHOICE="${LLM_INSTALL_BACKEND:-auto}"
BACKEND_REQUEST="$BACKEND_CHOICE"
INSTALL_MODE="${LLM_INSTALL_MODE:-auto}"
INSTALL_RELEASE="${LLM_INSTALL_RELEASE:-latest}"
INSTALL_RELEASE_DIR="${LLM_INSTALL_RELEASE_DIR:-}"
PY_DEPS_MODE="${LLM_INSTALL_PY_DEPS:-auto}"
DEPS_MODE="${LLM_INSTALL_DEPS:-auto}"
GO_MODE="${LLM_INSTALL_GO:-auto}"
GO_VERSION_OVERRIDE="${LLM_INSTALL_GO_VERSION:-}"
GO_BOOTSTRAP_ROOT="${LLM_INSTALL_GO_ROOT:-$BACKEND_ROOT/.llm-server-go}"
GO_CMD=""
NONINTERACTIVE="${LLM_INSTALL_NONINTERACTIVE:-0}"
MAIN_IMPL="${LLM_INSTALL_MAIN:-go}"
[[ ! -t 0 && ! -r /dev/tty ]] && NONINTERACTIVE=1   # piped installs can still ask through /dev/tty

SCRIPT_DIR=""
if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi

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
case "$DEPS_MODE" in
    auto|install|skip) ;;
    *) err "unknown dependency install mode: $DEPS_MODE"; exit 1 ;;
esac
case "$GO_MODE" in
    auto|system|download|skip) ;;
    *) err "unknown Go install mode: $GO_MODE"; exit 1 ;;
esac
case "$MAIN_IMPL" in
    go|bash) ;;
    *) err "unknown main implementation: $MAIN_IMPL"; exit 1 ;;
esac

say "═══ ggrun installer ═══"

# ── Stage 1: use local repo if present; clone only if source fallback needs it ──
SRC_DIR=""
if [[ -n "$SCRIPT_DIR" && -f "$SCRIPT_DIR/go/go.mod" && -f "$SCRIPT_DIR/scripts/setup-home.sh" ]]; then
    SRC_DIR="$SCRIPT_DIR"
    ok "Using local repo at $SRC_DIR"
elif [[ -f "./go/go.mod" && -f "./scripts/setup-home.sh" ]]; then
    SRC_DIR="$(pwd)"
    ok "Using local repo at $SRC_DIR"
fi

prepare_persistent_source_repo() {
    [[ -n "$SOURCE_REPO_DIR" ]] || return 1
    command -v git >/dev/null || { warn "git required to keep a source checkout for updates"; return 1; }
    if [[ -e "$SOURCE_REPO_DIR" && ! -d "$SOURCE_REPO_DIR/.git" ]]; then
        warn "$SOURCE_REPO_DIR exists but is not a git checkout; using temporary source instead"
        return 1
    fi
    if [[ -d "$SOURCE_REPO_DIR/.git" ]]; then
        say "── Updating source checkout: $SOURCE_REPO_DIR ($SOURCE_REF) ──"
        git -C "$SOURCE_REPO_DIR" fetch origin "$SOURCE_REF" --depth=1 >/dev/null 2>&1 || \
            git -C "$SOURCE_REPO_DIR" fetch origin "$SOURCE_REF" >/dev/null 2>&1 || return 1
        git -C "$SOURCE_REPO_DIR" checkout -q "$SOURCE_REF" >/dev/null 2>&1 || true
        git -C "$SOURCE_REPO_DIR" merge --ff-only FETCH_HEAD >/dev/null 2>&1 || \
            git -C "$SOURCE_REPO_DIR" checkout -q FETCH_HEAD >/dev/null 2>&1 || return 1
        ok "Source checkout ready at $SOURCE_REPO_DIR"
    else
        mkdir -p "$(dirname "$SOURCE_REPO_DIR")"
        say "── Cloning ggrun source for future updates: $SOURCE_REPO_DIR ($SOURCE_REF) ──"
        git clone --depth=1 --branch "$SOURCE_REF" "$REPO_URL" "$SOURCE_REPO_DIR" >/dev/null 2>&1 || return 1
        ok "Source checkout ready at $SOURCE_REPO_DIR"
    fi
    SRC_DIR="$SOURCE_REPO_DIR"
}

ensure_source_repo() {
    [[ -n "$SRC_DIR" ]] && return 0
    if prepare_persistent_source_repo; then
        return 0
    fi
    command -v git >/dev/null || { err "git required to fetch repo"; exit 1; }
    SRC_DIR="$(mktemp -d -t ggrun-install.XXXXXX)"
    say "── Cloning $REPO_URL ──"
    if git clone --depth=1 --branch "$SOURCE_REF" "$REPO_URL" "$SRC_DIR" >/dev/null 2>&1 || \
        git clone --depth=1 "$REPO_URL" "$SRC_DIR" >/dev/null 2>&1; then
        ok "Cloned to $SRC_DIR"
    else
        err "git clone failed"
        exit 1
    fi
    trap 'rm -rf "$SRC_DIR"' EXIT
}

if [[ -n "$SOURCE_REPO_DIR" ]]; then
    prepare_persistent_source_repo || true
fi

# ── Stage 2: detect platform + backend ──────────────────────────────────────
OS="$(uname -s)"

cuda_nvcc_path() {
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
        return 0
    fi
    if [[ -n "${CUDA_PATH:-}" && -x "${CUDA_PATH}/bin/nvcc" ]]; then
        printf '%s\n' "${CUDA_PATH}/bin/nvcc"
        return 0
    fi
    if [[ -x /usr/local/cuda/bin/nvcc ]]; then
        printf '%s\n' /usr/local/cuda/bin/nvcc
        return 0
    fi
    return 1
}

has_cuda_toolkit() {
    local nvcc
    nvcc="$(cuda_nvcc_path 2>/dev/null || true)"
    [[ -n "$nvcc" ]] || return 1
    "$nvcc" --version >/dev/null 2>&1
}

vulkan_available() {
    command -v vulkaninfo >/dev/null 2>&1 || return 1
    vulkaninfo --summary 2>/dev/null | grep -qi "GPU\|deviceName"
}

detect_backend() {
    if [[ "$OS" == "Darwin" ]]; then echo metal; return; fi
    if [[ "$OS" == MINGW* || "$OS" == MSYS* || "$OS" == CYGWIN* ]]; then
        err "Use install.ps1 for native Windows installs, or run this Bash installer on Linux/macOS."
        exit 1
    fi
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
        echo cuda; return
    fi
    if vulkan_available; then
        echo vulkan; return
    fi
    echo cpu
}
[[ "$BACKEND_CHOICE" == "auto" ]] && BACKEND_CHOICE="$(detect_backend)"
DETECTED_BACKEND="$BACKEND_CHOICE"

if (( ! NONINTERACTIVE )) && [[ "${LLM_INSTALL_PROMPT:-auto}" != "0" && "$BACKEND_REQUEST" == "auto" ]]; then
    say ""
    say "Setup choices"
    say "  Detected backend: $DETECTED_BACKEND"
    say "  Install location: $INSTALL_DIR"
    say "  Model directory:  $MODEL_DIR"
    if ask "Install/build a llama.cpp backend now so ggrun works out of the box? [Y/n]" y; then
        if ! ask "Use detected backend '$DETECTED_BACKEND'? [Y/n]" y; then
            read -r -p "Choose backend [cuda/vulkan/cpu/skip]: " reply </dev/tty || reply=""
            reply="${reply:-$DETECTED_BACKEND}"
            case "$reply" in
                cuda|vulkan|cpu|skip) BACKEND_CHOICE="$reply" ;;
                *) warn "Unknown backend '$reply'; using detected backend '$DETECTED_BACKEND'" ;;
            esac
        fi
        if [[ "$BACKEND_CHOICE" != "skip" && "$DEPS_MODE" == "auto" ]]; then
            if ask "Install missing system build dependencies if needed? [Y/n]" y; then
                DEPS_MODE="install"
            else
                DEPS_MODE="skip"
            fi
        fi
    else
        BACKEND_CHOICE="skip"
        warn "Backend install skipped. Configure LLAMA_SERVER manually before launching models."
    fi
    if [[ "$GO_MODE" == "auto" ]]; then
        if ask "Install a local Go toolchain if system Go is missing or too old? [Y/n]" y; then
            GO_MODE="auto"
        else
            GO_MODE="system"
        fi
    fi
    if [[ "$PY_DEPS_MODE" == "auto" ]]; then
        if ask "Install Python downloader helpers for HuggingFace model search/download if missing? [Y/n]" y; then
            PY_DEPS_MODE="install"
        else
            PY_DEPS_MODE="skip"
        fi
    fi
fi

ok "Selected backend: $BACKEND_CHOICE"

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
        cuda|vulkan|metal|cpu) printf 'ggrun-%s-%s.tar.gz\n' "$platform" "$backend" ;;
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


verify_release_checksum() {
    local tmp="$1" asset="$2" match
    [[ -f "$tmp/SHA256SUMS" ]] || return 1
    match="$(grep -F " $asset" "$tmp/SHA256SUMS" || true)"
    if [[ -z "$match" ]]; then
        warn "SHA256SUMS did not include $asset"
        return 1
    fi
    if command -v sha256sum >/dev/null 2>&1; then
        printf "%s\n" "$match" | (cd "$tmp" && sha256sum -c - >/dev/null)
    elif command -v shasum >/dev/null 2>&1; then
        printf "%s\n" "$match" | (cd "$tmp" && shasum -a 256 -c - >/dev/null)
    else
        warn "No SHA256 checker found"
        return 1
    fi
}

install_payload_file() {
    local src="$1" dst="$2" mode="${3:-0755}"
    [[ -f "$src" ]] || return 1
    install -m "$mode" "$src" "$dst"
}

install_go_as_main() {
    local go_bin="$1"
    [[ "$MAIN_IMPL" == "go" && -x "$go_bin" ]] || return 1
    if [[ -x "$INSTALL_DIR/ggrun" && ! -e "$INSTALL_DIR/llm-server-bash" ]]; then
        cp "$INSTALL_DIR/ggrun" "$INSTALL_DIR/llm-server-bash" 2>/dev/null || true
        chmod 0755 "$INSTALL_DIR/llm-server-bash" 2>/dev/null || true
    fi
    install -m 0755 "$go_bin" "$INSTALL_DIR/ggrun"
    ln -sf ggrun "$INSTALL_DIR/llm-server" 2>/dev/null || true  # back-compat: old command name
    ok "Installed Go ggrun as primary command"
}

go_required_version() {
    if [[ -n "$GO_VERSION_OVERRIDE" ]]; then
        printf '%s\n' "${GO_VERSION_OVERRIDE#go}"
        return
    fi
    if [[ -n "$SRC_DIR" && -f "$SRC_DIR/go/go.mod" ]]; then
        awk '$1 == "go" { print $2; exit }' "$SRC_DIR/go/go.mod"
        return
    fi
    printf '1.24.13\n'
}

go_version_parts() {
    local v="${1#go}" a b c
    v="${v%%[-+ ]*}"
    IFS=. read -r a b c <<<"$v"
    printf '%d %d %d\n' "${a:-0}" "${b:-0}" "${c:-0}"
}

go_version_at_least() {
    local have="$1" need="$2" ha hb hc na nb nc
    read -r ha hb hc < <(go_version_parts "$have")
    read -r na nb nc < <(go_version_parts "$need")
    (( ha > na )) && return 0
    (( ha < na )) && return 1
    (( hb > nb )) && return 0
    (( hb < nb )) && return 1
    (( hc >= nc ))
}

find_system_go() {
    local cmd have need
    command -v go >/dev/null 2>&1 || return 1
    cmd="$(command -v go)"
    have="$($cmd env GOVERSION 2>/dev/null || true)"
    if [[ -z "$have" ]]; then
        have="$($cmd version 2>/dev/null | awk '{ print $3; exit }')"
    fi
    [[ -n "$have" ]] || return 1
    need="$(go_required_version)"
    if go_version_at_least "$have" "$need"; then
        GO_CMD="$cmd"
        ok "Using system Go $have"
        return 0
    fi
    warn "System Go $have is older than required Go $need"
    return 1
}

go_download_platform() {
    local goos goarch arch
    arch="$(uname -m)"
    case "$OS" in
        Linux) goos="linux" ;;
        Darwin) goos="darwin" ;;
        *) return 1 ;;
    esac
    case "$arch" in
        x86_64|amd64) goarch="amd64" ;;
        arm64|aarch64) goarch="arm64" ;;
        armv6l|armv7l) goarch="armv6l" ;;
        *) return 1 ;;
    esac
    printf '%s-%s\n' "$goos" "$goarch"
}

download_go_toolchain() {
    local need platform root url tmp archive extracted_go
    need="$(go_required_version)"
    platform="$(go_download_platform)" || { warn "No Go toolchain download for $(uname -s)/$(uname -m)"; return 1; }
    root="$GO_BOOTSTRAP_ROOT/go$need.$platform"
    if [[ -x "$root/bin/go" ]] && go_version_at_least "$($root/bin/go env GOVERSION 2>/dev/null || true)" "$need"; then
        GO_CMD="$root/bin/go"
        ok "Using bundled Go at $root"
        return 0
    fi
    command -v curl >/dev/null 2>&1 || { warn "curl required to download Go"; return 1; }
    command -v tar >/dev/null 2>&1 || { warn "tar required to unpack Go"; return 1; }

    url="https://go.dev/dl/go$need.$platform.tar.gz"
    say "── Installing Go toolchain: go$need ($platform) ──"
    tmp="$(mktemp -d -t llm-server-go.XXXXXX)"
    archive="$tmp/go.tar.gz"
    if ! curl -fL "$url" -o "$archive"; then
        rm -rf "$tmp"
        warn "Go download failed: $url"
        return 1
    fi
    if ! tar -xzf "$archive" -C "$tmp"; then
        rm -rf "$tmp"
        warn "Go archive unpack failed"
        return 1
    fi
    extracted_go="$tmp/go"
    [[ -x "$extracted_go/bin/go" ]] || { rm -rf "$tmp"; warn "Downloaded Go archive did not contain bin/go"; return 1; }
    mkdir -p "$GO_BOOTSTRAP_ROOT"
    rm -rf "$root"
    mv "$extracted_go" "$root"
    rm -rf "$tmp"
    GO_CMD="$root/bin/go"
    ok "Installed Go at $root"
}

ensure_go_toolchain() {
    [[ "$MAIN_IMPL" == "go" && "$INSTALL_MODE" != "scripts" ]] || return 1
    case "$GO_MODE" in
        skip)
            find_system_go || return 1
            ;;
        system)
            find_system_go || { warn "Go is required; install Go or rerun with LLM_INSTALL_GO=auto"; return 1; }
            ;;
        download)
            download_go_toolchain
            ;;
        auto)
            find_system_go || download_go_toolchain
            ;;
    esac
}

build_go_binary() {
    local out="$1"
    [[ -n "$SRC_DIR" && -f "$SRC_DIR/go/go.mod" ]] || return 1
    ensure_go_toolchain || return 1
    # Stamp the version only on exact tag checkouts; branch builds keep the
    # in-source default so the update checker is not misled.
    local ldflags="-s -w"
    local ver
    ver="$(git -C "$SRC_DIR" describe --tags --exact-match 2>/dev/null || true)"
    [[ -n "$ver" ]] && ldflags="$ldflags -X github.com/raketenkater/ggrun/pkg/update.currentVersion=$ver"
    (cd "$SRC_DIR/go" && "$GO_CMD" build -trimpath -ldflags="$ldflags" -o "$out" ./cmd/ggrun)
}

link_backend_binary() {
    local server="$1"
    [[ -x "$server" ]] || return 1
    ln -sf "$server" "$INSTALL_DIR/llama-server"
    ok "Linked llama-server backend into $INSTALL_DIR"
}

install_source_file() {
    local rel="$1" name="${2:-$(basename "$1")}" mode="${3:-0755}"
    [[ -n "$SRC_DIR" && -f "$SRC_DIR/$rel" ]] || return 1
    install -m "$mode" "$SRC_DIR/$rel" "$INSTALL_DIR/$name"
    ok "Installed $name"
}

install_legacy_bash_shim() {
    [[ -n "$SRC_DIR" && -f "$SRC_DIR/legacy/bash/ggrun" ]] || return 0
    install -m 0755 "$SRC_DIR/legacy/bash/ggrun" "$INSTALL_DIR/llm-server-bash"
    ok "Installed llm-server-bash migration shim"
    if [[ "$MAIN_IMPL" == "bash" ]]; then
        install -m 0755 "$SRC_DIR/legacy/bash/ggrun" "$INSTALL_DIR/ggrun"
        ok "Installed legacy migration shim as primary command"
    fi
}

install_release_bundle() {
    local platform asset url sums_url tmp archive payload_root found_backend=0
    [[ "$BACKEND_CHOICE" == "skip" ]] && return 1
    # CUDA release bundles are optional manual assets. If none exists, auto mode
    # can still fall back to Vulkan or CPU before attempting a source build.
    command -v curl >/dev/null 2>&1 || return 1
    command -v tar >/dev/null 2>&1 || return 1
    platform="$(platform_slug)" || return 1
    asset="$(release_asset_name "$platform" "$BACKEND_CHOICE")" || return 1
    url="$(find_release_asset_url "$asset" || true)"
    [[ -n "$url" ]] || return 1

    say ""
    say "── Installing release bundle: $asset ──"
    tmp="$(mktemp -d -t ggrun-release.XXXXXX)"
    archive="$tmp/$asset"
    if ! curl -fL "$url" -o "$archive"; then
        rm -rf "$tmp"
        return 1
    fi
    sums_url="$(find_release_asset_url "SHA256SUMS" || true)"
    if [[ -n "$sums_url" ]]; then
        if ! curl -fL "$sums_url" -o "$tmp/SHA256SUMS"; then
            rm -rf "$tmp"
            warn "Checksum download failed"
            return 1
        fi
        if ! verify_release_checksum "$tmp" "$asset"; then
            rm -rf "$tmp"
            warn "Checksum verification failed for $asset"
            return 1
        fi
        ok "Verified checksum for $asset"
    elif [[ "${LLM_INSTALL_ALLOW_UNVERIFIED:-0}" == "1" ]]; then
        warn "No SHA256SUMS asset found; LLM_INSTALL_ALLOW_UNVERIFIED=1 set — installing UNVERIFIED bundle"
    else
        rm -rf "$tmp"
        err "No SHA256SUMS asset found; refusing to install an unverified bundle."
        err "Set LLM_INSTALL_ALLOW_UNVERIFIED=1 to override (not recommended)."
        return 1
    fi
    mkdir -p "$tmp/payload"
    if ! tar -xzf "$archive" -C "$tmp/payload"; then
        rm -rf "$tmp"
        return 1
    fi
    payload_root="$(find "$tmp/payload" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    [[ -n "$payload_root" ]] || payload_root="$tmp/payload"

    # llm-server-go is listed only for backward compatibility with pre-3.0.1
    # bundles, which shipped the binary under that name. New bundles ship it as
    # ggrun directly, and no ggrun-gui wrapper.
    for f in setup.sh setup-linux.sh setup-mac.sh ggrun llm-server-bash llm-server-go parse_gguf.py model_index.py download_any_gguf.py; do
        if install_payload_file "$payload_root/$f" "$INSTALL_DIR/$f"; then
            ok "Installed $f"
        elif install_payload_file "$payload_root/bin/$f" "$INSTALL_DIR/$f"; then
            ok "Installed $f"
        fi
    done
    # Old bundles need the -go binary promoted to the primary command.
    if [[ ! -x "$INSTALL_DIR/ggrun" && -x "$INSTALL_DIR/llm-server-go" ]]; then
        install_go_as_main "$INSTALL_DIR/llm-server-go" || true
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

choose_cuda_auto_fallback_backend() {
    [[ "$BACKEND_REQUEST" == "auto" && "$BACKEND_CHOICE" == "cuda" ]] || return 1
    has_cuda_toolkit && return 1
    if vulkan_available; then
        BACKEND_CHOICE="vulkan"
        warn "CUDA toolkit not found and no CUDA bundle is available; falling back to Vulkan."
    else
        BACKEND_CHOICE="cpu"
        warn "CUDA toolkit and a usable Vulkan device were not found; falling back to CPU."
    fi
    ok "Selected fallback backend: $BACKEND_CHOICE"
}

run_privileged() {
    if (( EUID == 0 )); then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        return 1
    fi
}

python3_usable() {
    command -v python3 >/dev/null 2>&1 \
        && python3 -c 'import sys; raise SystemExit(0 if sys.version_info.major == 3 else 1)' >/dev/null 2>&1
}

install_python_runtime() {
    [[ "$PY_DEPS_MODE" != "skip" ]] || return 1
    say "-- Installing Python 3 for model downloads --"
    if [[ "$OS" == "Darwin" ]]; then
        command -v brew >/dev/null 2>&1 || return 1
        brew install python
    elif command -v apt-get >/dev/null 2>&1; then
        run_privileged apt-get update \
            && run_privileged apt-get install -y python3 python3-pip python3-venv
    elif command -v dnf >/dev/null 2>&1; then
        run_privileged dnf install -y python3 python3-pip
    elif command -v yum >/dev/null 2>&1; then
        run_privileged yum install -y python3 python3-pip
    elif command -v pacman >/dev/null 2>&1; then
        run_privileged pacman -Sy --needed --noconfirm python python-pip
    elif command -v zypper >/dev/null 2>&1; then
        run_privileged zypper install -y python3 python3-pip
    else
        return 1
    fi
}

ensure_python_pip() {
    python3 -m pip --version >/dev/null 2>&1 && return 0
    # CPython normally bundles ensurepip. Distribution builds may omit it, in
    # which case install_python_runtime above installs the distro's pip package.
    python3 -m ensurepip --user >/dev/null 2>&1 \
        && python3 -m pip --version >/dev/null 2>&1
}

install_python_download_deps() {
    local args=(--user --quiet --upgrade huggingface_hub tqdm)
    python3 -m pip install "${args[@]}" >/dev/null 2>&1 \
        || python3 -m pip install --break-system-packages "${args[@]}" >/dev/null 2>&1
    python3 -c 'import huggingface_hub, tqdm' >/dev/null 2>&1
}

install_build_deps() {
    [[ "$DEPS_MODE" != "skip" ]] || return 1

    say "-- Installing build dependencies --"
    if [[ "$OS" == "Darwin" ]]; then
        if command -v brew >/dev/null 2>&1; then
            brew install cmake git
        else
            return 1
        fi
    elif command -v apt-get >/dev/null 2>&1; then
        local pkgs=(git cmake build-essential pkg-config libcurl4-openssl-dev)
        [[ "$BACKEND_CHOICE" == "vulkan" ]] && pkgs+=(libvulkan-dev glslang-tools vulkan-tools)
        run_privileged apt-get update && run_privileged apt-get install -y "${pkgs[@]}"
    elif command -v dnf >/dev/null 2>&1; then
        local pkgs=(git cmake make gcc gcc-c++ pkgconf-pkg-config libcurl-devel)
        [[ "$BACKEND_CHOICE" == "vulkan" ]] && pkgs+=(vulkan-loader-devel vulkan-headers glslang vulkan-tools)
        run_privileged dnf install -y "${pkgs[@]}"
    elif command -v yum >/dev/null 2>&1; then
        local pkgs=(git cmake make gcc gcc-c++ pkgconfig libcurl-devel)
        [[ "$BACKEND_CHOICE" == "vulkan" ]] && pkgs+=(vulkan-loader-devel glslang vulkan-tools)
        run_privileged yum install -y "${pkgs[@]}"
    elif command -v pacman >/dev/null 2>&1; then
        local pkgs=(git cmake make gcc pkgconf curl)
        [[ "$BACKEND_CHOICE" == "vulkan" ]] && pkgs+=(vulkan-headers glslang vulkan-tools)
        run_privileged pacman -Sy --needed --noconfirm "${pkgs[@]}"
    elif command -v zypper >/dev/null 2>&1; then
        local pkgs=(git cmake make gcc gcc-c++ pkg-config libcurl-devel)
        [[ "$BACKEND_CHOICE" == "vulkan" ]] && pkgs+=(vulkan-devel glslang-tools vulkan-tools)
        run_privileged zypper install -y "${pkgs[@]}"
    else
        return 1
    fi
}

missing_build_deps() {
    local missing=() dep
    for dep in git cmake make c++; do
        command -v "$dep" >/dev/null 2>&1 || missing+=("$dep")
    done
    printf '%s\n' "${missing[@]}"
}

ensure_build_deps() {
    local missing
    missing="$(missing_build_deps | paste -sd ' ' -)"
    if [[ -z "$missing" ]]; then
        return 0
    fi
    warn "Missing build dependencies: $missing"
    if install_build_deps; then
        missing="$(missing_build_deps | paste -sd ' ' -)"
        [[ -z "$missing" ]] && { ok "Build dependencies ready"; return 0; }
    fi
    err "Backend build dependencies are missing: $missing"
    if [[ "$OS" == "Darwin" ]]; then
        say "Install Apple's command-line tools with: xcode-select --install"
        say "If cmake is still missing, install Homebrew and run: brew install cmake"
    elif [[ "$OS" == "Linux" ]]; then
        say "Install your distribution's C/C++ build tools, git, and cmake."
    fi
    say "Or rerun with LLM_INSTALL_BACKEND=skip for a launcher-only install."
    return 1
}

backend_server_path() {
    printf '%s/bin/llama-server\n' "$BACKEND_BUILD"
}

refresh_backend_repo() {
    if [[ -d "$BACKEND_DIR/.git" ]]; then
        git -C "$BACKEND_DIR" pull --ff-only || warn "Could not fast-forward $BACKEND_DIR; using existing checkout"
    else
        git clone "$BACKEND_REPO" "$BACKEND_DIR"
    fi
}

build_backend() {
    ensure_build_deps || return 1
    if [[ "$OS" == "Linux" && "$BACKEND_CHOICE" == "cuda" ]] && ! has_cuda_toolkit; then
        err "CUDA toolkit/nvcc not found for CUDA backend."
        return 1
    fi
    refresh_backend_repo || return 1
    cmake_env=()
    if [[ "$BACKEND_CHOICE" == "cuda" ]]; then
        nvcc_path="$(cuda_nvcc_path 2>/dev/null || true)"
        [[ -n "$nvcc_path" ]] && cmake_env=(CUDACXX="$nvcc_path")
    fi
    env "${cmake_env[@]}" cmake -S "$BACKEND_DIR" -B "$BACKEND_BUILD" -DCMAKE_BUILD_TYPE=Release "${BACKEND_CMAKE[@]}" \
        && cmake --build "$BACKEND_BUILD" --config Release -j"$(nproc 2>/dev/null || echo 4)" -t llama-server
}

# ── Stage 3: install scripts ────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR" "$MODEL_DIR"
RELEASE_INSTALLED=0

if [[ "$INSTALL_MODE" == "auto" || "$INSTALL_MODE" == "release" ]]; then
    if install_release_bundle; then
        RELEASE_INSTALLED=1
    elif [[ "$INSTALL_MODE" == "release" ]]; then
        err "No compatible release bundle found for $(platform_slug 2>/dev/null || echo unknown)-$BACKEND_CHOICE"
        [[ "$BACKEND_CHOICE" == "cuda" ]] && err "CUDA release mode requires a manually attached CUDA bundle for this platform."
        exit 1
    else
        if choose_cuda_auto_fallback_backend && install_release_bundle; then
            RELEASE_INSTALLED=1
        elif [[ "$BACKEND_CHOICE" == "cuda" ]]; then
            warn "No compatible CUDA release bundle found; falling back to ik_llama.cpp source build."
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
    FILES=("setup.sh" "setup-linux.sh" "setup-mac.sh")
    for f in "${FILES[@]}"; do
        install_source_file "$f" "$f" 0755 || warn "$f not found in source; skipping"
    done
    install_source_file "tools/gguf/parse_gguf.py" "parse_gguf.py" 0755 || warn "parse_gguf.py not found in source; skipping"
    install_source_file "tools/models/model_index.py" "model_index.py" 0755 || warn "model_index.py not found in source; skipping"
    install_source_file "tools/download/download_any_gguf.py" "download_any_gguf.py" 0755 || warn "download_any_gguf.py not found in source; skipping"
    install_legacy_bash_shim
    # Build/install the Go binary directly as the single `ggrun` command —
    # no separate llm-server-go copy. `ggrun` with no args opens the GUI,
    # so no ggrun-gui wrapper is needed either.
    if [[ -f "$SRC_DIR/go/go.mod" && "$MAIN_IMPL" == "go" && "$INSTALL_MODE" != "scripts" ]]; then
        say "── Building Go ggrun ──"
        go_build_tmp="$(mktemp -t ggrun-build.XXXXXX)"
        if build_go_binary "$go_build_tmp" && install_go_as_main "$go_build_tmp"; then
            :
        elif [[ -x "$SRC_DIR/go/ggrun" ]] && install_go_as_main "$SRC_DIR/go/ggrun"; then
            :
        else
            warn "Could not build Go ggrun; rerun with LLM_INSTALL_GO=auto or use a release bundle."
        fi
        rm -f "$go_build_tmp"
    elif [[ -x "$SRC_DIR/go/ggrun" ]]; then
        install_go_as_main "$SRC_DIR/go/ggrun" || true
    fi
fi

if [[ "$MAIN_IMPL" == "go" && "$INSTALL_MODE" != "scripts" && ! -x "$INSTALL_DIR/ggrun" ]]; then
    err "Go ggrun was not installed. Install Go or rerun with LLM_INSTALL_GO=auto."
    exit 1
fi

# ── Stage 4: python deps (for downloader) ──────────────────────────────────
say ""
say "── Python dependencies ──"
if [[ "$PY_DEPS_MODE" == "skip" ]]; then
    warn "Skipped python dependency install. Downloader needs huggingface_hub + tqdm."
else
    if ! python3_usable; then
        warn "A usable Python 3 interpreter was not found."
        if ! install_python_runtime || ! python3_usable; then
            if [[ "$OS" == "Darwin" ]]; then
                err "Python 3 is needed for model search/download. Install Homebrew, then run: brew install python"
            else
                err "Python 3 is needed for model search/download. Install python3 and python3-pip with your package manager."
            fi
            if [[ "$PY_DEPS_MODE" == "install" ]]; then
                exit 1
            fi
            warn "Local GGUF serving still works; model search/download is unavailable until Python is installed."
        fi
    fi
    if python3_usable; then
        if python3 -c 'import huggingface_hub, tqdm' >/dev/null 2>&1; then
            ok "Python download dependencies already installed"
        elif ! ensure_python_pip; then
            err "Python 3 is installed, but pip is unavailable. Try: python3 -m ensurepip --user"
            [[ "$PY_DEPS_MODE" == "install" ]] && exit 1
            warn "Local GGUF serving still works; model search/download needs pip, huggingface_hub, and tqdm."
        elif [[ "$PY_DEPS_MODE" == "install" ]] || ask "Install huggingface_hub + tqdm via pip --user? [Y/n]" y; then
            if install_python_download_deps; then
                ok "Python download dependencies ready"
            else
                err "Could not install or import huggingface_hub and tqdm."
                say "Try: python3 -m pip install --user huggingface_hub tqdm"
                [[ "$PY_DEPS_MODE" == "install" ]] && exit 1
                warn "Local GGUF serving still works; model search/download is unavailable."
            fi
        fi
    fi
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
    backend_binary="$(backend_server_path)"
    if (( RELEASE_INSTALLED )); then
        ok "Using bundled backend at $INSTALL_DIR/llama-server"
    elif [[ -x "$backend_binary" ]]; then
        ok "Backend already built at $BACKEND_BUILD"
        link_backend_binary "$backend_binary" || true
    elif [[ "$INSTALL_MODE" == "scripts" ]]; then
        err "Scripts-only mode does not install a backend. Rerun without LLM_INSTALL_MODE=scripts or set LLM_INSTALL_BACKEND=skip intentionally."
        exit 1
    elif [[ "$INSTALL_MODE" == "release" ]]; then
        err "Release mode selected but no compatible backend bundle was installed. Rerun with LLM_INSTALL_MODE=build."
        exit 1
    else
        if build_backend; then
            ok "Built llama-server at $backend_binary"
            link_backend_binary "$backend_binary" || true
        else
            err "Backend build failed for $BACKEND_CHOICE."
            if [[ "$BACKEND_REQUEST" == "auto" && "$BACKEND_CHOICE" != "cpu" ]]; then
                fallback_built=0
                for fallback in vulkan cpu; do
                    [[ "$fallback" == "$BACKEND_CHOICE" ]] && continue
                    if [[ "$fallback" == "vulkan" ]]; then
                        warn "Retrying with Vulkan llama.cpp backend before CPU fallback."
                        BACKEND_CHOICE="vulkan"
                        BACKEND_REPO="https://github.com/ggml-org/llama.cpp.git"
                        BACKEND_DIR="$BACKEND_ROOT/llama.cpp"
                        BACKEND_BUILD="$BACKEND_DIR/build-vulkan"
                        BACKEND_CMAKE=(-DGGML_VULKAN=ON)
                    else
                        warn "Retrying with CPU llama.cpp backend so ggrun works out of the box."
                        BACKEND_CHOICE="cpu"
                        BACKEND_REPO="https://github.com/ggml-org/llama.cpp.git"
                        BACKEND_DIR="$BACKEND_ROOT/llama.cpp"
                        BACKEND_BUILD="$BACKEND_DIR/build"
                        BACKEND_CMAKE=()
                    fi
                    backend_binary="$(backend_server_path)"
                    if [[ -x "$backend_binary" ]] || build_backend; then
                        ok "Built $BACKEND_CHOICE llama-server at $backend_binary"
                        link_backend_binary "$backend_binary" || true
                        fallback_built=1
                        break
                    fi
                    warn "$BACKEND_CHOICE fallback backend failed."
                done
                if (( ! fallback_built )); then
                    err "Fallback backend builds failed. Install build dependencies and rerun setup."
                    exit 1
                fi
            else
                err "Install cannot finish without a backend. Rerun with LLM_INSTALL_BACKEND=skip only if you will configure LLAMA_SERVER manually."
                exit 1
            fi
        fi
    fi

    if [[ ! -x "$INSTALL_DIR/llama-server" ]]; then
        err "No llama-server binary was installed."
        exit 1
    fi
    ok "Backend ready: $INSTALL_DIR/llama-server"
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
say "╔════════════════════════════════════════════════════════════╗"
say "║ ggrun installer finished                             ║"
say "╚════════════════════════════════════════════════════════════╝"
say "CLI:       $INSTALL_DIR/ggrun"
say "GUI:       $INSTALL_DIR/ggrun   (no arguments opens the GUI)"
say "Models:    $MODEL_DIR"
if [[ -x "$INSTALL_DIR/llama-server" ]]; then
    say "Backend:   $INSTALL_DIR/llama-server"
else
    say "Backend:   not installed (launcher-only mode)"
fi
say ""
say "Next:"
say "  $INSTALL_DIR/ggrun            # interactive GUI"
say "  $INSTALL_DIR/ggrun detect"
say "  $INSTALL_DIR/ggrun <hf-repo> --download"
say "  $INSTALL_DIR/ggrun $MODEL_DIR/your-model.gguf"
