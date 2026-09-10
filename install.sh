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
#     auto: reuse local llama.cpp/ik_llama, then install CUDA/Vulkan/CPU
#           prebuilts. A backend counts only if it starts on this machine,
#           or only needs a GPU driver. CPU is installed when nothing runs.
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
#   LLM_INSTALL_SCAN_SYSTEM=0                             do not search the machine for
#                                                         existing llama.cpp / CUDA
#   ./install.sh --discover                               print what is already installed

set -Eeuo pipefail

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
PROBE_LAST_OUT=""
PROBE_LAST_KIND=""
# CI / no terminal: silent defaults. A real curl | bash still has /dev/tty.
[[ ! -t 0 && ! -r /dev/tty && -z "${LLM_INSTALL_NONINTERACTIVE:-}" ]] && NONINTERACTIVE=1

SCRIPT_DIR=""
if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi

say()  { printf '%s\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m⚠\033[0m %s\n' "$*"; }
err()  { LAST_ERR_MSG="$*"; printf '  \033[31m✗\033[0m %s\n' "$*" >&2; }
ask()  { # ask "prompt" default_yn
    local p="$1" d="${2:-n}" reply
    if (( NONINTERACTIVE )); then [[ "$d" == "y" ]]; return; fi
    read -r -p "$p " reply </dev/tty || reply=""
    reply="${reply:-$d}"
    [[ "$reply" =~ ^[Yy] ]]
}

show_help() {
    sed -n '2,/^set -Eeuo pipefail$/p' "${BASH_SOURCE[0]}" | sed '$d'
}

DISCOVER_ONLY=0
[[ "${LLM_INSTALL_DISCOVER_ONLY:-0}" == "1" ]] && DISCOVER_ONLY=1
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    --discover)
        DISCOVER_ONLY=1
        ;;
    "") ;;
    *)
        printf 'Unknown argument: %s\nRun %s --help for usage.\n' "$1" "$0" >&2
        exit 2
        ;;
esac

# ── Failure diagnostics + one-click GitHub issue ────────────────────────────
# Turn a failed install into "here's everything needed to fix it": on a fatal
# error, gather a sanitized diagnostic bundle and offer to file it as a
# pre-filled GitHub issue (one click, no account/token needed) or, if the gh
# CLI is set up, create it directly. Nothing leaves the machine until the user
# acts. Set LLM_INSTALL_NO_REPORT=1 to disable.
SRC_DIR="${SRC_DIR:-}"
INSTALL_STARTED=0
LAST_ERR_MSG=""
LAST_ERR_CMD=""
LAST_ERR_LINE=""
LAST_ERR_FN=""
REPORT_FILE=""

_redact() { sed -e "s#${HOME}#~#g" 2>/dev/null; }

_ver() { # _ver <bin> <version-args...>
    if command -v "$1" >/dev/null 2>&1; then "$@" 2>&1 | head -1; else echo 'not found'; fi
}

collect_diagnostics() {
    printf '### Environment\n'
    printf -- '- installer: install.sh (%s)\n' "$GITHUB_REPO"
    printf -- '- date: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null)"
    printf -- '- os: %s\n' "$(uname -srm 2>/dev/null)"
    [[ -r /etc/os-release ]] && printf -- '- distro: %s\n' "$( . /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-?}" )"
    printf -- '- shell: bash %s\n' "${BASH_VERSION:-?}"
    printf -- '- backend: requested=%s chosen=%s | mode=%s release=%s\n' "$BACKEND_REQUEST" "$BACKEND_CHOICE" "$INSTALL_MODE" "$INSTALL_RELEASE"
    printf '\n### Failure\n'
    printf -- '- message: %s\n' "${LAST_ERR_MSG:-<none captured>}"
    [[ -n "$LAST_ERR_CMD" ]] && printf -- '- command: `%s` (in %s, near line %s)\n' "$LAST_ERR_CMD" "${LAST_ERR_FN:-main}" "$LAST_ERR_LINE"
    printf '\n### Hardware\n'
    printf -- '- gpu: %s\n' "$(nvidia-smi -L 2>/dev/null | paste -sd'; ' - || echo 'nvidia-smi: none')"
    printf -- '- cpu: %s\n' "$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//' || uname -p 2>/dev/null)"
    printf -- '- avx: %s\n' "$(grep -m1 -oE 'avx512[a-z]*|avx2|avx' /proc/cpuinfo 2>/dev/null | paste -sd',' - || echo '?')"
    printf -- '- ram: %s\n' "$(awk '/MemTotal/{printf "%.0f GB", $2/1048576; exit}' /proc/meminfo 2>/dev/null || echo '?')"
    printf -- '- disk(%s): %s free\n' "$HOME" "$(df -h "$HOME" 2>/dev/null | awk 'NR==2{print $4}')"
    printf '\n### Tools\n'
    printf -- '- cmake: %s\n'   "$(_ver cmake --version)"
    printf -- '- gcc: %s\n'     "$(_ver gcc --version)"
    printf -- '- nvcc: %s\n'    "$(_ver nvcc --version)"
    printf -- '- git: %s\n'     "$(_ver git --version)"
    printf -- '- python3: %s\n' "$(_ver python3 --version)"
    printf -- '- go: %s\n'      "$(_ver go version)"
    printf -- '- glibc: %s\n'   "$(ldd --version 2>/dev/null | head -1 || echo '?')"
    printf -- '- vulkaninfo: %s\n' "$(command -v vulkaninfo >/dev/null 2>&1 && echo present || echo 'not found')"
}

urlencode() {
    local s="$1" o="" i c
    for ((i=0; i<${#s}; i++)); do
        c="${s:i:1}"
        case "$c" in
            [a-zA-Z0-9.~_-]) o+="$c" ;;
            *) printf -v c '%%%02X' "'$c"; o+="$c" ;;
        esac
    done
    printf '%s' "$o"
}

report_install_failure() {
    set +e
    local diag title body url msg="${LAST_ERR_MSG:-}"
    diag="$(collect_diagnostics | _redact)"
    REPORT_FILE="$HOME/ggrun-install-report.txt"
    printf '%s\n' "$diag" > "$REPORT_FILE" 2>/dev/null || REPORT_FILE=""

    say ""
    err "Install failed${msg:+: $msg}"
    [[ -n "$REPORT_FILE" ]] && say "  Saved diagnostics: $REPORT_FILE"

    title="install failed: ${msg:-$(uname -s 2>/dev/null) $BACKEND_CHOICE}"
    body="$diag

<!-- Add anything else above. Full report saved at: ${REPORT_FILE:-see terminal} -->"
    url="https://github.com/$GITHUB_REPO/issues/new?labels=install&title=$(urlencode "$title")&body=$(urlencode "$body")"

    if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
        if ask "Report this to the maintainers via GitHub now? [y/N]" n; then
            printf '%s' "$body" | gh issue create --repo "$GITHUB_REPO" --title "$title" --body-file - && return 0
        fi
    fi
    say "  Help us fix it — open a pre-filled issue (nothing is sent until you submit):"
    say "    $url"
}

# BASH_COMMAND is the innermost failing command while BASH_LINENO[0] is the line
# in the *calling* frame, so the two can name different places (issue 26 reported
# `IFS= read -r f` against the line of the call site). Record the enclosing
# function as well so a report identifies one unambiguous frame.
_on_err() { LAST_ERR_CMD="$BASH_COMMAND"; LAST_ERR_LINE="${BASH_LINENO[0]:-?}"; LAST_ERR_FN="${FUNCNAME[1]:-main}"; }
_on_exit() {
    local rc=$?
    set +e
    [[ -n "${SRC_DIR:-}" ]] && (( ! SRC_DIR_EXTERNAL )) && rm -rf "$SRC_DIR" 2>/dev/null
    if (( rc != 0 )) && (( rc != 130 )) && (( INSTALL_STARTED )) && [[ "${LLM_INSTALL_NO_REPORT:-0}" != 1 ]]; then
        report_install_failure "$rc"
    fi
}
trap '_on_err' ERR
trap '_on_exit' EXIT

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

if (( ! DISCOVER_ONLY )); then
    say "═══ ggrun installer ═══"
    INSTALL_STARTED=1
fi

# ── Stage 1: use local repo if present; clone only if source fallback needs it ──
SRC_DIR=""
SRC_DIR_EXTERNAL=0
if (( ! DISCOVER_ONLY )); then
    if [[ -n "$SCRIPT_DIR" && -f "$SCRIPT_DIR/go/go.mod" && -f "$SCRIPT_DIR/scripts/setup-home.sh" ]]; then
        SRC_DIR="$SCRIPT_DIR"
        SRC_DIR_EXTERNAL=1
        ok "Using local repo at $SRC_DIR"
    elif [[ -f "./go/go.mod" && -f "./scripts/setup-home.sh" ]]; then
        SRC_DIR="$(pwd)"
        SRC_DIR_EXTERNAL=1
        ok "Using local repo at $SRC_DIR"
    fi
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
}

if (( ! DISCOVER_ONLY )) && [[ -n "$SOURCE_REPO_DIR" ]]; then
    prepare_persistent_source_repo || true
fi

# ── Stage 2: detect platform + backend ──────────────────────────────────────
OS="$(uname -s)"

_discover_timeout() {
    if command -v timeout >/dev/null 2>&1; then
        timeout "$@"
    else
        shift
        "$@"
    fi
}

_discover_abspath() {
    readlink -f "$1" 2>/dev/null || printf '%s\n' "$1"
}

_discover_skip_path() {
    local p="$1"
    case "$p" in
        /proc/*|/sys/*|/dev/*|/run/*) return 0 ;;
    esac
    if [[ -n "${INSTALL_DIR:-}" ]]; then
        local dest
        dest="$(_discover_abspath "$INSTALL_DIR")"
        [[ "$p" == "$dest" || "$p" == "$dest"/* ]] && return 0
    fi
    return 1
}

_discover_locate() {
    local name="$1"
    if command -v plocate >/dev/null 2>&1; then
        # -b without a regex is a substring; llama-server-simulator.py must not match.
        _discover_timeout 8 plocate -b -r "^${name}$" 2>/dev/null \
            || _discover_timeout 8 plocate -b "$name" 2>/dev/null || true
    elif command -v locate >/dev/null 2>&1; then
        _discover_timeout 8 locate -b "\\$name" 2>/dev/null || true
    fi
    return 0
}

# A real backend is a native llama-server / ik_llama-server binary, not a
# Python simulator, shell stub, or examples/ helper with a similar name.
is_native_binary() {
    local hdr
    [[ -f "$1" && -r "$1" ]] || return 1
    hdr="$(head -c 4 "$1" 2>/dev/null || true)"
    [[ "$hdr" == $'\x7fELF' ]] && return 0
    [[ "${hdr:0:2}" == "MZ" ]] && return 0
    case "$hdr" in
        $'\xfe\xed\xfa\xcf'|$'\xcf\xfa\xed\xfe'|$'\xfe\xed\xfa\xce'|$'\xce\xfa\xed\xfe') return 0 ;;
    esac
    return 1
}

is_real_llama_server() {
    local p="$1" base
    [[ -n "$p" && -f "$p" && -x "$p" ]] || return 1
    base="$(basename "$p")"
    case "$base" in
        llama-server|llama-server.exe|ik_llama-server|ik_llama-server-cuda|ik_llama-server-vulkan) ;;
        *) return 1 ;;
    esac
    case "$p" in
        *simulator*|*llama-eval*|*/examples/*) return 1 ;;
    esac
    is_native_binary "$p"
}

# backend_actually_runs proves the candidate can execute, not merely that it is
# an ELF file with the right name.
#
# Reported as issue #28: a user's install adopted a llama-server already on the
# machine, and every check above passed -- correct filename, native binary, not
# a simulator -- while the binary could not load at all:
#
#   error while loading shared libraries: libllama-server-impl.so:
#   cannot open shared object file: No such file or directory
#
# Upstream split the server into a shared library
# (ggml-org/llama.cpp#23494), so a pre-split or partially installed build has
# the right name and an unsatisfiable NEEDED entry. Adoption then always
# succeeded and the launch always failed. Nothing here can be inferred from the
# file alone; the only way to know is to run it.
# run_bounded_probe executes a backend candidate with a hard time limit and no
# stdin, returning whatever it printed. A probe that times out returns nothing,
# which backend_actually_runs treats as "could not prove a failure" and accepts:
# an unresponsive --version is not evidence the binary is broken.
run_bounded_probe() {
    local p="$1"; shift
    if command -v timeout >/dev/null 2>&1; then
        timeout 10 "$p" "$@" 2>&1 </dev/null
    else
        "$p" "$@" 2>&1 </dev/null
    fi
}

backend_actually_runs() {
    local p="$1" out
    out="$(run_bounded_probe "$p" --version)"
    if [[ -z "$out" ]]; then
        out="$(run_bounded_probe "$p" --help)"
    fi
    # Reject ONLY on a definite loader or exec failure. Deliberately no
    # requirement on exit status or output content: llama-server exits non-zero
    # for --help on some builds, forks print different banners, and this runs on
    # hardware we cannot test. A false rejection would strand a user with a
    # working backend -- the same class of bug as #28, in the other direction.
    # Prove the failure, or accept.
    case "$out" in
        *"error while loading shared libraries"*|*"cannot open shared object file"*)
            BACKEND_RUN_ERROR="$(printf '%s' "$out" | head -n 1)"; return 1 ;;
        *"Exec format error"*|*"cannot execute binary file"*|*"symbol lookup error"*)
            BACKEND_RUN_ERROR="$(printf '%s' "$out" | head -n 1)"; return 1 ;;
    esac
    return 0
}

installed_real_server() {
    local p="$INSTALL_DIR/$1" t
    [[ -e "$p" || -L "$p" ]] || return 1
    t="$(_discover_abspath "$p")"
    is_real_llama_server "$t"
}

drop_fake_installed_backends() {
    local name p t
    for name in llama-server llama-server-vulkan llama-server-cuda ik_llama-server-cuda; do
        p="$INSTALL_DIR/$name"
        [[ -e "$p" || -L "$p" ]] || continue
        t="$(_discover_abspath "$p")"
        if ! is_real_llama_server "$t"; then
            warn "Ignoring $p (not a llama-server binary${t:+: $t})"
            rm -f "$p"
        fi
    done
}

find_nvidia_smi() {
    local c
    for c in \
        "$(command -v nvidia-smi 2>/dev/null || true)" \
        /usr/bin/nvidia-smi \
        /usr/local/bin/nvidia-smi \
        /usr/lib/nvidia/bin/nvidia-smi \
        /opt/nvidia/bin/nvidia-smi \
        /usr/lib64/nvidia/bin/nvidia-smi
    do
        [[ -n "$c" && -x "$c" ]] || continue
        printf '%s\n' "$(_discover_abspath "$c")"
        return 0
    done
    while IFS= read -r c; do
        [[ -n "$c" && -x "$c" ]] || continue
        _discover_skip_path "$c" && continue
        printf '%s\n' "$(_discover_abspath "$c")"
        return 0
    done < <(_discover_locate nvidia-smi | head -n 20)
    return 1
}

cuda_nvcc_path() {
    local c dir
    for c in \
        "$(command -v nvcc 2>/dev/null || true)" \
        "${CUDACXX:-}" \
        "${CUDA_HOME:+$CUDA_HOME/bin/nvcc}" \
        "${CUDA_PATH:+$CUDA_PATH/bin/nvcc}" \
        /usr/local/cuda/bin/nvcc \
        /usr/bin/nvcc \
        /opt/cuda/bin/nvcc \
        "$HOME/cuda/bin/nvcc" \
        "$HOME/.local/cuda/bin/nvcc"
    do
        [[ -n "$c" && -x "$c" ]] || continue
        printf '%s\n' "$(_discover_abspath "$c")"
        return 0
    done
    for dir in /usr/local/cuda-* /opt/cuda-*; do
        [[ -x "$dir/bin/nvcc" ]] || continue
        printf '%s\n' "$(_discover_abspath "$dir/bin/nvcc")"
        return 0
    done
    while IFS= read -r c; do
        [[ -n "$c" && -x "$c" && "$(basename "$c")" == nvcc ]] || continue
        _discover_skip_path "$c" && continue
        printf '%s\n' "$(_discover_abspath "$c")"
        return 0
    done < <(_discover_locate nvcc | head -n 20)
    return 1
}

has_cuda_toolkit() {
    local nvcc
    nvcc="$(cuda_nvcc_path 2>/dev/null || true)"
    [[ -n "$nvcc" ]] || return 1
    "$nvcc" --version >/dev/null 2>&1
}

vulkan_loader_present() {
    local lib
    for lib in /usr/lib64/libvulkan.so.1 /usr/lib/x86_64-linux-gnu/libvulkan.so.1 \
               /usr/lib/aarch64-linux-gnu/libvulkan.so.1 /usr/lib/libvulkan.so.1 \
               /usr/local/lib/libvulkan.so.1 /usr/local/lib64/libvulkan.so.1; do
        [[ -e "$lib" ]] && return 0
    done
    ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so\.1' && return 0
    return 1
}

vulkan_icd_present() {
    local f
    for f in /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json \
             /usr/lib64/nvidia/icd.d/*.json /usr/share/nvidia/nv_vulkan_wrapper.json; do
        [[ -e "$f" ]] && return 0
    done
    return 1
}

vulkan_available() {
    local info
    info="$(command -v vulkaninfo 2>/dev/null || true)"
    [[ -z "$info" && -x /usr/bin/vulkaninfo ]] && info=/usr/bin/vulkaninfo
    if [[ -n "$info" ]]; then
        "$info" --summary 2>/dev/null | grep -qi "GPU\|deviceName" && return 0
    fi
    vulkan_loader_present && vulkan_icd_present
}

has_nvidia_gpu() {
    local smi
    smi="$(find_nvidia_smi 2>/dev/null || true)"
    if [[ -n "$smi" ]] && "$smi" -L 2>/dev/null | grep -q GPU; then
        return 0
    fi
    [[ -e /dev/nvidia0 || -r /proc/driver/nvidia/version ]] && return 0
    [[ -e /usr/lib64/libcuda.so.1 || -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 || -e /usr/lib/libcuda.so.1 ]] && return 0
    if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -qiE 'VGA.*NVIDIA|3D.*NVIDIA|NVIDIA.*Controller'; then
        return 0
    fi
    return 1
}

FOUND_NVIDIA_SMI=""
FOUND_NVCC=""
FOUND_IK_SERVER=""
FOUND_VULKAN_SERVER=""
FOUND_CUDA_SERVER=""
FOUND_LLAMA_SERVER=""
FOUND_FORK_SERVERS=""

_path_score() {
    local p="$1" n=50
    case "$p" in
        *pre-rebrand*|*legacy*|*-prev-*|*/.cache/*) n=5 ;;
        "$HOME/ik_llama.cpp"/*) n=200 ;;
        "$HOME/llama.cpp"/*) n=180 ;;
        "$HOME/.local/bin"/*) n=160 ;;
        /usr/local/bin/*) n=150 ;;
        /usr/bin/*) n=140 ;;
        *fork-*|*fork_*) n=40 ;;
    esac
    [[ "$p" == *"/build-cuda/"* ]] && n=$((n + 8))
    [[ "$p" == *"/build-vulkan/"* ]] && n=$((n + 6))
    printf '%s\n' "$n"
}

_better_server() {
    local cur="$1" cand="$2"
    [[ -z "$cand" ]] && { printf '%s\n' "$cur"; return; }
    [[ -z "$cur" ]] && { printf '%s\n' "$cand"; return; }
    local sc cc
    sc="$(_path_score "$cur")"
    cc="$(_path_score "$cand")"
    if (( cc > sc )); then
        printf '%s\n' "$cand"
    else
        printf '%s\n' "$cur"
    fi
}

classify_llama_bin() {
    local path="$1" low
    low="$(printf '%s' "$path" | tr '[:upper:]' '[:lower:]')"
    case "$low" in
        *ik_llama*|*ik-llama*) printf 'ik\n' ;;
        *vulkan*) printf 'vulkan\n' ;;
        *fork-*|*fork_*) printf 'fork\n' ;;
        *cuda*) printf 'cuda\n' ;;
        *) printf 'llama\n' ;;
    esac
}

_collect_llama_bins() {
    local hit name root p
    for name in llama-server llama-server-cuda llama-server-vulkan \
                ik_llama-server ik_llama-server-cuda; do
        hit="$(command -v "$name" 2>/dev/null || true)"
        [[ -n "$hit" && -x "$hit" ]] && printf '%s\n' "$(_discover_abspath "$hit")"
    done
    for p in \
        "$HOME/ik_llama.cpp/build/bin/llama-server" \
        "$HOME/ik_llama.cpp/build-cuda/bin/llama-server" \
        "$HOME/llama.cpp/build/bin/llama-server" \
        "$HOME/llama.cpp/build-cuda/bin/llama-server" \
        "$HOME/llama.cpp/build-vulkan/bin/llama-server" \
        "$HOME/.local/bin/llama-server" \
        "$HOME/.local/bin/ik_llama-server-cuda" \
        "$HOME/ggrun/.bin/ik_llama-server-cuda" \
        "$HOME/ggrun/.bin/llama-server-vulkan" \
        "$HOME/ggrun/.bin/llama-server" \
        /usr/local/bin/llama-server \
        /usr/bin/llama-server \
        /opt/llama.cpp/build/bin/llama-server \
        /opt/ik_llama.cpp/build/bin/llama-server
    do
        [[ -x "$p" ]] && printf '%s\n' "$(_discover_abspath "$p")"
    done
    # Stay off /mnt and /media: those often hang on network mounts.
    # LLM_INSTALL_SCAN_ROOTS is a colon-separated override for tests and offline use.
    local roots="${LLM_INSTALL_SCAN_ROOTS:-}"
    if [[ -z "$roots" ]]; then
        roots="$HOME:$HOME/.local:/usr/local:/opt:/usr"
    fi
    local IFS=':'
    for root in $roots; do
        unset IFS
        [[ -d "$root" ]] || continue
        _discover_timeout 20 find "$root" -maxdepth 6 \
            \( -name .git -o -name .cache -o -name node_modules -o -name .Trash \
               -o -name Trash -o -name steamapps -o -name Proton -o -name proc \) -prune -o \
            -type f \( -name llama-server -o -name llama-server.exe -o -name 'ik_llama-server*' \) \
            -print 2>/dev/null || true
    done
    _discover_locate llama-server
    _discover_locate ik_llama-server
    return 0
}

scan_system_installs() {
    FOUND_NVIDIA_SMI="$(find_nvidia_smi 2>/dev/null || true)"
    FOUND_NVCC="$(cuda_nvcc_path 2>/dev/null || true)"
    FOUND_IK_SERVER=""
    FOUND_VULKAN_SERVER=""
    FOUND_CUDA_SERVER=""
    FOUND_LLAMA_SERVER=""
    FOUND_FORK_SERVERS=""
    [[ "${LLM_INSTALL_SCAN_SYSTEM:-1}" == "0" ]] && return 0

    local p kind seen=""
    while IFS= read -r p; do
        [[ -n "$p" && -x "$p" ]] || continue
        p="$(_discover_abspath "$p")"
        _discover_skip_path "$p" && continue
        is_real_llama_server "$p" || continue
        case " $seen " in
            *" $p "*) continue ;;
        esac
        seen+=" $p"
        kind="$(classify_llama_bin "$p")"
        case "$kind" in
            ik) FOUND_IK_SERVER="$(_better_server "$FOUND_IK_SERVER" "$p")" ;;
            vulkan) FOUND_VULKAN_SERVER="$(_better_server "$FOUND_VULKAN_SERVER" "$p")" ;;
            cuda) FOUND_CUDA_SERVER="$(_better_server "$FOUND_CUDA_SERVER" "$p")" ;;
            fork) FOUND_FORK_SERVERS+="$p"$'\n' ;;
            *) FOUND_LLAMA_SERVER="$(_better_server "$FOUND_LLAMA_SERVER" "$p")" ;;
        esac
    done < <(_collect_llama_bins | sed '/^$/d' | sort -u)
    return 0
}

report_system_installs() {
    say "── Already on this machine ──"
    if [[ -n "$FOUND_NVIDIA_SMI" ]]; then
        ok "NVIDIA driver: $FOUND_NVIDIA_SMI"
    else
        say "  NVIDIA driver: not found"
    fi
    if [[ -n "$FOUND_NVCC" ]]; then
        ok "CUDA toolkit:  $FOUND_NVCC"
    else
        say "  CUDA toolkit:  not found"
    fi
    if [[ -n "$FOUND_IK_SERVER" ]]; then
        ok "ik_llama.cpp:  $FOUND_IK_SERVER"
    else
        say "  ik_llama.cpp:  not found"
    fi
    if [[ -n "$FOUND_VULKAN_SERVER" ]]; then
        ok "llama.cpp Vulkan: $FOUND_VULKAN_SERVER"
    fi
    if [[ -n "$FOUND_CUDA_SERVER" ]]; then
        ok "llama.cpp CUDA: $FOUND_CUDA_SERVER"
    fi
    if [[ -n "$FOUND_LLAMA_SERVER" ]]; then
        ok "llama.cpp:     $FOUND_LLAMA_SERVER"
    fi
    if [[ -z "$FOUND_VULKAN_SERVER$FOUND_CUDA_SERVER$FOUND_LLAMA_SERVER" ]]; then
        say "  llama.cpp:     not found"
    fi
    if [[ -n "$FOUND_FORK_SERVERS" ]]; then
        local f
        while IFS= read -r f; do
            [[ -n "$f" ]] && ok "fork:          $f"
        done <<<"$FOUND_FORK_SERVERS"
    fi
    # FOUND_FORK_SERVERS always ends in a newline (see the fork case in
    # scan_system_installs), so the loop above ends on an empty line and its
    # final [[ -n "$f" ]] test is false. That false status would become this
    # function's return value and abort the whole installer under `set -e`,
    # which is exactly how an existing install used to break the upgrade.
    # Reporting what was found can never fail the install.
    return 0
}

print_discover_kv() {
    [[ -n "$FOUND_NVIDIA_SMI" ]] && printf 'nvidia_smi=%s\n' "$FOUND_NVIDIA_SMI"
    [[ -n "$FOUND_NVCC" ]] && printf 'nvcc=%s\n' "$FOUND_NVCC"
    [[ -n "$FOUND_IK_SERVER" ]] && printf 'ik_llama=%s\n' "$FOUND_IK_SERVER"
    [[ -n "$FOUND_VULKAN_SERVER" ]] && printf 'llama_vulkan=%s\n' "$FOUND_VULKAN_SERVER"
    [[ -n "$FOUND_CUDA_SERVER" ]] && printf 'llama_cuda=%s\n' "$FOUND_CUDA_SERVER"
    [[ -n "$FOUND_LLAMA_SERVER" ]] && printf 'llama=%s\n' "$FOUND_LLAMA_SERVER"
    local f
    while IFS= read -r f; do
        [[ -n "$f" ]] && printf 'fork=%s\n' "$f"
    done <<<"${FOUND_FORK_SERVERS:-}"
    return 0
}

link_existing_backend() {
    local src="$1" dest="$2"
    # Adoption is the only place this check belongs. Putting it in
    # is_real_llama_server made drop_fake_installed_backends delete a correctly
    # installed backend whenever the probe failed for any reason -- a false
    # rejection that strands a user with no backend at all, which is worse than
    # the bug it guards against. Here the cost of refusing is only that we fall
    # back to the bundle ggrun ships.
    [[ -x "$src" && -n "$dest" ]] || return 1
    BACKEND_RUN_ERROR=""
    if ! is_real_llama_server "$src" || ! backend_actually_runs "$src"; then
        # Say why, and say which binary. Issue #28's reporter could not tell
        # that ggrun was running a binary it had adopted from elsewhere on the
        # machine rather than the one it shipped.
        if [[ -n "$BACKEND_RUN_ERROR" ]]; then
            warn "Ignoring existing $dest at $src: $BACKEND_RUN_ERROR"
        fi
        return 1
    fi
    mkdir -p "$INSTALL_DIR"
    if [[ -e "$INSTALL_DIR/$dest" || -L "$INSTALL_DIR/$dest" ]]; then
        return 0
    fi
    ln -sfn "$src" "$INSTALL_DIR/$dest"
    ok "Using existing $dest at $src"
}

adopt_system_backends() {
    local used=0
    if [[ -n "$FOUND_IK_SERVER" ]]; then
        link_existing_backend "$FOUND_IK_SERVER" ik_llama-server-cuda && used=1
    fi
    if [[ -n "$FOUND_VULKAN_SERVER" ]]; then
        link_existing_backend "$FOUND_VULKAN_SERVER" llama-server-vulkan && used=1
    fi
    if [[ -n "$FOUND_CUDA_SERVER" && ! -e "$INSTALL_DIR/ik_llama-server-cuda" ]]; then
        link_existing_backend "$FOUND_CUDA_SERVER" llama-server-cuda && used=1
    fi
    if [[ -n "$FOUND_LLAMA_SERVER" && ! -e "$INSTALL_DIR/llama-server" && ! -e "$INSTALL_DIR/llama-server-vulkan" ]]; then
        link_existing_backend "$FOUND_LLAMA_SERVER" llama-server && used=1
    fi
    if [[ -n "$FOUND_CUDA_SERVER" && ! -e "$INSTALL_DIR/llama-server" && ! -e "$INSTALL_DIR/llama-server-vulkan" ]]; then
        link_existing_backend "$FOUND_CUDA_SERVER" llama-server && used=1
    fi
    if [[ -n "$FOUND_FORK_SERVERS" && ! -e "$INSTALL_DIR/llama-server" && ! -e "$INSTALL_DIR/llama-server-vulkan" ]]; then
        local fork
        fork="$(printf '%s\n' "$FOUND_FORK_SERVERS" | sed '/^$/d' | head -n 1)"
        if [[ -n "$fork" ]]; then
            link_existing_backend "$fork" llama-server && used=1
        fi
    fi
    link_default_llama_server
    (( used ))
}

if (( DISCOVER_ONLY )); then
    scan_system_installs || true
    print_discover_kv || true
    exit 0
fi

# Linux auto: CUDA first when an NVIDIA GPU is present, then Vulkan, then CPU.
# macOS: Metal. Never pick CUDA on a machine with no NVIDIA device.
auto_backend_candidates() {
    if [[ "$OS" == "Darwin" ]]; then
        echo metal
        return
    fi
    if [[ "$OS" == MINGW* || "$OS" == MSYS* || "$OS" == CYGWIN* ]]; then
        err "Use install.ps1 for native Windows installs, or run this Bash installer on Linux/macOS."
        exit 1
    fi
    if has_nvidia_gpu; then
        echo cuda
    fi
    echo vulkan
    echo cpu
}

detect_backend() {
    auto_backend_candidates | head -n 1
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
            if ask "Install missing system build dependencies (git, cmake, build-essential, etc.) via sudo? You may be asked for your password. [Y/n]" y; then
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
    if ! curl -fsL --show-error "$url" -o "$archive"; then
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
    local platform asset url sums_url tmp archive payload_root found_backend=0 dest_name="${1:-}"
    [[ "$BACKEND_CHOICE" == "skip" ]] && return 1
    # CUDA release bundles are published for supported Linux x86_64 hosts. If a
    # matching bundle is unavailable, auto mode can still fall back to Vulkan or
    # CPU before attempting a source build.
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
    if ! curl -fsL --show-error "$url" -o "$archive"; then
        rm -rf "$tmp"
        return 1
    fi
    sums_url="$(find_release_asset_url "SHA256SUMS" || true)"
    if [[ -n "$sums_url" ]]; then
        if ! curl -fsL --show-error "$sums_url" -o "$tmp/SHA256SUMS"; then
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
    for f in setup.sh setup-linux.sh setup-mac.sh ggrun llm-server-bash llm-server-go parse_gguf.py model_index.py download_any_gguf.py measure_bandwidth.py; do
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

    if [[ -z "$dest_name" ]]; then
        case "$BACKEND_CHOICE" in
            cuda) dest_name="ik_llama-server-cuda" ;;
            vulkan) dest_name="llama-server-vulkan" ;;
            *) dest_name="llama-server" ;;
        esac
    fi
    for candidate in "$payload_root/llama-server" "$payload_root/bin/llama-server"; do
        if [[ -f "$candidate" ]]; then
            # A checksummed CUDA ELF stays even if --version fails (missing
            # NCCL/cudart). Deleting it made setup compile and skip llama.cpp.
            if place_isolated_backend "$candidate" "$dest_name" 1; then
                found_backend=1
                ok "Installed $dest_name from $asset"
            fi
            break
        fi
    done

    rm -rf "$tmp"
    (( found_backend ))
}

host_libc() {
    if [[ -e /lib/ld-musl-x86_64.so.1 || -e /lib/ld-musl-aarch64.so.1 || -e /lib64/ld-musl-x86_64.so.1 ]]; then
        printf 'musl\n'
        return 0
    fi
    printf 'glibc\n'
}

# ggml-org and ggrun Linux prebuilts are glibc x86_64. Other hosts reuse a
# local llama-server or compile.
host_can_run_ubuntu_prebuilt() {
    [[ "${OS:-}" == "Linux" ]] || return 1
    case "$(uname -m)" in
        x86_64|amd64) ;;
        *) return 1 ;;
    esac
    [[ "$(host_libc)" == "glibc" ]]
}

# runs: binary starts on this machine.
# needs_gpu: binary is valid but Vulkan/CUDA runtime is missing.
# broken: wrong arch, musl vs glibc, or missing the bundle's own libraries.
classify_probe_output() {
    local out="$1"
    if [[ "$out" == *"Exec format error"* || "$out" == *"cannot execute binary file"* ]]; then
        printf 'broken\n'
        return 0
    fi
    if [[ "$out" == *GLIBC_* || "$out" == *"ld-linux"* && "$out" == *"not found"* ]]; then
        printf 'broken\n'
        return 0
    fi
    if [[ "$out" == *"cannot open shared object file"* || "$out" == *"error while loading shared libraries"* ]]; then
        if [[ "$out" == *libvulkan* || "$out" == *libcuda* || "$out" == *libcudart* \
            || "$out" == *libcublas* || "$out" == *libnccl* || "$out" == *libnvidia* \
            || "$out" == *vulkan* || "$out" == *ICD* || "$out" == *icd* || "$out" == *cuInit* ]]; then
            printf 'needs_gpu\n'
            return 0
        fi
        printf 'broken\n'
        return 0
    fi
    if [[ "$out" == *libvulkan* || "$out" == *libcuda* || "$out" == *libcudart* \
        || "$out" == *libcublas* || "$out" == *libnccl* || "$out" == *libnvidia* \
        || "$out" == *vulkan* || "$out" == *ICD* || "$out" == *icd* || "$out" == *cuInit* ]]; then
        printf 'needs_gpu\n'
        return 0
    fi
    if [[ "$out" == *[Vv]ersion* && "$out" != *"not found"* ]]; then
        printf 'runs\n'
        return 0
    fi
    printf 'needs_gpu\n'
}

probe_llama_server() {
    local bin="$1" libdir="${2:-}" out=""
    PROBE_LAST_OUT=""
    PROBE_LAST_KIND="broken"
    [[ -z "$libdir" ]] && libdir="$(dirname "$bin")"
    [[ -x "$bin" ]] || { printf 'broken\n'; return 1; }
    if command -v timeout >/dev/null 2>&1; then
        out="$(timeout 8 env LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$bin" --version 2>&1)" || true
    else
        out="$(env LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$bin" --version 2>&1)" || true
    fi
    PROBE_LAST_OUT="$out"
    PROBE_LAST_KIND="$(classify_probe_output "$out")"
    printf '%s\n' "$PROBE_LAST_KIND"
    [[ "$PROBE_LAST_KIND" != "broken" ]]
}

warn_probe_detail() {
    local line
    line="$(printf '%s' "${PROBE_LAST_OUT:-}" | tr '\n' ' ' | sed 's/^[[:space:]]\+//;s/[[:space:]]\+$//;s/[[:space:]]\+/ /g')"
    [[ ${#line} -gt 240 ]] && line="${line:0:240}…"
    [[ -n "$line" ]] && warn "  $line"
    # A probe that printed nothing leaves $line empty, making the test above
    # false and this function return 1. Every caller invokes it bare under
    # `set -e`, so that would abort the install before the `return 0` or the
    # cleanup that follows it -- turning "backend did not start, fall back"
    # into a hard failure. Printing a detail line can never fail the install.
    return 0
}

# Copy a .so into the isolated box. Follow toolkit symlinks so libcudart.so.12
# is a real file (or a symlink next to its target), not a dangling pointer.
copy_resolved_lib() {
    local lib="$1" box="$2" base dest real
    [[ -e "$lib" || -L "$lib" ]] || return 0
    base="$(basename "$lib")"
    dest="$box/$base"
    [[ -e "$dest" || -L "$dest" ]] && return 0
    if [[ -L "$lib" ]]; then
        real="$(readlink -f "$lib" 2>/dev/null || true)"
        if [[ -n "$real" && -f "$real" ]]; then
            if [[ "$(basename "$real")" != "$base" ]]; then
                [[ -e "$box/$(basename "$real")" || -L "$box/$(basename "$real")" ]] \
                    || install -m 0644 "$real" "$box/$(basename "$real")"
                ln -sfn "$(basename "$real")" "$dest"
            else
                install -m 0644 "$real" "$dest"
            fi
            return 0
        fi
        cp -a "$lib" "$dest" || true
        return 0
    fi
    install -m 0644 "$lib" "$dest"
}

# CUDA runtime only — never libcuda.so.1 (that is the driver).
harvest_cuda_runtime_libs() {
    local box="$1" dir base lib so line soname resolved
    local dirs=()
    [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]] && dirs+=("${CUDA_HOME}/lib64")
    [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib" ]] && dirs+=("${CUDA_HOME}/lib")
    [[ -n "${CUDA_PATH:-}" && -d "${CUDA_PATH}/lib64" ]] && dirs+=("${CUDA_PATH}/lib64")
    dirs+=(/usr/local/cuda/lib64 /usr/local/cuda/lib /usr/lib64 /usr/lib /usr/lib/x86_64-linux-gnu)
    for dir in /usr/local/cuda-*/lib64 /usr/local/cuda/targets/*/lib; do
        [[ -d "$dir" ]] && dirs+=("$dir")
    done
    for dir in "${dirs[@]}"; do
        [[ -d "$dir" ]] || continue
        for base in libcudart libcublas libcublasLt libnccl; do
            while IFS= read -r lib; do
                copy_resolved_lib "$lib" "$box"
            done < <(find "$dir" -maxdepth 1 \( -type f -o -type l \) -name "${base}.so*" 2>/dev/null | sort)
        done
    done
    command -v ldd >/dev/null 2>&1 || return 0
    for so in "$box"/llama-server "$box"/libggml.so "$box"/libggml.so.* "$box"/libllama.so "$box"/libllama.so.*; do
        [[ -e "$so" ]] || continue
        while IFS= read -r line; do
            soname="${line%% *}"
            case "$soname" in
                libcudart.so*|libcublas.so*|libcublasLt.so*|libnccl.so*) ;;
                *) continue ;;
            esac
            resolved="$(printf '%s\n' "$line" | awk '$2 == "=>" && $3 ~ /^\// { print $3 }')"
            [[ -n "$resolved" && -e "$resolved" ]] || continue
            copy_resolved_lib "$resolved" "$box"
            [[ -e "$box/$soname" || -L "$box/$soname" ]] || ln -sfn "$(basename "$resolved")" "$box/$soname"
        done < <(ldd "$so" 2>/dev/null || true)
    done
}

copy_backend_libs() {
    local src_dir="$1" box="$2" extra lib
    for extra in "$src_dir" "$src_dir/lib" "$(dirname "$src_dir")/lib"; do
        [[ -d "$extra" ]] || continue
        while IFS= read -r lib; do
            if [[ -L "$lib" ]]; then
                cp -a "$lib" "$box/$(basename "$lib")"
            else
                install -m 0644 "$lib" "$box/$(basename "$lib")"
            fi
        done < <(find "$extra" -maxdepth 1 \( -type f -o -type l \) \( -name 'lib*.so*' -o -name 'lib*.dylib' -o -name '*.dll' \) 2>/dev/null | sort)
    done
    harvest_cuda_runtime_libs "$box"
    local f base soname
    for f in "$box"/lib*.so.*; do
        [[ -e "$f" ]] || continue
        base="$(basename "$f")"
        soname="$(printf '%s\n' "$base" | sed -E 's/(\.so\.[0-9]+)\.[0-9].*/\1/')"
        [[ -n "$soname" && "$soname" != "$base" && ! -e "$box/$soname" ]] || continue
        ln -sfn "$base" "$box/$soname"
    done
}

backend_probe_kind() {
    local name="$1" p t
    p="$INSTALL_DIR/$name"
    [[ -e "$p" || -L "$p" ]] || { printf 'missing\n'; return 1; }
    t="$(_discover_abspath "$p")"
    probe_llama_server "$t" "$(dirname "$t")"
}

# Keep each backend's shared libraries next to its binary so ik_llama.cpp
# and llama.cpp can both live in .bin without overwriting each other.
place_isolated_backend() {
    local src="$1" dest_name="$2" keep="${3:-0}" box kind
    [[ -f "$src" && -n "$dest_name" ]] || return 1
    case "$dest_name" in
        ik_llama-server-cuda|llama-server-cuda) keep=1 ;;
    esac
    box="$INSTALL_DIR/backends/$dest_name"
    mkdir -p "$box"
    install -m 0755 "$src" "$box/llama-server"
    copy_backend_libs "$(dirname "$src")" "$box"
    ln -sfn "backends/$dest_name/llama-server" "$INSTALL_DIR/$dest_name"
    [[ -x "$box/llama-server" && -e "$INSTALL_DIR/$dest_name" ]] || return 1
    # Do not capture stdout: probe stores the --version text in PROBE_LAST_OUT.
    probe_llama_server "$box/llama-server" "$box" >/dev/null || true
    kind="${PROBE_LAST_KIND:-broken}"
    # The published CUDA libggml needs libnccl.so.2. That is not the NVIDIA
    # driver and is not in a typical CUDA toolkit. Fetch it into the box.
    if [[ "$kind" != "runs" ]] && [[ "$dest_name" == *cuda* ]] \
        && [[ "${PROBE_LAST_OUT}" == *libnccl* ]]; then
        if fetch_nccl_into_box "$box"; then
            probe_llama_server "$box/llama-server" "$box" >/dev/null || true
            kind="${PROBE_LAST_KIND:-broken}"
        fi
    fi
    case "$kind" in
        runs) return 0 ;;
        needs_gpu)
            warn "$dest_name installed but needs a GPU runtime on this machine (kept; llama.cpp will still be installed)"
            warn_probe_detail
            return 0
            ;;
        *)
            if [[ "$keep" == "1" ]]; then
                warn "$dest_name did not start; keeping the downloaded binary"
                warn_probe_detail
                return 0
            fi
            warn "$dest_name cannot start on this machine (wrong libc/arch or incomplete bundle)"
            warn_probe_detail
            rm -f "$INSTALL_DIR/$dest_name"
            rm -rf "$box"
            return 1
            ;;
    esac
}

box_has_libnccl() {
    local box="$1" f
    for f in "$box"/libnccl.so.2 "$box"/libnccl.so.2.*; do
        [[ -e "$f" ]] && return 0
    done
    return 1
}

# NVIDIA redist (no login). NCCL is a separate ~210 MB library from the driver.
# Override with LLM_INSTALL_NCCL_URL= (tests use a tiny local archive).
NCCL_REDIST_URL="${LLM_INSTALL_NCCL_URL:-https://developer.download.nvidia.com/compute/redist/nccl/v2.21.5/nccl_2.21.5-1+cuda12.4_x86_64.txz}"

fetch_nccl_into_box() {
    local box="$1" tmp archive lib
    [[ -d "$box" ]] || return 1
    box_has_libnccl "$box" && return 0
    command -v curl >/dev/null 2>&1 || return 1
    command -v tar >/dev/null 2>&1 || return 1
    local url="${LLM_INSTALL_NCCL_URL:-${NCCL_REDIST_URL:-https://developer.download.nvidia.com/compute/redist/nccl/v2.21.5/nccl_2.21.5-1+cuda12.4_x86_64.txz}}"
    say "Downloading NCCL (~210 MB) so the CUDA backend can load (not part of the NVIDIA driver)..."
    tmp="$(mktemp -d -t ggrun-nccl.XXXXXX)"
    archive="$tmp/nccl.txz"
    if ! curl -fL --retry 3 -A ggrun-installer "$url" -o "$archive"; then
        rm -rf "$tmp"
        warn "Could not download NCCL. CUDA needs libnccl.so.2 next to llama-server."
        return 1
    fi
    mkdir -p "$tmp/out"
    if ! tar -xJf "$archive" -C "$tmp/out" 2>/dev/null && ! tar -xf "$archive" -C "$tmp/out" 2>/dev/null; then
        rm -rf "$tmp"
        warn "Could not extract NCCL archive."
        return 1
    fi
    while IFS= read -r lib; do
        copy_resolved_lib "$lib" "$box"
    done < <(find "$tmp/out" \( -type f -o -type l \) -name 'libnccl.so*' 2>/dev/null | sort)
    rm -rf "$tmp"
    if box_has_libnccl "$box"; then
        ok "Installed libnccl into $box"
        return 0
    fi
    warn "NCCL archive did not contain libnccl.so.2"
    return 1
}

link_default_llama_server() {
    local name t kind best=""
    for name in ik_llama-server-cuda llama-server-vulkan llama-server-cuda llama-server; do
        [[ -e "$INSTALL_DIR/$name" || -L "$INSTALL_DIR/$name" ]] || continue
        kind="$(backend_probe_kind "$name" || true)"
        if [[ "$kind" == "runs" ]]; then
            best="$name"
            break
        fi
    done
    if [[ -z "$best" ]]; then
        for name in ik_llama-server-cuda llama-server-vulkan llama-server-cuda llama-server; do
            [[ -e "$INSTALL_DIR/$name" || -L "$INSTALL_DIR/$name" ]] || continue
            best="$name"
            break
        done
    fi
    [[ -n "$best" && "$best" != "llama-server" ]] || return 0
    ln -sfn "$best" "$INSTALL_DIR/llama-server"
}

# ggml-org ships Linux CPU and Vulkan prebuilts, not Linux CUDA.
# Used when a ggrun release has no matching bundle so the one-liner still
# leaves a working llama-server.
install_upstream_linux_prebuilt() {
    local want="$1" dest="${2:-}" api tmp archive url name server
    host_can_run_ubuntu_prebuilt || return 1
    command -v curl >/dev/null 2>&1 || return 1
    command -v tar >/dev/null 2>&1 || return 1
    case "$want" in
        vulkan) name="ubuntu-vulkan-x64" ;;
        cpu)    name="ubuntu-x64" ;;
        *) return 1 ;;
    esac
    say "Trying ggml-org llama.cpp $want prebuilt..."
    api="$(mktemp)"
    if ! curl -fsSL -A ggrun-installer "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" -o "$api"; then
        rm -f "$api"
        return 1
    fi
    url="$(python3 - "$api" "$name" <<'PY' 2>/dev/null || true
import json, sys
want = sys.argv[2]
data = json.load(open(sys.argv[1]))
for a in data.get("assets") or []:
    n = a.get("name") or ""
    if n.endswith("bin-" + want + ".tar.gz") or ("bin-" + want + ".tar.gz" in n):
        print(a.get("browser_download_url") or "")
        break
PY
)"
    rm -f "$api"
    # grep fallback if python is missing
    if [[ -z "$url" ]]; then
        url="$(curl -fsSL -A ggrun-installer "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" \
            | grep -Eo 'https://github.com/ggml-org/llama.cpp/releases/download/[^"]+bin-ubuntu[^"]+\.tar\.gz' \
            | grep -F "$name" | head -n 1 || true)"
    fi
    [[ -n "$url" ]] || return 1
    tmp="$(mktemp -d -t ggrun-llama-prebuilt.XXXXXX)"
    archive="$tmp/llama.tgz"
    if ! curl -fL --retry 3 -A ggrun-installer "$url" -o "$archive"; then
        rm -rf "$tmp"
        return 1
    fi
    mkdir -p "$tmp/out"
    if ! tar -xzf "$archive" -C "$tmp/out"; then
        rm -rf "$tmp"
        return 1
    fi
    server="$(find "$tmp/out" -type f -name llama-server -print -quit)"
    if [[ -z "$server" || ! -f "$server" ]]; then
        rm -rf "$tmp"
        return 1
    fi
    if [[ -z "$dest" ]]; then
        case "$want" in
            vulkan) dest="llama-server-vulkan" ;;
            *) dest="llama-server" ;;
        esac
    fi
    if ! place_isolated_backend "$server" "$dest"; then
        rm -rf "$tmp"
        return 1
    fi
    rm -rf "$tmp"
    BACKEND_CHOICE="$want"
    ok "Installed ggml-org llama.cpp $want as $dest"
    return 0
}

backend_kind_runs() {
    local name="$1" kind
    kind="$(backend_probe_kind "$name" || true)"
    [[ "$kind" == "runs" ]]
}

install_distro_packages() {
    local pkgs=("$@")
    [[ ${#pkgs[@]} -gt 0 ]] || return 1
    if command -v apt-get >/dev/null 2>&1; then
        run_privileged apt-get update && run_privileged apt-get install -y "${pkgs[@]}"
    elif command -v dnf >/dev/null 2>&1; then
        run_privileged dnf install -y "${pkgs[@]}"
    elif command -v yum >/dev/null 2>&1; then
        run_privileged yum install -y "${pkgs[@]}"
    elif command -v pacman >/dev/null 2>&1; then
        run_privileged pacman -Sy --needed --noconfirm "${pkgs[@]}"
    elif command -v zypper >/dev/null 2>&1; then
        run_privileged zypper install -y "${pkgs[@]}"
    else
        return 1
    fi
}

vulkan_runtime_packages() {
    if command -v apt-get >/dev/null 2>&1; then
        printf '%s\n' libvulkan1 vulkan-tools
        has_nvidia_gpu || printf '%s\n' mesa-vulkan-drivers
    elif command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
        printf '%s\n' vulkan-loader vulkan-tools
        has_nvidia_gpu || printf '%s\n' mesa-vulkan-drivers
    elif command -v pacman >/dev/null 2>&1; then
        printf '%s\n' vulkan-icd-loader vulkan-tools
        has_nvidia_gpu || printf '%s\n' vulkan-intel vulkan-radeon
    elif command -v zypper >/dev/null 2>&1; then
        printf '%s\n' libvulkan1 vulkan-tools
        has_nvidia_gpu || printf '%s\n' Mesa-vulkan-device-select
    fi
}

cuda_toolkit_packages() {
    if command -v apt-get >/dev/null 2>&1; then
        printf '%s\n' nvidia-cuda-toolkit
    elif command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
        printf '%s\n' cuda-toolkit
    elif command -v pacman >/dev/null 2>&1; then
        printf '%s\n' cuda
    elif command -v zypper >/dev/null 2>&1; then
        printf '%s\n' cuda
    fi
}

# Install Vulkan loader + ICD when the GPU binary cannot start without them.
# Never installs the NVIDIA proprietary driver.
ensure_vulkan_runtime() {
    vulkan_available && return 0
    [[ "$OS" == "Linux" ]] || return 1
    [[ "$DEPS_MODE" == "skip" ]] && return 1
    local pkgs=()
    mapfile -t pkgs < <(vulkan_runtime_packages)
    [[ ${#pkgs[@]} -gt 0 ]] || return 1
    say "Vulkan loader/ICD is not installed. llama.cpp Vulkan needs it on this distro."
    if (( NONINTERACTIVE )) && [[ "$DEPS_MODE" != "install" ]]; then
        warn "Install: ${pkgs[*]}"
        return 1
    fi
    if [[ "$DEPS_MODE" == "install" ]] || ask "Install ${pkgs[*]} with the system package manager? [Y/n]" y; then
        say "Installing Vulkan runtime: ${pkgs[*]}"
        install_distro_packages "${pkgs[@]}" || return 1
    else
        return 1
    fi
    vulkan_available
}

# Install nvcc only when this machine already has an NVIDIA driver.
ensure_cuda_toolkit() {
    has_cuda_toolkit && return 0
    has_nvidia_gpu || return 1
    [[ "$OS" == "Linux" ]] || return 1
    [[ "$DEPS_MODE" == "skip" ]] && return 1
    local pkgs=()
    mapfile -t pkgs < <(cuda_toolkit_packages)
    [[ ${#pkgs[@]} -gt 0 ]] || return 1
    say "CUDA toolkit (nvcc) is not installed. Needed to build ik_llama.cpp."
    if (( NONINTERACTIVE )) && [[ "$DEPS_MODE" != "install" ]]; then
        warn "Install: ${pkgs[*]}"
        return 1
    fi
    local default=n
    [[ "$BACKEND_REQUEST" == "cuda" ]] && default=y
    if [[ "$DEPS_MODE" == "install" ]] || ask "Install ${pkgs[*]}? This is a large download. [Y/n]" "$default"; then
        say "Installing CUDA toolkit: ${pkgs[*]}"
        install_distro_packages "${pkgs[@]}" || return 1
    else
        return 1
    fi
    has_cuda_toolkit
}

compiler_major() {
    local cc="$1" ver
    [[ -x "$cc" || -n "$(command -v "$cc" 2>/dev/null)" ]] || return 1
    ver="$("$cc" -dumpfullversion 2>/dev/null || "$cc" -dumpversion 2>/dev/null || true)"
    [[ -n "$ver" ]] || return 1
    printf '%s\n' "${ver%%.*}"
}

# nvcc ships a hard cap on the host GCC. CUDA 12.3 refuses GCC 13+.
cuda_max_host_gcc_for() {
    case "$1" in
        11.*) printf '11\n' ;;
        12.0|12.1|12.2|12.3) printf '12\n' ;;
        12.4|12.5) printf '13\n' ;;
        12.*) printf '13\n' ;;
        13.*) printf '14\n' ;;
        *) printf '12\n' ;;
    esac
}

cuda_max_host_gcc() {
    local nvcc ver
    nvcc="$(cuda_nvcc_path 2>/dev/null || true)"
    [[ -n "$nvcc" ]] || { printf '12\n'; return 0; }
    ver="$("$nvcc" --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2; exit}')"
    cuda_max_host_gcc_for "$ver"
}

# Newest g++ that this nvcc will accept. Fedora 42+ ships GCC 16; CUDA 12.3
# needs gcc-12 / g++-12 from the distro.
find_cuda_host_cxx() {
    local max cand n best="" best_n=0
    max="$(cuda_max_host_gcc)"
    for cand in \
        g++-14 g++-13 g++-12 g++-11 \
        /usr/bin/g++-14 /usr/bin/g++-13 /usr/bin/g++-12 /usr/bin/g++-11 \
        /usr/bin/g++14 /usr/bin/g++13 /usr/bin/g++12 \
        "$(command -v g++ 2>/dev/null || true)"
    do
        [[ -n "$cand" ]] || continue
        command -v "$cand" >/dev/null 2>&1 || [[ -x "$cand" ]] || continue
        n="$(compiler_major "$cand" || true)"
        [[ -n "$n" ]] || continue
        if (( n <= max && n >= 9 && n > best_n )); then
            best="$cand"
            best_n="$n"
        fi
    done
    [[ -n "$best" ]] || return 1
    printf '%s\n' "$(command -v "$best" 2>/dev/null || printf '%s' "$best")"
}

cuda_host_compiler_packages() {
    local max
    max="$(cuda_max_host_gcc)"
    if command -v apt-get >/dev/null 2>&1; then
        printf 'gcc-%s\n' "$max"
        printf 'g++-%s\n' "$max"
    elif command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
        printf 'gcc%s\n' "$max"
        printf 'gcc%s-c++\n' "$max"
    elif command -v zypper >/dev/null 2>&1; then
        printf 'gcc%s\n' "$max"
        printf 'gcc%s-c++\n' "$max"
    fi
}

ensure_cuda_host_compiler() {
    local max cur host pkgs=()
    CUDA_HOST_CXX="${CUDA_HOST_CXX:-}"
    CUDA_ALLOW_UNSUPPORTED="${CUDA_ALLOW_UNSUPPORTED:-0}"
    max="$(cuda_max_host_gcc)"
    cur="$(compiler_major g++ || compiler_major c++ || echo 99)"
    if host="$(find_cuda_host_cxx)"; then
        CUDA_HOST_CXX="$host"
        if (( cur > max )); then
            ok "CUDA host compiler: $host (system GCC $cur is newer than CUDA allows)"
        fi
        return 0
    fi
    [[ "$OS" == "Linux" ]] || return 1
    [[ "$DEPS_MODE" == "skip" ]] && return 1
    mapfile -t pkgs < <(cuda_host_compiler_packages)
    if [[ ${#pkgs[@]} -gt 0 ]]; then
        say "This CUDA toolkit does not support GCC $cur (max GCC $max)."
        if (( NONINTERACTIVE )) && [[ "$DEPS_MODE" != "install" ]]; then
            warn "Install a compatible host compiler: ${pkgs[*]}"
        elif [[ "$DEPS_MODE" == "install" ]] || ask "Install ${pkgs[*]} so CUDA can compile? [Y/n]" y; then
            install_distro_packages "${pkgs[@]}" || true
            if host="$(find_cuda_host_cxx)"; then
                CUDA_HOST_CXX="$host"
                ok "CUDA host compiler: $host"
                return 0
            fi
        fi
    fi
    warn "No GCC <= $max found. Passing nvcc --allow-unsupported-compiler (system GCC $cur)."
    CUDA_ALLOW_UNSUPPORTED=1
    return 0
}

# Last resort only: a checksummed CUDA tarball already landed, or there is
# no published bundle. Never compile just because --version failed.
install_cuda_backend_from_source() {
    has_cuda_toolkit || return 1
    BACKEND_CHOICE="cuda"
    BACKEND_REPO="https://github.com/ikawrakow/ik_llama.cpp.git"
    BACKEND_DIR="$BACKEND_ROOT/ik_llama.cpp"
    BACKEND_BUILD="$BACKEND_DIR/build"
    BACKEND_CMAKE=(-DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON)
    say "Building ik_llama.cpp from source (CUDA toolkit is on this machine)..."
    build_backend || return 1
    local built
    built="$(backend_server_path)"
    [[ -x "$built" ]] || return 1
    built="$(_discover_abspath "$built")"
    ln -sfn "$built" "$INSTALL_DIR/ik_llama-server-cuda"
    ok "Installed ik_llama.cpp CUDA at $built"
}

# Standard auto install: ik_llama.cpp (CUDA prebuilt) and llama.cpp (Vulkan).
# llama.cpp is always installed. CUDA is compiled only when no CUDA ELF landed
# and nvcc is already on this machine. CPU is last if nothing runs.
install_auto_release_backends() {
    local got=0
    if ! host_can_run_ubuntu_prebuilt && [[ "$OS" == "Linux" ]]; then
        warn "Linux prebuilts are glibc x86_64. This host is $(uname -m) $(host_libc); using a local llama-server or a source build."
    fi
    if installed_real_server ik_llama-server-cuda || installed_real_server llama-server-vulkan \
        || installed_real_server llama-server || installed_real_server llama-server-cuda; then
        got=1
    fi

    if ! vulkan_available; then
        ensure_vulkan_runtime && ok "Vulkan runtime is installed" || true
    fi

    if has_nvidia_gpu && ! installed_real_server ik_llama-server-cuda; then
        BACKEND_CHOICE="cuda"
        say "Installing ik_llama.cpp (CUDA)..."
        if install_release_bundle ik_llama-server-cuda || installed_real_server ik_llama-server-cuda; then
            got=1
        else
            warn "No published ggrun CUDA bundle for this host."
        fi
    elif installed_real_server ik_llama-server-cuda; then
        say "Keeping existing ik_llama.cpp (CUDA)."
        got=1
    fi

    if ! installed_real_server llama-server-vulkan; then
        BACKEND_CHOICE="vulkan"
        say "Installing llama.cpp (Vulkan)..."
        if install_release_bundle llama-server-vulkan || install_upstream_linux_prebuilt vulkan llama-server-vulkan; then
            got=1
        else
            warn "No Vulkan llama.cpp prebuilt for this host."
        fi
    else
        say "Keeping existing llama.cpp Vulkan backend."
        got=1
    fi

    if ! backend_kind_runs ik_llama-server-cuda && ! backend_kind_runs llama-server-vulkan \
        && ! backend_kind_runs llama-server-cuda && ! backend_kind_runs llama-server; then
        BACKEND_CHOICE="cpu"
        say "Installing llama.cpp (CPU) so ggrun can start without a GPU runtime..."
        if install_release_bundle llama-server || install_upstream_linux_prebuilt cpu llama-server; then
            got=1
        fi
    fi

    # Compile only if the CUDA tarball never landed. Do this after llama.cpp
    # so a gcc/dnf stall cannot skip the Vulkan/CPU pair.
    if has_nvidia_gpu && ! installed_real_server ik_llama-server-cuda; then
        if ! has_cuda_toolkit; then
            ensure_cuda_toolkit || true
        fi
        if has_cuda_toolkit && install_cuda_backend_from_source; then
            got=1
        else
            warn "CUDA llama-server did not build. Common cause: system GCC is newer than this CUDA toolkit allows."
            warn "Install a matching gcc/g++ (for CUDA 12.3: gcc12 / g++-12) or a newer CUDA toolkit."
        fi
    fi

    if backend_kind_runs llama-server-vulkan; then
        BACKEND_CHOICE="vulkan"
    elif backend_kind_runs ik_llama-server-cuda; then
        BACKEND_CHOICE="cuda"
    elif backend_kind_runs llama-server; then
        BACKEND_CHOICE="cpu"
    fi
    link_default_llama_server
    if installed_real_server ik_llama-server-cuda || installed_real_server llama-server-vulkan || installed_real_server llama-server; then
        got=1
    fi
    (( got ))
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

python_in_venv() {
    python3 -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)'
}

ensure_python_pip() {
    python3 -m pip --version >/dev/null 2>&1 && return 0
    # CPython normally bundles ensurepip. Distribution builds may omit it, in
    # which case install_python_runtime above installs the distro's pip package.
    local args=(--user)
    python_in_venv && args=()
    python3 -m ensurepip "${args[@]}" >/dev/null 2>&1 \
        && python3 -m pip --version >/dev/null 2>&1
}

install_python_download_deps() {
    local args=(--quiet --upgrade huggingface_hub tqdm)
    if ! python_in_venv; then
        args=(--user "${args[@]}")
    fi
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
    local nvcc_path="" host_cc="" cmake_args=()
    ensure_build_deps || return 1
    if [[ "$OS" == "Linux" && "$BACKEND_CHOICE" == "cuda" ]] && ! has_cuda_toolkit; then
        err "CUDA toolkit/nvcc not found for CUDA backend."
        return 1
    fi
    refresh_backend_repo || return 1
    cmake_env=()
    cmake_args=("${BACKEND_CMAKE[@]}")
    if [[ "$BACKEND_CHOICE" == "cuda" ]]; then
        nvcc_path="$(cuda_nvcc_path 2>/dev/null || true)"
        [[ -n "$nvcc_path" ]] && cmake_env+=(CUDACXX="$nvcc_path")
        ensure_cuda_host_compiler || true
        if [[ -n "${CUDA_HOST_CXX:-}" ]]; then
            host_cc="${CUDA_HOST_CXX}"
            cmake_args+=(-DCMAKE_CUDA_HOST_COMPILER="$host_cc")
            cmake_args+=(-DCMAKE_CXX_COMPILER="$host_cc")
            if [[ "$host_cc" == *g++* ]]; then
                cmake_args+=(-DCMAKE_C_COMPILER="${host_cc//g++/gcc}")
            elif [[ "$host_cc" == *c++* ]]; then
                cmake_args+=(-DCMAKE_C_COMPILER="${host_cc//c++/cc}")
            fi
        fi
        if [[ "${CUDA_ALLOW_UNSUPPORTED:-0}" == "1" ]]; then
            cmake_env+=(NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:+$NVCC_PREPEND_FLAGS }--allow-unsupported-compiler")
            cmake_args+=(-DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler")
        fi
        # A previous failed configure leaves a cache that ignores the new host
        # compiler. Only wipe when the server is not already built.
        if [[ ! -x "$(backend_server_path)" && -d "$BACKEND_BUILD" ]]; then
            rm -rf "$BACKEND_BUILD"
        fi
    fi
    env "${cmake_env[@]}" cmake -S "$BACKEND_DIR" -B "$BACKEND_BUILD" -DCMAKE_BUILD_TYPE=Release "${cmake_args[@]}" \
        && cmake --build "$BACKEND_BUILD" --config Release -j"$(nproc 2>/dev/null || echo 4)" -t llama-server
}

# ── Stage 3: install scripts ────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR" "$MODEL_DIR"
RELEASE_INSTALLED=0
drop_fake_installed_backends

if [[ "$INSTALL_MODE" == "auto" || "$INSTALL_MODE" == "release" ]]; then
    if [[ "${LLM_INSTALL_SCAN_SYSTEM:-1}" != "0" ]]; then
        say ""
        scan_system_installs
        report_system_installs
        if adopt_system_backends; then
            RELEASE_INSTALLED=1
        fi
    fi
    if [[ "$BACKEND_REQUEST" == "auto" && "$INSTALL_MODE" == "auto" ]]; then
        if install_auto_release_backends; then
            RELEASE_INSTALLED=1
        else
            warn "No prebuilt CUDA/Vulkan/CPU backend could be downloaded."
        fi
    elif installed_real_server ik_llama-server-cuda || installed_real_server llama-server-vulkan \
        || installed_real_server llama-server || installed_real_server llama-server-cuda; then
        RELEASE_INSTALLED=1
    elif install_release_bundle; then
        RELEASE_INSTALLED=1
    elif [[ "$INSTALL_MODE" == "release" ]]; then
        err "No compatible release bundle found for $(platform_slug 2>/dev/null || echo unknown)-$BACKEND_CHOICE"
        [[ "$BACKEND_CHOICE" == "cuda" ]] && err "CUDA release mode requires a matching CUDA bundle for this platform."
        exit 1
    elif [[ "$BACKEND_CHOICE" == "cuda" ]]; then
        warn "No compatible CUDA release bundle found; falling back to ik_llama.cpp source build."
    else
        warn "No compatible release bundle found; falling back to local script install + source build."
    fi
fi

say ""
say "── Installing scripts to $INSTALL_DIR ──"

install_ggrun_from_source() {
    [[ "$MAIN_IMPL" == "go" && "$INSTALL_MODE" != "scripts" ]] || return 0
    # Explicit release mode must retain the launcher covered by its checksum.
    [[ "$INSTALL_MODE" == "release" && "$RELEASE_INSTALLED" == "1" ]] && return 0
    ensure_source_repo
    if [[ -f "$SRC_DIR/go/go.mod" ]]; then
        say "── Building ggrun from this checkout ──"
        go_build_tmp="$(mktemp -t ggrun-build.XXXXXX)"
        if build_go_binary "$go_build_tmp" && install_go_as_main "$go_build_tmp"; then
            rm -f "$go_build_tmp"
            ok "Installed ggrun from source (not an older release binary)"
            return 0
        fi
        rm -f "$go_build_tmp"
        if [[ -x "$SRC_DIR/go/ggrun" ]] && install_go_as_main "$SRC_DIR/go/ggrun"; then
            ok "Installed prebuilt ggrun from this checkout"
            return 0
        fi
        # A failed build must surface to the caller (a self-update rolls back /
        # reports a failed update), not silently "keep the release binary" and
        # claim success. On a fresh install the prebuilt fallback above usually
        # saves us; reaching this line with no runnable binary is a real failure.
        if [[ -x "$INSTALL_DIR/ggrun" || -x "$INSTALL_DIR/ggrun.exe" ]]; then
            warn "Could not build ggrun from this checkout; keeping the existing binary."
        else
            warn "Could not build ggrun from this checkout and no existing binary is installed."
            return 1
        fi
    fi
}

ensure_source_repo
FILES=("setup.sh" "setup-linux.sh" "setup-mac.sh")
for f in "${FILES[@]}"; do
    install_source_file "$f" "$f" 0755 || warn "$f not found in source; skipping"
done
install_source_file "tools/gguf/parse_gguf.py" "parse_gguf.py" 0755 || warn "parse_gguf.py not found in source; skipping"
install_source_file "tools/models/model_index.py" "model_index.py" 0755 || warn "model_index.py not found in source; skipping"
install_source_file "tools/download/download_any_gguf.py" "download_any_gguf.py" 0755 || warn "download_any_gguf.py not found in source; skipping"
install_source_file "tools/hardware/measure_bandwidth.py" "measure_bandwidth.py" 0755 || warn "measure_bandwidth.py not found in source; skipping"
install_legacy_bash_shim
install_ggrun_from_source

if [[ "$OS" == "Linux" && "$BACKEND_CHOICE" == "cuda" && "$MAIN_IMPL" == "go" && "$INSTALL_MODE" != "scripts" && ! -e "$INSTALL_DIR/ik_llama-server-cuda" ]]; then
    [[ -d "$SRC_DIR/native/memguard" ]] || { err "CUDA source install is missing native/memguard"; exit 1; }
    command -v make >/dev/null 2>&1 || { err "make is required to build the CUDA allocation firewall"; exit 1; }
    command -v cc >/dev/null 2>&1 || { err "a C compiler is required to build the CUDA allocation firewall"; exit 1; }
    make -C "$SRC_DIR/native/memguard" libggrun-memguard.so
    install -m 0644 "$SRC_DIR/native/memguard/libggrun-memguard.so" "$INSTALL_DIR/libggrun-memguard.so"
    ok "Installed GPU memory-safety guard (prevents out-of-memory crashes)"
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
    if installed_real_server ik_llama-server-cuda || installed_real_server llama-server-vulkan || installed_real_server llama-server; then
        link_default_llama_server
        RELEASE_INSTALLED=1
    fi
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
        say "Building from source can take 10-30+ minutes depending on your CPU — the compiler output below is normal, not a hang."
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
say "║ ggrun installer finished                                   ║"
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
