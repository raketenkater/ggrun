#!/usr/bin/env bash
# Build a self-contained release archive for install.sh.
#
# Usage:
#   scripts/package-release.sh <asset-name> <llama-server-path> <output-dir>

set -euo pipefail

ASSET_NAME="${1:-}"
SERVER_BIN="${2:-}"
OUT_DIR="${3:-dist}"

if [[ -z "$ASSET_NAME" || -z "$SERVER_BIN" ]]; then
    echo "Usage: $0 <asset-name> <llama-server-path> <output-dir>" >&2
    exit 2
fi
if [[ ! -x "$SERVER_BIN" ]]; then
    echo "Error: llama-server binary not executable: $SERVER_BIN" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
WORK_DIR="$(mktemp -d -t ggrun-package.XXXXXX)"
PAYLOAD="$WORK_DIR/${ASSET_NAME%.tar.gz}"

cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

mkdir -p "$PAYLOAD/bin"

MEMGUARD_LIB=""
if [[ "$(uname -s)" == "Linux" && -d "$ROOT_DIR/native/memguard" ]]; then
    command -v cc >/dev/null 2>&1 || { echo "Error: 'cc' is required to build the Linux allocation firewall" >&2; exit 1; }
    make -C "$ROOT_DIR/native/memguard" libggrun-memguard.so
    MEMGUARD_LIB="$ROOT_DIR/native/memguard/libggrun-memguard.so"
fi

for f in LICENSE README.md CHANGELOG.md; do
    [[ -f "$ROOT_DIR/$f" ]] && install -m 0644 "$ROOT_DIR/$f" "$PAYLOAD/$f"
done
for f in setup.sh setup-linux.sh setup-mac.sh; do
    [[ -f "$ROOT_DIR/$f" ]] && install -m 0755 "$ROOT_DIR/$f" "$PAYLOAD/$f"
done
[[ -f "$ROOT_DIR/install.ps1" ]] && install -m 0644 "$ROOT_DIR/install.ps1" "$PAYLOAD/install.ps1"

install -m 0755 "$SERVER_BIN" "$PAYLOAD/bin/llama-server"

if [[ -x "$ROOT_DIR/go/ggrun" ]]; then
    install -m 0755 "$ROOT_DIR/go/ggrun" "$PAYLOAD/bin/ggrun"
fi
if [[ -n "$MEMGUARD_LIB" ]]; then
    install -m 0644 "$MEMGUARD_LIB" "$PAYLOAD/bin/libggrun-memguard.so"
fi
if [[ -f "$ROOT_DIR/legacy/bash/ggrun" ]]; then
    install -m 0755 "$ROOT_DIR/legacy/bash/ggrun" "$PAYLOAD/llm-server-bash"
fi

for spec in \
    "tools/gguf/parse_gguf.py:parse_gguf.py" \
    "tools/models/model_index.py:model_index.py" \
    "tools/download/download_any_gguf.py:download_any_gguf.py" \
    "tools/hardware/measure_bandwidth.py:measure_bandwidth.py"; do
    src="${spec%%:*}"
    dst="${spec##*:}"
    [[ -f "$ROOT_DIR/$src" ]] && install -m 0755 "$ROOT_DIR/$src" "$PAYLOAD/bin/$dst"
done

BIN_DIR="$(cd "$(dirname "$SERVER_BIN")" && pwd)"
while IFS= read -r lib; do
    install -m 0644 "$lib" "$PAYLOAD/bin/$(basename "$lib")"
done < <(find "$BIN_DIR" -maxdepth 1 -type f \( -name 'lib*.so*' -o -name 'lib*.dylib' -o -name '*.dll' \) 2>/dev/null | sort)

# Shared-library IK builds keep runtime libraries outside build/bin. Copy the
# project libraries referenced by the server under the names requested by its
# dynamic dependencies so the archive remains relocatable.
if command -v ldd >/dev/null 2>&1; then
    while IFS='|' read -r soname lib; do
        [[ -n "$soname" && -f "$lib" ]] || continue
        install -m 0644 "$lib" "$PAYLOAD/bin/$soname"
    done < <(
        ldd "$SERVER_BIN" 2>/dev/null |
            awk '$1 ~ /^lib(ggml|llama|mtmd)/ && $2 == "=>" && $3 ~ "^/" { print $1 "|" $3 }'
    )
fi

# CUDA runtime (not libcuda.so.1 — that is the driver) so a laptop can load
# the bundle without nvcc. Next release ships these; current hosts harvest
# the same names from a toolkit already on the machine.
if [[ "$ASSET_NAME" == *cuda* ]]; then
    copy_cuda_rt() {
        local lib="$1" dest="$PAYLOAD/bin/$(basename "$1")" real
        [[ -e "$dest" || -L "$dest" ]] && return 0
        if [[ -L "$lib" ]]; then
            real="$(readlink -f "$lib" 2>/dev/null || true)"
            if [[ -n "$real" && -f "$real" ]]; then
                install -m 0644 "$real" "$PAYLOAD/bin/$(basename "$real")"
                [[ "$(basename "$real")" == "$(basename "$lib")" ]] \
                    || ln -sfn "$(basename "$real")" "$dest"
                return 0
            fi
        fi
        install -m 0644 "$lib" "$dest"
    }
    for dir in \
        "${CUDA_HOME:-}/lib64" "${CUDA_PATH:-}/lib64" \
        /usr/local/cuda/lib64 /usr/local/cuda/lib \
        /usr/local/cuda-12.8/lib64 /usr/lib/x86_64-linux-gnu /usr/lib64
    do
        [[ -d "$dir" ]] || continue
        for base in libcudart libcublas libcublasLt libnccl; do
            while IFS= read -r lib; do
                copy_cuda_rt "$lib"
            done < <(find "$dir" -maxdepth 1 \( -type f -o -type l \) -name "${base}.so*" 2>/dev/null | sort)
        done
    done
    if command -v ldd >/dev/null 2>&1; then
        for so in "$PAYLOAD/bin/llama-server" "$PAYLOAD/bin"/libggml.so*; do
            [[ -e "$so" ]] || continue
            while IFS= read -r line; do
                soname="${line%% *}"
                case "$soname" in
                    libcudart.so*|libcublas.so*|libcublasLt.so*|libnccl.so*) ;;
                    *) continue ;;
                esac
                lib="$(printf '%s\n' "$line" | awk '$2 == "=>" && $3 ~ /^\// { print $3 }')"
                [[ -n "$lib" && -e "$lib" ]] || continue
                copy_cuda_rt "$lib"
            done < <(ldd "$so" 2>/dev/null || true)
        done
    fi
fi

# Versioned files (libfoo.so.0.0.1) also need the SONAME the binary loads.
for f in "$PAYLOAD/bin"/lib*.so.*; do
    [[ -e "$f" ]] || continue
    base="$(basename "$f")"
    soname="$(printf '%s\n' "$base" | sed -E 's/(\.so\.[0-9]+)\.[0-9].*/\1/')"
    [[ -n "$soname" && "$soname" != "$base" && ! -e "$PAYLOAD/bin/$soname" ]] || continue
    ln -sfn "$base" "$PAYLOAD/bin/$soname"
done

# Make the bundle actually relocatable.
#
# The copying above puts every library next to the binary, but that is not
# enough on ELF: the linker records where to look at BUILD time. The v3.2.8
# Linux bundles shipped with
#
#     Library runpath: [/tmp/llama.cpp/build/bin:]
#
# baked into bin/llama-server -- a directory that existed only on the build
# machine and has no $ORIGIN entry. The binary therefore cannot find its own
# libraries anywhere else, even sitting in the same directory as all of them.
# That is issue #28: libllama-server-impl.so IS in the tarball, and the loader
# still cannot see it.
#
# Rewrite RUNPATH to $ORIGIN so the loader looks beside the binary. Do this for
# the bundled libraries too: they load each other.
#
# Rewrite only what is actually mis-pointed: an ELF carrying a non-empty
# RUNPATH with no $ORIGIN in it. That is precisely the #28 condition.
#
# A file with no RUNPATH at all is a different animal and must be left alone.
# patchelf rewrites program headers, and a Go binary does not survive that: the
# fake backend used by the install smoke jobs packaged without complaint and
# then died on --version with "Segmentation fault (core dumped)". Nothing was
# wrong with it. We broke it by relocating a path it never had.
# Empty for anything that is not an ELF with a RUNPATH -- including a shell
# script, which readelf rejects. Under `set -o pipefail` that rejection would
# otherwise abort the whole packaging run.
elf_runpath() {
    readelf -d "$1" 2>/dev/null |
        awk '/R(UN)?PATH/ { sub(/.*\[/,""); sub(/\].*/,""); print; exit }' || true
}

patch_targets=()
if [[ "$ASSET_NAME" != *windows* && "$ASSET_NAME" != *darwin* && "$ASSET_NAME" != *macos* ]]; then
    if ! command -v readelf >/dev/null 2>&1; then
        # Without readelf we cannot tell a mis-pointed binary from a healthy
        # one, and both guesses ship a broken artifact.
        echo "Error: readelf is required to inspect a Linux bundle's RUNPATH." >&2
        echo "       Install binutils and re-run." >&2
        exit 1
    fi
    for elf in "$PAYLOAD/bin/llama-server" "$PAYLOAD/bin"/lib*.so*; do
        [[ -f "$elf" ]] || continue          # skip the SONAME symlinks
        runpath="$(elf_runpath "$elf")"
        [[ -n "$runpath" ]] || continue      # nothing baked in, nothing to relocate
        case "$runpath" in *'$ORIGIN'*) continue ;; esac
        patch_targets+=("$elf")
    done
fi

if [[ ${#patch_targets[@]} -gt 0 ]]; then
    if ! command -v patchelf >/dev/null 2>&1; then
        # Failing here is deliberate. A bundle whose RUNPATH points at the build
        # host is broken for every user, and it is invisible in a file listing --
        # which is exactly why it shipped through four releases.
        echo "Error: patchelf is required to produce a relocatable Linux bundle." >&2
        echo "       Install it (apt-get install patchelf) and re-run." >&2
        exit 1
    fi
    # Did it run before we touched it? Then it must still run afterwards.
    ran_before=0
    if timeout 30 "$PAYLOAD/bin/llama-server" --version >/dev/null 2>&1; then
        ran_before=1
    fi
    for elf in "${patch_targets[@]}"; do
        patchelf --set-rpath '$ORIGIN' "$elf" 2>/dev/null || true
    done
    if [[ "$ran_before" -eq 1 ]] &&
       ! timeout 30 "$PAYLOAD/bin/llama-server" --version >/dev/null 2>&1; then
        echo "Error: bin/llama-server ran before the RUNPATH rewrite and does not run after it." >&2
        echo "       Packaging corrupted the binary; refusing to ship it." >&2
        exit 1
    fi
    # Prove it. A silent patchelf failure would ship the same bug again.
    bad=0
    for elf in "${patch_targets[@]}"; do
        runpath="$(elf_runpath "$elf")"
        case "$runpath" in
            *'$ORIGIN'*) ;;
            *) echo "Error: $(basename "$elf") RUNPATH is not \$ORIGIN: $runpath" >&2; bad=1 ;;
        esac
    done
    [[ "$bad" -eq 0 ]] || exit 1
fi

# macOS has the identical bug in Mach-O clothing, and it went unnoticed for the
# same reason: the smoke test ran the binary inside its build tree. Extracted
# anywhere else the v3.2.9 candidate died with
#
#     dyld: Library not loaded: @rpath/libllama-server-impl.dylib
#
# The dylib is in the tarball, right beside the binary, and dyld still cannot
# see it because nothing in LC_RPATH resolves @rpath to "next to me".
#
# Add @loader_path -- and only to a Mach-O that actually references @rpath, on
# the same principle as the ELF rule above: a binary with no @rpath dependency
# has nothing to relocate, and editing it is pure risk. Then re-sign, because
# editing a Mach-O invalidates its signature and arm64 refuses to execute one
# that fails validation.
if [[ "$ASSET_NAME" == *macos* || "$ASSET_NAME" == *darwin* ]]; then
    # If the bundle ships dylibs, the tools are mandatory. Without otool we
    # cannot tell a mis-pointed binary from a healthy one, and skipping quietly
    # is how the broken bundle shipped in the first place.
    bundled_dylibs=()
    for lib in "$PAYLOAD/bin"/*.dylib; do
        [[ -f "$lib" ]] && bundled_dylibs+=("$lib")
    done
    if [[ ${#bundled_dylibs[@]} -gt 0 ]]; then
        for tool in otool install_name_tool codesign; do
            if ! command -v "$tool" >/dev/null 2>&1; then
                echo "Error: $tool is required to produce a relocatable macOS bundle." >&2
                echo "       Build macOS assets on macOS." >&2
                exit 1
            fi
        done
    fi

    macho_targets=()
    for macho in "$PAYLOAD/bin/llama-server" "${bundled_dylibs[@]+"${bundled_dylibs[@]}"}"; do
        [[ -f "$macho" ]] || continue
        otool -L "$macho" 2>/dev/null | grep -q '@rpath/' || continue
        macho_targets+=("$macho")
    done

    if [[ ${#macho_targets[@]} -gt 0 ]]; then
        ran_before=0
        if timeout 30 "$PAYLOAD/bin/llama-server" --version >/dev/null 2>&1; then
            ran_before=1
        fi
        for macho in "${macho_targets[@]}"; do
            if ! otool -l "$macho" 2>/dev/null | grep -q '@loader_path'; then
                install_name_tool -add_rpath @loader_path "$macho" 2>/dev/null || true
            fi
            codesign --force --sign - "$macho" >/dev/null 2>&1 || true
        done
        if [[ "$ran_before" -eq 1 ]] &&
           ! timeout 30 "$PAYLOAD/bin/llama-server" --version >/dev/null 2>&1; then
            echo "Error: bin/llama-server ran before the rpath rewrite and does not run after it." >&2
            echo "       Packaging corrupted the binary; refusing to ship it." >&2
            exit 1
        fi
        # Prove it.
        bad=0
        for macho in "${macho_targets[@]}"; do
            if ! otool -l "$macho" 2>/dev/null | grep -q '@loader_path'; then
                echo "Error: $(basename "$macho") has no @loader_path in LC_RPATH" >&2
                bad=1
            fi
        done
        [[ "$bad" -eq 0 ]] || exit 1
    fi
fi

(
    cd "$WORK_DIR"

    tar -czf "$OUT_DIR/$ASSET_NAME" "$(basename "$PAYLOAD")"
)

echo "$OUT_DIR/$ASSET_NAME"
