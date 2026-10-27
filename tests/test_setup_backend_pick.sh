#!/usr/bin/env bash
# setup-home must point LLAMA_SERVER at a backend that starts, not a
# CUDA path that cannot load.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d -t ggrun-setup-backend.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

APP_BIN="$TMP/bin"
APP_SRC="$TMP/src"
mkdir -p "$APP_BIN" "$APP_SRC"

eval "$(
    awk '
        /^usable_llama_server\(\)|^backend_starts\(\)|^backend_candidates\(\)|^cuda_bundle_on_older_gpu\(\)/ {keep=1}
        keep {print}
        keep && /^}/ {keep=0}
    ' "$ROOT/scripts/setup-home.sh"
)"

# Never query the host's GPUs: a stub nvidia-smi reports FAKE_CC.
mkdir -p "$TMP/fakebin"
cat >"$TMP/fakebin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' ${FAKE_CC:-8.6}
EOF
chmod +x "$TMP/fakebin/nvidia-smi"
export PATH="$TMP/fakebin:$PATH"

TRUE_BIN=""
for cand in /usr/bin/true /bin/true; do
    [[ -x "$cand" ]] || continue
    hdr="$(head -c 4 "$cand" 2>/dev/null || true)"
    [[ "$hdr" == $'\x7fELF' ]] || continue
    TRUE_BIN="$cand"
    break
done
[[ -n "$TRUE_BIN" ]] || { echo "  ✗ need ELF true"; exit 1; }

# CUDA stub cannot start. Vulkan is a native ELF that exits 0, laid out
# like the installer (symlink to backends/<name>/llama-server).
cat >"$APP_BIN/ik_llama-server-cuda" <<'EOF'
#!/usr/bin/env bash
echo "error while loading shared libraries: libnccl.so.2: cannot open shared object file" >&2
exit 127
EOF
chmod +x "$APP_BIN/ik_llama-server-cuda"
mkdir -p "$APP_BIN/backends/llama-server-vulkan"
cp "$TRUE_BIN" "$APP_BIN/backends/llama-server-vulkan/llama-server"
ln -sfn "backends/llama-server-vulkan/llama-server" "$APP_BIN/llama-server-vulkan"

got=""
while IFS= read -r cand; do
    if backend_starts "$cand"; then
        got="$cand"
        break
    fi
done < <(backend_candidates)
if [[ "$got" != "$APP_BIN/llama-server-vulkan" ]]; then
    echo "  ✗ expected vulkan, got ${got:-empty}"
    exit 1
fi
echo "  ✓ setup picks a starting Vulkan backend over a non-starting CUDA stub"

# A CUDA bundle that starts is the default on Turing+, but a Pascal/Volta GPU
# has no kernels in it, so Vulkan comes first there.
rm "$APP_BIN/ik_llama-server-cuda"
mkdir -p "$APP_BIN/backends/ik_llama-server-cuda"
cp "$TRUE_BIN" "$APP_BIN/backends/ik_llama-server-cuda/llama-server"
ln -sfn "backends/ik_llama-server-cuda/llama-server" "$APP_BIN/ik_llama-server-cuda"
pick() {
    local cand
    while IFS= read -r cand; do
        if backend_starts "$cand"; then
            printf '%s\n' "$cand"
            return
        fi
    done < <(backend_candidates)
}
for case in "8.6:$APP_BIN/ik_llama-server-cuda" "8.6 6.1:$APP_BIN/llama-server-vulkan" "7.0:$APP_BIN/llama-server-vulkan"; do
    got="$(FAKE_CC="${case%%:*}" pick)"
    if [[ "$got" != "${case#*:}" ]]; then
        echo "  ✗ compute capability ${case%%:*}: expected ${case#*:}, got ${got:-empty}"
        exit 1
    fi
done
got="$(FAKE_CC=6.1 BACKEND=cuda pick)"
[[ "$got" == "$APP_BIN/ik_llama-server-cuda" ]] || { echo "  ✗ explicit cuda on 6.1: got ${got:-empty}"; exit 1; }
# A CUDA build adopted from this machine was compiled for its GPU.
ln -sfn "$APP_BIN/backends/ik_llama-server-cuda/llama-server" "$APP_BIN/ik_llama-server-cuda"
got="$(FAKE_CC=6.1 pick)"
[[ "$got" == "$APP_BIN/ik_llama-server-cuda" ]] || { echo "  ✗ adopted CUDA build on 6.1: got ${got:-empty}"; exit 1; }
echo "  ✓ setup prefers Vulkan over the CUDA bundle on GPUs older than its kernels (compute capability < 7.5)"
