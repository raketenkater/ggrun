#!/usr/bin/env bash
# place_isolated_backend must keep a landed binary even when --version fails.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d -t ggrun-place-backend.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

# install.sh runs on source; pull only the helpers this test needs.
eval "$(
    awk '
        /^say\(\)/ {print; next}
        /^ok\(\)/ {print; next}
        /^warn\(\)/ {print; next}
        /^host_libc\(\)|^host_can_run_ubuntu_prebuilt\(\)|^classify_probe_output\(\)/ {keep=1}
        /^probe_llama_server\(\)|^warn_probe_detail\(\)|^copy_resolved_lib\(\)/ {keep=1}
        /^harvest_cuda_runtime_libs\(\)|^copy_backend_libs\(\)|^backend_probe_kind\(\)/ {keep=1}
        /^box_has_libnccl\(\)|^fetch_nccl_into_box\(\)/ {keep=1}
        /^place_isolated_backend\(\)|^link_default_llama_server\(\)/ {keep=1}
        /^_discover_abspath\(\)/ {keep=1}
        keep {print}
        keep && /^}/ {keep=0}
    ' "$ROOT/install.sh"
)"

INSTALL_DIR="$TMP/bin"
mkdir -p "$INSTALL_DIR" "$TMP/src"
# Never query the host's GPUs; a case below flips this to an old GPU.
OLD_GPU=0
cuda_bundle_supported() { (( ! OLD_GPU )); }

cat >"$TMP/src/llama-server" <<'EOF'
#!/usr/bin/env bash
echo "missing vulkan icd" >&2
exit 1
EOF
chmod +x "$TMP/src/llama-server"
printf 'dummy' >"$TMP/src/libggml.so"

if ! place_isolated_backend "$TMP/src/llama-server" llama-server-vulkan; then
    echo "  ✗ place_isolated_backend returned failure after --version failed"
    exit 1
fi

test -x "$INSTALL_DIR/backends/llama-server-vulkan/llama-server"
test -e "$INSTALL_DIR/llama-server-vulkan"
test -f "$INSTALL_DIR/backends/llama-server-vulkan/libggml.so"

echo "  ✓ keep vulkan backend when --version fails"

# Versioned bundle libs need a SONAME symlink.
printf 'dummy' >"$TMP/src/libllama-common.so.0.0.1"
if ! place_isolated_backend "$TMP/src/llama-server" llama-server-cpu; then
    echo "  ✗ place failed after adding versioned library"
    exit 1
fi
test -L "$INSTALL_DIR/backends/llama-server-cpu/libllama-common.so.0"
test -f "$INSTALL_DIR/backends/llama-server-cpu/libllama-common.so.0.0.1"
echo "  ✓ create SONAME symlink for libllama-common.so.0"

# Missing project libraries are a failed place, not "GPU runtime missing".
cat >"$TMP/src/llama-server" <<'EOF'
#!/usr/bin/env bash
echo "/tmp/llama-server: error while loading shared libraries: libllama-common.so.0: cannot open shared object file: No such file or directory" >&2
exit 127
EOF
chmod +x "$TMP/src/llama-server"
rm -f "$TMP/src"/lib*.so*
if place_isolated_backend "$TMP/src/llama-server" llama-server-broken; then
    echo "  ✗ place succeeded for a binary missing libllama"
    exit 1
fi
test ! -e "$INSTALL_DIR/llama-server-broken"
echo "  ✓ reject a bundle that cannot load its own libllama"

# Checksummed CUDA ELF is kept even when --version is "broken".
if ! place_isolated_backend "$TMP/src/llama-server" ik_llama-server-cuda 1; then
    echo "  ✗ keep=1 returned failure for a CUDA download"
    exit 1
fi
test -x "$INSTALL_DIR/backends/ik_llama-server-cuda/llama-server"
test -e "$INSTALL_DIR/ik_llama-server-cuda"
echo "  ✓ keep a downloaded CUDA ELF when --version fails"
rm -f "$INSTALL_DIR/ik_llama-server-cuda"
rm -rf "$INSTALL_DIR/backends/ik_llama-server-cuda"

# Missing libnccl is fetched into the CUDA box, then --version can succeed.
mkdir -p "$TMP/ncclpack/lib"
printf 'nccl-stub\n' >"$TMP/ncclpack/lib/libnccl.so.2.21.5"
ln -sfn libnccl.so.2.21.5 "$TMP/ncclpack/lib/libnccl.so.2"
tar -C "$TMP" -cJf "$TMP/nccl.txz" ncclpack
export LLM_INSTALL_NCCL_URL="file://$TMP/nccl.txz"
cat >"$TMP/src/llama-server" <<'EOF'
#!/usr/bin/env bash
dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -e "$dir/libnccl.so.2" || -e "$dir/libnccl.so.2.21.5" ]]; then
    echo "version: cuda-ok"
    exit 0
fi
echo "error while loading shared libraries: libnccl.so.2: cannot open shared object file" >&2
exit 127
EOF
chmod +x "$TMP/src/llama-server"
if ! place_isolated_backend "$TMP/src/llama-server" ik_llama-server-cuda; then
    echo "  ✗ CUDA place failed after NCCL fetch"
    exit 1
fi
test -e "$INSTALL_DIR/backends/ik_llama-server-cuda/libnccl.so.2" \
    -o -e "$INSTALL_DIR/backends/ik_llama-server-cuda/libnccl.so.2.21.5" \
    || { echo "  ✗ libnccl was not copied into the CUDA box"; exit 1; }
[[ "$(probe_llama_server "$INSTALL_DIR/backends/ik_llama-server-cuda/llama-server" "$INSTALL_DIR/backends/ik_llama-server-cuda")" == "runs" ]] \
    || { echo "  ✗ CUDA backend still does not run after NCCL fetch"; exit 1; }
echo "  ✓ fetch libnccl.so.2 so the CUDA backend can start"
rm -f "$INSTALL_DIR/ik_llama-server-cuda"
rm -rf "$INSTALL_DIR/backends/ik_llama-server-cuda"

# classify is distro-agnostic: GPU runtime vs broken vs runs
[[ "$(classify_probe_output "version: 1234")" == "runs" ]]
[[ "$(classify_probe_output "error while loading shared libraries: libvulkan.so.1: cannot open shared object file")" == "needs_gpu" ]]
[[ "$(classify_probe_output "error while loading shared libraries: libcuda.so.1: cannot open shared object file")" == "needs_gpu" ]]
[[ "$(classify_probe_output "error while loading shared libraries: libnccl.so.2: cannot open shared object file")" == "needs_gpu" ]]
[[ "$(classify_probe_output "error while loading shared libraries: libcublas.so.12: cannot open shared object file")" == "needs_gpu" ]]
[[ "$(classify_probe_output "error while loading shared libraries: libstdc++.so.6: cannot open shared object file")" == "broken" ]]
[[ "$(classify_probe_output "version \`GLIBC_2.38' not found")" == "broken" ]]
[[ "$(classify_probe_output "cannot execute binary file: Exec format error")" == "broken" ]]
echo "  ✓ probe classes GPU-missing vs broken vs runs"

# Default llama-server must be one that runs, not a GPU binary that cannot start.
mkdir -p "$INSTALL_DIR/backends/llama-server-vulkan" "$INSTALL_DIR/backends/llama-server"
cat >"$INSTALL_DIR/backends/llama-server-vulkan/llama-server" <<'EOF'
#!/usr/bin/env bash
echo "libvulkan.so.1: cannot open shared object file" >&2
exit 127
EOF
cat >"$INSTALL_DIR/backends/llama-server/llama-server" <<'EOF'
#!/usr/bin/env bash
echo "version: cpu-ok"
exit 0
EOF
chmod +x "$INSTALL_DIR/backends/llama-server-vulkan/llama-server" \
         "$INSTALL_DIR/backends/llama-server/llama-server"
ln -sfn "backends/llama-server-vulkan/llama-server" "$INSTALL_DIR/llama-server-vulkan"
ln -sfn "backends/llama-server/llama-server" "$INSTALL_DIR/llama-server"
link_default_llama_server
# llama-server already runs; leave it (do not point at vulkan).
[[ "$(readlink "$INSTALL_DIR/llama-server")" == "backends/llama-server/llama-server" ]]
echo "  ✓ default backend stays on the one that runs"

# An old GPU (Pascal/Volta) starts the CUDA bundle for --version but has no
# kernels in it, so a running Vulkan backend becomes the default instead.
mkdir -p "$INSTALL_DIR/backends/ik_llama-server-cuda"
cp "$INSTALL_DIR/backends/llama-server/llama-server" "$INSTALL_DIR/backends/ik_llama-server-cuda/llama-server"
cp "$INSTALL_DIR/backends/llama-server/llama-server" "$INSTALL_DIR/backends/llama-server-vulkan/llama-server"
ln -sfn "backends/ik_llama-server-cuda/llama-server" "$INSTALL_DIR/ik_llama-server-cuda"
rm "$INSTALL_DIR/llama-server"
link_default_llama_server
[[ "$(readlink "$INSTALL_DIR/llama-server")" == "ik_llama-server-cuda" ]]
rm "$INSTALL_DIR/llama-server"
OLD_GPU=1 link_default_llama_server
[[ "$(readlink "$INSTALL_DIR/llama-server")" == "llama-server-vulkan" ]]
rm "$INSTALL_DIR/llama-server"
OLD_GPU=1 BACKEND_REQUEST=cuda link_default_llama_server
[[ "$(readlink "$INSTALL_DIR/llama-server")" == "ik_llama-server-cuda" ]]
# A CUDA build adopted from this machine was compiled for its GPU.
ln -sfn "$INSTALL_DIR/backends/ik_llama-server-cuda/llama-server" "$INSTALL_DIR/ik_llama-server-cuda"
rm "$INSTALL_DIR/llama-server"
OLD_GPU=1 link_default_llama_server
[[ "$(readlink "$INSTALL_DIR/llama-server")" == "ik_llama-server-cuda" ]]
echo "  ✓ default backend is Vulkan, not the CUDA bundle, on GPUs older than its kernels"
