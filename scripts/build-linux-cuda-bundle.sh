#!/usr/bin/env bash
# Build a prebuilt Linux x86_64 CUDA release bundle (ik_llama.cpp) so that
# `install.sh` serves a CUDA GPU backend with NO toolchain on users' machines.
#
# Upstream llama.cpp publishes CUDA prebuilts for Windows only, and ik_llama.cpp
# (ggrun's fast CUDA backend) ships no binaries at all — so without this bundle a
# Linux NVIDIA user either needs a full CUDA toolkit to compile, or falls back to
# the slower prebuilt Vulkan backend. The release workflow runs this script in a
# CUDA development container. It can also be run manually when a new pinned
# backend revision needs testing.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IK_REPO="${IK_LLAMA_REPO:-https://github.com/ikawrakow/ik_llama.cpp.git}"
IK_REF="${IK_LLAMA_REF:-1fddd12ba861c4815a8633f14d9c5670692099cc}"
IK_DIR="${IK_LLAMA_DIR:-$ROOT_DIR/.ik_llama.cpp}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist}"
ASSET="ggrun-linux-x86_64-cuda.tar.gz"

for tool in nvcc cmake go git; do
    command -v "$tool" >/dev/null 2>&1 || { echo "Error: '$tool' not found on PATH" >&2; exit 1; }
done
# ik_llama's ggml target asks for CUDA20. Ubuntu 22.04's CMake 3.22 cannot.
cmake_ver="$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)"
cmake_maj="${cmake_ver%%.*}"
cmake_min="${cmake_ver#*.}"
if [ "${cmake_maj:-0}" -lt 3 ] || { [ "${cmake_maj}" -eq 3 ] && [ "${cmake_min%%.*}" -lt 25 ]; }; then
    echo "Error: CMake >= 3.25 is required (have ${cmake_ver:-unknown})" >&2
    exit 1
fi

# 1. Go launcher — package-release.sh bundles $ROOT_DIR/go/ggrun when present.
echo "==> Building ggrun (Go launcher)"
version="${GITHUB_REF_NAME:-}"
ldflags="-s -w"
case "$version" in
    v*) ldflags="$ldflags -X github.com/raketenkater/ggrun/pkg/update.currentVersion=$version" ;;
    *) ;;
esac
( cd "$ROOT_DIR/go" && go build -buildvcs=false -trimpath -ldflags="$ldflags" -o ggrun ./cmd/ggrun )

# 2. ik_llama.cpp CUDA llama-server.
if [[ -d "$IK_DIR/.git" ]]; then
    echo "==> Updating ik_llama.cpp ($IK_REF)"
    git -C "$IK_DIR" fetch --depth 1 origin "$IK_REF"
    # --force drops patches applied by an earlier build of this checkout.
    git -C "$IK_DIR" checkout --force --detach FETCH_HEAD
else
    echo "==> Cloning ik_llama.cpp ($IK_REF)"
    git init "$IK_DIR"
    git -C "$IK_DIR" remote add origin "$IK_REPO"
    git -C "$IK_DIR" fetch --depth 1 origin "$IK_REF"
    git -C "$IK_DIR" checkout --detach FETCH_HEAD
fi
# Upstream fixes newer than the pinned revision, applied in order. A patch that
# no longer applies fails here, not after the compile.
for patch in "$ROOT_DIR"/patches/ik_llama/*.patch; do
    [[ -e "$patch" ]] || continue
    echo "==> Applying $(basename "$patch")"
    git -C "$IK_DIR" apply --check "$patch"
    git -C "$IK_DIR" apply "$patch"
done
echo "==> Configuring + building llama-server (CUDA)"
# CI has nvcc, not the driver. -L stubs is not enough: ld then looks up
# libcuda.so.1 as a DT_NEEDED of libggml.so and still misses it unless the
# stub is on a default search path or rpath-link is set.
stub=""
for d in /usr/local/cuda/lib64/stubs \
         /usr/local/cuda/targets/x86_64-linux/lib/stubs \
         /usr/local/cuda-12.8/lib64/stubs; do
    if [[ -e "$d/libcuda.so" ]]; then
        stub="$d/libcuda.so"
        break
    fi
done
[[ -n "$stub" ]] || { echo "Error: CUDA driver stub libcuda.so not found" >&2; exit 1; }
for dest in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/local/lib; do
    [[ -d "$dest" ]] || continue
    ln -sfn "$stub" "$dest/libcuda.so"
    ln -sfn "$stub" "$dest/libcuda.so.1"
    echo "==> Installed driver stub at $dest/libcuda.so.1 -> $stub"
    break
done
export LIBRARY_PATH="$(dirname "$stub")${LIBRARY_PATH:+:$LIBRARY_PATH}"
# Fail in seconds, not after a two-hour compile, if the stub is unusable.
echo 'int main(void){return 0;}' | cc -xc - -o /tmp/cuda-stub-probe -lcuda \
    || { echo "Error: cc -lcuda cannot link the driver stub" >&2; exit 1; }
rm -f /tmp/cuda-stub-probe
stub_dir="$(dirname "$stub")"
# Default FA kernels only. ALL_QUANTS made CI take ~2h per attempt.
# NCCL is multi-GPU only and is not part of the driver or a typical toolkit.
# A laptop then fails to load libggml (DT_NEEDED libnccl.so.2). Leave it off
# so the published bundle starts with nvidia-smi + bundled cudart/cublas.
cmake -S "$IK_DIR" -B "$IK_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON \
    -DGGML_CUDA_NCCL=OFF \
    "-DCMAKE_CUDA_ARCHITECTURES=75;80;86;89" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${stub_dir} -Wl,-rpath-link,${stub_dir}" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${stub_dir} -Wl,-rpath-link,${stub_dir}"
cmake --build "$IK_DIR/build" --config Release -j"$(nproc 2>/dev/null || echo 4)" -t llama-server

SERVER="$IK_DIR/build/bin/llama-server"
[[ -x "$SERVER" ]] || { echo "Error: build did not produce $SERVER" >&2; exit 1; }

# 3. Package the relocatable bundle and record its checksum.
echo "==> Packaging $ASSET"
mkdir -p "$OUT_DIR"
"$ROOT_DIR/scripts/package-release.sh" "$ASSET" "$SERVER" "$OUT_DIR"
( cd "$OUT_DIR" && sha256sum "$ASSET" >> SHA256SUMS )

echo
echo "Built $OUT_DIR/$ASSET"
