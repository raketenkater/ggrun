#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
log="$(mktemp)"
trap 'rm -f "$log"' EXIT

GGRUN_MEMGUARD_LOG="$log" \
GGRUN_MEMGUARD_GPU_LIMITS_MB=1 \
GGRUN_MEMGUARD_PINNED_LIMIT_MB=0 \
LD_PRELOAD="$here/libggrun-memguard.so" \
    "$here/test-target"

grep -q '"event":"loaded"' "$log"
grep -q '"api":"cudaMalloc".*"result":2' "$log"
grep -q '"api":"cudaHostAlloc".*"result":2' "$log"
grep -q '"phase":"free".*"api":"cudaFree"' "$log"
grep -q '"api":"cudaMallocAsync".*"result":0' "$log"
echo "memguard event checks passed"
