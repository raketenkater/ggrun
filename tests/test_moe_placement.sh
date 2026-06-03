#!/usr/bin/env bash
# Regression tests for MoE placement decisions that do not require real GPUs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GO_BIN="${LLM_SERVER_GO_BIN:-$ROOT/go/llm-server}"
if [[ ! -x "$GO_BIN" ]]; then
    (cd "$ROOT/go" && go build -o llm-server ./cmd/llm-server)
fi
TMP="$(mktemp -d -t llm-server-moe-placement.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/bin" "$TMP/models" "$TMP/cache"

cat >"$TMP/llama-server" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
    --help|-h) echo "fake llama-server --reasoning --n-cpu-moe --kv-unified"; exit 0 ;;
    --version) echo "fake 0.0.0"; exit 0 ;;
esac
exit 0
EOF
chmod +x "$TMP/llama-server"

cat >"$TMP/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
    *"--query-gpu=driver_version"*) echo "580.0"; exit 0 ;;
    *"--query-gpu=index,name,memory.total,memory.free,pcie.link.width.current,pcie.link.gen.current,compute_cap"*)
        echo "0, RTX 4090, 24576, 24576, 16, 4, 8.9"
        echo "1, RTX 4090, 24576, 24576, 16, 4, 8.9"
        echo "2, RTX 4090, 24576, 24576, 16, 4, 8.9"
        exit 0
        ;;
    *) exit 0 ;;
esac
EOF
chmod +x "$TMP/bin/nvidia-smi"

python3 "$ROOT/tests/build_synthetic_gguf.py" \
    --out "$TMP/models/Kimi-K2.6-IQ3_K.gguf" \
    --arch kimi-linear --name 'Kimi-K2.6-IQ3_K' \
    --layers 61 --hkv 8 --kl 128 --vl 128 --embd 4096 --ff 14336 \
    --experts 384 --exp-used 8 --exp-ff 2048 --ctx-train 262144 \
    --tokenizer-pre kimi-k2 \
    --tensor 'blk.0.ffn_down_exps.weight:290000000000:138' \
    --tensor 'blk.0.ffn_down_shexp.weight:7000000000:138'

# Make the apparent file size larger than the RAM budget without consuming disk.
# This exercises the large-MoE mmap path used by Kimi-class user reports.
truncate -s 512G "$TMP/models/Kimi-K2.6-IQ3_K.gguf"

out=$(PATH="$TMP/bin:$PATH" LLAMA_SERVER="$TMP/llama-server" \
    LLM_ASSUME_YES=1 LLM_SERVER_UPDATE_CHECKED=1 \
    LLM_CACHE_DIR="$TMP/cache" LLM_MODEL_DIR="$TMP/models" \
    "$GO_BIN" --dry-run --server-bin "$TMP/llama-server" --ram-budget 64000 \
    "$TMP/models/Kimi-K2.6-IQ3_K.gguf" 2>&1)

if [[ "$out" != *"--n-cpu-moe 384"* ]]; then
    echo "expected Kimi-style model to enter MoE offload path"
    echo "$out"
    exit 1
fi

cmd_section="$out"
if [[ "$cmd_section" == *"--no-mmap"* ]]; then
    echo "oversized mmap-safe MoE dry-run must not pass --no-mmap"
    echo "$out"
    exit 1
fi

if [[ "$cmd_section" != *"ffn_(("*"_exps"* ]]; then
    echo "expected routed expert tensors in -ot regex"
    echo "$out"
    exit 1
fi

if [[ "$cmd_section" != *"_shexp"* ]]; then
    echo "expected shared expert tensors in -ot regex"
    echo "$out"
    exit 1
fi

python3 "$ROOT/tests/build_synthetic_gguf.py" \
    --out "$TMP/models/KV-Heavy-MoE.gguf" \
    --arch minimax-m2 --name 'KV-Heavy-MoE' \
    --layers 20 --hkv 64 --kl 128 --vl 128 --embd 3072 --ff 1536 \
    --experts 64 --exp-used 8 --exp-ff 1536 --ctx-train 196608 \
    --tokenizer-pre minimax-m2 \
    --tensor 'blk.0.ffn_down_exps.weight:40000000000:138'
truncate -s 64G "$TMP/models/KV-Heavy-MoE.gguf"

out=$(PATH="$TMP/bin:$PATH" LLAMA_SERVER="$TMP/llama-server" \
    LLM_ASSUME_YES=1 LLM_SERVER_UPDATE_CHECKED=1 \
    LLM_CACHE_DIR="$TMP/cache-kv-gpu" LLM_MODEL_DIR="$TMP/models" \
    "$GO_BIN" --dry-run --server-bin "$TMP/llama-server" \
    --ctx-size 196608 --kv-quality mid "$TMP/models/KV-Heavy-MoE.gguf" 2>&1)

if [[ "$out" == *"--no-kv-offload"* ]]; then
    echo "auto KV placement should keep KV on GPU when possible"
    echo "$out"
    exit 1
fi

if [[ "$out" != *"--ctx-size 196608"* ]]; then
    echo "expected requested context size to be preserved"
    echo "$out"
    exit 1
fi

out_cpu=$(PATH="$TMP/bin:$PATH" LLAMA_SERVER="$TMP/llama-server" \
    LLM_ASSUME_YES=1 LLM_SERVER_UPDATE_CHECKED=1 \
    LLM_CACHE_DIR="$TMP/cache-kv-cpu" LLM_MODEL_DIR="$TMP/models" \
    "$GO_BIN" --dry-run --server-bin "$TMP/llama-server" \
    --ctx-size 196608 --kv-quality mid --kv-placement cpu \
    "$TMP/models/KV-Heavy-MoE.gguf" 2>&1)

if [[ "$out_cpu" != *"--no-kv-offload"* ]]; then
    echo "expected explicit CPU KV placement to pass --no-kv-offload"
    echo "$out_cpu"
    exit 1
fi

if [[ "$out_cpu" == *"GPU KV reserve first"* ]]; then
    echo "CPU KV placement must not reserve GPU KV"
    echo "$out_cpu"
    exit 1
fi

printf 'LLM_KV_PLACEMENT="cpu"\n' > "$TMP/kv-placement.conf"
out_settings=$(PATH="$TMP/bin:$PATH" LLAMA_SERVER="$TMP/llama-server" \
    LLM_CONFIG="$TMP/kv-placement.conf" LLM_ASSUME_YES=1 LLM_SERVER_UPDATE_CHECKED=1 \
    LLM_CACHE_DIR="$TMP/cache-kv-settings" LLM_MODEL_DIR="$TMP/models" \
    "$GO_BIN" --dry-run --server-bin "$TMP/llama-server" \
    --ctx-size 196608 --kv-quality mid "$TMP/models/KV-Heavy-MoE.gguf" 2>&1)

if [[ "$out_settings" != *"--no-kv-offload"* ]]; then
    echo "expected LLM_KV_PLACEMENT setting to apply CPU KV placement"
    echo "$out_settings"
    exit 1
fi

echo "MoE placement regression: shared experts, mmap path, and KV-first placement covered"
