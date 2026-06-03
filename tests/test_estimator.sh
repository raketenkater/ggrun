#!/usr/bin/env bash
# Regression tests for the llm-server estimator and dry-run output.
#
# Builds tiny synthetic GGUFs, points llm-server at a fake llama-server
# binary so it doesn't need a real backend, runs --dry-run --cpu, and asserts
# the output contains the architecture/layer/KV strings we rely on.
#
# Usage: bash tests/test_estimator.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GO_BIN="${LLM_SERVER_GO_BIN:-$ROOT/go/llm-server}"
if [[ ! -x "$GO_BIN" ]]; then
    (cd "$ROOT/go" && go build -o llm-server ./cmd/llm-server)
fi
TMP="$(mktemp -d -t llm-server-tests.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

# Stand-in llama-server: --help must exit 0 cleanly with no "shared libraries"
# error so the binary-validity check in llm-server passes. Anything else just
# noops — --dry-run never actually invokes the binary.
cat >"$TMP/llama-server" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
    --help|-h) echo "fake llama-server (test stub)"; exit 0 ;;
    --version) echo "fake 0.0.0"; exit 0 ;;
esac
exit 0
EOF
chmod +x "$TMP/llama-server"

export LLAMA_SERVER="$TMP/llama-server"
export LLM_ASSUME_YES=1
export LLM_MODEL_DIR="$TMP/models"
mkdir -p "$LLM_MODEL_DIR"

PASS=0
FAIL=0

assert_contains() {
    local out="$1" needle="$2" label="$3"
    if [[ "$out" == *"$needle"* ]]; then
        echo "  ✓ $label"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $label"
        echo "    expected output to contain: $needle"
        echo "    actual output (last 30 lines):"
        echo "$out" | tail -30 | sed 's/^/      /'
        FAIL=$((FAIL + 1))
    fi
}

run_dry() {
    "$GO_BIN" --dry-run --cpu "$@" 2>&1
}

build_gguf() {
    python3 "$ROOT/tests/build_synthetic_gguf.py" "$@"
}

# ── Test 1: dense Llama-class ────────────────────────────────────────────
echo "Test: dense_llama"
build_gguf --out "$TMP/dense.gguf" --arch llama --name 'Test-Llama-7B' \
    --layers 32 --hkv 8 --kl 128 --vl 128 --embd 4096 --ff 14336 --ctx-train 8192
out=$(run_dry "$TMP/dense.gguf")
assert_contains "$out" "$TMP/dense.gguf" "dense_llama: model path included"
assert_contains "$out" "--ctx-size 32768" "dense_llama: context selected from metadata"
assert_contains "$out" "--cache-type-k q4_0" "dense_llama: KV cache type emitted"

# ── Test 2: MoE ──────────────────────────────────────────────────────────
echo "Test: moe_qwen35"
build_gguf --out "$TMP/moe.gguf" --arch qwen35moe --name 'Test-MoE-35B-A3B' \
    --layers 40 --hkv 2 --kl 256 --vl 256 --embd 2048 \
    --experts 256 --exp-used 8 --exp-ff 512 --ctx-train 262144 \
    --full-interval 4
out=$(run_dry "$TMP/moe.gguf")
assert_contains "$out" "$TMP/moe.gguf" "moe_qwen35: model path included"
assert_contains "$out" "--ctx-size 262144" "moe_qwen35: training context preserved"

# ── Test 3: MLA / DeepSeek-class ─────────────────────────────────────────
echo "Test: mla_deepseek"
build_gguf --out "$TMP/mla.gguf" --arch deepseek2 --name 'Test-DeepSeek' \
    --layers 61 --hkv 128 --kl 192 --vl 128 --embd 7168 --ff 18432 \
    --kv-lora 512 --q-lora 1536 --ctx-train 163840
out=$(run_dry "$TMP/mla.gguf")
assert_contains "$out" "$TMP/mla.gguf" "mla_deepseek: model path included"
assert_contains "$out" "--ctx-size 131072" "mla_deepseek: auto context selected"

# ── Test 4: ISWA / Gemma-class ───────────────────────────────────────────
echo "Test: iswa_gemma"
build_gguf --out "$TMP/iswa.gguf" --arch gemma3 --name 'Test-Gemma' \
    --layers 42 --hkv 4 --kl 256 --vl 256 --embd 3840 --ff 15360 \
    --swa 4096 --ctx-train 131072
out=$(run_dry "$TMP/iswa.gguf")
assert_contains "$out" "$TMP/iswa.gguf" "iswa_gemma: model path included"
assert_contains "$out" "--ctx-size 131072" "iswa_gemma: auto context selected"

# ── Test 5: SSM hybrid ───────────────────────────────────────────────────
echo "Test: ssm_hybrid"
build_gguf --out "$TMP/ssm.gguf" --arch qwen35 --name 'Test-Qwen35' \
    --layers 64 --hkv 4 --kl 256 --vl 256 --embd 5120 --ff 17408 \
    --ctx-train 262144 --full-interval 4 --ssm
out=$(run_dry "$TMP/ssm.gguf")
assert_contains "$out" "--no-context-shift" "ssm_hybrid: context shift disabled"

# ── Test 6: mistagged DeepSeek V4 Flash (deepseek2 arch + kl_mla<=rope_dim) ─
# Stock converters tag DeepSeek V4 Flash GGUFs as deepseek2 but emit V4
# metadata that crashes stock builds. llm-server should warn (not bail) so
# users with a fork-built llama-server can still proceed.
echo "Test: dsv4_flash_mistag_warns_but_proceeds"
build_gguf --out "$TMP/dsv4_mistag.gguf" --arch deepseek2 --name 'DeepSeek V4 Flash' \
    --layers 43 --hkv 1 --kl 512 --vl 512 --embd 4096 \
    --kv-lora 512 --q-lora 512 --kl-mla 64 --vl-mla 512 --rope-dim 64 \
    --ctx-train 1048576
out=$(run_dry "$TMP/dsv4_mistag.gguf" 2>&1 || true)
assert_contains "$out" "DeepSeek V4 Flash mistagged" "dsv4_flash_mistag: clear warning"
assert_contains "$out" "antirez/llama.cpp-deepseek-v4-flash" "dsv4_flash_mistag: points to fork"
assert_contains "$out" "PR #22378" "dsv4_flash_mistag: points to upstream PR"
# Warning must not abort the run; downstream command generation should still appear.
assert_contains "$out" "--ctx-size 1048576" "dsv4_flash_mistag: dry-run command still emitted"

# ── Test 7: max-context-fit suggestion stays out of non-interactive runs ─
echo "Test: max_ctx_suggestion_skipped_under_assume_yes"
out=$(run_dry "$TMP/dense.gguf")
if [[ "$out" == *"Use max context"* ]]; then
    echo "  ✗ max_ctx prompt leaked into LLM_ASSUME_YES=1 run"
    ((FAIL++))
else
    echo "  ✓ max_ctx prompt suppressed under LLM_ASSUME_YES"
    ((PASS++))
fi

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "Estimator regression: $PASS passed, $FAIL failed"
exit $(( FAIL > 0 ? 1 : 0 ))
