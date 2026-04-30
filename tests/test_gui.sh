#!/usr/bin/env bash
# Regression tests for llm-server-gui non-interactive entrypoints.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d -t llm-server-gui-tests.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/home" "$TMP/empty-model-dir"

cat >"$TMP/llama-server" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
    --help|-h) echo "fake llama-server (gui test stub)"; exit 0 ;;
    --version) echo "fake 0.0.0"; exit 0 ;;
esac
exit 0
EOF
chmod +x "$TMP/llama-server"

python3 "$ROOT/tests/build_synthetic_gguf.py" --out "$TMP/model.gguf" \
    --arch llama --name Gui-Direct-Smoke --layers 2 --hkv 1 --kl 16 --vl 16 \
    --embd 64 --ff 128 --ctx-train 2048

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
        echo "$out" | tail -30 | sed 's/^/      /'
        FAIL=$((FAIL + 1))
    fi
}

assert_not_contains() {
    local out="$1" needle="$2" label="$3"
    if [[ "$out" != *"$needle"* ]]; then
        echo "  ✓ $label"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $label"
        echo "    expected output not to contain: $needle"
        echo "$out" | tail -30 | sed 's/^/      /'
        FAIL=$((FAIL + 1))
    fi
}

echo "Test: direct --model bypasses first-run TUI"
out=$(HOME="$TMP/home" LLM_ASSUME_YES=1 LLM_MODEL_DIR="$TMP/empty-model-dir" \
    "$ROOT/llm-server-gui" --model "$TMP/model.gguf" --dry-run --cpu \
    --server-bin "$TMP/llama-server" 2>&1)
assert_contains "$out" "Running on CPU only" "direct model launches through llm-server"
assert_not_contains "$out" "First Run Setup" "direct model does not block on empty model dir"

if (( FAIL > 0 )); then
    echo ""
    echo "GUI regression: $PASS passed, $FAIL failed"
    exit 1
fi

echo ""
echo "GUI regression: $PASS passed, 0 failed"
