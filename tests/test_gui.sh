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
    --embd 64 --ff 128 --ctx-train 131072

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

echo "Test: interactive flow honors saved backend default"
mkdir -p "$TMP/home/ik_llama.cpp/build/bin" "$TMP/home/llama.cpp/build/bin" "$TMP/home/.config/llm-server" "$TMP/models"
cat >"$TMP/home/ik_llama.cpp/build/bin/llama-server" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
    --help|-h) echo "fake ik llama-server"; exit 0 ;;
    --version) echo "fake ik 0.0.0"; exit 0 ;;
esac
exit 0
EOF
cat >"$TMP/home/llama.cpp/build/bin/llama-server" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
    --help|-h) echo "fake llama-server"; exit 0 ;;
    --version) echo "fake 0.0.0"; exit 0 ;;
esac
exit 0
EOF
chmod +x "$TMP/home/ik_llama.cpp/build/bin/llama-server" "$TMP/home/llama.cpp/build/bin/llama-server"
printf 'LLM_BACKEND="ik_llama"\n' >"$TMP/home/.config/llm-server/config.sh"
cp "$TMP/model.gguf" "$TMP/models/model.gguf"

# Main menu: pick model 1 → quick boot screen: press D for dry-run → Enter
# to return → < to go back to main → q to quit.
out=$(printf '1\nD\n\n<\nq\n' | HOME="$TMP/home" LLM_ASSUME_YES=1 \
    LLM_MODEL_DIR="$TMP/models" LLM_SERVER_REPO="$TMP/no-repo" "$ROOT/llm-server-gui" 2>&1)
assert_contains "$out" "Backend:  ik_llama" "main menu shows saved backend"
assert_contains "$out" "Boot: model.gguf" "model number opens quick boot"
assert_contains "$out" "fit (launcher calculates)" "quick boot keeps fit as a mode"
assert_contains "$out" "--ctx-size 131072" "quick boot fit expands past standard context"
assert_contains "$out" "Binary: $TMP/home/ik_llama.cpp/build/bin/llama-server" "dry run uses saved backend binary"

echo "Test: interactive flow preserves explicit context passthrough"
out=$(printf '1\nD\n\n<\nq\n' | HOME="$TMP/home" LLM_ASSUME_YES=1 \
    LLM_MODEL_DIR="$TMP/models" LLM_SERVER_REPO="$TMP/no-repo" \
    "$ROOT/llm-server-gui" --ctx-size 1024 2>&1)
assert_contains "$out" "--ctx-size 1024" "dry run keeps explicit passthrough context"
assert_not_contains "$out" "--ctx-size 65536" "dry run does not add GUI default context"

echo "Test: interactive flow exposes KV placement"
out=$(printf 'c\nK\nc\nD\n\n<\nq\n' | HOME="$TMP/home" LLM_ASSUME_YES=1 \
    LLM_MODEL_DIR="$TMP/models" LLM_SERVER_REPO="$TMP/no-repo" \
    "$ROOT/llm-server-gui" 2>&1)
assert_contains "$out" "Advanced: model.gguf" "advanced configure opens explicitly"
assert_contains "$out" "--no-kv-offload" "dry run applies chosen CPU KV placement"

echo "Test: interactive flow launches chosen tuned config"
mkdir -p "$TMP/home/.cache/llm-server"
cat >"$TMP/home/.cache/llm-server/tune_model.gguf_test_ik.json" <<'JSON'
{
  "model": "model.gguf",
  "rounds": 4,
  "baseline_gen_tps": 10.0,
  "tuned_at": "2026-04-30T00:00:00Z",
  "best_config": {
    "name": "gui-test",
    "gen_tps": 12.0,
    "flags": {
      "--cache-type-k": "q8_0",
      "-ub": 256
    }
  }
}
JSON

# Open advanced configure → press t (tuned config picker) → 1 (pick first
# tuned cache) → D (dry-run) → Enter → < → q.
out=$(printf 'c\nt\n1\nD\n\n<\nq\n' | HOME="$TMP/home" LLM_ASSUME_YES=1 \
    LLM_MODEL_DIR="$TMP/models" LLM_SERVER_REPO="$TMP/no-repo" "$ROOT/llm-server-gui" 2>&1)
assert_contains "$out" "Tuned configs for model.gguf" "config chooser opens from configure screen"
assert_contains "$out" "12.00 tok/s" "config chooser shows measured performance"
assert_contains "$out" "Using selected AI-tuned config: tune_model.gguf_test_ik.json" "dry run uses the selected tuned config"
assert_contains "$out" "--cache-type-k q8_0" "selected tuned config overrides launch flags"

if (( FAIL > 0 )); then
    echo ""
    echo "GUI regression: $PASS passed, $FAIL failed"
    exit 1
fi

echo ""
echo "GUI regression: $PASS passed, 0 failed"
