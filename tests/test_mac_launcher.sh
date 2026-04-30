#!/usr/bin/env bash
# Linux-runnable regression tests for llm-server-mac using mocked macOS tools.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d -t llm-server-mac-tests.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/bin" "$TMP/models"

cat >"$TMP/llama-server" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
    --help|-h) echo "fake llama-server (test stub) --reasoning --parallel"; exit 0 ;;
    --version) echo "fake 0.0.0"; exit 0 ;;
esac
exit 0
EOF
chmod +x "$TMP/llama-server"

cat >"$TMP/bin/sysctl" <<'EOF'
#!/usr/bin/env bash
[[ "${1:-}" == "-n" ]] || exit 1
case "${2:-}" in
    hw.memsize) echo 68719476736 ;;
    hw.physicalcpu) echo 12 ;;
    hw.perflevel0.physicalcpu) echo 8 ;;
    machdep.cpu.brand_string) echo "Apple M4 Pro" ;;
    hw.model) echo "Apple M4 Pro" ;;
    *) exit 1 ;;
esac
EOF
chmod +x "$TMP/bin/sysctl"

cat >"$TMP/bin/vm_stat" <<'EOF'
#!/usr/bin/env bash
cat <<'OUT'
Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                               400000.
Pages inactive:                           600000.
OUT
EOF
chmod +x "$TMP/bin/vm_stat"

cat >"$TMP/bin/system_profiler" <<'EOF'
#!/usr/bin/env bash
cat <<'OUT'
Graphics/Displays:
    Apple M4 Pro:
      Chipset Model: Apple M4 Pro
      Metal Support: Metal 3
OUT
EOF
chmod +x "$TMP/bin/system_profiler"

cat >"$TMP/bin/lsof" <<'EOF'
#!/usr/bin/env bash
exit 1
EOF
chmod +x "$TMP/bin/lsof"

python3 "$ROOT/tests/build_synthetic_gguf.py" \
    --out "$TMP/model.gguf" --arch llama --name 'Mac-Test' \
    --layers 24 --hkv 4 --kl 128 --vl 128 --embd 2048 --ff 8192 --ctx-train 8192

python3 "$ROOT/tests/build_synthetic_gguf.py" \
    --out "$TMP/mmproj-F16.gguf" --arch clip --name 'Mac-Test' --basename 'Mac-Test' \
    --layers 2 --embd 128 --ff 256

python3 "$ROOT/tests/build_synthetic_gguf.py" \
    --out "$TMP/mmproj-other.gguf" --arch clip --name 'Other-Test' --basename 'Other-Test' \
    --layers 2 --embd 128 --ff 256

export PATH="$TMP/bin:$PATH"
export LLAMA_SERVER="$TMP/llama-server"
export LLM_MODEL_DIR="$TMP/models"

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
        echo "$out" | tail -40 | sed 's/^/      /'
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
        echo "    expected output to not contain: $needle"
        echo "$out" | tail -40 | sed 's/^/      /'
        FAIL=$((FAIL + 1))
    fi
}

echo "Test: mac metal dry-run"
out=$("$ROOT/llm-server-mac" --dry-run --parallel 1 "$TMP/model.gguf" --no-mmap 2>&1)
assert_contains "$out" "GPU: Apple M4 Pro (Metal, unified memory)" "detects mocked Metal GPU"
assert_contains "$out" "Strategy: metal_gpu" "selects Metal strategy"
assert_contains "$out" "-ngl 999" "offloads all layers to Metal"
assert_contains "$out" "--ctx-size 32768" "uses conservative mac default context"
assert_contains "$out" "--cache-type-k q8_0" "uses q8_0 KV by default"
assert_contains "$out" "--reasoning off" "disables reasoning by default when supported"
assert_contains "$out" "--parallel 1" "passes parallel slot selection through"
assert_contains "$out" "--no-mmap" "passes unknown llama-server flags through"

echo "Test: mac show-configs"
out=$("$ROOT/llm-server-mac" --show-configs 2>&1)
assert_contains "$out" "llm-server-mac cached configs" "show-configs works without model"

echo "Test: mac ai-tune parse"
out=$("$ROOT/llm-server-mac" --dry-run --ai-tune "$TMP/model.gguf" 2>&1)
assert_contains "$out" "AI Tune selected; dry-run shows the baseline macOS config before tuning" "ai-tune is explicit instead of misparsed"
assert_contains "$out" "Command:" "ai-tune dry-run still shows heuristic command"

echo "Test: mac tune-cache dry-run"
cat > "$TMP/tune-cache.json" <<EOF
{
  "model": "model.gguf",
  "best_config": {
    "name": "test-cache",
    "flags": {"-b": 1024, "-ub": 256, "--parallel": 1},
    "gen_tps": 42.0,
    "pp_tps": 100.0
  }
}
EOF
out=$("$ROOT/llm-server-mac" --dry-run --tune-cache "$TMP/tune-cache.json" "$TMP/model.gguf" 2>&1)
assert_contains "$out" "Using macOS AI-tuned config" "loads explicit mac tune cache"
assert_contains "$out" "-b 1024" "applies tuned batch flag"
assert_contains "$out" "-ub 256" "applies tuned ubatch flag"
assert_contains "$out" "--parallel 1" "applies tuned parallel flag"

echo "Test: mac vision dry-run"
out=$("$ROOT/llm-server-mac" --dry-run --vision "$TMP/model.gguf" 2>&1)
assert_contains "$out" "Vision: mmproj loaded from $TMP/mmproj-F16.gguf" "auto-detects matching local mmproj"
assert_contains "$out" "--mmproj $TMP/mmproj-F16.gguf" "passes detected mmproj to backend"

echo "Test: mac mismatched mmproj rejected"
if out=$("$ROOT/llm-server-mac" --dry-run --mmproj "$TMP/mmproj-other.gguf" "$TMP/model.gguf" 2>&1); then
    echo "  ✗ rejects mismatched explicit mmproj"
    echo "$out" | tail -40 | sed 's/^/      /'
    FAIL=$((FAIL + 1))
else
    assert_contains "$out" "mmproj metadata does not match" "rejects mismatched explicit mmproj"
fi

echo "Test: mac startup fallback"
out=$(LLM_SERVER_STARTUP_TIMEOUT=1 LLM_SERVER_TEST_STOP_AFTER_MAC_FALLBACK=1 "$ROOT/llm-server-mac" "$TMP/model.gguf" 2>&1)
assert_contains "$out" "Startup fallback: lowering Metal batch sizes." "applies first startup fallback"
assert_contains "$out" "-b 512 -ub 256" "fallback lowers batch sizes"

echo "Test: mac cpu dry-run"
out=$("$ROOT/llm-server-mac" --dry-run --cpu "$TMP/model.gguf" 2>&1)
assert_contains "$out" "GPU: skipped (--cpu flag)" "honors --cpu"
assert_contains "$out" "Strategy: cpu_only" "selects CPU strategy"
assert_contains "$out" "-ngl 0" "disables GPU layers on CPU"
assert_not_contains "$out" "-ngl 999" "does not keep Metal offload in CPU mode"

echo ""
echo "Mac launcher regression: $PASS passed, $FAIL failed"
exit $(( FAIL > 0 ? 1 : 0 ))
