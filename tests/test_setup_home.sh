#!/usr/bin/env bash
# Regression tests for the self-contained setup home.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d -t llm-server-setup-tests.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

APP_HOME="$TMP/llm-server"

echo "Test: setup-linux creates an isolated app home"
LLM_APP_HOME="$APP_HOME" \
LLM_SETUP_MODE=build \
LLM_SETUP_BACKEND=skip \
LLM_SETUP_PY_DEPS=skip \
LLM_SETUP_NONINTERACTIVE=1 \
"$ROOT/setup-linux.sh" >/tmp/llm-server-setup-test.log 2>&1

test -x "$APP_HOME/bin/llm-server"
test -x "$APP_HOME/bin/llm-server-gui"
test -x "$APP_HOME/bin/download_any_gguf.py"
test -x "$APP_HOME/bin/model_index.py"
test -f "$APP_HOME/env.sh"
test -f "$APP_HOME/config/config"
test -f "$APP_HOME/config/config.sh"
test -x "$APP_HOME/run"
test -x "$APP_HOME/gui"
test -d "$APP_HOME/models"
test -d "$APP_HOME/cache"
test -d "$APP_HOME/logs"
test -d "$APP_HOME/src"

cat >"$TMP/llama-server" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
    --help|-h) echo "fake llama-server (setup test stub)"; exit 0 ;;
    --version) echo "fake 0.0.0"; exit 0 ;;
esac
exit 0
EOF
chmod +x "$TMP/llama-server"

python3 "$ROOT/tests/build_synthetic_gguf.py" --out "$APP_HOME/models/setup-model.gguf" \
    --arch llama --name Setup-Home-Smoke --layers 2 --hkv 1 --kl 16 --vl 16 \
    --embd 64 --ff 128 --ctx-train 2048

out=$(HOME="$TMP/home" LLM_ASSUME_YES=1 \
    "$APP_HOME/run" --dry-run --cpu setup-model.gguf 2>&1)

if [[ "$out" != *"$APP_HOME/models/setup-model.gguf"* ]]; then
    echo "  ✗ app-home model directory was not used"
    echo "$out" | tail -40 | sed 's/^/    /'
    exit 1
fi

if [[ "$out" != *"Running on CPU only"* && "$out" != *"-ngl 0"* ]]; then
    echo "  ✗ installed launcher did not run dry-run path"
    echo "$out" | tail -40 | sed 's/^/    /'
    exit 1
fi

echo "  ✓ app home created"
echo "  ✓ installed launcher uses app-home models"

echo "Test: setup.sh auto-detects local Linux setup"
APP_HOME2="$TMP/auto-setup"
LLM_APP_HOME="$APP_HOME2" \
LLM_SETUP_MODE=build \
LLM_SETUP_BACKEND=skip \
LLM_SETUP_PY_DEPS=skip \
LLM_SETUP_NONINTERACTIVE=1 \
"$ROOT/setup.sh" >/tmp/llm-server-auto-setup-test.log 2>&1

test -x "$APP_HOME2/bin/llm-server"
test -x "$APP_HOME2/bin/llm-server-gui"
test -x "$APP_HOME2/bin/model_index.py"
echo "  ✓ setup.sh created app home"
