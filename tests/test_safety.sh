#!/usr/bin/env bash
# Regression tests for public safety guarantees that do not need real GPUs or
# backend model execution.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GO_BIN="${LLM_SERVER_GO_BIN:-$ROOT/go/ggrun}"
if [[ ! -x "$GO_BIN" ]]; then
    (cd "$ROOT/go" && go build -o ggrun ./cmd/ggrun)
fi
TMP="$(mktemp -d -t ggrun-safety.XXXXXX)"
LISTENER_PID=""
trap '[[ -n "${LISTENER_PID:-}" ]] && kill "$LISTENER_PID" 2>/dev/null || true; rm -rf "$TMP"' EXIT

cat >"$TMP/llama-server" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
    --help|-h) echo "usage: llama-server [--help] [--version]"; exit 0 ;;
    --version) echo "fake 0.0.0"; exit 0 ;;
esac
exit 0
EOF
chmod +x "$TMP/llama-server"

python3 "$ROOT/tests/build_synthetic_gguf.py" \
    --out "$TMP/model.gguf" --arch llama --name 'Safety-Test' \
    --layers 2 --hkv 1 --kl 32 --vl 32 --embd 128 --ff 256 --ctx-train 512

python3 "$ROOT/tests/build_synthetic_gguf.py" \
    --out "$TMP/mmproj-F16.gguf" --arch clip --name 'Safety-Test' --basename 'Safety-Test' \
    --layers 2 --embd 128 --ff 256

python3 "$ROOT/tests/build_synthetic_gguf.py" \
    --out "$TMP/mmproj-other.gguf" --arch clip --name 'Other-Test' --basename 'Other-Test' \
    --layers 2 --embd 128 --ff 256

python3 "$ROOT/tests/build_synthetic_gguf.py" \
    --out "$TMP/mmproj-partial.gguf" --arch clip --name 'Safety-Test' --basename 'Safety-Test' \
    --layers 2 --embd 128 --ff 256 --tensor 'clip.blk.0.weight:1024,1024:0'

# The foreign-listener check covered AI Tune's port pre-cleanup, which was
# removed with AI Tune (review D6); no other path kills a port listener.
export LLAMA_SERVER="$TMP/llama-server"
export LLM_MODEL_DIR="$TMP/models"
export LLM_ASSUME_YES=1

out=$("$GO_BIN" --cpu --dry-run --server-bin "$TMP/llama-server" --vision "$TMP/model.gguf" 2>&1)
if [[ "$out" != *"--mmproj $TMP/mmproj-F16.gguf"* ]]; then
    echo "expected matching local mmproj to be accepted"
    echo "$out"
    exit 1
fi

echo "Safety regression: matching local mmproj accepted"

if out=$("$GO_BIN" --cpu --dry-run --server-bin "$TMP/llama-server" --mmproj "$TMP/mmproj-other.gguf" "$TMP/model.gguf" 2>&1); then
    echo "expected mismatched explicit mmproj to fail"
    echo "$out"
    exit 1
fi
if [[ "$out" != *"mmproj metadata does not match"* ]]; then
    echo "expected clear mismatch error for explicit mmproj"
    echo "$out"
    exit 1
fi

echo "Safety regression: mismatched explicit mmproj rejected"

if out=$("$GO_BIN" --cpu --dry-run --server-bin "$TMP/llama-server" --mmproj "$TMP/mmproj-partial.gguf" "$TMP/model.gguf" 2>&1); then
    echo "expected incomplete explicit mmproj to fail"
    echo "$out"
    exit 1
fi
if [[ "$out" != *"mmproj metadata does not match"* && "$out" != *"incomplete file"* ]]; then
    echo "expected clear incomplete mmproj error"
    echo "$out"
    exit 1
fi

echo "Safety regression: incomplete mmproj rejected"

out=$("$GO_BIN" --cpu --dry-run --server-bin "$TMP/llama-server" "$TMP/model.gguf" 2>&1)
if [[ "$out" == *"Run AI Tune before launching"* ]]; then
    echo "first-run AI tune prompt leaked into dry-run/non-interactive path"
    echo "$out"
    exit 1
fi

echo "Safety regression: first-run AI tune prompt suppressed for dry-run"
