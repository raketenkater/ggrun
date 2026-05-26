#!/usr/bin/env bash
# Regression tests for public safety guarantees that do not need real GPUs or
# backend model execution.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d -t llm-server-safety.XXXXXX)"
LISTENER_PID=""
trap '[[ -n "${LISTENER_PID:-}" ]] && kill "$LISTENER_PID" 2>/dev/null || true; rm -rf "$TMP"' EXIT

cat >"$TMP/llama-server" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
    --help|-h) echo "fake llama-server (test stub)"; exit 0 ;;
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

mkdir -p "$TMP/qwen"
python3 "$ROOT/tests/build_synthetic_gguf.py" \
    --out "$TMP/qwen/Qwen3-0.6B-Q8_0.gguf" --arch qwen3 --name 'Qwen3 0.6B Instruct' --basename 'Qwen3' \
    --layers 2 --hkv 1 --kl 32 --vl 32 --embd 128 --ff 256 --ctx-train 512

python3 "$ROOT/tests/build_synthetic_gguf.py" \
    --out "$TMP/remote-mmproj-F32.gguf" --arch clip --name 'Qwen3' --basename 'Qwen3' \
    --layers 2 --embd 128 --ff 256

python3 "$ROOT/tests/build_synthetic_gguf.py" \
    --out "$TMP/mtp-draft.gguf" --arch mtp --name 'Safety-MTP-Draft' --basename 'Safety-MTP-Draft' \
    --layers 2 --hkv 1 --kl 32 --vl 32 --embd 128 --ff 256 --ctx-train 512

cat >"$TMP/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

url="${@: -1}"
if [[ "$url" == "https://huggingface.co/api/models/unsloth/Qwen3-0.6B-GGUF/tree/main?recursive=1" ]]; then
    printf '[{"path":"mmproj-F32.gguf"}]\n'
    exit 0
fi
if [[ "$url" == https://huggingface.co/api/models/*/tree/main?recursive=1 ]]; then
    printf '[]\n'
    exit 0
fi
if [[ " $* " == *" --head "* ]]; then
    [[ "$url" == "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/mmproj-F32.gguf" ]]
    exit
fi
out=""
while (($#)); do
    case "$1" in
        -o) out="$2"; shift 2 ;;
        *) shift ;;
    esac
done
[[ -n "$out" && "$url" == "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/mmproj-F32.gguf" ]]
cp "$REMOTE_MMPROJ" "$out"
EOF
chmod +x "$TMP/curl"

python3 - "$TMP/port" <<'PY' &
import socket
import sys
import time

s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 0))
s.listen()
with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write(str(s.getsockname()[1]))
    f.flush()
while True:
    time.sleep(1)
PY
LISTENER_PID=$!

for _ in $(seq 1 50); do
    [[ -s "$TMP/port" ]] && break
    kill -0 "$LISTENER_PID" 2>/dev/null
    sleep 0.1
done
[[ -s "$TMP/port" ]] || { echo "foreign listener failed to start"; exit 1; }
PORT="$(cat "$TMP/port")"

export LLAMA_SERVER="$TMP/llama-server"
export LLM_MODEL_DIR="$TMP/models"
export LLM_ASSUME_YES=1
export LLM_SERVER_TEST_STOP_AFTER_AI_TUNE_PRECLEANUP=1

out=$("$ROOT/llm-server" --cpu --ai-tune --port "$PORT" "$TMP/model.gguf" 2>&1)

if ! kill -0 "$LISTENER_PID" 2>/dev/null; then
    echo "foreign listener was killed unexpectedly"
    echo "$out"
    exit 1
fi

if [[ "$out" != *"refusing to kill"* ]]; then
    echo "expected refusal warning for foreign listener"
    echo "$out"
    exit 1
fi

echo "Safety regression: foreign listener survived AI-tune pre-cleanup"

out=$("$ROOT/llm-server" --cpu --dry-run --vision "$TMP/model.gguf" 2>&1)
if [[ "$out" != *"--mmproj $TMP/mmproj-F16.gguf"* ]]; then
    echo "expected matching local mmproj to be accepted"
    echo "$out"
    exit 1
fi

echo "Safety regression: matching local mmproj accepted"

out=$(REMOTE_MMPROJ="$TMP/remote-mmproj-F32.gguf" PATH="$TMP:$PATH" \
    "$ROOT/llm-server" --cpu --dry-run --vision "$TMP/qwen/Qwen3-0.6B-Q8_0.gguf" 2>&1)
if [[ "$out" != *"https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/mmproj-F32.gguf"* ]]; then
    echo "expected Qwen3 downloader to derive the specific Hugging Face repo from filename/model name"
    echo "$out"
    exit 1
fi
if [[ "$out" != *"--mmproj $TMP/qwen/mmproj-Qwen3-0.6B-F32.gguf"* ]]; then
    echo "expected downloaded Qwen3 mmproj to be passed to backend"
    echo "$out"
    exit 1
fi

echo "Safety regression: Qwen3 mmproj resolver derives model-specific repo"

if out=$("$ROOT/llm-server" --cpu --dry-run --mmproj "$TMP/mmproj-other.gguf" "$TMP/model.gguf" 2>&1); then
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

if out=$("$ROOT/llm-server" --cpu --dry-run --mmproj "$TMP/mmproj-partial.gguf" "$TMP/model.gguf" 2>&1); then
    echo "expected incomplete explicit mmproj to fail"
    echo "$out"
    exit 1
fi
if [[ "$out" != *"mmproj metadata does not match"* ]]; then
    echo "expected clear incomplete mmproj error"
    echo "$out"
    exit 1
fi

echo "Safety regression: incomplete mmproj rejected"

out=$("$ROOT/llm-server" --cpu --dry-run "$TMP/model.gguf" 2>&1)
if [[ "$out" == *"Run AI Tune before launching"* ]]; then
    echo "first-run AI tune prompt leaked into dry-run/non-interactive path"
    echo "$out"
    exit 1
fi

echo "Safety regression: first-run AI tune prompt suppressed for dry-run"

if out=$("$ROOT/llm-server" --cpu --dry-run "$TMP/mtp-draft.gguf" 2>&1); then
    echo "expected direct MTP draft launch to fail"
    echo "$out"
    exit 1
fi
if [[ "$out" != *"draft-only GGUF"* || "$out" != *"MTP draft architecture"* ]]; then
    echo "expected clear draft-only error for MTP model"
    echo "$out"
    exit 1
fi

echo "Safety regression: direct MTP draft launch rejected before health timeout"

if grep -Eq '\(\([[:space:]]*TOTAL_EXTRA[[:space:]]*\+=' "$ROOT/llm-server"; then
    echo "found set -e-unsafe TOTAL_EXTRA arithmetic command"
    exit 1
fi
bash -c 'set -euo pipefail; TOTAL_EXTRA=0; extra=0; TOTAL_EXTRA=$(( TOTAL_EXTRA + extra ))'

echo "Safety regression: VRAM headroom arithmetic survives zero extra layers"
