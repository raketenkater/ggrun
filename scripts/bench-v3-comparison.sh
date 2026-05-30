#!/usr/bin/env bash
# Produce reproducible v3 launch/benchmark artifacts for release posts.
# Compares raw llama-server, Bash llm-server, and Go llm-server on one model.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-}"
shift || true

GO_BIN="${LLM_SERVER_GO_BIN:-$ROOT/go/llm-server}"
SERVER_BIN="${LLAMA_SERVER:-}"
OUT_DIR="${BENCH_OUT_DIR:-$ROOT/benchmarks/v3-$(date -u +%Y%m%dT%H%M%SZ)}"
PORT_BASE=18081
CTX_SIZE="fit"
BACKEND="auto"
ROUNDS=1
RAW_FLAGS=()
COMMON_FLAGS=()

usage() {
    cat >&2 <<'EOF'
Usage: scripts/bench-v3-comparison.sh <model.gguf> [options]

Options:
  --go-bin <path>        Go llm-server binary (default: ./go/llm-server)
  --server-bin <path>    llama-server binary for raw baseline and wrappers
  --out-dir <dir>        Output directory (default: benchmarks/v3-<utc>)
  --port-base <n>        First port to use (default: 18081)
  --ctx-size <value>     fit|max|number passed to wrappers (default: fit)
  --backend <name>       auto|llama|ik_llama|vulkan passed to wrappers
  --rounds <n>           Repeat each target n times (default: 1)
  --raw-flag <arg>       Extra raw llama-server arg; repeat for values too
  --flag <arg>           Extra wrapper arg; repeat for values too

Examples:
  scripts/bench-v3-comparison.sh ~/ai_models/qwen.gguf --server-bin ~/llama.cpp/build/bin/llama-server
  LLM_SERVER_GO_BIN=go/llm-server scripts/bench-v3-comparison.sh model.gguf --backend vulkan --ctx-size 32768
EOF
    exit 2
}

[[ -n "$MODEL" ]] || usage
while [[ $# -gt 0 ]]; do
    case "$1" in
        --go-bin) GO_BIN="$2"; shift 2 ;;
        --server-bin) SERVER_BIN="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --port-base) PORT_BASE="$2"; shift 2 ;;
        --ctx-size) CTX_SIZE="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --raw-flag) RAW_FLAGS+=("$2"); shift 2 ;;
        --flag) COMMON_FLAGS+=("$2"); shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done

[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 2; }
[[ -x "$GO_BIN" ]] || { echo "Go binary not executable: $GO_BIN" >&2; exit 2; }
if [[ -z "$SERVER_BIN" ]]; then
    if command -v llama-server >/dev/null 2>&1; then
        SERVER_BIN="$(command -v llama-server)"
    else
        echo "llama-server not found; pass --server-bin" >&2
        exit 2
    fi
fi
[[ -x "$SERVER_BIN" ]] || { echo "llama-server not executable: $SERVER_BIN" >&2; exit 2; }

mkdir -p "$OUT_DIR"
MODEL_ABS="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"
SERVER_ABS="$(cd "$(dirname "$SERVER_BIN")" && pwd)/$(basename "$SERVER_BIN")"
GO_ABS="$(cd "$(dirname "$GO_BIN")" && pwd)/$(basename "$GO_BIN")"

wait_health() {
    local port="$1" deadline=$((SECONDS + 900))
    while (( SECONDS < deadline )); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 || \
           curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

bench_http() {
    local port="$1" model_name="$2"
    python3 - "$port" "$model_name" <<'PY'
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
model = sys.argv[2]
url = f"http://127.0.0.1:{port}/v1/chat/completions"

def chat(prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())
    elapsed = time.time() - start
    return data, elapsed

chat("Explain quantum computing in one sentence.", 32)
prompt = "Write a short story about a robot learning to paint. Include a beginning, middle, and end."
prefill, prefill_s = chat(prompt, 1)
gen, gen_s = chat(prompt, 128)
usage_p = prefill.get("usage", {})
usage_g = gen.get("usage", {})
timing_p = prefill.get("timings", {})
timing_g = gen.get("timings", {})
prompt_tokens = usage_p.get("prompt_tokens") or max(1, len(prompt) // 4)
gen_text = gen.get("choices", [{}])[0].get("message", {}).get("content", "")
gen_tokens = usage_g.get("completion_tokens") or max(1, len(gen_text) // 4)
prompt_tps = timing_p.get("prompt_per_second") or prompt_tokens / max(prefill_s, 1e-6)
# Some backends report an artificial 1e6 tok/s when a request stops after a
# single token. Treat short completions as a wall-clock result and mark them so
# release tables do not accidentally promote an invalid raw baseline.
short_completion = gen_tokens < 16
if short_completion:
    gen_tps = gen_tokens / max(gen_s, 1e-6)
else:
    gen_tps = timing_g.get("predicted_per_second") or gen_tokens / max(gen_s, 1e-6)
result = {
    "model": model,
    "prompt_tokens": prompt_tokens,
    "prompt_tps": prompt_tps,
    "gen_tokens": gen_tokens,
    "gen_tps": gen_tps,
    "short_completion": short_completion,
    "timestamp": int(time.time()),
}
print(json.dumps(result, indent=2))
PY
}

extract_gen_tps() {
    python3 - "$1" <<'PY'
import json, re, sys
text = open(sys.argv[1], encoding='utf-8', errors='replace').read()
# Prefer JSON blocks that start at the beginning of a line. Logs may contain
# stray quotes/braces, so do not parse the whole file as one stream.
lines = text.splitlines()
objs = []
for i, line in enumerate(lines):
    if not line.lstrip().startswith('{'):
        continue
    buf = []
    depth = 0
    for line2 in lines[i:]:
        buf.append(line2)
        depth += line2.count('{') - line2.count('}')
        if depth <= 0:
            break
    raw = '\n'.join(buf)
    try:
        doc = json.loads(raw)
    except Exception:
        continue
    objs.append(doc)
for doc in reversed(objs):
    if 'gen_tps' in doc:
        print(f"{float(doc['gen_tps']):.2f}")
        raise SystemExit(0)
# Bash human output.
match = re.search(r'Generation:\s+.*?@\s*([0-9.]+)\s*tok/s', text)
if match:
    print(match.group(1))
    raise SystemExit(0)
print('?')
PY
}

run_raw_once() {
    local round="$1" port="$2" log="$OUT_DIR/raw-$round.log" json="$OUT_DIR/raw-$round.json"
    echo "[raw] round $round on port $port"
    local raw_ctx_args=()
    if [[ "$CTX_SIZE" =~ ^[0-9]+$ ]]; then
        raw_ctx_args=(--ctx-size "$CTX_SIZE")
    fi
    "$SERVER_ABS" -m "$MODEL_ABS" --host 127.0.0.1 --port "$port" --jinja "${raw_ctx_args[@]}" "${RAW_FLAGS[@]}" >"$log" 2>&1 &
    local pid=$!
    if ! wait_health "$port"; then
        echo "raw server failed health check; see $log" >&2
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    fi
    bench_http "$port" "$(basename "$MODEL_ABS")" >"$json"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

run_wrapper_once() {
    local label="$1" bin="$2" round="$3" port="$4" out
    out="$OUT_DIR/$label-$round.out"
    echo "[$label] round $round on port $port"
    LLAMA_SERVER="$SERVER_ABS" "$bin" "$MODEL_ABS" --benchmark --port "$port" \
        --ctx-size "$CTX_SIZE" --backend "$BACKEND" --server-bin "$SERVER_ABS" \
        "${COMMON_FLAGS[@]}" >"$out" 2>&1
}

cat >"$OUT_DIR/metadata.txt" <<EOF
model=$MODEL_ABS
server_bin=$SERVER_ABS
go_bin=$GO_ABS
ctx_size=$CTX_SIZE
backend=$BACKEND
rounds=$ROUNDS
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

for round in $(seq 1 "$ROUNDS"); do
    base=$((PORT_BASE + (round - 1) * 10))
    run_raw_once "$round" "$base" || true
    run_wrapper_once bash "$ROOT/llm-server" "$round" "$((base + 1))" || true
    run_wrapper_once go "$GO_ABS" "$round" "$((base + 2))" || true
done

summary="$OUT_DIR/summary.md"
{
    echo "# llm-server v3 benchmark comparison"
    echo
    echo "Model: \`$MODEL_ABS\`"
    echo "Backend: \`$SERVER_ABS\`"
    echo "Go binary: \`$GO_ABS\`"
    echo
    echo "| Target | Round | Decode tok/s | Output |"
    echo "|---|---:|---:|---|"
    for round in $(seq 1 "$ROUNDS"); do
        for target in raw bash go; do
            file="$OUT_DIR/$target-$round.out"
            [[ "$target" == raw ]] && file="$OUT_DIR/raw-$round.json"
            [[ -f "$file" ]] || { echo "| $target | $round | failed | $(basename "$file") |"; continue; }
            tps="$(extract_gen_tps "$file")"
            echo "| $target | $round | $tps | $(basename "$file") |"
        done
    done
} >"$summary"

echo "Wrote benchmark artifacts to $OUT_DIR"
echo "$summary"
