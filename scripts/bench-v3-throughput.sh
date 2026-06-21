#!/usr/bin/env bash
# Measure OpenAI-compatible serving throughput for the Go ggrun launcher.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-}"
shift || true

GO_BIN="${LLM_SERVER_GO_BIN:-$ROOT/go/ggrun}"
SERVER_BIN="${LLAMA_SERVER:-}"
OUT_DIR="${BENCH_OUT_DIR:-$ROOT/.benchmarks/throughput-$(date -u +%Y%m%dT%H%M%SZ)}"
PORT=18200
CTX_SIZE=32768
CTX_PER_SLOT=""
BACKEND=auto
PARALLEL=1
CONCURRENCY=4
REQUESTS=12
MAX_TOKENS=256
PROMPT_PROFILE="${BENCH_PROMPT_PROFILE:-chat}"
EXTRA_FLAGS=()

usage() {
    cat >&2 <<'EOF'
Usage: scripts/bench-v3-throughput.sh <model.gguf> [options]

Options:
  --go-bin <path>        Go ggrun binary (default: ./go/ggrun)
  --server-bin <path>    llama-server binary used by Go launcher
  --out-dir <dir>        Output directory (default: .benchmarks/throughput-<utc>)
  --port <n>             Port to use (default: 18200)
  --ctx-size <value>     Total context passed to Go launcher (default: 32768)
  --ctx-per-slot <n>     Keep per-slot context fixed; total ctx = n * parallel
  --backend <name>       auto|llama|ik_llama|vulkan passed to Go launcher
  --parallel <n>         llama-server parallel slots (default: 1)
  --concurrency <n>      Concurrent HTTP clients (default: 4)
  --requests <n>         Total requests (default: 12)
  --max-tokens <n>       Max generated tokens per request (default: 256)
  --profile <name>       Prompt profile: chat|long|repeat|code|spec (default: chat)
  --flag <arg>           Extra Go launcher arg; repeat for values too

Example:
  scripts/bench-v3-throughput.sh ~/ai_models/model.gguf \
    --server-bin ~/llama.cpp/build-vulkan/bin/llama-server \
    --backend vulkan --parallel 4 --concurrency 8
EOF
    exit 2
}

[[ -n "$MODEL" ]] || usage
while [[ $# -gt 0 ]]; do
    case "$1" in
        --go-bin) GO_BIN="$2"; shift 2 ;;
        --server-bin) SERVER_BIN="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --ctx-size) CTX_SIZE="$2"; shift 2 ;;
        --ctx-per-slot) CTX_PER_SLOT="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --requests) REQUESTS="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --profile) PROMPT_PROFILE="$2"; shift 2 ;;
        --flag) EXTRA_FLAGS+=("$2"); shift 2 ;;
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
if [[ -n "$CTX_PER_SLOT" ]]; then
    [[ "$CTX_PER_SLOT" =~ ^[0-9]+$ && "$PARALLEL" =~ ^[0-9]+$ ]] || { echo "--ctx-per-slot and --parallel must be numeric" >&2; exit 2; }
    CTX_SIZE=$((CTX_PER_SLOT * PARALLEL))
fi
case "$PROMPT_PROFILE" in
    chat|long|repeat|code|spec) ;;
    *) echo "unknown prompt profile: $PROMPT_PROFILE" >&2; exit 2 ;;
esac

mkdir -p "$OUT_DIR"
MODEL_ABS="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"
SERVER_ABS="$(cd "$(dirname "$SERVER_BIN")" && pwd)/$(basename "$SERVER_BIN")"
GO_ABS="$(cd "$(dirname "$GO_BIN")" && pwd)/$(basename "$GO_BIN")"
LOG="$OUT_DIR/server.log"
JSON="$OUT_DIR/result.json"
SUMMARY="$OUT_DIR/summary.md"

wait_health() {
    local deadline=$((SECONDS + 900))
    while (( SECONDS < deadline )); do
        if [[ -n "$cleanup_pid" ]]; then
            local state
            state="$(ps -p "$cleanup_pid" -o stat= 2>/dev/null || true)"
            state="${state//[[:space:]]/}"
            if [[ -z "$state" || "$state" == Z* ]]; then
                return 1
            fi
        fi
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || \
           curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

cleanup_pid=""
cleanup() {
    if [[ -n "$cleanup_pid" ]]; then
        kill "$cleanup_pid" 2>/dev/null || true
        wait "$cleanup_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

cat >"$OUT_DIR/metadata.txt" <<EOF
model=$MODEL_ABS
server_bin=$SERVER_ABS
go_bin=$GO_ABS
ctx_size=$CTX_SIZE
ctx_per_slot=${CTX_PER_SLOT:-auto}
backend=$BACKEND
parallel=$PARALLEL
concurrency=$CONCURRENCY
requests=$REQUESTS
max_tokens=$MAX_TOKENS
prompt_profile=$PROMPT_PROFILE
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "[launch] Go ggrun on port $PORT, parallel=$PARALLEL, ctx-size=$CTX_SIZE"
LLAMA_SERVER="$SERVER_ABS" "$GO_ABS" "$MODEL_ABS" \
    --port "$PORT" --host 127.0.0.1 --ctx-size "$CTX_SIZE" \
    --backend "$BACKEND" --server-bin "$SERVER_ABS" --parallel "$PARALLEL" \
    "${EXTRA_FLAGS[@]}" >"$LOG" 2>&1 &
cleanup_pid=$!

if ! wait_health; then
    echo "server failed health check; see $LOG" >&2
    exit 1
fi

python3 - "$PORT" "$(basename "$MODEL_ABS")" "$CONCURRENCY" "$REQUESTS" "$MAX_TOKENS" "$PROMPT_PROFILE" >"$JSON" <<'PY'
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request

port = int(sys.argv[1])
model = sys.argv[2]
concurrency = int(sys.argv[3])
requests = int(sys.argv[4])
max_tokens = int(sys.argv[5])
profile = sys.argv[6]
url = f"http://127.0.0.1:{port}/v1/chat/completions"

def make_prompt(profile_name, request_id):
    if profile_name == "long":
        return (
            "Write a practical local LLM inference runbook for an engineer tuning llama.cpp serving. "
            "Cover request batching, KV cache size, GPU layer placement, split mode, speculative decoding, "
            "quality checks, and failure handling. Use numbered sections, include tradeoffs, and continue "
            f"until the runbook is complete. Request id: {request_id}."
        )
    if profile_name == "repeat":
        rows = [
            f"event {n:03d} | gpu={n % 4} | batch={1 + (n % 8):02d} | phase=decode | status=green | latency_ms={42 + n}"
            for n in range(1, 49)
        ]
        return (
            "Continue this telemetry ledger using exactly the same pipe-delimited format. "
            "Do not summarize. Continue with event 049 and produce as many following rows as possible.\n"
            + "\n".join(rows)
        )
    if profile_name == "code":
        snippet = """
def tune_candidate(name, ctx_size, batch, ubatch, parallel):
    result = run_server(name=name, ctx_size=ctx_size, batch=batch, ubatch=ubatch, parallel=parallel)
    if result.error:
        return {"name": name, "status": "failed", "score": 0}
    return {"name": name, "status": "ok", "score": result.decode_tps}
"""
        return (
            "Continue this Python benchmark module with validation helpers, deterministic candidate generation, "
            "and compact JSON output. Keep the same style and include runnable code only.\n"
            + snippet * 5
            + f"\n# request_id = {request_id}\n"
        )
    if profile_name == "spec":
        rows = [
            f"slot={n:03d} route=moe shard={n % 6} draft=ngram accept=track cache=kv batch={8 + (n % 5)} outcome=continue"
            for n in range(1, 65)
        ]
        return (
            "Continue the structured decode trace below. Preserve the exact key=value order and continue the sequence. "
            "Do not explain the trace; only append more trace rows.\n"
            + "\n".join(rows)
        )
    return (
        "Write a compact technical explanation of GPU inference serving. "
        f"Use short paragraphs and continue until the answer is complete. Request id: {request_id}"
    )

def run_one(i):
    prompt_text = make_prompt(profile, i)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0.1 if profile in {"repeat", "code", "spec"} else 0.2,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read().decode())
    elapsed = time.time() - start
    usage = data.get("usage", {})
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    completion_tokens = usage.get("completion_tokens") or max(1, len(text) // 4)
    prompt_tokens = usage.get("prompt_tokens") or max(1, len(prompt_text) // 4)
    timings = data.get("timings", {})
    draft_tokens = timings.get("draft_n") or 0
    draft_accepted = timings.get("draft_n_accepted") or 0
    return {
        "id": i,
        "latency_s": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "draft_tokens": draft_tokens,
        "draft_accepted": draft_accepted,
        "short_completion": completion_tokens < max(8, max_tokens // 2),
    }

started = time.time()
results = []
errors = []
with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
    future_map = {pool.submit(run_one, i): i for i in range(requests)}
    for fut in concurrent.futures.as_completed(future_map):
        try:
            results.append(fut.result())
        except Exception as exc:
            errors.append({"id": future_map[fut], "error": str(exc)})
elapsed = time.time() - started

completion_tokens = sum(r["completion_tokens"] for r in results)
prompt_tokens = sum(r["prompt_tokens"] for r in results)
draft_tokens = sum(r.get("draft_tokens", 0) for r in results)
draft_accepted = sum(r.get("draft_accepted", 0) for r in results)
latencies = [r["latency_s"] for r in results]
doc = {
    "model": model,
    "prompt_profile": profile,
    "max_tokens": max_tokens,
    "concurrency": concurrency,
    "requests": requests,
    "completed_requests": len(results),
    "errors": errors,
    "wall_time_s": elapsed,
    "prompt_tokens": prompt_tokens,
    "completion_tokens": completion_tokens,
    "draft_tokens": draft_tokens,
    "draft_accepted": draft_accepted,
    "draft_accept_rate": draft_accepted / draft_tokens if draft_tokens > 0 else 0,
    "aggregate_completion_tps": completion_tokens / elapsed if elapsed > 0 else 0,
    "aggregate_total_tps": (prompt_tokens + completion_tokens) / elapsed if elapsed > 0 else 0,
    "latency_avg_s": statistics.mean(latencies) if latencies else 0,
    "latency_p50_s": statistics.median(latencies) if latencies else 0,
    "latency_max_s": max(latencies) if latencies else 0,
    "short_completions": sum(1 for r in results if r["short_completion"]),
    "results": sorted(results, key=lambda r: r["id"]),
}
print(json.dumps(doc, indent=2))
PY

python3 - "$JSON" "$SUMMARY" "$MODEL_ABS" "$SERVER_ABS" "$GO_ABS" "$PARALLEL" "$CONCURRENCY" "$CTX_SIZE" "${CTX_PER_SLOT:-auto}" <<'PY'
import json
import sys

json_path, summary_path, model, server, go_bin, parallel, concurrency, ctx_size, ctx_per_slot = sys.argv[1:]
doc = json.load(open(json_path, encoding="utf-8"))
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("# ggrun v3 throughput benchmark\n\n")
    f.write(f"Model: `{model}`\n\n")
    f.write(f"Backend: `{server}`\n\n")
    f.write(f"Go binary: `{go_bin}`\n\n")
    f.write(f"Prompt profile: `{doc.get('prompt_profile', 'unknown')}`; max tokens: `{doc.get('max_tokens', 'unknown')}`\n\n")
    f.write(f"Total context: `{ctx_size}`; per-slot context: `{ctx_per_slot}`\n\n")
    f.write("| Parallel | Concurrency | Requests | Completion tok/s | Total tok/s | Draft accepted | Avg latency s | Errors |\n")
    f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    f.write(
        f"| {parallel} | {concurrency} | {doc['completed_requests']}/{doc['requests']} | "
        f"{doc['aggregate_completion_tps']:.2f} | {doc['aggregate_total_tps']:.2f} | "
        f"{doc.get('draft_accepted', 0)}/{doc.get('draft_tokens', 0)} ({doc.get('draft_accept_rate', 0):.1%}) | "
        f"{doc['latency_avg_s']:.2f} | {len(doc['errors'])} |\n"
    )
print(summary_path)
PY

echo "Wrote throughput artifacts to $OUT_DIR"
echo "$SUMMARY"
