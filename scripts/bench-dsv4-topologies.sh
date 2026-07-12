#!/usr/bin/env bash
# Compare DeepSeek-V4 MoE placement strategies under the real Claude Code
# service shape: 1,048,576 total context and four parallel slots.
set -uo pipefail

# ggrun and its placement cache use PCI bus ordering. Direct benchmark launches
# must use the same CUDA enumeration or a CUDA0 plan for the 3090 Ti can land on
# a 12 GiB card instead.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

APP_HOME="${GGRUN_APP_HOME:-/home/mik/ggrun}"
MODEL="${1:-$APP_HOME/models/UD-IQ4_XS/DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf}"
SERVER="${LLAMA_SERVER:-$APP_HOME/.bin/llama-server-cuda}"
FIT="${LLAMA_FIT_PARAMS:-$APP_HOME/.bin/llama-fit-params}"
PORT="${PORT:-8081}"
LOAD_TIMEOUT_S="${LOAD_TIMEOUT_S:-1200}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTDIR="${OUTDIR:-$APP_HOME/.benchmarks/dsv4-topology-$STAMP}"
RESULTS="$OUTDIR/results.md"
mkdir -p "$OUTDIR"

EXPERT='ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b)'
WEIGHTS='ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp)'
GATE_UP='ffn_(gate_up|up_gate|gate|up)_(ch|)exps'
DOWN='ffn_down_(ch|)exps'

declare -a CONFIGS=(
  baseline-router
  baseline-weights-only
  fractional-gate-up
  split-full-layer
  owner-gpu0-gpu2
  upstream-fit
)
if [[ -n "${CONFIGS_CSV:-}" ]]; then
  IFS=',' read -r -a CONFIGS <<<"$CONFIGS_CSV"
fi
declare -A UBATCH SPLIT NCPU OT SCALE_QUEUES

UBATCH[baseline-router]=256
SPLIT[baseline-router]='1,0,0'
NCPU[baseline-router]=37
OT[baseline-router]="blk\\.(0|1|2)\\.$EXPERT.*=CUDA2,blk\\.(3|4|5)\\.$EXPERT.*=CUDA1,exps=CPU"

UBATCH[baseline-launch-queues]=256
SPLIT[baseline-launch-queues]='1,0,0'
NCPU[baseline-launch-queues]=37
OT[baseline-launch-queues]="${OT[baseline-router]}"
SCALE_QUEUES[baseline-launch-queues]='4x'

UBATCH[baseline-weights-only]=256
SPLIT[baseline-weights-only]='1,0,0'
NCPU[baseline-weights-only]=37
OT[baseline-weights-only]="blk\\.(0|1|2)\\.$WEIGHTS.*=CUDA2,blk\\.(3|4|5)\\.$WEIGHTS.*=CUDA1,exps=CPU"

UBATCH[fractional-gate-up]=256
SPLIT[fractional-gate-up]='1,0,0'
NCPU[fractional-gate-up]=37
OT[fractional-gate-up]="blk\\.(0|1|2)\\.$EXPERT.*=CUDA2,blk\\.(3|4|5)\\.$EXPERT.*=CUDA1,blk\\.(6)\\.$GATE_UP.*=CUDA2,blk\\.(7)\\.$GATE_UP.*=CUDA1,exps=CPU"

UBATCH[split-full-layer]=256
SPLIT[split-full-layer]='1,0,0'
NCPU[split-full-layer]=36
OT[split-full-layer]="blk\\.(0|1|2)\\.$EXPERT.*=CUDA2,blk\\.(3|4|5)\\.$EXPERT.*=CUDA1,blk\\.(6)\\.$GATE_UP.*=CUDA2,blk\\.(6)\\.$DOWN.*=CUDA1,exps=CPU"

UBATCH[owner-gpu0-gpu2]=64
SPLIT[owner-gpu0-gpu2]='0.90,0,0.10'
NCPU[owner-gpu0-gpu2]=35
OT[owner-gpu0-gpu2]="blk\\.(0|1|2)\\.$EXPERT.*=CUDA0,blk\\.(3|4|5)\\.$EXPERT.*=CUDA1,blk\\.(6|7)\\.$EXPERT.*=CUDA2,exps=CPU"

UBATCH[upstream-fit]=256

say() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$OUTDIR/run.log"
}

base_args() {
  local ub=$1
  BASE_ARGS=(
    -m "$MODEL" --host 127.0.0.1 --port "$PORT"
    --ctx-size 1048576 --parallel 4
    --flash-attn on --cache-type-k f16 --cache-type-v f16 --kv-offload
    -b 2048 -ub "$ub" --threads 8 --threads-batch 8
    --jinja --no-mmap -cram 0 --ctx-checkpoints 0
    --reasoning-budget 0 --alias local --timeout 1800
  )
}

fit_args() {
  local ub=$1
  FIT_ARGS=(
    -m "$MODEL" --ctx-size 1048576 --parallel 4
    --flash-attn on --cache-type-k f16 --cache-type-v f16 --kv-offload
    -b 2048 -ub "$ub" --threads 8 --threads-batch 8 --no-mmap
  )
}

placement_args() {
  local label=$1
  if [[ "$label" == upstream-fit ]]; then
    PLACEMENT_ARGS=(--fit on --fit-target 512)
  else
    PLACEMENT_ARGS=(
      -ngl 999 --fit off --split-mode layer
      --tensor-split "${SPLIT[$label]}"
      -ot "${OT[$label]}"
      --n-cpu-moe "${NCPU[$label]}"
    )
  fi
}

wait_for_vram() {
  local limit=${1:-1000}
  for _ in $(seq 1 90); do
    local max_used=0 used
    while IFS= read -r used; do
      used=${used//[^0-9]/}
      [[ -n "$used" && "$used" -gt "$max_used" ]] && max_used=$used
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [[ "$max_used" -lt "$limit" ]] && return 0
    sleep 2
  done
  return 1
}

shutdown_server() {
  local pid=$1
  kill -INT "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  kill -TERM "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  wait_for_vram 1000 || true
}

active_server_pid=''
cleanup() {
  if [[ -n "$active_server_pid" ]]; then
    shutdown_server "$active_server_pid"
    active_server_pid=''
  fi
}
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

run_requests() {
  local label=$1
  python3 - "$PORT" "$OUTDIR/$label.requests.json" <<'PY'
import concurrent.futures
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
out_path = sys.argv[2]

def request(tag, prompt_tokens, output_tokens):
    words = " ".join(f"w{i % 997}" for i in range(prompt_tokens))
    prompt = f"Benchmark {tag}. Read the following data. {words}\nReply with the single word done."
    body = json.dumps({
        "model": "local",
        "max_tokens": output_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=7200) as response:
        data = json.loads(response.read().decode("utf-8", "replace"), strict=False)
    elapsed = time.monotonic() - started
    timings = data.get("timings", {})
    message = (data.get("choices") or [{}])[0].get("message", {})
    text = (message.get("content") or "") + " " + (message.get("reasoning_content") or "")
    return {
        "tag": tag,
        "elapsed_s": elapsed,
        "prompt_tps": timings.get("prompt_per_second", 0),
        "decode_tps": timings.get("predicted_per_second", 0),
        "prompt_n": timings.get("prompt_n", 0),
        "predicted_n": timings.get("predicted_n", 0),
        "sane": "done" in text.lower(),
    }

# Warm the graph and verify deterministic output before recording.
warmup = request("warmup", 64, 8)
serial = request("serial", 2048, 128)

started = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(request, f"concurrent-{i}", 1024 + i * 32, 64) for i in range(4)]
    concurrent = [future.result() for future in futures]
wall = time.monotonic() - started
total_predicted = sum(row["predicted_n"] for row in concurrent)

result = {
    "warmup": warmup,
    "serial": serial,
    "concurrent": concurrent,
    "concurrent_wall_s": wall,
    "aggregate_decode_tps": total_predicted / wall if wall > 0 else 0,
}
with open(out_path, "w") as fh:
    json.dump(result, fh, indent=2)
print(json.dumps(result))
PY
}

{
  echo "# DeepSeek-V4 topology benchmark — $STAMP"
  echo
  echo "Model: \`$MODEL\`"
  echo
  echo "Fixed service shape: ctx=1048576, parallel=4, f16 GPU KV, Flash Attention on."
  echo
  echo '| configuration | outcome | load | VRAM MiB (GPU0/GPU1/GPU2) | serial prefill t/s | serial decode t/s | four-request aggregate t/s | quality |'
  echo '|---|---|---:|---|---:|---:|---:|---|'
} > "$RESULTS"

for label in "${CONFIGS[@]}"; do
  say "starting $label"
  wait_for_vram 1000 || say "warning: VRAM did not fully drain before $label"
  base_args "${UBATCH[$label]}"
  fit_args "${UBATCH[$label]}"
  placement_args "$label"

  preflight="$OUTDIR/$label.preflight.log"
  if [[ "$label" == upstream-fit ]]; then
    "$FIT" "${FIT_ARGS[@]}" "${PLACEMENT_ARGS[@]}" --fit-print on > "$preflight" 2>&1 || true
  else
    "$FIT" "${FIT_ARGS[@]}" "${PLACEMENT_ARGS[@]}" --fit-print on > "$preflight" 2>&1 || true
  fi

  launch_log="$OUTDIR/$label.launch.log"
  started=$SECONDS
  if [[ -n "${SCALE_QUEUES[$label]:-}" ]]; then
    CUDA_SCALE_LAUNCH_QUEUES="${SCALE_QUEUES[$label]}" nohup "$SERVER" "${BASE_ARGS[@]}" "${PLACEMENT_ARGS[@]}" > "$launch_log" 2>&1 &
  else
    nohup "$SERVER" "${BASE_ARGS[@]}" "${PLACEMENT_ARGS[@]}" > "$launch_log" 2>&1 &
  fi
  server_pid=$!
  active_server_pid=$server_pid
  outcome=''
  while :; do
    if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      outcome=healthy
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      outcome=crashed
      break
    fi
    if (( SECONDS - started > LOAD_TIMEOUT_S )); then
      outcome=timeout
      break
    fi
    sleep 10
  done
  load_s=$((SECONDS - started))

  if [[ "$outcome" != healthy ]]; then
    echo "| $label | **$outcome** | ${load_s}s | - | - | - | - | - |" >> "$RESULTS"
    shutdown_server "$server_pid"
    active_server_pid=''
    continue
  fi

  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/)
  nvidia-smi dmon -s pucm -d 1 -c 1800 > "$OUTDIR/$label.dmon.log" 2>&1 &
  dmon_pid=$!
  request_json=$(run_requests "$label" 2>> "$OUTDIR/$label.requests.err")
  request_rc=$?
  kill "$dmon_pid" 2>/dev/null || true
  wait "$dmon_pid" 2>/dev/null || true

  if [[ "$request_rc" -ne 0 ]]; then
    echo "| $label | request-failed | ${load_s}s | $vram | - | - | - | - |" >> "$RESULTS"
  else
    read -r prefill decode aggregate sane < <(python3 - "$request_json" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
s = r["serial"]
all_sane = s["sane"] and all(x["sane"] for x in r["concurrent"])
print(f'{s["prompt_tps"]:.2f} {s["decode_tps"]:.3f} {r["aggregate_decode_tps"]:.3f} {"ok" if all_sane else "BAD"}')
PY
)
    echo "| $label | healthy | ${load_s}s | $vram | $prefill | $decode | $aggregate | $sane |" >> "$RESULTS"
  fi

  shutdown_server "$server_pid"
  active_server_pid=''
done

say "complete: $RESULTS"
