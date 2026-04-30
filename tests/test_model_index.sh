#!/usr/bin/env bash
# Regression tests for model_index.py model-management metadata.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d -t llm-server-model-index.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

MODEL_DIR="$TMP/models"
CACHE_DIR="$TMP/cache"
mkdir -p "$MODEL_DIR" "$CACHE_DIR"

python3 "$ROOT/tests/build_synthetic_gguf.py" --out "$MODEL_DIR/Test-A3B-Q4_K_M.gguf" \
    --arch qwen35moe --name Test-A3B --basename Test-A3B \
    --layers 4 --hkv 1 --kl 16 --vl 16 --embd 64 --ff 128 \
    --experts 8 --exp-used 2 --exp-ff 32 --ctx-train 4096

python3 "$ROOT/tests/build_synthetic_gguf.py" --out "$MODEL_DIR/mmproj-F16.gguf" \
    --arch clip --name Test-A3B --basename Test-A3B \
    --layers 1 --hkv 1 --kl 16 --vl 16 --embd 64 --ff 128 --ctx-train 4096

cat >"$CACHE_DIR/tune_Test-A3B-Q4_K_M.gguf_hw12345678_llama.json" <<'JSON'
{
  "model": "Test-A3B-Q4_K_M.gguf",
  "rounds": 3,
  "baseline_gen_tps": 10.0,
  "tuned_at": "2026-04-30T00:00:00Z",
  "best_config": {
    "gen_tps": 12.5,
    "flags": {
      "-b": 1024,
      "--cache-type-k": "q8_0"
    }
  }
}
JSON

echo "Test: model index scan"
python3 "$ROOT/model_index.py" --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" scan >/tmp/llm-server-model-index.json
test -f "$MODEL_DIR/.llm-server/models.json"

python3 - "$MODEL_DIR/.llm-server/models.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
models = data.get("models") or []
assert len(models) == 1, models
m = models[0]
assert m["file"] == "Test-A3B-Q4_K_M.gguf", m
assert m["moe"] is True, m
assert m["quant"] == "Q4_K_M", m
assert m["mmproj"], m
assert len(m["tune_configs"]) == 1, m
assert m["tune_configs"][0]["gen_tps"] == 12.5, m
PY

echo "Test: model index download metadata"
python3 "$ROOT/model_index.py" --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" \
    update-download --repo test/repo-GGUF --quant Q4_K_M --file Test-A3B-Q4_K_M.gguf >/tmp/llm-server-model-index-update.json

python3 - "$MODEL_DIR/.llm-server/models.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
m = data["models"][0]
assert m["download"]["repo"] == "test/repo-GGUF", m
assert m["download"]["quant"] == "Q4_K_M", m
PY

echo "Test: model index GUI rows"
gui=$(python3 "$ROOT/model_index.py" --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" scan --format gui)
[[ "$gui" == *"Test-A3B-Q4_K_M.gguf"* ]]
[[ "$gui" == *"vision"* ]]
[[ "$gui" == *"1 cfg"* ]]

echo "  ✓ model index captures local models, mmproj, tune configs, and download metadata"
