#!/usr/bin/env bash
# Compare Go dry-run behavior with the Bash launcher on synthetic fixtures.
# This intentionally checks normalized backend argv semantics, not full text:
# Bash prints a report while Go prints a compact command line.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GO_BIN="${LLM_SERVER_GO_BIN:-$ROOT/go/llm-server}"

if [[ ! -x "$GO_BIN" ]]; then
    echo "Go binary not executable: $GO_BIN" >&2
    echo "Build it first, for example: (cd go && go build -o llm-server ./cmd/llm-server)" >&2
    exit 2
fi

TMP="$(mktemp -d -t llm-server-go-bash-parity.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin" "$TMP/cache" "$TMP/models"

SERVER="$TMP/llama-server"
cat >"$SERVER" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
    --help|-h)
        echo "fake llama-server --reasoning --no-kv-offload --kv-offload --spec-type --model-draft --mmproj"
        exit 0
        ;;
    --version)
        echo "fake 0.0.0"
        exit 0
        ;;
esac
exit 0
EOF
chmod +x "$SERVER"

python3 "$ROOT/tests/build_synthetic_gguf.py" \
    --out "$TMP/models/parity.gguf" --arch llama --name 'Parity-Test' --basename 'Parity-Test' \
    --layers 4 --hkv 1 --kl 32 --vl 32 --embd 128 --ff 256 --ctx-train 4096 >/dev/null

python3 "$ROOT/tests/build_synthetic_gguf.py" \
    --out "$TMP/models/mmproj-F16.gguf" --arch clip --name 'Parity-Test' --basename 'Parity-Test' \
    --layers 2 --embd 128 --ff 256 >/dev/null

MODEL="$TMP/models/parity.gguf"
MMPROJ="$TMP/models/mmproj-F16.gguf"

run_bash() {
    PATH="$TMP/bin:$PATH" \
    LLAMA_SERVER="$SERVER" \
    LLM_SCRIPT_DIR="$ROOT" \
    LLM_CONFIG="$TMP/missing-config" \
    LLM_CACHE_DIR="$TMP/cache/bash" \
    LLM_MODEL_DIR="$TMP/models" \
    LLM_ASSUME_YES=1 \
    LLM_SERVER_UPDATE_CHECKED=1 \
    "$ROOT/llm-server" "$@" 2>&1
}

run_go() {
    PATH="$TMP/bin:$PATH" \
    LLAMA_SERVER="$SERVER" \
    LLM_SCRIPT_DIR="$ROOT" \
    LLM_CONFIG="$TMP/missing-config" \
    LLM_CACHE_DIR="$TMP/cache/go" \
    LLM_MODEL_DIR="$TMP/models" \
    LLM_ASSUME_YES=1 \
    LLM_SERVER_UPDATE_CHECKED=1 \
    "$GO_BIN" "$@" 2>&1
}

command_from_output() {
    local input_file
    input_file="$(mktemp "$TMP/output.XXXXXX")"
    cat >"$input_file"
    python3 - "$SERVER" "$input_file" <<'PYSCRIPT'
import sys
server = sys.argv[1]
with open(sys.argv[2], "r", encoding="utf-8") as f:
    text = f.read()
if "Command:" in text:
    lines = text.splitlines()
    out = []
    seen = False
    for line in lines:
        if seen:
            stripped = line.strip()
            if not stripped:
                continue
            stripped = stripped.rstrip('\\').strip()
            out.append(stripped)
        elif line.strip() == "Command:":
            seen = True
    if out:
        print(" ".join(out))
        raise SystemExit(0)
for line in reversed(text.splitlines()):
    stripped = line.strip()
    if not stripped or stripped.startswith("[spec]"):
        continue
    if server in stripped or " -m " in f" {stripped} ":
        print(stripped)
        raise SystemExit(0)
print("could not find backend command in output:\n" + text, file=sys.stderr)
raise SystemExit(1)
PYSCRIPT
    rm -f "$input_file"
}

assert_has() {
    local label="$1" cmd="$2" token="$3"
    python3 - "$label" "$cmd" "$token" <<'PYSCRIPT'
import shlex, sys
label, cmd, token = sys.argv[1:]
tokens = shlex.split(cmd)
if token not in tokens:
    print(f"{label}: missing token {token!r}\ncmd: {cmd}", file=sys.stderr)
    raise SystemExit(1)
PYSCRIPT
}

assert_flag_value() {
    local label="$1" cmd="$2" flag="$3" want="$4"
    python3 - "$label" "$cmd" "$flag" "$want" <<'PYSCRIPT'
import shlex, sys
label, cmd, flag, want = sys.argv[1:]
tokens = shlex.split(cmd)
for i, tok in enumerate(tokens):
    if tok == flag and i + 1 < len(tokens):
        if tokens[i + 1] == want:
            raise SystemExit(0)
        print(f"{label}: {flag} value {tokens[i + 1]!r}, expected {want!r}\ncmd: {cmd}", file=sys.stderr)
        raise SystemExit(1)
print(f"{label}: missing flag {flag!r}\ncmd: {cmd}", file=sys.stderr)
raise SystemExit(1)
PYSCRIPT
}

check_common_cpu_contract() {
    local label="$1" cmd="$2" port="$3" ctx="$4"
    assert_has "$label" "$cmd" "$SERVER"
    assert_flag_value "$label" "$cmd" "-m" "$MODEL"
    assert_flag_value "$label" "$cmd" "--port" "$port"
    assert_flag_value "$label" "$cmd" "--ctx-size" "$ctx"
    assert_flag_value "$label" "$cmd" "-ngl" "0"
    assert_has "$label" "$cmd" "--flash-attn"
    assert_flag_value "$label" "$cmd" "--reasoning" "off"
}

compare_case() {
    local name="$1" port="$2" ctx="$3"
    shift 3
    local bash_out go_out bash_cmd go_cmd
    if ! bash_out="$(run_bash "$@")"; then
        echo "bash/$name dry-run failed" >&2
        echo "$bash_out" >&2
        exit 1
    fi
    if ! go_out="$(run_go "$@")"; then
        echo "go/$name dry-run failed" >&2
        echo "$go_out" >&2
        exit 1
    fi
    bash_cmd="$(printf '%s\n' "$bash_out" | command_from_output)"
    go_cmd="$(printf '%s\n' "$go_out" | command_from_output)"
    check_common_cpu_contract "bash/$name" "$bash_cmd" "$port" "$ctx"
    check_common_cpu_contract "go/$name" "$go_cmd" "$port" "$ctx"
    echo "  ✓ $name"
}

echo "Test: Go dry-run parity with Bash"
compare_case "flag-first cpu dry-run" 9090 2048 \
    --cpu --dry-run --server-bin "$SERVER" --ctx-size 2048 --port 9090 "$MODEL"

compare_case "model-first cpu dry-run" 9091 1024 \
    "$MODEL" --dry-run --cpu --server-bin "$SERVER" --ctx-size 1024 --port 9091

if ! bash_vision_out="$(run_bash --cpu --dry-run --server-bin "$SERVER" --mmproj "$MMPROJ" --ctx-size 2048 --port 9092 "$MODEL")"; then
    echo "bash/vision dry-run failed" >&2
    echo "$bash_vision_out" >&2
    exit 1
fi
if ! go_vision_out="$(run_go --cpu --dry-run --server-bin "$SERVER" --mmproj "$MMPROJ" --ctx-size 2048 --port 9092 "$MODEL")"; then
    echo "go/vision dry-run failed" >&2
    echo "$go_vision_out" >&2
    exit 1
fi
bash_vision_cmd="$(printf '%s\n' "$bash_vision_out" | command_from_output)"
go_vision_cmd="$(printf '%s\n' "$go_vision_out" | command_from_output)"
check_common_cpu_contract "bash/vision" "$bash_vision_cmd" 9092 2048
check_common_cpu_contract "go/vision" "$go_vision_cmd" 9092 2048
assert_flag_value "bash/vision" "$bash_vision_cmd" "--mmproj" "$MMPROJ"
assert_flag_value "go/vision" "$go_vision_cmd" "--mmproj" "$MMPROJ"
echo "  ✓ explicit mmproj dry-run"

echo "Go/Bash dry-run parity: passed"
