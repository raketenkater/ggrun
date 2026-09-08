#!/usr/bin/env bash
# First-run smoke test: can ggrun plan a launch on a machine it has never seen?
#
# Every gate in the optimizer is fail-closed, so a cold machine gets the most
# conservative plan ggrun can build. That is safe by construction but it is not
# self-evidently *launchable*, and the failure mode -- a new user whose first
# launch cannot be planned at all -- is the worst one the product has.
#
# This is the cheap layer of the three in docs/core-standard-launch-todos.md
# ("FIRST -- cold start"). It exercises real detection, the real fit oracle and
# the real planner against an empty cache, without paying for a model load. It
# does not prove the plan is fast; only that a machine with no evidence still
# gets a complete one.
#
# Usage: scripts/first-run-smoke.sh <model.gguf> [more args...]
set -uo pipefail

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
    echo "usage: $0 <model.gguf> [extra ggrun args]" >&2
    exit 2
fi
shift || true

if [[ ! -f "$MODEL" ]]; then
    echo "FAIL: model not found: $MODEL" >&2
    exit 2
fi

GGRUN_BIN="${GGRUN_BIN:-ggrun}"
if ! command -v "$GGRUN_BIN" >/dev/null 2>&1; then
    echo "FAIL: $GGRUN_BIN not on PATH" >&2
    exit 2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# A pristine cache and a config that points at it. Nothing this run reads may
# come from the developer's own measured history, which is the entire point.
mkdir -p "$TMP/cache" "$TMP/logs"
cat > "$TMP/llm.conf" <<EOF
CACHE_DIR=$TMP/cache
LOG_DIR=$TMP/logs
EOF

echo "cold cache: $TMP/cache (empty: $(find "$TMP/cache" -type f | wc -l) files)"

out="$TMP/dryrun.out"
LLM_CONFIG="$TMP/llm.conf" "$GGRUN_BIN" dry-run "$MODEL" "$@" > "$out" 2>&1
rc=$?

fail() { echo "FAIL: $1"; echo "--- output (last 40 lines) ---"; tail -40 "$out"; exit 1; }

[[ $rc -eq 0 ]] || fail "dry-run exited $rc on a cold cache"

# A plan is only a plan if it names the model and the server it would run.
grep -q -- "llama-server" "$out" || fail "no llama-server argv in the dry-run output"
grep -qF -- "$(basename "$MODEL")" "$out" || fail "argv does not reference the model"

# Flags every complete plan must carry. A cold machine may choose conservative
# values for these; it may not omit them.
for flag in "-ngl" "--ctx-size"; do
    grep -q -- "$flag" "$out" || fail "argv is missing $flag"
done

# Cold start must never emit a mode it has not established support for. row and
# tensor abort at load on backends that do not implement them, and a first run
# has probed nothing.
if grep -qE -- '--split-mode (row|tensor)' "$out"; then
    fail "cold plan selected a split mode it has no capability evidence for"
fi

# Fail-closed means no seats and no cache funded by displacement on a machine
# with no measured margin and no displacement proof.
if grep -qE 'seat\(s\)' "$out" && ! grep -q 'margin unmeasured' "$out"; then
    echo "WARN: seats offered on a cold cache; check the margin evidence path"
fi

echo "PASS: cold cache produced a complete plan"
echo "  argv: $(grep -m1 -o 'llama-server .*' "$out" | cut -c1-160)"

# Report what the cold run learned, which is the number the bootstrap-cost
# question in the TODO actually turns on.
learned=$(find "$TMP/cache" -type f 2>/dev/null | wc -l)
echo "  evidence files written by a single dry-run: $learned"
