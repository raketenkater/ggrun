#!/usr/bin/env bash
# Guard and capture the idle-hardware evidence needed by the ggrun core release.
# This script never stops a process. Busy hardware is a hard refusal.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
CANDIDATE="$REPO_DIR/go/ggrun"
OUT_ROOT="$REPO_DIR/.benchmarks/core-readiness"
MODE=check

usage() {
  cat <<'EOF'
Usage: scripts/core-hardware-window.sh [check|plan|capture|bandwidth] [options]

Modes:
  check       Refuse if a model process or NVIDIA compute process is active.
  plan        Print the ordered hardware matrix; does not require idle hardware.
  capture     After the idle check, record candidate, git, PCI, CPU, RAM, and GPU evidence.
  bandwidth   Run capture, then ggrun detect --bandwidth and a cached follow-up detect.

Options:
  --candidate PATH   Candidate ggrun binary (default: go/ggrun)
  --out-root DIR     Evidence parent (default: .benchmarks/core-readiness)
  -h, --help         Show this help

The script never kills, drains, reloads, installs, or launches a model server.
EOF
}

die() {
  printf 'hardware-window: %s\n' "$*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    check|plan|capture|bandwidth)
      MODE=$1
      shift
      ;;
    --candidate)
      (($# >= 2)) || die "--candidate requires a path"
      CANDIDATE=$2
      shift 2
      ;;
    --out-root)
      (($# >= 2)) || die "--out-root requires a directory"
      OUT_ROOT=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

print_plan() {
  cat <<'EOF'
Core hardware window order

0. Guard: no ggrun/llama/FreeToken server and no NVIDIA compute process.
1. Capture: candidate SHA/version, git diff, CPU/RAM/PCI/GPU topology, backend versions.
2. Bandwidth: measured host and pinned H2D/D2H profile; second detect must reuse it.
3. Dense: fully resident model on fastest suitable GPU.
4. MoE: heterogeneous offload with measured ubatch 512/1024/2048 candidates.
5. Hybrid/SWA: cold, append, older branch, identical replay.
6. Long context: cross checkpoint boundary, then concurrent decode and graph growth.
7. KV compatibility: odd head dimension must reject incompatible block-quantized KV.
8. CLI: invalid GPU/model, occupied port, unknown command, bare benchmark, non-TTY.
9. Agentic capacity: serial baseline, full-context slots where valid, exact-model replica.
10. Separate comparison lane: ggrun vs FreeToken real-agent suite on one identical GPU grant.

Evidence contract: exact argv + model/backend/hardware identity + load/TTFT/prefill/decode
+ cache + RAM/VRAM/power + stability. A load-only result is not a pass.
EOF
}

if [[ "$MODE" == plan ]]; then
  print_plan
  exit 0
fi

[[ -x "$CANDIDATE" ]] || die "candidate is missing or not executable: $CANDIDATE"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required on this release rig"

busy_processes=$(ps -eo pid=,comm=,args= | awk -v self="$$" '
  $1 == self { next }
  $2 == "ggrun" || $2 == "ft" || $2 ~ /^llama-(server|cli)/ { print; next }
  /python(3)? .*freetoken(\.cli)? .*serve/ { print }
')
gpu_processes=$(nvidia-smi \
  --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits 2>/dev/null || true)

if [[ -n "$busy_processes" || -n "$gpu_processes" ]]; then
  printf 'hardware-window: REFUSED — hardware is busy; nothing was stopped.\n' >&2
  if [[ -n "$busy_processes" ]]; then
    printf '\nModel-related processes:\n%s\n' "$busy_processes" >&2
  fi
  if [[ -n "$gpu_processes" ]]; then
    printf '\nNVIDIA compute processes (pid, name, MiB):\n%s\n' "$gpu_processes" >&2
  fi
  exit 2
fi

printf 'hardware-window: idle guard passed; no model or NVIDIA compute process found.\n'
if [[ "$MODE" == check ]]; then
  exit 0
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
out_dir="$OUT_ROOT/$stamp"
mkdir -p "$out_dir"

printf '%s\n' "$CANDIDATE" >"$out_dir/candidate.path"
"$CANDIDATE" version >"$out_dir/candidate.version.txt"
# A wrapper or symlink can front any build: hash the executable that answered
# version --json, and keep the entry point's own hash apart from it.
"$CANDIDATE" version --json >"$out_dir/candidate.build.json"
candidate_exe=$(python3 -c 'import json, sys; b = json.load(open(sys.argv[1])); print(b.get("executable_resolved") or b.get("executable") or "")' \
  "$out_dir/candidate.build.json") || die "candidate version --json is not JSON: $CANDIDATE"
[[ -n "$candidate_exe" ]] || die "candidate did not report its executable: $CANDIDATE"
sha256sum "$candidate_exe" >"$out_dir/candidate.sha256"
sha256sum "$CANDIDATE" >"$out_dir/candidate.entry.sha256"
git -C "$REPO_DIR" rev-parse HEAD >"$out_dir/git-head.txt"
git -C "$REPO_DIR" status --short >"$out_dir/git-status.txt"
git -C "$REPO_DIR" diff --binary >"$out_dir/source.patch"
uname -a >"$out_dir/uname.txt"
lscpu >"$out_dir/lscpu.txt"
awk '/MemTotal|MemAvailable|HugePages_Total|Hugepagesize/ {print}' /proc/meminfo \
  >"$out_dir/memory.txt"
nvidia-smi -q >"$out_dir/nvidia-smi-q.txt"
nvidia-smi \
  --query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,power.limit \
  --format=csv,noheader >"$out_dir/gpus.csv"
if command -v lspci >/dev/null 2>&1; then
  lspci -Dvv >"$out_dir/lspci.txt"
fi

for backend in "$REPO_DIR"/.bin/*server*; do
  [[ -x "$backend" ]] || continue
  name=$(basename "$backend")
  timeout 10 "$backend" --version >"$out_dir/backend-$name.version.txt" 2>&1 || true
done

printf 'hardware-window: captured readiness evidence -> %s\n' "$out_dir"
if [[ "$MODE" == capture ]]; then
  exit 0
fi

"$CANDIDATE" detect --bandwidth >"$out_dir/detect-bandwidth.json" \
  2>"$out_dir/detect-bandwidth.log"
"$CANDIDATE" detect >"$out_dir/detect-cached.json" \
  2>"$out_dir/detect-cached.log"

printf 'hardware-window: bandwidth and cached follow-up captured -> %s\n' "$out_dir"
