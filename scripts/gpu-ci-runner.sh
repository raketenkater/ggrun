#!/usr/bin/env bash
# Run ggrun's GPU end-to-end job on this machine.
#
# No GitHub-hosted runner has a GPU, on any plan, so every hosted job proves the
# CPU path only: VRAM admission, placement across devices and split mode are
# untested by all of them. The install-e2e workflow carries a `gpu` job for
# exactly that gap, and it needs a runner on hardware that has cards.
#
# The runner is brought up for one run and taken down again, deliberately.
# ggrun is a public repository, so a permanently-online runner on a personal
# workstation is a standing target for anyone who can open a pull request. The
# workflow itself only accepts a manual dispatch from main, which closes the
# fork-PR path; this script keeps the window narrow on top of that.
#
#   scripts/gpu-ci-runner.sh setup   configure the runner once
#   scripts/gpu-ci-runner.sh run     bring it up, dispatch, watch, take it down
#   scripts/gpu-ci-runner.sh status  what is registered and online

set -euo pipefail

REPO="${GGRUN_CI_REPO:-raketenkater/ggrun}"
RUNNER_DIR="${GGRUN_RUNNER_DIR:-$HOME/actions-runner}"
RUNNER_VERSION="${GGRUN_RUNNER_VERSION:-2.337.0}"
RUNNER_NAME="${GGRUN_RUNNER_NAME:-$(hostname)-gpu}"
WORKFLOW="install-e2e.yml"
# The job downloads a 563 MB Q4_0 model and serves it. Below this there is no
# point starting: ggrun would either refuse to fit or quietly place on the CPU,
# and the job asserts a real device allocation.
MIN_FREE_VRAM_MB="${GGRUN_MIN_FREE_VRAM_MB:-3000}"

die() { echo "error: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "$1 is required but not installed"; }

free_vram_mb() {
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null |
        awk -F', *' '{ total += $1 - $2 } END { print total + 0 }'
}

runner_status() {
    gh api "repos/$REPO/actions/runners" \
        --jq ".runners[] | select(.name==\"$RUNNER_NAME\") | .status" 2>/dev/null || true
}

cmd_setup() {
    need curl; need tar; need gh
    [[ $EUID -ne 0 ]] || die "do not configure or run the runner as root"
    if [[ -x "$RUNNER_DIR/run.sh" ]]; then
        echo "runner already configured at $RUNNER_DIR"
        return 0
    fi
    mkdir -p "$RUNNER_DIR"
    cd "$RUNNER_DIR"
    local tarball="actions-runner-linux-x64-$RUNNER_VERSION.tar.gz"
    echo "==> downloading runner $RUNNER_VERSION"
    curl -fL --retry 3 -o "$tarball" \
        "https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/$tarball"
    tar xzf "$tarball" && rm -f "$tarball"
    echo "==> registering as '$RUNNER_NAME' with label 'gpu'"
    # The registration token is short-lived and minted per call, so it is never
    # written to disk or to a shell history entry.
    ./config.sh --url "https://github.com/$REPO" \
        --token "$(gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token)" \
        --labels gpu --name "$RUNNER_NAME" --unattended --replace
    echo "==> configured. Nothing is running yet; use: $0 run"
}

# The gpu job is gated on this variable as well as the runner label, so that a
# repository with no GPU machine skips the job instead of queueing a job that
# can never be picked up.
ensure_enabled() {
    if gh api "repos/$REPO/actions/variables/GGRUN_GPU_RUNNER" --jq .value 2>/dev/null | grep -qx true; then
        return 0
    fi
    echo "==> setting GGRUN_GPU_RUNNER=true"
    gh api -X PATCH "repos/$REPO/actions/variables/GGRUN_GPU_RUNNER" -f value=true >/dev/null 2>&1 ||
        gh api -X POST "repos/$REPO/actions/variables" -f name=GGRUN_GPU_RUNNER -f value=true >/dev/null
}

cmd_run() {
    need gh; need nvidia-smi
    [[ $EUID -ne 0 ]] || die "do not run the runner as root"
    [[ -x "$RUNNER_DIR/run.sh" ]] || die "runner not configured; run: $0 setup"

    local free; free="$(free_vram_mb)"
    echo "==> free VRAM across all cards: ${free} MiB"
    if [[ "$free" -lt "$MIN_FREE_VRAM_MB" ]]; then
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
        die "only ${free} MiB free, need ${MIN_FREE_VRAM_MB}; stop a loaded model first"
    fi

    ensure_enabled

    echo "==> starting runner"
    "$RUNNER_DIR/run.sh" >"$RUNNER_DIR/run.log" 2>&1 &
    local runner_pid=$!
    # Take the runner down however this script exits, including Ctrl+C.
    trap 'echo "==> stopping runner"; kill "$runner_pid" 2>/dev/null || true; wait "$runner_pid" 2>/dev/null || true' EXIT

    for _ in $(seq 1 40); do
        [[ "$(runner_status)" == "online" ]] && break
        sleep 2
    done
    [[ "$(runner_status)" == "online" ]] || {
        tail -20 "$RUNNER_DIR/run.log"
        die "runner did not come online"
    }
    echo "==> runner online"

    # Record where the run list stood, so the dispatched run is identified by
    # being new rather than by a timestamp this script would have to trust.
    #
    # Filter by event here, exactly as the lookup below does. Without the
    # filter this recorded the newest run of ANY event -- usually a push -- so
    # the newest *dispatch* run was already different from it and the loop
    # accepted a stale run on its first try. It then watched a run that had
    # finished over an hour earlier and reported that old failure as this one.
    local before; before="$(gh api "repos/$REPO/actions/workflows/$WORKFLOW/runs?per_page=1&event=workflow_dispatch" --jq '.workflow_runs[0].id // 0')"
    echo "==> dispatching $WORKFLOW on main"
    # Optional workflow fields, e.g. run -f gpu_model_repo=... -f gpu_model_quant=...
    gh workflow run "$WORKFLOW" --repo "$REPO" --ref main "$@"

    local run_id=""
    for _ in $(seq 1 30); do
        run_id="$(gh api "repos/$REPO/actions/workflows/$WORKFLOW/runs?per_page=1&event=workflow_dispatch" --jq '.workflow_runs[0].id // 0')"
        [[ -n "$run_id" && "$run_id" != "0" && "$run_id" != "$before" ]] && break
        run_id=""
        sleep 2
    done
    [[ -n "$run_id" ]] || die "dispatched run did not appear"
    echo "==> run $run_id: https://github.com/$REPO/actions/runs/$run_id"

    local status=""
    while :; do
        status="$(gh api "repos/$REPO/actions/runs/$run_id" --jq .status)"
        [[ "$status" == "completed" ]] && break
        gh api "repos/$REPO/actions/runs/$run_id/jobs" \
            --jq '.jobs[] | select(.name=="gpu") | "    gpu: \(.status) \(.conclusion // "")"' 2>/dev/null || true
        sleep 20
    done

    echo "==> finished"
    gh api "repos/$REPO/actions/runs/$run_id/jobs" --jq '.jobs[] | "  \(.name): \(.conclusion // .status)"'
    local gpu_result
    gpu_result="$(gh api "repos/$REPO/actions/runs/$run_id/jobs" --jq '.jobs[] | select(.name=="gpu") | .conclusion')"
    case "$gpu_result" in
        success) echo "==> the GPU job passed" ;;
        skipped) die "the gpu job was SKIPPED: check GGRUN_GPU_RUNNER and that the workflow is on main" ;;
        "")      die "no gpu job in this run; is the workflow on main?" ;;
        *)       die "the gpu job did not pass: $gpu_result" ;;
    esac
}

cmd_status() {
    need gh
    echo "configured locally: $([[ -x "$RUNNER_DIR/run.sh" ]] && echo "yes ($RUNNER_DIR)" || echo no)"
    echo "registered runners:"
    gh api "repos/$REPO/actions/runners" \
        --jq '.runners[] | "  \(.name)  \(.status)  labels=\([.labels[].name] | join(","))"' 2>/dev/null ||
        echo "  (none)"
    # gh prints the API error body on stdout for a 404, so an unset variable
    # would otherwise be reported as a wall of JSON.
    local enabled
    enabled="$(gh api "repos/$REPO/actions/variables/GGRUN_GPU_RUNNER" --jq .value 2>/dev/null || true)"
    case "$enabled" in *'"message"'* | "") enabled="(unset)" ;; esac
    echo "GGRUN_GPU_RUNNER: $enabled"
    command -v nvidia-smi >/dev/null 2>&1 && echo "free VRAM: $(free_vram_mb) MiB"
}

case "${1:-}" in
    setup)  cmd_setup ;;
    run)    shift; cmd_run "$@" ;;
    status) cmd_status ;;
    *)      sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
