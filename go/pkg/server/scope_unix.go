//go:build linux

package server

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const scopedParentWatchScript = `parent=$1
shift
cg=$(awk -F: '$1 == "0" { print $3 }' /proc/self/cgroup 2>/dev/null)
if [ -n "$cg" ] && [ -w "/sys/fs/cgroup${cg}/memory.oom.group" ]; then
  printf '1\n' > "/sys/fs/cgroup${cg}/memory.oom.group" 2>/dev/null || true
fi
setsid "$@" &
child=$!
watcher=
terminate_child() {
  trap - HUP INT TERM
  if [ -n "$watcher" ]; then
    kill "$watcher" 2>/dev/null || true
  fi
  kill -TERM -"$child" 2>/dev/null || true
  sleep 2
  kill -KILL -"$child" 2>/dev/null || true
}
trap 'terminate_child; exit 143' HUP INT TERM
(
  while kill -0 "$parent" 2>/dev/null; do
    sleep 1
  done
  kill -TERM -"$child" 2>/dev/null
  sleep 2
  kill -KILL -"$child" 2>/dev/null
) &
watcher=$!
wait "$child"
status=$?
trap - HUP INT TERM
kill "$watcher" 2>/dev/null
wait "$watcher" 2>/dev/null
exit "$status"`

func scopedCommandArgs(args []string, memoryMaxMB int) ([]string, error) {
	return scopedCommandArgsWithUnit(args, memoryMaxMB, "ggrun-test.scope")
}

func scopedCommandArgsWithUnit(args []string, memoryMaxMB int, unit string) ([]string, error) {
	return scopedCommandArgsWithLimits(args, 0, memoryMaxMB, unit)
}

func scopedCommandArgsWithLimits(args []string, memoryHighMB, memoryMaxMB int, unit string) ([]string, error) {
	if len(args) == 0 {
		return nil, fmt.Errorf("start server: empty argv")
	}
	if memoryMaxMB <= 0 {
		return args, nil
	}
	systemdRun, err := exec.LookPath("systemd-run")
	if err != nil {
		return nil, fmt.Errorf("backend memory containment requires systemd-run: %w", err)
	}
	out := []string{
		systemdRun,
		"--user",
		"--scope",
		"--quiet",
	}
	if unit != "" {
		// Do not use systemd-run --collect here. A failed/OOM-killed transient
		// scope can otherwise disappear before Process.Stop reads memory.peak and
		// memory.events, losing the evidence needed to classify the failure.
		out = append(out, "--unit", unit)
	}
	out = append(out,
		"-p", "MemoryAccounting=yes",
	)
	if memoryHighMB > 0 {
		if memoryHighMB > memoryMaxMB {
			memoryHighMB = memoryMaxMB
		}
		out = append(out, "-p", fmt.Sprintf("MemoryHigh=%dM", memoryHighMB))
	}
	out = append(out,
		"-p", fmt.Sprintf("MemoryMax=%dM", memoryMaxMB),
		"-p", "MemorySwapMax=0",
		"-p", "OOMPolicy=kill",
		"-p", "KillMode=control-group",
		"--",
	)
	wrapper := []string{"/bin/sh", "-c", scopedParentWatchScript, "ggrun-scope-watch", strconv.Itoa(os.Getpid())}
	out = append(out, wrapper...)
	return append(out, args...), nil
}

func stopScopeUnit(unit string) error {
	if unit == "" {
		return nil
	}
	return exec.Command("systemctl", "--user", "stop", unit).Run()
}

func resetFailedScopeUnit(unit string) error {
	if unit == "" {
		return nil
	}
	return exec.Command("systemctl", "--user", "reset-failed", unit).Run()
}

func waitScopeUnitStopped(unit string, timeout time.Duration) error {
	if unit == "" {
		return nil
	}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if !scopeUnitActive(unit) {
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}
	return fmt.Errorf("systemd scope %s did not stop within %s", unit, timeout)
}

func scopeUnitActive(unit string) bool {
	return exec.Command("systemctl", "--user", "is-active", "--quiet", unit).Run() == nil
}

func scopeMemoryPeakBytes(unit string) (uint64, error) {
	peak, _, peakErr, _ := scopeMemoryStats(unit)
	return peak, peakErr
}

func scopeMemoryStats(unit string) (uint64, uint64, error, error) {
	cgroup, err := scopeControlGroup(unit)
	if err != nil {
		return scopeUnitMemoryStats(unit)
	}
	data, err := os.ReadFile("/sys/fs/cgroup" + cgroup + "/memory.peak")
	if err != nil {
		return scopeUnitMemoryStats(unit)
	}
	peak, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		return 0, 0, fmt.Errorf("parse scope memory.peak: %w", err), nil
	}
	oomKills, oomErr := scopeMemoryOOMKillCountAt(cgroup)
	if oomErr != nil {
		return scopeUnitMemoryStats(unit)
	}
	return peak, oomKills, nil, oomErr
}

func scopeUnitMemoryStats(unit string) (uint64, uint64, error, error) {
	peakOut, peakErr := exec.Command("systemctl", "--user", "show", "--property=MemoryPeak", "--value", unit).Output()
	peak := uint64(0)
	if peakErr == nil {
		peak, peakErr = strconv.ParseUint(strings.TrimSpace(string(peakOut)), 10, 64)
	}
	if peakErr != nil {
		peakErr = fmt.Errorf("read scope MemoryPeak property: %w", peakErr)
	}
	resultOut, resultErr := exec.Command("systemctl", "--user", "show", "--property=Result", "--value", unit).Output()
	oomKills := uint64(0)
	if resultErr == nil && strings.TrimSpace(string(resultOut)) == "oom-kill" {
		oomKills = 1
	}
	if resultErr != nil {
		resultErr = fmt.Errorf("read scope Result property: %w", resultErr)
	}
	return peak, oomKills, peakErr, resultErr
}

func scopeControlGroup(unit string) (string, error) {
	if unit == "" {
		return "", fmt.Errorf("empty systemd scope unit")
	}
	out, err := exec.Command("systemctl", "--user", "show", "--property=ControlGroup", "--value", unit).Output()
	if err != nil {
		return "", fmt.Errorf("read scope control group: %w", err)
	}
	cgroup := strings.TrimSpace(string(out))
	if cgroup == "" {
		return "", fmt.Errorf("scope %s has no control group", unit)
	}
	return cgroup, nil
}

func scopeMemoryOOMKillCount(unit string) (uint64, error) {
	cgroup, err := scopeControlGroup(unit)
	if err != nil {
		return 0, err
	}
	return scopeMemoryOOMKillCountAt(cgroup)
}

func scopeMemoryOOMKillCountAt(cgroup string) (uint64, error) {
	data, err := os.ReadFile("/sys/fs/cgroup" + cgroup + "/memory.events")
	if err != nil {
		return 0, fmt.Errorf("read scope memory.events: %w", err)
	}
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 || fields[0] != "oom_kill" {
			continue
		}
		return strconv.ParseUint(fields[1], 10, 64)
	}
	return 0, nil
}
