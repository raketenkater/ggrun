//go:build linux

package server

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// scopedParentWatchScript runs inside the backend's scope. Its teardown must
// not be interruptible: ggrun's parent-death SIGTERM arrives once per exiting
// ggrun thread, and a second TERM used to kill this shell after its trap had
// reset itself, before the backend group was signalled (the backend outlived
// a stopped ggrun). The script holds no "${": systemd-run expands those.
const scopedParentWatchScript = `parent=$1
shift
cg=$(awk -F: '$1 == "0" { print $3 }' /proc/self/cgroup 2>/dev/null)
if [ -n "$cg" ] && [ -w "/sys/fs/cgroup$cg/memory.oom.group" ]; then
  printf '1\n' > "/sys/fs/cgroup$cg/memory.oom.group" 2>/dev/null || true
fi
child=
watcher=
terminate_child() {
  trap '' HUP INT TERM
  c=$child
  [ -n "$c" ] || c=$!
  if [ -n "$c" ]; then
    kill -TERM -"$c" 2>/dev/null || true
  fi
  if [ -n "$watcher" ]; then
    kill "$watcher" 2>/dev/null || true
  fi
  if [ -n "$c" ]; then
    sleep 2
    kill -KILL -"$c" 2>/dev/null || true
  fi
}
trap 'terminate_child; exit 143' HUP INT TERM
setsid "$@" &
child=$!
(
  trap '' HUP INT
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
trap '' HUP INT TERM
kill "$watcher" 2>/dev/null
kill -KILL -"$child" 2>/dev/null
wait "$watcher" 2>/dev/null
exit "$status"`

// scopeModeEnvOverride lets tests select a mode without being root and without a
// real systemd. Read only here so the override cannot leak into other decisions.
// "none" is also the operator opt-out for hosts without systemd (Docker): the
// backend runs uncontained while ggrun's pre-launch admission still applies.
const scopeModeEnvOverride = "GGRUN_SCOPE_MODE"

// scopeOptOutHint names the opt-out in every containment refusal.
const scopeOptOutHint = " On a host without systemd (for example a container), set " +
	"GGRUN_SCOPE_MODE=none to run the backend without cgroup memory containment."

var warnUncontainedOnce sync.Once

// resolveScopeMode selects an owner once, before starting a contained process.
// A directory or inherited session variable does not establish a reachable bus.
func resolveScopeMode() scopeMode {
	switch strings.TrimSpace(os.Getenv(scopeModeEnvOverride)) {
	case "user":
		return scopeUser
	case "system":
		return scopeSystem
	case "unsupported":
		return scopeUnsupported
	case "none":
		return scopeNone
	}
	return scopeModeForUID(os.Geteuid())
}

func scopeModeForUID(uid int) scopeMode {
	mode := scopeUser
	if uid == 0 {
		mode = scopeSystem
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	args := systemctlArgs(mode, "show", "--property=Version", "--value")
	out, err := exec.CommandContext(ctx, args[0], args[1:]...).Output()
	if err != nil || strings.TrimSpace(string(out)) == "" {
		return scopeUnsupported
	}
	return mode
}

// systemctlArgs builds a systemctl invocation against the instance that owns the
// scope. Every read, teardown and property write must go through this so a
// scope created in one instance is never addressed in the other.
func systemctlArgs(mode scopeMode, verb string, rest ...string) []string {
	out := []string{"systemctl"}
	if mode == scopeUser {
		out = append(out, "--user")
	}
	out = append(out, verb)
	return append(out, rest...)
}

// systemctlCmd builds the exec.Cmd for a systemctl call against the instance that
// owns the scope. Paired with systemctlArgs so argv construction and execution
// can never disagree about --user.
func systemctlCmd(mode scopeMode, verb string, rest ...string) *exec.Cmd {
	a := systemctlArgs(mode, verb, rest...)
	return exec.Command(a[0], a[1:]...)
}

func scopedCommandArgs(args []string, memoryMaxMB int) ([]string, error) {
	return scopedCommandArgsWithUnit(args, memoryMaxMB, "ggrun-test.scope")
}

func scopedCommandArgsWithUnit(args []string, memoryMaxMB int, unit string) ([]string, error) {
	out, _, err := scopedCommandArgsWithLimits(args, 0, memoryMaxMB, unit)
	return out, err
}

func scopedCommandArgsWithLimits(args []string, memoryHighMB, memoryMaxMB int, unit string) ([]string, scopeMode, error) {
	if len(args) == 0 {
		return nil, scopeUnsupported, fmt.Errorf("start server: empty argv")
	}
	if memoryMaxMB <= 0 {
		return args, scopeUnsupported, nil
	}
	// The explicit opt-out is checked before systemd-run is required, so a host
	// without systemd can launch at all. It is never chosen automatically.
	if strings.TrimSpace(os.Getenv(scopeModeEnvOverride)) == "none" {
		warnUncontainedOnce.Do(func() {
			fmt.Fprintf(os.Stderr, "[launch] WARNING: GGRUN_SCOPE_MODE=none: the backend runs WITHOUT cgroup memory "+
				"containment; the %d MiB limit is not enforced and a runtime overrun can exhaust host RAM. "+
				"Pre-launch admission still applies.\n", memoryMaxMB)
		})
		return args, scopeNone, nil
	}
	systemdRun, err := exec.LookPath("systemd-run")
	if err != nil {
		return nil, scopeUnsupported, fmt.Errorf("backend memory containment requires systemd-run: %w."+scopeOptOutHint, err)
	}
	// Refuse before building any argv when no instance can own a scope. This must
	// be an explicit ggrun-authored error rather than a raw systemd failure
	// surfacing later from cmd.Start, so the failure classifies deterministically
	// (support.go matches this phrase) instead of landing in
	// unclassified_launch_failure.
	mode := resolveScopeMode()
	if mode == scopeUnsupported {
		return nil, scopeUnsupported, fmt.Errorf(
			"backend memory containment unavailable: no reachable systemd manager. " +
				"Run in a systemd login session with a working user manager, or as root " +
				"on a host with a working system manager. " +
				"Refusing to start an uncontained backend." + scopeOptOutHint)
	}
	out := []string{
		systemdRun,
		"--scope",
		"--quiet",
	}
	if mode == scopeUser {
		out = append(out, "--user")
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
		// SIGTERM only the wrapper; its trap terminates the setsid child group
		// once. control-group also signalled the child directly, so the wrapper's
		// own TERM became llama-server's fatal \"second interrupt\".
		"-p", "KillMode=mixed",
		"--",
	)
	wrapper := []string{"/bin/sh", "-c", scopedParentWatchScript, "ggrun-scope-watch", strconv.Itoa(os.Getpid())}
	out = append(out, wrapper...)
	return append(out, args...), mode, nil
}

func stopScopeUnit(unit string, mode scopeMode) error {
	if unit == "" {
		return nil
	}
	err := systemctlCmd(mode, "stop", unit).Run()
	// A transient scope may disappear between the activity check and stop.
	// systemctl returns exit 5 for that already-stopped state; teardown has
	// nevertheless achieved its only required outcome.
	if err != nil {
		if active, stateErr := scopeUnitActive(unit, mode); stateErr == nil && !active {
			return nil
		}
	}
	return err
}

func resetFailedScopeUnit(unit string, mode scopeMode) error {
	if unit == "" {
		return nil
	}
	return systemctlCmd(mode, "reset-failed", unit).Run()
}

func waitScopeUnitStopped(unit string, mode scopeMode, timeout time.Duration) error {
	if unit == "" {
		return nil
	}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		active, err := scopeUnitActive(unit, mode)
		if err != nil {
			return err
		}
		if !active {
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}
	return fmt.Errorf("systemd scope %s did not stop within %s", unit, timeout)
}

func scopeUnitActive(unit string, mode scopeMode) (bool, error) {
	err := systemctlCmd(mode, "is-active", "--quiet", unit).Run()
	if err == nil {
		return true, nil
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) && (exitErr.ExitCode() == 3 || exitErr.ExitCode() == 4) {
		return false, nil // inactive or absent unit
	}
	return false, fmt.Errorf("read scope %s activity: %w", unit, err)
}

func scopeMemoryStats(unit string, mode scopeMode) (uint64, uint64, error, error) {
	cgroup, err := scopeControlGroup(unit, mode)
	if err != nil {
		return scopeUnitMemoryStats(unit, mode)
	}
	data, err := os.ReadFile("/sys/fs/cgroup" + cgroup + "/memory.peak")
	if err != nil {
		return scopeUnitMemoryStats(unit, mode)
	}
	peak, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		return 0, 0, fmt.Errorf("parse scope memory.peak: %w", err), nil
	}
	oomKills, oomErr := scopeMemoryOOMKillCountAt(cgroup)
	if oomErr != nil {
		return scopeUnitMemoryStats(unit, mode)
	}
	return peak, oomKills, nil, oomErr
}

func scopeUnitMemoryStats(unit string, mode scopeMode) (uint64, uint64, error, error) {
	peakOut, peakErr := systemctlCmd(mode, "show", "--property=MemoryPeak", "--value", unit).Output()
	peak := uint64(0)
	if peakErr == nil {
		peak, peakErr = strconv.ParseUint(strings.TrimSpace(string(peakOut)), 10, 64)
	}
	if peakErr != nil {
		peakErr = fmt.Errorf("read scope MemoryPeak property: %w", peakErr)
	}
	resultOut, resultErr := systemctlCmd(mode, "show", "--property=Result", "--value", unit).Output()
	oomKills := uint64(0)
	if resultErr == nil && strings.TrimSpace(string(resultOut)) == "oom-kill" {
		oomKills = 1
	}
	if resultErr != nil {
		resultErr = fmt.Errorf("read scope Result property: %w", resultErr)
	}
	return peak, oomKills, peakErr, resultErr
}

func scopeControlGroup(unit string, mode scopeMode) (string, error) {
	if unit == "" {
		return "", fmt.Errorf("empty systemd scope unit")
	}
	out, err := systemctlCmd(mode, "show", "--property=ControlGroup", "--value", unit).Output()
	if err != nil {
		return "", fmt.Errorf("read scope control group: %w", err)
	}
	cgroup := strings.TrimSpace(string(out))
	if cgroup == "" {
		return "", fmt.Errorf("scope %s has no control group", unit)
	}
	return cgroup, nil
}

// ScopeNonReclaimableMB reports the backend's current non-reclaimable host
// footprint (anon + shmem + slab) from its own cgroup's memory.stat. This is
// the measured post-launch footprint Fix B sizes the running scope's MemoryMax
// against: model/server anon RSS, CPU-expert CUDA host buffers (shmem), KV,
// context checkpoints, and prompt cache. Page cache is excluded as reclaimable.
func (p *Process) ScopeNonReclaimableMB() (int, error) {
	if p == nil || p.scopeUnit == "" {
		return 0, fmt.Errorf("backend has no memory scope")
	}
	cgroup, err := scopeControlGroup(p.scopeUnit, p.scopeMode)
	if err != nil {
		return 0, err
	}
	return scopeNonReclaimableMB(cgroup)
}

// SetMemoryMaxMB raises the running backend scope's hard ceiling to memoryMaxMB
// MiB. It is the post-launch half of measured-footprint containment: once the
// backend is healthy and the canary has run, ggrun re-sizes the scope to the
// real measured footprint plus headroom instead of the pre-launch plan estimate.
// The clamp to the whole-host ceiling is the caller's responsibility.
func (p *Process) SetMemoryMaxMB(memoryMaxMB int) error {
	if p == nil || p.scopeUnit == "" {
		return fmt.Errorf("backend has no memory scope")
	}
	if err := setScopeMemoryMaxMB(p.scopeUnit, p.scopeMode, memoryMaxMB); err != nil {
		return err
	}
	p.recordMemoryLimits(0, memoryMaxMB)
	return nil
}

// SetMemoryHighMB updates the running backend scope's reclaim/throttle
// boundary without lowering its hard ceiling. Mmap-backed launches use this
// after the first real canary. The kernel throttles and reclaims every charged
// page above memory.high, clean mapped model pages included, so the caller
// must leave room for the mapped working set, not only the anonymous state.
func (p *Process) SetMemoryHighMB(memoryHighMB int) error {
	if p == nil || p.scopeUnit == "" {
		return fmt.Errorf("backend has no memory scope")
	}
	if err := setScopeMemoryHighMB(p.scopeUnit, p.scopeMode, memoryHighMB); err != nil {
		return err
	}
	p.recordMemoryLimits(memoryHighMB, 0)
	return nil
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

// scopeNonReclaimableMB reports the scope's non-reclaimable host memory:
// anon + shmem + slab from memory.stat. This is exactly what the backend's
// resident footprint is -- model/server anonymous RSS, CUDA host buffers for
// CPU experts (shmem), KV, context checkpoints, and prompt cache. Page cache
// (file-backed experts) is deliberately excluded: the kernel can evict it
// under pressure, so no plan or limit should reserve it.
func scopeNonReclaimableMB(cgroup string) (int, error) {
	if cgroup == "" {
		return 0, fmt.Errorf("empty scope control group")
	}
	statData, err := os.ReadFile("/sys/fs/cgroup" + cgroup + "/memory.stat")
	if err != nil {
		return 0, fmt.Errorf("read scope memory.stat: %w", err)
	}
	var totalBytes int64
	for _, line := range strings.Split(string(statData), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			continue
		}
		switch fields[0] {
		case "anon", "shmem", "slab":
			if v, convErr := strconv.ParseInt(fields[1], 10, 64); convErr == nil && v > 0 {
				totalBytes += v
			}
		}
	}
	return int(totalBytes / (1024 * 1024)), nil
}

// setScopeMemoryMaxMB raises the running scope's hard memory ceiling by writing
// the cgroup's memory.max directly. The user owns the scope, so a direct write
// is allowed and avoids systemd's transient-unit set-property quirks; a
// systemctl --user set-property fallback keeps DBus in the loop when the
// direct write is not permitted.
func setScopeMemoryMaxMB(unit string, mode scopeMode, memoryMaxMB int) error {
	return setScopeMemoryLimitMB(unit, mode, "memory.max", "MemoryMax", memoryMaxMB)
}

func setScopeMemoryHighMB(unit string, mode scopeMode, memoryHighMB int) error {
	return setScopeMemoryLimitMB(unit, mode, "memory.high", "MemoryHigh", memoryHighMB)
}

func setScopeMemoryLimitMB(unit string, mode scopeMode, cgroupFile, systemdProperty string, limitMB int) error {
	if unit == "" {
		return fmt.Errorf("empty scope unit")
	}
	if limitMB <= 0 {
		return fmt.Errorf("invalid %s %d MiB", cgroupFile, limitMB)
	}
	cgroup, err := scopeControlGroup(unit, mode)
	if err != nil {
		return err
	}
	bytes := uint64(limitMB) * 1024 * 1024
	path := "/sys/fs/cgroup" + cgroup + "/" + cgroupFile
	if err := os.WriteFile(path, []byte(strconv.FormatUint(bytes, 10)), 0o644); err == nil {
		return nil
	}
	// Fallback: systemctl set-property on the transient unit. This goes through
	// DBus and may fail for a scope that was never fully registered, which the
	// caller treats as a non-fatal signal.
	if systemctlCmd(mode, "set-property", unit, systemdProperty, fmt.Sprintf("%d", bytes)).Run() == nil {
		return nil
	}
	return fmt.Errorf("set scope %s at %s", cgroupFile, path)
}

// cgroupFSRoot and scopeControlGroupFn are swapped in tests.
var (
	cgroupFSRoot        = "/sys/fs/cgroup"
	scopeControlGroupFn = scopeControlGroup
	scopeLimitWait      = 5 * time.Second
)

// errScopeLimitNotInForce means the backend runs in a scope whose hard memory
// limit is not the one ggrun asked for.
var errScopeLimitNotInForce = errors.New("backend memory limit is not in force")

// verifyScopeMemoryLimit reads the scope's memory.max back after start. The
// containment promise rests on it: systemd can accept MemoryMax= and still
// leave memory.max at "max" (no memory controller delegated to the user
// manager, a cgroup v1 host). A scope that cannot be found within the wait is
// left to the normal readiness path, which reports an exited backend.
func verifyScopeMemoryLimit(unit string, mode scopeMode, wantMB int) error {
	if unit == "" || wantMB <= 0 {
		return nil
	}
	deadline := time.Now().Add(scopeLimitWait)
	for {
		cgroup, err := scopeControlGroupFn(unit, mode)
		if err == nil {
			data, readErr := os.ReadFile(cgroupFSRoot + cgroup + "/memory.max")
			if readErr == nil {
				return checkMemoryMax(strings.TrimSpace(string(data)), wantMB)
			}
		}
		if time.Now().After(deadline) {
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// checkMemoryMax accepts the requested limit within page rounding, or a
// stricter one; "max" or a looser limit is not in force.
func checkMemoryMax(value string, wantMB int) error {
	if value == "max" {
		return fmt.Errorf("%w: memory.max is \"max\", wanted %d MiB (is the memory controller delegated to this systemd manager?)", errScopeLimitNotInForce, wantMB)
	}
	got, err := strconv.ParseUint(value, 10, 64)
	if err != nil {
		return fmt.Errorf("%w: unreadable memory.max %q", errScopeLimitNotInForce, value)
	}
	want := uint64(wantMB) << 20
	const pageSlack = 64 << 10
	if got > want+pageSlack {
		return fmt.Errorf("%w: memory.max is %d MiB, wanted %d MiB", errScopeLimitNotInForce, got>>20, wantMB)
	}
	return nil
}

func scopeOptOutHintText() string { return scopeOptOutHint }
