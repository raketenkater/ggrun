//go:build !linux

package server

import "fmt"
import "time"

func scopedCommandArgs(args []string, memoryMaxMB int) ([]string, error) {
	return scopedCommandArgsWithUnit(args, memoryMaxMB, "")
}

func scopedCommandArgsWithUnit(args []string, memoryMaxMB int, _ string) ([]string, error) {
	if len(args) == 0 {
		return nil, fmt.Errorf("start server: empty argv")
	}
	if memoryMaxMB > 0 {
		return nil, fmt.Errorf("backend memory containment is only implemented on Linux/systemd")
	}
	return args, nil
}

func scopedCommandArgsWithLimits(args []string, memoryHighMB, memoryMaxMB int, unit string) ([]string, scopeMode, error) {
	_ = memoryHighMB
	out, err := scopedCommandArgsWithUnit(args, memoryMaxMB, unit)
	return out, scopeUnsupported, err
}

func stopScopeUnit(string, scopeMode) error { return nil }

func resetFailedScopeUnit(string, scopeMode) error { return nil }

func scopeUnitActive(string, scopeMode) (bool, error) { return false, nil }

func scopeControlGroup(string, scopeMode) (string, error) {
	return "", fmt.Errorf("backend memory scopes are only implemented on Linux/systemd")
}

func waitScopeUnitStopped(string, scopeMode, time.Duration) error { return nil }

func scopeMemoryPeakBytes(string, scopeMode) (uint64, error) {
	return 0, fmt.Errorf("backend memory scopes are only implemented on Linux/systemd")
}

func scopeMemoryOOMKillCount(string, scopeMode) (uint64, error) {
	return 0, fmt.Errorf("backend memory scopes are only implemented on Linux/systemd")
}

func scopeMemoryStats(string, scopeMode) (uint64, uint64, error, error) {
	err := fmt.Errorf("backend memory scopes are only implemented on Linux/systemd")
	return 0, 0, err, err
}

func scopeNonReclaimableMB(string) (int, error) {
	return 0, fmt.Errorf("backend memory scopes are only implemented on Linux/systemd")
}

func setScopeMemoryMaxMB(string, scopeMode, int) error {
	return fmt.Errorf("backend memory scopes are only implemented on Linux/systemd")
}

func setScopeMemoryHighMB(string, scopeMode, int) error {
	return fmt.Errorf("backend memory scopes are only implemented on Linux/systemd")
}

func (p *Process) ScopeNonReclaimableMB() (int, error) {
	return 0, fmt.Errorf("backend memory scopes are only implemented on Linux/systemd")
}

func (p *Process) SetMemoryMaxMB(int) error {
	return fmt.Errorf("backend memory scopes are only implemented on Linux/systemd")
}

func (p *Process) SetMemoryHighMB(int) error {
	return fmt.Errorf("backend memory scopes are only implemented on Linux/systemd")
}
