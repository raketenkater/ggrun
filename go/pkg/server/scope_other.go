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

func scopedCommandArgsWithLimits(args []string, memoryHighMB, memoryMaxMB int, unit string) ([]string, error) {
	_ = memoryHighMB
	return scopedCommandArgsWithUnit(args, memoryMaxMB, unit)
}

func stopScopeUnit(string) error { return nil }

func resetFailedScopeUnit(string) error { return nil }

func waitScopeUnitStopped(string, time.Duration) error { return nil }

func scopeMemoryPeakBytes(string) (uint64, error) {
	return 0, fmt.Errorf("backend memory scopes are only implemented on Linux/systemd")
}

func scopeMemoryOOMKillCount(string) (uint64, error) {
	return 0, fmt.Errorf("backend memory scopes are only implemented on Linux/systemd")
}

func scopeMemoryStats(string) (uint64, uint64, error, error) {
	err := fmt.Errorf("backend memory scopes are only implemented on Linux/systemd")
	return 0, 0, err, err
}
