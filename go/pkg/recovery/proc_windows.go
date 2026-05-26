//go:build windows

package recovery

import (
	"os"
	"syscall"
)

func setProcessGroupAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP}
}

func killProcGroup(pid int) {
	p, _ := os.FindProcess(pid)
	if p != nil {
		_ = p.Kill()
	}
}

func procAlive(pid int) bool {
	proc, err := os.FindProcess(pid)
	// os.FindProcess does not verify liveness on Windows, but this helper is
	// only used as a non-destructive health-loop guard. Never probe with Kill.
	return err == nil && proc != nil
}
