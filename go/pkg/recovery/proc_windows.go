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
	if p != nil { _ = p.Kill() }
}

func procAlive(pid int) bool {
	p, err := os.FindProcess(pid)
	return err == nil && p != nil
}
