//go:build !windows

package recovery

import "syscall"

func setProcessGroupAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Setpgid: true}
}

func killProcGroup(pid int) {
	_ = syscall.Kill(-pid, syscall.SIGTERM)
	_ = syscall.Kill(-pid, syscall.SIGKILL)
}

func procAlive(pid int) bool {
	return syscall.Kill(pid, syscall.Signal(0)) == nil
}
