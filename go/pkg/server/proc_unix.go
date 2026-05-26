//go:build !windows

package server

import (
	"os/exec"
	"syscall"
	"time"
)

// setSysProcAttr configures the child process to run in its own group.
func setSysProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// killProcessTree sends SIGTERM then SIGKILL to the process group.
func killProcessTree(pid int) {
	_ = syscall.Kill(-pid, syscall.SIGTERM)
	time.Sleep(2 * time.Second)
	_ = syscall.Kill(-pid, syscall.SIGKILL)
}

// isProcessAlive checks if a process is still running on Unix.
func isProcessAlive(pid int) bool {
	return syscall.Kill(pid, syscall.Signal(0)) == nil
}
