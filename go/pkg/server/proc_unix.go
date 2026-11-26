//go:build !windows

package server

import (
	"os/exec"
	"syscall"
	"time"
)

// setSysProcAttr configures the child process to run in its own group.
func setSysProcAttr(cmd *exec.Cmd) {
	attr := &syscall.SysProcAttr{Setpgid: true}
	setParentDeathSignal(attr)
	cmd.SysProcAttr = attr
}

// bindToParentLifetime is a no-op on Unix: the parent-death signal and the
// scope wrapper's parent watch already end the backend with ggrun.
func bindToParentLifetime(*exec.Cmd) (func(), error) { return nil, nil }

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
