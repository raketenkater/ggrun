//go:build windows

package server

import (
	"os"
	"os/exec"
	"syscall"
	"time"
)

// setSysProcAttr configures the child process on Windows.
func setSysProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP,
	}
}

// killProcessTree terminates a process on Windows.
func killProcessTree(pid int) {
	p, err := os.FindProcess(pid)
	if err != nil {
		return
	}
	_ = p.Kill()
	time.Sleep(2 * time.Second)
	_ = p.Kill()
}

// isProcessAlive checks if a Windows process is still running.
func isProcessAlive(pid int) bool {
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return p.Signal(os.Kill) != nil // non-nil = process is dead/not found
}
