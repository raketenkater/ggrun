//go:build windows

package server

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

const stillActiveExitCode = 259

// setSysProcAttr configures the child process on Windows.
func setSysProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP,
	}
}

// bindToParentLifetime puts the started backend in a job object that kills
// every process in it when its last handle closes. Only ggrun holds that
// handle, so when ggrun exits for any reason (Ctrl+C during model load
// included) Windows ends llama-server too; CREATE_NEW_PROCESS_GROUP keeps the
// console's Ctrl+C from reaching the backend directly. The returned release
// closes the handle and so kills whatever is still in the job: call it only
// once the backend has exited or must be torn down.
func bindToParentLifetime(cmd *exec.Cmd) (func(), error) {
	if cmd == nil || cmd.Process == nil {
		return nil, fmt.Errorf("no started process")
	}
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return nil, fmt.Errorf("create job object: %w", err)
	}
	var info windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
	info.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	if _, err := windows.SetInformationJobObject(job, windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info))); err != nil {
		_ = windows.CloseHandle(job)
		return nil, fmt.Errorf("set kill-on-close: %w", err)
	}
	proc, err := windows.OpenProcess(windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE, false, uint32(cmd.Process.Pid))
	if err != nil {
		_ = windows.CloseHandle(job)
		return nil, fmt.Errorf("open backend process: %w", err)
	}
	defer windows.CloseHandle(proc)
	if err := windows.AssignProcessToJobObject(job, proc); err != nil {
		_ = windows.CloseHandle(job)
		return nil, fmt.Errorf("assign backend to job object: %w", err)
	}
	var once sync.Once
	return func() { once.Do(func() { _ = windows.CloseHandle(job) }) }, nil
}

// killProcessTree terminates a process tree on Windows.
func killProcessTree(pid int) {
	_ = exec.Command("taskkill", "/PID", strconv.Itoa(pid), "/T").Run()
	time.Sleep(2 * time.Second)
	if !isProcessAlive(pid) {
		return
	}
	_ = exec.Command("taskkill", "/PID", strconv.Itoa(pid), "/T", "/F").Run()
	if !isProcessAlive(pid) {
		return
	}
	p, err := os.FindProcess(pid)
	if err != nil {
		return
	}
	_ = p.Kill()
}

// isProcessAlive checks if a Windows process is still running.
func isProcessAlive(pid int) bool {
	handle, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return false
	}
	defer windows.CloseHandle(handle)
	var code uint32
	if err := windows.GetExitCodeProcess(handle, &code); err != nil {
		return false
	}
	return code == stillActiveExitCode
}
