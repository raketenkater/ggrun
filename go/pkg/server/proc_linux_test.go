//go:build linux

package server

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestSetSysProcAttrRequestsParentDeathTermination(t *testing.T) {
	cmd := exec.Command("sleep", "1")
	setSysProcAttr(cmd)
	if cmd.SysProcAttr == nil {
		t.Fatal("missing Unix process attributes")
	}
	if !cmd.SysProcAttr.Setpgid {
		t.Fatal("backend must remain isolated in its own process group")
	}
	if cmd.SysProcAttr.Pdeathsig != syscall.SIGTERM {
		t.Fatalf("Pdeathsig=%v, want SIGTERM", cmd.SysProcAttr.Pdeathsig)
	}
}

func TestParentDeathSignalTerminatesChild(t *testing.T) {
	if os.Getenv("GGRUN_PDEATH_HELPER") == "1" {
		pidFile := os.Getenv("GGRUN_PDEATH_PID_FILE")
		cmd := exec.Command("sleep", "60")
		setSysProcAttr(cmd)
		if err := cmd.Start(); err != nil {
			os.Exit(2)
		}
		if err := os.WriteFile(pidFile, []byte(strconv.Itoa(cmd.Process.Pid)), 0o600); err != nil {
			_ = cmd.Process.Kill()
			os.Exit(3)
		}
		os.Exit(0)
	}

	pidFile := filepath.Join(t.TempDir(), "child.pid")
	helper := exec.Command(os.Args[0], "-test.run=^TestParentDeathSignalTerminatesChild$")
	helper.Env = append(os.Environ(),
		"GGRUN_PDEATH_HELPER=1",
		"GGRUN_PDEATH_PID_FILE="+pidFile,
	)
	if out, err := helper.CombinedOutput(); err != nil {
		t.Fatalf("parent helper failed: %v: %s", err, out)
	}
	data, err := os.ReadFile(pidFile)
	if err != nil {
		t.Fatal(err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Kill(-pid, syscall.SIGKILL)
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if !isProcessAlive(pid) {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("child pid %d survived its ggrun parent", pid)
}
