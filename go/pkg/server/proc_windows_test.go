//go:build windows

package server

import (
	"os/exec"
	"testing"
	"time"
)

// Invariant: a backend ggrun started cannot outlive ggrun's hold on it. When
// ggrun exits abruptly (Ctrl+C during model load) Windows closes the job
// handle, exactly as release does here, and the backend must die with it.
func TestBackendDiesWhenItsJobHandleCloses(t *testing.T) {
	cmd := exec.Command("ping", "-n", "120", "127.0.0.1")
	setSysProcAttr(cmd)
	if err := cmd.Start(); err != nil {
		t.Skipf("cannot start a long-lived child: %v", err)
	}
	done := make(chan struct{})
	go func() { _ = cmd.Wait(); close(done) }()
	defer func() { _ = cmd.Process.Kill(); <-done }()

	release, err := bindToParentLifetime(cmd)
	if err != nil {
		t.Fatalf("bind backend to ggrun's lifetime: %v", err)
	}
	if !isProcessAlive(cmd.Process.Pid) {
		t.Fatal("binding killed the backend")
	}
	release()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("backend survived the close of ggrun's job handle")
	}
	release() // a second release (Stop after Kill) is harmless
}
