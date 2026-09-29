package server

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// The final MiMo probe logged 4.4 s of weight loading and then nothing for the
// rest of its 30-minute timeout. The supervisor must say that it gave up on a
// live, silent process, not leave the log's last line as the only account.
func TestStalledLoaderReportsTimeoutAndSilence(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses a POSIX shell script")
	}
	script := filepath.Join(t.TempDir(), "stalled.sh")
	if err := os.WriteFile(script, []byte("#!/bin/sh\necho 'load_tensors: loading model tensors' >&2\nexec sleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	timeout := 700 * time.Millisecond
	p, err := StartWithTimeout([]string{script}, 59997, timeout)
	if err == nil {
		_ = p.Stop()
		t.Fatal("stalled loader reported ready")
	}
	var failure *StartupFailure
	if !errors.As(err, &failure) {
		t.Fatalf("startup error is untyped: %v", err)
	}
	if failure.Outcome != "timeout" {
		t.Fatalf("outcome = %q, want timeout", failure.Outcome)
	}
	if failure.Elapsed < timeout || failure.Timeout != timeout {
		t.Fatalf("elapsed %s / timeout %s do not show the full wait", failure.Elapsed, failure.Timeout)
	}
	if failure.LastOutputAge < timeout/2 {
		t.Fatalf("last output age %s does not show the loader went silent", failure.LastOutputAge)
	}
	if p.IsRunning() {
		t.Fatal("timed-out loader was left running")
	}
}

func TestExitedLoaderIsNotReportedAsTimeout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses a POSIX shell script")
	}
	script := filepath.Join(t.TempDir(), "crash.sh")
	if err := os.WriteFile(script, []byte("#!/bin/sh\necho 'CUDA error: out of memory' >&2\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	_, err := StartWithTimeout([]string{script}, 59996, 5*time.Second)
	var failure *StartupFailure
	if !errors.As(err, &failure) || failure.Outcome != "exited" {
		t.Fatalf("crashed loader = %v (%#v)", err, failure)
	}
	if failure.Elapsed >= 5*time.Second {
		t.Fatalf("exit waited for the whole timeout: %s", failure.Elapsed)
	}
}
