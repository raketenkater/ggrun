package boundexec

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fakeTool writes an executable that sleeps, standing in for a hung vendor tool.
func fakeTool(t *testing.T, name string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture")
	}
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte("#!/bin/sh\nsleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestOutputReturnsTypedTimeoutWithinTheDeadline(t *testing.T) {
	tool := fakeTool(t, "nvidia-smi")
	start := time.Now()
	_, err := Output(200*time.Millisecond, tool, "--query-gpu=index")
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("hung for %s", elapsed)
	}
	var te *TimeoutError
	if !errors.As(err, &te) || te.Command != "nvidia-smi" || te.Wait != 200*time.Millisecond {
		t.Fatalf("want a typed timeout naming nvidia-smi, got %v", err)
	}
	if !IsTimeout(err) {
		t.Fatal("IsTimeout false")
	}
}

func TestOrdinaryFailureIsNotATimeout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture")
	}
	_, err := Output(5*time.Second, "sh", "-c", "exit 3")
	var exitErr *exec.ExitError
	if IsTimeout(err) || !errors.As(err, &exitErr) {
		t.Fatalf("want the exit error, got %v", err)
	}
}

func TestReportPrintsOneLinePerCommand(t *testing.T) {
	var buf bytes.Buffer
	reportTo, reported = &buf, map[string]bool{}
	t.Cleanup(func() { reportTo, reported = os.Stderr, map[string]bool{} })
	err := &TimeoutError{Command: "nvidia-smi", Wait: 10 * time.Second}
	Report(err, "GPU state unknown")
	Report(err, "GPU state unknown")
	Report(errors.New("other"), "ignored")
	if got := buf.String(); strings.Count(got, "\n") != 1 || !strings.Contains(got, "nvidia-smi did not answer within 10s; GPU state unknown") {
		t.Fatalf("report = %q", got)
	}
}
