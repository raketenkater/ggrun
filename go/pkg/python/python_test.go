package python

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// A python3 that resolves on PATH but does not run (the Windows Store alias
// exits 9009) must not be chosen over a working python.
func TestCommandSkipsInterpreterThatDoesNotRun(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses shell-script stand-ins")
	}
	dir := t.TempDir()
	write := func(name, body string) string {
		t.Helper()
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte("#!/bin/sh\n"+body), 0o755); err != nil {
			t.Fatal(err)
		}
		return p
	}
	t.Setenv("PATH", dir)
	write("python3", "exit 9009\n")
	if got, ok := Command(); ok {
		t.Fatalf("Command() = %q, want none when only a broken stub exists", got)
	}
	want := write("python", "exit 0\n")
	if got, ok := Command(); !ok || got != want {
		t.Fatalf("Command() = %q %v, want %q", got, ok, want)
	}
}
