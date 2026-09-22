//go:build windows

package tui

import (
	"context"
	"os"
	"os/exec"
	"syscall"
	"testing"
	"time"

	"github.com/charmbracelet/x/term"
	"golang.org/x/sys/windows"
)

// Use separate processes so console attachment never changes the test runner.
func TestTerminalWindowsConsoleAndDetachedInput(t *testing.T) {
	for _, tc := range []struct {
		name  string
		flags uint32
	}{
		{"console", windows.CREATE_NEW_CONSOLE},
		{"detached", windows.DETACHED_PROCESS},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestTerminalWindowsChild$")
			cmd.Env = append(os.Environ(), "GGRUN_TEST_CONSOLE="+tc.name)
			cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: tc.flags, HideWindow: true}
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("console fixture: %v\n%s", err, out)
			}
		})
	}
}

func TestTerminalWindowsChild(t *testing.T) {
	mode := os.Getenv("GGRUN_TEST_CONSOLE")
	if mode == "" {
		t.Skip("subprocess fixture")
	}
	if term.IsTerminal(os.Stdin.Fd()) {
		t.Fatal("fixture stdin must be redirected")
	}
	want := mode == "console"
	if got := terminalAvailable(); got != want {
		t.Fatalf("redirected input with %s: terminalAvailable=%v, want %v", mode, got, want)
	}
	if want {
		console, err := os.Open("CONIN$")
		if err != nil {
			t.Fatal(err)
		}
		defer console.Close()
		os.Stdin = console
		if !terminalAvailable() {
			t.Fatal("attached console input was rejected")
		}
	}
}
