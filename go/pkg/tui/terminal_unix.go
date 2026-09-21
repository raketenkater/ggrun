//go:build !windows

package tui

import (
	"os"

	"github.com/charmbracelet/x/term"
)

// terminalAvailable reports whether bubbletea has a terminal to drive the
// interactive UI on, on Unix.
//
// It mirrors bubbletea's own input selection rather than approximating it. That
// logic (bubbletea tea.go, `standardInput` case) is:
//
//   - if os.Stdin IS a terminal, use it and stop;
//   - if os.Stdin is NOT a terminal, open /dev/tty and use that instead.
//
// Both halves matter. Checking stdin alone would refuse `echo x | ggrun` — a
// pipe into a process that still owns a controlling terminal — which bubbletea
// handles correctly today via the /dev/tty fallback. That case was verified: with
// stdin piped inside a pty, IsTerminal(stdin) is false while /dev/tty opens.
//
// Checking /dev/tty alone (the original implementation) is the mirror-image bug:
// it cannot work on Windows at all, and it rejects a detached session that still
// has a usable stdout.
func terminalAvailable() bool {
	if term.IsTerminal(os.Stdin.Fd()) || term.IsTerminal(os.Stdout.Fd()) {
		return true
	}
	// bubbletea's non-TTY-stdin fallback: if this opens, the UI can run.
	f, err := os.Open("/dev/tty")
	if err != nil {
		return false
	}
	_ = f.Close()
	return true
}
