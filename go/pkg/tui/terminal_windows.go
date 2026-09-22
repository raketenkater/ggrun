//go:build windows

package tui

import (
	"os"

	"github.com/charmbracelet/x/term"
)

// terminalAvailable reports whether bubbletea has a terminal to drive the
// interactive UI on, on Windows.
//
// Same two-step shape as the Unix file, with the platform's own console device:
// stdin if it is a console, otherwise bubbletea's Windows fallback, which opens
// "CONIN$" (bubbletea tty_windows.go:58-64 — the direct analogue of /dev/tty, not
// a no-op, as an earlier version of this comment wrongly claimed).
//
// Stdout is deliberately not consulted, for the reason recorded in
// terminal_unix.go: bubbletea assigns it unconditionally and never tests it.
func terminalAvailable() bool {
	if term.IsTerminal(os.Stdin.Fd()) {
		return true
	}
	// bubbletea's Windows input fallback. CONIN$ resolves whenever the process
	// has a console attached, even if its std handles were redirected.
	f, err := os.Open("CONIN$")
	if err != nil {
		return false
	}
	_ = f.Close()
	return true
}
