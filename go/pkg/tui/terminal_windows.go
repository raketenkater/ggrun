//go:build windows

package tui

import (
	"os"

	"github.com/charmbracelet/x/term"
)

// terminalAvailable reports whether bubbletea has a terminal to drive the
// interactive UI on, on Windows.
//
// It mirrors bubbletea's input selection exactly, the same way the Unix file
// does. Bubbletea's logic is: if os.Stdin IS a terminal use it, otherwise open
// the fallback console device and use that. On Windows that fallback is
// openInputTTY, which opens "CONIN$" (bubbletea tty_windows.go) — the direct
// analogue of /dev/tty, not a no-op.
//
// So this file does the same three-way check as the Unix one, with CONIN$ in
// place of /dev/tty. An earlier version of this file checked only the console
// handles, on the mistaken belief that bubbletea could not rescue a non-console
// stdin on Windows; that would have refused `echo x | ggrun` on Windows while
// allowing it on Unix, for no reason.
func terminalAvailable() bool {
	if term.IsTerminal(os.Stdin.Fd()) || term.IsTerminal(os.Stdout.Fd()) {
		return true
	}
	// bubbletea's Windows input fallback. CONIN$ is resolvable whenever the
	// process has a console attached, even if its std handles were redirected.
	f, err := os.Open("CONIN$")
	if err != nil {
		return false
	}
	_ = f.Close()
	return true
}
