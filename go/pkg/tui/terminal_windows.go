//go:build windows

package tui

import (
	"os"

	"github.com/charmbracelet/x/term"
)

// terminalAvailable reports whether bubbletea has a terminal to drive the
// interactive UI on, on Windows.
//
// The Unix implementation falls back to opening /dev/tty when stdin is not a
// terminal. Windows has no equivalent path: the console is reached through the
// CONIN$ / CONOUT$ device names, and bubbletea's openInputTTY is a no-op there
// (tty_windows.go returns a nil file rather than reopening the console), so a
// non-console stdin genuinely cannot be rescued. Reporting the console handles
// is therefore the honest check: a real console works, a redirected pipe or
// NUL does not and the caller prints the subcommand guidance.
func terminalAvailable() bool {
	return term.IsTerminal(os.Stdin.Fd()) || term.IsTerminal(os.Stdout.Fd())
}
