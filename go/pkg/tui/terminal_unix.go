//go:build !windows

package tui

import (
	"os"

	"github.com/charmbracelet/x/term"
)

// terminalAvailable reports whether bubbletea has a terminal to drive the
// interactive UI on, on Unix.
//
// It mirrors bubbletea's OWN input selection, which is the only thing that
// decides whether the program can run. In bubbletea v1.3.10 `initInput`:
//
//	if term.IsTerminal(f.Fd()) { break }   // os.Stdin is usable
//	f, err := openInputTTY()               // otherwise /dev/tty
//
// Two things about that are easy to get wrong, and an earlier version of this
// file got both:
//
//   - STDOUT IS NOT CONSULTED. Bubbletea assigns `p.output = os.Stdout`
//     unconditionally (tea.go:262) and never tests it for terminal-ness. An
//     earlier version OR-ed `IsTerminal(os.Stdout)` into this check, which has no
//     basis in bubbletea and made the function return true for a piped stdin
//     whenever the process happened to own a terminal — the test below caught it,
//     but only when run under a real pty, because `go test` outside one has
//     non-terminal stdout and the clause could not fire.
//
//   - The /dev/tty fallback IS load-bearing, for the opposite case: a piped stdin
//     in a process that still owns a controlling terminal. Bubbletea opens
//     /dev/tty there, so `echo x | ggrun` works, and refusing it would be a
//     regression. Verified under a pty: IsTerminal(stdin) false, /dev/tty opens.
//
// So: stdin first, then the platform's console device. Nothing else.
func terminalAvailable() bool {
	if term.IsTerminal(os.Stdin.Fd()) {
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
