//go:build !windows

package tui

import (
	"os"

	"github.com/charmbracelet/x/term"
)

// terminalAvailable follows Bubble Tea's input selection on Unix: use
// terminal stdin, or open /dev/tty when input is redirected. Stdout does not
// establish that an interactive input device is available.
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
