//go:build windows

package tui

import (
	"os"

	"github.com/charmbracelet/x/term"
)

// terminalAvailable follows Bubble Tea's input selection on Windows: use
// terminal stdin, or open CONIN$ when input is redirected. Stdout does not
// establish that an interactive input device is available.
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
