//go:build !windows

package recovery

import (
	"os"

	"golang.org/x/sys/unix"
)

func startupTerminalColumnsOS() int {
	ws, err := unix.IoctlGetWinsize(int(os.Stderr.Fd()), unix.TIOCGWINSZ)
	if err != nil || ws == nil || ws.Col <= 0 {
		return 0
	}
	return int(ws.Col)
}
