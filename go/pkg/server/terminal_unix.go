//go:build !windows

package server

import (
	"os"

	"golang.org/x/sys/unix"
)

func terminalColumnsOS() int {
	ws, err := unix.IoctlGetWinsize(int(os.Stderr.Fd()), unix.TIOCGWINSZ)
	if err != nil || ws == nil || ws.Col <= 0 {
		return 0
	}
	return int(ws.Col)
}
