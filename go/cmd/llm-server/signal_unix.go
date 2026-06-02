//go:build !windows

package main

import (
	"os"
	"syscall"
)

// shutdownSignals returns the Unix signals for graceful shutdown.
func shutdownSignals() []os.Signal {
	return []os.Signal{syscall.SIGINT, syscall.SIGTERM}
}
