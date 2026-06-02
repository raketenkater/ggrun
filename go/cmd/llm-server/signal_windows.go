//go:build windows

package main

import "os"

// shutdownSignals returns Windows signals for graceful shutdown.
func shutdownSignals() []os.Signal {
	return []os.Signal{os.Interrupt}
}
