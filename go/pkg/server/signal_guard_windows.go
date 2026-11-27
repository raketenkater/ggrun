//go:build windows

package server

// Windows ties the backend to ggrun with a job object (bindToParentLifetime);
// there is no Unix shutdown signal to guard.
func installSignalGuard() {}

// OwnShutdownSignals is a no-op on Windows.
func OwnShutdownSignals() {}

func blockIfStopping() {}

func trackLive(*Process) {}

func untrackLive(*Process) {}
