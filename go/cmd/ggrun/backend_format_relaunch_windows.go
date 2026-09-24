//go:build windows

package main

import "errors"

// relaunchAfterBackendUpdate cannot replace the process on Windows; the user
// starts the same command again on the updated backend.
func relaunchAfterBackendUpdate() error {
	return errors.New("backend updated; start the same ggrun command again")
}

// relaunchLaunch cannot replace the process on Windows; the launch fails as
// before and the user may start it again.
func relaunchLaunch(launchArgs []string, marker string) error {
	return errors.New("re-planning needs a new process on Windows; start the same command again")
}
