//go:build windows

package main

import "errors"

// relaunchAfterBackendUpdate cannot replace the process on Windows; the user
// starts the same command again on the updated backend.
func relaunchAfterBackendUpdate() error {
	return errors.New("backend updated; start the same ggrun command again")
}
