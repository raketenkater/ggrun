//go:build !windows

package main

import (
	"fmt"
	"os"
	"syscall"
)

// relaunchAfterBackendUpdate replaces this process with the same command so
// backend detection, placement and admission all start again on the new build.
func relaunchAfterBackendUpdate() error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	env := append(os.Environ(), backendFormatRetryEnv+"=1")
	fmt.Fprintln(os.Stderr, "[launch] backend updated; relaunching")
	return syscall.Exec(exe, os.Args, env)
}
