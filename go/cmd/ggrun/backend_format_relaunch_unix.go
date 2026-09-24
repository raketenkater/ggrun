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

// relaunchLaunch replaces this process with `ggrun launch <args>` and the
// given marker set, so detection, placement and admission start again. Using
// the launch's own recorded argv makes a TUI launch and a CLI launch re-plan
// identically.
func relaunchLaunch(launchArgs []string, marker string) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	if len(launchArgs) == 0 {
		return fmt.Errorf("no recorded launch arguments")
	}
	argv := append([]string{exe, "launch"}, launchArgs...)
	return syscall.Exec(exe, argv, append(os.Environ(), marker+"=1"))
}
