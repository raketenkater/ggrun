//go:build !windows && !linux

package main

import "syscall"

// setBackendReuseOptions binds without SO_REUSEADDR whatever the style: the
// per-style behaviour is measured on Linux only and BSD reuse semantics
// differ, so other systems keep the probe any socket on the port blocks.
func setBackendReuseOptions(fd int, _ portBindStyle) error {
	return syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 0)
}
