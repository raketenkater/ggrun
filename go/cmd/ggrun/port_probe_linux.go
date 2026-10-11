//go:build linux

package main

import "golang.org/x/sys/unix"

// setBackendReuseOptions gives the probe socket the backend's reuse option.
// Linux lets a bind pass TIME_WAIT sockets only when both set SO_REUSEADDR or
// both set SO_REUSEPORT; an unknown style sets neither, so they block it.
func setBackendReuseOptions(fd int, style portBindStyle) error {
	reuseAddr := 0
	if style == bindStyleReuseAddr {
		reuseAddr = 1
	}
	if err := unix.SetsockoptInt(fd, unix.SOL_SOCKET, unix.SO_REUSEADDR, reuseAddr); err != nil {
		return err
	}
	if style != bindStyleReusePort {
		return nil
	}
	return unix.SetsockoptInt(fd, unix.SOL_SOCKET, unix.SO_REUSEPORT, 1)
}
