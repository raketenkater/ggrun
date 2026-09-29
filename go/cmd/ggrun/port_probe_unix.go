//go:build !windows

package main

import (
	"context"
	"net"
	"syscall"
)

// listenLikeBackend binds the way llama.cpp-family servers do, without
// SO_REUSEADDR. Go's default listener sets it, so a port whose previous
// server's connections sit in TIME_WAIT looked free to ggrun while the
// backend's own bind failed after a full model load.
func listenLikeBackend(addr string) (net.Listener, error) {
	lc := net.ListenConfig{Control: func(_, _ string, c syscall.RawConn) error {
		var serr error
		if err := c.Control(func(fd uintptr) {
			serr = syscall.SetsockoptInt(int(fd), syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 0)
		}); err != nil {
			return err
		}
		return serr
	}}
	return lc.Listen(context.Background(), "tcp", addr)
}
