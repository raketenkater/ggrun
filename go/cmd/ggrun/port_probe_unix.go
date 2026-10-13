//go:build !windows

package main

import (
	"context"
	"errors"
	"net"
	"syscall"
	"time"
)

// predecessorHoldsPort reports whether a server still accepts connections on
// addr. Only a refusal proves nothing listens: a timeout may be a listener
// whose backlog is full.
func predecessorHoldsPort(addr string) bool {
	conn, err := net.DialTimeout("tcp", addr, 500*time.Millisecond)
	if err == nil {
		_ = conn.Close()
		return true
	}
	return !errors.Is(err, syscall.ECONNREFUSED)
}

// backendPortBusy returns why a backend binding addr with style would fail
// now, or nil. A live listener is busy whatever the bind says: SO_REUSEPORT
// would let the backend share a running server's port.
func backendPortBusy(addr string, style portBindStyle) error {
	if predecessorHoldsPort(addr) {
		return errors.New("a server is listening")
	}
	ln, err := listenLikeBackend(addr, style)
	if err != nil {
		return err
	}
	return ln.Close()
}

// listenLikeBackend binds with the backend's reuse option. Go's default
// listener sets SO_REUSEADDR, so a port whose previous server's connections
// sit in TIME_WAIT looked free to ggrun while the backend's own bind failed
// after a full model load.
func listenLikeBackend(addr string, style portBindStyle) (net.Listener, error) {
	lc := net.ListenConfig{Control: func(_, _ string, c syscall.RawConn) error {
		var serr error
		if err := c.Control(func(fd uintptr) {
			serr = setBackendReuseOptions(int(fd), style)
		}); err != nil {
			return err
		}
		return serr
	}}
	return lc.Listen(context.Background(), "tcp", addr)
}
