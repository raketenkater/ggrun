//go:build windows

package main

import "net"

// predecessorHoldsPort binds as the backend will. Go does not set SO_REUSEADDR
// on Windows, so the default listener already matches.
func predecessorHoldsPort(addr string) bool {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return true
	}
	_ = ln.Close()
	return false
}

// backendPortBusy adds nothing on Windows: predecessorHoldsPort already bound
// the launch address the way the backend does.
func backendPortBusy(string, portBindStyle) error {
	return nil
}
