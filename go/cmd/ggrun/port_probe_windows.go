//go:build windows

package main

import "net"

// listenLikeBackend binds as the backend will. Go does not set SO_REUSEADDR
// on Windows, so the default listener already matches.
func listenLikeBackend(addr string) (net.Listener, error) {
	return net.Listen("tcp", addr)
}
