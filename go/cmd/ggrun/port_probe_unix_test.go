//go:build linux

package main

import (
	"io"
	"net"
	"testing"
	"time"
)

// A port whose previous server closed its connections first holds them in
// TIME_WAIT. Go's default bind succeeds there, the backend's does not; the
// launch must see the port as busy instead of loading into a failed bind.
func TestPortProbeSeesTimeWaitLikeTheBackend(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	accepted := make(chan net.Conn, 1)
	go func() {
		c, err := ln.Accept()
		if err == nil {
			accepted <- c
		}
	}()
	client, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	server := <-accepted
	_ = server.Close() // the server closes first: its side enters TIME_WAIT
	_, _ = io.Copy(io.Discard, client)
	_ = client.Close()
	_ = ln.Close()
	time.Sleep(100 * time.Millisecond)

	if probe, err := listenLikeBackend(addr); err == nil {
		_ = probe.Close()
		t.Fatalf("backend-style bind of %s succeeded with a server-side TIME_WAIT socket", addr)
	}
	if plain, err := net.Listen("tcp", addr); err != nil {
		t.Skipf("fixture did not leave a TIME_WAIT-only port: %v", err)
	} else {
		_ = plain.Close()
	}
}
