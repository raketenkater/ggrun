//go:build linux

package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

var bindStyleNames = map[portBindStyle]string{
	bindStyleUnknown:   "unknown",
	bindStyleReuseAddr: "reuseaddr",
	bindStyleReusePort: "reuseport",
}

// backendListen opens a listener with the options the backend's own source
// sets before bind: mainline tools/server/server-http.cpp sets SO_REUSEADDR
// (b8560+); cpp-httplib's default_socket_options sets SO_REUSEPORT alone
// (ik_llama 1fddd12, httplib 0.46), so Go's SO_REUSEADDR is cleared for it.
// Written apart from setBackendReuseOptions so the probe is checked against
// the backend, not against itself.
func backendListen(t *testing.T, addr string, style portBindStyle) (net.Listener, error) {
	t.Helper()
	lc := net.ListenConfig{Control: func(_, _ string, c syscall.RawConn) error {
		var serr error
		if err := c.Control(func(fd uintptr) {
			switch style {
			case bindStyleReuseAddr:
				serr = unix.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_REUSEADDR, 1)
			case bindStyleReusePort:
				serr = unix.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_REUSEADDR, 0)
				if serr == nil {
					serr = unix.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_REUSEPORT, 1)
				}
			default:
				serr = unix.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_REUSEADDR, 0)
			}
		}); err != nil {
			return err
		}
		return serr
	}}
	ln, err := lc.Listen(context.Background(), "tcp", addr)
	if errors.Is(err, unix.ENOPROTOOPT) {
		t.Skipf("the kernel lacks the %s socket option: %v", bindStyleNames[style], err)
	}
	return ln, err
}

// tcpStates returns the kernel TCP states (hex, as in /proc/net/tcp) of the
// sockets whose local address is addr.
func tcpStates(t *testing.T, addr string) []string {
	t.Helper()
	tcp, err := net.ResolveTCPAddr("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	ip := tcp.IP.To4()
	local := fmt.Sprintf("%02X%02X%02X%02X:%04X", ip[3], ip[2], ip[1], ip[0], tcp.Port)
	data, err := os.ReadFile("/proc/net/tcp")
	if err != nil {
		t.Skipf("cannot read socket states: %v", err)
	}
	var states []string
	for _, line := range strings.Split(string(data), "\n")[1:] {
		if fields := strings.Fields(line); len(fields) > 3 && fields[1] == local {
			states = append(states, fields[3])
		}
	}
	return states
}

// timeWaitAddr leaves a 127.0.0.2 port whose only sockets are a backend-style
// server's TIME_WAIT connections, as a clean stop does: the server closes its
// connection first, then its listener.
func timeWaitAddr(t *testing.T, style portBindStyle) string {
	t.Helper()
	ln, err := backendListen(t, "127.0.0.2:0", style)
	if err != nil {
		t.Skipf("127.0.0.2 is not bindable here: %v", err)
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
	deadline := time.Now().Add(2 * time.Second)
	for {
		states := tcpStates(t, addr)
		timeWaitOnly := len(states) > 0
		for _, state := range states {
			timeWaitOnly = timeWaitOnly && state == "06" // TCP_TIME_WAIT
		}
		if timeWaitOnly {
			return addr
		}
		if time.Now().After(deadline) {
			t.Skipf("fixture did not leave a TIME_WAIT-only port: states %v", states)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func splitAddr(t *testing.T, addr string) (string, int) {
	t.Helper()
	host, portText, err := net.SplitHostPort(addr)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal(err)
	}
	return host, port
}

var allBindStyles = []portBindStyle{bindStyleUnknown, bindStyleReuseAddr, bindStyleReusePort}

// A clean stop leaves the server's accepted connections in TIME_WAIT for 60 s.
// The probe must answer as the backend's own bind would: free for the style
// the previous server used, busy for an unknown style, and for a mismatched
// style whatever the backend's bind reports.
func TestPortProbeAnswersLikeTheBackendOverTimeWait(t *testing.T) {
	for _, predecessor := range []portBindStyle{bindStyleReuseAddr, bindStyleReusePort} {
		t.Run(bindStyleNames[predecessor], func(t *testing.T) {
			addr := timeWaitAddr(t, predecessor)
			for _, style := range allBindStyles {
				probeErr := backendPortBusy(addr, style)
				ln, bindErr := backendListen(t, addr, style)
				if bindErr == nil {
					_ = ln.Close()
				}
				if (probeErr == nil) != (bindErr == nil) {
					t.Errorf("%s probe says %v, the %s backend's bind says %v", bindStyleNames[style], probeErr, bindStyleNames[style], bindErr)
				}
				if style == predecessor && probeErr != nil {
					t.Errorf("TIME_WAIT left in the %s style blocked the same style: %v", bindStyleNames[style], probeErr)
				}
				if style == bindStyleUnknown && probeErr == nil {
					t.Error("an unknown style reported a TIME_WAIT port free")
				}
			}
		})
	}
}

// A live listener blocks every style. SO_REUSEPORT would even let the backend
// bind beside it and share the port with a running server.
func TestPortProbeLiveListenerAlwaysBlocks(t *testing.T) {
	for _, holder := range []portBindStyle{bindStyleReuseAddr, bindStyleReusePort} {
		t.Run(bindStyleNames[holder], func(t *testing.T) {
			ln, err := backendListen(t, "127.0.0.2:0", holder)
			if err != nil {
				t.Skipf("127.0.0.2 is not bindable here: %v", err)
			}
			defer ln.Close()
			addr := ln.Addr().String()
			for _, style := range allBindStyles {
				if err := backendPortBusy(addr, style); err == nil {
					t.Errorf("%s probe reported a live %s listener's port free", bindStyleNames[style], bindStyleNames[holder])
				}
			}
			host, port := splitAddr(t, addr)
			var warn bytes.Buffer
			start := time.Now()
			if waitForBackendPort(host, port, holder, 300*time.Millisecond, &warn) {
				t.Fatal("the wait reported a live listener's port free")
			}
			if elapsed := time.Since(start); elapsed > 5*time.Second {
				t.Fatalf("the wait took %s against a 300ms timeout", elapsed)
			}
			if !strings.Contains(warn.String(), "not yet bindable") {
				t.Errorf("the wait must be explained: %q", warn.String())
			}
		})
	}
}

// The relaunch: a clean stop, then the same backend on the same port. Neither
// wait may outlast the TIME_WAIT the backend binds over (~55 s before); an
// unknown style still waits, within its bound.
func TestRelaunchDoesNotWaitOutTimeWaitOfTheSameStyle(t *testing.T) {
	for _, predecessor := range []portBindStyle{bindStyleReuseAddr, bindStyleReusePort} {
		t.Run(bindStyleNames[predecessor], func(t *testing.T) {
			host, port := splitAddr(t, timeWaitAddr(t, predecessor))
			var warn bytes.Buffer
			start := time.Now()
			if !waitForPredecessorPort(host, port, 5*time.Second, &warn) {
				t.Fatal("TIME_WAIT holds no VRAM, but the predecessor wait timed out on it")
			}
			if !waitForBackendPort(host, port, predecessor, 5*time.Second, &warn) {
				t.Fatal("the backend wait timed out on TIME_WAIT of its own style")
			}
			if elapsed := time.Since(start); elapsed > 2*time.Second || warn.Len() > 0 {
				t.Errorf("the relaunch waited %s: %q", elapsed, warn.String())
			}
			start = time.Now()
			if waitForBackendPort(host, port, bindStyleUnknown, 300*time.Millisecond, &warn) {
				t.Fatal("an unknown style passed TIME_WAIT")
			}
			if elapsed := time.Since(start); elapsed > 5*time.Second {
				t.Fatalf("the unknown-style wait took %s against a 300ms timeout", elapsed)
			}
		})
	}
}
