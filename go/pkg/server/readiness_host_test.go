package server

import (
	"net"
	"net/http"
	"runtime"
	"testing"
	"time"
)

func TestReadinessHostFollowsTheBackendBind(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want string
	}{
		{nil, "localhost"},
		{[]string{"--host", "0.0.0.0"}, "localhost"},
		{[]string{"--host", "::"}, "localhost"},
		{[]string{"--host", "127.0.0.1"}, "127.0.0.1"},
		{[]string{"--host", "192.168.178.97"}, "192.168.178.97"},
		{[]string{"--host=127.0.0.2"}, "127.0.0.2"},
		{[]string{"--host", "fd00::5"}, "[fd00::5]"},
	} {
		if got := readinessHost(tc.args); got != tc.want {
			t.Errorf("readinessHost(%v) = %q, want %q", tc.args, got, tc.want)
		}
	}
}

// A backend bound to one specific address must be seen as ready. Polling
// localhost regardless of --host reported a fully loaded server as never ready.
func TestWaitReadyPollsASpecificBindAddress(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("127.0.0.2 is a local address only on Linux")
	}
	ln, err := net.Listen("tcp", "127.0.0.2:0")
	if err != nil {
		t.Skipf("cannot bind 127.0.0.2: %v", err)
	}
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})}
	go func() { _ = srv.Serve(ln) }()
	defer srv.Close()

	port := ln.Addr().(*net.TCPAddr).Port
	p := &Process{Port: port, host: readinessHost([]string{"--host", "127.0.0.2"}), done: make(chan struct{})}
	if err := p.waitReady(5 * time.Second); err != nil {
		t.Fatalf("a server bound to 127.0.0.2 was not seen as ready: %v", err)
	}
}
