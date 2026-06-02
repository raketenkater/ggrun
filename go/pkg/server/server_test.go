package server

import (
	"testing"
	"time"
)

func TestProcessIsRunning(t *testing.T) {
	// Start a dummy HTTP server to test readiness logic
	// We can't easily test subprocess here, but we can test the struct
	p := &Process{Port: 99999}
	if p.IsRunning() {
		t.Fatalf("expected not running for nil process")
	}
}

func TestWaitReadyTimeout(t *testing.T) {
	p := &Process{Port: 59999} // no server here
	err := p.waitReady(100 * time.Millisecond)
	if err == nil {
		t.Fatalf("expected timeout")
	}
}
