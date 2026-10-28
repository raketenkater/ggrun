package main

import (
	"sync/atomic"
	"testing"
	"time"
)

// Invariant: a backend stop that ggrun requested never reaches the OOM
// recorder, including when a health check is already in flight as the stop
// begins; a backend that really dies mid-session is still recorded once.
func TestClaudeHealthMonitorRequestedStopIsNotACrash(t *testing.T) {
	const tick = 5 * time.Millisecond

	t.Run("backend dies mid-session", func(t *testing.T) {
		var up atomic.Bool
		up.Store(true)
		var died atomic.Int32
		m := startClaudeHealthMonitor(tick, up.Load, func() { died.Add(1) })
		time.Sleep(4 * tick)
		up.Store(false)
		deadline := time.Now().Add(2 * time.Second)
		for died.Load() == 0 && time.Now().Before(deadline) {
			time.Sleep(tick)
		}
		m.stop()
		m.stop()
		if got := died.Load(); got != 1 {
			t.Fatalf("a real mid-session death was recorded %d times, want 1", got)
		}
	})

	t.Run("stop, then the backend goes down", func(t *testing.T) {
		var up atomic.Bool
		up.Store(true)
		var died atomic.Int32
		m := startClaudeHealthMonitor(tick, up.Load, func() { died.Add(1) })
		time.Sleep(4 * tick)
		m.stop()
		up.Store(false) // p.Stop()
		time.Sleep(20 * tick)
		if died.Load() != 0 {
			t.Fatal("a requested stop was recorded as a mid-session death")
		}
	})

	t.Run("stop while a health check is in flight", func(t *testing.T) {
		entered := make(chan struct{}, 1)
		release := make(chan bool)
		var died atomic.Int32
		m := startClaudeHealthMonitor(tick, func() bool {
			select {
			case entered <- struct{}{}:
			default:
			}
			return <-release
		}, func() { died.Add(1) })
		<-entered
		m.cancel() // the stop request lands before the check answers
		stopped := make(chan struct{})
		go func() { m.stop(); close(stopped) }()
		select {
		case <-stopped:
			t.Fatal("stop returned while a health check was still in flight")
		case <-time.After(10 * tick):
		}
		// The check overlapping the stop sees the backend going down.
		release <- false
		select {
		case <-stopped:
		case <-time.After(2 * time.Second):
			t.Fatal("stop did not return after the in-flight check finished")
		}
		if died.Load() != 0 {
			t.Fatal("a check overlapping a requested stop reached the OOM recorder")
		}
	})

	var m *claudeHealthMonitor
	m.stop() // nil monitor: no session monitor was started
}
