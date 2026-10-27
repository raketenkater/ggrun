package main

import (
	"context"
	"time"
)

// claudeHealthMonitor polls the backend during a --claude-code session so a
// mid-session crash is recorded at once. stop cancels it and waits for a check
// already in flight, so a stop ggrun itself requested is never reported as
// "backend died mid-session" or scanned for an OOM.
type claudeHealthMonitor struct {
	cancel context.CancelFunc
	done   chan struct{}
}

// startClaudeHealthMonitor calls onDied at most once, the first time alive
// reports the backend down, unless stop was requested meanwhile.
func startClaudeHealthMonitor(interval time.Duration, alive func() bool, onDied func()) *claudeHealthMonitor {
	ctx, cancel := context.WithCancel(context.Background())
	m := &claudeHealthMonitor{cancel: cancel, done: make(chan struct{})}
	go func() {
		defer close(m.done)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
			if alive() {
				continue
			}
			if ctx.Err() != nil {
				// The check overlapped a requested stop.
				return
			}
			onDied()
			return
		}
	}()
	return m
}

// stop must precede every requested backend stop. It is idempotent.
func (m *claudeHealthMonitor) stop() {
	if m == nil {
		return
	}
	m.cancel()
	<-m.done
}
