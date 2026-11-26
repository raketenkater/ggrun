package main

import (
	"context"
	"fmt"
	"io"
	"time"

	"github.com/raketenkater/ggrun/pkg/claudeauto"
)

// The serving display reads only atomic counters. In particular, it must not
// enqueue HTTP health, metrics or slot tasks on a saturated model backend.
func startRemoteClaudeStatus(out io.Writer, snapshot func() claudeauto.RouterAdmissionStatus) context.CancelFunc {
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		runRemoteClaudeStatus(ctx, ticker.C, out, snapshot)
	}()
	return cancel
}

func runRemoteClaudeStatus(ctx context.Context, ticks <-chan time.Time, out io.Writer, snapshot func() claudeauto.RouterAdmissionStatus) {
	var previous claudeauto.RouterAdmissionStatus
	havePrevious := false
	show := func() bool {
		if ctx.Err() != nil {
			return false
		}
		current := snapshot()
		if havePrevious && current == previous {
			return true
		}
		previous, havePrevious = current, true
		limit := "unlimited"
		if current.Limit > 0 {
			limit = fmt.Sprint(current.Limit)
		}
		_, err := fmt.Fprintf(out, "[remote] ready | active %d | queued %d | configured main limit %s | Ctrl+C stops this server\n", current.Active, current.Queued, limit)
		return err == nil
	}
	if !show() {
		return
	}
	for {
		select {
		case <-ctx.Done():
			return
		case _, ok := <-ticks:
			if !ok || !show() {
				return
			}
		}
	}
}
