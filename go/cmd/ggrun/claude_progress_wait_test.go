package main

import (
	"strings"
	"testing"
)

// W7: the status line shows the router's expected wait beside "N queued", and
// only when the router has an estimate (unknown stays unknown).
func TestClaudeProgressShowsExpectedWait(t *testing.T) {
	idle := formatClaudeProgress(claudeProgressState{Queued: 2, ExpectedWaitMS: 95_000})
	if !strings.Contains(idle, "2 queued (~1m35s wait)") {
		t.Fatalf("idle status %q", idle)
	}
	busy := formatClaudeProgress(claudeProgressState{Active: 1, Queued: 1, ExpectedWaitMS: 12_000,
		Requests: []claudeRequestProgress{{Slot: 0, Stage: "generating", Generated: 5}}})
	if !strings.Contains(busy, "1 queued (~12s wait)") {
		t.Fatalf("busy status %q", busy)
	}
	if s := formatClaudeProgress(claudeProgressState{Queued: 2}); !strings.Contains(s, "2 queued") || strings.Contains(s, "wait") {
		t.Fatalf("estimate invented: %q", s)
	}
}
