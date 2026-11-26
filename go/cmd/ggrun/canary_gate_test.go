package main

import (
	"errors"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/benchmark"
	"github.com/raketenkater/ggrun/pkg/controller"
)

// Invariant: only a speed probe that timed out on a healthy backend degrades
// the profile; every real failure still rejects it and nothing unverified is
// promoted. GLM-5.3 (2026-10-26) was rejected for a probe that queued behind a
// cancelled calibration prefill although the same plan had just served three
// Claude Code tasks.
func TestCacheCanaryGate(t *testing.T) {
	cases := []struct {
		name   string
		canary *benchmark.CacheCanaryResult
		err    error
		want   controller.State
	}{
		{"healthy but too slow to measure", &benchmark.CacheCanaryResult{TooSlow: true, Reason: "endpoint answered too slowly to measure: x"}, nil, controller.StateDegraded},
		{"probe error", nil, errors.New("prefill speed probe: timeout"), controller.StateRejected},
		{"error wins over a too-slow flag", &benchmark.CacheCanaryResult{TooSlow: true}, errors.New("boom"), controller.StateRejected},
		{"no result", nil, nil, controller.StateRejected},
		{"wrong answer", &benchmark.CacheCanaryResult{Reason: "bad"}, nil, controller.StateRejected},
		{"functional", &benchmark.CacheCanaryResult{Functional: true}, nil, controller.StateFunctionalVerified},
	}
	for _, c := range cases {
		state, reason := cacheCanaryGate(c.canary, c.err)
		if state != c.want {
			t.Fatalf("%s: state %s, want %s (%s)", c.name, state, c.want, reason)
		}
		if state == controller.StateDegraded && !strings.Contains(reason, "too slowly to measure") {
			t.Fatalf("%s: degraded reason %q", c.name, reason)
		}
		if state == controller.StateRejected && !strings.HasPrefix(reason, "functional canary failed") {
			t.Fatalf("%s: rejection reason %q", c.name, reason)
		}
	}
}

// The canary's health judgement for a slow backend comes from the process,
// never from /health, which ik_llama.cpp answers from its busy scheduler.
func TestBackendStillHealthy(t *testing.T) {
	if backendStillHealthy(nil) != nil {
		t.Fatal("no backend must leave the runner's own check")
	}
	b := newFakeBackend()
	healthy := backendStillHealthy(b)
	if !healthy() {
		t.Fatal("running backend without a fatal line judged unhealthy")
	}
	b.appendLog("CUDA error: out of memory\n")
	if healthy() {
		t.Fatal("backend with a fatal line judged healthy")
	}
	stopped := newFakeBackend()
	_ = stopped.Stop()
	if backendStillHealthy(stopped)() {
		t.Fatal("exited backend judged healthy")
	}
}
