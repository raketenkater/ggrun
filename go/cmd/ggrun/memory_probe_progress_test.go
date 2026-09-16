package main

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

func ctxStrategy(n int) *placement.Strategy {
	return &placement.Strategy{ContextSize: n, ContextAuto: true}
}

// The two exhaustion modes have opposite fixes, so the report has to tell them
// apart. This is the case observed on GLM-5.3-Flash 2026-09-15: a correct
// descent, one 1,024-token granule per round, against deficits worth far more.
func TestProgressReportsTooSlowRatherThanStuck(t *testing.T) {
	p := newProbeSearchProgress(ctxStrategy(530432))
	p.record(ctxStrategy(530432), ctxStrategy(529408), 2267, 60000)
	p.record(ctxStrategy(529408), ctxStrategy(528384), 4590, 120000)

	var sb strings.Builder
	p.report(&sb)
	out := sb.String()

	if !strings.Contains(out, "too slowly") {
		t.Fatalf("a creeping search was not reported as too slow:\n%s", out)
	}
	if strings.Contains(out, "cycling") {
		t.Fatalf("a descending search was misreported as cycling:\n%s", out)
	}
	if p.moved() != 2048 {
		t.Fatalf("moved: got %d, want 2048", p.moved())
	}
}

// A search whose context never moves is cycling between shapes. That needs a
// different fix from a slow descent and must not be described as slow.
func TestProgressReportsStuckWhenContextNeverMoves(t *testing.T) {
	p := newProbeSearchProgress(ctxStrategy(500736))
	p.record(ctxStrategy(500736), ctxStrategy(500736), 523, 0)
	p.record(ctxStrategy(500736), ctxStrategy(500736), 2291, 0)

	var sb strings.Builder
	p.report(&sb)
	out := sb.String()

	if !strings.Contains(out, "cycling") {
		t.Fatalf("a non-moving search was not reported as cycling:\n%s", out)
	}
	if strings.Contains(out, "too slowly") {
		t.Fatalf("a cycling search was misreported as slow:\n%s", out)
	}
}

func TestProgressReportsEveryRoundItTook(t *testing.T) {
	p := newProbeSearchProgress(ctxStrategy(530432))
	p.record(ctxStrategy(530432), ctxStrategy(529408), 2267, 60000)
	p.record(ctxStrategy(529408), ctxStrategy(528384), 4590, 120000)
	p.record(ctxStrategy(528384), ctxStrategy(527360), 4571, 119000)

	var sb strings.Builder
	p.report(&sb)
	out := sb.String()

	for _, want := range []string{"530432", "529408", "528384", "527360", "2267 MiB", "4590 MiB", "4571 MiB"} {
		if !strings.Contains(out, want) {
			t.Fatalf("report omitted %q:\n%s", want, out)
		}
	}
}

// An exhausted budget with no rounds means the first plan never produced a
// measurement — a different failure again, and one the report must not
// misdescribe as a search problem.
func TestProgressHandlesNoRounds(t *testing.T) {
	p := newProbeSearchProgress(ctxStrategy(131072))
	var sb strings.Builder
	p.report(&sb)
	if !strings.Contains(sb.String(), "no re-plan rounds") {
		t.Fatalf("empty search not reported:\n%s", sb.String())
	}
}

// The suggestion is advice for the next invocation, so it must be below the
// deepest point the search actually reached — never above it.
func TestSuggestedContextIsBelowTheDeepestProbe(t *testing.T) {
	p := newProbeSearchProgress(ctxStrategy(530432))
	p.record(ctxStrategy(530432), ctxStrategy(529408), 2267, 60000)

	got := p.suggestedContext()
	if got <= 0 {
		t.Fatalf("no suggestion produced: %d", got)
	}
	if got >= 529408 {
		t.Fatalf("suggestion %d is not below the deepest probed context 529408", got)
	}
}

func TestProgressIsNilSafe(t *testing.T) {
	var p *probeSearchProgress
	p.record(ctxStrategy(1), ctxStrategy(2), 3, 4)
	if p.moved() != 0 || p.needed() != 0 || p.suggestedContext() != 0 {
		t.Fatal("nil progress returned non-zero values")
	}
}
