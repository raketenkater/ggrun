package placement

import "testing"

func moeBase(ctx, ncpumoe, parallel int) *Strategy {
	return &Strategy{ContextSize: ctx, NCPUMoE: ncpumoe, Parallel: parallel,
		Type: MoEOffload, KVPlacement: "gpu"}
}

// On a host-offloaded MoE, context and expert residency compete for the same
// VRAM and the planner never weighs them against each other. Measured on
// GLM-5.3-Flash: reclaiming 3,893 MiB moved residency not at all (42-43 of 48
// layers stayed on host) while the window grew 24.5%.
func TestContextCandidatesOfferedOnlyWhereTheTradeExists(t *testing.T) {
	auto := Options{AutoContextMax: 1048576}

	got := calibrationContextNeighbors(moeBase(400000, 42, 1), auto)
	if len(got) == 0 {
		t.Fatal("no context candidates for a host-offloaded MoE")
	}
	for _, c := range got {
		if c >= 400000 {
			t.Errorf("candidate %d is not smaller than the base window", c)
		}
		if c < 200000 {
			t.Errorf("candidate %d falls below half the base window", c)
		}
	}

	// A resident model has no experts on the host, so there is nothing to trade.
	if n := calibrationContextNeighbors(&Strategy{ContextSize: 400000, Parallel: 1,
		Type: MoEOffload, NCPUMoE: 0}, auto); len(n) != 0 {
		t.Errorf("offered %v candidates with no experts on the host", n)
	}
	// Not an offloaded MoE at all.
	if n := calibrationContextNeighbors(&Strategy{ContextSize: 400000, Parallel: 1,
		NCPUMoE: 42}, auto); len(n) != 0 {
		t.Errorf("offered %v candidates for a non-MoEOffload base", n)
	}
}

// Per-agent context is a protected quantity: an explicit --ctx-size is a user
// constraint, not a coordinate the search may move.
func TestExplicitContextIsNeverSearched(t *testing.T) {
	// An explicit --ctx-size arrives as a non-zero opts.ContextSize.
	explicit := Options{ContextSize: 400000, AutoContextMax: 1048576}
	if n := calibrationContextNeighbors(moeBase(400000, 42, 1), explicit); len(n) != 0 {
		t.Errorf("an explicit-context request was searched: %v", n)
	}
}

// The two serving modes signal "automatic" differently and both must be
// honoured. Claude Code carries a ceiling in AutoContextMax while ContextSize
// stays 0; plain serving resolves the window itself and marks the strategy
// ContextAuto. Gating on AutoContextMax alone excluded every plain launch, which
// is where GLM-5.3-Flash was measured — the candidates silently never appeared.
func TestBothAutomaticContextSignalsAreHonoured(t *testing.T) {
	claudeCode := Options{AutoContextMax: 1048576}
	base := moeBase(400000, 42, 1)
	base.ContextAuto = false
	if n := calibrationContextNeighbors(base, claudeCode); len(n) == 0 {
		t.Error("Claude Code's AutoContextMax signal was not honoured")
	}

	plain := Options{}
	autoBase := moeBase(400000, 42, 1)
	autoBase.ContextAuto = true
	if n := calibrationContextNeighbors(autoBase, plain); len(n) == 0 {
		t.Error("plain serving's ContextAuto signal was not honoured")
	}

	// Neither signal means nothing is known to be automatic; do not search.
	neither := moeBase(400000, 42, 1)
	neither.ContextAuto = false
	if n := calibrationContextNeighbors(neither, Options{}); len(n) != 0 {
		t.Errorf("searched a window with no automatic signal: %v", n)
	}
}

// CTXFLOOR: at 16,384 a six-turn session truncated to five turns and looked
// fastest because it did less work. A candidate must never collapse the window.
func TestContextCandidatesRespectTheFloor(t *testing.T) {
	auto := Options{AutoContextMax: 1048576}
	// A window already at the minimum has nothing safe below it.
	small := calibrationContextNeighbors(moeBase(contextMinimum, 42, 1), auto)
	for _, c := range small {
		if c < contextMinimum {
			t.Errorf("candidate %d fell below contextMinimum %d", c, contextMinimum)
		}
	}
	// With four slots the per-slot floor scales with the slot count.
	for _, c := range calibrationContextNeighbors(moeBase(400000, 42, 4), auto) {
		if c < contextMinimum*4 {
			t.Errorf("candidate %d fell below the four-slot floor", c)
		}
	}
}
