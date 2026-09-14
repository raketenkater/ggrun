package placement

import "testing"

// Slot candidates used to be compared on total context, which every one of them
// necessarily changes. The result was that none survived: Qwen3.8-Flash-Next in
// Claude Code mode reported "parallel 4..4" — batch, ubatch and topology
// variants, and no slot-count variant at all — so the slot cost could never be
// measured, let alone fixed.
//
// The comparable quantity is the window one agent gets, which is also what the
// contract forbids reducing silently.
func TestSlotCandidatesAreComparedOnPerAgentContext(t *testing.T) {
	base := &Strategy{ContextSize: 1046528, Parallel: 4, KVPlacement: "gpu"}

	// Same per-agent window, fewer slots: comparable.
	fewer := &Strategy{ContextSize: 261632, Parallel: 1, KVPlacement: "gpu"}
	if !sameCalibrationPerAgentContext(base, fewer) {
		t.Errorf("a candidate holding %d tokens per agent was rejected against the base's %d",
			fewer.ContextSize/fewer.Parallel, base.ContextSize/base.Parallel)
	}
	if !sameCalibrationResidency(base, fewer) {
		t.Error("a slot candidate preserving per-agent context was rejected as a residency change")
	}

	// Same slot count must still require the same total, so batch/ubatch and
	// topology candidates keep their existing comparison.
	if sameCalibrationPerAgentContext(base, &Strategy{ContextSize: 900000, Parallel: 4, KVPlacement: "gpu"}) {
		t.Error("a same-width candidate with a different total context was accepted")
	}

	// Fewer slots but a smaller window per agent is a quality cut, not a tuning
	// move, and must not enter the search.
	if sameCalibrationPerAgentContext(base, &Strategy{ContextSize: 100000, Parallel: 1, KVPlacement: "gpu"}) {
		t.Error("a candidate that shrinks per-agent context was accepted")
	}
	// More slots at the same per-agent window is legal in the other direction.
	if !sameCalibrationPerAgentContext(&Strategy{ContextSize: 261632, Parallel: 1},
		&Strategy{ContextSize: 523264, Parallel: 2}) {
		t.Error("widening at a preserved per-agent window was rejected")
	}
}

func TestPerAgentContextComparisonRejectsUnusableInputs(t *testing.T) {
	ok := &Strategy{ContextSize: 1000, Parallel: 1}
	for name, args := range map[string][2]*Strategy{
		"nil base":      {nil, ok},
		"nil candidate": {ok, nil},
		"both nil":      {nil, nil},
	} {
		if sameCalibrationPerAgentContext(args[0], args[1]) {
			t.Errorf("%s was accepted", name)
		}
	}
}

// The residency guard must keep doing its original job: a candidate that moves
// a GPU-resident plan onto the host is not a comparable tuning candidate, even
// when its per-agent window matches.
func TestResidencyGuardStillRejectsAHostFallback(t *testing.T) {
	base := &Strategy{ContextSize: 1046528, Parallel: 4, KVPlacement: "gpu"}
	host := &Strategy{ContextSize: 261632, Parallel: 1, KVPlacement: "cpu"}
	if sameCalibrationResidency(base, host) {
		t.Error("a host-KV candidate was accepted against a GPU-resident base")
	}
	mmapped := &Strategy{ContextSize: 261632, Parallel: 1, KVPlacement: "gpu", MMapRequired: true}
	if sameCalibrationResidency(base, mmapped) {
		t.Error("an mmap-required candidate was accepted against a resident base")
	}
}
