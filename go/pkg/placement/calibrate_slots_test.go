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

// The scaling gate is the request being automatic, not the strategy carrying
// ContextAuto. A Claude Code base arrives with ContextAuto false even though its
// window was derived, so gating on the strategy made the scaling inert exactly
// where it was needed: slot candidates kept the base's full total, which for
// parallel-1 meant a million tokens of KV on one slot — no saving and not
// feasible. Verified against this machine's real capabilities and the real
// Qwen3.8-Flash-Next profile, where the candidates now come out as
// 786,432/3, 524,288/2 and 262,144/1, all at 262,144 per agent.
func TestSlotScalingGateUsesTheAutomaticContextRequest(t *testing.T) {
	base := &Strategy{ContextSize: 1048576, Parallel: 4, ContextAuto: false}

	automatic := Options{AutoContextMax: 1048576}
	if !slotCandidateScalesContext(automatic, base, 1) {
		t.Error("an automatic-context request did not scale a slot candidate")
	}
	// An explicit window is a user constraint: the total stays put and the
	// candidate redistributes it, which is the historical behaviour.
	if slotCandidateScalesContext(Options{}, base, 1) {
		t.Error("an explicit-context request scaled a slot candidate")
	}
	// Same width is not a slot candidate at all.
	if slotCandidateScalesContext(automatic, base, 4) {
		t.Error("a same-width candidate was scaled")
	}
}

// A slot candidate must keep KV where the baseline has it. Left free, the packer
// spends the KV freed by cutting slots on expert residency and then relocates
// the cache to the host — a residency change sameCalibrationResidency refuses to
// compare, so the candidate is discarded before it can be measured.
//
// Observed on Qwen3.8-Flash-Next: unpinned, parallel-1 and parallel-2 both came
// back kv=cpu against a kv=gpu base. Pinned, both stay GPU-resident and
// n-cpu-moe falls from 48 to 46.
func TestSlotCandidatesKeepTheBaselineKVPlacement(t *testing.T) {
	base := &Strategy{ContextSize: 254976, Parallel: 3, KVPlacement: "gpu"}
	opts := Options{AutoContextMax: 1048576, KVPlacement: ""}

	got := slotCandidateOptions(opts, base, 1)
	if got.KVPlacement != "gpu" {
		t.Errorf("slot candidate KV placement = %q, want the baseline's %q", got.KVPlacement, base.KVPlacement)
	}
	if got.ContextSize != 84992 {
		t.Errorf("slot candidate context = %d, want 84992 (per-agent preserved)", got.ContextSize)
	}

	// An explicit-context request keeps the historical behaviour: the total is a
	// user constraint, so neither it nor KV placement is rewritten here.
	explicit := slotCandidateOptions(Options{KVPlacement: ""}, base, 1)
	if explicit.ContextSize != 0 || explicit.KVPlacement != "" {
		t.Errorf("an explicit-context request was rewritten: ctx=%d kv=%q", explicit.ContextSize, explicit.KVPlacement)
	}

	// A baseline with no recorded placement leaves the candidate's choice alone.
	noPlacement := slotCandidateOptions(opts, &Strategy{ContextSize: 254976, Parallel: 3}, 1)
	if noPlacement.KVPlacement != "" {
		t.Errorf("KV placement was invented from an empty baseline: %q", noPlacement.KVPlacement)
	}
}
