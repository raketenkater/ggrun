package placement

import (
	"strings"
	"testing"
)

// The 2026-09-08 serve: compute buffers 4855/5114/4855 MiB against a ~4500 MiB
// routed-expert layer.
func measuredComputeBuffers() map[int]int {
	return map[int]int{0: 4855, 1: 5114, 2: 4855}
}

func seatsWithLayerSize(mb int) ExpertSeatReport {
	return ExpertSeatReport{ExpertLayerMB: mb, ExpertsOnCPU: 40}
}

// TestBatchSignalRejectsTheObviousMove: shrinking the batch to reclaim 14.8 GB
// of compute buffer for resident expert layers looks compelling, but on
// prefill-heavy traffic it sells the dominant phase, so the signal must point
// the other way. (This rig's own traffic measured 42/58 and holds instead --
// see TestBatchSummaryDistinguishesHoldFromUnmeasured. The fixture here is a
// prefill-heavy workload, not a claim about this hardware.)
func TestBatchSignalRejectsTheObviousMove(t *testing.T) {
	mix := ParseAgentPhaseTimings(agentShapedLog)
	got := AnalyzeBatchSize(mix, seatsWithLayerSize(4500), measuredComputeBuffers())

	if !got.Observed {
		t.Fatalf("a measured prefill-dominant mix should produce a signal: %s", got.Summary())
	}
	if got.Direction != BatchRaise {
		t.Errorf("direction: got %s, want raise on a prefill-dominant workload", got.Direction)
	}
	if got.ComputeBufMB != 14824 {
		t.Errorf("compute buffer: got %d MiB, want 14824", got.ComputeBufMB)
	}
	// The size of the trade must be reported, since that is what makes it
	// worth considering at all.
	if got.LayersWorth != 3 {
		t.Errorf("layers worth: got %d, want 3", got.LayersWorth)
	}
	if next := BatchChallenger(got, 2048); next != 4096 {
		t.Errorf("challenger: got %d, want 4096", next)
	}
}

// TestBatchSignalLowersOnDecodeBoundWork: the same component must reach the
// opposite conclusion on the opposite workload, or it is encoding this rig's
// answer rather than reading the machine.
func TestBatchSignalLowersOnDecodeBoundWork(t *testing.T) {
	mix := ParseAgentPhaseTimings(syntheticLog)
	got := AnalyzeBatchSize(mix, seatsWithLayerSize(4500), measuredComputeBuffers())

	if !got.Observed {
		t.Fatalf("a measured decode-dominant mix should produce a signal: %s", got.Summary())
	}
	if got.Direction != BatchLower {
		t.Errorf("direction: got %s, want lower on a decode-dominant workload", got.Direction)
	}
	if next := BatchChallenger(got, 2048); next != 1024 {
		t.Errorf("challenger: got %d, want 1024", next)
	}
}

// TestBatchSignalHoldsWhenNothingToReclaim: decode-bound work whose compute
// buffer is smaller than one expert layer has no trade available. Shrinking the
// batch would cost prefill and buy no residency at all.
func TestBatchSignalHoldsWhenNothingToReclaim(t *testing.T) {
	mix := ParseAgentPhaseTimings(syntheticLog)
	got := AnalyzeBatchSize(mix, seatsWithLayerSize(4500), map[int]int{0: 300, 1: 300})

	if got.Observed {
		t.Errorf("no whole expert layer is reclaimable; signal should hold: %s", got.Summary())
	}
	if BatchChallenger(got, 2048) != 0 {
		t.Error("an unobserved signal must not produce a challenger")
	}
}

// TestBatchSignalFailsClosedOnMissingEvidence: every input this depends on can
// be absent on a cold machine, and none of them may be guessed.
func TestBatchSignalFailsClosedOnMissingEvidence(t *testing.T) {
	mix := ParseAgentPhaseTimings(agentShapedLog)

	cases := []struct {
		name    string
		mix     AgentPhaseTiming
		seats   ExpertSeatReport
		compute map[int]int
	}{
		{"unmeasured mix", AgentPhaseTiming{}, seatsWithLayerSize(4500), measuredComputeBuffers()},
		{"unknown expert layer size", mix, seatsWithLayerSize(0), measuredComputeBuffers()},
		{"no compute buffers", mix, seatsWithLayerSize(4500), nil},
		{"zero compute buffers", mix, seatsWithLayerSize(4500), map[int]int{0: 0, 1: 0}},
	}
	for _, tc := range cases {
		got := AnalyzeBatchSize(tc.mix, tc.seats, tc.compute)
		if got.Observed {
			t.Errorf("%s: must not produce a signal", tc.name)
		}
		if BatchChallenger(got, 2048) != 0 {
			t.Errorf("%s: must not produce a challenger", tc.name)
		}
	}
}

// TestBatchSignalHoldsOnEvenMix: near an even split a bounded challenger is
// unlikely to separate from noise, so the launch is left alone.
func TestBatchSignalHoldsOnEvenMix(t *testing.T) {
	even := AgentPhaseTiming{
		PrefillMS: 1000, PrefillTokens: 500,
		DecodeMS: 1000, DecodeTokens: 100, Samples: 10,
	}
	got := AnalyzeBatchSize(even, seatsWithLayerSize(4500), measuredComputeBuffers())
	if got.Observed {
		t.Errorf("an even mix should hold; got %s", got.Summary())
	}
}

// TestBatchChallengerRespectsFloor: halving must stop before the batch stops
// being a batch, where prefill collapses toward per-token cost and the
// reclaimed VRAM cannot pay for it.
func TestBatchChallengerRespectsFloor(t *testing.T) {
	signal := BatchSizeSignal{Observed: true, Direction: BatchLower}
	if got := BatchChallenger(signal, 128); got != 64 {
		t.Errorf("128 should halve to 64; got %d", got)
	}
	if got := BatchChallenger(signal, 64); got != 0 {
		t.Errorf("64 is the floor; got %d", got)
	}
	if got := BatchChallenger(signal, 0); got != 0 {
		t.Errorf("an unknown current batch cannot be halved; got %d", got)
	}
}

// TestBatchSummaryDistinguishesHoldFromUnmeasured: a measured mix that declines
// to move the batch is a working decision, not a failed probe. The real
// 2026-09-08 session (42.3% prefill) lands exactly here, and reporting it as
// "unmeasured" would send a reader hunting a broken measurement.
func TestBatchSummaryDistinguishesHoldFromUnmeasured(t *testing.T) {
	realMix := AgentPhaseTiming{
		PrefillMS: 311700, PrefillTokens: 10321,
		DecodeMS: 425900, DecodeTokens: 3086, Samples: 32,
	}
	held := AnalyzeBatchSize(realMix, seatsWithLayerSize(4500), measuredComputeBuffers())
	if held.Observed {
		t.Fatalf("a 42/58 mix is too even to move the batch; got %s", held.Summary())
	}
	if got := held.Summary(); !strings.Contains(got, "hold") || strings.Contains(got, "unmeasured") {
		t.Errorf("a measured hold must not read as unmeasured; got %q", got)
	}

	none := AnalyzeBatchSize(AgentPhaseTiming{}, seatsWithLayerSize(4500), measuredComputeBuffers())
	if got := none.Summary(); !strings.Contains(got, "unmeasured") {
		t.Errorf("a genuinely unmeasured mix must say so; got %q", got)
	}
}
