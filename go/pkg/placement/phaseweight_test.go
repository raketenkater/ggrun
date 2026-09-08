package placement

import (
	"math"
	"testing"
)

// Rows in the backend's own format, from the 2026-09-08 agent-shaped session.
const phaseLog = `
5.21.429.647 I slot print_timing: id  0 | task 0 | prompt eval time =   55102.28 ms /  2906 tokens (   18.96 ms per token,    52.74 tokens per second)
5.21.429.652 I slot print_timing: id  0 | task 0 |        eval time =   16364.49 ms /   119 tokens (  138.68 ms per token,     7.21 tokens per second)
6.08.324.260 I slot print_timing: id  0 | task 3 | prompt eval time =    2150.25 ms /    26 tokens (   82.70 ms per token,    12.09 tokens per second)
6.08.324.265 I slot print_timing: id  0 | task 3 |        eval time =   44585.03 ms /   300 tokens (  149.11 ms per token,     6.71 tokens per second)
`

// Agent-shaped traffic: large prompts, short replies.
const agentShapedLog = `
I slot print_timing: id  0 | task 0 | prompt eval time =   55102.28 ms /  2906 tokens (   18.96 ms per token,    52.74 tokens per second)
I slot print_timing: id  0 | task 0 |        eval time =   16364.49 ms /   119 tokens (  138.68 ms per token,     7.21 tokens per second)
I slot print_timing: id  0 | task 1 | prompt eval time =  174320.00 ms /  6096 tokens (   28.60 ms per token,    34.97 tokens per second)
I slot print_timing: id  0 | task 1 |        eval time =   41000.00 ms /   295 tokens (  138.98 ms per token,     7.20 tokens per second)
`

// The synthetic A/B workload: tiny prompts, long replies. Decode-dominant.
const syntheticLog = `
I slot print_timing: id  0 | task 3 | prompt eval time =    2150.25 ms /    26 tokens (   82.70 ms per token,    12.09 tokens per second)
I slot print_timing: id  0 | task 3 |        eval time =   44585.03 ms /   300 tokens (  149.11 ms per token,     6.71 tokens per second)
`

// A request that produced nothing. Counting it as a decode sample drags the
// rate toward zero.
const degenerateLog = `
5.21.429.652 I slot print_timing: id  0 | task 9 |        eval time =       0.00 ms /     1 tokens (    0.00 ms per token,     0.00 tokens per second)
`

func TestParseAgentPhaseTimingsSplitsByLabel(t *testing.T) {
	got := ParseAgentPhaseTimings(phaseLog)

	if !got.Measured() {
		t.Fatal("both phases are present; the mix must read as measured")
	}
	if got.PrefillTokens != 2932 {
		t.Errorf("prefill tokens: got %d, want 2932", got.PrefillTokens)
	}
	if got.DecodeTokens != 419 {
		t.Errorf("decode tokens: got %d, want 419", got.DecodeTokens)
	}
	if got.Samples != 4 {
		t.Errorf("samples: got %d, want 4", got.Samples)
	}
	// The regression this parser exists for: an earlier hand-rolled version swept
	// `prompt eval time` rows into the decode set, which invented a
	// "cache degrades with use" curve out of nothing. Assert the exact totals of
	// the two rows of each label -- "which phase is bigger" is not a leak test,
	// because a decode-heavy workload legitimately produces a bigger decode.
	if math.Abs(got.PrefillMS-57252.53) > 0.01 {
		t.Errorf("prefill ms: got %.2f, want 57252.53 (the two prompt-eval rows)", got.PrefillMS)
	}
	if math.Abs(got.DecodeMS-60949.52) > 0.01 {
		t.Errorf("decode ms: got %.2f, want 60949.52 (the two eval rows)", got.DecodeMS)
	}
}

func TestParseAgentPhaseTimingsDropsDegenerateRows(t *testing.T) {
	if got := ParseAgentPhaseTimings(degenerateLog); got.Samples != 0 {
		t.Errorf("a 1-token 0.00 ms row is not a sample; got %d", got.Samples)
	}
	// It must also not poison a real observation.
	got := ParseAgentPhaseTimings(phaseLog + degenerateLog)
	if got.Samples != 4 {
		t.Errorf("samples: got %d, want 4", got.Samples)
	}
	if rate := got.DecodeTPS(); rate < 6 || rate > 8 {
		t.Errorf("decode rate: got %.2f tok/s, want ~7", rate)
	}
}

// TestPhaseShareTracksTheWorkload: the share must follow the traffic, not a
// baked-in assumption about either phase. The two fixtures are deliberately on
// opposite sides.
//
// Neither fixture is a claim about this rig. The real 12-turn measurement here
// came out 42.3% prefill / 57.7% decode -- close to even and leaning decode --
// after an estimate from a single turn had said 80% prefill.
func TestPhaseShareTracksTheWorkload(t *testing.T) {
	// Large prompts with short replies must read as prefill-heavy...
	got := ParseAgentPhaseTimings(agentShapedLog)
	share := got.PrefillTimeShare()
	if share <= 0.5 {
		t.Errorf("prefill share: got %.2f, want > 0.5 on large-prompt traffic", share)
	}
	if share >= 1 {
		t.Errorf("prefill share must be a fraction; got %.2f", share)
	}
	// ...and small prompts with long replies as decode-heavy, or the weight is
	// not reading the workload at all.
	if synth := ParseAgentPhaseTimings(syntheticLog).PrefillTimeShare(); synth >= 0.5 {
		t.Errorf("a 26-token-prompt workload should be decode-dominant; got %.2f", synth)
	}
}

func TestUnmeasuredMixWeighsNothing(t *testing.T) {
	only := ParseAgentPhaseTimings(`prompt eval time =  100.00 ms /  50 tokens`)
	if only.Measured() {
		t.Error("one phase is not a mix")
	}
	if only.PrefillTimeShare() != 0 {
		t.Error("an unmeasured mix must report no share rather than 100% of one phase")
	}
	// An unscored candidate must not outrank a scored one.
	if cost := AgentTurnCostMS(only, 50, 7); cost != 0 {
		t.Errorf("unmeasured mix must not produce a score; got %.1f", cost)
	}
}

// TestAgentTurnCostRanksByWeightedTime is the objective invariant 5 asks for:
// prefill and decode both count, in the proportion actually observed.
func TestAgentTurnCostRanksByWeightedTime(t *testing.T) {
	mix := ParseAgentPhaseTimings(agentShapedLog)

	// Trading prefill away for decode: on a prefill-dominant mix this must lose,
	// which is exactly the "shrink the batch to free VRAM for expert layers"
	// move that looked attractive before the mix was measured.
	fastPrefill := AgentTurnCostMS(mix, 52.0, 7.0)
	fastDecode := AgentTurnCostMS(mix, 26.0, 8.5)
	if fastPrefill <= 0 || fastDecode <= 0 {
		t.Fatal("both candidates should score on a measured mix")
	}
	if fastPrefill >= fastDecode {
		t.Errorf("halving prefill for a 21%% decode gain should lose on this mix: "+
			"prefill-fast %.0f ms vs decode-fast %.0f ms", fastPrefill, fastDecode)
	}
	// A candidate faster in both phases must win outright.
	if both := AgentTurnCostMS(mix, 60.0, 8.0); both >= fastPrefill {
		t.Errorf("a candidate faster in both phases must win: %.0f vs %.0f", both, fastPrefill)
	}
	// Missing rates cannot score.
	if AgentTurnCostMS(mix, 0, 7) != 0 || AgentTurnCostMS(mix, 50, 0) != 0 {
		t.Error("a candidate with an unknown rate must not score")
	}
}

func TestAgentPhaseTimingAccumulatesAcrossSessions(t *testing.T) {
	cacheDir := t.TempDir()
	model := "GLM-5.3-Flash.gguf"
	one := ParseAgentPhaseTimings(phaseLog)

	for i := 0; i < 3; i++ {
		if err := RecordAgentPhaseTiming(cacheDir, model, one); err != nil {
			t.Fatalf("record: %v", err)
		}
	}
	got := MeasuredAgentPhaseTiming(cacheDir, model)
	if got.Samples != 12 {
		t.Fatalf("samples: got %d, want 12 accumulated", got.Samples)
	}
	if got.PrefillTokens != one.PrefillTokens*3 {
		t.Errorf("prefill tokens: got %d, want %d", got.PrefillTokens, one.PrefillTokens*3)
	}
	// Accumulation must not drift the ratio it exists to measure.
	if math.Abs(got.PrefillTimeShare()-one.PrefillTimeShare()) > 1e-9 {
		t.Errorf("share drifted under accumulation: %.6f vs %.6f",
			got.PrefillTimeShare(), one.PrefillTimeShare())
	}
}

// TestThinEvidenceWeighsNothing: one request's accident must not become the
// rig's workload model.
func TestThinEvidenceWeighsNothing(t *testing.T) {
	cacheDir := t.TempDir()
	model := "GLM-5.3-Flash.gguf"
	if err := RecordAgentPhaseTiming(cacheDir, model, ParseAgentPhaseTimings(phaseLog)); err != nil {
		t.Fatalf("record: %v", err)
	}
	if got := MeasuredAgentPhaseTiming(cacheDir, model); got.Measured() {
		t.Errorf("4 samples is below the %d-sample floor; got %s", minAgentPhaseSamples, got.Summary())
	}
	// An unmeasured mix must not be written at all.
	if err := RecordAgentPhaseTiming(cacheDir, "other.gguf", AgentPhaseTiming{PrefillMS: 5}); err != nil {
		t.Fatalf("record: %v", err)
	}
	if got := MeasuredAgentPhaseTiming(cacheDir, "other.gguf"); got.Measured() {
		t.Error("a one-phase observation must not be persisted as a weight")
	}
}
