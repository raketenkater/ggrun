package placement

import "testing"

// measuredAgentSamples mirrors the shape of the 11692 real agent requests
// observed on 2026-09-03: a long tail of tiny turns and a small number of large
// ones, max 143718 tokens.
func measuredAgentSamples() []int {
	s := make([]int, 0, 1200)
	for i := 0; i < 1000; i++ {
		s = append(s, 1002)
	}
	for i := 0; i < 150; i++ {
		s = append(s, 32484)
	}
	for i := 0; i < 40; i++ {
		s = append(s, 78958)
	}
	s = append(s, 143718)
	return s
}

// TestAgentContextCeilingCutsTheMeasuredOverProvision is the regression this
// file exists for: auto mode served 942080 tokens of context against a workload
// whose largest observed request was 143718, spending 17.4 GB of KV+compute and
// leaving every expert layer on the host.
func TestAgentContextCeilingCutsTheMeasuredOverProvision(t *testing.T) {
	d := AgentContextDemandFromSamples(measuredAgentSamples())
	if d.MaxTokens != 143718 {
		t.Fatalf("max should be the largest observed request, got %d", d.MaxTokens)
	}
	ceiling := AgentContextCeiling(d, 1)
	if ceiling <= 0 {
		t.Fatal("a 1191-sample distribution must be treated as evidence")
	}
	if ceiling >= 942080 {
		t.Fatalf("ceiling %d must be far below the 942080 that starved the experts", ceiling)
	}
	// It must still clear the largest request actually seen, with room to grow.
	if ceiling < d.MaxTokens {
		t.Fatalf("ceiling %d would truncate the largest observed request %d", ceiling, d.MaxTokens)
	}
	if ceiling < d.MaxTokens*agentContextGrowthFactor {
		t.Fatalf("ceiling %d is below the stated %dx growth policy over max %d",
			ceiling, agentContextGrowthFactor, d.MaxTokens)
	}
}

// TestAgentContextCeilingScalesWithSlots is what makes the ceiling correct for
// multi-agent work: llama.cpp splits --ctx-size across --parallel slots, so a
// per-agent ceiling must be bought once per slot or every agent gets a fraction
// of what it needs.
func TestAgentContextCeilingScalesWithSlots(t *testing.T) {
	d := AgentContextDemandFromSamples(measuredAgentSamples())
	one := AgentContextCeiling(d, 1)
	four := AgentContextCeiling(d, 4)
	// Compare against the pre-rounding requirement, not one*4: each ceiling is
	// rounded up to a whole granule independently, so one*4 overshoots by up to
	// three granules and would fail a correct implementation.
	if want := d.MaxTokens * agentContextGrowthFactor * 4; four < want {
		t.Fatalf("4 slots must buy 4 agents' worth of context: got %d, need >= %d", four, want)
	}
	if four <= one {
		t.Fatalf("more slots must buy more total context: 1 slot %d, 4 slots %d", one, four)
	}
	// Per-slot share must still clear a single agent's demand.
	if four/4 < d.MaxTokens {
		t.Fatalf("per-slot share %d truncates an agent whose largest request was %d", four/4, d.MaxTokens)
	}
}

// TestAgentContextCeilingIgnoresThinEvidence keeps a quiet first session from
// pinning the deployment to a tiny context. Below the sample gate the automatic
// path must keep its existing capacity-driven behaviour.
func TestAgentContextCeilingIgnoresThinEvidence(t *testing.T) {
	thin := AgentContextDemandFromSamples([]int{900, 1200, 3000})
	if got := AgentContextCeiling(thin, 1); got != 0 {
		t.Fatalf("3 samples must not size a deployment, got ceiling %d", got)
	}
	if got := AgentContextCeiling(AgentContextDemand{}, 1); got != 0 {
		t.Fatalf("an unobserved deployment must offer no ceiling, got %d", got)
	}
}

// TestRecordedDemandNeverShrinks guards the same failure RecordCompanionVRAM
// guards: a ceiling recomputed during a quiet period must not fall below a
// long conversation already seen, or the next one is truncated.
func TestRecordedDemandNeverShrinks(t *testing.T) {
	dir := t.TempDir()
	big := AgentContextDemandFromSamples(measuredAgentSamples())
	if err := RecordAgentContextDemand(dir, big); err != nil {
		t.Fatalf("record: %v", err)
	}
	quiet := AgentContextDemandFromSamples([]int{800, 900, 1000})
	if err := RecordAgentContextDemand(dir, quiet); err != nil {
		t.Fatalf("record quiet period: %v", err)
	}
	got := MeasuredAgentContextDemand(dir)
	if got.MaxTokens != big.MaxTokens {
		t.Fatalf("a quiet period lowered the ceiling from %d to %d", big.MaxTokens, got.MaxTokens)
	}
	if got.Samples < big.Samples {
		t.Fatalf("sample count regressed from %d to %d", big.Samples, got.Samples)
	}
}

// TestAutoContextCapHonoursMeasuredDemand pins the wiring: the demand ceiling
// must actually reach autoContextCap, and must only ever lower it.
func TestAutoContextCapHonoursMeasuredDemand(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "m.gguf", CTXTrain: 1048576}

	if got := autoContextCap(model, Options{CacheDir: dir, Parallel: 1}); got < 1000000 {
		t.Fatalf("with no evidence the cap must stay at the model's native context, got %d", got)
	}
	if err := RecordAgentContextDemand(dir, AgentContextDemandFromSamples(measuredAgentSamples())); err != nil {
		t.Fatalf("record: %v", err)
	}
	capped := autoContextCap(model, Options{CacheDir: dir, Parallel: 1})
	if capped >= 1048576 {
		t.Fatalf("measured demand must lower the automatic cap, got %d", capped)
	}
	if capped < 143718 {
		t.Fatalf("the cap must still clear the largest observed request, got %d", capped)
	}
}

// TestMeasuredDemandNeverRaisesTheCap keeps the ceiling one-directional: it is
// a demand limit, never a licence to buy more context than the model or the
// user's own --auto-context-max allows.
func TestMeasuredDemandNeverRaisesTheCap(t *testing.T) {
	dir := t.TempDir()
	huge := AgentContextDemandFromSamples(func() []int {
		s := make([]int, 0, 400)
		for i := 0; i < 400; i++ {
			s = append(s, 900000)
		}
		return s
	}())
	if err := RecordAgentContextDemand(dir, huge); err != nil {
		t.Fatalf("record: %v", err)
	}
	model := &ModelProfile{Path: "m.gguf", CTXTrain: 131072}
	got := autoContextCap(model, Options{CacheDir: dir, Parallel: 1})
	if got > 131072 {
		t.Fatalf("demand must never exceed the model's native context, got %d", got)
	}
	small := autoContextCap(model, Options{CacheDir: dir, Parallel: 1, AutoContextMax: 65536})
	if small > 65536 {
		t.Fatalf("demand must never override an explicit --auto-context-max, got %d", small)
	}
}
