package placement

import (
	"testing"
)

func pinStrategy(nCPUMoE int, ot string) *Strategy {
	return &Strategy{
		ContextSize: 287744, UBatchSize: 256, Parallel: 1,
		NCPUMoE: nCPUMoE, OTString: ot, SplitMode: "layer",
		TensorSplit: []float64{0.12, 0.85, 0.02},
	}
}

// TestBootstrapPinBreaksTheAlternatingShapeLivelock is the regression this file
// exists for.
//
// Measured 2026-09-07 with identical user settings across four launches:
// --n-cpu-moe went 41 -> 40 -> 41 -> 40, producing four distinct
// allocation-placement identities for one ctx/ubatch. Each launch measured a
// baseline for the shape it served, and that very evidence moved the next plan
// to the other shape, whose baseline was unmeasured. Hot experts bootstrapped
// forever and never once engaged.
func TestBootstrapPinBreaksTheAlternatingShapeLivelock(t *testing.T) {
	dir := t.TempDir()
	model := "/models/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf"

	// Launch 1 bootstraps: it serves shape 41 cache-free to measure it.
	measured := pinStrategy(41, `blk\.(3|4)\..*=CUDA1,exps=CPU`)
	if err := RecordHotExpertBootstrapPin(dir, model, measured); err != nil {
		t.Fatalf("record pin: %v", err)
	}

	// Launch 2: new evidence makes the planner derive shape 40 instead.
	derived := pinStrategy(40, `blk\.(3|4|5)\..*=CUDA1,exps=CPU`)
	pin := MeasuredHotExpertBootstrapPin(dir, model)
	if !pin.AppliesTo(derived) {
		t.Fatal("the pin must apply: same ctx/ubatch/parallel, only the topology drifted")
	}
	if !applyHotExpertBootstrapPin(derived, pin) {
		t.Fatal("the pin must replay onto the drifted plan")
	}
	if derived.NCPUMoE != 41 {
		t.Fatalf("replay must restore the measured shape: got n-cpu-moe %d, want 41", derived.NCPUMoE)
	}
	if derived.OTString != measured.OTString {
		t.Fatalf("replay must restore the measured expert pins:\n got %q\nwant %q",
			derived.OTString, measured.OTString)
	}
}

// TestBootstrapPinRefusesADifferentShape keeps the pin from replaying a
// topology whose allocation no longer describes this launch. A pin is a
// measured MoE layout, not a licence to override the fit stage.
func TestBootstrapPinRefusesADifferentShape(t *testing.T) {
	dir := t.TempDir()
	model := "/models/m.gguf"
	if err := RecordHotExpertBootstrapPin(dir, model, pinStrategy(41, "ot")); err != nil {
		t.Fatalf("record: %v", err)
	}
	pin := MeasuredHotExpertBootstrapPin(dir, model)

	for name, s := range map[string]*Strategy{
		"different context":    {ContextSize: 131072, UBatchSize: 256, Parallel: 1, NCPUMoE: 40},
		"different ubatch":     {ContextSize: 287744, UBatchSize: 64, Parallel: 1, NCPUMoE: 40},
		"different slot count": {ContextSize: 287744, UBatchSize: 256, Parallel: 4, NCPUMoE: 40},
	} {
		if pin.AppliesTo(s) {
			t.Fatalf("%s must not accept the pin", name)
		}
		if applyHotExpertBootstrapPin(s, pin) {
			t.Fatalf("%s must not be rewritten by the pin", name)
		}
	}
}

// TestBootstrapPinRoundTripsTheOTString guards the serialisation: an -ot value
// contains '=' (blk...=CUDA1), so a naive key/value split truncates the expert
// pins and would replay a topology that seats no experts at all.
func TestBootstrapPinRoundTripsTheOTString(t *testing.T) {
	dir := t.TempDir()
	model := "/models/m.gguf"
	ot := `blk\.(3|4)\.ffn_((gate_up|gate|up|down)_(ch|)exps|gate_inp).*=CUDA1,blk\.(5)\..*=CUDA2,exps=CPU`
	if err := RecordHotExpertBootstrapPin(dir, model, pinStrategy(41, ot)); err != nil {
		t.Fatalf("record: %v", err)
	}
	got := MeasuredHotExpertBootstrapPin(dir, model)
	if got.OTString != ot {
		t.Fatalf("OT string did not survive the round trip:\n got %q\nwant %q", got.OTString, ot)
	}
	if len(got.TensorSplit) != 3 || got.TensorSplit[1] != 0.85 {
		t.Fatalf("tensor split did not round-trip: %v", got.TensorSplit)
	}
	if got.SplitMode != "layer" {
		t.Fatalf("split mode did not round-trip: %q", got.SplitMode)
	}
}

// TestClearedPinStopsReplaying pins the lifecycle: once the cache engages the
// pin must go, or every later launch replays a frozen topology after the
// evidence has moved on.
func TestClearedPinStopsReplaying(t *testing.T) {
	dir := t.TempDir()
	model := "/models/m.gguf"
	if err := RecordHotExpertBootstrapPin(dir, model, pinStrategy(41, "ot")); err != nil {
		t.Fatalf("record: %v", err)
	}
	if !MeasuredHotExpertBootstrapPin(dir, model).Valid() {
		t.Fatal("precondition: pin should exist")
	}
	if err := ClearHotExpertBootstrapPin(dir, model); err != nil {
		t.Fatalf("clear: %v", err)
	}
	if MeasuredHotExpertBootstrapPin(dir, model).Valid() {
		t.Fatal("a cleared pin must not replay")
	}
	// Clearing an absent pin is not an error: the cache can engage on a launch
	// that never bootstrapped.
	if err := ClearHotExpertBootstrapPin(dir, model); err != nil {
		t.Fatalf("clearing an absent pin must be a no-op, got %v", err)
	}
}

// TestPinIsNoOpWhenShapeAlreadyMatches avoids a misleading banner line: when
// the planner already derived the pinned topology there is nothing to replay.
func TestPinIsNoOpWhenShapeAlreadyMatches(t *testing.T) {
	dir := t.TempDir()
	model := "/models/m.gguf"
	s := pinStrategy(41, "ot")
	if err := RecordHotExpertBootstrapPin(dir, model, s); err != nil {
		t.Fatalf("record: %v", err)
	}
	same := pinStrategy(41, "ot")
	if applyHotExpertBootstrapPin(same, MeasuredHotExpertBootstrapPin(dir, model)) {
		t.Fatal("replaying an identical topology must report no change")
	}
}

// TestBootstrapIsMarkedNotRecordedDuringPlanning guards the mistake this cost
// one live run: finalizeHotExpertCache runs while candidates are still being
// evaluated, so the plan it holds is not necessarily the one that launches.
// Recording the pin there wrote a topology no launch ever served (measured
// 2026-09-07: pinned n-cpu-moe 38 against a launched 40), which recreates the
// alternating-shape livelock instead of breaking it. The bootstrap may only
// MARK the plan; the launcher writes the pin from the final argv.
func TestBootstrapIsMarkedNotRecordedDuringPlanning(t *testing.T) {
	dir := t.TempDir()
	model := "/models/m.gguf"

	// A planning-time strategy that is flagged but never launched.
	planning := pinStrategy(38, "candidate-ot")
	planning.HotExpertBootstrapPending = true

	// Nothing may exist on disk from planning alone.
	if MeasuredHotExpertBootstrapPin(dir, model).Valid() {
		t.Fatal("planning must not write a pin")
	}

	// The launcher records the plan that actually started.
	launched := pinStrategy(40, "launched-ot")
	launched.HotExpertBootstrapPending = true
	if err := RecordHotExpertBootstrapPin(dir, model, launched); err != nil {
		t.Fatalf("record: %v", err)
	}
	got := MeasuredHotExpertBootstrapPin(dir, model)
	if got.NCPUMoE != 40 {
		t.Fatalf("the pin must describe the launched plan, got n-cpu-moe %d, want 40", got.NCPUMoE)
	}
	if got.OTString != "launched-ot" {
		t.Fatalf("the pin must carry the launched expert pins, got %q", got.OTString)
	}
}
