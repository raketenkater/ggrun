package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

// acceptedNCPUMoE only arms after an exact preflight fits. On a launch where
// nothing ever fits it stays zero, so it cannot stop a context re-plan from
// undoing the derate ladder. That is the Qwen3.8-Flash-Next reviewer-seat
// failure: n-cpu-moe went 46, 45, 46, 40, 44, 45, 46, 44, 45 — every derate up,
// every context re-plan down — until the replan budget ran out.
func TestResidencyFloorRatchetsWithoutAProvenFit(t *testing.T) {
	r := newLaunchMemoryRecovery()
	if got := r.expertResidencyFloor(); got != 0 {
		t.Fatalf("a launch that has derated nothing has floor %d, want 0", got)
	}

	// The observed rounds of the real failure, in order.
	for _, n := range []int{46, 45, 46, 40, 44, 45, 46, 44, 45} {
		r.observeExpertResidency(&placement.Strategy{NCPUMoE: n})
	}
	if got := r.expertResidencyFloor(); got != 46 {
		t.Errorf("floor = %d after the observed rounds, want 46 (the high-water mark)", got)
	}

	// Nothing here fitted, so the proven-fit ledger must still be untouched.
	if r.acceptedNCPUMoE != 0 {
		t.Errorf("acceptedNCPUMoE = %d without any proven fit, want 0", r.acceptedNCPUMoE)
	}
}

// The floor only moves toward more experts on the CPU, which is the direction
// that frees VRAM. A floor that could fall would be unsafe to re-impose.
func TestResidencyFloorOnlyMovesTowardTheCPU(t *testing.T) {
	r := newLaunchMemoryRecovery()
	r.observeExpertResidency(&placement.Strategy{NCPUMoE: 40})
	r.observeExpertResidency(&placement.Strategy{NCPUMoE: 12})
	if got := r.expertResidencyFloor(); got != 40 {
		t.Errorf("floor fell to %d, want 40", got)
	}
	r.observeExpertResidency(nil)
	if got := r.expertResidencyFloor(); got != 40 {
		t.Errorf("a nil strategy moved the floor to %d, want 40", got)
	}
	var missing *launchMemoryRecovery
	missing.observeExpertResidency(&placement.Strategy{NCPUMoE: 9})
	if got := missing.expertResidencyFloor(); got != 0 {
		t.Errorf("nil recovery reported floor %d, want 0", got)
	}
}

// holdExpertResidency must leave a plan alone unless it actually lost ground,
// and must never fail a launch closed on its own: a re-plan that still fits is
// better than none, and the caller's exact preflight stays the authority.
func TestHoldExpertResidencyLeavesGoodPlansAlone(t *testing.T) {
	model := &placement.ModelProfile{IsMoE: true, NumLayers: 48, ExpertBytes: 48 << 20}
	plan := &placement.Strategy{NCPUMoE: 44}

	for name, floor := range map[string]int{
		"no floor recorded":         0,
		"plan already at the floor": 44,
		"plan beyond the floor":     40,
	} {
		if got := holdExpertResidency(nil, model, placement.Options{}, plan, floor, 0); got != plan {
			t.Errorf("%s: plan was replaced, want it kept", name)
		}
	}

	if got := holdExpertResidency(nil, model, placement.Options{}, nil, 46, 0); got != nil {
		t.Error("a nil plan was replaced")
	}
	// No device to charge the penalty to.
	if got := holdExpertResidency(nil, model, placement.Options{}, plan, 46, -1); got != plan {
		t.Error("a plan was replaced with no failing device to penalise")
	}
	// A non-MoE model has no expert layers to price, so there is no penalty to
	// compute and the plan must survive untouched.
	dense := &placement.ModelProfile{NumLayers: 48}
	if got := holdExpertResidency(nil, dense, placement.Options{}, plan, 46, 0); got != plan {
		t.Error("a dense model's plan was replaced")
	}
}

// When the re-plan cannot be re-packed (nil caps make ReplanAfterOOM fail), the
// original plan is returned rather than an error. Failing closed here would
// turn a recoverable launch into a dead one.
func TestHoldExpertResidencyFallsBackToTheRecomputedPlan(t *testing.T) {
	model := &placement.ModelProfile{IsMoE: true, NumLayers: 48, ExpertBytes: 48 << 20}
	plan := &placement.Strategy{NCPUMoE: 40}
	if got := holdExpertResidency(nil, model, placement.Options{}, plan, 46, 0); got != plan {
		t.Error("a failed re-pack did not fall back to the recomputed plan")
	}
}
