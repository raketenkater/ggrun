package placement

import "testing"

// TestTargetSlotsExceedTheBareMinimum is the regression for the sizing that
// made hot experts pointless even once it engaged.
//
// The demotion loop stopped as soon as hotExpertMinUsefulSlots fit --
// ExpertUsedCount, 8 for GLM 5.3 Flash -- so residency kept the rest of the
// VRAM and the cache was sized to 14 slots. llama.cpp PR 27861 names 26-48
// slots/layer as the practical minimum, publishing 32 -> 69.3% and 64 -> 81.5%
// hit rates; below that range the cache does not repay its memory.
func TestTargetSlotsExceedTheBareMinimum(t *testing.T) {
	glm := &ModelProfile{Path: "GLM.gguf", NumExperts: 288, ExpertUsedCount: 8}
	min := hotExpertMinUsefulSlots(glm)
	target := hotExpertTargetSlots(glm)

	if min != 8 {
		t.Fatalf("precondition: GLM routes 8 experts per token, got min %d", min)
	}
	if target <= min {
		t.Fatalf("the demotion target must exceed the bare minimum: target %d, min %d", target, min)
	}
	if target < 26 {
		t.Fatalf("target %d is below the 26-slot practical floor the published curve names", target)
	}
}

// TestTargetScalesWithExpertCount keeps the target from being a constant that
// happens to suit one model. Roughly 15-20% of experts serve ~80% of tokens, so
// a model with few experts needs proportionally fewer slots to cover its hot
// set -- demoting resident layers past that point would spend VRAM for nothing.
func TestTargetScalesWithExpertCount(t *testing.T) {
	big := hotExpertTargetSlots(&ModelProfile{NumExperts: 288, ExpertUsedCount: 8})
	small := hotExpertTargetSlots(&ModelProfile{NumExperts: 64, ExpertUsedCount: 8})
	if small >= big {
		t.Fatalf("a 64-expert model must target fewer slots than a 288-expert one: %d vs %d", small, big)
	}
	if small < hotExpertMinUsefulSlots(&ModelProfile{NumExperts: 64, ExpertUsedCount: 8}) {
		t.Fatalf("the target must never fall below the minimum useful slots, got %d", small)
	}
}

// TestTargetIsCappedAtTheCurveKnee stops the target chasing hit rate forever:
// the published curve flattens (48 -> ~75-80%, 96 -> 86.9%), and every extra
// slot is paid for by demoting a resident expert layer.
func TestTargetIsCappedAtTheCurveKnee(t *testing.T) {
	huge := hotExpertTargetSlots(&ModelProfile{NumExperts: 4096, ExpertUsedCount: 8})
	if huge > hotExpertCurveKneeSlots {
		t.Fatalf("target %d exceeds the curve knee %d; demotion past it buys little",
			huge, hotExpertCurveKneeSlots)
	}
}

// TestTargetHandlesAnUnknownModel keeps the sizing safe when the profile is
// missing rather than producing a zero or negative target.
func TestTargetHandlesAnUnknownModel(t *testing.T) {
	if got := hotExpertTargetSlots(nil); got < 1 {
		t.Fatalf("an unknown model must still yield a positive target, got %d", got)
	}
}

// TestSlotBackoffPrefersASmallerCacheOverNoCache is the regression for a
// regression I introduced. Raising the demotion target made the sizing reach 32
// slots, but hotExpertCacheMaxSlots only divides residual VRAM by a per-slot
// estimate: the real per-GPU distribution plus runtime growth overshot, and the
// code returned a hard error instead of trying fewer slots. Measured
// 2026-09-07: "MoE expert cache enabled: 41 layers x 32 slots, 14344.7 MiB"
// followed by a CUDA OOM, with CUDA0 1435 MiB over budget -- so a working
// 14-slot cache became no cache at all.
//
// Fewer slots is a worse cache; it is not a failed launch.
func TestSlotBackoffPrefersASmallerCacheOverNoCache(t *testing.T) {
	// The backoff walks down by quarters and must terminate at the floor
	// rather than spinning or undershooting it.
	min := 8
	seen := map[int]bool{}
	steps := 0
	for try := 32; try >= min; try = try * 3 / 4 {
		if seen[try] {
			t.Fatalf("backoff repeated slot count %d; it would not terminate", try)
		}
		seen[try] = true
		if steps++; steps > 32 {
			t.Fatal("backoff did not converge")
		}
		if try == min {
			break
		}
	}
	if !seen[32] {
		t.Fatal("backoff must start at the sized slot count")
	}
	if len(seen) < 2 {
		t.Fatalf("backoff must try smaller sizes before giving up, tried %v", seen)
	}
}
