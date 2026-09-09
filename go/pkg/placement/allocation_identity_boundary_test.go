package placement

import "testing"

func TestAllocationIdentityCanonicalizationRequiresCompleteLayerMode(t *testing.T) {
	model := &ModelProfile{NumLayers: 46}
	base := identityStrategy([]float64{0.23, 0.65, 0.12})
	jitter := identityStrategy([]float64{0.23, 0.64, 0.12})
	if AllocationPlacementIdentity(base, model) != AllocationPlacementIdentity(jitter, model) {
		t.Fatal("complete layer splits with identical backend ownership should canonicalize")
	}
	row := *base
	row.SplitMode = "row"
	rowJitter := *jitter
	rowJitter.SplitMode = "row"
	if AllocationPlacementIdentity(&row, model) == AllocationPlacementIdentity(&rowJitter, model) {
		t.Fatal("row tensor ratios must remain identity-sensitive")
	}
	partial := *base
	partial.GPULayers = 20
	partialJitter := *jitter
	partialJitter.GPULayers = 20
	if AllocationPlacementIdentity(&partial, model) == AllocationPlacementIdentity(&partialJitter, model) {
		t.Fatal("partial offload must remain identity-sensitive")
	}
}

func TestAllocationIdentityUsesStrictFloat32Boundary(t *testing.T) {
	s := identityStrategy([]float64{0.5, 0.5})
	s.GPULayers = 4
	got := splitIdentityForStrategy(s, &ModelProfile{NumLayers: 3})
	if got != "2,1/out1" {
		t.Fatalf("strict upper-bound ownership = %q, want 2,1/out1", got)
	}
}

func TestAllocationIdentityUsesEmittedRatiosBeforeFloat32Ownership(t *testing.T) {
	model := &ModelProfile{NumLayers: 3}
	rounded := identityStrategy([]float64{0.504, 0.496})
	actual := identityStrategy([]float64{0.50, 0.50})
	if AllocationPlacementIdentity(rounded, model) != AllocationPlacementIdentity(actual, model) {
		t.Fatal("identical emitted splits must have identical allocation identity")
	}
	if got := splitIdentityForStrategy(rounded, model); got != "2,1/out1" {
		t.Fatalf("identity used un-emitted planner precision: %s", got)
	}
}
