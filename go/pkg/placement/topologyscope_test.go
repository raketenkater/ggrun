package placement

import "testing"

func topoStrategy(nCPUMoE int, ot string) *Strategy {
	return &Strategy{
		Type: MoEOffload, ContextSize: 287744, UBatchSize: 256, Parallel: 1,
		BatchSize: 2048, KVPlacement: "gpu", KVQuality: "q8_0", KVType: "q8_0",
		NCPUMoE: nCPUMoE, OTString: ot, SplitMode: "layer",
		TensorSplit: []float64{0.12, 0.85, 0.02},
	}
}

// TestProbeFromAnotherLayoutIsNotSpent is the regression for the defect that
// produced days of unexplained crashes. The probe key omits the expert layout,
// so two topologies share one entry: observed 2026-09-07, consecutive launches
// alternated --n-cpu-moe 41/40/41/40 and a graph reserve measured under one was
// spent under the other, leaving too little for cudaGraphInstantiate.
func TestProbeFromAnotherLayoutIsNotSpent(t *testing.T) {
	planning := topoStrategy(40, `blk\.(3|4|5)\..*=CUDA1,exps=CPU`)
	measured := topoStrategy(41, `blk\.(3|4)\..*=CUDA1,exps=CPU`)

	pc := &probeCache{
		AllocationPlacementIdentity:    AllocationPlacementIdentity(measured, nil),
		ComputeBufMB:                   4871,
		ComputeBufByGPU:                map[int]int{1: 4871},
		RuntimeGraphGrowthByGPU:        map[int]int{1: 1650},
		RuntimeGraphGrowthFromOOMByGPU: map[int]bool{1: false},
		ModelByGPU:                     map[int]int{1: 15582},
		ModelHostMB:                    116868,
		KVPerLayerMB:                   33,
		PromptCacheBytesPerToken:       512,
		ContextTotalMB:                 3188,
	}

	scoped := scopeProbeToTopology(pc, planning)
	if scoped.RuntimeGraphGrowthByGPU != nil {
		t.Fatalf("a graph reserve from another layout must not be spent: %v", scoped.RuntimeGraphGrowthByGPU)
	}
	if scoped.ComputeBufByGPU != nil || scoped.ComputeBufMB != 0 {
		t.Fatalf("compute buffers are per-layout and must be dropped: %v/%d",
			scoped.ComputeBufByGPU, scoped.ComputeBufMB)
	}
	if scoped.ModelByGPU != nil || scoped.ModelHostMB != 0 {
		t.Fatal("per-device weight residency is decided by the layout and must be dropped")
	}
	// Layout-independent costs survive: dropping them would force needless
	// re-measurement of things the layout does not move.
	if scoped.KVPerLayerMB != 33 || scoped.PromptCacheBytesPerToken != 512 || scoped.ContextTotalMB != 3188 {
		t.Fatalf("per-token costs must survive a layout change: %+v", scoped)
	}
	// The stored record itself is untouched, so the layout that measured it can
	// still read it back.
	if pc.RuntimeGraphGrowthByGPU[1] != 1650 {
		t.Fatal("scoping must not mutate the cached entry")
	}
}

// TestProbeFromTheSameLayoutIsSpentUnchanged keeps the guard from throwing away
// evidence that does describe this launch.
func TestProbeFromTheSameLayoutIsSpentUnchanged(t *testing.T) {
	s := topoStrategy(40, "ot")
	pc := &probeCache{
		AllocationPlacementIdentity: AllocationPlacementIdentity(s, nil),
		ComputeBufByGPU:             map[int]int{1: 4871},
		RuntimeGraphGrowthByGPU:     map[int]int{1: 1650},
	}
	if got := scopeProbeToTopology(pc, s); got != pc {
		t.Fatal("a matching layout must be spent as-is")
	}
}

// TestUnidentifiedProbeIsStillSpent pins the deliberate limit of this guard.
// Identity is stamped by RecordMeasuredAllocation, while compute buffers arrive
// separately from the fit-params oracle -- measured on this rig, only 6 of 40
// entries with compute buffers carried an identity. Rejecting the unidentified
// majority would push nearly every plan back onto the cold estimate, a worse
// regression than the collision. Closing that gap means stamping identity in
// every recorder, not tightening this rule.
func TestUnidentifiedProbeIsStillSpent(t *testing.T) {
	s := topoStrategy(40, "ot")
	pc := &probeCache{ComputeBufByGPU: map[int]int{1: 4871}}
	scoped := scopeProbeToTopology(pc, s)
	if scoped.ComputeBufByGPU == nil {
		t.Fatal("an entry with no identity is unverifiable, not disproven; it must still be spent")
	}
}
