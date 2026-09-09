package placement

import "fmt"

// The probe cache is keyed by model|ctx|ubatch|kvQuality|kvPlacement|backend|
// gpuSig|parallel. That key says nothing about how the expert layers are laid
// out across the devices -- no --n-cpu-moe, no -ot pins, no tensor split --
// yet most of what the entry stores is a per-device number that exists only
// because of that layout.
//
// So two different expert topologies share one entry and read each other's
// measurements. Observed on this rig 2026-09-07: three distinct allocation
// placements under ctx=1048576 ubatch=64, three more under ctx=287744
// ubatch=1024, and consecutive launches alternating --n-cpu-moe 41/40/41/40
// while consuming a single cached record.
//
// The consequences were chased for days as separate bugs. A graph reserve of
// 1650 MiB measured under one layout was spent under another, left ~2.7 GiB
// on CUDA1 for a graph that wanted more, and the server aborted in
// cudaGraphInstantiate. The hot-expert bootstrap promised "the next launch of
// this shape" and never got one. Each was patched where it surfaced --
// relaxed lookups, scope guards, a bootstrap pin -- which is compensating
// machinery for a key that is missing a field.
//
// The entry already records which layout produced it, in
// AllocationPlacementIdentity. It simply was not consulted on the way in. This
// file consults it: a strategy may read only the measurements its own
// topology produced, and anything else reads as unmeasured, which every
// consumer already handles by falling back to the conservative estimate.
//
// Deliberately a read-side filter rather than a re-keying. Re-keying the files
// would strand every existing measurement with no way to tell which of them
// were still valid; filtering keeps each entry, spends only the parts that
// provably describe this launch, and lets the rest be re-measured under a key
// that can now be verified.

// Layout-dependent values are the ones scopeProbeToTopology drops on a
// mismatch. Everything it keeps is a property of the model or the runtime
// signature that the layout does not move: KVPerLayerMB, PromptCache*,
// CheckpointMB, ContextTotalMB and ContextHostMB are per-token or per-entry
// costs. ModelHostMB is dropped, because --n-cpu-moe is exactly what decides
// how much weight stays on the host.

// probeMatchesTopology reports whether a cached entry was produced by the same
// expert layout the caller is planning.
//
// Only a POSITIVE mismatch rejects. An entry with no recorded identity is
// spent as before, because identity is stamped by RecordMeasuredAllocation
// while compute buffers arrive separately from the fit-params oracle: measured
// on this rig, only 6 of 40 entries carrying compute buffers also carry an
// identity. Failing closed on the unidentified 85% would push almost every
// plan back onto the cold estimate -- a larger regression than the collision
// being fixed, and one that would land on first-launch behaviour everywhere.
//
// The remaining gap is therefore coverage, not correctness of this rule: every
// recorder that writes a layout-dependent value should stamp the identity, at
// which point this guard becomes total and unidentified entries can be
// rejected outright. Until then it removes the collisions it can prove.
func probeMatchesTopology(pc *probeCache, s *Strategy) bool {
	if pc == nil || s == nil {
		return true
	}
	if pc.AllocationPlacementIdentity == "" {
		return true // unverifiable, not disproven
	}
	return pc.AllocationPlacementIdentity == AllocationPlacementIdentity(s, nil)
}

// scopeProbeToTopology returns the entry with every layout-dependent value
// dropped unless it was measured under this strategy's own layout.
//
// The returned cache is a shallow copy: callers keep their pointer semantics
// and the on-disk record is untouched, so a measurement stays available to the
// layout that produced it.
func scopeProbeToTopology(pc *probeCache, s *Strategy) *probeCache {
	if pc == nil {
		return nil
	}
	if probeMatchesTopology(pc, s) {
		return pc
	}
	scoped := *pc
	scoped.ComputeBufMB = 0
	scoped.ComputeBufByGPU = nil
	scoped.ComputeBufExpertOnlyByGPU = nil
	scoped.ComputeBufEvidence = ""
	scoped.ContextByGPU = nil
	scoped.ModelByGPU = nil
	scoped.ModelHostMB = 0
	scoped.UnaccountedByGPU = nil
	scoped.UnaccountedHostMB = 0
	scoped.RuntimeGraphGrowthByGPU = nil
	scoped.RuntimeGraphGrowthEstimatedByGPU = nil
	scoped.RuntimeGraphGrowthFromOOMByGPU = nil
	scoped.AllocationEvidence = ""
	scoped.AllocationPlacementIdentity = ""
	scoped.FreeVRAMAtProbe = nil
	return &scoped
}

// describeTopologyScope explains a rejection for a launch banner.
func describeTopologyScope(pc *probeCache, s *Strategy) string {
	if pc == nil || s == nil || probeMatchesTopology(pc, s) {
		return ""
	}
	return fmt.Sprintf(
		"cached per-device measurements were taken under a different expert layout (%s, planning %s); re-measuring",
		shortIdentity(pc.AllocationPlacementIdentity), shortIdentity(AllocationPlacementIdentity(s, nil)))
}

func shortIdentity(id string) string {
	if len(id) > 12 {
		return id[:12]
	}
	return id
}
