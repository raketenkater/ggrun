package placement

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// The GPU identity hash is used four different ways: as a filename for the
// system probe, as a cross-key match filter for runtime graph growth, inside the
// .probe key, and inside the .place key. Three of those four hold measurements
// that have nothing to do with link bandwidth.
//
// Until placementProbeCacheVersion 9 all four also hashed the measured link
// bandwidth at integer precision. Measured bandwidth is not stable: re-running
// detect on this project's three-card rig moved it +2 / +3 / -1 MB/s on values of
// ~12,190 / ~12,318 / ~6,270. Each of those single-digit wobbles minted a new
// signature, so every cached measurement written under the previous one became
// unreachable. On 2026-09-17 the store held 643 probes and 535 of them were
// stranded that way.
//
// The consequence was not theoretical. The live Qwen3.8-Flash-Next server started
// 2026-09-16 09:13:31; the first probe under the new signature was written
// 09:15:02. It launched with no usable allocation evidence, fell back to the
// conservative cold-start path, and emitted --n-cpu-moe 29 where the same
// coordinates had planned 25 the day before. Four expert layers, silently, with
// no warning.
//
// These tests pin the boundary: a re-measurement must not move a fit key, and a
// genuinely different link must still move a plan key.

// fitBox is the identity fixture. Bandwidth is varied deliberately by the caller.
func fitBox(bandwidthMBps ...int) []detect.GPU {
	gpus := []detect.GPU{
		{Index: 0, Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12282, Driver: "580",
			ComputeCap: "8.9", PCIBusID: "00000000:17:00.0", PCIGen: 3, PCILanes: 16},
		{Index: 1, Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24564, Driver: "580",
			ComputeCap: "8.6", PCIBusID: "00000000:65:00.0", PCIGen: 3, PCILanes: 16},
		{Index: 2, Name: "NVIDIA GeForce RTX 3060", VRAMTotalMB: 12288, Driver: "580",
			ComputeCap: "8.6", PCIBusID: "00000000:B3:00.0", PCIGen: 3, PCILanes: 8},
	}
	for i, bw := range bandwidthMBps {
		if i < len(gpus) {
			gpus[i].BandwidthMBps = bw
		}
	}
	return gpus
}

// The defect in one assertion: the same hardware, measured twice, must keep the
// same fit key.
func TestProbeCacheSurvivesABandwidthRemeasurement(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "/models/qwen.gguf", NumLayers: 48, NumExperts: 512, EmbeddingLength: 2560}

	measured := fitBox(12192, 12321, 6269)
	if err := writeProbeCacheForModel(dir, model, 262144, 256, "q8_0", "gpu", "llama", measured, 1,
		map[int]int{0: 1159, 1: 1036, 2: 828}, map[int]int{0: 5504}, nil, 0); err != nil {
		t.Fatal(err)
	}

	// The second detection run, on the identical machine, one to three MB/s off.
	remeasured := fitBox(12194, 12318, 6270)
	got := loadProbeCache(dir, model, 262144, 256, "q8_0", "gpu", "llama", remeasured, 1)

	if got == nil {
		t.Fatalf("re-measuring link bandwidth orphaned the probe cache: "+
			"identity %s vs %s", gpuIdentityHash(measured), gpuIdentityHash(remeasured))
	}
	if got.ComputeBufByGPU[0] != 1159 || got.ComputeBufByGPU[1] != 1036 || got.ComputeBufByGPU[2] != 828 {
		t.Errorf("probe survived but its per-device allocations changed: %#v", got.ComputeBufByGPU)
	}
	if got.RuntimeGraphGrowthByGPU[0] != 5504 {
		t.Errorf("probe survived but its growth figure changed: %#v", got.RuntimeGraphGrowthByGPU)
	}
}

// The inverse, so the test above cannot pass by making the hash ignore hardware
// entirely: a real link change must still move a PLAN key.
//
// This supplies the split each plan would produce, which is what the SAVE side
// has. It cannot guard the LOOKUP side: the launch computes the key before any
// split exists, so this would still pass if the bandwidth class were removed.
// TestPlanKeySeparatesALinkChangeWithoutATensorSplit drives the empty split the
// lookup actually has and is the test that catches that regression.
func TestPlacementKeyStillSeparatesARealBandwidthChange(t *testing.T) {
	model := &ModelProfile{Path: "/models/qwen.gguf", NumLayers: 48, NumExperts: 512, EmbeddingLength: 2560}
	dir := t.TempDir()
	args := func(gpus []detect.GPU, split string) string {
		return PlacementCachePathFor(dir, model, 262144, 256, "q8_0", "gpu", "llama", gpus, 1, split, false)
	}

	gpus := fitBox(12192, 12321, 6269)
	// Same devices, a genuinely narrower link on the third card. The placement
	// weight is the bandwidth proportion the split is built from, so it moves even
	// though the ORDER does not.
	fast := args(gpus, splitCompactKey([]float64{
		gpuPlacementWeight(gpus[0], "link"), gpuPlacementWeight(gpus[1], "link"), gpuPlacementWeight(gpus[2], "link")}))
	slowG := fitBox(12192, 12321, 1200)
	slow := args(slowG, splitCompactKey([]float64{
		gpuPlacementWeight(slowG[0], "link"), gpuPlacementWeight(slowG[1], "link"), gpuPlacementWeight(slowG[2], "link")}))

	if fast == slow {
		t.Error("a real link change did not move the plan key: " +
			"a plan packed for a fast link would be reused for a slow one")
	}
}

// NOTE: a TestPlacementKeySurvivesBandwidthNoise lived here and was VACUOUS.
// It called PlacementCachePathFor with a hardcoded split and two fixtures that
// differed only in BandwidthMBps, so both calls produced byte-identical keys
// whether or not the key carried a bandwidth term — it could not fail, and a
// second senior review caught it. The property it meant to assert (the key must
// not flap on measurement noise) is covered properly by
// TestPlanKeySeparatesALinkChangeWithoutATensorSplit below, which drives an
// EMPTY split the way the real lookup path does and asserts both halves:
// stable under +-3 MB/s, and different for a real link change.

// What a class boundary actually costs, measured rather than asserted.
//
// A test with this name used to live here and was MISLEADING. It called
// PlacementCachePathFor with a hardcoded split ("0.29,0.63,0.08") and asserted
// the key did not move across a boundary. But the launch path computes the key
// BEFORE the split exists (placement.go:1786-1787, against a fresh Strategy), so
// the split component is a constant at lookup time — 9be1ec2 found that and
// restored the class for exactly that reason. Handing the function a split is
// therefore the one input the real lookup cannot have, and the test passed for a
// reason that does not hold where it matters.
//
// The property that DOES hold, and the one worth pinning, is a split of
// responsibilities between the two keys:
//
//   - the PROBE key carries gpuIdentityHash only (probeCachePath), so a
//     bandwidth wobble cannot strand a measurement. This is the guarantee the
//     branch exists for, and it is asserted below across a real boundary.
//   - the PLAN key also carries bandwidthRatioClass, so it can move at an edge.
//     The cost is one recompute from probes that are still there — not lost
//     evidence. Measured on this rig's own cards, 6,098 and 6,099 MB/s sit on
//     either side of the 49.5% edge and mint different .place keys while
//     producing an IDENTICAL fresh plan (same split, same n-cpu-moe). So the
//     churn is real and buys nothing, but it is safe, and removing the class is
//     not the fix: 97404a3 tried that and left the key with no bandwidth signal
//     at all, which would reuse a plan packed for a fast link on a slow one.
//
// Recorded as an accepted cost with a test rather than left as an unverified
// claim in a commit message.
func TestProbeKeySurvivesAClassBoundaryThatMovesThePlanKey(t *testing.T) {
	model := &ModelProfile{Path: "/models/qwen.gguf", NumLayers: 48, NumExperts: 512, EmbeddingLength: 2560}
	dir := t.TempDir()
	// The lookup path's own inputs: an EMPTY split, because the key is computed
	// before any split exists.
	lookupKey := func(gpus []detect.GPU) string {
		return PlacementCachePathFor(dir, model, 262144, 256, "q8_0", "gpu", "llama", gpus, 1,
			splitCompactKey(nil), false)
	}
	probeKey := func(gpus []detect.GPU) string {
		return probeCachePath(dir, model, 262144, 256, "q8_0", "gpu", "llama", gpus, 1)
	}

	// This rig's third card measures ~6,270 MB/s and the ratio edge sits at
	// 49.5% of the 3090 Ti's ~12,320, so 6,098 and 6,099 straddle it. That is a
	// real coincidence of this hardware, not a contrived pair.
	lo, hi := fitBox(12192, 12321, 6098), fitBox(12192, 12321, 6099)
	if bandwidthRatioClass(lo) == bandwidthRatioClass(hi) {
		t.Fatalf("fixture no longer straddles a class boundary: %s == %s",
			bandwidthRatioClass(lo), bandwidthRatioClass(hi))
	}

	// The guarantee: allocation/geometry evidence is NOT keyed by link
	// bandwidth, so a re-measurement across the edge cannot strand it.
	if probeKey(lo) != probeKey(hi) {
		t.Error("a bandwidth measurement crossing a class boundary moved the PROBE " +
			"key: a re-measurement would strand allocation evidence, which is the " +
			"defect this branch exists to remove")
	}

	// The accepted cost, pinned so it cannot regress silently in either
	// direction: the plan key does move here.
	if lookupKey(lo) == lookupKey(hi) {
		t.Error("the plan key no longer separates a class boundary. That is only " +
			"safe if a bandwidth signal survives somewhere else in the key — " +
			"97404a3 removed the class on the belief that tensorSplit covered it, " +
			"and 9be1ec2 showed the split is a constant at lookup time, so the key " +
			"was left with no bandwidth signal at all")
	}
}

// The churn above is only acceptable because it costs a recompute, not a
// measurement: the fresh plan must come out the same on both sides of the edge.
//
// This is the half that makes the boundary a nuisance rather than a defect, and
// it was previously only asserted in prose. If a boundary crossing ever changed
// the PACKING, the same re-measurement that today costs a recompute would start
// changing where expert layers land.
func TestAClassBoundaryDoesNotChangeTheFreshPlan(t *testing.T) {
	model := &ModelProfile{
		Path: "m.gguf", SizeBytes: 30 * 1024 * 1024 * 1024,
		NumLayers: 48, NumExperts: 256, EmbeddingLength: 2560,
		FeedForwardLength: 9728, IsMoE: true, ContextSize: 32768,
	}
	plan := func(bw int) *Strategy {
		gpus := fitBox(12192, 12321, bw)
		for i := range gpus {
			gpus[i].VRAMUsedMB = 0
		}
		caps := &detect.Capabilities{
			GPUs: gpus, RAM: detect.RAMInfo{TotalMB: 131072},
			CPU: detect.CPUInfo{Cores: 14},
		}
		s, err := Compute(caps, model, Options{KVPlacement: "gpu", KVQuality: "mid", ContextSize: 32768})
		if err != nil {
			t.Fatalf("compute at %d MB/s: %v", bw, err)
		}
		return s
	}

	lo, hi := plan(6098), plan(6099)
	if bandwidthRatioClass(fitBox(12192, 12321, 6098)) == bandwidthRatioClass(fitBox(12192, 12321, 6099)) {
		t.Fatal("fixture no longer straddles a class boundary")
	}
	if lo.NCPUMoE != hi.NCPUMoE {
		t.Errorf("a class boundary changed the expert offload: n-cpu-moe %d -> %d. "+
			"A re-measurement that crosses the edge would then change where expert "+
			"layers are packed, not merely force a recompute", lo.NCPUMoE, hi.NCPUMoE)
	}
	if splitCompactKey(lo.TensorSplit) != splitCompactKey(hi.TensorSplit) {
		t.Errorf("a class boundary changed the tensor split: %v -> %v",
			lo.TensorSplit, hi.TensorSplit)
	}
}

// The class was removed because tensorSplit already separates plans whose links
// differ enough to matter. This pins that reasoning at the level that actually
// decides it.
//
// The real split is `effective_free_VRAM * bandwidth / total` (placement.go:2581),
// a NORMALISED proportion, and it is serialised into the key at two decimals by
// splitCompactKey. Normalisation is what makes it stable: the observed run-to-run
// spread (+2 / -3 / +1 MB/s here) moves each share by far less than 0.005, so the
// two-decimal string is unchanged. A genuinely narrower link moves it enough to
// show. This helper reproduces the real formula rather than supplying raw weights,
// because raw weights are NOT what the key carries.
func TestBandwidthMovesTheSplitWeightsNotJustTheOrder(t *testing.T) {
	effectiveVRAM := []float64{12282, 24564, 12288}
	splitKey := func(gpus []detect.GPU) string {
		var total float64
		for i, g := range gpus {
			total += effectiveVRAM[i] * gpuPlacementWeight(g, "link")
		}
		share := make([]float64, len(gpus))
		for i, g := range gpus {
			share[i] = effectiveVRAM[i] * gpuPlacementWeight(g, "link") / total
		}
		return splitCompactKey(share)
	}

	fast := splitKey(fitBox(12192, 12321, 6269))
	if slow := splitKey(fitBox(12192, 12321, 1200)); slow == fast {
		t.Error("a genuinely slower third card did not move the split shares: " +
			"the plan would be reused for a different topology")
	}
	// The observed run-to-run spread must not move it, or the key is still
	// unstable and the fix has only relocated the defect.
	if jit := splitKey(fitBox(12194, 12318, 6270)); jit != fast {
		t.Errorf("measurement noise moved the split shares: %s vs %s", fast, jit)
	}
}

// The stable identity must not move when bandwidth does, on any card, including
// one whose link class changes. This is the property the .probe key depends on.
func TestIdentityHashIgnoresBandwidthEntirely(t *testing.T) {
	unmeasured := fitBox()
	measured := fitBox(12192, 12321, 6269)

	if gpuIdentityHash(unmeasured) != gpuIdentityHash(measured) {
		t.Error("gpuIdentityHash moved when only bandwidth was set: " +
			"fit evidence would still be orphaned by a re-measurement")
	}
}

// Determinism and discrimination, so the derivation change cannot quietly make
// the key order-dependent or constant. Mirrors the intent of
// TestPlacementCachePathFor_KeyedByKVAndCtx.
func TestIdentityHashIsDeterministicAndDiscriminating(t *testing.T) {
	gpus := fitBox(12192, 12321, 6269)
	if a, b := gpuIdentityHash(gpus), gpuIdentityHash(gpus); a != b {
		t.Errorf("identity hash is not deterministic: %s != %s", a, b)
	}

	// Input order must not matter (the hash sorts its parts).
	shuffled := []detect.GPU{gpus[2], gpus[0], gpus[1]}
	if gpuIdentityHash(gpus) != gpuIdentityHash(shuffled) {
		t.Error("identity hash depends on the order the GPUs were detected in")
	}

	// A capacity change is real hardware and must separate.
	bigger := fitBox(12192, 12321, 6269)
	bigger[0].VRAMTotalMB = 24576
	if gpuIdentityHash(gpus) == gpuIdentityHash(bigger) {
		t.Error("a VRAM capacity change did not move the identity hash")
	}

	// So must a link-width change, which is the topology the stable hash keeps.
	narrower := fitBox(12192, 12321, 6269)
	narrower[2].PCILanes = 4
	if gpuIdentityHash(gpus) == gpuIdentityHash(narrower) {
		t.Error("a PCIe width change did not move the identity hash")
	}
}

// The identity must survive a card moving to a different slot.
//
// g.Index is a SLOT ORDINAL: detect assigns it by sorting on PCIBusID
// (detect.go:185-189), so it is already a deterministic function of a field the
// hash keeps. Including it meant the identity changed when a card was reseated —
// or when detection simply ordered the same cards differently — and every cached
// measurement was orphaned again, for no information gained. The hash already
// discarded ORDER by sorting its parts, so Index contributed only itself.
func TestIdentitySurvivesACardMovingSlots(t *testing.T) {
	base := fitBox(12192, 12321, 6269)

	// Same three physical cards, detected in a different order. This is what a
	// reseat, a driver reload, or a machine that enumerates in another order
	// produces.
	reordered := []detect.GPU{base[2], base[0], base[1]}
	for i := range reordered {
		reordered[i].Index = i // detect renumbers from the sort
	}

	if gpuIdentityHash(base) != gpuIdentityHash(reordered) {
		t.Error("rescanning the same cards in a different order changed the " +
			"hardware identity: cached evidence would be orphaned by a reseat")
	}
}

// The identity must still be a function of the physical cards, not of how many
// there are. Dropping Index must not make two different machines collide.
func TestIdentityStillSeparatesDifferentCardSets(t *testing.T) {
	three := fitBox(12192, 12321, 6269)
	// Truncate deliberately: fitBox always builds the full three-card fixture
	// (its variadic argument only sets bandwidths), so a two-card machine has to
	// be expressed by slicing rather than by passing fewer bandwidths.
	two := fitBox(12192, 12321)[:2]

	if gpuIdentityHash(three) == gpuIdentityHash(two) {
		t.Error("a two-card machine shares an identity with a three-card one")
	}
	// A different card in the same slot must separate too.
	swapped := fitBox(12192, 12321, 6269)
	swapped[2].VRAMTotalMB = 8192
	if gpuIdentityHash(three) == gpuIdentityHash(swapped) {
		t.Error("a capacity change did not change the identity")
	}
}

// PCIBusID is the field that now leads the record, and it is `omitempty` on
// detect.GPU — so a hand-written or partial inventory can omit it. That must not
// silently collapse two different cards into one identity.
func TestIdentityWithoutBusIDsStillSeparatesByNameAndCapacity(t *testing.T) {
	a := fitBox(12192, 12321, 6269)
	b := fitBox(12192, 12321, 6269)
	for i := range a {
		a[i].PCIBusID = ""
		b[i].PCIBusID = ""
	}
	// Same cards with no bus id: identity must match (there is nothing else to
	// distinguish them by, and they ARE the same hardware).
	if gpuIdentityHash(a) != gpuIdentityHash(b) {
		t.Error("identical cards without bus ids produced different identities")
	}
	// But a genuinely different card must still separate without bus ids.
	c := fitBox(12192, 12321, 6269)
	for i := range c {
		c[i].PCIBusID = ""
	}
	c[0].Name = "NVIDIA GeForce RTX 5080"
	if gpuIdentityHash(a) == gpuIdentityHash(c) {
		t.Error("a different card collided once bus ids were absent; nothing " +
			"distinguishes the set but name and capacity, so those must carry it")
	}
}

// The plan key is computed BEFORE the tensor split exists, so it cannot rely on
// tensorSplit to separate two links. This is the regression 97404a3 introduced.
//
// The lookup path is:
//
//	1786  cacheSplitKey := splitCompactKey(s.TensorSplit)   // s is fresh: empty
//	1787  s.PlacementCachePath = PlacementCachePathFor(..., cacheSplitKey, ...)
//	1794  if that path exists -> reuse the cached plan
//	1900  s.TensorSplit = normalizeSplit(cache.TensorSplit) // comes FROM the cache
//
// so at lookup time the split component of the key is a constant. Before 97404a3
// the key also carried a bandwidth class, which did differ between links;
// removing it left the key with NO bandwidth signal, and a plan packed for a fast
// link is now returned for a slow one.
//
// The boundary test below therefore drives PlacementCachePathFor the way the
// launch path does — with the split it actually has at that moment, which is
// empty — rather than supplying a split by hand, which is why the older tests
// could not see this.
func TestPlanKeySeparatesALinkChangeWithoutATensorSplit(t *testing.T) {
	model := &ModelProfile{Path: "/models/qwen.gguf", NumLayers: 48, NumExperts: 512, EmbeddingLength: 2560}
	dir := t.TempDir()

	// Empty split: exactly what the lookup path has.
	lookupKey := func(gpus []detect.GPU) string {
		return PlacementCachePathFor(dir, model, 262144, 256, "q8_0", "gpu", "llama", gpus, 1,
			splitCompactKey(nil), false)
	}

	fast := lookupKey(fitBox(12192, 12321, 6269))
	// The 3060 degrades to a genuinely narrower link — a real hardware change
	// that reorders devices and changes what should be packed.
	slow := lookupKey(fitBox(12192, 12321, 1200))
	if fast == slow {
		t.Error("a real link change did not move the plan key at lookup time: " +
			"a plan packed for a fast link would be reused for a slow one, because " +
			"the key carries no bandwidth signal until the split exists")
	}

	// And it must still be stable against the noise this rig produces, or the
	// fix has merely restored the flapping key that was removed.
	if a, b := lookupKey(fitBox(12192, 12321, 6269)), lookupKey(fitBox(12194, 12318, 6270)); a != b {
		t.Error("the plan key flaps on measurement noise again")
	}
}

// An explicit --ctx-checkpoints must reach the argv even when a launch reuses a
// promoted verified config.
//
// The reuse branch restores the saved CRAM and MaxCheckpoints and, before the
// fix, skipped the cache-policy helper entirely because that call is guarded by
// `s.CRAM == 0`. So an override was silently dropped on every launch after a
// config had been promoted — the same "override does nothing on one path" defect
// that collapsing the three derivation sites was meant to remove, relocated to a
// fourth site.
//
// TESTING LIMIT, stated rather than hidden: the reuse path is not reachable from
// an in-package test. It requires opts.VerifiedConfigScopeKey to match the key the
// LAUNCHER builds (main.go:2823), and that key ends with
// planLogicVersion, a constant in cmd/ggrun — not exported to this package. A
// test here can set any key it likes and the lookup will miss, which is how an
// earlier version of this test passed with the fix deleted.
//
// The reuse path is therefore covered end to end by the CLI-level check recorded
// in the ledger (save a config, relaunch with --ctx-checkpoints N, assert N
// reaches the argv). What is pinned here is the resolution RULE the reuse branch
// must follow, so the rule itself cannot regress silently.
func TestExplicitCheckpointOverrideBeatsASavedConfigValue(t *testing.T) {
	model := &ModelProfile{Path: "/models/qwen.gguf", NumLayers: 48, NumExperts: 512, EmbeddingLength: 2560}
	caps := &detect.Capabilities{
		GPUs: fitBox(12192, 12321, 6269),
		RAM:  detect.RAMInfo{TotalMB: 217096, FreeMB: 200000},
		CPU:  detect.CPUInfo{Cores: 14, Threads: 28},
	}

	// The state a reuse hit restores: a complete saved decision with a non-zero
	// CRAM, which is exactly why the helper's `s.CRAM == 0` guard skips it.
	saved := &Strategy{CRAM: 13824, MaxCheckpoints: 16}

	// Explicit override must win over the saved value.
	over := &Strategy{CRAM: saved.CRAM, MaxCheckpoints: saved.MaxCheckpoints}
	applyRuntimeCachePolicy(model, over, caps, 90000, 4000,
		Options{MaxCheckpoints: 32, MaxCheckpointsSet: true})
	if over.MaxCheckpoints != 32 {
		t.Errorf("MaxCheckpoints = %d, want the explicit 32: an override must "+
			"outrank a value saved by an earlier launch", over.MaxCheckpoints)
	}

	// An explicit 0 (disable) is a real decision and must also win.
	off := &Strategy{CRAM: saved.CRAM, MaxCheckpoints: saved.MaxCheckpoints}
	applyRuntimeCachePolicy(model, off, caps, 90000, 4000,
		Options{MaxCheckpoints: 0, MaxCheckpointsSet: true})
	if off.MaxCheckpoints != 0 {
		t.Errorf("MaxCheckpoints = %d, want 0: --ctx-checkpoints 0 asks to disable "+
			"checkpoints and was ignored", off.MaxCheckpoints)
	}

	// Without the setter flag the field's zero value must NOT clobber anything,
	// or every launch would silently disable checkpoints.
	unset := &Strategy{CRAM: saved.CRAM, MaxCheckpoints: saved.MaxCheckpoints}
	applyRuntimeCachePolicy(model, unset, caps, 90000, 4000, Options{MaxCheckpoints: 32})
	if unset.MaxCheckpoints == 32 {
		t.Error("an unset override still overrode: MaxCheckpointsSet must gate it")
	}
}

// The plan key must not depend on the order detection enumerated the cards in.
//
// bandwidthRatioClass built one class per element in CALLER order and joined with
// "-", so {12192,12321,6269} gave "90-100-50" while the reversed slice gave
// "50-100-90" — a different plan key for the same machine. That is precisely the
// re-orphaning mechanism that dropping g.Index removed, reintroduced one layer up:
// a reseat, a driver reload, or a differently-ordered enumeration would discard a
// still-correct plan. gpuIdentityHash already sorts its parts, and
// TestIdentitySurvivesACardMovingSlots asserts the property for identity; this
// asserts it for the ratio term.
func TestPlanKeyIsInvariantToDeviceEnumerationOrder(t *testing.T) {
	model := &ModelProfile{Path: "/models/qwen.gguf", NumLayers: 48, NumExperts: 512, EmbeddingLength: 2560}
	dir := t.TempDir()
	key := func(gpus []detect.GPU) string {
		// Empty split: what the lookup path actually has.
		return PlacementCachePathFor(dir, model, 262144, 256, "q8_0", "gpu", "llama", gpus, 1,
			splitCompactKey(nil), false)
	}

	base := fitBox(12192, 12321, 6269)
	reversed := []detect.GPU{base[2], base[1], base[0]}
	for i := range reversed {
		reversed[i].Index = i // detection renumbers from its own sort
	}

	if key(base) != key(reversed) {
		t.Error("re-enumerating the same cards in a different order changed the " +
			"plan key: a reseat or driver reload would orphan a still-correct plan")
	}
}

// A plan must not be reused for a machine that differs only in WHICH physical
// card has the fast link.
//
// bandwidthRatioClass used to sort its bare per-card classes to stay independent
// of enumeration order. That reduced a per-card assignment to a multiset, so
// {4070 fast, 3060 slow} and {4070 slow, 3060 fast} produced the same class, the
// same plan key, and — because the offloaded-MoE path reuses a keyed .place file
// wholesale — the same plan. Their fresh plans differ (on this fixture the split
// moves from 0.26/0.61/0.13 to 0.12/0.64/0.24 and n-cpu-moe from 40 to 42), so the
// reused plan packed experts for hardware that is not there.
//
// This drives the real lifecycle rather than comparing keys: Compute, persist the
// plan the way a successful launch does, then Compute again. A sentinel split in
// the saved entry makes reuse unambiguous, and the same-hardware reload asserts
// the cache IS consulted on this path — without that half the test could pass
// on a path that never reads .place, which is exactly how the dense fixture tried
// first behaved.
func TestPlanIsNotReusedWhenTheFastLinkMovesToAnotherCard(t *testing.T) {
	model := &ModelProfile{
		Path:      "/models/DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf",
		SizeBytes: 137903959808, TotalSizeMB: 131515,
		NumLayers: 43, IsMoE: true, NumExperts: 256, ExpertUsedCount: 6, ExpertFF: 2048,
		ExpertBytes: 131240296448, NonExpertBytes: 6658320448,
		TokenEmbdBytes: 562626560, OutputBytes: 434380800, ShexpBytes: 1149763584,
		ContextSize: 262144, CTXTrain: 1048576, HiddenSize: 4096, EmbeddingLength: 4096,
		HeadCountKV: 1, KeyLength: 512, ValueLength: 512, ModelArch: "deepseek4",
		MeasuredKVBytesPerTok: map[string]float64{"f16": 7012},
	}
	caps := func(bw ...int) *detect.Capabilities {
		return &detect.Capabilities{GPUs: fitBox(bw...),
			RAM: detect.RAMInfo{TotalMB: 128730, FreeMB: 123424}, CPU: detect.CPUInfo{Cores: 8}}
	}
	opts := func(dir string) Options {
		return Options{ContextSize: 262144, KVPlacement: "gpu", KVQuality: "high",
			BackendTag: "llama", Parallel: 1, CacheDir: dir}
	}
	sentinel := []float64{0.5, 0.3, 0.2}
	dir := t.TempDir()

	first, err := Compute(caps(12192, 12321, 6269), model, opts(dir))
	if err != nil {
		t.Fatal(err)
	}
	if first.Type != MoEOffload {
		t.Fatalf("fixture must exercise the offloaded-MoE plan cache, got %s", first.Type)
	}
	entry := StrategyToCacheEntry(first)
	entry.TensorSplit = sentinel
	if err := SavePlacementCache(first.PlacementCachePath, entry); err != nil {
		t.Fatal(err)
	}

	// Sanity: on unchanged hardware the saved plan IS reused. If this ever stops
	// holding, the assertion below proves nothing.
	same, err := Compute(caps(12192, 12321, 6269), model, opts(dir))
	if err != nil {
		t.Fatal(err)
	}
	if splitCompactKey(same.TensorSplit) != splitCompactKey(sentinel) {
		t.Fatalf("same-hardware relaunch did not reuse the saved plan (split %v): "+
			"this path no longer reads .place, so the swap check below is vacuous", same.TensorSplit)
	}

	// Same cards, but the 4070 and 3060 swap link speeds.
	swapped, err := Compute(caps(6269, 12321, 12192), model, opts(dir))
	if err != nil {
		t.Fatal(err)
	}
	fresh, err := Compute(caps(6269, 12321, 12192), model, opts(t.TempDir()))
	if err != nil {
		t.Fatal(err)
	}
	if splitCompactKey(fresh.TensorSplit) == splitCompactKey(sentinel) {
		t.Fatal("fixture invalid: the fresh swapped plan equals the sentinel")
	}
	if splitCompactKey(swapped.TensorSplit) != splitCompactKey(fresh.TensorSplit) ||
		swapped.NCPUMoE != fresh.NCPUMoE {
		t.Errorf("moving the fast link to another card reused the saved plan: got split %v "+
			"n-cpu-moe %d, fresh plan for this hardware is split %v n-cpu-moe %d",
			swapped.TensorSplit, swapped.NCPUMoE, fresh.TensorSplit, fresh.NCPUMoE)
	}
}

// Per-device evidence must stay bound to the card that produced it.
//
// Probe rows are stored by device index (PROBED_COMPUTE_BUF_MB_CUDA<i>) under a
// key that sorts per-card identities, so they are only valid while the index is
// a deterministic function of the identity. NVIDIA guarantees that (detect
// renumbers by sorted bus id). Vulkan has no bus id and numbers by enumeration
// order, so two different cards enumerating the other way round used to share an
// identity with swapped indices, and each card inherited the other's measured
// compute buffer. The busy-machine guard (probeMeasuredUnderDuress) only caught
// it when capacities differed, so this uses two equal-capacity cards.
func TestProbeEvidenceIsNotMisappliedWhenBusIDlessCardsReorder(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "/models/qwen.gguf", NumLayers: 48, NumExperts: 512, EmbeddingLength: 2560}
	amd := detect.GPU{Name: "AMD Radeon RX 7800 XT", VRAMTotalMB: 16368, Driver: "2.0.301", ComputeCap: "1.3.280"}
	arc := detect.GPU{Name: "Intel(R) Arc(tm) A770 Graphics", VRAMTotalMB: 16368, Driver: "2.0.301", ComputeCap: "1.3.280"}
	order := func(first, second detect.GPU) []detect.GPU {
		first.Index, second.Index = 0, 1
		return []detect.GPU{first, second}
	}

	written := order(amd, arc)
	if err := writeProbeCacheForModel(dir, model, 32768, 256, "q8_0", "gpu", "llama", written, 1,
		map[int]int{0: 3000, 1: 900}, nil, nil, 0); err != nil {
		t.Fatal(err)
	}

	// Stable enumeration must keep the evidence, or this is just churn.
	if got := loadProbeCache(dir, model, 32768, 256, "q8_0", "gpu", "llama", order(amd, arc), 1); got == nil ||
		got.ComputeBufByGPU[0] != 3000 || got.ComputeBufByGPU[1] != 900 {
		t.Fatalf("an unchanged enumeration lost or altered its evidence: %+v", got)
	}

	// Reordered: the evidence must be invalidated, never re-attached by index.
	if got := loadProbeCache(dir, model, 32768, 256, "q8_0", "gpu", "llama", order(arc, amd), 1); got != nil {
		t.Errorf("a reordered bus-id-less set loaded evidence by index: CUDA0 (%s) got %d MiB and "+
			"CUDA1 (%s) got %d MiB, which were measured on the other card",
			arc.Name, got.ComputeBufByGPU[0], amd.Name, got.ComputeBufByGPU[1])
	}
}

// The ordinal must NOT enter the identity when bus ids already fix the order,
// or a reseat — the case gpuIdentityHash was built to survive — would orphan
// the whole NVIDIA corpus again.
func TestBusIDIdentityStaysOrderFree(t *testing.T) {
	gpus := fitBox(12192, 12321, 6269)
	renumbered := []detect.GPU{gpus[2], gpus[0], gpus[1]}
	for i := range renumbered {
		renumbered[i].Index = i
	}
	if gpuIdentityHash(gpus) != gpuIdentityHash(renumbered) {
		t.Error("with a bus id on every card the identity changed when indices moved")
	}
}
