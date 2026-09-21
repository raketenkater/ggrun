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
// The key does NOT carry bandwidth any more, so it separates these two plans
// through tensorSplit — the split weights are proportional to free VRAM times
// bandwidth, and the caller serialises them into the key. This test therefore
// supplies the split each plan would actually produce, which is what the launch
// path does, rather than a fixed literal that could not differ.
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

// And the plan key must still be stable against noise, or the fix has only moved
// the problem from the probe cache to the placement cache. These are the values
// this rig actually produces (4070 / 3090 Ti / 3060), moved by the observed
// run-to-run spread.
func TestPlacementKeySurvivesBandwidthNoise(t *testing.T) {
	model := &ModelProfile{Path: "/models/qwen.gguf", NumLayers: 48, NumExperts: 512, EmbeddingLength: 2560}
	// One fixed directory: t.TempDir() inside the closure would make every call
	// differ by path alone and the comparison would prove nothing.
	dir := t.TempDir()
	args := func(gpus []detect.GPU) string {
		return PlacementCachePathFor(dir, model, 262144, 256, "q8_0", "gpu", "llama", gpus, 1, "0.29,0.63,0.08", false)
	}

	base := args(fitBox(12192, 12321, 6269))
	jittered := args(fitBox(12210, 12305, 6255))
	if base != jittered {
		t.Error("plan key moved on measurement noise this rig produces: " +
			"a re-measured link would discard a still-correct plan")
	}
}

// The edge case that a coarse CLASS could not survive, and the reason the class
// was removed rather than widened.
//
// A 100 MB/s class grid has an edge every 100 MB/s, so a value sitting on one
// changed class on a 1 MB/s move: 12,149 classified as 121 and 12,150 as 122.
// That minted a new plan key for a plan that had not changed. Measured directly
// before the removal, orderGPUsByBandwidth returned the identical order [1 0 2]
// for 12,149 and 12,151 — the key moved while the placement did not.
//
// No bucket width fixes this in general: any grid has edges, and the rig's
// smallest real gap (129 MB/s) bounds how coarse the bucket may be. The plan key
// does not need the class because it already carries tensorSplit, which is the
// quantity a bandwidth change actually moves.
func TestPlacementKeyIsStableAcrossAClassBoundary(t *testing.T) {
	model := &ModelProfile{Path: "/models/qwen.gguf", NumLayers: 48, NumExperts: 512, EmbeddingLength: 2560}
	dir := t.TempDir()
	args := func(gpus []detect.GPU) string {
		return PlacementCachePathFor(dir, model, 262144, 256, "q8_0", "gpu", "llama", gpus, 1, "0.29,0.63,0.08", false)
	}

	// The exact pair the goal handoff cites as a boundary failure.
	if a, b := args(fitBox(12149, 12321, 6269)), args(fitBox(12151, 12321, 6269)); a != b {
		t.Error("a 2 MB/s change moved the plan key — a class boundary is still " +
			"in the key, and a re-measurement near it would discard a correct plan")
	}
	// And a value sitting exactly on the old grid edge.
	if a, b := args(fitBox(12150, 12321, 6269)), args(fitBox(12149, 12321, 6269)); a != b {
		t.Error("a 1 MB/s change moved the plan key at a grid edge")
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
