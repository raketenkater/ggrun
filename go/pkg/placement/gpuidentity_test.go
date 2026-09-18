package placement

import (
	"strings"
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
// entirely: a real bandwidth change must still move a PLAN key. Bandwidth orders
// devices for packing (orderGPUsByBandwidth), so a plan computed for a 1 GB/s
// link is not a plan for a 15 GB/s one.
func TestPlacementKeyStillSeparatesARealBandwidthChange(t *testing.T) {
	model := &ModelProfile{Path: "/models/qwen.gguf", NumLayers: 48, NumExperts: 512, EmbeddingLength: 2560}
	dir := t.TempDir()
	args := func(gpus []detect.GPU) string {
		return PlacementCachePathFor(dir, model, 262144, 256, "q8_0", "gpu", "llama", gpus, 1, "0.29,0.63,0.08", false)
	}

	fast := args(fitBox(12192, 12321, 6269))
	// CUDA2 degraded from x8 Gen3 to a genuinely narrower link.
	slow := args(fitBox(12192, 12321, 1200))

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

// The class must separate every distinct link on this rig, including the two
// Gen3 x16 cards whose 129 MB/s gap is the tightest real margin. A bucket coarser
// than that gap would merge the 4070 and the 3090 Ti and lose their ordering.
func TestBandwidthClassSeparatesEveryRealLink(t *testing.T) {
	links := map[string]int{
		"4070 Gen3 x16":    12192,
		"3090 Ti Gen3 x16": 12321,
		"3060 Gen3 x8":     6269,
		"theoretical x16":  15760,
		"theoretical x8":   7880,
	}
	seen := map[string]string{}
	for label, mbps := range links {
		c := bandwidthClassMBps(mbps)
		if other, dup := seen[c]; dup {
			t.Errorf("%s (%d) and %s share bandwidth class %s: "+
				"their ordering would be lost", label, mbps, other, c)
		}
		seen[c] = label
	}
}

// Missing evidence is its own class, never folded into a measured bucket: an
// unmeasured device must not inherit a plan key from a measured slow one.
func TestUnknownBandwidthDoesNotCollideWithAMeasuredLink(t *testing.T) {
	if bandwidthClassMBps(0) == bandwidthClassMBps(1000) {
		t.Error("an unmeasured link shares a plan key with a measured one")
	}
	if !strings.Contains(bandwidthClassMBps(0), "unknown") {
		t.Errorf("unknown bandwidth should be labelled, got %q", bandwidthClassMBps(0))
	}
	// A plausible low measurement must not be mistaken for absence either.
	if bandwidthClassMBps(0) == bandwidthClassMBps(500) {
		t.Error("an unmeasured link collides with a slow measured link")
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
