package placement

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// TestHealthyServeRetiresAnOOMSizedReserve is the regression this exists for.
// Measured 2026-09-04 on GLM 5.3 Flash: CUDA0 carried 4007 MiB recorded from a
// cudaMalloc failure while a healthy serve of the plan actually under
// consideration measured 334 MiB on the same device. Both are exact sizes, so
// the estimated flag could not separate them, and the plain maximum kept the
// OOM figure -- roughly 3.9 GB withheld on that card alone.
func TestHealthyServeRetiresAnOOMSizedReserve(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 12282}}
	model := &ModelProfile{Path: "GLM.gguf", NumLayers: 45}

	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 287744, 512, "q8_0", "gpu", "llama",
		gpus, 1, 0, 4007, false); err != nil {
		t.Fatalf("seed OOM growth: %v", err)
	}
	if got := RelatedModelRuntimeGraphGrowth(dir, model, gpus, 1, "llama"); got[0] != 4007 {
		t.Fatalf("precondition: OOM growth should carry when nothing better exists, got %v", got)
	}

	// A launch that reached a serving state measures the same device far lower.
	if err := RecordRuntimeGraphGrowth(dir, model, 287744, 256, "q8_0", "gpu", "llama",
		gpus, 1, map[int]int{0: 334}); err != nil {
		t.Fatalf("record serve growth: %v", err)
	}
	got := RelatedModelRuntimeGraphGrowth(dir, model, gpus, 1, "llama")
	if got[0] != 334 {
		t.Fatalf("a healthy serve must retire the OOM-sized reserve: got %d, want 334", got[0])
	}
}

// TestOOMGrowthStillCarriesForUnmeasuredDevices keeps the change from throwing
// away evidence: retiring an OOM figure is only justified for a device a
// healthy launch has actually measured. Every other device keeps its OOM
// reserve.
func TestOOMGrowthStillCarriesForUnmeasuredDevices(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{
		{Index: 0, VRAMTotalMB: 12282},
		{Index: 1, VRAMTotalMB: 24564},
	}
	model := &ModelProfile{Path: "GLM.gguf", NumLayers: 45}

	for dev, mb := range map[int]int{0: 4007, 1: 3000} {
		if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 287744, 512, "q8_0", "gpu", "llama",
			gpus, 1, dev, mb, false); err != nil {
			t.Fatalf("seed OOM growth for CUDA%d: %v", dev, err)
		}
	}
	// Only CUDA0 is measured by a healthy serve.
	if err := RecordRuntimeGraphGrowth(dir, model, 287744, 256, "q8_0", "gpu", "llama",
		gpus, 1, map[int]int{0: 334}); err != nil {
		t.Fatalf("record serve growth: %v", err)
	}
	got := RelatedModelRuntimeGraphGrowth(dir, model, gpus, 1, "llama")
	if got[0] != 334 {
		t.Fatalf("CUDA0 should use its serve measurement, got %d", got[0])
	}
	if got[1] != 3000 {
		t.Fatalf("CUDA1 has no healthy measurement and must keep its OOM reserve, got %d", got[1])
	}
}

// TestOOMGrowthNeverRaisesAServeMeasurement pins the other direction. A later
// crash under some other plan must not re-inflate a device whose serving cost
// is known, or the reserve ratchets up forever and the retirement is pointless.
//
// NOTE (2026-09-07): this rule is why a cudaGraphInstantiate abort cannot
// currently correct an insufficient reserve -- the abort names no size, so it
// arrives estimated and is refused. Fixing that needs a signal this merge does
// not have: whether the crash occurred while the device was actually holding
// `prior`. Widening the rule to "any OOM may raise" reintroduces the estimate
// compounding it was written to prevent.
func TestOOMGrowthNeverRaisesAServeMeasurement(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 12282}}
	model := &ModelProfile{Path: "GLM.gguf", NumLayers: 45}

	if err := RecordRuntimeGraphGrowth(dir, model, 287744, 256, "q8_0", "gpu", "llama",
		gpus, 1, map[int]int{0: 334}); err != nil {
		t.Fatalf("record serve growth: %v", err)
	}
	// Same exact signature, now an OOM claims a much larger size.
	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 287744, 256, "q8_0", "gpu", "llama",
		gpus, 1, 0, 4007, false); err != nil {
		t.Fatalf("record OOM growth: %v", err)
	}
	exact := RuntimeGraphGrowthByGPU(dir, model, 287744, 256, "q8_0", "gpu", "llama", gpus, 1)
	if exact[0] != 334 {
		t.Fatalf("an OOM must not raise a serving measurement for the same signature: got %d, want 334", exact[0])
	}
}

// TestEstimatedGrowthRemainsWeakestEvidence keeps the pre-existing rule intact:
// a guessed fraction of the card must still lose to either kind of measurement.
func TestEstimatedGrowthRemainsWeakestEvidence(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 12282}}
	model := &ModelProfile{Path: "GLM.gguf", NumLayers: 45}

	// estimated=true: a fraction of the card, not an observation of it.
	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 287744, 256, "q8_0", "gpu", "llama",
		gpus, 1, 0, 6000, true); err != nil {
		t.Fatalf("seed estimated growth: %v", err)
	}
	if err := RecordRuntimeGraphGrowth(dir, model, 287744, 256, "q8_0", "gpu", "llama",
		gpus, 1, map[int]int{0: 334}); err != nil {
		t.Fatalf("record serve growth: %v", err)
	}
	exact := RuntimeGraphGrowthByGPU(dir, model, 287744, 256, "q8_0", "gpu", "llama", gpus, 1)
	if exact[0] != 334 {
		t.Fatalf("a measurement must replace a guess even downwards: got %d, want 334", exact[0])
	}
	// And an estimate must never be carried across signatures at all.
	if got := RelatedModelRuntimeGraphGrowth(dir, model, gpus, 1, "llama"); got[0] != 334 {
		t.Fatalf("related lookup should carry the measurement, got %v", got)
	}
}

// TestGrowthEvidenceRankOrdersTheThreeSources documents the ladder directly.
func TestGrowthEvidenceRankOrdersTheThreeSources(t *testing.T) {
	estimate := growthEvidenceRank(true, false)
	oom := growthEvidenceRank(false, true)
	serve := growthEvidenceRank(false, false)
	if !(estimate < oom && oom < serve) {
		t.Fatalf("expected estimate < oom < serve, got %d, %d, %d", estimate, oom, serve)
	}
	// An estimate read from an OOM is still an estimate.
	if growthEvidenceRank(true, true) != estimate {
		t.Fatal("a guessed size must rank as an estimate regardless of where it came from")
	}
}

func TestCrashNeverLowersAReserve(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{{Index: 1, VRAMTotalMB: 24564}}
	model := &ModelProfile{Path: "GLM.gguf", NumLayers: 45}

	if err := RecordRuntimeGraphGrowth(dir, model, 287744, 256, "q8_0", "gpu", "llama",
		gpus, 1, map[int]int{1: 3000}); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 287744, 256, "q8_0", "gpu", "llama",
		gpus, 1, 1, 900, true); err != nil {
		t.Fatalf("record: %v", err)
	}
	got := RuntimeGraphGrowthByGPU(dir, model, 287744, 256, "q8_0", "gpu", "llama", gpus, 1)
	if got[1] != 3000 {
		t.Fatalf("a smaller crash figure must not lower the reserve: got %d, want 3000", got[1])
	}
}
