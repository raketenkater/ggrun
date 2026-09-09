package placement

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// TestLedgerUsesMeasuredGrowthAcrossTheUBatchLadder is the regression this
// guards. Growth is recorded against the exact runtime signature that produced
// it, but the ubatch ladder descends after the measurement: measured 2026-09-04
// on GLM 5.3 Flash, the serve recorded growth at ubatch 256 and the winning plan
// replanned to ubatch 128, whose probe row carries compute buffers and no
// growth. The exact lookup then read zero and the ledger reserved nothing for
// the one quantity no no-alloc oracle predicts.
func TestLedgerUsesMeasuredGrowthAcrossTheUBatchLadder(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{
		{Index: 0, VRAMTotalMB: 12282},
		{Index: 1, VRAMTotalMB: 24564},
	}
	model := &ModelProfile{Path: "GLM.gguf", NumLayers: 45, SizeBytes: 8 << 30}

	// A serve measured at ubatch 256 recorded exact growth for both devices.
	if err := RecordRuntimeGraphGrowth(dir, model, 287744, 256, "q8_0", "gpu", "llama",
		gpus, 1, map[int]int{0: 334, 1: 1650}); err != nil {
		t.Fatalf("seed measured growth: %v", err)
	}
	// The ladder then plans at ubatch 128, a signature with no growth row.
	s := &Strategy{
		ContextSize: 287744, UBatchSize: 128, KVQuality: "q8_0", KVPlacement: "gpu",
		KVType: "q8_0", Parallel: 1, TensorSplit: []float64{0.5, 0.5},
	}
	opts := Options{CacheDir: dir, BackendTag: "llama", Parallel: 1}

	related := RelatedModelRuntimeGraphGrowth(dir, model, gpus, 1, "llama")
	if related[0] != 334 || related[1] != 1650 {
		t.Fatalf("precondition: measured growth must be transferable, got %v", related)
	}

	ledger := BuildResourceLedger(&detect.Capabilities{GPUs: gpus}, model, s, opts)
	byGPU := map[int]DeviceResourceLedger{}
	for _, d := range ledger.Devices {
		byGPU[d.GPU] = d
	}
	if got := byGPU[0].RuntimeMB; got != 334 {
		t.Fatalf("CUDA0 must reserve its measured growth across the ladder: got %d, want 334", got)
	}
	if got := byGPU[1].RuntimeMB; got != 1650 {
		t.Fatalf("CUDA1 must reserve its measured growth across the ladder: got %d, want 1650", got)
	}
}

// TestLedgerGrowthFallbackNeverLowersAnExactReserve keeps the fallback
// one-directional. A device that has an exact growth row for this very
// signature must keep it: a neighbouring signature's figure is weaker evidence
// and must never replace, or lower, the reserve measured for this plan.
func TestLedgerGrowthFallbackNeverLowersAnExactReserve(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 12282}}
	model := &ModelProfile{Path: "GLM.gguf", NumLayers: 45, SizeBytes: 8 << 30}

	// Exact row for the planned signature is LARGER than the neighbour's.
	if err := RecordRuntimeGraphGrowth(dir, model, 287744, 128, "q8_0", "gpu", "llama",
		gpus, 1, map[int]int{0: 4007}); err != nil {
		t.Fatalf("seed exact growth: %v", err)
	}
	if err := RecordRuntimeGraphGrowth(dir, model, 287744, 256, "q8_0", "gpu", "llama",
		gpus, 1, map[int]int{0: 334}); err != nil {
		t.Fatalf("seed neighbour growth: %v", err)
	}
	s := &Strategy{
		ContextSize: 287744, UBatchSize: 128, KVQuality: "q8_0", KVPlacement: "gpu",
		KVType: "q8_0", Parallel: 1, TensorSplit: []float64{1.0},
	}
	opts := Options{CacheDir: dir, BackendTag: "llama", Parallel: 1}

	ledger := BuildResourceLedger(&detect.Capabilities{GPUs: gpus}, model, s, opts)
	for _, d := range ledger.Devices {
		if d.GPU == 0 && d.RuntimeMB != 4007 {
			t.Fatalf("the exact reserve for this signature must win, got %d, want 4007", d.RuntimeMB)
		}
	}
}

// TestLedgerGrowthFallbackStaysZeroWithoutEvidence pins the no-history case: a
// deployment that has never measured growth must not acquire a reserve out of
// nowhere, or the fallback becomes the static margin invariant 8 forbids.
func TestLedgerGrowthFallbackStaysZeroWithoutEvidence(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 12282}}
	model := &ModelProfile{Path: "GLM.gguf", NumLayers: 45, SizeBytes: 8 << 30}
	s := &Strategy{
		ContextSize: 287744, UBatchSize: 128, KVQuality: "q8_0", KVPlacement: "gpu",
		KVType: "q8_0", Parallel: 1, TensorSplit: []float64{1.0},
	}
	opts := Options{CacheDir: dir, BackendTag: "llama", Parallel: 1}

	ledger := BuildResourceLedger(&detect.Capabilities{GPUs: gpus}, model, s, opts)
	for _, d := range ledger.Devices {
		if d.RuntimeMB != 0 {
			t.Fatalf("an unmeasured deployment must reserve no growth, got %d", d.RuntimeMB)
		}
	}
}
