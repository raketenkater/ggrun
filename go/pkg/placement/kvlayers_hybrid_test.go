package placement

import "testing"

// Qwen3.8-27B-UD-Q5_K_S on ik_llama.cpp, --tensor-split 0.27,0.58,0.15,
// --ctx-size 65536, q8_0, flash attention, 1 and 4 slots.
func qwen38HybridProfile() *ModelProfile {
	return &ModelProfile{
		ModelArch: "qwen35", NumLayers: 65, NextNPredictLayers: 1, HasSSM: 1, FullAttnInterval: 4,
		HeadCountKV: 4, KeyLength: 256, ValueLength: 256,
		SSMConvKernel: 4, SSMStateSize: 128, SSMGroupCount: 16, SSMInnerSize: 6144, SSMTimeStepRank: 48,
	}
}

func TestDeltaNetHybridKVMatchesTheBackendTotal(t *testing.T) {
	model := qwen38HybridProfile()
	// Sums of the backend's per-device "KV buffer size" lines. Which device
	// holds which block is the separate layer-split question.
	for _, tc := range []struct {
		slots    int
		measured float64
	}{
		{1, 726.76 + 1317.52 + 281.35},
		{4, 867.03 + 1598.07 + 309.41},
	} {
		got, ok := kvLayerTotalMB(model, kvCacheShape{Context: 65536, Slots: tc.slots, KVType: "q8_0"})
		if !ok || float64(got) < tc.measured || float64(got) > tc.measured+2 {
			t.Errorf("slots=%d: priced %d MiB (ok=%v), backend allocated %.2f", tc.slots, got, ok, tc.measured)
		}
	}
	// The context fit prices one slot: attention ("KV self size = 2176.00
	// MiB") plus one slot of recurrent state.
	if got := computeKVTotalMB(model, 65536, "q8_0", false); got < 2325 || got > 2327 {
		t.Fatalf("model-wide KV %d MiB, want 2176 attention + 149.63 state", got)
	}
}

// The recurrent state is per slot and does not grow with context.
func TestDeltaNetHybridRecurrentStateScalesWithSlotsOnly(t *testing.T) {
	model := qwen38HybridProfile()
	at := func(ctx, slots int) int {
		mb, ok := kvLayerTotalMB(model, kvCacheShape{Context: ctx, Slots: slots, KVType: "q8_0"})
		if !ok {
			t.Fatal("delta-net hybrid layout not recognised")
		}
		return mb
	}
	if d := at(65536, 4) - at(65536, 1); d < 448 || d > 450 {
		t.Fatalf("3 extra slots added %d MiB, backend measured 448.88", d)
	}
	if d := at(131072, 1) - at(65536, 1); d != 2176 {
		t.Fatalf("65,536 more tokens added %d MiB, want 2176 of attention KV", d)
	}
}

// Missing geometry (an older parse_gguf.py) keeps the previous path.
func TestDeltaNetHybridWithoutSSMGeometryKeepsTheScalarPath(t *testing.T) {
	model := qwen38HybridProfile()
	model.SSMInnerSize = 0
	if _, ok := modelKVLayerLayout(model); ok {
		t.Fatal("a hybrid without SSM geometry must not be priced per block")
	}
}
