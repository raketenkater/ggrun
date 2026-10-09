package placement

import "testing"

// Gemma 4 26B A4B (bartowski IQ4_XS) header: 30 blocks, head_count_kv and the
// window pattern stated per block only, global blocks 2 x 512, windowed blocks
// 8 x 256 with a 1,024-token window.
func gemma4MoEProfile() *ModelProfile {
	hkv := make([]int, 30)
	swa := make([]int, 30)
	for i := range hkv {
		hkv[i], swa[i] = 8, 1
		if (i+1)%6 == 0 {
			hkv[i], swa[i] = 2, 0
		}
	}
	return &ModelProfile{
		ModelArch: "gemma4", NumLayers: 30, HeadCount: 16, KeyLength: 512, ValueLength: 512,
		KeyLengthSWA: 256, ValueLengthSWA: 256, SlidingWindow: 1024,
		HeadCountKVByLayer: hkv, SWAPattern: swa,
	}
}

// ik_llama 1fddd12 at --ctx-size 262144, q8_0, one slot: "allocating 29920.01
// MiB on device 0" for the KV cache.
func TestFullContextWindowedKVMatchesIKAllocation(t *testing.T) {
	model := gemma4MoEProfile()
	model.FullContextWindowedKV = true
	if got := computeKVTotalMB(model, 262144, "q8_0", false); got < 29920 || got > 29922 {
		t.Fatalf("priced %d MiB, ik_llama allocated 29,920", got)
	}
}

// A windowed backend keeps the window, and a per-block-only model is never 0.
func TestGemma4WindowedKVIsPricedPerBlock(t *testing.T) {
	model := gemma4MoEProfile()
	got := computeKVTotalMB(model, 262144, "q8_0", false)
	if got <= 0 || got >= 29920/2 {
		t.Fatalf("windowed Gemma 4 KV priced %d MiB", got)
	}
	// 5 global blocks hold the context: 262,144 x 2 x 1,024 x 1.0625 B.
	if got < 2720 {
		t.Fatalf("windowed Gemma 4 KV %d MiB is below its 5 global blocks (2,720)", got)
	}
}
