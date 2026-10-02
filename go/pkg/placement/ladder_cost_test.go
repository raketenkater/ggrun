package placement

import (
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// A generic sparse MoE (not any recorded model): 40 MoE layers of 1 GiB
// routed experts, 8 of 128 routed, on one 16 GiB GPU. Measured compute rows
// make the base microbatch leave no room for an expert layer while ubatch 64
// frees room for a couple.
func ladderFixture(t *testing.T) (*detect.Capabilities, *ModelProfile, *Strategy, Options) {
	t.Helper()
	dir := t.TempDir()
	model := &ModelProfile{Path: filepath.Join(dir, "moe.gguf"), NumLayers: 40, IsMoE: true,
		NumExperts: 128, ExpertUsedCount: 8, HiddenSize: 4096, EmbeddingLength: 4096,
		ExpertBytes: 40 << 30, NonExpertBytes: 6 << 30, HeadCountKV: 4, KeyLength: 128, ValueLength: 128}
	model.SizeBytes = model.ExpertBytes + model.NonExpertBytes
	model.TotalSizeMB = int(model.SizeBytes >> 20)
	caps := &detect.Capabilities{HostMemoryBandwidthMBps: 60000,
		RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 120000},
		GPUs: []detect.GPU{{Index: 0, Name: "gpu", VRAMTotalMB: 16384, MemBandwidthMBps: 900000, BandwidthMBps: 15760}}}
	opts := Options{CacheDir: dir, BackendTag: "llama", BackendCacheTag: "llama", BackendHelp: mainlineHelp,
		RequireMeasuredBuffers: true, KVPlacement: "gpu", KVQuality: "q8_0", ContextSize: 32768, Parallel: 1}
	base := &Strategy{Type: MoEOffload, ContextSize: 32768, BatchSize: 2048, UBatchSize: 512, Parallel: 1,
		KVPlacement: "gpu", KVQuality: "q8_0", KVType: "q8_0", FlashAttention: true}
	for ub, mb := range map[int]int{512: 7600, 256: 7400, 128: 7200, 64: 4600} {
		if err := RecordMeasuredComputeBuffers(dir, model, 32768, ub, "q8_0", "gpu", "llama", caps.GPUs, 1, map[int]int{0: mb}); err != nil {
			t.Fatal(err)
		}
	}
	return caps, model, base, opts
}

func runLadder(t *testing.T, caps *detect.Capabilities, model *ModelProfile, base *Strategy, opts Options) *Strategy {
	t.Helper()
	kv := computeKVTotalMB(model, base.ContextSize, base.KVType, false)
	pre := *base
	s, err := buildMoEOffload(cloneStrategy(base), caps, model, model.TotalSizeMB, kv, opts)
	got, err := maximizeMoEGPUFitByUBatch(&pre, s, err, caps, model, model.TotalSizeMB, kv, opts)
	if err != nil || got == nil {
		t.Fatalf("ladder failed: %v", err)
	}
	return got
}

func TestStagedLadderKeepsMicrobatchOverAFewExpertLayers(t *testing.T) {
	caps, model, base, opts := ladderFixture(t)
	_, moeCount := moeLayerRange(model)
	legacy := opts
	legacy.BackendTag = "ik_llama"
	old := runLadder(t, caps, model, base, legacy)
	if old.UBatchSize >= base.UBatchSize || old.NCPUMoE >= moeCount {
		t.Fatalf("fixture does not reproduce the legacy trade: ubatch %d, %d/%d CPU layers", old.UBatchSize, old.NCPUMoE, moeCount)
	}
	staged := runLadder(t, caps, model, base, opts)
	if staged.UBatchSize != base.UBatchSize {
		t.Fatalf("staged prefill gave up ubatch %d for %d GPU expert layer(s)", base.UBatchSize, moeCount-staged.NCPUMoE)
	}
	// Multi-slot serving keeps the legacy residency ladder.
	multi := cloneStrategy(base)
	multi.Parallel = 2
	want := runLadder(t, caps, model, multi, legacy)
	if got := runLadder(t, caps, model, multi, opts); got.UBatchSize != want.UBatchSize || got.NCPUMoE != want.NCPUMoE {
		t.Fatalf("multi-slot plan left the legacy ladder: ubatch %d, want %d", got.UBatchSize, want.UBatchSize)
	}
}

func TestStagedLadderStillDescendsForMostExperts(t *testing.T) {
	caps, model, base, opts := ladderFixture(t)
	caps.GPUs[0].VRAMTotalMB = 49152
	// Large-context graphs that only a small microbatch keeps off the card.
	for ub, mb := range map[int]int{512: 38000, 256: 30000, 128: 16000, 64: 4000} {
		if err := RecordMeasuredComputeBuffers(opts.CacheDir, model, 32768, ub, "q8_0", "gpu", "llama", caps.GPUs, 1, map[int]int{0: mb}); err != nil {
			t.Fatal(err)
		}
	}
	_, moeCount := moeLayerRange(model)
	got := runLadder(t, caps, model, base, opts)
	if got.UBatchSize >= base.UBatchSize || moeCount-got.NCPUMoE < moeCount/2 {
		t.Fatalf("kept ubatch %d with %d/%d GPU expert layers; most experts should move to GPU", got.UBatchSize, moeCount-got.NCPUMoE, moeCount)
	}
}
