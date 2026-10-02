package placement

import (
	"math"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestObservePromptCacheReconstructsEntryBeforeFailedSave(t *testing.T) {
	logText := `
slot create_check: id  0 | task 58 | created context checkpoint 1 of 16 (pos_min = 0, pos_max = 2047, n_tokens = 2048, size = 871.657 MiB)
slot create_check: id  0 | task 58 | created context checkpoint 2 of 16 (pos_min = 0, pos_max = 4095, n_tokens = 4096, size = 871.673 MiB)
slot create_check: id  0 | task 77 | created context checkpoint 5 of 16 (pos_min = 0, pos_max = 6361, n_tokens = 6362, size = 871.690 MiB)
slot create_check: id  0 | task 77 | created context checkpoint 6 of 16 (pos_min = 0, pos_max = 6371, n_tokens = 6372, size = 871.690 MiB)
slot create_check: id  0 | task 87 | created context checkpoint 3 of 16 (pos_min = 0, pos_max = 4245, n_tokens = 4246, size = 871.674 MiB)
slot create_check: id  0 | task 97 | created context checkpoint 4 of 16 (pos_min = 0, pos_max = 4255, n_tokens = 4256, size = 871.674 MiB)
 - saving prompt with length 4256, total state size = 1050.393 MiB
`

	obs := ObservePromptCache(logText)
	want := 1050.393 + 6*871.690
	if math.Abs(obs.LargestEntryMB-want) > 0.001 {
		t.Fatalf("LargestEntryMB = %.3f, want %.3f", obs.LargestEntryMB, want)
	}
	if obs.Evicted != 0 || obs.Skipped != 0 {
		t.Fatalf("save-only observation counted pressure: evicted=%d skipped=%d", obs.Evicted, obs.Skipped)
	}
	if math.Abs(obs.LargestCheckpointMB-871.690) > 0.001 {
		t.Fatalf("LargestCheckpointMB = %.3f, want 871.690", obs.LargestCheckpointMB)
	}
}

func TestRecurrentCheckpointFootprintUsesColdFloorMeasurementAndSlots(t *testing.T) {
	model := &ModelProfile{HasSSM: 1}
	if got := checkpointFootprintMB(model, 16, 1, 0, 262144); got != 2048 {
		t.Fatalf("cold recurrent reserve = %d MiB, want 2048", got)
	}
	if got := checkpointFootprintMB(model, 16, 4, 0, 262144); got != 8192 {
		t.Fatalf("four-slot recurrent reserve = %d MiB, want 8192", got)
	}
	if got := checkpointFootprintMB(model, 16, 1, 0, 262144, 871.690); got != 13952 {
		t.Fatalf("measured recurrent reserve = %d MiB, want 13952", got)
	}
	if got := checkpointFootprintMB(&ModelProfile{}, 16, 1, 0, 262144); got != 0 {
		t.Fatalf("non-recurrent model without SWA reserved %d MiB", got)
	}
}

func TestApplyMeasuredCheckpointObservationRaisesHostPlan(t *testing.T) {
	model := &ModelProfile{HasSSM: 1}
	strategy := &Strategy{
		ContextSize: 262144, Parallel: 1, MaxCheckpoints: 16,
		PlannedHostFootprintMB: 60000,
	}
	before, after, changed := ApplyMeasuredCheckpointObservation(model, strategy, PromptCacheObservation{LargestCheckpointMB: 871.690})
	if !changed || before != 2048 || after != 13952 {
		t.Fatalf("checkpoint adjustment = %d -> %d changed=%v, want 2048 -> 13952", before, after, changed)
	}
	if strategy.PlannedHostFootprintMB != 71904 || math.Abs(strategy.MeasuredCheckpointMB-871.690) > 0.001 {
		t.Fatalf("adjusted strategy = %+v", strategy)
	}
}

func TestCheckpointObservationRoundTrip(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "recurrent.gguf", Basename: "recurrent.gguf", SizeBytes: 1234, HasSSM: 1}
	gpus := []detect.GPU{{Index: 0, Name: "gpu", VRAMTotalMB: 24576}}
	const (
		ctx      = 32768
		ubatch   = 256
		parallel = 2
		tag      = "llama-test"
	)
	obs := PromptCacheObservation{LargestCheckpointMB: 871.690}
	if err := RecordPromptCacheObservation(dir, model, ctx, ubatch, "mid", "gpu", tag, gpus, parallel, obs); err != nil {
		t.Fatal(err)
	}
	strategy := &Strategy{ContextSize: ctx, UBatchSize: ubatch, KVQuality: "mid", KVPlacement: "gpu", Parallel: parallel}
	LoadMeasuredPromptCache(dir, model, strategy, tag, gpus)
	if math.Abs(strategy.MeasuredCheckpointMB-871.690) > 0.001 {
		t.Fatalf("checkpoint round trip = %.3f, want 871.690", strategy.MeasuredCheckpointMB)
	}
}

func TestObservePromptCacheDoesNotUndercutBackendFullEntryMeasurement(t *testing.T) {
	logText := `
slot create_check: id 0 | task 1 | created context checkpoint 16 of 16 (size = 74.930 MiB)
srv prompt_save: - saving prompt with length 86885, total state size = 2451.686 MiB (draft: 0.000 MiB)
srv alloc: - prompt state size 3650.558 MiB exceeds cache size limit 3584.000 MiB, skipping
`

	obs := ObservePromptCache(logText)
	if obs.LargestEntryMB < 3650.558 || obs.LargestEntryMB > 3650.6 {
		t.Fatalf("LargestEntryMB = %.3f, want at least backend measurement 3650.558", obs.LargestEntryMB)
	}
	if obs.Skipped != 1 {
		t.Fatalf("Skipped = %d, want 1", obs.Skipped)
	}
}

func TestMeasuredPromptCacheTargetRoundsUpWithoutBreakingHostCap(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 60000, FreeMB: 60000},
	}
	s := &Strategy{
		Type:                       MoEOffload,
		Parallel:                   1,
		PlannedHostFootprintMB:     30000,
		MeasuredPromptCacheEntryMB: 3650.558,
	}

	cram, _ := computeCRAM(caps, &ModelProfile{}, s, 0, 0)
	if cram != 7680 {
		t.Fatalf("CRAM = %d MiB, want 7680 MiB (ceil of two measured entries)", cram)
	}

	// The capacity rounding must never override the existing two-thirds host
	// safety ceiling. With only 9 GiB after load, the quantized cap is 5632 MiB.
	s.PlannedHostFootprintMB = 51000
	cram, _ = computeCRAM(caps, &ModelProfile{}, s, 0, 0)
	if cram != 5632 {
		t.Fatalf("capped CRAM = %d MiB, want 5632 MiB", cram)
	}
}

// A prompt-cache measurement re-sizes the host cache budget of the chosen
// plan; it must not move context, microbatch, split or expert layout.
func TestRefreshRuntimeCachePolicyKeepsTheChosenPlan(t *testing.T) {
	dir := t.TempDir()
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 12000}, {Index: 1, VRAMTotalMB: 24000}},
		RAM: detect.RAMInfo{TotalMB: 200000, FreeMB: 180000}}
	model := &ModelProfile{Path: dir + "/m.gguf", NumLayers: 48, IsMoE: true, TotalSizeMB: 130000,
		HeadCountKV: 4, KeyLength: 128, ValueLength: 128}
	s := &Strategy{Type: MoEOffload, ContextSize: 662528, UBatchSize: 1024, BatchSize: 2048, Parallel: 1,
		TensorSplit: []float64{.28, .66, .06}, NCPUMoE: 49, KVType: "q8_0", KVQuality: "q8_0", KVPlacement: "gpu",
		PlannedHostFootprintMB: 120000, CRAM: 8704}
	opts := Options{CacheDir: dir, BackendCacheTag: "llama"}
	if err := RecordMeasuredPromptCache(dir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, "llama", caps.GPUs, 1, 4096); err != nil {
		t.Fatal(err)
	}
	next := RefreshRuntimeCachePolicy(caps, model, s, opts)
	if next.ContextSize != s.ContextSize || next.UBatchSize != 1024 || next.NCPUMoE != 49 ||
		splitCompactKey(next.TensorSplit) != splitCompactKey(s.TensorSplit) {
		t.Fatalf("cache refresh re-planned the placement: %+v", next)
	}
	if next.MeasuredPromptCacheBPT != 4096 {
		t.Fatalf("measurement not loaded: %v", next.MeasuredPromptCacheBPT)
	}
	if s.MeasuredPromptCacheBPT != 0 {
		t.Fatal("refresh mutated the chosen strategy")
	}
}
