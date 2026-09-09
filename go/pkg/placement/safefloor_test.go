package placement

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func safeFloorModel() *ModelProfile {
	return &ModelProfile{
		Path:      "/models/big-moe-00001-of-00004.gguf",
		Basename:  "big-moe.gguf",
		SizeBytes: 137903959808, TotalSizeMB: 131515,
		NumLayers: 43, IsMoE: true, NumExperts: 256, ExpertUsedCount: 6, ExpertFF: 2048,
		ExpertBytes: 131240296448, NonExpertBytes: 6658320448,
		TokenEmbdBytes: 562626560, OutputBytes: 434380800, ShexpBytes: 1149763584,
		ContextSize: 1048576, CTXTrain: 1048576, HiddenSize: 4096, EmbeddingLength: 4096,
		HeadCountKV: 1, KeyLength: 512, ValueLength: 512, ModelArch: "deepseek4",
		MeasuredKVBytesPerTok: map[string]float64{"f16": 6912.25},
	}
}

func safeFloorCaps() *detect.Capabilities {
	return &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "small-a", VRAMTotalMB: 12282, BandwidthMBps: 15760, VRAMUsedMB: 400},
			{Index: 1, Name: "big", VRAMTotalMB: 24564, BandwidthMBps: 15760, VRAMUsedMB: 300},
			{Index: 2, Name: "small-b", VRAMTotalMB: 12288, BandwidthMBps: 7880, VRAMUsedMB: 7800},
		},
		RAM: detect.RAMInfo{TotalMB: 217096, FreeMB: 206381},
		CPU: detect.CPUInfo{Cores: 14},
	}
}

func safeFloorOpts() Options {
	return Options{
		ContextSize: 1048576, KVPlacement: "auto", KVQuality: "auto",
		BackendTag: "llama", Parallel: 1, UBatchSize: 256, BatchSize: 2048,
		HotExperts: "on",
	}
}

// The floor exists to serve, so it must actually resolve something on a rig
// where the requested plan does not fit, and it must be tagged so nothing
// downstream can mistake it for a measured winner (invariant 7).
func TestSafeFloorResolvesAndIsNeverAMeasuredWinner(t *testing.T) {
	strategy, rung, err := ComputeSafeFloor(safeFloorCaps(), safeFloorModel(), safeFloorOpts(), SafeFloorConstraints{})
	if err != nil || strategy == nil {
		t.Fatalf("safe floor produced no placement: %v", err)
	}
	if !IsSafeFloorStrategy(strategy) {
		t.Fatalf("floor placement is not tagged as one: ContextFitTier=%q", strategy.ContextFitTier)
	}
	if strategy.PerformanceTuned || strategy.VerifiedConfigReused || strategy.BatchTuned {
		t.Fatalf("a floor placement must never claim measured/tuned provenance: %+v", strategy)
	}
	if rung.Name == "" {
		t.Fatal("floor did not report which rung it used")
	}
	if rung.String() == "" || !strings.Contains(rung.String(), rung.Name) {
		t.Fatalf("rung description is not usable for the banner: %q", rung.String())
	}
}

// Parallel is the one coordinate no rung may surrender: RelatedModelRuntime
// GraphGrowth only transfers a measured runtime-graph-growth reading to another
// key when the slot count matches, and that reading is the single quantity a
// floor launch uniquely produces.
func TestSafeFloorNeverSurrendersParallel(t *testing.T) {
	for _, parallel := range []int{1, 2, 4} {
		opts := safeFloorOpts()
		opts.Parallel = parallel
		strategy, _, err := ComputeSafeFloor(safeFloorCaps(), safeFloorModel(), opts, SafeFloorConstraints{})
		if err != nil || strategy == nil {
			t.Fatalf("parallel %d: no floor placement: %v", parallel, err)
		}
		if strategy.Parallel != parallel {
			t.Fatalf("floor changed the slot count: got %d want %d", strategy.Parallel, parallel)
		}
	}
}

// Invariant 3: automatic search may move only coordinates the user left
// automatic. A floor that needs a pinned one must refuse and say so, and the
// --safe-floor opt-in is the only thing that unlocks it.
func TestSafeFloorRefusesPinnedCoordinateWithoutOptIn(t *testing.T) {
	// A rig with no usable GPU forces the ladder down to the rungs that
	// surrender context and KV placement.
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "tiny", VRAMTotalMB: 2048, BandwidthMBps: 8000, VRAMUsedMB: 1900}},
		RAM:  detect.RAMInfo{TotalMB: 217096, FreeMB: 206381},
		CPU:  detect.CPUInfo{Cores: 14},
	}
	pinned := SafeFloorConstraints{ContextExplicit: true, KVPlacementExplicit: true}

	_, _, err := ComputeSafeFloor(caps, safeFloorModel(), safeFloorOpts(), pinned)
	if err == nil {
		t.Fatal("floor surrendered an explicitly pinned coordinate without the opt-in")
	}
	if !strings.Contains(err.Error(), "--safe-floor") {
		t.Fatalf("refusal must name the opt-in that unlocks it: %v", err)
	}

	pinned.AllowExplicit = true
	strategy, rung, err := ComputeSafeFloor(caps, safeFloorModel(), safeFloorOpts(), pinned)
	if err != nil || strategy == nil {
		t.Fatalf("with --safe-floor the pinned coordinates may be surrendered: %v", err)
	}
	if len(rung.Surrenders) == 0 {
		t.Fatal("a rung that gave up pinned coordinates must name them")
	}
}

// The CPU-only rung is mandatory: it is the only tier that does not depend on
// what companions occupy or on the driver-side CUDA-graph executable that no
// no-alloc oracle predicts.
func TestSafeFloorReachesCPUOnlyWhenNoGPUCanServe(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "tiny", VRAMTotalMB: 1024, BandwidthMBps: 8000, VRAMUsedMB: 1000}},
		RAM:  detect.RAMInfo{TotalMB: 217096, FreeMB: 206381},
		CPU:  detect.CPUInfo{Cores: 14},
	}
	strategy, rung, err := ComputeSafeFloor(caps, safeFloorModel(), safeFloorOpts(),
		SafeFloorConstraints{AllowExplicit: true})
	if err != nil || strategy == nil {
		t.Fatalf("floor must still resolve when no GPU can serve: %v", err)
	}
	if strategy.Type != CPUOnly && rung.Name != "cpu-only" {
		t.Fatalf("expected the ladder to reach a CPU-serviceable rung, got type=%v rung=%s",
			strategy.Type, rung.Name)
	}
}
