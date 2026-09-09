package placement

import "testing"

func identityStrategy(split []float64) *Strategy {
	return &Strategy{
		Type: MoEOffload, ContextSize: 287744, UBatchSize: 256, BatchSize: 2048,
		Parallel: 1, KVPlacement: "gpu", KVQuality: "q8_0", KVType: "q8_0",
		NCPUMoE: 38, SplitMode: "layer", GPULayers: 999, TensorSplit: split,
	}
}

// TestJitteredSplitKeepsOneIdentity is the regression for the defect that kept
// hot experts from ever engaging.
//
// Measured 2026-09-07 across five launches of one model with identical
// settings, the planner emitted 0.23,0.65,0.12 three times and 0.23,0.64,0.12
// twice. Both seat the same layers on the same devices and allocate
// identically, but the identity hashed the raw floats, so no plan could match
// an allocation any launch had recorded -- including its own minutes earlier.
// hotExpertCacheCandidate needs BuildResourceLedger(...).Exact, Exact needs a
// matching identity, and the identity never matched.
func TestJitteredSplitKeepsOneIdentity(t *testing.T) {
	model := &ModelProfile{Path: "GLM.gguf", NumLayers: 45}
	a := AllocationPlacementIdentity(identityStrategy([]float64{0.23, 0.65, 0.12}), model)
	b := AllocationPlacementIdentity(identityStrategy([]float64{0.23, 0.64, 0.12}), model)
	if a == "" || b == "" {
		t.Fatal("identity must be produced for a complete strategy")
	}
	if a != b {
		t.Fatalf("splits that seat identical layers must share one identity:\n 0.65 -> %s\n 0.64 -> %s", a, b)
	}
}

// TestDifferentLayerSeatingIsADifferentPlan keeps the relaxation honest: the
// identity must still separate splits that actually move layers between
// devices, or a measurement would be spent on a plan it never described.
func TestDifferentLayerSeatingIsADifferentPlan(t *testing.T) {
	model := &ModelProfile{Path: "GLM.gguf", NumLayers: 45}
	balanced := AllocationPlacementIdentity(identityStrategy([]float64{0.33, 0.34, 0.33}), model)
	lopsided := AllocationPlacementIdentity(identityStrategy([]float64{0.10, 0.80, 0.10}), model)
	if balanced == lopsided {
		t.Fatal("splits that seat different layers on different devices are different plans")
	}
}

// TestIdentityStillSeparatesRealPlanChanges guards the rest of the tuple: the
// split is only one component, and relaxing it must not blur anything else.
func TestIdentityStillSeparatesRealPlanChanges(t *testing.T) {
	model := &ModelProfile{Path: "GLM.gguf", NumLayers: 45}
	base := identityStrategy([]float64{0.23, 0.65, 0.12})
	ref := AllocationPlacementIdentity(base, model)

	for name, mutate := range map[string]func(*Strategy){
		"context":   func(s *Strategy) { s.ContextSize = 131072 },
		"ubatch":    func(s *Strategy) { s.UBatchSize = 128 },
		"n-cpu-moe": func(s *Strategy) { s.NCPUMoE = 40 },
		"kv type":   func(s *Strategy) { s.KVType = "q4_0" },
		"ot pins":   func(s *Strategy) { s.OTString = `blk\.(3)\..*=CUDA1,exps=CPU` },
	} {
		s := identityStrategy([]float64{0.23, 0.65, 0.12})
		mutate(s)
		if AllocationPlacementIdentity(s, model) == ref {
			t.Fatalf("changing %s must change the identity", name)
		}
	}
}

// TestIdentityWithoutAModelStillAbsorbsFormattingNoise covers the fallback: no
// layer count is available, so the split is rounded rather than formatted at
// full float precision. It cannot know where layer boundaries fall, so it only
// claims to remove formatting noise.
func TestIdentityWithoutAModelStillAbsorbsFormattingNoise(t *testing.T) {
	a := AllocationPlacementIdentity(identityStrategy([]float64{0.23, 0.65, 0.12}), nil)
	b := AllocationPlacementIdentity(identityStrategy([]float64{0.230000000000001, 0.65, 0.12}), nil)
	if a != b {
		t.Fatal("float formatting noise must not create a new plan identity")
	}
}

// TestNonArgvFieldsDoNotChangeTheIdentity is the regression for the drift that
// kept hot experts from engaging even after the tensor-split fix.
//
// Measured 2026-09-07: two launches emitting byte-identical argv recorded
// allocation identities 6edcc5420b53 and cfd72c5f5cd6, so neither could match
// the other's measurement and the cache-free baseline read unmeasured forever.
// The identity carried seven fields that never reach the memory-shaping argv;
// any of them drifting minted a new plan out of the same launch.
func TestNonArgvFieldsDoNotChangeTheIdentity(t *testing.T) {
	model := &ModelProfile{Path: "GLM.gguf", NumLayers: 45}
	ref := AllocationPlacementIdentity(identityStrategy([]float64{0.23, 0.65, 0.12}), model)

	for name, mutate := range map[string]func(*Strategy){
		"backend tag":     func(s *Strategy) { s.BackendTag = "some-other-build" },
		"cram":            func(s *Strategy) { s.CRAM = 11776 },
		"max checkpoints": func(s *Strategy) { s.MaxCheckpoints = 32 },
		"checkpoint step": func(s *Strategy) { s.CheckpointMinStep = 512 },
		"mmap required":   func(s *Strategy) { s.MMapRequired = true },
		"cuda graphs":     func(s *Strategy) { s.UseCUDAGraphs = true },
		"mmproj size":     func(s *Strategy) { s.MMProjSizeMB = 512 },
	} {
		s := identityStrategy([]float64{0.23, 0.65, 0.12})
		mutate(s)
		if got := AllocationPlacementIdentity(s, model); got != ref {
			t.Fatalf("%s does not shape the memory argv and must not change the identity", name)
		}
	}
}

// TestArgvShapingFieldsStillChangeTheIdentity is the other half: relaxing the
// tuple must not let a measurement be spent on a plan that allocates
// differently. Every field below becomes a memory-shaping flag.
func TestArgvShapingFieldsStillChangeTheIdentity(t *testing.T) {
	model := &ModelProfile{Path: "GLM.gguf", NumLayers: 45}
	ref := AllocationPlacementIdentity(identityStrategy([]float64{0.23, 0.65, 0.12}), model)

	for name, mutate := range map[string]func(*Strategy){
		"context":        func(s *Strategy) { s.ContextSize = 131072 },
		"batch":          func(s *Strategy) { s.BatchSize = 512 },
		"ubatch":         func(s *Strategy) { s.UBatchSize = 128 },
		"kv type":        func(s *Strategy) { s.KVType = "q4_0" },
		"kv v type":      func(s *Strategy) { s.KVTypeV = "q4_0" },
		"parallel":       func(s *Strategy) { s.Parallel = 4 },
		"gpu layers":     func(s *Strategy) { s.GPULayers = 40 },
		"split mode":     func(s *Strategy) { s.SplitMode = "row" },
		"main gpu":       func(s *Strategy) { s.MainGPU = 2 },
		"n-cpu-moe":      func(s *Strategy) { s.NCPUMoE = 40 },
		"ot pins":        func(s *Strategy) { s.OTString = `blk\.(3)\..*=CUDA1,exps=CPU` },
		"mmap":           func(s *Strategy) { s.MMap = true },
		"mlock":          func(s *Strategy) { s.MLock = true },
		"flash attn":     func(s *Strategy) { s.FlashAttention = !identityStrategy(nil).FlashAttention },
		"swa full":       func(s *Strategy) { s.SWAFull = true },
		"kv placement":   func(s *Strategy) { s.KVPlacement = "cpu" },
		"expert seating": func(s *Strategy) { s.TensorSplit = []float64{0.10, 0.80, 0.10} },
	} {
		s := identityStrategy([]float64{0.23, 0.65, 0.12})
		mutate(s)
		if AllocationPlacementIdentity(s, model) == ref {
			t.Fatalf("%s shapes the memory argv and must change the identity", name)
		}
	}
}
