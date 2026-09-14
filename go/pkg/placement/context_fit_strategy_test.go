package placement

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func strategyFitCaps(gpus ...detect.GPU) *detect.Capabilities {
	return &detect.Capabilities{
		GPUs: gpus,
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 16},
	}
}

func strategyFitDenseModel(sizeMB, native int) *ModelProfile {
	return &ModelProfile{
		Path:        "fit-dense.gguf",
		TotalSizeMB: sizeMB,
		SizeBytes:   int64(sizeMB) << 20,
		NumLayers:   32,
		HiddenSize:  4096,
		HeadCountKV: 8,
		KeyLength:   128,
		ValueLength: 128,
		ContextSize: native,
		CTXTrain:    native,
		ModelArch:   "llama",
		MeasuredKVBytesPerTok: map[string]float64{
			"q8_0": 64 * 1024,
			"q4_0": 32 * 1024,
		},
	}
}

func strategyFitOptions() Options {
	return Options{
		KVPlacement:        "gpu",
		KVQuality:          "mid",
		BatchSize:          2048,
		UBatchSize:         512,
		BatchSizeExplicit:  true,
		UBatchSizeExplicit: true,
		SkipCachedConfig:   true,
	}
}

func TestAutomaticContextFitAccountsCompanionBeforeBoundary(t *testing.T) {
	caps := strategyFitCaps(detect.GPU{Index: 0, VRAMTotalMB: 12000})
	model := strategyFitDenseModel(6000, 131072)
	opts := strategyFitOptions()

	without, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("fit without companion: %v", err)
	}
	opts.Companions = []CompanionReservation{{
		Name: "reviewer", VRAMMB: 2000, GPUPreference: []int{0},
	}}
	with, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("fit with companion: %v", err)
	}
	if with.ContextSize >= without.ContextSize {
		t.Fatalf("companion did not reduce the fit boundary: with=%d without=%d",
			with.ContextSize, without.ContextSize)
	}
	if got := with.PlanFreeVRAM[0]; got != 10000 {
		t.Fatalf("plan snapshot ignored companion reservation: GPU0 free=%d, want 10000", got)
	}
	if len(with.CompanionPlacements) != 1 || with.CompanionPlacements[0].GPU != 0 {
		t.Fatalf("companion placement missing from final strategy: %+v", with.CompanionPlacements)
	}
	if with.ContextFitRejected != with.ContextSize+contextGranularity {
		t.Fatalf("adjacent boundary evidence = %d, want %d",
			with.ContextFitRejected, with.ContextSize+contextGranularity)
	}
	if !strings.Contains(with.ContextFitEvidence, "next granule") {
		t.Fatalf("missing adjacent rejection evidence: %q", with.ContextFitEvidence)
	}
}

func TestAutomaticContextFitUsesOnlySelectedCurrentFreeVRAM(t *testing.T) {
	caps := strategyFitCaps(
		detect.GPU{Index: 0, VRAMTotalMB: 24000, VRAMUsedMB: 0},
		detect.GPU{Index: 1, VRAMTotalMB: 12000, VRAMUsedMB: 3000},
	)
	model := strategyFitDenseModel(5000, 131072)
	opts := strategyFitOptions()

	all, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("unrestricted fit: %v", err)
	}
	opts.GPUs = []int{1}
	selected, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("selected-device fit: %v", err)
	}
	if selected.ContextSize >= all.ContextSize {
		t.Fatalf("fit counted unselected GPU0: selected=%d unrestricted=%d",
			selected.ContextSize, all.ContextSize)
	}
	// The backend sees a restricted CUDA_VISIBLE_DEVICES namespace, so selected
	// physical GPU1 is deliberately reindexed as CUDA0 in the plan.
	if len(selected.PlanFreeVRAM) != 1 || selected.PlanFreeVRAM[0] != 9000 {
		t.Fatalf("effective selected inventory = %+v, want only visible CUDA0=9000", selected.PlanFreeVRAM)
	}
}

func TestAutomaticContextFitDoesNotLowerExplicitQualityPreset(t *testing.T) {
	caps := strategyFitCaps(detect.GPU{Index: 0, VRAMTotalMB: 11776})
	model := strategyFitDenseModel(9000, 32768)

	fixed := strategyFitOptions()
	fixed.KVQuality = "mid"
	mid, err := Compute(caps, model, fixed)
	if err != nil {
		t.Fatalf("mid-quality fit: %v", err)
	}
	if mid.KVType != "q8_0" {
		t.Fatalf("explicit mid quality silently became %s", mid.KVType)
	}

	automatic := fixed
	automatic.KVQuality = "auto"
	auto, err := Compute(caps, model, automatic)
	if err != nil {
		t.Fatalf("automatic-quality fit: %v", err)
	}
	if auto.KVType != "q4_0" || auto.ContextFitTier != "gpu-resident" {
		t.Fatalf("auto policy did not use its legal resident fallback: %+v", auto)
	}
}

func TestAutomaticContextFitAccountsFinalParallelRuntimePolicy(t *testing.T) {
	caps := strategyFitCaps(detect.GPU{Index: 0, VRAMTotalMB: 24000})
	model := strategyFitDenseModel(4000, 131072)
	opts := strategyFitOptions()
	opts.AutoContextMax = 131072
	opts.Parallel = 4
	opts.AutoParallel = true
	opts.ParallelSlotTarget = 65536
	opts.RuntimePolicy = func(s *Strategy) {
		if s.Parallel > 1 {
			s.BatchSize = 128
			s.UBatchSize = 128
		}
	}

	got, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("parallel fit: %v", err)
	}
	if got.ContextSize != 131072 || got.Parallel != 2 {
		t.Fatalf("fit shape = ctx %d / parallel %d, want 131072 / 2",
			got.ContextSize, got.Parallel)
	}
	if got.BatchSize != 128 || got.UBatchSize != 128 {
		t.Fatalf("placement ignored effective runtime batch graph: %d/%d",
			got.BatchSize, got.UBatchSize)
	}
	if got.ContextSize/got.Parallel < 65536 {
		t.Fatalf("automatic slots violate their context target: %d per slot",
			got.ContextSize/got.Parallel)
	}
}

func TestAutomaticContextNativeLimitIsPerSlot(t *testing.T) {
	caps := strategyFitCaps(detect.GPU{Index: 0, VRAMTotalMB: 65536})
	model := strategyFitDenseModel(4000, 131072)
	for _, slots := range []int{1, 2, 4} {
		opts := strategyFitOptions()
		opts.Parallel = slots
		got, err := Compute(caps, model, opts)
		if err != nil {
			t.Fatal(err)
		}
		if got.ContextSize != slots*model.CTXTrain {
			t.Fatalf("%d slots: context=%d, want %d total (%d per slot)", slots, got.ContextSize, slots*model.CTXTrain, model.CTXTrain)
		}
	}
}

func TestAutomaticContextTotalPolicyCapStillWins(t *testing.T) {
	caps := strategyFitCaps(detect.GPU{Index: 0, VRAMTotalMB: 65536})
	model := strategyFitDenseModel(4000, 131072)
	opts := strategyFitOptions()
	opts.Parallel = 4
	opts.AutoContextMax = 262144
	got, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatal(err)
	}
	if got.ContextSize != 262144 {
		t.Fatalf("total context policy lost: %d", got.ContextSize)
	}
}

func TestAutomaticContextCapTracksReducedSlotCandidate(t *testing.T) {
	model := strategyFitDenseModel(4000, 65536)
	opts := strategyFitOptions()
	opts.Parallel = 4
	for _, slots := range []int{4, 2, 1} {
		if got := autoContextCapForSlots(model, opts, slots); got != slots*65536 {
			t.Fatalf("%d-slot candidate inherited wrong cap %d", slots, got)
		}
	}
}

func TestAutomaticContextNativeTotalDoesNotOverflow(t *testing.T) {
	model := &ModelProfile{CTXTrain: int(^uint(0) >> 1)}
	got := autoContextCapForSlots(model, Options{}, 4)
	if got <= 0 || uint64(got) > uint64(^uint32(0)) || got%contextGranularity != 0 {
		t.Fatalf("invalid capped native total: %d", got)
	}
}
