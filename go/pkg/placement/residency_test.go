package placement

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// glmRig is the measured 2026-09-02 GLM 5.3 Flash launch: 137 GB of weights on
// 48 GB of VRAM, all GPU-resident experts pinned to a single owner (CUDA1),
// --tensor-split 0.17,0.77,0.06. The numbers below are the backend's own
// per-device ledger from that serve.
func glmRig() ([]detect.GPU, MeasuredAllocation) {
	gpus := []detect.GPU{
		{Index: 0, VRAMTotalMB: 12282}, // RTX 4070
		{Index: 1, VRAMTotalMB: 24564}, // RTX 3090 Ti
		{Index: 2, VRAMTotalMB: 12288}, // RTX 3060
	}
	alloc := MeasuredAllocation{
		Evidence:   "live-allocated",
		ModelByGPU: map[int]int{0: 1678, 1: 9217, 2: 496},
		ContextByGPU: map[int]int{
			0: 628, 1: 2826, 2: 0,
		},
		// The compute/graph residue. CUDA0 and CUDA1 are layer owners and pay a
		// real graph; CUDA2 owns nothing and pays 0.61 MiB.
		UnaccountedByGPU: map[int]int{0: 3115, 1: 2814, 2: 1},
	}
	return gpus, alloc
}

func glmModel() *ModelProfile {
	// 45 layers, 41 of them routed-expert layers left on the host by that plan.
	// ~2.4 GB per routed-expert layer at Q2_K_XL.
	routed := make([]int64, 45)
	for i := range routed {
		routed[i] = 2400 << 20
	}
	return &ModelProfile{
		Path:                   "GLM-5.3-UD-Q2_K_XL.gguf",
		NumLayers:              45,
		NumExperts:             288,
		RoutedExpertLayerBytes: routed,
	}
}

// TestExpertSeatsDetectMeasuredUnderPacking is the regression this file exists
// for. numGPUsExcluded reported zero excluded GPUs for this exact placement --
// CUDA2's 0.06 share maps to 2.7 dense layers, which passes its ">= 1 layer"
// test -- while CUDA2 finished the run at 4% occupancy holding no expert layer
// at all. Occupancy must catch what participation missed.
func TestExpertSeatsDetectMeasuredUnderPacking(t *testing.T) {
	cacheDir := t.TempDir()
	gpus, alloc := glmRig()
	model := glmModel()
	s := &Strategy{NCPUMoE: 41, ContextSize: 643072, UBatchSize: 64}

	if err := RecordRuntimeGraphGrowth(cacheDir, model, s.ContextSize, s.UBatchSize,
		"high", "gpu", "llama", gpus, 1, map[int]int{0: 1000, 1: 1000, 2: 1000}); err != nil {
		t.Fatalf("seed measured runtime growth: %v", err)
	}

	caps := &detect.Capabilities{GPUs: gpus}
	report := ComputeExpertSeats(cacheDir, caps, model, s, alloc, 1, "llama")

	if !report.UnderPacked() {
		t.Fatalf("expected the measured GLM placement to read as under-packed; got %s", report.Summary())
	}
	// The starved card must be the one offering the most seats: it is 96% empty.
	var cuda2 ExpertSeat
	for _, seat := range report.Seats {
		if seat.GPUIndex == 2 {
			cuda2 = seat
		}
	}
	if cuda2.Seats < 2 {
		t.Fatalf("CUDA2 held 496 of 12288 MiB and must offer seats, got %d (%s)", cuda2.Seats, report.Summary())
	}
	if cuda2.OccupancyPct() > 10 {
		t.Fatalf("CUDA2 occupancy should read near-empty, got %d%%", cuda2.OccupancyPct())
	}
	// And the whole rig must report real idle capacity, not a rounding artefact.
	if report.IdleMB < 10000 {
		t.Fatalf("expected five figures of idle VRAM on this measured run, got %d MiB", report.IdleMB)
	}
	if report.TotalSeats() < 3 {
		t.Fatalf("expected at least 3 further expert layers to fit, got %d (%s)",
			report.TotalSeats(), report.Summary())
	}
	t.Logf("measured GLM rig: %s", report.Summary())
	t.Logf("seats offered: %d further expert layer(s) of %d MiB each",
		report.TotalSeats(), report.ExpertLayerMB)
}

// TestExpertSeatsOfferNoSeatsWithoutMeasuredMargin pins the user's own
// requirement -- "the margin should be a measured one" -- and invariant 4.
// Without exact runtime-growth evidence for a device, the packer is offered
// nothing rather than a default margin, because a default margin here is the
// static fudge reserve invariant 8 forbids and is exactly what an aggressive
// packer would lean on hardest.
func TestExpertSeatsOfferNoSeatsWithoutMeasuredMargin(t *testing.T) {
	cacheDir := t.TempDir() // deliberately empty: nothing measured
	gpus, alloc := glmRig()
	model := glmModel()
	s := &Strategy{NCPUMoE: 41, ContextSize: 643072, UBatchSize: 64}

	caps := &detect.Capabilities{GPUs: gpus}
	report := ComputeExpertSeats(cacheDir, caps, model, s, alloc, 1, "llama")

	if report.TotalSeats() != 0 {
		t.Fatalf("unmeasured margin must offer zero seats, got %d (%s)",
			report.TotalSeats(), report.Summary())
	}
	if report.UnderPacked() {
		t.Fatal("an unmeasured rig must not be declared under-packed: there is no evidence to pack against")
	}
	for _, seat := range report.Seats {
		if seat.MarginMeasured {
			t.Fatalf("CUDA%d claims a measured margin from an empty cache", seat.GPUIndex)
		}
	}
	// Idle VRAM is still reported: the operator should see the capacity even
	// when ggrun refuses to spend it.
	if report.IdleMB <= 0 {
		t.Fatal("idle capacity must stay visible even when no seats are offered")
	}
}

// TestEntryCostChargesExpertFreeGPUAtMeasuredOwnerRate guards the accounting
// that keeps a re-pack from walking into the graph_reserve OOM that cost 46
// minutes on 2026-09-02. A device owning no expert layers pays no real graph
// yet; handing it a layer costs it one first. That charge is taken from what a
// real owner was measured to pay on this same launch, never from a formula.
func TestEntryCostChargesExpertFreeGPUAtMeasuredOwnerRate(t *testing.T) {
	gpus, alloc := glmRig()
	entry := entryCostByGPU(gpus, alloc)

	// CUDA0 already pays the highest observed graph: it has bought the ticket.
	if entry[0] != 0 {
		t.Fatalf("CUDA0 already pays the owner rate and must be charged nothing, got %d", entry[0])
	}
	// CUDA2 pays 1 MiB against an owner rate of 3115.
	if want := 3115 - 1; entry[2] != want {
		t.Fatalf("CUDA2 must be charged the full owner graph (%d), got %d", want, entry[2])
	}
	if entry[1] != 3115-2814 {
		t.Fatalf("CUDA1 must be charged only the shortfall, got %d", entry[1])
	}
}

// TestFullyPackedRigIsNotUnderPacked keeps the detector from firing forever:
// once no device can seat another whole expert layer, the plan is at its
// measured maximum and a re-pack must stop.
func TestFullyPackedRigIsNotUnderPacked(t *testing.T) {
	cacheDir := t.TempDir()
	model := glmModel()
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 12282}}
	s := &Strategy{NCPUMoE: 41, ContextSize: 643072, UBatchSize: 64}
	if err := RecordRuntimeGraphGrowth(cacheDir, model, s.ContextSize, s.UBatchSize,
		"high", "gpu", "llama", gpus, 1, map[int]int{0: 1000}); err != nil {
		t.Fatalf("seed measured runtime growth: %v", err)
	}
	// Leave less free than one 2400 MiB expert layer plus the measured margin.
	alloc := MeasuredAllocation{
		Evidence:         "live-allocated",
		ModelByGPU:       map[int]int{0: 9000},
		ContextByGPU:     map[int]int{0: 0},
		UnaccountedByGPU: map[int]int{0: 500},
	}
	caps := &detect.Capabilities{GPUs: gpus}
	report := ComputeExpertSeats(cacheDir, caps, model, s, alloc, 1, "llama")
	if report.UnderPacked() {
		t.Fatalf("a rig with no room for another whole expert layer must not read under-packed (%s)",
			report.Summary())
	}
}

// TestExpertSeatsIgnoreRigWithNoExpertsOnHost keeps the detector honest about
// what it is for: idle VRAM is not a defect when every expert is already
// resident. There is nothing left to pack.
func TestExpertSeatsIgnoreRigWithNoExpertsOnHost(t *testing.T) {
	cacheDir := t.TempDir()
	gpus, alloc := glmRig()
	model := glmModel()
	s := &Strategy{NCPUMoE: 0, ContextSize: 643072, UBatchSize: 64}
	if err := RecordRuntimeGraphGrowth(cacheDir, model, s.ContextSize, s.UBatchSize,
		"high", "gpu", "llama", gpus, 1, map[int]int{0: 1000, 1: 1000, 2: 1000}); err != nil {
		t.Fatalf("seed measured runtime growth: %v", err)
	}
	caps := &detect.Capabilities{GPUs: gpus}
	report := ComputeExpertSeats(cacheDir, caps, model, s, alloc, 1, "llama")
	if report.UnderPacked() {
		t.Fatal("no experts on the host means nothing to pack, however much VRAM is idle")
	}
	if report.TotalSeats() == 0 {
		t.Fatal("seats should still be reported for diagnostics even when nothing needs them")
	}
}

// glmQ3Rig is the 2026-09-08 GLM 5.3 Flash UD-Q3_K_XL control serve: the same
// three cards, carrying 41357 of 49134 MiB (84%). UnaccountedByGPU holds the
// backend's compute buffers, which is what that term means everywhere it is
// consumed (parseLiveAllocationFromLog stores computeMB+overhead there,
// optimizer.go reads it as graphMB, entryCostByGPU charges new owners at it).
func glmQ3Rig() ([]detect.GPU, MeasuredAllocation) {
	gpus := []detect.GPU{
		{Index: 0, VRAMTotalMB: 12282},
		{Index: 1, VRAMTotalMB: 24564},
		{Index: 2, VRAMTotalMB: 12288},
	}
	alloc := MeasuredAllocation{
		Evidence:         "live-allocated",
		ModelByGPU:       map[int]int{0: 5089, 1: 14242, 2: 4013},
		ContextByGPU:     map[int]int{0: 580, 1: 2318, 2: 290},
		UnaccountedByGPU: map[int]int{0: 4855, 1: 5114, 2: 4855},
	}
	return gpus, alloc
}

func glmQ3Model() *ModelProfile {
	// 45 layers; ~4.5 GB per routed-expert layer at UD-Q3_K_XL, measured as the
	// 8964 MiB difference between a 5-layer and a 3-layer resident plan.
	routed := make([]int64, 45)
	for i := range routed {
		routed[i] = 4500 << 20
	}
	return &ModelProfile{
		Path:                   "GLM-5.3-Flash-UD-Q3_K_XL.gguf",
		NumLayers:              45,
		NumExperts:             288,
		RoutedExpertLayerBytes: routed,
	}
}

// TestExpertSeatsRefuseSeatsOnAFullRig is the invariant the occupancy fix
// exists for: a device may only be offered a seat that its own free VRAM can
// actually hold.
//
// Free space on this serve is 1758 / 2889 / 3130 MiB against a 4500 MiB expert
// layer, so the answer is zero seats on every device even though 7777 MiB is
// unclaimed rig-wide. Fragmentation is not capacity: seats are per-device
// because an expert layer lands on one card.
//
// The margin is seeded deliberately, so a passing test proves the zero came
// from arithmetic rather than from the fail-closed unmeasured-margin path --
// which would hide a regression here behind the right answer.
func TestExpertSeatsRefuseSeatsOnAFullRig(t *testing.T) {
	cacheDir := t.TempDir()
	gpus, alloc := glmQ3Rig()
	model := glmQ3Model()
	s := &Strategy{NCPUMoE: 40, ContextSize: 287744, UBatchSize: 256}

	if err := RecordRuntimeGraphGrowth(cacheDir, model, s.ContextSize, s.UBatchSize,
		"q8_0", "gpu", "llama", gpus, 1, map[int]int{0: 64, 1: 64, 2: 64}); err != nil {
		t.Fatalf("seed measured runtime growth: %v", err)
	}

	caps := &detect.Capabilities{GPUs: gpus}
	report := ComputeExpertSeats(cacheDir, caps, model, s, alloc, 1, "llama")

	for _, seat := range report.Seats {
		if !seat.MarginMeasured {
			t.Fatalf("CUDA%d margin should be measured; the zero-seat result must come from "+
				"arithmetic, not from missing evidence", seat.GPUIndex)
		}
		if seat.Seats != 0 {
			t.Errorf("CUDA%d offered %d seat(s) with %d MiB free against a %d MiB expert layer",
				seat.GPUIndex, seat.Seats, seat.FreeMB(), report.ExpertLayerMB)
		}
	}
	if report.TotalSeats() != 0 {
		t.Errorf("rig offered %d seat(s); 7777 MiB spread across three cards cannot hold a "+
			"%d MiB layer on any of them", report.TotalSeats(), report.ExpertLayerMB)
	}
	// 40 expert layers are on the host, but the rig cannot take one back. Calling
	// that under-packed would send the packer into an OOM it cannot avoid.
	if report.UnderPacked() {
		t.Errorf("a full rig must not read as under-packed: %s", report.Summary())
	}
}

// TestExpertSeatsOccupancyIncludesComputeBuffer proves the reported occupancy
// is the one the devices actually reached. Omitting the compute buffers turned
// 84% into 54% and invented 14682 MiB of idle VRAM.
func TestExpertSeatsOccupancyIncludesComputeBuffer(t *testing.T) {
	cacheDir := t.TempDir()
	gpus, alloc := glmQ3Rig()
	caps := &detect.Capabilities{GPUs: gpus}
	s := &Strategy{NCPUMoE: 40, ContextSize: 287744, UBatchSize: 256}

	report := ComputeExpertSeats(cacheDir, caps, glmQ3Model(), s, alloc, 1, "llama")

	if got, want := report.CapacityMB-report.IdleMB, 41356; got != want {
		t.Errorf("occupied VRAM: got %d MiB, want %d", got, want)
	}
	if got, want := report.IdleMB, 7778; got != want {
		t.Errorf("idle VRAM: got %d MiB, want %d", got, want)
	}
	if got := pctOf(report.CapacityMB-report.IdleMB, report.CapacityMB); got < 83 || got > 85 {
		t.Errorf("occupancy: got %d%%, want ~84%%", got)
	}
}
