package placement

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Qwen3.8-27B UD-IQ2_XXS, --cpu --ram-budget 16G: the plan filled the scope
// with weights + KV + runtime buffers, then -cram 6144 and 16 checkpoints were
// added on top and the first long prompt was killed by the cgroup. Everything a
// CPU-only plan puts in host RAM must fit the budget together.
func TestCPUOnlyPromptCacheAndCheckpointsFitTheBudget(t *testing.T) {
	model := qwen38HybridProfile()
	model.Path = "qwen38.gguf"
	model.SizeBytes = 7266070528
	model.ContextSize = 262144
	model.HiddenSize = 5120
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 16384, FreeMB: 16384},
		CPU: detect.CPUInfo{Cores: 14},
	}
	s, err := Compute(caps, model, Options{CPUMode: true, ContextSize: 65536, KVPlacement: "cpu", KVQuality: "q8_0"})
	if err != nil {
		t.Fatal(err)
	}
	kv := computeKVTotalMB(model, s.ContextSize, s.KVType, false)
	weights := int(model.SizeBytes / 1024 / 1024)
	if s.PlannedHostFootprintMB < weights+kv {
		t.Fatalf("host footprint %d MiB omits weights %d + KV %d", s.PlannedHostFootprintMB, weights, kv)
	}
	checkpoints := max(0, s.MaxCheckpoints) * hybridCheckpointMB(model, 0)
	if total := s.PlannedHostFootprintMB + s.CRAM + checkpoints; total > caps.RAM.FreeMB {
		t.Fatalf("footprint %d + cram %d + checkpoints %d = %d MiB over the %d MiB budget",
			s.PlannedHostFootprintMB, s.CRAM, checkpoints, total, caps.RAM.FreeMB)
	}
	if s.MaxCheckpoints != 0 && s.MaxCheckpoints < hybridCheckpointMinimum {
		t.Fatalf("%d checkpoints is below the useful branch window", s.MaxCheckpoints)
	}
}

// The launcher's containment gate requires footprint + max(--cgroup-headroom,
// CRAM) under the ceiling. An automatic CPU-only plan must leave that room
// itself (matrix6: 14,309 + 4,096 > 16,384 refused the launch, and the gate's
// re-plan charged checkpoints again and found no context at all).
func TestCPUOnlyAutoContextLeavesTheContainmentReserve(t *testing.T) {
	model := qwen38HybridProfile()
	model.Path = "qwen38.gguf"
	model.SizeBytes = 7266070528
	model.ContextSize = 262144
	model.CTXTrain = 262144
	model.HiddenSize = 5120
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 16384, FreeMB: 16384},
		CPU: detect.CPUInfo{Cores: 14},
	}
	const headroom = 4096
	s, err := Compute(caps, model, Options{CPUMode: true, AutoContextMax: 262144, KVPlacement: "cpu", KVQuality: "q8_0", HostGrowthReserveMB: headroom})
	if err != nil {
		t.Fatal(err)
	}
	if s.ContextSize < contextMinimum {
		t.Fatalf("context %d below the floor", s.ContextSize)
	}
	if got := s.PlannedHostFootprintMB + max(headroom, s.CRAM); got > caps.RAM.FreeMB {
		t.Fatalf("footprint %d + reserve %d = %d MiB over the %d MiB ceiling",
			s.PlannedHostFootprintMB, max(headroom, s.CRAM), got, caps.RAM.FreeMB)
	}
	checkpoints := max(0, s.MaxCheckpoints) * hybridCheckpointMB(model, 0)
	if s.CRAM+checkpoints > caps.RAM.FreeMB-s.PlannedHostFootprintMB {
		t.Fatalf("cram %d + checkpoints %d exceed the %d MiB left", s.CRAM, checkpoints, caps.RAM.FreeMB-s.PlannedHostFootprintMB)
	}
	t.Logf("ctx %d, footprint %d, cram %d, checkpoints %d", s.ContextSize, s.PlannedHostFootprintMB, s.CRAM, s.MaxCheckpoints)

	// matrix7: the backend then reported 149.662 MiB per checkpoint. Priced at
	// the flat 128 MiB floor, that raised the footprint 352 MiB and the saved
	// config's relaunch was refused by the same gate. The plan already holds
	// the recurrent state, so the measurement must not move it.
	footprint := s.PlannedHostFootprintMB
	if _, _, changed := ApplyMeasuredCheckpointObservation(model, s, PromptCacheObservation{LargestCheckpointMB: 149.662}); changed {
		t.Fatalf("the measured checkpoint changed a plan that already priced the recurrent state")
	}
	if s.PlannedHostFootprintMB != footprint || s.PlannedHostFootprintMB+max(headroom, s.CRAM) > caps.RAM.FreeMB {
		t.Fatalf("relaunch footprint %d (planned %d) no longer passes the gate", s.PlannedHostFootprintMB, footprint)
	}
}

// A delta-net checkpoint is the slot's recurrent state: 149.63 MiB computed,
// 149.662 measured by ik_llama for Qwen3.8-27B.
func TestHybridCheckpointIsTheRecurrentState(t *testing.T) {
	model := qwen38HybridProfile()
	if got := recurrentStateMiBPerSlot(model); got < 149.6 || got > 149.7 {
		t.Fatalf("recurrent state %.2f MiB, want 149.63", got)
	}
	if got := hybridCheckpointMB(model, 0); got != 150 {
		t.Fatalf("checkpoint priced %d MiB, want 150", got)
	}
	if got := checkpointFootprintMB(model, 16, 1, 2176, 65536); got != 16*150 {
		t.Fatalf("16 checkpoints priced %d MiB, want 2400", got)
	}
	// A smaller state keeps the floor; a larger measurement still wins.
	small := qwen38HybridProfile()
	small.SSMInnerSize, small.SSMTimeStepRank = 4096, 32
	if got := hybridCheckpointMB(small, 0); got != hybridCheckpointReservePerCheckpointMB {
		t.Fatalf("small state priced %d MiB, want the %d MiB floor", got, hybridCheckpointReservePerCheckpointMB)
	}
	if got := hybridCheckpointMB(model, 200.2); got != 201 {
		t.Fatalf("measurement 200.2 priced %d MiB, want 201", got)
	}
}

// Qwen3.6-35B-A3B UD-IQ2_XXS, --cpu --ram-budget 16G (matrix7): the cold host
// estimate charged CUDA host staging and 2 GiB of graph scratch, 3,156 MiB in
// all, so with the containment reserve no context fit and the recommended CPU
// pick could not start. ik_llama's guarded load peaked at 11,331 MiB at 81,920
// tokens: weights 10,247.8 + KV 912.8 + compute 244.5 (the ub x vocab logits).
func TestCPUOnlyMoEFitsUnderTheContainmentReserve(t *testing.T) {
	model := &ModelProfile{
		Path: "qwen36.gguf", ModelArch: "qwen35moe", SizeBytes: 10756586464,
		NumLayers: 40, FullAttnInterval: 4, HasSSM: 1, IsMoE: true,
		HeadCountKV: 2, KeyLength: 256, ValueLength: 256,
		SSMConvKernel: 4, SSMStateSize: 128, SSMGroupCount: 16, SSMInnerSize: 4096, SSMTimeStepRank: 32,
		EmbeddingLength: 2048, HiddenSize: 2048, NumExperts: 256, ExpertUsedCount: 8,
		ExpertFF: 512, ExpertSharedFF: 512, VocabSize: 248320,
		ContextSize: 262144, CTXTrain: 262144,
	}
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 16384, FreeMB: 16384},
		CPU: detect.CPUInfo{Cores: 14},
	}
	if got := cpuOnlyRuntimeOverheadMB(model, 256, 10258); got < 245 || got > 1024 {
		t.Fatalf("CPU-only runtime estimate %d MiB; the backend used 244.5 of compute and ~85 more", got)
	}
	const headroom = 4096
	s, err := Compute(caps, model, Options{CPUMode: true, AutoContextMax: 262144, KVPlacement: "cpu", KVQuality: "q8_0", HostGrowthReserveMB: headroom})
	if err != nil {
		t.Fatal(err)
	}
	if s.ContextSize < 65536 {
		t.Fatalf("context %d; the same scope served 81,920 with room to spare", s.ContextSize)
	}
	measuredLoad := 10248 + computeKVTotalMB(model, s.ContextSize, s.KVType, false) + 245
	if s.PlannedHostFootprintMB < measuredLoad {
		t.Fatalf("footprint %d MiB is under the measured load %d", s.PlannedHostFootprintMB, measuredLoad)
	}
	if got := s.PlannedHostFootprintMB + max(headroom, s.CRAM); got > caps.RAM.FreeMB {
		t.Fatalf("footprint %d + reserve %d over the %d MiB ceiling", s.PlannedHostFootprintMB, max(headroom, s.CRAM), caps.RAM.FreeMB)
	}
	t.Logf("ctx %d, footprint %d, cram %d, checkpoints %d", s.ContextSize, s.PlannedHostFootprintMB, s.CRAM, s.MaxCheckpoints)
}
