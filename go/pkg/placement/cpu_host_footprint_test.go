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
	checkpoints := max(0, s.MaxCheckpoints) * hybridCheckpointReservePerCheckpointMB
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
	checkpoints := max(0, s.MaxCheckpoints) * hybridCheckpointReservePerCheckpointMB
	if s.CRAM+checkpoints > caps.RAM.FreeMB-s.PlannedHostFootprintMB {
		t.Fatalf("cram %d + checkpoints %d exceed the %d MiB left", s.CRAM, checkpoints, caps.RAM.FreeMB-s.PlannedHostFootprintMB)
	}
	t.Logf("ctx %d, footprint %d, cram %d, checkpoints %d", s.ContextSize, s.PlannedHostFootprintMB, s.CRAM, s.MaxCheckpoints)
}
