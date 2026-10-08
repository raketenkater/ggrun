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
