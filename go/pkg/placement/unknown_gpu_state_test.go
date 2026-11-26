package placement

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Invariant 8: when the GPU probe timed out, the planner refuses instead of
// treating the empty GPU list as a CPU-only host.
func TestPlannerRefusesUnknownGPUState(t *testing.T) {
	caps := &detect.Capabilities{
		RAM:             detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:             detect.CPUInfo{Cores: 8, Threads: 16},
		GPUStateUnknown: "nvidia-smi did not answer within 10s",
	}
	model := &ModelProfile{Path: "/m/x.gguf", Basename: "x.gguf", TotalSizeMB: 4000}
	if _, err := Compute(caps, model, Options{ContextSize: 4096}); err == nil || !strings.Contains(err.Error(), "GPU state unknown") {
		t.Fatalf("Compute planned on unknown GPU state: %v", err)
	}
	if _, err := Compute(caps, model, Options{}); err == nil || !strings.Contains(err.Error(), "GPU state unknown") {
		t.Fatalf("automatic context planned on unknown GPU state: %v", err)
	}
	if _, err := ReplanAfterOOM(caps, model, Options{ContextSize: 4096}, map[int]int{0: 512}); err == nil || !strings.Contains(err.Error(), "GPU state unknown") {
		t.Fatalf("replan ran on unknown GPU state: %v", err)
	}
	caps.GPUStateUnknown = ""
	if _, err := Compute(caps, model, Options{ContextSize: 4096}); err != nil && strings.Contains(err.Error(), "GPU state unknown") {
		t.Fatalf("known state refused: %v", err)
	}
}
