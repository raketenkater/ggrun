package placement

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestUnmeasuredDevicesReserveTheBackendDefaultMargin(t *testing.T) {
	gpus := []detect.GPU{{Index: 0, Name: "a", VRAMTotalMB: 12282}, {Index: 1, Name: "b", VRAMTotalMB: 24564}}
	dir := t.TempDir()
	t.Setenv("HOME", t.TempDir()) // no legacy ~/.cache/ggrun probe to migrate
	if got := SystemCUDAOverheadByGPU(dir, gpus); got != nil {
		t.Fatalf("measurement view invented overhead: %v", got)
	}
	got := PlanningCUDAOverheadByGPU(dir, gpus)
	if got[0] != UnmeasuredCUDAOverheadMB || got[1] != UnmeasuredCUDAOverheadMB {
		t.Fatalf("unmeasured planning overhead = %v", got)
	}
}
