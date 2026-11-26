package placement

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// W9d / P-14: every stored reserve the plan charges is printed with its size,
// date and source. Golden output; print only.
func TestLearnedReserveLinesGolden(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{
		{Index: 0, Name: "GPU A", VRAMTotalMB: 12282, PCIBusID: "00000000:17:00.0"},
		{Index: 1, Name: "GPU B", VRAMTotalMB: 24564, PCIBusID: "00000000:65:00.0"},
	}
	path := systemProbePath(dir, gpus)
	body := strings.Join([]string{
		"# System probe (post-launch per-device measurement)",
		"SYS_PROBE_SCHEMA=3",
		"# Generated: 2026-11-14T11:02:23Z",
		"SYS_CUDA_OVERHEAD_MB_CUDA0=255",
		"SYS_CUDA_OVERHEAD_MB_CUDA1=393",
		"SYS_CUDA_OVERHEAD_MB=393",
		"SYS_HOST_OVERHEAD_MB=719",
		"",
	}, "\n")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	got := strings.Join(LearnedReserveLines("dry-run", LearnedReserves(dir, gpus)), "\n")
	want := strings.Join([]string{
		"[dry-run] learned reserve CUDA0 255 MiB (measured 2026-11-14T11:02:23Z; System probe (post-launch per-device measurement); " + path + "; `ggrun probe reset` clears it)",
		"[dry-run] learned reserve CUDA1 393 MiB (measured 2026-11-14T11:02:23Z; System probe (post-launch per-device measurement); " + path + "; `ggrun probe reset` clears it)",
		"[dry-run] learned reserve host 719 MiB (measured 2026-11-14T11:02:23Z; System probe (post-launch per-device measurement); " + path + "; `ggrun probe reset` clears it)",
	}, "\n")
	if got != want {
		t.Fatalf("got:\n%s\nwant:\n%s", got, want)
	}
	// The listing reports what the planner applies, not a parallel reading.
	byGPU := SystemCUDAOverheadByGPU(dir, gpus)
	for _, r := range LearnedReserves(dir, gpus) {
		if r.Device == "CUDA0" && r.MB != byGPU[0] || r.Device == "CUDA1" && r.MB != byGPU[1] {
			t.Fatalf("listing %+v disagrees with planner %v", r, byGPU)
		}
	}
	if filepath.Dir(path) != dir {
		t.Fatalf("path %s", path)
	}
}

func TestNoStoredReservesPrintsNothing(t *testing.T) {
	if lines := LearnedReserveLines("dry-run", LearnedReserves(t.TempDir(), []detect.GPU{{Index: 0, Name: "x", VRAMTotalMB: 1}})); len(lines) != 0 {
		t.Fatalf("lines for an empty cache: %v", lines)
	}
}
