package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// measuredOverheadCache returns a cache dir whose system probe records mb of
// measured CUDA overhead on every GPU, for tests whose subject is not the
// unmeasured-overhead reserve (UnmeasuredCUDAOverheadMB).
func measuredOverheadCache(t *testing.T, gpus []detect.GPU, mb int) string {
	t.Helper()
	dir := t.TempDir()
	seedMeasuredOverhead(t, dir, gpus, mb)
	return dir
}

// seedMeasuredOverhead adds that probe for one GPU set to dir; a --gpus
// selection is a different set (renumbered) with its own probe.
func seedMeasuredOverhead(t *testing.T, dir string, gpus []detect.GPU, mb int) {
	t.Helper()
	var b strings.Builder
	for _, g := range gpus {
		fmt.Fprintf(&b, "SYS_CUDA_OVERHEAD_MB_CUDA%d=%d\n", g.Index, mb)
	}
	fmt.Fprintf(&b, "SYS_CUDA_OVERHEAD_MB=%d\n", mb)
	if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("system_%s.cache", gpuIdentityHash(gpus))), []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

// rigOverheadCache records the CUDA overhead this project's rig measured
// (~/.cache/ggrun system probe, 2026-07-08): 488, 311 and 367 MiB on CUDA0-2.
func rigOverheadCache(t *testing.T, gpus []detect.GPU) string {
	t.Helper()
	dir := t.TempDir()
	rig := []int{488, 311, 367}
	var b strings.Builder
	most := 0
	for i, g := range gpus {
		mb := rig[i%len(rig)]
		most = max(most, mb)
		fmt.Fprintf(&b, "SYS_CUDA_OVERHEAD_MB_CUDA%d=%d\n", g.Index, mb)
	}
	fmt.Fprintf(&b, "SYS_CUDA_OVERHEAD_MB=%d\n", most)
	if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("system_%s.cache", gpuIdentityHash(gpus))), []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}
