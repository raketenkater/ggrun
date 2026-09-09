package placement

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestComputeUpdatePreservesOOMSource(t *testing.T) {
	dir := t.TempDir()
	gpus, model := growthCarryFixture()
	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1, 0, 900, false); err != nil {
		t.Fatal(err)
	}
	if err := RecordMeasuredComputeBuffers(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1, map[int]int{0: 200}); err != nil {
		t.Fatal(err)
	}
	pc := loadProbeCache(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1)
	if pc == nil || !pc.RuntimeGraphGrowthFromOOMByGPU[0] {
		t.Fatal("compute-only update promoted OOM evidence to a healthy serve")
	}
	if err := RecordRuntimeGraphGrowth(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1, map[int]int{0: 100}); err != nil {
		t.Fatal(err)
	}
	if got := RuntimeGraphGrowthByGPU(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1)[0]; got != 100 {
		t.Fatalf("healthy measurement could not retire the OOM reserve: %d", got)
	}
}

func TestGrowthUpdatesPreserveComputeRole(t *testing.T) {
	for _, clear := range []bool{false, true} {
		dir := t.TempDir()
		gpus, model := growthCarryFixture()
		if err := RecordMeasuredComputeBuffers(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1, map[int]int{0: 200}, map[int]bool{0: true}); err != nil {
			t.Fatal(err)
		}
		if err := RecordRuntimeGraphGrowth(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1, map[int]int{0: 100}); err != nil {
			t.Fatal(err)
		}
		if clear {
			if err := ClearRuntimeGraphGrowth(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1); err != nil {
				t.Fatal(err)
			}
		}
		pc := loadProbeCache(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1)
		if pc == nil || pc.ComputeBufByGPU[0] != 200 || !pc.ComputeBufExpertOnlyByGPU[0] {
			t.Fatalf("growth update changed compute role (clear=%v): %+v", clear, pc)
		}
	}
}

func TestRetainedComputeKeepsItsRole(t *testing.T) {
	for _, observed := range []bool{false, true} {
		dir := t.TempDir()
		gpus, model := growthCarryFixture()
		evidence := "oracle-planned"
		if observed {
			evidence = "live-allocated"
		}
		if err := writeProbeCacheForModel(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1, map[int]int{0: 4000}, nil, nil, 0,
			probeMeasurements{ComputeBufEvidence: evidence}); err != nil {
			t.Fatal(err)
		}
		if err := RecordMeasuredComputeBuffers(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1, map[int]int{0: 200}, map[int]bool{0: true}); err != nil {
			t.Fatal(err)
		}
		pc := loadProbeCache(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1)
		if pc == nil || pc.ComputeBufByGPU[0] != 4000 || pc.ComputeBufExpertOnlyByGPU[0] {
			t.Fatalf("rejected reading changed retained role (observed=%v): %+v", observed, pc)
		}
	}
}

func TestPreIntegrityProbeCannotClaimHealthyMetadata(t *testing.T) {
	for _, oldSchema := range []int{8, 9} {
		t.Run(fmt.Sprintf("schema-%d", oldSchema), func(t *testing.T) {
			dir := t.TempDir()
			gpus, model := growthCarryFixture()
			if err := RecordRuntimeGraphGrowth(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1, map[int]int{0: 900}); err != nil {
				t.Fatal(err)
			}
			path := probeCachePath(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 0)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			data = []byte(strings.ReplaceAll(string(data), fmt.Sprintf("PROBE_CACHE_SCHEMA=%d", probeCacheSchema), fmt.Sprintf("PROBE_CACHE_SCHEMA=%d", oldSchema)))
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			if pc := loadProbeCache(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1); pc != nil {
				t.Fatal("old exact record retained unverifiable metadata")
			}
			if got := RelatedModelRuntimeGraphGrowth(dir, model, gpus, 1, "llama")[0]; got != 900 {
				t.Fatalf("old growth should remain fallback evidence: %d", got)
			}
			if err := RecordRuntimeGraphGrowth(dir, model, 32768, 256, "high", "gpu", "llama", gpus, 1, map[int]int{0: 100}); err != nil {
				t.Fatal(err)
			}
			if got := RelatedModelRuntimeGraphGrowth(dir, model, gpus, 1, "llama")[0]; got != 100 {
				t.Fatalf("old mislabelled growth outranked fresh serving evidence: %d", got)
			}
		})
	}
}

func TestNewComputeReadingKeepsItsOwnRole(t *testing.T) {
	dir := t.TempDir()
	gpus, model := growthCarryFixture()
	if err := writeProbeCacheForModel(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1, map[int]int{0: 4000}, nil, nil, 0,
		probeMeasurements{ComputeBufEvidence: "live-allocated"}); err != nil {
		t.Fatal(err)
	}
	if err := RecordMeasuredComputeBuffers(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1, map[int]int{1: 200}, map[int]bool{1: true}); err != nil {
		t.Fatal(err)
	}
	pc := loadProbeCache(dir, model, 32768, 128, "high", "gpu", "llama", gpus, 1)
	if pc == nil || pc.ComputeBufByGPU[1] != 200 || !pc.ComputeBufExpertOnlyByGPU[1] || pc.ComputeBufExpertOnlyByGPU[0] {
		t.Fatalf("sparse update did not preserve per-device roles: %+v", pc)
	}
}
