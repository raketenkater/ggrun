package placement

import (
	"math"
	"reflect"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestHotCacheUsesEmittedRouterPlacement(t *testing.T) {
	caps, model, s, opts := hotExpertFixture()
	model.NumLayers = 3
	model.RoutedExpertLayerBytes = []int64{4 * hotExpertTestMiB, 4 * hotExpertTestMiB, 4 * hotExpertTestMiB}
	s.OTString = "exps=CPU"
	s.TensorSplit = []float64{0.504, 0.496} // argv emits 0.50,0.50; strict boundary sends layer 2 to GPU7.
	shape, err := hotExpertCacheShapeFor(caps, model, s, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(shape.layersByGPU, map[int]int{3: 2, 7: 1}) {
		t.Fatalf("cache charged different owners than emitted argv: %v", shape.layersByGPU)
	}
}

func TestHotCacheRejectsUnprovenRouterSplit(t *testing.T) {
	for _, split := range [][]float64{{0, 0}, {math.NaN(), 1}, {-1, 2}} {
		caps, model, s, opts := hotExpertFixture()
		s.TensorSplit = split
		if _, err := hotExpertCacheShapeFor(caps, model, s, opts); err == nil {
			t.Fatalf("unproven router split accepted: %v", split)
		}
	}
}

func TestCacheMemoryCannotBecomeMeasuredGraphGrowth(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 12282}}
	model := &ModelProfile{Path: "m.gguf", NumLayers: 45}
	s := &Strategy{ContextSize: 32768, UBatchSize: 256, KVQuality: "high", KVPlacement: "gpu", Parallel: 1, HotExpertCacheSlots: 16, HotExpertCacheVRAMByGPU: map[int]int{0: 2000}}
	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, "llama", gpus, 1, 0, 700, false); err != nil {
		t.Fatal(err)
	}
	restore := serveVRAMSampler
	defer func() { serveVRAMSampler = restore }()
	serveVRAMSampler = func(int) int { return 5500 } // 3000 logged + cache + driver residual. Actual cache distribution unknown.
	if got := RecordRuntimeGraphGrowthFromServe(dir, model, s, "llama", gpus, serveLogTwoRounds, nil); len(got) != 0 {
		t.Fatalf("unattributed cache memory recorded as graph growth: %v", got)
	}
	pc := loadProbeCache(dir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, "llama", gpus, 1)
	if pc == nil || pc.RuntimeGraphGrowthByGPU[0] != 700 || !pc.RuntimeGraphGrowthFromOOMByGPU[0] {
		t.Fatalf("missing attribution erased prior OOM evidence: %+v", pc)
	}
}

func TestHotCacheAccountingScopeRejectsOldMixedResidual(t *testing.T) {
	old := "llama|hot-experts=16,inserts=2"
	got := ScopedBackendRuntimeFeatureTag(old, false, 16, 2)
	if got == old {
		t.Fatal("cache plus graph residual remained reusable as graph-only evidence")
	}
	if off := ScopedBackendRuntimeFeatureTag(old, false, 0, 0); off != "llama" {
		t.Fatalf("cache-free scope changed: %s", off)
	}
	if again := ScopedBackendRuntimeFeatureTag(got, false, 16, 2); again != got {
		t.Fatalf("scope is not idempotent: %s -> %s", got, again)
	}
}

func TestHotCacheRejectsPartialRouterOffload(t *testing.T) {
	for _, n := range []int{0, 1, 4} {
		caps, m, s, o := hotExpertFixture()
		s.GPULayers = n
		if _, err := hotExpertCacheShapeFor(caps, m, s, o); err == nil {
			t.Fatalf("partial layer range admitted as full router placement: %d", n)
		}
	}
}
