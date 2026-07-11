package placement

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func countOTLayersByDevice(ot string) map[int]int {
	out := map[int]int{}
	for _, part := range strings.Split(ot, ",") {
		m := otDevicePattern.FindStringSubmatch(part)
		if m == nil {
			continue
		}
		dev, _ := strconv.Atoi(m[2])
		out[dev] += len(strings.Split(m[1], "|"))
	}
	return out
}

func TestReplanAfterOOMReducesFailedDevice(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "3090 Ti", VRAMTotalMB: 24576, VRAMUsedMB: 800, BandwidthMBps: 31504},
			{Index: 1, Name: "3060", VRAMTotalMB: 12288, VRAMUsedMB: 600, BandwidthMBps: 12000},
			{Index: 2, Name: "4070", VRAMTotalMB: 12288, VRAMUsedMB: 600, BandwidthMBps: 25203},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 120000},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "V4.gguf", TotalSizeMB: 146 * 1024, SizeBytes: 146 * 1024 * 1024 * 1024,
		NumLayers: 43, IsMoE: true, NumExperts: 256,
		ExpertBytes: int64(43 * 3289 * 1024 * 1024), NonExpertBytes: int64(7680 * 1024 * 1024),
		ContextSize: 32768, EmbeddingLength: 4096, HeadCountKV: 1, KeyLength: 512, ValueLength: 512,
		ExpertUsedCount: 6, ExpertFF: 2048,
	}
	opts := Options{ContextSize: 32768, KVQuality: "low", KVPlacement: "cpu", CacheDir: t.TempDir()}

	base, err := Compute(caps, model, opts)
	if err != nil || base.Type != MoEOffload {
		t.Fatalf("base compute: type=%v err=%v", base.Type, err)
	}
	baseC := countOTLayersByDevice(base.OTString)

	// Device 2 OOM'd by ~8 GB — the re-plan must shed layers from it.
	replan, err := ReplanAfterOOM(caps, model, opts, map[int]int{2: 8000})
	if err != nil || replan == nil {
		t.Fatalf("replan: %v", err)
	}
	newC := countOTLayersByDevice(replan.OTString)
	if newC[2] >= baseC[2] {
		t.Errorf("re-plan should reduce device 2 layers: base=%d new=%d (ot=%s)", baseC[2], newC[2], replan.OTString)
	}
	if replan.OTString == "" || !strings.Contains(replan.OTString, "exps=CPU") {
		t.Errorf("re-plan produced no valid MoE -ot: %q", replan.OTString)
	}
}

func TestPlacementCachePathFor_KeyedByKVAndCtx(t *testing.T) {
	m := &ModelProfile{Path: "model.gguf", NumLayers: 43, NumExperts: 256, EmbeddingLength: 4096}
	gpus := []detect.GPU{{Index: 0, Name: "3090 Ti"}, {Index: 1, Name: "3060"}}
	dir := "/tmp/cache"

	gpuKV := PlacementCachePathFor(dir, m, 1048576, 512, "mid", "gpu", "llama", gpus, 0, "")
	cpuKV := PlacementCachePathFor(dir, m, 1048576, 512, "mid", "cpu", "llama", gpus, 0, "")
	smallCtx := PlacementCachePathFor(dir, m, 131072, 512, "mid", "gpu", "llama", gpus, 0, "")

	if gpuKV == cpuKV {
		t.Errorf("kv=gpu and kv=cpu must not share a placement cache file:\n  %s", gpuKV)
	}
	if gpuKV == smallCtx {
		t.Errorf("different context sizes must not share a placement cache file")
	}
	// Deterministic + lands in the cache dir with a .place extension.
	if again := PlacementCachePathFor(dir, m, 1048576, 512, "mid", "gpu", "llama", gpus, 0, ""); again != gpuKV {
		t.Errorf("path must be deterministic: %s != %s", again, gpuKV)
	}
	if filepath.Dir(gpuKV) != dir || filepath.Ext(gpuKV) != ".place" {
		t.Errorf("unexpected path %s (want %s/*.place)", gpuKV, dir)
	}
}

func TestPlacementCacheRejectsLegacyMissingMMap(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy.place")
	content := `CACHED_OT_STRING="exps=CPU"
CACHED_TENSOR_SPLIT="0.86,0.03,0.11"
CACHED_SPLIT_MODE="layer"
CACHED_NCPUMOE="34"
CACHED_BATCH="2048"
CACHED_UBATCH="512"
CACHED_PARALLEL="4"
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}}}
	if _, err := LoadPlacementCache(path, caps, 0); err == nil || !strings.Contains(err.Error(), "CACHED_MMAP") {
		t.Fatalf("expected legacy cache without mmap mode to be rejected, got %v", err)
	}
}

func TestPlacementCacheRoundTripPreservesSubPins(t *testing.T) {
	// A placement WITH sub-layer gate+up pins must survive save -> load exactly,
	// so the squeeze isn't silently dropped on the next launch.
	ot := `blk\.(0|1|2)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,` +
		`blk\.(8)\.ffn_(gate_up|up_gate|gate|up)_exps.*=CUDA0,exps=CPU`
	strat := &Strategy{
		Type:        MoEOffload,
		OTString:    ot,
		TensorSplit: []float64{0.86, 0.03, 0.11},
		SplitMode:   "layer",
		NCPUMoE:     33,
		BatchSize:   2048,
		UBatchSize:  512,
		Parallel:    1,
		MMap:        false,
	}
	entry := StrategyToCacheEntry(strat)
	if entry.OTString != ot {
		t.Fatalf("StrategyToCacheEntry dropped OTString")
	}

	path := filepath.Join(t.TempDir(), "x.place")
	if err := SavePlacementCache(path, entry); err != nil {
		t.Fatalf("save: %v", err)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}}}
	loaded, err := LoadPlacementCache(path, caps, 0)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.OTString != ot {
		t.Errorf("sub-pin -ot not preserved:\n got=%s\nwant=%s", loaded.OTString, ot)
	}
	if loaded.NCPUMoE != 33 || len(loaded.TensorSplit) != 3 {
		t.Errorf("cache round-trip lost fields: ncpumoe=%d split=%v", loaded.NCPUMoE, loaded.TensorSplit)
	}
}
