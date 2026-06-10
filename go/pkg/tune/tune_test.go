package tune

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTuneCachePathUsesSplitGGUFTotalSize(t *testing.T) {
	tmpDir := t.TempDir()
	modelDir := filepath.Join(tmpDir, "model")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	shards := map[string]int{
		"Split-Model-00001-of-00003.gguf": 2,
		"Split-Model-00002-of-00003.gguf": 3,
		"Split-Model-00003-of-00003.gguf": 5,
	}
	for name, size := range shards {
		if err := os.WriteFile(filepath.Join(modelDir, name), make([]byte, size), 0644); err != nil {
			t.Fatalf("write shard: %v", err)
		}
	}

	path := TuneCachePath(tmpDir, filepath.Join(modelDir, "Split-Model-00001-of-00003.gguf"), []string{"GPU"}, false, "ik_llama")
	if !strings.Contains(path, "_10_hw") {
		t.Fatalf("expected total split size in cache path, got %s", path)
	}
}

func TestCache(t *testing.T) {
	tmpDir := t.TempDir()
	c := NewCache(tmpDir)

	// Add first entry
	e1 := Entry{
		Timestamp:    1,
		ModelPath:    "/models/test.gguf",
		HardwareHash: "abc123",
		Round:        0,
		Result:       BenchmarkResult{GenTPS: 10.5},
		Best:         true,
	}
	if err := c.Add(e1); err != nil {
		t.Fatalf("add: %v", err)
	}

	// Add better entry
	e2 := Entry{
		Timestamp:    2,
		ModelPath:    "/models/test.gguf",
		HardwareHash: "abc123",
		Round:        1,
		Result:       BenchmarkResult{GenTPS: 15.2},
		Best:         true,
	}
	if err := c.Add(e2); err != nil {
		t.Fatalf("add: %v", err)
	}

	best, err := c.FindBest("/models/test.gguf", "abc123")
	if err != nil {
		t.Fatalf("find best: %v", err)
	}
	if best == nil {
		t.Fatalf("expected best entry")
	}
	if best.Result.GenTPS != 15.2 {
		t.Fatalf("expected 15.2 tps, got %f", best.Result.GenTPS)
	}
	if best.Round != 1 {
		t.Fatalf("expected round 1, got %d", best.Round)
	}
}

func TestKey(t *testing.T) {
	k := Key("model.gguf", "10GB", "hw1", "vision", "ik_llama")
	if k == "" {
		t.Fatalf("key empty")
	}
}

func TestHardwareHash(t *testing.T) {
	h := HardwareHash([]string{"RTX 4070", "RTX 3090"}, 36864)
	if h == "" {
		t.Fatalf("hash empty")
	}
}

func TestCacheKeepsFasterBest(t *testing.T) {
	tmpDir := t.TempDir()
	c := NewCache(tmpDir)

	fast := Entry{Timestamp: 1, ModelPath: "/models/test.gguf", HardwareHash: "abc123", Round: 0, Result: BenchmarkResult{GenTPS: 20}, Best: true}
	if err := c.Add(fast); err != nil {
		t.Fatalf("add fast: %v", err)
	}
	slow := Entry{Timestamp: 2, ModelPath: "/models/test.gguf", HardwareHash: "abc123", Round: 1, Result: BenchmarkResult{GenTPS: 10}, Best: true}
	if err := c.Add(slow); err != nil {
		t.Fatalf("add slow: %v", err)
	}

	best, err := c.FindBest("/models/test.gguf", "abc123")
	if err != nil {
		t.Fatalf("find best: %v", err)
	}
	if best == nil || best.Result.GenTPS != 20 || best.Round != 0 {
		t.Fatalf("expected original fast best, got %#v", best)
	}
}

func TestCacheBestIsScopedByBackendAndVision(t *testing.T) {
	tmpDir := t.TempDir()
	c := NewCache(tmpDir)

	ik := Entry{Timestamp: 1, ModelPath: "/models/test.gguf", HardwareHash: "abc123", Backend: "ik_llama", Result: BenchmarkResult{GenTPS: 50}, Best: true}
	vk := Entry{Timestamp: 2, ModelPath: "/models/test.gguf", HardwareHash: "abc123", Backend: "vulkan", Result: BenchmarkResult{GenTPS: 30}, Best: true}
	vision := Entry{Timestamp: 3, ModelPath: "/models/test.gguf", HardwareHash: "abc123", Backend: "ik_llama", Vision: true, Result: BenchmarkResult{GenTPS: 20}, Best: true}
	for _, e := range []Entry{ik, vk, vision} {
		if err := c.Add(e); err != nil {
			t.Fatalf("add: %v", err)
		}
	}

	fasterIK := Entry{Timestamp: 4, ModelPath: "/models/test.gguf", HardwareHash: "abc123", Backend: "ik_llama", Result: BenchmarkResult{GenTPS: 55}, Best: true}
	if err := c.Add(fasterIK); err != nil {
		t.Fatalf("add faster ik: %v", err)
	}

	entries, err := c.Load()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	bestByScope := map[string]int{}
	for _, e := range entries {
		if e.Best {
			bestByScope[e.Backend+"/"+boolString(e.Vision)]++
		}
	}
	if bestByScope["ik_llama/false"] != 1 {
		t.Fatalf("expected one non-vision IK best, got %d", bestByScope["ik_llama/false"])
	}
	if bestByScope["vulkan/false"] != 1 {
		t.Fatalf("expected Vulkan best to survive, got %d", bestByScope["vulkan/false"])
	}
	if bestByScope["ik_llama/true"] != 1 {
		t.Fatalf("expected vision IK best to survive, got %d", bestByScope["ik_llama/true"])
	}
}

func TestSaveTuneFileMarksFinalRunCompleteWithSkippedRounds(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := filepath.Join(tmpDir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}

	c := NewCache(tmpDir)
	baseline := &Entry{
		ModelPath: modelPath,
		Round:     0,
		Name:      "baseline",
		Result:    BenchmarkResult{GenTPS: 37.8, PromptTPS: 247.6},
	}
	entries := []Entry{
		*baseline,
		{ModelPath: modelPath, Round: 6, Name: "threads", Result: BenchmarkResult{GenTPS: 37.6, PromptTPS: 248}},
	}

	path, err := c.SaveTuneFile(modelPath, baseline, baseline, 8, "vulkan", false, 1, []string{"GPU"}, entries, true)
	if err != nil {
		t.Fatalf("save tune file: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read tune file: %v", err)
	}
	var doc map[string]interface{}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("unmarshal tune file: %v", err)
	}
	if doc["complete"] != true {
		t.Fatalf("expected complete tune file, got %#v", doc["complete"])
	}
	if got := doc["completed_rounds"]; got != float64(8) {
		t.Fatalf("expected completed_rounds 8, got %#v", got)
	}
}

func boolString(v bool) string {
	if v {
		return "true"
	}
	return "false"
}
