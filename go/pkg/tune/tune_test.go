package tune

import (
	"testing"
)

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
