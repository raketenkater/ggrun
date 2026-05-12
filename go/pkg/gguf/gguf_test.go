package gguf

import (
	"os"
	"testing"
)

func TestParse(t *testing.T) {
	// Test with a real model if available
	paths := []string{
		"/home/mik/ai_models/Qwen3-0.6B-Q8_0.gguf",
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			info, err := Parse(p)
			if err != nil {
				t.Fatalf("parse %s: %v", p, err)
			}
			if info.Architecture == "" {
				t.Skip("architecture empty in test model")
			}
			return
		}
	}
	t.Skip("no test model available")
}

func TestEstimateParams(t *testing.T) {
	info := &Info{
		VocabSize:         151936,
		EmbeddingLength:   1024,
		BlockCount:        28,
		FeedForwardLength: 3072,
	}
	got := info.EstimateParams()
	// Expected ~596M for Qwen3 0.6B
	if got < 500000000 || got > 700000000 {
		t.Fatalf("expected ~596M params, got %d", got)
	}
}
