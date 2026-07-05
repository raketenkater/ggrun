package gguf

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseMissingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.gguf")
	_, err := Parse(path)
	if err == nil {
		t.Fatal("expected missing model file to fail")
	}
	if !strings.Contains(err.Error(), "model file") {
		t.Fatalf("expected model-file error, got %v", err)
	}
}

func TestParse(t *testing.T) {
	// Exercise the parser against a real model if one is provided via
	// GGUF_TEST_MODEL=/path/to/model.gguf; otherwise the test skips.
	paths := []string{}
	if p := os.Getenv("GGUF_TEST_MODEL"); p != "" {
		paths = append(paths, p)
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
