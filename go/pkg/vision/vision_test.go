package vision

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFindLocal(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := filepath.Join(tmpDir, "qwen3-vl.gguf")
	os.WriteFile(modelPath, []byte("model"), 0644)

	// No mmproj yet
	_, err := FindLocal(modelPath)
	if err == nil {
		t.Fatalf("expected error when no mmproj exists")
	}

	// Create mmproj
	mmprojPath := filepath.Join(tmpDir, "mmproj-F16.gguf")
	os.WriteFile(mmprojPath, []byte("proj"), 0644)

	found, err := FindLocal(modelPath)
	if err != nil {
		t.Fatalf("find local: %v", err)
	}
	if found != mmprojPath {
		t.Fatalf("expected %s, got %s", mmprojPath, found)
	}
}

func TestMatchesModel(t *testing.T) {
	// Generic mmproj in same directory matches any model
	if !matchesModel("mmproj-F16.gguf", "any-model") {
		t.Fatalf("expected generic mmproj to match")
	}
	// Named mmproj matching model
	if !matchesModel("mmproj-Qwen3-vl-F16.gguf", "Qwen3-VL") {
		t.Fatalf("expected match")
	}
	// No match for different models
	if matchesModel("mmproj-qwen3-vl-F16.gguf", "mistral") {
		t.Fatalf("expected no match")
	}
}

func TestLabel(t *testing.T) {
	if got := Label("mmproj-F16.gguf"); got != "f16" {
		t.Fatalf("expected f16, got %s", got)
	}
	if got := Label("mmproj-bf16.gguf"); got != "bf16" {
		t.Fatalf("expected bf16, got %s", got)
	}
}
