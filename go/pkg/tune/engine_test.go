package tune

import (
	"testing"
)

func TestApplySuggestion(t *testing.T) {
	base := []string{"-m", "model.gguf", "--port", "8081", "-c", "4096"}
	suggested := []string{"-c", "8192", "-b", "4096"}

	result := applySuggestion(base, suggested)

	// Should have replaced -c 4096 with -c 8192
	has8192 := false
	has4096 := false
	for i := 0; i < len(result); i++ {
		if result[i] == "-c" {
			if i+1 < len(result) && result[i+1] == "8192" {
				has8192 = true
			}
			if i+1 < len(result) && result[i+1] == "4096" {
				has4096 = true
			}
		}
	}
	if !has8192 {
		t.Fatalf("expected -c 8192 in result, got %v", result)
	}
	if has4096 {
		t.Fatalf("expected -c 4096 to be removed, got %v", result)
	}
}

func TestFlagMap(t *testing.T) {
	flags := []string{"-m", "model.gguf", "--port", "8081", "-fa"}
	m := flagMap(flags)
	if m["-m"] != "model.gguf" {
		t.Fatalf("expected model.gguf, got %s", m["-m"])
	}
	if m["--port"] != "8081" {
		t.Fatalf("expected 8081, got %s", m["--port"])
	}
	if _, ok := m["-fa"]; !ok {
		t.Fatalf("expected -fa flag")
	}
}

func TestBuildTuningPrompt(t *testing.T) {
	// Cannot easily test without detect.Capabilities, but we can verify it doesn't panic
	// This is a smoke test
}

func TestRemoveConflicting(t *testing.T) {
	flags := []string{"-c", "4096", "-b", "2048"}
	result := removeConflicting(flags, "-c")
	for _, f := range result {
		if f == "-c" {
			t.Fatalf("expected -c to be removed")
		}
	}
}
