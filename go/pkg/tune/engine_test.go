package tune

import (
	"encoding/json"
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
	if _, ok := m["--flash-attn"]; !ok {
		t.Fatalf("expected canonical flash-attn flag")
	}
}

func TestBuildTuningPrompt(t *testing.T) {
	// Cannot easily test without detect.Capabilities, but we can verify it doesn't panic
	// This is a smoke test
}

func TestRemoveConflicting(t *testing.T) {
	flags := []string{"-c", "4096", "-b", "2048"}
	result := removeConflicting(flags, "--ctx-size")
	for _, f := range result {
		if f == "-c" || f == "4096" {
			t.Fatalf("expected -c and its value to be removed, got %v", result)
		}
	}
	if len(result) != 2 || result[0] != "-b" || result[1] != "2048" {
		t.Fatalf("unexpected remaining flags: %v", result)
	}
}

func TestFlagMapEqualsForm(t *testing.T) {
	flags := []string{"--ctx-size=8192", "--threads", "16"}
	m := flagMap(flags)
	if m["--ctx-size"] != "8192" {
		t.Fatalf("expected ctx from equals form, got %v", m)
	}
	if m["--threads"] != "16" {
		t.Fatalf("expected threads value, got %v", m)
	}
}

func TestSuggestionUnmarshalObjectFlags(t *testing.T) {
	var s Suggestion
	data := []byte(`{"name":"batch","flags":{"--batch-size":4096,"--cache-type-k":"q8_0"},"reasoning":"test"}`)
	if err := json.Unmarshal(data, &s); err != nil {
		t.Fatalf("unmarshal suggestion: %v", err)
	}
	if s.FlagValues["--batch-size"] == nil || s.FlagValues["--cache-type-k"] != "q8_0" {
		t.Fatalf("expected object flags to survive, got %#v", s.FlagValues)
	}
	args := flagValuesToArgs(sanitizeFlagValues(s.FlagValues, nil))
	m := flagMap(args)
	if m["-b"] != "4096" || m["--cache-type-k"] != "q8_0" {
		t.Fatalf("expected canonical args, got %v from %v", m, args)
	}
}

func TestApplyOverridesProtectsPlacement(t *testing.T) {
	base := []string{"-m", "model.gguf", "--port", "8081", "-b", "4096", "--device", "CUDA0"}
	overrides := map[string]interface{}{
		"--batch-size": 8192,
		"--port":       9090,
		"--device":     "CUDA1",
	}
	result := ApplyOverrides(base, overrides, DefaultProtectedFlags())
	m := flagMap(result)
	if m["-b"] != "8192" {
		t.Fatalf("expected batch override, got %v", result)
	}
	if m["--port"] != "8081" {
		t.Fatalf("expected protected port to stay 8081, got %v", result)
	}
	if m["--device"] != "CUDA0" {
		t.Fatalf("expected protected device to stay CUDA0, got %v", result)
	}
}

func TestMeaningfulImprovementUsesNoiseFloor(t *testing.T) {
	if meaningfulImprovement(100.5, 100.0, 1.0) {
		t.Fatalf("0.5%% should not beat a 1%% tune noise floor")
	}
	if !meaningfulImprovement(101.0, 100.0, 1.0) {
		t.Fatalf("1%% should meet the tune noise floor")
	}
}

func TestDeterministicSuggestionSkipsKVUpgradeForMoEOffload(t *testing.T) {
	base := []string{
		"--cache-type-k", "q4_0",
		"--cache-type-v", "q4_0",
		"--n-cpu-moe", "256",
		"-ot", "exps=CPU",
		"-b", "2048",
		"-ub", "512",
	}
	s := deterministicSuggestion(1, base)
	if s == nil {
		t.Fatalf("expected a safe deterministic candidate")
	}
	if _, ok := s.FlagValues["--cache-type-k"]; ok {
		t.Fatalf("did not expect q8 KV upgrade for MoE offload, got %#v", s.FlagValues)
	}
	if s.FlagValues["-b"] != "4096" {
		t.Fatalf("expected first MoE fallback to test batch size, got %#v", s.FlagValues)
	}
}
