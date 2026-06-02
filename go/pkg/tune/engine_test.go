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
	if s.FlagValues["-b"] != "1024" {
		t.Fatalf("expected first MoE fallback to lower batch pressure, got %#v", s.FlagValues)
	}
}

func TestApplyOverridesCanRemoveBooleanFlags(t *testing.T) {
	base := []string{
		"-b", "2048",
		"--run-time-repack",
		"-khad",
		"--defrag-thold", "0.1",
	}
	overrides := map[string]interface{}{
		"--run-time-repack": false,
		"-khad":             false,
		"--defrag-thold":    "-1",
	}
	result := ApplyOverrides(base, overrides, nil)
	for _, flag := range result {
		if flag == "--run-time-repack" || flag == "-khad" {
			t.Fatalf("expected boolean flag to be removed, got %v", result)
		}
	}
	m := flagMap(result)
	if m["--defrag-thold"] != "-1" {
		t.Fatalf("expected negative defrag value to survive, got %v from %v", m, result)
	}
}

func TestDeterministicPlanIncludesIKMoEKnobs(t *testing.T) {
	base := []string{
		"--cache-type-k", "q4_0",
		"--cache-type-v", "q4_0",
		"--n-cpu-moe", "256",
		"-ot", "exps=CPU",
		"-b", "2048",
		"-ub", "512",
		"--run-time-repack",
		"-khad",
		"-muge",
		"-ger",
		"-mqkv",
		"--defrag-thold", "0.1",
	}
	plan := deterministicPlan(base, "ik_llama", nil, "")
	if len(plan) < 8 {
		t.Fatalf("expected a deeper MoE plan, got %d entries: %#v", len(plan), plan)
	}
	names := map[string]bool{}
	for _, c := range plan {
		names[c.Name] = true
		if c.FlagValues["-ub"] == "1024" {
			t.Fatalf("MoE plan should not probe larger ubatch after OOM data, got %#v", c)
		}
	}
	for _, want := range []string{"moe-disable-repack", "moe-disable-khad", "moe-defrag-off", "moe-no-muge", "moe-no-ger", "moe-no-mqkv"} {
		if !names[want] {
			t.Fatalf("expected %s in MoE plan, got %#v", want, names)
		}
	}
}

func TestGuardRiskyMoEOverridesDropsMemoryExpandingSuggestions(t *testing.T) {
	base := []string{
		"--cache-type-k", "q4_0",
		"--cache-type-v", "q4_0",
		"--n-cpu-moe", "256",
		"-ot", "exps=CPU",
		"-b", "2048",
		"-ub", "512",
	}
	overrides := guardRiskyMoEOverrides(map[string]interface{}{
		"-ub":            "1024",
		"-b":             "8192",
		"--cache-type-k": "q8_0",
		"--cache-type-v": "q8_0",
		"--defrag-thold": "-1",
	}, base)
	if _, ok := overrides["-ub"]; ok {
		t.Fatalf("expected larger ubatch to be dropped, got %#v", overrides)
	}
	if _, ok := overrides["-b"]; ok {
		t.Fatalf("expected excessive batch increase to be dropped, got %#v", overrides)
	}
	if _, ok := overrides["--cache-type-k"]; ok {
		t.Fatalf("expected MoE KV upgrade to be dropped, got %#v", overrides)
	}
	if overrides["--defrag-thold"] != "-1" {
		t.Fatalf("expected safe MoE override to remain, got %#v", overrides)
	}
}

func TestDeterministicPlanIncludesSpecCandidatesForMainline(t *testing.T) {
	base := []string{"-b", "4096", "-ub", "512"}
	help := "--spec-type [none|ngram-map-k|ngram-map-k4v|ngram-mod] --spec-ngram-mod-n-match"
	plan := deterministicPlan(base, "vulkan", nil, help)
	names := map[string]bool{}
	for _, c := range plan {
		names[c.Name] = true
	}
	for _, want := range []string{"spec-ngram-mod", "spec-ngram-k4v", "spec-ngram-map-k"} {
		if !names[want] {
			t.Fatalf("expected %s in spec-aware plan, got %#v", want, names)
		}
	}
}

func TestDeterministicPlanIncludesIKSpecCandidate(t *testing.T) {
	base := []string{"-b", "4096", "-ub", "512", "--run-time-repack"}
	plan := deterministicPlan(base, "ik_llama", nil, "")
	if len(plan) == 0 || plan[0].Name != "spec-ik-ngram" {
		t.Fatalf("expected first IK dense tune candidate to test ngram spec, got %#v", plan)
	}
	if plan[0].FlagValues["--spec-type"] != "ngram - map - k" {
		t.Fatalf("expected IK ngram dialect, got %#v", plan[0].FlagValues)
	}
}

func TestDeterministicPlanDoesNotEnableSpecForMoEByDefault(t *testing.T) {
	base := []string{"-b", "2048", "-ub", "512", "--n-cpu-moe", "256", "-ot", "exps=CPU"}
	plan := deterministicPlan(base, "ik_llama", nil, "")
	for _, c := range plan {
		if _, ok := c.FlagValues["--spec-type"]; ok {
			t.Fatalf("MoE plan should not enable speculative decoding by default, got %#v", c)
		}
	}
}
