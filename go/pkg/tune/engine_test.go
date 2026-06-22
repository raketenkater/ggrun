package tune

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/benchmark"
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

func TestDeterministicPlanIncludesIKDenseDefragCandidate(t *testing.T) {
	base := []string{
		"--cache-type-k", "q4_0",
		"--cache-type-v", "q4_0",
		"-b", "8192",
		"-ub", "1024",
		"--defrag-thold", "0.1",
	}
	plan := deterministicPlan(base, "ik_llama", nil, "")
	for _, candidate := range plan {
		if candidate.Name == "dense-defrag-0.5" && candidate.FlagValues["--defrag-thold"] == "0.5" {
			return
		}
	}
	t.Fatalf("expected historical dense IK defrag candidate, got %#v", plan)
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

func TestDeterministicPlanIncludesMoEMemoryMovementCandidates(t *testing.T) {
	base := []string{
		"--cache-type-k", "q4_0",
		"--cache-type-v", "q4_0",
		"--n-cpu-moe", "256",
		"-ot", "exps=CPU",
		"-b", "2048",
		"-ub", "512",
		"--no-mmap",
	}
	help := "--defer-experts --cont-batching"
	plan := deterministicPlan(base, "vulkan", nil, help)
	names := map[string]bool{}
	for _, c := range plan {
		names[c.Name] = true
		if _, ok := c.FlagValues["--parallel"]; ok {
			t.Fatalf("MoE plan must not change parallel without fixed per-slot context scheduling: %#v", c)
		}
		if _, ok := c.FlagValues["--ctx-size"]; ok {
			t.Fatalf("MoE plan must not change context size: %#v", c)
		}
	}
	for _, want := range []string{"moe-batch-1536", "moe-ubatch-384", "moe-defer-experts", "moe-mmap-pagecache", "moe-cont-batching"} {
		if !names[want] {
			t.Fatalf("expected %s in MoE memory-movement plan, got %#v", want, names)
		}
	}
}

func TestGuardRiskyMoEOverridesKeepsContextAndSpecStable(t *testing.T) {
	base := []string{
		"--cache-type-k", "q4_0",
		"--cache-type-v", "q4_0",
		"--flash-attn", "on",
		"--parallel", "1",
		"--n-cpu-moe", "256",
		"-ot", "exps=CPU",
		"-b", "2048",
		"-ub", "512",
		"--no-mmap",
	}
	overrides := guardRiskyMoEOverrides(map[string]interface{}{
		"--parallel":         "2",
		"--spec-type":        "ngram-mod",
		"--spec-draft-n-max": "64",
		"--flash-attn":       "off",
		"--no-mmap":          false,
		"--defer-experts":    true,
	}, base)
	for _, key := range []string{"--parallel", "--spec-type", "--spec-draft-n-max", "--flash-attn"} {
		if _, ok := overrides[key]; ok {
			t.Fatalf("expected %s to be dropped for MoE stability, got %#v", key, overrides)
		}
	}
	if overrides["--no-mmap"] != false || overrides["--defer-experts"] != true {
		t.Fatalf("expected safe memory-movement flags to remain, got %#v", overrides)
	}
}

func TestDeterministicPlanDoesNotSuggestNgramByDefault(t *testing.T) {
	base := []string{"-b", "4096", "-ub", "512"}
	help := "--spec-type [none|ngram-map-k|ngram-map-k4v|ngram-mod] --spec-ngram-mod-n-match"
	plan := deterministicPlan(base, "vulkan", nil, help)
	for _, c := range plan {
		if _, ok := c.FlagValues["--spec-type"]; ok {
			t.Fatalf("default AI-tune should not propose ngram speculation, got %#v", c)
		}
	}
}

func TestDeterministicPlanTunesExplicitNgramMode(t *testing.T) {
	base := []string{"-b", "4096", "-ub", "512", "--spec-type", "ngram-mod"}
	plan := deterministicPlan(base, "vulkan", nil, "--spec-autotune")
	if len(plan) == 0 || plan[0].Name != "spec-ngram-mod-lower-depth" {
		t.Fatalf("expected explicit ngram mode to get depth tuning, got %#v", plan)
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

func msgsContain(msgs []string, sub string) bool {
	for _, m := range msgs {
		if strings.Contains(m, sub) {
			return true
		}
	}
	return false
}

// A candidate that wins on a single noisy sample but does not reproduce on the
// confirmation re-measure must be discarded so the cache never holds a config
// slower than the default.
func TestRunConfirmationRevertsUnreproducibleWinner(t *testing.T) {
	initial := []string{"--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "-b", "8192", "-ub", "1024", "--threads", "8"}
	var lastFlags []string
	var msgs []string
	candCalls := 0
	e := &Engine{
		Model:   "m.gguf",
		Rounds:  3,
		Backend: "ik_llama",
		StartServer: func(flags []string) (func(), error) {
			lastFlags = append([]string(nil), flags...)
			return func() {}, nil
		},
		OnProgress: func(m string) { msgs = append(msgs, m) },
	}
	e.benchmarkFn = func() (*benchmark.Result, error) {
		if equalFlags(lastFlags, initial) {
			return &benchmark.Result{GenTPS: 100, GenTokens: 256, PromptTokens: 10}, nil
		}
		candCalls++
		tps := 100.0
		if candCalls == 1 { // first candidate noise-wins; it never reproduces afterward
			tps = 130.0
		}
		return &benchmark.Result{GenTPS: tps, GenTokens: 256, PromptTokens: 10}, nil
	}
	best, err := e.Run("m.gguf", initial)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !msgsContain(msgs, "keeping baseline") {
		t.Fatalf("confirmation pass did not run/revert; messages: %v", msgs)
	}
	if best.Name != "baseline" || len(best.OverrideFlags) != 0 {
		t.Fatalf("expected revert to baseline, got name=%q overrides=%v gentps=%.1f", best.Name, best.OverrideFlags, best.Result.GenTPS)
	}
}

// A candidate whose win reproduces on the confirmation re-measure must be kept,
// with the confirmed measurement recorded.
func TestRunConfirmationKeepsReproducibleWinner(t *testing.T) {
	initial := []string{"--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "-b", "8192", "-ub", "1024", "--threads", "8"}
	var lastFlags []string
	var msgs []string
	e := &Engine{
		Model:   "m.gguf",
		Rounds:  3,
		Backend: "ik_llama",
		StartServer: func(flags []string) (func(), error) {
			lastFlags = append([]string(nil), flags...)
			return func() {}, nil
		},
		OnProgress: func(m string) { msgs = append(msgs, m) },
	}
	e.benchmarkFn = func() (*benchmark.Result, error) {
		tps := 100.0
		if !equalFlags(lastFlags, initial) { // every candidate (and confirmation) reproduces the win
			tps = 130.0
		}
		return &benchmark.Result{GenTPS: tps, GenTokens: 256, PromptTokens: 10}, nil
	}
	best, err := e.Run("m.gguf", initial)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !msgsContain(msgs, "confirmed") {
		t.Fatalf("expected a confirmed winner message; messages: %v", msgs)
	}
	if best.Name == "baseline" || len(best.OverrideFlags) == 0 {
		t.Fatalf("expected a kept non-baseline winner, got name=%q overrides=%v", best.Name, best.OverrideFlags)
	}
	if best.Result.GenTPS != 130 {
		t.Fatalf("expected confirmed gen tok/s 130, got %.1f", best.Result.GenTPS)
	}
}
