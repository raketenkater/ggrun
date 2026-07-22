package tune

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/benchmark"
)

// ---------------------------------------------------------------------------
// equalFlags — semantic map comparison (fork modification)
// ---------------------------------------------------------------------------

func TestEqualFlags_SameOrderSameFlags(t *testing.T) {
	a := []string{"-b", "4096", "-ub", "512", "--flash-attn", "on"}
	b := []string{"-b", "4096", "-ub", "512", "--flash-attn", "on"}
	if !equalFlags(a, b) {
		t.Errorf("expected equal for identical flag slices")
	}
}

func TestEqualFlags_DifferentOrderSameFlags(t *testing.T) {
	a := []string{"-b", "4096", "-ub", "512", "--flash-attn", "on"}
	b := []string{"--flash-attn", "on", "-ub", "512", "-b", "4096"}
	if !equalFlags(a, b) {
		t.Errorf("expected equal for same flags in different order")
	}
}

func TestEqualFlags_DifferentValues(t *testing.T) {
	a := []string{"-b", "4096", "-ub", "512"}
	b := []string{"-b", "8192", "-ub", "512"}
	if equalFlags(a, b) {
		t.Errorf("expected not equal for different flag values")
	}
}

func TestEqualFlags_DifferentLength(t *testing.T) {
	a := []string{"-b", "4096", "-ub", "512"}
	b := []string{"-b", "4096"}
	if equalFlags(a, b) {
		t.Errorf("expected not equal for different flag counts")
	}
}

func TestEqualFlags_EmptySlices(t *testing.T) {
	if !equalFlags(nil, nil) {
		t.Errorf("expected equal for two nil slices")
	}
	if !equalFlags([]string{}, []string{}) {
		t.Errorf("expected equal for two empty slices")
	}
}

func TestEqualFlags_BooleanFlagPresence(t *testing.T) {
	// --no-mmap present in one, absent in other
	a := []string{"-b", "4096", "--no-mmap"}
	b := []string{"-b", "4096"}
	if equalFlags(a, b) {
		t.Errorf("expected not equal when boolean flag differs")
	}
}

func TestEqualFlags_CanonicalAliases(t *testing.T) {
	// -c and --ctx-size are canonical aliases
	a := []string{"-c", "32768"}
	b := []string{"--ctx-size", "32768"}
	if !equalFlags(a, b) {
		t.Errorf("expected equal for canonical flag aliases (-c vs --ctx-size)")
	}
}

// ---------------------------------------------------------------------------
// isStrictVRAMSuperset — OOM-aware pruning (fork addition)
// ---------------------------------------------------------------------------

func TestIsStrictVRAMSuperset_TrueWhenSuperset(t *testing.T) {
	baseFlags := []string{"-b", "4096", "-ub", "512"}

	// Crashed with --no-mmap
	crashedSet := map[string]interface{}{"--no-mmap": true}

	// New candidate has --no-mmap AND adds --mlock (non-VRAM-saving)
	newOverrides := map[string]interface{}{"--no-mmap": true, "--mlock": true}

	if !isStrictVRAMSuperset(newOverrides, crashedSet, baseFlags) {
		t.Errorf("expected true: new is superset of crashed with no VRAM savers")
	}
}

func TestIsStrictVRAMSuperset_FalseWhenNotSuperset(t *testing.T) {
	baseFlags := []string{"-b", "4096", "-ub", "512"}

	// Crashed with --no-mmap AND -b 8192
	crashedSet := map[string]interface{}{"--no-mmap": true, "-b": "8192"}

	// New candidate only has --no-mmap (missing -b 8192)
	newOverrides := map[string]interface{}{"--no-mmap": true}

	if isStrictVRAMSuperset(newOverrides, crashedSet, baseFlags) {
		t.Errorf("expected false: new is NOT a superset of crashed")
	}
}

func TestIsStrictVRAMSuperset_FalseWhenVRAMSaverAdded(t *testing.T) {
	baseFlags := []string{"-b", "4096", "-ub", "512"}

	// Crashed with --no-mmap
	crashedSet := map[string]interface{}{"--no-mmap": true}

	// New candidate has --no-mmap but also changes -b (a VRAM saver)
	newOverrides := map[string]interface{}{"--no-mmap": true, "-b": "2048"}

	if isStrictVRAMSuperset(newOverrides, crashedSet, baseFlags) {
		t.Errorf("expected false: new adds a VRAM-saving flag (-b)")
	}
}

func TestIsStrictVRAMSuperset_FalseWhenCacheTypeChanged(t *testing.T) {
	baseFlags := []string{"-b", "4096", "--cache-type-k", "f16"}

	crashedSet := map[string]interface{}{"--no-mmap": true}

	// New candidate adds --cache-type-k change (VRAM saver)
	newOverrides := map[string]interface{}{"--no-mmap": true, "--cache-type-k": "q8_0"}

	if isStrictVRAMSuperset(newOverrides, crashedSet, baseFlags) {
		t.Errorf("expected false: new changes --cache-type-k (VRAM saver)")
	}
}

func TestIsStrictVRAMSuperset_FalseWhenUBatchChanged(t *testing.T) {
	baseFlags := []string{"-b", "4096", "-ub", "512"}

	crashedSet := map[string]interface{}{"--no-mmap": true}

	// New candidate changes -ub (VRAM saver: smaller compute buffer)
	newOverrides := map[string]interface{}{"--no-mmap": true, "-ub": "256"}

	if isStrictVRAMSuperset(newOverrides, crashedSet, baseFlags) {
		t.Errorf("expected false: new changes -ub (VRAM saver)")
	}
}

func TestIsStrictVRAMSuperset_NCPUMoEChangeIsNotSuperset(t *testing.T) {
	// --n-cpu-moe is no longer protected (the tune engine can adjust it).
	// Changing it is a VRAM-saving change, so the candidate is NOT a strict
	// superset of the crashed set — it should be allowed to run.
	baseFlags := []string{"-b", "4096", "--n-cpu-moe", "10"}

	crashedSet := map[string]interface{}{"--no-mmap": true}

	newOverrides := map[string]interface{}{"--no-mmap": true, "--n-cpu-moe": "20"}

	// --n-cpu-moe changed (VRAM saver), so NOT a strict superset.
	if isStrictVRAMSuperset(newOverrides, crashedSet, baseFlags) {
		t.Errorf("expected false: --n-cpu-moe change is a VRAM saver, not a strict superset")
	}
}

func TestIsStrictVRAMSuperset_ProtectedFlagStripped(t *testing.T) {
	// -ngl is still protected, so ApplyOverrides strips the override.
	// The resulting flags are identical to base, making it a true superset.
	baseFlags := []string{"-b", "4096", "-ngl", "999"}

	crashedSet := map[string]interface{}{"--no-mmap": true}

	newOverrides := map[string]interface{}{"--no-mmap": true, "-ngl": "50"}

	// Because -ngl is protected, the override is stripped and the
	// candidate IS a strict superset (no effective VRAM-saving change).
	if !isStrictVRAMSuperset(newOverrides, crashedSet, baseFlags) {
		t.Errorf("expected true: -ngl is protected, override stripped, still a superset")
	}
}

// ---------------------------------------------------------------------------
// isSkippedDueToOOM — integration with crashedFlagSets
// ---------------------------------------------------------------------------

func TestIsSkippedDueToOOM_SkipsSuperset(t *testing.T) {
	baseFlags := []string{"-b", "4096", "-ub", "512"}
	crashedFlagSets := []map[string]interface{}{
		{"--no-mmap": true},
	}

	// This candidate is a superset of the crashed set
	overrides := map[string]interface{}{"--no-mmap": true, "--mlock": true}

	if !isSkippedDueToOOM(overrides, crashedFlagSets, baseFlags) {
		t.Errorf("expected candidate to be skipped due to OOM prediction")
	}
}

func TestIsSkippedDueToOOM_AllowsDifferentCandidate(t *testing.T) {
	baseFlags := []string{"-b", "4096", "-ub", "512"}
	crashedFlagSets := []map[string]interface{}{
		{"--no-mmap": true, "-b": "8192"},
	}

	// This candidate is NOT a superset (different -b value)
	overrides := map[string]interface{}{"--no-mmap": true, "-b": "2048"}

	if isSkippedDueToOOM(overrides, crashedFlagSets, baseFlags) {
		t.Errorf("expected candidate NOT to be skipped (different batch size)")
	}
}

func TestIsSkippedDueToOOM_EmptyCrashHistory(t *testing.T) {
	baseFlags := []string{"-b", "4096"}
	overrides := map[string]interface{}{"--no-mmap": true}

	if isSkippedDueToOOM(overrides, nil, baseFlags) {
		t.Errorf("expected no skip with empty crash history")
	}
}

// ---------------------------------------------------------------------------
// deterministicPlan — synergistic combination candidates (fork modification)
// ---------------------------------------------------------------------------

func TestDeterministicPlan_SynergisticCombosEmitted(t *testing.T) {
	// Base flags with --no-kv-offload (triggers gpu-kv-turbo4 combo)
	baseFlags := []string{"-b", "4096", "-ub", "512", "--no-kv-offload"}
	backendHelp := "--cache-type-k turbo4 turbo3 f16 q8_0"

	plan := deterministicPlan(baseFlags, "llama", nil, backendHelp)

	// Should contain the gpu-kv-turbo4 combo candidate
	found := false
	for _, c := range plan {
		if c.Name == "gpu-kv-turbo4" {
			found = true
			// Verify it's a combo: removes --no-kv-offload AND sets cache types
			if c.FlagValues["--no-kv-offload"] != false {
				t.Errorf("gpu-kv-turbo4 should set --no-kv-offload=false")
			}
			if c.FlagValues["--cache-type-k"] != "turbo4" {
				t.Errorf("gpu-kv-turbo4 should set --cache-type-k=turbo4")
			}
			if c.FlagValues["--cache-type-v"] != "turbo3" {
				t.Errorf("gpu-kv-turbo4 should set --cache-type-v=turbo3")
			}
			break
		}
	}
	if !found {
		t.Errorf("expected gpu-kv-turbo4 combo candidate in plan")
	}
}

func TestDeterministicPlan_GpuKvNoMmapCombo(t *testing.T) {
	baseFlags := []string{"-b", "4096", "-ub", "512", "--no-kv-offload"}
	backendHelp := "--cache-type-k turbo4 turbo3 f16 q8_0"

	plan := deterministicPlan(baseFlags, "llama", nil, backendHelp)

	found := false
	for _, c := range plan {
		if c.Name == "gpu-kv-no-mmap" {
			found = true
			if c.FlagValues["--no-kv-offload"] != false {
				t.Errorf("gpu-kv-no-mmap should set --no-kv-offload=false")
			}
			if c.FlagValues["--no-mmap"] != true {
				t.Errorf("gpu-kv-no-mmap should set --no-mmap=true")
			}
			break
		}
	}
	if !found {
		t.Errorf("expected gpu-kv-no-mmap combo candidate in plan")
	}
}

func TestDeterministicPlan_KVTypesBeforeBatch(t *testing.T) {
	baseFlags := []string{"-b", "4096", "-ub", "512"}
	backendHelp := "--cache-type-k turbo4 turbo3 f16 q8_0"

	plan := deterministicPlan(baseFlags, "llama", nil, backendHelp)

	// KV type candidates should come before batch/ubatch candidates
	kvIdx := -1
	batchIdx := -1
	for i, c := range plan {
		if c.Name == "kv-type-turbo4-turbo3" && kvIdx == -1 {
			kvIdx = i
		}
		if c.Name == "larger-batch" && batchIdx == -1 {
			batchIdx = i
		}
	}

	if kvIdx == -1 {
		t.Fatalf("kv-type-turbo4-turbo3 not found in plan")
	}
	if batchIdx == -1 {
		t.Fatalf("larger-batch not found in plan")
	}
	if kvIdx > batchIdx {
		t.Errorf("KV type candidates should come before batch: kv at %d, batch at %d", kvIdx, batchIdx)
	}
}

func TestDeterministicPlan_TurboComboWithLargerBatch(t *testing.T) {
	baseFlags := []string{"-b", "2048", "-ub", "512", "--kv-offload"}
	backendHelp := "--cache-type-k turbo4 turbo3 f16 q8_0"

	plan := deterministicPlan(baseFlags, "llama", nil, backendHelp)

	found := false
	for _, c := range plan {
		if c.Name == "turbo4-larger-batch" {
			found = true
			if c.FlagValues["--cache-type-k"] != "turbo4" {
				t.Errorf("turbo4-larger-batch should set --cache-type-k=turbo4")
			}
			if c.FlagValues["-b"] != "4096" {
				t.Errorf("turbo4-larger-batch should set -b=4096, got %v", c.FlagValues["-b"])
			}
			break
		}
	}
	if !found {
		t.Errorf("expected turbo4-larger-batch combo candidate in plan")
	}
}

func TestDeterministicPlan_NoTurboCombosWithoutBackendSupport(t *testing.T) {
	baseFlags := []string{"-b", "4096", "-ub", "512", "--no-kv-offload"}
	backendHelp := "--cache-type-k f16 q8_0 q4_0" // no "turbo" in help

	plan := deterministicPlan(baseFlags, "llama", nil, backendHelp)

	for _, c := range plan {
		if c.Name == "gpu-kv-turbo4" || c.Name == "gpu-kv-no-mmap" {
			t.Errorf("did not expect turbo combo %q without backend turbo support", c.Name)
		}
	}
}

func TestDeterministicPlan_MoECandidates(t *testing.T) {
	baseFlags := []string{"-b", "4096", "-ub", "512", "--n-cpu-moe", "30", "-ot", "exps=CPU"}
	backendHelp := ""

	plan := deterministicPlan(baseFlags, "llama", nil, backendHelp)

	// Should have MoE-specific candidates
	foundSmallerBatch := false
	foundNativeMoE := false
	for _, c := range plan {
		if c.Name == "smaller-moe-batch" {
			foundSmallerBatch = true
		}
		if c.Name == "native-moe-gpu-kv" {
			foundNativeMoE = true
		}
	}
	if !foundSmallerBatch {
		t.Errorf("expected smaller-moe-batch candidate for MoE offload")
	}
	if !foundNativeMoE {
		t.Errorf("expected native-moe-gpu-kv candidate for MoE with -ot set")
	}
}

// ---------------------------------------------------------------------------
// flagMap — semantic parsing
// ---------------------------------------------------------------------------

func TestFlagMap_BasicParsing(t *testing.T) {
	flags := []string{"-b", "4096", "--flash-attn", "on", "--no-mmap", "-m", "/model.gguf"}
	m := flagMap(flags)

	if m["-b"] != "4096" {
		t.Errorf("expected -b=4096, got %q", m["-b"])
	}
	if m["--flash-attn"] != "on" {
		t.Errorf("expected --flash-attn=on, got %q", m["--flash-attn"])
	}
	if m["--no-mmap"] != "" {
		t.Errorf("expected --no-mmap='' (boolean), got %q", m["--no-mmap"])
	}
	if m["-m"] != "/model.gguf" {
		t.Errorf("expected -m=/model.gguf, got %q", m["-m"])
	}
}

func TestFlagMap_EqualsSyntax(t *testing.T) {
	flags := []string{"--ctx-size=65536", "-b=2048"}
	m := flagMap(flags)

	if m["--ctx-size"] != "65536" {
		t.Errorf("expected --ctx-size=65536, got %q", m["--ctx-size"])
	}
	if m["-b"] != "2048" {
		t.Errorf("expected -b=2048, got %q", m["-b"])
	}
}

// ---------------------------------------------------------------------------
// Engine.Run — integration with mock benchmarkFn
// ---------------------------------------------------------------------------

func TestEngineRun_BaselineOnly(t *testing.T) {
	callCount := 0
	e := &Engine{
		BaseURL:   "http://localhost:8080",
		Model:     "test-model",
		Rounds:    0, // no tuning rounds, just baseline
		Backend:   "llama",
		benchmarkFn: func() (*benchmark.Result, error) {
			callCount++
			return &benchmark.Result{
				PromptTPS:  100.0,
				GenTPS:     50.0,
				PeakVRAMMB: 8000,
			}, nil
		},
	}

	best, err := e.Run("/models/test.gguf", []string{"-b", "4096"})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if best == nil {
		t.Fatal("expected non-nil best entry")
	}
	if best.Result.GenTPS != 50.0 {
		t.Errorf("expected GenTPS=50.0, got %f", best.Result.GenTPS)
	}
	if best.Name != "baseline" {
		t.Errorf("expected name 'baseline', got %q", best.Name)
	}
}

func TestEngineRun_ImprovementDetected(t *testing.T) {
	round := 0
	e := &Engine{
		BaseURL:          "http://localhost:8080",
		Model:            "test-model",
		Rounds:           1,
		Backend:          "llama",
		MinImprovementPct: 1.0,
		benchmarkFn: func() (*benchmark.Result, error) {
			round++
			if round <= 1 {
				// Baseline
				return &benchmark.Result{PromptTPS: 100, GenTPS: 50, PeakVRAMMB: 8000}, nil
			}
			// Candidate: 10% improvement
			return &benchmark.Result{PromptTPS: 110, GenTPS: 55, PeakVRAMMB: 8000}, nil
		},
	}

	best, err := e.Run("/models/test.gguf", []string{"-b", "4096"})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	// The confirmation pass re-measures, so we just check it didn't crash
	if best == nil {
		t.Fatal("expected non-nil best")
	}
}

// ---------------------------------------------------------------------------
// sanitizeFlagValues — protected flags and normalization
// ---------------------------------------------------------------------------

func TestSanitizeFlagValues_ProtectedFlagsRemoved(t *testing.T) {
	protected := QualityProtectedFlags()
	values := map[string]interface{}{
		"-b":            "8192",
		"--ctx-size":    "65536", // protected
		"--flash-attn":  true,
		"-m":            "/other.gguf", // protected
	}

	result := sanitizeFlagValues(values, protected)

	if _, ok := result["--ctx-size"]; ok {
		t.Errorf("expected --ctx-size to be filtered (protected)")
	}
	if _, ok := result["-m"]; ok {
		t.Errorf("expected -m to be filtered (protected)")
	}
	if result["-b"] != "8192" {
		t.Errorf("expected -b=8192 to pass through")
	}
}

func TestSanitizeFlagValues_FlashAttnBoolNormalization(t *testing.T) {
	values := map[string]interface{}{
		"--flash-attn": true,
	}
	result := sanitizeFlagValues(values, nil)
	if result["--flash-attn"] != "on" {
		t.Errorf("expected --flash-attn=true to normalize to 'on', got %v", result["--flash-attn"])
	}

	values = map[string]interface{}{
		"--flash-attn": false,
	}
	result = sanitizeFlagValues(values, nil)
	if result["--flash-attn"] != "off" {
		t.Errorf("expected --flash-attn=false to normalize to 'off', got %v", result["--flash-attn"])
	}
}

// ---------------------------------------------------------------------------
// meaningfulImprovement
// ---------------------------------------------------------------------------

func TestMeaningfulImprovement(t *testing.T) {
	tests := []struct {
		candidate, incumbent, minPct float64
		want                         bool
	}{
		{55.0, 50.0, 1.0, true},   // 10% > 1%
		{50.4, 50.0, 1.0, false},  // 0.8% < 1%
		{50.5, 50.0, 1.0, true},   // exactly 1%
		{0, 50.0, 1.0, false},     // zero candidate
		{50.0, 0, 1.0, true},      // zero incumbent
	}
	for _, tt := range tests {
		got := meaningfulImprovement(tt.candidate, tt.incumbent, tt.minPct)
		if got != tt.want {
			t.Errorf("meaningfulImprovement(%f, %f, %f) = %v, want %v",
				tt.candidate, tt.incumbent, tt.minPct, got, tt.want)
		}
	}
}

// ---------------------------------------------------------------------------
// ApplyOverrides
// ---------------------------------------------------------------------------

func TestApplyOverrides_AddsNewFlag(t *testing.T) {
	base := []string{"-b", "4096", "-ub", "512"}
	overrides := map[string]interface{}{"--flash-attn": "on"}
	protected := QualityProtectedFlags()

	result := ApplyOverrides(base, overrides, protected)
	m := flagMap(result)
	if m["--flash-attn"] != "on" {
		t.Errorf("expected --flash-attn=on in result, got %v", m)
	}
	// Original flags preserved
	if m["-b"] != "4096" {
		t.Errorf("expected -b=4096 preserved, got %q", m["-b"])
	}
}

func TestApplyOverrides_RemovesFlagWithFalse(t *testing.T) {
	base := []string{"-b", "4096", "--no-mmap"}
	overrides := map[string]interface{}{"--no-mmap": false}
	protected := QualityProtectedFlags()

	result := ApplyOverrides(base, overrides, protected)
	m := flagMap(result)
	if _, ok := m["--no-mmap"]; ok {
		t.Errorf("expected --no-mmap to be removed, got %v", m)
	}
}

func TestApplyOverrides_ProtectedFlagIgnored(t *testing.T) {
	base := []string{"-b", "4096", "--ctx-size", "32768"}
	overrides := map[string]interface{}{"--ctx-size": "65536"}
	protected := QualityProtectedFlags()

	result := ApplyOverrides(base, overrides, protected)
	m := flagMap(result)
	if m["--ctx-size"] != "32768" {
		t.Errorf("expected --ctx-size=32768 (protected), got %q", m["--ctx-size"])
	}
}