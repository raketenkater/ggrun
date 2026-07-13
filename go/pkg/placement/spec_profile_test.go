package placement

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestSpecPerformanceProfileRoundTripAndExactScope(t *testing.T) {
	dir := t.TempDir()
	companion := filepath.Join(dir, "mtp.gguf")
	if err := os.WriteFile(companion, []byte("GGUF-test"), 0o600); err != nil {
		t.Fatal(err)
	}
	target := &ModelProfile{
		Path: "model.gguf", SizeBytes: 1234, TotalSizeMB: 1, ModelArch: "qwen35",
		NumLayers: 33, EmbeddingLength: 2560, VocabSize: 248320,
		TokenizerHash: "abc", NextNPredictLayers: 1, ContextSize: 262144,
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "RTX", VRAMTotalMB: 24576}}}
	opts := Options{CacheDir: dir, ContextSize: 262144, Parallel: 4, BackendIdentity: "llama-commit-a", SamplingProfile: "default"}
	scope := NewSpecProfileScope(target, caps, opts, "mtp", companion)
	profile := SpecPerformanceProfile{
		Scope: scope, LaunchIdentity: "test-launch", DraftMax: 2, BaselineTPS: 100, SpeculativeTPS: 112, ImprovementPct: 12, WallImprovementPct: 9,
		PromptCases: 9, RepeatedRounds: 3, MaxPromptTokens: 60000,
		CorrectnessPassed: true, StabilityPassed: true, ParallelLoadPassed: true, Complete: true,
	}
	path, err := SaveSpecPerformanceProfile(dir, profile)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadSpecPerformanceProfile(dir, scope)
	if err != nil {
		t.Fatal(err)
	}
	if ok, reason := loaded.AutoEligible(); !ok {
		t.Fatalf("eligible profile rejected: %s", reason)
	}
	if path != SpecProfilePath(dir, scope) || loaded.ScopeKey != scope.Key() {
		t.Fatalf("profile key/path mismatch: path=%s key=%s", path, loaded.ScopeKey)
	}

	changed := scope
	changed.BackendIdentity = "llama-commit-b"
	if _, err := LoadSpecPerformanceProfile(dir, changed); err == nil {
		t.Fatal("backend change must invalidate the performance profile")
	}
}

func TestSpecPerformanceProfileEligibilityGates(t *testing.T) {
	base := SpecPerformanceProfile{
		Scope: SpecProfileScope{ContextSize: 1048576, Parallel: 4}, LaunchIdentity: "test-launch", DraftMax: 2,
		BaselineTPS: 100, SpeculativeTPS: 103, ImprovementPct: 3, WallImprovementPct: 3,
		PromptCases: 9, RepeatedRounds: 2, MaxPromptTokens: 60000,
		CorrectnessPassed: true, StabilityPassed: true, ParallelLoadPassed: true, Complete: true,
	}
	if ok, reason := base.AutoEligible(); !ok {
		t.Fatalf("valid profile rejected: %s", reason)
	}
	short := base
	short.MaxPromptTokens = 4096
	if ok, _ := short.AutoEligible(); ok {
		t.Fatal("1M profile without a 60k request must be rejected")
	}
	noisy := base
	noisy.ImprovementPct = 1
	if ok, _ := noisy.AutoEligible(); ok {
		t.Fatal("gain below the noise floor must be rejected")
	}
	serialOnly := base
	serialOnly.ParallelLoadPassed = false
	if ok, _ := serialOnly.AutoEligible(); ok {
		t.Fatal("parallel profile without load validation must be rejected")
	}
}

func TestSpecPerformanceProfileLongContextGateUsesPerSlotCapacity(t *testing.T) {
	profile := SpecPerformanceProfile{
		Scope: SpecProfileScope{ContextSize: 131072, Parallel: 4}, LaunchIdentity: "test-launch", DraftMax: 2,
		BaselineTPS: 100, SpeculativeTPS: 103, ImprovementPct: 3, WallImprovementPct: 3,
		PromptCases: 9, RepeatedRounds: 2, MaxPromptTokens: 32000,
		CorrectnessPassed: true, StabilityPassed: true, ParallelLoadPassed: true, Complete: true,
	}
	if ok, reason := profile.AutoEligible(); !ok {
		t.Fatalf("unexpected per-slot long-context rejection: %s", reason)
	}

	profile.Scope.ContextSize = 262144 // 65k per slot, so a 60k proof is required.
	if ok, _ := profile.AutoEligible(); ok {
		t.Fatal("missing 60k proof accepted when a slot has enough capacity")
	}
}

func TestSpecProfileScopeTracksGPUSetAndEveryShard(t *testing.T) {
	dir := t.TempDir()
	first := filepath.Join(dir, "model-00001-of-00002.gguf")
	second := filepath.Join(dir, "model-00002-of-00002.gguf")
	if err := os.WriteFile(first, []byte("one"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(second, []byte("two"), 0o600); err != nil {
		t.Fatal(err)
	}
	model := &ModelProfile{Path: first, SizeBytes: 6, ModelArch: "test"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "A", VRAMTotalMB: 100}, {Index: 1, Name: "B", VRAMTotalMB: 200}}}
	a := NewSpecProfileScope(model, caps, Options{GPUs: []int{1, 0}}, "mtp", "")
	b := NewSpecProfileScope(model, caps, Options{GPUs: []int{0, 1}}, "mtp", "")
	if a.GPUSet != "0,1" || a.Key() != b.Key() {
		t.Fatalf("GPU set is not canonical: %#v / %#v", a, b)
	}
	if err := os.WriteFile(second, []byte("changed-size"), 0o600); err != nil {
		t.Fatal(err)
	}
	c := NewSpecProfileScope(model, caps, Options{GPUs: []int{0, 1}}, "mtp", "")
	if c.TargetIdentity == a.TargetIdentity || c.Key() == a.Key() {
		t.Fatal("changing a non-primary shard did not invalidate the profile")
	}
}
