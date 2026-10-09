package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestBackendHasNoWindowedKV(t *testing.T) {
	gemma := &placement.ModelProfile{ModelArch: "gemma4", SlidingWindow: 1024}
	ik := &backendInfo{Tag: "ik_llama", IsIK: true, Help: "--split-mode-graph"}
	for _, tc := range []struct {
		name  string
		be    *backendInfo
		model *placement.ModelProfile
		want  bool
	}{
		{"ik windowed model", ik, gemma, true},
		{"mainline", &backendInfo{Tag: "llama", Help: "--swa-full"}, gemma, false},
		{"ik build with --swa-full", &backendInfo{Tag: "ik_llama", IsIK: true, Help: "--swa-full"}, gemma, false},
		{"no window", ik, &placement.ModelProfile{ModelArch: "qwen3"}, false},
		{"deepseek4", ik, &placement.ModelProfile{ModelArch: "deepseek4", SlidingWindow: 128}, false},
		{"mla", ik, &placement.ModelProfile{ModelArch: "x", SlidingWindow: 128, KVLoraRank: 512}, false},
	} {
		if got := backendHasNoWindowedKV(tc.be, tc.model); got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
}

// Re-resolving to another backend clears the fact.
func TestBackendFeatureCompatibilitySetsWindowedKVFact(t *testing.T) {
	model := &placement.ModelProfile{ModelArch: "gemma4", SlidingWindow: 1024}
	applyBackendFeatureCompatibility(&launchRequest{}, model, &backendInfo{Tag: "ik_llama", IsIK: true})
	if !model.FullContextWindowedKV {
		t.Fatal("ik_llama without --swa-full must price windowed layers at full context")
	}
	applyBackendFeatureCompatibility(&launchRequest{}, model, &backendInfo{Tag: "llama", Help: "--swa-full"})
	if model.FullContextWindowedKV {
		t.Fatal("a windowed backend must price the window")
	}
}
