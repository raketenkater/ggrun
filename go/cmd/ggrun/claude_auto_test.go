package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestClaudeReviewerGPUCandidatesPreservesLargestGPU(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24564, BandwidthMBps: 15754},
		{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 985},
		{Index: 2, VRAMTotalMB: 12282, BandwidthMBps: 3938},
	}}
	got := claudeReviewerGPUCandidates(caps, &launchRequest{})
	want := []int{1, 2, 0}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func TestClaudeReviewerGPUCandidatesKeepSparsePhysicalSelection(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0}, {Index: 1}, {Index: 2}}}
	got := claudeReviewerGPUCandidates(caps, &launchRequest{GPUsFlag: "2,1,2,9"})
	want := []int{2, 1}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("got %v, want physical selection %v", got, want)
	}
}

func TestClaudeReviewerArgsUsesIsolatedDeviceAsLocalMain(t *testing.T) {
	args := claudeReviewerArgs("server", "reviewer.gguf", 1234, 0, "--reasoning ARG --cache-type-k TYPE --cache-type-v TYPE")
	for _, want := range []string{"--device", "CUDA0", "-mg", "0", "--reasoning", "off", "--ctx-size", "65536", "--cache-type-k", "q8_0", "--cache-type-v"} {
		if !hasArg(args, want) {
			t.Fatalf("missing %q in %v", want, args)
		}
	}
	for _, flag := range []string{"--cache-type-k", "--cache-type-v"} {
		if !hasArgValue(args, flag, "q8_0") {
			t.Fatalf("expected %s q8_0 in %v", flag, args)
		}
	}
}

func TestClaudeReviewerArgsKeepsOlderBackendCompatibility(t *testing.T) {
	args := claudeReviewerArgs("server", "reviewer.gguf", 1234, -1, "--reasoning ARG")
	for _, unsupported := range []string{"--cache-type-k", "--cache-type-v"} {
		if hasArg(args, unsupported) {
			t.Fatalf("unexpected unsupported %q in %v", unsupported, args)
		}
	}
}

func TestClaudeReviewerCPUFallbackHidesAccelerators(t *testing.T) {
	got := claudeReviewerCPUEnv()
	for _, want := range []string{"CUDA_VISIBLE_DEVICES=-1", "HIP_VISIBLE_DEVICES=-1", "ROCR_VISIBLE_DEVICES=-1"} {
		if !hasArg(got, want) {
			t.Fatalf("missing %q in %v", want, got)
		}
	}
}

func TestClaudeAutoReviewerNeededDefaultsOnForAuto(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "")
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	if !claudeAutoReviewerNeeded(nil) {
		t.Fatal("default local Auto launch must start its reviewer")
	}
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "acceptEdits")
	if claudeAutoReviewerNeeded(nil) {
		t.Fatal("non-Auto permission mode should not spend memory on a reviewer")
	}
}
