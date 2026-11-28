package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

// The draft preflight of a replayed record saved without a dialect names the
// launch backend's device: Vulkan on Vulkan, CUDA as before on CUDA.
func TestDraftPreflightOfLegacyReplayUsesTheLaunchDialect(t *testing.T) {
	for tag, want := range map[string]string{"vulkan": "Vulkan2", "llama": "CUDA2"} {
		vc := &placement.VerifiedConfig{
			StrategyType: placement.SingleGPU, MainGPU: 0, ContextSize: 8192, Parallel: 1,
			Draft: &placement.DraftConfig{Type: placement.DraftModel, Path: "/m/draft.gguf", DraftGPU: 2},
		}
		s := placement.VerifiedToStrategy(vc, placement.Options{BackendTag: tag}, nil)
		args := draftPreflightServerArgs(s)
		got := ""
		for i := 0; i+1 < len(args); i++ {
			if args[i] == "--device" {
				got = args[i+1]
			}
		}
		if got != want {
			t.Fatalf("%s: draft preflight --device = %q, want %q (args %v)", tag, got, want, args)
		}
	}
}
