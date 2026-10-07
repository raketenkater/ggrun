package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

// The --claude-code launch of Qwen3.8-27B (4 slots, 628,736 tokens) failed its
// KV-cache allocation on CUDA0 by 308 MiB and failed closed: context recovery
// accepted only compute-buffer failures, although KV shrinks with context too.
func TestKVCacheOOMDeratesAutomaticContext(t *testing.T) {
	args := []string{"llama-server", "--ctx-size", "628736", "-b", "1024", "-ub", "512",
		"--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--parallel", "4"}
	strategy := placementStrategyForTest{ctx: 628736, parallel: 4}
	outcome := preflightOutcome{Device: 0, AllocMB: 6711, AllocMBMeasured: true, DeficitMB: 308, IsKVCache: true}
	req := &launchRequest{CtxFlag: "fit", ClaudeCode: true}
	target, ok := automaticContextRecoveryTarget(req, strategy.strategy(), args, outcome)
	if !ok {
		t.Fatal("a KV-cache OOM under automatic context must derate context")
	}
	if target >= 628736 || target < 4*claudeSlotMin {
		t.Fatalf("target %d outside (Claude 4-slot floor %d, current 628736)", target, 4*claudeSlotMin)
	}
	// Sized to the deficit, not a blind cut: a 308 MiB shortfall on a 6.7 GB KV
	// row needs only a few percent of context.
	if target < 628736*85/100 {
		t.Fatalf("target %d cut far more than a 308 MiB deficit needs", target)
	}

	// Explicit context stays a constraint.
	explicit := strategy.strategy()
	explicit.ContextAuto = false
	if _, ok := automaticContextRecoveryTarget(&launchRequest{CtxFlag: "628736"}, explicit, args, outcome); ok {
		t.Fatal("recovery lowered an explicit context")
	}
	// A weights allocation is neither compute nor KV: context is not its lever.
	weights := outcome
	weights.IsKVCache = false
	if _, ok := automaticContextRecoveryTarget(req, strategy.strategy(), args, weights); ok {
		t.Fatal("a non-KV, non-compute OOM must not derate context")
	}
}

func TestKVCacheAllocationFailureIsRecognised(t *testing.T) {
	log := "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 6710.78 MiB on device 0: cudaMalloc failed: out of memory\n" +
		"llama_kv_cache_init: failed to allocate buffer for kv cache\n" +
		"llama_init_from_model: llama_kv_cache_init() failed for self-attention cache\n"
	if !kvCacheAllocationFailed(log) {
		t.Fatal("KV-cache allocation failure not recognised")
	}
	if kvCacheAllocationFailed("ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 1024\n") {
		t.Fatal("compute-buffer failure taken for a KV failure")
	}
	if got := allocationOOMOutcome(preflightOutcome{}, &ikAllocationOOMError{Device: 0, AllocMB: 6711, DeficitMB: 308, IsKVCache: true}); !got.IsKVCache {
		t.Fatal("KV flag lost between the probe error and the recovery outcome")
	}
}

type placementStrategyForTest struct{ ctx, parallel int }

func (p placementStrategyForTest) strategy() *placement.Strategy {
	return &placement.Strategy{ContextSize: p.ctx, ContextAuto: true, BatchSize: 1024, UBatchSize: 512, KVType: "q8_0", Parallel: p.parallel}
}
