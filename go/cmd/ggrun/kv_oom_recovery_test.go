package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
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

// The derated context must be re-planned around the failed device's deficit.
// A free re-plan rebalanced layers onto the device that had just run short.
func TestKVContextDerateKeepsTheFailedDevicesDeficit(t *testing.T) {
	cfg := &config.Config{CacheDir: t.TempDir()}
	model := &placement.ModelProfile{Path: "dense.gguf", Basename: "dense", TotalSizeMB: 18 * 1024, SizeBytes: 18 << 30,
		NumLayers: 64, HeadCountKV: 4, KeyLength: 128, ValueLength: 128}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 12282}, {Index: 1, VRAMTotalMB: 24564}},
		RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 131072}, CPU: detect.CPUInfo{Cores: 16},
	}
	be := &backendInfo{Tag: "ik_llama", Identity: "ik"}
	req := &launchRequest{CtxFlag: "fit", Parallel: 4, ParallelSet: true}
	current, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil || current == nil || len(current.TensorSplit) != 2 {
		t.Skipf("fixture did not produce a two-GPU split: %v", err)
	}
	current.ContextAuto = true
	args := buildLaunchServerArgs(req, cfg, be, caps, model, current)
	outcome := preflightOutcome{Device: 1, AllocMB: 6000, AllocMBMeasured: true, DeficitMB: 700, IsKVCache: true}

	free, _, err := recomputeAutomaticContextRecovery(req, cfg, model, be, caps, current, args, outcome, 0, nil)
	if err != nil || free == nil {
		t.Skipf("no context target for this fixture: %v", err)
	}
	held, _, err := recomputeAutomaticContextRecovery(req, cfg, model, be, caps, current, args, outcome, 0, map[int]int{1: 700})
	if err != nil || held == nil {
		t.Fatalf("penalised context re-plan failed: %v", err)
	}
	if held.ContextSize != free.ContextSize {
		t.Fatalf("both re-plans must use the same deficit-sized context: %d vs %d", held.ContextSize, free.ContextSize)
	}
	t.Logf("ctx %d: free split %v, with deficit %v", free.ContextSize, free.TensorSplit, held.TensorSplit)
	if held.TensorSplit[1] >= free.TensorSplit[1] {
		t.Fatalf("re-plan gave the short device more: split %v vs %v", held.TensorSplit, free.TensorSplit)
	}
}

// A small compute-buffer shortfall must not cost most of the context when the
// same cut frees the device's KV share.
func TestSmallComputeShortfallCutsContextBySize(t *testing.T) {
	args := []string{"llama-server", "--ctx-size", "518144", "-ub", "64", "--parallel", "4",
		"--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--tensor-split", "0.27,0.58,0.15"}
	current := &placement.Strategy{ContextSize: 518144, ContextAuto: true, UBatchSize: 64, KVType: "q8_0", Parallel: 4}
	outcome := preflightOutcome{Device: 0, AllocMB: 79, AllocMBMeasured: true, DeficitMB: 29, IsComputeBuffer: true}
	req := &launchRequest{CtxFlag: "fit", ClaudeCode: true}
	floor, ok := automaticContextRecoveryTargetWithKV(req, current, args, outcome, 0)
	if !ok || floor != 4*claudeSlotMin {
		t.Fatalf("without the KV share the old sizing falls to the floor: %d %v", floor, ok)
	}
	sized, ok := automaticContextRecoveryTargetWithKV(req, current, args, outcome, 5000)
	if !ok || sized < 518144*95/100 || sized >= 518144 {
		t.Fatalf("a 29 MiB shortfall with a 5 GB KV share cut context to %d", sized)
	}
	model := &placement.ModelProfile{NumLayers: 64, HeadCountKV: 4, KeyLength: 128, ValueLength: 128}
	if kv := deviceKVShareMB(model, args, 0); kv <= 0 {
		t.Fatalf("device 0 holds 27%% of the KV: estimated %d MiB", kv)
	}
	if kv := deviceKVShareMB(model, append(args, "--no-kv-offload"), 0); kv != 0 {
		t.Fatalf("KV on the CPU is not freed on the GPU: %d", kv)
	}
}

// An oracle shortfall that is neither a compute nor a KV allocation, on a
// device holding KV under automatic context, is met by a small context cut.
func TestKVBackedOracleShortfallDeratesContext(t *testing.T) {
	args := []string{"llama-server", "--ctx-size", "369664", "-ub", "64", "--parallel", "1",
		"--cache-type-k", "q8_0", "--tensor-split", "0.30,0.55,0.15"}
	current := &placement.Strategy{ContextSize: 369664, ContextAuto: true, UBatchSize: 64, KVType: "q8_0", Parallel: 1}
	outcome := preflightOutcome{Device: 0, AllocMB: 151, DeficitMB: 151}
	req := &launchRequest{CtxFlag: "fit"}
	target, ok := automaticContextRecoveryTargetWithKV(req, current, args, outcome, 5929)
	if !ok || target >= 369664 || target < 369664*90/100 {
		t.Fatalf("151 MiB short with 5,929 MiB of KV on the device: target %d ok=%v", target, ok)
	}
	if _, ok := automaticContextRecoveryTargetWithKV(req, current, args, outcome, 0); ok {
		t.Fatal("an untyped shortfall on a device without KV is not a context problem")
	}
	explicit := *current
	explicit.ContextAuto = false
	if _, ok := automaticContextRecoveryTargetWithKV(&launchRequest{CtxFlag: "369664"}, &explicit, args, outcome, 5929); ok {
		t.Fatal("an explicit context stays a constraint")
	}
}
