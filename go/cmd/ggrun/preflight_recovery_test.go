package main

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func preflightRecoveryFixture() (*placement.ModelProfile, *detect.Capabilities, []string, *placement.Strategy) {
	model := &placement.ModelProfile{
		NumLayers:    60,
		LeadingDense: 3,
		ExpertBytes:  int64(57 * 2500 * 1024 * 1024),
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576},
		{Index: 1, VRAMTotalMB: 12288},
	}}
	ot := `blk\.(3|4|5|6)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,blk\.(7|8|9|10)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA1,exps=CPU`
	args := []string{
		"llama-server", "-b", "2048", "-ub", "512",
		"--tensor-split", "0.67,0.33", "--split-mode", "layer",
		"-ot", ot, "--n-cpu-moe", "49",
	}
	strategy := &placement.Strategy{
		BatchSize: 2048, UBatchSize: 512, TensorSplit: []float64{0.67, 0.33},
		SplitMode: "layer", OTString: ot, NCPUMoE: 49,
	}
	return model, caps, args, strategy
}

func TestAllocationOOMOutcomePreservesExactRecoveryEvidence(t *testing.T) {
	got := allocationOOMOutcome(preflightOutcome{Device: -1}, &ikAllocationOOMError{
		Device: 0, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if !got.DoesNotFit || got.Device != 0 || got.AllocMB != 2132 || got.DeficitMB != 74 || !got.IsComputeBuffer {
		t.Fatalf("allocation outcome lost exact evidence: %+v", got)
	}
}

func TestUnchangedComputeReplanMovesExpertBeforeLoweringUBatch(t *testing.T) {
	model, caps, args, unchanged := preflightRecoveryFixture()
	next, entry, method, ok := selectChangedPreflightRecovery(args, unchanged, model, caps, preflightOutcome{
		Device: 0, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if !ok || method != "expert-derate" {
		t.Fatalf("compute recovery = method %q ok=%v", method, ok)
	}
	fingerprint := effectiveMemoryArgsFingerprint(next)
	if entry == nil || entry.NCPUMoE != 50 || !strings.Contains(fingerprint, "n-cpu-moe=50") || !strings.Contains(fingerprint, "ubatch=512") {
		t.Fatalf("compute recovery did not move one expert while preserving ubatch: entry=%+v args=%v", entry, next)
	}
}

func TestComputeRecoveryNeverRaisesDeratedUBatch(t *testing.T) {
	model, caps, args, candidate := preflightRecoveryFixture()
	args = replaceUBatchArg(args, 256)
	candidate.UBatchSize = 512
	candidate.NCPUMoE++
	next, entry, method, ok := selectChangedPreflightRecovery(args, candidate, model, caps, preflightOutcome{
		Device: 0, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if !ok || method != "expert-derate" || entry == nil || entry.NCPUMoE != 50 || !strings.Contains(effectiveMemoryArgsFingerprint(next), "ubatch=256") {
		t.Fatalf("recovery raised a derated ubatch: method=%q entry=%+v ok=%v args=%v", method, entry, ok, next)
	}
}

func TestComputeRecoveryLowersUBatchWhenFailedDeviceHasNoExpert(t *testing.T) {
	model, caps, args, _ := preflightRecoveryFixture()
	for i := range args {
		if args[i] == "-ot" && i+1 < len(args) {
			args[i+1] = strings.ReplaceAll(args[i+1], "CUDA1", "CUDA0")
			break
		}
	}
	next, entry, method, ok := selectChangedPreflightRecovery(args, nil, model, caps, preflightOutcome{
		Device: 1, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if !ok || method != "ubatch-derate" || entry == nil || entry.UBatchSize != 256 {
		t.Fatalf("compute fallback = method %q entry=%+v ok=%v args=%v", method, entry, ok, next)
	}
}

// GLM-5.3-Flash produced a 17,495 MiB CUDA0 graph allocation at ubatch 256
// against a 12,666 MiB deficit. Walking 256 -> 128 -> 64 spent two four-minute
// model loads even though the first measured allocation already proved that a
// one-rung reduction could not reclaim enough memory. Recovery must use the
// measured ratio to jump to the first rung with a plausible fit.
func TestComputeRecoveryJumpsUBatchByMeasuredDeficit(t *testing.T) {
	model, caps, args, _ := preflightRecoveryFixture()
	args = replaceUBatchArg(args, 256)
	for i := range args {
		if args[i] == "-ot" && i+1 < len(args) {
			// CUDA0 owns no movable expert, so graph-size recovery is the next
			// applicable lever.
			args[i+1] = strings.ReplaceAll(args[i+1], "CUDA0", "CUDA1")
			break
		}
	}
	next, entry, method, ok := selectChangedPreflightRecovery(args, nil, model, caps, preflightOutcome{
		Device: 0, AllocMB: 17495, AllocMBMeasured: true, DeficitMB: 12666, IsComputeBuffer: true,
	})
	if !ok || method != "ubatch-derate" || entry == nil || entry.UBatchSize != 64 {
		t.Fatalf("deficit-sized compute recovery = method %q entry=%+v ok=%v args=%v", method, entry, ok, next)
	}
}

// At ubatch 64 the GLM preflight repeatedly changed only context by 1,024
// tokens. Each retry reclaimed 4-5 MiB while CUDA2 was still short by ~2.5 GiB,
// consuming the entire retry budget. The failed device owned one routed expert;
// moving that expert is relevant and large enough, while the context nudge is
// neither.
func TestGLMContextNudgeCannotBeatFailedDeviceExpertRelief(t *testing.T) {
	model := &placement.ModelProfile{
		NumLayers: 43, LeadingDense: 1,
		ExpertBytes: int64(42 * 3000 * 1024 * 1024),
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576},
		{Index: 1, VRAMTotalMB: 24576},
		{Index: 2, VRAMTotalMB: 12288},
	}}
	ot := `blk\.(1)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA2,exps=CPU`
	args := []string{
		"llama-server", "--ctx-size", "1048576", "-b", "2048", "-ub", "64",
		"--tensor-split", "0.29,0.61,0.10", "--split-mode", "layer",
		"-ot", ot, "--n-cpu-moe", "41", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
	}
	candidate := &placement.Strategy{
		ContextSize: 1047552, BatchSize: 2048, UBatchSize: 64,
		TensorSplit: []float64{0.29, 0.61, 0.10}, SplitMode: "layer",
		OTString: ot, NCPUMoE: 41,
	}
	next, entry, method, ok := selectChangedPreflightRecovery(args, candidate, model, caps, preflightOutcome{
		Device: 2, AllocMB: 4632, AllocMBMeasured: true, DeficitMB: 2524, IsComputeBuffer: true,
	})
	if !ok || method != "expert-derate" || entry == nil || entry.NCPUMoE != 42 {
		t.Fatalf("GLM recovery = method %q entry=%+v ok=%v args=%v", method, entry, ok, next)
	}
	fingerprint := effectiveMemoryArgsFingerprint(next)
	if !strings.Contains(fingerprint, "ctx=1048576") {
		t.Fatalf("irrelevant context nudge won over failed-device relief: %v", next)
	}
}

func TestAutomaticContextFallbackTargetIsDeficitSizedAndExplicitContextIsImmutable(t *testing.T) {
	model := &placement.ModelProfile{NumLayers: 32, HeadCountKV: 8, KeyLength: 128, ValueLength: 128}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}}}
	args := []string{
		"llama-server", "--ctx-size", "262144", "-b", "2048", "-ub", "64",
		"--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--parallel", "1",
	}
	strategy := &placement.Strategy{
		ContextSize: 262144, ContextAuto: true, BatchSize: 2048, UBatchSize: 64,
		KVType: "q8_0", Parallel: 1,
	}
	outcome := preflightOutcome{
		Device: 0, AllocMB: 4096, AllocMBMeasured: true, DeficitMB: 2048, IsComputeBuffer: true,
	}
	autoReq := &launchRequest{CtxFlag: "fit"}
	target, ok := automaticContextRecoveryTarget(autoReq, strategy, args, outcome)
	if !ok || target >= 262144 || target < 32768 {
		t.Fatalf("automatic context fallback target=%d ok=%v", target, ok)
	}
	if target%1024 != 0 {
		t.Fatalf("context fallback target was not rounded to 1024 tokens: %d", target)
	}
	if _, _, _, patched := applyMemoryRecoverySelection(autoReq, strategy, args, nil, model, caps, outcome); patched {
		t.Fatal("memory selection patched context without a full placement recompute")
	}

	explicitStrategy := &placement.Strategy{
		ContextSize: 262144, ContextAuto: false, BatchSize: 2048, UBatchSize: 64,
		KVType: "q8_0", Parallel: 1,
	}
	if _, ok := automaticContextRecoveryTarget(&launchRequest{CtxFlag: "262144"}, explicitStrategy, args, outcome); ok {
		t.Fatal("memory recovery silently lowered an explicit context")
	}
	explicitCandidate := &placement.Strategy{
		ContextSize: 65536, BatchSize: 2048, UBatchSize: 64, KVType: "q8_0", Parallel: 1,
	}
	if _, _, _, ok := applyMemoryRecoverySelection(
		&launchRequest{CtxFlag: "262144"}, explicitStrategy, args, explicitCandidate, model, caps, outcome,
	); ok {
		t.Fatal("a re-plan candidate overrode an explicit context")
	}
}

func TestFailedDeviceReliefNormalizesTensorSplitShares(t *testing.T) {
	current := []string{"llama-server", "--tensor-split", "2,8", "--split-mode", "layer"}
	equivalent := []string{"llama-server", "--tensor-split", "1,4", "--split-mode", "layer"}
	if candidateRelievesFailedDevice(current, equivalent, 0) {
		t.Fatal("an equivalent normalized tensor split was mistaken for failed-device relief")
	}
	relieved := []string{"llama-server", "--tensor-split", "1,9", "--split-mode", "layer"}
	if !candidateRelievesFailedDevice(current, relieved, 0) {
		t.Fatal("a real failed-device tensor-share reduction was not recognized")
	}
}

func TestUnchangedWeightReplanMovesExpertLayer(t *testing.T) {
	model, caps, args, unchanged := preflightRecoveryFixture()
	next, entry, method, ok := selectChangedPreflightRecovery(args, unchanged, model, caps, preflightOutcome{
		Device: 1, AllocMB: 2500, DeficitMB: 74,
	})
	if !ok || method != "expert-derate" {
		t.Fatalf("weight recovery = method %q ok=%v", method, ok)
	}
	if entry == nil || entry.NCPUMoE != 50 || !strings.Contains(effectiveMemoryArgsFingerprint(next), "n-cpu-moe=50") {
		t.Fatalf("weight recovery did not move one expert layer: entry=%+v args=%v", entry, next)
	}
}

func TestComputeRecoveryMovesExpertBeforeChangedPackerLowersUBatch(t *testing.T) {
	model, caps, args, candidate := preflightRecoveryFixture()
	candidate.UBatchSize = 256
	next, entry, method, ok := selectChangedPreflightRecovery(args, candidate, model, caps, preflightOutcome{
		Device: 0, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if !ok || method != "expert-derate" || entry == nil || entry.NCPUMoE != 50 {
		t.Fatalf("layer-first compute recovery = method %q entry=%+v ok=%v", method, entry, ok)
	}
	if !strings.Contains(effectiveMemoryArgsFingerprint(next), "ubatch=512") {
		t.Fatalf("compute recovery lowered ubatch before moving an expert: %v", next)
	}
}

func TestSharedMemoryRecoveryAppliesLayerFirstStrategyAndArgsTogether(t *testing.T) {
	model, caps, args, current := preflightRecoveryFixture()
	candidate := *current
	candidate.UBatchSize = 256

	next, nextArgs, method, ok := applyMemoryRecoverySelection(
		nil, current, args, &candidate, model, caps,
		preflightOutcome{Device: 0, AllocMB: 2132, DeficitMB: 1, IsComputeBuffer: true},
	)
	if !ok || method != "expert-derate" || next != current {
		t.Fatalf("shared recovery = strategy %p current %p method %q ok=%v", next, current, method, ok)
	}
	fingerprint := effectiveMemoryArgsFingerprint(nextArgs)
	if next.NCPUMoE != 50 || next.UBatchSize != 512 ||
		!strings.Contains(fingerprint, "n-cpu-moe=50") || !strings.Contains(fingerprint, "ubatch=512") {
		t.Fatalf("strategy/argv recovery drifted: strategy=%+v args=%v", next, nextArgs)
	}
}

func TestSharedMemoryRecoveryKeepsCompleteCandidateArgs(t *testing.T) {
	model, caps, args, current := preflightRecoveryFixture()
	candidate := *current
	candidate.NCPUMoE = current.NCPUMoE + 1
	complete := patchPlacementArgs(args, &candidate)
	complete = append(complete, "-cram", "7777", "--ctx-checkpoints", "8")

	next, nextArgs, method, ok := applyMemoryRecoverySelection(
		nil, current, args, &candidate, model, caps,
		preflightOutcome{Device: 0, AllocMB: 2132, DeficitMB: 1, IsComputeBuffer: true},
		complete,
	)
	if !ok || method != "replanned" || next != &candidate {
		t.Fatalf("complete recovery candidate = strategy %p want %p method=%q ok=%v", next, &candidate, method, ok)
	}
	if argIntValue(nextArgs, "-cram") != 7777 || argIntValue(nextArgs, "--ctx-checkpoints") != 8 {
		t.Fatalf("complete recovery argv lost context-derived flags: %v", nextArgs)
	}
}

func TestEffectiveDuplicateOverrideForbidsIdenticalRetry(t *testing.T) {
	model, caps, args, candidate := preflightRecoveryFixture()
	args = replaceUBatchArg(args, 64)
	for i := range args {
		if args[i] == "-ot" && i+1 < len(args) {
			args[i+1] = strings.ReplaceAll(args[i+1], "CUDA0", "CUDA1")
			break
		}
	}
	args = append(args, "-ub", "64") // later user value remains authoritative
	candidate.UBatchSize = 256
	_, _, _, ok := selectChangedPreflightRecovery(args, candidate, model, caps, preflightOutcome{
		Device: 0, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if ok {
		t.Fatal("recovery must reject a syntactically changed but effectively identical retry")
	}
}

func TestNoGenericWeightLeverFailsClosed(t *testing.T) {
	model := &placement.ModelProfile{NumLayers: 32}
	args := []string{"llama-server", "-ub", "512"}
	_, _, _, ok := selectChangedPreflightRecovery(args, nil, model, &detect.Capabilities{}, preflightOutcome{
		Device: 0, AllocMB: 1024, DeficitMB: 1,
	})
	if ok {
		t.Fatal("weight OOM without a placement lever must fail closed")
	}
}

// A ggrun-generated --swa-full is withdrawn before any weight is shed. It is
// the largest reclaimable block on a memory failure (measured 2026-08-03:
// 6196 MiB of KV on CUDA0 versus 871 MiB without) and it was buying no prefix
// reuse on this model. A --swa-full the user typed is an instruction and stays.
func TestGeneratedSWAFullIsWithdrawnBeforeSheddingWeights(t *testing.T) {
	baseArgs := []string{
		"llama-server", "-m", "model.gguf", "-ub", "512", "--swa-full", "--n-cpu-moe", "39",
	}
	model := &placement.ModelProfile{ModelArch: "deepseek4", IsMoE: true, NumLayers: 43}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24564}}}
	outcome := preflightOutcome{Device: 0, AllocMB: 9307, AllocMBMeasured: true, DeficitMB: 1200, IsComputeBuffer: true}

	// Config-sourced: ExtraArgs carries it, OriginalArgs does not.
	generated := &launchRequest{ExtraArgs: []string{"--swa-full"}}
	_, nextArgs, method, ok := applyMemoryRecoverySelection(
		generated, &placement.Strategy{UBatchSize: 512, NCPUMoE: 39}, baseArgs, nil, model, caps, outcome)
	if !ok || method != "swa-full-withdrawn" {
		t.Fatalf("generated --swa-full was not withdrawn first: method=%q ok=%v", method, ok)
	}
	if hasArg(nextArgs, "--swa-full") {
		t.Fatalf("--swa-full survived withdrawal: %v", nextArgs)
	}
	if hasArg(generated.ExtraArgs, "--swa-full") {
		t.Fatalf("--swa-full must also leave the request so rebuilds cannot reintroduce it: %v", generated.ExtraArgs)
	}

	// Typed on the command line: never withdrawn, fall through to the ladder.
	typed := &launchRequest{ExtraArgs: []string{"--swa-full"}, OriginalArgs: []string{"--swa-full"}}
	_, _, method, _ = applyMemoryRecoverySelection(
		typed, &placement.Strategy{UBatchSize: 512, NCPUMoE: 39}, baseArgs, nil, model, caps, outcome)
	if method == "swa-full-withdrawn" {
		t.Fatal("an explicitly typed --swa-full must not be withdrawn")
	}
}

func TestOracleDeficitRetainsCompleteAutomaticContextReplan(t *testing.T) {
	model := fitTestModel(131072, 6000)
	caps := fitTestCaps(12000)
	req := &launchRequest{CtxFlag: "fit", Parallel: 1, ParallelSet: true, KVQuality: "mid", KVPlacement: "gpu", RAMLimitPercent: 95}
	be := fitTestBackend()
	cfg := &config.Config{CacheDir: t.TempDir()}
	opts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	current, err := placement.Compute(caps, model, opts)
	if err != nil {
		t.Fatal(err)
	}
	args := buildLaunchServerArgs(req, cfg, be, caps, model, current)
	outcome := preflightOutcome{Device: 0, DeficitMB: 1000, DoesNotFit: true, Evidence: memoryPlanEvidence{Level: memoryEvidenceOraclePlanned}}
	next, nextArgs, method, err := recoverPreflightOOM(req, cfg, model, be, caps, caps, nil, current, args, map[int]int{}, outcome)
	if err != nil {
		t.Fatalf("oracle-backed full context replan discarded: %v", err)
	}
	if method != "context-replanned" || next.ContextSize >= current.ContextSize || next.ContextSize < 32768 {
		t.Fatalf("recovery=%s ctx %d -> %d", method, current.ContextSize, next.ContextSize)
	}
	if next.Parallel != 1 || next.KVType != current.KVType || !next.ContextAuto {
		t.Fatalf("recovery changed workload policy: %+v", next)
	}
	rebuilt := buildLaunchServerArgs(req, cfg, be, caps, model, next)
	if launchArgsIdentity(nextArgs) != launchArgsIdentity(rebuilt) {
		t.Fatal("recovery returned a partial argv overlay")
	}
}

func TestOracleContextRecoveryPreservesExplicitContext(t *testing.T) {
	model := fitTestModel(131072, 6000)
	caps := fitTestCaps(12000)
	req := &launchRequest{CtxFlag: "65536", Parallel: 1, ParallelSet: true, KVQuality: "mid", KVPlacement: "gpu", RAMLimitPercent: 95}
	be := fitTestBackend()
	cfg := &config.Config{CacheDir: t.TempDir()}
	current, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil {
		t.Fatal(err)
	}
	args := buildLaunchServerArgs(req, cfg, be, caps, model, current)
	outcome := preflightOutcome{Device: 0, DeficitMB: 1000, DoesNotFit: true, Evidence: memoryPlanEvidence{Level: memoryEvidenceOraclePlanned}}
	next, _, _, err := recoverPreflightOOM(req, cfg, model, be, caps, caps, nil, current, args, map[int]int{}, outcome)
	if err == nil && next.ContextSize != 65536 {
		t.Fatalf("explicit context changed to %d", next.ContextSize)
	}
}
