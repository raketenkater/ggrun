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
	next, nextArgs, method, err := recoverPreflightOOM(req, cfg, model, be, caps, caps, nil, current, args, map[int]int{}, outcome, nil)
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
	next, _, _, err := recoverPreflightOOM(req, cfg, model, be, caps, caps, nil, current, args, map[int]int{}, outcome, nil)
	if err == nil && next.ContextSize != 65536 {
		t.Fatalf("explicit context changed to %d", next.ContextSize)
	}
}

// A context this launch disproved must stay disproved. The argv identity
// ledger cannot enforce that: the recompute emits a different argv, so nothing
// matches and the loop climbs back into the rejected range.
func TestRejectedAutomaticContextCapsTheBackendMeasuredRecompute(t *testing.T) {
	model := fitTestModel(131072, 6000)
	caps := fitTestCaps(12000)
	req := &launchRequest{CtxFlag: "fit", Parallel: 1, ParallelSet: true, KVQuality: "mid", KVPlacement: "gpu", RAMLimitPercent: 95}
	be := fitTestBackend()
	cfg := &config.Config{CacheDir: t.TempDir()}
	current, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil {
		t.Fatal(err)
	}
	if !current.ContextAuto || current.ContextSize <= 0 {
		t.Fatalf("fixture did not produce an automatic context: %+v", current)
	}

	recovery := newLaunchMemoryRecovery()
	recovery.rejectContext(current, 0)

	// This is the backend-measured recompute: it re-enters Compute from the
	// original automatic request, which is why it reproduces the rejection.
	replan := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	replan.SkipPlacementCache = true
	unbounded, err := placement.Compute(caps, model, replan)
	if err != nil {
		t.Fatal(err)
	}
	if unbounded.ContextSize != current.ContextSize {
		t.Fatalf("fixture no longer reproduces the climb: %d -> %d", current.ContextSize, unbounded.ContextSize)
	}

	// Exactly what the launch loop applies before it recomputes.
	replan = boundByProvenLimits(replan, recovery)
	if replan.AutoContextMax <= 0 || replan.AutoContextMax >= current.ContextSize {
		t.Fatalf("ceiling %d does not exclude the rejected context %d", replan.AutoContextMax, current.ContextSize)
	}
	bounded, err := placement.Compute(caps, model, replan)
	if err != nil {
		t.Fatal(err)
	}
	if bounded.ContextSize >= current.ContextSize {
		t.Fatalf("recompute proposed %d at or above the rejected %d", bounded.ContextSize, current.ContextSize)
	}
	if !bounded.ContextAuto || bounded.Parallel != current.Parallel || bounded.KVType != current.KVType {
		t.Fatalf("the ceiling changed workload policy: %+v", bounded)
	}
}

func TestContextCeilingOnlyRatchetsDown(t *testing.T) {
	recovery := newLaunchMemoryRecovery()
	if recovery.automaticContextCeiling() != 0 {
		t.Fatal("a launch with no rejection must not cap its own context")
	}
	// An explicit context is a user constraint, not a coordinate to move.
	recovery.rejectContext(&placement.Strategy{ContextSize: 65536}, 0)
	if recovery.automaticContextCeiling() != 0 {
		t.Fatal("an explicit context was recorded as an automatic rejection")
	}
	// The GLM-5.3-Flash sequence observed on 2026-09-14.
	for _, ctx := range []int{592896, 385024, 589824} {
		recovery.rejectContext(&placement.Strategy{ContextSize: ctx, ContextAuto: true}, 0)
	}
	if got := recovery.automaticContextCeiling(); got != 385023 {
		t.Fatalf("ceiling %d; a later larger rejection must not raise it", got)
	}
}

// The backend-measured re-plan refines placement from measured buffers. It may
// not spend an accepted plan's proof on a larger context: on GLM-5.3-Flash it
// discarded an accepted 555,008-token plan twice and failed against the same
// limit, exhausting the re-plan budget before any weights loaded.
func TestAcceptedContextBoundsTheMeasuredReplan(t *testing.T) {
	recovery := newLaunchMemoryRecovery()
	recovery.rejectContext(&placement.Strategy{ContextSize: 670720, ContextAuto: true}, 0)
	recovery.rejectContext(&placement.Strategy{ContextSize: 563200, ContextAuto: true}, 0)
	recovery.acceptContext(&placement.Strategy{ContextSize: 555008, ContextAuto: true})
	if got := recovery.automaticContextCeiling(); got != 555008 {
		t.Fatalf("ceiling %d; an accepted plan must bound the re-plan at itself", got)
	}
	// The observed climb: 562,176 sits under the rejected 563,200 but above the
	// accepted 555,008, and it is exactly what broke the launch.
	opts := boundByProvenLimits(placement.Options{}, recovery)
	if opts.AutoContextMax >= 562176 {
		t.Fatalf("AutoContextMax %d still admits the plan that broke the launch", opts.AutoContextMax)
	}
	// Recovery falling further must ratchet down, never back up.
	recovery.acceptContext(&placement.Strategy{ContextSize: 500736, ContextAuto: true})
	if got := recovery.automaticContextCeiling(); got != 500736 {
		t.Fatalf("ceiling %d after a lower accepted plan", got)
	}
	// An explicit context is a user constraint, not a coordinate to record.
	recovery.acceptContext(&placement.Strategy{ContextSize: 1024})
	if got := recovery.automaticContextCeiling(); got != 500736 {
		t.Fatalf("an explicit context moved the ceiling to %d", got)
	}
}

// recoverPreflightOOM already refuses to recompute a derated ubatch back up.
// The measured re-plan did not, so a GLM-5.3-Flash plan accepted at ubatch 128
// came back at 256 and overshot all three devices at the same context.
func TestMeasuredReplanKeepsADeratedUBatch(t *testing.T) {
	recovery := newLaunchMemoryRecovery()
	recovery.acceptContext(&placement.Strategy{ContextSize: 500736, ContextAuto: true, UBatchSize: 128})
	if got := boundByProvenLimits(placement.Options{UBatchSize: 512}, recovery); got.UBatchSize != 128 {
		t.Fatalf("measured re-plan recomputed ubatch back to %d", got.UBatchSize)
	}
	// An automatic request carries no ubatch of its own; the proven one applies.
	if got := boundByProvenLimits(placement.Options{}, recovery); got.UBatchSize != 128 {
		t.Fatalf("automatic ubatch request ignored the proven derating: %d", got.UBatchSize)
	}
	// Like the context ceiling, it only ratchets down.
	recovery.acceptContext(&placement.Strategy{ContextSize: 400384, ContextAuto: true, UBatchSize: 256})
	if got := boundByProvenLimits(placement.Options{}, recovery); got.UBatchSize != 128 {
		t.Fatalf("a later larger ubatch raised the pin to %d", got.UBatchSize)
	}
	if got := recovery.automaticContextCeiling(); got != 400384 {
		t.Fatalf("context ceiling %d did not follow the lower accepted plan", got)
	}
}

// Recovery derives its own candidate from the original automatic request, so it
// needs the same ledger the measured re-plan uses. At --parallel 2 on
// Qwen3.8-27B this candidate returned to 524,288 tokens after recovery had
// already derated to 301,056, and the launch spent its whole re-plan budget
// without ever loading weights.
func TestRecoveryCandidateRespectsTheProvenContextCeiling(t *testing.T) {
	model := fitTestModel(131072, 6000)
	caps := fitTestCaps(12000)
	req := &launchRequest{CtxFlag: "fit", Parallel: 1, ParallelSet: true, KVQuality: "mid", KVPlacement: "gpu", RAMLimitPercent: 95}
	be := fitTestBackend()
	cfg := &config.Config{CacheDir: t.TempDir()}
	current, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil {
		t.Fatal(err)
	}
	args := buildLaunchServerArgs(req, cfg, be, caps, model, current)
	outcome := preflightOutcome{Device: 0, DeficitMB: 1000, DoesNotFit: true,
		Evidence: memoryPlanEvidence{Level: memoryEvidenceOraclePlanned}}

	// This launch has already disproved everything at or above 40,960 tokens.
	const rejected = 40960
	if current.ContextSize <= rejected {
		t.Skipf("fixture context %d is already below the rejection", current.ContextSize)
	}
	recovery := newLaunchMemoryRecovery()
	recovery.rejectContext(&placement.Strategy{ContextSize: rejected, ContextAuto: true}, 0)

	next, _, _, err := recoverPreflightOOM(req, cfg, model, be, caps, caps, nil,
		current, args, map[int]int{}, outcome, recovery)
	if err != nil {
		t.Fatalf("recovery failed closed against its own ceiling: %v", err)
	}
	if next == nil {
		t.Fatal("recovery returned no candidate")
	}
	if next.ContextSize >= rejected {
		t.Fatalf("recovery proposed %d at or above the rejected %d", next.ContextSize, rejected)
	}
	// A nil ledger must stay legal for callers that keep no per-launch state.
	if _, _, _, err := recoverPreflightOOM(req, cfg, model, be, caps, caps, nil,
		current, args, map[int]int{}, outcome, nil); err != nil {
		t.Fatalf("recovery without a ledger regressed: %v", err)
	}
}

// The ceiling excludes the rejected context itself and nothing more. A
// deficit-sized step was tried here and reverted: it made GLM-5.3-Flash leap
// 662,528 -> 236,544 tokens and the plans computed at that depth were refused
// by the recovery guards, failing a launch that converges without it.
func TestContextCeilingExcludesTheRejectedContext(t *testing.T) {
	strategy := &placement.Strategy{ContextSize: 816128, ContextAuto: true, KVType: "q8_0"}
	recovery := newLaunchMemoryRecovery()
	recovery.rejectContext(strategy, 76800)
	if got := recovery.automaticContextCeiling(); got != strategy.ContextSize-1 {
		t.Fatalf("ceiling %d, want one token below %d regardless of the reclaim", got, strategy.ContextSize)
	}
	// The ceiling only ratchets down.
	lower := &placement.Strategy{ContextSize: 500736, ContextAuto: true, KVType: "q8_0"}
	recovery.rejectContext(lower, 0)
	if got := recovery.automaticContextCeiling(); got != lower.ContextSize-1 {
		t.Fatalf("ceiling %d did not follow the lower rejection", got)
	}
	recovery.rejectContext(strategy, 999999)
	if got := recovery.automaticContextCeiling(); got != lower.ContextSize-1 {
		t.Fatalf("a higher rejection widened the ceiling to %d", got)
	}
}

// Unknown geometry yields no step rather than a guessed one.
func TestContextReclaimTokensRefusesToGuess(t *testing.T) {
	model := &placement.ModelProfile{NumLayers: 32, HeadCountKV: 8, KeyLength: 128, ValueLength: 128}
	strategy := &placement.Strategy{ContextSize: 262144, ContextAuto: true, KVType: "q8_0"}
	args := []string{"llama-server", "--cache-type-k", "q8_0"}
	if got := contextReclaimTokens(model, strategy, args, 0, 0); got != 0 {
		t.Fatalf("no deficit gave %d tokens", got)
	}
	if got := contextReclaimTokens(nil, strategy, args, 1000, 0); got != 0 {
		t.Fatalf("no model gave %d tokens", got)
	}
	if got := contextReclaimTokens(model, &placement.Strategy{ContextAuto: true}, args, 1000, 0); got != 0 {
		t.Fatalf("no context gave %d tokens", got)
	}
}

// The re-plan budget exists to stop churn. A descent that keeps shrinking the
// measured shortfall is not churn, and a 1,024-token nudge that trims a few MiB
// is not progress. Both have to be distinguishable.
func TestDeficitProgressSeparatesConvergenceFromNudging(t *testing.T) {
	// The GLM-5.3-Flash --parallel 2 descent: every step is progress.
	for _, step := range [][2]int{{1443, 323}, {323, 136}, {136, 30}} {
		if !deficitProgress(step[0], step[1]) {
			t.Fatalf("%d -> %d MiB is convergence and was charged as churn", step[0], step[1])
		}
	}
	// The nudge pathology: a 2,524 MiB shortfall trimmed by 5 MiB a round.
	if deficitProgress(2524, 2519) {
		t.Fatal("a 5 MiB trim against a 2524 MiB deficit was credited as progress")
	}
	// Growing, equal, and unknown deficits are never progress.
	for _, step := range [][2]int{{100, 100}, {100, 140}, {0, 50}, {50, 0}} {
		if deficitProgress(step[0], step[1]) {
			t.Fatalf("%d -> %d MiB was credited as progress", step[0], step[1])
		}
	}
}

// The oracle path accepted any smaller context, so a single 1,024-token granule
// worth ~7 MiB satisfied a ~100 MiB deficit and returned before ubatch or
// expert relief was ever considered. Qwen3.8-Flash-Next never launched on main:
// 101 -> 94 -> 87 MiB with ubatch stuck at 256.
func TestOracleContextDropMustCoverTheDeficit(t *testing.T) {
	model := &placement.ModelProfile{NumLayers: 48, HeadCountKV: 8, KeyLength: 128, ValueLength: 128}
	args := []string{
		"llama-server", "--ctx-size", "261120", "-b", "2048", "-ub", "256",
		"--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--parallel", "1",
	}
	current := &placement.Strategy{
		ContextSize: 261120, ContextAuto: true, UBatchSize: 256, KVType: "q8_0", Parallel: 1,
	}
	outcome := preflightOutcome{Device: 0, DeficitMB: 101, DoesNotFit: true,
		Evidence: memoryPlanEvidence{Level: memoryEvidenceOraclePlanned}}

	needed := contextReclaimTokens(model, current, args, outcome.DeficitMB, 0)
	if needed <= 1024 {
		t.Skipf("fixture KV geometry makes one granule sufficient (needed=%d)", needed)
	}

	// One granule down: the observed nudge. Must be refused.
	nudge := *current
	nudge.ContextSize = current.ContextSize - 1024
	if oracleContextDropCoversDeficit(model, current, args, &nudge, outcome) {
		t.Fatalf("a 1024-token nudge was accepted against a %d MiB deficit", outcome.DeficitMB)
	}

	// A drop sized to the deficit is a real recovery and must be accepted.
	sized := *current
	sized.ContextSize = current.ContextSize - needed
	if !oracleContextDropCoversDeficit(model, current, args, &sized, outcome) {
		t.Fatalf("a deficit-sized drop of %d tokens was refused", needed)
	}

	// Unknown geometry must not veto a candidate the rest of the guard accepted.
	noKV := &placement.Strategy{ContextSize: 261120, ContextAuto: true}
	if !oracleContextDropCoversDeficit(model, noKV, []string{"llama-server"}, &nudge, outcome) {
		t.Fatal("unknown KV geometry rejected a candidate instead of deferring to preflight")
	}
}

// A deficit lands on one device, so the exchange rate is that device's KV
// share. Sizing against the aggregate credits the failing GPU with savings that
// land on its neighbours and produces a step several times too small.
func TestContextReclaimUsesTheFailingDeviceKVShare(t *testing.T) {
	model := &placement.ModelProfile{NumLayers: 48, HeadCountKV: 8, KeyLength: 128, ValueLength: 128}
	args := []string{"llama-server", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"}
	// CUDA1 owns the majority of layers; CUDA2 owns a small tail.
	strategy := &placement.Strategy{
		ContextSize: 262144, ContextAuto: true, KVType: "q8_0", KVPlacement: "gpu",
		TensorSplit: []float64{0.12, 0.76, 0.12},
	}
	// A deficit small enough that both devices can cover it above the floor, so
	// the comparison is about apportionment rather than about bowing out.
	big := contextReclaimTokens(model, strategy, args, 200, 1)
	small := contextReclaimTokens(model, strategy, args, 200, 2)
	if big <= 0 || small <= 0 {
		t.Fatalf("no reclaim computed: majority=%d minority=%d", big, small)
	}
	// The same deficit on a device owning fewer layers needs a larger context cut.
	if small <= big {
		t.Fatalf("minority-owner device asked for %d tokens, majority %d; share ignored", small, big)
	}
}

// Freeing host KV does not relieve a GPU. Crediting it against a device deficit
// is the aggregate error in a different direction.
func TestContextReclaimRefusesToCreditHostKVAgainstADeviceDeficit(t *testing.T) {
	model := &placement.ModelProfile{NumLayers: 48, HeadCountKV: 8, KeyLength: 128, ValueLength: 128}
	args := []string{"llama-server", "--cache-type-k", "q8_0"}
	hostKV := &placement.Strategy{
		ContextSize: 262144, ContextAuto: true, KVType: "q8_0", KVPlacement: "cpu",
		TensorSplit: []float64{0.5, 0.5},
	}
	if got := contextReclaimTokens(model, hostKV, args, 2000, 0); got != 0 {
		t.Fatalf("host-resident KV was credited with %d tokens of GPU relief", got)
	}
	gpuKV := *hostKV
	gpuKV.KVPlacement = "gpu"
	if got := contextReclaimTokens(model, &gpuKV, args, 2000, 0); got <= 0 {
		t.Fatalf("GPU-resident KV produced no reclaim: %d", got)
	}
}

// An explicit context is a user constraint, not a coordinate recovery may move.
// The ledger must record nothing for it however the deficit is measured.
func TestExplicitContextIsNeverRecordedByRecovery(t *testing.T) {
	recovery := newLaunchMemoryRecovery()
	explicit := &placement.Strategy{ContextSize: 131072, KVType: "q8_0", KVPlacement: "gpu"}
	recovery.rejectContext(explicit, 4096)
	if got := recovery.automaticContextCeiling(); got != 0 {
		t.Fatalf("explicit context produced ceiling %d", got)
	}
	recovery.acceptContext(explicit)
	if got := recovery.automaticContextCeiling(); got != 0 {
		t.Fatalf("explicit context accepted into the ceiling: %d", got)
	}
	// Its ubatch is still a compute lever and is legitimately recorded.
	explicit.UBatchSize = 128
	recovery.acceptContext(explicit)
	if got := boundByProvenLimits(placement.Options{UBatchSize: 512}, recovery); got.UBatchSize != 128 {
		t.Fatalf("proven ubatch %d not preserved for an explicit-context launch", got.UBatchSize)
	}
}

// Competing deficits on different devices are each judged against their own
// device's KV share, which is what decides whether a proposed drop covers the
// shortfall. The ceiling itself stays one token below the rejected context.
func TestCompetingDeviceDeficitsAreJudgedPerDevice(t *testing.T) {
	model := &placement.ModelProfile{NumLayers: 48, HeadCountKV: 8, KeyLength: 128, ValueLength: 128}
	args := []string{"llama-server", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"}
	strategy := &placement.Strategy{
		ContextSize: 262144, ContextAuto: true, KVType: "q8_0", KVPlacement: "gpu",
		TensorSplit: []float64{0.12, 0.76, 0.12},
	}
	majority := contextReclaimTokens(model, strategy, args, 200, 1)
	minority := contextReclaimTokens(model, strategy, args, 200, 2)
	if majority <= 0 || minority <= majority {
		t.Fatalf("per-device judgement collapsed: majority=%d minority=%d", majority, minority)
	}
	recovery := newLaunchMemoryRecovery()
	recovery.rejectContext(strategy, majority)
	recovery.rejectContext(strategy, minority)
	if got := recovery.automaticContextCeiling(); got != strategy.ContextSize-1 {
		t.Fatalf("ceiling %d; recovery must not force a deficit-sized step", got)
	}
}

// Unknown geometry must leave recovery a legal fallback rather than veto it.
func TestUnknownGeometryLeavesRecoveryAFallback(t *testing.T) {
	model := &placement.ModelProfile{NumLayers: 48, HeadCountKV: 8, KeyLength: 128, ValueLength: 128}
	noSplit := &placement.Strategy{ContextSize: 262144, ContextAuto: true}
	if got := contextReclaimTokens(model, noSplit, []string{"llama-server"}, 1500, 0); got != 0 {
		t.Fatalf("unknown KV type produced a guessed rate of %d tokens", got)
	}
	recovery := newLaunchMemoryRecovery()
	recovery.rejectContext(noSplit, 0)
	if got := recovery.automaticContextCeiling(); got != noSplit.ContextSize-1 {
		t.Fatalf("unknown geometry gave ceiling %d, want one token below %d", got, noSplit.ContextSize)
	}
	outcome := preflightOutcome{Device: 0, DeficitMB: 1500, DoesNotFit: true,
		Evidence: memoryPlanEvidence{Level: memoryEvidenceOraclePlanned}}
	candidate := *noSplit
	candidate.ContextSize = noSplit.ContextSize - 1024
	if !oracleContextDropCoversDeficit(model, noSplit, []string{"llama-server"}, &candidate, outcome) {
		t.Fatal("unknown geometry vetoed the only remaining candidate")
	}
}

// Context is only a lever when it can cover the deficit and still leave a usable
// window. Demanding more context than exists yields no computable plan, so
// recovery returns nothing and the launch fails closed. Observed on
// GLM-5.3-Flash: a 2,432 MiB CUDA0 deficit against that device's KV share asks
// for a cut deeper than the whole context.
func TestContextBowsOutWhenItCannotCoverTheDeficit(t *testing.T) {
	model := &placement.ModelProfile{NumLayers: 48, HeadCountKV: 8, KeyLength: 128, ValueLength: 128}
	args := []string{"llama-server", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"}
	strategy := &placement.Strategy{
		ContextSize: 65536, ContextAuto: true, KVType: "q8_0", KVPlacement: "gpu",
		TensorSplit: []float64{0.24, 0.59, 0.17},
	}
	// A deficit far beyond what this device's KV share can free.
	if got := contextReclaimTokens(model, strategy, args, 100000, 0); got != 0 {
		t.Fatalf("context claimed %d tokens of relief it cannot deliver", got)
	}
	// A deficit it can cover while leaving a usable window is still reported.
	small := contextReclaimTokens(model, strategy, args, 20, 0)
	if small <= 0 || strategy.ContextSize-small < contextRecoveryFloorTokens {
		t.Fatalf("a coverable deficit produced %d tokens", small)
	}
}
