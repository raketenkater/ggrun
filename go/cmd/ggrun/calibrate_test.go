package main

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/benchmark"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func calibrateTestSetup(sizeMB int) (*launchRequest, *config.Config, *placement.ModelProfile, *backendInfo, *detect.Capabilities) {
	cfg := config.Defaults()
	cfg.CacheDir = ""
	model := &placement.ModelProfile{
		Path: "model.gguf", Basename: "model", IsMoE: true,
		TotalSizeMB: sizeMB, NumLayers: 60, NumExperts: 128,
	}
	be := &backendInfo{Tag: "ik_llama", Identity: "ik-build-4641", Help: "--reasoning ARG"}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 32000},
			{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 8000},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU: detect.CPUInfo{Cores: 16},
	}
	req := &launchRequest{Port: 8081, Calibrate: calibrateAuto}
	return req, cfg, model, be, caps
}

func TestCalibrationAutoOwnsBoundedStandardLaunchSearch(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(60 * 1024) // 60 GB MoE
	strategy := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	got := calibrationPlan(req, cfg, model, be, caps, strategy)
	if len(got) < 2 || len(got) > calibrationAutoMaxCandidates {
		t.Fatalf("automatic standard launch search is not bounded/useful: %d candidates", len(got))
	}
}

func TestCalibrationPlanForcedOnIgnoresSizeGate(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(60 * 1024)
	req.Calibrate = calibrateOn
	strategy := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	if got := calibrationPlan(req, cfg, model, be, caps, strategy); len(got) < 2 {
		t.Fatalf("forced calibration must run on a big MoE, got %d candidates", len(got))
	}
}

func TestCalibrationBudgetsAreFinite(t *testing.T) {
	auto := calibrationBudgetFor(calibrateAuto)
	forced := calibrationBudgetFor(calibrateOn)
	if auto.MaxCandidates != 4 || auto.MaxFailures != 3 || auto.MaxElapsed != 20*time.Minute {
		t.Fatalf("unexpected automatic budget: %#v", auto)
	}
	if forced.MaxCandidates < auto.MaxCandidates || forced.MaxFailures < auto.MaxFailures || forced.MaxElapsed <= auto.MaxElapsed {
		t.Fatalf("forced budget must remain finite but larger: auto=%#v forced=%#v", auto, forced)
	}
}

func TestAutomaticAdmissionPlanKeepsRankedFallbacks(t *testing.T) {
	base := &placement.Strategy{BatchSize: 128, UBatchSize: 128}
	first := &placement.Strategy{BatchSize: 2048, UBatchSize: 2048}
	exact := &placement.Strategy{BatchSize: 1024, UBatchSize: 1024}
	last := &placement.Strategy{BatchSize: 512, UBatchSize: 512}
	candidates := []placement.CalibrationCandidate{
		{Name: "default", Strategy: base},
		{Name: "ubatch-2048", Strategy: first, Estimate: placement.CandidateEstimate{AgentCost: 0.5}},
		{Name: "ubatch-1024", Strategy: exact, Estimate: placement.CandidateEstimate{AgentCost: 0.51}},
		{Name: "ubatch-512", Strategy: last, Estimate: placement.CandidateEstimate{AgentCost: 0.6}},
	}
	got := selectAutomaticCalibrationAdmissionPlan(candidates, func(strategy *placement.Strategy) bool {
		return strategy == exact
	}, 4)
	want := []string{"default", "ubatch-1024", "ubatch-2048", "ubatch-512"}
	if len(got) != len(want) {
		t.Fatalf("admission plan length=%d, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].Name != want[i] {
			t.Fatalf("admission plan[%d]=%q, want %q: %+v", i, got[i].Name, want[i], got)
		}
	}
}

func TestAutomaticCalibrationSelectsOneMeasuredFinalist(t *testing.T) {
	base := &placement.Strategy{BatchSize: 128, UBatchSize: 128, Parallel: 2}
	unmeasured := &placement.Strategy{BatchSize: 256, UBatchSize: 256, Parallel: 2}
	measuredSameGraph := &placement.Strategy{BatchSize: 256, UBatchSize: 128, Parallel: 2}
	candidates := []placement.CalibrationCandidate{
		{Name: "default", Strategy: base},
		{Name: "batch-256-ubatch-256", Strategy: unmeasured},
		{Name: "batch-256-ubatch-128", Strategy: measuredSameGraph},
	}
	got := selectAutomaticCalibrationFinalist(candidates, func(strategy *placement.Strategy) bool {
		return strategy == measuredSameGraph
	})
	if len(got) != 2 || got[0].Name != "default" || got[1].Name != "batch-256-ubatch-128" {
		t.Fatalf("auto plan did not retain exactly the measured finalist: %+v", got)
	}
	got = selectAutomaticCalibrationFinalist(candidates, nil)
	if len(got) != 2 || got[1].Name != "batch-256-ubatch-256" {
		t.Fatalf("no-evidence fallback did not retain the closest calculated finalist: %+v", got)
	}
}

func TestAutomaticCalibrationUsesRankedCostNotCandidateName(t *testing.T) {
	base := &placement.Strategy{Type: placement.MoEOffload, BatchSize: 128, UBatchSize: 64}
	giant := &placement.Strategy{Type: placement.MoEOffload, BatchSize: 8192, UBatchSize: 8192}
	owner := &placement.Strategy{Type: placement.MoEOffload, BatchSize: 128, UBatchSize: 64, MainGPU: 1}
	got := selectAutomaticCalibrationFinalist([]placement.CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: placement.CandidateEstimate{Feasible: true, AgentCost: 1.0}},
		{Name: "batch-8192-ubatch-8192", Strategy: giant, Estimate: placement.CandidateEstimate{Feasible: true, AgentCost: 0.4}},
		{Name: "moe-owner-1", Strategy: owner, Estimate: placement.CandidateEstimate{Feasible: true, AgentCost: 0.9}},
	}, func(strategy *placement.Strategy) bool { return strategy == giant })
	if len(got) != 2 || got[1].Name != "batch-8192-ubatch-8192" {
		t.Fatalf("auto plan ignored ranked cost/exact evidence because of a candidate name: %+v", got)
	}
}

func TestCalibrationCandidateFilterKeepsBaselineAndDropsRejectedArgv(t *testing.T) {
	base := &placement.Strategy{BatchSize: 128}
	rejected := &placement.Strategy{BatchSize: 256}
	usable := &placement.Strategy{BatchSize: 512}
	got := filterCalibrationCandidates([]placement.CalibrationCandidate{
		{Name: "default", Strategy: base},
		{Name: "rejected", Strategy: rejected},
		{Name: "usable", Strategy: usable},
	}, func(candidate *placement.Strategy) bool { return candidate == rejected })
	if len(got) != 2 || got[0].Strategy != base || got[1].Strategy != usable {
		t.Fatalf("filtered candidates=%+v", got)
	}
}

func TestTelemetryDirectedFinalistCannotReclassifyTightLaunch(t *testing.T) {
	strategy := &placement.Strategy{
		Type: placement.MoEOffload, Residency: placement.ResidencyTight,
		ResourceLedger: &placement.ResourceLedger{Exact: true, Fits: true},
	}
	signal := placement.DeviceBalanceSignal{Observed: true, Imbalanced: true, BusyGPU: 0, IdleGPU: 1, BusySM: 98, IdleSM: 1}
	if candidate, ok := telemetryDirectedCalibrationFinalist(nil, nil, nil, nil, nil, strategy, signal, nil); ok {
		t.Fatalf("tight launch escaped its proven topology boundary: %+v", candidate)
	}
	if strategy.Residency != placement.ResidencyTight {
		t.Fatalf("telemetry mutated residency to %q", strategy.Residency)
	}
}

func TestAutomaticCalibrationKeepsNonResidentLastResortStable(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(60 * 1024)
	base := &placement.Strategy{
		Type: placement.DenseCPUOffload, MMapRequired: true,
		ContextSize: 32768, Parallel: 1, BatchSize: 512, UBatchSize: 128,
	}
	alternate := *base
	alternate.BatchSize = 1024
	got := automaticCalibrationFinalistPlan(req, cfg, model, be, caps, []placement.CalibrationCandidate{
		{Name: "default", Strategy: base},
		{Name: "batch-1024-ubatch-128", Strategy: &alternate},
	})
	if len(got) != 1 || got[0].Name != "default" || got[0].Strategy.Residency != placement.ResidencyNonResident {
		t.Fatalf("standard launch challenged a non-resident last-resort plan: %+v", got)
	}
}

func TestCalibrationPlanKeepsPowerOfTwoParallelCurveInsideBoundedBudget(t *testing.T) {
	base := &placement.Strategy{Parallel: 3}
	candidates := []placement.CalibrationCandidate{{Name: "default", Strategy: base}}
	for i := 0; i < 12; i++ {
		candidates = append(candidates, placement.CalibrationCandidate{
			Name:     fmt.Sprintf("batch-%d-ubatch-64", 128+i*64),
			Strategy: &placement.Strategy{Parallel: 3, BatchSize: 128 + i*64, UBatchSize: 64},
		})
	}
	for _, parallel := range []int{3, 8, 4, 2, 1} {
		candidates = append(candidates, placement.CalibrationCandidate{
			Name: fmt.Sprintf("parallel-%d", parallel), Strategy: &placement.Strategy{Parallel: parallel},
		})
	}

	got := prioritizeParallelCalibrationCurve(candidates)
	if len(got) != len(candidates) || got[0].Name != "default" {
		t.Fatalf("parallel diversification lost candidates or baseline: %+v", got)
	}
	want := []string{"parallel-1", "parallel-2", "parallel-4", "parallel-8"}
	for i, name := range want {
		if got[i+1].Name != name {
			t.Fatalf("parallel curve position %d = %q, want %q; plan=%+v", i, got[i+1].Name, name, got)
		}
	}
	// The forced nine-candidate budget now contains baseline + the full p1/p2/p4
	// curve even when batch search generated many coordinates first.
	for _, name := range want[:3] {
		found := false
		for _, candidate := range got[:calibrationForcedMaxCandidates] {
			if candidate.Name == name {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("bounded plan omitted %s: %+v", name, got[:calibrationForcedMaxCandidates])
		}
	}
}

func TestAutomaticAdmissionFallbackIncludesParallelFamily(t *testing.T) {
	strategy := func(parallel int) *placement.Strategy { return &placement.Strategy{Parallel: parallel} }
	candidates := []placement.CalibrationCandidate{
		{Name: "default", Strategy: strategy(1)},
		{Name: "ubatch-2048", Strategy: strategy(1)},
		{Name: "batch-2048-ubatch-1024", Strategy: strategy(1)},
		{Name: "parallel-2", Strategy: strategy(2)},
		{Name: "parallel-4", Strategy: strategy(4)},
	}
	got := selectAutomaticCalibrationAdmissionPlan(candidates, nil, calibrationAutoMaxCandidates)
	foundParallel := false
	for _, candidate := range got[2:] {
		if strings.HasPrefix(candidate.Name, "parallel-") {
			foundParallel = true
			break
		}
	}
	if !foundParallel {
		t.Fatalf("automatic admission fallbacks were all from one coordinate family: %+v", got)
	}
}

func TestAutomaticWorkloadCandidateSetCapsSlotExperimentsAtDemand(t *testing.T) {
	strategy := func(parallel int) *placement.Strategy {
		return &placement.Strategy{Parallel: parallel}
	}
	req := &launchRequest{ClaudeCode: true, ClaudeProfile: claudeProfileParallel, Parallel: 1}
	candidates := []placement.CalibrationCandidate{
		{Name: "default", Strategy: strategy(4)},
		{Name: "parallel-3", Strategy: strategy(3)},
		{Name: "parallel-2", Strategy: strategy(2)},
		{Name: "parallel-1", Strategy: strategy(1)},
		{Name: "batch-256-ubatch-128", Strategy: strategy(4)},
	}
	got := automaticWorkloadCandidateSet(req, candidates)
	want := []string{"default", "parallel-2", "parallel-1", "batch-256-ubatch-128"}
	if len(got) != len(want) {
		t.Fatalf("candidate count=%d, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].Name != want[i] {
			t.Fatalf("candidate[%d]=%q, want %q: %+v", i, got[i].Name, want[i], got)
		}
	}
}

func TestDefaultWonDecisionRetainsCompleteAgentEvidence(t *testing.T) {
	result := &benchmark.Result{
		Parallel: 2, GenTPS: 40, PromptTPS: 800, MixedGenTPS: 20,
		GenTokens: 128, PromptTokens: 4000, MixedGenTokens: 64, GenTimeS: 10,
		AgentSamples: 2, AgentTurnTimeS: 10, AgentTurnMaxS: 11,
		AgentScenarioTimeS: 30, AgentScenarioMaxS: 31, AgentPromptBytes: 4096,
		AgentCachedTokens: 1800, AgentNewPromptTokens: 80,
	}
	decision := newCalibrationDecision("scope", &placement.ModelProfile{Path: "/models/test.gguf"}, result,
		calibrationMeasurement{Name: "default", Result: result, Score: 1})
	if decision == nil || decision.Winner != "default" || decision.ModelBasename != "test.gguf" ||
		decision.DefaultTurnTimeS != decision.WinnerTurnTimeS {
		t.Fatalf("default-won evidence is incomplete: %+v", decision)
	}
	if automaticCalibrationEvidenceValid(decision) {
		t.Fatal("a default result without a measured challenger became automatic evidence")
	}
	annotateOptimizationDecision(decision, []placement.CalibrationCandidate{
		{Name: "default", Strategy: &placement.Strategy{}},
		{Name: "ubatch-512", Strategy: &placement.Strategy{}},
	}, []calibrationMeasurement{
		{Name: "default", Result: result, Score: 1},
		{Name: "ubatch-512", Result: result, Score: 1},
	})
	if !automaticCalibrationEvidenceValid(decision) {
		t.Fatalf("measured default-won comparison was rejected: %+v", decision)
	}
}

func TestCalibrationScoreUsesColdIngestPlusCacheBackedTurn(t *testing.T) {
	baseline := &benchmark.Result{
		Parallel: 2, GenTPS: 40, PromptTPS: 800, MixedGenTPS: 20,
		GenTokens: 128, PromptTokens: 4000, MixedGenTokens: 64, GenTimeS: 10,
		AgentSamples: 2, AgentTurnTimeS: 10, AgentTurnMaxS: 11,
		AgentScenarioTimeS: 30, AgentScenarioMaxS: 31, AgentPromptBytes: 4096,
		AgentCachedTokens: 1800, AgentNewPromptTokens: 80,
	}
	fastPrefillButSlowAppend := &benchmark.Result{
		Parallel: 2, GenTPS: 60, PromptTPS: 1200, MixedGenTPS: 30,
		GenTokens: 128, PromptTokens: 4000, MixedGenTokens: 64, GenTimeS: 8,
		AgentSamples: 2, AgentTurnTimeS: 12, AgentTurnMaxS: 13,
		AgentScenarioTimeS: 22, AgentScenarioMaxS: 23, AgentPromptBytes: 4096,
		AgentCachedTokens: 1800, AgentNewPromptTokens: 80,
	}
	if got, want := calibrationScore(fastPrefillButSlowAppend, baseline), 30.0/22.0; math.Abs(got-want) > 1e-9 {
		t.Fatalf("agent score=%v, want %v", got, want)
	}
	missingReuse := *fastPrefillButSlowAppend
	missingReuse.AgentCachedTokens = 0
	if validCalibrationResult(&missingReuse) {
		t.Fatal("parallel calibration accepted a result without cache-reuse evidence")
	}
	mismatchedPrompt := *fastPrefillButSlowAppend
	mismatchedPrompt.AgentPromptBytes++
	if got := calibrationScore(&mismatchedPrompt, baseline); got != 0 {
		t.Fatalf("mismatched agent prompt geometry received score %v", got)
	}
	winner := *fastPrefillButSlowAppend
	winner.AgentScenarioTimeS = 20
	winner.AgentScenarioMaxS = 32
	winnerMeasurement := calibrationMeasurement{Result: &winner, Score: calibrationScore(&winner, baseline)}
	if calibrationCandidateBetter(winnerMeasurement, calibrationMeasurement{Result: baseline, Score: 1}) {
		t.Fatal("candidate with a regressed confirmation sample was accepted")
	}
	winner.AgentScenarioMaxS = 21
	winnerMeasurement = calibrationMeasurement{Result: &winner, Score: calibrationScore(&winner, baseline)}
	if !calibrationCandidateBetter(winnerMeasurement, calibrationMeasurement{Result: baseline, Score: 1}) {
		t.Fatal("repeatable lower-turn-time candidate was rejected")
	}
}

func TestCalibrationScoreUsesSerialRequestWallTime(t *testing.T) {
	baseline := &benchmark.Result{GenTPS: 10, PromptTPS: 100, GenTokens: 256, PromptTokens: 100, GenTimeS: 10}
	fasterTurn := &benchmark.Result{GenTPS: 9, PromptTPS: 80, GenTokens: 256, PromptTokens: 100, GenTimeS: 8}
	if got := calibrationScore(fasterTurn, baseline); math.Abs(got-1.25) > 1e-9 {
		t.Fatalf("turn-time score = %v, want 1.25", got)
	}
	if calibrationCandidateBetter(
		calibrationMeasurement{Result: fasterTurn, Score: 1.25},
		calibrationMeasurement{Result: baseline, Score: 1},
	) {
		t.Fatal("faster aggregate turn with material prefill/decode regressions was accepted")
	}
	balanced := &benchmark.Result{GenTPS: 10.5, PromptTPS: 98, GenTokens: 256, PromptTokens: 100, GenTimeS: 9}
	if !calibrationCandidateBetter(
		calibrationMeasurement{Result: balanced, Score: 10.0 / 9.0},
		calibrationMeasurement{Result: baseline, Score: 1},
	) {
		t.Fatal("balanced phase-safe workflow improvement was rejected")
	}
	invalid := &benchmark.Result{GenTPS: 100, PromptTPS: 0, GenTokens: 256, PromptTokens: 100, GenTimeS: 1}
	if got := calibrationScore(invalid, baseline); got != 0 {
		t.Fatalf("incomplete candidate received score %v", got)
	}
}

func TestCalibrationConfirmationNormalizesDifferentSlotWidths(t *testing.T) {
	serial := &benchmark.Result{
		Parallel: 1, AgentSamples: 2, AgentTurnTimeS: 4, AgentTurnMaxS: 5,
		AgentWorkloadLanes: 4, AgentWorkloadTimeS: 16, AgentWorkloadMaxS: 20,
	}
	parallel := &benchmark.Result{
		Parallel: 2, AgentSamples: 2, AgentTurnTimeS: 6, AgentTurnMaxS: 7,
		AgentWorkloadLanes: 4, AgentWorkloadTimeS: 12, AgentWorkloadMaxS: 14,
	}
	if !calibrationCandidateBetter(
		calibrationMeasurement{Result: parallel, Score: 16.0 / 12.0},
		calibrationMeasurement{Result: serial, Score: 1},
	) {
		t.Fatal("faster normalized two-slot workflow was rejected by raw per-wave latency")
	}
}

func TestCalibrationScoreComparesIdenticalWorkAcrossSlotCounts(t *testing.T) {
	serial := &benchmark.Result{
		GenTPS: 20, PromptTPS: 200, GenTokens: 128, PromptTokens: 1000, MixedGenTokens: 64, MixedGenTPS: 10,
		AgentSamples: 2, AgentTurnTimeS: 8, AgentTurnMaxS: 8,
		AgentScenarioTimeS: 16, AgentScenarioMaxS: 16, AgentPromptBytes: 4096,
		AgentCachedTokens: 500, AgentNewPromptTokens: 50,
		AgentWorkloadLanes: 2, AgentWorkloadTimeS: 16,
	}
	parallel := &benchmark.Result{
		GenTPS: 35, PromptTPS: 300, GenTokens: 128, PromptTokens: 1000, MixedGenTokens: 64, MixedGenTPS: 12,
		AgentSamples: 2, AgentTurnTimeS: 10, AgentTurnMaxS: 10,
		AgentScenarioTimeS: 10, AgentScenarioMaxS: 10, AgentPromptBytes: 4096,
		AgentCachedTokens: 500, AgentNewPromptTokens: 50,
		AgentWorkloadLanes: 2, AgentWorkloadTimeS: 10,
	}
	if got := calibrationScore(parallel, serial); math.Abs(got-1.6) > 1e-9 {
		t.Fatalf("normalized workload score=%v, want 1.6", got)
	}
	for _, change := range []func(*benchmark.Result){
		func(r *benchmark.Result) { r.AgentWorkloadLanes++ },
		func(r *benchmark.Result) { r.GenTokens++ },
		func(r *benchmark.Result) { r.AgentPromptBytes++ },
	} {
		mismatch := *parallel
		change(&mismatch)
		if got := calibrationScore(&mismatch, serial); got != 0 {
			t.Fatalf("different requested work received score %v", got)
		}
	}
	for _, score := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), 0} {
		if calibrationCandidateBetter(calibrationMeasurement{Result: parallel, Score: score}, calibrationMeasurement{Result: serial, Score: 1}) {
			t.Fatalf("invalid score %v promoted", score)
		}
	}
}

func TestApplyCalibrationDecisionReplaysExploredBoundary(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	cfg.CacheDir = t.TempDir()
	base := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	scopeKey := calibrationScopeKey(req, model, be, caps, base)
	boundary := &placement.OptimizationBoundary{
		CandidateCount: 14, FeasibleCount: 9, MinBatch: 128, MaxBatch: 512,
		MinUBatch: 64, MaxUBatch: 2048, MinParallel: 1, MaxParallel: 4,
	}
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: "default",
		ValidationLevel:    placement.CalibrationValidationWorkflow,
		BaselineResidency:  placement.ResidencyRoomy,
		BaselineBottleneck: "GPU 1 layer service",
		Finalist:           "single-gpu-0",
		FinalistOutcome:    "baseline-won",
		ExploredBoundary:   boundary,
		DefaultTurnTimeS:   10, WinnerTurnTimeS: 10,
	}); err != nil {
		t.Fatal(err)
	}
	got := applyCalibrationDecision(req, cfg, model, be, caps, base)
	if got != base || !got.PerformanceTuned {
		t.Fatalf("default-won was not applied in place: %+v", got)
	}
	if got.Residency != placement.ResidencyRoomy || got.OptimizationBottleneck != "GPU 1 layer service" {
		t.Fatalf("default-won did not restore residency evidence: %+v", got)
	}
	if got.OptimizationBoundary == nil || got.OptimizationBoundary.CandidateCount != 14 || got.OptimizationBoundary.MaxUBatch != 2048 {
		t.Fatalf("default-won dropped the explored boundary: %+v", got.OptimizationBoundary)
	}
	if req.AppliedCalibration == nil || req.AppliedCalibration.FinalistOutcome != "baseline-won" {
		t.Fatalf("inspect handle missing: %+v", req.AppliedCalibration)
	}

	named := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: "kv-alternate",
		ValidationLevel:    placement.CalibrationValidationWorkflow,
		BaselineResidency:  placement.ResidencyRoomy,
		BaselineBottleneck: "CPU expert bandwidth",
		Finalist:           "kv-alternate",
		FinalistOutcome:    "promoted",
		ExploredBoundary:   boundary,
		DefaultTurnTimeS:   10, WinnerTurnTimeS: 8,
	}); err != nil {
		t.Fatal(err)
	}
	got = applyCalibrationDecision(req, cfg, model, be, caps, named)
	if got.KVPlacement != "gpu" || got.OptimizationBoundary == nil || got.OptimizationBoundary.CandidateCount != 14 {
		t.Fatalf("named winner dropped recorded boundary: %+v", got)
	}
	if !got.PerformanceTuned || req.AppliedCalibration == nil || req.AppliedCalibration.Winner != "kv-alternate" {
		t.Fatalf("named winner inspect handle missing: %+v", req.AppliedCalibration)
	}
}

func TestAutomaticPlanOnTightLaunchKeepsProvenShape(t *testing.T) {
	base := &placement.Strategy{
		Type: placement.MultiGPUDense, MainGPU: 0, TensorSplit: []float64{0.5, 0.5},
		BatchSize: 2048, UBatchSize: 512, Residency: placement.ResidencyTight,
		ResourceLedger: &placement.ResourceLedger{Exact: true, Fits: true},
	}
	same := *base
	same.BatchSize = 4096
	same.Residency = ""
	reshuffle := *base
	reshuffle.TensorSplit = []float64{0.9, 0.1}
	reshuffle.Residency = ""
	candidates := []placement.CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: placement.CandidateEstimate{Feasible: true, AgentCost: 1}},
		{Name: "topology-fast", Strategy: &reshuffle, Estimate: placement.CandidateEstimate{Feasible: true, AgentCost: 0.4}},
		{Name: "batch-4096-ubatch-512", Strategy: &same, Estimate: placement.CandidateEstimate{Feasible: true, AgentCost: 0.9}},
	}
	got := selectAutomaticCalibrationFinalist(placement.TightLiveCandidates(candidates), nil)
	if len(got) != 2 || got[1].Name != "batch-4096-ubatch-512" {
		t.Fatalf("tight auto plan live-tested a topology reshuffle: %+v", got)
	}
}

func TestExplainServingOptimizationDoesNotReplaceMeasuredWinner(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(16 * 1024)
	winner := &placement.Strategy{
		Type: placement.MoEOffload, KVPlacement: "gpu", NCPUMoE: 20,
		BatchSize: 256, UBatchSize: 256, Parallel: 2, ContextSize: 65536,
		PerformanceTuned: true, Residency: placement.ResidencyRoomy,
		OptimizationBoundary: &placement.OptimizationBoundary{CandidateCount: 11, FeasibleCount: 7, MaxUBatch: 1024},
	}
	got := explainServingOptimization(req, cfg, model, be, caps, winner, true)
	if got != winner {
		t.Fatal("full-frontier explain replaced a measured winner")
	}
	if got.OptimizationBoundary.CandidateCount != 11 || got.BatchSize != 256 {
		t.Fatalf("measured winner was rewritten: %+v", got)
	}
}

func TestOptimizationSummaryNamesKVSlotsAndBoundary(t *testing.T) {
	strategy := &placement.Strategy{
		Residency:   placement.ResidencyRoomy,
		ContextSize: 131072, Parallel: 2, BatchSize: 256, UBatchSize: 128,
		KVPlacement: "gpu", KVType: "q8_0", OptimizationBottleneck: "GPU 0 layer service",
		ResourceLedger:       &placement.ResourceLedger{Evidence: "backend-measured", Fits: true},
		OptimizationBoundary: &placement.OptimizationBoundary{CandidateCount: 8, FeasibleCount: 5, ExactCount: 1, MinBatch: 128, MaxBatch: 512, MinUBatch: 64, MaxUBatch: 256, MinParallel: 1, MaxParallel: 4, Topologies: []string{"moe"}},
	}
	lines := optimizationSummaryLines("dry-run", strategy, false)
	if len(lines) != 2 {
		t.Fatalf("summary lines=%d, want 2: %q", len(lines), lines)
	}
	if !strings.Contains(lines[0], "kv gpu/q8_0") || !strings.Contains(lines[0], "parallel 2") || !strings.Contains(lines[0], "per agent") {
		t.Fatalf("summary omitted serving knobs: %q", lines[0])
	}
	if !strings.Contains(lines[1], "8 candidates") || !strings.Contains(lines[1], "ubatch 64..256") {
		t.Fatalf("summary omitted explored boundary: %q", lines[1])
	}
}

func TestOptimizationDecisionRecordsFinalistOutcomeAndBoundary(t *testing.T) {
	boundary := &placement.OptimizationBoundary{CandidateCount: 14, FeasibleCount: 9}
	baseline := &placement.Strategy{
		Residency: placement.ResidencyRoomy, OptimizationBottleneck: "GPU 1 layer service",
		OptimizationBoundary: boundary,
	}
	finalist := &placement.Strategy{}
	candidates := []placement.CalibrationCandidate{
		{Name: "default", Strategy: baseline},
		{Name: "single-gpu-0", Strategy: finalist, Estimate: placement.CandidateEstimate{AgentCost: 0.5, Confidence: "derived"}},
	}
	decision := &placement.CalibrationDecision{Winner: "default"}
	annotateOptimizationDecision(decision, candidates, []calibrationMeasurement{{Name: "default"}, {Name: "single-gpu-0"}})
	if decision.Finalist != "single-gpu-0" || decision.FinalistOutcome != "baseline-won" ||
		decision.BaselineResidency != placement.ResidencyRoomy || decision.ExploredBoundary != boundary {
		t.Fatalf("optimization evidence was not attached: %+v", decision)
	}
}

func TestAutomaticCalibrationPromotionRequiresCompleteAgentEvidence(t *testing.T) {
	complete := &placement.CalibrationDecision{
		Finalist: "ubatch-512", FinalistOutcome: "baseline-won",
		DefaultAgentSamples: 2, WinnerAgentSamples: 2,
		DefaultTurnTimeS: 10, WinnerTurnTimeS: 9,
		DefaultTurnMaxS: 11, WinnerTurnMaxS: 10,
		DefaultCachedTokens: 1000, WinnerCachedTokens: 1000,
		DefaultNewPromptTokens: 50, WinnerNewPromptTokens: 50,
		DefaultMixedTPS: 20, WinnerMixedTPS: 21,
		AgentPromptBytes: 4096,
	}
	if !automaticCalibrationEvidenceValid(complete) {
		t.Fatal("complete repeated agent evidence was rejected")
	}
	missingReuse := *complete
	missingReuse.WinnerCachedTokens = 0
	if automaticCalibrationEvidenceValid(&missingReuse) {
		t.Fatal("winner without cache-reuse evidence became automatic-eligible")
	}
	oneSample := *complete
	oneSample.DefaultAgentSamples = 1
	if automaticCalibrationEvidenceValid(&oneSample) {
		t.Fatal("one noisy baseline sample became automatic-eligible")
	}
	unavailable := *complete
	unavailable.FinalistOutcome = "unavailable"
	if automaticCalibrationEvidenceValid(&unavailable) {
		t.Fatal("baseline-only evidence became automatic-eligible after challenger admission failed")
	}
	if !automaticCalibrationAdmissionEvidenceValid(&placement.CalibrationDecision{
		Winner: "default", Finalist: "ubatch-512", FinalistOutcome: "unavailable",
		FinalistFailureClass: "cuda-oom", FinalistFailureReason: "exact candidate CUDA OOM",
	}) {
		t.Fatal("exact unavailable finalist was not recognized as admission-only evidence")
	}
	if automaticCalibrationAdmissionEvidenceValid(&placement.CalibrationDecision{
		Winner: "default", Finalist: "ubatch-512", FinalistOutcome: "unavailable",
	}) {
		t.Fatal("unexplained unavailable finalist became persistent negative evidence")
	}
	if automaticCalibrationAdmissionEvidenceValid(complete) {
		t.Fatal("a measured comparison was mislabeled admission-only")
	}
}

func TestCalibrationPlanCachesUnavailableAdmissionWithoutPerformancePromotion(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(60 * 1024)
	cfg.CacheDir = t.TempDir()
	strategy := &placement.Strategy{
		Type: placement.MoEOffload, KVPlacement: "cpu", KVQuality: "mid",
		ContextSize: 32768, Parallel: 1, BatchSize: 2048, UBatchSize: 512,
		NCPUMoE: 40,
	}
	first := calibrationPlan(req, cfg, model, be, caps, strategy)
	if len(first) < 2 {
		t.Fatalf("test setup produced no automatic finalist: %+v", first)
	}
	scope := calibrationScopeKey(req, model, be, caps, strategy)
	decision := placement.CalibrationDecision{
		ScopeKey: scope, ModelBasename: model.Path,
		Winner: "default", ValidationLevel: placement.CalibrationValidationAdmission,
		Finalist: first[1].Name, FinalistOutcome: "unavailable",
		FinalistFailureClass: "memory", FinalistFailureReason: "exact candidate failed memory admission",
	}
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, decision); err != nil {
		t.Fatal(err)
	}
	if got := calibrationPlan(req, cfg, model, be, caps, strategy); len(got) != 0 {
		t.Fatalf("identical failed admission was scheduled again: %+v", got)
	}
	applied := applyCalibrationDecision(req, cfg, model, be, caps, strategy)
	if applied != strategy || applied.PerformanceTuned || req.AppliedCalibration != nil {
		t.Fatalf("admission-only evidence became a performance winner: strategy=%+v request=%+v", applied, req)
	}
}

func TestExactCalibrationCandidateRejectsRecoveredArgv(t *testing.T) {
	candidate := []string{"llama-server", "-ub", "1024", "--n-cpu-moe", "40"}
	if !exactCalibrationCandidate(candidate, append([]string(nil), candidate...)) {
		t.Fatal("identical candidate argv was rejected")
	}
	recovered := []string{"llama-server", "-ub", "512", "--n-cpu-moe", "41"}
	if exactCalibrationCandidate(candidate, recovered) {
		t.Fatal("memory-recovered argv was accepted under the original candidate name")
	}
}

func TestCalibrationAdvisorIncidentOffersOnlyMeasuredCandidates(t *testing.T) {
	req, _, model, be, caps := calibrateTestSetup(60 * 1024)
	result := &benchmark.Result{GenTPS: 10, PromptTPS: 100, GenTokens: 256, PromptTokens: 100, GenTimeS: 10}
	measurements := []calibrationMeasurement{
		{Name: "default", Strategy: &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu"}, Result: result, Score: 1},
		{Name: "kv-alternate", Strategy: &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "gpu"}, Result: result, Score: 1.02},
	}
	incident := calibrationAdvisorIncident(req, model, be, caps, "scope", measurements)
	if incident.Mode != "optimizer" || len(incident.Candidates) != 2 || len(incident.AllowedActions) != 2 {
		t.Fatalf("unexpected optimizer incident: %#v", incident)
	}
	for _, candidate := range incident.Candidates {
		if !candidate.Verified || candidate.Metrics["decode_tps"] <= 0 || candidate.Metrics["relative_turn_score"] <= 0 || candidate.Metrics["agent_workflow_seconds"] <= 0 {
			t.Fatalf("unverified/incomplete candidate leaked into incident: %#v", candidate)
		}
	}
}

func TestCalibrationPlanSkipsSingleGPU(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	req.Calibrate = calibrateOn
	caps.GPUs = caps.GPUs[:1] // one GPU only
	strategy := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	if got := calibrationPlan(req, cfg, model, be, caps, strategy); got != nil {
		t.Fatalf("single GPU offers no placement alternatives, got %d", len(got))
	}
}

func TestCalibrationPlanSkipsWhenDecisionCached(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	cfg.CacheDir = t.TempDir()
	strategy := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	// Seed a prior decision for this exact scope.
	scopeKey := calibrationScopeKey(req, model, be, caps, strategy)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: "kv-alternate", DefaultTPS: 20, WinnerTPS: 24,
		ValidationLevel: placement.CalibrationValidationWorkflow,
	}); err != nil {
		t.Fatalf("seed decision: %v", err)
	}
	if got := calibrationPlan(req, cfg, model, be, caps, strategy); got != nil {
		t.Fatalf("a cached decision must suppress re-calibration, got %d candidates", len(got))
	}
}

func TestApplyCalibrationDecisionReplaysParallelAgentBatchWinner(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(16 * 1024)
	cfg.CacheDir = t.TempDir()
	req.ClaudeCode = true
	req.Parallel = 2
	req.ParallelSet = true
	model.IsMoE = false
	model.HasSSM = 1
	model.CTXTrain = 131072

	measuredBase := &placement.Strategy{
		Type: placement.MultiGPUDense, HasSSM: true, Parallel: 2,
		ContextSize: 131072, BatchSize: 4096, UBatchSize: 512,
		TensorSplit: []float64{0.25, 0.75}, MainGPU: 1,
	}
	claudeCodeSlotAdjust(measuredBase, model, true, true, false, false)
	if measuredBase.BatchSize != 128 || measuredBase.UBatchSize != 128 {
		t.Fatalf("fixture did not reach effective baseline: %+v", measuredBase)
	}
	scope := calibrationScopeKey(req, model, be, caps, measuredBase)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scope, Winner: "batch-256-ubatch-256", DefaultTPS: 40, WinnerTPS: 44,
		ValidationLevel: placement.CalibrationValidationWorkflow,
	}); err != nil {
		t.Fatal(err)
	}

	cold := &placement.Strategy{
		Type: placement.MultiGPUDense, HasSSM: true, Parallel: 2,
		ContextSize: 131072, BatchSize: 4096, UBatchSize: 512,
		TensorSplit: []float64{0.25, 0.75}, MainGPU: 1,
	}
	got := applyCalibrationDecision(req, cfg, model, be, caps, cold)
	if got.BatchSize != 256 || got.UBatchSize != 256 || !got.BatchTuned {
		t.Fatalf("cached agent batch winner was not replayed: %+v", got)
	}
	// The ordinary finalization pass is intentionally idempotent: it must not
	// reset a measured winner to the conservative 128/128 baseline.
	claudeCodeSlotAdjust(got, model, true, true, false, false)
	if got.BatchSize != 256 || got.UBatchSize != 256 {
		t.Fatalf("final fairness pass erased cached winner: %+v", got)
	}
}

func TestCalibrationPlanSmallMoEOffered(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	req.Calibrate = calibrateOn
	strategy := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	got := calibrationPlan(req, cfg, model, be, caps, strategy)
	if len(got) < 2 {
		t.Fatalf("a small multi-GPU MoE with no cached decision should calibrate, got %d", len(got))
	}
	if got[0].Name != "default" {
		t.Fatalf("candidate 0 must be the default, got %q", got[0].Name)
	}
}

func TestApplyCalibrationDecisionRestoresWinner(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	cfg.CacheDir = t.TempDir()
	base := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	scopeKey := calibrationScopeKey(req, model, be, caps, base)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: "kv-alternate", DefaultTPS: 20, WinnerTPS: 24.5,
		ValidationLevel: placement.CalibrationValidationWorkflow,
	}); err != nil {
		t.Fatalf("seed decision: %v", err)
	}
	got := applyCalibrationDecision(req, cfg, model, be, caps, base)
	if got == base {
		t.Fatal("cached kv-alternate winner was not applied")
	}
	if got.KVPlacement != "gpu" {
		t.Fatalf("kv-alternate should restore KV=gpu, got %q", got.KVPlacement)
	}
	// The base estimate must be untouched for the next caller.
	if base.KVPlacement != "cpu" {
		t.Fatalf("base strategy mutated to %q", base.KVPlacement)
	}
	if req.CalibrationScreened {
		t.Fatal("workflow-validated winner was mislabeled as a screened configuration")
	}
}

func TestScreenedCalibrationDecisionIsExplicitOnly(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	cfg.CacheDir = t.TempDir()
	base := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	scopeKey := calibrationScopeKey(req, model, be, caps, base)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: "kv-alternate", DefaultTPS: 20, WinnerTPS: 24.5,
		ValidationLevel: placement.CalibrationValidationScreened,
	}); err != nil {
		t.Fatalf("seed screened decision: %v", err)
	}
	if got := applyCalibrationDecision(req, cfg, model, be, caps, base); got != base || req.CalibrationScreened {
		t.Fatalf("auto launch consumed screened evidence: got=%+v marked=%t", got, req.CalibrationScreened)
	}
	if got := calibrationPlan(req, cfg, model, be, caps, base); len(got) < 2 {
		t.Fatalf("screened explicit evidence blocked the automatic workflow optimizer: %+v", got)
	}
	req.Calibrate = calibrateOn
	got := applyCalibrationDecision(req, cfg, model, be, caps, base)
	if got == base || got.KVPlacement != "gpu" || !req.CalibrationScreened {
		t.Fatalf("explicit screen did not apply and mark its provisional winner: got=%+v marked=%t", got, req.CalibrationScreened)
	}
}

func TestApplyCalibrationDecisionRestoresMeasuredUBatchWinner(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	cfg.CacheDir = t.TempDir()
	base := &placement.Strategy{
		Type: placement.MoEOffload, KVPlacement: "cpu", KVQuality: "mid", KVType: "q8_0",
		ContextSize: 8192, Parallel: 1, BatchSize: 2048, UBatchSize: 512, NCPUMoE: 40,
	}
	scopeKey := calibrationScopeKey(req, model, be, caps, base)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scopeKey, ModelBasename: filepath.Base(model.Path), Winner: "ubatch-1024",
		DefaultTPS: 10, WinnerTPS: 10.2, DefaultPromptTPS: 63.4, WinnerPromptTPS: 99.4,
		ValidationLevel: placement.CalibrationValidationWorkflow,
	}); err != nil {
		t.Fatalf("seed decision: %v", err)
	}
	got := applyCalibrationDecision(req, cfg, model, be, caps, base)
	if got == base || got.UBatchSize != 1024 {
		candidates := calibrationCandidates(req, cfg, model, be, caps, base)
		names := make([]string, 0, len(candidates))
		for _, candidate := range candidates {
			names = append(names, candidate.Name)
		}
		t.Fatalf("cached ubatch winner was not restored (candidates=%v): %+v", names, got)
	}
	if base.UBatchSize != 512 {
		t.Fatalf("base strategy mutated to ubatch %d", base.UBatchSize)
	}
}

func TestApplyCalibrationDecisionIgnoresDefaultWinner(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	cfg.CacheDir = t.TempDir()
	base := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	scopeKey := calibrationScopeKey(req, model, be, caps, base)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: "default", DefaultTPS: 22, WinnerTPS: 22,
	}); err != nil {
		t.Fatalf("seed decision: %v", err)
	}
	got := applyCalibrationDecision(req, cfg, model, be, caps, base)
	if got != base {
		t.Fatal("a default winner must leave the estimated strategy in place")
	}
}

func TestParseCalibrateMode(t *testing.T) {
	for _, ok := range []string{"auto", "on", "off", "AUTO", " On "} {
		if _, err := parseCalibrateMode(ok); err != nil {
			t.Fatalf("parseCalibrateMode(%q): %v", ok, err)
		}
	}
	if _, err := parseCalibrateMode("yes"); err == nil {
		t.Fatal("parseCalibrateMode must reject unknown modes")
	}
}

func TestRuntimeOOMInvalidatesCalibrationDecision(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	cfg.CacheDir = t.TempDir()
	base := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	scopeKey := calibrationScopeKey(req, model, be, caps, base)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: "kv-alternate", DefaultTPS: 20, WinnerTPS: 24,
		ValidationLevel: placement.CalibrationValidationWorkflow,
	}); err != nil {
		t.Fatalf("seed decision: %v", err)
	}
	if _, err := placement.LoadCalibrationDecision(cfg.CacheDir, scopeKey); err != nil {
		t.Fatalf("decision not present before invalidation: %v", err)
	}
	// A runtime OOM must invalidate the calibration decision for the same scope,
	// so the OOM'd placement is never re-declared the winner.
	if err := invalidateRuntimeOOMLaunch(req, cfg, model, be, caps, base, nil, "runtime oom"); err != nil {
		t.Fatalf("invalidate: %v", err)
	}
	if _, err := placement.LoadCalibrationDecision(cfg.CacheDir, scopeKey); err == nil {
		t.Fatal("calibration decision survived a runtime OOM; stale winner would be re-applied")
	}
}

// The runtime OOM invalidation must delete the decision even when the serving
// strategy is the calibration winner (kv-alternate), whose scope key differs
// from the default strategy the decision was saved under. This is the real
// flow: runCalibration saves under the default key, the winner serves, then a
// runtime OOM invalidates with the winner strategy in hand.
func TestRuntimeOOMInvalidatesWinnerStrategyCalibrationDecision(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	cfg.CacheDir = t.TempDir()
	// Compute the default strategy and seed a decision under its scope key —
	// exactly how runCalibration saves.
	base, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil || base == nil {
		t.Fatalf("base compute: %v", err)
	}
	defaultKey := calibrationScopeKey(req, model, be, caps, base)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: defaultKey, Winner: "kv-alternate", DefaultTPS: 20, WinnerTPS: 24,
	}); err != nil {
		t.Fatalf("seed decision: %v", err)
	}
	// A calibration candidate (kv-alternate) with a different scope key serves.
	winner := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "gpu", NCPUMoE: 40}
	winnerKey := calibrationScopeKey(req, model, be, caps, winner)
	if winnerKey == defaultKey {
		t.Fatal("test setup: winner and default keys must differ for this scenario")
	}
	// A runtime OOM on the winner must delete the decision saved under the
	// default key.
	if err := invalidateRuntimeOOMLaunch(req, cfg, model, be, caps, winner, nil, "runtime oom"); err != nil {
		t.Fatalf("invalidate: %v", err)
	}
	if _, err := placement.LoadCalibrationDecision(cfg.CacheDir, defaultKey); err == nil {
		t.Fatal("calibration decision saved under the default key survived an OOM on the winner")
	}
}

func TestRuntimeOOMInvalidatesUBatchWinnerCalibrationDecision(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	cfg.CacheDir = t.TempDir()
	base := &placement.Strategy{
		Type: placement.MoEOffload, KVPlacement: "cpu", KVQuality: "mid", KVType: "q8_0",
		ContextSize: 8192, Parallel: 1, BatchSize: 2048, UBatchSize: 512, NCPUMoE: 40,
	}
	defaultKey := calibrationScopeKey(req, model, be, caps, base)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: defaultKey, ModelBasename: filepath.Base(model.Path), Winner: "ubatch-1024",
		DefaultTPS: 10, WinnerTPS: 10.2,
	}); err != nil {
		t.Fatalf("seed decision: %v", err)
	}
	winner := *base
	winner.UBatchSize = 1024
	if calibrationScopeKey(req, model, be, caps, &winner) == defaultKey {
		t.Fatal("test setup: winner ubatch must have a distinct serving scope")
	}
	if err := invalidateRuntimeOOMLaunch(req, cfg, model, be, caps, &winner, nil, "runtime oom"); err != nil {
		t.Fatalf("invalidate: %v", err)
	}
	if _, err := placement.LoadCalibrationDecision(cfg.CacheDir, defaultKey); err == nil {
		t.Fatal("ubatch winner decision survived runtime OOM invalidation")
	}
}

func TestMeasuredRecomputeReappliesCachedCalibrationWinner(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	cfg.CacheDir = t.TempDir()
	base, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil || base == nil {
		t.Fatalf("base compute: %v", err)
	}
	scopeKey := calibrationScopeKey(req, model, be, caps, base)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: "kv-alternate", DefaultTPS: 20, WinnerTPS: 24,
		ValidationLevel: placement.CalibrationValidationWorkflow,
	}); err != nil {
		t.Fatalf("seed decision: %v", err)
	}
	// A corrective recompute (cache bypassed) must re-apply the cached winner,
	// not replace it with an unbenchmarked estimate.
	recomputed, err := placement.Compute(caps, model, func() placement.Options {
		o := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
		o.SkipPlacementCache = true
		o.CacheFile = ""
		return o
	}())
	if err != nil {
		t.Fatalf("recompute: %v", err)
	}
	applied := recomputeAndApplyCalibration(req, cfg, model, be, caps, recomputed)
	if applied == recomputed {
		t.Fatal("cached calibration winner was not re-applied to the corrected recompute")
	}
}

// --no-cached-config must bypass a cached calibration winner: the launch is the
// user's escape hatch from a stale measured decision, so it re-derives from the
// fresh estimate instead of re-applying the last winner.
func TestApplyCalibrationDecisionSkipsWithNoCachedConfig(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	cfg.CacheDir = t.TempDir()
	base := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	scopeKey := calibrationScopeKey(req, model, be, caps, base)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: "kv-alternate", DefaultTPS: 20, WinnerTPS: 24.5,
	}); err != nil {
		t.Fatalf("seed decision: %v", err)
	}
	req.NoCachedConfig = true
	got := applyCalibrationDecision(req, cfg, model, be, caps, base)
	if got != base {
		t.Fatal("--no-cached-config must not re-apply a cached calibration winner")
	}
	if base.KVPlacement != "cpu" {
		t.Fatalf("base strategy mutated to %q", base.KVPlacement)
	}
}
