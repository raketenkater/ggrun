package main

// Pins for "the calibration cost is paid once": what a saved decision and a
// verified config suppress on the next identical launch, and where today's
// policy pays the search again. Tests only; no product behaviour is changed.
// A test named ..._KnownDefect asserts today's behaviour and says what must
// flip when the defect is fixed.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/benchmark"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// reusePinResult is a complete agent workload result: two samples, cache reuse,
// mixed decode. automaticCalibrationEvidenceValid accepts a comparison of two
// such results.
func reusePinResult() *benchmark.Result {
	return &benchmark.Result{
		Parallel: 1, GenTPS: 40, PromptTPS: 800, MixedGenTPS: 20,
		GenTokens: 128, PromptTokens: 4000, MixedGenTokens: 64, GenTimeS: 10,
		AgentSamples: 2, AgentTurnTimeS: 10, AgentTurnMaxS: 11,
		AgentScenarioTimeS: 30, AgentScenarioMaxS: 31, AgentPromptBytes: 4096,
		AgentCachedTokens: 1800, AgentNewPromptTokens: 80,
	}
}

// reusePinFixture is the 60 GB MoE launch of
// TestCalibrationPlanCachesUnavailableAdmissionWithoutPerformancePromotion,
// which yields an automatic plan with a calculated finalist.
func reusePinFixture(t *testing.T) (*launchRequest, *config.Config, *placement.ModelProfile, *backendInfo, *detect.Capabilities, *placement.Strategy) {
	t.Helper()
	req, cfg, model, be, caps := calibrateTestSetup(60 * 1024)
	cfg.CacheDir = t.TempDir()
	strategy := &placement.Strategy{
		Type: placement.MoEOffload, KVPlacement: "cpu", KVQuality: "mid",
		ContextSize: 32768, Parallel: 1, BatchSize: 2048, UBatchSize: 512,
		NCPUMoE: 40,
	}
	return req, cfg, model, be, caps, strategy
}

// reusePinMeasuredDecision builds the decision runCalibration returns when the
// baseline beat the finalist it actually loaded (measured[1], after any
// telemetry- or phase-directed swap): scopedCalibrationDecision for the
// default, then annotateOptimizationDecision over the measured order.
func reusePinMeasuredDecision(t *testing.T, scope placement.CalibrationScopeKey, model *placement.ModelProfile, measured []placement.CalibrationCandidate) *placement.CalibrationDecision {
	t.Helper()
	if len(measured) < 2 {
		t.Fatalf("need a baseline and a finalist, got %d candidates", len(measured))
	}
	result := reusePinResult()
	measurements := []calibrationMeasurement{
		{Name: "default", Strategy: measured[0].Strategy, Result: result, Score: 1},
		{Name: measured[1].Name, Strategy: measured[1].Strategy, Result: result, Score: 0.95},
	}
	decision := scopedCalibrationDecision(scope, model, result, measurements[0])
	if decision == nil {
		t.Fatal("no decision for a complete baseline result")
	}
	annotateOptimizationDecision(decision, measured, measurements)
	if decision.Winner != "default" || decision.Finalist != measured[1].Name || decision.FinalistOutcome != "baseline-won" {
		t.Fatalf("decision does not record the measured finalist: winner=%q finalist=%q outcome=%q",
			decision.Winner, decision.Finalist, decision.FinalistOutcome)
	}
	return decision
}

// reusePinDegradedDecision is the save in the launch's degraded-profile block
// (main.go, "baseline kept for ... identical launches skip this search"):
// degradedBaselinePersistable, then ValidationLevel baseline-bounded.
func reusePinDegradedDecision(t *testing.T, scope placement.CalibrationScopeKey, model *placement.ModelProfile, measured []placement.CalibrationCandidate) placement.CalibrationDecision {
	t.Helper()
	decision := reusePinMeasuredDecision(t, scope, model, measured)
	if !degradedBaselinePersistable(calibrateAuto, decision, true) {
		t.Fatalf("a functional degraded profile would not keep this baseline-won decision: %+v", decision)
	}
	decision.ValidationLevel = placement.CalibrationValidationBaselineBounded
	return *decision
}

func reusePinNames(candidates []placement.CalibrationCandidate) []string {
	names := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		names = append(names, candidate.Name)
	}
	return names
}

// After the baseline is measured, runCalibration may replace the calculated
// finalist with a phase-directed one (calibrate.go "replacing calculated
// finalist ... with phase-directed ..."). On the same frontier the two
// selectors pick different finalists in an ordinary case: a scheduler-limited
// mixed phase points at the smallest batch/ubatch, the cost ranking at the
// cheapest estimate. The decision then names the directed finalist, and it
// suppresses a retry only for that name, never for the calculated one that
// the next launch's calibrationPlan compares against.
func TestPhaseDirectedFinalistIsNotTheCalculatedFinalist_KnownDefect(t *testing.T) {
	roomy := func(batch, ubatch int) *placement.Strategy {
		return &placement.Strategy{
			Type: placement.MoEOffload, Residency: placement.ResidencyRoomy,
			ContextSize: 32768, Parallel: 1, BatchSize: batch, UBatchSize: ubatch,
			KVPlacement: "cpu", KVQuality: "mid", NCPUMoE: 40,
			ResourceLedger: &placement.ResourceLedger{Fits: true},
		}
	}
	frontier := []placement.CalibrationCandidate{
		{Name: "default", Strategy: roomy(2048, 512)},
		{Name: "batch-1024-ubatch-512", Strategy: roomy(1024, 512), Estimate: placement.CandidateEstimate{Feasible: true, AgentCost: 0.8}},
		{Name: "batch-64-ubatch-64", Strategy: roomy(64, 64), Estimate: placement.CandidateEstimate{Feasible: true, AgentCost: 0.95}},
	}
	calculated := selectAutomaticCalibrationAdmissionPlan(frontier, nil, calibrationAutoMaxCandidates)
	if len(calculated) < 2 || calculated[1].Name != "batch-1024-ubatch-512" {
		t.Fatalf("calculated finalist = %v, want batch-1024-ubatch-512 second", reusePinNames(calculated))
	}
	directed, ok := placement.SelectBottleneckFinalist(frontier, placement.WorkloadBottleneck{
		Primary: placement.BottleneckScheduler, PrimaryPhase: benchmark.AgentPhaseMixed, Confidence: "measured",
	})
	if !ok || directed.Name != "batch-64-ubatch-64" {
		t.Fatalf("phase-directed finalist = %q (ok=%v), want batch-64-ubatch-64", directed.Name, ok)
	}
	measured := prioritizeCalibrationFinalist(calculated, directed)
	decision := reusePinDegradedDecision(t, placement.CalibrationScopeKey{ModelIdentity: "m"}, &placement.ModelProfile{Path: "m.gguf"}, measured)
	if decision.Finalist != directed.Name {
		t.Errorf("saved finalist = %q, want the measured directed %q", decision.Finalist, directed.Name)
	}
	if !decision.SuppressesAutomaticAdmissionRetry(directed.Name) {
		t.Error("the decision does not even suppress the finalist it measured")
	}
	// KNOWN DEFECT: calibrationPlan asks about the calculated finalist. When
	// fixed, an identical scope must not be re-searched after this decision
	// (this assertion flips, or the plan stops comparing by finalist name).
	if decision.SuppressesAutomaticAdmissionRetry(calculated[1].Name) {
		t.Error("defect appears fixed: the decision now suppresses the calculated finalist; update this pin")
	}
}

// End to end at the policy level: the degraded-profile launch saves a
// baseline-bounded decision naming the finalist it measured; the next
// identical launch calls calibrationPlan for the same request, model, hardware
// and baseline (same scope key). If the measured finalist was the calculated
// one, the search is suppressed (control). If it was any other candidate, as
// after the phase-directed swap above, the full search is planned again: a
// challenger load and a baseline restore on every launch, although the launch
// printed that identical launches skip it.
func TestCalibrationPlan_DegradedProfileRepeatsSearch_KnownDefect(t *testing.T) {
	req, cfg, model, be, caps, strategy := reusePinFixture(t)
	first := calibrationPlan(req, cfg, model, be, caps, strategy)
	if len(first) < 2 {
		t.Fatalf("fixture produced no automatic finalist: %v", reusePinNames(first))
	}
	calculated := first[1].Name
	frontier := analyzedMeasuredCalibrationFrontier(req, cfg, model, be, caps, strategy, nil)
	var directed placement.CalibrationCandidate
	for _, candidate := range frontier[1:] {
		if candidate.Strategy != nil && candidate.Name != calculated && candidate.Name != "default" {
			directed = candidate
			break
		}
	}
	if directed.Name == "" {
		t.Fatalf("frontier offers no second challenger: %v", reusePinNames(frontier))
	}
	t.Logf("calculated finalist %s, measured stand-in %s, frontier %v", calculated, directed.Name, reusePinNames(frontier))
	scope := calibrationScope(req, model, be, caps, strategy)

	save := func(d placement.CalibrationDecision) {
		t.Helper()
		if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, d); err != nil {
			t.Fatal(err)
		}
	}

	// Control: the measured finalist is the calculated one.
	save(reusePinDegradedDecision(t, scope, model, first))
	if got := calibrationPlan(req, cfg, model, be, caps, strategy); len(got) != 0 {
		t.Errorf("control: a bounded decision for the calculated finalist did not suppress the search: %v", reusePinNames(got))
	}

	// The measured finalist differs from the calculated one.
	save(reusePinDegradedDecision(t, scope, model, prioritizeCalibrationFinalist(first, directed)))
	if loaded, err := placement.LoadCalibrationDecision(cfg.CacheDir, scope.String()); err != nil || loaded.Finalist != directed.Name {
		t.Fatalf("the bounded decision is not the one loaded for this scope: %+v err=%v", loaded, err)
	}
	got := calibrationPlan(req, cfg, model, be, caps, strategy)
	// KNOWN DEFECT: when fixed, this identical launch plans nothing (len 0).
	if len(got) < 2 {
		t.Errorf("defect appears fixed: the identical launch no longer re-searches (%v); update this pin", reusePinNames(got))
	} else if got[1].Name != calculated {
		t.Errorf("repeated plan finalist = %q, want the calculated %q", got[1].Name, calculated)
	}
}

// A release that bumps CalibrationSchemaVersion retires every saved decision:
// a decision written by the previous schema (27 in v3.2.13, 29 on main) loads
// as an error that does not name the schema, and the next identical launch
// plans the full search again. The user is told, in internal terms, by
// explainCalibrationMiss: "... was recorded with different SchemaVersion".
func TestCalibrationPlan_OlderSchemaDecisionRepeatsSearch_KnownCost(t *testing.T) {
	req, cfg, model, be, caps, strategy := reusePinFixture(t)
	first := calibrationPlan(req, cfg, model, be, caps, strategy)
	if len(first) < 2 {
		t.Fatalf("fixture produced no automatic finalist: %v", reusePinNames(first))
	}
	scope := calibrationScope(req, model, be, caps, strategy)
	decision := reusePinMeasuredDecision(t, scope, model, first)
	if !automaticCalibrationEvidenceValid(decision) {
		t.Fatalf("complete comparison was rejected: %+v", decision)
	}
	decision.ValidationLevel = placement.CalibrationValidationWorkflow
	decision.SchemaVersion = placement.CalibrationSchemaVersion - 2 // the v3.2.13 -> main distance
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, *decision); err != nil {
		t.Fatal(err)
	}

	if _, err := placement.LoadCalibrationDecision(cfg.CacheDir, scope.String()); err == nil {
		t.Error("an older-schema decision was accepted")
	} else if err.Error() != "calibration decision scope mismatch" {
		t.Errorf("older-schema load error = %q; pin the text the user would see", err)
	}
	file, differing := placement.ExplainCalibrationScopeMiss(cfg.CacheDir, placement.CalibrationModelBasename(model.Path), scope)
	if file == "" || len(differing) != 1 || differing[0] != "SchemaVersion" {
		t.Errorf("miss explanation file=%q differing=%v, want only SchemaVersion", file, differing)
	}

	read := captureStdout(t)
	got := calibrationPlan(req, cfg, model, be, caps, strategy)
	out := read()
	if len(got) < 2 {
		t.Errorf("an older-schema decision suppressed the search (%v); update this pin", reusePinNames(got))
	}
	if !strings.Contains(out, "was recorded with different SchemaVersion") {
		t.Errorf("the repeated search was not explained; stdout=%q", out)
	}

	// Control: the same decision under the current schema suppresses it.
	decision.SchemaVersion = placement.CalibrationSchemaVersion
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, *decision); err != nil {
		t.Fatal(err)
	}
	if got := calibrationPlan(req, cfg, model, be, caps, strategy); len(got) != 0 {
		t.Errorf("control: a current-schema workflow decision did not suppress the search: %v", reusePinNames(got))
	}
}

// A planLogicVersion bump retires every verified config. The record file is
// named from the first 12 key characters, so the older record is found and
// rejected as "unusable (verified config scope mismatch)" rather than reported
// missing; nothing says that an update retired it. The next active launch
// overwrites the same file.
func TestVerifiedConfig_OlderPlanLogicRecordIsUnusable_KnownCost(t *testing.T) {
	cacheDir := t.TempDir()
	cfg := &config.Config{CacheDir: cacheDir}
	model := &placement.ModelProfile{Path: "/models/target.gguf", SizeBytes: 1 << 30, TotalSizeMB: 1024, ModelArch: "llama", NumLayers: 32}
	backend := &backendInfo{Tag: "llama", Identity: "backend-build", Path: "/usr/bin/llama-server"}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "gpu", VRAMTotalMB: 24576, BandwidthMBps: 30000}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	strategy := &placement.Strategy{
		Type: placement.SingleGPU, ContextSize: 8192, KVPlacement: "gpu", KVType: "q8_0",
		BatchSize: 2048, UBatchSize: 512, Parallel: 1, MMap: true, FlashAttention: true,
		BackendTag: "llama", Threads: 8, ThreadsBatch: 8,
	}
	req := &launchRequest{CtxFlag: "8192", KVQuality: "mid", KVPlacement: "gpu", Parallel: 1}
	key := verifiedConfigScopeKey(req, model, backend, caps)
	suffix := "-plan" + planLogicVersion
	if !strings.HasSuffix(key, suffix) {
		t.Fatalf("verified key %q is not bound to plan logic %q", key, planLogicVersion)
	}
	older := strings.TrimSuffix(key, suffix) + "-plan-previous"
	if placement.VerifiedConfigPath(cacheDir, older) != placement.VerifiedConfigPath(cacheDir, key) {
		t.Errorf("an older plan-logic record no longer shares the file; the miss now reads as 'no verified config'")
	}
	if _, err := placement.SaveVerifiedConfig(cacheDir, placement.VerifiedConfigToRecord(
		older, filepath.Base(model.Path), strategy, backend.Identity, backend.Path, "", "")); err != nil {
		t.Fatal(err)
	}
	_, err := placement.LoadVerifiedConfig(cacheDir, key)
	switch {
	case err == nil:
		t.Error("a verified config from older plan logic was replayed")
	case os.IsNotExist(err):
		t.Errorf("older record reported as missing: %v", err)
	case err.Error() != "verified config scope mismatch":
		t.Errorf("older record error = %q; pin the text the user would see", err)
	}
	saveVerifiedConfigForLaunch(cfg, req, model, backend, caps, strategy, nil)
	if loaded, err := placement.LoadVerifiedConfig(cacheDir, key); err != nil || loaded.ScopeKey != key {
		t.Errorf("the next active launch did not replace the older record: %+v err=%v", loaded, err)
	}
}

// reusePinMoELaunch is the MoE fixture of TestComputeReusesVerifiedConfigForMoE
// (placement.Compute plans it), reached through the launch's own option
// builder, so the verified-config key is the one a real launch computes.
func reusePinMoELaunch(t *testing.T) (*config.Config, *placement.ModelProfile, *backendInfo, *detect.Capabilities, func() *launchRequest) {
	t.Helper()
	dir := t.TempDir()
	cfg := config.Defaults()
	cfg.CacheDir = dir
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "gpu-a", VRAMTotalMB: 24576, VRAMUsedMB: 512, BandwidthMBps: 31504},
			{Index: 1, Name: "gpu-b", VRAMTotalMB: 12288, VRAMUsedMB: 512, BandwidthMBps: 12000},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 120000},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &placement.ModelProfile{
		Path: filepath.Join(dir, "moe.gguf"), SizeBytes: 146 << 30, TotalSizeMB: 146 * 1024,
		NumLayers: 43, IsMoE: true, NumExperts: 256,
		ExpertBytes: int64(43 * 3289 * 1024 * 1024), NonExpertBytes: int64(7680 * 1024 * 1024),
		ContextSize: 32768, EmbeddingLength: 4096, HeadCountKV: 1, KeyLength: 512, ValueLength: 512,
		ExpertUsedCount: 6, ExpertFF: 2048, ModelArch: "llama",
	}
	be := &backendInfo{Tag: "llama", Identity: "b", Path: "/p"}
	newReq := func() *launchRequest {
		return &launchRequest{Port: 8081, Calibrate: calibrateAuto, CtxFlag: "8192", KVQuality: "low", KVPlacement: "cpu", Parallel: 1}
	}
	return cfg, model, be, caps, newReq
}

func reusePinCompute(t *testing.T, req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities) *placement.Strategy {
	t.Helper()
	strategy, err := placement.Compute(caps, model, placementOptionsFromRequestCaps(req, model, be, cfg.CacheDir, caps))
	if err != nil || strategy == nil {
		t.Fatalf("compute: %v", err)
	}
	return strategy
}

// The good path. First launch: no saved state, calibrationPlan schedules the
// bounded search; the baseline wins with complete evidence on an active
// profile, so the decision (agent-workflow) and the verified config are saved.
// Identical relaunch with a fresh request: placement.Compute replays the
// verified config and calibrationPlan returns nil, so one load and no
// challenger. If the verified config is missed (stale free VRAM, start-boundary
// re-plan) the decision alone still suppresses the search for the same
// baseline.
func TestCalibrationPaidOnce_CalibratedLaunchThenIdenticalRelaunchStartsDirect(t *testing.T) {
	cfg, model, be, caps, newReq := reusePinMoELaunch(t)
	req := newReq()
	first := reusePinCompute(t, req, cfg, model, be, caps)
	if first.VerifiedConfigReused {
		t.Fatal("an empty cache replayed a verified config")
	}
	plan := calibrationPlan(req, cfg, model, be, caps, first)
	if len(plan) < 2 {
		t.Errorf("precondition: the first launch schedules no search (%v), so nothing is paid", reusePinNames(plan))
	} else {
		decision := reusePinMeasuredDecision(t, calibrationScope(req, model, be, caps, first), model, plan)
		if !automaticCalibrationEvidenceValid(decision) {
			t.Fatalf("complete comparison was rejected: %+v", decision)
		}
		decision.ValidationLevel = placement.CalibrationValidationWorkflow
		if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, *decision); err != nil {
			t.Fatal(err)
		}
	}
	saveVerifiedConfigForLaunch(cfg, req, model, be, caps, first, nil)

	again := newReq()
	second := reusePinCompute(t, again, cfg, model, be, caps)
	if !second.VerifiedConfigReused {
		t.Fatal("the identical relaunch did not replay the verified config")
	}
	if got := calibrationPlan(again, cfg, model, be, caps, second); len(got) != 0 {
		t.Errorf("the verified relaunch scheduled calibration loads: %v", reusePinNames(got))
	}
	if len(plan) >= 2 {
		if got := calibrationPlan(again, cfg, model, be, caps, first); len(got) != 0 {
			t.Errorf("without the verified replay the saved decision did not suppress the search: %v", reusePinNames(got))
		}
	}
}

// After a release that bumps CalibrationSchemaVersion, a scope that still has
// a verified config is not re-searched: the replay sets VerifiedConfigReused
// and calibrationPlan returns nil, although PerformanceTuned is dropped and
// the decision no longer loads. The cost of the bump falls only on launches
// that miss the verified config (degraded profiles never have one). This also
// pins that the verified record's "cannot suppress a newer optimizer" intent
// (PerformanceEvidenceSchema) does not reach calibrationPlan.
func TestCalibrationPlan_SchemaBumpWithVerifiedConfigStartsDirect(t *testing.T) {
	cfg, model, be, caps, newReq := reusePinMoELaunch(t)
	req := newReq()
	first := reusePinCompute(t, req, cfg, model, be, caps)
	plan := calibrationPlan(req, cfg, model, be, caps, first)
	if len(plan) >= 2 {
		decision := reusePinMeasuredDecision(t, calibrationScope(req, model, be, caps, first), model, plan)
		decision.ValidationLevel = placement.CalibrationValidationWorkflow
		decision.SchemaVersion = placement.CalibrationSchemaVersion - 2
		if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, *decision); err != nil {
			t.Fatal(err)
		}
	} else {
		t.Errorf("precondition: the first launch schedules no search (%v)", reusePinNames(plan))
	}
	tuned := *first
	tuned.PerformanceTuned = true
	record := placement.VerifiedConfigToRecord(verifiedConfigScopeKey(req, model, be, caps),
		filepath.Base(model.Path), &tuned, be.Identity, be.Path, "", "")
	record.PerformanceEvidenceSchema = placement.CalibrationSchemaVersion - 2
	if _, err := placement.SaveVerifiedConfig(cfg.CacheDir, record); err != nil {
		t.Fatal(err)
	}

	again := newReq()
	second := reusePinCompute(t, again, cfg, model, be, caps)
	if !second.VerifiedConfigReused {
		t.Fatal("a calibration-schema bump retired the verified config")
	}
	if second.PerformanceTuned {
		t.Error("older performance evidence was restored as current")
	}
	if got := calibrationPlan(again, cfg, model, be, caps, second); len(got) != 0 {
		t.Errorf("the verified relaunch after a schema bump scheduled calibration loads: %v", reusePinNames(got))
	}
	if len(plan) >= 2 {
		if got := calibrationPlan(again, cfg, model, be, caps, first); len(got) < 2 {
			t.Errorf("without the verified replay the older-schema decision still suppressed the search: %v", reusePinNames(got))
		}
	}
}
