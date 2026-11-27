package main

// Pins for "the calibration cost is paid once": what a saved decision and a
// verified config suppress on the next identical launch, and where today's
// policy pays the search again. A test named ..._KnownCost asserts today's
// behaviour and says what must flip if that cost is removed.

import (
	"fmt"
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
// default, annotateOptimizationDecision over the measured order, then
// recordCalculatedFinalist with the plan's calculated finalist.
func reusePinMeasuredDecision(t *testing.T, scope placement.CalibrationScopeKey, model *placement.ModelProfile, calculated placement.CalibrationCandidate, measured []placement.CalibrationCandidate) *placement.CalibrationDecision {
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
	recordCalculatedFinalist(decision, calculated.Name, placement.CalibrationStrategyFingerprint(calculated.Strategy), measured[1].Name)
	if decision.Winner != "default" || decision.Finalist != measured[1].Name || decision.FinalistOutcome != "baseline-won" {
		t.Fatalf("decision does not record the measured finalist: winner=%q finalist=%q outcome=%q",
			decision.Winner, decision.Finalist, decision.FinalistOutcome)
	}
	return decision
}

// reusePinDegradedDecision is the save in the launch's degraded-profile block
// (main.go, "baseline kept for ... identical launches skip this search"):
// degradedBaselinePersistable, then ValidationLevel baseline-bounded.
func reusePinDegradedDecision(t *testing.T, scope placement.CalibrationScopeKey, model *placement.ModelProfile, calculated placement.CalibrationCandidate, measured []placement.CalibrationCandidate) placement.CalibrationDecision {
	t.Helper()
	decision := reusePinMeasuredDecision(t, scope, model, calculated, measured)
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
// cheapest estimate. The decision names the directed finalist it measured and
// records the calculated one, so it covers the next plan with that calculated
// finalist and no other.
func TestPhaseDirectedFinalistDecisionCoversTheCalculatedPlan(t *testing.T) {
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
	decision := reusePinDegradedDecision(t, placement.CalibrationScopeKey{ModelIdentity: "m"}, &placement.ModelProfile{Path: "m.gguf"}, calculated[1], measured)
	if decision.Finalist != directed.Name || decision.CalculatedFinalist != calculated[1].Name {
		t.Errorf("saved finalist = %q calculated = %q, want measured %q and calculated %q",
			decision.Finalist, decision.CalculatedFinalist, directed.Name, calculated[1].Name)
	}
	if !decision.SuppressesAutomaticAdmissionRetry(directed.Name) {
		t.Error("the decision does not even suppress the finalist it measured")
	}
	fingerprint := placement.CalibrationStrategyFingerprint
	if !decision.SuppressesAutomaticSearch(calculated[1].Name, fingerprint(calculated[1].Strategy)) {
		t.Error("the decision does not cover the plan it searched")
	}
	if decision.SuppressesAutomaticSearch(directed.Name, fingerprint(directed.Strategy)) {
		t.Error("the decision covers a plan whose calculated finalist is the directed one")
	}
	// Same name, different strategy: a recomputed candidate (expert-pack and
	// the like) is a different experiment and is searched again.
	if decision.SuppressesAutomaticSearch(calculated[1].Name, fingerprint(roomy(512, 512))) {
		t.Error("the decision covers a calculated finalist of the same name but another strategy")
	}
	if decision.SuppressesAutomaticSearch(calculated[1].Name, "") {
		t.Error("the decision covers a calculated finalist without a strategy")
	}
	// A record written before the field existed keeps the older comparison.
	legacy := decision
	legacy.CalculatedFinalist = ""
	legacy.CalculatedFingerprint = ""
	if legacy.SuppressesAutomaticSearch(calculated[1].Name, fingerprint(calculated[1].Strategy)) ||
		!legacy.SuppressesAutomaticSearch(directed.Name, fingerprint(directed.Strategy)) {
		t.Error("a record without calculated_finalist no longer compares against its finalist")
	}
}

// End to end at the policy level: the degraded-profile launch saves a
// baseline-bounded decision naming the finalist it measured; the next
// identical launch calls calibrationPlan for the same request, model, hardware
// and baseline (same scope key) and plans nothing, whether the measured
// finalist was the calculated one or a directed replacement. A record without
// calculated_finalist behaves as before; a decision recorded for another
// calculated finalist or another scope never suppresses the search.
func TestCalibrationPlan_DegradedProfileSkipsIdenticalSearch(t *testing.T) {
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
	redirected := prioritizeCalibrationFinalist(first, directed)

	save := func(d placement.CalibrationDecision) {
		t.Helper()
		if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, d); err != nil {
			t.Fatal(err)
		}
	}
	searches := func(label string) {
		t.Helper()
		got := calibrationPlan(req, cfg, model, be, caps, strategy)
		if len(got) < 2 {
			t.Errorf("%s: the search was suppressed: %v", label, reusePinNames(got))
		} else if got[1].Name != calculated {
			t.Errorf("%s: plan finalist = %q, want the calculated %q", label, got[1].Name, calculated)
		}
	}

	// Control: the measured finalist is the calculated one.
	save(reusePinDegradedDecision(t, scope, model, first[1], first))
	if got := calibrationPlan(req, cfg, model, be, caps, strategy); len(got) != 0 {
		t.Errorf("control: a bounded decision for the calculated finalist did not suppress the search: %v", reusePinNames(got))
	}

	// The measured finalist differs from the calculated one.
	bounded := reusePinDegradedDecision(t, scope, model, first[1], redirected)
	save(bounded)
	if loaded, err := placement.LoadCalibrationDecision(cfg.CacheDir, scope.String()); err != nil ||
		loaded.Finalist != directed.Name || loaded.CalculatedFinalist != calculated {
		t.Fatalf("the bounded decision is not the one loaded for this scope: %+v err=%v", loaded, err)
	}
	if got := calibrationPlan(req, cfg, model, be, caps, strategy); len(got) != 0 {
		t.Errorf("the identical launch re-searched after a directed finalist was measured: %v", reusePinNames(got))
	}

	// The same decision for another scope does not suppress that scope's
	// search. Only the backend identity changes, so the frontier is the same.
	other := *be
	other.Identity = be.Identity + "-rebuilt"
	if otherScope := calibrationScope(req, model, &other, caps, strategy); otherScope.String() == scope.String() {
		t.Fatal("a different backend identity did not change the scope key")
	}
	if got := calibrationPlan(req, cfg, model, &other, caps, strategy); len(got) < 2 {
		t.Errorf("a decision for scope X suppressed scope Y: %v", reusePinNames(got))
	}

	// A record written without calculated_finalist (main before this field)
	// keeps the older comparison: a directed finalist does not cover the plan.
	legacy := bounded
	legacy.CalculatedFinalist, legacy.CalculatedFingerprint = "", ""
	save(legacy)
	searches("record without calculated_finalist")

	// Same calculated name, another strategy (a recomputed expert-pack, for
	// example): a different experiment, so it is searched.
	shifted := bounded
	shifted.CalculatedFingerprint = placement.CalibrationStrategyFingerprint(directed.Strategy)
	save(shifted)
	searches("same calculated name, another strategy")

	// A decision recorded for a different calculated finalist (a changed
	// candidate set) still searches, even though its measured finalist is the
	// current calculated one: the two comparisons are never combined.
	save(reusePinDegradedDecision(t, scope, model, directed, first))
	searches("decision for another calculated finalist")
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
	decision := reusePinMeasuredDecision(t, scope, model, first[1], first)
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

// reusePinMoELaunch is the MoE geometry of TestComputeReusesVerifiedConfigForMoE
// (placement.Compute plans it), shrunk to about 60 GB so the experts fit host
// RAM and the launch is resident (the 146 GB original is non-resident, whose
// last-resort lane is never challenged). It is reached through the launch's
// own option builder, so the verified-config key is the one a real launch
// computes.
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
		Path: filepath.Join(dir, "moe.gguf"), SizeBytes: int64(43*1300+4096) << 20, TotalSizeMB: 43*1300 + 4096,
		NumLayers: 43, IsMoE: true, NumExperts: 256,
		ExpertBytes: int64(43*1300) << 20, NonExpertBytes: int64(4096) << 20,
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

// reusePinPlanShape says why a first launch planned no search.
func reusePinPlanShape(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, s *placement.Strategy) string {
	return fmt.Sprintf("type=%s residency=%s mmapRequired=%v ncpumoe=%d ctx=%d frontier=%v",
		s.Type, s.Residency, s.MMapRequired, s.NCPUMoE, s.ContextSize,
		reusePinNames(calibrationCandidates(req, cfg, model, be, caps, s)))
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
		t.Errorf("precondition: the first launch schedules no search (%v), so nothing is paid; %s",
			reusePinNames(plan), reusePinPlanShape(req, cfg, model, be, caps, first))
	} else {
		decision := reusePinMeasuredDecision(t, calibrationScope(req, model, be, caps, first), model, plan[1], plan)
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
		decision := reusePinMeasuredDecision(t, calibrationScope(req, model, be, caps, first), model, plan[1], plan)
		decision.ValidationLevel = placement.CalibrationValidationWorkflow
		decision.SchemaVersion = placement.CalibrationSchemaVersion - 2
		if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, *decision); err != nil {
			t.Fatal(err)
		}
	} else {
		t.Errorf("precondition: the first launch schedules no search (%v); %s",
			reusePinNames(plan), reusePinPlanShape(req, cfg, model, be, caps, first))
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
