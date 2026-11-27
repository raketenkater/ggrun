package main

// runCalibration-level tests for the calculated finalist a decision records:
// each path that builds a decision is driven with fake processes and a forced
// phase-directed swap, and the saved decision must name the finalist the plan
// calculated before the swap, so the next identical plan is recognised.

import (
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// newFinalistHarness is a budgetHarness on the reuse-pin MoE fixture: its
// candidates are calibrationPlan's automatic plan, and the phase-directed
// selector returns a frontier member other than the calculated finalist.
func newFinalistHarness(t *testing.T, cacheDir string, baselineLoad, baselineWorkload time.Duration) (*budgetHarness, placement.CalibrationCandidate) {
	t.Helper()
	h := newBudgetHarness(t, baselineLoad, baselineWorkload)
	req, cfg, model, be, caps, strategy := reusePinFixture(t)
	cfg.CacheDir = cacheDir
	h.req, h.cfg, h.model, h.be, h.caps, h.strategy = req, cfg, model, be, caps, strategy
	h.serverArgs = buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
	h.recovery = newLaunchMemoryRecovery()
	h.recovery.observeProductionLoad(h.serverArgs, baselineLoad)
	h.candidates = calibrationPlan(req, cfg, model, be, caps, strategy)
	if len(h.candidates) < 2 {
		t.Fatalf("fixture produced no automatic finalist: %v", reusePinNames(h.candidates))
	}
	var directed placement.CalibrationCandidate
	for _, candidate := range analyzedMeasuredCalibrationFrontier(req, cfg, model, be, caps, strategy, nil)[1:] {
		if candidate.Strategy != nil && candidate.Name != h.candidates[1].Name && candidate.Name != "default" {
			directed = candidate
			break
		}
	}
	if directed.Name == "" {
		t.Fatal("frontier offers no second challenger")
	}
	h.fx.directFinalist = func(*launchRequest, *config.Config, *placement.ModelProfile, *backendInfo, *detect.Capabilities, *placement.Strategy, placement.WorkloadBottleneck, *launchMemoryRecovery) (placement.CalibrationCandidate, bool) {
		return directed, true
	}
	return h, directed
}

// assertRecordsCalculated checks the decision names the measured (directed)
// finalist and records the calculated one, captured before the swap.
func assertRecordsCalculated(t *testing.T, h *budgetHarness, calculated placement.CalibrationCandidate, directed placement.CalibrationCandidate, d *placement.CalibrationDecision) {
	t.Helper()
	if d == nil {
		t.Fatalf("no decision returned; events %v", h.events)
	}
	if d.Finalist != directed.Name {
		t.Errorf("decision finalist = %q, want the directed %q", d.Finalist, directed.Name)
	}
	if d.CalculatedFinalist != calculated.Name || d.CalculatedFinalist == d.Finalist {
		t.Errorf("decision calculated finalist = %q, want %q captured before the swap (finalist %q)",
			d.CalculatedFinalist, calculated.Name, d.Finalist)
	}
	if want := placement.CalibrationStrategyFingerprint(calculated.Strategy); d.CalculatedFingerprint != want {
		t.Errorf("decision calculated fingerprint = %q, want the calculated strategy's %q", d.CalculatedFingerprint, want)
	}
}

func saveFinalistDecision(t *testing.T, h *budgetHarness, d placement.CalibrationDecision) {
	t.Helper()
	if _, err := placement.SaveCalibrationDecision(h.cfg.CacheDir, d); err != nil {
		t.Fatal(err)
	}
}

func planAgain(h *budgetHarness) []placement.CalibrationCandidate {
	return calibrationPlan(h.req, h.cfg, h.model, h.be, h.caps, h.strategy)
}

// Measured path: the directed finalist is loaded, measured and loses. On a
// degraded profile the bounded decision covers the next identical plan. On an
// active profile the agent-workflow decision suppresses the search with or
// without the new fields, as before.
func TestRunCalibrationMeasuredDirectedFinalistRecordsCalculated(t *testing.T) {
	h, directed := newFinalistHarness(t, t.TempDir(), 30*time.Second, time.Minute)
	calculated := h.candidates[1]
	_, _, _, d := h.run()
	assertRecordsCalculated(t, h, calculated, directed, d)
	if d == nil {
		return
	}
	if d.Winner != "default" || d.FinalistOutcome != "baseline-won" || h.count("candidate workload") != 1 {
		t.Fatalf("fixture did not measure the directed finalist once and keep the baseline: %+v events %v", d, h.events)
	}

	bounded := *d
	if !degradedBaselinePersistable(calibrateAuto, &bounded, true) {
		t.Fatalf("a functional degraded profile would not keep this decision: %+v", bounded)
	}
	bounded.ValidationLevel = placement.CalibrationValidationBaselineBounded
	saveFinalistDecision(t, h, bounded)
	if got := planAgain(h); len(got) != 0 {
		t.Errorf("degraded: the identical launch re-searched: %v", reusePinNames(got))
	}

	// Active-profile regression: unchanged behaviour.
	active := *d
	if !automaticCalibrationEvidenceValid(&active) {
		t.Fatalf("complete comparison was rejected: %+v", active)
	}
	active.ValidationLevel = placement.CalibrationValidationWorkflow
	saveFinalistDecision(t, h, active)
	if got := planAgain(h); len(got) != 0 {
		t.Errorf("active: a workflow decision did not suppress the search: %v", reusePinNames(got))
	}
	active.CalculatedFinalist, active.CalculatedFingerprint = "", ""
	saveFinalistDecision(t, h, active)
	if got := planAgain(h); len(got) != 0 {
		t.Errorf("active: a workflow decision without the fields did not suppress the search: %v", reusePinNames(got))
	}
}

// Unavailable path: every challenger is refused by the memory oracle while
// the baseline keeps serving. The admission-only decision names the directed
// finalist and records the calculated one.
func TestRunCalibrationUnavailableDirectedFinalistRecordsCalculated(t *testing.T) {
	h, directed := newFinalistHarness(t, t.TempDir(), 30*time.Second, time.Minute)
	h.fx.prescreen = func(*launchRequest, *config.Config, *placement.ModelProfile, *backendInfo, *detect.Capabilities, *detect.Capabilities, *placement.Strategy, []string) (bool, string, string) {
		h.events = append(h.events, "prescreen refused")
		return true, "oracle-refused", "the exact argv does not fit"
	}
	calculated := h.candidates[1]
	_, _, _, d := h.run()
	assertRecordsCalculated(t, h, calculated, directed, d)
	if d == nil {
		return
	}
	if d.FinalistOutcome != "unavailable" || h.count("start") != 0 {
		t.Fatalf("fixture did not refuse every challenger before a load: %+v events %v", d, h.events)
	}
	if !automaticCalibrationAdmissionEvidenceValid(d) {
		t.Fatalf("not admission evidence: %+v", d)
	}
	admission := *d
	admission.ValidationLevel = placement.CalibrationValidationAdmission
	saveFinalistDecision(t, h, admission)
	if got := planAgain(h); len(got) != 0 {
		t.Errorf("the identical launch re-searched after an admission-only decision: %v", reusePinNames(got))
	}
}

// Budget-bound path: the baseline measurement spends the elapsed budget, so
// the directed finalist is recorded as unmeasured. Below the attempt cap the
// next identical launch searches again; the second attempt counts against the
// same decision and then the search is suppressed.
func TestRunCalibrationBudgetBoundDirectedFinalistRecordsCalculated(t *testing.T) {
	dir := t.TempDir()
	persist := func(h *budgetHarness, d *placement.CalibrationDecision) {
		t.Helper()
		if !degradedBaselinePersistable(calibrateAuto, d, true) {
			t.Fatalf("a functional degraded profile would not keep this decision: %+v", d)
		}
		bounded := *d
		bounded.ValidationLevel = placement.CalibrationValidationBaselineBounded
		saveFinalistDecision(t, h, bounded)
	}

	h, directed := newFinalistHarness(t, dir, time.Minute, 21*time.Minute)
	calculated := h.candidates[1]
	_, _, _, d := h.run()
	assertRecordsCalculated(t, h, calculated, directed, d)
	if d == nil {
		return
	}
	if d.FinalistOutcome != placement.FinalistOutcomeUnmeasured || d.FinalistAttempts != 1 {
		t.Fatalf("fixture did not record one unmeasured attempt: %+v", d)
	}
	persist(h, d)
	if got := planAgain(h); len(got) < 2 {
		t.Fatalf("below the attempt cap the identical launch did not search: %v", reusePinNames(got))
	}

	again, _ := newFinalistHarness(t, dir, time.Minute, 21*time.Minute)
	_, _, _, d2 := again.run()
	assertRecordsCalculated(t, again, calculated, directed, d2)
	if d2 == nil {
		return
	}
	if d2.FinalistAttempts != placement.MaxUnmeasuredFinalistAttempts {
		t.Fatalf("second attempt = %d, want %d", d2.FinalistAttempts, placement.MaxUnmeasuredFinalistAttempts)
	}
	persist(again, d2)
	if got := planAgain(again); len(got) != 0 {
		t.Errorf("at the attempt cap the identical launch re-searched: %v", reusePinNames(got))
	}
}

// A primary finalist that fails admission while a fallback is measured and
// loses keeps the older comparison: the decision records no calculated
// finalist, so the unmeasured primary is not skipped on the next launch.
func TestRecordCalculatedFinalistSkipsAdmissionFallbackDecisions(t *testing.T) {
	d := &placement.CalibrationDecision{Winner: "default", Finalist: "fallback"}
	recordCalculatedFinalist(d, "primary", "fp", "primary")
	if d.CalculatedFinalist != "" || d.CalculatedFingerprint != "" {
		t.Errorf("a fallback decision recorded the calculated finalist: %+v", d)
	}
	recordCalculatedFinalist(d, "primary", "fp", "fallback")
	if d.CalculatedFinalist != "primary" || d.CalculatedFingerprint != "fp" {
		t.Errorf("a decision about the lead finalist did not record the calculated one: %+v", d)
	}
	for _, empty := range []*placement.CalibrationDecision{nil, {Winner: "default"}} {
		recordCalculatedFinalist(empty, "primary", "fp", "primary")
		if empty != nil && empty.CalculatedFinalist != "" {
			t.Errorf("a decision without a finalist recorded one: %+v", empty)
		}
	}
}

// The printed line promises a skipped search only where the next identical
// plan is suppressed.
func TestBoundedBaselineKeptLineIsTrue(t *testing.T) {
	covered := &placement.CalibrationDecision{Winner: "default", ValidationLevel: placement.CalibrationValidationBaselineBounded,
		Finalist: "directed", FinalistOutcome: "baseline-won", CalculatedFinalist: "calculated", CalculatedFingerprint: "fp"}
	if line := boundedBaselineKeptLine(covered, "cal.json"); !strings.Contains(line, "identical launches skip this search while the calculated finalist stays calculated") {
		t.Errorf("covered decision line = %q", line)
	}
	below := *covered
	below.FinalistOutcome = placement.FinalistOutcomeUnmeasured
	below.FinalistFailureClass, below.FinalistFailureReason, below.FinalistAttempts = "elapsed-budget", "reason", 1
	fallback := *covered
	fallback.CalculatedFinalist, fallback.CalculatedFingerprint = "", ""
	for name, d := range map[string]*placement.CalibrationDecision{"unmeasured below the cap": &below, "admission fallback": &fallback} {
		line := boundedBaselineKeptLine(d, "cal.json")
		if strings.Contains(line, "skip") || !strings.Contains(line, "the next identical launch searches again") {
			t.Errorf("%s: line = %q", name, line)
		}
	}
}
