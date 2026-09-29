package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/advisor"
	"github.com/raketenkater/ggrun/pkg/benchmark"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
)

// budgetHarness drives runCalibration through time with fake processes: the
// clock only moves when a fake effect says the step took that long.
type budgetHarness struct {
	t          *testing.T
	clock      *admissionClock
	start      time.Time
	req        *launchRequest
	cfg        *config.Config
	model      *placement.ModelProfile
	be         *backendInfo
	caps       *detect.Capabilities
	strategy   *placement.Strategy
	serverArgs []string
	recovery   *launchMemoryRecovery
	baseline   *server.Process
	candidates []placement.CalibrationCandidate

	baselineLoad time.Duration
	events       []string
	starts       []time.Duration // timeout given to each exact admission
	restarts     []harnessRestart

	fx calibrationSideEffects
}

type harnessRestart struct {
	args   []string
	window time.Duration
	at     time.Time
}

func completeAgentResult(scale float64) *benchmark.Result {
	return &benchmark.Result{
		Parallel: 1, GenTPS: 40 / scale, PromptTPS: 800 / scale, MixedGenTPS: 20 / scale,
		GenTokens: 128, PromptTokens: 4000, MixedGenTokens: 64, GenTimeS: 10 * scale,
		AgentSamples: 2, AgentTurnTimeS: 10 * scale, AgentTurnMaxS: 11 * scale,
		AgentScenarioTimeS: 30 * scale, AgentScenarioMaxS: 31 * scale, AgentPromptBytes: 4096,
		AgentCachedTokens: 1800, AgentNewPromptTokens: 80,
		AgentWorkloadLanes: 1, AgentWorkloadTimeS: 30 * scale, AgentWorkloadMaxS: 31 * scale,
	}
}

// newBudgetHarness builds an automatic 20-minute experiment whose baseline was
// observed to load in baselineLoad and whose workload takes baselineWorkload.
func newBudgetHarness(t *testing.T, baselineLoad, baselineWorkload time.Duration, extra ...string) *budgetHarness {
	t.Helper()
	req, cfg, model, be, caps := calibrateTestSetup(40000)
	cfg.CacheDir = t.TempDir()
	strategy := &placement.Strategy{Type: placement.SingleGPU, UBatchSize: 512, BatchSize: 512, Parallel: 1, Residency: placement.ResidencyTight}
	h := &budgetHarness{
		t: t, clock: &admissionClock{t: time.Unix(1_800_000_000, 0)},
		req: req, cfg: cfg, model: model, be: be, caps: caps, strategy: strategy,
		recovery: newLaunchMemoryRecovery(), baseline: &server.Process{}, baselineLoad: baselineLoad,
	}
	h.start = h.clock.now()
	h.serverArgs = buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
	h.recovery.observeProductionLoad(h.serverArgs, baselineLoad)
	h.candidates = []placement.CalibrationCandidate{{Name: "default", Strategy: strategy}}
	for i, name := range append([]string{"ubatch-1024"}, extra...) {
		ub := 1024 << i
		h.candidates = append(h.candidates, placement.CalibrationCandidate{Name: name,
			Strategy: &placement.Strategy{Type: placement.SingleGPU, UBatchSize: ub, BatchSize: ub, Parallel: 1, Residency: placement.ResidencyTight}})
	}
	workloads := 0
	h.fx = calibrationSideEffects{
		now: h.clock.now,
		probePrefill: func(*benchmark.Runner) (benchmark.PrefillProbe, error) {
			return benchmark.PrefillProbe{}, errors.New("no pilot")
		},
		oracleAvailable: func(*backendInfo, *placement.ModelProfile) bool { return true },
		prescreen: func(*launchRequest, *config.Config, *placement.ModelProfile, *backendInfo, *detect.Capabilities, *detect.Capabilities, *placement.Strategy, []string) (bool, string, string) {
			h.events = append(h.events, "prescreen")
			return false, "", ""
		},
		runWorkload: func(r *benchmark.Runner, slots, lanes int) (*benchmark.Result, error) {
			workloads++
			if workloads == 1 {
				h.events = append(h.events, "baseline workload")
				h.clock.advance(baselineWorkload)
				return completeAgentResult(1), nil
			}
			h.events = append(h.events, "candidate workload")
			h.clock.advance(time.Minute)
			return completeAgentResult(1.2), nil
		},
		stopAndWait: func(_ *server.Process, label string, _ *detect.Capabilities, _ time.Duration) bool {
			h.events = append(h.events, "stop "+label)
			h.clock.advance(calibrationReleaseCost)
			return true
		},
		startExact: func(_ *launchRequest, _ *config.Config, _ *placement.ModelProfile, s *placement.Strategy, _ *backendInfo, _ *detect.Capabilities, args []string, timeout time.Duration, _ *launchMemoryRecovery) (*server.Process, *placement.Strategy, []string, error) {
			h.events = append(h.events, "start")
			h.starts = append(h.starts, timeout)
			h.clock.advance(baselineLoad)
			return &server.Process{}, s, args, nil
		},
	}
	h.fx.restart = func(_ *launchRequest, _ *config.Config, _ *placement.ModelProfile, s *placement.Strategy, _ *backendInfo, _ *detect.Capabilities, args []string, window time.Duration, _ *launchMemoryRecovery) (*server.Process, *placement.Strategy, []string) {
		h.events = append(h.events, "restart")
		h.restarts = append(h.restarts, harnessRestart{args: args, window: window, at: h.clock.now()})
		h.clock.advance(baselineLoad)
		return &server.Process{}, s, args
	}
	return h
}

func (h *budgetHarness) run() (*server.Process, *placement.Strategy, []string, *placement.CalibrationDecision) {
	h.t.Helper()
	savedFx, savedVRAM := calibrationEffects, runtimeVRAMUsedMB
	calibrationEffects = h.fx
	runtimeVRAMUsedMB = func(int) int { return 0 }
	defer func() { calibrationEffects, runtimeVRAMUsedMB = savedFx, savedVRAM }()
	return runCalibration(h.req, h.cfg, h.model, h.be, h.caps, h.strategy, h.serverArgs, time.Hour,
		h.baseline, h.recovery, h.caps, h.candidates)
}

func (h *budgetHarness) deadline() time.Time {
	return h.start.Add(calibrationBudgetFor(effectiveCalibrationMode(h.req)).MaxElapsed)
}

// restoreFloor is the baseline's priced restoration: its oracle admission and
// its load. A winner restart must also leave room for its own release.
func (h *budgetHarness) restoreFloor() time.Duration {
	return oracleAdmissionCost + h.cost(0).Restore
}

func (h *budgetHarness) emergencyDeadline() time.Time {
	return h.deadline().Add(restartAdmissionWindow(h.model))
}

// cost is what the controller prices for the one challenger, computed with the
// production estimator so a change of constants does not break the tests.
func (h *budgetHarness) cost(baselineWorkload time.Duration) challengerCost {
	return estimateChallengerCost(challengerCostInputs{
		BaselineLoad: h.baselineLoad, BaselineLoadObserved: true, BaselineWorkload: baselineWorkload,
		Ceiling: autoStartupTimeout(h.model), BaselineRunning: true,
	})
}

func (h *budgetHarness) ledger(kind string) []launchWorkRecord {
	h.t.Helper()
	data, err := os.ReadFile(launchWorkLedgerPath(h.cfg.CacheDir))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		h.t.Fatal(err)
	}
	var out []launchWorkRecord
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var rec launchWorkRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			h.t.Fatal(err)
		}
		if rec.Kind == kind {
			out = append(out, rec)
		}
	}
	return out
}

func (h *budgetHarness) count(event string) int {
	n := 0
	for _, e := range h.events {
		if strings.HasPrefix(e, event) {
			n++
		}
	}
	return n
}

func TestCalibrationExperimentReserveIsNotSpendable(t *testing.T) {
	clock := &admissionClock{t: time.Unix(1_800_000_000, 0)}
	exp := newCalibrationExperiment(20*time.Minute, 16*time.Minute, clock.now)
	clock.advance(10 * time.Minute)
	if got := exp.until(4 * time.Minute); got != 6*time.Minute {
		t.Fatalf("until(reserve)=%s, want 6m", got)
	}
	clock.advance(7 * time.Minute)
	if got := exp.until(4 * time.Minute); got > 0 {
		t.Fatalf("the reserve was spendable: until=%s", got)
	}
	clock.advance(5 * time.Minute) // 2m past the optimization deadline
	if got := exp.overrun(); got != 2*time.Minute {
		t.Fatalf("overrun=%s, want 2m", got)
	}
	// Restoration shares the fixed emergency deadline; it is never a fresh window.
	if got := exp.serviceWindow(16*time.Minute, 0, 0); got != 14*time.Minute {
		t.Fatalf("baseline restoration window=%s, want the 14m left of the emergency window", got)
	}
	// A winner restart must leave the baseline's restoration inside that window.
	if got := exp.serviceWindow(16*time.Minute, 3*time.Minute, 0); got != 11*time.Minute {
		t.Fatalf("winner restart window=%s, want 11m", got)
	}
	// Past the emergency deadline the baseline still gets its priced restoration,
	// capped at one restart window; a winner gets nothing.
	clock.advance(20 * time.Minute)
	if got := exp.serviceWindow(16*time.Minute, 0, 3*time.Minute); got != 3*time.Minute {
		t.Fatalf("exhausted emergency window gave the baseline %s, want its 3m restoration", got)
	}
	if got := exp.serviceWindow(16*time.Minute, 3*time.Minute, 0); got >= minUsefulLoadWindow {
		t.Fatalf("exhausted emergency window still admitted a winner restart of %s", got)
	}
	if got := exp.serviceWindow(2*time.Minute, 0, 3*time.Minute); got != 2*time.Minute {
		t.Fatalf("restoration floor exceeded one restart window: %s", got)
	}
}

func TestRestorationReserveCoversCandidateReleaseAndRestore(t *testing.T) {
	cost := estimateChallengerCost(challengerCostInputs{
		BaselineLoad: 11*time.Minute + 30*time.Second, BaselineLoadObserved: true,
		BaselineWorkload: 3 * time.Minute, Ceiling: 30 * time.Minute, BaselineRunning: true,
	})
	if cost.Reserve() != calibrationReleaseCost+cost.Restore {
		t.Fatalf("reserve %s is not the candidate release plus restoration (%s)", cost.Reserve(), cost)
	}
	// The baseline's release is paid before the candidate starts, so it is in
	// the total but outside the reserve.
	if cost.Total()-cost.Reserve() < calibrationReleaseCost+cost.Admission+cost.Load+cost.Workload {
		t.Fatalf("total does not cover admission, load, workload and the baseline release outside the reserve: %s", cost)
	}
	// A release is priced at its real bound: Process.Stop's exit and scope
	// waits plus the resource wait.
	if calibrationReleaseCost < 60*time.Second {
		t.Fatalf("release priced at %s, below its 60s bound", calibrationReleaseCost)
	}
}

// Before the fix, affordability used the remaining time read before the oracle
// prescreen; the prescreen's own time was only noticed after the healthy
// baseline had been stopped.
func TestPrescreenTimeConsumesTheExperimentBudget(t *testing.T) {
	const baselineWorkload = 2 * time.Minute
	for _, tc := range []struct {
		name      string
		prescreen time.Duration
		wantStop  bool
	}{
		{"slow prescreen keeps the baseline", 55 * time.Second, false},
		{"instant prescreen control", 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newBudgetHarness(t, 5*time.Minute, baselineWorkload)
			cost := h.cost(baselineWorkload)
			stale := calibrationBudgetFor(calibrateAuto).MaxElapsed - baselineWorkload
			if !cost.Affordable(stale) || cost.Affordable(stale-55*time.Second) {
				t.Fatalf("fixture must fit before the prescreen and not after it: %s, stale remaining %s", cost, stale)
			}
			prescreen := h.fx.prescreen
			h.fx.prescreen = func(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps, base *detect.Capabilities, s *placement.Strategy, args []string) (bool, string, string) {
				h.clock.advance(tc.prescreen)
				return prescreen(req, cfg, model, be, caps, base, s, args)
			}
			p, _, args, decision := h.run()
			stops := h.count("stop default before")
			if !tc.wantStop {
				if stops != 0 || len(h.starts) != 0 || len(h.restarts) != 0 {
					t.Fatalf("baseline was disturbed after the prescreen spent the margin: %v", h.events)
				}
				if p != h.baseline || formatCommand(args) != formatCommand(h.serverArgs) {
					t.Fatal("the serving baseline was not returned")
				}
				if decision == nil || decision.FinalistFailureClass != "restore-budget" || decision.FinalistAttempts != 1 {
					t.Fatalf("refusal was not recorded as a bounded restore-budget attempt: %+v", decision)
				}
				if !strings.Contains(decision.FinalistFailureReason, "17m5s remains") {
					t.Fatalf("refusal did not report the refreshed remaining time: %q", decision.FinalistFailureReason)
				}
				if refusals := h.ledger("budget-refusal"); len(refusals) != 1 || refusals[0].Outcome != "not-started" || refusals[0].LoadedWeights {
					t.Fatalf("budget refusal ledger=%+v", refusals)
				}
				return
			}
			if stops != 1 {
				t.Fatalf("control did not run the experiment: %v", h.events)
			}
		})
	}
}

// The candidate workload runs sequential waves; a per-request timeout derived
// from the whole remaining budget let it run about three times that budget.
// Model the real runner: without a deadline every wave takes its full timeout.
func TestSlowWorkloadLeavesTheRestorationReserve(t *testing.T) {
	const baselineWorkload = time.Minute
	h := newBudgetHarness(t, time.Minute, baselineWorkload, "ubatch-2048")
	cost := h.cost(baselineWorkload)
	baselineRun := h.fx.runWorkload
	calls := 0
	var candidateDeadline time.Time
	h.fx.runWorkload = func(r *benchmark.Runner, slots, lanes int) (*benchmark.Result, error) {
		calls++
		if calls == 1 {
			return baselineRun(r, slots, lanes)
		}
		candidateDeadline = r.Deadline
		elapsed := 9 * r.Timeout
		if !r.Deadline.IsZero() && r.Deadline.Sub(h.clock.now()) < elapsed {
			h.clock.t = r.Deadline
			return nil, fmt.Errorf("agent sample 2: %w", benchmark.ErrDeadline)
		}
		h.clock.advance(elapsed)
		return nil, errors.New("agent sample 2: request timeout")
	}
	p, s, args, decision := h.run()
	if candidateDeadline.IsZero() || candidateDeadline.After(h.deadline().Add(-cost.Reserve())) {
		t.Fatalf("candidate workload deadline %v reaches into the %s reserve before %v", candidateDeadline, cost.Reserve(), h.deadline())
	}
	if end := h.clock.now(); end.After(h.deadline()) {
		t.Fatalf("experiment ended %s past its budget: %v", end.Sub(h.deadline()), h.events)
	}
	if len(h.starts) != 1 {
		t.Fatalf("a workload cut off by the deadline must end the search, got %d starts: %v", len(h.starts), h.events)
	}
	if len(h.restarts) != 1 || formatCommand(h.restarts[0].args) != formatCommand(h.serverArgs) || h.restarts[0].window < cost.Restore {
		t.Fatalf("baseline restoration=%+v, want one restart of the baseline with at least %s", h.restarts, cost.Restore)
	}
	if p == nil || p == h.baseline || s != h.strategy || formatCommand(args) != formatCommand(h.serverArgs) {
		t.Fatal("restored baseline is not what is served")
	}
	if decision == nil || decision.FinalistFailureClass != "elapsed-budget" || decision.FinalistAttempts != 1 {
		t.Fatalf("workload cut-off not recorded as one budget-bound attempt: %+v", decision)
	}
	if n := len(h.ledger("budget-refusal")); n != 0 {
		t.Fatalf("a workload cut-off was misreported as %d restore-budget refusal(s)", n)
	}
	if n := len(h.ledger("emergency-restore")); n != 0 {
		t.Fatalf("restoration needed the emergency window: %d records", n)
	}
}

// Regression guard: a challenger admission that uses its entire window still
// leaves the workload and the restoration reserve.
func TestFailedChallengerRestoresWithinTheReserve(t *testing.T) {
	const baselineWorkload = time.Minute
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"memory refusal", exactAdmissionError(exactAdmissionMemory, " on CUDA1", nil)},
		{"admission budget", &admissionBudgetError{Phase: "challenger", Reason: "window spent"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newBudgetHarness(t, time.Minute, baselineWorkload)
			cost := h.cost(baselineWorkload)
			var startedAt time.Time
			h.fx.startExact = func(_ *launchRequest, _ *config.Config, _ *placement.ModelProfile, _ *placement.Strategy, _ *backendInfo, _ *detect.Capabilities, _ []string, timeout time.Duration, _ *launchMemoryRecovery) (*server.Process, *placement.Strategy, []string, error) {
				startedAt = h.clock.now()
				h.starts = append(h.starts, timeout)
				h.clock.advance(timeout)
				return nil, nil, nil, tc.err
			}
			p, s, _, _ := h.run()
			if limit := h.deadline().Sub(startedAt) - cost.Workload - cost.Reserve(); h.starts[0] > limit {
				t.Fatalf("challenger admission got %s, more than the %s left after workload and reserve", h.starts[0], limit)
			}
			if len(h.restarts) != 1 || formatCommand(h.restarts[0].args) != formatCommand(h.serverArgs) {
				t.Fatalf("baseline restoration=%+v", h.restarts)
			}
			if end := h.clock.now(); end.After(h.deadline()) {
				t.Fatalf("restoration ended %s past the budget", end.Sub(h.deadline()))
			}
			if p == nil || s != h.strategy {
				t.Fatal("baseline was not restored")
			}
			if n := len(h.ledger("emergency-restore")); n != 0 {
				t.Fatalf("restoration needed the emergency window: %d records", n)
			}
		})
	}
}

// Winner and fallback restarts used to receive a fresh restart window each, so
// they stacked past any budget. Both now share one emergency deadline, the
// winner leaves the baseline's restoration inside it, the overrun is reported,
// and a winner that cannot be restarted is not recorded as baseline-won.
func TestWinnerRestartFailureUsesOneReportedEmergencyWindow(t *testing.T) {
	for _, tc := range []struct {
		name        string
		baselineUp  bool
		wantOutcome string
	}{
		{"baseline restored", true, "ready"},
		{"baseline also fails", false, "failed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newBudgetHarness(t, time.Minute, time.Minute)
			restoreFloor := h.restoreFloor() + calibrationReleaseCost
			fast := h.fx.runWorkload
			calls := 0
			h.fx.runWorkload = func(r *benchmark.Runner, slots, lanes int) (*benchmark.Result, error) {
				calls++
				if calls == 1 {
					return fast(r, slots, lanes)
				}
				h.clock.advance(time.Minute)
				return completeAgentResult(0.5), nil // the finalist wins
			}
			h.fx.restart = func(_ *launchRequest, _ *config.Config, _ *placement.ModelProfile, s *placement.Strategy, _ *backendInfo, _ *detect.Capabilities, args []string, window time.Duration, _ *launchMemoryRecovery) (*server.Process, *placement.Strategy, []string) {
				h.restarts = append(h.restarts, harnessRestart{args: args, window: window, at: h.clock.now()})
				if len(h.restarts) == 1 {
					// The winner's restart times out, ending past the budget.
					if h.clock.now().Add(window).Add(restoreFloor).After(h.emergencyDeadline()) {
						t.Fatalf("winner window %s does not leave the %s baseline restoration before the emergency deadline", window, restoreFloor)
					}
					h.clock.advance(window)
					return nil, nil, nil
				}
				if h.clock.now().Add(window).After(h.emergencyDeadline()) {
					t.Fatalf("baseline window %s runs past the shared emergency deadline", window)
				}
				h.clock.advance(time.Minute)
				if !tc.baselineUp {
					return nil, nil, nil
				}
				return &server.Process{}, s, args
			}
			p, s, args, decision := h.run()
			if len(h.restarts) != 2 {
				t.Fatalf("restarts=%d, want the winner then the baseline", len(h.restarts))
			}
			if formatCommand(h.restarts[0].args) == formatCommand(h.serverArgs) || formatCommand(h.restarts[1].args) != formatCommand(h.serverArgs) {
				t.Fatal("restart order is not winner then measured baseline")
			}
			if !tc.baselineUp {
				if p != nil || decision != nil {
					t.Fatalf("failed restoration returned process=%v decision=%+v", p, decision)
				}
			} else {
				if p == nil || s != h.strategy || formatCommand(args) != formatCommand(h.serverArgs) {
					t.Fatal("restored baseline is not what is served")
				}
				if decision == nil || decision.FinalistOutcome == "baseline-won" || automaticCalibrationEvidenceValid(decision) ||
					decision.FinalistFailureClass != "winner-restart" || decision.FinalistAttempts != 1 {
					t.Fatalf("an unrestartable winner became comparative evidence: %+v", decision)
				}
			}
			records := h.ledger("emergency-restore")
			if len(records) == 0 {
				t.Fatal("the overrun of the optimization budget was not recorded")
			}
			last := records[len(records)-1]
			if last.Outcome != tc.wantOutcome || last.LoadedWeights || last.Phase != "restore" ||
				last.ElapsedSec != 60 || last.TimeoutSec != h.restarts[1].window.Seconds() ||
				!strings.Contains(last.Reason, "optimization budget") {
				t.Fatalf("emergency record=%+v", last)
			}
		})
	}
}

// Stopping a serving baseline's measurement early protects nothing: it keeps
// serving either way. A slow baseline stays measured and is recorded as one
// bounded attempt instead of repeating the full measurement every launch.
func TestSlowBaselineKeepsServing(t *testing.T) {
	h := newBudgetHarness(t, time.Minute, 21*time.Minute)
	var baselineDeadline time.Time
	run := h.fx.runWorkload
	h.fx.runWorkload = func(r *benchmark.Runner, slots, lanes int) (*benchmark.Result, error) {
		baselineDeadline = r.Deadline
		return run(r, slots, lanes)
	}
	p, _, _, decision := h.run()
	if !baselineDeadline.IsZero() {
		t.Fatal("the serving baseline's measurement was given a deadline")
	}
	if p != h.baseline || h.count("stop") != 0 || len(h.starts) != 0 || len(h.restarts) != 0 {
		t.Fatalf("slow baseline was disturbed: %v", h.events)
	}
	if decision == nil || decision.FinalistAttempts != 1 || decision.FinalistFailureClass != "elapsed-budget" {
		t.Fatalf("slow baseline search not recorded as one bounded attempt: %+v", decision)
	}
}

// winsAfter makes the finalist win and moves the clock to at when the measured
// candidate is released, so a restart begins late in the experiment.
func (h *budgetHarness) winsAfter(at func() time.Time) {
	run := h.fx.runWorkload
	calls := 0
	h.fx.runWorkload = func(r *benchmark.Runner, slots, lanes int) (*benchmark.Result, error) {
		calls++
		if calls == 1 {
			return run(r, slots, lanes)
		}
		h.clock.advance(time.Minute)
		return completeAgentResult(0.5), nil
	}
	stop := h.fx.stopAndWait
	h.fx.stopAndWait = func(p *server.Process, label string, base *detect.Capabilities, timeout time.Duration) bool {
		ok := stop(p, label, base, timeout)
		if strings.HasPrefix(label, "measured candidate") {
			h.clock.t = at()
		}
		return ok
	}
}

// When a slow release pushes the winner restart late, the emergency deadline,
// not the restart cap, binds: the winner must leave the baseline's priced
// restoration and the failed winner's release inside it.
func TestLateWinnerRestartLeavesTheBaselineRestoration(t *testing.T) {
	h := newBudgetHarness(t, time.Minute, time.Minute)
	h.winsAfter(func() time.Time { return h.deadline().Add(8 * time.Minute) })
	keep := h.restoreFloor() + calibrationReleaseCost
	h.fx.restart = func(_ *launchRequest, _ *config.Config, _ *placement.ModelProfile, s *placement.Strategy, _ *backendInfo, _ *detect.Capabilities, args []string, window time.Duration, _ *launchMemoryRecovery) (*server.Process, *placement.Strategy, []string) {
		h.restarts = append(h.restarts, harnessRestart{args: args, window: window, at: h.clock.now()})
		h.clock.advance(window)
		if len(h.restarts) == 1 {
			return nil, nil, nil
		}
		return &server.Process{}, s, args
	}
	h.run()
	if len(h.restarts) != 2 {
		t.Fatalf("restarts=%+v, want the winner then the baseline", h.restarts)
	}
	winner := h.restarts[0]
	if winner.window >= restartAdmissionWindow(h.model) {
		t.Fatalf("fixture must make the emergency deadline bind, winner window %s", winner.window)
	}
	if winner.at.Add(winner.window).Add(keep).After(h.emergencyDeadline()) {
		t.Fatalf("winner window %s leaves no room for the %s baseline restoration", winner.window, keep)
	}
	if h.restarts[1].window < h.restoreFloor() {
		t.Fatalf("baseline got %s, less than its %s priced restoration", h.restarts[1].window, h.restoreFloor())
	}
}

// Availability over timer: once bounded process stops have consumed the whole
// emergency window, the measured baseline still receives its priced
// restoration, reported as an overrun, instead of leaving no server.
func TestBaselineRestoredAfterTheEmergencyWindowIsSpent(t *testing.T) {
	t.Run("after a measured winner", func(t *testing.T) {
		h := newBudgetHarness(t, time.Minute, time.Minute)
		h.winsAfter(func() time.Time { return h.emergencyDeadline().Add(time.Minute) })
		p, s, args, decision := h.run()
		if len(h.restarts) != 1 || formatCommand(h.restarts[0].args) != formatCommand(h.serverArgs) {
			t.Fatalf("restarts=%+v, want only the baseline", h.restarts)
		}
		if want := min(h.restoreFloor(), restartAdmissionWindow(h.model)); h.restarts[0].window != want {
			t.Fatalf("baseline window=%s, want its priced restoration %s", h.restarts[0].window, want)
		}
		if p == nil || s != h.strategy || formatCommand(args) != formatCommand(h.serverArgs) {
			t.Fatal("baseline was not restored")
		}
		if decision == nil || decision.FinalistFailureClass != "winner-restart" {
			t.Fatalf("skipped winner not recorded as a bounded attempt: %+v", decision)
		}
		if n := len(h.ledger("emergency-restore")); n != 1 {
			t.Fatalf("emergency-restore records=%d, want 1", n)
		}
	})
	t.Run("after a failed measurement", func(t *testing.T) {
		h := newBudgetHarness(t, time.Minute, time.Minute)
		run := h.fx.runWorkload
		calls := 0
		h.fx.runWorkload = func(r *benchmark.Runner, slots, lanes int) (*benchmark.Result, error) {
			calls++
			if calls == 1 {
				return run(r, slots, lanes)
			}
			return nil, errors.New("HTTP 500")
		}
		stop := h.fx.stopAndWait
		h.fx.stopAndWait = func(p *server.Process, label string, base *detect.Capabilities, timeout time.Duration) bool {
			ok := stop(p, label, base, timeout)
			if strings.Contains(label, "after failed measurement") {
				h.clock.t = h.emergencyDeadline().Add(time.Minute)
			}
			return ok
		}
		p, s, _, _ := h.run()
		if len(h.restarts) != 1 || h.restarts[0].window != min(h.restoreFloor(), restartAdmissionWindow(h.model)) {
			t.Fatalf("restarts=%+v, want one baseline restoration of %s", h.restarts, h.restoreFloor())
		}
		if p == nil || s != h.strategy {
			t.Fatal("baseline was not restored")
		}
	})
}

// The support optimizer runs only inside the optimization budget, and only
// when its whole lifecycle fits: the helper's start and release ignore its
// context.
func TestSupportOptimizerIsBoundedByTheOptimizationBudget(t *testing.T) {
	saved := runSupportIncidentFn
	defer func() { runSupportIncidentFn = saved }()
	for _, tc := range []struct {
		name     string
		leftover time.Duration // optimization time left when the advisor would start
		wantCall bool
	}{
		{"budget left", 30 * time.Minute, true},
		{"budget too short", calibrationAdvisorBound - time.Second, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newBudgetHarness(t, time.Minute, time.Minute)
			h.req.Calibrate = calibrateOn
			h.req.SupportExpert = "on"
			h.winsAfter(func() time.Time { return h.deadline().Add(-tc.leftover) })
			called := false
			var until, remaining time.Duration
			runSupportIncidentFn = func(ctx context.Context, _ *config.Config, _ *detect.Capabilities, _ string, _ advisor.Incident, _ bool) (advisor.Decision, advisor.RunReport, error) {
				called = true
				deadline, ok := ctx.Deadline()
				if !ok {
					t.Fatal("support optimizer context has no deadline")
				}
				until, remaining = time.Until(deadline), h.deadline().Sub(h.clock.now())
				return advisor.Decision{}, advisor.RunReport{}, errors.New("stubbed")
			}
			h.run()
			if called != tc.wantCall {
				t.Fatalf("support optimizer called=%v, want %v", called, tc.wantCall)
			}
			if called && (until > remaining || until < remaining-10*time.Second) {
				t.Fatalf("support optimizer deadline in %s, want the %s optimization time left", until, remaining)
			}
		})
	}
}

// A stop whose resource wait timed out must not hand back an exited process
// as the server: the TUI cell printed "Server running" over nothing when a
// concurrent download kept memory from settling. The measured baseline is
// restored instead, at either stop site.
func TestExitedProcessAfterFailedReleaseRestoresTheBaseline(t *testing.T) {
	for _, label := range []string{"default before", "measured candidate"} {
		t.Run(label, func(t *testing.T) {
			h := newBudgetHarness(t, time.Minute, time.Minute)
			stop := h.fx.stopAndWait
			h.fx.stopAndWait = func(p *server.Process, l string, base *detect.Capabilities, timeout time.Duration) bool {
				stop(p, l, base, timeout)
				return !strings.HasPrefix(l, label) // this release never settles; the process has exited
			}
			p, s, args, _ := h.run()
			if p == nil || p == h.baseline {
				t.Fatalf("returned %v, want a restored baseline, not the exited process", p)
			}
			if len(h.restarts) != 1 || formatCommand(h.restarts[0].args) != formatCommand(h.serverArgs) {
				t.Fatalf("restarts=%+v, want one baseline restoration", h.restarts)
			}
			if s != h.strategy || formatCommand(args) != formatCommand(h.serverArgs) {
				t.Fatal("restored configuration is not the measured baseline")
			}
		})
	}
}
