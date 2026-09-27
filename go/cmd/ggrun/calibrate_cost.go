package main

import (
	"fmt"
	"time"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// An optional challenger must pay for its whole experiment before a healthy
// baseline is stopped: admission, the candidate's load, the workload, releasing
// both processes and restoring the baseline. The MiMo launch that motivated this
// stopped an 11.5-minute baseline to try a finalist inside a 20-minute budget;
// the finalist's probe and the restoration could not both fit, and the user was
// left with no server (STARTUP-MIMO-20260924).
//
// Costs come from this launch's observations, never from file size. Loads are
// inflated by loadCostUncertainty because a repeat load of the same weights is
// not guaranteed to be as fast as the one observed (page cache, repack, I/O).
const (
	loadCostUncertainty = 1.25
	// One release is a process stop (server.Process.Stop waits up to 15s for
	// exit plus 15s for its scope) and then up to 30s for RAM/VRAM to return
	// (stopCalibrationProcessAndWait). The reserve must price the real bound.
	calibrationReleaseCost = 60 * time.Second
	// A no-allocation oracle run is metadata work; this bounds it generously.
	oracleAdmissionCost = 30 * time.Second
	// The support optimizer's lifecycle outside its query context: helper
	// startup (advisor.Runner default 5m), its release wait and a release.
	calibrationAdvisorBound = 5*time.Minute + 30*time.Second + calibrationReleaseCost
)

type challengerCostInputs struct {
	// BaselineLoad is the observed production load time of the serving baseline.
	BaselineLoad         time.Duration
	BaselineLoadObserved bool
	// BaselineWorkload is how long measuring the baseline took; the candidate
	// runs the identical workload.
	BaselineWorkload time.Duration
	// Ceiling is the per-process startup ceiling, used when no load was observed.
	Ceiling time.Duration
	// ProbeNeeded is true when admission must run a contained full-load probe:
	// no same-build oracle and no reusable exact allocation evidence.
	ProbeNeeded bool
	// BaselineRunning is true when the baseline is still live and must be
	// stopped first; otherwise one release is already paid.
	BaselineRunning bool
}

type challengerCost struct {
	Admission, Load, Workload, Release, Restore time.Duration
}

func (c challengerCost) Total() time.Duration {
	return c.Admission + c.Load + c.Workload + c.Release + c.Restore
}

func (c challengerCost) Affordable(remaining time.Duration) bool {
	return c.Total() <= remaining
}

// Reserve is the part of the experiment that must stay available once the
// candidate runs: releasing the candidate and restoring the baseline. Candidate
// admission and its workload may never spend it.
func (c challengerCost) Reserve() time.Duration {
	return calibrationReleaseCost + c.Restore
}

func (c challengerCost) String() string {
	return fmt.Sprintf("admission=%s load=%s workload=%s release=%s restore=%s reserve=%s",
		c.Admission.Round(time.Second), c.Load.Round(time.Second), c.Workload.Round(time.Second),
		c.Release.Round(time.Second), c.Restore.Round(time.Second), c.Reserve().Round(time.Second))
}

// calibrationExperiment is the one clock of an optimization experiment, fixed
// when the experiment starts. The optimization deadline bounds every optional
// step: pilots, prescreens, challenger admission and its workload. Only putting
// a measured configuration back into service may run past it, and only until
// emergencyDeadline, which is fixed at creation and shared by every restart:
// leaving no server to meet a timer is worse than a bounded, reported overrun.
type calibrationExperiment struct {
	started, deadline, emergencyDeadline time.Time
	now                                  func() time.Time
}

func newCalibrationExperiment(budget, emergencyGrace time.Duration, now func() time.Time) *calibrationExperiment {
	if now == nil {
		now = time.Now
	}
	start := now()
	return &calibrationExperiment{
		started: start, deadline: start.Add(budget),
		emergencyDeadline: start.Add(budget + emergencyGrace), now: now,
	}
}

// remaining is the optimization time left; it may be negative.
func (e *calibrationExperiment) remaining() time.Duration { return e.deadline.Sub(e.now()) }

// until is the optimization time left once reserve is held back.
func (e *calibrationExperiment) until(reserve time.Duration) time.Duration {
	return e.remaining() - reserve
}

// overrun is how far the experiment has run past its optimization deadline.
func (e *calibrationExperiment) overrun() time.Duration {
	return max(0, e.now().Sub(e.deadline))
}

// serviceWindow bounds a restart that puts a measured configuration back into
// service. It never exceeds one restart window, ends by the shared emergency
// deadline after holding back keep (the baseline's restoration when a winner
// restart could still fail), and is never below floor: when earlier bounded
// steps (process stops) already consumed the emergency window, the baseline
// still receives its priced restoration rather than being abandoned.
func (e *calibrationExperiment) serviceWindow(restartCap, keep, floor time.Duration) time.Duration {
	window := min(restartCap, e.emergencyDeadline.Sub(e.now())-keep)
	if floor > 0 && window < floor {
		window = min(max(floor, minUsefulLoadWindow), restartCap)
	}
	return window
}

func estimateChallengerCost(in challengerCostInputs) challengerCost {
	load := in.BaselineLoad
	if !in.BaselineLoadObserved || load <= 0 {
		// Nothing observed: assume the worst load the supervisor would permit.
		load = in.Ceiling
	}
	load = time.Duration(float64(load) * loadCostUncertainty)
	cost := challengerCost{
		Admission: oracleAdmissionCost,
		Load:      load,
		Workload:  time.Duration(float64(in.BaselineWorkload) * loadCostUncertainty),
		Release:   calibrationReleaseCost,
		Restore:   load,
	}
	if in.ProbeNeeded {
		cost.Admission = load
	}
	if in.BaselineRunning {
		cost.Release += calibrationReleaseCost
	}
	return cost
}

// candidateAllocationEvidenceCached reports whether admission can reuse exact
// guarded allocation evidence for this candidate instead of probing it.
func candidateAllocationEvidenceCached(req *launchRequest, cfg *config.Config, be *backendInfo, caps *detect.Capabilities, model *placement.ModelProfile, strategy *placement.Strategy, args []string) bool {
	if cfg == nil || cfg.CacheDir == "" {
		return false
	}
	runtimeCaps, _ := runtimeGPUCapabilitiesForLaunch(caps, req, strategy)
	if runtimeCaps == nil {
		runtimeCaps = caps
	}
	_, ok := loadMemoryEvidence(cfg.CacheDir, memoryEvidenceKey(be, model, runtimeCaps, args))
	return ok
}

// prescreenCalibrationCandidate runs the candidate's memory admission through
// the same-build oracle only: its admission work allows zero weight-loading
// starts, so an oracle failure can never become a contained probe beside the
// live baseline. It reports a refusal only for outcomes exact admission would
// also refuse; anything inconclusive leaves the decision to the real admission.
//
// It must price the moment after the baseline stops, as exact admission will.
// caps are the launch's detection-time capabilities (the admission's input,
// with the reviewer reservation applied on top), and resourceBaseline is the
// live usage captured after companions started and before the main model did,
// which is what the devices return to once the baseline is gone. Reading live
// usage now would count the serving baseline itself; using the snapshot as the
// detection input would count the companions twice.
func prescreenCalibrationCandidate(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps, resourceBaseline *detect.Capabilities, strategy *placement.Strategy, args []string) (bool, string, string) {
	if cfg == nil || caps == nil || resourceBaseline == nil || strategy == nil {
		return false, "", ""
	}
	afterStop := map[int]int{}
	for _, gpu := range resourceBaseline.GPUs {
		afterStop[gpu.Index] = gpu.VRAMUsedMB
	}
	runtimeCaps, _ := runtimeGPUCapabilitiesWithUsage(caps, req, strategy, func(physical int) int { return afterStop[physical] })
	if runtimeCaps == nil {
		return false, "", ""
	}
	work := newAdmissionWork("prescreen", time.Minute, time.Minute, 0, cfg.CacheDir, nil)
	outcome := preflightPlacement(req, be, &configForPreflight{CacheDir: cfg.CacheDir, Work: work}, runtimeCaps, model, strategy, args)
	switch {
	case outcome.Err != nil || outcome.ProbeUnavailable != "":
		return false, "", ""
	case outcome.BackendAdjustment != nil:
		return true, string(exactAdmissionCompat), "backend compatibility adjustment required: " + outcome.BackendAdjustment.Reason
	case outcome.CompanionRejected:
		return true, string(exactAdmissionCompanion), "the speculative companion was rejected"
	case outcome.DoesNotFit:
		return true, string(exactAdmissionMemory), fmt.Sprintf("CUDA%d deficit %d MiB against the launch resources", outcome.Device, outcome.DeficitMB)
	}
	return false, "", ""
}
