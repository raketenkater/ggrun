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
	// Each release waits up to this long for resources (stopCalibrationProcessAndWait).
	calibrationReleaseCost = 30 * time.Second
	// A no-allocation oracle run is metadata work; this bounds it generously.
	oracleAdmissionCost = 30 * time.Second
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

func (c challengerCost) String() string {
	return fmt.Sprintf("admission=%s load=%s workload=%s release=%s restore=%s",
		c.Admission.Round(time.Second), c.Load.Round(time.Second), c.Workload.Round(time.Second),
		c.Release.Round(time.Second), c.Restore.Round(time.Second))
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
