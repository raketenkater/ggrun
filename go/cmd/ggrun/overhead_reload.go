package main

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
)

// A fresh install plans and admits its first launch before the post-launch
// system probe has measured any per-GPU CUDA overhead, so every unmeasured
// device reserves placement.UnmeasuredCUDAOverheadMB. That reserve is the OOM
// margin for a device nobody has measured, and it stays. What it costs is the
// first launch's plan: on a three-GPU MoE launch (2026-11-26) the 1,024 MiB
// guess made the full plan miss preflight, recovery derated it to 44% less
// context at ubatch 64 with one GPU idle, the probe then measured 136-323 MiB,
// and the next launch fitted the full plan on all three GPUs. The server kept
// the derated plan for as long as it ran.
//
// Once the first launch has measured the overhead, the launch reloads at most
// once into the plan a relaunch would choose. It warns first, admits the new
// plan through the ordinary start boundary, and restores the plan that was
// serving when the new one is not admitted or comes up smaller.

// overheadReload is the serving process with the plan, argv and memory ledger
// it was admitted under. A nil process means nothing is serving.
type overheadReload struct {
	process  *server.Process
	strategy *placement.Strategy
	args     []string
	recovery *launchMemoryRecovery
}

// measuredOverheadView is the measured-only per-GPU CUDA overhead for the GPU
// set the post-launch probe records under. It never contains the unmeasured
// reserve, so a device without a measurement is absent.
func measuredOverheadView(cacheDir string, caps *detect.Capabilities) map[int]int {
	if caps == nil || len(caps.GPUs) == 0 {
		return nil
	}
	return placement.SystemCUDAOverheadByGPU(cacheDir, caps.GPUs)
}

// measuredOverheadRelief returns the overhead now measured on each device that
// had no measurement before this launch and measured below the reserve planning
// used for it. Only those devices were planned against the guess, and only a
// smaller measurement can make a re-plan larger. before and after must be
// measured-only views: a planning view reads the guess as a measurement.
func measuredOverheadRelief(before, after map[int]int, gpus []detect.GPU) map[int]int {
	var out map[int]int
	for _, g := range gpus {
		if before[g.Index] > 0 {
			continue
		}
		if v := after[g.Index]; v > 0 && v < placement.UnmeasuredCUDAOverheadMB {
			if out == nil {
				out = map[int]int{}
			}
			out[g.Index] = v
		}
	}
	return out
}

func formatOverheadRelief(relief map[int]int) string {
	idx := make([]int, 0, len(relief))
	for i := range relief {
		idx = append(idx, i)
	}
	sort.Ints(idx)
	parts := make([]string, 0, len(idx))
	for _, i := range idx {
		parts = append(parts, fmt.Sprintf("CUDA%d=%d MiB", i, relief[i]))
	}
	return strings.Join(parts, ", ")
}

// overheadReloadEligible decides whether the measurement may reload this
// launch. It skips the launches calibrationPlan already keeps to one load:
// --calibrate off (lab harnesses key on the first ready line), the one-shot
// --benchmark and --worker-benchmark ("load once"), and a plan that is a
// reused verified config or a tuned winner, which is not a first launch.
//
// The reload is admitted with a fresh memory ledger, as the next launch would
// be, because every rejection in this one was priced against the guessed
// reserve. That is true only when the serving load is the lifecycle's only
// weight load: a contained probe or a real CUDA OOM is evidence a fresh ledger
// would forget, so such a launch keeps its plan and the next launch re-plans.
func overheadReloadEligible(req *launchRequest, served *placement.Strategy, relief map[int]int, recovery *launchMemoryRecovery) (bool, string) {
	if len(relief) == 0 {
		return false, ""
	}
	switch {
	case effectiveCalibrationMode(req) == calibrateOff:
		return false, "--calibrate is off"
	case req != nil && (req.Benchmark || req.WorkerBenchmark):
		return false, "a one-shot benchmark loads once"
	case served != nil && (served.VerifiedConfigReused || served.PerformanceTuned):
		return false, "the serving plan is a reused verified config"
	case recovery == nil || recovery.weightLoads != 1:
		return false, "this launch loaded model weights more than once"
	}
	return true, ""
}

// nonResidentStrategy is the optimizer's non-resident class: such a launch
// serves its stable fallback and is never reloaded automatically.
func nonResidentStrategy(s *placement.Strategy) bool {
	return s.MMapRequired || s.Type == placement.CPUOnly || s.Type == placement.DenseCPUOffload
}

// overheadReloadUpgrade reports whether candidate is the serving plan's shape
// with strictly more of what the guessed reserve took away: more context or a
// larger ubatch, and less of neither. Slots, KV cache types and KV placement
// stay the same, no expert layer moves to the CPU, and both plans are
// resident: a non-resident launch is not reloaded automatically. Anything else
// is a different configuration, which only the optimizer's live comparison may
// choose.
func overheadReloadUpgrade(served, candidate *placement.Strategy) bool {
	if served == nil || candidate == nil {
		return false
	}
	if max(served.Parallel, 1) != max(candidate.Parallel, 1) ||
		served.KVType != candidate.KVType || served.KVTypeV != candidate.KVTypeV ||
		served.KVPlacement != candidate.KVPlacement {
		return false
	}
	if candidate.NCPUMoE > served.NCPUMoE || nonResidentStrategy(served) || nonResidentStrategy(candidate) {
		return false
	}
	if candidate.ContextSize < served.ContextSize || candidate.UBatchSize < served.UBatchSize {
		return false
	}
	return candidate.ContextSize > served.ContextSize || candidate.UBatchSize > served.UBatchSize
}

// overheadReloadCandidate plans the request again now that the measured
// overhead is on disk, as the next launch would, and returns it only when it
// is an upgrade with a different, valid argv.
func overheadReloadCandidate(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo,
	caps *detect.Capabilities, served *placement.Strategy, servedArgs []string,
) (*placement.Strategy, []string, string) {
	opts := startBoundaryReplanOptions(placementOptionsFromRequestCaps(req, model, be, cfg.CacheDir, caps))
	opts.SkipPlacementCache = true
	// The reload is optional: a plan that would have to ask about a dense CPU
	// offload is not an upgrade, so the question is never asked for it.
	opts.DenseCPUOffloadPrompt = nil
	next, err := placement.Compute(caps, model, opts)
	if err != nil || next == nil {
		return nil, nil, fmt.Sprintf("the re-plan failed: %v", err)
	}
	next = recomputeAndApplyCalibration(req, cfg, model, be, caps, next)
	claudeCodeSlotAdjust(next, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	if !overheadReloadUpgrade(served, next) {
		return nil, nil, fmt.Sprintf("the re-plan (ctx %d, ubatch %d) is not a larger plan of the same shape", next.ContextSize, next.UBatchSize)
	}
	args := buildLaunchServerArgs(req, cfg, be, caps, model, next)
	if formatCommand(args) == formatCommand(servedArgs) {
		return nil, nil, "the re-plan did not change the launch"
	}
	if err := validateBackendLaunchArgs(be, args); err != nil {
		return nil, nil, fmt.Sprintf("the re-plan's argv is invalid: %v", err)
	}
	return next, args, ""
}

// overheadReloadEffects are the process effects of the reload. Tests replace
// them to drive every branch without a backend or GPU.
type overheadReloadEffects struct {
	plan        func(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, served *placement.Strategy, servedArgs []string) (*placement.Strategy, []string, string)
	stderr      io.Writer
	stopAndWait func(p *server.Process, label string, baseline *detect.Capabilities, timeout time.Duration) bool
	running     func(p *server.Process) bool
	kill        func(p *server.Process)
	release     func(p *server.Process, baseline *detect.Capabilities, timeout time.Duration) error
	start       func(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, args []string, timeout time.Duration, recovery *launchMemoryRecovery) (*server.Process, *placement.Strategy, []string, error)
	restore     func(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, args []string, timeout time.Duration, recovery *launchMemoryRecovery) (*server.Process, *placement.Strategy, []string, error)
}

var overheadReloadFx = overheadReloadEffects{
	plan:        overheadReloadCandidate,
	stderr:      os.Stderr,
	stopAndWait: stopCalibrationProcessAndWait,
	running:     func(p *server.Process) bool { return p != nil && p.IsRunning() },
	kill:        func(p *server.Process) { p.Kill() },
	release:     stopFailedLaunchBeforeAdvisor,
	start:       startLaunchWithCUDAOOMRecoveryState,
	restore:     restoreLaunchWithCUDAOOMRecoveryState,
}

// replanAfterMeasuredOverhead is the launch's one reload after the first load
// measured CUDA overhead. The bool reports whether the reload was attempted;
// the returned reload is then what serves (a nil process means nothing does).
func replanAfterMeasuredOverhead(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo,
	caps, runtimeCaps *detect.Capabilities, overheadBefore map[int]int, served overheadReload, resourceBaseline *detect.Capabilities,
) (overheadReload, bool) {
	if req == nil || cfg == nil || model == nil || be == nil || caps == nil || runtimeCaps == nil ||
		served.process == nil || served.strategy == nil {
		return served, false
	}
	relief := measuredOverheadRelief(overheadBefore, measuredOverheadView(cfg.CacheDir, runtimeCaps), runtimeCaps.GPUs)
	if eligible, why := overheadReloadEligible(req, served.strategy, relief, served.recovery); !eligible {
		if why != "" {
			fmt.Fprintf(os.Stderr, "[launch] measured CUDA overhead %s; not reloading because %s; the next launch plans with it\n",
				formatOverheadRelief(relief), why)
		}
		return served, false
	}
	fx := overheadReloadFx
	candidate, candidateArgs, why := fx.plan(req, cfg, model, be, caps, served.strategy, served.args)
	if candidate == nil {
		fmt.Fprintf(os.Stderr, "[launch] measured CUDA overhead %s; keeping the serving plan: %s\n", formatOverheadRelief(relief), why)
		return served, false
	}
	return reloadIntoMeasuredOverheadPlan(fx, req, cfg, model, be, caps, served, candidate, candidateArgs, relief, resourceBaseline), true
}

// reloadIntoMeasuredOverheadPlan stops the serving plan, admits candidate with
// a fresh memory ledger, and serves the result when it is still an upgrade.
// Otherwise the serving plan is restored under its own ledger, which now
// rejects the candidate argv. At most one backend runs at any point: nothing
// starts until the previous process is gone, and a process that will not stop
// is killed before the reload reports that nothing serves, because the caller
// then exits and would otherwise leave it behind.
func reloadIntoMeasuredOverheadPlan(fx overheadReloadEffects, req *launchRequest, cfg *config.Config, model *placement.ModelProfile,
	be *backendInfo, caps *detect.Capabilities, served overheadReload, candidate *placement.Strategy, candidateArgs []string,
	relief map[int]int, resourceBaseline *detect.Capabilities,
) overheadReload {
	window := restartAdmissionWindow(model)
	cost := "one more model load"
	if d, ok := served.recovery.productionLoadCost(served.args); ok {
		cost = fmt.Sprintf("one more model load (the last took %s)", d.Round(time.Second))
	}
	fmt.Fprintf(fx.stderr,
		"[launch] measured CUDA overhead %s is below the %d MiB reserved before it was measured; reloading once into the plan the next launch would choose (ctx %d -> %d, ubatch %d -> %d). This costs %s; the serving plan comes back if the new one is not admitted.\n",
		formatOverheadRelief(relief), placement.UnmeasuredCUDAOverheadMB,
		served.strategy.ContextSize, candidate.ContextSize, served.strategy.UBatchSize, candidate.UBatchSize, cost)
	// abandon releases a failed process and kills it when it is still running.
	abandon := func(p *server.Process) bool {
		if relErr := fx.release(p, resourceBaseline, failedStartReleaseWait); relErr != nil && fx.running(p) {
			fmt.Fprintf(fx.stderr, "[launch] %v; killing it\n", relErr)
			fx.kill(p)
			return !fx.running(p)
		}
		return true
	}
	restoreServed := func() overheadReload {
		p, s, a, err := fx.restore(req, cfg, model, served.strategy, be, caps, served.args, window, served.recovery)
		if err != nil {
			fmt.Fprintf(fx.stderr, "[launch] restoring the serving plan failed: %v\n", err)
			abandon(p)
			return overheadReload{recovery: served.recovery}
		}
		return overheadReload{process: p, strategy: s, args: a, recovery: served.recovery}
	}
	if !fx.stopAndWait(served.process, "the first-launch plan", resourceBaseline, 30*time.Second) {
		if fx.running(served.process) {
			fmt.Fprintln(fx.stderr, "[launch] the serving plan did not stop; keeping it")
			return served
		}
		fmt.Fprintln(fx.stderr, "[launch] the stopped plan's memory did not settle; restoring it instead of loading the new plan")
		return restoreServed()
	}
	fresh := newLaunchMemoryRecovery()
	p, s, a, err := fx.start(req, cfg, model, candidate, be, caps, candidateArgs, window, fresh)
	if err == nil && p != nil && overheadReloadUpgrade(served.strategy, s) {
		fmt.Fprintf(fx.stderr, "[launch] serving the plan made with the measured overhead (ctx %d, ubatch %d)\n", s.ContextSize, s.UBatchSize)
		return overheadReload{process: p, strategy: s, args: a, recovery: fresh}
	}
	if err != nil {
		served.recovery.reject(candidateArgs)
		fmt.Fprintf(fx.stderr, "[launch] the measured-overhead plan was not admitted (%v); restoring the plan that was serving\n", err)
	} else {
		fmt.Fprintln(fx.stderr, "[launch] the measured-overhead plan was admitted smaller than the serving plan; restoring the plan that was serving")
	}
	if !abandon(p) {
		fmt.Fprintln(fx.stderr, "[launch] the measured-overhead process is still running after a kill; not starting another backend")
		return overheadReload{recovery: served.recovery}
	}
	return restoreServed()
}
