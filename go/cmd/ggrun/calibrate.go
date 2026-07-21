package main

import (
	"fmt"
	"os"
	"time"

	"github.com/raketenkater/ggrun/pkg/benchmark"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
)

// calibrateSizeGateMB bounds the models calibration runs on automatically.
// Calibration restarts the server once per candidate; on a huge MoE that costs
// 15+ minutes per reload, so auto mode skips anything at or above this size.
// --calibrate on forces it regardless of size.
const calibrateSizeGateMB = 40 * 1024 // 40 GB

// calibrationMinImprovementPct is the margin a challenger must beat the default
// by before it is cached as the winner. Below this the two placements are
// indistinguishable at micro-probe precision and the estimated default stands.
const calibrationMinImprovementPct = 3.0

// calibrationPlan decides whether this launch should run first-launch
// calibration and, if so, the candidate set to measure. It is a pure function
// of the request, model, hardware, and whether a prior decision exists, so the
// policy is unit-testable without starting a server.
//
// Returns nil when there is nothing to do: calibration disabled, a decision
// already cached for this exact scope, or the placement has no alternatives
// (single GPU, CPU-only, non-MoE single-GPU, symmetric dense split).
func calibrationPlan(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy) []placement.CalibrationCandidate {
	if req == nil || cfg == nil || model == nil || be == nil || caps == nil || strategy == nil {
		return nil
	}
	mode := req.Calibrate
	if mode == "" {
		mode = calibrateAuto
	}
	if mode == calibrateOff {
		return nil
	}
	// The auto gate: skip models too large to restart cheaply. A forced
	// --calibrate on ignores the size gate but still respects the scope cache.
	if mode == calibrateAuto && model.TotalSizeMB >= calibrateSizeGateMB {
		return nil
	}
	candidates := calibrationCandidates(req, cfg, model, be, caps, strategy)
	if len(candidates) < 2 {
		return nil
	}
	// One decision per scope: once any candidate has won, later launches apply
	// it directly instead of re-measuring.
	scopeKey := calibrationScopeKey(req, model, be, caps, strategy)
	if _, err := placement.LoadCalibrationDecision(cfg.CacheDir, scopeKey); err == nil {
		return nil
	}
	return candidates
}

// applyCalibrationDecision returns the strategy to serve with when a prior
// calibration decision exists for this scope. The winner is re-derived by name
// from the deterministic candidate generator rather than deserialized, so the
// full placement (KV placement, main GPU, split) is reproduced exactly and can
// never be a partial overlay. Context, batch, and slots always come from the
// current request. Returns the input strategy unchanged when no decision
// applies or the winner's candidate no longer exists for this hardware.
func applyCalibrationDecision(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy) *placement.Strategy {
	if req == nil || cfg == nil || model == nil || be == nil || caps == nil || strategy == nil {
		return strategy
	}
	if req.Calibrate == calibrateOff {
		return strategy
	}
	scopeKey := calibrationScopeKey(req, model, be, caps, strategy)
	decision, err := placement.LoadCalibrationDecision(cfg.CacheDir, scopeKey)
	if err != nil || decision.Winner == "" || decision.Winner == "default" {
		return strategy
	}
	for _, cand := range calibrationCandidates(req, cfg, model, be, caps, strategy) {
		if cand.Name == decision.Winner {
			fmt.Printf("[calibrate] applying cached winner %s (%.1f vs default %.1f tok/s)\n", decision.Winner, decision.WinnerTPS, decision.DefaultTPS)
			return cand.Strategy
		}
	}
	return strategy
}

// calibrationScopeKey builds the opaque cache key for this launch shape. It
// mirrors placement.NewCalibrationScopeKey on the request's resolved options so
// a decision is valid only for the exact model + backend + hardware + workload
// + runtime knobs it was measured under.
func calibrationCandidates(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy) []placement.CalibrationCandidate {
	cacheDir := ""
	if cfg != nil {
		cacheDir = cfg.CacheDir
	}
	opts := placementOptionsFromRequest(req, model, be, cacheDir)
	candidates := placement.CalibrationCandidates(caps, model, strategy, opts)
	if req == nil || req.ForceMMap || len(candidates) < 2 {
		return candidates
	}
	// Do not stop a resident server and then ask for a new disk-paging policy
	// halfway through calibration. An mmap-dependent alternate is eligible only
	// when the user approved mmap on the original command line or launch prompt.
	out := candidates[:1]
	for _, cand := range candidates[1:] {
		if cand.Strategy != nil && !cand.Strategy.MMapRequired {
			out = append(out, cand)
		}
	}
	return out
}

func calibrationScopeKey(req *launchRequest, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy) string {
	opts := placementOptionsFromRequest(req, model, be, "")
	key := placement.NewCalibrationScopeKey(model, caps, opts, strategy)
	return key.String()
}

// runCalibration measures each candidate placement with a live micro-benchmark
// and returns the strategy to actually serve with. The default (candidate 0)
// is measured first and is the fallback on any failure; a challenger only
// replaces it when it wins by calibrationMinImprovementPct. The winning
// decision is cached under the scope key so this launch shape never pays the
// restart cost again.
//
// The returned process is left running for the winning strategy. On any error
// the caller's already-running default process is retained.
func runCalibration(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy, serverArgs []string, timeout time.Duration, p *server.Process) (*server.Process, *placement.Strategy, []string) {
	candidates := calibrationPlan(req, cfg, model, be, caps, strategy)
	if len(candidates) < 2 {
		return p, strategy, serverArgs
	}
	scopeKey := calibrationScopeKey(req, model, be, caps, strategy)
	fmt.Printf("[calibrate] first launch of this model/hardware/workload: measuring %d placements\n", len(candidates))

	baseURL := fmt.Sprintf("http://localhost:%d", req.Port)
	bench := func() (float64, error) {
		runner := &benchmark.Runner{BaseURL: baseURL, Model: model.Basename, Timeout: 5 * time.Minute}
		res, err := runner.Run()
		if err != nil {
			return 0, err
		}
		return res.GenTPS, nil
	}

	// The default is already running: measure it in place.
	defaultTPS, err := bench()
	if err != nil {
		fmt.Fprintf(os.Stderr, "[calibrate] baseline measurement failed (%v); serving default placement\n", err)
		return p, strategy, serverArgs
	}
	bestName := "default"
	bestTPS := defaultTPS
	bestStrategy := strategy
	bestArgs := serverArgs
	fmt.Printf("[calibrate] default: %.1f tok/s\n", defaultTPS)

	curP := p
	for _, cand := range candidates[1:] {
		candArgs := buildLaunchServerArgs(req, cfg, be, caps, model, cand.Strategy)
		if fmt.Sprintf("%v", candArgs) == fmt.Sprintf("%v", serverArgs) {
			continue // candidate serializes identically to the default; nothing to measure
		}
		fmt.Printf("[calibrate] measuring %s...\n", cand.Name)
		_ = curP.Stop()
		cp, _, _, serr := startLaunchWithCUDAOOMRecovery(req, cfg, model, cand.Strategy, be, caps, candArgs, timeout)
		if serr != nil {
			fmt.Fprintf(os.Stderr, "[calibrate] %s failed to start (%v); skipping\n", cand.Name, serr)
			if curP = restartDefault(req, cfg, model, bestStrategy, be, caps, bestArgs, timeout); curP == nil {
				return nil, bestStrategy, bestArgs
			}
			continue
		}
		tps, berr := bench()
		if berr != nil {
			fmt.Fprintf(os.Stderr, "[calibrate] %s measurement failed (%v); skipping\n", cand.Name, berr)
			_ = cp.Stop()
			if curP = restartDefault(req, cfg, model, bestStrategy, be, caps, bestArgs, timeout); curP == nil {
				return nil, bestStrategy, bestArgs
			}
			continue
		}
		fmt.Printf("[calibrate] %s: %.1f tok/s\n", cand.Name, tps)
		if tps > bestTPS*(1.0+calibrationMinImprovementPct/100.0) {
			// Keep this process running; it is the new winner. The old default is
			// already stopped from the candidate handoff above.
			bestName, bestTPS = cand.Name, tps
			bestStrategy, bestArgs = cand.Strategy, candArgs
			curP = cp
		} else {
			_ = cp.Stop()
			if curP = restartDefault(req, cfg, model, bestStrategy, be, caps, bestArgs, timeout); curP == nil {
				return nil, bestStrategy, bestArgs
			}
		}
	}

	decision := placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: bestName,
		DefaultTPS: defaultTPS, WinnerTPS: bestTPS,
		Improvement: (bestTPS - defaultTPS) / defaultTPS * 100.0,
	}
	if path, serr := placement.SaveCalibrationDecision(cfg.CacheDir, decision); serr != nil {
		fmt.Fprintf(os.Stderr, "[calibrate] failed to cache decision: %v\n", serr)
	} else {
		fmt.Printf("[calibrate] winner %s (%.1f vs default %.1f tok/s); cached %s\n", bestName, bestTPS, defaultTPS, path)
	}

	// curP is running the winning strategy. If the winner is not the default the
	// process already serves it; if it is the default and we never left it, p is
	// still that same process.
	return curP, bestStrategy, bestArgs
}

// restartDefault brings the current best strategy back up after a failed or
// losing candidate stopped it. A nil return means the restart failed; the
// caller then reports the error and stops any reviewer before exiting, since
// leaving the user with no server after a calibration that already measured a
// working default is worse than failing loudly.
func restartDefault(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, serverArgs []string, timeout time.Duration) *server.Process {
	p, _, _, err := startLaunchWithCUDAOOMRecovery(req, cfg, model, strategy, be, caps, serverArgs, timeout)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[calibrate] restart of best placement failed: %v\n", err)
		return nil
	}
	return p
}
