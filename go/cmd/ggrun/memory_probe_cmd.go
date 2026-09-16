package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/memprobe"
	"github.com/raketenkater/ggrun/pkg/placement"
)

type memoryProbeDeviceOutput struct {
	Name          string `json:"name"`
	ModelMB       int    `json:"model_mb"`
	ContextMB     int    `json:"context_mb"`
	ComputeMB     int    `json:"compute_mb"`
	UnaccountedMB int    `json:"unaccounted_mb"`
	TotalMB       int    `json:"total_mb"`
}

type memoryProbeOutput struct {
	Schema          string                    `json:"schema"`
	Backend         string                    `json:"backend"`
	BackendIdentity string                    `json:"backend_identity"`
	Evidence        memoryEvidenceLevel       `json:"evidence"`
	Attempts        int                       `json:"attempts"`
	Devices         []memoryProbeDeviceOutput `json:"devices"`
	Host            memprobe.HostMemory       `json:"host"`
	Coverage        memprobe.Coverage         `json:"coverage"`
	ServerArgv      []string                  `json:"server_argv"`
}

// memoryProbeRecomputeOptions builds the options for the probe loop's post-fit
// recompute. It exists as a named function so the convergence wiring is
// testable: the defect it fixes was not a bad bound but a missing one, and a
// future edit that drops `recovery` here would otherwise be invisible.
func memoryProbeRecomputeOptions(
	req *launchRequest,
	model *placement.ModelProfile,
	be *backendInfo,
	cacheDir string,
	recovery *launchMemoryRecovery,
) placement.Options {
	opts := placementOptionsFromRequest(req, model, be, cacheDir)
	opts.SkipPlacementCache = true
	return boundByProvenLimits(opts, recovery)
}

func cmdMemoryProbe(args []string) {
	wantJSON := hasArg(args, "--json")
	filtered := make([]string, 0, len(args))
	for _, arg := range args {
		if arg != "--json" {
			filtered = append(filtered, arg)
		}
	}
	req, err := parseLaunchArgs(filtered)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(2)
	}
	if req.ModelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: ggrun memory-probe <model.gguf> [--json] [--allow-live-memory-probe]")
		os.Exit(2)
	}

	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}
	cfg := loadConfigOrExit()
	req.ModelPath = resolveModelPath(req.ModelPath, cfg.ModelDir)
	model, err := parseModel(req.ModelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing model: %v\n", err)
		os.Exit(1)
	}
	be := resolveLaunchBackend(req, model, caps)
	if be == nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", backendUnavailableMessage(req))
		os.Exit(1)
	}
	strategy, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %s\n", placementErrorMessage(err))
		os.Exit(1)
	}
	claudeCodeSlotAdjust(strategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	runtimeCaps, visibleToPhysical := runtimeGPUCapabilities(caps, req)
	oomPenalty := map[int]int{}
	// This command previews the launch plan, so it has to converge the same way a
	// launch does. Without this ledger the probe loop is the launch loop minus
	// every convergence ratchet: recoverPreflightOOM received nil, so
	// observeExpertResidency recorded nothing, boundByProvenLimits bounded
	// nothing, and expertResidencyFloor returned zero. Observed on
	// GLM-5.3-Flash 2026-09-15: `memory-probe -ctx fit` cycled n-cpu-moe 40 <-> 41
	// at a constant 500,736-token context, flipping the -ot expert assignment
	// between CUDA1 and CUDA2 each round, and exhausted all six attempts on a
	// configuration the launch path plans successfully.
	recovery := newLaunchMemoryRecovery()
	progress := newProbeSearchProgress(strategy)

	for attempt := 1; attempt <= memoryProbeMaxAttempts; attempt++ {
		if err := confirmRequiredMMap(req, strategy, os.Stdin, os.Stderr, stdinIsTerminal()); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		serverArgs := buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
		outcome := preflightPlacement(req, be, &configForPreflight{CacheDir: cfg.CacheDir}, runtimeCaps, model, strategy, serverArgs)
		if consent, ok := outcome.Err.(*liveMemoryProbeConsentError); ok {
			if err := confirmLiveMemoryProbe(req, consent.Reason, os.Stdin, os.Stderr, stdinIsTerminal()); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			rememberLiveMemoryProbeConsent(cfg, os.Stderr)
			continue
		}
		if outcome.Err != nil {
			fmt.Fprintf(os.Stderr, "Error: memory probe failed closed: %v\n", outcome.Err)
			os.Exit(1)
		}
		// A launch treats an unavailable probe as "fall back to the estimate",
		// but this command exists to produce the measurement, so reporting
		// success without one would be a lie.
		if outcome.ProbeUnavailable != "" {
			fmt.Fprintf(os.Stderr, "Error: memory probe unavailable on this host: %s\n", outcome.ProbeUnavailable)
			os.Exit(1)
		}
		if outcome.CompanionRejected {
			fmt.Fprintln(os.Stderr, "Error: selected backend rejected the speculative companion; rerun with --spec off")
			os.Exit(1)
		}
		if outcome.DoesNotFit {
			serverArgs := buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
			// Same order as the launch path: record the disproof before recovering
			// from it, so the next proposal is bounded by it.
			recovery.reject(serverArgs)
			recovery.rejectContext(strategy,
				contextReclaimTokens(model, strategy, serverArgs, outcome.DeficitMB, outcome.Device))
			if contextDeficitOutstripsDevice(model, strategy, serverArgs, outcome.DeficitMB, outcome.Device) {
				recovery.rejectContextOutstripped(strategy)
			}
			next, nextArgs, method, replanErr := recoverPreflightOOM(
				req, cfg, model, be, caps, runtimeCaps, visibleToPhysical,
				strategy, serverArgs, oomPenalty, outcome, recovery,
			)
			if replanErr != nil {
				fmt.Fprintf(os.Stderr, "Error: measured placement does not fit and recovery failed closed: %v\n", replanErr)
				os.Exit(1)
			}
			progress.record(strategy, next, outcome.DeficitMB,
				contextReclaimTokens(model, strategy, serverArgs, outcome.DeficitMB, outcome.Device))
			strategy = next
			fmt.Fprintf(os.Stderr,
				"[memory-probe] %s after CUDA%d allocation %d MiB (deficit %d MiB, next=%s)\n",
				method, outcome.Device, outcome.AllocMB, outcome.DeficitMB, formatCommand(nextArgs),
			)
			continue
		}

		// This argv fitted. Record what it proves before recomputing, and bound the
		// recompute by it, exactly as the launch path does: without this the
		// recompute walks back to a context or ubatch this probe already
		// disproved, the argv differs so the identity check below never matches,
		// and the loop burns its whole attempt budget.
		if outcome.Evidence.Level != memoryEvidenceNone {
			recovery.acceptContext(strategy)
		}
		opts := memoryProbeRecomputeOptions(req, model, be, cfg.CacheDir, recovery)
		next, replanErr := placement.Compute(caps, model, opts)
		if replanErr != nil {
			fmt.Fprintf(os.Stderr, "Error: measured placement recompute failed: %v\n", replanErr)
			os.Exit(1)
		}
		claudeCodeSlotAdjust(next, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
		nextArgs := buildLaunchServerArgs(req, cfg, be, caps, model, next)
		if formatCommand(nextArgs) != formatCommand(serverArgs) {
			strategy = next
			continue
		}

		result := memoryProbeOutput{
			Schema: "ggrun-memory-probe-v1", Backend: be.Tag, BackendIdentity: be.Identity,
			Evidence: outcome.Evidence.Level, Attempts: attempt, Host: outcome.Evidence.Host,
			Coverage: outcome.Evidence.Coverage, ServerArgv: serverArgs,
		}
		for _, device := range outcome.Evidence.Devices {
			result.Devices = append(result.Devices, memoryProbeDeviceOutput{
				Name: device.Name, ModelMB: device.ModelMB, ContextMB: device.ContextMB,
				ComputeMB: device.ComputeMB, UnaccountedMB: device.UnaccountedMB, TotalMB: device.TotalMB(),
			})
		}
		if wantJSON {
			if err := json.NewEncoder(os.Stdout).Encode(result); err != nil {
				fmt.Fprintf(os.Stderr, "Error writing memory probe: %v\n", err)
				os.Exit(1)
			}
			return
		}
		fmt.Printf("Memory probe stable after %d attempt(s): %s evidence\n", attempt, result.Evidence)
		for _, device := range result.Devices {
			fmt.Printf("  %s: %d MiB (model=%d context=%d compute=%d unaccounted=%d)\n",
				device.Name, device.TotalMB, device.ModelMB, device.ContextMB, device.ComputeMB, device.UnaccountedMB)
		}
		return
	}
	fmt.Fprintf(os.Stderr, "Error: memory probe did not reach a fixed point after %d attempts\n", memoryProbeMaxAttempts)
	// "N attempts" alone cannot distinguish a search that is stuck from one that
	// is merely slow, and the difference decides whether the user should retry,
	// name a smaller -ctx, or report a bug. Diagnosing the GLM-5.3-Flash case
	// took a hand-diff of the per-round argv; this reports it directly.
	progress.report(os.Stderr)
	os.Exit(1)
}
