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
		fmt.Fprintln(os.Stderr, "Error: no llama-server binary found")
		os.Exit(1)
	}
	strategy, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %s\n", placementErrorMessage(err))
		os.Exit(1)
	}
	claudeCodeSlotAdjust(strategy, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet)
	runtimeCaps, visibleToPhysical := runtimeGPUCapabilities(caps, req)
	oomPenalty := map[int]int{}

	for attempt := 1; attempt <= 6; attempt++ {
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
			continue
		}
		if outcome.Err != nil {
			fmt.Fprintf(os.Stderr, "Error: memory probe failed closed: %v\n", outcome.Err)
			os.Exit(1)
		}
		if outcome.CompanionRejected {
			fmt.Fprintln(os.Stderr, "Error: selected backend rejected the speculative companion; rerun with --spec off")
			os.Exit(1)
		}
		if outcome.DoesNotFit {
			serverArgs := buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
			next, nextArgs, method, replanErr := recoverPreflightOOM(
				req, cfg, model, be, caps, runtimeCaps, visibleToPhysical,
				strategy, serverArgs, oomPenalty, outcome,
			)
			if replanErr != nil {
				fmt.Fprintf(os.Stderr, "Error: measured placement does not fit and recovery failed closed: %v\n", replanErr)
				os.Exit(1)
			}
			strategy = next
			fmt.Fprintf(os.Stderr,
				"[memory-probe] %s after CUDA%d allocation %d MiB (deficit %d MiB, next=%s)\n",
				method, outcome.Device, outcome.AllocMB, outcome.DeficitMB, formatCommand(nextArgs),
			)
			continue
		}

		opts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
		opts.SkipPlacementCache = true
		next, replanErr := placement.Compute(caps, model, opts)
		if replanErr != nil {
			fmt.Fprintf(os.Stderr, "Error: measured placement recompute failed: %v\n", replanErr)
			os.Exit(1)
		}
		claudeCodeSlotAdjust(next, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet)
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
	fmt.Fprintln(os.Stderr, "Error: memory probe did not reach a fixed point after 6 attempts")
	os.Exit(1)
}
