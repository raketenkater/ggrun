package main

import (
	"fmt"
	"sort"
	"strings"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// recoverPreflightOOM converts one measured allocation failure into a strictly
// different launch shape. Failed graph allocations are useful component
// measurements, but are never promoted to complete allocation evidence.
func recoverPreflightOOM(
	req *launchRequest,
	cfg *config.Config,
	model *placement.ModelProfile,
	be *backendInfo,
	caps, runtimeCaps *detect.Capabilities,
	visibleToPhysical map[int]int,
	strategy *placement.Strategy,
	serverArgs []string,
	oomPenalty map[int]int,
	outcome preflightOutcome,
) (*placement.Strategy, []string, string, error) {
	if !outcome.DoesNotFit || outcome.Device < 0 || model == nil || strategy == nil {
		return nil, nil, "", fmt.Errorf("invalid preflight allocation failure")
	}
	if outcome.AllocMB <= 0 {
		outcome.AllocMB = maxPreflightInt(outcome.DeficitMB, 1)
	}
	if outcome.DeficitMB <= 0 {
		outcome.DeficitMB = 1
	}

	var candidate *placement.Strategy
	var replanErr error
	if outcome.IsComputeBuffer && cfg != nil && be != nil && runtimeCaps != nil {
		cacheBackendTag := scopedProbeBackendTag(req, model, be)
		recordErr := placement.RecordMeasuredComputeBuffers(
			cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize,
			strategy.KVQuality, strategy.KVPlacement, cacheBackendTag,
			runtimeCaps.GPUs, strategy.Parallel, map[int]int{outcome.Device: outcome.AllocMB},
		)
		if recordErr == nil {
			opts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
			opts.SkipPlacementCache = true
			opts.CacheFile = ""
			candidate, replanErr = placement.Compute(caps, model, opts)
		}
	}

	if candidate == nil && !outcome.IsComputeBuffer {
		physicalDev := physicalGPUIndex(outcome.Device, visibleToPhysical)
		oomPenalty[physicalDev] += outcome.DeficitMB
		candidate, replanErr = placement.ReplanAfterOOM(
			caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir), oomPenalty,
		)
	}

	nextArgs, entry, method, changed := selectChangedPreflightRecovery(
		serverArgs, candidate, model, runtimeCaps, outcome,
	)
	if !changed {
		detail := ""
		if replanErr != nil {
			detail = ": " + replanErr.Error()
		}
		return nil, nil, "", fmt.Errorf(
			"CUDA%d allocation of %d MiB exceeded the guard by %d MiB, but neither exact re-planning nor deterministic derating changed the effective memory configuration%s",
			outcome.Device, outcome.AllocMB, outcome.DeficitMB, detail,
		)
	}
	if method == "replanned" {
		return candidate, nextArgs, method, nil
	}
	if entry != nil {
		if entry.OTString != "" {
			applyDeratedPlacementEntry(strategy, entry)
		} else if entry.UBatchSize > 0 {
			strategy.UBatchSize = entry.UBatchSize
		}
	}
	return strategy, nextArgs, method, nil
}

// selectChangedPreflightRecovery accepts a packer result only when it changes
// the backend's effective memory flags. Otherwise it applies the generic,
// monotonic fallback: graph OOMs lower ubatch; weight OOMs move expert layers
// off the failed device. Returning changed=false forbids an identical reload.
func selectChangedPreflightRecovery(
	currentArgs []string,
	candidate *placement.Strategy,
	model *placement.ModelProfile,
	caps *detect.Capabilities,
	outcome preflightOutcome,
) ([]string, *placement.CacheEntry, string, bool) {
	currentFingerprint := effectiveMemoryArgsFingerprint(currentArgs)
	if candidate != nil {
		nextArgs := patchPlacementArgs(currentArgs, candidate)
		if effectiveMemoryArgsFingerprint(nextArgs) != currentFingerprint {
			return nextArgs, nil, "replanned", true
		}
	}

	// The exact deficit is the minimum memory that must be reclaimed from the
	// loaded state. Passing the full allocation here could drop many expert
	// layers even when the guard was exceeded by only a few MiB.
	derateMB := maxPreflightInt(outcome.DeficitMB, 1)
	nextArgs, entry, ok := placement.DerateCUDAOOMArgs(
		currentArgs, model, caps, outcome.Device, derateMB, outcome.IsComputeBuffer,
	)
	if !ok || effectiveMemoryArgsFingerprint(nextArgs) == currentFingerprint {
		return nil, nil, "", false
	}
	method := "expert-derate"
	if outcome.IsComputeBuffer && entry != nil && entry.UBatchSize > 0 {
		method = "ubatch-derate"
	}
	return nextArgs, entry, method, true
}

var memoryArgCanonical = map[string]string{
	"-m": "model", "--model": "model",
	"-c": "ctx", "--ctx-size": "ctx", "--ctx": "ctx",
	"-b": "batch", "--batch-size": "batch",
	"-ub": "ubatch", "--ubatch-size": "ubatch",
	"-ctk": "cache-k", "--cache-type-k": "cache-k",
	"-ctv": "cache-v", "--cache-type-v": "cache-v",
	"-np": "parallel", "--parallel": "parallel",
	"-ngl": "gpu-layers", "--n-gpu-layers": "gpu-layers", "--gpu-layers": "gpu-layers",
	"-ts": "tensor-split", "--tensor-split": "tensor-split",
	"-sm": "split-mode", "--split-mode": "split-mode",
	"-ot": "override-tensor", "--override-tensor": "override-tensor",
	"-ncmoe": "n-cpu-moe", "--n-cpu-moe": "n-cpu-moe",
	"-fa": "flash-attn", "--flash-attn": "flash-attn",
	"-mg": "main-gpu", "--main-gpu": "main-gpu",
	"-dev": "device", "--device": "device",
}

// effectiveMemoryArgsFingerprint follows the backend's last-value-wins argv
// behavior. This catches retries where ggrun changed an earlier generated flag
// but a later user override kept the effective placement identical.
func effectiveMemoryArgsFingerprint(args []string) string {
	values := map[string]string{}
	for i := 0; i < len(args); i++ {
		canonical, ok := memoryArgCanonical[args[i]]
		if !ok || i+1 >= len(args) || strings.HasPrefix(args[i+1], "-") {
			continue
		}
		values[canonical] = args[i+1]
		i++
	}
	for _, flag := range []string{"--no-kv-offload", "--no-mmap", "--mmap"} {
		if hasArg(args, flag) {
			values[flag] = "true"
		}
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var parts []string
	for _, key := range keys {
		parts = append(parts, key+"="+values[key])
	}
	return strings.Join(parts, "\n")
}
