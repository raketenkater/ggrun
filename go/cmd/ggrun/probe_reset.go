package main

import (
	"fmt"
	"os"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// cmdProbeReset clears learned runtime graph growth for one runtime signature.
//
// ggrun reserves VRAM for growth a real request needed beyond the load-time
// graph reserve, learned from an allocation failure. That reserve only ever
// rises: nothing shrinks it, nothing ages it out, and a CUDA VMM failure
// carries no allocation size, so the value stored is often a flat fraction of
// the card rather than an observation.
//
// The consequence is a scope permanently taxed by a launch that would not fail
// today. On this project a 24 GiB card ended up reserving 4914 MiB -- 3.6
// expert layers of a RAM-resident MoE, in the part of the pipeline that is most
// of decode cost -- and one of the two contributing aborts was a malformed
// launch, not a capacity limit.
//
// Clearing is safe to offer because the failure it re-exposes is contained: a
// CUDA out-of-memory aborts the backend process, which ggrun's recovery derates
// and restarts. Measured compute buffers and KV probes are expensive to obtain
// and are deliberately kept.
func cmdProbeReset(args []string) {
	if err := runProbeReset(args); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func runProbeReset(args []string) error {
	req, err := parseLaunchArgs(args)
	if err != nil {
		return err
	}
	if req.ModelPath == "" {
		return fmt.Errorf("usage: ggrun probe reset <model.gguf> [--ctx-size N] [--parallel N] [--kv cpu|gpu] [--kv-quality Q]")
	}
	cfg := loadConfigOrExit()
	req.ModelPath = resolveModelPath(req.ModelPath, cfg.ModelDir)
	model, err := parseModel(req.ModelPath)
	if err != nil {
		return fmt.Errorf("parse model: %w", err)
	}
	caps, err := detect.Detect()
	if err != nil {
		return fmt.Errorf("detect hardware: %w", err)
	}
	be := resolveLaunchBackend(req, model, caps)
	if be == nil {
		return fmt.Errorf("backend resolution: %s; the probe scope is keyed by backend identity", backendUnavailableMessage(req))
	}

	// The signature must match the launch whose reserve is being cleared, so
	// derive it the same way a launch does rather than from the raw flags.
	strategy, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil {
		return fmt.Errorf("compute placement scope: %w", err)
	}
	claudeCodeSlotAdjust(strategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	tag := placement.SparseLookupScopedTag(scopedProbeBackendTag(req, model, be), strategy.SparseLookupPolicy)

	before := placement.RuntimeGraphGrowthByGPU(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, tag, caps.GPUs, strategy.Parallel)
	if len(before) == 0 {
		fmt.Printf("[probe] no learned runtime growth for ctx=%d ubatch=%d kv=%s/%s parallel=%d\n",
			strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, strategy.Parallel)
		return nil
	}
	for gpu, mb := range before {
		fmt.Printf("[probe] clearing CUDA%d reserve: %d MiB\n", gpu, mb)
	}
	if err := placement.ClearRuntimeGraphGrowth(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, tag, caps.GPUs, strategy.Parallel); err != nil {
		return fmt.Errorf("clear runtime growth: %w", err)
	}
	fmt.Println("[probe] measured compute buffers and KV probes were kept.")
	fmt.Println("[probe] the next launch will pack more onto the GPU and re-learn if it genuinely cannot fit.")
	return nil
}
