package main

import (
	"strconv"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// fitMarginFlag is ik_llama.cpp's safety margin (MiB per device) for its own
// --fit layer placement. ik subtracts it from the free VRAM the driver reports;
// without the flag it keeps ikDefaultFitMarginMB.
const fitMarginFlag = "--fit-margin"

// ikDefaultFitMarginMB is ik_llama's own margin when --fit-margin is absent
// (k_default_mem_margin at the bundled 1fddd12).
const ikDefaultFitMarginMB = 1024

// fitMarginStepMB rounds the margin up so small free-VRAM drift between
// launches does not change the argv, which keys cached memory evidence.
const fitMarginStepMB = 256

// allocationGuardLimitsMB is the per-device ceiling the contained allocation
// probe enforces: the planner's free VRAM less the CUDA overhead it reserves.
// It is indexed like caps.GPUs.
func allocationGuardLimitsMB(cacheDir string, caps *detect.Capabilities) []int {
	if caps == nil {
		return nil
	}
	overheadByGPU := placement.PlanningCUDAOverheadByGPU(cacheDir, caps.GPUs)
	limits := make([]int, len(caps.GPUs))
	for i, gpu := range caps.GPUs {
		limits[i] = max(gpu.VRAMFreeMB()-overheadByGPU[gpu.Index], 0)
	}
	return limits
}

// backendFitsItself reports whether args hand layer placement to the backend's
// own fit pass (--fit, or --fit on) rather than an explicit ggrun layout.
func backendFitsItself(args []string) bool {
	for i, a := range args {
		if a == "--fit" {
			return i+1 >= len(args) || args[i+1] != "off"
		}
	}
	return false
}

// guardedFitArgs binds a backend-run --fit to the allocation guard's budget.
//
// The backend sizes its fit from the free VRAM the driver reports, less its
// margin. The guard allows only the planner's free VRAM less the overhead it
// reserves, and the planner withholds memory the driver cannot know about: a
// companion's reservation, an OOM recovery's escalated reserve, the user's VRAM
// headroom. On a fresh install the Auto reviewer's 6144 MiB reservation against
// a 5045 MiB reviewer left ik's fit 1099 MiB above the guard on a 12 GB card;
// every re-plan raised ggrun's reserve, which the fit never saw, and the four
// allowed loads ran out (Qwen3.8-27B IQ2_XXS, --claude-code, 2026-10-14).
//
// The margin is the largest gap between the free memory the backend will see
// at load and the guard limit over the devices, rounded up, so the backend's budget is within the guard on every
// device. It only ever tightens: a gap at or below ik's default leaves args
// unchanged. An explicit --fit-margin is the user's and is kept. Unknown driver
// readings leave args unchanged.
func guardedFitArgs(args []string, help string, userArgs []string, limitsMB []int, driverFreeMB map[int]int, gpus []detect.GPU) []string {
	if !backendFitsItself(args) || !helpHasExactFlag(help, fitMarginFlag) || hasArg(userArgs, fitMarginFlag) {
		return args
	}
	margin := 0
	for i, g := range gpus {
		free, ok := driverFreeMB[g.Index]
		if !ok || i >= len(limitsMB) {
			return args
		}
		// A companion planned but not yet started still takes its seat before
		// the backend loads: the free memory the backend will see is at most
		// what the runtime view leaves after planned and live use.
		if g.VRAMTotalMB > 0 {
			free = min(free, g.VRAMTotalMB-g.VRAMUsedMB)
		}
		margin = max(margin, free-limitsMB[i])
	}
	if margin <= ikDefaultFitMarginMB {
		// ik's own default is already within the guard.
		return args
	}
	margin = (margin + fitMarginStepMB - 1) / fitMarginStepMB * fitMarginStepMB
	return setArgValue(args, fitMarginFlag, strconv.Itoa(margin))
}

// fitDriverFreeVRAM reads the driver's free VRAM per runtime device. It is
// indirected so tests can describe a host.
var fitDriverFreeVRAM = failedBackendFreeVRAM

// bindFitToGuard applies guardedFitArgs to every argv the launcher builds, so
// the argv admission proves, the argv re-plans compare against, and the argv
// that is launched all carry the same margin. The guard's limits come from the
// same runtime view of the selected GPUs the admission uses.
func bindFitToGuard(args []string, req *launchRequest, cacheDir string, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy) []string {
	if be == nil || req == nil || !backendFitsItself(args) || !helpHasExactFlag(be.Help, fitMarginFlag) {
		return args
	}
	runtimeCaps, _ := runtimeGPUCapabilitiesForLaunch(caps, req, strategy)
	if runtimeCaps == nil || len(runtimeCaps.GPUs) == 0 {
		return args
	}
	return guardedFitArgs(args, be.Help, req.ExtraArgs, allocationGuardLimitsMB(cacheDir, runtimeCaps),
		fitDriverFreeVRAM(runtimeCaps), runtimeCaps.GPUs)
}
