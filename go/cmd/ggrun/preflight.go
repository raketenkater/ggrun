package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// Launch preflight: before paying a real model load (15+ minutes for a big MoE
// on --no-mmap), ask the backend itself whether the computed placement fits.
// llama.cpp's `llama-fit-params --fit-print on` loads the model with
// no_alloc=true, builds the exact startup graphs, and prints per-device
// model/context/compute MiB — the same allocator accounting a real load will
// use, without committing a byte of VRAM (~1s). ggrun's own placement math
// predicts these numbers; the preflight catches the cases where prediction and
// backend disagree BEFORE the load, and feeds the measured deficit back into
// the re-planner. No fit-params binary next to the backend (ik_llama, forks) →
// preflight is skipped and behavior is unchanged.

// preflightDevice is one row of `llama-fit-params --fit-print on` output:
// planned MiB per device for model weights, context (KV) and compute buffers.
type preflightDevice struct {
	Name      string // "CUDA0", "Host", ...
	ModelMB   int
	ContextMB int
	ComputeMB int
}

// TotalMB is the device's planned VRAM demand at load time.
func (d preflightDevice) TotalMB() int { return d.ModelMB + d.ContextMB + d.ComputeMB }

func preflightContextTotalMB(devs []preflightDevice) int {
	total := 0
	for _, d := range devs {
		if d.ContextMB > 0 {
			total += d.ContextMB
		}
	}
	return total
}

// findFitParamsBin locates the llama-fit-params binary belonging to the given
// server binary: a sibling of the resolved binary (backend build dir), then a
// sibling of the unresolved path (.bin), then PATH. Empty when unavailable.
func findFitParamsBin(serverBin string) string {
	if serverBin == "" {
		return ""
	}
	var candidates []string
	if resolved, err := filepath.EvalSymlinks(serverBin); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(resolved), "llama-fit-params"))
	}
	candidates = append(candidates, filepath.Join(filepath.Dir(serverBin), "llama-fit-params"))
	for _, c := range candidates {
		if fi, err := os.Stat(c); err == nil && !fi.IsDir() {
			return c
		}
	}
	if p, err := exec.LookPath("llama-fit-params"); err == nil {
		return p
	}
	return ""
}

// preflightArgValueFlags are the launch flags that shape memory allocation.
// Everything else (server networking, sampling, logging) is stripped: the
// fit-params arg parser only accepts its own example's flag set, and none of
// the stripped flags change where bytes land.
var preflightArgValueFlags = map[string]bool{
	"-m": true, "--model": true,
	"-c": true, "--ctx-size": true, "--ctx": true,
	"-b": true, "--batch-size": true,
	"-ub": true, "--ubatch-size": true,
	"-ctk": true, "--cache-type-k": true,
	"-ctv": true, "--cache-type-v": true,
	"-np": true, "--parallel": true,
	"-ngl": true, "--n-gpu-layers": true, "--gpu-layers": true,
	"-ts": true, "--tensor-split": true,
	"-sm": true, "--split-mode": true,
	"-ot": true, "--override-tensor": true,
	"-ncmoe": true, "--n-cpu-moe": true,
	"-fa": true, "--flash-attn": true,
	"-mg": true, "--main-gpu": true,
}

// preflightArgs filters real launch args down to the memory-shaping subset.
func preflightArgs(serverArgs []string) []string {
	out := []string{"--fit-print", "on"}
	for i := 0; i < len(serverArgs); i++ {
		a := serverArgs[i]
		if !preflightArgValueFlags[a] || i+1 >= len(serverArgs) {
			continue
		}
		// No legitimate value of these flags starts with "-"; a following flag
		// means the user passed the flag bare — drop it rather than mis-pair.
		if v := serverArgs[i+1]; !strings.HasPrefix(v, "-") {
			out = append(out, a, v)
			i++
		}
	}
	return out
}

// runFitPreflight executes the no-alloc accounting and parses the per-device
// rows. serverArgs are the real launch args (binary path at index 0).
func runFitPreflight(fitBin string, serverArgs []string) ([]preflightDevice, error) {
	args := preflightArgs(serverArgs[1:])
	cmd := exec.Command(fitBin, args...)
	// Same device numbering contract as the real server launch (server.go):
	// placement indices are PCI-ordered, CUDA's default order is fastest-first.
	env := os.Environ()
	filtered := env[:0]
	for _, e := range env {
		if !strings.HasPrefix(e, "CUDA_DEVICE_ORDER=") {
			filtered = append(filtered, e)
		}
	}
	cmd.Env = append(filtered, "CUDA_DEVICE_ORDER=PCI_BUS_ID")

	done := make(chan struct{})
	var out []byte
	var err error
	go func() {
		out, err = cmd.Output()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Minute):
		_ = cmd.Process.Kill()
		<-done
		return nil, fmt.Errorf("fit-params preflight timed out")
	}
	if err != nil {
		return nil, fmt.Errorf("fit-params preflight failed: %w", err)
	}

	var devs []preflightDevice
	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Fields(line)
		if len(f) < 4 {
			continue
		}
		model, err1 := strconv.Atoi(f[1])
		ctx, err2 := strconv.Atoi(f[2])
		comp, err3 := strconv.Atoi(f[3])
		if err1 != nil || err2 != nil || err3 != nil {
			continue
		}
		devs = append(devs, preflightDevice{Name: f[0], ModelMB: model, ContextMB: ctx, ComputeMB: comp})
	}
	if len(devs) == 0 {
		return nil, fmt.Errorf("fit-params preflight produced no device rows")
	}
	return devs, nil
}

// preflightWorstDeficit compares the backend's planned per-GPU demand against
// free VRAM. The only extra terms are measured per-device CUDA context overhead
// and measured runtime graph growth for the exact runtime signature. Missing
// entries mean unknown and contribute 0; there are no percentage reserves or
// static fallback margins hidden in this calculation. Device rows are matched
// to GPUs by CUDA index == detect index, both PCI-ordered under
// CUDA_DEVICE_ORDER=PCI_BUS_ID.
func cudaDeviceIndex(name string) (int, bool) {
	if !strings.HasPrefix(name, "CUDA") {
		return 0, false
	}
	idx, err := strconv.Atoi(strings.TrimPrefix(name, "CUDA"))
	return idx, err == nil
}

func preflightWorstDeficit(devs []preflightDevice, gpus []detect.GPU, overheadByGPU, runtimeGrowthByGPU map[int]int) (int, int, string) {
	worstDev, worstDeficit := -1, 0
	var summary []string
	for _, d := range devs {
		idx, ok := cudaDeviceIndex(d.Name)
		if !ok {
			continue
		}
		for _, g := range gpus {
			if g.Index != idx {
				continue
			}
			overheadMB := overheadByGPU[idx]
			runtimeMB := runtimeGrowthByGPU[idx]
			fitMB := d.TotalMB()
			need := fitMB + overheadMB + runtimeMB
			free := g.VRAMFreeMB()
			summary = append(summary, fmt.Sprintf("%s %d/%d MiB (fit=%d overhead=%d runtime=%d)", d.Name, need, free, fitMB, overheadMB, runtimeMB))
			if deficit := need - free; deficit > worstDeficit {
				worstDev, worstDeficit = idx, deficit
			}
		}
	}
	return worstDev, worstDeficit, strings.Join(summary, ", ")
}

// preflightPlacement runs the fit-params gate for one launch attempt. It
// returns (device, deficitMB, true) when the placement provably cannot load —
// the caller feeds the deficit into the re-planner instead of paying a real
// load to learn the same thing. Any infrastructure failure (no binary,
// unsupported arch, parse error) skips the gate: the preflight must never
// block a launch the backend could have served.
func preflightPlacement(be *backendInfo, cfg *configForPreflight, caps *detect.Capabilities, model *placement.ModelProfile, strategy *placement.Strategy, serverArgs []string) (int, int, bool) {
	if be == nil || caps == nil || len(caps.GPUs) == 0 {
		return -1, 0, false
	}
	fitBin := findFitParamsBin(be.Path)
	if fitBin == "" {
		return -1, 0, false
	}
	devs, err := runFitPreflight(fitBin, serverArgs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[launch] preflight skipped: %v\n", err)
		return -1, 0, false
	}
	// Feed the backend's measured context and compute buffers back into placement
	// BEFORE checking fit, regardless of outcome. A re-plan below
	// (ReplanAfterOOM -> Compute) must see these real numbers immediately, not
	// the first-launch formulas that produced this (possibly wrong) strategy.
	if model != nil && strategy != nil {
		computeByGPU := map[int]int{}
		for _, d := range devs {
			if idx, ok := cudaDeviceIndex(d.Name); ok {
				computeByGPU[idx] = d.ComputeMB
			}
		}
		placement.RecordMeasuredContextMB(cfg.CacheDir, model, strategy.ContextSize, strategy.KVType, preflightContextTotalMB(devs))
		_ = placement.RecordMeasuredComputeBuffers(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, be.Tag, caps.GPUs, strategy.Parallel, computeByGPU)
	}
	overheadByGPU := placement.SystemCUDAOverheadByGPU(cfg.CacheDir, caps.GPUs)
	var runtimeGrowthByGPU map[int]int
	if model != nil && strategy != nil {
		runtimeGrowthByGPU = placement.RuntimeGraphGrowthByGPU(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, be.Tag, caps.GPUs, strategy.Parallel)
	}
	dev, deficit, summary := preflightWorstDeficit(devs, caps.GPUs, overheadByGPU, runtimeGrowthByGPU)
	if deficit > 0 {
		fmt.Fprintf(os.Stderr, "[launch] preflight: placement does not fit (%s) - re-planning before load\n", summary)
		if model != nil && model.IsMoE && strategy != nil {
			// The re-plan below may fall through pkg/placement's ubatch-fit
			// ladder (maximizeMoEGPUFitByUBatch). Without this, a ladder rung
			// that was never measured here falls back to the first-launch
			// heuristic — the same wrong-by-4x estimate that produced this
			// deficit in the first place, just at a different ubatch.
			measureUBatchLadderCandidates(fitBin, serverArgs, cfg, caps, model, strategy, be.Tag)
		}
		return dev, deficit, true
	}
	fmt.Printf("[launch] preflight: placement fits (%s)\n", summary)
	return -1, 0, false
}

// measureUBatchLadderCandidates runs the no-alloc preflight for every
// UBatchFitLadder rung smaller than the current placement's ubatch and
// records each one's real per-GPU compute buffer, so a downstream ubatch-fit
// retry always has measured data to work from rather than the heuristic
// first-launch estimate. Best-effort: a failed candidate run just means that
// rung stays on the heuristic, same as before this function existed.
func measureUBatchLadderCandidates(fitBin string, serverArgs []string, cfg *configForPreflight, caps *detect.Capabilities, model *placement.ModelProfile, strategy *placement.Strategy, backendTag string) {
	for _, ub := range placement.UBatchFitLadder {
		if ub >= strategy.UBatchSize {
			continue
		}
		candArgs := replaceUBatchArg(serverArgs, ub)
		devs, err := runFitPreflight(fitBin, candArgs)
		if err != nil {
			continue
		}
		computeByGPU := map[int]int{}
		for _, d := range devs {
			if idx, ok := cudaDeviceIndex(d.Name); ok {
				computeByGPU[idx] = d.ComputeMB
			}
		}
		_ = placement.RecordMeasuredComputeBuffers(cfg.CacheDir, model, strategy.ContextSize, ub, strategy.KVQuality, strategy.KVPlacement, backendTag, caps.GPUs, strategy.Parallel, computeByGPU)
	}
}

// replaceUBatchArg returns a copy of args with -ub/--ubatch-size's value set
// to ub, appending the flag if absent. Leaves the input slice untouched.
func replaceUBatchArg(args []string, ub int) []string {
	out := append([]string(nil), args...)
	val := strconv.Itoa(ub)
	for i, a := range out {
		if (a == "-ub" || a == "--ubatch-size") && i+1 < len(out) {
			out[i+1] = val
			return out
		}
	}
	return append(out, "-ub", val)
}

// configForPreflight is the slice of config the preflight needs; a tiny struct
// keeps preflightPlacement testable without a full config.Config.
type configForPreflight struct {
	CacheDir string
}
