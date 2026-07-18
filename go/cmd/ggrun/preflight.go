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
	// A PATH fallback is safe only when the server was itself selected by name.
	// For an absolute/custom fork path it could pair a fork with mainline's
	// fit-params and produce false compatibility or memory results.
	if filepath.Base(serverBin) == serverBin {
		if p, err := exec.LookPath("llama-fit-params"); err == nil {
			return p
		}
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
	"-dev": true, "--device": true,
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
		detail := ""
		if exitErr, ok := err.(*exec.ExitError); ok {
			detail = strings.TrimSpace(string(exitErr.Stderr))
		}
		if detail != "" {
			const maxDetail = 600
			if len(detail) > maxDetail {
				detail = detail[len(detail)-maxDetail:]
			}
			return nil, fmt.Errorf("fit-params preflight failed: %w: %s", err, detail)
		}
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

// backendSpecCandidateValidator returns a cached, no-allocation load probe for
// the selected backend. It catches private GGML tensor types and draft
// architectures that look compatible in metadata but the binary cannot load.
func backendSpecCandidateValidator(be *backendInfo) func(string) error {
	if be == nil {
		return nil
	}
	fitBin := findFitParamsBin(be.Path)
	if fitBin == "" {
		return nil
	}
	results := map[string]error{}
	return func(path string) error {
		if err, ok := results[path]; ok {
			return err
		}
		_, err := runFitPreflight(fitBin, []string{
			"llama-fit-candidate", "-m", path,
			"-c", "512", "-b", "128", "-ub", "64", "-ngl", "all",
		})
		results[path] = err
		return err
	}
}

// draftPreflightServerArgs maps the server's separate draft configuration back
// to the ordinary model/context flags understood by llama-fit-params. Running
// the no-allocation oracle once for the target and once for the companion gives
// us backend-measured model/KV/graph bytes for both without loading either.
func draftPreflightServerArgs(strategy *placement.Strategy) []string {
	if strategy == nil || strategy.Draft == nil || strategy.Draft.Path == "" {
		return nil
	}
	draft := strategy.Draft
	args := []string{"llama-fit-draft", "-m", draft.Path}
	draftCTX := strategy.ContextSize
	if draft.SupportsDraftCTX && draft.CTXSizeDraft > 0 {
		draftCTX = draft.CTXSizeDraft
	}
	if draftCTX > 0 {
		args = append(args, "-c", strconv.Itoa(draftCTX))
	}
	if strategy.BatchSize > 0 {
		args = append(args, "-b", strconv.Itoa(strategy.BatchSize))
	}
	if strategy.UBatchSize > 0 {
		args = append(args, "-ub", strconv.Itoa(strategy.UBatchSize))
	}
	if draft.KVTypeDraft != "" {
		args = append(args, "-ctk", draft.KVTypeDraft, "-ctv", draft.KVTypeDraft)
	}
	if strategy.Parallel > 0 {
		args = append(args, "-np", strconv.Itoa(strategy.Parallel))
	}
	ngl := draft.GPULayersDraft
	if ngl == "" {
		ngl = "all"
	}
	args = append(args, "-ngl", ngl)
	if draft.DraftGPU >= 0 {
		args = append(args, "--device", draftDeviceForPreflight(strategy.BackendTag, draft.DraftGPU))
	}
	if strategy.FlashAttention {
		args = append(args, "--flash-attn", "on")
	}
	return args
}

func draftDeviceForPreflight(backendTag string, gpu int) string {
	if strings.Contains(strings.ToLower(backendTag), "vulkan") {
		return fmt.Sprintf("Vulkan%d", gpu)
	}
	return fmt.Sprintf("CUDA%d", gpu)
}

func mergePreflightDevices(groups ...[]preflightDevice) []preflightDevice {
	order := []string{}
	byName := map[string]preflightDevice{}
	for _, group := range groups {
		for _, d := range group {
			if _, ok := byName[d.Name]; !ok {
				order = append(order, d.Name)
			}
			merged := byName[d.Name]
			merged.Name = d.Name
			merged.ModelMB += d.ModelMB
			merged.ContextMB += d.ContextMB
			merged.ComputeMB += d.ComputeMB
			byName[d.Name] = merged
		}
	}
	out := make([]preflightDevice, 0, len(order))
	for _, name := range order {
		out = append(out, byName[name])
	}
	return out
}

func isEmbeddedMainlineMTP(strategy *placement.Strategy) bool {
	return strategy != nil && strategy.Draft != nil &&
		strategy.Draft.Type == placement.DraftMTP && strategy.Draft.Path == "" &&
		strings.EqualFold(strategy.Draft.SpecType, "draft-mtp")
}

// embeddedMTPPreflightReservation supplies the context+compute rows that the
// standalone fit-params frontend cannot request. llama-server creates a second
// LLAMA_CONTEXT_TYPE_MTP against the target model, so weights are already in the
// target rows but KV and graph buffers are not. The exact MTP layer KV formula is
// metadata-derived; charging that full amount plus at least one full target graph
// reserve to every active CUDA device is intentionally conservative near a limit.
func embeddedMTPPreflightReservation(model *placement.ModelProfile, strategy *placement.Strategy, target []preflightDevice) ([]preflightDevice, error) {
	if !isEmbeddedMainlineMTP(strategy) || model == nil {
		return nil, nil
	}
	if !strings.EqualFold(strategy.KVPlacement, "gpu") {
		return nil, fmt.Errorf("embedded MTP Auto requires verified GPU KV placement")
	}
	kvType := strategy.Draft.KVTypeDraft
	if kvType == "" {
		kvType = "f16" // llama.cpp's draft-context default
	}
	contextMB := placement.EmbeddedMTPContextMB(model, strategy.ContextSize, kvType)
	if contextMB <= 0 {
		return nil, fmt.Errorf("embedded MTP context cannot be derived from GGUF metadata")
	}
	const computeFloorMB = 1024
	rows := make([]preflightDevice, 0, len(target))
	for _, d := range target {
		if _, ok := cudaDeviceIndex(d.Name); !ok {
			continue
		}
		computeMB := d.ComputeMB
		if computeMB < computeFloorMB {
			computeMB = computeFloorMB
		}
		rows = append(rows, preflightDevice{Name: d.Name, ContextMB: contextMB, ComputeMB: computeMB})
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("embedded MTP has no measured CUDA device rows")
	}
	return rows, nil
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
// returns (device, deficitMB, doesNotFit, companionRejected). A rejected
// companion is distinct from a memory deficit: the caller disables speculation
// and recomputes the target-only placement instead of launching a known-bad pair.
// the caller feeds the deficit into the re-planner instead of paying a real
// load to learn the same thing. Any infrastructure failure (no binary,
// unsupported arch, parse error) skips the gate: the preflight must never
// block a launch the backend could have served.
func preflightPlacement(req *launchRequest, be *backendInfo, cfg *configForPreflight, caps *detect.Capabilities, model *placement.ModelProfile, strategy *placement.Strategy, serverArgs []string) (int, int, bool, bool) {
	if be == nil || caps == nil || len(caps.GPUs) == 0 {
		return -1, 0, false, false
	}
	cacheBackendTag := scopedProbeBackendTag(req, model, be)
	fitBin := findFitParamsBin(be.Path)
	if fitBin == "" {
		if isEmbeddedMainlineMTP(strategy) {
			fmt.Fprintln(os.Stderr, "[launch] embedded MTP has no selected-backend memory oracle; disabling speculation")
			return -1, 0, false, true
		}
		return -1, 0, false, false
	}
	targetDevs, err := runFitPreflight(fitBin, serverArgs)
	if err != nil {
		if isEmbeddedMainlineMTP(strategy) {
			fmt.Fprintf(os.Stderr, "[launch] embedded MTP target preflight failed; disabling speculation: %v\n", err)
			return -1, 0, false, true
		}
		fmt.Fprintf(os.Stderr, "[launch] preflight skipped: %v\n", err)
		return -1, 0, false, false
	}
	devs := targetDevs
	companionRejected := false
	if draftArgs := draftPreflightServerArgs(strategy); len(draftArgs) > 0 {
		draftDevs, draftErr := runFitPreflight(fitBin, draftArgs)
		if draftErr != nil {
			fmt.Fprintf(os.Stderr, "[launch] companion rejected by selected backend; disabling speculation: %v\n", draftErr)
			companionRejected = true
		} else {
			devs = mergePreflightDevices(targetDevs, draftDevs)
		}
	}
	if isEmbeddedMainlineMTP(strategy) {
		reservation, reserveErr := embeddedMTPPreflightReservation(model, strategy, targetDevs)
		if reserveErr != nil {
			fmt.Fprintf(os.Stderr, "[launch] embedded MTP memory cannot be proven; disabling speculation: %v\n", reserveErr)
			return -1, 0, false, true
		}
		devs = mergePreflightDevices(devs, reservation)
	}
	// Feed the backend's measured context and compute buffers back into placement
	// BEFORE checking fit, regardless of outcome. A re-plan below
	// (ReplanAfterOOM -> Compute) must see these real numbers immediately, not
	// the first-launch formulas that produced this (possibly wrong) strategy.
	if model != nil && strategy != nil {
		computeByGPU := map[int]int{}
		for _, d := range targetDevs {
			if idx, ok := cudaDeviceIndex(d.Name); ok {
				computeByGPU[idx] = d.ComputeMB
			}
		}
		placement.RecordMeasuredContextMB(cfg.CacheDir, model, strategy.ContextSize, strategy.KVType, preflightContextTotalMB(targetDevs))
		_ = placement.RecordMeasuredComputeBuffers(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, cacheBackendTag, caps.GPUs, strategy.Parallel, computeByGPU)
	}
	overheadByGPU := placement.SystemCUDAOverheadByGPU(cfg.CacheDir, caps.GPUs)
	var runtimeGrowthByGPU map[int]int
	if model != nil && strategy != nil {
		runtimeGrowthByGPU = placement.RuntimeGraphGrowthByGPU(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, cacheBackendTag, caps.GPUs, strategy.Parallel)
	}
	dev, deficit, summary := preflightWorstDeficit(devs, caps.GPUs, overheadByGPU, runtimeGrowthByGPU)
	if isEmbeddedMainlineMTP(strategy) && deficit > 0 {
		_, targetDeficit, _ := preflightWorstDeficit(targetDevs, caps.GPUs, overheadByGPU, runtimeGrowthByGPU)
		if targetDeficit == 0 {
			fmt.Fprintf(os.Stderr, "[launch] embedded MTP reservation does not fit (%s); disabling speculation and keeping the proven target placement\n", summary)
			return -1, 0, false, true
		}
	}
	if deficit > 0 {
		fmt.Fprintf(os.Stderr, "[launch] preflight: placement does not fit (%s) - re-planning before load\n", summary)
		if model != nil && model.IsMoE && strategy != nil {
			// The re-plan below may fall through pkg/placement's ubatch-fit
			// ladder (maximizeMoEGPUFitByUBatch). Without this, a ladder rung
			// that was never measured here falls back to the first-launch
			// heuristic — the same wrong-by-4x estimate that produced this
			// deficit in the first place, just at a different ubatch.
			measureUBatchLadderCandidates(fitBin, serverArgs, cfg, caps, model, strategy, cacheBackendTag)
		}
		return dev, deficit, true, companionRejected
	}
	fmt.Printf("[launch] preflight: placement fits (%s)\n", summary)
	return -1, 0, false, companionRejected
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
