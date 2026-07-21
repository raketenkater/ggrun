package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/memprobe"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
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
	Name          string // "CUDA0", "Host", ...
	ModelMB       int
	ContextMB     int
	ComputeMB     int
	UnaccountedMB int // allocator peak not identified by optional backend log labels
}

type memoryEvidenceLevel string

const (
	memoryEvidenceNone          memoryEvidenceLevel = "none"
	memoryEvidenceOraclePlanned memoryEvidenceLevel = "oracle-planned"
	memoryEvidenceAllocated     memoryEvidenceLevel = "allocation-verified"
)

type memoryPlanEvidence struct {
	Level    memoryEvidenceLevel
	Backend  string
	Devices  []preflightDevice
	Host     memprobe.HostMemory
	Coverage memprobe.Coverage
}

type preflightOutcome struct {
	Device            int
	AllocMB           int
	DeficitMB         int
	IsComputeBuffer   bool
	DoesNotFit        bool
	CompanionRejected bool
	Evidence          memoryPlanEvidence
	Err               error
}

const memoryEvidenceSchemaVersion = memprobe.SchemaVersion

type ikAllocationOOMError struct {
	Device          int
	AllocMB         int
	DeficitMB       int
	IsComputeBuffer bool
}

func (e *ikAllocationOOMError) Error() string {
	return fmt.Sprintf("guarded allocation probe CUDA%d allocation failed at %d MiB (exact deficit %d MiB)", e.Device, e.AllocMB, e.DeficitMB)
}

type liveMemoryProbeConsentError struct {
	Reason string
}

func (e *liveMemoryProbeConsentError) Error() string {
	return e.Reason + "; rerun with --allow-live-memory-probe or approve the interactive prompt"
}

var ikBufferLinePattern = regexp.MustCompile(`(CUDA[0-9]+|CUDA_Host|CPU|Host)[^=\n]*buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB`)

func parseIKAllocationDevices(logData string) []preflightDevice {
	byName := map[string]*preflightDevice{}
	var order []string
	for _, line := range strings.Split(logData, "\n") {
		match := ikBufferLinePattern.FindStringSubmatch(line)
		if match == nil {
			continue
		}
		name := match[1]
		if name == "CUDA_Host" || name == "CPU" {
			name = "Host"
		}
		value, err := strconv.ParseFloat(match[2], 64)
		if err != nil || value <= 0 {
			continue
		}
		mb := int(math.Ceil(value))
		device := byName[name]
		if device == nil {
			device = &preflightDevice{Name: name}
			byName[name] = device
			order = append(order, name)
		}
		lower := strings.ToLower(line)
		switch {
		case strings.Contains(lower, "compute buffer"):
			device.ComputeMB += mb
		case strings.Contains(lower, "kv buffer"), strings.Contains(lower, "recurrent buffer"):
			device.ContextMB += mb
		case strings.Contains(lower, "llm_load_tensors"):
			device.ModelMB += mb
		}
	}
	result := make([]preflightDevice, 0, len(order))
	for _, name := range order {
		device := byName[name]
		if device.TotalMB() > 0 {
			result = append(result, *device)
		}
	}
	return result
}

func memoryEvidenceKey(be *backendInfo, model *placement.ModelProfile, caps *detect.Capabilities, serverArgs []string) string {
	h := sha256.New()
	_, _ = io.WriteString(h, fmt.Sprintf("schema=%d\n", memoryEvidenceSchemaVersion))
	if be != nil {
		_, _ = io.WriteString(h, "backend="+be.Identity+"\n")
	}
	if model != nil {
		_, _ = io.WriteString(h, fmt.Sprintf("model=%s\nsize=%d\n", model.Path, model.SizeBytes))
		if stat, err := os.Stat(model.Path); err == nil {
			_, _ = io.WriteString(h, fmt.Sprintf("mtime=%d\n", stat.ModTime().UnixNano()))
		}
	}
	if caps != nil {
		for _, gpu := range caps.GPUs {
			_, _ = io.WriteString(h, fmt.Sprintf("gpu=%d:%s:%d\n", gpu.Index, gpu.Name, gpu.VRAMTotalMB))
		}
	}
	// The ik canary is sensitive to host-allocation flags such as --no-mmap,
	// which mainline fit-params intentionally strips. Hash the exact backend
	// argv; network-only differences merely create an extra safe cache entry.
	_, _ = io.WriteString(h, strings.Join(serverArgs, "\x00"))
	return hex.EncodeToString(h.Sum(nil)[:16])
}

func loadMemoryEvidence(cacheDir, key string) (memoryPlanEvidence, bool) {
	plan, ok := memprobe.Load(cacheDir, key)
	if !ok || plan.Evidence != memprobe.EvidenceGuardedAllocated || len(plan.Devices) == 0 {
		return memoryPlanEvidence{}, false
	}
	devices := make([]preflightDevice, 0, len(plan.Devices))
	for _, device := range plan.Devices {
		devices = append(devices, preflightDevice{
			Name:          device.ID,
			ModelMB:       bytesToMiBCeil(device.ModelBytes),
			ContextMB:     bytesToMiBCeil(device.ContextBytes),
			ComputeMB:     bytesToMiBCeil(device.ComputeBytes),
			UnaccountedMB: bytesToMiBCeil(device.UnaccountedBytes),
		})
	}
	if plan.Host.ModelBytes+plan.Host.ContextBytes+plan.Host.ComputeBytes+plan.Host.UnaccountedBytes > 0 {
		devices = append(devices, preflightDevice{
			Name:          "Host",
			ModelMB:       bytesToMiBCeil(plan.Host.ModelBytes),
			ContextMB:     bytesToMiBCeil(plan.Host.ContextBytes),
			ComputeMB:     bytesToMiBCeil(plan.Host.ComputeBytes),
			UnaccountedMB: bytesToMiBCeil(plan.Host.UnaccountedBytes),
		})
	}
	return memoryPlanEvidence{Level: memoryEvidenceAllocated, Backend: plan.BackendIdentity, Devices: devices, Host: plan.Host, Coverage: plan.Coverage}, true
}

func saveMemoryEvidence(cacheDir, key string, evidence memoryPlanEvidence) error {
	if cacheDir == "" || key == "" || len(evidence.Devices) == 0 {
		return nil
	}
	plan := memprobe.Plan{
		Key:             key,
		Evidence:        memprobe.EvidenceGuardedAllocated,
		BackendIdentity: evidence.Backend,
		Outcome:         "fit",
		Coverage:        memprobe.Coverage{GuardLoaded: true, DeviceAllocations: true, CgroupV2: true, Complete: true},
		Host:            evidence.Host,
	}
	for _, device := range evidence.Devices {
		if device.Name == "Host" {
			plan.Host.ModelBytes = uint64(maxPreflightInt(device.ModelMB, 0)) * 1024 * 1024
			plan.Host.ContextBytes = uint64(maxPreflightInt(device.ContextMB, 0)) * 1024 * 1024
			plan.Host.ComputeBytes = uint64(maxPreflightInt(device.ComputeMB, 0)) * 1024 * 1024
			plan.Host.UnaccountedBytes = uint64(maxPreflightInt(device.UnaccountedMB, 0)) * 1024 * 1024
			continue
		}
		plan.Devices = append(plan.Devices, memprobe.DeviceMemory{
			ID:               device.Name,
			PeakBytes:        uint64(maxPreflightInt(device.TotalMB(), 0)) * 1024 * 1024,
			ModelBytes:       uint64(maxPreflightInt(device.ModelMB, 0)) * 1024 * 1024,
			ContextBytes:     uint64(maxPreflightInt(device.ContextMB, 0)) * 1024 * 1024,
			ComputeBytes:     uint64(maxPreflightInt(device.ComputeMB, 0)) * 1024 * 1024,
			UnaccountedBytes: uint64(maxPreflightInt(device.UnaccountedMB, 0)) * 1024 * 1024,
		})
	}
	_, err := memprobe.Save(cacheDir, plan)
	return err
}

func bytesToMiBCeil(value uint64) int {
	if value == 0 {
		return 0
	}
	return int((value + 1024*1024 - 1) / (1024 * 1024))
}

func maxPreflightInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func setArgValue(args []string, name, value string) []string {
	out := append([]string(nil), args...)
	for i := 0; i+1 < len(out); i++ {
		if out[i] == name {
			out[i+1] = value
			return out
		}
	}
	return append(out, name, value)
}

func backendSupportsAllocationDryRun(be *backendInfo) bool {
	if be == nil {
		return false
	}
	for _, field := range strings.Fields(be.Help) {
		if strings.Trim(field, " ,;[]()") == "--dry-run" || strings.HasPrefix(field, "--dry-run=") {
			return true
		}
	}
	return false
}

func reconcileGuardedDevices(parsed []preflightDevice, summary memprobe.Summary) []preflightDevice {
	devices := append([]preflightDevice(nil), parsed...)
	byName := make(map[string]int, len(devices))
	for i, device := range devices {
		byName[device.Name] = i
	}
	for _, measured := range summary.DeviceSlice() {
		peakMB := bytesToMiBCeil(measured.PeakBytes)
		idx, ok := byName[measured.ID]
		if !ok {
			devices = append(devices, preflightDevice{Name: measured.ID, UnaccountedMB: peakMB})
			byName[measured.ID] = len(devices) - 1
			continue
		}
		if unexplained := peakMB - devices[idx].TotalMB(); unexplained > 0 {
			devices[idx].UnaccountedMB += unexplained
		}
	}
	return devices
}

func guardedPlanDevices(devices []preflightDevice, summary memprobe.Summary) ([]memprobe.DeviceMemory, memprobe.HostMemory) {
	measured := summary.Devices
	host := summary.Host
	for _, device := range devices {
		if device.Name == "Host" {
			host.ModelBytes = uint64(maxPreflightInt(device.ModelMB, 0)) * 1024 * 1024
			host.ContextBytes = uint64(maxPreflightInt(device.ContextMB, 0)) * 1024 * 1024
			host.ComputeBytes = uint64(maxPreflightInt(device.ComputeMB, 0)) * 1024 * 1024
			host.UnaccountedBytes = uint64(maxPreflightInt(device.UnaccountedMB, 0)) * 1024 * 1024
			continue
		}
		idx, ok := cudaDeviceIndex(device.Name)
		if !ok {
			continue
		}
		entry := measured[idx]
		entry.ID = device.Name
		entry.ModelBytes = uint64(maxPreflightInt(device.ModelMB, 0)) * 1024 * 1024
		entry.ContextBytes = uint64(maxPreflightInt(device.ContextMB, 0)) * 1024 * 1024
		entry.ComputeBytes = uint64(maxPreflightInt(device.ComputeMB, 0)) * 1024 * 1024
		entry.UnaccountedBytes = uint64(maxPreflightInt(device.UnaccountedMB, 0)) * 1024 * 1024
		measured[idx] = entry
	}
	ordered := memprobe.Summary{Devices: measured}.DeviceSlice()
	return ordered, host
}

func runGuardedAllocationPreflight(req *launchRequest, be *backendInfo, cfg *configForPreflight, caps *detect.Capabilities, model *placement.ModelProfile, serverArgs []string) (memoryPlanEvidence, error) {
	key := memoryEvidenceKey(be, model, caps, serverArgs)
	if evidence, ok := loadMemoryEvidence(cfg.CacheDir, key); ok {
		return evidence, nil
	}
	memoryMaxMB := backendMemoryMaxMB(req, caps)
	if memoryMaxMB <= 0 {
		return memoryPlanEvidence{}, fmt.Errorf("allocation probe requires a positive backend MemoryMax")
	}
	if _, err := os.Stat("/sys/fs/cgroup/cgroup.controllers"); err != nil {
		return memoryPlanEvidence{}, fmt.Errorf("allocation probe requires Linux cgroup v2 containment: %w", err)
	}
	dryRun := backendSupportsAllocationDryRun(be)
	guardLibrary := memprobe.FindGuardLibrary()
	if !dryRun && !req.AllowLiveMemoryProbe {
		return memoryPlanEvidence{}, &liveMemoryProbeConsentError{Reason: "the selected backend has no advertised --dry-run allocation mode, so measurement requires one contained live model load"}
	}
	if guardLibrary == "" && !req.AllowLiveMemoryProbe {
		return memoryPlanEvidence{}, &liveMemoryProbeConsentError{Reason: "the CUDA allocation firewall is unavailable, so only the slower cgroup-contained live fallback is possible"}
	}
	port, err := freeLoopbackPort()
	if err != nil {
		return memoryPlanEvidence{}, fmt.Errorf("allocate memory-probe port: %w", err)
	}
	args := setArgValue(serverArgs, "--port", strconv.Itoa(port))
	args = setArgValue(args, "--host", "127.0.0.1")
	if dryRun && !hasArg(args, "--dry-run") {
		args = append(args, "--dry-run")
	}
	envOverrides := []string{"GGML_CUDA_NO_PINNED=1"}
	guardLogPath := ""
	if guardLibrary != "" {
		guardLog, createErr := os.CreateTemp("", "ggrun-memory-probe-*.jsonl")
		if createErr != nil {
			return memoryPlanEvidence{}, fmt.Errorf("create allocation guard log: %w", createErr)
		}
		guardLogPath = guardLog.Name()
		_ = guardLog.Close()
		defer os.Remove(guardLogPath)
		overheadByGPU := placement.SystemCUDAOverheadByGPU(cfg.CacheDir, caps.GPUs)
		gpuLimitsMB := make([]int, len(caps.GPUs))
		for i, gpu := range caps.GPUs {
			gpuLimitsMB[i] = gpu.VRAMFreeMB() - overheadByGPU[gpu.Index]
			if gpuLimitsMB[i] < 0 {
				gpuLimitsMB[i] = 0
			}
		}
		envOverrides = memprobe.GuardEnvironment(guardLibrary, guardLogPath, gpuLimitsMB, 0, os.Getenv("LD_PRELOAD"))
	}
	timeout := 2 * time.Minute
	if !dryRun {
		timeout = autoStartupTimeout(model)
	}
	p, startErr := server.StartWithTimeoutToOptions(args, port, timeout, io.Discard, io.Discard, server.StartOptions{
		EnvOverrides: envOverrides,
		MemoryHighMB: memoryMaxMB,
		MemoryMaxMB:  memoryMaxMB,
	})
	logData := ""
	var cgroupPeakBytes uint64
	var cgroupOOMKills uint64
	cgroupStatsComplete := false
	if p != nil {
		logData = p.LogBuf.String()
		var peakErr, oomErr error
		cgroupPeakBytes, peakErr = p.MemoryPeakBytes()
		cgroupOOMKills, oomErr = p.MemoryOOMKillCount()
		cgroupStatsComplete = peakErr == nil && oomErr == nil
		_ = p.Stop()
	}
	summary := memprobe.Summary{Devices: map[int]memprobe.DeviceMemory{}}
	if guardLogPath != "" {
		var parseErr error
		summary, parseErr = memprobe.ParseGuardLog(guardLogPath)
		if parseErr != nil {
			return memoryPlanEvidence{}, fmt.Errorf("read CUDA allocation firewall evidence: %w", parseErr)
		}
		if summary.Denied != nil && (summary.Denied.Kind == "device" || summary.Denied.Kind == "managed") {
			deficitBytes := summary.Denied.ActiveBytes + summary.Denied.Bytes
			if deficitBytes > summary.Denied.LimitBytes {
				deficitBytes -= summary.Denied.LimitBytes
			} else {
				deficitBytes = 1
			}
			_, _, isComputeBuffer, _ := startupLogCUDAOOMDetailed(logData)
			return memoryPlanEvidence{}, &ikAllocationOOMError{
				Device:          summary.Denied.Device,
				AllocMB:         bytesToMiBCeil(summary.Denied.Bytes),
				DeficitMB:       bytesToMiBCeil(deficitBytes),
				IsComputeBuffer: isComputeBuffer,
			}
		}
	}
	summary.Host.CgroupPeakBytes = cgroupPeakBytes
	summary.Host.CgroupLimitBytes = uint64(memoryMaxMB) * 1024 * 1024
	if startErr != nil {
		if cgroupOOMKills > 0 {
			return memoryPlanEvidence{}, fmt.Errorf("contained backend exceeded the %d MiB host-memory cap (cgroup peak %d MiB, oom_kill=%d)", memoryMaxMB, bytesToMiBCeil(cgroupPeakBytes), cgroupOOMKills)
		}
		if device, allocMB, isComputeBuffer, ok := startupLogCUDAOOMDetailed(logData); ok {
			deficit := oomOvershoot(caps, device, allocMB)
			if deficit < 1 {
				deficit = 1
			}
			return memoryPlanEvidence{}, &ikAllocationOOMError{
				Device: device, AllocMB: allocMB, DeficitMB: deficit, IsComputeBuffer: isComputeBuffer,
			}
		}
		return memoryPlanEvidence{}, fmt.Errorf("contained backend memory probe did not complete: %w", startErr)
	}
	parsed := parseIKAllocationDevices(logData)
	coverageComplete := summary.Loaded && summary.DeviceEvents && guardLibrary != "" && cgroupStatsComplete
	devices := reconcileGuardedDevices(parsed, summary)
	if !coverageComplete && !req.AllowLiveMemoryProbe {
		return memoryPlanEvidence{}, &liveMemoryProbeConsentError{Reason: "the automatic memory probe completed without full CUDA/cgroup evidence, so only an explicitly approved one-use result is available"}
	}
	level := memoryEvidenceAllocated
	if !coverageComplete {
		level = memoryEvidenceOraclePlanned
	}
	coverage := memprobe.Coverage{
		GuardLoaded: summary.Loaded, DeviceAllocations: summary.DeviceEvents,
		PinnedAllocations: summary.PinnedEvents, CgroupV2: cgroupStatsComplete,
		Complete: coverageComplete,
	}
	evidence := memoryPlanEvidence{Level: level, Backend: be.Identity, Devices: devices, Host: summary.Host, Coverage: coverage}
	if len(evidence.Devices) == 0 {
		return memoryPlanEvidence{}, fmt.Errorf("contained backend probe reached health but produced neither allocator events nor parseable memory buffers")
	}
	if coverageComplete {
		planDevices, host := guardedPlanDevices(devices, summary)
		host.CgroupLimitBytes = uint64(memoryMaxMB) * 1024 * 1024
		plan := memprobe.Plan{
			Key:             key,
			Evidence:        memprobe.EvidenceGuardedAllocated,
			BackendIdentity: be.Identity,
			ModelIdentity:   model.Path,
			ShapeIdentity:   key,
			Outcome:         "fit",
			Coverage:        coverage,
			Devices:         planDevices,
			Host:            host,
			BackendArgs:     append([]string(nil), serverArgs...),
		}
		if _, err := memprobe.Save(cfg.CacheDir, plan); err != nil {
			fmt.Fprintf(os.Stderr, "[launch] warning: could not persist guarded allocation evidence: %v\n", err)
		}
	} else {
		fmt.Fprintln(os.Stderr, "[launch] warning: allocation coverage was incomplete; using this explicitly approved live result once and not caching it")
	}
	return evidence, nil
}

// TotalMB is the device's planned VRAM demand at load time.
func (d preflightDevice) TotalMB() int {
	return d.ModelMB + d.ContextMB + d.ComputeMB + d.UnaccountedMB
}

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
	oldLibraryPath := ""
	for _, e := range env {
		if !strings.HasPrefix(e, "CUDA_DEVICE_ORDER=") {
			if strings.HasPrefix(e, "LD_LIBRARY_PATH=") {
				oldLibraryPath = strings.TrimPrefix(e, "LD_LIBRARY_PATH=")
				continue
			}
			filtered = append(filtered, e)
		}
	}
	libraryPath := fitPreflightLibraryPath(fitBin)
	if oldLibraryPath != "" {
		libraryPath += string(os.PathListSeparator) + oldLibraryPath
	}
	cmd.Env = append(filtered,
		"CUDA_DEVICE_ORDER=PCI_BUS_ID",
		"LD_LIBRARY_PATH="+libraryPath,
	)

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

func fitPreflightLibraryPath(fitBin string) string {
	dirs := []string{filepath.Dir(fitBin)}
	if resolved, err := filepath.EvalSymlinks(fitBin); err == nil {
		resolvedDir := filepath.Dir(resolved)
		if resolvedDir != dirs[0] {
			dirs = append([]string{resolvedDir}, dirs...)
		}
	}
	return strings.Join(dirs, string(os.PathListSeparator))
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
			merged.UnaccountedMB += d.UnaccountedMB
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

// preflightPlacement runs the selected backend's memory gate for one launch attempt. It
// returns a normalized outcome with the evidence level that supports it. A rejected
// companion is distinct from a memory deficit: the caller disables speculation
// and recomputes the target-only placement instead of launching a known-bad pair.
// the caller feeds the deficit into the re-planner instead of paying a real
// load to learn the same thing. Mainline oracle infrastructure failures remain
// best-effort; an ik allocation canary failure is blocking because treating an
// incomplete contained allocation as proof would immediately repeat it in the
// real server.
func preflightPlacement(req *launchRequest, be *backendInfo, cfg *configForPreflight, caps *detect.Capabilities, model *placement.ModelProfile, strategy *placement.Strategy, serverArgs []string) preflightOutcome {
	outcome := preflightOutcome{Device: -1, Evidence: memoryPlanEvidence{Level: memoryEvidenceNone}}
	if be == nil || caps == nil || len(caps.GPUs) == 0 {
		return outcome
	}
	cacheBackendTag := scopedProbeBackendTag(req, model, be)
	fitBin := findFitParamsBin(be.Path)
	allocationProbe := false
	var targetDevs []preflightDevice
	if fitBin == "" {
		allocationProbe = true
		evidence, err := runGuardedAllocationPreflight(req, be, cfg, caps, model, serverArgs)
		if err != nil {
			if oom, ok := err.(*ikAllocationOOMError); ok {
				return allocationOOMOutcome(outcome, oom)
			}
			outcome.Err = err
			return outcome
		}
		outcome.Evidence = evidence
		targetDevs = evidence.Devices
	} else {
		var err error
		targetDevs, err = runFitPreflight(fitBin, serverArgs)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[launch] selected-backend memory oracle failed; switching to contained allocation probe: %v\n", err)
			allocationProbe = true
			fitBin = ""
			evidence, allocationErr := runGuardedAllocationPreflight(req, be, cfg, caps, model, serverArgs)
			if allocationErr != nil {
				if oom, ok := allocationErr.(*ikAllocationOOMError); ok {
					return allocationOOMOutcome(outcome, oom)
				}
				outcome.Err = allocationErr
				return outcome
			}
			outcome.Evidence = evidence
			targetDevs = evidence.Devices
		} else {
			outcome.Evidence = memoryPlanEvidence{
				Level:   memoryEvidenceOraclePlanned,
				Backend: be.Tag,
				Devices: append([]preflightDevice(nil), targetDevs...),
			}
		}
	}
	devs := targetDevs
	companionRejected := false
	if !allocationProbe {
		if draftArgs := draftPreflightServerArgs(strategy); len(draftArgs) > 0 {
			draftDevs, draftErr := runFitPreflight(fitBin, draftArgs)
			if draftErr != nil {
				fmt.Fprintf(os.Stderr, "[launch] companion rejected by selected backend; disabling speculation: %v\n", draftErr)
				companionRejected = true
			} else {
				devs = mergePreflightDevices(targetDevs, draftDevs)
			}
		}
	}
	if !allocationProbe && isEmbeddedMainlineMTP(strategy) {
		reservation, reserveErr := embeddedMTPPreflightReservation(model, strategy, targetDevs)
		if reserveErr != nil {
			fmt.Fprintf(os.Stderr, "[launch] embedded MTP memory cannot be proven; disabling speculation: %v\n", reserveErr)
			outcome.CompanionRejected = true
			return outcome
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
			outcome.CompanionRejected = true
			return outcome
		}
	}
	if deficit > 0 {
		fmt.Fprintf(os.Stderr, "[launch] preflight: placement does not fit (%s) - re-planning before load\n", summary)
		if fitBin != "" && model != nil && model.IsMoE && strategy != nil {
			// The re-plan below may fall through pkg/placement's ubatch-fit
			// ladder (maximizeMoEGPUFitByUBatch). Without this, a ladder rung
			// that was never measured here falls back to the first-launch
			// heuristic — the same wrong-by-4x estimate that produced this
			// deficit in the first place, just at a different ubatch.
			measureUBatchLadderCandidates(fitBin, serverArgs, cfg, caps, model, strategy, cacheBackendTag)
		}
		outcome.Device = dev
		outcome.DeficitMB = deficit
		outcome.DoesNotFit = true
		outcome.CompanionRejected = companionRejected
		return outcome
	}
	fmt.Printf("[launch] preflight: placement fits (%s)\n", summary)
	outcome.CompanionRejected = companionRejected
	return outcome
}

func allocationOOMOutcome(outcome preflightOutcome, oom *ikAllocationOOMError) preflightOutcome {
	if oom == nil {
		return outcome
	}
	outcome.Device = oom.Device
	outcome.AllocMB = maxPreflightInt(oom.AllocMB, 1)
	outcome.DeficitMB = maxPreflightInt(oom.DeficitMB, 1)
	outcome.IsComputeBuffer = oom.IsComputeBuffer
	outcome.DoesNotFit = true
	return outcome
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
