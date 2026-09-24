package main

import (
	"bytes"
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

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/memprobe"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
	"net"
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
	Device  int
	AllocMB int
	// AllocMBMeasured records that AllocMB is a size the backend actually
	// printed, not one synthesized from the deficit. Only a real reported
	// allocation may be written back as compute-buffer evidence: a graph-capture
	// abort (cudaGraphInstantiate) names no size at all, and storing the
	// stand-in as a measurement teaches the planner a compute buffer that was
	// never observed — in either direction.
	AllocMBMeasured   bool
	DeficitMB         int
	IsComputeBuffer   bool
	DoesNotFit        bool
	CompanionRejected bool
	Evidence          memoryPlanEvidence
	Err               error
	BackendAdjustment *backendLaunchAdjustment
	// ProbeUnavailable is set when the host cannot run the guarded allocation
	// probe at all, as opposed to running it and learning something. It is not
	// an error: a launch continues on the planner's estimate, exactly as it does
	// on a machine with no GPU. `ggrun memory-probe` asked for the measurement
	// specifically, so it still treats this as a failure.
	ProbeUnavailable string
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

type backendLaunchAdjustment struct {
	RemoveFlag string
	KVQuality  string
	// KVQualityV promotes only the V-cache leg (--cache-type-v) to the given
	// type while leaving K at its requested quant. Some backends reject a
	// quantized V cache outright ("model does not support a quantized V cache
	// (q4_0)") even when a compressed K is fine; the V leg is the only one that
	// must be forced to f16.
	KVQualityV string
	Feature    string
	// Advisory marks an adjustment learned from a backend that WARNED and then
	// started successfully, as opposed to one that refused to start. The two
	// need different handling for an explicitly user-supplied flag: refusing to
	// override the user is right when the launch is already dead, but aborting a
	// launch the backend was perfectly willing to run would turn a working
	// command line into an error.
	Advisory      bool
	RequireReplan bool
	Reason        string
}

type backendLaunchAdjustmentError struct {
	Adjustment backendLaunchAdjustment
}

func (e *backendLaunchAdjustmentError) Error() string {
	return e.Adjustment.Reason
}

type liveMemoryProbeConsentError struct {
	Reason string
}

func (e *liveMemoryProbeConsentError) Error() string {
	return e.Reason + "; rerun with --allow-live-memory-probe or approve the interactive prompt"
}

// waitForPredecessorPort blocks until the requested server port is bindable,
// so placement is not computed against the VRAM of a predecessor still
// exiting. The settle rounds in detect cannot catch this: a server that has
// not started releasing yet reads as perfectly stable, and the information
// that its 43 GB is about to vanish is not in the numbers. The port is: if the
// previous server still holds it, the machine ggrun is about to measure is not
// the machine it will launch into -- and launching would fail to bind anyway.
// Returns false if the port never freed within the timeout, which is worth a
// warning but not an abort: the holder may be an unrelated persistent server,
// and then its VRAM use is genuinely occupied and the plan is right to see it.
func waitForPredecessorPort(port int, timeout time.Duration, warn io.Writer) bool {
	if port <= 0 {
		return true
	}
	deadline := time.Now().Add(timeout)
	waited := false
	for {
		ln, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
		if err == nil {
			_ = ln.Close()
			if waited {
				// The process releases the port before the driver finishes
				// releasing its VRAM; give the frees a moment to land and let
				// the normal settle rounds absorb the tail.
				time.Sleep(1500 * time.Millisecond)
			}
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		if !waited {
			fmt.Fprintf(warn, "[launch] port %d is still held by the previous server; waiting for it to exit so placement sees the real free VRAM\n", port)
			waited = true
		}
		time.Sleep(500 * time.Millisecond)
	}
}

// memoryProbeUnavailableError marks a host that can never run the guarded
// allocation probe, as distinct from a probe that ran and found a deficit.
//
// The probe contains a live model load in a cgroup v2 memory scope, which only
// exists on Linux -- backendMemoryMaxMB returns 0 off Linux by construction. So
// on macOS/Metal and Windows/CUDA, both supported backends, every launch that
// reached this path failed outright with "requires a positive backend
// MemoryMax". Refusing to launch because the *optional* measurement is
// impossible rejects the platform rather than protecting it; a host with no GPU
// already skips preflight and launches on the planner's estimate, and this is
// the same situation.
type memoryProbeUnavailableError struct {
	Reason string
}

func (e *memoryProbeUnavailableError) Error() string { return e.Reason }

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
		_, _ = io.WriteString(h, "model="+placement.SpecTargetIdentity(model)+"\n")
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
	// CUDA guard events and the cgroup peak are two halves of the same exact
	// allocation proof. Preserve an unlabelled host peak in the returned device
	// rows as well as the persisted memprobe plan, because these rows are fed
	// into placement immediately during this launch.
	if peakMB := bytesToMiBCeil(summary.Host.CgroupPeakBytes); peakMB > 0 {
		if idx, ok := byName["Host"]; ok {
			if unexplained := peakMB - devices[idx].TotalMB(); unexplained > 0 {
				devices[idx].UnaccountedMB += unexplained
			}
		} else {
			devices = append(devices, preflightDevice{Name: "Host", UnaccountedMB: peakMB})
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
	// cgroup peak is the exact resident host footprint of the contained process.
	// Backends do not consistently label host model buffers (qwen4exp PLE is one
	// example), so retain every peak byte not explained by parsed model/context/
	// compute rows as unaccounted instead of silently reducing a 60+ GiB process
	// to its small host compute buffer.
	accountedHost := host.ModelBytes + host.ContextBytes + host.ComputeBytes + host.UnaccountedBytes
	if host.CgroupPeakBytes > accountedHost {
		host.UnaccountedBytes += host.CgroupPeakBytes - accountedHost
	}
	ordered := memprobe.Summary{Devices: measured}.DeviceSlice()
	return ordered, host
}

// persistFailedAllocationProbe keeps the complete backend and firewall evidence
// for a load that could not be classified automatically. The terminal's final
// 20 lines are usually just tensor-override bookkeeping followed by
// "unable to load model"; the actionable allocator/loader error can be hundreds
// of lines earlier. Keeping one file per exact memory-evidence key makes the
// failure debuggable without weakening containment or repeating a 100+ GiB load
// merely to recover stderr.
func persistFailedAllocationProbe(cacheDir, key, backendLog, guardLogPath string) string {
	if strings.TrimSpace(cacheDir) == "" || strings.TrimSpace(key) == "" {
		return ""
	}
	dir := filepath.Join(cacheDir, "memory-probes")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return ""
	}
	var data strings.Builder
	data.WriteString(backendLog)
	if data.Len() > 0 && !strings.HasSuffix(backendLog, "\n") {
		data.WriteByte('\n')
	}
	if guardLogPath != "" {
		if guardData, err := os.ReadFile(guardLogPath); err == nil && len(guardData) > 0 {
			data.WriteString("\n--- ggrun CUDA allocation firewall ---\n")
			data.Write(guardData)
			if guardData[len(guardData)-1] != '\n' {
				data.WriteByte('\n')
			}
		}
	}
	path := filepath.Join(dir, "failed-"+key+".log")
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(data.String()), 0o600); err != nil {
		return ""
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return ""
	}
	return path
}

func backendLoadFailureDiagnostic(logData string) string {
	best := ""
	for _, line := range strings.Split(logData, "\n") {
		trimmed := strings.TrimSpace(line)
		lower := strings.ToLower(trimmed)
		if trimmed == "" || strings.Contains(lower, "unable to load model") {
			continue
		}
		if strings.Contains(lower, "failed") || strings.Contains(lower, "error:") ||
			strings.Contains(lower, "cannot allocate") || strings.Contains(lower, "unable to allocate") ||
			strings.Contains(lower, "exception") {
			best = trimmed
		}
	}
	if len(best) > 500 {
		best = best[:500]
	}
	return best
}

// modelFormatRejectionMarkers are llama.cpp loader errors about the GGUF's own
// metadata or tensor layout. They mean the backend predates the model's format
// (for example a newer revision of a supported architecture that adds layers),
// so no memory, context or KV setting can make the load succeed. Allocation
// failures ("unable to allocate ... buffer") are deliberately absent: they are
// reported through the same "error loading model" prefix but are memory errors.
var modelFormatRejectionMarkers = []string{
	"error loading model hyperparameters",
	"error loading model architecture",
	"error loading model vocabulary",
	"unknown model architecture",
	"wrong array length",
	"missing tensor",
	"has wrong shape",
	"wrong number of tensors",
}

// modelFormatRejection returns the backend's loader line when it rejected the
// model file itself, or "" when the log shows no such rejection.
func modelFormatRejection(logData string) string {
	for _, line := range strings.Split(logData, "\n") {
		lower := strings.ToLower(line)
		for _, marker := range modelFormatRejectionMarkers {
			if !strings.Contains(lower, marker) {
				continue
			}
			detail := strings.TrimSpace(line)
			if i := strings.LastIndex(lower, "error loading model"); i >= 0 {
				detail = strings.TrimSpace(line[i:])
			}
			if len(detail) > 300 {
				detail = detail[:300]
			}
			return detail
		}
	}
	return ""
}

// backendModelFormatError marks a backend that cannot read the model file. It
// is not a memory result: the launch must not suggest memory controls, and the
// actionable fix is a newer backend.
type backendModelFormatError struct {
	Backend string
	Detail  string
	LogPath string
}

func (e *backendModelFormatError) Error() string {
	msg := fmt.Sprintf("backend %s cannot read this model file; it is older than the model's format: %s. Update it with: ggrun backend update", e.Backend, e.Detail)
	if e.LogPath != "" {
		msg += " (backend log: " + e.LogPath + ")"
	}
	return msg
}

// backendUnclassifiedProbeError marks a contained allocation preflight that
// failed with a backend/model error no ggrun rule classified: no memory OOM
// (device or firewall/cgroup), no backendAdjustmentFromLog rule, and no known
// flag. It carries a bounded excerpt of the backend's own startup log so the
// crash-diagnosis hook can ask the advisor what happened. It is advisory-only —
// the wrapped Cause is the real launch error and nothing about this type changes
// launch behavior.
type backendUnclassifiedProbeError struct {
	// LogExcerpt is a bounded excerpt of the backend startup log around the
	// actionable allocator/loader error, for the advisor's diagnosis. Never fed
	// back into a launch command line.
	LogExcerpt string
	Cause      error
}

func (e *backendUnclassifiedProbeError) Error() string {
	if e == nil || e.Cause == nil {
		return "backend launch failure was not classified"
	}
	return e.Cause.Error()
}

func (e *backendUnclassifiedProbeError) Unwrap() error {
	return e.Cause
}

// backendUnclassifiedLogExcerpt returns a bounded excerpt of a backend startup
// log for the crash-diagnosis hook. It prefers a window around the actionable
// error line over a blind tail: the allocator/loader error can be hundreds of
// lines above the generic "unable to load model" tail (the comment on
// persistFailedAllocationProbe makes the same point). Falls back to the log's
// tail when no error line is found.
func backendUnclassifiedLogExcerpt(logData string) string {
	if strings.TrimSpace(logData) == "" {
		return ""
	}
	detail := backendLoadFailureDiagnostic(logData)
	if detail != "" {
		if idx := strings.Index(logData, detail); idx >= 0 {
			const window = 400
			start := idx - window
			if start < 0 {
				start = 0
			}
			end := idx + len(detail) + window
			if end > len(logData) {
				end = len(logData)
			}
			return strings.TrimSpace(logData[start:end])
		}
	}
	const maxTail = 2048
	if len(logData) > maxTail {
		logData = logData[len(logData)-maxTail:]
	}
	return strings.TrimSpace(logData)
}

// backendAdjustmentFromLog classifies options that a backend accepts during
// parsing but rejects only after it has inspected the model or created its
// context. Each rule removes one optional flag generated by ggrun; callers
// separately refuse to override an explicit user passthrough.
func backendAdjustmentFromLog(logData string) *backendLaunchAdjustment {
	lower := strings.ToLower(logData)
	if (strings.Contains(lower, "swa_full is not supported") ||
		strings.Contains(lower, "swa-full is not supported") ||
		strings.Contains(lower, "full swa cache is not supported") ||
		strings.Contains(lower, "full swa cache is unsupported")) &&
		(strings.Contains(lower, "disabl") || strings.Contains(lower, "unsupported") || strings.Contains(lower, "not supported")) {
		return &backendLaunchAdjustment{
			RemoveFlag:    "--swa-full",
			Feature:       "swa-full",
			RequireReplan: true,
			Reason:        "selected backend reports that this model cannot use full SWA cache",
		}
	}
	if strings.Contains(lower, "deepseek4 k-cache supports only f16, bf16, and q8_0") &&
		strings.Contains(lower, "requested ") {
		return &backendLaunchAdjustment{
			KVQuality: "q8_0",
			Reason:    "selected backend reports that DeepSeek4 K-cache supports only F16, BF16, or Q8_0",
		}
	}
	if strings.Contains(lower, "k-cache hadamard is not supported") &&
		strings.Contains(lower, "use an untransformed k-cache") {
		return &backendLaunchAdjustment{
			RemoveFlag: "-khad",
			Reason:     "selected backend reports that this model does not support K-cache Hadamard",
		}
	}
	if strings.Contains(lower, "quantized v cache") &&
		(strings.Contains(lower, "not support") || strings.Contains(lower, "unsupported")) {
		return &backendLaunchAdjustment{
			KVQualityV: "f16",
			Reason:     "selected backend reports that this model does not support a quantized V cache; promoting only the V cache to f16",
		}
	}
	return nil
}

func runGuardedAllocationPreflight(req *launchRequest, be *backendInfo, cfg *configForPreflight, caps *detect.Capabilities, model *placement.ModelProfile, serverArgs []string) (memoryPlanEvidence, error) {
	key := memoryEvidenceKey(be, model, caps, serverArgs)
	if evidence, ok := loadMemoryEvidence(cfg.CacheDir, key); ok {
		return evidence, nil
	}
	memoryMaxMB := backendMemoryMaxMB(req, caps)
	if memoryMaxMB <= 0 {
		return memoryPlanEvidence{}, &memoryProbeUnavailableError{
			Reason: "allocation probe needs a positive backend MemoryMax, which only Linux hosts report",
		}
	}
	if _, err := os.Stat("/sys/fs/cgroup/cgroup.controllers"); err != nil {
		return memoryPlanEvidence{}, &memoryProbeUnavailableError{
			Reason: fmt.Sprintf("allocation probe needs Linux cgroup v2 containment: %v", err),
		}
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
		// Total host allocation is already fail-closed under this probe's cgroup
		// MemoryMax. Do not add a second pinned-only ceiling: zero in memguard means
		// deny all, not unlimited, and ik_llama's CPU expert buffer legitimately
		// uses CUDA host registration even with GGML_CUDA_NO_PINNED requested.
		envOverrides = memprobe.GuardEnvironment(guardLibrary, guardLogPath, gpuLimitsMB, 0, os.Getenv("LD_PRELOAD"))
	}
	timeout := allocationProbeTimeout(model)
	// Same scope regime as the production launch (backendStartOptions): under
	// mmap the plan's full file-backed footprint is charged to the cgroup as
	// reclaimable page cache, so a hard cap at the resident budget OOM-kills a
	// probe the launch would survive. Routing through the shared builder keeps
	// probe and launch from ever diverging.
	probeOpts := backendStartOptions(req, caps, be, envOverrides, args)
	p, startErr := server.StartWithTimeoutToOptions(args, port, timeout, io.Discard, io.Discard, probeOpts)
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
		failureLogPath := persistFailedAllocationProbe(cfg.CacheDir, key, logData, guardLogPath)
		withEvidence := func(message string) string {
			if failureLogPath == "" {
				return message
			}
			return message + " (full probe evidence: " + failureLogPath + ")"
		}
		if summary.Denied != nil && (summary.Denied.Kind == "pinned" || summary.Denied.Kind == "mlock") {
			message := fmt.Sprintf(
				"contained backend allocation firewall denied %s host allocation of %d MiB at %d MiB active (limit %d MiB)",
				summary.Denied.Kind, bytesToMiBCeil(summary.Denied.Bytes),
				bytesToMiBCeil(summary.Denied.ActiveBytes), bytesToMiBCeil(summary.Denied.LimitBytes),
			)
			return memoryPlanEvidence{}, fmt.Errorf("%s: %w", withEvidence(message), startErr)
		}
		if cgroupOOMKills > 0 {
			return memoryPlanEvidence{}, fmt.Errorf("%s", withEvidence(fmt.Sprintf("contained backend exceeded the %d MiB host-memory cap (cgroup peak %d MiB, oom_kill=%d)", memoryMaxMB, bytesToMiBCeil(cgroupPeakBytes), cgroupOOMKills)))
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
		// A loader that rejects the file itself cannot be helped by any launch
		// flag, so it is reported before flag adjustments are considered.
		if detail := modelFormatRejection(logData); detail != "" {
			return memoryPlanEvidence{}, &backendModelFormatError{Backend: be.Path, Detail: detail, LogPath: failureLogPath}
		}
		// Resource failures take precedence when more than one diagnostic is
		// present. Only mutate optional launch flags once the firewall, cgroup,
		// and CUDA log all agree that allocation itself succeeded.
		if adjustment := backendAdjustmentFromLog(logData); adjustment != nil {
			return memoryPlanEvidence{}, &backendLaunchAdjustmentError{Adjustment: *adjustment}
		}
		message := failedAllocationPreflightMessage(logData, failureLogPath)
		cause := fmt.Errorf("%s: %w", message, startErr)
		// The unclassified path: no OOM, no firewall/cgroup rejection, and no
		// backendAdjustmentFromLog rule matched the log. Mark it explicitly so the
		// launch caller can consult the advisor for a diagnosis. The wrapper is
		// advisory-only and wraps the exact error the caller would have seen.
		return memoryPlanEvidence{}, &backendUnclassifiedProbeError{LogExcerpt: backendUnclassifiedLogExcerpt(logData), Cause: cause}
	}
	// Some model/backend incompatibilities are reported as startup warnings and
	// the server continues after silently disabling the feature. That is not a
	// successful verification of the requested memory shape. This log contains
	// only the contained startup/model-load phase (no user requests), so it is a
	// trusted place to convert the warning into a typed full re-plan.
	if adjustment := backendAdjustmentFromLog(logData); adjustment != nil {
		return memoryPlanEvidence{}, &backendLaunchAdjustmentError{Adjustment: *adjustment}
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
		return memoryPlanEvidence{}, fmt.Errorf("contained backend probe reached health but produced neither allocator events nor parseable memory buffers; this is unexpected — try again, or check that the selected backend and its memory guard library installed correctly")
	}
	if coverageComplete {
		planDevices, host := guardedPlanDevices(devices, summary)
		host.CgroupLimitBytes = uint64(memoryMaxMB) * 1024 * 1024
		plan := memprobe.Plan{
			Key:             key,
			Evidence:        memprobe.EvidenceGuardedAllocated,
			BackendIdentity: be.Identity,
			ModelIdentity:   placement.SpecTargetIdentity(model),
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

// failedAllocationPreflightMessage keeps an unclassified backend failure from
// becoming a dead end. It retains the backend's strongest diagnostic and the
// persisted evidence path, then names bounded launch controls that can avoid a
// cache/shape incompatibility while the optional advisor is unavailable.
func failedAllocationPreflightMessage(logData, failureLogPath string) string {
	message := "contained backend memory probe did not complete"
	if detail := backendLoadFailureDiagnostic(logData); detail != "" {
		message += ": " + detail
	}
	message += "; try --kv-quality f16, a smaller --ctx-size, or --kv-placement cpu"
	if failureLogPath != "" {
		message += " (full probe evidence: " + failureLogPath + ")"
	}
	return message
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
//
// modelArch guards against pairing a fork-only architecture with mainline's
// fit-params. A fork backend (e.g. nanbeige42, laguna) may have no fit-params
// sibling of its own, and mainline llama-fit-params cannot load the fork-only
// architecture — it aborts with "unknown model architecture: 'nanbeige'". ggrun
// already knows the served GGUF's architecture, so a candidate that cannot load
// it is rejected rather than run (which would crash the whole launch).
func findFitParamsBin(serverBin, modelArch string) string {
	if serverBin == "" {
		return ""
	}
	okForArch := func(path string) bool {
		if modelArch == "" {
			return true
		}
		supported, probed := backends.BackendSupportsArch(path, modelArch)
		// A failed probe must not block a launch: we could not answer, so do
		// not refuse on the unknown (the fork path the user selected still gets
		// to try its own measurement or the contained probe fallback).
		return !probed || supported
	}
	var candidates []string
	if resolved, err := filepath.EvalSymlinks(serverBin); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(resolved), "llama-fit-params"))
	}
	candidates = append(candidates, filepath.Join(filepath.Dir(serverBin), "llama-fit-params"))
	for _, c := range candidates {
		if fi, err := os.Stat(c); err == nil && !fi.IsDir() {
			if okForArch(c) {
				return c
			}
			// The sibling exists but cannot load this architecture — keep
			// looking (a sibling of the fork build dir is preferred but must
			// be compatible) before falling back to PATH.
			continue
		}
	}
	// A PATH fallback is safe only when the server was itself selected by name.
	// For an absolute/custom fork path it could pair a fork with mainline's
	// fit-params and produce false compatibility or memory results.
	if filepath.Base(serverBin) == serverBin {
		if p, err := exec.LookPath("llama-fit-params"); err == nil && okForArch(p) {
			return p
		}
	}
	return ""
}

// preflightArgValueFlags are the launch flags that shape memory allocation.
// Server networking, sampling and logging are omitted. Memory policy flags
// must survive even when a particular fit-params build rejects them: an oracle
// error selects the contained exact-argv probe, whereas dropping a flag can
// incorrectly admit a different allocation shape.
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
	"--override-kv": true,
}

// These switches change KV placement, graph shape or host/model allocation.
var preflightBoolFlags = map[string]bool{
	"--swa-full": true,
	"-kvo":       true, "--kv-offload": true, "-nkvo": true, "--no-kv-offload": true,
	"-kvu": true, "--kv-unified": true, "-no-kvu": true, "--no-kv-unified": true,
	"--op-offload": true, "--no-op-offload": true,
	"--mmap": true, "--no-mmap": true, "--mlock": true,
	"--repack": true, "-nr": true, "--no-repack": true, "--no-host": true,
}

// preflightArgs preserves order, including repeated last-wins overrides.
func preflightArgs(serverArgs []string) []string {
	out := []string{"--fit-print", "on"}
	for i := 0; i < len(serverArgs); i++ {
		a := serverArgs[i]
		if preflightBoolFlags[a] {
			out = append(out, a)
			continue
		}
		flag, value, inline := strings.Cut(a, "=")
		if inline && preflightBoolFlags[flag] {
			out = append(out, a) // Preserve backend-specific boolean syntax or rejection.
			continue
		}
		if !preflightArgValueFlags[flag] {
			continue
		}
		if inline {
			out = append(out, flag, value)
			continue
		}
		if i+1 >= len(serverArgs) {
			out = append(out, flag) // Let the oracle reject the malformed shape.
			continue
		}
		value = serverArgs[i+1]
		// A negative numeric value (for example -ngl -1) is a value, not a flag.
		// Do not consume a following option when the value is missing.
		if strings.HasPrefix(value, "-") {
			if _, err := strconv.Atoi(value); err != nil {
				out = append(out, flag)
				continue
			}
		}
		out = append(out, flag, value)
		i++
	}
	return out
}

// runFitPreflight executes the no-alloc accounting and parses the per-device
// rows. serverArgs are the real launch args (binary path at index 0).
// runFitPreflight returns the measured per-device rows and the backend's own
// stderr. The stderr matters: llama.cpp reports "swa_full is not supported by
// this model" as a load-time WARNING on an otherwise successful start, so it
// never reached backendAdjustmentFromLog, which only ran on startup failures.
// The flag then survived the whole launch. Measured 2026-08-02 on deepseek4:
// carrying --swa-full cost 6364 MiB of context instead of 871 MiB — 5.5 GiB of
// the main GPU spent on a feature the model silently ignored.
func runFitPreflight(fitBin string, serverArgs []string) ([]preflightDevice, string, error) {
	args := server.LoadModeArgs(append([]string{fitBin}, preflightArgs(serverArgs[1:])...))[1:]
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

	var stderr bytes.Buffer
	cmd.Stderr = &stderr
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
		return nil, stderr.String(), fmt.Errorf("fit-params preflight timed out; falling back automatically for this launch")
	}
	if err != nil {
		detail := strings.TrimSpace(stderr.String())
		if detail != "" {
			const maxDetail = 600
			if len(detail) > maxDetail {
				detail = detail[len(detail)-maxDetail:]
			}
			return nil, stderr.String(), fmt.Errorf("fit-params preflight failed: %w: %s", err, detail)
		}
		return nil, stderr.String(), fmt.Errorf("fit-params preflight failed: %w", err)
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
		return nil, stderr.String(), fmt.Errorf("fit-params preflight produced no device rows")
	}
	return devs, stderr.String(), nil
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
func backendSpecCandidateValidator(be *backendInfo, modelArch string) func(string) error {
	if be == nil {
		return nil
	}
	fitBin := findFitParamsBin(be.Path, modelArch)
	if fitBin == "" {
		return nil
	}
	results := map[string]error{}
	return func(path string) error {
		if err, ok := results[path]; ok {
			return err
		}
		_, _, err := runFitPreflight(fitBin, []string{
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

func hasExternalSpecCompanion(strategy *placement.Strategy) bool {
	return strategy != nil && strategy.Draft != nil && strategy.Draft.Type != placement.DraftNone && strategy.Draft.Path != ""
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
		return nil, fmt.Errorf("embedded MTP Auto requires verified GPU KV placement; rerun with --spec off to skip this check")
	}
	kvType := strategy.Draft.KVTypeDraft
	if kvType == "" {
		kvType = "f16" // llama.cpp's draft-context default
	}
	contextMB := placement.EmbeddedMTPContextMB(model, strategy.ContextSize, kvType)
	if contextMB <= 0 {
		return nil, fmt.Errorf("embedded MTP context cannot be derived from GGUF metadata; rerun with --spec off to skip this check")
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
		return nil, fmt.Errorf("embedded MTP has no measured CUDA device rows; rerun with --spec off to skip this check")
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
			// Decompose fit into the components the probe already measured. The
			// success path prints model/context/compute per device; the failure
			// path printed only the total, so a plan that does NOT fit -- the one
			// case where knowing why matters -- could not be attributed without
			// re-deriving it by hand. That cost three wrong attributions on
			// 2026-09-15 (see PLANOVER), including one that blamed compute buffers
			// the probe cache then showed scale linearly.
			summary = append(summary, fmt.Sprintf(
				"%s %d/%d MiB (fit=%d [model=%d context=%d compute=%d unaccounted=%d] overhead=%d runtime=%d)",
				d.Name, need, free, fitMB,
				d.ModelMB, d.ContextMB, d.ComputeMB, d.UnaccountedMB,
				overheadMB, runtimeMB))
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
	cacheBackendTag := scopedProbeBackendTagForStrategy(req, model, be, strategy)
	fitBin := findFitParamsBin(be.Path, model.ModelArch)
	allocationProbe := false
	var targetDevs []preflightDevice
	var fitStderr string
	if fitBin == "" {
		allocationProbe = true
		evidence, err := runGuardedAllocationPreflight(req, be, cfg, caps, model, serverArgs)
		if err != nil {
			if adjustment, ok := err.(*backendLaunchAdjustmentError); ok {
				outcome.BackendAdjustment = &adjustment.Adjustment
				return outcome
			}
			if oom, ok := err.(*ikAllocationOOMError); ok {
				return allocationOOMOutcome(outcome, oom)
			}
			if unavailable, ok := err.(*memoryProbeUnavailableError); ok {
				outcome.ProbeUnavailable = unavailable.Reason
				return outcome
			}
			outcome.Err = err
			return outcome
		}
		outcome.Evidence = evidence
		targetDevs = evidence.Devices
	} else {
		var err error
		targetDevs, fitStderr, err = runFitPreflight(fitBin, serverArgs)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[launch] selected-backend memory oracle failed; switching to contained allocation probe: %v\n", err)
			allocationProbe = true
			fitBin = ""
			evidence, allocationErr := runGuardedAllocationPreflight(req, be, cfg, caps, model, serverArgs)
			if allocationErr != nil {
				if adjustment, ok := allocationErr.(*backendLaunchAdjustmentError); ok {
					outcome.BackendAdjustment = &adjustment.Adjustment
					return outcome
				}
				if oom, ok := allocationErr.(*ikAllocationOOMError); ok {
					return allocationOOMOutcome(outcome, oom)
				}
				if unavailable, ok := allocationErr.(*memoryProbeUnavailableError); ok {
					outcome.ProbeUnavailable = unavailable.Reason
					return outcome
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
	// The oracle loads the model for real (no_alloc), so any capability warning
	// the model triggers is already on its stderr — before a byte of VRAM is
	// committed. Acting on it here drops the dead flag and re-plans against the
	// smaller context, instead of carrying it through the entire launch because
	// the backend "only warned" and never failed.
	if adjustment := backendAdjustmentFromLog(fitStderr); adjustment != nil {
		// An explicit user flag is not ggrun's to remove. The backend only
		// warned here — it loaded fine — so the launch continues; say plainly
		// that the flag buys nothing, because silently carrying it is how
		// --swa-full cost 5.5 GiB of the main GPU on deepseek4 for a feature the
		// model ignored.
		if adjustment.RemoveFlag != "" && userExplicitBackendFlag(req, adjustment.RemoveFlag) {
			fmt.Fprintf(os.Stderr,
				"[launch] %s is a no-op on this model (%s), but you asked for it explicitly, so it stays. Drop it to reclaim the VRAM it reserves.\n",
				adjustment.RemoveFlag, adjustment.Reason,
			)
		} else {
			advisory := *adjustment
			advisory.Advisory = true
			outcome.BackendAdjustment = &advisory
			return outcome
		}
	}
	devs := targetDevs
	companionRejected := false
	if !allocationProbe {
		if draftArgs := draftPreflightServerArgs(strategy); len(draftArgs) > 0 {
			draftDevs, _, draftErr := runFitPreflight(fitBin, draftArgs)
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
		allocation := placement.MeasuredAllocation{
			Evidence:          string(outcome.Evidence.Level),
			PlacementIdentity: placement.AllocationPlacementIdentity(strategy),
			ContextByGPU:      map[int]int{},
			ModelByGPU:        map[int]int{},
			UnaccountedByGPU:  map[int]int{},
		}
		for _, d := range targetDevs {
			if idx, ok := cudaDeviceIndex(d.Name); ok {
				computeByGPU[idx] = d.ComputeMB
				allocation.ContextByGPU[idx] = d.ContextMB
				allocation.ModelByGPU[idx] = d.ModelMB
				allocation.UnaccountedByGPU[idx] = d.UnaccountedMB
				continue
			}
			if d.Name == "Host" {
				allocation.ContextHostMB = d.ContextMB
				allocation.ModelHostMB = d.ModelMB
				allocation.UnaccountedHostMB = d.UnaccountedMB
			}
		}
		if !allocationProbe || !hasExternalSpecCompanion(strategy) {
			allocation.ContextTotalMB = preflightContextTotalMB(targetDevs)
			if err := placement.RecordMeasuredAllocation(cfg.CacheDir, model, strategy.ContextSize,
				strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, cacheBackendTag,
				caps.GPUs, strategy.Parallel, allocation); err != nil {
				fmt.Fprintf(os.Stderr, "[launch] warning: could not persist scoped allocation evidence: %v\n", err)
			}
		}
		_ = placement.RecordMeasuredComputeBuffers(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, cacheBackendTag, caps.GPUs, strategy.Parallel, computeByGPU)
	}
	overheadByGPU := placement.SystemCUDAOverheadByGPU(cfg.CacheDir, caps.GPUs)
	var runtimeGrowthByGPU map[int]int
	if model != nil && strategy != nil {
		runtimeGrowthByGPU = placement.RuntimeGraphGrowthByGPU(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, cacheBackendTag, caps.GPUs, strategy.Parallel)
		// Cold-start carry: agree with the fit loop — when the keyed growth is
		// unmeasured, reserve the measured related-key growth so preflight and
		// placement both pack around it on a cold key (preflight must not fit
		// while the real launch would OOM, or vice versa).
		if len(runtimeGrowthByGPU) == 0 {
			if related := placement.RelatedModelRuntimeGraphGrowth(cfg.CacheDir, model, caps.GPUs, strategy.Parallel, cacheBackendTag); len(related) > 0 {
				runtimeGrowthByGPU = related
			}
		}
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
	outcome.AllocMBMeasured = oom.AllocMB > 0
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
		devs, _, err := runFitPreflight(fitBin, candArgs)
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

// allocationProbeTimeout bounds the contained allocation probe. A --dry-run
// probe used a fixed two minutes on the assumption that it touches no weights,
// but ik_llama's dry run still reads and repacks every expert tensor: a fresh
// install timed out at expert layer 56/60 of a 148.5 GiB MiniMax-M3 and failed
// closed on its first launch. The bound is a ceiling, not a cost, so a dry run
// gets the same size-scaled budget as a load.
func allocationProbeTimeout(model *placement.ModelProfile) time.Duration {
	return autoStartupTimeout(model)
}
