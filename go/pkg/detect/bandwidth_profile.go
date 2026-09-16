package detect

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	bandwidthProfileSchema = 1
	bandwidthProbeMinBytes = int64(16 << 20)
	bandwidthProbeMaxBytes = int64(1024 << 20)
	bandwidthProbeBytes    = int64(128 << 20)
	bandwidthProbeMinIters = 4
	maxPlausibleMBps       = 1_000_000
)

// BandwidthProfile is an explicitly measured hardware ceiling. It is keyed to
// physical hardware rather than a driver version: a driver update should not
// discard useful evidence, while moving/replacing a GPU must not reuse it.
type BandwidthProfile struct {
	Schema             int                       `json:"schema"`
	HardwareKey        string                    `json:"hardware_key"`
	MeasuredAt         time.Time                 `json:"measured_at"`
	Source             string                    `json:"source"`
	Bytes              int64                     `json:"bytes"`
	MinIterations      int                       `json:"min_iterations"`
	HostCopyMBps       int                       `json:"host_copy_mbps"`
	HostCopyIterations int                       `json:"host_copy_iterations"`
	HostCopyWorkers    int                       `json:"host_copy_workers"`
	GPUs               []GPUBandwidthMeasurement `json:"gpus"`
}

// GPUBandwidthMeasurement records pinned-memory transfers for one physical GPU.
// MoE expert streaming is host-to-device, so H2DMBps is the value placement uses.
type GPUBandwidthMeasurement struct {
	PCIBusID      string `json:"pci_bus_id"`
	H2DMBps       int    `json:"h2d_mbps"`
	D2HMBps       int    `json:"d2h_mbps"`
	H2DIterations int    `json:"h2d_iterations"`
	D2HIterations int    `json:"d2h_iterations"`
}

type bandwidthHelperOutput struct {
	Bytes              int                       `json:"bytes"`
	MinIterations      int                       `json:"min_iterations"`
	HostCopyMBps       int                       `json:"host_copy_mbps"`
	HostCopyIterations int                       `json:"host_copy_iterations"`
	HostCopyWorkers    int                       `json:"host_copy_workers"`
	GPUs               []GPUBandwidthMeasurement `json:"gpus"`
}

// BandwidthProfilePath returns the stable per-user cache path. The profile is
// intentionally independent of a model-cache setting so every ggrun app home
// on the same machine sees the same physical-hardware measurement.
func BandwidthProfilePath() (string, error) {
	if p := strings.TrimSpace(os.Getenv("LLM_BANDWIDTH_PROFILE")); p != "" {
		return filepath.Clean(p), nil
	}
	base, err := os.UserCacheDir()
	if err != nil || strings.TrimSpace(base) == "" {
		home, homeErr := os.UserHomeDir()
		if homeErr != nil || strings.TrimSpace(home) == "" {
			if err != nil {
				return "", fmt.Errorf("locate user cache directory: %w", err)
			}
			return "", fmt.Errorf("locate user cache directory")
		}
		base = filepath.Join(home, ".cache")
	}
	return filepath.Join(base, "ggrun", "hardware-bandwidth.json"), nil
}

// HardwareBandwidthKey returns a stable fingerprint of the hardware relevant
// to the profile. Volatile values (free RAM/VRAM, clocks, driver) are omitted.
func HardwareBandwidthKey(caps *Capabilities) string {
	if caps == nil {
		return ""
	}
	type identityGPU struct {
		PCIBusID    string `json:"pci_bus_id"`
		Name        string `json:"name"`
		VRAMTotalMB int    `json:"vram_total_mb"`
		ComputeCap  string `json:"compute_cap"`
	}
	type identity struct {
		OS         string        `json:"os"`
		Arch       string        `json:"arch"`
		CPUModel   string        `json:"cpu_model"`
		CPUCores   int           `json:"cpu_cores"`
		CPUThreads int           `json:"cpu_threads"`
		RAMTotalMB int           `json:"ram_total_mb"`
		GPUs       []identityGPU `json:"gpus"`
	}
	id := identity{
		OS:         strings.ToLower(strings.TrimSpace(caps.OS)),
		Arch:       strings.ToLower(strings.TrimSpace(caps.Arch)),
		CPUModel:   strings.TrimSpace(caps.CPU.Model),
		CPUCores:   caps.CPU.Cores,
		CPUThreads: caps.CPU.Threads,
		RAMTotalMB: caps.RAM.TotalMB,
		GPUs:       make([]identityGPU, 0, len(caps.GPUs)),
	}
	for _, gpu := range caps.GPUs {
		id.GPUs = append(id.GPUs, identityGPU{
			PCIBusID:    canonicalPCIBusID(gpu.PCIBusID),
			Name:        strings.TrimSpace(gpu.Name),
			VRAMTotalMB: gpu.VRAMTotalMB,
			ComputeCap:  strings.TrimSpace(gpu.ComputeCap),
		})
	}
	sort.Slice(id.GPUs, func(i, j int) bool {
		if id.GPUs[i].PCIBusID != id.GPUs[j].PCIBusID {
			return id.GPUs[i].PCIBusID < id.GPUs[j].PCIBusID
		}
		if id.GPUs[i].Name != id.GPUs[j].Name {
			return id.GPUs[i].Name < id.GPUs[j].Name
		}
		return id.GPUs[i].VRAMTotalMB < id.GPUs[j].VRAMTotalMB
	})
	data, _ := json.Marshal(id)
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func canonicalPCIBusID(busID string) string {
	busID = strings.ToLower(strings.TrimSpace(busID))
	parts := strings.SplitN(busID, ":", 2)
	if len(parts) != 2 {
		return busID
	}
	if domain, err := strconv.ParseUint(parts[0], 16, 32); err == nil {
		return fmt.Sprintf("%04x:%s", domain, parts[1])
	}
	return busID
}

func plausibleBandwidth(v int) bool {
	return v > 0 && v <= maxPlausibleMBps
}

// ApplyBandwidthProfile applies a profile only if it completely and exactly
// matches the current hardware. Validation finishes before caps is mutated, so
// a partial/corrupt profile can never skew only one GPU in a placement.
func ApplyBandwidthProfile(caps *Capabilities, profile *BandwidthProfile) bool {
	if caps == nil || profile == nil || profile.Schema != bandwidthProfileSchema ||
		profile.HardwareKey == "" || profile.HardwareKey != HardwareBandwidthKey(caps) ||
		profile.MeasuredAt.IsZero() || strings.TrimSpace(profile.Source) == "" ||
		profile.Bytes < bandwidthProbeMinBytes || profile.Bytes > bandwidthProbeMaxBytes ||
		profile.MinIterations <= 0 || profile.HostCopyIterations <= 0 ||
		profile.HostCopyWorkers <= 0 || profile.HostCopyWorkers > 256 ||
		!plausibleBandwidth(profile.HostCopyMBps) || len(profile.GPUs) != len(caps.GPUs) {
		return false
	}
	byBus := make(map[string]GPUBandwidthMeasurement, len(profile.GPUs))
	for _, measurement := range profile.GPUs {
		busID := canonicalPCIBusID(measurement.PCIBusID)
		if busID == "" || measurement.H2DIterations <= 0 || measurement.D2HIterations <= 0 ||
			!plausibleBandwidth(measurement.H2DMBps) || !plausibleBandwidth(measurement.D2HMBps) {
			return false
		}
		if _, duplicate := byBus[busID]; duplicate {
			return false
		}
		byBus[busID] = measurement
	}
	for _, gpu := range caps.GPUs {
		if _, ok := byBus[canonicalPCIBusID(gpu.PCIBusID)]; !ok {
			return false
		}
	}

	for i := range caps.GPUs {
		measurement := byBus[canonicalPCIBusID(caps.GPUs[i].PCIBusID)]
		caps.GPUs[i].BandwidthMBps = measurement.H2DMBps
		caps.GPUs[i].BandwidthSource = "measured_pinned_h2d"
	}
	caps.HostMemoryBandwidthMBps = profile.HostCopyMBps
	caps.HostMemoryBandwidthSource = "measured_parallel_memcpy"
	return true
}

// LoadBandwidthProfile reads a cached profile without applying it.
func LoadBandwidthProfile(path string) (*BandwidthProfile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var profile BandwidthProfile
	if err := json.Unmarshal(data, &profile); err != nil {
		return nil, fmt.Errorf("parse bandwidth profile: %w", err)
	}
	return &profile, nil
}

// ApplyCachedBandwidthProfile loads the stable cache and applies it when valid.
func ApplyCachedBandwidthProfile(caps *Capabilities) error {
	path, err := BandwidthProfilePath()
	if err != nil {
		return err
	}
	profile, err := LoadBandwidthProfile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	ApplyBandwidthProfile(caps, profile)
	return nil
}

// MeasureBandwidth runs the bundled dependency-free CUDA driver probe. It uses
// pinned host memory and addresses GPUs by PCI bus ID, matching ggrun's CUDA
// ordering rather than trusting an unrelated driver enumeration.
func MeasureBandwidth(caps *Capabilities) (*BandwidthProfile, error) {
	if caps == nil || len(caps.GPUs) == 0 {
		return nil, fmt.Errorf("bandwidth measurement needs at least one CUDA GPU")
	}
	script := findBandwidthScript()
	if script == "" {
		return nil, fmt.Errorf("measure_bandwidth.py could not be located or extracted: %w", errBandwidthScriptUnavailable())
	}
	args := []string{
		script,
		"--bytes", strconv.FormatInt(bandwidthProbeBytes, 10),
		"--min-iterations", strconv.Itoa(bandwidthProbeMinIters),
		"--host-workers", strconv.Itoa(max(1, caps.CPU.Cores)),
	}
	for _, gpu := range caps.GPUs {
		if strings.TrimSpace(gpu.PCIBusID) == "" {
			return nil, fmt.Errorf("GPU %d (%s) has no PCI bus ID; the CUDA bandwidth probe cannot match it safely", gpu.Index, gpu.Name)
		}
		args = append(args, "--gpu-bus-id", gpu.PCIBusID)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, bandwidthPythonCommand(), args...)
	out, err := cmd.Output()
	if err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return nil, fmt.Errorf("bandwidth measurement timed out after 5 minutes")
		}
		if exitErr, ok := err.(*exec.ExitError); ok && len(exitErr.Stderr) > 0 {
			return nil, fmt.Errorf("measure_bandwidth.py: %s", strings.TrimSpace(string(exitErr.Stderr)))
		}
		return nil, fmt.Errorf("measure_bandwidth.py: %w", err)
	}
	var measured bandwidthHelperOutput
	if err := json.Unmarshal(out, &measured); err != nil {
		return nil, fmt.Errorf("parse bandwidth measurement: %w", err)
	}
	profile := &BandwidthProfile{
		Schema:             bandwidthProfileSchema,
		HardwareKey:        HardwareBandwidthKey(caps),
		MeasuredAt:         time.Now().UTC(),
		Source:             "cuda_driver_pinned_copy",
		Bytes:              int64(measured.Bytes),
		MinIterations:      measured.MinIterations,
		HostCopyMBps:       measured.HostCopyMBps,
		HostCopyIterations: measured.HostCopyIterations,
		HostCopyWorkers:    measured.HostCopyWorkers,
		GPUs:               measured.GPUs,
	}
	copyCaps := *caps
	copyCaps.GPUs = append([]GPU(nil), caps.GPUs...)
	if !ApplyBandwidthProfile(&copyCaps, profile) {
		return nil, fmt.Errorf("bandwidth helper returned incomplete or implausible measurements for this hardware")
	}
	return profile, nil
}

// SaveBandwidthProfile atomically persists a measured profile.
func SaveBandwidthProfile(path string, profile *BandwidthProfile) error {
	if profile == nil || strings.TrimSpace(path) == "" {
		return fmt.Errorf("bandwidth profile path and data are required")
	}
	data, err := json.MarshalIndent(profile, "", "  ")
	if err != nil {
		return fmt.Errorf("encode bandwidth profile: %w", err)
	}
	data = append(data, '\n')
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create bandwidth profile directory: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".hardware-bandwidth-*.tmp")
	if err != nil {
		return fmt.Errorf("create bandwidth profile temp file: %w", err)
	}
	tmpPath := tmp.Name()
	ok := false
	defer func() {
		_ = tmp.Close()
		if !ok {
			_ = os.Remove(tmpPath)
		}
	}()
	if err := tmp.Chmod(0o600); err != nil {
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		return fmt.Errorf("write bandwidth profile: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("sync bandwidth profile: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close bandwidth profile: %w", err)
	}
	// Unix rename replaces atomically. Windows does not replace an existing
	// destination, so an explicit refresh removes only this exact cache file
	// after the complete temp file has been synced.
	if runtime.GOOS == "windows" {
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("replace bandwidth profile: %w", err)
		}
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("install bandwidth profile: %w", err)
	}
	ok = true
	return nil
}

// MeasureAndSaveBandwidth measures, validates, and writes the default profile.
func MeasureAndSaveBandwidth(caps *Capabilities) (*BandwidthProfile, string, error) {
	profile, err := MeasureBandwidth(caps)
	if err != nil {
		return nil, "", err
	}
	path, err := BandwidthProfilePath()
	if err != nil {
		return nil, "", err
	}
	if err := SaveBandwidthProfile(path, profile); err != nil {
		return nil, path, err
	}
	return profile, path, nil
}

func bandwidthPythonCommand() string {
	for _, name := range []string{"python3", "python", "py"} {
		if path, err := exec.LookPath(name); err == nil {
			return path
		}
	}
	return "python3"
}

func findBandwidthScript() string {
	var candidates []string
	addRoot := func(root string) {
		if strings.TrimSpace(root) == "" {
			return
		}
		candidates = append(candidates,
			filepath.Join(root, "measure_bandwidth.py"),
			filepath.Join(root, "tools", "hardware", "measure_bandwidth.py"),
		)
	}
	addRoot(os.Getenv("LLM_SCRIPT_DIR"))
	addRoot(os.Getenv("LLM_SERVER_HOME"))
	if appHome := os.Getenv("LLM_APP_HOME"); appHome != "" {
		candidates = append(candidates,
			filepath.Join(appHome, ".bin", "measure_bandwidth.py"),
			filepath.Join(appHome, "bin", "measure_bandwidth.py"),
			filepath.Join(appHome, ".src", "ggrun", "tools", "hardware", "measure_bandwidth.py"),
		)
	}
	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		candidates = append(candidates,
			filepath.Join(exeDir, "measure_bandwidth.py"),
			filepath.Join(exeDir, "..", "tools", "hardware", "measure_bandwidth.py"),
			filepath.Join(exeDir, "..", "..", "tools", "hardware", "measure_bandwidth.py"),
		)
	}
	if wd, err := os.Getwd(); err == nil {
		candidates = append(candidates,
			filepath.Join(wd, "tools", "hardware", "measure_bandwidth.py"),
			filepath.Join(wd, "..", "tools", "hardware", "measure_bandwidth.py"),
			filepath.Join(wd, "measure_bandwidth.py"),
		)
	}
	if path, err := exec.LookPath("measure_bandwidth.py"); err == nil {
		candidates = append(candidates, path)
	}
	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate
		}
	}
	// No on-disk copy. Fall back to the embedded probe so the command works
	// from a bare `go install` binary. On-disk copies deliberately win, so a
	// developer editing tools/ or a packaged install still overrides this.
	if path, err := extractBandwidthScript(); err == nil {
		return path
	}
	return ""
}
