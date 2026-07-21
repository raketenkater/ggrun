package memprobe

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

const SchemaVersion = 1

type EvidenceLevel string

const (
	EvidenceNone             EvidenceLevel = "none"
	EvidenceOraclePlanned    EvidenceLevel = "oracle-planned"
	EvidenceGuardedAllocated EvidenceLevel = "guarded-allocated"
	EvidenceLiveAllocated    EvidenceLevel = "live-allocated"
)

type AllocationEvent struct {
	Type        string `json:"type"`
	Event       string `json:"event,omitempty"`
	Phase       string `json:"phase,omitempty"`
	API         string `json:"api,omitempty"`
	Kind        string `json:"kind,omitempty"`
	PID         int    `json:"pid,omitempty"`
	Device      int    `json:"device,omitempty"`
	Bytes       uint64 `json:"bytes,omitempty"`
	ActiveBytes uint64 `json:"active_bytes,omitempty"`
	PeakBytes   uint64 `json:"peak_bytes,omitempty"`
	LimitBytes  uint64 `json:"limit_bytes,omitempty"`
	Result      int    `json:"result,omitempty"`
}

type DeviceMemory struct {
	ID               string `json:"id"`
	RequestedBytes   uint64 `json:"requested_bytes"`
	PeakBytes        uint64 `json:"peak_bytes"`
	LimitBytes       uint64 `json:"limit_bytes"`
	DeniedBytes      uint64 `json:"denied_bytes,omitempty"`
	ModelBytes       uint64 `json:"model_bytes,omitempty"`
	ContextBytes     uint64 `json:"context_bytes,omitempty"`
	ComputeBytes     uint64 `json:"compute_bytes,omitempty"`
	UnaccountedBytes uint64 `json:"unaccounted_bytes,omitempty"`
}

type HostMemory struct {
	PinnedPeakBytes   uint64 `json:"pinned_peak_bytes"`
	PinnedDeniedBytes uint64 `json:"pinned_denied_bytes,omitempty"`
	CgroupPeakBytes   uint64 `json:"cgroup_peak_bytes,omitempty"`
	CgroupLimitBytes  uint64 `json:"cgroup_limit_bytes,omitempty"`
	ModelBytes        uint64 `json:"model_bytes,omitempty"`
	ContextBytes      uint64 `json:"context_bytes,omitempty"`
	ComputeBytes      uint64 `json:"compute_bytes,omitempty"`
	UnaccountedBytes  uint64 `json:"unaccounted_bytes,omitempty"`
}

type Coverage struct {
	GuardLoaded       bool `json:"guard_loaded"`
	DeviceAllocations bool `json:"device_allocations"`
	PinnedAllocations bool `json:"pinned_allocations"`
	CgroupV2          bool `json:"cgroup_v2"`
	Complete          bool `json:"complete"`
}

type Plan struct {
	SchemaVersion   int            `json:"schema_version"`
	Key             string         `json:"key"`
	Evidence        EvidenceLevel  `json:"evidence"`
	BackendIdentity string         `json:"backend_identity"`
	ModelIdentity   string         `json:"model_identity"`
	ShapeIdentity   string         `json:"shape_identity"`
	Outcome         string         `json:"outcome"`
	Coverage        Coverage       `json:"coverage"`
	Devices         []DeviceMemory `json:"devices"`
	Host            HostMemory     `json:"host"`
	BackendArgs     []string       `json:"backend_args,omitempty"`
	Warnings        []string       `json:"warnings,omitempty"`
}

type Summary struct {
	Loaded       bool
	DeviceEvents bool
	PinnedEvents bool
	Devices      map[int]DeviceMemory
	Host         HostMemory
	Denied       *AllocationEvent
}

func ParseGuardLog(path string) (Summary, error) {
	result := Summary{Devices: map[int]DeviceMemory{}}
	f, err := os.Open(path)
	if err != nil {
		return result, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	buf := make([]byte, 64*1024)
	scanner.Buffer(buf, 1024*1024)
	for scanner.Scan() {
		var event AllocationEvent
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			return result, fmt.Errorf("parse memguard event: %w", err)
		}
		if event.Type == "guard" && event.Event == "loaded" {
			result.Loaded = true
			continue
		}
		if event.Type != "allocation" {
			continue
		}
		if event.Kind == "device" || event.Kind == "managed" {
			result.DeviceEvents = true
			device := result.Devices[event.Device]
			device.ID = "CUDA" + strconv.Itoa(event.Device)
			if event.Bytes > device.RequestedBytes {
				device.RequestedBytes = event.Bytes
			}
			if event.PeakBytes > device.PeakBytes {
				device.PeakBytes = event.PeakBytes
			}
			if event.LimitBytes > 0 {
				device.LimitBytes = event.LimitBytes
			}
			if event.Phase == "denied" {
				device.DeniedBytes = event.Bytes
			}
			result.Devices[event.Device] = device
		} else if event.Kind == "pinned" || event.Kind == "mlock" {
			result.PinnedEvents = true
			if event.PeakBytes > result.Host.PinnedPeakBytes {
				result.Host.PinnedPeakBytes = event.PeakBytes
			}
			if event.Phase == "denied" {
				result.Host.PinnedDeniedBytes = event.Bytes
			}
		}
		if event.Phase == "denied" {
			deviceDenial := event.Kind == "device" || event.Kind == "managed"
			currentDeviceDenial := result.Denied != nil && (result.Denied.Kind == "device" || result.Denied.Kind == "managed")
			if result.Denied == nil || (deviceDenial && !currentDeviceDenial) {
				copy := event
				result.Denied = &copy
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return result, err
	}
	return result, nil
}

func (s Summary) DeviceSlice() []DeviceMemory {
	devices := make([]DeviceMemory, 0, len(s.Devices))
	indices := make([]int, 0, len(s.Devices))
	for index := range s.Devices {
		indices = append(indices, index)
	}
	sort.Ints(indices)
	for _, index := range indices {
		devices = append(devices, s.Devices[index])
	}
	return devices
}

func CacheKey(parts ...string) string {
	h := sha256.New()
	for _, part := range parts {
		_, _ = h.Write([]byte(part))
		_, _ = h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil)[:16])
}

func Save(cacheDir string, plan Plan) (string, error) {
	if cacheDir == "" || plan.Key == "" {
		return "", errors.New("memory plan cache requires a directory and key")
	}
	plan.SchemaVersion = SchemaVersion
	path := filepath.Join(cacheDir, "memory-probes", "probe-"+plan.Key+".json")
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return "", err
	}
	data, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		return "", err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o600); err != nil {
		return "", err
	}
	if err := os.Rename(tmp, path); err != nil {
		return "", err
	}
	return path, nil
}

func Load(cacheDir, key string) (Plan, bool) {
	var plan Plan
	if cacheDir == "" || key == "" {
		return plan, false
	}
	path := filepath.Join(cacheDir, "memory-probes", "probe-"+key+".json")
	data, err := os.ReadFile(path)
	if err != nil || json.Unmarshal(data, &plan) != nil {
		return Plan{}, false
	}
	if plan.SchemaVersion != SchemaVersion || plan.Key != key || !plan.Coverage.Complete {
		return Plan{}, false
	}
	return plan, true
}

func FindGuardLibrary() string {
	if runtime.GOOS != "linux" {
		return ""
	}
	if configured := strings.TrimSpace(os.Getenv("GGRUN_MEMGUARD_LIBRARY")); configured != "" {
		if regularFile(configured) {
			return configured
		}
		return ""
	}
	var candidates []string
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		candidates = append(candidates,
			filepath.Join(dir, "libggrun-memguard.so"),
			filepath.Join(dir, "..", "lib", "libggrun-memguard.so"),
		)
	}
	if cwd, err := os.Getwd(); err == nil {
		candidates = append(candidates,
			filepath.Join(cwd, "native", "memguard", "libggrun-memguard.so"),
			filepath.Join(cwd, "..", "native", "memguard", "libggrun-memguard.so"),
			filepath.Join(cwd, "..", "..", "native", "memguard", "libggrun-memguard.so"),
		)
	}
	for _, candidate := range candidates {
		candidate = filepath.Clean(candidate)
		if regularFile(candidate) {
			return candidate
		}
	}
	return ""
}

func GuardEnvironment(library, logPath string, gpuLimitsMB []int, pinnedLimitMB int, inheritedPreload string) []string {
	limits := make([]string, len(gpuLimitsMB))
	for i, limit := range gpuLimitsMB {
		if limit < 0 {
			limit = 0
		}
		limits[i] = strconv.Itoa(limit)
	}
	preload := library
	if strings.TrimSpace(inheritedPreload) != "" {
		preload += string(os.PathListSeparator) + inheritedPreload
	}
	return []string{
		"LD_PRELOAD=" + preload,
		"GGRUN_MEMGUARD_LOG=" + logPath,
		"GGRUN_MEMGUARD_GPU_LIMITS_MB=" + strings.Join(limits, ","),
		"GGRUN_MEMGUARD_PINNED_LIMIT_MB=" + strconv.Itoa(pinnedLimitMB),
		"GGML_CUDA_NO_PINNED=1",
	}
}

func regularFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}
