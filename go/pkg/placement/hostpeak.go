package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// The per-instance memory scope is sized from a plan estimate:
// measuredFootprintPlannedFloor returns PlannedHostFootprintMB + CRAM, and the
// launcher clamps that to the configured whole-host ceiling. When the estimate
// is low the backend is SIGKILLed inside its own scope, which is the design
// working -- a runaway dies in its own cgroup rather than taking the machine.
//
// What was missing is the second half. runtimeCgroupOOM already reads the peak
// the scope recorded at the kill, and that figure went only into a log line:
// nothing stored it, so the next launch derived the same estimate and could be
// killed at the same point. Measured 2026-09-06 on GLM 5.3 Flash: the scope was
// re-sized to 127742 MiB from a 115411 MiB measurement, and the process was
// killed at that ceiling while the prompt cache was still growing (evicting
// 473 MiB entries). ggrun learned nothing and would plan identically again.
//
// This file keeps the peak. It is deliberately the least dangerous kind of
// evidence in the system: it can only RAISE a ceiling, and the launcher still
// clamps the result to the user's configured whole-host limit, so a wrong or
// stale value costs headroom rather than safety. That asymmetry is why it is
// kept per model rather than per topology -- over-reserving a little on a
// layout that needed less is a far cheaper error than being killed again.

// HostPeak is the largest host footprint a launch of this model was observed to
// reach before its scope killed it, or while serving.
type HostPeak struct {
	PeakMB int
	// FromKill records that the peak came from a cgroup kill rather than a
	// healthy sample, so a banner can say why the ceiling moved.
	FromKill bool
	// ContextSize and NCPUMoE describe the layout that reached it. Diagnostic
	// only: the value is applied per model (see the file comment).
	ContextSize int
	NCPUMoE     int
	ObservedAt  time.Time
}

func (p HostPeak) Measured() bool { return p.PeakMB > 0 }

// Describe renders the peak for a launch banner.
func (p HostPeak) Describe() string {
	if !p.Measured() {
		return ""
	}
	how := "observed while serving"
	if p.FromKill {
		how = "observed at a cgroup kill"
	}
	return fmt.Sprintf("%d MiB %s (ctx %d, n-cpu-moe %d)", p.PeakMB, how, p.ContextSize, p.NCPUMoE)
}

func hostPeakPath(cacheDir, modelBasename string) string {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	safe := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '.', r == '_':
			return r
		default:
			return '_'
		}
	}, modelBasename)
	return filepath.Join(cacheDir, "host_peak_"+safe+".peak")
}

// RecordMeasuredHostPeak keeps the largest host footprint seen for this model.
//
// Monotonic, like RecordCompanionVRAM and for the same reason: a ceiling must
// cover the worst case ever observed, and a quiet run that peaked lower is not
// evidence that the busy one will not happen again.
func RecordMeasuredHostPeak(cacheDir, modelPath string, peak HostPeak) error {
	if peak.PeakMB <= 0 {
		return nil
	}
	path := hostPeakPath(cacheDir, CalibrationModelBasename(modelPath))
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	release, err := acquirePlacementLock(path+".lock", 5*time.Second)
	if err != nil {
		return err
	}
	defer release()
	if prior := MeasuredHostPeak(cacheDir, modelPath); prior.PeakMB >= peak.PeakMB {
		return nil
	}
	body := fmt.Sprintf("# Observed host footprint peak for %s\n"+
		"HOST_PEAK_MB=%d\nHOST_PEAK_FROM_KILL=%t\nHOST_PEAK_CTX=%d\nHOST_PEAK_NCPUMOE=%d\nHOST_PEAK_AT=%d\n",
		CalibrationModelBasename(modelPath), peak.PeakMB, peak.FromKill,
		peak.ContextSize, peak.NCPUMoE, time.Now().Unix())
	return atomicWriteFile(path, []byte(body), 0o644)
}

// MeasuredHostPeak returns the stored peak, or a zero value.
func MeasuredHostPeak(cacheDir, modelPath string) HostPeak {
	data, err := os.ReadFile(hostPeakPath(cacheDir, CalibrationModelBasename(modelPath)))
	if err != nil {
		return HostPeak{}
	}
	var p HostPeak
	for _, line := range strings.Split(string(data), "\n") {
		key, raw, ok := strings.Cut(strings.TrimSpace(line), "=")
		if !ok {
			continue
		}
		switch key {
		case "HOST_PEAK_MB":
			p.PeakMB, _ = strconv.Atoi(raw)
		case "HOST_PEAK_FROM_KILL":
			p.FromKill = raw == "true"
		case "HOST_PEAK_CTX":
			p.ContextSize, _ = strconv.Atoi(raw)
		case "HOST_PEAK_NCPUMOE":
			p.NCPUMoE, _ = strconv.Atoi(raw)
		case "HOST_PEAK_AT":
			if v, convErr := strconv.ParseInt(raw, 10, 64); convErr == nil {
				p.ObservedAt = time.Unix(v, 0)
			}
		}
	}
	return p
}

// HostPeakFloorMB returns the scope floor this model's history demands, or 0
// when nothing has been observed.
//
// A kill at peak P proves the scope must exceed P, so the floor is P plus the
// caller's own headroom; a healthy peak only needs to be covered. Both are
// clamped by the caller to the configured whole-host ceiling, so this can never
// raise a scope past the limit the user set.
func HostPeakFloorMB(cacheDir, modelPath string, headroomMB int) int {
	peak := MeasuredHostPeak(cacheDir, modelPath)
	if !peak.Measured() {
		return 0
	}
	if peak.FromKill {
		if headroomMB < 0 {
			headroomMB = 0
		}
		return peak.PeakMB + headroomMB
	}
	return peak.PeakMB
}
