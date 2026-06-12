//go:build !windows

package detect

import (
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// detectPhysicalCores returns the number of physical CPU cores on Linux/Darwin.
func detectPhysicalCores() int {
	// Darwin: use sysctl
	if runtime.GOOS == "darwin" {
		out, err := exec.Command("sysctl", "-n", "hw.physicalcpu").Output()
		if err == nil {
			if n, e := strconv.Atoi(strings.TrimSpace(string(out))); e == nil && n > 0 {
				return n
			}
		}
	}

	// Linux: parse /proc/cpuinfo for physical cores
	data, err := os.ReadFile("/proc/cpuinfo")
	if err == nil {
		seen := make(map[string]bool)
		var physID, coreID string
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			switch {
			case strings.HasPrefix(line, "physical id"):
				if parts := strings.SplitN(line, ":", 2); len(parts) == 2 {
					physID = strings.TrimSpace(parts[1])
				}
			case strings.HasPrefix(line, "core id"):
				if parts := strings.SplitN(line, ":", 2); len(parts) == 2 {
					coreID = strings.TrimSpace(parts[1])
				}
				if physID != "" && coreID != "" {
					seen[physID+":"+coreID] = true
					physID = ""
					coreID = ""
				}
			}
		}
		if n := len(seen); n > 0 {
			return n
		}
	}

	// Fallback: logical cores / 2 (HT assumption)
	n := runtime.NumCPU()
	if n >= 4 {
		return n / 2
	}
	return n
}

// detectRAMFreeMB returns available RAM in MB on Linux/Darwin.
func detectRAMFreeMB() int {
	if runtime.GOOS == "darwin" {
		// macOS: use sysctl + vm_stat (approximate)
		out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
		if err == nil {
			totalBytes, _ := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
			// Return ~80% of total as "free" (macOS manages memory aggressively)
			return int(totalBytes / 1024 / 1024 * 80 / 100)
		}
	}
	// Linux: /proc/meminfo
	data, err := os.ReadFile("/proc/meminfo")
	if err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "MemAvailable:") {
				parts := strings.Fields(line)
				if len(parts) >= 2 {
					kb, _ := strconv.Atoi(parts[1])
					return kb / 1024
				}
			}
		}
	}
	return 4096 // fallback
}

func detectRAMWindows() RAMInfo {
	return RAMInfo{}
}
