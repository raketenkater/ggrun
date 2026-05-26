package detect

import (
	"os/exec"
	"strings"
)

// detectVulkanGPUs discovers Vulkan-capable GPUs via vulkaninfo.
func detectVulkanGPUs() []GPU {
	// Try vulkaninfo --summary for device enumeration
	out, err := exec.Command("vulkaninfo", "--summary").Output()
	if err != nil {
		return nil
	}

	var gpus []GPU
	var current *GPU
	inDevice := false

	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)

		// Device blocks start with GPU name
		if strings.HasPrefix(line, "deviceName") {
			parts := strings.SplitN(line, "=", 2)
			if len(parts) == 2 {
			name := strings.TrimSpace(parts[1])
			if strings.Contains(strings.ToLower(name), "llvmpipe") { continue }
			gpus = append(gpus, GPU{Name: name})
			current = &gpus[len(gpus)-1]
				inDevice = true
			}
		}

		if !inDevice || current == nil {
			continue
		}

		// driverVersion
		if strings.HasPrefix(line, "driverVersion") {
			parts := strings.SplitN(line, "=", 2)
			if len(parts) == 2 {
				current.Driver = strings.TrimSpace(parts[1])
			}
		}

		// apiVersion
		if strings.HasPrefix(line, "apiVersion") {
			parts := strings.SplitN(line, "=", 2)
			if len(parts) == 2 {
				current.ComputeCap = strings.TrimSpace(parts[1])
			}
		}
	}

	// Re-index GPUs
	for i := range gpus {
		gpus[i].Index = i
		// Estimate VRAM from device name (vulkaninfo doesn't expose VRAM)
		gpus[i].VRAMTotalMB = estimateVRAMFromName(gpus[i].Name)
	}

	return gpus
}

// estimateVRAMFromName tries to guess VRAM from common GPU name patterns.
func estimateVRAMFromName(name string) int {
	lower := strings.ToLower(name)
	switch {
	case strings.Contains(lower, "4090"): return 24576
	case strings.Contains(lower, "4080"): return 16384
	case strings.Contains(lower, "4070 ti"): return 12288
	case strings.Contains(lower, "4070"): return 12288
	case strings.Contains(lower, "4060 ti"): return 16384
	case strings.Contains(lower, "4060"): return 8192
	case strings.Contains(lower, "3090 ti"): return 24576
	case strings.Contains(lower, "3090"): return 24576
	case strings.Contains(lower, "3080 ti"): return 12288
	case strings.Contains(lower, "3080"): return 10240
	case strings.Contains(lower, "3070 ti"): return 8192
	case strings.Contains(lower, "3070"): return 8192
	case strings.Contains(lower, "3060 ti"): return 8192
	case strings.Contains(lower, "3060"): return 12288
	case strings.Contains(lower, "7900 xtx"): return 24576
	case strings.Contains(lower, "7900 xt"): return 20480
	case strings.Contains(lower, "7800 xt"): return 16384
	case strings.Contains(lower, "7700 xt"): return 12288
	case strings.Contains(lower, "7600"): return 8192
	case strings.Contains(lower, "a770"): return 16384
	case strings.Contains(lower, "a750"): return 8192
	case strings.Contains(lower, "a580"): return 8192
	case strings.Contains(lower, "a380"): return 6144
	default:
		return 8192 // safe default
	}
}
