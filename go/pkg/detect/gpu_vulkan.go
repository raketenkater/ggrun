package detect

import (
	"os/exec"
	"strings"
)

type vulkanDevice struct {
	name       string
	deviceType string
	driver     string
	apiVersion string
}

// detectVulkanGPUs discovers Vulkan-capable GPUs via vulkaninfo.
func detectVulkanGPUs() []GPU {
	out, err := exec.Command("vulkaninfo", "--summary").Output()
	if err != nil {
		return nil
	}
	return parseVulkanGPUs(string(out))
}

func parseVulkanGPUs(summary string) []GPU {
	var gpus []GPU
	var dev vulkanDevice
	inBlock := false

	flush := func() {
		if dev.name == "" {
			dev = vulkanDevice{}
			return
		}
		if skipVulkanDevice(dev) {
			dev = vulkanDevice{}
			return
		}
		vramMB := estimateVulkanVRAM(dev.name, dev.deviceType)
		if vramMB <= 0 {
			dev = vulkanDevice{}
			return
		}
		gpus = append(gpus, GPU{
			Index:       len(gpus),
			Name:        dev.name,
			VRAMTotalMB: vramMB,
			Driver:      dev.driver,
			ComputeCap:  dev.apiVersion,
		})
		dev = vulkanDevice{}
	}

	for _, raw := range strings.Split(summary, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		if isVulkanGPUHeader(line) {
			if inBlock {
				flush()
			}
			inBlock = true
			continue
		}
		if !inBlock {
			continue
		}
		key, value, ok := splitVulkanKV(line)
		if !ok {
			continue
		}
		switch key {
		case "apiVersion":
			dev.apiVersion = value
		case "driverVersion":
			dev.driver = value
		case "deviceType":
			dev.deviceType = value
		case "deviceName":
			dev.name = value
		}
	}
	if inBlock {
		flush()
	}
	return gpus
}

func isVulkanGPUHeader(line string) bool {
	return strings.HasPrefix(line, "GPU") && strings.Contains(line, ":")
}

func splitVulkanKV(line string) (string, string, bool) {
	parts := strings.SplitN(line, "=", 2)
	if len(parts) != 2 {
		return "", "", false
	}
	return strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1]), true
}

func skipVulkanDevice(dev vulkanDevice) bool {
	lowerName := strings.ToLower(dev.name)
	lowerType := strings.ToLower(dev.deviceType)
	if strings.Contains(lowerType, "cpu") || strings.Contains(lowerType, "software") {
		return true
	}
	for _, marker := range []string{"llvmpipe", "lavapipe", "swiftshader", "software rasterizer"} {
		if strings.Contains(lowerName, marker) {
			return true
		}
	}
	return false
}

func estimateVulkanVRAM(name, deviceType string) int {
	if strings.Contains(strings.ToLower(deviceType), "integrated") {
		return 2048
	}
	return estimateVRAMFromName(name)
}

// estimateVRAMFromName tries to guess VRAM from common GPU name patterns.
func estimateVRAMFromName(name string) int {
	lower := strings.ToLower(name)
	switch {
	case strings.Contains(lower, "4090"):
		return 24576
	case strings.Contains(lower, "4080"):
		return 16384
	case strings.Contains(lower, "4070 ti"):
		return 12288
	case strings.Contains(lower, "4070"):
		return 12288
	case strings.Contains(lower, "4060 ti"):
		return 16384
	case strings.Contains(lower, "4060"):
		return 8192
	case strings.Contains(lower, "3090 ti"):
		return 24576
	case strings.Contains(lower, "3090"):
		return 24576
	case strings.Contains(lower, "3080 ti"):
		return 12288
	case strings.Contains(lower, "3080"):
		return 10240
	case strings.Contains(lower, "3070 ti"):
		return 8192
	case strings.Contains(lower, "3070"):
		return 8192
	case strings.Contains(lower, "3060 ti"):
		return 8192
	case strings.Contains(lower, "3060"):
		return 12288
	case strings.Contains(lower, "7900 xtx"):
		return 24576
	case strings.Contains(lower, "7900 xt"):
		return 20480
	case strings.Contains(lower, "7800 xt"):
		return 16384
	case strings.Contains(lower, "7700 xt"):
		return 12288
	case strings.Contains(lower, "7600"):
		return 8192
	case strings.Contains(lower, "a770"):
		return 16384
	case strings.Contains(lower, "a750"):
		return 8192
	case strings.Contains(lower, "a580"):
		return 8192
	case strings.Contains(lower, "a380"):
		return 6144
	default:
		return 4096
	}
}
