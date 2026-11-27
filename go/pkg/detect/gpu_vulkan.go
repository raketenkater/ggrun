package detect

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"
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
	gpus := parseVulkanGPUs(string(out), detectRAM().TotalMB)
	if len(gpus) == 0 {
		return gpus
	}
	// Best-effort: replace the name-heuristic VRAM with the real DEVICE_LOCAL
	// heap size when full vulkaninfo is available. Keyed by device name, so it
	// stays correct regardless of how many devices --summary skipped; any parse
	// failure or unmatched device just leaves the heuristic in place. This path
	// only runs for non-NVIDIA rigs (NVIDIA is detected via nvidia-smi first).
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	full, err := exec.CommandContext(ctx, "vulkaninfo").Output()
	if err != nil {
		return gpus
	}
	heaps := parseVulkanHeapVRAM(string(full))
	for i := range gpus {
		if mb, ok := heaps[gpus[i].Name]; ok && mb > 0 {
			gpus[i].VRAMTotalMB = mb
		}
	}
	return gpus
}

// parseVulkanGPUs parses `vulkaninfo --summary` output. ramTotalMB is the
// host's system RAM (used only to synthesize a unified-memory budget for
// AMD integrated GPUs; see synthesizeIntegratedVRAMMB) — passed in rather
// than read here so the parser stays pure and unit-testable off-hardware.
func parseVulkanGPUs(summary string, ramTotalMB int) []GPU {
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
		// AMD APUs (e.g. Strix Halo, Ryzen AI Max, gfx11xx) expose an
		// integrated GPU whose DEVICE_LOCAL heap is unified system RAM, and
		// llama.cpp's HIP/ROCm backend on such systems can address most of
		// system RAM (~96-122 GiB on a 128 GB box; see llama.cpp #20472).
		// estimateVulkanVRAM returns a flat 2048 MB for integrated GPUs and
		// parseVulkanHeapVRAM skips them, so the tiny heuristic would
		// otherwise survive and placement would reject every real model.
		// Mirror detectAppleSilicon's unified-memory synthesis, scoped to
		// AMD: Intel iGPUs are aperture-capped rather than unified-memory
		// SoCs, so synthesizing RAM for them would fabricate VRAM that
		// their driver cannot actually back. Split out so the sizing rule
		// is unit-testable off-hardware, same as appleSiliconGPU.
		if isAMDIntegrated(dev.name, dev.deviceType) {
			if synth := synthesizeIntegratedVRAMMB(ramTotalMB); synth > vramMB {
				vramMB = synth
			}
		}
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

// isAMDIntegrated reports whether the Vulkan device is an AMD/ATI integrated
// GPU (APU). Only AMD APUs back their iGPU with unified system RAM that the
// HIP/ROCm backend can largely address; Intel iGPUs are aperture-capped and
// must keep the conservative heuristic.
func isAMDIntegrated(name, deviceType string) bool {
	if !strings.Contains(strings.ToLower(deviceType), "integrated") {
		return false
	}
	lowerName := strings.ToLower(name)
	return strings.Contains(lowerName, "amd") || strings.Contains(lowerName, "radeon")
}

// synthesizeIntegratedVRAMMB sizes the VRAM budget for an AMD integrated GPU
// (APU, e.g. Strix Halo gfx1151) whose DEVICE_LOCAL heap is unified system
// RAM. The flat 2048 MB heuristic (estimateVulkanVRAM) would reject every
// real model; llama.cpp's HIP backend on such systems addresses most of
// system RAM (~96-122 GiB on 128 GB, per llama.cpp PR #20472). 0.8x leaves
// ~20% for the OS/desktop and mirrors Apple's Metal fraction (0.75x,
// detectAppleSilicon) as a conservative unified-memory ceiling. Split out so
// the rule is unit-testable off-hardware, same as appleSiliconGPU.
func synthesizeIntegratedVRAMMB(ramTotalMB int) int {
	if ramTotalMB <= 0 {
		return 0
	}
	return ramTotalMB * 8 / 10
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

// parseVulkanHeapVRAM reads the largest DEVICE_LOCAL memory heap for each
// discrete GPU from full `vulkaninfo` output, returning device name -> VRAM in
// MB. It is deliberately conservative: integrated GPUs (whose DEVICE_LOCAL heap
// is shared system RAM) are skipped, and anything it can't parse cleanly is
// omitted so the caller falls back to the name heuristic.
func parseVulkanHeapVRAM(full string) map[string]int {
	result := map[string]int{}
	curName, curType := "", ""
	var heapBytes int64
	heapHasSize := false

	for _, raw := range strings.Split(full, "\n") {
		line := strings.TrimSpace(raw)
		switch {
		case strings.HasPrefix(line, "deviceName"):
			if _, v, ok := splitVulkanKV(line); ok {
				curName = v
			}
		case strings.HasPrefix(line, "deviceType"):
			if _, v, ok := splitVulkanKV(line); ok {
				curType = v
			}
		case strings.HasPrefix(line, "memoryHeaps["):
			heapBytes = 0
			heapHasSize = false
		case strings.HasPrefix(line, "size") && strings.Contains(line, "="):
			if b, ok := vulkanHeapSizeBytes(line); ok {
				heapBytes = b
				heapHasSize = true
			}
		case strings.Contains(line, "DEVICE_LOCAL"):
			if curName != "" && heapHasSize && heapBytes > 0 &&
				!strings.Contains(strings.ToUpper(curType), "INTEGRATED") {
				if mb := int(heapBytes / (1024 * 1024)); mb > result[curName] {
					result[curName] = mb
				}
			}
			heapHasSize = false
		}
	}
	return result
}

// vulkanHeapSizeBytes extracts the byte count from a vulkaninfo heap line like
// "size = 12878610432 (0x2ff800000) (11.99 GiB)".
func vulkanHeapSizeBytes(line string) (int64, bool) {
	idx := strings.Index(line, "=")
	if idx < 0 {
		return 0, false
	}
	rest := strings.TrimSpace(line[idx+1:])
	end := 0
	for end < len(rest) && rest[end] >= '0' && rest[end] <= '9' {
		end++
	}
	if end == 0 {
		return 0, false
	}
	n, err := strconv.ParseInt(rest[:end], 10, 64)
	if err != nil {
		return 0, false
	}
	return n, true
}
