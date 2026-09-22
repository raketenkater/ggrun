package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// listedDevice is one device as the backend itself enumerates it.
type listedDevice struct {
	Index   int
	Name    string
	TotalMB int
}

var vulkanListedDevice = regexp.MustCompile(`^\s*Vulkan(\d+):\s*(.+?)\s*\((\d+) MiB`)

func parseVulkanDeviceList(out string) []listedDevice {
	var devices []listedDevice
	for _, line := range strings.Split(out, "\n") {
		m := vulkanListedDevice.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		index, _ := strconv.Atoi(m[1])
		total, _ := strconv.Atoi(m[3])
		devices = append(devices, listedDevice{Index: index, Name: m[2], TotalMB: total})
	}
	return devices
}

// listVulkanDevices asks the backend for its own, unfiltered device order.
func listVulkanDevices(path string) ([]listedDevice, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, path, "--list-devices")
	env := make([]string, 0, len(os.Environ()))
	for _, kv := range os.Environ() {
		if !strings.HasPrefix(kv, "GGML_VK_VISIBLE_DEVICES=") {
			env = append(env, kv)
		}
	}
	cmd.Env = env
	out, err := cmd.CombinedOutput()
	devices := parseVulkanDeviceList(string(out))
	if len(devices) == 0 {
		return nil, fmt.Errorf("%s --list-devices reported no Vulkan devices (%v)", path, err)
	}
	return devices, nil
}

// matchBackendDevices maps each detected GPU (ggrun's index order) to the
// backend's device index: same name, nearest total memory, each backend device
// used once. Two identical cards match in order, which is harmless: they are
// interchangeable for placement.
func matchBackendDevices(gpus []detect.GPU, listed []listedDevice) ([]int, error) {
	if len(listed) < len(gpus) {
		return nil, fmt.Errorf("backend lists %d devices, detection found %d", len(listed), len(gpus))
	}
	used := make([]bool, len(listed))
	perm := make([]int, len(gpus))
	for i, g := range gpus {
		best, bestDiff := -1, 0
		for j, d := range listed {
			if used[j] || !strings.EqualFold(strings.TrimSpace(d.Name), strings.TrimSpace(g.Name)) {
				continue
			}
			diff := d.TotalMB - g.VRAMTotalMB
			if diff < 0 {
				diff = -diff
			}
			if best < 0 || diff < bestDiff {
				best, bestDiff = j, diff
			}
		}
		// Reported totals differ slightly between APIs (24,564 MiB from NVML,
		// 24,810 MiB from Vulkan on the same 3090 Ti); a mismatch far beyond
		// that is a different card.
		if best < 0 || bestDiff > g.VRAMTotalMB/10+256 {
			return nil, fmt.Errorf("no backend device matches %s (%d MiB)", g.Name, g.VRAMTotalMB)
		}
		used[best] = true
		perm[i] = listed[best].Index
	}
	return perm, nil
}

// vulkanDeviceOrderEnv makes a Vulkan backend number its devices the way ggrun
// does. Placement, the tensor split, -ot device names, --main-gpu and the
// per-device evidence parsed from backend logs all use ggrun's index (NVIDIA
// PCI order), but ggml-vulkan enumerates in its own order: on the reference rig
// Vulkan0 is the 3090 Ti while ggrun's device 0 is the 4070, so a 0.28/0.58/0.14
// split put the 3090 Ti's 58% on the 12 GiB card and MiMo-V2.6 failed to
// allocate. GGML_VK_VISIBLE_DEVICES takes an ordered list and renames the
// devices by position, so listing the backend's devices in ggrun's order aligns
// every one of those at once. It also carries the --gpus selection, which the
// generic path wrote as ggrun indices and so selected the wrong physical card.
func vulkanDeviceOrderEnv(req *launchRequest, be *backendInfo, caps *detect.Capabilities, list func(string) ([]listedDevice, error)) (string, error) {
	if be == nil || !strings.EqualFold(be.Dialect, "vulkan") || caps == nil || len(caps.GPUs) == 0 {
		return "", nil
	}
	listed, err := list(be.Path)
	if err != nil {
		return "", err
	}
	perm, err := matchBackendDevices(caps.GPUs, listed)
	if err != nil {
		return "", err
	}
	selected := make([]int, len(caps.GPUs))
	for i := range selected {
		selected[i] = i
	}
	restricted := req != nil && strings.TrimSpace(req.GPUsFlag) != ""
	if restricted {
		if selected, err = parseGPUIndices(req.GPUsFlag); err != nil {
			return "", err
		}
	}
	parts := make([]string, 0, len(selected))
	identity := true
	for pos, idx := range selected {
		if idx < 0 || idx >= len(perm) {
			return "", fmt.Errorf("--gpus index %d is not a detected GPU", idx)
		}
		if perm[idx] != pos {
			identity = false
		}
		parts = append(parts, strconv.Itoa(perm[idx]))
	}
	if identity && !restricted {
		return "", nil
	}
	value := strings.Join(parts, ",")
	if err := os.Setenv("GGML_VK_VISIBLE_DEVICES", value); err != nil {
		return "", err
	}
	return "GGML_VK_VISIBLE_DEVICES=" + value, nil
}

// applyVulkanDeviceOrder runs vulkanDeviceOrderEnv for a launch and reports it.
func applyVulkanDeviceOrder(req *launchRequest, be *backendInfo, caps *detect.Capabilities) {
	env, err := vulkanDeviceOrderEnv(req, be, caps, listVulkanDevices)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[launch] warning: could not align Vulkan device order with detected GPUs: %v\n", err)
		return
	}
	if env != "" {
		fmt.Printf("[launch] Vulkan device order: %s (backend devices listed in detected GPU order)\n", env)
	}
}
