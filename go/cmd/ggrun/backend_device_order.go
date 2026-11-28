package main

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/devicelist"
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
	return listVulkanDevicesWith(path, "", false)
}

// listVulkanDevicesUnder lists the devices the backend sees with
// GGML_VK_VISIBLE_DEVICES=visible, the value a launch is about to export.
func listVulkanDevicesUnder(path, visible string) ([]listedDevice, error) {
	return listVulkanDevicesWith(path, visible, true)
}

func listVulkanDevicesWith(path, visible string, restrict bool) ([]listedDevice, error) {
	env := make([]string, 0, len(os.Environ())+1)
	for _, kv := range os.Environ() {
		if !strings.HasPrefix(kv, "GGML_VK_VISIBLE_DEVICES=") {
			env = append(env, kv)
		}
	}
	if restrict {
		env = append(env, "GGML_VK_VISIBLE_DEVICES="+visible)
	}
	listing := devicelist.Run(devicelist.Probe{Binary: path, Env: env, Timeout: devicelist.DefaultTimeout})
	devices := parseVulkanDeviceList(listing.Output)
	if len(devices) == 0 {
		return nil, fmt.Errorf("%s --list-devices reported no Vulkan devices (%v)", path, listing.Err)
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
	env, _, err := vulkanDeviceOrderEnvChecked(req, be, caps, list, nil)
	return env, err
}

// vulkanDeviceOrderEnvChecked is vulkanDeviceOrderEnv with the restriction of
// restrictedVulkanDeviceOrder when listUnder is set (vulkanRestrictionApplies).
// note is a line for the user, or empty.
func vulkanDeviceOrderEnvChecked(req *launchRequest, be *backendInfo, caps *detect.Capabilities,
	list func(string) ([]listedDevice, error), listUnder func(path, visible string) ([]listedDevice, error)) (env, note string, err error) {
	if !vulkanDeviceOrderApplies(be, caps) {
		return "", "", nil
	}
	listed, err := list(be.Path)
	if err != nil {
		return "", "", err
	}
	var key, value string
	if listUnder == nil {
		key, value, err = vulkanDeviceOrder(req, be, caps, listed)
	} else {
		key, value, note, err = restrictedVulkanDeviceOrder(req, be, caps, listed, func(visible string) ([]listedDevice, error) {
			return listUnder(be.Path, visible)
		})
	}
	if err != nil || key == "" {
		return "", note, err
	}
	if err := os.Setenv(key, value); err != nil {
		return "", note, err
	}
	return key + "=" + value, note, nil
}

// vulkanDeviceOrderApplies reports whether vulkanDeviceOrderEnv lists devices:
// only a Vulkan backend on a host with detected GPUs.
func vulkanDeviceOrderApplies(be *backendInfo, caps *detect.Capabilities) bool {
	return be != nil && strings.EqualFold(be.Dialect, "vulkan") && caps != nil && len(caps.GPUs) > 0
}

// vulkanRestrictionApplies reports whether a Vulkan launch must be restricted
// to its own GPUs even when the backend already numbers them in ggrun's order:
// with GGRUN_EXPERIMENTAL_AMD=1 on Linux, a Vulkan backend on NVIDIA cards
// whose host also has a GPU of another vendor (the backend's API differs from
// the primary group's). Without the restriction the backend also opens that
// other card when the counts happen to line up. Hosts with one GPU vendor, and
// every host without the opt-in, never get here: no inventory is read and no
// extra listing runs.
func vulkanRestrictionApplies(be *backendInfo, caps *detect.Capabilities, inventory func() []detect.PCIDisplay) bool {
	if !vulkanDeviceOrderApplies(be, caps) || inventory == nil || !detect.ExperimentalAMD() {
		return false
	}
	return detect.OtherVendorBesideNVIDIA(caps, inventory())
}

const vulkanVisibleDevicesKey = "GGML_VK_VISIBLE_DEVICES"

// vulkanDeviceOrder is the variable and value vulkanDeviceOrderEnv exports for
// the backend's own device listing, or two empty strings when the order is
// already ggrun's and --gpus is not given. It runs nothing and reads no
// process state.
func vulkanDeviceOrder(req *launchRequest, be *backendInfo, caps *detect.Capabilities, listed []listedDevice) (key, value string, err error) {
	if !vulkanDeviceOrderApplies(be, caps) {
		return "", "", nil
	}
	parts, _, identity, restricted, err := vulkanDeviceSelection(req, caps, listed)
	if err != nil {
		return "", "", err
	}
	if identity && !restricted {
		return "", "", nil
	}
	return vulkanVisibleDevicesKey, strings.Join(parts, ","), nil
}

// vulkanDeviceSelection maps the launch's GPUs (all detected ones, or --gpus)
// to the backend's device indices: parts are those indices in ggrun's order,
// selected the ggrun indices, identity whether they already coincide.
func vulkanDeviceSelection(req *launchRequest, caps *detect.Capabilities, listed []listedDevice) (parts []string, selected []int, identity, restricted bool, err error) {
	perm, err := matchBackendDevices(caps.GPUs, listed)
	if err != nil {
		return nil, nil, false, false, err
	}
	selected = make([]int, len(caps.GPUs))
	for i := range selected {
		selected[i] = i
	}
	restricted = req != nil && strings.TrimSpace(req.GPUsFlag) != ""
	if restricted {
		if selected, err = parseGPUIndices(req.GPUsFlag); err != nil {
			return nil, nil, false, false, err
		}
	}
	parts = make([]string, 0, len(selected))
	identity = true
	for pos, idx := range selected {
		if idx < 0 || idx >= len(perm) {
			return nil, nil, false, false, fmt.Errorf("--gpus index %d is not a detected GPU", idx)
		}
		if perm[idx] != pos {
			identity = false
		}
		parts = append(parts, strconv.Itoa(perm[idx]))
	}
	return parts, selected, identity, restricted, nil
}

// restrictedVulkanDeviceOrder always exports the launch's devices, also when
// vulkanDeviceOrder would export nothing, and checks the value first: the
// backend is listed once under it and must show exactly the selected GPUs, in
// ggrun's order. On any mismatch the launch keeps vulkanDeviceOrder's result
// (today's behaviour) and note says so. It runs only what listUnder runs.
func restrictedVulkanDeviceOrder(req *launchRequest, be *backendInfo, caps *detect.Capabilities, listed []listedDevice,
	listUnder func(visible string) ([]listedDevice, error)) (key, value, note string, err error) {
	key, value, err = vulkanDeviceOrder(req, be, caps, listed)
	if err != nil || !vulkanDeviceOrderApplies(be, caps) {
		return key, value, "", err
	}
	parts, selected, _, _, err := vulkanDeviceSelection(req, caps, listed)
	if err != nil {
		return key, value, "", nil
	}
	restricted := strings.Join(parts, ",")
	visible, checkErr := listUnder(restricted)
	if checkErr == nil {
		checkErr = vulkanVisibleListingMatches(caps, selected, visible)
	}
	if checkErr != nil {
		kept := "no restriction"
		if key != "" {
			kept = key + "=" + value
		}
		return key, value, fmt.Sprintf("warning: restricting the Vulkan backend to the detected GPUs failed its check (%v); keeping %s", checkErr, kept), nil
	}
	return vulkanVisibleDevicesKey, restricted,
		fmt.Sprintf("Vulkan backend restricted to the %d detected GPU(s) because this host also has a GPU of another vendor; checked against the backend's own listing", len(selected)), nil
}

// vulkanVisibleListingMatches reports whether a listing taken under the
// exported GGML_VK_VISIBLE_DEVICES shows exactly the selected GPUs, renumbered
// 0..n-1 in ggrun's order, with matching names and totals.
func vulkanVisibleListingMatches(caps *detect.Capabilities, selected []int, visible []listedDevice) error {
	if len(visible) != len(selected) {
		return fmt.Errorf("the backend lists %d devices under the restriction, want %d", len(visible), len(selected))
	}
	for pos, idx := range selected {
		if idx < 0 || idx >= len(caps.GPUs) {
			return fmt.Errorf("GPU %d is not a detected GPU", idx)
		}
		g, d := caps.GPUs[idx], visible[pos]
		diff := d.TotalMB - g.VRAMTotalMB
		if diff < 0 {
			diff = -diff
		}
		if d.Index != pos || !strings.EqualFold(strings.TrimSpace(d.Name), strings.TrimSpace(g.Name)) || diff > g.VRAMTotalMB/10+256 {
			return fmt.Errorf("Vulkan%d is %s (%d MiB), want %s (%d MiB)", d.Index, d.Name, d.TotalMB, g.Name, g.VRAMTotalMB)
		}
	}
	return nil
}

// applyVulkanDeviceOrder runs vulkanDeviceOrderEnv for a launch and reports it.
func applyVulkanDeviceOrder(req *launchRequest, be *backendInfo, caps *detect.Capabilities) {
	var listUnder func(string, string) ([]listedDevice, error)
	if vulkanRestrictionApplies(be, caps, detect.HostPCIDisplays) {
		listUnder = listVulkanDevicesUnder
	}
	env, note, err := vulkanDeviceOrderEnvChecked(req, be, caps, listVulkanDevices, listUnder)
	if note != "" {
		fmt.Fprintf(os.Stderr, "[launch] %s\n", note)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "[launch] warning: could not align Vulkan device order with detected GPUs: %v\n", err)
		return
	}
	if env != "" {
		fmt.Printf("[launch] Vulkan device order: %s (backend devices listed in detected GPU order)\n", env)
	}
}
