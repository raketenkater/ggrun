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

// listedDevice is one device as the backend itself enumerates it. Driver is
// the Vulkan driver name from ggml_vulkan's own device line, when the backend
// printed one.
type listedDevice struct {
	Index   int
	Name    string
	TotalMB int
	Driver  string
}

var (
	vulkanListedDevice = regexp.MustCompile(`^\s*Vulkan(\d+):\s*(.+?)\s*\((\d+) MiB`)
	// ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | ...
	vulkanDeviceDriver = regexp.MustCompile(`ggml_vulkan: (\d+) = .*\(([^()]*)\) \| uma:`)
	// ggml_vulkan: Invalid device index 4 in GGML_VK_VISIBLE_DEVICES.
	vulkanInvalidIndex = regexp.MustCompile(`Invalid device index (\d+) in GGML_VK_VISIBLE_DEVICES`)
)

func parseVulkanDeviceList(out string) []listedDevice {
	drivers := map[int]string{}
	for _, line := range strings.Split(out, "\n") {
		if m := vulkanDeviceDriver.FindStringSubmatch(line); m != nil {
			index, _ := strconv.Atoi(m[1])
			drivers[index] = strings.TrimSpace(m[2])
		}
	}
	var devices []listedDevice
	for _, line := range strings.Split(out, "\n") {
		m := vulkanListedDevice.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		index, _ := strconv.Atoi(m[1])
		total, _ := strconv.Atoi(m[3])
		devices = append(devices, listedDevice{Index: index, Name: m[2], TotalMB: total, Driver: drivers[index]})
	}
	return devices
}

// listVulkanDevices asks the backend for its own, unfiltered device order.
func listVulkanDevices(path string) ([]listedDevice, error) {
	listing := devicelist.Run(devicelist.Probe{Binary: path, Env: vulkanListingEnv("", false), Timeout: devicelist.DefaultTimeout})
	devices := parseVulkanDeviceList(listing.Output)
	if len(devices) == 0 {
		return nil, fmt.Errorf("%s --list-devices reported no Vulkan devices (%v)", path, listing.Err)
	}
	return devices, nil
}

// listVulkanOutputUnder is the backend's listing output with
// GGML_VK_VISIBLE_DEVICES=visible; an invalid index is reported there.
func listVulkanOutputUnder(path, visible string) (string, error) {
	listing := devicelist.Run(devicelist.Probe{Binary: path, Env: vulkanListingEnv(visible, true), Timeout: devicelist.DefaultTimeout})
	if listing.Output == "" && listing.Err != nil {
		return "", fmt.Errorf("%s --list-devices: %w", path, listing.Err)
	}
	return listing.Output, nil
}

func vulkanListingEnv(visible string, restrict bool) []string {
	env := make([]string, 0, len(os.Environ())+1)
	for _, kv := range os.Environ() {
		if !strings.HasPrefix(kv, "GGML_VK_VISIBLE_DEVICES=") {
			env = append(env, kv)
		}
	}
	if restrict {
		env = append(env, "GGML_VK_VISIBLE_DEVICES="+visible)
	}
	return env
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
	list func(string) ([]listedDevice, error), listUnder func(path, visible string) (string, error)) (env, note string, err error) {
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
		key, value, note, err = restrictedVulkanDeviceOrder(req, be, caps, listed, func(visible string) (string, error) {
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

// vulkanRawProbeLimit bounds the raw device count learned in one listing.
const vulkanRawProbeLimit = 16

// restrictedVulkanDeviceOrder always exports the launch's devices, also when
// vulkanDeviceOrder would export nothing, and checks the value first.
//
// GGML_VK_VISIBLE_DEVICES takes indices into the Vulkan loader's raw physical
// device list, while the default listing (the one ggrun's order is matched
// on) numbers a filtered list: GPUs only, one entry per card when two drivers
// expose it. The two coincide only when nothing is filtered ahead of a GPU (a
// CPU renderer listed first, or RADV beside AMDVLK, NVK beside NVIDIA's
// driver, break that). So the raw list is listed too (vulkanRawDevices), each
// launch device is found there by name, driver and total, and those raw
// indices are exported. The backend is listed once more under that value and
// must show exactly the launch's devices, in ggrun's order, with the same
// names, drivers and totals. On any doubt the launch keeps vulkanDeviceOrder's
// result (today's behaviour) and note says so. It runs only what listUnder
// runs: at most three listings.
func restrictedVulkanDeviceOrder(req *launchRequest, be *backendInfo, caps *detect.Capabilities, listed []listedDevice,
	listUnder func(visible string) (string, error)) (key, value, note string, err error) {
	key, value, err = vulkanDeviceOrder(req, be, caps, listed)
	if err != nil || !vulkanDeviceOrderApplies(be, caps) {
		return key, value, "", err
	}
	keep := func(reason error) (string, string, string, error) {
		kept := "no restriction"
		if key != "" {
			kept = key + "=" + value + " (not verified)"
		}
		return key, value, fmt.Sprintf("warning: restricting the Vulkan backend to the detected GPUs failed its check (%v); keeping %s", reason, kept), nil
	}
	launch, err := vulkanLaunchDevices(req, caps, listed)
	if err != nil {
		return keep(err)
	}
	raw, err := vulkanRawDevices(listUnder)
	if err != nil {
		return keep(err)
	}
	indices, err := vulkanRawIndices(listed, launch, raw)
	if err != nil {
		return keep(err)
	}
	restricted := strings.Join(indices, ",")
	out, err := listUnder(restricted)
	if err == nil {
		err = vulkanVisibleListingMatches(launch, parseVulkanDeviceList(out))
	}
	if err != nil {
		return keep(err)
	}
	return vulkanVisibleDevicesKey, restricted,
		fmt.Sprintf("Vulkan backend restricted to the %d detected GPU(s) because this host also has a GPU of another vendor; checked against the backend's own listing", len(launch)), nil
}

// vulkanLaunchDevices is the default-listing device of each launch GPU, in
// ggrun's order (all detected GPUs, or --gpus).
func vulkanLaunchDevices(req *launchRequest, caps *detect.Capabilities, listed []listedDevice) ([]listedDevice, error) {
	parts, _, _, _, err := vulkanDeviceSelection(req, caps, listed)
	if err != nil {
		return nil, err
	}
	byIndex := map[int]listedDevice{}
	for _, d := range listed {
		byIndex[d.Index] = d
	}
	launch := make([]listedDevice, 0, len(parts))
	for _, part := range parts {
		idx, _ := strconv.Atoi(part)
		d, ok := byIndex[idx]
		if !ok {
			return nil, fmt.Errorf("the backend's listing has no Vulkan%d", idx)
		}
		launch = append(launch, d)
	}
	return launch, nil
}

// vulkanRawDevices lists the raw physical device list: indices 0..15 make the
// backend name the first invalid index, which is the raw count n; the list
// under 0..n-1 then numbers every raw device by its raw index.
func vulkanRawDevices(listUnder func(visible string) (string, error)) ([]listedDevice, error) {
	indexList := func(n int) string {
		parts := make([]string, n)
		for i := range parts {
			parts[i] = strconv.Itoa(i)
		}
		return strings.Join(parts, ",")
	}
	out, err := listUnder(indexList(vulkanRawProbeLimit))
	if err != nil {
		return nil, err
	}
	devices := parseVulkanDeviceList(out)
	if m := vulkanInvalidIndex.FindStringSubmatch(out); m != nil {
		n, _ := strconv.Atoi(m[1])
		if n <= 0 {
			return nil, fmt.Errorf("the backend has no Vulkan device")
		}
		if out, err = listUnder(indexList(n)); err != nil {
			return nil, err
		}
		devices = parseVulkanDeviceList(out)
		if len(devices) != n {
			return nil, fmt.Errorf("the backend listed %d of its %d raw Vulkan devices", len(devices), n)
		}
	} else if len(devices) != vulkanRawProbeLimit {
		return nil, fmt.Errorf("the backend did not report its raw Vulkan device count")
	}
	for i, d := range devices {
		if d.Index != i {
			return nil, fmt.Errorf("the raw Vulkan listing is out of order at Vulkan%d", d.Index)
		}
	}
	return devices, nil
}

// vulkanRawIndices finds each launch device in the raw list by name, driver
// and total. The k-th launch device of one kind (same name and driver in the
// default listing) is the k-th raw device of that kind: the filter keeps the
// raw order. When the default listing and the raw list disagree on how many
// devices of a kind exist (a duplicate driver with the same name and no
// driver line), the mapping is ambiguous and refused.
func vulkanRawIndices(listed, launch []listedDevice, raw []listedDevice) ([]string, error) {
	kind := func(d listedDevice) string {
		return strings.ToLower(strings.TrimSpace(d.Name)) + "\x00" + strings.ToLower(strings.TrimSpace(d.Driver))
	}
	sameCard := func(a, b listedDevice) bool {
		diff := a.TotalMB - b.TotalMB
		if diff < 0 {
			diff = -diff
		}
		return kind(a) == kind(b) && diff <= 256
	}
	indices := make([]string, 0, len(launch))
	for _, d := range launch {
		occurrence, inDefault := 0, 0
		for _, other := range listed {
			if sameCard(other, d) {
				if other.Index < d.Index {
					occurrence++
				}
				inDefault++
			}
		}
		var candidates []listedDevice
		for _, r := range raw {
			if sameCard(r, d) {
				candidates = append(candidates, r)
			}
		}
		if len(candidates) != inDefault {
			return nil, fmt.Errorf("the raw Vulkan list has %d devices like %s (%s), the default listing %d", len(candidates), d.Name, driverOrUnknown(d.Driver), inDefault)
		}
		indices = append(indices, strconv.Itoa(candidates[occurrence].Index))
	}
	return indices, nil
}

func driverOrUnknown(driver string) string {
	if driver == "" {
		return "driver not reported"
	}
	return driver
}

// vulkanVisibleListingMatches reports whether a listing taken under the
// exported GGML_VK_VISIBLE_DEVICES shows exactly the launch devices,
// renumbered 0..n-1 in ggrun's order, with the same names, drivers and totals.
func vulkanVisibleListingMatches(launch, visible []listedDevice) error {
	if len(visible) != len(launch) {
		return fmt.Errorf("the backend lists %d devices under the restriction, want %d", len(visible), len(launch))
	}
	for pos, want := range launch {
		d := visible[pos]
		diff := d.TotalMB - want.TotalMB
		if diff < 0 {
			diff = -diff
		}
		if d.Index != pos || !strings.EqualFold(strings.TrimSpace(d.Name), strings.TrimSpace(want.Name)) ||
			!strings.EqualFold(d.Driver, want.Driver) || diff > 256 {
			return fmt.Errorf("Vulkan%d is %s (%s, %d MiB), want %s (%s, %d MiB)", d.Index,
				d.Name, driverOrUnknown(d.Driver), d.TotalMB, want.Name, driverOrUnknown(want.Driver), want.TotalMB)
		}
	}
	return nil
}

// hostPCIDisplays is the launch's inventory reader (a test seam).
var hostPCIDisplays = detect.HostPCIDisplays

// applyVulkanDeviceOrder runs vulkanDeviceOrderEnv for a launch and reports it.
func applyVulkanDeviceOrder(req *launchRequest, be *backendInfo, caps *detect.Capabilities) {
	var listUnder func(string, string) (string, error)
	if vulkanRestrictionApplies(be, caps, hostPCIDisplays) {
		listUnder = listVulkanOutputUnder
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
