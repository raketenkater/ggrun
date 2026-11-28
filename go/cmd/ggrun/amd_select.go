package main

import (
	"fmt"
	"os"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/devicelist"
	"github.com/raketenkater/ggrun/pkg/server"
)

// Selecting a GPU of another vendor: --gpus amd:<n> and --gpus pci:<address>.
//
// Detection lists the GPUs nvidia-smi reports. An AMD card beside them is in
// the OS inventory only (detect.HostPCIDisplays). With GGRUN_EXPERIMENTAL_AMD=1
// on Linux and an inventory with GPUs of more than one vendor, launch and
// dry-run accept inventory selectors. Such a card becomes plannable only when
// the launch's own backend build lists it under the launch's environment
// (applyVendorSelection):
//   - Vulkan: no listing prints a PCI address, so the card is found in the
//     backend's default listing by its total memory, leaving out every device
//     named like a detected GPU or another vendor; more than one candidate is
//     refused. The export is the card's raw Vulkan index (vulkanRawIndices),
//     and the listing under it must show exactly the selected cards;
//   - HIP (ROCm): ROCR_VISIBLE_DEVICES names each card by its KFD unique id,
//     so the listing under it shows only those; count and totals must match.
//     ROCr surfaces the listed GPUs in the order of the list's tokens (its
//     device filter ranks each token by position; ROCm 6.3 amd_topology.cpp
//     and amd_filter_device.cpp), and HIP numbers them in that order, so
//     ROCm<k> is the k-th card of the list, which is written in PCI address
//     order. The post-load check verifies it.
//
// The cards are appended after the detected GPUs and --gpus is rewritten to
// their indexes, so every reader of --gpus sees the shape it already handles
// and no index-keyed nvidia-smi query can name an NVIDIA card for them. After
// load the backend's "using device" lines must name the planned PCI addresses
// (checkUsingDevices), or the launch stops. One launch never mixes vendors.
// Numeric --gpus keeps meaning the detected indexes; without the opt-in, and in
// every other command, a selector is today's parse error.

// gpuSelectorsAccepted is set by the commands that resolve inventory
// selectors (launch and dry-run) before they parse their arguments.
var gpuSelectorsAccepted bool

// vendorDevice is one selected inventory card.
type vendorDevice struct {
	Display detect.PCIDisplay
	Token   string // the --gpus token that named it
	Host    int    // its index in the launch's GPU list, after the detected GPUs
	Row     string // the backend's device name: prefix + ordinal, e.g. "ROCm0"
	Name    string // as the backend lists it
	TotalMB int    // as the backend lists it
}

// vendorSelection is a resolved AMD selection. Dialect and the visibility
// fields are set once the backend has listed the cards.
type vendorSelection struct {
	Flag     string         // the --gpus value as given
	Devices  []vendorDevice // in PCI address order, which is the ordinal order
	Dialect  string         // "vulkan" or "rocm"
	VisKey   string
	VisValue string
	Unset    []string // variables the backend must not inherit
}

type gpuSelector struct {
	Token string
	Index int    // a detected GPU index; -1 otherwise
	AMD   int    // amd:<n>; -1 otherwise
	PCI   string // pci:<address>, canonical
}

func hasVendorSelector(raw string) bool {
	for _, token := range strings.Split(raw, ",") {
		t := strings.ToLower(strings.TrimSpace(token))
		if strings.HasPrefix(t, "amd:") || strings.HasPrefix(t, "pci:") {
			return true
		}
	}
	return false
}

func parseGPUSelectors(raw string) ([]gpuSelector, error) {
	var out []gpuSelector
	for _, token := range strings.Split(strings.TrimSpace(raw), ",") {
		token = strings.TrimSpace(token)
		lower := strings.ToLower(token)
		s := gpuSelector{Token: token, Index: -1, AMD: -1}
		switch {
		case strings.HasPrefix(lower, "amd:"):
			n, err := strconv.Atoi(strings.TrimSpace(token[len("amd:"):]))
			if err != nil || n < 0 {
				return nil, fmt.Errorf("%q: amd:<n> takes a non-negative number", token)
			}
			s.AMD = n
		case strings.HasPrefix(lower, "pci:"):
			address, ok := detect.CanonicalPCIAddress(token[len("pci:"):])
			if !ok {
				return nil, fmt.Errorf("%q: pci:<address> takes a PCI address such as 0000:0c:00.0", token)
			}
			s.PCI = address
		default:
			n, err := strconv.Atoi(token)
			if err != nil || n < 0 {
				return nil, fmt.Errorf("%q is not a non-negative GPU index, amd:<n> or pci:<address>", token)
			}
			s.Index = n
		}
		out = append(out, s)
	}
	return out, nil
}

// gpusFlagSyntaxError is parseLaunchArgs' --gpus check: today's for numeric
// indexes; the selector syntax only for launch and dry-run with the opt-in.
func gpusFlagSyntaxError(raw string) error {
	_, err := parseGPUIndices(raw)
	if err == nil || !gpuSelectorsAccepted || !detect.ExperimentalAMD() || !hasVendorSelector(raw) {
		return err
	}
	_, err = parseGPUSelectors(raw)
	return err
}

func vendorLabel(vendor string) string {
	switch vendor {
	case "nvidia":
		return "NVIDIA"
	case "amd":
		return "AMD"
	case "intel":
		return "Intel"
	}
	return vendor
}

func inventoryGPUList(inv []detect.PCIDisplay) string {
	var parts []string
	for _, d := range inv {
		if d.Vendor == "nvidia" || d.Vendor == "amd" || d.Vendor == "intel" {
			parts = append(parts, fmt.Sprintf("%s %s (%s)", vendorLabel(d.Vendor), d.Address, driverOrNone(d.Driver)))
		}
	}
	if len(parts) == 0 {
		return "none"
	}
	return strings.Join(parts, ", ")
}

func driverOrNone(driver string) string {
	if driver == "" {
		return "no driver"
	}
	return driver
}

// resolveGPUSelectors resolves the amd: and pci: tokens of --gpus against the
// OS inventory, before a backend is chosen. Without such a token it reads
// nothing and changes nothing. A selection of NVIDIA cards is rewritten to
// their detected indexes, and the note says so. A selection of AMD cards is
// kept on req for applyVendorSelection, which needs the backend. Everything
// else is refused: a single-vendor host, mixed vendors, an unknown or unusable
// card, a card whose memory is unknown, and a separate Claude Code reviewer
// (it seats on the NVIDIA cards by their indexes).
func resolveGPUSelectors(req *launchRequest, caps *detect.Capabilities, displays func() []detect.PCIDisplay) (string, error) {
	if req == nil || !hasVendorSelector(req.GPUsFlag) {
		return "", nil
	}
	flag := req.GPUsFlag
	if !detect.ExperimentalAMD() {
		return "", fmt.Errorf("--gpus %s: amd:<n> and pci:<address> need GGRUN_EXPERIMENTAL_AMD=1 on Linux", flag)
	}
	selectors, err := parseGPUSelectors(flag)
	if err != nil {
		return "", fmt.Errorf("--gpus: %w", err)
	}
	var inv []detect.PCIDisplay
	if displays != nil {
		inv = displays()
	}
	if n := inventoryGPUVendors(inv); n < 2 {
		return "", fmt.Errorf("--gpus %s: amd:<n> and pci:<address> select a card on a host with GPUs of more than one vendor; the PCI inventory shows %d (%s); use numeric --gpus", flag, n, inventoryGPUList(inv))
	}
	byAddress := map[string]detect.PCIDisplay{}
	var amd []detect.PCIDisplay
	for _, d := range inv {
		byAddress[d.Address] = d
		if d.Vendor == "amd" && d.Driver == "amdgpu" {
			amd = append(amd, d)
		}
	}
	detectedAt := map[string]int{}
	if caps != nil {
		for _, g := range caps.GPUs {
			if address, ok := detect.CanonicalPCIAddress(g.PCIBusID); ok {
				detectedAt[address] = g.Index
			}
		}
	}
	type pick struct {
		token    string
		display  detect.PCIDisplay
		detected int
	}
	var picks []pick
	named := map[string]string{}
	byVendor := map[string][]string{}
	var vendorOrder []string
	for _, s := range selectors {
		p := pick{token: s.Token, detected: -1}
		switch {
		case s.Index >= 0:
			var g *detect.GPU
			if caps != nil {
				for i := range caps.GPUs {
					if caps.GPUs[i].Index == s.Index {
						g = &caps.GPUs[i]
					}
				}
			}
			if g == nil {
				return "", fmt.Errorf("--gpus %s: no detected GPU with index %d", flag, s.Index)
			}
			address, ok := detect.CanonicalPCIAddress(g.PCIBusID)
			d, found := byAddress[address]
			if !ok || !found || d.Vendor != "nvidia" {
				return "", fmt.Errorf("--gpus %s: GPU %d (%s) is not an NVIDIA card of the PCI inventory, so it cannot be combined with amd:<n> or pci:<address>", flag, s.Index, g.Name)
			}
			p.display, p.detected = d, s.Index
		case s.AMD >= 0:
			if s.AMD >= len(amd) {
				return "", fmt.Errorf("--gpus %s: amd:%d names no card; this host has %d AMD GPU(s) driven by amdgpu (GPUs: %s)", flag, s.AMD, len(amd), inventoryGPUList(inv))
			}
			p.display = amd[s.AMD]
		default:
			d, ok := byAddress[s.PCI]
			if !ok {
				return "", fmt.Errorf("--gpus %s: no GPU at PCI %s (GPUs: %s)", flag, s.PCI, inventoryGPUList(inv))
			}
			p.display = d
			if i, ok := detectedAt[s.PCI]; ok {
				p.detected = i
			}
		}
		if other, dup := named[p.display.Address]; dup {
			return "", fmt.Errorf("--gpus %s: %s and %s name the same card (%s)", flag, other, p.token, p.display.Address)
		}
		named[p.display.Address] = p.token
		v := p.display.Vendor
		if _, ok := byVendor[v]; !ok {
			vendorOrder = append(vendorOrder, v)
		}
		byVendor[v] = append(byVendor[v], fmt.Sprintf("%s at %s", p.token, p.display.Address))
		picks = append(picks, p)
	}
	if len(vendorOrder) > 1 {
		var parts []string
		for _, v := range vendorOrder {
			parts = append(parts, vendorLabel(v)+": "+strings.Join(byVendor[v], ", "))
		}
		return "", fmt.Errorf("--gpus %s mixes GPU vendors (%s); one launch runs on one vendor's GPUs: select only NVIDIA GPUs (numeric or pci:<address>) or only AMD GPUs (amd:<n> or pci:<address>)", flag, strings.Join(parts, "; "))
	}
	switch vendorOrder[0] {
	case "nvidia":
		indices := make([]int, 0, len(picks))
		for _, p := range picks {
			if p.detected < 0 {
				return "", fmt.Errorf("--gpus %s: the NVIDIA card at %s is not among the GPUs nvidia-smi reports", flag, p.display.Address)
			}
			indices = append(indices, p.detected)
		}
		sort.Ints(indices)
		parts := make([]string, len(indices))
		for i, idx := range indices {
			parts[i] = strconv.Itoa(idx)
		}
		req.GPUsFlag = strings.Join(parts, ",")
		return fmt.Sprintf("--gpus %s selects detected GPU(s) %s", flag, req.GPUsFlag), nil
	case "amd":
	default:
		p := picks[0]
		return "", fmt.Errorf("--gpus %s: the card at %s is a %s GPU; only NVIDIA and AMD GPUs can be selected", flag, p.display.Address, vendorLabel(p.display.Vendor))
	}
	// The cards are appended after the detected GPUs, which must be the NVIDIA
	// cards nvidia-smi reports: GPUs detected another way (vulkaninfo, without
	// a PCI address) may already include the AMD card.
	if caps != nil {
		for _, g := range caps.GPUs {
			address, ok := detect.CanonicalPCIAddress(g.PCIBusID)
			if d, found := byAddress[address]; !ok || !found || d.Vendor != "nvidia" {
				return "", fmt.Errorf("--gpus %s: detected GPU %d (%s) is not an NVIDIA card nvidia-smi reports with a PCI address, so an AMD card cannot be added beside it", flag, g.Index, g.Name)
			}
		}
	}
	if req.ClaudeCode && !req.ClaudeReviewerDisabled {
		return "", fmt.Errorf("--gpus %s: the separate Claude Code reviewer is placed on the NVIDIA GPUs by index and is not supported beside an AMD selection yet; add --claude-reviewer off", flag)
	}
	sel := &vendorSelection{Flag: flag}
	for _, p := range picks {
		d := p.display
		if d.Driver != "amdgpu" {
			return "", fmt.Errorf("--gpus %s: the AMD card at %s is driven by %s, not amdgpu", flag, d.Address, driverOrNone(d.Driver))
		}
		if d.VRAMTotalMB <= 0 {
			why := d.MemoryNote
			if why == "" {
				why = "amdgpu reports no VRAM total"
			}
			next := "check /sys/bus/pci/devices/" + d.Address + "/mem_info_vram_total"
			if strings.Contains(why, "runtime-") {
				// Reading the memory files would wake the card, so the inventory
				// does not; the user decides to wake it.
				next = "wake it (run any GPU job on it, or as root write \"on\" to /sys/bus/pci/devices/" + d.Address +
					"/power/control) and check /sys/bus/pci/devices/" + d.Address + "/power/runtime_status reads \"active\""
			}
			return "", fmt.Errorf("--gpus %s: the memory of %s at %s is unknown (%s); %s, then retry", flag, d.Name, d.Address, why, next)
		}
		sel.Devices = append(sel.Devices, vendorDevice{Display: d, Token: p.token})
	}
	sort.Slice(sel.Devices, func(i, j int) bool { return sel.Devices[i].Display.Address < sel.Devices[j].Display.Address })
	req.vendorSelection = sel
	return "", nil
}

// vendorListers are the backend listings applyVendorSelection runs.
type vendorListers struct {
	ListVulkan      func(path string) ([]listedDevice, error)
	ListVulkanUnder func(path, visible string) (string, error)
	ListUnderEnv    func(path string, env []string) (string, error)
	Environ         func() []string
}

func launchVendorListers() vendorListers {
	return vendorListers{ListVulkan: listVulkanDevices, ListVulkanUnder: listVulkanOutputUnder, ListUnderEnv: listDevicesUnderEnv, Environ: os.Environ}
}

// listDevicesUnderEnv is the backend's listing under exactly env.
func listDevicesUnderEnv(path string, env []string) (string, error) {
	res := devicelist.Run(devicelist.Probe{Binary: path, Env: env, Timeout: devicelist.DefaultTimeout})
	if res.TimedOut {
		return "", fmt.Errorf("%s --list-devices timed out", path)
	}
	if res.Output == "" && res.Err != nil {
		return "", fmt.Errorf("%s --list-devices: %w", path, res.Err)
	}
	return res.Output, nil
}

const rocmVisibleDevicesKey = "ROCR_VISIBLE_DEVICES"

// rocmHiddenVisibility are the variables a HIP build also reads as a device
// selection, by AMD index (CUDA_VISIBLE_DEVICES among them): unset, so only
// the ROCr list applies.
var rocmHiddenVisibility = []string{"CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}

func rocmListingEnv(base []string, value string) []string {
	env := make([]string, 0, len(base)+1)
	for _, kv := range base {
		key, _, _ := strings.Cut(kv, "=")
		if key == rocmVisibleDevicesKey || slices.Contains(rocmHiddenVisibility, key) {
			continue
		}
		env = append(env, kv)
	}
	return append(env, rocmVisibleDevicesKey+"="+value)
}

// rocmDeviceID is ROCr's name for a card: GPU- and the amdgpu unique id as 16
// hex digits. A missing or zero id has no name.
func rocmDeviceID(uniqueID string) (string, bool) {
	v, err := strconv.ParseUint(strings.TrimPrefix(strings.ToLower(strings.TrimSpace(uniqueID)), "0x"), 16, 64)
	if err != nil || v == 0 {
		return "", false
	}
	return fmt.Sprintf("GPU-%016x", v), true
}

// applyVendorSelection lists the selected AMD cards with the launch's backend,
// sets the dialect and the visibility, appends the cards to the launch's GPUs
// and rewrites --gpus to their indexes. Without an AMD selection it returns
// caps unchanged and runs nothing. Any doubt refuses the launch.
func applyVendorSelection(req *launchRequest, be *backendInfo, caps *detect.Capabilities, in vendorListers) (*detect.Capabilities, string, error) {
	if req == nil || req.vendorSelection == nil {
		return caps, "", nil
	}
	sel := req.vendorSelection
	if be == nil || be.IsIK || strings.EqualFold(be.Dialect, "ik_llama") || strings.EqualFold(be.Dialect, "metal") {
		return nil, "", fmt.Errorf("--gpus %s: an AMD card needs a mainline llama.cpp Vulkan or HIP (ROCm) build; pass its llama-server with --server-bin", sel.Flag)
	}
	var err error
	if strings.EqualFold(be.Dialect, "vulkan") {
		err = selectOnVulkan(sel, be, caps, in)
	} else {
		err = selectOnROCm(sel, be, caps, in)
	}
	if err != nil {
		return nil, "", fmt.Errorf("--gpus %s: %w", sel.Flag, err)
	}
	var out detect.Capabilities
	if caps != nil {
		out = *caps
	}
	out.GPUs = append([]detect.GPU(nil), out.GPUs...)
	base := 0
	for _, g := range out.GPUs {
		if g.Index >= base {
			base = g.Index + 1
		}
	}
	hosts := make([]string, 0, len(sel.Devices))
	for k := range sel.Devices {
		sel.Devices[k].Host = base + k
		out.GPUs = append(out.GPUs, vendorDeviceGPU(sel.Devices[k]))
		hosts = append(hosts, strconv.Itoa(sel.Devices[k].Host))
	}
	req.GPUsFlag = strings.Join(hosts, ",")
	be.Dialect = sel.Dialect
	return &out, sel.describe(), nil
}

func (sel *vendorSelection) describe() string {
	parts := make([]string, 0, len(sel.Devices))
	for _, d := range sel.Devices {
		parts = append(parts, fmt.Sprintf("%s = %s at %s (backend %s, GPU %d)", d.Token, d.Name, d.Display.Address, d.Row, d.Host))
	}
	vis := sel.VisKey + "=" + sel.VisValue
	if len(sel.Unset) > 0 {
		vis += ", " + strings.Join(sel.Unset, ", ") + " unset"
	}
	return fmt.Sprintf("AMD selection: %s; %s; the backend's device lines are checked after load", strings.Join(parts, ", "), vis)
}

var pcieLaneMBps = map[int]int{1: 250, 2: 500, 3: 985, 4: 1969, 5: 3938}

// vendorDeviceGPU is the launch's view of a selected card: name and listing
// from the backend, memory, address and slot link from the inventory.
func vendorDeviceGPU(d vendorDevice) detect.GPU {
	g := detect.GPU{
		Index: d.Host, Name: d.Name, VRAMTotalMB: d.Display.VRAMTotalMB, VRAMUsedMB: d.Display.VRAMUsedMB,
		Driver: d.Display.Driver, PCIBusID: d.Display.Address, PCIGen: d.Display.LinkGen, PCILanes: d.Display.LinkWidth,
	}
	if perLane := pcieLaneMBps[g.PCIGen]; perLane > 0 && g.PCILanes > 0 {
		g.BandwidthMBps, g.BandwidthSource = perLane*g.PCILanes, "sysfs_link"
	}
	return g
}

// selectOnVulkan maps the selection to the backend's raw Vulkan indexes and
// checks the listing under them.
func selectOnVulkan(sel *vendorSelection, be *backendInfo, caps *detect.Capabilities, in vendorListers) error {
	if in.ListVulkan == nil || in.ListVulkanUnder == nil {
		return fmt.Errorf("the Vulkan backend's device listing is not available")
	}
	listed, err := in.ListVulkan(be.Path)
	if err != nil {
		return err
	}
	launch, err := vulkanVendorDevices(sel, caps, listed)
	if err != nil {
		return err
	}
	listUnder := func(visible string) (string, error) { return in.ListVulkanUnder(be.Path, visible) }
	raw, err := vulkanRawDevices(listUnder)
	if err != nil {
		return fmt.Errorf("the raw Vulkan device list could not be read: %v", err)
	}
	indices, err := vulkanRawIndices(listed, launch, raw)
	if err != nil {
		return fmt.Errorf("the raw Vulkan index of the selection is ambiguous: %v", err)
	}
	value := strings.Join(indices, ",")
	out, err := listUnder(value)
	if err == nil {
		err = vulkanVisibleListingMatches(launch, parseVulkanDeviceList(out))
	}
	if err != nil {
		return fmt.Errorf("the listing under %s=%s does not show the selection: %v", vulkanVisibleDevicesKey, value, err)
	}
	for k := range sel.Devices {
		sel.Devices[k].Row = "Vulkan" + strconv.Itoa(k)
		sel.Devices[k].Name, sel.Devices[k].TotalMB = launch[k].Name, launch[k].TotalMB
	}
	sel.Dialect, sel.VisKey, sel.VisValue, sel.Unset = "vulkan", vulkanVisibleDevicesKey, value, nil
	return nil
}

// vulkanVendorDevices finds each selected card in the backend's default
// listing: a device whose total matches the card's and that is neither named
// like a detected GPU nor for another vendor or a CPU renderer. A card needs
// exactly one candidate. Cards that share one candidate set are paired with it
// in order only when the selection holds all of them (identical cards; the
// post-load check is then the authority on which is which); otherwise the
// match is ambiguous and refused.
func vulkanVendorDevices(sel *vendorSelection, caps *detect.Capabilities, listed []listedDevice) ([]listedDevice, error) {
	detected := map[string]bool{}
	if caps != nil {
		for _, g := range caps.GPUs {
			detected[strings.ToLower(strings.TrimSpace(g.Name))] = true
		}
	}
	excluded := func(name, vendor string) bool {
		for _, other := range []string{"nvidia", "intel", "amd", "llvmpipe", "lavapipe", "swiftshader"} {
			if other != vendor && strings.Contains(name, other) {
				return true
			}
		}
		return false
	}
	candidates := make([][]listedDevice, len(sel.Devices))
	keys := make([]string, len(sel.Devices))
	for i, d := range sel.Devices {
		var idx []string
		for _, l := range listed {
			name := strings.ToLower(strings.TrimSpace(l.Name))
			if detected[name] || excluded(name, d.Display.Vendor) || !admissionTotalMatches(l.TotalMB, d.Display.VRAMTotalMB) {
				continue
			}
			candidates[i] = append(candidates[i], l)
			idx = append(idx, strconv.Itoa(l.Index))
		}
		keys[i] = strings.Join(idx, ",")
	}
	launch := make([]listedDevice, len(sel.Devices))
	used := map[int]bool{}
	for i, d := range sel.Devices {
		c := candidates[i]
		if len(c) == 0 {
			return nil, fmt.Errorf("the backend's Vulkan listing shows no device like %s at %s (%d MiB); listed: %s", d.Display.Name, d.Display.Address, d.Display.VRAMTotalMB, describeListed(listed))
		}
		pick := c[0]
		if len(c) > 1 {
			var same []int
			for j := range sel.Devices {
				if keys[j] == keys[i] {
					same = append(same, j)
				}
			}
			if len(same) != len(c) {
				return nil, fmt.Errorf("%s at %s matches %d Vulkan devices by name and total (%s) and no Vulkan listing gives a PCI address; the match is ambiguous: select all of those cards, or use a HIP (ROCm) build", d.Display.Name, d.Display.Address, len(c), describeListed(c))
			}
			pick = c[slices.Index(same, i)]
		}
		if used[pick.Index] {
			return nil, fmt.Errorf("Vulkan%d matches more than one selected card", pick.Index)
		}
		used[pick.Index] = true
		launch[i] = pick
	}
	return launch, nil
}

func describeListed(listed []listedDevice) string {
	parts := make([]string, 0, len(listed))
	for _, l := range listed {
		parts = append(parts, fmt.Sprintf("Vulkan%d %s (%d MiB)", l.Index, l.Name, l.TotalMB))
	}
	if len(parts) == 0 {
		return "none"
	}
	return strings.Join(parts, ", ")
}

// selectOnROCm lists the backend under ROCR_VISIBLE_DEVICES set to the
// selection's KFD ids. ROCm rows make it the ROCm dialect; Vulkan rows mean a
// Vulkan build whose path does not say so; anything else is refused.
func selectOnROCm(sel *vendorSelection, be *backendInfo, caps *detect.Capabilities, in vendorListers) error {
	ids := make([]string, 0, len(sel.Devices))
	for _, d := range sel.Devices {
		id, ok := rocmDeviceID(d.Display.KFDUniqueID)
		if !ok {
			return fmt.Errorf("%s at %s has no KFD unique id (no amdgpu KFD node matches its PCI address), so ROCm cannot be pointed at it", d.Display.Name, d.Display.Address)
		}
		ids = append(ids, id)
	}
	value := strings.Join(ids, ",")
	if in.ListUnderEnv == nil {
		return fmt.Errorf("the backend's device listing is not available")
	}
	var base []string
	if in.Environ != nil {
		base = in.Environ()
	}
	out, err := in.ListUnderEnv(be.Path, rocmListingEnv(base, value))
	if err != nil {
		return err
	}
	rows := parseListedMemory("ROCm", out)
	if len(rows) == 0 {
		if len(parseVulkanDeviceList(out)) > 0 {
			return selectOnVulkan(sel, be, caps, in)
		}
		return fmt.Errorf("%s lists no ROCm or Vulkan device under %s=%s; an AMD card needs a Vulkan or HIP (ROCm) build of llama-server, passed with --server-bin", be.Path, rocmVisibleDevicesKey, value)
	}
	if len(rows) != len(sel.Devices) {
		return fmt.Errorf("%s lists %d ROCm devices under %s=%s, the selection has %d", be.Path, len(rows), rocmVisibleDevicesKey, value, len(sel.Devices))
	}
	for k, r := range rows {
		d := sel.Devices[k]
		if r.Ordinal != k || !admissionTotalMatches(r.TotalMB, d.Display.VRAMTotalMB) {
			return fmt.Errorf("ROCm%d is %s (%d MiB), the selection expects %s at %s (%d MiB) there", r.Ordinal, r.Name, r.TotalMB, d.Display.Name, d.Display.Address, d.Display.VRAMTotalMB)
		}
	}
	for k := range sel.Devices {
		sel.Devices[k].Row = "ROCm" + strconv.Itoa(k)
		sel.Devices[k].Name, sel.Devices[k].TotalMB = rows[k].Name, rows[k].TotalMB
	}
	sel.Dialect, sel.VisKey, sel.VisValue = "rocm", rocmVisibleDevicesKey, value
	sel.Unset = append([]string(nil), rocmHiddenVisibility...)
	return nil
}

// gpuVisibilityUnset is what the launch unsets beside the visibility key.
func gpuVisibilityUnset(req *launchRequest, key string) []string {
	if req == nil || req.vendorSelection == nil || key == "" || key != req.vendorSelection.VisKey {
		return nil
	}
	return req.vendorSelection.Unset
}

// usingDevicePattern is llama.cpp's line for each device a model loads on,
// printed before any tensor is loaded: "<func>: using device ROCm0 (<desc>)
// (<id>) - <n> MiB free". The description may hold parentheses; the id is
// the last group, "unknown id" when the backend reports none. The "skipping
// device ... already using device" line does not match.
var usingDevicePattern = regexp.MustCompile(`(?:^|\s)\w+: using device (\S+) \((.*)\) \(([^()]*)\) - \d+ MiB free`)

// usingDeviceNoIDPattern is the same line from a build that prints no id
// field at all: "<func>: using device Vulkan0 (<desc>) - <n> MiB free".
var usingDeviceNoIDPattern = regexp.MustCompile(`(?:^|\s)\w+: using device (\S+) \(.*\) - \d+ MiB free`)

// deviceOrderMismatchError stops a launch whose backend loaded on devices
// other than the plan's.
type deviceOrderMismatchError struct{ msg string }

func (e *deviceOrderMismatchError) Error() string { return e.msg }

// checkUsingDevices requires every "using device" line of the backend to name
// a row of the selection at its planned PCI address. The backend prints lines
// only for the devices it uses (a plan on one of two selected cards names one),
// so a selected row without a line is fine; requireLine asks for at least one
// line, which a successful load always prints.
func checkUsingDevices(sel *vendorSelection, log string, requireLine bool) error {
	if sel == nil {
		return nil
	}
	ids := map[string]string{}
	var order, noID []string
	for _, line := range strings.Split(log, "\n") {
		m := usingDevicePattern.FindStringSubmatch(line)
		if m == nil {
			if n := usingDeviceNoIDPattern.FindStringSubmatch(line); n != nil && !slices.Contains(noID, n[1]) {
				noID = append(noID, n[1])
			}
			continue
		}
		if _, seen := ids[m[1]]; !seen {
			order = append(order, m[1])
		}
		ids[m[1]] = strings.TrimSpace(m[3])
	}
	planned := map[string]vendorDevice{}
	plannedOrder := make([]string, 0, len(sel.Devices))
	for _, d := range sel.Devices {
		planned[d.Row] = d
		plannedOrder = append(plannedOrder, d.Row+"="+d.Display.Address)
	}
	backendOrder := make([]string, 0, len(order))
	for _, row := range order {
		backendOrder = append(backendOrder, row+"="+ids[row])
	}
	const stopped = "; the launch is stopped before serving because the backend's devices are not the plan's"
	fail := func(next, format string, args ...any) error {
		return &deviceOrderMismatchError{msg: fmt.Sprintf(format, args...) + stopped + "; " + next}
	}
	next := "please report the backend's log"
	if len(sel.Devices) > 1 {
		next = "select one card with --gpus pci:<address>, which needs no order, and please report both orders"
	}
	const noIDNext = "use a llama.cpp build that reports each device's PCI id"
	if len(order) == 0 {
		// A failed start without a device proof goes to today's handling.
		if !requireLine {
			return nil
		}
		if len(noID) > 0 {
			return fail(noIDNext, "the backend's \"using device\" line for %s has no PCI id, so the device order cannot be checked", strings.Join(noID, ", "))
		}
		return fail("check the backend's log for its device lines", "the backend printed no \"using device\" line (planned: %s)", strings.Join(plannedOrder, ", "))
	}
	for _, row := range order {
		d, ok := planned[row]
		if !ok {
			return fail(next, "the backend uses %s (%s), which the plan does not contain (planned: %s)", row, ids[row], strings.Join(plannedOrder, ", "))
		}
		if strings.EqualFold(ids[row], "unknown id") {
			if !requireLine {
				continue
			}
			return fail(noIDNext, "the backend reports no PCI id for %s (planned at %s), so the device order cannot be checked", row, d.Display.Address)
		}
		if pci, ok := detect.CanonicalPCIAddress(ids[row]); !ok || pci != d.Display.Address {
			return fail(next,
				"backend device %s is %s, the plan expected %s (%s); planned order %s, backend order %s",
				row, ids[row], d.Display.Address, d.Name, strings.Join(plannedOrder, ", "), strings.Join(backendOrder, ", "))
		}
	}
	return nil
}

// postLoadDeviceCheck is checkUsingDevices on a started backend's log; nil
// without an AMD selection.
func postLoadDeviceCheck(req *launchRequest, p *server.Process) error {
	if req == nil || req.vendorSelection == nil {
		return nil
	}
	log := ""
	if p != nil && p.LogBuf != nil {
		log = p.LogBuf.String()
	}
	return admissionUnknown(checkUsingDevices(req.vendorSelection, log, true))
}

// failedLoadDeviceCheck is the same check on a failed start: lines naming
// other devices stop the launch; no line at all is today's failure handling.
func failedLoadDeviceCheck(req *launchRequest, log string) error {
	if req == nil || req.vendorSelection == nil {
		return nil
	}
	return admissionUnknown(checkUsingDevices(req.vendorSelection, log, false))
}
