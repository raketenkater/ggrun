package detect

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

// PCIDisplay is one display or compute-accelerator PCI function as the Linux
// kernel lists it under /sys/bus/pci/devices: the operating system's GPU
// inventory. It records presence, identity, PCI address, the slot link and
// whole-card memory. It never adds an entry to Capabilities.GPUs: a GPU is
// plannable only once a backend can be pointed at it, and that is decided by
// the backend's own device listing, not by this inventory.
type PCIDisplay struct {
	Address  string `json:"pci_address"` // canonical, e.g. 0000:19:00.0
	Vendor   string `json:"vendor"`      // nvidia, amd, intel, or the vendor id
	VendorID string `json:"vendor_id"`   // e.g. 0x1002
	DeviceID string `json:"device_id"`   // e.g. 0x66a1
	Class    string `json:"class"`       // e.g. 0x038000
	Name     string `json:"name"`
	// Driver is the bound kernel driver; empty when none is bound.
	Driver string `json:"driver,omitempty"`
	// VRAMTotalMB and VRAMUsedMB come from amdgpu's mem_info_vram_* files.
	// Zero is unknown; MemoryNote says why when it is known.
	VRAMTotalMB int    `json:"vram_total_mb,omitempty"`
	VRAMUsedMB  int    `json:"vram_used_mb,omitempty"`
	MemoryNote  string `json:"memory_note,omitempty"`
	// LinkGen and LinkWidth describe the slot link: read on the topmost PCI
	// function of the card (a card with its own PCIe switch reports the
	// switch's internal link on the GPU function), with the generation capped
	// by the bridge above it. LinkAt is the function they were read on.
	LinkGen   int    `json:"link_gen,omitempty"`
	LinkWidth int    `json:"link_width,omitempty"`
	LinkAt    string `json:"link_at,omitempty"`
	// KFDUniqueID is the amdgpu unique id (hex), as ROCm names the device
	// (GPU-<id>); GFXTarget is the KFD graphics target, e.g. gfx906.
	KFDUniqueID string `json:"kfd_unique_id,omitempty"`
	GFXTarget   string `json:"gfx_target,omitempty"`
}

// ExperimentalAMD reports the opt-in for multi-vendor behaviour
// (GGRUN_EXPERIMENTAL_AMD=1). It is ignored outside Linux.
func ExperimentalAMD() bool {
	return runtime.GOOS == "linux" && os.Getenv("GGRUN_EXPERIMENTAL_AMD") == "1"
}

// HostPCIDisplays is this host's inventory when ExperimentalAMD is set, and
// nil otherwise: without the opt-in nothing is read.
func HostPCIDisplays() []PCIDisplay {
	if !ExperimentalAMD() {
		return nil
	}
	displays, err := ReadPCIDisplays("/")
	if err != nil {
		return nil
	}
	return displays
}

// Known AMD compute parts by PCI device id, named after the PCI ID
// repository (pci-ids.ucw.cz). Anything else is "AMD GPU <device id>".
var amdPCIDeviceNames = map[string]string{
	"0x66a0": "AMD Radeon Pro / Radeon Instinct (Vega 20)",
	"0x66a1": "AMD Radeon Pro VII / Radeon Instinct MI50 (Vega 20)",
	"0x66a2": "AMD Vega 20",
	"0x66a3": "AMD Radeon Pro Vega II (Vega 20)",
	"0x66a7": "AMD Radeon Pro Vega 20",
	"0x66af": "AMD Radeon VII (Vega 20)",
	"0x738c": "AMD Instinct MI100",
	"0x738e": "AMD Instinct MI100",
	"0x7408": "AMD Instinct MI250X",
	"0x740c": "AMD Instinct MI250X / MI250",
	"0x740f": "AMD Instinct MI210",
	"0x74a1": "AMD Instinct MI300X",
	"0x74a5": "AMD Instinct MI325X",
}

var pciVendorNames = map[string]string{"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}

// ReadPCIDisplays parses the display (class 0x03) and processing-accelerator
// (class 0x12) functions under root/sys/bus/pci/devices, sorted by address.
// root is "/" on a host and a fixture directory in tests. Every value is a
// plain file read; a missing attribute leaves its field unknown.
func ReadPCIDisplays(root string) ([]PCIDisplay, error) {
	devicesDir := filepath.Join(root, "sys", "bus", "pci", "devices")
	entries, err := os.ReadDir(devicesDir)
	if err != nil {
		return nil, fmt.Errorf("PCI device list unavailable: %w", err)
	}
	kfd := readKFDNodes(root)
	var out []PCIDisplay
	for _, entry := range entries {
		address, ok := CanonicalPCIAddress(entry.Name())
		if !ok {
			continue
		}
		dir := filepath.Join(devicesDir, entry.Name())
		class := strings.ToLower(readSysfs(dir, "class"))
		if !strings.HasPrefix(class, "0x03") && !strings.HasPrefix(class, "0x12") {
			continue
		}
		d := PCIDisplay{
			Address:  address,
			VendorID: strings.ToLower(readSysfs(dir, "vendor")),
			DeviceID: strings.ToLower(readSysfs(dir, "device")),
			Class:    class,
		}
		d.Vendor = pciVendorNames[d.VendorID]
		if d.Vendor == "" {
			d.Vendor = d.VendorID
		}
		if target, err := os.Readlink(filepath.Join(dir, "driver")); err == nil {
			d.Driver = filepath.Base(target)
		}
		d.Name = pciDisplayName(d)
		if d.Driver == "amdgpu" {
			readAMDGPUMemory(dir, &d)
			d.KFDUniqueID = strings.ToLower(readSysfs(dir, "unique_id"))
			if node, ok := kfd[address]; ok {
				if d.KFDUniqueID == "" {
					d.KFDUniqueID = node.uniqueID
				}
				d.GFXTarget = node.gfxTarget
			}
		}
		readSlotLink(dir, &d)
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Address < out[j].Address })
	return out, nil
}

func pciDisplayName(d PCIDisplay) string {
	if d.Vendor == "amd" {
		if name, ok := amdPCIDeviceNames[d.DeviceID]; ok {
			return name
		}
		return "AMD GPU " + d.DeviceID
	}
	switch d.Vendor {
	case "nvidia":
		return "NVIDIA GPU " + d.DeviceID
	case "intel":
		return "Intel GPU " + d.DeviceID
	}
	return "PCI display " + d.VendorID + ":" + d.DeviceID
}

// readAMDGPUMemory reads whole-card VRAM from amdgpu, unless the card is
// runtime-suspended: reading the memory attributes would wake it.
func readAMDGPUMemory(dir string, d *PCIDisplay) {
	if status := readSysfs(dir, filepath.Join("power", "runtime_status")); status != "" && status != "active" {
		d.MemoryNote = "not read: the card is runtime-" + status
		return
	}
	if total, ok := readSysfsInt(dir, "mem_info_vram_total"); ok && total > 0 {
		d.VRAMTotalMB = int(total / (1 << 20))
	}
	if used, ok := readSysfsInt(dir, "mem_info_vram_used"); ok && used >= 0 {
		d.VRAMUsedMB = int(used / (1 << 20))
	}
}

// readSlotLink walks up from the GPU function through PCI functions of the
// same vendor (the card's own PCIe switch) and reads the link of the topmost
// one, which is the slot link. Its generation is the lower of its own maximum
// and that of the bridge above it, so a Gen 4 card in a Gen 3 slot reads 3.
func readSlotLink(dir string, d *PCIDisplay) {
	resolved, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return
	}
	top := resolved
	for {
		parent := filepath.Dir(top)
		if _, ok := CanonicalPCIAddress(filepath.Base(parent)); !ok {
			break
		}
		if !strings.EqualFold(readSysfs(parent, "vendor"), d.VendorID) {
			break
		}
		top = parent
	}
	gen := pcieGenFromSpeed(readSysfs(top, "max_link_speed"))
	bridge := filepath.Dir(top)
	if _, ok := CanonicalPCIAddress(filepath.Base(bridge)); ok {
		if bridgeGen := pcieGenFromSpeed(readSysfs(bridge, "max_link_speed")); bridgeGen > 0 && (gen == 0 || bridgeGen < gen) {
			gen = bridgeGen
		}
	}
	if gen == 0 {
		gen = pcieGenFromSpeed(readSysfs(top, "current_link_speed"))
	}
	width, _ := readSysfsInt(top, "current_link_width")
	if width <= 0 {
		width, _ = readSysfsInt(top, "max_link_width")
	}
	if gen <= 0 || width <= 0 {
		return
	}
	d.LinkGen, d.LinkWidth = gen, int(width)
	if address, ok := CanonicalPCIAddress(filepath.Base(top)); ok {
		d.LinkAt = address
	}
}

// pcieGenFromSpeed maps a sysfs link speed ("8.0 GT/s PCIe") to a generation.
func pcieGenFromSpeed(speed string) int {
	fields := strings.Fields(speed)
	if len(fields) == 0 {
		return 0
	}
	gts, err := strconv.ParseFloat(fields[0], 64)
	if err != nil || gts <= 0 {
		return 0
	}
	for gen, ceiling := range []float64{2.5, 5, 8, 16, 32, 64} {
		if gts <= ceiling {
			return gen + 1
		}
	}
	return 6
}

type kfdNode struct {
	uniqueID  string
	gfxTarget string
}

// readKFDNodes maps PCI addresses to the KFD topology's GPU nodes. KFD is
// part of the in-kernel amdgpu driver; no ROCm installation is needed.
func readKFDNodes(root string) map[string]kfdNode {
	nodes := map[string]kfdNode{}
	dirs, _ := filepath.Glob(filepath.Join(root, "sys", "class", "kfd", "kfd", "topology", "nodes", "*", "properties"))
	for _, path := range dirs {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		props := map[string]uint64{}
		for _, line := range strings.Split(string(data), "\n") {
			fields := strings.Fields(line)
			if len(fields) != 2 {
				continue
			}
			if v, err := strconv.ParseUint(fields[1], 10, 64); err == nil {
				props[fields[0]] = v
			}
		}
		location, ok := props["location_id"]
		if !ok || props["gfx_target_version"] == 0 {
			continue // a CPU node
		}
		address := fmt.Sprintf("%04x:%02x:%02x.%x", props["domain"], location>>8, (location>>3)&0x1f, location&0x7)
		node := kfdNode{}
		if id, ok := props["unique_id"]; ok && id != 0 {
			node.uniqueID = fmt.Sprintf("%016x", id)
		}
		v := props["gfx_target_version"]
		node.gfxTarget = fmt.Sprintf("gfx%d%x%x", v/10000, (v/100)%100, v%100)
		nodes[address] = node
	}
	return nodes
}

// CanonicalPCIAddress normalizes a PCI address from sysfs ("0000:65:00.0")
// or nvidia-smi ("00000000:65:00.0") to "dddd:bb:dd.f" in lower case.
func CanonicalPCIAddress(s string) (string, bool) {
	parts := strings.Split(strings.TrimSpace(s), ":")
	if len(parts) != 3 {
		return "", false
	}
	devfn := strings.Split(parts[2], ".")
	if len(devfn) != 2 {
		return "", false
	}
	domain, err1 := strconv.ParseUint(parts[0], 16, 32)
	bus, err2 := strconv.ParseUint(parts[1], 16, 8)
	dev, err3 := strconv.ParseUint(devfn[0], 16, 8)
	fn, err4 := strconv.ParseUint(devfn[1], 16, 8)
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil || dev > 0x1f || fn > 7 {
		return "", false
	}
	return fmt.Sprintf("%04x:%02x:%02x.%x", domain, bus, dev, fn), true
}

func readSysfs(dir, name string) string {
	data, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func readSysfsInt(dir, name string) (int64, bool) {
	v, err := strconv.ParseInt(readSysfs(dir, name), 10, 64)
	return v, err == nil
}

// reachableByAnotherAPI reports whether a GPU-API backend could open this
// card: a known GPU vendor with a bound driver that is not a VM passthrough.
// A server board's BMC display (an unknown vendor) does not count.
func (d PCIDisplay) reachableByAnotherAPI() bool {
	known := d.Vendor == "nvidia" || d.Vendor == "amd" || d.Vendor == "intel"
	return known && d.Driver != "" && d.Driver != "vfio-pci"
}

// detectedFromNVIDIA reports whether caps' GPUs are the NVIDIA cards
// nvidia-smi reported, each found in the inventory by its PCI address.
func detectedFromNVIDIA(caps *Capabilities, inv []PCIDisplay) (map[string]bool, bool) {
	if caps == nil || caps.GPUStateUnknown != "" || len(caps.GPUs) == 0 {
		return nil, false
	}
	byAddress := map[string]PCIDisplay{}
	for _, d := range inv {
		byAddress[d.Address] = d
	}
	used := map[string]bool{}
	for _, g := range caps.GPUs {
		address, ok := CanonicalPCIAddress(g.PCIBusID)
		if !ok || byAddress[address].Vendor != "nvidia" {
			return nil, false
		}
		used[address] = true
	}
	return used, true
}

// OtherVendorBesideNVIDIA reports whether caps holds the NVIDIA cards
// nvidia-smi reported and the inventory also shows a GPU of another vendor
// that a non-CUDA backend (Vulkan) could open. It is the multi-vendor rule:
// only then may a launch run an extra device listing.
func OtherVendorBesideNVIDIA(caps *Capabilities, inv []PCIDisplay) bool {
	if _, ok := detectedFromNVIDIA(caps, inv); !ok {
		return false
	}
	for _, d := range inv {
		if d.Vendor != "nvidia" && d.reachableByAnotherAPI() {
			return true
		}
	}
	return false
}

// UnusableGPUNotes is one line per inventory GPU that ggrun cannot select for
// a launch, with the reason. It runs no backend: a card an installed backend
// could list is still reported while ggrun cannot point a launch at it. It is
// empty when caps' GPUs cannot be matched to the inventory (they came from
// vulkaninfo, which reports no PCI address), so a GPU ggrun does use is never
// reported.
func UnusableGPUNotes(caps *Capabilities, inv []PCIDisplay) []string {
	if caps == nil || caps.GPUStateUnknown != "" {
		return nil
	}
	used, fromNVIDIA := detectedFromNVIDIA(caps, inv)
	if !fromNVIDIA && len(caps.GPUs) > 0 {
		return nil
	}
	var notes []string
	for _, d := range inv {
		known := d.Vendor == "nvidia" || d.Vendor == "amd" || d.Vendor == "intel"
		if !known || used[d.Address] {
			continue
		}
		var reason string
		switch {
		case d.Driver == "":
			reason = "no kernel driver is bound to it"
		case d.Driver == "vfio-pci":
			reason = "it is bound to vfio-pci (reserved for a virtual machine)"
		case d.Vendor == "nvidia":
			reason = "nvidia-smi does not report it (driver " + d.Driver + ")"
		case fromNVIDIA:
			reason = "ggrun launches only on the NVIDIA GPUs that nvidia-smi reports; this " + vendorTitle(d.Vendor) + " GPU cannot be selected beside them yet"
		default:
			reason = "ggrun detected no usable GPU (no nvidia-smi answer and no vulkaninfo device)"
		}
		notes = append(notes, fmt.Sprintf("%s at %s (%s%s) is not used by ggrun: %s",
			d.Name, d.Address, driverLabel(d.Driver), memoryLabel(d), reason))
	}
	return notes
}

func vendorTitle(vendor string) string {
	switch vendor {
	case "amd":
		return "AMD"
	case "intel":
		return "Intel"
	}
	return vendor
}

func driverLabel(driver string) string {
	if driver == "" {
		return "no driver"
	}
	return "driver " + driver
}

func memoryLabel(d PCIDisplay) string {
	label := ""
	if d.VRAMTotalMB > 0 {
		label += fmt.Sprintf(", %d MiB", d.VRAMTotalMB)
	}
	if d.LinkGen > 0 && d.LinkWidth > 0 {
		label += fmt.Sprintf(", PCIe Gen%d x%d", d.LinkGen, d.LinkWidth)
	}
	return label
}
