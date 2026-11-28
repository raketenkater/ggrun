package detect

import (
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// fixtureFunction writes one PCI function's attributes under parent (a
// directory in the fixture's /sys/devices tree) and links it from
// /sys/bus/pci/devices like the kernel does. driver may be empty.
func fixtureFunction(t *testing.T, root, parent, address, driver string, attrs map[string]string) string {
	t.Helper()
	dir := filepath.Join(parent, address)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, value := range attrs {
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(value+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if driver != "" {
		target := filepath.Join(root, "sys", "bus", "pci", "drivers", driver)
		if err := os.MkdirAll(target, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(target, filepath.Join(dir, "driver")); err != nil {
			t.Fatal(err)
		}
	}
	if _, ok := CanonicalPCIAddress(address); ok && attrs["class"] != "" {
		link := filepath.Join(root, "sys", "bus", "pci", "devices", address)
		if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(dir, link); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// fixtureHost is a synthetic two-vendor host as its kernel would list it: two
// NVIDIA cards, an AMD Vega 20 card behind its own PCIe switch (upstream and
// downstream port) in a Gen 3 slot, a BMC display, an NPU and an SR-IOV
// virtual function. amdgpuDriver binds the AMD card.
func fixtureHost(t *testing.T, amdgpuDriver string, amdAttrs map[string]string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("the sysfs fixture uses symlinks")
	}
	root := t.TempDir()
	devices := filepath.Join(root, "sys", "devices")
	// BMC display on the chipset.
	pch := fixtureFunction(t, root, filepath.Join(devices, "pci0000:00"), "0000:00:1c.0", "pcieport", map[string]string{"vendor": "0x8086", "max_link_speed": "8.0 GT/s PCIe"})
	fixtureFunction(t, root, pch, "0000:03:00.0", "ast", map[string]string{"vendor": "0x1a03", "device": "0x2000", "class": "0x030000"})
	// MI50 behind its own switch.
	port := fixtureFunction(t, root, filepath.Join(devices, "pci0000:16"), "0000:16:00.0", "pcieport", map[string]string{
		"vendor": "0x8086", "max_link_speed": "8.0 GT/s PCIe", "current_link_speed": "8.0 GT/s PCIe", "current_link_width": "16"})
	up := fixtureFunction(t, root, port, "0000:17:00.0", "pcieport", map[string]string{
		"vendor": "0x1002", "max_link_speed": "16.0 GT/s PCIe", "current_link_speed": "8.0 GT/s PCIe", "current_link_width": "16", "max_link_width": "16"})
	down := fixtureFunction(t, root, up, "0000:18:00.0", "pcieport", map[string]string{
		"vendor": "0x1002", "max_link_speed": "16.0 GT/s PCIe", "current_link_speed": "16.0 GT/s PCIe", "current_link_width": "16"})
	attrs := map[string]string{
		"vendor": "0x1002", "device": "0x66a1", "class": "0x038000",
		"mem_info_vram_total": "34342961152", "mem_info_vram_used": "10485760",
		"max_link_speed": "16.0 GT/s PCIe", "current_link_speed": "16.0 GT/s PCIe", "current_link_width": "16",
		"power/runtime_status": "active",
	}
	for k, v := range amdAttrs {
		attrs[k] = v
	}
	fixtureFunction(t, root, down, "0000:19:00.0", amdgpuDriver, attrs)
	// NVIDIA cards, link idle at 2.5 GT/s.
	for _, card := range []struct{ port, address, device, width, slotSpeed string }{
		{"0000:64:00.0", "0000:65:00.0", "0x2203", "16", "8.0 GT/s PCIe"},
		{"0000:b2:00.0", "0000:b3:00.0", "0x2504", "8", "2.5 GT/s PCIe"}, // idle
	} {
		bridge := fixtureFunction(t, root, filepath.Join(devices, "pci0000:"+card.port[5:7]), card.port, "pcieport", map[string]string{
			"vendor": "0x8086", "max_link_speed": "8.0 GT/s PCIe", "current_link_speed": card.slotSpeed, "current_link_width": card.width})
		fixtureFunction(t, root, bridge, card.address, "nvidia", map[string]string{
			"vendor": "0x10de", "device": card.device, "class": "0x030000",
			"max_link_speed": "16.0 GT/s PCIe", "current_link_speed": "2.5 GT/s PCIe", "current_link_width": card.width})
		fixtureFunction(t, root, bridge, card.address[:len(card.address)-1]+"1", "snd_hda_intel", map[string]string{
			"vendor": "0x10de", "device": "0x1aef", "class": "0x040300"})
	}
	// Not GPUs: an NPU (class 0x12, intel_vpu) and an SR-IOV virtual function
	// of a display device.
	fixtureFunction(t, root, pch, "0000:00:0b.0", "intel_vpu", map[string]string{"vendor": "0x8086", "device": "0x7d1d", "class": "0x120000"})
	vf := fixtureFunction(t, root, pch, "0000:04:00.1", "", map[string]string{"vendor": "0x1002", "device": "0x7461", "class": "0x038000"})
	if err := os.Symlink(filepath.Join(pch, "0000:03:00.0"), filepath.Join(vf, "physfn")); err != nil {
		t.Fatal(err)
	}
	// KFD topology: node 0 is the CPU, node 1 the MI50 (location_id 0x1900).
	kfd := filepath.Join(root, "sys", "class", "kfd", "kfd", "topology", "nodes")
	for node, props := range map[string]string{
		"0": "cpu_cores_count 28\nsimd_count 0\ngfx_target_version 0\nlocation_id 0\ndomain 0\n",
		"1": "simd_count 240\ngfx_target_version 90006\nlocation_id 6400\ndomain 0\nunique_id 81985529216486895\n",
	} {
		if err := os.MkdirAll(filepath.Join(kfd, node), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(kfd, node, "properties"), []byte(props), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func nvidiaFixtureCaps() *Capabilities {
	return &Capabilities{GPUs: []GPU{
		{Index: 0, Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24564, PCIBusID: "00000000:65:00.0"},
		{Index: 1, Name: "NVIDIA GeForce RTX 3060", VRAMTotalMB: 12288, PCIBusID: "00000000:B3:00.0"},
	}}
}

func TestReadPCIDisplaysOnTheReferenceHost(t *testing.T) {
	root := fixtureHost(t, "amdgpu", nil)
	got, err := ReadPCIDisplays(root)
	if err != nil {
		t.Fatal(err)
	}
	want := []PCIDisplay{
		{Address: "0000:03:00.0", Vendor: "0x1a03", VendorID: "0x1a03", DeviceID: "0x2000", Class: "0x030000", Name: "PCI display 0x1a03:0x2000", Driver: "ast"},
		{Address: "0000:19:00.0", Vendor: "amd", VendorID: "0x1002", DeviceID: "0x66a1", Class: "0x038000",
			Name: "AMD Radeon Pro VII / Radeon Instinct MI50 (Vega 20)", Driver: "amdgpu",
			VRAMTotalMB: 32752, VRAMUsedMB: 10,
			// The GPU function says Gen 4 x16 (the card's switch); the slot
			// link is read on the root port above the switch's upstream port.
			LinkGen: 3, LinkWidth: 16, LinkAt: "0000:16:00.0",
			KFDUniqueID: "0123456789abcdef", GFXTarget: "gfx906"},
		{Address: "0000:65:00.0", Vendor: "nvidia", VendorID: "0x10de", DeviceID: "0x2203", Class: "0x030000", Name: "NVIDIA GPU 0x2203", Driver: "nvidia",
			LinkGen: 3, LinkWidth: 16, LinkAt: "0000:64:00.0"},
		{Address: "0000:b3:00.0", Vendor: "nvidia", VendorID: "0x10de", DeviceID: "0x2504", Class: "0x030000", Name: "NVIDIA GPU 0x2504", Driver: "nvidia",
			LinkGen: 1, LinkWidth: 8, LinkAt: "0000:b2:00.0"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("inventory:\n got %+v\nwant %+v", got, want)
	}
}

// The device's own unique_id attribute wins over the KFD node's.
func TestReadPCIDisplaysPrefersTheDeviceUniqueID(t *testing.T) {
	root := fixtureHost(t, "amdgpu", map[string]string{"unique_id": "00000000000000ab"})
	got, err := ReadPCIDisplays(root)
	if err != nil {
		t.Fatal(err)
	}
	if got[1].KFDUniqueID != "00000000000000ab" {
		t.Fatalf("unique id = %q", got[1].KFDUniqueID)
	}
}

// A runtime-suspended card is not woken to read its memory.
func TestReadPCIDisplaysLeavesASuspendedCardAsleep(t *testing.T) {
	root := fixtureHost(t, "amdgpu", map[string]string{"power/runtime_status": "suspended"})
	got, err := ReadPCIDisplays(root)
	if err != nil {
		t.Fatal(err)
	}
	if got[1].VRAMTotalMB != 0 || got[1].VRAMUsedMB != 0 || !strings.Contains(got[1].MemoryNote, "suspended") {
		t.Fatalf("suspended card = %+v", got[1])
	}
}

// A compute-only part in class 0x12 counts when a GPU driver drives it.
func TestReadPCIDisplaysKeepsAGPUDrivenAccelerator(t *testing.T) {
	root := fixtureHost(t, "amdgpu", map[string]string{"class": "0x120000"})
	got, err := ReadPCIDisplays(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 4 || got[1].Address != "0000:19:00.0" || got[1].Class != "0x120000" {
		t.Fatalf("inventory = %+v", got)
	}
}

func TestReadPCIDisplaysWithoutSysfs(t *testing.T) {
	if _, err := ReadPCIDisplays(t.TempDir()); err == nil {
		t.Fatal("a root without /sys/bus/pci must be an error, not an empty inventory")
	}
}

func TestOtherVendorBesideNVIDIA(t *testing.T) {
	root := fixtureHost(t, "amdgpu", nil)
	inv, err := ReadPCIDisplays(root)
	if err != nil {
		t.Fatal(err)
	}
	if !OtherVendorBesideNVIDIA(nvidiaFixtureCaps(), inv) {
		t.Fatal("NVIDIA cards beside a bound MI50 must count as a multi-vendor host")
	}
	var nvidiaOnly []PCIDisplay
	for _, d := range inv {
		if d.Vendor != "amd" {
			nvidiaOnly = append(nvidiaOnly, d)
		}
	}
	if OtherVendorBesideNVIDIA(nvidiaFixtureCaps(), nvidiaOnly) {
		t.Fatal("a BMC display beside NVIDIA cards is not another GPU vendor")
	}
	vulkanCaps := &Capabilities{GPUs: []GPU{{Index: 0, Name: "AMD Instinct MI60 / MI50 (RADV VEGA20)", VRAMTotalMB: 32752}}}
	if OtherVendorBesideNVIDIA(vulkanCaps, inv) {
		t.Fatal("GPUs detected without PCI addresses (vulkaninfo) are not an NVIDIA primary group")
	}
	unknown := nvidiaFixtureCaps()
	unknown.GPUStateUnknown = "nvidia-smi did not answer"
	if OtherVendorBesideNVIDIA(unknown, inv) {
		t.Fatal("an unknown GPU state must not trigger anything")
	}
	vfio, err := ReadPCIDisplays(fixtureHost(t, "vfio-pci", nil))
	if err != nil {
		t.Fatal(err)
	}
	if OtherVendorBesideNVIDIA(nvidiaFixtureCaps(), vfio) {
		t.Fatal("a card passed through to a VM cannot be opened by a host backend")
	}
}

func TestUnusableGPUNotes(t *testing.T) {
	inv, err := ReadPCIDisplays(fixtureHost(t, "amdgpu", nil))
	if err != nil {
		t.Fatal(err)
	}
	notes := UnusableGPUNotes(nvidiaFixtureCaps(), inv)
	amd := ""
	for _, d := range inv {
		if d.Vendor == "amd" {
			amd = d.Address
		}
	}
	want := []string{"AMD Radeon Pro VII / Radeon Instinct MI50 (Vega 20) at " + amd + " (driver amdgpu, 32752 MiB, PCIe Gen3 x16) is not used by ggrun: " +
		"launches use the NVIDIA GPUs that nvidia-smi reports unless --gpus selects this AMD GPU: --gpus pci:" + amd + " with a Vulkan or HIP (ROCm) build of llama-server (--server-bin)"}
	if !reflect.DeepEqual(notes, want) {
		t.Fatalf("notes:\n got %q\nwant %q", notes, want)
	}
	unbound, err := ReadPCIDisplays(fixtureHost(t, "", nil))
	if err != nil {
		t.Fatal(err)
	}
	if notes := UnusableGPUNotes(nvidiaFixtureCaps(), unbound); len(notes) != 1 || !strings.HasSuffix(notes[0], "no kernel driver is bound to it") {
		t.Fatalf("unbound card notes = %q", notes)
	}
	vulkanCaps := &Capabilities{GPUs: []GPU{{Index: 0, Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24810}}}
	if notes := UnusableGPUNotes(vulkanCaps, inv); notes != nil {
		t.Fatalf("GPUs without PCI addresses cannot be matched, so nothing may be called unusable: %q", notes)
	}
}

func TestHostPCIDisplaysNeedsTheOptIn(t *testing.T) {
	t.Setenv("GGRUN_EXPERIMENTAL_AMD", "")
	if ExperimentalAMD() || HostPCIDisplays() != nil {
		t.Fatal("without GGRUN_EXPERIMENTAL_AMD=1 nothing is read")
	}
	t.Setenv("GGRUN_EXPERIMENTAL_AMD", "1")
	if ExperimentalAMD() != (runtime.GOOS == "linux") {
		t.Fatal("the opt-in applies on Linux only")
	}
}

func TestCanonicalPCIAddress(t *testing.T) {
	for in, want := range map[string]string{
		"00000000:B3:00.0": "0000:b3:00.0",
		"0000:19:00.0":     "0000:19:00.0",
		"0001:03:1f.7":     "0001:03:1f.7",
	} {
		if got, ok := CanonicalPCIAddress(in); !ok || got != want {
			t.Fatalf("CanonicalPCIAddress(%q) = %q, %v; want %q", in, got, ok, want)
		}
	}
	for _, bad := range []string{"", "pci0000:16", "0000:19:00", "0000:19:20.0", "0000:19:00.8", "zz:19:00.0"} {
		if _, ok := CanonicalPCIAddress(bad); ok {
			t.Fatalf("CanonicalPCIAddress(%q) accepted", bad)
		}
	}
}
