package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/devicelist"
)

// A synthetic multi-vendor host (made-up addresses and ids, round sizes): a
// BMC display, two NVIDIA cards (synthTwoNVIDIA stands in for nvidia-smi's
// answer) and one AMD card with its amdgpu memory files and a KFD node.
const (
	synthKFDID   = "5a17e0c0ffee1234"
	synthROCrID  = "GPU-" + synthKFDID
	synthKFDDec  = "6491904507409994292" // synthKFDID in decimal, as KFD prints it
	synthAMDUsed = 1024                  // MiB
)

func writeSysfs(t *testing.T, dir, name, value string) {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(value+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

// fakeMultiVendorSysfs writes the host above under a fixture root and returns
// it. The AMD card has no unique_id file, so its id comes from the KFD node
// matched by location_id and domain.
func fakeMultiVendorSysfs(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	devices := filepath.Join(root, "sys", "bus", "pci", "devices")
	card := func(address, class, vendor, device, driver string) string {
		t.Helper()
		dir := filepath.Join(devices, address)
		writeSysfs(t, dir, "class", class)
		writeSysfs(t, dir, "vendor", vendor)
		writeSysfs(t, dir, "device", device)
		if err := os.Symlink(filepath.Join(root, "sys", "bus", "pci", "drivers", driver), filepath.Join(dir, "driver")); err != nil {
			t.Fatal(err)
		}
		return dir
	}
	card("0000:01:00.0", "0x030000", "0x1a03", "0x2000", "ast")
	card(synthNV0, "0x030000", "0x10de", "0x2203", "nvidia")
	card(synthNV1, "0x030000", "0x10de", "0x2504", "nvidia")
	amd := card(synthAMD, "0x038000", "0x1002", "0x66a1", "amdgpu")
	writeSysfs(t, amd, "mem_info_vram_total", strconv.FormatInt(synthAMDT*testMiB, 10))
	writeSysfs(t, amd, "mem_info_vram_used", strconv.FormatInt(synthAMDUsed*testMiB, 10))
	writeSysfs(t, amd, filepath.Join("power", "runtime_status"), "active")
	nodes := filepath.Join(root, "sys", "class", "kfd", "kfd", "topology", "nodes")
	writeSysfs(t, filepath.Join(nodes, "0"), "properties", "cpu_cores_count 16\nsimd_count 0\ngfx_target_version 0")
	// location_id 3072 = bus 0x0c, device 0, function 0.
	writeSysfs(t, filepath.Join(nodes, "1"), "properties",
		"cpu_cores_count 0\nsimd_count 240\nlocation_id 3072\ndomain 0\ngfx_target_version 90006\nunique_id "+synthKFDDec)
	return root
}

func sysfsDisplays(t *testing.T, root string, reads *int) func() []detect.PCIDisplay {
	return func() []detect.PCIDisplay {
		if reads != nil {
			*reads++
		}
		inv, err := detect.ReadPCIDisplays(root)
		if err != nil {
			t.Fatal(err)
		}
		return inv
	}
}

// selectorsOn turns on the opt-in and the launch's selector parsing.
func selectorsOn(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("the opt-in is Linux only")
	}
	t.Setenv("GGRUN_EXPERIMENTAL_AMD", "1")
	saved := gpuSelectorsAccepted
	gpuSelectorsAccepted = true
	t.Cleanup(func() { gpuSelectorsAccepted = saved })
}

// Without the opt-in, and in commands other than launch and dry-run, a
// selector is today's parse error, byte for byte.
func TestGPUSelectorsParseOnlyWithTheOptIn(t *testing.T) {
	const today = `--gpus: "amd:0" is not a non-negative GPU index`
	saved := gpuSelectorsAccepted
	t.Cleanup(func() { gpuSelectorsAccepted = saved })
	for _, c := range []struct {
		optIn, accepted bool
	}{{false, false}, {false, true}, {true, false}} {
		t.Setenv("GGRUN_EXPERIMENTAL_AMD", map[bool]string{true: "1", false: ""}[c.optIn])
		gpuSelectorsAccepted = c.accepted
		if _, err := parseLaunchArgs([]string{"model.gguf", "--gpus", "amd:0"}); err == nil || err.Error() != today {
			t.Fatalf("opt-in %v accepted %v: %v, want %q", c.optIn, c.accepted, err, today)
		}
	}
	selectorsOn(t)
	for _, ok := range []string{"amd:0", "pci:0000:0c:00.0", "AMD:1,pci:0000:0C:00.0", "0,1"} {
		if _, err := parseLaunchArgs([]string{"model.gguf", "--gpus", ok}); err != nil {
			t.Fatalf("%s: %v", ok, err)
		}
	}
	for _, bad := range []string{"amd:x", "amd:-1", "pci:zz", "amd:0,"} {
		if _, err := parseLaunchArgs([]string{"model.gguf", "--gpus", bad}); err == nil {
			t.Fatalf("%s parsed", bad)
		}
	}
}

func TestResolveGPUSelectorsAgainstTheInventory(t *testing.T) {
	selectorsOn(t)
	root := fakeMultiVendorSysfs(t)
	reads := 0
	displays := sysfsDisplays(t, root, &reads)

	req := &launchRequest{GPUsFlag: "0,1"}
	if note, err := resolveGPUSelectors(req, synthTwoNVIDIA(), displays); err != nil || note != "" || reads != 0 || req.GPUsFlag != "0,1" || req.vendorSelection != nil {
		t.Fatalf("numeric --gpus: note %q err %v reads %d flag %q", note, err, reads, req.GPUsFlag)
	}

	for _, flag := range []string{"amd:0", "pci:0000:0c:00.0", "PCI:0000:0C:00.0"} {
		req := &launchRequest{GPUsFlag: flag}
		note, err := resolveGPUSelectors(req, synthTwoNVIDIA(), displays)
		if err != nil || note != "" || req.vendorSelection == nil || len(req.vendorSelection.Devices) != 1 {
			t.Fatalf("%s: note %q err %v sel %+v", flag, note, err, req.vendorSelection)
		}
		d := req.vendorSelection.Devices[0].Display
		if d.Address != synthAMD || d.KFDUniqueID != synthKFDID || d.VRAMTotalMB != synthAMDT || d.VRAMUsedMB != synthAMDUsed || req.GPUsFlag != flag {
			t.Fatalf("%s: device %+v flag %q", flag, d, req.GPUsFlag)
		}
	}

	// pci: of NVIDIA cards is their detected index; nothing else changes.
	req = &launchRequest{GPUsFlag: "pci:0000:0b:00.0,0"}
	if note, err := resolveGPUSelectors(req, synthTwoNVIDIA(), displays); err != nil || req.GPUsFlag != "0,1" || req.vendorSelection != nil || !strings.Contains(note, "selects detected GPU(s) 0,1") {
		t.Fatalf("NVIDIA pci: note %q err %v flag %q", note, err, req.GPUsFlag)
	}

	for _, c := range []struct {
		flag string
		req  *launchRequest
		want string
	}{
		{"amd:0,0", nil, "mixes GPU vendors (AMD: amd:0 at 0000:0c:00.0; NVIDIA: 0 at 0000:0a:00.0)"},
		{"pci:0000:0a:00.0,amd:0", nil, "mixes GPU vendors"},
		{"amd:0,pci:0000:0c:00.0", nil, "name the same card"},
		{"amd:1", nil, "amd:1 names no card"},
		{"pci:0000:0d:00.0", nil, "no GPU at PCI 0000:0d:00.0"},
		{"pci:0000:01:00.0", nil, "only NVIDIA and AMD GPUs can be selected"},
		{"amd:0,7", nil, "no detected GPU with index 7"},
		{"amd:0", &launchRequest{GPUsFlag: "amd:0", ClaudeCode: true}, "--claude-reviewer off"},
	} {
		req := c.req
		if req == nil {
			req = &launchRequest{GPUsFlag: c.flag}
		}
		if _, err := resolveGPUSelectors(req, synthTwoNVIDIA(), displays); err == nil || !strings.Contains(err.Error(), c.want) || req.vendorSelection != nil {
			t.Fatalf("%s: %v, want %q", c.flag, err, c.want)
		}
	}
	if _, err := resolveGPUSelectors(&launchRequest{GPUsFlag: "amd:0", ClaudeCode: true, ClaudeReviewerDisabled: true}, synthTwoNVIDIA(), displays); err != nil {
		t.Fatalf("reviewer off: %v", err)
	}

	// A single-vendor host and a host without the opt-in refuse.
	if _, err := resolveGPUSelectors(&launchRequest{GPUsFlag: "pci:0000:0a:00.0"}, synthTwoNVIDIA(), func() []detect.PCIDisplay { return synthInventory(false) }); err == nil || !strings.Contains(err.Error(), "more than one vendor") {
		t.Fatalf("single vendor: %v", err)
	}
	t.Setenv("GGRUN_EXPERIMENTAL_AMD", "")
	if _, err := resolveGPUSelectors(&launchRequest{GPUsFlag: "amd:0"}, synthTwoNVIDIA(), displays); err == nil || !strings.Contains(err.Error(), "GGRUN_EXPERIMENTAL_AMD=1") {
		t.Fatalf("opt-in off: %v", err)
	}
}

// selectedAMD is a resolved amd:0 on the fixture host.
func selectedAMD(t *testing.T, flag string) *launchRequest {
	t.Helper()
	root := fakeMultiVendorSysfs(t)
	req := &launchRequest{GPUsFlag: flag}
	if _, err := resolveGPUSelectors(req, synthTwoNVIDIA(), sysfsDisplays(t, root, nil)); err != nil || req.vendorSelection == nil {
		t.Fatalf("%s: %v", flag, err)
	}
	return req
}

var vkTestMI50 = fakeVulkanDevice{name: "AMD Test MI50 (RADV TEST)", driver: "radv", totalMB: synthAMDT, uuid: "amd", priority: 1}

func vendorVulkanListers(loader fakeVulkanLoader, calls *[]string) vendorListers {
	return vendorListers{
		ListVulkan: func(string) ([]listedDevice, error) { return loader.defaultListing(), nil },
		ListVulkanUnder: func(_, visible string) (string, error) {
			*calls = append(*calls, visible)
			return loader.output(visible, true), nil
		},
		ListUnderEnv: func(string, []string) (string, error) { return loader.output("", false), nil },
		Environ:      func() []string { return nil },
	}
}

// Vulkan: the card is found by total among the devices not named like a
// detected GPU, and exported by its raw index (a CPU renderer sorts first, so
// the default position 1 would name another device); the listing under it
// must show the card.
func TestAMDSelectionOnVulkan(t *testing.T) {
	selectorsOn(t)
	req := selectedAMD(t, "amd:0")
	loader := fakeVulkanLoader{devices: []fakeVulkanDevice{vkCPU, vkNV24, vkTestMI50, vkNV12}, driverLines: true}
	var calls []string
	be := &backendInfo{Path: "/x/llama-server-vulkan", Tag: "vulkan", Dialect: "vulkan"}
	caps, note, err := applyVendorSelection(req, be, synthTwoNVIDIA(), vendorVulkanListers(loader, &calls))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(calls, []string{vkRawProbe, "0,1,2,3", "2"}) {
		t.Fatalf("listings %q", calls)
	}
	sel := req.vendorSelection
	if sel.Dialect != "vulkan" || sel.VisKey != "GGML_VK_VISIBLE_DEVICES" || sel.VisValue != "2" || len(sel.Unset) != 0 || be.Dialect != "vulkan" || req.GPUsFlag != "2" {
		t.Fatalf("selection %+v dialect %q flag %q", sel, be.Dialect, req.GPUsFlag)
	}
	want := detect.GPU{Index: 2, Name: vkTestMI50.name, VRAMTotalMB: synthAMDT, VRAMUsedMB: synthAMDUsed, Driver: "amdgpu", PCIBusID: synthAMD}
	if len(caps.GPUs) != 3 || !reflect.DeepEqual(caps.GPUs[2], want) || !reflect.DeepEqual(caps.GPUs[:2], synthTwoNVIDIA().GPUs) {
		t.Fatalf("caps %+v", caps.GPUs)
	}
	if !strings.Contains(note, "amd:0 = AMD Test MI50 (RADV TEST) at 0000:0c:00.0 (backend Vulkan0, GPU 2); GGML_VK_VISIBLE_DEVICES=2;") {
		t.Fatalf("note %q", note)
	}
	if key, value := gpuVisibilityEnv(req, backendDialect(be)); key != "GGML_VK_VISIBLE_DEVICES" || value != "2" {
		t.Fatalf("visibility %s=%s", key, value)
	}
	// The launch's own Vulkan order step leaves the checked value alone.
	before := devicelist.StartedListings()
	applyVulkanDeviceOrder(req, be, caps)
	if devicelist.StartedListings() != before {
		t.Fatal("the Vulkan order step listed again after an AMD selection")
	}
}

// Two identical AMD cards: one selected is ambiguous and refused; both
// selected pair with the listing in order, and the post-load check decides.
func TestAMDSelectionOnVulkanRefusesAnAmbiguousMatch(t *testing.T) {
	selectorsOn(t)
	twin := func(address, id string) detect.PCIDisplay {
		return detect.PCIDisplay{Address: address, Vendor: "amd", Driver: "amdgpu", Name: "AMD GPU 0x66a1", VRAMTotalMB: synthAMDT, KFDUniqueID: id}
	}
	inv := append(synthInventory(false), twin(synthAMD, "1111"), twin("0000:0d:00.0", "2222"))
	a := fakeVulkanDevice{name: "AMD Test MI50 (RADV TEST)", driver: "radv", totalMB: synthAMDT, uuid: "a", priority: 1}
	b := a
	b.uuid = "b"
	loader := fakeVulkanLoader{devices: []fakeVulkanDevice{vkNV24, a, b, vkNV12}, driverLines: true}
	vulkan := func() *backendInfo { return &backendInfo{Path: "/x/llama-server-vulkan", Dialect: "vulkan"} }

	req := &launchRequest{GPUsFlag: "amd:0"}
	if _, err := resolveGPUSelectors(req, synthTwoNVIDIA(), func() []detect.PCIDisplay { return inv }); err != nil {
		t.Fatal(err)
	}
	var calls []string
	if _, _, err := applyVendorSelection(req, vulkan(), synthTwoNVIDIA(), vendorVulkanListers(loader, &calls)); err == nil ||
		!strings.Contains(err.Error(), "matches 2 Vulkan devices by name and total") || !strings.Contains(err.Error(), "ambiguous") || len(calls) != 0 {
		t.Fatalf("one of two identical cards: %v, listings %q", err, calls)
	}

	req = &launchRequest{GPUsFlag: "amd:1,amd:0"}
	if _, err := resolveGPUSelectors(req, synthTwoNVIDIA(), func() []detect.PCIDisplay { return inv }); err != nil {
		t.Fatal(err)
	}
	if _, _, err := applyVendorSelection(req, vulkan(), synthTwoNVIDIA(), vendorVulkanListers(loader, &calls)); err != nil {
		t.Fatal(err)
	}
	sel := req.vendorSelection
	if sel.VisValue != "1,2" || req.GPUsFlag != "2,3" || sel.Devices[0].Display.Address != synthAMD || sel.Devices[0].Row != "Vulkan0" || sel.Devices[1].Row != "Vulkan1" {
		t.Fatalf("both cards: %+v flag %q", sel, req.GPUsFlag)
	}

	// No device of the card's size apart from the detected ones: refused.
	req = selectedAMD(t, "amd:0")
	if _, _, err := applyVendorSelection(req, vulkan(), synthTwoNVIDIA(), vendorVulkanListers(fakeVulkanLoader{devices: []fakeVulkanDevice{vkNV24, vkNV12}}, &calls)); err == nil ||
		!strings.Contains(err.Error(), "shows no device like") {
		t.Fatalf("absent card: %v", err)
	}
}

// HIP: one listing under ROCR_VISIBLE_DEVICES=GPU-<KFD id> with every other
// device selection removed; the dialect comes from the listing's ROCm rows.
func TestAMDSelectionOnHIP(t *testing.T) {
	selectorsOn(t)
	const hipListing = "ggml_cuda_init: found 1 ROCm devices:\nAvailable devices:\n  ROCm0: AMD Instinct Test (32768 MiB, 32000 MiB free)\n"
	req := selectedAMD(t, "pci:0000:0c:00.0")
	var gotEnv []string
	listers := vendorListers{
		ListUnderEnv: func(path string, env []string) (string, error) {
			if path != "/x/hip/bin/llama-server" {
				t.Fatalf("listed %s", path)
			}
			gotEnv = env
			return hipListing, nil
		},
		Environ: func() []string {
			return []string{"PATH=/bin", "CUDA_VISIBLE_DEVICES=0", "HIP_VISIBLE_DEVICES=1", "GPU_DEVICE_ORDINAL=0", "ROCR_VISIBLE_DEVICES=7", "HOME=/h"}
		},
	}
	be := &backendInfo{Path: "/x/hip/bin/llama-server", Tag: "llama", Dialect: "llama"}
	caps, note, err := applyVendorSelection(req, be, synthTwoNVIDIA(), listers)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"PATH=/bin", "HOME=/h", "ROCR_VISIBLE_DEVICES=" + synthROCrID}; !reflect.DeepEqual(gotEnv, want) {
		t.Fatalf("listing env %q, want %q", gotEnv, want)
	}
	sel := req.vendorSelection
	if sel.Dialect != "rocm" || be.Dialect != "rocm" || be.Tag != "llama" || sel.Devices[0].Row != "ROCm0" || req.GPUsFlag != "2" {
		t.Fatalf("selection %+v backend %+v flag %q", sel, be, req.GPUsFlag)
	}
	if caps.GPUs[2].Name != "AMD Instinct Test" || caps.GPUs[2].PCIBusID != synthAMD || caps.GPUs[2].Index != 2 {
		t.Fatalf("caps %+v", caps.GPUs)
	}
	if !strings.Contains(note, "backend ROCm0, GPU 2); ROCR_VISIBLE_DEVICES="+synthROCrID+", CUDA_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES, GPU_DEVICE_ORDINAL unset;") {
		t.Fatalf("note %q", note)
	}

	// The launch's environment, exactly: the ROCr list set, the index
	// variables HIP would also read unset, nothing else touched.
	for _, kv := range []string{"CUDA_VISIBLE_DEVICES=0", "HIP_VISIBLE_DEVICES=1", "GPU_DEVICE_ORDINAL=0", "ROCR_VISIBLE_DEVICES=7"} {
		key, value, _ := strings.Cut(kv, "=")
		t.Setenv(key, value)
	}
	before := sortedEnviron()
	shown := applyGPUVisibility(req, backendDialect(be))
	if shown != "ROCR_VISIBLE_DEVICES="+synthROCrID+" (unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL)" {
		t.Fatalf("shown %q", shown)
	}
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"} {
		if _, set := os.LookupEnv(key); set {
			t.Fatalf("%s is still set", key)
		}
	}
	after := sortedEnviron()
	var removed, added []string
	for _, kv := range before {
		if !containsString(after, kv) {
			removed = append(removed, kv)
		}
	}
	for _, kv := range after {
		if !containsString(before, kv) {
			added = append(added, kv)
		}
	}
	if !reflect.DeepEqual(added, []string{"ROCR_VISIBLE_DEVICES=" + synthROCrID}) ||
		!reflect.DeepEqual(removed, []string{"CUDA_VISIBLE_DEVICES=0", "GPU_DEVICE_ORDINAL=0", "HIP_VISIBLE_DEVICES=1", "ROCR_VISIBLE_DEVICES=7"}) {
		t.Fatalf("environment: added %q removed %q", added, removed)
	}
}

func containsString(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// A CUDA build, a wrong device count, a card without a KFD id and an ik build
// are refused; a Vulkan build whose path does not say so is found by its
// listing.
func TestAMDSelectionRefusesWhatCannotDriveTheCard(t *testing.T) {
	selectorsOn(t)
	listing := func(out string) vendorListers {
		return vendorListers{ListUnderEnv: func(string, []string) (string, error) { return out, nil }, Environ: func() []string { return nil }}
	}
	llama := func() *backendInfo { return &backendInfo{Path: "/x/llama-server", Tag: "llama", Dialect: "llama"} }
	for _, c := range []struct {
		name    string
		be      *backendInfo
		listers vendorListers
		want    string
	}{
		{"CUDA build", llama(), listing("Available devices:\n  CUDA0: NVIDIA Test GPU 24G (24576 MiB, 20000 MiB free)\n"), "lists no ROCm or Vulkan device"},
		{"two ROCm rows", llama(), listing("Available devices:\n  ROCm0: AMD Instinct Test (32768 MiB, 32000 MiB free)\n  ROCm1: AMD Instinct Test (32768 MiB, 32000 MiB free)\n"), "lists 2 ROCm devices"},
		{"another card", llama(), listing("Available devices:\n  ROCm0: AMD Small Test (8192 MiB, 8000 MiB free)\n"), "ROCm0 is AMD Small Test (8192 MiB)"},
		{"ik build", &backendInfo{Path: "/x/ik/llama-server", Tag: "ik_llama", Dialect: "ik_llama", IsIK: true}, listing(""), "mainline llama.cpp Vulkan or HIP"},
	} {
		req := selectedAMD(t, "amd:0")
		if _, _, err := applyVendorSelection(req, c.be, synthTwoNVIDIA(), c.listers); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Fatalf("%s: %v, want %q", c.name, err, c.want)
		}
		if req.GPUsFlag != "amd:0" || c.be.Dialect == "rocm" {
			t.Fatalf("%s: a refusal changed the request or the backend: %q %q", c.name, req.GPUsFlag, c.be.Dialect)
		}
	}

	req := selectedAMD(t, "amd:0")
	req.vendorSelection.Devices[0].Display.KFDUniqueID = ""
	if _, _, err := applyVendorSelection(req, llama(), synthTwoNVIDIA(), listing("")); err == nil || !strings.Contains(err.Error(), "no KFD unique id") {
		t.Fatalf("no KFD id: %v", err)
	}

	req = selectedAMD(t, "amd:0")
	loader := fakeVulkanLoader{devices: []fakeVulkanDevice{vkNV24, vkTestMI50, vkNV12}, driverLines: true}
	var calls []string
	be := llama()
	if _, _, err := applyVendorSelection(req, be, synthTwoNVIDIA(), vendorVulkanListers(loader, &calls)); err != nil || be.Dialect != "vulkan" || req.vendorSelection.VisValue != "1" {
		t.Fatalf("unnamed Vulkan build: %v dialect %q value %q", err, be.Dialect, req.vendorSelection.VisValue)
	}

	// No AMD selection: nothing is listed and caps come back as they were.
	caps := synthTwoNVIDIA()
	got, note, err := applyVendorSelection(&launchRequest{GPUsFlag: "1"}, llama(), caps, vendorListers{})
	if err != nil || note != "" || got != caps {
		t.Fatalf("no selection: %v %q", err, note)
	}
}

func TestROCmDeviceID(t *testing.T) {
	for in, want := range map[string]string{synthKFDID: synthROCrID, "0x00ab": "GPU-00000000000000ab", "AB": "GPU-00000000000000ab"} {
		if got, ok := rocmDeviceID(in); !ok || got != want {
			t.Fatalf("%q -> %q %v, want %q", in, got, ok, want)
		}
	}
	for _, bad := range []string{"", "0", "0000000000000000", "xyz"} {
		if got, ok := rocmDeviceID(bad); ok {
			t.Fatalf("%q -> %q", bad, got)
		}
	}
}

// After load, the backend's own "using device" lines must name the planned
// cards at their PCI addresses, or the launch stops.
func TestPostLoadDeviceCheck(t *testing.T) {
	rocm := &vendorSelection{Devices: []vendorDevice{{Row: "ROCm0", Name: "AMD Instinct Test", Display: detect.PCIDisplay{Address: synthAMD}}}}
	vulkan := &vendorSelection{Devices: []vendorDevice{{Row: "Vulkan0", Name: "AMD Test MI50 (RADV TEST)", Display: detect.PCIDisplay{Address: synthAMD}}}}
	line := func(row, desc, id string) string {
		return "0.00.090.001 I llama_prepare_model_devices: using device " + row + " (" + desc + ") (" + id + ") - 32000 MiB free\n"
	}
	for _, c := range []struct {
		name string
		sel  *vendorSelection
		log  string
		want string // "" = passes
	}{
		{"ROCm card", rocm, "load_backend: loaded ROCm backend\n" + line("ROCm0", "AMD Instinct Test", "0000:0c:00.0"), ""},
		{"nested parentheses", vulkan, line("Vulkan0", "AMD Test MI50 (RADV TEST)", "0000:0c:00.0"), ""},
		{"CUDA-style id", rocm, line("ROCm0", "AMD Instinct Test", "00000000:0C:00.0"), ""},
		{"other card", rocm, line("ROCm0", "NVIDIA Test GPU 24G", "0000:0a:00.0"), "backend device ROCm0 is 0000:0a:00.0, the plan expected 0000:0c:00.0"},
		{"no id", rocm, line("ROCm0", "AMD Instinct Test", "unknown id"), "is unknown id"},
		{"no line", rocm, "llama_model_loader: loaded meta data\n", `no "using device" line for ROCm0`},
		{"skipping line only", vulkan,
			"llama_prepare_model_devices: skipping device Vulkan0 (X) with id 0000:0c:00.0 - already using device Vulkan1 (X) with the same id\n",
			`no "using device" line for Vulkan0`},
		{"extra device", rocm, line("ROCm0", "AMD Instinct Test", "0000:0c:00.0") + line("ROCm1", "AMD Instinct Test", "0000:0d:00.0"), "also uses ROCm1 (0000:0d:00.0)"},
	} {
		err := checkUsingDevices(c.sel, c.log)
		if (c.want == "") != (err == nil) || (err != nil && !strings.Contains(err.Error(), c.want)) {
			t.Fatalf("%s: %v, want %q", c.name, err, c.want)
		}
		var mismatch *deviceOrderMismatchError
		if err != nil && !errors.As(err, &mismatch) {
			t.Fatalf("%s: %T is not a device order stop", c.name, err)
		}
	}
	if err := postLoadDeviceCheck(&launchRequest{GPUsFlag: "0"}, nil); err != nil {
		t.Fatalf("no AMD selection: %v", err)
	}
	err := postLoadDeviceCheck(&launchRequest{vendorSelection: rocm}, nil)
	var mismatch *deviceOrderMismatchError
	if err == nil || !errors.As(err, &mismatch) || verifiedConfigStaleAfter(err) {
		t.Fatalf("a stop must keep the verified config and be a device order stop: %v", err)
	}
}

// ROCm rows map through the admission table like Vulkan rows: the listing and
// sysfs readings, the smaller one charged.
func TestAdmissionRowsMapROCmRows(t *testing.T) {
	inv := synthInventory(true)
	gatedHostFixture(t, inv, true)
	fakeAdmissionListing(t, "Available devices:\n  ROCm0: AMD Instinct Test (32768 MiB, 30000 MiB free)\n", nil)
	fakeAMDGPUSysfs(t, synthAMD, synthAMDT*testMiB, 4096*testMiB, "active")
	rocmBe := &backendInfo{Path: "/x/hip/llama-server", Tag: "llama", Dialect: "rocm"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "AMD Instinct Test", VRAMTotalMB: synthAMDT, PCIBusID: synthAMD}}}
	if !nonCUDAAdmissionApplies(rocmBe, caps, func() []detect.PCIDisplay { return inv }) {
		t.Fatal("a ROCm launch on the AMD card beside NVIDIA cards must take the mapped rows")
	}
	if nonCUDAAdmissionApplies(rocmBe, synthTwoNVIDIA(), func() []detect.PCIDisplay { return inv }) {
		t.Fatal("ROCm with no AMD card in caps must not")
	}
	if nonCUDAAdmissionApplies(rocmBe, caps, func() []detect.PCIDisplay { return synthInventory(false)[:1] }) {
		t.Fatal("a single-vendor host must not")
	}
	if got := admissionRowPrefix(rocmBe); got != "ROCm" {
		t.Fatalf("prefix %q", got)
	}
	rows, err := buildAdmissionRows(rocmBe, caps, inv, false)
	if err != nil {
		t.Fatal(err)
	}
	d, ok := rows.device("ROCm0")
	if !ok || !d.FreeOK || d.FreeMB != 28672 || d.Vendor != "amd" || !reflect.DeepEqual(d.Sources, []string{"listing=30000", "sysfs=28672"}) {
		t.Fatalf("ROCm row %+v", d)
	}
	if err := rows.checkRows([]preflightDevice{{Name: "ROCm0", ModelMB: 16000}, {Name: "Host", ModelMB: 600}}); err != nil {
		t.Fatal(err)
	}
	if err := rows.checkRows([]preflightDevice{{Name: "ROCm0", ModelMB: 16000}, {Name: "CUDA0", ModelMB: 10}}); err == nil {
		t.Fatal("a CUDA row on a ROCm launch must refuse")
	}
}

func TestROCmNamesInPreflightAndGPUCheck(t *testing.T) {
	if got := draftDeviceForPreflight("rocm", 1); got != "ROCm1" {
		t.Fatalf("draft device %q", got)
	}
	for _, tag := range []string{"", "llama", "ik_llama"} {
		if got := draftDeviceForPreflight(tag, 1); got != "CUDA1" {
			t.Fatalf("tag %q: %q", tag, got)
		}
	}
	out := "Available devices:\n  ROCm0: AMD Instinct Test (32768 MiB, 32000 MiB free)\n"
	if capable, probed := backendGPUListing(out, "available devices:"); capable || !probed {
		t.Fatalf("today's check: capable %v probed %v", capable, probed)
	}
	if capable, probed := backendGPUListing(out, "available devices:", "rocm"); !capable || !probed {
		t.Fatalf("ROCm dialect: capable %v probed %v", capable, probed)
	}
}

// Through planLaunch (the dry run): an AMD selection on HIP and on Vulkan
// gives the same argv but for the device names, and exactly the environment
// of its dialect; a pci: selection of an NVIDIA card is the numeric plan;
// mixed vendors and --inventory are refused.
func TestPlanLaunchAMDSelection(t *testing.T) {
	launchOrderTestEnv(t)
	c := launchOrderCase{goldenPlanCase: goldenPlanCase{name: "mixed/dense-small/amd-select", hw: "mixed3", model: "dense-small"}}
	launchOrderRunCase(t, c, func(t *testing.T) {
		selectorsOn(t)
		goldenPlanUnsetenv(t, "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
		root := fakeMultiVendorSysfs(t)
		const name = "AMD Instinct Test"
		loader := fakeVulkanLoader{devices: []fakeVulkanDevice{vkCPU, vkNV24,
			{name: name, driver: "radv", totalMB: synthAMDT, uuid: "amd", priority: 1}, vkNV12}, driverLines: true}
		hip := "Available devices:\n  ROCm0: " + name + " (32768 MiB, 32000 MiB free)\n"
		type outcome struct {
			plan     *launchPlan
			err      *planError
			json     string
			hipEnv   []string
			listings int
		}
		run := func(gpus, dialect string, keepInventory bool) outcome {
			t.Helper()
			var o outcome
			req, model := planLaunchFixture(t, c.hw, c.model, "--gpus", gpus)
			// The inventory's host (RAM, CPU) with the synthetic NVIDIA cards
			// nvidia-smi would report on the fixture host.
			host, err := planLaunchSilentInventory(req.InventoryPath)
			if err != nil {
				t.Fatal(err)
			}
			host.GPUs = synthTwoNVIDIA().GPUs
			if !keepInventory {
				req.InventoryPath = ""
			}
			be := goldenPlanBackend(t, "")
			be.Dialect = dialect
			in := goldenPlanInputs(t, c.name, model, be, nil)
			in.Detect = func() (*detect.Capabilities, error) { return host, nil }
			in.HostDisplays = sysfsDisplays(t, root, nil)
			in.ListVulkan = func(string) ([]listedDevice, error) { o.listings++; return loader.defaultListing(), nil }
			in.ListVulkanUnder = func(_, visible string) (string, error) { o.listings++; return loader.output(visible, true), nil }
			in.ListUnderEnv = func(_ string, env []string) (string, error) { o.listings++; o.hipEnv = env; return hip, nil }
			planLaunchCollect(&in)
			o.plan, o.err = planLaunch(req, in)
			if o.err == nil && o.plan != nil && o.plan.Refusal == nil {
				data, err := json.Marshal(goldenPlanJSONOf(o.plan))
				if err != nil {
					t.Fatal(err)
				}
				o.json = string(data)
			}
			return o
		}
		argvOf := func(o outcome) string {
			t.Helper()
			if o.err != nil || o.plan == nil || o.plan.Refusal != nil {
				t.Fatalf("plan failed: %+v", o.err)
			}
			return strings.Join(o.plan.ServerArgv, " ")
		}

		h := run("amd:0", "llama", false)
		hipArgv := argvOf(h)
		if !strings.Contains(hipArgv, " --device ROCm0 ") || strings.Contains(hipArgv, "CUDA") || strings.Contains(hipArgv, "Vulkan") {
			t.Fatalf("HIP argv %s", hipArgv)
		}
		if h.plan.EnvPrefix != "ROCR_VISIBLE_DEVICES="+synthROCrID || !reflect.DeepEqual(h.plan.EnvUnset, rocmHiddenVisibility) {
			t.Fatalf("HIP env prefix %q unset %q", h.plan.EnvPrefix, h.plan.EnvUnset)
		}
		if want := map[string]string{"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "ROCR_VISIBLE_DEVICES": synthROCrID}; !reflect.DeepEqual(h.plan.Environment, want) {
			t.Fatalf("HIP environment %v, want %v", h.plan.Environment, want)
		}
		if h.listings != 1 || !containsString(h.hipEnv, "ROCR_VISIBLE_DEVICES="+synthROCrID) || h.plan.Backend.Dialect != "rocm" {
			t.Fatalf("HIP listings %d env %q dialect %q", h.listings, h.hipEnv, h.plan.Backend.Dialect)
		}

		v := run("pci:0000:0c:00.0", "vulkan", false)
		vkArgv := argvOf(v)
		if !strings.Contains(vkArgv, " --device Vulkan0 ") || strings.Contains(vkArgv, "CUDA") || strings.Contains(vkArgv, "ROCm") {
			t.Fatalf("Vulkan argv %s", vkArgv)
		}
		if v.plan.EnvPrefix != "GGML_VK_VISIBLE_DEVICES=2" || len(v.plan.EnvUnset) != 0 {
			t.Fatalf("Vulkan env prefix %q unset %q", v.plan.EnvPrefix, v.plan.EnvUnset)
		}
		if want := map[string]string{"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "GGML_VK_VISIBLE_DEVICES": "2"}; !reflect.DeepEqual(v.plan.Environment, want) {
			t.Fatalf("Vulkan environment %v, want %v", v.plan.Environment, want)
		}
		if strings.ReplaceAll(vkArgv, "Vulkan0", "ROCm0") != hipArgv {
			t.Fatalf("the dialects plan different argv:\nVulkan %s\nHIP    %s", vkArgv, hipArgv)
		}

		// pci: of an NVIDIA card is the numeric plan, byte for byte.
		numeric, byPCI := run("1", "llama", false), run("pci:0000:0b:00.0", "llama", false)
		if numeric.json == "" || numeric.json != byPCI.json || byPCI.listings != 0 || byPCI.plan.EnvPrefix != "CUDA_VISIBLE_DEVICES=1" {
			t.Fatalf("NVIDIA pci: plan differs:\n%s\n%s", numeric.json, byPCI.json)
		}

		if o := run("amd:0,0", "llama", false); o.err == nil || o.err.Code != 1 || !strings.Contains(o.err.Message, "mixes GPU vendors") || o.listings != 0 {
			t.Fatalf("mixed vendors: %+v", o.err)
		}
		if o := run("amd:0", "llama", true); o.err == nil || o.err.Code != 2 || !strings.Contains(o.err.Message, "cannot be used with --inventory") {
			t.Fatalf("--inventory: %+v", o.err)
		}
	})
}
