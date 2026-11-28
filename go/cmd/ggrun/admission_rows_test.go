package main

import (
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
	"github.com/raketenkater/ggrun/pkg/placement"
)

// Synthetic multi-vendor host: two NVIDIA cards and one AMD card in the OS
// inventory (multiVendorInventory). Values are made up; no real host.
func admissionNVIDIA24() detect.GPU {
	return detect.GPU{Index: 0, Name: "NVIDIA Test GPU 24G", VRAMTotalMB: 24564, VRAMUsedMB: 3000, PCIBusID: "00000000:65:00.0"}
}

func admissionAMD32(index int) detect.GPU {
	return detect.GPU{Index: index, Name: "AMD Test GPU 32G (RADV)", VRAMTotalMB: 32752, PCIBusID: "0000:19:00.0"}
}

func vulkanTestBackend() *backendInfo {
	return &backendInfo{Path: "/x/llama-server-vulkan", Tag: "vulkan", Dialect: "vulkan"}
}

// gatedHostFixture opts in and serves inventory to the launch; reads counts
// the inventory reads.
func gatedHostFixture(t *testing.T, inventory []detect.PCIDisplay, optIn bool) *int {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("the opt-in is Linux only")
	}
	reads := 0
	saved := hostPCIDisplays
	hostPCIDisplays = func() []detect.PCIDisplay { reads++; return inventory }
	t.Cleanup(func() { hostPCIDisplays = saved })
	value := ""
	if optIn {
		value = "1"
	}
	t.Setenv("GGRUN_EXPERIMENTAL_AMD", value)
	return &reads
}

// fakeAMDGPUSysfs writes amdgpu's memory files for pci under a fixture root.
func fakeAMDGPUSysfs(t *testing.T, pci string, totalBytes, usedBytes int64, status string) {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, "sys", "bus", "pci", "devices", pci)
	if err := os.MkdirAll(filepath.Join(dir, "power"), 0o755); err != nil {
		t.Fatal(err)
	}
	for name, value := range map[string]string{
		"mem_info_vram_total":  strconv.FormatInt(totalBytes, 10),
		"mem_info_vram_used":   strconv.FormatInt(usedBytes, 10),
		"power/runtime_status": status,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(value+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	saved := amdgpuSysfsRoot
	amdgpuSysfsRoot = root
	t.Cleanup(func() { amdgpuSysfsRoot = saved })
}

func fakeAdmissionListing(t *testing.T, out string, err error) *int {
	t.Helper()
	calls := 0
	saved := admissionListing
	admissionListing = func(string) (string, error) { calls++; return out, err }
	t.Cleanup(func() { admissionListing = saved })
	return &calls
}

const testMiB = int64(1) << 20

// The gate: opt-in, Linux, a non-CUDA dialect, an inventory with GPUs of two
// vendors. The inventory is read only after the opt-in and the dialect.
func TestNonCUDAAdmissionGate(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the opt-in is Linux only")
	}
	nvidiaOnly := multiVendorInventory(false)
	mixed := multiVendorInventory(true)
	vfio := append(multiVendorInventory(false), detect.PCIDisplay{Address: "0000:19:00.0", Vendor: "amd", Driver: "vfio-pci"})
	amdOnly := []detect.PCIDisplay{{Address: "0000:03:00.0", Vendor: "0x1a03", Driver: "ast"}, {Address: "0000:19:00.0", Vendor: "amd", Driver: "amdgpu"}}
	workaround := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "AMD Test GPU 32G (RADV)", VRAMTotalMB: 32752}}}
	cuda := &backendInfo{Path: "/x/llama-server", Tag: "llama", Dialect: "llama"}
	for _, c := range []struct {
		name  string
		optIn bool
		be    *backendInfo
		caps  *detect.Capabilities
		inv   []detect.PCIDisplay
		want  bool
		reads int
	}{
		{"opt-in off", false, vulkanTestBackend(), twoNVIDIACaps(), mixed, false, 0},
		{"NVIDIA-only host, Vulkan", true, vulkanTestBackend(), twoNVIDIACaps(), nvidiaOnly, false, 1},
		{"CUDA backend beside AMD", true, cuda, twoNVIDIACaps(), mixed, false, 0},
		{"Vulkan on NVIDIA beside AMD", true, vulkanTestBackend(), twoNVIDIACaps(), mixed, true, 1},
		{"vulkaninfo-only AMD card beside NVIDIA", true, vulkanTestBackend(), workaround, mixed, true, 1},
		{"other vendor bound to vfio-pci", true, vulkanTestBackend(), twoNVIDIACaps(), vfio, false, 1},
		{"AMD-only host with a BMC display", true, vulkanTestBackend(), workaround, amdOnly, false, 1},
		{"no GPU in caps", true, vulkanTestBackend(), &detect.Capabilities{}, mixed, false, 0},
	} {
		value := ""
		if c.optIn {
			value = "1"
		}
		t.Setenv("GGRUN_EXPERIMENTAL_AMD", value)
		reads := 0
		before := devicelist.StartedListings()
		got := nonCUDAAdmissionApplies(c.be, c.caps, func() []detect.PCIDisplay { reads++; return c.inv })
		if got != c.want || reads != c.reads || devicelist.StartedListings() != before {
			t.Errorf("%s: applies=%v reads=%d listings=%d, want %v reads=%d listings=0",
				c.name, got, reads, devicelist.StartedListings()-before, c.want, c.reads)
		}
	}
}

// Each row maps to its view GPU; free memory is the minimum of the ok
// readers. The listing reports the AMD card's whole heap as free while sysfs
// shows another process holding most of it: the listing figure is not used.
func TestAdmissionRowsMapRowsAndReadEachVendor(t *testing.T) {
	fakeAdmissionListing(t, "Available devices:\n"+
		"  Vulkan0: NVIDIA Test GPU 24G (24810 MiB, 20000 MiB free)\n"+
		"  Vulkan1: AMD Test GPU 32G (RADV) (32752 MiB, 32752 MiB free)\n", nil)
	// One byte past 26,314 MiB used: free rounds down to 6,437.
	fakeAMDGPUSysfs(t, "0000:19:00.0", 32752*testMiB, 26314*testMiB+1, "active")
	caps := &detect.Capabilities{GPUs: []detect.GPU{admissionNVIDIA24(), admissionAMD32(1)}}
	rows, err := buildAdmissionRows(vulkanTestBackend(), caps, multiVendorInventory(true), false)
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := rows.index("Vulkan1"); !ok || v != 1 {
		t.Fatalf("Vulkan1 -> %d %v, want GPU 1", v, ok)
	}
	for _, name := range []string{"Vulkan2", "CUDA0", "Host"} {
		if _, ok := rows.index(name); ok {
			t.Fatalf("%s resolved to a device", name)
		}
	}
	nv, _ := rows.device("Vulkan0")
	if !nv.FreeOK || nv.FreeMB != 20000 || nv.Vendor != "nvidia" || !reflect.DeepEqual(nv.Sources, []string{"nvml=21564", "listing=20000"}) {
		t.Fatalf("NVIDIA row: %+v", nv)
	}
	amd, _ := rows.device("Vulkan1")
	if !amd.FreeOK || amd.FreeMB != 6437 || amd.PCI != "0000:19:00.0" || !reflect.DeepEqual(amd.Sources, []string{"sysfs=6437"}) {
		t.Fatalf("AMD row: %+v", amd)
	}
	if !strings.Contains(strings.Join(amd.Reasons, ";"), "whole heap as free") {
		t.Fatalf("the listing's total-as-free figure was not set aside: %q", amd.Reasons)
	}
	if got := rows.freeMB(caps.GPUs[1]); got != 6437 {
		t.Fatalf("admission free for GPU 1 = %d", got)
	}
	if !strings.Contains(rows.summary(), "Vulkan1 = GPU 1 (AMD Test GPU 32G (RADV), 0000:19:00.0): free 6437 MiB") {
		t.Fatalf("summary %q", rows.summary())
	}
	over := rows.overheadByGPU(t.TempDir(), caps.GPUs)
	if over[0] != placement.UnmeasuredCUDAOverheadMB || over[1] != placement.UnmeasuredCUDAOverheadMB {
		t.Fatalf("gated overhead %v", over)
	}
}

// A listing that does not show exactly the plan's GPUs in view order leaves
// the order unverified, and the attempt is refused.
func TestAdmissionRowsRefuseAnUnverifiedOrder(t *testing.T) {
	two := &detect.Capabilities{GPUs: []detect.GPU{admissionNVIDIA24(), admissionAMD32(1)}}
	nvLine := "  Vulkan0: NVIDIA Test GPU 24G (24810 MiB, 20000 MiB free)\n"
	amdLine := "  Vulkan1: AMD Test GPU 32G (RADV) (32752 MiB, 30000 MiB free)\n"
	for _, c := range []struct {
		name string
		caps *detect.Capabilities
		out  string
		err  error
	}{
		{"unrestricted: an extra device", two, nvLine + amdLine + "  Vulkan2: NVIDIA Test GPU 12G (12288 MiB, 12000 MiB free)\n", nil},
		{"swapped order", two, "  Vulkan0: AMD Test GPU 32G (RADV) (32752 MiB, 30000 MiB free)\n  Vulkan1: NVIDIA Test GPU 24G (24810 MiB, 20000 MiB free)\n", nil},
		{"another card of that name", two, nvLine + "  Vulkan1: AMD Test GPU 32G (RADV) (16368 MiB, 16000 MiB free)\n", nil},
		{"listing failed", two, "", errors.New("exit status 1")},
		{"GPU indexes not 0..n-1", &detect.Capabilities{GPUs: []detect.GPU{admissionAMD32(1)}}, amdLine, nil},
	} {
		fakeAdmissionListing(t, c.out, c.err)
		if _, err := buildAdmissionRows(vulkanTestBackend(), c.caps, multiVendorInventory(true), false); err == nil || !strings.Contains(err.Error(), "could not be verified") {
			t.Errorf("%s: err %v", c.name, err)
		}
	}
}

// Unknown stays unknown: no ok reader means the device is refused, and the
// refusal names the readers.
func TestAdmissionRowsUnknownMemoryFailsClosed(t *testing.T) {
	heapAsFree := "  Vulkan0: AMD Test GPU 32G (RADV) (32752 MiB, 32752 MiB free)\n"
	oracle := []preflightDevice{{Name: "Vulkan0", ModelMB: 16773, ContextMB: 1241, ComputeMB: 368}, {Name: "Host", ModelMB: 682}}
	one := &detect.Capabilities{GPUs: []detect.GPU{admissionAMD32(0)}}

	fakeAdmissionListing(t, heapAsFree, nil)
	fakeAMDGPUSysfs(t, "0000:19:00.0", 32752*testMiB, 10*testMiB, "suspended")
	rows, err := buildAdmissionRows(vulkanTestBackend(), one, multiVendorInventory(true), false)
	if err != nil {
		t.Fatal(err)
	}
	err = rows.checkMemory(oracle)
	if err == nil || !strings.Contains(err.Error(), "runtime-suspended") || !strings.Contains(err.Error(), "whole heap as free") {
		t.Fatalf("suspended card with no budget: %v", err)
	}

	// The lab path: detected through vulkaninfo, no PCI address.
	noPCI := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "AMD Test GPU 32G (RADV)", VRAMTotalMB: 32752}}}
	if rows, err = buildAdmissionRows(vulkanTestBackend(), noPCI, multiVendorInventory(true), false); err != nil {
		t.Fatal(err)
	}
	if err := rows.checkMemory(oracle); err == nil || !strings.Contains(err.Error(), "no PCI address") {
		t.Fatalf("no PCI and no budget: %v", err)
	}
	fakeAdmissionListing(t, "  Vulkan0: AMD Test GPU 32G (RADV) (32752 MiB, 6696 MiB free)\n", nil)
	if rows, err = buildAdmissionRows(vulkanTestBackend(), noPCI, multiVendorInventory(true), false); err != nil {
		t.Fatal(err)
	}
	if err := rows.checkMemory(oracle); err != nil || rows.freeMB(noPCI.GPUs[0]) != 6696 {
		t.Fatalf("a budget-backed listing is a reading: err %v free %d", err, rows.freeMB(noPCI.GPUs[0]))
	}

	// While the baseline serves, live readings are not used: an NVIDIA card
	// keeps its detection figure, the AMD card is unknown.
	fakeAdmissionListing(t, "  Vulkan0: NVIDIA Test GPU 24G (24810 MiB, 1000 MiB free)\n  Vulkan1: AMD Test GPU 32G (RADV) (32752 MiB, 1000 MiB free)\n", nil)
	fakeAMDGPUSysfs(t, "0000:19:00.0", 32752*testMiB, 10*testMiB, "active")
	two := &detect.Capabilities{GPUs: []detect.GPU{admissionNVIDIA24(), admissionAMD32(1)}}
	if rows, err = buildAdmissionRows(vulkanTestBackend(), two, multiVendorInventory(true), true); err != nil {
		t.Fatal(err)
	}
	if nv, _ := rows.device("Vulkan0"); !nv.FreeOK || nv.FreeMB != 21564 {
		t.Fatalf("NVIDIA row while the baseline serves: %+v", nv)
	}
	if amd, _ := rows.device("Vulkan1"); amd.FreeOK {
		t.Fatalf("AMD row while the baseline serves: %+v", amd)
	}
}

// Two cards the listing cannot tell apart are each charged against the
// smaller free figure until a load proves which is which.
func TestAdmissionRowsIdenticalCardsChargeTheSmallerFree(t *testing.T) {
	fakeAdmissionListing(t, "  Vulkan0: NVIDIA Test GPU 24G (24810 MiB, 21000 MiB free)\n  Vulkan1: NVIDIA Test GPU 24G (24810 MiB, 9000 MiB free)\n", nil)
	a, b := admissionNVIDIA24(), admissionNVIDIA24()
	b.Index, b.PCIBusID, b.VRAMUsedMB = 1, "00000000:B3:00.0", 15000
	rows, err := buildAdmissionRows(vulkanTestBackend(), &detect.Capabilities{GPUs: []detect.GPU{a, b}}, multiVendorInventory(true), false)
	if err != nil {
		t.Fatal(err)
	}
	if rows.freeMB(a) != 9000 || rows.freeMB(b) != 9000 {
		t.Fatalf("identical cards: free %d and %d, want 9000 each", rows.freeMB(a), rows.freeMB(b))
	}
}

// A row the map does not contain refuses the attempt unless it is empty.
func TestAdmissionRowsRefuseRowsOutsideTheMap(t *testing.T) {
	rows := &admissionRows{prefix: "Vulkan", devices: []admissionDevice{{Row: "Vulkan0", View: 0, FreeOK: true, FreeMB: 30000}}}
	host := preflightDevice{Name: "Host", ModelMB: 682}
	for _, foreign := range []preflightDevice{
		{Name: "SYCL0", ModelMB: 900, ContextMB: 10, ComputeMB: 10},
		{Name: "Vulkan3", ModelMB: 1},
		{Name: "CUDA0", ComputeMB: 100},
		{Name: "garbage", ContextMB: 5},
	} {
		err := rows.checkRows([]preflightDevice{{Name: "Vulkan0", ModelMB: 1}, foreign, host})
		if err == nil || !strings.Contains(err.Error(), "device map does not contain") {
			t.Errorf("%s: err %v", foreign.Name, err)
		}
	}
	if err := rows.checkRows([]preflightDevice{{Name: "Vulkan0", ModelMB: 1}, {Name: "Vulkan3"}, host}); err != nil {
		t.Fatalf("an empty row of an idle device: %v", err)
	}
	var cuda *admissionRows
	if err := cuda.checkRows([]preflightDevice{{Name: "SYCL0", ModelMB: 900}}); err != nil {
		t.Fatalf("today's path checks nothing: %v", err)
	}
}

// The three-term MTP charge on a delta-net hybrid shaped like the CUDA
// yardstick (64 + 1 blocks, ctx 32,768, -b 8192, one slot, n_max 2): it
// covers the measured MTP-on cost (1,182 MiB) without leaning on the compute
// row. The CUDA charge on the same inputs is today's, unchanged.
func TestEmbeddedMTPReservationForNonCUDARows(t *testing.T) {
	model := placementMTPYardstick()
	strategy := &placement.Strategy{
		ContextSize: 32768, BatchSize: 8192, Parallel: 1, KVPlacement: "gpu",
		Draft: &placement.DraftConfig{Type: placement.DraftMTP, SpecType: "draft-mtp", DraftMax: 2},
	}
	kv := placement.EmbeddedMTPContextMB(model, 32768, "f16")
	if kv <= 0 {
		t.Fatal("no MTP context for the fixture")
	}
	cudaRows, err := embeddedMTPPreflightReservation(model, strategy, []preflightDevice{{Name: "CUDA0", ComputeMB: 1114}, {Name: "Host", ComputeMB: 10}})
	if err != nil || !reflect.DeepEqual(cudaRows, []preflightDevice{{Name: "CUDA0", ContextMB: kv, ComputeMB: 1114}}) {
		t.Fatalf("CUDA charge changed: %#v %v", cudaRows, err)
	}

	adm := &admissionRows{prefix: "Vulkan", devices: []admissionDevice{{Row: "Vulkan0", View: 0, FreeOK: true, FreeMB: 30000}}}
	got, err := embeddedMTPPreflightReservationRows(adm, model, strategy, []preflightDevice{{Name: "Vulkan0", ModelMB: 16773, ContextMB: 1241, ComputeMB: 368}, {Name: "Host", ModelMB: 682}})
	if err != nil {
		t.Fatal(err)
	}
	// nextn ceil(334.75), rs ceil(2 x 149.625), h_nextn ceil(4 x 5120 x 8193 B)
	want := []preflightDevice{{Name: "Vulkan0", ModelMB: 335, ContextMB: kv + 300 + 161, ComputeMB: 1024}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("non-CUDA MTP charge:\n got  %#v\n want %#v", got, want)
	}
	onCUDAShape, err := embeddedMTPPreflightReservationRows(adm, model, strategy, []preflightDevice{{Name: "Vulkan0", ComputeMB: 1114}})
	if err != nil || onCUDAShape[0].TotalMB() < 1182 {
		t.Fatalf("charge %#v below the measured 1,182 MiB (%v)", onCUDAShape, err)
	}

	// A term that cannot be derived refuses MTP, never charges zero.
	unknown := *model
	unknown.NonExpertLayerBytes = nil
	if _, err := embeddedMTPPreflightReservationRows(adm, &unknown, strategy, []preflightDevice{{Name: "Vulkan0", ComputeMB: 368}}); err == nil {
		t.Fatal("unknown NextN weights must refuse embedded MTP")
	}
	if _, err := embeddedMTPPreflightReservationRows(adm, model, strategy, []preflightDevice{{Name: "Host", ComputeMB: 368}}); err == nil || !strings.Contains(err.Error(), "no measured Vulkan device rows") {
		t.Fatalf("no device row: %v", err)
	}
}

// placementMTPYardstick is a synthetic 64 + 1-block delta-net hybrid whose
// recurrent state is 149.625 MiB per slot and whose NextN block is 334.75 MiB.
func placementMTPYardstick() *placement.ModelProfile {
	layers := make([]int64, 65)
	for i := range layers {
		layers[i] = 256 * testMiB
	}
	layers[64] = 351010816
	return &placement.ModelProfile{
		ModelArch: "qwen35", NumLayers: 65, NextNPredictLayers: 1, HasSSM: 1, FullAttnInterval: 4,
		HeadCountKV: 4, KeyLength: 256, ValueLength: 256, EmbeddingLength: 5120,
		SSMConvKernel: 4, SSMStateSize: 128, SSMGroupCount: 16, SSMInnerSize: 6144, SSMTimeStepRank: 48,
		NonExpertLayerBytes: layers,
	}
}

// gatedPreflightFixture is a fake backend directory: a llama-server that
// lists one Vulkan device and a llama-fit-params that prints the oracle rows.
func gatedPreflightFixture(t *testing.T, listing, oracle string) *backendInfo {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell backend fixture")
	}
	dir := t.TempDir()
	server := "#!/bin/sh\n[ \"$1\" = --list-devices ] && printf '" + listing + "'\nexit 0\n"
	if err := os.WriteFile(filepath.Join(dir, "llama-server"), []byte(server), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "llama-fit-params"), []byte("#!/bin/sh\nprintf '"+oracle+"'\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return &backendInfo{Path: filepath.Join(dir, "llama-server"), Tag: "vulkan", Dialect: "vulkan"}
}

const gatedOracleRows = `Vulkan0 16773 1241 368\nHost 682 0 144\n`

// End to end through preflightPlacement on the AMD card beside NVIDIA cards:
// the listing reports total as free while sysfs shows 26,314 MiB in use, so
// the plan is short and is re-planned instead of admitted. Without the
// opt-in, or on a single-vendor host, it is today's empty admission, with no
// listing and no inventory read beyond the opt-in check.
func TestGatedPreflightRefusesAnOverCommitTheListingHides(t *testing.T) {
	be := gatedPreflightFixture(t, `Available devices:\n  Vulkan0: AMD Test GPU 32G (RADV) (32752 MiB, 32752 MiB free)\n`, gatedOracleRows)
	fakeAMDGPUSysfs(t, "0000:19:00.0", 32752*testMiB, 26314*testMiB, "active")
	caps := &detect.Capabilities{GPUs: []detect.GPU{admissionAMD32(0)}}
	dir := t.TempDir()
	model := &placement.ModelProfile{Path: filepath.Join(dir, "m.gguf"), NumLayers: 65}
	plain := func() *placement.Strategy {
		return &placement.Strategy{ContextSize: 32768, BatchSize: 8192, Parallel: 1, KVPlacement: "gpu"}
	}
	run := func(optIn bool, inventory []detect.PCIDisplay) (preflightOutcome, int, int) {
		reads := gatedHostFixture(t, inventory, optIn)
		before := devicelist.StartedListings()
		out := preflightPlacement(&launchRequest{}, be, &configForPreflight{CacheDir: t.TempDir()}, caps, model, plain(), []string{be.Path, "-m", model.Path})
		return out, devicelist.StartedListings() - before, *reads
	}

	today, listings, reads := run(false, multiVendorInventory(true))
	if today.Err != nil || today.DoesNotFit || today.Device != -1 || listings != 0 || reads != 0 {
		t.Fatalf("opt-in off: %+v listings=%d reads=%d", today, listings, reads)
	}
	single, listings, _ := run(true, multiVendorInventory(false))
	if !reflect.DeepEqual(single, today) || listings != 0 {
		t.Fatalf("single-vendor host differs from today:\n%+v\n%+v (listings %d)", single, today, listings)
	}

	gated, listings, _ := run(true, multiVendorInventory(true))
	free := 32752 - 26314
	want := 16773 + 1241 + 368 + placement.UnmeasuredCUDAOverheadMB - free
	if gated.Err != nil || !gated.DoesNotFit || gated.Device != 0 || gated.DeficitMB != want || listings != 1 {
		t.Fatalf("gated: %+v listings=%d, want a %d MiB deficit on GPU 0 after one listing", gated, listings, want)
	}
	if gated.DeviceComputeMB != 368 || gated.DeviceContextMB != 1241 {
		t.Fatalf("deficit rows: compute %d context %d", gated.DeviceComputeMB, gated.DeviceContextMB)
	}

	// The same card without a PCI address (detected through vulkaninfo): the
	// listing's figure is the only one and it is the whole heap, so the
	// attempt fails closed instead of admitting against 32 GiB.
	noPCI := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "AMD Test GPU 32G (RADV)", VRAMTotalMB: 32752}}}
	gatedHostFixture(t, multiVendorInventory(true), true)
	out := preflightPlacement(&launchRequest{}, be, &configForPreflight{CacheDir: t.TempDir()}, noPCI, model, plain(), []string{be.Path, "-m", model.Path})
	if out.Err == nil || !strings.Contains(out.Err.Error(), "free memory of Vulkan0") {
		t.Fatalf("unknown free memory: %+v", out)
	}
	// While a calibration baseline serves, the same unknown is inconclusive.
	out = preflightPlacement(&launchRequest{}, be, &configForPreflight{CacheDir: t.TempDir(), BaselineServing: true}, noPCI, model, plain(), []string{be.Path, "-m", model.Path})
	if out.Err != nil || out.ProbeUnavailable == "" || out.DoesNotFit {
		t.Fatalf("prescreen: %+v", out)
	}
}

// A CUDA backend on the same multi-vendor host is today's admission with the
// opt-in on: same outcome, no inventory read, no listing.
func TestGatedPreflightLeavesCUDALaunchesAlone(t *testing.T) {
	be := gatedPreflightFixture(t, `Available devices:\n  CUDA0: NVIDIA Test GPU 24G (24564 MiB, 21564 MiB free)\n`, `CUDA0 16773 1241 368\nHost 682 0 144\n`)
	be.Tag, be.Dialect = "llama", "llama"
	caps := &detect.Capabilities{GPUs: []detect.GPU{admissionNVIDIA24()}}
	model := &placement.ModelProfile{Path: filepath.Join(t.TempDir(), "m.gguf"), NumLayers: 65}
	run := func(optIn bool) (preflightOutcome, int, int) {
		reads := gatedHostFixture(t, multiVendorInventory(true), optIn)
		before := devicelist.StartedListings()
		s := &placement.Strategy{ContextSize: 32768, BatchSize: 8192, Parallel: 1, KVPlacement: "gpu"}
		out := preflightPlacement(&launchRequest{}, be, &configForPreflight{CacheDir: t.TempDir()}, caps, model, s, []string{be.Path, "-m", model.Path})
		return out, devicelist.StartedListings() - before, *reads
	}
	off, _, _ := run(false)
	on, listings, reads := run(true)
	if !reflect.DeepEqual(on, off) || listings != 0 || reads != 0 {
		t.Fatalf("CUDA admission changed with the opt-in:\n%+v\n%+v (listings %d, reads %d)", on, off, listings, reads)
	}
	if off.DoesNotFit || off.Device != -1 || off.Err != nil {
		t.Fatalf("the CUDA row was not admitted as today: %+v", off)
	}
}

// Embedded MTP on a gated Vulkan row is admitted when the reservation fits,
// rejected as a companion (target-only) when only the target fits, and still
// refused without the opt-in, as today.
func TestGatedPreflightProvesEmbeddedMTP(t *testing.T) {
	fakeAMDGPUSysfs(t, "0000:19:00.0", 32752*testMiB, 10*testMiB, "active")
	caps := &detect.Capabilities{GPUs: []detect.GPU{admissionAMD32(0)}}
	model := placementMTPYardstick()
	model.ModelArch = "" // the fake oracle carries no architecture names
	model.Path = filepath.Join(t.TempDir(), "m.gguf")
	mtp := func() *placement.Strategy {
		return &placement.Strategy{ContextSize: 32768, BatchSize: 8192, Parallel: 1, KVPlacement: "gpu",
			Draft: &placement.DraftConfig{Type: placement.DraftMTP, SpecType: "draft-mtp", DraftMax: 2}}
	}
	run := func(optIn bool, freeMB int) preflightOutcome {
		be := gatedPreflightFixture(t, `Available devices:\n  Vulkan0: AMD Test GPU 32G (RADV) (32752 MiB, `+strconv.Itoa(freeMB)+` MiB free)\n`, gatedOracleRows)
		gatedHostFixture(t, multiVendorInventory(true), optIn)
		return preflightPlacement(&launchRequest{}, be, &configForPreflight{CacheDir: t.TempDir()}, caps, model, mtp(), []string{be.Path, "-m", model.Path})
	}
	if today := run(false, 30000); !today.CompanionRejected || today.Err != nil {
		t.Fatalf("opt-in off: embedded MTP must stay refused as today: %+v", today)
	}
	if roomy := run(true, 30000); roomy.CompanionRejected || roomy.DoesNotFit || roomy.Err != nil {
		t.Fatalf("gated, room for MTP: %+v", roomy)
	}
	// 19,500 MiB free: the target's 19,406 MiB fit, the MTP charge does not.
	if tight := run(true, 19500); !tight.CompanionRejected || tight.DoesNotFit || tight.Err != nil {
		t.Fatalf("gated, room for the target only: %+v", tight)
	}
}

// Scope keys gain the admission-rows component only on the gated path, so a
// record written while the rows were skipped is not found there and every
// other key is unchanged.
func TestAdmissionRowsScopeOnlyOnTheGatedPath(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the opt-in is Linux only")
	}
	cuda := &backendInfo{Path: "/x/llama-server", Tag: "llama", Dialect: "llama", Identity: "cuda-build"}
	vulkan := &backendInfo{Path: "/x/llama-server-vulkan", Tag: "vulkan", Dialect: "vulkan", Identity: "vulkan-build"}
	keys := func(optIn bool, be *backendInfo, inventory []detect.PCIDisplay) (string, string, string) {
		gatedHostFixture(t, inventory, optIn)
		req, _, model, _, caps := calibrateTestSetup(20 * 1024)
		scope := calibrationScope(req, model, be, caps, nil)
		plain := scope
		plain.AdmissionRows = ""
		return verifiedConfigScopeKey(req, model, be, caps), scope.String(), plain.String()
	}
	for _, c := range []struct {
		name  string
		optIn bool
		be    *backendInfo
		inv   []detect.PCIDisplay
	}{
		{"opt-in off", false, vulkan, multiVendorInventory(true)},
		{"single-vendor host", true, vulkan, multiVendorInventory(false)},
		{"CUDA backend", true, cuda, multiVendorInventory(true)},
	} {
		offVerified, offScope, _ := keys(false, c.be, c.inv)
		verified, scope, plain := keys(c.optIn, c.be, c.inv)
		if verified != offVerified || scope != offScope || scope != plain {
			t.Errorf("%s: keys changed off the gated path", c.name)
		}
	}
	offVerified, offScope, _ := keys(false, vulkan, multiVendorInventory(true))
	verified, scope, plain := keys(true, vulkan, multiVendorInventory(true))
	if verified == offVerified || scope == offScope || plain != offScope {
		t.Fatal("the gated path must get its own verified-config and calibration scope, and nothing else in the key may change")
	}
}
