package main

import (
	"errors"
	"fmt"
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

// A synthetic multi-vendor host: two NVIDIA cards, one AMD card and a BMC
// display in the OS inventory, at made-up PCI addresses with round sizes.
const (
	synthNV0  = "0000:0a:00.0"
	synthNV1  = "0000:0b:00.0"
	synthAMD  = "0000:0c:00.0"
	testMiB   = int64(1) << 20
	synthAMDT = 32768 // MiB
)

func synthInventory(withAMD bool) []detect.PCIDisplay {
	inv := []detect.PCIDisplay{
		{Address: "0000:01:00.0", Vendor: "0x1a03", Driver: "ast"},
		{Address: synthNV0, Vendor: "nvidia", Driver: "nvidia"},
		{Address: synthNV1, Vendor: "nvidia", Driver: "nvidia"},
	}
	if withAMD {
		inv = append(inv, detect.PCIDisplay{Address: synthAMD, Vendor: "amd", Driver: "amdgpu"})
	}
	return inv
}

func admissionNVIDIA24() detect.GPU {
	return detect.GPU{Index: 0, Name: "NVIDIA Test GPU 24G", VRAMTotalMB: 24576, VRAMUsedMB: 3000, PCIBusID: "00000000:0A:00.0"}
}

func admissionAMD32(index int) detect.GPU {
	return detect.GPU{Index: index, Name: "AMD Test GPU 32G (RADV)", VRAMTotalMB: synthAMDT, PCIBusID: synthAMD}
}

func synthTwoNVIDIA() *detect.Capabilities {
	b := admissionNVIDIA24()
	b.Index, b.Name, b.VRAMTotalMB, b.PCIBusID = 1, "NVIDIA Test GPU 12G", 12288, "00000000:0B:00.0"
	return &detect.Capabilities{GPUs: []detect.GPU{admissionNVIDIA24(), b}}
}

func vulkanTestBackend() *backendInfo {
	return &backendInfo{Path: "/x/llama-server-vulkan", Tag: "vulkan", Dialect: "vulkan"}
}

// gatedHostFixture sets the opt-in and serves inventory to the launch; the
// result counts the inventory reads.
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

func amdListing(freeMB int) string {
	return fmt.Sprintf("  Vulkan0: AMD Test GPU 32G (RADV) (%d MiB, %d MiB free)\n", synthAMDT, freeMB)
}

// The gate: opt-in, Linux, a non-CUDA dialect, an inventory with GPUs of two
// vendors. The inventory is read only after the opt-in and the dialect.
func TestNonCUDAAdmissionGate(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the opt-in is Linux only")
	}
	nvidiaOnly := synthInventory(false)
	mixed := synthInventory(true)
	vfio := append(synthInventory(false), detect.PCIDisplay{Address: synthAMD, Vendor: "amd", Driver: "vfio-pci"})
	amdOnly := []detect.PCIDisplay{{Address: "0000:01:00.0", Vendor: "0x1a03", Driver: "ast"}, {Address: synthAMD, Vendor: "amd", Driver: "amdgpu"}}
	workaround := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "AMD Test GPU 32G (RADV)", VRAMTotalMB: synthAMDT}}}
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
		{"opt-in off", false, vulkanTestBackend(), synthTwoNVIDIA(), mixed, false, 0},
		{"NVIDIA-only host, Vulkan", true, vulkanTestBackend(), synthTwoNVIDIA(), nvidiaOnly, false, 1},
		{"CUDA backend beside AMD", true, cuda, synthTwoNVIDIA(), mixed, false, 0},
		{"Vulkan on NVIDIA beside AMD", true, vulkanTestBackend(), synthTwoNVIDIA(), mixed, true, 1},
		{"vulkaninfo-only AMD card beside NVIDIA", true, vulkanTestBackend(), workaround, mixed, true, 1},
		{"other vendor bound to vfio-pci", true, vulkanTestBackend(), synthTwoNVIDIA(), vfio, false, 1},
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

// The launch-path switches built on the gate: the staged-prefill raise and
// the MoE post-launch system probes run off the gate and not on it.
func TestGatedLaunchSkipsCUDAOnlyLaunchSteps(t *testing.T) {
	caps := synthTwoNVIDIA()
	for _, c := range []struct {
		optIn bool
		inv   []detect.PCIDisplay
		be    *backendInfo
		apply bool
	}{
		{false, synthInventory(true), vulkanTestBackend(), true},
		{true, synthInventory(false), vulkanTestBackend(), true},
		{true, synthInventory(true), &backendInfo{Path: "/x/llama-server", Dialect: "llama"}, true},
		{true, synthInventory(true), vulkanTestBackend(), false},
	} {
		gatedHostFixture(t, c.inv, c.optIn)
		if got := stagedPrefillRaiseApplies(c.be, caps); got != c.apply {
			t.Errorf("opt-in %v, dialect %s: staged raise applies=%v, want %v", c.optIn, c.be.Dialect, got, c.apply)
		}
		if got := postLaunchSystemProbesApply(c.be, caps); got != c.apply {
			t.Errorf("opt-in %v, dialect %s: post-launch probes apply=%v, want %v", c.optIn, c.be.Dialect, got, c.apply)
		}
	}
}

// A gated refusal for want of proof keeps a reused verified config; every
// other launch failure still retires it.
func TestVerifiedConfigKeptAfterAGatedRefusal(t *testing.T) {
	refusal := fmt.Errorf("memory preflight failed closed: %w", admissionUnknown(errors.New("free memory of Vulkan0 is unknown")))
	if verifiedConfigStaleAfter(nil) || verifiedConfigStaleAfter(refusal) {
		t.Fatal("no failure, or a gated refusal for want of proof, must keep the verified config")
	}
	if !verifiedConfigStaleAfter(errors.New("memory preflight did not converge")) {
		t.Fatal("an ordinary launch failure must retire the verified config")
	}
}

// Each row maps to its view GPU; free memory is the minimum of the ok
// readers. The listing reports the AMD card's whole heap as free while sysfs
// shows another process holding most of it: the listing figure is not used.
func TestAdmissionRowsMapRowsAndReadEachVendor(t *testing.T) {
	fakeAdmissionListing(t, "Available devices:\n"+
		"  Vulkan0: NVIDIA Test GPU 24G (24800 MiB, 20000 MiB free)\n"+
		"  Vulkan1: AMD Test GPU 32G (RADV) (32768 MiB, 32768 MiB free)\n", nil)
	// One byte past 26,000 MiB used: free rounds down to 6,767.
	fakeAMDGPUSysfs(t, synthAMD, synthAMDT*testMiB, 26000*testMiB+1, "active")
	caps := &detect.Capabilities{GPUs: []detect.GPU{admissionNVIDIA24(), admissionAMD32(1)}}
	rows, err := buildAdmissionRows(vulkanTestBackend(), caps, synthInventory(true), false)
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
	if !nv.FreeOK || nv.FreeMB != 20000 || nv.Vendor != "nvidia" || !reflect.DeepEqual(nv.Sources, []string{"nvml=21576", "listing=20000"}) {
		t.Fatalf("NVIDIA row: %+v", nv)
	}
	amd, _ := rows.device("Vulkan1")
	if !amd.FreeOK || amd.FreeMB != 6767 || amd.PCI != synthAMD || !reflect.DeepEqual(amd.Sources, []string{"sysfs=6767"}) {
		t.Fatalf("AMD row: %+v", amd)
	}
	if !strings.Contains(strings.Join(amd.Reasons, ";"), "whole heap as free") {
		t.Fatalf("the listing's total-as-free figure was not set aside: %q", amd.Reasons)
	}
	if got := rows.freeMB(caps.GPUs[1]); got != 6767 {
		t.Fatalf("admission free for GPU 1 = %d", got)
	}
	if !strings.Contains(rows.summary(), "Vulkan1 = GPU 1 (AMD Test GPU 32G (RADV), "+synthAMD+"): free 6767 MiB") {
		t.Fatalf("summary %q", rows.summary())
	}
	over := rows.overheadByGPU(t.TempDir(), caps.GPUs)
	if over[0] != placement.UnmeasuredCUDAOverheadMB || over[1] != placement.UnmeasuredCUDAOverheadMB {
		t.Fatalf("gated overhead %v", over)
	}
}

// A listing that does not show exactly the plan's GPUs in view order leaves
// the order unverified, and the attempt is refused with a next step.
func TestAdmissionRowsRefuseAnUnverifiedOrder(t *testing.T) {
	two := &detect.Capabilities{GPUs: []detect.GPU{admissionNVIDIA24(), admissionAMD32(1)}}
	nvLine := "  Vulkan0: NVIDIA Test GPU 24G (24800 MiB, 20000 MiB free)\n"
	amdLine := "  Vulkan1: AMD Test GPU 32G (RADV) (32768 MiB, 30000 MiB free)\n"
	for _, c := range []struct {
		name string
		caps *detect.Capabilities
		out  string
		err  error
	}{
		{"unrestricted: an extra device", two, nvLine + amdLine + "  Vulkan2: NVIDIA Test GPU 12G (12288 MiB, 12000 MiB free)\n", nil},
		{"swapped order", two, "  Vulkan0: AMD Test GPU 32G (RADV) (32768 MiB, 30000 MiB free)\n  Vulkan1: NVIDIA Test GPU 24G (24800 MiB, 20000 MiB free)\n", nil},
		{"another card of that name", two, nvLine + "  Vulkan1: AMD Test GPU 32G (RADV) (16384 MiB, 16000 MiB free)\n", nil},
		{"listing failed", two, "", errors.New("exit status 1")},
		{"GPU indexes not 0..n-1", &detect.Capabilities{GPUs: []detect.GPU{admissionAMD32(1)}}, amdLine, nil},
	} {
		fakeAdmissionListing(t, c.out, c.err)
		_, err := buildAdmissionRows(vulkanTestBackend(), c.caps, synthInventory(true), false)
		if err == nil || !strings.Contains(err.Error(), "could not be verified") || !strings.Contains(err.Error(), "--gpus") {
			t.Errorf("%s: err %v", c.name, err)
		}
	}
}

// Unknown stays unknown: no ok reader means the device is refused, and the
// refusal names the readers.
func TestAdmissionRowsUnknownMemoryFailsClosed(t *testing.T) {
	heapAsFree := amdListing(synthAMDT)
	oracle := []preflightDevice{{Name: "Vulkan0", ModelMB: 16000, ContextMB: 1200, ComputeMB: 400}, {Name: "Host", ModelMB: 600}}
	one := &detect.Capabilities{GPUs: []detect.GPU{admissionAMD32(0)}}

	fakeAdmissionListing(t, heapAsFree, nil)
	fakeAMDGPUSysfs(t, synthAMD, synthAMDT*testMiB, 10*testMiB, "suspended")
	rows, err := buildAdmissionRows(vulkanTestBackend(), one, synthInventory(true), false)
	if err != nil {
		t.Fatal(err)
	}
	err = rows.checkMemory(oracle)
	if err == nil || !strings.Contains(err.Error(), "runtime-suspended") || !strings.Contains(err.Error(), "whole heap as free") {
		t.Fatalf("suspended card with no budget: %v", err)
	}

	// Runtime PM disabled ("unsupported"): the card never suspends, so sysfs
	// is read.
	fakeAMDGPUSysfs(t, synthAMD, synthAMDT*testMiB, 10*testMiB, "unsupported")
	if rows, err = buildAdmissionRows(vulkanTestBackend(), one, synthInventory(true), false); err != nil {
		t.Fatal(err)
	}
	if err := rows.checkMemory(oracle); err != nil || rows.freeMB(one.GPUs[0]) != synthAMDT-10 {
		t.Fatalf("runtime PM unsupported: err %v free %d", err, rows.freeMB(one.GPUs[0]))
	}

	// Detected through vulkaninfo: no PCI address.
	noPCI := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "AMD Test GPU 32G (RADV)", VRAMTotalMB: synthAMDT}}}
	if rows, err = buildAdmissionRows(vulkanTestBackend(), noPCI, synthInventory(true), false); err != nil {
		t.Fatal(err)
	}
	if err := rows.checkMemory(oracle); err == nil || !strings.Contains(err.Error(), "no PCI address") {
		t.Fatalf("no PCI and no budget: %v", err)
	}
	fakeAdmissionListing(t, amdListing(6000), nil)
	if rows, err = buildAdmissionRows(vulkanTestBackend(), noPCI, synthInventory(true), false); err != nil {
		t.Fatal(err)
	}
	if err := rows.checkMemory(oracle); err != nil || rows.freeMB(noPCI.GPUs[0]) != 6000 {
		t.Fatalf("a budget-backed listing is a reading: err %v free %d", err, rows.freeMB(noPCI.GPUs[0]))
	}

	// While the baseline serves, live readings are not used: an NVIDIA card
	// keeps its detection figure, the AMD card is unknown.
	fakeAdmissionListing(t, "  Vulkan0: NVIDIA Test GPU 24G (24800 MiB, 1000 MiB free)\n  Vulkan1: AMD Test GPU 32G (RADV) (32768 MiB, 1000 MiB free)\n", nil)
	fakeAMDGPUSysfs(t, synthAMD, synthAMDT*testMiB, 10*testMiB, "active")
	two := &detect.Capabilities{GPUs: []detect.GPU{admissionNVIDIA24(), admissionAMD32(1)}}
	if rows, err = buildAdmissionRows(vulkanTestBackend(), two, synthInventory(true), true); err != nil {
		t.Fatal(err)
	}
	if nv, _ := rows.device("Vulkan0"); !nv.FreeOK || nv.FreeMB != 21576 {
		t.Fatalf("NVIDIA row while the baseline serves: %+v", nv)
	}
	if amd, _ := rows.device("Vulkan1"); amd.FreeOK {
		t.Fatalf("AMD row while the baseline serves: %+v", amd)
	}
}

// Two cards the listing cannot tell apart (same name, totals within the order
// check's tolerance) are each charged against the smaller free figure.
func TestAdmissionRowsIdenticalCardsChargeTheSmallerFree(t *testing.T) {
	fakeAdmissionListing(t, "  Vulkan0: NVIDIA Test GPU 24G (24800 MiB, 21000 MiB free)\n  Vulkan1: NVIDIA Test GPU 24G (24400 MiB, 9000 MiB free)\n", nil)
	a, b := admissionNVIDIA24(), admissionNVIDIA24()
	b.Index, b.PCIBusID, b.VRAMUsedMB, b.VRAMTotalMB = 1, "00000000:0B:00.0", 15000, 24176
	rows, err := buildAdmissionRows(vulkanTestBackend(), &detect.Capabilities{GPUs: []detect.GPU{a, b}}, synthInventory(true), false)
	if err != nil {
		t.Fatal(err)
	}
	if rows.freeMB(a) != 9000 || rows.freeMB(b) != 9000 {
		t.Fatalf("identical cards: free %d and %d, want 9000 each", rows.freeMB(a), rows.freeMB(b))
	}
}

// The oracle's rows must match the map: one row per mapped device, and no
// nonzero row outside it.
func TestAdmissionRowsMustMatchTheMap(t *testing.T) {
	rows := &admissionRows{prefix: "Vulkan", devices: []admissionDevice{
		{Row: "Vulkan0", View: 0, FreeOK: true, FreeMB: 30000},
		{Row: "Vulkan1", View: 1, FreeOK: true, FreeMB: 30000},
	}}
	host := preflightDevice{Name: "Host", ModelMB: 600}
	both := []preflightDevice{{Name: "Vulkan0", ModelMB: 1}, {Name: "Vulkan1"}, host}
	if err := rows.checkRows(both); err != nil {
		t.Fatalf("a full row set with an idle device: %v", err)
	}
	for name, devs := range map[string][]preflightDevice{
		"empty":     nil,
		"Host only": {host},
		"partial":   {{Name: "Vulkan0", ModelMB: 1}, host},
		"duplicate": {{Name: "Vulkan0", ModelMB: 1}, {Name: "Vulkan0"}, {Name: "Vulkan1"}, host},
	} {
		if err := rows.checkRows(devs); err == nil || !strings.Contains(err.Error(), "want exactly one") {
			t.Errorf("%s: err %v", name, err)
		}
	}
	for _, foreign := range []preflightDevice{
		{Name: "SYCL0", ModelMB: 900, ContextMB: 10, ComputeMB: 10},
		{Name: "Vulkan3", ModelMB: 1},
		{Name: "CUDA0", ComputeMB: 100},
		{Name: "garbage", ContextMB: 5},
	} {
		if err := rows.checkRows(append(append([]preflightDevice(nil), both...), foreign)); err == nil || !strings.Contains(err.Error(), "device map does not contain") {
			t.Errorf("%s: err %v", foreign.Name, err)
		}
	}
	if err := rows.checkRows(append(append([]preflightDevice(nil), both...), preflightDevice{Name: "Vulkan3"})); err != nil {
		t.Fatalf("an empty row of an idle device outside the map: %v", err)
	}
	var cuda *admissionRows
	if err := cuda.checkRows(nil); err != nil {
		t.Fatalf("today's path checks nothing: %v", err)
	}
}

// The three-term MTP charge on a delta-net hybrid (64 + 1 blocks, ctx
// 32,768, -b 8192, one slot, n_max 2). The CUDA charge on the same inputs is
// today's, unchanged. The snapshots and h_nextn are not context.
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
	cudaRows, err := embeddedMTPPreflightReservation(model, strategy, []preflightDevice{{Name: "CUDA0", ComputeMB: 1100}, {Name: "Host", ComputeMB: 10}})
	if err != nil || !reflect.DeepEqual(cudaRows, []preflightDevice{{Name: "CUDA0", ContextMB: kv, ComputeMB: 1100}}) {
		t.Fatalf("CUDA charge changed: %#v %v", cudaRows, err)
	}

	adm := &admissionRows{prefix: "Vulkan", devices: []admissionDevice{{Row: "Vulkan0", View: 0, FreeOK: true, FreeMB: 30000}}}
	got, err := embeddedMTPPreflightReservationRows(adm, model, strategy, []preflightDevice{{Name: "Vulkan0", ModelMB: 16000, ContextMB: 1200, ComputeMB: 400}, {Name: "Host", ModelMB: 600}})
	if err != nil {
		t.Fatal(err)
	}
	// nextn ceil(334.75), rs ceil(2 x 149.625), h_nextn ceil(4 x 5120 x 8193 B)
	want := []preflightDevice{{Name: "Vulkan0", ModelMB: 335, ContextMB: kv, ComputeMB: 1024, UnaccountedMB: 300 + 161}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("non-CUDA MTP charge:\n got  %#v\n want %#v", got, want)
	}

	// A term that cannot be derived refuses MTP, never charges zero.
	unknown := *model
	unknown.NonExpertLayerBytes = nil
	if _, err := embeddedMTPPreflightReservationRows(adm, &unknown, strategy, []preflightDevice{{Name: "Vulkan0", ComputeMB: 400}}); err == nil {
		t.Fatal("unknown NextN weights must refuse embedded MTP")
	}
	if _, err := embeddedMTPPreflightReservationRows(adm, model, strategy, []preflightDevice{{Name: "Host", ComputeMB: 400}}); err == nil || !strings.Contains(err.Error(), "no measured Vulkan device rows") {
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
// prints listing for --list-devices and, unless oracle is "", a
// llama-fit-params that prints oracle (or fails when it is "FAIL").
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
	fit := "#!/bin/sh\nprintf '" + oracle + "'\n"
	if oracle == "FAIL" {
		fit = "#!/bin/sh\necho 'failed to load model' >&2\nexit 1\n"
	}
	if oracle != "" {
		if err := os.WriteFile(filepath.Join(dir, "llama-fit-params"), []byte(fit), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	return &backendInfo{Path: filepath.Join(dir, "llama-server"), Tag: "vulkan", Dialect: "vulkan"}
}

const (
	gatedOracleRows = `Vulkan0 16000 1200 400\nHost 600 0 100\n`
	gatedListing    = `Available devices:\n  Vulkan0: AMD Test GPU 32G (RADV) (32768 MiB, 32768 MiB free)\n`
)

func plainPlan() *placement.Strategy {
	return &placement.Strategy{ContextSize: 32768, BatchSize: 8192, Parallel: 1, KVPlacement: "gpu"}
}

// End to end through preflightPlacement on the AMD card beside NVIDIA cards:
// the listing reports total as free while sysfs shows 26,000 MiB in use, so
// the plan is short and is re-planned instead of admitted. Without the
// opt-in, or on a single-vendor host, it is today's empty admission, with no
// listing and no inventory read beyond the opt-in check.
func TestGatedPreflightRefusesAnOverCommitTheListingHides(t *testing.T) {
	be := gatedPreflightFixture(t, gatedListing, gatedOracleRows)
	fakeAMDGPUSysfs(t, synthAMD, synthAMDT*testMiB, 26000*testMiB, "active")
	caps := &detect.Capabilities{GPUs: []detect.GPU{admissionAMD32(0)}}
	model := &placement.ModelProfile{Path: filepath.Join(t.TempDir(), "m.gguf"), NumLayers: 65}
	run := func(optIn bool, inventory []detect.PCIDisplay) (preflightOutcome, int, int) {
		reads := gatedHostFixture(t, inventory, optIn)
		before := devicelist.StartedListings()
		out := preflightPlacement(&launchRequest{}, be, &configForPreflight{CacheDir: t.TempDir()}, caps, model, plainPlan(), []string{be.Path, "-m", model.Path})
		return out, devicelist.StartedListings() - before, *reads
	}

	today, listings, reads := run(false, synthInventory(true))
	if today.Err != nil || today.DoesNotFit || today.Device != -1 || listings != 0 || reads != 0 {
		t.Fatalf("opt-in off: %+v listings=%d reads=%d", today, listings, reads)
	}
	single, listings, _ := run(true, synthInventory(false))
	if !reflect.DeepEqual(single, today) || listings != 0 {
		t.Fatalf("single-vendor host differs from today:\n%+v\n%+v (listings %d)", single, today, listings)
	}

	gated, listings, _ := run(true, synthInventory(true))
	free := synthAMDT - 26000
	want := 16000 + 1200 + 400 + placement.UnmeasuredCUDAOverheadMB - free
	if gated.Err != nil || !gated.DoesNotFit || gated.Device != 0 || gated.DeficitMB != want || listings != 1 {
		t.Fatalf("gated: %+v listings=%d, want a %d MiB deficit on GPU 0 after one listing", gated, listings, want)
	}
	if gated.DeviceComputeMB != 400 || gated.DeviceContextMB != 1200 {
		t.Fatalf("deficit rows: compute %d context %d", gated.DeviceComputeMB, gated.DeviceContextMB)
	}

	// The same card without a PCI address (detected through vulkaninfo): the
	// listing's figure is the only one and it is the whole heap, so the
	// attempt fails closed instead of admitting against 32 GiB.
	noPCI := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "AMD Test GPU 32G (RADV)", VRAMTotalMB: synthAMDT}}}
	gatedHostFixture(t, synthInventory(true), true)
	out := preflightPlacement(&launchRequest{}, be, &configForPreflight{CacheDir: t.TempDir()}, noPCI, model, plainPlan(), []string{be.Path, "-m", model.Path})
	var unknown *admissionUnknownError
	if out.Err == nil || !strings.Contains(out.Err.Error(), "free memory of Vulkan0") || !errors.As(out.Err, &unknown) {
		t.Fatalf("unknown free memory: %+v", out)
	}
	// While a calibration baseline serves, the same unknown is inconclusive.
	out = preflightPlacement(&launchRequest{}, be, &configForPreflight{CacheDir: t.TempDir(), BaselineServing: true}, noPCI, model, plainPlan(), []string{be.Path, "-m", model.Path})
	if out.Err != nil || out.ProbeUnavailable == "" || out.DoesNotFit {
		t.Fatalf("prescreen: %+v", out)
	}
}

// Gated, every way the oracle can fail to prove the device refuses the
// attempt, never falls back to the contained probe and never admits "fits ()".
func TestGatedPreflightRefusesWithoutOracleProof(t *testing.T) {
	fakeAMDGPUSysfs(t, synthAMD, synthAMDT*testMiB, 10*testMiB, "active")
	caps := &detect.Capabilities{GPUs: []detect.GPU{admissionAMD32(0)}}
	model := &placement.ModelProfile{Path: filepath.Join(t.TempDir(), "m.gguf"), NumLayers: 65}
	for _, c := range []struct {
		name, oracle, want string
	}{
		{"no fit-params", "", "has no llama-fit-params"},
		{"oracle failed", "FAIL", "memory oracle failed"},
		{"Host row only", `Host 600 0 100\n`, "want exactly one"},
		{"row of another backend", `CUDA0 16000 1200 400\nHost 600 0 100\n`, "rows for Vulkan0"},
	} {
		be := gatedPreflightFixture(t, gatedListing, c.oracle)
		gatedHostFixture(t, synthInventory(true), true)
		before := devicelist.StartedListings()
		out := preflightPlacement(&launchRequest{}, be, &configForPreflight{CacheDir: t.TempDir()}, caps, model, plainPlan(), []string{be.Path, "-m", model.Path})
		var unknown *admissionUnknownError
		if out.Err == nil || !strings.Contains(out.Err.Error(), c.want) || !errors.As(out.Err, &unknown) || out.DoesNotFit || out.Evidence.Level == memoryEvidenceAllocated {
			t.Errorf("%s: %+v", c.name, out)
		}
		if c.oracle == "" || c.oracle == "FAIL" {
			if n := devicelist.StartedListings() - before; n != 0 {
				t.Errorf("%s: %d listings before the oracle proved anything", c.name, n)
			}
		}
	}
}

// A CUDA backend on the same multi-vendor host is today's admission with the
// opt-in on: same outcome, no inventory read, no listing.
func TestGatedPreflightLeavesCUDALaunchesAlone(t *testing.T) {
	be := gatedPreflightFixture(t, `Available devices:\n  CUDA0: NVIDIA Test GPU 24G (24576 MiB, 21576 MiB free)\n`, `CUDA0 16000 1200 400\nHost 600 0 100\n`)
	be.Tag, be.Dialect = "llama", "llama"
	caps := &detect.Capabilities{GPUs: []detect.GPU{admissionNVIDIA24()}}
	model := &placement.ModelProfile{Path: filepath.Join(t.TempDir(), "m.gguf"), NumLayers: 65}
	run := func(optIn bool) (preflightOutcome, int, int) {
		reads := gatedHostFixture(t, synthInventory(true), optIn)
		before := devicelist.StartedListings()
		out := preflightPlacement(&launchRequest{}, be, &configForPreflight{CacheDir: t.TempDir()}, caps, model, plainPlan(), []string{be.Path, "-m", model.Path})
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
// refused without the opt-in, as today. Attention-only here: the fake oracle
// carries no architecture names, so the snapshot term is covered by
// TestEmbeddedMTPReservationForNonCUDARows.
func TestGatedPreflightProvesEmbeddedMTP(t *testing.T) {
	fakeAMDGPUSysfs(t, synthAMD, synthAMDT*testMiB, 10*testMiB, "active")
	caps := &detect.Capabilities{GPUs: []detect.GPU{admissionAMD32(0)}}
	model := placementMTPYardstick()
	model.ModelArch, model.HasSSM = "", 0
	model.Path = filepath.Join(t.TempDir(), "m.gguf")
	mtp := func() *placement.Strategy {
		return &placement.Strategy{ContextSize: 32768, BatchSize: 8192, Parallel: 1, KVPlacement: "gpu",
			Draft: &placement.DraftConfig{Type: placement.DraftMTP, SpecType: "draft-mtp", DraftMax: 2}}
	}
	run := func(optIn bool, freeMB int) preflightOutcome {
		be := gatedPreflightFixture(t, `Available devices:\n`+strings.TrimSuffix(amdListing(freeMB), "\n")+`\n`, gatedOracleRows)
		gatedHostFixture(t, synthInventory(true), optIn)
		return preflightPlacement(&launchRequest{}, be, &configForPreflight{CacheDir: t.TempDir()}, caps, model, mtp(), []string{be.Path, "-m", model.Path})
	}
	if today := run(false, 30000); !today.CompanionRejected || today.Err != nil {
		t.Fatalf("opt-in off: embedded MTP must stay refused as today: %+v", today)
	}
	if roomy := run(true, 30000); roomy.CompanionRejected || roomy.DoesNotFit || roomy.Err != nil {
		t.Fatalf("gated, room for MTP: %+v", roomy)
	}
	// 19,000 MiB free: the target's 18,624 MiB fit, the MTP charge does not.
	if tight := run(true, 19000); !tight.CompanionRejected || tight.DoesNotFit || tight.Err != nil {
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
		{"opt-in off", false, vulkan, synthInventory(true)},
		{"single-vendor host", true, vulkan, synthInventory(false)},
		{"CUDA backend", true, cuda, synthInventory(true)},
	} {
		offVerified, offScope, _ := keys(false, c.be, c.inv)
		verified, scope, plain := keys(c.optIn, c.be, c.inv)
		if verified != offVerified || scope != offScope || scope != plain {
			t.Errorf("%s: keys changed off the gated path", c.name)
		}
	}
	offVerified, offScope, _ := keys(false, vulkan, synthInventory(true))
	verified, scope, plain := keys(true, vulkan, synthInventory(true))
	if verified == offVerified || scope == offScope || plain != offScope {
		t.Fatal("the gated path must get its own verified-config and calibration scope, and nothing else in the key may change")
	}
}
