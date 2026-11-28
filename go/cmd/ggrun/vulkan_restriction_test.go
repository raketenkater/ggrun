package main

import (
	"encoding/json"
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
)

// fakeVulkanDevice is one physical device a fake Vulkan loader enumerates.
type fakeVulkanDevice struct {
	name, driver string
	totalMB      int
	cpu          bool   // a CPU renderer such as llvmpipe
	uuid         string // the same card through two drivers shares it
	priority     int    // ggml's driver preference among one card's drivers: lower wins
}

// fakeVulkanLoader models both of ggml-vulkan's index spaces: without
// GGML_VK_VISIBLE_DEVICES it lists a filtered list (GPUs only, one entry per
// card, the preferred driver; a preferred duplicate replaces the earlier
// entry and goes to the end); with it, the variable's raw indices in order,
// unfiltered, and an invalid index ends the listing.
type fakeVulkanLoader struct {
	devices     []fakeVulkanDevice
	driverLines bool // whether ggml_vulkan's "<i> = <name> (<driver>)" lines are printed
}

func (l fakeVulkanLoader) defaultIndices() []int {
	var out []int
	for i, d := range l.devices {
		if d.cpu {
			continue
		}
		found := -1
		for k, j := range out {
			if l.devices[j].uuid == d.uuid {
				found = k
			}
		}
		if found < 0 {
			out = append(out, i)
			continue
		}
		if d.priority < l.devices[out[found]].priority {
			out = append(out[:found:found], out[found+1:]...)
			out = append(out, i)
		}
	}
	return out
}

func (l fakeVulkanLoader) output(visible string, set bool) string {
	indices := l.defaultIndices()
	if set {
		indices = nil
		for _, token := range strings.Split(visible, ",") {
			i, err := strconv.Atoi(token)
			if err != nil || i >= len(l.devices) {
				return fmt.Sprintf("ggml_vulkan: Invalid device index %s in GGML_VK_VISIBLE_DEVICES.\nterminate called after throwing an instance of 'std::runtime_error'\n", token)
			}
			indices = append(indices, i)
		}
	}
	var b strings.Builder
	if l.driverLines {
		for pos, i := range indices {
			fmt.Fprintf(&b, "ggml_vulkan: %d = %s (%s) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32\n", pos, l.devices[i].name, l.devices[i].driver)
		}
	}
	b.WriteString("Available devices:\n")
	for pos, i := range indices {
		fmt.Fprintf(&b, "  Vulkan%d: %s (%d MiB, %d MiB free)\n", pos, l.devices[i].name, l.devices[i].totalMB, l.devices[i].totalMB-100)
	}
	return b.String()
}

func (l fakeVulkanLoader) defaultListing() []listedDevice {
	return parseVulkanDeviceList(l.output("", false))
}

func (l fakeVulkanLoader) lister(calls *[]string) func(string) (string, error) {
	return func(visible string) (string, error) {
		*calls = append(*calls, visible)
		return l.output(visible, true), nil
	}
}

// A synthetic host: two NVIDIA cards found by nvidia-smi, and an AMD card.
func twoNVIDIACaps() *detect.Capabilities {
	return &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, Name: "NVIDIA Test GPU 24G", VRAMTotalMB: 24564, PCIBusID: "00000000:65:00.0"},
		{Index: 1, Name: "NVIDIA Test GPU 12G", VRAMTotalMB: 12288, PCIBusID: "00000000:B3:00.0"},
	}}
}

var (
	vkNV24   = fakeVulkanDevice{name: "NVIDIA Test GPU 24G", driver: "NVIDIA", totalMB: 24810, uuid: "nv24", priority: 1}
	vkNV12   = fakeVulkanDevice{name: "NVIDIA Test GPU 12G", driver: "NVIDIA", totalMB: 12534, uuid: "nv12", priority: 1}
	vkAMD    = fakeVulkanDevice{name: "AMD Test GPU (RADV TEST)", driver: "radv", totalMB: 32752, uuid: "amd", priority: 1}
	vkCPU    = fakeVulkanDevice{name: "llvmpipe (LLVM 17.0.6, 256 bits)", driver: "llvmpipe", totalMB: 32000, cpu: true, uuid: "cpu"}
	vkNV24NV = fakeVulkanDevice{name: "NVIDIA Test GPU 24G", driver: "NVK", totalMB: 24810, uuid: "nv24", priority: 2}
	vkNV12NV = fakeVulkanDevice{name: "NVIDIA Test GPU 12G", driver: "NVK", totalMB: 12534, uuid: "nv12", priority: 2}
	vkAMDVLK = fakeVulkanDevice{name: "AMD Test GPU", driver: "AMD open-source driver", totalMB: 32752, uuid: "amd", priority: 2}
)

const vkRawProbe = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

// The restriction exports raw indices, found by name, driver and total, and
// keeps them only when the listing under them shows the launch's cards.
func TestRestrictedVulkanOrderExportsRawIndices(t *testing.T) {
	for _, c := range []struct {
		name    string
		loader  fakeVulkanLoader
		req     *launchRequest
		caps    *detect.Capabilities
		today   string // vulkanDeviceOrder's value, from the filtered listing
		want    string // the exported raw value
		listing []string
	}{
		{
			name:   "a CPU renderer sorts first",
			loader: fakeVulkanLoader{devices: []fakeVulkanDevice{vkCPU, vkNV24, vkAMD, vkNV12}, driverLines: true},
			req:    &launchRequest{}, caps: twoNVIDIACaps(), today: "0,2", want: "1,3",
			listing: []string{vkRawProbe, "0,1,2,3", "1,3"},
		},
		{
			name:   "a second driver for every card",
			loader: fakeVulkanLoader{devices: []fakeVulkanDevice{vkNV24, vkNV24NV, vkNV12, vkNV12NV, vkAMD, vkAMDVLK}, driverLines: true},
			req:    &launchRequest{}, caps: twoNVIDIACaps(), today: "", want: "0,2",
			listing: []string{vkRawProbe, "0,1,2,3,4,5", "0,2"},
		},
		{
			name:   "the preferred driver comes later and is moved to the end",
			loader: fakeVulkanLoader{devices: []fakeVulkanDevice{vkAMDVLK, vkNV24, vkAMD, vkNV12}, driverLines: true},
			req:    &launchRequest{}, caps: twoNVIDIACaps(), today: "0,2", want: "1,3",
			listing: []string{vkRawProbe, "0,1,2,3", "1,3"},
		},
		{
			name:   "--gpus picks one card",
			loader: fakeVulkanLoader{devices: []fakeVulkanDevice{vkCPU, vkNV24, vkAMD, vkNV12}, driverLines: true},
			req:    &launchRequest{GPUsFlag: "1"}, caps: twoNVIDIACaps(), today: "2", want: "3",
			listing: []string{vkRawProbe, "0,1,2,3", "3"},
		},
		{
			name: "two identical cards keep their order",
			loader: fakeVulkanLoader{devices: []fakeVulkanDevice{
				vkNV24, vkAMD, {name: vkNV24.name, driver: "NVIDIA", totalMB: 24810, uuid: "nv24b", priority: 1}}, driverLines: true},
			req: &launchRequest{},
			caps: &detect.Capabilities{GPUs: []detect.GPU{
				{Index: 0, Name: "NVIDIA Test GPU 24G", VRAMTotalMB: 24564, PCIBusID: "00000000:17:00.0"},
				{Index: 1, Name: "NVIDIA Test GPU 24G", VRAMTotalMB: 24564, PCIBusID: "00000000:65:00.0"},
			}},
			today: "0,2", want: "0,2",
			listing: []string{vkRawProbe, "0,1,2", "0,2"},
		},
	} {
		listed := c.loader.defaultListing()
		if _, v, err := vulkanDeviceOrder(c.req, vulkanBackend(), c.caps, listed); err != nil || v != c.today {
			t.Fatalf("%s: today's value %q, %v; want %q", c.name, v, err, c.today)
		}
		var calls []string
		key, value, note, err := restrictedVulkanDeviceOrder(c.req, vulkanBackend(), c.caps, listed, c.loader.lister(&calls))
		if err != nil || key != "GGML_VK_VISIBLE_DEVICES" || value != c.want || !strings.Contains(note, "restricted") {
			t.Fatalf("%s: got %q=%q note %q err %v; want %q", c.name, key, value, note, err, c.want)
		}
		if !reflect.DeepEqual(calls, c.listing) {
			t.Fatalf("%s: listings %q, want %q", c.name, calls, c.listing)
		}
	}
}

// On any doubt the launch keeps today's value and says so.
func TestRestrictedVulkanOrderFallsBackToToday(t *testing.T) {
	three := fakeVulkanLoader{devices: []fakeVulkanDevice{vkNV24, vkNV12, vkAMD}, driverLines: true}
	for _, c := range []struct {
		name      string
		loader    fakeVulkanLoader
		listUnder func(fakeVulkanLoader) func(string) (string, error)
		reason    string
	}{
		{
			// Without driver lines a same-named duplicate cannot be told apart.
			name: "same-named duplicate, no driver lines",
			loader: fakeVulkanLoader{devices: []fakeVulkanDevice{
				vkNV24, {name: vkNV24.name, driver: "NVK", totalMB: 24810, uuid: "nv24", priority: 2}, vkNV12, vkAMD}},
			reason: "raw Vulkan list has 2 devices like NVIDIA Test GPU 24G",
		},
		{
			name: "backend ignores the variable", loader: three,
			listUnder: func(l fakeVulkanLoader) func(string) (string, error) {
				return func(string) (string, error) { return l.output("", false), nil }
			},
			reason: "did not report its raw Vulkan device count",
		},
		{
			name: "the restricted listing shows another card", loader: three,
			listUnder: func(l fakeVulkanLoader) func(string) (string, error) {
				return func(visible string) (string, error) {
					if visible == "0,1" {
						return l.output("0,2", true), nil
					}
					return l.output(visible, true), nil
				}
			},
			reason: "Vulkan1 is AMD Test GPU (RADV TEST) (radv, 32752 MiB), want NVIDIA Test GPU 12G (NVIDIA, 12534 MiB)",
		},
		{
			name: "listing fails", loader: three,
			listUnder: func(fakeVulkanLoader) func(string) (string, error) {
				return func(string) (string, error) { return "", fmt.Errorf("no answer") }
			},
			reason: "no answer",
		},
	} {
		listUnder := c.loader.lister(new([]string))
		if c.listUnder != nil {
			listUnder = c.listUnder(c.loader)
		}
		key, value, note, err := restrictedVulkanDeviceOrder(&launchRequest{}, vulkanBackend(), twoNVIDIACaps(), c.loader.defaultListing(), listUnder)
		if err != nil || key != "" || value != "" {
			t.Fatalf("%s: got %q=%q err %v; want today's empty value", c.name, key, value, err)
		}
		if !strings.Contains(note, "failed its check") || !strings.Contains(note, c.reason) || !strings.HasSuffix(note, "keeping no restriction") {
			t.Fatalf("%s: note %q, want reason %q", c.name, note, c.reason)
		}
	}
	// A non-identity value stays exported when its check fails, as today, and
	// the note says it was not verified.
	cpuFirst := fakeVulkanLoader{devices: []fakeVulkanDevice{vkCPU, vkNV24, vkAMD, vkNV12}, driverLines: true}
	key, value, note, err := restrictedVulkanDeviceOrder(&launchRequest{}, vulkanBackend(), twoNVIDIACaps(), cpuFirst.defaultListing(),
		func(string) (string, error) { return cpuFirst.output("", false), nil })
	if err != nil || key != "GGML_VK_VISIBLE_DEVICES" || value != "0,2" || !strings.HasSuffix(note, "keeping GGML_VK_VISIBLE_DEVICES=0,2 (not verified)") {
		t.Fatalf("got %q=%q note %q err %v", key, value, note, err)
	}
}

// Today's mapping error stays an error and nothing else is listed.
func TestRestrictedVulkanOrderKeepsTodaysError(t *testing.T) {
	other := parseVulkanDeviceList("  Vulkan0: AMD Test GPU (RADV TEST) (32752 MiB, 1 MiB free)\n  Vulkan1: NVIDIA Test GPU 12G (12534 MiB, 1 MiB free)\n")
	_, _, _, err := restrictedVulkanDeviceOrder(&launchRequest{}, vulkanBackend(), twoNVIDIACaps(), other, func(string) (string, error) {
		t.Fatal("listed again after the mapping failed")
		return "", nil
	})
	if err == nil {
		t.Fatal("a listing without one of the detected cards must stay an error")
	}
}

func multiVendorInventory(withAMD bool) []detect.PCIDisplay {
	inv := []detect.PCIDisplay{
		{Address: "0000:03:00.0", Vendor: "0x1a03", Driver: "ast"},
		{Address: "0000:65:00.0", Vendor: "nvidia", Driver: "nvidia"},
		{Address: "0000:b3:00.0", Vendor: "nvidia", Driver: "nvidia"},
	}
	if withAMD {
		inv = append(inv, detect.PCIDisplay{Address: "0000:19:00.0", Vendor: "amd", Driver: "amdgpu"})
	}
	return inv
}

// The rule: opt-in, Linux, a Vulkan backend on the NVIDIA cards, and a GPU of
// another vendor in the OS inventory. Anything else reads nothing more.
func TestVulkanRestrictionAppliesOnlyBesideAnotherVendor(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the opt-in is Linux only")
	}
	read := 0
	inventory := func(withAMD bool) func() []detect.PCIDisplay {
		return func() []detect.PCIDisplay { read++; return multiVendorInventory(withAMD) }
	}
	t.Setenv("GGRUN_EXPERIMENTAL_AMD", "")
	if vulkanRestrictionApplies(vulkanBackend(), twoNVIDIACaps(), inventory(true)) || read != 0 {
		t.Fatalf("without the opt-in: applied or read the inventory (%d reads)", read)
	}
	t.Setenv("GGRUN_EXPERIMENTAL_AMD", "1")
	if !vulkanRestrictionApplies(vulkanBackend(), twoNVIDIACaps(), inventory(true)) {
		t.Fatal("a Vulkan backend on NVIDIA cards beside an AMD GPU must be restricted")
	}
	if vulkanRestrictionApplies(vulkanBackend(), twoNVIDIACaps(), inventory(false)) {
		t.Fatal("a single-vendor host must keep today's path")
	}
	read = 0
	cuda := &backendInfo{Path: "/x/llama-server", Dialect: "llama"}
	if vulkanRestrictionApplies(cuda, twoNVIDIACaps(), inventory(true)) || read != 0 {
		t.Fatal("a CUDA backend is never restricted and reads no inventory")
	}
	if vulkanRestrictionApplies(vulkanBackend(), twoNVIDIACaps(), nil) {
		t.Fatal("no inventory reader means no restriction")
	}
}

// fakeVulkanBackend writes a shell backend whose raw list is llvmpipe, the
// 24G card, the AMD card and the 12G card, and whose default list leaves out
// the CPU renderer.
func fakeVulkanBackend(t *testing.T) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake backends are shell scripts")
	}
	script := `#!/bin/sh
raw() {
  case "$1" in
  0) echo "llvmpipe (LLVM 17.0.6, 256 bits)|llvmpipe|32000" ;;
  1) echo "NVIDIA Test GPU 24G|NVIDIA|24810" ;;
  2) echo "AMD Test GPU (RADV TEST)|radv|32752" ;;
  3) echo "NVIDIA Test GPU 12G|NVIDIA|12534" ;;
  *) return 1 ;;
  esac
}
if [ -n "${GGML_VK_VISIBLE_DEVICES+set}" ]; then list=$(echo "$GGML_VK_VISIBLE_DEVICES" | tr ',' ' '); else list="1 2 3"; fi
for i in $list; do
  raw "$i" >/dev/null || { echo "ggml_vulkan: Invalid device index $i in GGML_VK_VISIBLE_DEVICES." >&2; exit 134; }
done
pos=0
for i in $list; do
  line=$(raw "$i"); name=${line%%|*}; rest=${line#*|}; driver=${rest%%|*}
  echo "ggml_vulkan: $pos = $name ($driver) | uma: 0 | fp16: 1" >&2
  pos=$((pos+1))
done
echo "Available devices:"
pos=0
for i in $list; do
  line=$(raw "$i"); name=${line%%|*}; rest=${line#*|}; mib=${rest#*|}
  echo "  Vulkan$pos: $name ($mib MiB, $mib MiB free)"
  pos=$((pos+1))
done
`
	bin := filepath.Join(t.TempDir(), "llama-server-vulkan")
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return bin
}

// The listings run under their own visibility; the unfiltered listing never
// inherits this process's value, and no listing changes it.
func TestVulkanListingsRunUnderTheirOwnVisibility(t *testing.T) {
	bin := fakeVulkanBackend(t)
	t.Setenv("GGML_VK_VISIBLE_DEVICES", "0")
	all, err := listVulkanDevices(bin)
	want := []listedDevice{
		{Index: 0, Name: "NVIDIA Test GPU 24G", TotalMB: 24810, Driver: "NVIDIA"},
		{Index: 1, Name: "AMD Test GPU (RADV TEST)", TotalMB: 32752, Driver: "radv"},
		{Index: 2, Name: "NVIDIA Test GPU 12G", TotalMB: 12534, Driver: "NVIDIA"},
	}
	if err != nil || !reflect.DeepEqual(all, want) {
		t.Fatalf("unfiltered listing = %+v, %v", all, err)
	}
	out, err := listVulkanOutputUnder(bin, "0,3")
	if got := parseVulkanDeviceList(out); err != nil || len(got) != 2 || got[0].Driver != "llvmpipe" || got[1].Name != "NVIDIA Test GPU 12G" {
		t.Fatalf("listing under 0,3 = %+v, %v", got, err)
	}
	out, _ = listVulkanOutputUnder(bin, "0,7")
	if m := vulkanInvalidIndex.FindStringSubmatch(out); m == nil || m[1] != "7" {
		t.Fatalf("an invalid index was not reported: %q", out)
	}
	if os.Getenv("GGML_VK_VISIBLE_DEVICES") != "0" {
		t.Fatal("a listing changed this process's environment")
	}
}

// The launch's own wiring, end to end through applyVulkanDeviceOrder. Without
// the opt-in it is today's code: one listing and the filtered value, exported
// as before (on this raw list today's 0,2 names the CPU renderer and the AMD
// card: the known issue the opt-in path corrects). With it: three more
// listings and the raw value.
func TestApplyVulkanDeviceOrderWiring(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the opt-in is Linux only")
	}
	reads := 0
	saved := hostPCIDisplays
	hostPCIDisplays = func() []detect.PCIDisplay { reads++; return multiVendorInventory(true) }
	t.Cleanup(func() { hostPCIDisplays = saved })

	for _, c := range []struct {
		optIn     string
		env       string
		listings  int
		reads     int
		stderrHas string
	}{
		{optIn: "", env: "0,2", listings: 1, reads: 0},
		{optIn: "1", env: "1,3", listings: 4, reads: 1, stderrHas: "[launch] Vulkan backend restricted to the 2 detected GPU(s)"},
	} {
		be := &backendInfo{Path: fakeVulkanBackend(t), Dialect: "vulkan"}
		goldenPlanUnsetenv(t, pureEnvKeys...)
		t.Setenv("GGRUN_EXPERIMENTAL_AMD", c.optIn)
		reads = 0
		before := devicelist.StartedListings()
		errOut := captureStderrForPureEnv(t)
		out := captureStdout(t)
		applyVulkanDeviceOrder(&launchRequest{}, be, twoNVIDIACaps())
		printed, warned := out(), errOut()
		if got := os.Getenv("GGML_VK_VISIBLE_DEVICES"); got != c.env {
			t.Fatalf("opt-in %q: exported %q, want %q", c.optIn, got, c.env)
		}
		if got := devicelist.StartedListings() - before; got != c.listings || reads != c.reads {
			t.Fatalf("opt-in %q: %d listings and %d inventory reads, want %d and %d", c.optIn, got, reads, c.listings, c.reads)
		}
		if printed != "[launch] Vulkan device order: GGML_VK_VISIBLE_DEVICES="+c.env+" (backend devices listed in detected GPU order)\n" {
			t.Fatalf("opt-in %q: stdout %q", c.optIn, printed)
		}
		if (c.stderrHas == "") != (warned == "") || !strings.Contains(warned, c.stderrHas) {
			t.Fatalf("opt-in %q: stderr %q", c.optIn, warned)
		}
	}
}

// The export branch: with the restriction, the checked value is what lands in
// the process environment; a failed check exports today's value.
func TestVulkanDeviceOrderEnvCheckedExportsTheCheckedValue(t *testing.T) {
	loader := fakeVulkanLoader{devices: []fakeVulkanDevice{vkCPU, vkNV24, vkAMD, vkNV12}, driverLines: true}
	list := func(string) ([]listedDevice, error) { return loader.defaultListing(), nil }
	goldenPlanUnsetenv(t, pureEnvKeys...)
	before := sortedEnviron()
	env, note, err := vulkanDeviceOrderEnvChecked(&launchRequest{}, vulkanBackend(), twoNVIDIACaps(), list,
		func(path, visible string) (string, error) {
			if path != vulkanBackend().Path {
				t.Fatalf("listed %q", path)
			}
			return loader.output(visible, true), nil
		})
	if err != nil || env != "GGML_VK_VISIBLE_DEVICES=1,3" || !strings.Contains(note, "restricted") {
		t.Fatalf("env %q note %q err %v", env, note, err)
	}
	assertOnlyAssignment(t, "checked restriction", before, sortedEnviron(), "GGML_VK_VISIBLE_DEVICES=1,3")

	goldenPlanUnsetenv(t, pureEnvKeys...)
	before = sortedEnviron()
	env, note, err = vulkanDeviceOrderEnvChecked(&launchRequest{}, vulkanBackend(), twoNVIDIACaps(), list,
		func(string, string) (string, error) { return loader.output("", false), nil })
	if err != nil || env != "GGML_VK_VISIBLE_DEVICES=0,2" || !strings.Contains(note, "failed its check") {
		t.Fatalf("env %q note %q err %v", env, note, err)
	}
	assertOnlyAssignment(t, "failed check", before, sortedEnviron(), "GGML_VK_VISIBLE_DEVICES=0,2")
}

// Through planLaunch: on a single-vendor host, or without the opt-in, the
// dry run of a Vulkan backend is byte-identical and runs no extra listing.
// Beside another vendor's GPU it exports the checked restriction and changes
// nothing else; a failed check is today's plan plus one note. A CUDA backend
// is untouched either way.
func TestPlanLaunchVulkanRestrictionOnlyBesideAnotherVendor(t *testing.T) {
	launchOrderTestEnv(t)
	c := launchOrderCase{goldenPlanCase: goldenPlanCase{name: "mixed3/dense-small/vulkan-restriction", hw: "mixed3", model: "dense-small"}}
	launchOrderRunCase(t, c, func(t *testing.T) {
		nv := func(gb int, uuid string) fakeVulkanDevice {
			return fakeVulkanDevice{name: fmt.Sprintf("NVIDIA synthetic %d GB", gb), driver: "NVIDIA", totalMB: gb << 10, uuid: uuid, priority: 1}
		}
		loader := fakeVulkanLoader{devices: []fakeVulkanDevice{nv(12, "a"), nv(24, "b"), nv(12, "c"), vkAMD}, driverLines: true}
		inventory := func(withAMD bool) []detect.PCIDisplay {
			inv := []detect.PCIDisplay{
				{Address: "0000:01:00.0", Vendor: "nvidia", Driver: "nvidia"},
				{Address: "0000:02:00.0", Vendor: "nvidia", Driver: "nvidia"},
				{Address: "0000:03:00.0", Vendor: "nvidia", Driver: "nvidia"},
			}
			if withAMD {
				inv = append(inv, detect.PCIDisplay{Address: "0000:19:00.0", Vendor: "amd", Driver: "amdgpu"})
			}
			return inv
		}
		type outcome struct {
			plan    string
			prefix  string
			checked []string
			notes   []string
		}
		run := func(optIn bool, withAMD bool, checkPasses bool, cuda ...bool) outcome {
			t.Helper()
			goldenPlanUnsetenv(t, "GGRUN_EXPERIMENTAL_AMD", "GGML_VK_VISIBLE_DEVICES")
			if optIn {
				t.Setenv("GGRUN_EXPERIMENTAL_AMD", "1")
			}
			req, model := planLaunchFixture(t, c.hw, c.model)
			caps, err := planLaunchSilentInventory(req.InventoryPath)
			if err != nil {
				t.Fatal(err)
			}
			req.InventoryPath = ""
			be := goldenPlanBackend(t, "")
			onCUDA := len(cuda) > 0 && cuda[0]
			if !onCUDA {
				be.Tag, be.Dialect = "vulkan", "vulkan"
			}
			in := goldenPlanInputs(t, c.name, model, be, nil)
			in.Detect = func() (*detect.Capabilities, error) { return caps, nil }
			if !onCUDA {
				in.ListVulkan = func(string) ([]listedDevice, error) { return loader.defaultListing(), nil }
			}
			var o outcome
			in.HostDisplays = func() []detect.PCIDisplay {
				if onCUDA {
					t.Error("a CUDA plan read the PCI inventory")
				}
				return inventory(withAMD)
			}
			in.ListVulkanUnder = func(_, visible string) (string, error) {
				o.checked = append(o.checked, visible)
				if !checkPasses && visible == "0,1,2" {
					return loader.output("", false), nil
				}
				return loader.output(visible, true), nil
			}
			notes := planLaunchCollect(&in)
			plan, planErr := planLaunch(req, in)
			if planErr != nil || plan == nil || plan.Refusal != nil {
				t.Fatalf("plan failed: %v", planErr)
			}
			data, err := json.Marshal(goldenPlanJSONOf(plan))
			if err != nil {
				t.Fatal(err)
			}
			o.plan, o.prefix = string(data), plan.EnvPrefix
			for _, n := range *notes {
				o.notes = append(o.notes, n.Text)
			}
			return o
		}

		// The daily path: a CUDA backend beside an AMD GPU, opt-in or not,
		// plans byte for byte the same and reads nothing more.
		if cudaOn, cudaOff := run(true, true, true, true), run(false, true, true, true); cudaOn.plan != cudaOff.plan ||
			cudaOn.prefix != cudaOff.prefix || len(cudaOn.checked)+len(cudaOff.checked) != 0 ||
			strings.Join(cudaOn.notes, "") != strings.Join(cudaOff.notes, "") {
			t.Fatalf("CUDA plan changed with the opt-in:\n%s\n%s", cudaOn.plan, cudaOff.plan)
		}

		today := run(false, true, true)
		if today.prefix != "" || len(today.checked) != 0 {
			t.Fatalf("without the opt-in: prefix %q, checked %q", today.prefix, today.checked)
		}
		for _, other := range []outcome{run(false, false, true), run(true, false, true)} {
			if other.plan != today.plan || other.prefix != "" || len(other.checked) != 0 || strings.Join(other.notes, "") != strings.Join(today.notes, "") {
				t.Fatalf("single-vendor or opt-out plan differs from today's:\n%s\n%s\nchecked %q", other.plan, today.plan, other.checked)
			}
		}

		restricted := run(true, true, true)
		if restricted.prefix != "GGML_VK_VISIBLE_DEVICES=0,1,2" || !reflect.DeepEqual(restricted.checked, []string{vkRawProbe, "0,1,2,3", "0,1,2"}) {
			t.Fatalf("restricted: prefix %q, checked %q", restricted.prefix, restricted.checked)
		}
		if !strings.Contains(restricted.plan, `"GGML_VK_VISIBLE_DEVICES":"0,1,2"`) {
			t.Fatalf("restricted plan environment lacks the value: %s", restricted.plan)
		}
		if strings.Replace(restricted.plan, `"GGML_VK_VISIBLE_DEVICES":"0,1,2",`, "", 1) != today.plan &&
			strings.Replace(restricted.plan, `,"GGML_VK_VISIBLE_DEVICES":"0,1,2"`, "", 1) != today.plan {
			t.Fatalf("the restriction changed more than the environment:\n%s\n%s", restricted.plan, today.plan)
		}
		if !strings.Contains(strings.Join(restricted.notes, ""), "[dry-run] Vulkan backend restricted") {
			t.Fatalf("notes %q", restricted.notes)
		}

		failed := run(true, true, false)
		if failed.plan != today.plan || failed.prefix != "" || len(failed.checked) != 3 {
			t.Fatalf("failed check must keep today's plan: prefix %q checked %q\n%s", failed.prefix, failed.checked, failed.plan)
		}
		if !strings.Contains(strings.Join(failed.notes, ""), "failed its check") {
			t.Fatalf("failed check notes %q", failed.notes)
		}
	})
}
