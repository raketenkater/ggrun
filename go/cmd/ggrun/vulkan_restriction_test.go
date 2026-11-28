package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// The rig's NVIDIA cards (rigGPUs) beside an MI50, as the OS inventory lists
// them. The BMC display never counts as another GPU vendor.
func rigInventory(withAMD bool) []detect.PCIDisplay {
	inv := []detect.PCIDisplay{
		{Address: "0000:03:00.0", Vendor: "0x1a03", Driver: "ast"},
		{Address: "0000:17:00.0", Vendor: "nvidia", Driver: "nvidia"},
		{Address: "0000:65:00.0", Vendor: "nvidia", Driver: "nvidia"},
		{Address: "0000:b3:00.0", Vendor: "nvidia", Driver: "nvidia"},
	}
	if withAMD {
		inv = append(inv, detect.PCIDisplay{Address: "0000:19:00.0", Vendor: "amd", Driver: "amdgpu", VRAMTotalMB: 32752})
	}
	return inv
}

// The Vulkan build's listing with the MI50 present, in its own order.
const rigVulkanListWithMI50 = `Available devices:
  Vulkan0: NVIDIA GeForce RTX 3090 Ti (24810 MiB, 24321 MiB free)
  Vulkan1: AMD Instinct MI60 / MI50 (RADV VEGA20) (32752 MiB, 32700 MiB free)
  Vulkan2: NVIDIA GeForce RTX 4070 (12528 MiB, 12075 MiB free)
  Vulkan3: NVIDIA GeForce RTX 3060 (12534 MiB, 12125 MiB free)
`

// renumberedListing is what ggml-vulkan lists under GGML_VK_VISIBLE_DEVICES:
// the named devices, renumbered by position.
func renumberedListing(t *testing.T, full []listedDevice, visible string) []listedDevice {
	t.Helper()
	var out []listedDevice
	for pos, token := range strings.Split(visible, ",") {
		idx, err := strconv.Atoi(token)
		if err != nil || idx < 0 || idx >= len(full) {
			t.Fatalf("bad visible list %q", visible)
		}
		out = append(out, listedDevice{Index: pos, Name: full[idx].Name, TotalMB: full[idx].TotalMB})
	}
	return out
}

// An order that already matches ggrun's exported nothing, so the backend also
// opened the MI50. Beside another vendor's GPU the launch now exports its own
// cards, after the backend confirmed the value lists exactly them.
func TestRestrictedVulkanOrderExportsTheMatchingOrderAfterItsCheck(t *testing.T) {
	// Detected order 4070, 3090 Ti, 3060 with the 3090 Ti first in Vulkan:
	// today's value is 2,0,3 and the restriction keeps it, checked.
	full := parseVulkanDeviceList(rigVulkanListWithMI50)
	var checked []string
	listUnder := func(visible string) ([]listedDevice, error) {
		checked = append(checked, visible)
		return renumberedListing(t, full, visible), nil
	}
	key, value, note, err := restrictedVulkanDeviceOrder(&launchRequest{}, vulkanBackend(), rigGPUs(), full, listUnder)
	if err != nil || key != "GGML_VK_VISIBLE_DEVICES" || value != "2,0,3" || !strings.Contains(note, "restricted") {
		t.Fatalf("got %q=%q note %q err %v", key, value, note, err)
	}
	// The same cards already in ggrun's order: today exports nothing.
	same := []listedDevice{
		{Index: 0, Name: "NVIDIA GeForce RTX 4070", TotalMB: 12528},
		{Index: 1, Name: "NVIDIA GeForce RTX 3090 Ti", TotalMB: 24810},
		{Index: 2, Name: "NVIDIA GeForce RTX 3060", TotalMB: 12534},
		{Index: 3, Name: "AMD Instinct MI60 / MI50 (RADV VEGA20)", TotalMB: 32752},
	}
	if k, v, _ := vulkanDeviceOrder(&launchRequest{}, vulkanBackend(), rigGPUs(), same); k != "" || v != "" {
		t.Fatalf("today's order must export nothing for an identity order, got %q=%q", k, v)
	}
	checked = nil
	key, value, _, err = restrictedVulkanDeviceOrder(&launchRequest{}, vulkanBackend(), rigGPUs(), same, func(visible string) ([]listedDevice, error) {
		checked = append(checked, visible)
		return renumberedListing(t, same, visible), nil
	})
	if err != nil || key != "GGML_VK_VISIBLE_DEVICES" || value != "0,1,2" || len(checked) != 1 || checked[0] != "0,1,2" {
		t.Fatalf("identity order: got %q=%q err %v, checked %q", key, value, err, checked)
	}
	// --gpus is honoured the same way.
	key, value, _, err = restrictedVulkanDeviceOrder(&launchRequest{GPUsFlag: "1"}, vulkanBackend(), rigGPUs(), same, func(visible string) ([]listedDevice, error) {
		return renumberedListing(t, same, visible), nil
	})
	if err != nil || key != "GGML_VK_VISIBLE_DEVICES" || value != "1" {
		t.Fatalf("--gpus 1: got %q=%q err %v", key, value, err)
	}
}

// When the check fails, the launch keeps today's value and says so.
func TestRestrictedVulkanOrderFallsBackToTodayOnAMismatch(t *testing.T) {
	same := []listedDevice{
		{Index: 0, Name: "NVIDIA GeForce RTX 4070", TotalMB: 12528},
		{Index: 1, Name: "NVIDIA GeForce RTX 3090 Ti", TotalMB: 24810},
		{Index: 2, Name: "NVIDIA GeForce RTX 3060", TotalMB: 12534},
		{Index: 3, Name: "AMD Instinct MI60 / MI50 (RADV VEGA20)", TotalMB: 32752},
	}
	for _, c := range []struct {
		name      string
		listUnder func(string) ([]listedDevice, error)
	}{
		{"variable ignored", func(string) ([]listedDevice, error) { return same, nil }},
		{"wrong card", func(string) ([]listedDevice, error) {
			return []listedDevice{same[0], same[3], same[2]}, nil
		}},
		{"listing failed", func(string) ([]listedDevice, error) { return nil, errors.New("no Vulkan devices") }},
	} {
		key, value, note, err := restrictedVulkanDeviceOrder(&launchRequest{}, vulkanBackend(), rigGPUs(), same, c.listUnder)
		if err != nil || key != "" || value != "" {
			t.Fatalf("%s: got %q=%q err %v; want today's empty value", c.name, key, value, err)
		}
		if !strings.Contains(note, "failed its check") || !strings.HasSuffix(note, "keeping no restriction") {
			t.Fatalf("%s: note %q", c.name, note)
		}
	}
	// A non-identity value stays exported when its check fails, as today.
	full := parseVulkanDeviceList(rigVulkanListWithMI50)
	key, value, note, err := restrictedVulkanDeviceOrder(&launchRequest{}, vulkanBackend(), rigGPUs(), full,
		func(string) ([]listedDevice, error) { return full, nil })
	if err != nil || key != "GGML_VK_VISIBLE_DEVICES" || value != "2,0,3" || !strings.HasSuffix(note, "keeping GGML_VK_VISIBLE_DEVICES=2,0,3") {
		t.Fatalf("got %q=%q note %q err %v", key, value, note, err)
	}
}

// Today's mapping error stays an error and nothing else is listed.
func TestRestrictedVulkanOrderKeepsTodaysError(t *testing.T) {
	other := parseVulkanDeviceList("  Vulkan0: AMD Radeon RX 7800 XT (16368 MiB, 1 MiB free)\n  Vulkan1: NVIDIA GeForce RTX 4070 (12528 MiB, 1 MiB free)\n  Vulkan2: NVIDIA GeForce RTX 3060 (12534 MiB, 1 MiB free)\n")
	_, _, _, err := restrictedVulkanDeviceOrder(&launchRequest{}, vulkanBackend(), rigGPUs(), other, func(string) ([]listedDevice, error) {
		t.Fatal("listed again after the mapping failed")
		return nil, nil
	})
	if err == nil || !strings.Contains(err.Error(), "3090 Ti") {
		t.Fatalf("err = %v", err)
	}
}

// The rule: opt-in, Linux, a Vulkan backend on the NVIDIA cards, and a GPU of
// another vendor in the OS inventory. Anything else reads nothing more.
func TestVulkanRestrictionAppliesOnlyBesideAnotherVendor(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the opt-in is Linux only")
	}
	read := 0
	inventory := func(withAMD bool) func() []detect.PCIDisplay {
		return func() []detect.PCIDisplay { read++; return rigInventory(withAMD) }
	}
	t.Setenv("GGRUN_EXPERIMENTAL_AMD", "")
	if vulkanRestrictionApplies(vulkanBackend(), rigGPUs(), inventory(true)) || read != 0 {
		t.Fatalf("without the opt-in: applied or read the inventory (%d reads)", read)
	}
	t.Setenv("GGRUN_EXPERIMENTAL_AMD", "1")
	if !vulkanRestrictionApplies(vulkanBackend(), rigGPUs(), inventory(true)) {
		t.Fatal("a Vulkan backend on NVIDIA cards beside an MI50 must be restricted")
	}
	if vulkanRestrictionApplies(vulkanBackend(), rigGPUs(), inventory(false)) {
		t.Fatal("a single-vendor host must keep today's path")
	}
	read = 0
	cuda := &backendInfo{Path: "/x/llama-server", Dialect: "llama"}
	if vulkanRestrictionApplies(cuda, rigGPUs(), inventory(true)) || read != 0 {
		t.Fatal("a CUDA backend is never restricted and reads no inventory")
	}
	if vulkanRestrictionApplies(vulkanBackend(), rigGPUs(), nil) {
		t.Fatal("no inventory reader means no restriction")
	}
}

// The listing under a value really runs with that value, and the unfiltered
// listing never inherits the process's own.
func TestVulkanListingsRunUnderTheirOwnVisibility(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake backends are shell scripts")
	}
	bin := filepath.Join(t.TempDir(), "llama-server-vulkan")
	script := "#!/bin/sh\necho 'Available devices:'\necho \"  Vulkan0: seen [$GGML_VK_VISIBLE_DEVICES] (1000 MiB, 1 MiB free)\"\n"
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GGML_VK_VISIBLE_DEVICES", "7")
	all, err := listVulkanDevices(bin)
	if err != nil || len(all) != 1 || all[0].Name != "seen []" {
		t.Fatalf("unfiltered listing = %+v, %v", all, err)
	}
	under, err := listVulkanDevicesUnder(bin, "2,0")
	if err != nil || len(under) != 1 || under[0].Name != "seen [2,0]" {
		t.Fatalf("listing under 2,0 = %+v, %v", under, err)
	}
	if os.Getenv("GGML_VK_VISIBLE_DEVICES") != "7" {
		t.Fatal("a listing changed this process's environment")
	}
}

// Through planLaunch: on a single-vendor host, or without the opt-in, the
// dry run of a Vulkan backend is byte-identical and runs no extra listing.
// Beside another vendor's GPU it exports the checked restriction and changes
// nothing else; a failed check is today's plan plus one note.
func TestPlanLaunchVulkanRestrictionOnlyBesideAnotherVendor(t *testing.T) {
	launchOrderTestEnv(t)
	c := launchOrderCase{goldenPlanCase: goldenPlanCase{name: "mixed3/dense-small/vulkan-restriction", hw: "mixed3", model: "dense-small"}}
	launchOrderRunCase(t, c, func(t *testing.T) {
		full := []listedDevice{
			{Index: 0, Name: "NVIDIA synthetic 12 GB", TotalMB: 12288},
			{Index: 1, Name: "NVIDIA synthetic 24 GB", TotalMB: 24576},
			{Index: 2, Name: "NVIDIA synthetic 12 GB", TotalMB: 12288},
			{Index: 3, Name: "AMD Instinct MI60 / MI50 (RADV VEGA20)", TotalMB: 32752},
		}
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
				in.ListVulkan = func(string) ([]listedDevice, error) { return full, nil }
			}
			var o outcome
			in.HostDisplays = func() []detect.PCIDisplay {
				if onCUDA {
					t.Error("a CUDA plan read the PCI inventory")
				}
				return inventory(withAMD)
			}
			in.ListVulkanUnder = func(_, visible string) ([]listedDevice, error) {
				o.checked = append(o.checked, visible)
				if !checkPasses {
					return full, nil
				}
				return renumberedListing(t, full, visible), nil
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
		if restricted.prefix != "GGML_VK_VISIBLE_DEVICES=0,1,2" || len(restricted.checked) != 1 || restricted.checked[0] != "0,1,2" {
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
		if failed.plan != today.plan || failed.prefix != "" || len(failed.checked) != 1 {
			t.Fatalf("failed check must keep today's plan: prefix %q checked %q\n%s", failed.prefix, failed.checked, failed.plan)
		}
		if !strings.Contains(strings.Join(failed.notes, ""), "failed its check") {
			t.Fatalf("failed check notes %q", failed.notes)
		}
	})
}

// The launch's export branch: with the restriction, the checked value is what
// lands in the process environment, and the note says so.
func TestVulkanDeviceOrderEnvCheckedExportsTheCheckedValue(t *testing.T) {
	goldenPlanUnsetenv(t, pureEnvKeys...)
	same := []listedDevice{
		{Index: 0, Name: "NVIDIA GeForce RTX 4070", TotalMB: 12528},
		{Index: 1, Name: "NVIDIA GeForce RTX 3090 Ti", TotalMB: 24810},
		{Index: 2, Name: "NVIDIA GeForce RTX 3060", TotalMB: 12534},
		{Index: 3, Name: "AMD Instinct MI60 / MI50 (RADV VEGA20)", TotalMB: 32752},
	}
	before := sortedEnviron()
	env, note, err := vulkanDeviceOrderEnvChecked(&launchRequest{}, vulkanBackend(), rigGPUs(),
		func(string) ([]listedDevice, error) { return same, nil },
		func(path, visible string) ([]listedDevice, error) {
			if path != vulkanBackend().Path {
				t.Fatalf("listed %q", path)
			}
			return renumberedListing(t, same, visible), nil
		})
	if err != nil || env != "GGML_VK_VISIBLE_DEVICES=0,1,2" || !strings.Contains(note, "restricted") {
		t.Fatalf("env %q note %q err %v", env, note, err)
	}
	assertOnlyAssignment(t, "checked restriction", before, sortedEnviron(), "GGML_VK_VISIBLE_DEVICES=0,1,2")

	// A failed check exports today's value: nothing, for this order.
	goldenPlanUnsetenv(t, pureEnvKeys...)
	before = sortedEnviron()
	env, note, err = vulkanDeviceOrderEnvChecked(&launchRequest{}, vulkanBackend(), rigGPUs(),
		func(string) ([]listedDevice, error) { return same, nil },
		func(string, string) ([]listedDevice, error) { return same, nil })
	if err != nil || env != "" || !strings.Contains(note, "failed its check") {
		t.Fatalf("env %q note %q err %v", env, note, err)
	}
	assertOnlyAssignment(t, "failed check", before, sortedEnviron(), "")
}
