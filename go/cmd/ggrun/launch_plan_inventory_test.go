package main

import (
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// An inventory dry run plans for the machine the file describes: planLaunch
// must not detect this host's hardware, read its live VRAM or resident
// reviewer, or run the backend to list Vulkan devices. These tests stand in
// for this host with readers that fail when an inventory plan calls them, and
// check that a plan without an inventory still calls each one.

// inventoryTestReserved is a 24 GB card whose inventory withholds 4096 MiB
// from planning (vram_reserved_mb): the driver on that machine would still
// report it free, so the ik --fit binding must tighten the backend's margin.
var inventoryTestReserved = goldenPlanInventory(strings.Replace(goldenPlanGPU(0, 24576, 16, 1008000),
	`"vram_used_mb":300`, `"vram_used_mb":300,"vram_reserved_mb":4096`, 1))

// inventoryTestFixture is planLaunchFixture for an inventory given as text.
func inventoryTestFixture(t *testing.T, inventory, modelName string, args ...string) (*launchRequest, *placement.ModelProfile) {
	t.Helper()
	t.Setenv("LLM_CACHE_DIR", t.TempDir())
	path := filepath.Join(t.TempDir(), "inventory.json")
	if err := os.WriteFile(path, []byte(inventory), 0o600); err != nil {
		t.Fatal(err)
	}
	model := goldenPlanModels[modelName]()
	full := append([]string{model.Path}, args...)
	full = append(full, "--inventory", path, "--emit-server-argv-json")
	req, err := parseLaunchArgs(full)
	if err != nil {
		t.Fatalf("parse %v: %v", full, err)
	}
	return req, model
}

// inventoryTestHost is this host as the tests describe it: its hardware is
// the same inventory text, its live readers return the given values, and
// every call is counted. fatal makes each call fail the test.
type inventoryTestHost struct {
	inventory      string
	fatal          bool
	driverFreeMB   func(runtime *detect.Capabilities) map[int]int
	residentMB     int
	listing        []listedDevice
	detect, list   int
	used, free     int
	residentCalled int
	nanoCapable    bool
	nano           int
}

// install replaces the live readers for the rest of t.
func (h *inventoryTestHost) install(t *testing.T) {
	t.Helper()
	savedUsed, savedFree, savedResident, savedNano := runtimeVRAMUsedMB, fitDriverFreeVRAM, residentReviewerVRAM, claudeNanoGPUCapable
	t.Cleanup(func() {
		runtimeVRAMUsedMB, fitDriverFreeVRAM, residentReviewerVRAM, claudeNanoGPUCapable = savedUsed, savedFree, savedResident, savedNano
	})
	runtimeVRAMUsedMB = func(int) int {
		h.used++
		if h.fatal {
			t.Error("an inventory plan read this host's used VRAM")
		}
		return 0
	}
	fitDriverFreeVRAM = func(runtime *detect.Capabilities) map[int]int {
		h.free++
		if h.fatal {
			t.Error("an inventory plan read this host's driver free VRAM")
		}
		if h.driverFreeMB == nil {
			return nil
		}
		return h.driverFreeMB(runtime)
	}
	residentReviewerVRAM = func(string) int {
		h.residentCalled++
		if h.fatal {
			t.Error("an inventory plan scanned this host for a resident reviewer")
		}
		return h.residentMB
	}
	claudeNanoGPUCapable = func(string) bool {
		h.nano++
		if h.fatal {
			t.Error("an inventory plan ran the Nano reviewer backend to list devices")
		}
		return h.nanoCapable
	}
}

// inputs are goldenPlanInputs with this host's Detect and Vulkan listing.
func (h *inventoryTestHost) inputs(t *testing.T, name string, model *placement.ModelProfile, be *backendInfo) planInputs {
	t.Helper()
	in := goldenPlanInputs(t, name, model, be, nil)
	in.Inventory = planLaunchSilentInventory
	in.Detect = func() (*detect.Capabilities, error) {
		h.detect++
		if h.fatal {
			t.Error("an inventory plan detected this host's hardware")
		}
		return detect.LoadCapabilities([]byte(h.inventory))
	}
	in.ListVulkan = func(string) ([]listedDevice, error) {
		h.list++
		if h.fatal {
			t.Error("an inventory plan ran the backend to list Vulkan devices")
		}
		return h.listing, nil
	}
	return in
}

func inventoryTestVulkanBackend(t *testing.T) *backendInfo {
	be := goldenPlanBackend(t, "")
	be.Dialect = "vulkan"
	return be
}

// argValue is the value after flag in args, "" when absent.
func inventoryTestArgValue(args []string, flag string) string {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag {
			return args[i+1]
		}
	}
	return ""
}

// (1) With an inventory, no host reader runs: mainline, an ik --fit plan that
// reaches the --fit-margin binding, --claude-code, and a Vulkan backend.
func TestInventoryPlanTouchesNoHostReader(t *testing.T) {
	launchOrderTestEnv(t)
	for _, tc := range []struct {
		name, inventory, model string
		backend                func(t *testing.T) *backendInfo
		args                   []string
		check                  func(t *testing.T, plan *launchPlan, notes []planNote)
	}{
		{name: "mainline/gpu12/dense-small", inventory: goldenPlanInventories["gpu12"], model: "dense-small",
			backend: func(t *testing.T) *backendInfo { return goldenPlanBackend(t, "") }},
		{name: "ik/reserved24/dense-large", inventory: inventoryTestReserved, model: "dense-large",
			backend: func(t *testing.T) *backendInfo { return goldenPlanBackend(t, "ik") },
			check: func(t *testing.T, plan *launchPlan, _ []planNote) {
				if !backendFitsItself(plan.ServerArgv) {
					t.Fatalf("the ik case does not reach the --fit binding: %v", plan.ServerArgv)
				}
			}},
		{name: "ik/gpu12/dense-small/claude-code", inventory: goldenPlanInventories["gpu12"], model: "dense-small",
			backend: func(t *testing.T) *backendInfo { return goldenPlanBackend(t, "ik") }, args: []string{"--claude-code"},
			check: func(t *testing.T, plan *launchPlan, _ []planNote) {
				if plan.Req.ReviewerReservation == nil {
					t.Fatal("the --claude-code case plans no reviewer reservation")
				}
			}},
		{name: "vulkan/mixed3/dense-small", inventory: goldenPlanInventories["mixed3"], model: "dense-small",
			backend: inventoryTestVulkanBackend,
			check: func(t *testing.T, plan *launchPlan, notes []planNote) {
				want := planNote{Stream: planStderr, Text: "[dry-run] Vulkan device order taken from the inventory; not checked against the backend's own listing\n"}
				found := false
				for _, n := range notes {
					found = found || n == want
				}
				if !found {
					t.Fatalf("notes %+v lack %+v", notes, want)
				}
			}},
	} {
		c := launchOrderCase{goldenPlanCase: goldenPlanCase{name: tc.name}}
		launchOrderRunCase(t, c, func(t *testing.T) {
			host := &inventoryTestHost{inventory: tc.inventory, fatal: true}
			host.install(t)
			req, model := inventoryTestFixture(t, tc.inventory, tc.model, tc.args...)
			in := host.inputs(t, tc.name, model, tc.backend(t))
			notes := planLaunchCollect(&in)
			plan, planErr := planLaunch(req, in)
			if planErr != nil || plan == nil || plan.Refusal != nil {
				t.Fatalf("plan %+v, error %+v; want a plan", plan, planErr)
			}
			if host.detect+host.list+host.used+host.free+host.residentCalled != 0 {
				t.Fatalf("host readers called: %+v", host)
			}
			if tc.check != nil {
				tc.check(t, plan, *notes)
			}
		})
	}
}

// (2) and (3) The values an inventory plan uses come from the inventory, and
// the same request without one still reads the host, once per reader as before.
func TestInventoryPlanDerivesLiveReadingsFromTheInventory(t *testing.T) {
	launchOrderTestEnv(t)

	t.Run("fit-margin", func(t *testing.T) {
		launchOrderRunCase(t, launchOrderCase{goldenPlanCase: goldenPlanCase{name: "ik/reserved24/dense-large"}}, func(t *testing.T) {
			// This host's driver reports 2048 MiB less free than the inventory's.
			host := &inventoryTestHost{inventory: inventoryTestReserved, driverFreeMB: func(runtime *detect.Capabilities) map[int]int {
				free := map[int]int{}
				for _, g := range runtime.GPUs {
					free[g.Index] = g.VRAMTotalMB - 300 - 2048
				}
				return free
			}}
			host.install(t)
			plan := func(withInventory bool) *launchPlan {
				req, model := inventoryTestFixture(t, inventoryTestReserved, "dense-large")
				if !withInventory {
					req.InventoryPath = ""
				}
				p, planErr := planLaunch(req, host.inputs(t, "fit-margin", model, goldenPlanBackend(t, "ik")))
				if planErr != nil || p == nil || p.Refusal != nil {
					t.Fatalf("plan %+v, error %+v; want a plan", p, planErr)
				}
				return p
			}

			inv := plan(true)
			if host.detect+host.used+host.free != 0 {
				t.Fatalf("the inventory plan read the host: %+v", host)
			}
			gpu := inv.Caps.GPUs[0]
			limits := allocationGuardLimitsMB(inv.Config.CacheDir, inv.Caps)
			gap := gpu.VRAMTotalMB - gpu.VRAMUsedMB - limits[0]
			if gap <= ikDefaultFitMarginMB {
				t.Fatalf("fixture gap %d MiB does not exceed ik's default margin; it cannot show the binding", gap)
			}
			want := strconv.Itoa((gap + fitMarginStepMB - 1) / fitMarginStepMB * fitMarginStepMB)
			if got := inventoryTestArgValue(inv.ServerArgv, fitMarginFlag); got != want {
				t.Fatalf("inventory plan %s = %q, want %q from the inventory's total - used (argv %v)", fitMarginFlag, got, want, inv.ServerArgv)
			}

			live := plan(false)
			if host.detect != 1 || host.used == 0 || host.free == 0 {
				t.Fatalf("the plan without an inventory did not read the host as before: %+v", host)
			}
			if got := inventoryTestArgValue(live.ServerArgv, fitMarginFlag); got == "" || got == want {
				t.Fatalf("host plan %s = %q; the fixture must tell the host's reading from the inventory's %q", fitMarginFlag, got, want)
			}
		})
	})

	t.Run("reviewer-reservation", func(t *testing.T) {
		launchOrderRunCase(t, launchOrderCase{goldenPlanCase: goldenPlanCase{name: "gpu12/dense-small/claude-code"}}, func(t *testing.T) {
			host := &inventoryTestHost{inventory: goldenPlanInventories["gpu12"], residentMB: 1000}
			host.install(t)
			reservation := func(withInventory bool) int {
				req, model := inventoryTestFixture(t, goldenPlanInventories["gpu12"], "dense-small", "--claude-code")
				if !withInventory {
					req.InventoryPath = ""
				}
				p, planErr := planLaunch(req, host.inputs(t, "reviewer", model, goldenPlanBackend(t, "")))
				if planErr != nil || p == nil {
					t.Fatalf("plan %+v, error %+v; want a plan", p, planErr)
				}
				if p.Req.ReviewerReservation == nil {
					return 0
				}
				return p.Req.ReviewerReservation.VRAMMB
			}
			inv := reservation(true)
			if host.residentCalled != 0 || host.detect != 0 {
				t.Fatalf("the inventory plan read the host: %+v", host)
			}
			if inv <= host.residentMB {
				t.Fatalf("inventory reservation %d MiB; the fixture needs more than the host's resident %d", inv, host.residentMB)
			}
			live := reservation(false)
			if host.residentCalled == 0 || host.detect != 1 {
				t.Fatalf("the plan without an inventory did not read the host as before: %+v", host)
			}
			if live != inv-host.residentMB {
				t.Fatalf("host reservation %d MiB, want %d - %d resident", live, inv, host.residentMB)
			}
		})
	})

	t.Run("nano-reviewer", func(t *testing.T) {
		launchOrderRunCase(t, launchOrderCase{goldenPlanCase: goldenPlanCase{name: "gpu24/dense-small/claude-reviewer-nanbeige"}}, func(t *testing.T) {
			// The Nano artifact and its reviewed backend are installed; this
			// host's reviewer backend lists no CUDA device.
			savedReady, savedBackend := claudeNanoArtifactReady, claudeNanoBackend
			claudeNanoArtifactReady = func(string) bool { return true }
			claudeNanoBackend = func() (string, error) { return "/usr/bin/ggrun-golden-nano-llama-server", nil }
			t.Cleanup(func() { claudeNanoArtifactReady, claudeNanoBackend = savedReady, savedBackend })
			host := &inventoryTestHost{inventory: goldenPlanInventories["gpu24"]}
			host.install(t)
			reviewer := func(withInventory bool) string {
				req, model := inventoryTestFixture(t, goldenPlanInventories["gpu24"], "dense-small", "--claude-code", "--claude-reviewer", "nanbeige")
				if !withInventory {
					req.InventoryPath = ""
				}
				p, planErr := planLaunch(req, host.inputs(t, "nano", model, goldenPlanBackend(t, "")))
				if planErr != nil || p == nil || p.Req.ReviewerReservation == nil {
					t.Fatalf("plan %+v, error %+v; want a plan with a reviewer seat", p, planErr)
				}
				return p.Req.ReviewerReservation.Name
			}
			if got := reviewer(true); got != claudeNanoCompanionName || host.nano != 0 {
				t.Fatalf("inventory plan seats %q after %d device listings, want %q from the inventory's GPUs and none", got, host.nano, claudeNanoCompanionName)
			}
			if got := reviewer(false); got == claudeNanoCompanionName || host.nano == 0 {
				t.Fatalf("host plan seats %q after %d device listings; the host's listing must decide", got, host.nano)
			}
		})
	})

	t.Run("vulkan-order", func(t *testing.T) {
		launchOrderRunCase(t, launchOrderCase{goldenPlanCase: goldenPlanCase{name: "vulkan/mixed3/dense-small"}}, func(t *testing.T) {
			// This host's backend enumerates the 24 GB card first.
			host := &inventoryTestHost{inventory: goldenPlanInventories["mixed3"], listing: []listedDevice{
				{Index: 0, Name: "NVIDIA synthetic 24 GB", TotalMB: 24576},
				{Index: 1, Name: "NVIDIA synthetic 12 GB", TotalMB: 12288},
				{Index: 2, Name: "NVIDIA synthetic 12 GB", TotalMB: 12288},
			}}
			host.install(t)
			plan := func(withInventory bool) *launchPlan {
				req, model := inventoryTestFixture(t, goldenPlanInventories["mixed3"], "dense-small")
				if !withInventory {
					req.InventoryPath = ""
				}
				p, planErr := planLaunch(req, host.inputs(t, "vulkan", model, inventoryTestVulkanBackend(t)))
				if planErr != nil || p == nil || p.Refusal != nil {
					t.Fatalf("plan %+v, error %+v; want a plan", p, planErr)
				}
				return p
			}
			inv := plan(true)
			if host.list != 0 {
				t.Fatalf("the inventory plan listed the backend's devices: %+v", host)
			}
			if strings.Contains(inv.EnvPrefix, "GGML_VK_VISIBLE_DEVICES") || inv.Environment["GGML_VK_VISIBLE_DEVICES"] != "" {
				t.Fatalf("inventory order is the identity, yet the plan reorders: %q %v", inv.EnvPrefix, inv.Environment)
			}
			live := plan(false)
			if host.list != 1 {
				t.Fatalf("the plan without an inventory listed devices %d times, want 1", host.list)
			}
			if live.EnvPrefix != "GGML_VK_VISIBLE_DEVICES=1,0,2" {
				t.Fatalf("host plan env prefix %q, want the host listing's order 1,0,2", live.EnvPrefix)
			}
		})
	})
}

// (4) An inventory plan leaves the package readers as it found them, after a
// plan and after an error return.
func TestInventoryPlanRestoresTheLiveReaders(t *testing.T) {
	launchOrderTestEnv(t)
	launchOrderRunCase(t, launchOrderCase{goldenPlanCase: goldenPlanCase{name: "gpu12/dense-small/restore"}}, func(t *testing.T) {
		host := &inventoryTestHost{inventory: goldenPlanInventories["gpu12"], residentMB: 7, nanoCapable: true}
		host.install(t)
		check := func(step string) {
			t.Helper()
			before := *host
			runtimeVRAMUsedMB(0)
			fitDriverFreeVRAM(&detect.Capabilities{GPUs: []detect.GPU{{Index: 0}}})
			if residentReviewerVRAM("marker") != 7 || !claudeNanoGPUCapable("backend") || host.used != before.used+1 || host.free != before.free+1 || host.residentCalled != before.residentCalled+1 || host.nano != before.nano+1 {
				t.Fatalf("%s: the live readers are not the ones installed before the plan", step)
			}
		}

		req, model := inventoryTestFixture(t, goldenPlanInventories["gpu12"], "dense-small", "--claude-code")
		in := host.inputs(t, "restore", model, goldenPlanBackend(t, "ik"))
		if plan, planErr := planLaunch(req, in); planErr != nil || plan == nil {
			t.Fatalf("plan %+v, error %+v; want a plan", plan, planErr)
		}
		check("after a plan")

		req, model = inventoryTestFixture(t, goldenPlanInventories["gpu12"], "dense-small")
		in = host.inputs(t, "restore", model, goldenPlanBackend(t, ""))
		in.Model = func(string) (*placement.ModelProfile, error) {
			// The inventory readers are in place by now.
			if runtimeVRAMUsedMB(0) != 300 || residentReviewerVRAM("marker") != 0 {
				t.Error("the inventory readers are not installed while planning")
			}
			return nil, errors.New("not a gguf")
		}
		plan, planErr := planLaunch(req, in)
		if plan != nil || planErr == nil || planErr.Message != "Error parsing model: not a gguf" {
			t.Fatalf("plan %+v, error %+v; want the model error", plan, planErr)
		}
		check("after an error return")
		if host.detect != 0 {
			t.Fatalf("Detect ran %d times on inventory plans", host.detect)
		}
	})
}
