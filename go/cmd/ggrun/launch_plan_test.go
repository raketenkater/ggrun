package main

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// planLaunchFixture writes a case's inventory into a fresh cache and parses
// the dry run's arguments the way the golden replays do.
func planLaunchFixture(t *testing.T, hw, modelName string, args ...string) (*launchRequest, *placement.ModelProfile) {
	t.Helper()
	t.Setenv("LLM_CACHE_DIR", t.TempDir())
	inventory, ok := goldenPlanInventories[hw]
	if !ok {
		inventory = launchOrderInventories[hw]
	}
	if inventory == "" {
		t.Fatalf("no inventory %q", hw)
	}
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

// planLaunchSilentInventory reads an inventory like loadInventory without its
// "[inventory]" line, so a test sees only what planLaunch and its callees print.
func planLaunchSilentInventory(path string) (*detect.Capabilities, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return detect.LoadCapabilities(data)
}

func planLaunchCollect(in *planInputs) *[]planNote {
	notes := &[]planNote{}
	in.Note = func(n planNote) { *notes = append(*notes, n) }
	return notes
}

func planLaunchExpectError(t *testing.T, plan *launchPlan, planErr *planError, code int, message string) {
	t.Helper()
	if plan != nil || planErr == nil {
		t.Fatalf("plan %v, error %v; want only an error", plan, planErr)
	}
	if planErr.Code != code || planErr.Message != message {
		t.Fatalf("error = exit %d %q, want exit %d %q", planErr.Code, planErr.Message, code, message)
	}
	if n := planErr.note(); n.Stream != planStderr || n.Text != message+"\n" {
		t.Fatalf("error prints %+v, want %q on stderr", n, message+"\n")
	}
}

// The "Model does not fit" placement error is not in the console golden: a
// dry run without an inventory appends placementErrorMessage's /proc scan for
// running llama-server processes, which another test of the same `go test
// ./...` can start. An inventory plan does not scan this host, so here the
// message is pinned exactly, without the hint.
func TestPlanLaunchReturnsModelDoesNotFitAsAValue(t *testing.T) {
	launchOrderTestEnv(t)
	c := launchOrderCase{goldenPlanCase: goldenPlanCase{name: "gpu12-ram32/moe/no-mmap", hw: "gpu12-ram32", model: "moe", args: []string{"--no-mmap"}}}
	launchOrderRunCase(t, c, func(t *testing.T) {
		req, model := planLaunchFixture(t, c.hw, c.model, c.args...)
		in := goldenPlanInputs(t, c.name, model, goldenPlanBackend(t, ""), nil)
		in.Inventory = planLaunchSilentInventory
		notes := planLaunchCollect(&in)
		plan, planErr := planLaunch(req, in)
		if plan != nil || planErr == nil {
			t.Fatalf("plan %v, error %v; want a placement error", plan, planErr)
		}
		const cause = "automatic context fit found no admissible plan: Model does not fit on this system: no GPU has free VRAM after CUDA/compute overhead"
		if planErr.Err == nil || planErr.Err.Error() != cause {
			t.Fatalf("cause = %v, want %q", planErr.Err, cause)
		}
		if !goldenPlanIsPlacementError(planErr) || planErr.Code != 1 {
			t.Fatalf("error = exit %d %q, want a placement error with exit 1", planErr.Code, planErr.Message)
		}
		if want := "Error computing placement: " + cause; planErr.Message != want {
			t.Fatalf("message = %q, want %q without this host's running-server hint", planErr.Message, want)
		}
		if n := planErr.note(); n.Stream != planStderr || n.Text != planErr.Message+"\n" {
			t.Fatalf("error prints %+v, want the message on stderr", n)
		}
		if len(*notes) != 0 {
			t.Fatalf("notes before the error: %+v", *notes)
		}
	})
}

// The dry run's fit refusal is not in the console golden and no fixture
// reaches it through planLaunch: placement.Compute refuses an automatic-context
// dense offload whose GPU KV does not fit (computeResolvedStrategy), and with
// an explicit context the estimated ledger of every fixture tried carries no
// KV on a GPU. A measured allocation in the cache may reach it; that is not
// verified. The helper planLaunch calls for it is pinned here; the branch in
// planLaunch and cmdDryRun that carries it is not exercised.
func TestPlanFitRefusalIsExitOneOnStderr(t *testing.T) {
	over := &placement.Strategy{Type: placement.DenseCPUOffload, ResourceLedger: &placement.ResourceLedger{Devices: []placement.DeviceResourceLedger{
		{GPU: 0, FreeMB: 1587, ContextMB: 4000, SlackMB: -2758},
		{GPU: 1, FreeMB: 3040, ContextMB: 0, SlackMB: -40},
	}}}
	refusal := planFitRefusal(over)
	const want = "[dry-run] plan does not fit as estimated: CUDA0 KV 4000 MiB, slack -2758 MiB. Lower --ctx-size or use --kv-placement cpu."
	if refusal == nil || refusal.Code != 1 || refusal.Message != want || refusal.Err != nil {
		t.Fatalf("refusal = %+v, want exit 1 %q", refusal, want)
	}
	if n := refusal.note(); n.Stream != planStderr || n.Text != want+"\n" {
		t.Fatalf("refusal prints %+v, want %q on stderr", n, want+"\n")
	}
	fits := &placement.Strategy{Type: placement.DenseCPUOffload, ResourceLedger: &placement.ResourceLedger{Devices: []placement.DeviceResourceLedger{
		{GPU: 0, ContextMB: 4000, SlackMB: 0},
	}}}
	for name, s := range map[string]*placement.Strategy{"nil": nil, "fits": fits, "no ledger": {Type: placement.DenseCPUOffload}} {
		if got := planFitRefusal(s); got != nil {
			t.Fatalf("%s: refusal %+v", name, got)
		}
	}
	if !dryRunPlanPolicy.FitRefusal {
		t.Fatal("the dry run's policy does not check the fit refusal")
	}
}

// Every error that ends a dry run before placement, as a value: exit code,
// exact message and stream, and the order in which they win.
func TestPlanLaunchErrorsAreValuesInTodaysOrder(t *testing.T) {
	launchOrderTestEnv(t)
	goldenPlanUnsetenv(t, "LLAMA_API_KEY")
	c := launchOrderCase{goldenPlanCase: goldenPlanCase{name: "gpu12/dense-small/errors", hw: "gpu12", model: "dense-small"}}
	launchOrderRunCase(t, c, func(t *testing.T) {
		inputs := func(req *launchRequest, model *placement.ModelProfile) (planInputs, *[]planNote) {
			in := goldenPlanInputs(t, c.name, model, goldenPlanBackend(t, ""), nil)
			in.Inventory = planLaunchSilentInventory
			notes := planLaunchCollect(&in)
			return in, notes
		}

		// Without an inventory, Detect failing ends the plan first; the
		// exposure warning is already out.
		req, model := planLaunchFixture(t, c.hw, c.model, "--host", "0.0.0.0")
		req.InventoryPath = ""
		in, notes := inputs(req, model)
		in.Detect = func() (*detect.Capabilities, error) { return nil, errors.New("no probe") }
		plan, planErr := planLaunch(req, in)
		planLaunchExpectError(t, plan, planErr, 1, "Error detecting hardware: no probe")
		warning := exposedHostWarning(req.Host, req.Port, req.ExtraArgs, os.Getenv)
		if want := []planNote{{Stream: planStderr, Text: warning + "\n"}}; warning == "" || !reflect.DeepEqual(*notes, want) {
			t.Fatalf("notes = %+v, want the exposure warning %+v", *notes, want)
		}

		// A missing inventory: exit 2. Detect is not run for an inventory
		// plan, so its failure cannot win.
		req, model = planLaunchFixture(t, c.hw, c.model)
		req.InventoryPath = filepath.Join(t.TempDir(), "missing.json")
		_, readErr := os.ReadFile(req.InventoryPath)
		in, notes = inputs(req, model)
		in.Detect = func() (*detect.Capabilities, error) {
			t.Error("an inventory plan detected this host's hardware")
			return nil, errors.New("no probe")
		}
		plan, planErr = planLaunch(req, in)
		planLaunchExpectError(t, plan, planErr, 2, fmt.Sprintf("Error reading inventory: %v", readErr))
		if len(*notes) != 0 {
			t.Fatalf("notes = %+v", *notes)
		}

		// A config that does not load: exit 2, after the hardware.
		req, model = planLaunchFixture(t, c.hw, c.model)
		t.Setenv("LLM_PORT", "not-a-port")
		_, cfgErr := config.Load()
		if cfgErr == nil {
			t.Fatal("LLM_PORT=not-a-port loads")
		}
		in, _ = inputs(req, model)
		plan, planErr = planLaunch(req, in)
		planLaunchExpectError(t, plan, planErr, 2, fmt.Sprintf("Error loading config: %v", cfgErr))
		goldenPlanUnsetenv(t, "LLM_PORT")

		// A model that does not parse: exit 1.
		req, model = planLaunchFixture(t, c.hw, c.model)
		in, _ = inputs(req, model)
		in.Model = func(string) (*placement.ModelProfile, error) { return nil, errors.New("not a gguf") }
		plan, planErr = planLaunch(req, in)
		planLaunchExpectError(t, plan, planErr, 1, "Error parsing model: not a gguf")

		// No backend: exit 1 with the reason selection left on the request.
		req, model = planLaunchFixture(t, c.hw, c.model)
		in, _ = inputs(req, model)
		in.ResolveBackend = func(req *launchRequest, _ *placement.ModelProfile, _ *detect.Capabilities) *backendInfo {
			req.BackendUnavailableReason = "No installed backend supports the llama architecture."
			return nil
		}
		plan, planErr = planLaunch(req, in)
		planLaunchExpectError(t, plan, planErr, 1, "Error: No installed backend supports the llama architecture.")
	})
}

// planLaunch refuses a --gpus selection the hardware view does not have with
// the launch's own text and exit status ("Error: " + validateRequestedGPUs,
// exit 1), before the model is read, a backend is resolved or anything is
// placed. A valid selection plans as before.
func TestPlanLaunchRejectsMissingGPUsLikeTheLaunch(t *testing.T) {
	launchOrderTestEnv(t)
	goldenPlanUnsetenv(t, "LLAMA_API_KEY")
	const (
		gpu12   = "NVIDIA synthetic 12 GB"
		gpu24   = "NVIDIA synthetic 24 GB"
		mixedHW = "0 (" + gpu12 + "), 1 (" + gpu24 + "), 2 (" + gpu12 + ")"
	)
	for _, tc := range []struct {
		name, hw, gpus, want string
		detected             bool // hardware from Detect instead of --inventory
	}{
		{name: "inventory/one-missing", hw: "gpu12", gpus: "9", want: "Error: --gpus 9: no GPU with index 9; this machine has 0 (" + gpu12 + ")"},
		{name: "inventory/several-missing", hw: "mixed3", gpus: "0,5,7", want: "Error: --gpus 0,5,7: no GPU with index 5, 7; this machine has " + mixedHW},
		{name: "inventory/no-gpus", hw: "cpu", gpus: "0", want: "Error: --gpus 0: no GPUs were detected on this machine"},
		{name: "detected/one-missing", hw: "mixed3", gpus: "3", want: "Error: --gpus 3: no GPU with index 3; this machine has " + mixedHW, detected: true},
	} {
		c := launchOrderCase{goldenPlanCase: goldenPlanCase{name: "planlaunch-gpus/" + tc.name, hw: tc.hw, model: "dense-small"}}
		launchOrderRunCase(t, c, func(t *testing.T) {
			req, model := planLaunchFixture(t, c.hw, c.model, "--gpus", tc.gpus)
			in := goldenPlanInputs(t, c.name, model, goldenPlanBackend(t, ""), nil)
			in.Inventory = planLaunchSilentInventory
			if tc.detected {
				req.InventoryPath = ""
				caps, err := detect.LoadCapabilities([]byte(goldenPlanInventories[tc.hw]))
				if err != nil {
					t.Fatal(err)
				}
				in.Detect = func() (*detect.Capabilities, error) { return caps, nil }
			}
			in.Model = func(string) (*placement.ModelProfile, error) {
				t.Error("the model was read after a --gpus index the hardware does not have")
				return model, nil
			}
			in.ResolveBackend = func(*launchRequest, *placement.ModelProfile, *detect.Capabilities) *backendInfo {
				t.Error("a backend was resolved after a --gpus index the hardware does not have")
				return nil
			}
			in.ListVulkan = func(string) ([]listedDevice, error) {
				t.Error("Vulkan devices were listed after a --gpus index the hardware does not have")
				return nil, nil
			}
			notes := planLaunchCollect(&in)
			plan, planErr := planLaunch(req, in)
			planLaunchExpectError(t, plan, planErr, 1, tc.want)
			// The launch prints "Error: " + validateRequestedGPUs(caps, req) and exits 1.
			caps, err := detect.LoadCapabilities([]byte(goldenPlanInventories[tc.hw]))
			if err != nil {
				t.Fatal(err)
			}
			if launchErr := validateRequestedGPUs(caps, req); launchErr == nil || "Error: "+launchErr.Error() != planErr.Message {
				t.Fatalf("launch check = %v, plan message %q", launchErr, planErr.Message)
			}
			if len(*notes) != 0 {
				t.Fatalf("notes = %+v", *notes)
			}
		})
	}

	c := launchOrderCase{goldenPlanCase: goldenPlanCase{name: "planlaunch-gpus/valid", hw: "mixed3", model: "dense-small"}}
	launchOrderRunCase(t, c, func(t *testing.T) {
		req, model := planLaunchFixture(t, c.hw, c.model, "--gpus", "1")
		in := goldenPlanInputs(t, c.name, model, goldenPlanBackend(t, ""), nil)
		in.Inventory = planLaunchSilentInventory
		plan, planErr := planLaunch(req, in)
		if planErr != nil || plan == nil || plan.Refusal != nil || len(plan.ServerArgv) == 0 {
			t.Fatalf("plan %+v, error %+v; want a plan", plan, planErr)
		}
	})
}

// planLaunchCapture redirects os.Stdout and os.Stderr; the returned function
// restores both and returns what was written to each.
func planLaunchCapture(t *testing.T) func() (string, string) {
	t.Helper()
	origOut, origErr := os.Stdout, os.Stderr
	outR, outW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	errR, errW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout, os.Stderr = outW, errW
	t.Cleanup(func() { os.Stdout, os.Stderr = origOut, origErr })
	var out, errOut strings.Builder
	done := make(chan struct{}, 2)
	go func() { _, _ = io.Copy(&out, outR); done <- struct{}{} }()
	go func() { _, _ = io.Copy(&errOut, errR); done <- struct{}{} }()
	return func() (string, string) {
		os.Stdout, os.Stderr = origOut, origErr
		_ = outW.Close()
		_ = errW.Close()
		<-done
		<-done
		return out.String(), errOut.String()
	}
}

// planLaunchResidualPrefixes are the lines the callees of planLaunch still
// print themselves on these fixtures: the Claude sampling summary and slot
// notes (buildLaunchServerArgs, claudeCodeSlotAdjust), batch normalisation,
// and placement.Compute's own lines. The list may only shrink.
var planLaunchResidualPrefixes = map[planStream][]string{
	planStdout: {"[claude-code] sampling: ", "[claude-code] context ", "[claude-code] warning: only ", "[launch] normalized batch/ubatch "},
	planStderr: {"[placement] "},
}

// planLaunch in dry-run mode exports nothing (the dry run used to export
// CUDA_VISIBLE_DEVICES for --gpus) and prints nothing of its own: its notes
// reach the sink only, and the console holds only the callees' residual lines.
func TestPlanLaunchLeavesProcessEnvironmentAndConsoleAlone(t *testing.T) {
	launchOrderTestEnv(t)
	goldenPlanUnsetenv(t, "LLAMA_API_KEY")
	for _, tc := range []struct {
		name         string
		args         []string
		wantNotes    int
		wantSampling int
	}{
		{name: "gpu12/dense-small/gpus-0+exposed-host", args: []string{"--gpus", "0", "--host", "0.0.0.0"}, wantNotes: 1},
		{name: "gpu12/dense-small/claude-code+gpus-0", args: []string{"--claude-code", "--gpus", "0"}, wantSampling: 1},
	} {
		c := launchOrderCase{goldenPlanCase: goldenPlanCase{name: tc.name, hw: "gpu12", model: "dense-small", args: tc.args}}
		launchOrderRunCase(t, c, func(t *testing.T) {
			req, model := planLaunchFixture(t, c.hw, c.model, c.args...)
			in := goldenPlanInputs(t, c.name, model, goldenPlanBackend(t, ""), nil)
			in.Inventory = planLaunchSilentInventory
			notes := planLaunchCollect(&in)
			// announceClaudeSampling prints a summary once per process; clear
			// what an earlier test left so the residual line is deterministic.
			claudeSamplingNoticeMu.Lock()
			savedNotice := claudeSamplingNoticeLast
			claudeSamplingNoticeLast = ""
			claudeSamplingNoticeMu.Unlock()
			t.Cleanup(func() {
				claudeSamplingNoticeMu.Lock()
				claudeSamplingNoticeLast = savedNotice
				claudeSamplingNoticeMu.Unlock()
			})

			before := sortedEnviron()
			stop := planLaunchCapture(t)
			plan, planErr := planLaunch(req, in)
			stdout, stderr := stop()
			after := sortedEnviron()

			if planErr != nil || plan == nil || plan.Refusal != nil || len(plan.ServerArgv) == 0 {
				t.Fatalf("plan %+v, error %+v; want a plan", plan, planErr)
			}
			if !reflect.DeepEqual(before, after) {
				t.Fatalf("planLaunch changed the process environment: added %q, removed %q", environAdded(before, after), environAdded(after, before))
			}
			if os.Getenv("CUDA_VISIBLE_DEVICES") != "" || os.Getenv("GGML_VK_VISIBLE_DEVICES") != "" {
				t.Fatal("planLaunch exported a device visibility variable")
			}
			if plan.EnvPrefix != "CUDA_VISIBLE_DEVICES=0" || plan.Environment["CUDA_VISIBLE_DEVICES"] != "0" {
				t.Fatalf("plan env prefix %q, environment %v; want CUDA_VISIBLE_DEVICES=0 as a value", plan.EnvPrefix, plan.Environment)
			}
			if len(*notes) != tc.wantNotes {
				t.Fatalf("notes = %+v, want %d", *notes, tc.wantNotes)
			}
			if tc.wantNotes == 1 {
				want := planNote{Stream: planStderr, Text: exposedHostWarning(req.Host, req.Port, req.ExtraArgs, os.Getenv) + "\n"}
				if (*notes)[0] != want || !strings.HasPrefix(want.Text, "[launch] warning: the model server listens on 0.0.0.0:") {
					t.Fatalf("note = %+v, want the exposure warning %+v", (*notes)[0], want)
				}
			}
			for _, n := range *notes {
				if strings.Contains(stdout, n.Text) || strings.Contains(stderr, n.Text) {
					t.Fatalf("note %q reached the console", n.Text)
				}
			}
			sampling := 0
			for stream, text := range map[planStream]string{planStdout: stdout, planStderr: stderr} {
				for _, line := range strings.SplitAfter(text, "\n") {
					if line == "" {
						continue
					}
					allowed := false
					for _, prefix := range planLaunchResidualPrefixes[stream] {
						allowed = allowed || strings.HasPrefix(line, prefix)
					}
					if !allowed || !strings.HasSuffix(line, "\n") {
						t.Fatalf("planLaunch printed %q on stream %d; only the callees' residual lines may print", line, stream)
					}
					if strings.HasPrefix(line, "[claude-code] sampling: ") {
						sampling++
					}
				}
			}
			if sampling != tc.wantSampling {
				t.Fatalf("%d sampling lines, want %d (stdout %q)", sampling, tc.wantSampling, stdout)
			}
		})
	}
}

// The plan's memory limits are the ones the launch starts the backend with:
// startLaunchProcess's backendStartOptions call on the same request, hardware,
// backend and argv. Resident plans keep MemoryHigh == MemoryMax at the budget;
// mmap-backed CPU experts raise MemoryMax to the host reclaim ceiling, which
// backendMemoryMaxMB alone (the old memory_max_mb) under-reported.
func TestPlanLaunchMemoryLimitsAreTheLaunchStartLimits(t *testing.T) {
	launchOrderTestEnv(t)
	if runtime.GOOS != "linux" {
		t.Skip("backend memory scopes are Linux-only")
	}
	for _, tc := range []struct {
		name, backend, hw, model string
		args                     []string
		mmapBand                 bool
	}{
		{name: "resident/gpu12/dense-small", hw: "gpu12", model: "dense-small"},
		{name: "resident-cpu-experts/mixed3/moe", hw: "mixed3", model: "moe"},
		{name: "mmap-experts/gpu12-ram32/moe", hw: "gpu12-ram32", model: "moe", args: []string{"--mmap"}, mmapBand: true},
		{name: "mmap-experts/mixed3/moe/claude-code", hw: "mixed3", model: "moe", args: []string{"--claude-code", "--mmap"}, mmapBand: true},
	} {
		c := launchOrderCase{goldenPlanCase: goldenPlanCase{name: tc.name, hw: tc.hw, model: tc.model, args: tc.args}}
		launchOrderRunCase(t, c, func(t *testing.T) {
			req, model := planLaunchFixture(t, c.hw, c.model, c.args...)
			in := goldenPlanInputs(t, c.name, model, goldenPlanBackend(t, tc.backend), nil)
			in.Inventory = planLaunchSilentInventory
			plan, planErr := planLaunch(req, in)
			if planErr != nil || plan == nil || plan.Refusal != nil {
				t.Fatalf("plan %+v, error %+v; want a plan", plan, planErr)
			}
			start := backendStartOptions(plan.Req, plan.Caps, plan.Backend, hostExpertPinningEnv(plan.Backend, plan.ServerArgv), plan.ServerArgv)
			if plan.MemoryHighMB != start.MemoryHighMB || plan.MemoryMaxMB != start.MemoryMaxMB {
				t.Fatalf("plan limits high=%d max=%d, launch start limits high=%d max=%d", plan.MemoryHighMB, plan.MemoryMaxMB, start.MemoryHighMB, start.MemoryMaxMB)
			}
			budget := backendMemoryMaxMB(plan.Req, plan.Caps)
			if budget <= 0 || plan.MemoryHighMB != budget {
				t.Fatalf("MemoryHigh %d, want the plan budget %d", plan.MemoryHighMB, budget)
			}
			if gotBand := plan.MemoryMaxMB > plan.MemoryHighMB; gotBand != tc.mmapBand {
				t.Fatalf("high=%d max=%d (argv %q): reclaim band %v, want %v", plan.MemoryHighMB, plan.MemoryMaxMB, plan.ServerArgv, gotBand, tc.mmapBand)
			}
			if tc.mmapBand && plan.MemoryMaxMB != hostReclaimCeilingMB(plan.Req, plan.Caps) {
				t.Fatalf("mmap-backed MemoryMax %d, want the host reclaim ceiling %d", plan.MemoryMaxMB, hostReclaimCeilingMB(plan.Req, plan.Caps))
			}
		})
	}
}

// A lazy sparse-lookup argv (no golden fixture has a reviewed loader) gets the
// banded start limits through planBackendProcess, the function planLaunch calls.
func TestPlanBackendProcessLazyLookupLimitsAreTheLaunchStartLimits(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("backend memory scopes and the reviewed loader are Linux-only")
	}
	req := &launchRequest{RAMLimitPercent: 90, CgroupHeadroomMB: 4096}
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 200000, FreeMB: 200000}}
	be := reviewedLazyBackend()
	for _, argv := range [][]string{{"/b/llama-server", "--no-mmap"}, {"/b/llama-server", "--no-mmap", "--lazy-mode", "on"}} {
		_, high, max := planBackendProcess(req, caps, be, argv, "")
		start := backendStartOptions(req, caps, be, hostExpertPinningEnv(be, argv), argv)
		if high != start.MemoryHighMB || max != start.MemoryMaxMB || max <= 0 {
			t.Fatalf("%q: plan high=%d max=%d, launch start high=%d max=%d", argv, high, max, start.MemoryHighMB, start.MemoryMaxMB)
		}
		if lazy := argsLazySparseLookup(be, argv); lazy != (max > high) {
			t.Fatalf("%q: lazy=%v but high=%d max=%d", argv, lazy, high, max)
		}
	}
}

// The plan's environment carries what the launch adds for its argv
// (GGML_CUDA_NO_PINNED=1 for ik with CPU experts, hostExpertPinningEnv) and
// nothing from the caller's environment beyond launchPlanEnvironment's
// allow-list, even when the caller exports the same variable.
func TestPlanLaunchEnvironmentHasLaunchOverridesAndNoCallerLeak(t *testing.T) {
	launchOrderTestEnv(t)
	allowed := map[string]bool{"CUDA_DEVICE_ORDER": true, "CUDA_SCALE_LAUNCH_QUEUES": true, "LD_LIBRARY_PATH": true}
	for _, tc := range []struct {
		name, backend, hw, model string
		wantNoPinned             bool
	}{
		{name: "ik/mixed3/moe", backend: "ik", hw: "mixed3", model: "moe", wantNoPinned: true},
		{name: "ik/gpu24/moe", backend: "ik", hw: "gpu24", model: "moe", wantNoPinned: true},
		{name: "ik/gpu24/dense-large", backend: "ik", hw: "gpu24", model: "dense-large"},
		{name: "mainline/mixed3/moe", hw: "mixed3", model: "moe"},
	} {
		c := launchOrderCase{goldenPlanCase: goldenPlanCase{name: tc.name, hw: tc.hw, model: tc.model}}
		c.env = map[string]string{"GGML_CUDA_NO_PINNED": "1", "GGRUN_TEST_UNRELATED_SECRET": "do-not-report", "HF_TOKEN": "do-not-report"}
		launchOrderRunCase(t, c, func(t *testing.T) {
			req, model := planLaunchFixture(t, c.hw, c.model)
			in := goldenPlanInputs(t, c.name, model, goldenPlanBackend(t, tc.backend), nil)
			in.Inventory = planLaunchSilentInventory
			plan, planErr := planLaunch(req, in)
			if planErr != nil || plan == nil || plan.Refusal != nil {
				t.Fatalf("plan %+v, error %+v; want a plan", plan, planErr)
			}
			pinning := hostExpertPinningEnv(plan.Backend, plan.ServerArgv)
			if got := plan.Environment["GGML_CUDA_NO_PINNED"]; (got == "1") != tc.wantNoPinned || (len(pinning) > 0) != tc.wantNoPinned {
				t.Fatalf("GGML_CUDA_NO_PINNED=%q in the plan, launch pinning env %q; want present=%v (argv %q)", got, pinning, tc.wantNoPinned, plan.ServerArgv)
			}
			for key, value := range plan.Environment {
				if allowed[key] || (tc.wantNoPinned && key == "GGML_CUDA_NO_PINNED") {
					continue
				}
				t.Errorf("plan environment carries %s=%q, outside the allow-list and the launch's own overrides", key, value)
			}
		})
	}
}

// cmdDryRun hands planning to planLaunch: a planning call in its body would be
// a second planner the goldens do not witness.
func TestCmdDryRunDoesNotPlanItself(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "main.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var body *ast.BlockStmt
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Recv == nil && fn.Name.Name == "cmdDryRun" {
			body = fn.Body
		}
	}
	if body == nil {
		t.Fatal("cmdDryRun not found in main.go")
	}
	forbidden := map[string]bool{
		"detect.Detect": true, "loadInventory": true, "config.Load": true, "loadConfigOrExit": true,
		"resolveModelPath": true, "parseModel": true, "warnModelCompatibility": true,
		"resolveLaunchBackend": true, "selectBackendForModel": true, "selectBackend": true,
		"normalizeArchKVRequest": true, "applyBackendFeatureCompatibility": true, "preflightBackendArch": true,
		"applyCachedBackendCapabilities": true, "applyRememberedClaudeSelfClassify": true,
		"claudeReviewerReservation": true, "placementOptionsFromRequest": true, "placementOptionsFromRequestCaps": true,
		"placement.Compute": true, "claudeCodeSlotAdjust": true, "applyCalibrationDecision": true,
		"explainServingOptimization": true, "dryRunFitRefusal": true, "buildLaunchServerArgs": true,
		"gpuVisibilityEnv": true, "applyGPUVisibility": true, "vulkanDeviceOrder": true,
		"vulkanDeviceOrderEnv": true, "applyVulkanDeviceOrder": true, "listVulkanDevices": true,
		"launchPlanEnvironment": true, "backendMemoryMaxMB": true, "claudeLaunchLogScope": true,
		"planBackendProcess": true, "backendStartOptions": true, "hostExpertPinningEnv": true,
		"os.Setenv": true, "exposedHostWarning": true,
	}
	planCalls := 0
	ast.Inspect(body, func(n ast.Node) bool {
		var name string
		switch e := n.(type) {
		case *ast.Ident:
			name = e.Name
		case *ast.SelectorExpr:
			if pkg, ok := e.X.(*ast.Ident); ok {
				name = pkg.Name + "." + e.Sel.Name
			}
		default:
			return true
		}
		if forbidden[name] {
			t.Errorf("cmdDryRun refers to %s; planning belongs in planLaunch", name)
		}
		if name == "planLaunch" {
			planCalls++
		}
		return true
	})
	if planCalls != 1 {
		t.Fatalf("cmdDryRun refers to planLaunch %d times, want once", planCalls)
	}
}

// withStdoutOnStderr is the dry run's JSON-mode guard: inside it, stdout
// prints (direct ones and stdout notes) reach stderr in order with stderr's
// own lines; on every way out, a plan error, a refused plan and a panic
// included, os.Stdout is what it was before.
func TestWithStdoutOnStderrRestoresStdout(t *testing.T) {
	origOut, origErr := os.Stdout, os.Stderr
	t.Cleanup(func() { os.Stdout, os.Stderr = origOut, origErr })
	dir := t.TempDir()
	create := func(name string) *os.File {
		f, err := os.Create(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = f.Close() })
		return f
	}
	out, errOut := create("stdout"), create("stderr")
	os.Stdout, os.Stderr = out, errOut
	restored := func(step string) {
		t.Helper()
		if os.Stdout != out || os.Stderr != errOut {
			os.Stdout, os.Stderr = origOut, origErr
			t.Fatalf("%s: os.Stdout/os.Stderr not restored", step)
		}
	}

	// An error returned by planLaunch itself.
	var planErr *planError
	withStdoutOnStderr(func() {
		if os.Stdout != errOut {
			t.Error("os.Stdout inside the guard is not os.Stderr")
		}
		fmt.Print("direct stdout line\n")
		printPlanNote(planNote{Stream: planStdout, Text: "stdout note\n"})
		printPlanNote(planNote{Stream: planStderr, Text: "stderr note\n"})
		_, planErr = planLaunch(&launchRequest{Host: "127.0.0.1"}, planInputs{
			Detect: func() (*detect.Capabilities, error) { return nil, errors.New("no probe") },
		})
	})
	restored("plan error")
	if planErr == nil || planErr.Message != "Error detecting hardware: no probe" {
		t.Fatalf("planErr = %+v", planErr)
	}

	// A refused plan.
	var plan *launchPlan
	withStdoutOnStderr(func() {
		fmt.Println("refusal path line")
		plan = &launchPlan{Refusal: &planError{Code: 1, Message: "refused"}}
	})
	restored("refusal")
	if plan.Refusal == nil {
		t.Fatal("refusal lost")
	}

	// A panic.
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("no panic")
			}
		}()
		withStdoutOnStderr(func() {
			fmt.Println("before panic")
			panic("boom")
		})
	}()
	restored("panic")

	fmt.Print("after the guard\n")
	read := func(f *os.File) string {
		b, err := os.ReadFile(f.Name())
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	if got := read(out); got != "after the guard\n" {
		t.Errorf("stdout = %q, want only the line printed after the guard", got)
	}
	if got, want := read(errOut), "direct stdout line\nstdout note\nstderr note\nrefusal path line\nbefore panic\n"; got != want {
		t.Errorf("stderr = %q, want %q", got, want)
	}
}
