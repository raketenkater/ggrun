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

// The "Model does not fit" placement error is not in the console golden: its
// message carries placementErrorMessage's /proc scan for running llama-server
// processes, which another test of the same `go test ./...` can start. Here
// the cause is pinned exactly and the hint is allowed in either form.
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
		base := "Error computing placement: " + cause
		const hint = "\n\nActive llama-server process(es) are currently consuming memory; if you are switching models, stop the current server and retry:\n"
		if planErr.Message != base && !strings.HasPrefix(planErr.Message, base+hint) {
			t.Fatalf("message = %q, want %q, optionally followed by the running-server hint", planErr.Message, base)
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

		// Detect fails first, before a missing inventory is read; the exposure
		// warning is already out.
		req, model := planLaunchFixture(t, c.hw, c.model, "--host", "0.0.0.0")
		req.InventoryPath = filepath.Join(t.TempDir(), "missing.json")
		in, notes := inputs(req, model)
		in.Detect = func() (*detect.Capabilities, error) { return nil, errors.New("no probe") }
		plan, planErr := planLaunch(req, in)
		planLaunchExpectError(t, plan, planErr, 1, "Error detecting hardware: no probe")
		warning := exposedHostWarning(req.Host, req.Port, req.ExtraArgs, os.Getenv)
		if want := []planNote{{Stream: planStderr, Text: warning + "\n"}}; warning == "" || !reflect.DeepEqual(*notes, want) {
			t.Fatalf("notes = %+v, want the exposure warning %+v", *notes, want)
		}

		// A missing inventory: exit 2.
		req, model = planLaunchFixture(t, c.hw, c.model)
		req.InventoryPath = filepath.Join(t.TempDir(), "missing.json")
		_, readErr := os.ReadFile(req.InventoryPath)
		in, notes = inputs(req, model)
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
