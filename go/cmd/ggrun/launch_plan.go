package main

import (
	"fmt"
	"os"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// planLaunch computes the dry run's initial plan (review W11, step 5): the
// adjusted request, hardware view, model, backend, strategy, server argv,
// environment additions, memory limit and launch scope. It never calls
// os.Exit, never writes the process environment and prints nothing itself:
// its notes go to in.Note in the order and on the stream cmdDryRun used to
// print them, and an error that ends the dry run comes back as a planError.
//
// The functions it calls still print some lines directly (backend selection,
// the arch KV rule, placement.Compute, slot adjustment, calibration reuse,
// batch normalisation, the Claude sampling summary, printVRAMLedger under
// GGRUN_TRACE_PLACEMENT); their text is unchanged. In JSON mode cmdDryRun
// runs planLaunch under withStdoutOnStderr, so their stdout lines and the
// stdout notes reach stderr instead.
// resolveLaunchBackend can still exit through preflightIKOnlyArch.
func planLaunch(req *launchRequest, in planInputs) (*launchPlan, *planError) {
	note := in.Note
	if note == nil {
		note = func(planNote) {}
	}
	if w := exposedHostWarning(req.Host, req.Port, req.ExtraArgs, os.Getenv); w != "" {
		note(planNote{Stream: planStderr, Text: w + "\n"})
	}

	caps, err := in.Detect()
	if err != nil {
		return nil, &planError{Code: 1, Message: fmt.Sprintf("Error detecting hardware: %v", err), Err: err}
	}
	// A dry-run may plan for a DIFFERENT machine, or for this one while another
	// process holds its VRAM. Planning-only: real launches refuse the flag
	// outright (see loadInventory), because exact-argv admission and observed
	// allocations are the authority for what actually fits.
	if in.Policy.Inventory && req.InventoryPath != "" {
		caps, err = in.Inventory(req.InventoryPath)
		if err != nil {
			return nil, &planError{Code: 2, Message: fmt.Sprintf("Error reading inventory: %v", err), Err: err}
		}
	}

	cfg, err := config.Load()
	if err != nil {
		return nil, &planError{Code: 2, Message: fmt.Sprintf("Error loading config: %v", err), Err: err}
	}
	req.ModelPath = resolveModelPath(req.ModelPath, cfg.ModelDir)

	model, err := in.Model(req.ModelPath)
	if err != nil {
		return nil, &planError{Code: 1, Message: fmt.Sprintf("Error parsing model: %v", err), Err: err}
	}
	warnModelCompatibility(model)

	be := in.ResolveBackend(req, model, caps)
	if be == nil {
		return nil, &planError{Code: 1, Message: "Error: " + backendUnavailableMessage(req)}
	}
	applyCachedBackendCapabilities(req, cfg.CacheDir, model, be)
	// Match cmdLaunch exactly: Claude Auto's helper consumes VRAM before the
	// main model is packed. Without this reservation a dry-run can claim one
	// additional expert layer fits and disagree with the real launch.
	applyRememberedClaudeSelfClassify(req, cfg.CacheDir, model, caps, planNoteWriter{stream: planStderr, note: note})
	req.ReviewerReservation = claudeReviewerReservation(req, caps, cfg.CacheDir)

	placementOpts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	strategy, err := placement.Compute(caps, model, placementOpts)
	if err != nil {
		return nil, &planError{Code: 1, Message: "Error computing placement: " + placementErrorMessage(err), Err: err}
	}
	claudeCodeSlotAdjust(strategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	strategy = applyCalibrationDecision(req, cfg, model, be, caps, strategy)
	strategy = explainServingOptimization(req, cfg, model, be, caps, strategy, in.Policy.FullFrontier)

	if os.Getenv("GGRUN_TRACE_PLACEMENT") != "" {
		printVRAMLedger(strategy)
		note(planNote{Stream: planStdout, Text: fmt.Sprintf("[trace] NCPUMoE=%d OT=%s\n", strategy.NCPUMoE, strategy.OTString)})
	}
	plan := &launchPlan{Req: req, Config: cfg, Caps: caps, Model: model, Backend: be, Strategy: strategy}
	if in.Policy.FitRefusal {
		if plan.Refusal = planFitRefusal(strategy); plan.Refusal != nil {
			return plan, nil
		}
	}

	plan.ServerArgv = buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
	if key, value := gpuVisibilityEnv(req, backendDialect(be)); key != "" {
		plan.EnvPrefix = key + "=" + value
	}
	if vulkanDeviceOrderApplies(be, caps) {
		listed, vkErr := in.ListVulkan(be.Path)
		key, value := "", ""
		if vkErr == nil {
			key, value, vkErr = vulkanDeviceOrder(req, be, caps, listed)
		}
		if vkErr != nil {
			note(planNote{Stream: planStderr, Text: fmt.Sprintf("[dry-run] warning: could not align Vulkan device order with detected GPUs: %v\n", vkErr)})
		} else if key != "" {
			plan.EnvPrefix = key + "=" + value
		}
	}
	plan.Environment = launchPlanEnvironment(plan.ServerArgv, plan.EnvPrefix, be.Path)
	plan.MemoryMaxMB = backendMemoryMaxMB(req, caps)
	// The name a crash log would be filed under. Exposed because it is a
	// hash: when recovery silently fails to find a previous OOM, the only
	// way to see why is to compare this between two runs.
	plan.LaunchScope = claudeLaunchLogScope(req, model, be, plan.ServerArgv)
	return plan, nil
}

// launchPlan is planLaunch's result.
type launchPlan struct {
	Req      *launchRequest // the request as planning adjusted it (same pointer as the input)
	Config   *config.Config // the loaded config; the summary prints learned reserves from its cache
	Caps     *detect.Capabilities
	Model    *placement.ModelProfile
	Backend  *backendInfo
	Strategy *placement.Strategy
	// Refusal is the dry run's fit refusal. When set, the plan stops at the
	// strategy: no argv, environment, memory limit or scope is computed.
	Refusal     *planError
	ServerArgv  []string
	EnvPrefix   string            // the GPU visibility or Vulkan order assignment the summary shows
	Environment map[string]string // launchPlanEnvironment, the plan JSON's "environment"
	MemoryMaxMB int               // backendMemoryMaxMB, the plan JSON's "memory_max_mb"
	LaunchScope string            // claudeLaunchLogScope, the plan JSON's "launch_scope"
}

// planPolicy holds what the dry run does that a launch does not (D1, D31, D32
// of the W11 design). The dry run is its only user so far.
type planPolicy struct {
	Inventory    bool // read --inventory in place of the detected hardware
	FullFrontier bool // explainServingOptimization's fullFrontier
	FitRefusal   bool // refuse a plan whose own ledger says GPU KV does not fit
}

var dryRunPlanPolicy = planPolicy{Inventory: true, FullFrontier: true, FitRefusal: true}

// planInputs are planLaunch's readers and its note sink. dryRunPlanInputs
// gives the production ones; the golden tests inject fakes for hardware,
// model, backend and the Vulkan listing.
type planInputs struct {
	Policy         planPolicy
	Detect         func() (*detect.Capabilities, error)
	Inventory      func(path string) (*detect.Capabilities, error)
	Model          func(path string) (*placement.ModelProfile, error)
	ResolveBackend func(req *launchRequest, model *placement.ModelProfile, caps *detect.Capabilities) *backendInfo
	ListVulkan     func(path string) ([]listedDevice, error)
	// Note receives each note when it happens, so a sink that prints keeps
	// today's order against the lines the called functions print directly.
	Note func(planNote)
}

func dryRunPlanInputs(note func(planNote)) planInputs {
	return planInputs{
		Policy:         dryRunPlanPolicy,
		Detect:         detect.Detect,
		Inventory:      loadInventory,
		Model:          parseModel,
		ResolveBackend: resolveLaunchBackend,
		ListVulkan:     listVulkanDevices,
		Note:           note,
	}
}

type planStream int

const (
	planStdout planStream = iota
	planStderr
)

// planNote is one piece of console text: Text is exactly what is printed,
// trailing newline included.
type planNote struct {
	Stream planStream
	Text   string
}

// printPlanNote is cmdDryRun's sink.
func printPlanNote(n planNote) {
	if n.Stream == planStdout {
		fmt.Fprint(os.Stdout, n.Text)
		return
	}
	fmt.Fprint(os.Stderr, n.Text)
}

// withStdoutOnStderr runs fn with os.Stdout set to os.Stderr and puts the
// previous os.Stdout back when fn returns or panics. The dry run uses it in
// JSON mode so stdout carries only the plan document. It swaps the Go
// variable, not file descriptor 1: everything that looks up os.Stdout when it
// prints (fmt.Print*, printPlanNote, a child given cmd.Stdout = os.Stdout)
// follows it; a child with no Stdout set gets the null device either way.
// Not safe beside goroutines that use os.Stdout concurrently; planning starts
// none that outlive it.
func withStdoutOnStderr(fn func()) {
	saved := os.Stdout
	os.Stdout = os.Stderr
	defer func() { os.Stdout = saved }()
	fn()
}

// planNoteWriter turns a callee's writes into notes on one stream.
type planNoteWriter struct {
	stream planStream
	note   func(planNote)
}

func (w planNoteWriter) Write(p []byte) (int, error) {
	w.note(planNote{Stream: w.stream, Text: string(p)})
	return len(p), nil
}

// planError ends a dry run: Message is printed on stderr as one line and the
// process exits with Code (2 for input errors, 1 for everything else). Err is
// the cause when there is one; for a placement error it is the raw error
// without placementErrorMessage's running-server hint.
type planError struct {
	Code    int
	Message string
	Err     error
}

func (e *planError) Error() string { return e.Message }

// note is the stderr text cmdDryRun prints for e.
func (e *planError) note() planNote {
	return planNote{Stream: planStderr, Text: e.Message + "\n"}
}

// planFitRefusal is the dry run's refusal of a plan whose per-device ledger
// says KV on a GPU does not fit there: exit 1, the dryRunFitRefusal text on
// stderr. nil when the plan fits.
func planFitRefusal(strategy *placement.Strategy) *planError {
	if refusal := dryRunFitRefusal(strategy); refusal != "" {
		return &planError{Code: 1, Message: refusal}
	}
	return nil
}
