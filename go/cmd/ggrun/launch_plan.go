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

	// A dry-run may plan for a DIFFERENT machine, or for this one while another
	// process holds its VRAM. Planning-only: real launches refuse the flag
	// outright (see loadInventory), because exact-argv admission and observed
	// allocations are the authority for what actually fits. An inventory plan
	// reads nothing from this host's devices: no detection, and the live
	// readings planning takes later come from the inventory (useInventoryReadings).
	fromInventory := in.Policy.Inventory && req.InventoryPath != ""
	var caps *detect.Capabilities
	var err error
	if fromInventory {
		caps, err = in.Inventory(req.InventoryPath)
		if err != nil {
			return nil, &planError{Code: 2, Message: fmt.Sprintf("Error reading inventory: %v", err), Err: err}
		}
		defer useInventoryReadings(caps, req)()
	} else {
		caps, err = in.Detect()
		if err != nil {
			return nil, &planError{Code: 1, Message: fmt.Sprintf("Error detecting hardware: %v", err), Err: err}
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
		// The running-server hint scans this host's processes, which say
		// nothing about an inventory's machine.
		msg := err.Error()
		if !fromInventory {
			msg = placementErrorMessage(err)
		}
		return nil, &planError{Code: 1, Message: "Error computing placement: " + msg, Err: err}
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
		var listed []listedDevice
		var vkErr error
		if fromInventory {
			// No backend runs: the inventory's order stands for the backend's.
			listed = inventoryVulkanListing(caps)
			note(planNote{Stream: planStderr, Text: "[dry-run] Vulkan device order taken from the inventory; not checked against the backend's own listing\n"})
		} else {
			listed, vkErr = in.ListVulkan(be.Path)
		}
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

// useInventoryReadings points the live readers planning consults at inv for
// the rest of an inventory plan and returns the function that puts the
// previous readers back; planLaunch defers it, so every return and a panic
// restore them. The readers are package variables (test seams) read by
// runtimeGPUCapabilitiesForLaunch, bindFitToGuard and
// claudeReviewerReservation, which the launch shares; swapping them here keeps
// those bodies and the launch untouched. Not safe beside goroutines that call
// the readers concurrently: planning starts none that outlive it, and nothing
// plans concurrently with a dry run in one process.
//
//   - used VRAM: the inventory's own vram_used_mb, so the live floor adds
//     nothing to what the inventory already charges;
//   - driver free VRAM: total - used of the inventory card, what a driver on
//     that machine would report (ggrun's own vram_reserved_mb is withheld by
//     the planner, not by the driver);
//   - resident reviewer: none; an inventory cannot describe running processes.
func useInventoryReadings(inv *detect.Capabilities, req *launchRequest) (restore func()) {
	savedUsed, savedFree, savedResident := runtimeVRAMUsedMB, fitDriverFreeVRAM, residentReviewerVRAM
	byIndex := map[int]detect.GPU{}
	if inv != nil {
		for _, g := range inv.GPUs {
			byIndex[g.Index] = g
		}
	}
	runtimeVRAMUsedMB = func(physical int) int { return byIndex[physical].VRAMUsedMB }
	fitDriverFreeVRAM = func(runtime *detect.Capabilities) map[int]int {
		if runtime == nil || len(runtime.GPUs) == 0 {
			return nil
		}
		_, visibleToPhysical := runtimeGPUCapabilities(inv, req)
		free := map[int]int{}
		for _, g := range runtime.GPUs {
			if card, ok := byIndex[physicalGPUIndex(g.Index, visibleToPhysical)]; ok {
				free[g.Index] = max(card.VRAMTotalMB-card.VRAMUsedMB, 0)
			}
		}
		return free
	}
	residentReviewerVRAM = func(string) int { return 0 }
	return func() {
		runtimeVRAMUsedMB, fitDriverFreeVRAM, residentReviewerVRAM = savedUsed, savedFree, savedResident
	}
}

// inventoryVulkanListing is the device listing an inventory plan uses in place
// of the backend's: the inventory's GPUs in their own order.
func inventoryVulkanListing(inv *detect.Capabilities) []listedDevice {
	listed := make([]listedDevice, len(inv.GPUs))
	for i, g := range inv.GPUs {
		listed[i] = listedDevice{Index: i, Name: g.Name, TotalMB: g.VRAMTotalMB}
	}
	return listed
}

// planPolicy holds what the dry run does that a launch does not (D1, D31, D32
// of the W11 design). The dry run is its only user so far.
type planPolicy struct {
	Inventory    bool // read --inventory in place of the detected hardware and the live readings
	FullFrontier bool // explainServingOptimization's fullFrontier
	FitRefusal   bool // refuse a plan whose own ledger says GPU KV does not fit
}

var dryRunPlanPolicy = planPolicy{Inventory: true, FullFrontier: true, FitRefusal: true}

// planInputs are planLaunch's readers and its note sink. dryRunPlanInputs
// gives the production ones; the golden tests inject fakes for hardware,
// model, backend and the Vulkan listing. With an inventory, Detect and
// ListVulkan are not called.
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
