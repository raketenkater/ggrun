package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// planLaunch computes the dry run's initial plan (review W11, step 5): the
// adjusted request, hardware view, model, backend, strategy, server argv,
// environment additions, memory limits and launch scope. It never calls
// os.Exit, never writes the process environment and prints nothing itself:
// its notes go to in.Note in the order and on the stream cmdDryRun used to
// print them, and an error that ends the dry run comes back as a planError.
//
// With --claude-resume it runs the launch's resume-context step on its first
// plan (resumeContextPlan): the re-plan line and its warnings reach in.Note on
// the streams the launch prints them on.
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
		// Refused, never renumbered: saved host records store per-card values by
		// index under an identity that ignores the index (validateInventoryNumbering).
		if err := validateInventoryNumbering(caps.GPUs); err != nil {
			return nil, &planError{Code: 2, Message: fmt.Sprintf("Error reading inventory: %v", err), Err: err}
		}
		defer useInventoryReadings(caps, req)()
	} else {
		caps, err = in.Detect()
		if err != nil {
			return nil, &planError{Code: 1, Message: fmt.Sprintf("Error detecting hardware: %v", err), Err: err}
		}
	}

	// The launch's own check, on the hardware view just taken: a --gpus index
	// this machine (or inventory) does not have is rejected here, before the
	// model is read or anything is planned. Shared, not a planPolicy field.
	if gpuErr := requestedGPUsPlanError(caps, req); gpuErr != nil {
		return nil, gpuErr
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

	// The dry run's plan for one request: placement, the Claude slot adjustment
	// and a calibration decision for that exact scope. The first plan and the
	// resume re-plan below both use it.
	computeStrategy := func(candidate *launchRequest) (*placement.Strategy, error) {
		next, computeErr := placement.Compute(caps, model, placementOptionsFromRequest(candidate, model, be, cfg.CacheDir))
		if computeErr != nil {
			return nil, computeErr
		}
		claudeCodeSlotAdjust(next, model, candidate.ClaudeCode, candidate.ParallelSet, candidate.BatchSizeSet, candidate.UBatchSizeSet)
		return applyCalibrationDecision(candidate, cfg, model, be, caps, next), nil
	}
	strategy, err := computeStrategy(req)
	if err != nil {
		// The running-server hint scans this host's processes, which say
		// nothing about an inventory's machine.
		msg := err.Error()
		if !fromInventory {
			msg = placementErrorMessage(err)
		}
		return nil, &planError{Code: 1, Message: "Error computing placement: " + msg, Err: err}
	}
	// The launch's own step (cmdLaunch runs ensureResumeContext on its first
	// plan): with --claude-resume, re-plan so the resumed session fits. Shared,
	// not a planPolicy field. The session's size is read from this machine's
	// session records and Claude Code transcripts even for an inventory plan:
	// the session lives where the user launches, and the inventory describes
	// hardware, not conversations.
	strategy, unread := resumeContextPlan(req, cfg.CacheDir, model, strategy, computeStrategy,
		planNoteWriter{stream: planStdout, note: note}, planNoteWriter{stream: planStderr, note: note})
	if unread {
		note(planNote{Stream: planStderr, Text: fmt.Sprintf("[dry-run] could not read the size of Claude session %s on this machine; a launch plans the same context without it, and stops with an error after the server starts if no resumable record of the session exists here\n", req.ClaudeResume)})
	}
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
			// Never for an inventory plan: it reads nothing from this host.
			if !fromInventory && in.ListVulkanUnder != nil && vulkanRestrictionApplies(be, caps, in.HostDisplays) {
				var restrictNote string
				key, value, restrictNote, vkErr = restrictedVulkanDeviceOrder(req, be, caps, listed, func(visible string) (string, error) {
					return in.ListVulkanUnder(be.Path, visible)
				})
				if restrictNote != "" {
					note(planNote{Stream: planStderr, Text: "[dry-run] " + restrictNote + "\n"})
				}
			} else {
				key, value, vkErr = vulkanDeviceOrder(req, be, caps, listed)
			}
		}
		if vkErr != nil {
			note(planNote{Stream: planStderr, Text: fmt.Sprintf("[dry-run] warning: could not align Vulkan device order with detected GPUs: %v\n", vkErr)})
		} else if key != "" {
			plan.EnvPrefix = key + "=" + value
		}
	}
	plan.Environment, plan.MemoryHighMB, plan.MemoryMaxMB = planBackendProcess(req, caps, be, plan.ServerArgv, plan.EnvPrefix)
	// The name a crash log would be filed under. Exposed because it is a
	// hash: when recovery silently fails to find a previous OOM, the only
	// way to see why is to compare this between two runs.
	plan.LaunchScope = claudeLaunchLogScope(req, model, be, plan.ServerArgv)
	return plan, nil
}

// requestedGPUsPlanError is validateRequestedGPUs as a planError, with the text
// and exit status cmdLaunch gives the same input: "Error: <message>", exit 1.
func requestedGPUsPlanError(caps *detect.Capabilities, req *launchRequest) *planError {
	if err := validateRequestedGPUs(caps, req); err != nil {
		return &planError{Code: 1, Message: fmt.Sprintf("Error: %v", err), Err: err}
	}
	return nil
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
	// strategy: no argv, environment, memory limits or scope is computed.
	Refusal      *planError
	ServerArgv   []string
	EnvPrefix    string            // the GPU visibility or Vulkan order assignment the summary shows
	Environment  map[string]string // planBackendProcess, the plan JSON's "environment"
	MemoryHighMB int               // planBackendProcess, the plan JSON's "memory_high_mb"
	MemoryMaxMB  int               // planBackendProcess, the plan JSON's "memory_max_mb"
	LaunchScope  string            // claudeLaunchLogScope, the plan JSON's "launch_scope"
}

// planBackendProcess is the environment and memory scope a launch starts the
// backend with for serverArgs. The limits and the environment overrides come
// from the call startLaunchProcess makes, backendStartOptions(req, caps, be,
// hostExpertPinningEnv(be, serverArgs), serverArgs), so they are the values the
// backend's scope gets: MemoryMax raised to the host reclaim ceiling for
// mmap-backed experts and banded for lazy sparse lookup. The overrides (today
// GGML_CUDA_NO_PINNED=1 for ik with CPU experts) are laid over
// launchPlanEnvironment last, because the launch applies them after
// server.ChildEnv (server.OverrideEnv, or /usr/bin/env inside the scope). They
// are values the plan derives; nothing beyond launchPlanEnvironment's
// allow-list is read from this process's environment.
func planBackendProcess(req *launchRequest, caps *detect.Capabilities, be *backendInfo, serverArgs []string, envPrefix string) (env map[string]string, memoryHighMB, memoryMaxMB int) {
	start := backendStartOptions(req, caps, be, hostExpertPinningEnv(be, serverArgs), serverArgs)
	env = launchPlanEnvironment(serverArgs, envPrefix, be.Path)
	for _, item := range start.EnvOverrides {
		if key, value, ok := strings.Cut(item, "="); ok && key != "" {
			env[key] = value
		}
	}
	return env, start.MemoryHighMB, start.MemoryMaxMB
}

// useInventoryReadings points the live readers planning consults at inv for
// the rest of an inventory plan and returns the function that puts the
// previous readers back; planLaunch defers it, so every return and a panic
// restore them. The readers are package variables (test seams) read by
// runtimeGPUCapabilitiesForLaunch, bindFitToGuard, claudeReviewerReservation
// and resolveClaudeCompanionProfile, which the launch shares; swapping them here keeps
// those bodies and the launch untouched. Not safe beside goroutines that call
// the readers concurrently: planning starts none that outlive it, and nothing
// plans concurrently with a dry run in one process.
//
//   - used VRAM: the inventory's own vram_used_mb, so the live floor adds
//     nothing to what the inventory already charges;
//   - driver free VRAM: total - used of the inventory card, what a driver on
//     that machine would report (ggrun's own vram_reserved_mb is withheld by
//     the planner, not by the driver);
//   - resident reviewer: none; an inventory cannot describe running processes;
//   - Nano reviewer GPU check (--claude-reviewer nanbeige; it runs the reviewer
//     backend with --list-devices and needs a CUDA device there): capable only
//     when an inventory GPU's name says NVIDIA. detect.GPU carries no vendor,
//     and ComputeCap holds the Vulkan API version on Vulkan inventories, so the
//     name is the only marker; a card it cannot identify counts as not
//     capable, which keeps the Qwen reviewer a non-CUDA machine could run.
func useInventoryReadings(inv *detect.Capabilities, req *launchRequest) (restore func()) {
	savedUsed, savedFree, savedResident, savedNano := runtimeVRAMUsedMB, fitDriverFreeVRAM, residentReviewerVRAM, claudeNanoGPUCapable
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
	claudeNanoGPUCapable = func(string) bool {
		if inv == nil {
			return false
		}
		for _, g := range inv.GPUs {
			if strings.Contains(strings.ToLower(g.Name), "nvidia") {
				return true
			}
		}
		return false
	}
	return func() {
		runtimeVRAMUsedMB, fitDriverFreeVRAM, residentReviewerVRAM, claudeNanoGPUCapable = savedUsed, savedFree, savedResident, savedNano
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
	// HostDisplays and ListVulkanUnder serve the Vulkan restriction
	// (vulkanRestrictionApplies): this host's PCI display inventory and a
	// listing's output under a given GGML_VK_VISIBLE_DEVICES. Nil leaves
	// today's order.
	HostDisplays    func() []detect.PCIDisplay
	ListVulkanUnder func(path, visible string) (string, error)
	// Note receives each note when it happens, so a sink that prints keeps
	// today's order against the lines the called functions print directly.
	Note func(planNote)
}

func dryRunPlanInputs(note func(planNote)) planInputs {
	return planInputs{
		Policy:          dryRunPlanPolicy,
		Detect:          detect.Detect,
		Inventory:       loadInventory,
		Model:           parseModel,
		ResolveBackend:  resolveLaunchBackend,
		ListVulkan:      listVulkanDevices,
		HostDisplays:    detect.HostPCIDisplays,
		ListVulkanUnder: listVulkanOutputUnder,
		Note:            note,
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
