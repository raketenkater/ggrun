package main

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
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

// Launch-order golden plans (review W11, step 2, tests only).
//
// launch_plans_golden.json witnesses cmdDryRun's planning order only. This
// file replays cmdLaunch's pre-start planning sequence, from parseLaunchArgs to
// the first buildLaunchServerArgs, with the same fakes as goldenPlanDryRun, and
// records:
//
//   - testdata/launch_plans_launchorder_golden.json: the golden matrix's cases
//     in launch order.
//   - testdata/launch_plans_golden_ext.json: extra cases (GPU restriction with
//     a seated reviewer, --cpu on a GPU host, mmap, calibration and cache
//     switches, seeded caches, launch-only consents) in both orders.
//   - launchOrderDivergence below: for every case, which launch-only step
//     changed the strategy, the request or the outcome, and whether the
//     resulting argv, environment or outcome differs from the dry-run order.
//
// Regenerate (Linux only):
//
//	go test ./cmd/ggrun -run 'TestGoldenPlan(LaunchOrder|ExtendedCases)' -update-launch-order
//
// Without a local build, a missing or differing file is logged as numbered
// gzip+base64 lines with a per-file tag, and a differing divergence table is
// logged as Go literal lines:
//
//	grep -o 'launch-order-golden-b64 [0-9]* [A-Za-z0-9+/=]*' log | sort -k2,2n |
//	  awk '{printf "%s", $3}' | base64 -d | gunzip > testdata/launch_plans_launchorder_golden.json
//	grep -o 'launch-ext-golden-b64 [0-9]* [A-Za-z0-9+/=]*' log | sort -k2,2n |
//	  awk '{printf "%s", $3}' | base64 -d | gunzip > testdata/launch_plans_golden_ext.json
//
// Launch-order fakes, beyond those of goldenPlanDryRun:
//   - the arguments are the dry-run replay's (with --inventory and
//     --emit-server-argv-json), so the request differs only by what the steps
//     do; refuseInventoryOnLaunch is skipped and the inventory replaces
//     detect.Detect.
//   - every consent is non-interactive with no --yes: an empty reader and
//     interactive=false for the reviewer fallback, the proactive reviewer drop
//     and the mmap consent; confirmDenseCPUOffloadWith(non-terminal) for the
//     dense offload prompt; the advisor consent IO and the support helper are
//     stubbed and never run.
//   - the backend offers (reviewed install, unsupported-arch routes,
//     accelerated mainline) are skipped: a non-terminal launch without --yes
//     declines them.
//   - applyVulkanDeviceOrder is replaced by vulkanDeviceOrderEnv with a lister
//     that fails the test (the fixtures are CUDA backends).
//   - containHostMemory gets validateHostMemoryContainment; the launch's
//     ...Waiting validator re-reads live RAM and sleeps on a refusal.
//   - resumeSessionTokens is stubbed (a 200000-token session).
//
// Excluded, because they are lifecycle or need a process, a port or the
// network: prepareRemoteClaude, ensureClaudeBeforeLaunch, exposedHostWarning,
// waitForPredecessorPort, the pending-release wait, guardPortFree,
// waitForBackendPort, and everything from the first buildLaunchServerArgs on
// (recoverPreviousClaudeRuntimeOOM, validateAndRepairBackendArgs, process
// start). rememberClaudeSelfClassify runs; it writes only after an
// interactive approval, which never happens here.

var updateLaunchOrderGolden = flag.Bool("update-launch-order", false, "rewrite testdata/launch_plans_launchorder_golden.json and testdata/launch_plans_golden_ext.json")

const (
	launchOrderGoldenPath = "testdata/launch_plans_launchorder_golden.json"
	launchExtGoldenPath   = "testdata/launch_plans_golden_ext.json"
)

// launchOrderGoldenRAM is the golden inventories' host memory. The extended
// cases add one host with 32 GB, too little to hold the MoE model's experts,
// so the mmap consent and host-memory containment steps have a plan to act on.
const launchOrderGoldenRAM = `"ram":{"total_mb":131072,"free_mb":122880}`

var launchOrderInventories = map[string]string{
	"gpu12-ram32": strings.Replace(goldenPlanInventory(goldenPlanGPU(0, 12288, 16, 504000)),
		launchOrderGoldenRAM, `"ram":{"total_mb":32768,"free_mb":28672}`, 1),
}

// launchOrderCase is a golden case plus what the extended file adds: extra
// environment for the run and a seeded cache ("verified-config" or
// "self-classify-choice").
type launchOrderCase struct {
	goldenPlanCase
	env  map[string]string
	seed string
}

// launchOrderRecord is one case in one order. Plan is the dry run's
// ggrun-server-plan-v1 object, built the same way in both orders; Strategy is
// the part of the plan the JSON does not show (companion seats are json:"-").
type launchOrderRecord struct {
	Case     string               `json:"case"`
	Order    string               `json:"order"`
	Args     []string             `json:"args"`
	Seed     string               `json:"seed,omitempty"`
	Env      map[string]string    `json:"env,omitempty"`
	Outcome  string               `json:"outcome"`
	Error    string               `json:"error,omitempty"`
	Plan     *goldenPlanJSON      `json:"plan,omitempty"`
	Strategy *launchOrderStrategy `json:"strategy,omitempty"`
	// PinningEnv is what the launch adds to the child environment for these
	// argv (hostExpertPinningEnv); the dry run's environment never shows it.
	PinningEnv []string `json:"pinning_env,omitempty"`
}

type launchOrderStrategy struct {
	Type                  placement.StrategyType         `json:"type"`
	ContextSize           int                            `json:"context_size"`
	Parallel              int                            `json:"parallel"`
	MMapRequired          bool                           `json:"mmap_required,omitempty"`
	VerifiedConfigReused  bool                           `json:"verified_config_reused,omitempty"`
	ReviewerReservationMB int                            `json:"reviewer_reservation_mb,omitempty"`
	CompanionSeats        []placement.CompanionPlacement `json:"companion_seats,omitempty"`
}

// launchOrderRun is a replay's record plus what the divergence table needs.
type launchOrderRun struct {
	record    launchOrderRecord
	firstPlan string   // fingerprint of the first plan, before any launch-only step
	steps     []string // launch-only steps that changed or refused the plan
	req       *launchRequest
	strategy  *placement.Strategy
	model     *placement.ModelProfile
	be        *backendInfo
	caps      *detect.Capabilities
}

// launchOrderPrepare seeds a run's fresh cache directory before any step reads it.
type launchOrderPrepare func(t *testing.T, cacheDir string, req *launchRequest, model *placement.ModelProfile, caps *detect.Capabilities)

// launchOrderFingerprint covers the strategy fields the argv is built from and
// the request fields the launch-only steps rewrite.
func launchOrderFingerprint(s *placement.Strategy, req *launchRequest) string {
	if s == nil || req == nil {
		return "nil"
	}
	reserved := 0
	if req.ReviewerReservation != nil {
		reserved = req.ReviewerReservation.VRAMMB
	}
	return fmt.Sprintf("%s ctx=%d np=%d b=%d ub=%d ncmoe=%d ot=%q ts=%v sm=%s mg=%d cram=%d mmap=%t/%t kv=%s/%s ckpt=%d seats=%v vc=%t | ctx=%q np=%d/%t nommap=%t forcemmap=%t norev=%t reserve=%d hostreserve=%d",
		s.Type, s.ContextSize, s.Parallel, s.BatchSize, s.UBatchSize, s.NCPUMoE, s.OTString, s.TensorSplit, s.SplitMode, s.MainGPU, s.CRAM,
		s.MMap, s.MMapRequired, s.KVPlacement, s.KVType, s.MaxCheckpoints, s.CompanionPlacements, s.VerifiedConfigReused,
		req.CtxFlag, req.Parallel, req.ParallelSet, req.NoMMap, req.ForceMMap, req.ClaudeReviewerDisabled, reserved, req.PlacementHostReserveMB)
}

func launchOrderProjection(s *placement.Strategy, req *launchRequest) *launchOrderStrategy {
	if s == nil {
		return nil
	}
	p := &launchOrderStrategy{
		Type: s.Type, ContextSize: s.ContextSize, Parallel: s.Parallel,
		MMapRequired: s.MMapRequired, VerifiedConfigReused: s.VerifiedConfigReused,
		CompanionSeats: append([]placement.CompanionPlacement(nil), s.CompanionPlacements...),
	}
	if req != nil && req.ReviewerReservation != nil {
		p.ReviewerReservationMB = req.ReviewerReservation.VRAMMB
	}
	return p
}

// launchOrderReplay plans one case in dry-run order (as goldenPlanDryRun) or
// in launch order (as cmdLaunch up to its first buildLaunchServerArgs).
func launchOrderReplay(t *testing.T, c launchOrderCase, launch bool, prepare launchOrderPrepare) launchOrderRun {
	t.Helper()
	order := "dry-run"
	if launch {
		order = "launch"
	}
	run := launchOrderRun{record: launchOrderRecord{Case: c.name, Order: order, Args: append([]string{}, c.args...), Seed: c.seed, Env: c.env}}
	if run.record.Args == nil {
		run.record.Args = []string{}
	}
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES"} {
		if err := os.Unsetenv(key); err != nil {
			t.Fatal(err)
		}
	}
	cacheDir := t.TempDir()
	t.Setenv("LLM_CACHE_DIR", cacheDir)
	inventory := filepath.Join(t.TempDir(), "inventory.json")
	hw, ok := goldenPlanInventories[c.hw]
	if !ok {
		hw = launchOrderInventories[c.hw]
	}
	if hw == "" || (!ok && strings.Contains(hw, launchOrderGoldenRAM)) {
		t.Fatalf("%s: no inventory %q, or its RAM was not replaced", c.name, c.hw)
	}
	if err := os.WriteFile(inventory, []byte(hw), 0o600); err != nil {
		t.Fatal(err)
	}
	model := goldenPlanModels[c.model]()
	args := append([]string{model.Path}, c.args...)
	args = append(args, "--inventory", inventory, "--emit-server-argv-json")

	req, err := parseLaunchArgs(args)
	if err != nil {
		run.record.Outcome, run.record.Error = "parse-error", err.Error()
		return run
	}
	caps, err := loadInventory(req.InventoryPath)
	if err != nil {
		t.Fatalf("%s: inventory: %v", c.name, err)
	}
	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("%s: config: %v", c.name, err)
	}
	if cfg.CacheDir != cacheDir {
		t.Fatalf("%s: cache dir %q, want the case's own %q", c.name, cfg.CacheDir, cacheDir)
	}
	req.ModelPath = resolveModelPath(req.ModelPath, cfg.ModelDir)
	warnModelCompatibility(model)
	if prepare != nil {
		prepare(t, cfg.CacheDir, req, model, caps)
	}
	run.req, run.model, run.caps = req, model, caps

	noVulkan := func(string) ([]listedDevice, error) {
		t.Fatalf("%s: a CUDA fixture listed Vulkan devices", c.name)
		return nil, nil
	}
	var strategy *placement.Strategy
	var serverArgs []string
	var envPrefix string
	be := goldenPlanBackend(t, c.backend)
	run.be = be

	if !launch {
		normalizeArchKVRequest(req, model)
		applyBackendFeatureCompatibility(req, model, be)
		preflightBackendArch(model, be, caps, req.AppHome)
		applyCachedBackendCapabilities(req, cfg.CacheDir, model, be)
		applyRememberedClaudeSelfClassify(req, cfg.CacheDir, model, caps, io.Discard)
		req.ReviewerReservation = claudeReviewerReservation(req, caps, cfg.CacheDir)
		strategy, err = placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
		if err != nil {
			run.record.Outcome, run.record.Error = "placement-error", err.Error()
			return run
		}
		claudeCodeSlotAdjust(strategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
		strategy = applyCalibrationDecision(req, cfg, model, be, caps, strategy)
		run.firstPlan = launchOrderFingerprint(strategy, req)
		strategy = explainServingOptimization(req, cfg, model, be, caps, strategy, true)
		run.strategy = strategy
		if refusal := dryRunFitRefusal(strategy); refusal != "" {
			run.record.Outcome, run.record.Error = "fit-refusal", refusal
			return run
		}
		serverArgs = buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
		envPrefix = applyGPUVisibility(req, backendDialect(be))
		if vkEnv, vkErr := vulkanDeviceOrderEnv(req, be, caps, noVulkan); vkErr != nil {
			t.Fatalf("%s: vulkan device order: %v", c.name, vkErr)
		} else if vkEnv != "" {
			envPrefix = vkEnv
		}
	} else {
		req.ProfilePolicyIdentity = requestedLaunchPolicyIdentity(req, model)
		if err := validateRequestedGPUs(caps, req); err != nil {
			run.steps = append(run.steps, "validate-gpus")
			run.record.Outcome, run.record.Error = "gpu-error", err.Error()
			return run
		}
		// resolveLaunchBackend after selection, as in the dry-run replay.
		normalizeArchKVRequest(req, model)
		applyBackendFeatureCompatibility(req, model, be)
		preflightBackendArch(model, be, caps, req.AppHome)
		applyCachedBackendCapabilities(req, cfg.CacheDir, model, be)
		envPrefix = applyGPUVisibility(req, backendDialect(be))
		if vkEnv, vkErr := vulkanDeviceOrderEnv(req, be, caps, noVulkan); vkErr != nil {
			t.Fatalf("%s: vulkan device order: %v", c.name, vkErr)
		} else if vkEnv != "" {
			envPrefix = vkEnv
		}
		forgetClaudeSelfClassifyForExplicitReviewer(req, cfg.CacheDir, model, caps, io.Discard)
		applyRememberedClaudeSelfClassify(req, cfg.CacheDir, model, caps, io.Discard)
		req.ReviewerReservation = claudeReviewerReservation(req, caps, cfg.CacheDir)
		req.DenseOffloadPrompt = confirmDenseCPUOffloadWith(cfg.AssumeYes, strings.NewReader(""), io.Discard, false)

		computeStrategy := func(candidateReq *launchRequest) (*placement.Strategy, error) {
			candidate, computeErr := placement.Compute(caps, model, placementOptionsFromRequestCaps(candidateReq, model, be, cfg.CacheDir, caps))
			if computeErr != nil {
				return nil, computeErr
			}
			return applyCalibrationDecision(candidateReq, cfg, model, be, caps, candidate), nil
		}
		noInput := func() io.Reader { return strings.NewReader("") }

		strategy, err = computeStrategy(req)
		if err != nil {
			run.steps = append(run.steps, "placement-fallback")
			strategy, err = tryResidentWithoutClaudeReviewer(req, err, noInput(), io.Discard, false, computeStrategy)
			if err != nil {
				firstCode := classifyAdvisorFailure(err)
				if replanned, replanErr := deterministicReplanOnPlacementFailure(req, model, be, cfg, caps, nil); replanErr == nil {
					strategy, err = replanned, nil
				} else {
					strategy, err = escalatePlacementFailure(req, cfg, model, be, caps, firstCode, replanErr, computeStrategy)
				}
				if err != nil {
					run.record.Outcome, run.record.Error = "placement-error", err.Error()
					return run
				}
			}
		}
		run.firstPlan = launchOrderFingerprint(strategy, req)
		if strategy.VerifiedConfigReused {
			run.steps = append(run.steps, "verified-replay")
		}
		// step records name when next differs from the current strategy, or
		// when the step rewrote the request.
		step := func(name, before string, next *placement.Strategy) {
			if next != strategy || launchOrderFingerprint(next, req) != before {
				run.steps = append(run.steps, name)
			}
			strategy = next
		}
		before := launchOrderFingerprint(strategy, req)
		step("reviewer-drop", before, proactivelyDropReviewerForVRAMModel(req, caps, model, strategy, computeStrategy, noInput(), io.Discard, false, cfg.CacheDir))
		before = launchOrderFingerprint(strategy, req)
		step("resume-context", before, ensureResumeContext(req, cfg.CacheDir, model, strategy, computeStrategy))
		// Non-interactive, confirmRequiredMMap either passes or refuses; the
		// declined branch (errMMapDeclined) needs an interactive "no".
		if err := confirmRequiredMMap(req, strategy, noInput(), io.Discard, false); err != nil {
			if errors.Is(err, errMMapDeclined) {
				t.Fatalf("%s: a non-interactive mmap consent declined instead of refusing", c.name)
			}
			run.strategy = strategy
			run.steps = append(run.steps, "mmap-consent")
			run.record.Outcome, run.record.Error = "mmap-refusal", err.Error()
			run.record.Strategy = launchOrderProjection(strategy, req)
			return run
		}
		rememberClaudeSelfClassify(req, cfg.CacheDir, model, caps, io.Discard)
		before = launchOrderFingerprint(strategy, req)
		contained, containErr := containHostMemory(req, caps, strategy, computeStrategy, validateHostMemoryContainment)
		if containErr != nil {
			run.strategy = strategy
			run.steps = append(run.steps, "host-memory-containment")
			run.record.Outcome, run.record.Error = "containment-refusal", containErr.Error()
			run.record.Strategy = launchOrderProjection(strategy, req)
			return run
		}
		step("host-memory-containment", before, contained)
		before = launchOrderFingerprint(strategy, req)
		claudeCodeSlotAdjust(strategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
		step("slot-adjust-2", before, strategy)
		run.strategy = strategy
		serverArgs = buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
		run.record.PinningEnv = hostExpertPinningEnv(be, serverArgs)
	}

	run.record.Outcome = "plan"
	run.record.Plan = &goldenPlanJSON{
		Schema:               "ggrun-server-plan-v1",
		ModelPath:            req.ModelPath,
		BackendTag:           be.Tag,
		BackendID:            be.Identity,
		ClaudeProfile:        effectiveClaudeProfile(req),
		LaunchScope:          claudeLaunchLogScope(req, model, be, serverArgs),
		MemoryMaxMB:          backendMemoryMaxMB(req, caps),
		Environment:          launchPlanEnvironment(serverArgs, envPrefix, be.Path),
		ServerArgv:           serverArgs,
		Residency:            strategy.Residency,
		Bottleneck:           strategy.OptimizationBottleneck,
		ResourceLedger:       strategy.ResourceLedger,
		OptimizationBoundary: strategy.OptimizationBoundary,
		CachedOptimization:   req.AppliedCalibration,
	}
	run.record.Strategy = launchOrderProjection(strategy, req)
	return run
}

// launchOrderDivergence is the divergence table: for each case, the launch-only
// steps that changed or refused the plan (in launch order), then which of the
// first plan, outcome, argv and environment differ from the dry-run order.
// "first-plan" means the plan differed before any launch-only step ran
// (caps-aware placement options, the dense-offload hook, calibration's slot
// adjustment order).
var launchOrderDivergence = []string{
	"gpu24/dense-small/auto+verified-config: verified-replay",
	"gpu24/dense-small/auto+verified-config: first-plan",
	"gpu24/dense-small/auto+verified-config: argv",
	"gpu24/dense-small/claude-code+self-review-on: reviewer-drop",
	"gpu24/dense-small/claude-code+self-review-on: argv",
	"mixed3/moe/claude-resume: resume-context",
	"mixed3/moe/claude-resume: argv",
	"gpu12-ram32/moe/auto: mmap-consent",
	"gpu12-ram32/moe/auto: outcome",
	"gpu12-ram32/moe/no-mmap: placement-fallback",
	"gpu12-ram32/moe/claude-code-mmap: placement-fallback",
}

// launchOrderDivergenceOf compares one case's two orders.
func launchOrderDivergenceOf(dry, launch launchOrderRun) []string {
	var out []string
	add := func(step string) { out = append(out, dry.record.Case+": "+step) }
	for _, s := range launch.steps {
		add(s)
	}
	if dry.firstPlan != "" && launch.firstPlan != "" && dry.firstPlan != launch.firstPlan {
		add("first-plan")
	}
	if dry.record.Outcome != launch.record.Outcome {
		add("outcome")
		return out
	}
	if dry.record.Plan != nil && launch.record.Plan != nil {
		if !reflect.DeepEqual(dry.record.Plan.ServerArgv, launch.record.Plan.ServerArgv) {
			add("argv")
		}
		if !reflect.DeepEqual(dry.record.Plan.Environment, launch.record.Plan.Environment) {
			add("environment")
		}
	}
	return out
}

// launchOrderTestEnv is TestGoldenPlanSyntheticMatrix's environment plus the
// switches and seams only the launch-only steps read.
func launchOrderTestEnv(t *testing.T) string {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("golden launch plans are recorded on Linux")
	}
	home := t.TempDir()
	unset := []string{
		"LD_LIBRARY_PATH", "LLM_SERVER_LIB_HUB", "CUDA_DEVICE_ORDER", "CUDA_SCALE_LAUNCH_QUEUES",
		"CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES", "GGML_OP_OFFLOAD_MIN_BATCH",
		"LLAMA_ARG_LOG_VERBOSITY", "GGRUN_CLAUDE_AUTO_REVIEWER", "GGRUN_CLAUDE_REVIEWER_MODEL",
		"GGRUN_CLAUDE_REVIEWER_BIN", "GGRUN_TRACE_PLACEMENT", "LLM_SERVER_SKIP_MMPROJ_CHECK",
		"GGRUN_CLAUDE_AUTO_SELF_REVIEW", "GGRUN_ADVISOR_CONSENT",
	}
	for _, key := range config.EnvKeys {
		unset = append(unset, key, config.GGRUNAlias(key))
	}
	goldenPlanUnsetenv(t, unset...)
	t.Setenv("HOME", home)
	t.Setenv("XDG_CACHE_HOME", filepath.Join(home, ".cache"))
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(home, ".config"))
	t.Setenv("LLM_CONFIG", filepath.Join(home, "missing-config"))
	t.Setenv("LLM_APP_HOME", filepath.Join(home, "app"))
	t.Setenv("LLM_SERVER_SKIP_DRAFT_DOWNLOAD", "1")
	restoreResident, restoreResume := residentReviewerVRAM, resumeSessionTokens
	residentReviewerVRAM = func(string) int { return 0 }
	resumeSessionTokens = func(_, value string) (string, int, bool) { return value, 200000, true }
	t.Cleanup(func() { residentReviewerVRAM, resumeSessionTokens = restoreResident, restoreResume })
	stubPlacementAdvisorConsentIO(t, "", false)
	called := stubSupportIncidentCount(t)
	t.Cleanup(func() {
		if *called != 0 {
			t.Errorf("the support helper ran %d times during a replay", *called)
		}
	})
	return home
}

// launchOrderRunCase runs one case's subtest: env, live-VRAM stubs, then fn.
func launchOrderRunCase(t *testing.T, c launchOrderCase, fn func(t *testing.T)) {
	t.Helper()
	t.Run(c.name, func(t *testing.T) {
		goldenPlanUnsetenv(t, "CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES")
		for k, v := range c.env {
			t.Setenv(k, v)
		}
		goldenPlanStubLiveVRAM(t)
		fn(t)
	})
}

func launchOrderMarshal(t *testing.T, record launchOrderRecord) []byte {
	t.Helper()
	b, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// launchOrderCheckStable fails when two runs of one order disagree or when a
// record carries a temporary path.
func launchOrderCheckStable(t *testing.T, home string, first, second launchOrderRun) {
	t.Helper()
	a, b := launchOrderMarshal(t, first.record), launchOrderMarshal(t, second.record)
	if !bytes.Equal(a, b) {
		t.Errorf("%s (%s): two identical runs disagree\nfirst:  %s\nsecond: %s", first.record.Case, first.record.Order, a, b)
	}
	if !reflect.DeepEqual(first.steps, second.steps) || first.firstPlan != second.firstPlan {
		t.Errorf("%s (%s): two identical runs took different steps: %v / %v", first.record.Case, first.record.Order, first.steps, second.steps)
	}
	if bytes.Contains(a, []byte(t.Name())) || bytes.Contains(a, []byte(home)) {
		t.Errorf("%s (%s): record carries a temporary path, so no golden can hold it: %s", first.record.Case, first.record.Order, a)
	}
}

// launchOrderCompareFile compares records with a golden file. Every outcome
// short of a match logs the computed file under tag; it reports with Errorf so
// a recording run emits every file and the table before failing.
func launchOrderCompareFile(t *testing.T, path, tag string, records []launchOrderRecord) {
	t.Helper()
	got, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	if *updateLaunchOrderGolden {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		launchOrderLogEncoded(t, tag, got)
		t.Errorf("%s is missing; recover it from the %s lines above or regenerate with -update-launch-order", path, tag)
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(got, want) {
		return
	}
	var wantRecords []launchOrderRecord
	if err := json.Unmarshal(want, &wantRecords); err != nil {
		launchOrderLogEncoded(t, tag, got)
		t.Errorf("%s is not a launch-order record list: %v", path, err)
		return
	}
	for i := 0; i < max(len(records), len(wantRecords)); i++ {
		var g, w []byte
		if i < len(records) {
			g = launchOrderMarshal(t, records[i])
		}
		if i < len(wantRecords) {
			w = launchOrderMarshal(t, wantRecords[i])
		}
		if !bytes.Equal(g, w) {
			t.Errorf("%s record %d changed\nwant: %s\ngot:  %s", path, i, w, g)
		}
	}
	launchOrderLogEncoded(t, tag, got)
	t.Errorf("launch-order records differ from %s; if the change is intended, regenerate with -update-launch-order", path)
}

// launchOrderCheckDivergence compares the computed table with the cases'
// entries of launchOrderDivergence.
func launchOrderCheckDivergence(t *testing.T, cases []launchOrderCase, got []string) {
	t.Helper()
	names := map[string]bool{}
	for _, c := range cases {
		names[c.name] = true
	}
	want := []string{}
	for _, entry := range launchOrderDivergence {
		name, _, _ := strings.Cut(entry, ": ")
		if names[name] {
			want = append(want, entry)
		}
	}
	if got == nil {
		got = []string{}
	}
	if reflect.DeepEqual(got, want) {
		return
	}
	var b strings.Builder
	for _, entry := range got {
		fmt.Fprintf(&b, "\n\t%q,", entry)
	}
	t.Logf("launch-order divergence, computed:%s", b.String())
	t.Errorf("launch-order divergence differs from launchOrderDivergence\nwant: %q\ngot:  %q", want, got)
}

// launchOrderLogEncoded is goldenPlanLogEncoded with a per-file tag.
func launchOrderLogEncoded(t *testing.T, tag string, data []byte) {
	t.Helper()
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	if _, err := zw.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	encoded := base64.StdEncoding.EncodeToString(buf.Bytes())
	const width = 1000
	chunks := (len(encoded) + width - 1) / width
	t.Logf("%s sha256 %x bytes %d chunks %d", tag, sha256.Sum256(data), len(data), chunks)
	for i := 0; i < chunks; i++ {
		t.Logf("%s-b64 %04d %s", tag, i, encoded[i*width:min((i+1)*width, len(encoded))])
	}
}

// TestGoldenPlanLaunchOrder records the golden matrix in launch order. Its
// dry-run replay is checked against launch_plans_golden.json, so the two
// orders are compared on the same fakes.
func TestGoldenPlanLaunchOrder(t *testing.T) {
	home := launchOrderTestEnv(t)
	wantDry, err := os.ReadFile(goldenPlanPath)
	if err != nil {
		t.Fatal(err)
	}
	var dryGolden []goldenPlanRecord
	if err := json.Unmarshal(wantDry, &dryGolden); err != nil {
		t.Fatal(err)
	}
	base := goldenPlanCases()
	if len(dryGolden) != len(base) {
		t.Fatalf("%s holds %d records for %d cases", goldenPlanPath, len(dryGolden), len(base))
	}
	var cases []launchOrderCase
	var records []launchOrderRecord
	var divergence []string
	for i, gc := range base {
		c := launchOrderCase{goldenPlanCase: gc}
		cases = append(cases, c)
		var dry, first, second launchOrderRun
		launchOrderRunCase(t, c, func(t *testing.T) {
			dry = launchOrderReplay(t, c, false, nil)
			first = launchOrderReplay(t, c, true, nil)
			second = launchOrderReplay(t, c, true, nil)
		})
		dryRecord := goldenPlanRecord{Case: dry.record.Case, Args: dry.record.Args, Outcome: dry.record.Outcome, Error: dry.record.Error, Plan: dry.record.Plan}
		if g, w := goldenPlanMarshal(t, dryRecord), goldenPlanMarshal(t, dryGolden[i]); !bytes.Equal(g, w) {
			t.Errorf("%s: the dry-run replay here no longer matches %s\nwant: %s\ngot:  %s", c.name, goldenPlanPath, w, g)
		}
		launchOrderCheckStable(t, home, first, second)
		records = append(records, first.record)
		divergence = append(divergence, launchOrderDivergenceOf(dry, first)...)
	}
	launchOrderCompareFile(t, launchOrderGoldenPath, "launch-order-golden", records)
	launchOrderCheckDivergence(t, cases, divergence)
}

// launchOrderExtCases are the extended cases. --gpus with a seated reviewer
// exercises the companion seat under a restriction; the --mmap variants let
// the steps after the mmap consent run.
func launchOrderExtCases() []launchOrderCase {
	mk := func(name, backend, hw, model string, args ...string) launchOrderCase {
		return launchOrderCase{goldenPlanCase: goldenPlanCase{name: name, backend: backend, hw: hw, model: model, args: args}}
	}
	selfReview := mk("gpu24/dense-small/claude-code+self-review-on", "", "gpu24", "dense-small", "--claude-code")
	selfReview.env = map[string]string{"GGRUN_CLAUDE_AUTO_SELF_REVIEW": "on"}
	verified := mk("gpu24/dense-small/auto+verified-config", "", "gpu24", "dense-small")
	verified.seed = "verified-config"
	choice := mk("gpu12/dense-small/claude-code+self-classify-choice", "", "gpu12", "dense-small", "--claude-code")
	choice.seed = "self-classify-choice"
	return []launchOrderCase{
		mk("mixed3/dense-small/claude-code-gpus-1", "", "mixed3", "dense-small", "--claude-code", "--gpus", "1"),
		mk("mixed3/moe/claude-code-gpus-1,2", "", "mixed3", "moe", "--claude-code", "--gpus", "1,2"),
		mk("mixed3/dense-large/claude-code-gpus-0,2", "", "mixed3", "dense-large", "--claude-code", "--gpus", "0,2"),
		mk("gpu24/dense-small/cpu", "", "gpu24", "dense-small", "--cpu"),
		mk("mixed3/moe/no-mmap", "", "mixed3", "moe", "--no-mmap"),
		mk("gpu24/moe/claude-code-calibrate-off", "", "gpu24", "moe", "--claude-code", "--calibrate", "off"),
		mk("gpu24/moe/claude-code-no-cached-config", "", "gpu24", "moe", "--claude-code", "--no-cached-config"),
		mk("ik-backend/gpu24/moe/auto", "ik", "gpu24", "moe"),
		verified,
		choice,
		mk("gpu12/dense-large/claude-code", "", "gpu12", "dense-large", "--claude-code"),
		mk("gpu12/dense-large/mmap", "", "gpu12", "dense-large", "--mmap"),
		mk("gpu24/dense-small/claude-code", "", "gpu24", "dense-small", "--claude-code"),
		selfReview,
		mk("mixed3/moe/claude-code-mmap", "", "mixed3", "moe", "--claude-code", "--mmap"),
		mk("mixed3/moe/claude-resume", "", "mixed3", "moe", "--claude-resume", "golden-session"),
		mk("gpu12-ram32/moe/auto", "", "gpu12-ram32", "moe"),
		mk("gpu12-ram32/moe/mmap", "", "gpu12-ram32", "moe", "--mmap"),
		mk("gpu12-ram32/moe/no-mmap", "", "gpu12-ram32", "moe", "--no-mmap"),
		mk("gpu12-ram32/dense-large/no-mmap", "", "gpu12-ram32", "dense-large", "--no-mmap"),
		mk("gpu12-ram32/moe/claude-code-mmap", "", "gpu12-ram32", "moe", "--claude-code", "--mmap"),
	}
}

// launchOrderSeed returns the prepare hook for a case's seed. A verified
// config is the case's own launch-order plan with half its context, saved the
// way a promoted launch saves it, so a replay is visible in the argv.
func launchOrderSeed(t *testing.T, c launchOrderCase) launchOrderPrepare {
	t.Helper()
	switch c.seed {
	case "":
		return nil
	case "self-classify-choice":
		return func(t *testing.T, cacheDir string, req *launchRequest, model *placement.ModelProfile, caps *detect.Capabilities) {
			if err := saveClaudeReviewerChoice(cacheDir, claudeReviewerChoiceScope(req, model, caps), claudeReviewerChoiceSelfClassify); err != nil {
				t.Fatalf("%s: seed reviewer choice: %v", c.name, err)
			}
		}
	case "verified-config":
		seed := launchOrderReplay(t, c, true, nil)
		if seed.record.Outcome != "plan" || seed.strategy == nil {
			t.Fatalf("%s: the seeding launch did not plan: %s %s", c.name, seed.record.Outcome, seed.record.Error)
		}
		saved := *seed.strategy
		saved.ContextSize = seed.strategy.ContextSize / 2
		return func(t *testing.T, cacheDir string, _ *launchRequest, _ *placement.ModelProfile, _ *detect.Capabilities) {
			saveVerifiedConfigForLaunch(&config.Config{CacheDir: cacheDir}, seed.req, seed.model, seed.be, seed.caps, &saved, nil)
			if _, err := os.Stat(placement.VerifiedConfigPath(cacheDir, verifiedConfigScopeKey(seed.req, seed.model, seed.be, seed.caps))); err != nil {
				t.Fatalf("%s: seed verified config: %v", c.name, err)
			}
		}
	}
	t.Fatalf("%s: unknown seed %q", c.name, c.seed)
	return nil
}

// TestGoldenPlanExtendedCases records the extended cases in both orders.
func TestGoldenPlanExtendedCases(t *testing.T) {
	home := launchOrderTestEnv(t)
	cases := launchOrderExtCases()
	var records []launchOrderRecord
	var divergence []string
	for _, c := range cases {
		var dry, dry2, launch, launch2 launchOrderRun
		launchOrderRunCase(t, c, func(t *testing.T) {
			prepare := launchOrderSeed(t, c)
			dry = launchOrderReplay(t, c, false, prepare)
			dry2 = launchOrderReplay(t, c, false, prepare)
			launch = launchOrderReplay(t, c, true, prepare)
			launch2 = launchOrderReplay(t, c, true, prepare)
		})
		launchOrderCheckStable(t, home, dry, dry2)
		launchOrderCheckStable(t, home, launch, launch2)
		// A seed must bite, or the case witnesses nothing.
		switch c.seed {
		case "verified-config":
			if launch.record.Strategy == nil || !launch.record.Strategy.VerifiedConfigReused {
				t.Errorf("%s: the launch did not replay the seeded verified config: %+v", c.name, launch.record.Strategy)
			}
		case "self-classify-choice":
			for _, r := range []launchOrderRun{dry, launch} {
				if r.req == nil || !r.req.ClaudeReviewerDisabled || r.req.ReviewerReservation != nil {
					t.Errorf("%s (%s): the remembered self-classify choice did not drop the reviewer", c.name, r.record.Order)
				}
			}
		}
		records = append(records, dry.record, launch.record)
		divergence = append(divergence, launchOrderDivergenceOf(dry, launch)...)
	}
	launchOrderCompareFile(t, launchExtGoldenPath, "launch-ext-golden", records)
	launchOrderCheckDivergence(t, cases, divergence)
}
