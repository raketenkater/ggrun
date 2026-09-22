package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/advisor"
	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/recovery"
	"github.com/raketenkater/ggrun/pkg/server"
)

type supportStatus struct {
	Policy          string           `json:"policy"`
	OnlineResearch  bool             `json:"online_research"`
	Roles           []string         `json:"roles"`
	Model           advisor.Artifact `json:"model"`
	ModelPath       string           `json:"model_path"`
	ModelVerified   bool             `json:"model_verified"`
	ModelError      string           `json:"model_error,omitempty"`
	BackendPath     string           `json:"backend_path,omitempty"`
	BackendVerified bool             `json:"backend_verified"`
	BackendError    string           `json:"backend_error,omitempty"`
	Ready           bool             `json:"ready"`
	WorkerReady     bool             `json:"worker_ready"`
	RuntimePolicy   string           `json:"runtime_policy"`
}

func cmdSupport(args []string) {
	subcommand := "status"
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		subcommand = strings.ToLower(args[0])
		args = args[1:]
	}
	switch subcommand {
	case "help", "-h", "--help":
		printSupportUsage()
	case "status", "doctor":
		cfg := loadConfigOrExit()
		status := currentSupportStatus(cfg, nil)
		if hasArg(args, "--json") {
			data, _ := json.MarshalIndent(status, "", "  ")
			fmt.Println(string(data))
			return
		}
		printSupportStatus(status)
		if subcommand == "doctor" && !status.Ready {
			os.Exit(1)
		}
	case "install":
		installSupportExpert(args)
	case "analyze", "optimize":
		if len(args) == 0 {
			fmt.Fprintf(os.Stderr, "Usage: ggrun support %s <typed-incident.json> [--online]\n", subcommand)
			os.Exit(2)
		}
		analyzeSupportIncident(args[0], subcommand == "optimize", hasArg(args[1:], "--online"))
	case "latest":
		cfg := loadConfigOrExit()
		data, err := os.ReadFile(filepath.Join(cfg.CacheDir, "advisor", "latest.json"))
		if err != nil {
			fmt.Fprintf(os.Stderr, "No support analysis recorded: %v\n", err)
			os.Exit(1)
		}
		_, _ = os.Stdout.Write(data)
	default:
		printSupportUsage()
		os.Exit(2)
	}
}

func printSupportUsage() {
	fmt.Fprint(os.Stderr, `Usage: ggrun support <subcommand>

  status [--json]        Show model/backend integrity and native controller policy
  install [--model-only] Install the pinned NanoBeige artifact and, when needed,
                         a separate pinned CPU llama.cpp backend that supports it
  doctor                 Exit non-zero unless artifact and backend are verified
  analyze <incident>     Analyze a typed support incident (optional --online)
  optimize <incident>    Rank only ggrun-generated candidates in a typed incident
  latest                 Print the most recent sanitized analysis record

Diagnostic/optimizer runs are CPU-only and unload before main-model placement.
In Claude Code mode the same verified artifact may instead occupy a separately
reserved worker seat for Auto review and explicit cheap-tier work. It has no
shell/flag authority in either role.
`)
}

func advisorModelPath(cfg *config.Config) string {
	if cfg != nil && strings.TrimSpace(cfg.SupportModel) != "" {
		return filepath.Clean(cfg.SupportModel)
	}
	if cfg == nil {
		return ""
	}
	return advisor.ArtifactPath(cfg.CacheDir, advisor.DefaultArtifact)
}

func currentSupportStatus(cfg *config.Config, caps *detect.Capabilities) supportStatus {
	status := supportStatus{
		Policy: "auto", Roles: []string{"support_expert", "candidate_optimizer"}, Model: advisor.DefaultArtifact,
		RuntimePolicy: "diagnostics use an ephemeral CPU-only helper released before placement; Claude Code may reserve a separate measured worker seat",
	}
	if cfg != nil {
		status.Policy = cfg.SupportExpert
		status.OnlineResearch = cfg.SupportOnline
	}
	status.ModelPath = advisorModelPath(cfg)
	if err := advisor.VerifyArtifact(status.ModelPath, advisor.DefaultArtifact); err != nil {
		status.ModelError = err.Error()
	} else {
		status.ModelVerified = true
	}
	if caps == nil {
		caps, _ = detect.Detect()
	}
	backendPath, backendErr := reviewedSupportBackend()
	status.BackendPath = backendPath
	if backendErr != nil {
		status.BackendError = backendErr.Error()
	} else {
		status.BackendVerified = true
	}
	status.Ready = status.ModelVerified && status.BackendVerified
	status.WorkerReady = status.Ready && claudeNanoGPUCapable(status.BackendPath)
	if status.WorkerReady {
		status.Roles = append(status.Roles, "claude_code_worker")
	}
	return status
}

func printSupportStatus(status supportStatus) {
	fmt.Println("ggrun native support expert / optimizer")
	fmt.Printf("  policy:          %s\n", status.Policy)
	fmt.Printf("  roles:           %s\n", strings.Join(status.Roles, ", "))
	fmt.Printf("  online research: %t (official llama.cpp sources only)\n", status.OnlineResearch)
	if status.ModelVerified {
		fmt.Printf("  model:           verified %s (%s, SHA-256 %s…)\n", status.ModelPath, status.Model.Quantization, status.Model.SHA256[:12])
	} else {
		fmt.Printf("  model:           not ready — %s\n", status.ModelError)
	}
	if status.BackendVerified {
		fmt.Printf("  backend:         verified reviewed nanbeige support — %s\n", status.BackendPath)
	} else {
		fmt.Printf("  backend:         not ready — %s\n", status.BackendError)
	}
	fmt.Printf("  main runtime:    %s\n", status.RuntimePolicy)
	fmt.Printf("  ready:           %t\n", status.Ready)
	fmt.Printf("  Claude worker:   %t (requires a GPU-capable reviewed backend)\n", status.WorkerReady)
	if !status.ModelVerified {
		fmt.Println("  install:         ggrun support install")
	} else if !status.BackendVerified {
		fmt.Println("  backend install: ggrun backend install nanbeige42")
	}
}

func installSupportExpert(args []string) {
	cfg := loadConfigOrExit()
	modelOnly := hasArg(args, "--model-only")
	for _, arg := range args {
		if arg != "--model-only" {
			fmt.Fprintf(os.Stderr, "unknown support install option %q\n", arg)
			os.Exit(2)
		}
	}
	fmt.Printf("[support] installing pinned %s (%s, %.2f GiB, %s)\n",
		advisor.DefaultArtifact.Model, advisor.DefaultArtifact.Quantization,
		float64(advisor.DefaultArtifact.SizeBytes)/(1024*1024*1024), advisor.DefaultArtifact.License)
	lastPercent := -5
	path, err := (advisor.Installer{}).Install(context.Background(), cfg.CacheDir, advisor.DefaultArtifact,
		func(received, total int64) {
			if total <= 0 {
				return
			}
			percent := int(received * 100 / total)
			if percent >= lastPercent+5 || percent == 100 {
				fmt.Printf("[support] model download %d%%\n", percent)
				lastPercent = percent
			}
		})
	if err != nil {
		fmt.Fprintf(os.Stderr, "support model install failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("[support] verified artifact: %s\n", path)
	status := currentSupportStatus(cfg, nil)
	if !status.BackendVerified && !modelOnly {
		fmt.Println("[support] no installed backend proves nanbeige support; building the pinned official backend for this host")
		cmdBackendInstall([]string{"nanbeige42"})
		status = currentSupportStatus(cfg, nil)
	}
	printSupportStatus(status)
	if !status.Ready && !modelOnly {
		os.Exit(1)
	}
}

func findSupportBackend(cfg *config.Config, caps *detect.Capabilities, preferred string) string {
	seen := map[string]bool{}
	var candidates []string
	add := func(path string) {
		path = strings.TrimSpace(path)
		if path == "" || seen[path] {
			return
		}
		seen[path] = true
		candidates = append(candidates, path)
	}
	if be := backends.ByTag("nanbeige42"); be != nil {
		add(be.Path)
	}
	add(preferred)
	if cfg != nil {
		add(cfg.LlamaServer)
		for _, path := range backendSearchPaths(cfg.AppHome) {
			add(path)
		}
	}
	for _, be := range backends.Load() {
		add(be.Path)
	}
	if caps != nil {
		for _, be := range caps.Backends {
			add(be.Path)
		}
	}
	if path, err := exec.LookPath("llama-server"); err == nil {
		add(path)
	}
	for _, path := range candidates {
		if info, err := os.Stat(path); err != nil || info.IsDir() {
			continue
		}
		if supported, probed := backends.BackendSupportsArch(path, "nanbeige"); probed && supported {
			return path
		}
	}
	return ""
}

func reviewedSupportBackend() (string, error) {
	recipe := backends.RecipeByName("nanbeige42")
	if recipe == nil {
		return "", errors.New("reviewed nanbeige42 backend recipe is unavailable")
	}
	registered := backends.ByTag(recipe.Tag)
	if registered == nil {
		return "", errors.New("reviewed nanbeige42 backend is not installed")
	}
	if !strings.EqualFold(strings.TrimSpace(registered.GitURL), strings.TrimSpace(recipe.GitURL)) ||
		!strings.EqualFold(strings.TrimSpace(registered.Commit), strings.TrimSpace(recipe.Commit)) {
		return "", fmt.Errorf("registered nanbeige42 backend lacks reviewed source provenance (need commit %s)", recipe.Commit[:12])
	}
	if !sameStrings(registered.AppliedPatches, recipe.PatchNames()) {
		return "", errors.New("registered nanbeige42 backend lacks the reviewed GGUF compatibility patch; run ggrun backend update nanbeige42")
	}
	if !registered.HelperOnly || !strings.EqualFold(strings.TrimSpace(registered.RouteArch), strings.TrimSpace(recipe.RouteArch)) {
		return "", errors.New("registered nanbeige42 backend does not preserve helper-only routing policy")
	}
	info, err := os.Stat(registered.Path)
	if err != nil || info.IsDir() {
		if err == nil {
			err = errors.New("path is a directory")
		}
		return "", fmt.Errorf("reviewed nanbeige42 backend is unavailable: %w", err)
	}
	if supported, probed := backends.BackendSupportsArch(registered.Path, "nanbeige"); !probed || !supported {
		return "", errors.New("reviewed nanbeige42 binary fails architecture conformance")
	}
	return registered.Path, nil
}

func analyzeSupportIncident(path string, forceOptimizer, online bool) {
	file, err := os.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open typed incident: %v\n", err)
		os.Exit(1)
	}
	defer file.Close()
	decoder := json.NewDecoder(io.LimitReader(file, 2<<20))
	decoder.DisallowUnknownFields()
	var incident advisor.Incident
	if err := decoder.Decode(&incident); err != nil {
		fmt.Fprintf(os.Stderr, "decode typed incident: %v\n", err)
		os.Exit(1)
	}
	if forceOptimizer {
		incident.Mode = advisor.ModeOptimizer
	}
	if err := incident.Normalize(); err != nil {
		fmt.Fprintf(os.Stderr, "invalid typed incident: %v\n", err)
		os.Exit(1)
	}
	cfg := loadConfigOrExit()
	caps, _ := detect.Detect()
	decision, report, runErr := runSupportIncidentFn(context.Background(), cfg, caps, "", incident, online || cfg.SupportOnline)
	var decisionPtr *advisor.Decision
	if runErr == nil {
		decisionPtr = &decision
	}
	history, historyErr := advisor.SaveAnalysis(cfg.CacheDir, incident, decisionPtr, report, runErr)
	if historyErr != nil {
		fmt.Fprintf(os.Stderr, "[support] history save failed: %v\n", historyErr)
	}
	if runErr != nil {
		fmt.Fprintf(os.Stderr, "support analysis failed: %v\n", runErr)
		os.Exit(1)
	}
	data, _ := json.MarshalIndent(decision, "", "  ")
	fmt.Println(string(data))
	fmt.Printf("[support] helper stopped; release verified=%t; analysis=%s\n", report.ReleaseVerified, history)
}

func runSupportIncident(ctx context.Context, cfg *config.Config, caps *detect.Capabilities, preferredBackend string, incident advisor.Incident, online bool) (advisor.Decision, advisor.RunReport, error) {
	if cfg == nil {
		return advisor.Decision{}, advisor.RunReport{}, errors.New("support expert has no configuration")
	}
	backendPath, backendErr := reviewedSupportBackend()
	if backendErr != nil {
		return advisor.Decision{}, advisor.RunReport{}, backendErr
	}
	runner := advisor.Runner{
		BackendPath: backendPath, ModelPath: advisorModelPath(cfg), Artifact: advisor.DefaultArtifact,
		CacheDir: cfg.CacheDir, Online: online, MemoryMaxMB: 8192,
	}
	return runner.Analyze(ctx, incident)
}

// runSupportIncidentFn is the support-incident execution seam. Production runs
// the reviewed helper backend via runSupportIncident; tests override this
// package variable to record incidents or return canned decisions without
// launching a helper process.
var runSupportIncidentFn = runSupportIncident

// escalationSignal is the deterministic-recovery state threaded into the
// Layer-2 escalation gate. retriesExhausted reports that the deterministic
// controller has already spent its full recovery budget for this failure;
// sameCodeRecurred reports that the same classified failure class was observed
// again after a deterministic attempt. Both are computed by the caller from
// deterministic evidence, never from the advisor.
type escalationSignal struct {
	retriesExhausted bool
	sameCodeRecurred bool
}

// shouldEscalateToAdvisor is the Layer-2 gate: the advisor is consulted only
// for genuinely novel/unclassified failures, or a classified failure that
// keeps recurring after the deterministic recovery budget is exhausted.
// Classified failures that have not exhausted the deterministic budget stay
// with the controller's own recovery paths. The OR is load-bearing: a novel
// failure escalates on its own, regardless of retry state.
func shouldEscalateToAdvisor(code string, signal escalationSignal) bool {
	// A deterministic environment/flag class is the controller's own problem and
	// no bounded placement reshape fixes it; escalating would only spend a helper
	// process and online budget to re-derive "no_action". Never consult the
	// advisor for these.
	if deterministicFailureCode(code) {
		return false
	}
	if code == "unclassified_launch_failure" {
		return true
	}
	return signal.retriesExhausted && signal.sameCodeRecurred
}

// ConsentTier is the Layer-3 cost class the controller attaches to an advisor
// action before asking for approval. Cheap actions only reshape placement the
// next time the deterministic engine runs (auto-approved in an interactive
// launch, still required explicitly when non-interactive); expensive actions
// consume a helper process, online research budget, and a full recompute of a
// previously-working plan, so they always prompt.
type ConsentTier int

const (
	ConsentTierCheap     ConsentTier = iota // deterministic placement reshape only
	ConsentTierExpensive                    // helper process + full recompute of a working plan
)

// AdvisorConsentPlan is the Layer-3 approval request the controller builds
// before consulting the advisor on an escalated failure. It is pure data: the
// escalation is already classified and the action the advisor may take is
// bounded by ValidateDecision and applyAdvisorDecision before any part of it
// reaches a launch. The plan only describes the consultation to a human and
// records its cost class. It never carries a command line, and no part of it is
// ever fed back to a shell.
type AdvisorConsentPlan struct {
	Action  advisor.ActionID // empty for a pre-consultation plan; set when gating a specific action
	Feature advisor.FeatureID
	Tier    ConsentTier
	Summary string
}

// advisorConsentPlanForEscalation classifies the Layer-3 cost of consulting the
// advisor for a failure class, BEFORE the advisor runs. A classified, recurring
// failure is the cheap case: deterministic evidence already bounds what the
// advisor could say to a placement reshape, so an interactive launch
// auto-approves the consultation. A novel/unclassified failure is the expensive
// case: the advisor needs its helper model and online research budget, so it
// always prompts. Every consultation runs a helper process, so the tier is the
// escalation's cost class, not the advisor's eventual (still bounded) answer.
func advisorConsentPlanForEscalation(code string) AdvisorConsentPlan {
	plan := AdvisorConsentPlan{}
	switch code {
	case "unclassified_launch_failure":
		plan.Tier = ConsentTierExpensive
		plan.Summary = "investigate a novel launch failure with its support model and online research, then re-plan the launch"
	case "cuda_oom_after_deterministic_recovery":
		plan.Tier = ConsentTierExpensive
		plan.Summary = "consult its support model about the repeated GPU out-of-memory failure and re-plan the launch"
	default:
		plan.Tier = ConsentTierCheap
		plan.Summary = "consult its support model about the recurring " + code + " failure and reshape the launch"
	}
	return plan
}

// advisorConsentTierForAction classifies the Layer-3 cost of a SPECIFIC bounded
// advisor action. Every controller-owned placement action — the three new ones
// and the existing reshaples — is a deterministic reshape: it mutates typed
// request state (ubatch rung, VRAM penalty, a passthrough bool flag) that the
// next Compute consumes, so it is ConsentTierCheap and, on an interactive
// launch, does not prompt. The tier is attached to a plan via
// advisorConsentPlanForAction for post-consultation gating, and is deliberately
// independent of the pre-consultation escalation tier, which describes the
// consultation itself.
func advisorConsentTierForAction(action advisor.ActionID) ConsentTier {
	_ = action // the whole current controller-owned taxonomy is cheap placement reshaples
	return ConsentTierCheap
}

// advisorConsentPlanForAction builds the Layer-3 approval plan that gates a
// specific bounded advisor action after the consultation has returned it. The
// tier comes from advisorConsentTierForAction (cheap reshaples auto-approve
// interactively), and the summary names the action so the human knows exactly
// which bounded knob the advisor wants to turn.
func advisorConsentPlanForAction(action advisor.ActionID, feature advisor.FeatureID) AdvisorConsentPlan {
	plan := AdvisorConsentPlan{Action: action, Feature: feature, Tier: advisorConsentTierForAction(action)}
	switch action {
	case advisor.ActionProposeUBatch:
		plan.Summary = "set a concrete micro-batch derate rung from the {256,128,64} ladder"
	case advisor.ActionProposeLayerDistribution:
		plan.Summary = "rebalance expert layers off a target device to reclaim VRAM"
	case advisor.ActionToggleSWAFull:
		plan.Summary = "toggle the generated swa-full feature"
	default:
		plan.Summary = "apply its bounded " + string(action) + " adjustment and reshape the launch"
	}
	return plan
}

// advisorConsentRerunHint maps the blocked advisor consultation back to the
// explicit opt-in a non-interactive user can pass. It is the Layer-3 analogue
// of the --allow-live-memory-probe / --mmap rerun hints in the other consent
// gates: the gate is the GGRUN_ADVISOR_CONSENT environment variable, so the
// hint names that. A feature-specific deterministic alternative is offered when
// a single flag turns the generated feature off without the advisor at all.
func advisorConsentRerunHint(plan AdvisorConsentPlan) string {
	if plan.Feature == advisor.FeatureSWAFull &&
		(plan.Action == advisor.ActionRemoveGeneratedFeature || plan.Action == advisor.ActionToggleSWAFull) {
		return "rerun with GGRUN_ADVISOR_CONSENT=on, or --no-swa-full to remove the generated feature deterministically"
	}
	return "rerun with GGRUN_ADVISOR_CONSENT=on to approve advisor actions"
}

// ErrAdvisorDeclined is the Layer-3 consent sentinel. A declined advisor action
// is a policy choice, not a synthetic failure: the caller must return the
// ORIGINAL launch error unchanged (never a new one naming this sentinel) so a
// user who said no sees the real reason their launch failed, exactly like
// errMMapDeclined and the live-memory-probe decline.
var ErrAdvisorDeclined = errors.New("advisor action declined")

// advisorConsentPolicy is the GGRUN_ADVISOR_CONSENT gate. "prompt" (default)
// asks interactively and fails closed when stdin is not a terminal; "on"
// auto-approves every advisor action without asking; "off" auto-declines every
// advisor action so the advisor is never consulted and the original launch
// error is returned. Invalid values are treated as the default so a typo can
// never silently weaken the gate into auto-approval.
func advisorConsentPolicy() string {
	policy := strings.ToLower(strings.TrimSpace(os.Getenv("GGRUN_ADVISOR_CONSENT")))
	switch policy {
	case "prompt", "on", "off":
		return policy
	default:
		return "prompt"
	}
}

// confirmAdvisorAction is the Layer-3 user-consent gate, the fourth sibling of
// the consent family (confirmRequiredMMap, confirmMainModelReviewerFallback,
// confirmLiveMemoryProbe). It approves only the *decision to consult the
// advisor* and the bounded action the advisor may take — it never approves a
// raw command line, and it is orthogonal to resource-safety verification: a
// declined action must not weaken the ErrResourceReleaseUnverified boundary,
// and the caller must not restart the main model until helper release is proven.
//
// Cheap tiers auto-approve when the launch is interactive (the action is just a
// deterministic reshape of the next placement, reversible by rerunning).
// Expensive tiers always prompt, because they run a helper process, spend
// online research, and recompute a previously-working placement. Any tier fails
// closed when the launch is not interactive, returning a rerun hint instead of
// silently consulting a helper model.
func confirmAdvisorAction(plan AdvisorConsentPlan, input io.Reader, output io.Writer, interactive bool) error {
	switch advisorConsentPolicy() {
	case "on":
		return nil
	case "off":
		return ErrAdvisorDeclined
	}
	if !interactive {
		// Fail closed: never silently consult a helper model in a launch that
		// cannot answer a prompt. Surface the deterministic opt-in instead.
		return fmt.Errorf("support advisor would %s; %s", plan.Summary, advisorConsentRerunHint(plan))
	}
	if plan.Tier == ConsentTierCheap {
		// A cheap action only reshapes the next deterministic placement; in an
		// interactive launch that is reversible by rerunning, so it is approved
		// without a prompt, mirroring how a resident --no-mmap retry proceeds.
		return nil
	}
	prompt := "Support advisor wants to " + plan.Summary + ". Allow it to run its helper model and re-plan the launch? [y/N] "
	fmt.Fprint(output, prompt)
	answer, err := bufio.NewReader(input).ReadString('\n')
	if err != nil && len(answer) == 0 {
		return fmt.Errorf("read advisor action confirmation: %w", err)
	}
	switch strings.ToLower(strings.TrimSpace(answer)) {
	case "y", "yes":
		return nil
	default:
		return ErrAdvisorDeclined
	}
}

// deterministicReplanOnPlacementFailure is the Layer-1 deterministic recompute
// that runs before any Layer-2 escalation on a placement failure. It bypasses a
// possibly-poisoned placement cache (SkipPlacementCache plus a cleared
// CacheFile, exactly like the startup OOM recovery) and preserves a derated
// ubatch from the failed attempt. placement.Compute already descends
// UBatchFitLadder internally, so this replan is not asked to guess a smaller
// ubatch — it only removes the cache that may have produced the failing plan
// and lets the packer retry clean. Returns the new strategy, or the original
// error when the recompute cannot fit.
func deterministicReplanOnPlacementFailure(req *launchRequest, model *placement.ModelProfile, be *backendInfo, cfg *config.Config, caps *detect.Capabilities, prior *placement.Strategy) (*placement.Strategy, error) {
	if req == nil || model == nil || be == nil || cfg == nil || caps == nil {
		return nil, fmt.Errorf("deterministic placement replan is missing launch context")
	}
	opts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	if prior != nil && prior.UBatchSize > 0 {
		// Preserve a derated ubatch: a retry must not recompute the original
		// automatic request back up to the rung that already failed.
		opts.UBatchSize = prior.UBatchSize
	}
	opts.SkipPlacementCache = true
	opts.CacheFile = ""
	return placement.Compute(caps, model, opts)
}

// escalatePlacementFailure is the placement-side Layer-2/Layer-3 boundary, the
// sibling of retryStartWithAdvisor for a placement (not backend-start) failure.
// The caller has already run the Layer-1 deterministic replan
// (deterministicReplanOnPlacementFailure) and handed the surviving error here.
// The Layer-2 gate decides whether the failure class is novel/unclassified (or a
// classified class that recurred after the deterministic budget) before any
// advisor consultation; the Layer-3 consent gate then approves the consultation
// before maybeAnalyzeLaunchFailure starts the advisor's helper process. Every
// declined/absent path returns the ORIGINAL replanErr — never a synthetic error —
// and consent is orthogonal to resource safety.
//
// computeStrategy is the caller's placement recompute closure; it is injected so
// the whole boundary is testable with the runSupportIncidentFn seam without
// starting a helper process. Returns the replacement strategy and nil on success.
func escalatePlacementFailure(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, firstCode string, replanErr error, computeStrategy func(*launchRequest) (*placement.Strategy, error)) (*placement.Strategy, error) {
	replanCode := classifyAdvisorFailure(replanErr)
	sameCodeRecurred := replanCode == firstCode
	escalation := escalationSignal{retriesExhausted: true, sameCodeRecurred: sameCodeRecurred}
	if !shouldEscalateToAdvisor(replanCode, escalation) {
		return nil, replanErr
	}
	plan := advisorConsentPlanForEscalation(replanCode)
	if consentErr := confirmAdvisorAction(plan, os.Stdin, os.Stderr, stdinIsTerminal()); consentErr != nil {
		fmt.Fprintf(os.Stderr, "[support] %v\n", consentErr)
		return nil, replanErr
	}
	decision, analyzed := maybeAnalyzeLaunchFailure(req, cfg, model, be, caps, nil, replanErr, "placement", escalation)
	if !analyzed || !applyAdvisorDecision(req, model, nil, decision) {
		return nil, replanErr
	}
	strategy, err := computeStrategy(req)
	if err != nil {
		return nil, err
	}
	return strategy, nil
}

func launchFailureIncident(req *launchRequest, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy, cause error, stage string) advisor.Incident {
	code := classifyAdvisorFailure(cause)
	arch := ""
	if model != nil {
		arch = model.ModelArch
	}
	backendFamily, backendIdentity := "", ""
	if be != nil {
		backendFamily, backendIdentity = backendDialect(be), evidenceBackendCacheTag(be)
	}
	settings := map[string]string{"stage": stage}
	if req != nil {
		settings["context"] = req.CtxFlag
		settings["kv_placement"] = req.KVPlacement
		settings["kv_quality"] = req.KVQuality
		settings["parallel"] = strconv.Itoa(req.Parallel)
		settings["swa_full"] = strconv.FormatBool(hasArg(req.ExtraArgs, "--swa-full"))
	}
	if strategy != nil {
		settings["resolved_context"] = strconv.Itoa(strategy.ContextSize)
		settings["ubatch"] = strconv.Itoa(strategy.UBatchSize)
		settings["cpu_moe_layers"] = strconv.Itoa(strategy.NCPUMoE)
		settings["checkpoints"] = strconv.Itoa(strategy.MaxCheckpoints)
		settings["context_allocation_mb"] = strconv.Itoa(strategy.ContextAllocationMB)
	}
	hardware := map[string]string{}
	if caps != nil {
		hardware["ram_total_mb"] = strconv.Itoa(caps.RAM.TotalMB)
		hardware["ram_free_mb"] = strconv.Itoa(caps.RAM.FreeMB)
		for _, gpu := range caps.GPUs {
			prefix := fmt.Sprintf("gpu%d_", gpu.Index)
			hardware[prefix+"name"] = gpu.Name
			hardware[prefix+"total_mb"] = strconv.Itoa(gpu.VRAMTotalMB)
			hardware[prefix+"free_mb"] = strconv.Itoa(gpu.VRAMFreeMB())
			hardware[prefix+"bandwidth_mbps"] = strconv.Itoa(gpu.BandwidthMBps)
		}
	}
	observation := advisor.Observation{Code: code, Component: stage, Source: "ggrun_controller", Confidence: "measured"}
	allowed := []advisor.ActionID{advisor.ActionNoAction, advisor.ActionRemeasureAllocation}
	if req != nil && strategy != nil && strategy.UBatchSize > 64 && !req.UBatchSizeSet {
		allowed = append(allowed, advisor.ActionLowerUBatch)
	}
	// propose_ubatch is offered only when the strategy sits at or above the top of
	// the derate ladder {256,128,64}: every accepted proposal is then a genuine
	// micro-batch derate, and the advisor can never propose the no-op rung 512.
	if req != nil && strategy != nil && strategy.UBatchSize >= 256 && !req.UBatchSizeSet {
		allowed = append(allowed, advisor.ActionProposeUBatch)
	}
	if req != nil && hasArg(req.ExtraArgs, "--swa-full") && !userExplicitBackendFlag(req, "--swa-full") {
		allowed = append(allowed, advisor.ActionRemoveGeneratedFeature)
		// toggle_swa_full is the general sibling that may also flip a
		// controller-generated swa-full back ON; it is equally never offered for
		// a user-explicit --swa-full/--no-swa-full choice.
		allowed = append(allowed, advisor.ActionToggleSWAFull)
	}
	// propose_layer_distribution is the general MoE rebalancing sibling of
	// move_expert_layer (which is reserved for a device-scoped OOM). Whenever the
	// controller knows the per-layer expert cost, the advisor may name a device
	// and a 1..2 layer count; the packer turns it into a VRAM budget reduction.
	if model != nil && expertLayerVRAMMB(model) > 0 {
		allowed = append(allowed, advisor.ActionProposeLayerDistribution)
	}
	if cause != nil {
		if device, bytesMB, ok := recovery.ParseCUDAOOM(cause.Error()); ok {
			observation.Device = device
			observation.Bytes = uint64(bytesMB) * 1024 * 1024
			// A device-scoped OOM is exactly the case where shedding one expert
			// layer from the failing card beats cutting global prefill
			// throughput. The deterministic ladder already prefers that order
			// (preflight_recovery.selectChangedPreflightRecovery); without this
			// the advisor was structurally unable to agree with it, and could
			// only ever answer "lower ubatch".
			if device >= 0 && expertLayerVRAMMB(model) > 0 {
				allowed = append(allowed, advisor.ActionMoveExpertLayer)
			}
		}
	}
	scope := strings.Join([]string{arch, backendIdentity, stage, code, settings["resolved_context"], settings["ubatch"]}, "|")
	sum := sha256.Sum256([]byte(scope))
	incident := advisor.Incident{
		ID: "launch-" + hex.EncodeToString(sum[:8]), Mode: advisor.ModeSupport,
		Architecture: arch, BackendFamily: backendFamily, BackendIdentity: backendIdentity,
		Workload: requestWorkloadProfile(req, model), ProfileState: stage,
		Hardware: hardware, Settings: settings, Observations: []advisor.Observation{observation}, AllowedActions: allowed,
	}
	_ = incident.Normalize()
	return incident
}

// expertLayerVRAMMB is the VRAM one routed-expert layer occupies, which is the
// unit a move_expert_layer decision reclaims. Returns 0 when the model profile
// cannot describe it, which withholds the action rather than guessing.
func expertLayerVRAMMB(model *placement.ModelProfile) int {
	if model == nil || !model.IsMoE || model.ExpertBytes <= 0 {
		return 0
	}
	moeLayers := model.NumLayers - model.LeadingDense
	if moeLayers <= 0 {
		moeLayers = model.NumLayers
	}
	if moeLayers <= 0 {
		return 0
	}
	return int(model.ExpertBytes / int64(moeLayers) / (1024 * 1024))
}

// classifyAdvisorFailure is the Layer-1 pre-flight probe that turns a raw
// launch failure into a typed taxonomy class. The deterministic controller
// owns every class except "unclassified_launch_failure"; a classified class
// reaches the advisor only when it recurs after the deterministic budget (the
// Layer-2 gate), and the pre-flight environment classes below (port collision,
// permission boundary, missing cgroup, a flag the repair loop cannot remove)
// never reach the advisor at all, because no bounded placement reshape can fix
// them — escalating would only spend a helper process and online budget to
// re-derive "no action".
func classifyAdvisorFailure(cause error) string {
	message := ""
	if cause != nil {
		message = strings.ToLower(cause.Error())
	}
	var rejected *backendArgValidationError
	if cause != nil {
		errors.As(cause, &rejected)
	}
	switch {
	case strings.Contains(message, "cuda") && strings.Contains(message, "out of memory"):
		return "cuda_oom_after_deterministic_recovery"
	case strings.Contains(message, "model does not fit"):
		return "model_does_not_fit"
	case strings.Contains(message, "memory preflight"):
		return "memory_preflight_failed"
	case strings.Contains(message, "unsupported") || strings.Contains(message, "does not support"):
		return "backend_capability_unsupported"
	// Pre-flight environment probes come BEFORE backend_start_failed: a bind
	// failure surfaces wrapped inside "server not ready: server process exited
	// during startup: bind: address already in use", and the environment class
	// must win so the deterministic controller routes it to a user action
	// instead of treating it as a start failure that could escalate.
	case strings.Contains(message, "address already in use") ||
		strings.Contains(message, "already in use") ||
		strings.Contains(message, "cannot assign requested address") ||
		strings.Contains(message, "bind: address"):
		return "port_in_use"
	case strings.Contains(message, "permission denied") ||
		strings.Contains(message, "operation not permitted") ||
		strings.Contains(message, "access denied"):
		return "permission_denied"
	case strings.Contains(message, "requires systemd-run") ||
		// The containment refusal is ggrun-authored and stable. Matching it
		// explicitly keeps a root/container launch out of unclassified_launch_failure,
		// which is where a raw systemd-run dbus error used to land (issue #64).
		strings.Contains(message, "backend memory containment unavailable") ||
		(strings.Contains(message, "cgroup") &&
			(strings.Contains(message, "containment") || strings.Contains(message, "memory") || strings.Contains(message, "limit"))):
		return "memory_cgroup_limit"
	// flag_rejected_no_fix: the backend rejected a flag the deterministic repair
	// loop (validateAndRepairBackendArgs) cannot remove. The loop removes only
	// ggrun-generated allowlisted flags; a rejected user-explicit flag, a
	// non-repairable generated flag, or a repair loop that stalled is a
	// user-action failure, never an advisor placement problem. Detect the
	// structured error first so "unknown argument" stays routed to the repairable
	// backend_flag_rejected class only when the flag IS repairable.
	case (rejected != nil && rejected.Flag != "" && !repairableGeneratedBackendFlag(rejected.Flag)) ||
		strings.Contains(message, "backend rejected explicitly supplied") ||
		strings.Contains(message, "backend argument repair") ||
		strings.Contains(message, "refusing to change user input"):
		return "flag_rejected_no_fix"
	case strings.Contains(message, "unknown argument") ||
		strings.Contains(message, "unrecognized argument") ||
		strings.Contains(message, "invalid argument"):
		return "backend_flag_rejected"
	case strings.Contains(message, "server not ready") || strings.Contains(message, "failed to start"):
		return "backend_start_failed"
	case strings.Contains(message, "cache"):
		return "cache_verification_failed"
	default:
		return "unclassified_launch_failure"
	}
}

// deterministicFailureCode reports whether a classified failure is owned by the
// deterministic controller with NO advisor-reshapable action. The advisor can
// only propose bounded placement reshaples; a port collision, a permission
// boundary, a missing cgroup containment, or a flag the controller refuses to
// change are environment/user-action problems that no reshape fixes. Escalating
// them would spend a helper process and online research budget to re-derive the
// same "no_action" — so the Layer-2 gate treats them as deterministic.
func deterministicFailureCode(code string) bool {
	switch code {
	case "port_in_use", "permission_denied", "memory_cgroup_limit", "flag_rejected_no_fix":
		return true
	default:
		return false
	}
}

func maybeAnalyzeLaunchFailure(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy, cause error, stage string, escalation escalationSignal) (advisor.Decision, bool) {
	mode := "auto"
	online := false
	if cfg != nil {
		mode, online = cfg.SupportExpert, cfg.SupportOnline
	}
	if req != nil {
		if req.SupportExpert != "" {
			mode = req.SupportExpert
		}
		online = req.SupportOnline
	}
	if mode == "off" {
		return advisor.Decision{}, false
	}
	// Force-online research on escalation. The advisor is consulted only for a
	// genuinely novel/unclassified failure (or a classified class that recurred
	// after the deterministic budget), so its evidence pool should not be silently
	// limited to bundled knowledge: official issue search plus the pinned model
	// card are exactly what make the helper useful on the rare failure that
	// reaches it. Online research stays OFF when the user explicitly named
	// --no-support-online (a user instruction that no failure class overrides).
	// The consent gate (Layer 3) has already approved the consultation before this
	// point, and the research tier degrades gracefully if the network is down.
	if req == nil || !req.SupportOnlineSet {
		online = true
	}
	// Layer-2 escalation gate. Classified, non-recurring failures are the
	// deterministic controller's own problem; escalating them would only ever
	// re-derive the same bounded action at the cost of a helper process and (in
	// Layer 3) a user prompt. Only a novel/unclassified failure, or a classified
	// one that has resisted the full deterministic budget by recurring with the
	// same code, reaches the advisor.
	if !shouldEscalateToAdvisor(classifyAdvisorFailure(cause), escalation) {
		return advisor.Decision{}, false
	}
	incident := launchFailureIncident(req, model, be, caps, strategy, cause, stage)
	preferred := ""
	if be != nil {
		preferred = be.Path
	}
	decision, report, runErr := runSupportIncidentFn(context.Background(), cfg, caps, preferred, incident, online)
	var decisionPtr *advisor.Decision
	if runErr == nil {
		decisionPtr = &decision
	}
	history, historyErr := advisor.SaveAnalysis(cfg.CacheDir, incident, decisionPtr, report, runErr)
	if historyErr != nil {
		fmt.Fprintf(os.Stderr, "[support] could not persist analysis: %v\n", historyErr)
	}
	if runErr != nil {
		fmt.Fprintf(os.Stderr, "[support] optional expert unavailable: %v\n", runErr)
		if mode == "auto" && !statusArtifactReady(cfg) {
			fmt.Fprintln(os.Stderr, "[support] install the optional native expert/optimizer with: ggrun support install")
		}
		return advisor.Decision{}, false
	}
	fmt.Fprintf(os.Stderr, "[support] decision=%s confidence=%.2f — %s\n", decision.Action, decision.Confidence, decision.Rationale)
	fmt.Fprintf(os.Stderr, "[support] helper stopped and released resources before main placement; record=%s\n", history)
	return decision, true
}

func statusArtifactReady(cfg *config.Config) bool {
	return advisor.VerifyArtifact(advisorModelPath(cfg), advisor.DefaultArtifact) == nil
}

// adviseUnclassifiedLaunchFailure is the crash-diagnosis hook. It consults the
// local assistant model (the advisor) about a launch failure that no ggrun rule
// classified — no memory OOM, no backendAdjustmentFromLog match, no known flag —
// and surfaces the diagnosis on stderr. This is the user's ask: "integrate the
// local assistant model for things like that when models crash and ggrun does
// not know why".
//
// It is STRICTLY advisory: the incident's AllowedActions is only no_action, so
// ValidateDecision rejects any recommendation the model makes and nothing it
// says can reach argv or placement. The launch still fails closed exactly as it
// would have without the consultation; the only observable effect is the printed
// rationale and a persisted analysis record. Mode is governed by the same
// support_expert / --support-expert policy as the optimizer: off never consults,
// auto consults only when an already-verified artifact is installed, on consults
// and reports a missing artifact as a diagnostic. A helper-process or research
// failure is non-fatal and logged, never escalated.
func adviseUnclassifiedLaunchFailure(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, logExcerpt string) {
	mode := "auto"
	online := false
	if cfg != nil {
		mode, online = cfg.SupportExpert, cfg.SupportOnline
	}
	if req != nil {
		if req.SupportExpert != "" {
			mode = req.SupportExpert
		}
		online = req.SupportOnline
	}
	if mode == "off" {
		return
	}
	if mode == "auto" && !statusArtifactReady(cfg) {
		fmt.Fprintln(os.Stderr, "[support] optional crash diagnosis skipped; install it with: ggrun support install")
		return
	}
	incident := unclassifiedFailureIncident(req, model, be, caps, logExcerpt)
	preferred := ""
	if be != nil {
		preferred = be.Path
	}
	decision, report, runErr := runSupportIncidentFn(context.Background(), cfg, caps, preferred, incident, online)
	var decisionPtr *advisor.Decision
	if runErr == nil {
		decisionPtr = &decision
	}
	history, historyErr := advisor.SaveAnalysis(cfg.CacheDir, incident, decisionPtr, report, runErr)
	if historyErr != nil {
		fmt.Fprintf(os.Stderr, "[support] could not persist crash diagnosis: %v\n", historyErr)
	}
	if runErr != nil {
		fmt.Fprintf(os.Stderr, "[support] crash diagnosis unavailable; the launch error above is authoritative: %v\n", runErr)
		if mode == "auto" && !statusArtifactReady(cfg) {
			fmt.Fprintln(os.Stderr, "[support] install the optional native expert/optimizer with: ggrun support install")
		}
		return
	}
	// Only a rationale is surfaced. The advisor was offered exactly one allowed
	// action (no_action), so even if the model emitted a recommendation it was
	// rejected by ValidateDecision and could not reach a launch.
	fmt.Fprintf(os.Stderr, "[support] diagnosis (advisory, no launch action taken): %s\n", decision.Rationale)
	if history != "" {
		fmt.Fprintf(os.Stderr, "[support] diagnosis record=%s\n", history)
	}
}

// unclassifiedFailureIncident builds the advisory-only advisor incident for a
// launch failure that no ggrun rule classified. The backend's own log excerpt is
// the primary evidence; model/backend metadata gives the advisor the shape of the
// failed launch. AllowedActions is exactly no_action, making the consultation
// diagnostic by construction: the model may explain what it sees, never change
// what ggrun does.
func unclassifiedFailureIncident(req *launchRequest, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, logExcerpt string) advisor.Incident {
	arch := ""
	if model != nil {
		arch = model.ModelArch
	}
	backendFamily, backendIdentity := "", ""
	if be != nil {
		backendFamily, backendIdentity = backendDialect(be), evidenceBackendCacheTag(be)
	}
	settings := map[string]string{"stage": "backend_start", "role": "crash_diagnosis_advisory"}
	if req != nil {
		settings["context"] = req.CtxFlag
		settings["kv_placement"] = req.KVPlacement
		settings["kv_quality"] = req.KVQuality
		settings["parallel"] = strconv.Itoa(req.Parallel)
	}
	hardware := map[string]string{}
	if caps != nil {
		hardware["ram_total_mb"] = strconv.Itoa(caps.RAM.TotalMB)
		hardware["ram_free_mb"] = strconv.Itoa(caps.RAM.FreeMB)
		for _, gpu := range caps.GPUs {
			prefix := fmt.Sprintf("gpu%d_", gpu.Index)
			hardware[prefix+"name"] = gpu.Name
			hardware[prefix+"total_mb"] = strconv.Itoa(gpu.VRAMTotalMB)
			hardware[prefix+"free_mb"] = strconv.Itoa(gpu.VRAMFreeMB())
		}
	}
	// The hook fires precisely because no ggrun rule classified the failure, so
	// the observation code is the unclassified class by construction. The
	// advisor's evidence is the backend log excerpt, not a re-derived taxonomy.
	code := "unclassified_launch_failure"
	observation := advisor.Observation{
		Code: code, Component: "backend_start", Source: "ggrun_controller",
		Confidence: "measured", Attributes: map[string]string{"backend_log_excerpt": logExcerpt},
	}
	scope := strings.Join([]string{arch, backendIdentity, code, settings["context"]}, "|")
	sum := sha256.Sum256([]byte(scope))
	incident := advisor.Incident{
		ID: "diag-" + hex.EncodeToString(sum[:8]), Mode: advisor.ModeSupport,
		Architecture: arch, BackendFamily: backendFamily, BackendIdentity: backendIdentity,
		Workload: requestWorkloadProfile(req, model), ProfileState: "crash_diagnosis_advisory",
		Hardware: hardware, Settings: settings, Observations: []advisor.Observation{observation},
		AllowedActions: []advisor.ActionID{advisor.ActionNoAction},
	}
	_ = incident.Normalize()
	return incident
}

func applyAdvisorDecision(req *launchRequest, model *placement.ModelProfile, strategy *placement.Strategy, decision advisor.Decision) bool {
	if req == nil {
		return false
	}
	switch decision.Action {
	case advisor.ActionRemeasureAllocation:
		// retryStartWithAdvisor always bypasses the placement cache and performs
		// a complete deterministic Compute, so this typed action needs no model-
		// supplied parameter or direct argv mutation.
		return true
	case advisor.ActionLowerUBatch:
		if req.UBatchSizeSet || strategy == nil || int(decision.UBatch) >= strategy.UBatchSize {
			return false
		}
		req.UBatchSize, req.UBatchSizeSet = int(decision.UBatch), true
		if req.BatchSizeSet && req.BatchSize < req.UBatchSize {
			return false
		}
		return true
	case advisor.ActionProposeUBatch:
		// Same strict-derate discipline as lower_ubatch; the {256,128,64} ladder
		// is enforced by ValidateDecision, so the apply side only needs to refuse
		// a proposal that is not strictly smaller than the current rung.
		if req.UBatchSizeSet || strategy == nil || int(decision.UBatch) >= strategy.UBatchSize {
			return false
		}
		req.UBatchSize, req.UBatchSizeSet = int(decision.UBatch), true
		if req.BatchSizeSet && req.BatchSize < req.UBatchSize {
			return false
		}
		return true
	case advisor.ActionRemoveGeneratedFeature:
		if decision.Feature != advisor.FeatureSWAFull || userExplicitBackendFlag(req, "--swa-full") || !hasArg(req.ExtraArgs, "--swa-full") {
			return false
		}
		req.ExtraArgs = setPassthroughBoolFlag(req.ExtraArgs, "--swa-full", false)
		return true
	case advisor.ActionToggleSWAFull:
		// Generated-only: a user-explicit --swa-full/--no-swa-full is never
		// touched. Value=false removes the generated feature (the useful direction
		// on a fit failure); Value=true confirms it, which is a harmless no-op.
		if decision.Feature != advisor.FeatureSWAFull || userExplicitBackendFlag(req, "--swa-full") || !hasArg(req.ExtraArgs, "--swa-full") {
			return false
		}
		req.ExtraArgs = setPassthroughBoolFlag(req.ExtraArgs, "--swa-full", decision.Value)
		return true
	case advisor.ActionMoveExpertLayer, advisor.ActionProposeLayerDistribution:
		perLayerMB := expertLayerVRAMMB(model)
		if perLayerMB <= 0 || decision.Device < 0 || decision.Count < 1 {
			return false
		}
		if req.AdvisorVRAMPenaltyMB == nil {
			req.AdvisorVRAMPenaltyMB = map[int]int{}
		}
		req.AdvisorVRAMPenaltyMB[int(decision.Device)] += perLayerMB * int(decision.Count)
		return true
	default:
		return false
	}
}

// retryStartWithAdvisor is the single advisor-controlled retry boundary. The
// caller has already stopped every main/reviewer process and verified release
// (stopFailedLaunchBeforeAdvisor). The Layer-3 consent gate then asks the user
// before this function consults the advisor's helper process; Analyze must stop
// NanoBeige and verify resource release before this function recomputes
// placement and starts anything. No model-produced string reaches argv.
func retryStartWithAdvisor(req *launchRequest, cfg *config.Config, model *placement.ModelProfile,
	be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy, cause error,
	timeout time.Duration, memoryRecovery *launchMemoryRecovery,
) (*server.Process, *placement.Strategy, []string, *claudeAutoRuntime, error) {
	// The launch loop hands the failure here only after its deterministic
	// OOM/preflight recovery budget is spent. Thread that state into the Layer-2
	// escalation predicate: a classified failure escalates only when the same
	// class recurred across the deterministic attempts (memoryRecovery logged
	// rejections); a novel/unclassified failure escalates regardless.
	escalation := escalationSignal{
		retriesExhausted: true,
		sameCodeRecurred: memoryRecovery != nil && memoryRecovery.hasRejections(),
	}
	code := classifyAdvisorFailure(cause)
	if !shouldEscalateToAdvisor(code, escalation) {
		// Not a Layer-2 escalation: this is the deterministic controller's own
		// problem, so the original failure is returned untouched. No prompt, no
		// helper process.
		return nil, strategy, nil, nil, cause
	}
	// Layer-3 consent gate, inserted after the caller's stop-main -> verified
	// release and before the advisor's helper process (runSupportIncident, inside
	// maybeAnalyzeLaunchFailure) is started. The gate approves the CONSULTATION,
	// never a raw command line: the action the advisor may take is still bounded
	// by ValidateDecision -> applyAdvisorDecision. Consent is orthogonal to
	// resource safety — a decline here must not weaken the
	// ErrResourceReleaseUnverified boundary, and the caller must not restart the
	// main model until helper release is proven.
	plan := advisorConsentPlanForEscalation(code)
	if err := confirmAdvisorAction(plan, os.Stdin, os.Stderr, stdinIsTerminal()); err != nil {
		// A declined, auto-refused (GGRUN_ADVISOR_CONSENT=off), or unconfirmable
		// (non-interactive) consultation is a policy choice, never a synthetic
		// launch failure. Surface the rerun hint on stderr and return the
		// ORIGINAL launch error so the user sees the real reason their launch
		// failed, exactly like the mmap and live-memory-probe declines.
		fmt.Fprintf(os.Stderr, "[support] %v\n", err)
		return nil, strategy, nil, nil, cause
	}
	decision, analyzed := maybeAnalyzeLaunchFailure(req, cfg, model, be, caps, strategy, cause, "backend_start", escalation)
	if !analyzed {
		return nil, strategy, nil, nil, cause
	}
	// Post-consultation Layer-3 gate for the SPECIFIC bounded action the advisor
	// returned. The escalation gate above approved the consultation; this gate
	// approves the action. Every current action is a ConsentTierCheap placement
	// reshape, so an interactive launch auto-approves; a non-interactive launch
	// still fails closed. A declined action must return the ORIGINAL launch error,
	// never a synthetic one.
	actionPlan := advisorConsentPlanForAction(decision.Action, decision.Feature)
	if err := confirmAdvisorAction(actionPlan, os.Stdin, os.Stderr, stdinIsTerminal()); err != nil {
		fmt.Fprintf(os.Stderr, "[support] %v\n", err)
		return nil, strategy, nil, nil, cause
	}
	if !applyAdvisorDecision(req, model, strategy, decision) {
		return nil, strategy, nil, nil, cause
	}

	opts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	opts.SkipPlacementCache = true
	var next *placement.Strategy
	var err error
	if len(req.AdvisorVRAMPenaltyMB) > 0 {
		// ReplanAfterOOM shrinks the named device's usable VRAM and then runs the
		// ordinary packer, so every other GPU is re-packed around the reduction
		// instead of the failing card simply losing a layer to system RAM.
		next, err = placement.ReplanAfterOOM(caps, model, opts, req.AdvisorVRAMPenaltyMB)
	} else {
		next, err = placement.Compute(caps, model, opts)
	}
	if err != nil {
		return nil, strategy, nil, nil, fmt.Errorf("advisor-approved re-plan failed: %w", err)
	}
	next = applyCalibrationDecision(req, cfg, model, be, caps, next)
	claudeCodeSlotAdjust(next, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	if err := validateHostMemoryContainment(req, caps, next); err != nil {
		return nil, strategy, nil, nil, fmt.Errorf("advisor-approved re-plan violates host containment: %w", err)
	}
	nextArgs := buildLaunchServerArgs(req, cfg, be, caps, model, next)
	if err := validateBackendLaunchArgs(be, nextArgs); err != nil {
		return nil, strategy, nil, nil, err
	}

	fmt.Fprintf(os.Stderr, "[support] applying bounded %s decision and retrying once after full placement recompute\n", decision.Action)
	reviewer, err := startClaudeAutoReviewer(req, cfg, caps, next.CompanionPlacements)
	if err != nil {
		return nil, strategy, nil, nil, err
	}
	if req.ClaudeCode && reviewer == nil {
		reviewer = &claudeAutoRuntime{reviewerGPU: -1}
	}
	fmt.Printf("[launch] %s\n", formatCommand(nextArgs))
	process, next, nextArgs, err := startLaunchWithCUDAOOMRecoveryState(req, cfg, model, next, be, caps, nextArgs, timeout, memoryRecovery)
	if err != nil {
		reviewer.stop()
		return nil, next, nextArgs, nil, fmt.Errorf("advisor-approved retry failed: %w", err)
	}
	return process, next, nextArgs, reviewer, nil
}
