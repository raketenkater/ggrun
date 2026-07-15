package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/libhub"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/specbench"
)

type specTestConfiguration struct {
	Name       string           `json:"name"`
	DraftMax   int              `json:"draft_max,omitempty"`
	Context    int              `json:"context_size"`
	Parallel   int              `json:"parallel"`
	DraftPath  string           `json:"draft_path,omitempty"`
	LaunchID   string           `json:"launch_identity,omitempty"`
	Result     specbench.Result `json:"result"`
	Skipped    bool             `json:"skipped,omitempty"`
	SkipReason string           `json:"skip_reason,omitempty"`
}

type specTestReport struct {
	Model           string                  `json:"model"`
	BackendIdentity string                  `json:"backend_identity"`
	MeasuredAt      string                  `json:"measured_at"`
	Rounds          int                     `json:"rounds"`
	Configurations  []specTestConfiguration `json:"configurations"`
}

func cmdSpecTest(args []string) {
	if err := runSpecTest(args); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// runSpecTest benchmarks the stable target-only placement and the exact MTP
// placement at ceilings 1-4. Auto never consumes a result unless this command
// proves correctness, repeated stability, end-to-end speed, and the requested
// parallel/long-context workload for the exact artifact/backend/hardware scope.
func runSpecTest(args []string) error {
	req, err := parseLaunchArgs(args)
	if err != nil {
		return err
	}
	if req.ModelPath == "" {
		return fmt.Errorf("usage: ggrun spec-test <model.gguf> [--rounds 2] [--ctx 1048576] [--parallel 4]")
	}
	if err := guardPortFree(req.Port, "spec-test"); err != nil {
		return err
	}

	cfg, loadErr := config.Load()
	if loadErr != nil {
		return fmt.Errorf("load config: %w", loadErr)
	}
	rounds, err := tuneRoundsFromArgs(args, 2)
	if err != nil {
		return err
	}
	if rounds < 2 {
		return fmt.Errorf("spec-test requires at least 2 rounds")
	}
	req.ModelPath = resolveModelPath(req.ModelPath, cfg.ModelDir)
	model, err := parseModel(req.ModelPath)
	if err != nil {
		return fmt.Errorf("parse model: %w", err)
	}
	caps, err := detect.Detect()
	if err != nil {
		return fmt.Errorf("detect hardware: %w", err)
	}
	be := resolveLaunchBackend(req, model, caps)
	if be == nil {
		return fmt.Errorf("no llama-server binary found")
	}
	if env := applyGPUVisibility(req, backendDialect(be)); env != "" {
		fmt.Printf("[spec-test] GPU restriction: %s\n", env)
	}
	if hubDir, ok, setupErr := libhub.Setup(be.Path); setupErr != nil {
		fmt.Fprintf(os.Stderr, "[spec-test] warning: lib hub: %v\n", setupErr)
	} else if ok {
		os.Setenv("LLM_SERVER_LIB_HUB", hubDir)
		defer libhub.Cleanup(hubDir)
	}

	report := specTestReport{
		Model: req.ModelPath, BackendIdentity: be.Identity,
		MeasuredAt: time.Now().UTC().Format(time.RFC3339), Rounds: rounds,
	}

	// The core matrix is deterministic and compares target-only against MTP,
	// not the model's variable-length thinking budget. Reasoning models can spend
	// the entire short answer budget in reasoning_content and never emit the
	// verification marker, making a healthy baseline look broken. The extended
	// matrix tracks thinking on/off separately; keep this first proof explicit.
	testReq := *req
	testReq.Benchmark = true
	fmt.Println("[spec-test] deterministic matrix: reasoning off")
	baselineReq := testReq
	baselineReq.SpecMode = "off"
	baseline, baselineStrategy, err := runSpecConfiguration(&baselineReq, cfg, caps, model, be, rounds, 0)
	if err != nil {
		report.Configurations = append(report.Configurations, baseline)
		reportPath, saveErr := saveSpecTestReport(cfg.CacheDir, report)
		if saveErr != nil {
			return fmt.Errorf("baseline: %w (also failed to save raw report: %v)", err, saveErr)
		}
		fmt.Printf("[spec-test] baseline failures: %s\n", specFailureSummary(baseline.Result))
		return fmt.Errorf("baseline: %w; raw report: %s", err, reportPath)
	}
	report.Configurations = append(report.Configurations, baseline)
	fmt.Printf("[spec-test] baseline: %.2f tok/s, %.2fs mean wall, %.2f prompt tok/s\n",
		baseline.Result.MedianGenerateTPS, baseline.Result.MeanWallSeconds, baseline.Result.MedianPromptTPS)

	bestEligibleIndex := -1
	bestEligibleWallGain := -1e9
	bestStableIndex := -1
	bestStableWallGain := -1e9
	for ceiling := 1; ceiling <= 4; ceiling++ {
		mtpReq := testReq
		mtpReq.SpecMode = "mtp"
		mtpReq.SpecDraftMax = ceiling
		result, strategy, runErr := runSpecConfiguration(&mtpReq, cfg, caps, model, be, rounds, ceiling)
		if runErr != nil {
			result.Skipped = true
			result.SkipReason = runErr.Error()
			fmt.Printf("[spec-test] MTP ceiling %d skipped: %v\n", ceiling, runErr)
		} else if strategy.ContextSize != baselineStrategy.ContextSize || strategy.Parallel != baselineStrategy.Parallel {
			result.Skipped = true
			result.SkipReason = "MTP changed context or parallelism relative to baseline"
			fmt.Printf("[spec-test] MTP ceiling %d rejected: %s\n", ceiling, result.SkipReason)
		} else {
			decodeGain := percentGain(result.Result.MedianGenerateTPS, baseline.Result.MedianGenerateTPS)
			wallGain := inversePercentGain(result.Result.MeanWallSeconds, baseline.Result.MeanWallSeconds)
			promptRegression := percentRegression(result.Result.MedianPromptTPS, baseline.Result.MedianPromptTPS)
			lengthDelta := absolutePercentDelta(result.Result.MeanGenerated, baseline.Result.MeanGenerated)
			fmt.Printf("[spec-test] MTP ceiling %d: %.2f tok/s (%+.2f%%), %.2fs wall (%+.2f%%), accept %.1f%%\n",
				ceiling, result.Result.MedianGenerateTPS, decodeGain, result.Result.MeanWallSeconds, wallGain, result.Result.DraftAcceptRate*100)
			if result.Result.CorrectnessPassed && result.Result.StabilityPassed {
				if wallGain > bestStableWallGain {
					bestStableIndex = len(report.Configurations)
					bestStableWallGain = wallGain
				}
				longProofOK := result.Context/result.Parallel < 60000 || result.Result.MaxPromptTokens >= 60000
				if decodeGain >= 2 && wallGain >= 2 && promptRegression <= 5 && lengthDelta <= 10 && longProofOK && wallGain > bestEligibleWallGain {
					bestEligibleIndex = len(report.Configurations)
					bestEligibleWallGain = wallGain
				}
			}
		}
		report.Configurations = append(report.Configurations, result)
	}

	reportPath, err := saveSpecTestReport(cfg.CacheDir, report)
	if err != nil {
		return fmt.Errorf("save raw report: %w", err)
	}
	fmt.Printf("[spec-test] raw report: %s\n", reportPath)
	bestIndex := bestEligibleIndex
	if bestIndex < 0 {
		bestIndex = bestStableIndex
	}
	if bestIndex < 0 {
		return fmt.Errorf("no MTP ceiling completed correctness and stability checks; Auto remains off")
	}

	best := report.Configurations[bestIndex]
	opts := placementOptionsFromRequest(&testReq, model, be, cfg.CacheDir)
	opts.ContextSize = best.Context
	opts.Parallel = best.Parallel
	scope := placement.NewSpecProfileScope(model, caps, opts, "mtp", best.DraftPath)
	decodeGain := percentGain(best.Result.MedianGenerateTPS, baseline.Result.MedianGenerateTPS)
	wallGain := inversePercentGain(best.Result.MeanWallSeconds, baseline.Result.MeanWallSeconds)
	promptRegression := percentRegression(best.Result.MedianPromptTPS, baseline.Result.MedianPromptTPS)
	lengthDelta := absolutePercentDelta(best.Result.MeanGenerated, baseline.Result.MeanGenerated)
	profile := placement.SpecPerformanceProfile{
		Scope: scope, LaunchIdentity: best.LaunchID, DraftMax: best.DraftMax,
		BaselineTPS: baseline.Result.MedianGenerateTPS, SpeculativeTPS: best.Result.MedianGenerateTPS,
		ImprovementPct: decodeGain, WallImprovementPct: wallGain, PromptRegressionPct: promptRegression, OutputLengthDeltaPct: lengthDelta,
		PromptCases: len(specbench.Prompts(false)), RepeatedRounds: rounds, MaxPromptTokens: best.Result.MaxPromptTokens,
		CorrectnessPassed: best.Result.CorrectnessPassed, StabilityPassed: best.Result.StabilityPassed,
		ParallelLoadPassed: best.Parallel <= 1 || best.Result.StabilityPassed, Complete: true,
	}
	profilePath, err := placement.SaveSpecPerformanceProfile(cfg.CacheDir, profile)
	if err != nil {
		return fmt.Errorf("save performance profile: %w", err)
	}
	if ok, reason := profile.AutoEligible(); !ok {
		fmt.Printf("[spec-test] profile saved but Auto remains off: %s\n", reason)
	} else {
		fmt.Printf("[spec-test] verified MTP ceiling %d; Auto is enabled for this exact runtime scope\n", best.DraftMax)
	}
	fmt.Printf("[spec-test] profile: %s\n", profilePath)
	return nil
}

func specFailureSummary(result specbench.Result) string {
	failures := make([]string, 0)
	for _, sample := range result.Samples {
		if sample.Correct {
			continue
		}
		reason := sample.Error
		if reason == "" {
			reason = "verification failed"
		}
		failures = append(failures, sample.Prompt+": "+reason)
		if len(failures) == 5 {
			break
		}
	}
	if len(failures) == 0 {
		return "stability proof incomplete"
	}
	return strings.Join(failures, "; ")
}

func runSpecConfiguration(req *launchRequest, cfg *config.Config, caps *detect.Capabilities, model *placement.ModelProfile, be *backendInfo, rounds, ceiling int) (specTestConfiguration, *placement.Strategy, error) {
	name := "baseline"
	if ceiling > 0 {
		name = fmt.Sprintf("mtp-%d", ceiling)
	}
	opts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	strategy, err := placement.Compute(caps, model, opts)
	if err != nil {
		return specTestConfiguration{Name: name, DraftMax: ceiling}, nil, err
	}
	claudeCodeSlotAdjust(strategy, req.ClaudeCode, req.ParallelSet)
	if ceiling > 0 && (strategy.Draft == nil || strategy.Draft.Type != placement.DraftMTP) {
		return specTestConfiguration{Name: name, DraftMax: ceiling}, strategy, fmt.Errorf("no compatible MTP path for selected backend")
	}
	serverArgs := buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
	fmt.Printf("[spec-test] loading %s: %s\n", name, formatCommand(serverArgs))
	p, finalStrategy, finalArgs, err := startLaunchWithCUDAOOMRecovery(req, cfg, model, strategy, be, caps, serverArgs, autoStartupTimeout(model))
	if err != nil {
		return specTestConfiguration{Name: name, DraftMax: ceiling}, finalStrategy, err
	}
	defer p.Stop()
	if ceiling > 0 && (finalStrategy.Draft == nil || finalStrategy.Draft.Type != placement.DraftMTP || finalStrategy.Draft.DraftMax != ceiling) {
		return specTestConfiguration{Name: name, DraftMax: ceiling}, finalStrategy, fmt.Errorf("startup safety re-plan disabled or changed MTP")
	}
	parallel := finalStrategy.Parallel
	if parallel < 1 {
		parallel = 1
	}
	include60K := finalStrategy.ContextSize/parallel >= 60000
	timeout := 30 * time.Minute
	if model.IsMoE {
		timeout = 2 * time.Hour
	}
	modelName := filepath.Base(req.ModelPath)
	if req.ClaudeCode {
		modelName = "local"
	}
	result := (&specbench.Runner{
		BaseURL: fmt.Sprintf("http://127.0.0.1:%d", req.Port), Model: modelName,
		Timeout: timeout, Rounds: rounds, Parallel: parallel, Include60K: include60K,
	}).Run()
	draftPath := ""
	if finalStrategy.Draft != nil {
		draftPath = finalStrategy.Draft.Path
	}
	configuration := specTestConfiguration{
		Name: name, DraftMax: ceiling, Context: finalStrategy.ContextSize,
		Parallel: parallel, DraftPath: draftPath, LaunchID: specLaunchIdentity(finalArgs), Result: result,
	}
	if !result.CorrectnessPassed || !result.StabilityPassed {
		return configuration, finalStrategy, fmt.Errorf("request correctness or repeated stability failed")
	}
	return configuration, finalStrategy, nil
}

func percentGain(candidate, baseline float64) float64 {
	if baseline <= 0 {
		return -100
	}
	return (candidate/baseline - 1) * 100
}

func inversePercentGain(candidate, baseline float64) float64 {
	if candidate <= 0 || baseline <= 0 {
		return -100
	}
	return (baseline/candidate - 1) * 100
}

func percentRegression(candidate, baseline float64) float64 {
	if baseline <= 0 || candidate >= baseline {
		return 0
	}
	return (baseline - candidate) / baseline * 100
}

func absolutePercentDelta(candidate, baseline float64) float64 {
	if baseline <= 0 {
		return 100
	}
	delta := (candidate - baseline) / baseline * 100
	if delta < 0 {
		return -delta
	}
	return delta
}

func saveSpecTestReport(cacheDir string, report specTestReport) (string, error) {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	dir := filepath.Join(cacheDir, "spec-results")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, fmt.Sprintf("spec-%d.json", time.Now().UTC().Unix()))
	tmp := path + fmt.Sprintf(".%d.tmp", os.Getpid())
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return "", err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return "", err
	}
	return path, nil
}
