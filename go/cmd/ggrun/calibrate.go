package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/advisor"
	"github.com/raketenkater/ggrun/pkg/benchmark"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
)

// calibrationMinImprovementPct is the margin a challenger must beat the default
// by before it is cached as the winner. Below this the two placements are
// indistinguishable at micro-probe precision and the estimated default stands.
const calibrationMinImprovementPct = 3.0

// A faster aggregate agent turn is not a valid core winner if it achieves that
// result by materially sacrificing either prompt ingestion or generation.
// Five percent is above the micro-screen noise floor while still permitting
// small phase trade-offs whose end-to-end workflow improvement is repeatable.
const calibrationMaxPhaseRegressionPct = 5.0

const (
	// Standard launch measures the already-running baseline plus at most one
	// successfully admitted challenger. It may walk a short ranked admission
	// ladder when an optimistic finalist cannot start exactly; this avoids
	// declaring the baseline tuned merely because the first estimate missed its
	// memory boundary. A wider successful live sweep still belongs to explicit
	// maintenance mode.
	calibrationAutoMaxCandidates   = 4
	calibrationForcedMaxCandidates = 9
	calibrationAutoFailureBudget   = 3
	calibrationForcedFailureBudget = 4
	calibrationAutoMaxElapsed      = 20 * time.Minute
	calibrationForcedMaxElapsed    = 60 * time.Minute
	calibrationAdvisorTiePct       = 3.0
	// Automatic launch sizes one identical per-lane prompt from a short baseline
	// prefill so two repeated baseline/finalist scenarios fit inside the finite
	// controller budget even for a slow CPU-expert MoE. Maintenance calibration
	// keeps benchmark.Runner's full prompt geometry.
	calibrationAutoAgentWaveTarget = time.Minute
	calibrationAutoProbeTimeout    = 2 * time.Minute
)

type calibrationBudget struct {
	MaxCandidates int
	MaxFailures   int
	MaxElapsed    time.Duration
}

func calibrationBudgetFor(mode string) calibrationBudget {
	if mode == calibrateOn {
		return calibrationBudget{calibrationForcedMaxCandidates, calibrationForcedFailureBudget, calibrationForcedMaxElapsed}
	}
	return calibrationBudget{calibrationAutoMaxCandidates, calibrationAutoFailureBudget, calibrationAutoMaxElapsed}
}

func effectiveCalibrationMode(req *launchRequest) string {
	if req == nil || req.Calibrate == "" {
		return calibrateAuto
	}
	return req.Calibrate
}

// calibrationPlan decides whether this launch should run first-launch
// calibration and, if so, the candidate set to measure. It is a pure function
// of the request, model, hardware, and whether a prior decision exists, so the
// policy is unit-testable without starting a server.
//
// Returns nil when there is nothing to do: optimization is disabled, a
// decision is already cached for this exact scope, or the active workload
// exposes no bounded full-placement alternate. Automatic mode owns this search
// for ordinary launch; --calibrate on only widens its maintenance budget.
func calibrationPlan(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy, rejected ...func(*placement.Strategy) bool) []placement.CalibrationCandidate {
	if req == nil || cfg == nil || model == nil || be == nil || caps == nil || strategy == nil {
		return nil
	}
	if strategy.PerformanceTuned {
		return nil
	}
	mode := effectiveCalibrationMode(req)
	if mode == calibrateOff {
		return nil
	}
	if mode == calibrateAuto && (req.Benchmark || req.WorkerBenchmark) {
		// One-shot benchmark commands measure the requested baseline; recursively
		// benchmarking optimizer candidates first would make their output
		// non-reproducible. Explicit maintenance mode may still request it.
		return nil
	}
	scopeKey := calibrationScopeKey(req, model, be, caps, strategy)
	candidates := calibrationCandidates(req, cfg, model, be, caps, strategy)
	if len(rejected) > 0 && rejected[0] != nil {
		candidates = filterCalibrationCandidates(candidates, rejected[0])
	}
	if len(candidates) < 2 {
		return nil
	}
	var cachedDecision *placement.CalibrationDecision
	if decision, err := placement.LoadCalibrationDecision(cfg.CacheDir, scopeKey); err == nil {
		cachedDecision = decision
		// A reusable recorded decision (including "default won") makes the
		// controller finite. A merely screened explicit result is intentionally
		// not reusable by auto mode and therefore must not suppress the core
		// workflow search that can promote complete evidence. Likewise, a named
		// winner that no longer exists in the deterministic candidate set cannot
		// block a fresh bounded search.
		reusable := mode == calibrateOn || decision.AutomaticEligible()
		if reusable && decision.Winner == "default" {
			return nil
		}
		if reusable {
			for _, candidate := range candidates {
				if candidate.Name == decision.Winner {
					return nil
				}
			}
		}
		if mode == calibrateAuto && decision.ValidationLevel == placement.CalibrationValidationAdmission {
			// Negative evidence is coordinate-local. Remove every exact finalist
			// this scope already disproved, then let the bounded planner choose the
			// next legal coordinate. Returning nil here froze the entire optimizer
			// after one oversized ubatch failed and prevented topology/hot-cache
			// evidence from ever being collected.
			filtered := candidates[:1]
			for _, candidate := range candidates[1:] {
				if !decision.SuppressesAutomaticAdmissionRetry(candidate.Name) {
					filtered = append(filtered, candidate)
				}
			}
			candidates = filtered
			if len(candidates) < 2 {
				return nil
			}
		}
	}
	budget := calibrationBudgetFor(mode)
	if mode == calibrateAuto {
		candidates = automaticCalibrationFinalistPlan(req, cfg, model, be, caps, candidates)
	} else {
		// Explicit maintenance is the scientific sweep. Batch search can emit
		// dozens of coordinates before slot count is appended; without family
		// ordering the nine-load budget never reached p1/p2/p4 at all.
		candidates = prioritizeParallelCalibrationCurve(candidates)
	}
	if len(candidates) < 2 {
		return nil
	}
	// Compare negative admission evidence with the finalist that automatic mode
	// will actually load, not the first raw candidate. The finalist planner can
	// reorder the calculated set, and comparing before that step caused the same
	// failed argv to be retried on every launch.
	if mode == calibrateAuto && cachedDecision != nil &&
		cachedDecision.SuppressesAutomaticAdmissionRetry(candidates[1].Name) {
		return nil
	}
	if len(candidates) > budget.MaxCandidates {
		candidates = candidates[:budget.MaxCandidates]
	}
	return candidates
}

// prioritizeParallelCalibrationCurve guarantees the bounded maintenance plan
// observes the useful power-of-two scheduler curve before spending the rest of
// its load budget on dense batch coordinates. Every candidate remains in the
// result; this changes experiment order, not feasibility or promotion policy.
func prioritizeParallelCalibrationCurve(candidates []placement.CalibrationCandidate) []placement.CalibrationCandidate {
	if len(candidates) < 2 {
		return candidates
	}
	out := make([]placement.CalibrationCandidate, 0, len(candidates))
	out = append(out, candidates[0])
	used := make([]bool, len(candidates))
	used[0] = true
	for _, want := range []int{1, 2, 4, 8} {
		for i := 1; i < len(candidates); i++ {
			candidate := candidates[i]
			if used[i] || !strings.HasPrefix(candidate.Name, "parallel-") ||
				candidate.Strategy == nil || candidate.Strategy.Parallel != want {
				continue
			}
			out = append(out, candidate)
			used[i] = true
			break
		}
	}
	for i := 1; i < len(candidates); i++ {
		if !used[i] {
			out = append(out, candidates[i])
		}
	}
	return out
}

// filterCalibrationCandidates removes only challengers. Candidate zero is the
// currently serving baseline and must always remain available for measurement
// and restoration. This lets a recovered launch continue into the performance
// lane without ever retrying an argv already disproved in the same lifecycle.
func filterCalibrationCandidates(candidates []placement.CalibrationCandidate, rejected func(*placement.Strategy) bool) []placement.CalibrationCandidate {
	if len(candidates) < 2 || rejected == nil {
		return candidates
	}
	out := make([]placement.CalibrationCandidate, 0, len(candidates))
	out = append(out, candidates[0])
	for _, candidate := range candidates[1:] {
		if candidate.Strategy == nil || rejected(candidate.Strategy) {
			continue
		}
		out = append(out, candidate)
	}
	return out
}

// automaticCalibrationFinalistPlan reduces the calculated candidate set to the
// stable baseline plus one live finalist. The placement package first rejects
// candidates that do not fit its common device/host ledger and orders the rest
// with a topology-aware relative performance model. Existing exact allocation
// evidence wins ties because it can skip an otherwise speculative admission;
// the live agent benchmark remains the only authority that can promote a win.
func automaticCalibrationFinalistPlan(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, candidates []placement.CalibrationCandidate) []placement.CalibrationCandidate {
	if len(candidates) < 2 {
		return candidates
	}
	candidates = automaticWorkloadCandidateSet(req, candidates)
	if len(candidates) < 2 {
		return candidates
	}
	cacheDir := ""
	if cfg != nil {
		cacheDir = cfg.CacheDir
	}
	opts := placementOptionsFromRequest(req, model, be, cacheDir)
	candidates = placement.AnalyzeCandidateFrontier(caps, model, opts, candidates)
	if len(candidates) > 0 && candidates[0].Strategy != nil &&
		candidates[0].Strategy.Residency == placement.ResidencyNonResident {
		// Required mmap/dense CPU offload is the last-resort fit lane. Its stable
		// baseline may collect evidence, but ordinary launch must not pay another
		// very long load to chase a low-confidence performance estimate. An
		// explicit --calibrate on maintenance run remains available.
		return candidates[:1]
	}
	ranked := candidates
	if len(candidates) > 0 && candidates[0].Strategy != nil &&
		candidates[0].Strategy.Residency == placement.ResidencyTight &&
		candidates[0].Strategy.ResourceLedger != nil &&
		candidates[0].Strategy.ResourceLedger.Exact {
		// Tight launches keep the allocation-proven shape, except a denser
		// GPU-expert pack may still be live-tested so a too-conservative baseline
		// can recover. Topology reshuffles remain calculated-only. Without an
		// exact ledger the launch is merely unmeasured, not proven tight.
		if tight := placement.TightLiveCandidates(candidates); len(tight) >= 2 {
			ranked = tight
		} else {
			return candidates[:1]
		}
	} else {
		feasible := make([]placement.CalibrationCandidate, 0, len(candidates))
		if len(candidates) > 0 {
			feasible = append(feasible, candidates[0])
		}
		for _, candidate := range candidates[1:] {
			if candidate.Estimate.Feasible || calibrationCandidateIsHotExperts(candidate.Name) {
				feasible = append(feasible, candidate)
			}
		}
		if len(feasible) >= 2 {
			ranked = feasible
		}
	}
	// An estimated miss is not a product gate on a roomy launch. Contained
	// admission remains the fail-closed proof.
	return selectAutomaticCalibrationAdmissionPlan(ranked, func(strategy *placement.Strategy) bool {
		return strategy != nil && strategy.ResourceLedger != nil &&
			strategy.ResourceLedger.Exact && strategy.ResourceLedger.Fits
	}, calibrationAutoMaxCandidates)
}

// automaticWorkloadCandidateSet removes slot-width experiments that cannot
// serve another runnable request in the declared workload. An idle llama.cpp
// slot does not speed an active sequence; it only consumes sequence state and
// divides the guaranteed context window. Keep the serving baseline and every
// non-slot coordinate so a configured wider baseline can still be compared
// with the useful p1/p2 curve. Explicit maintenance retains the full p1..p8
// experiment set through its separate path.
func automaticWorkloadCandidateSet(req *launchRequest, candidates []placement.CalibrationCandidate) []placement.CalibrationCandidate {
	if len(candidates) < 2 || candidates[0].Strategy == nil {
		return candidates
	}
	demand := max(1, requestWorkloadConcurrency(req))
	baselineParallel := max(1, candidates[0].Strategy.Parallel)
	out := make([]placement.CalibrationCandidate, 0, len(candidates))
	out = append(out, candidates[0])
	for _, candidate := range candidates[1:] {
		if candidate.Strategy != nil {
			parallel := max(1, candidate.Strategy.Parallel)
			if parallel != baselineParallel && parallel > demand {
				continue
			}
		}
		out = append(out, candidate)
	}
	return out
}

// selectAutomaticCalibrationAdmissionPlan keeps one calculated primary plus a
// short sequence of already-ranked fallbacks. runCalibration stops after the
// first challenger that both starts exactly and completes its measurement, so
// these extra entries increase admission robustness without turning ordinary
// launch into a successful-candidate sweep.
func selectAutomaticCalibrationAdmissionPlan(candidates []placement.CalibrationCandidate, hasExactAllocation func(*placement.Strategy) bool, limit int) []placement.CalibrationCandidate {
	if len(candidates) < 2 || limit < 2 {
		return candidates
	}
	primary := selectAutomaticCalibrationFinalist(candidates, hasExactAllocation)
	if len(primary) < 2 {
		return primary
	}
	out := make([]placement.CalibrationCandidate, 0, min(limit, len(candidates)))
	out = append(out, candidates[0], primary[1])
	if calibrationCandidateIsHotExperts(primary[1].Name) && len(out) < limit {
		primarySlots := 0
		if primary[1].Strategy != nil {
			primarySlots = primary[1].Strategy.HotExpertCacheSlots
		}
		for _, candidate := range candidates[1:] {
			if candidate.Name == primary[1].Name || !calibrationCandidateIsHotExperts(candidate.Name) {
				continue
			}
			if candidate.Strategy == nil || candidate.Strategy.HotExpertCacheSlots >= primarySlots {
				continue
			}
			out = append(out, candidate)
			break
		}
	}
	// If the predicted primary is a batch/topology coordinate, keep one legal
	// slot-count fallback in the bounded admission ladder. This is especially
	// important after a high-ubatch candidate fails: retrying two more members
	// of the same family teaches nothing about aggregate agent throughput.
	if !strings.HasPrefix(primary[1].Name, "parallel-") && len(out) < limit {
		for _, candidate := range candidates[1:] {
			if strings.HasPrefix(candidate.Name, "parallel-") {
				out = append(out, candidate)
				break
			}
		}
	}
	for _, candidate := range candidates[1:] {
		if len(out) >= limit {
			break
		}
		duplicate := false
		for _, selected := range out[1:] {
			if candidate.Name == selected.Name {
				duplicate = true
				break
			}
		}
		if duplicate {
			continue
		}
		out = append(out, candidate)
	}
	return out
}

func selectAutomaticCalibrationFinalist(candidates []placement.CalibrationCandidate, hasExactAllocation func(*placement.Strategy) bool) []placement.CalibrationCandidate {
	if len(candidates) < 2 {
		return candidates
	}
	if hot, ok := firstHotExpertCalibrationCandidate(candidates); ok {
		// Auto asked to turn the cache on. The relative cost model does not
		// price cache hits, so a cheaper-looking KV/topology neighbor must not
		// consume the single live A/B slot. Exact allocation still prefers a
		// measured cache-on shape among hot-expert candidates.
		if hasExactAllocation != nil {
			for _, candidate := range candidates[1:] {
				if !calibrationCandidateIsHotExperts(candidate.Name) {
					continue
				}
				if hasExactAllocation(candidate.Strategy) {
					return []placement.CalibrationCandidate{candidates[0], candidate}
				}
			}
		}
		return []placement.CalibrationCandidate{candidates[0], hot}
	}
	if hasExactAllocation != nil {
		bestCost := candidates[1].Estimate.AgentCost
		for _, candidate := range candidates[1:] {
			candidateCost := candidate.Estimate.AgentCost
			withinEstimateTie := bestCost <= 0 || candidateCost <= 0 ||
				candidateCost <= bestCost*(1+calibrationAdvisorTiePct/100)
			if withinEstimateTie && hasExactAllocation(candidate.Strategy) {
				return []placement.CalibrationCandidate{candidates[0], candidate}
			}
		}
	}
	// No alternate shares exact allocation evidence yet. The candidate generator
	// has already run every option through a complete placement computation and
	// orders nearest coordinate changes first, so validate only its first
	// calculated finalist. Contained admission remains the fail-closed proof.
	return []placement.CalibrationCandidate{candidates[0], candidates[1]}
}

func calibrationCandidateIsHotExperts(name string) bool {
	return strings.HasPrefix(strings.TrimSpace(name), "hot-experts-")
}

func firstHotExpertCalibrationCandidate(candidates []placement.CalibrationCandidate) (placement.CalibrationCandidate, bool) {
	for _, candidate := range candidates[1:] {
		if !calibrationCandidateIsHotExperts(candidate.Name) {
			continue
		}
		if candidate.Strategy == nil || candidate.Strategy.HotExpertCacheSlots <= 0 {
			continue
		}
		return candidate, true
	}
	return placement.CalibrationCandidate{}, false
}

func keepHotExpertAutoFinalist(candidates []placement.CalibrationCandidate, incoming placement.CalibrationCandidate) bool {
	if len(candidates) < 2 || !calibrationCandidateIsHotExperts(candidates[1].Name) {
		return false
	}
	return !calibrationCandidateIsHotExperts(incoming.Name)
}

// applyCalibrationDecision returns the strategy to serve with when a prior
// calibration decision exists for this scope. The winner is re-derived by name
// from the deterministic candidate generator rather than deserialized, so the
// full placement (KV placement, main GPU, split) is reproduced exactly and can
// never be a partial overlay. Context, batch, and slots always come from the
// current request. Returns the input strategy unchanged when no decision
// applies or the winner's candidate no longer exists for this hardware.
func applyCalibrationDecision(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy) *placement.Strategy {
	if req == nil || cfg == nil || model == nil || be == nil || caps == nil || strategy == nil {
		return strategy
	}
	if req.Calibrate == calibrateOff {
		return strategy
	}
	// --no-cached-config is the escape hatch for a stale cached decision: do not
	// re-apply a measured calibration winner to this launch either.
	if req.NoCachedConfig {
		return strategy
	}
	// The decision was measured after the Claude slot/fairness baseline was
	// finalized. Reproduce that same base before deriving its scope key; looking
	// up against placement's cold 4096/512 pair can never find a decision saved
	// under the effective 128/128 workload. A returned batch candidate carries
	// BatchTuned, so later idempotent slot adjustment preserves the measured win.
	claudeCodeSlotAdjust(strategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	scopeKey := calibrationScopeKey(req, model, be, caps, strategy)
	decision, err := placement.LoadCalibrationDecision(cfg.CacheDir, scopeKey)
	if err != nil || decision.Winner == "" {
		return strategy
	}
	mode := effectiveCalibrationMode(req)
	if mode != calibrateOn && !decision.AutomaticEligible() {
		return strategy
	}
	if decision.Winner == "default" {
		strategy.PerformanceTuned = true
		attachOptimizationEvidence(strategy, decision)
		req.AppliedCalibration = decision
		printOptimizationReuse(decision)
		return strategy
	}
	for _, cand := range calibrationCandidates(req, cfg, model, be, caps, strategy) {
		if cand.Name == decision.Winner {
			req.CalibrationScreened = !decision.AutomaticEligible()
			cand.Strategy.PerformanceTuned = true
			attachOptimizationEvidence(cand.Strategy, decision)
			req.AppliedCalibration = decision
			printOptimizationReuse(decision)
			return cand.Strategy
		}
	}
	return strategy
}

// attachOptimizationEvidence copies the finite recorded search onto the serving
// strategy so later launches and dry-runs explain the same settled boundary
// instead of re-analyzing a one-candidate placeholder.
func attachOptimizationEvidence(strategy *placement.Strategy, decision *placement.CalibrationDecision) {
	if strategy == nil || decision == nil {
		return
	}
	if decision.ExploredBoundary != nil {
		strategy.OptimizationBoundary = decision.ExploredBoundary
	}
	if strategy.Residency == "" && decision.BaselineResidency != "" {
		strategy.Residency = decision.BaselineResidency
	}
	if strategy.OptimizationBottleneck == "" && decision.BaselineBottleneck != "" {
		strategy.OptimizationBottleneck = decision.BaselineBottleneck
	}
}

func printOptimizationReuse(decision *placement.CalibrationDecision) {
	if decision == nil || decision.Winner == "" {
		return
	}
	label := "optimize"
	if decision.ValidationLevel == placement.CalibrationValidationScreened {
		label = "calibrate"
	}
	outcome := decision.FinalistOutcome
	if outcome == "" {
		if decision.Winner == "default" {
			outcome = "baseline-won"
		} else {
			outcome = "promoted"
		}
	}
	explored := ""
	if b := decision.ExploredBoundary; b != nil {
		explored = fmt.Sprintf("; explored %d candidates (%d feasible), batch %d..%d, ubatch %d..%d, parallel %d..%d",
			b.CandidateCount, b.FeasibleCount, b.MinBatch, b.MaxBatch, b.MinUBatch, b.MaxUBatch, b.MinParallel, b.MaxParallel)
	}
	measured := ""
	if decision.MeasuredAt != "" {
		measured = ", measured " + decision.MeasuredAt
	}
	fmt.Printf("[%s] reusing %s %s (%s%s, agent workload %.2fs vs default %.2fs%s)\n",
		label, decision.ValidationLevel, decision.Winner, outcome, measured,
		decision.WinnerTurnTimeS, decision.DefaultTurnTimeS, explored)
}

// explainServingOptimization annotates the strategy that launch/dry-run will
// actually serve. A cached winner already carries its recorded boundary; a cold
// estimate may compute the full frontier for inspectability, but that ranking
// never replaces a measured winner with an unbenchmarked neighbor.
func explainServingOptimization(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy, fullFrontier bool) *placement.Strategy {
	if strategy == nil {
		return nil
	}
	if strategy.OptimizationBoundary != nil && (!fullFrontier || strategy.PerformanceTuned) {
		return strategy
	}
	cacheDir := ""
	if cfg != nil {
		cacheDir = cfg.CacheDir
	}
	opts := placementOptionsFromRequest(req, model, be, cacheDir)
	if fullFrontier && !strategy.PerformanceTuned {
		frontier := calibrationCandidates(req, cfg, model, be, caps, strategy)
		if len(frontier) == 0 {
			frontier = []placement.CalibrationCandidate{{Name: "default", Strategy: strategy}}
		}
		frontier = placement.AnalyzeCandidateFrontier(caps, model, opts, frontier)
		if len(frontier) > 0 && frontier[0].Strategy != nil {
			return frontier[0].Strategy
		}
		return strategy
	}
	if strategy.OptimizationBoundary != nil {
		return strategy
	}
	analyzed := placement.AnalyzeCandidateFrontier(caps, model, opts, []placement.CalibrationCandidate{{Name: "default", Strategy: strategy}})
	if len(analyzed) > 0 && analyzed[0].Strategy != nil {
		return analyzed[0].Strategy
	}
	return strategy
}

// recomputeAndApplyCalibration re-derives placement from current evidence and
// re-applies the cached calibration winner on top, so a corrective recompute
// never replaces the applied winner with an unbenchmarked estimate. Mirror of
// the computeStrategy path in main(); SkipPlacementCache and CacheFile are
// caller-managed because a mid-retry placement is never persisted.
func recomputeAndApplyCalibration(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, s *placement.Strategy) *placement.Strategy {
	return applyCalibrationDecision(req, cfg, model, be, caps, s)
}

// calibrationScopeKey builds the opaque cache key for this launch shape. It
// mirrors placement.NewCalibrationScopeKey on the request's resolved options so
// a decision is valid only for the exact model + backend + hardware + workload
// + runtime knobs it was measured under.
func calibrationCandidates(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy) []placement.CalibrationCandidate {
	cacheDir := ""
	if cfg != nil {
		cacheDir = cfg.CacheDir
	}
	opts := placementOptionsFromRequest(req, model, be, cacheDir)
	candidates := placement.CalibrationCandidates(caps, model, strategy, opts)
	if req == nil || req.ForceMMap || len(candidates) < 2 {
		return candidates
	}
	// Do not stop a resident server and then ask for a new disk-paging policy
	// halfway through calibration. An mmap-dependent alternate is eligible only
	// when the user approved mmap on the original command line or launch prompt.
	out := candidates[:1]
	for _, cand := range candidates[1:] {
		if cand.Strategy != nil && !cand.Strategy.MMapRequired {
			out = append(out, cand)
		}
	}
	return out
}

func calibrationScopeKey(req *launchRequest, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy) string {
	opts := placementOptionsFromRequest(req, model, be, "")
	key := placement.NewCalibrationScopeKey(model, caps, opts, strategy)
	return key.String()
}

type calibrationMeasurement struct {
	Name     string
	Strategy *placement.Strategy
	Args     []string
	Result   *benchmark.Result
	Score    float64
}

func exactCalibrationCandidate(candidateArgs, measuredArgs []string) bool {
	return formatCommand(candidateArgs) == formatCommand(measuredArgs)
}

func validCalibrationResult(result *benchmark.Result) bool {
	if result == nil || result.GenTPS <= 0 || result.PromptTPS <= 0 || result.GenTokens <= 0 || result.PromptTokens <= 0 {
		return false
	}
	if math.IsNaN(result.GenTPS) || math.IsInf(result.GenTPS, 0) ||
		math.IsNaN(result.PromptTPS) || math.IsInf(result.PromptTPS, 0) {
		return false
	}
	if result.AgentSamples > 0 {
		return result.MixedGenTokens > 0 && result.MixedGenTPS > 0 &&
			result.AgentSamples >= 2 && result.AgentTurnTimeS > 0 && result.AgentTurnMaxS > 0 &&
			result.AgentScenarioTimeS > 0 && result.AgentScenarioMaxS > 0 && result.AgentPromptBytes > 0 &&
			result.AgentCachedTokens > 0 && result.AgentNewPromptTokens > 0 &&
			!math.IsNaN(result.MixedGenTPS) && !math.IsInf(result.MixedGenTPS, 0)
	}
	return result.GenTimeS > 0 && !math.IsNaN(result.GenTimeS) && !math.IsInf(result.GenTimeS, 0)
}

func automaticCalibrationEvidenceValid(decision *placement.CalibrationDecision) bool {
	if decision == nil {
		return false
	}
	// A baseline copied into the winner fields after every challenger failed is
	// useful stability evidence, but it is not comparative performance evidence.
	// Require an actually measured challenger before automatic launch may call
	// either that challenger or the baseline a workflow winner.
	challengerMeasured := decision.Finalist != "" &&
		(decision.FinalistOutcome == "promoted" || decision.FinalistOutcome == "baseline-won")
	return challengerMeasured &&
		decision.DefaultAgentSamples >= 2 && decision.WinnerAgentSamples >= 2 &&
		decision.DefaultTurnTimeS > 0 && decision.WinnerTurnTimeS > 0 &&
		decision.DefaultTurnMaxS > 0 && decision.WinnerTurnMaxS > 0 &&
		decision.AgentPromptBytes > 0 &&
		decision.DefaultCachedTokens > 0 && decision.WinnerCachedTokens > 0 &&
		decision.DefaultNewPromptTokens > 0 && decision.WinnerNewPromptTokens > 0 &&
		decision.DefaultMixedTPS > 0 && decision.WinnerMixedTPS > 0
}

func automaticCalibrationAdmissionEvidenceValid(decision *placement.CalibrationDecision) bool {
	if decision == nil {
		return false
	}
	// This state records only deterministic exact-admission or feature-lifecycle
	// failure. A benchmark timeout, malformed response, incomplete workload, or
	// untested estimate must retry later rather than becoming a negative result.
	return decision.Winner == "default" &&
		decision.Finalist != "" &&
		decision.FinalistOutcome == "unavailable" &&
		decision.FinalistFailureClass != "" &&
		decision.FinalistFailureReason != ""
}

func exactAdmissionFailureEvidence(err error) (string, string) {
	var failure *exactAdmissionFailure
	if !errors.As(err, &failure) || failure == nil {
		return "", ""
	}
	return string(failure.class), failure.Error()
}

func calibrationTurnTime(result *benchmark.Result) float64 {
	if result == nil {
		return 0
	}
	if result.AgentWorkloadTimeS > 0 {
		return result.AgentWorkloadTimeS
	}
	if result.AgentScenarioTimeS > 0 {
		return result.AgentScenarioTimeS
	}
	if result.AgentSamples > 0 {
		return result.AgentTurnTimeS
	}
	// Runner.Run's generation request includes both prompt ingestion and decode,
	// so its request wall time is already the end-to-end serial turn. Adding the
	// separate prefill probe would double-count prompt work.
	return result.GenTimeS
}

// calibrationWorstWorkloadTime compares confirmation samples at the same
// requested agent concurrency. Raw per-wave latency is not comparable across
// slot widths because a wider server intentionally performs more work in each
// wave.
func calibrationWorstWorkloadTime(result *benchmark.Result) float64 {
	if result == nil {
		return 0
	}
	if result.AgentWorkloadMaxS > 0 {
		return result.AgentWorkloadMaxS
	}
	if result.AgentScenarioMaxS > 0 {
		return result.AgentScenarioMaxS
	}
	worst := result.AgentTurnMaxS
	if worst <= 0 {
		return 0
	}
	lanes := max(1, result.AgentWorkloadLanes)
	active := max(1, result.Parallel)
	if active > lanes {
		active = lanes
	}
	waves := (lanes + active - 1) / active
	return worst * float64(waves)
}

// calibrationScore is a direct relative workflow-time objective: baseline
// cold-ingest-plus-append wall time divided by the candidate's identical
// scenario. Throughput remains visible for diagnosis and tie-breaking but
// cannot outweigh a slower user-visible workflow.
func calibrationScore(result, baseline *benchmark.Result) float64 {
	if !validCalibrationResult(result) || !validCalibrationResult(baseline) {
		return 0
	}
	if result.AgentSamples > 0 && result.AgentPromptBytes != baseline.AgentPromptBytes {
		return 0
	}
	baseTime, resultTime := calibrationTurnTime(baseline), calibrationTurnTime(result)
	if baseTime <= 0 || resultTime <= 0 {
		return 0
	}
	return baseTime / resultTime
}

func calibrationCandidateBetter(candidate, current calibrationMeasurement) bool {
	if candidate.Score <= current.Score*(1+calibrationMinImprovementPct/100) {
		return false
	}
	if candidate.Strategy != nil && current.Strategy != nil &&
		candidate.Strategy.HotExpertCacheSlots > current.Strategy.HotExpertCacheSlots {
		// This backend feature is decode-only. A faster workflow score without a
		// material pure-decode gain is noise or an unrelated phase effect, not
		// evidence that caching host experts helped.
		if candidate.Result == nil || current.Result == nil || current.Result.GenTPS <= 0 ||
			candidate.Result.GenTPS <= current.Result.GenTPS*(1+calibrationMinImprovementPct/100) {
			return false
		}
	}
	if candidate.Result != nil && current.Result != nil {
		floor := 1 - calibrationMaxPhaseRegressionPct/100
		for _, phase := range [][2]float64{
			{candidate.Result.PromptTPS, current.Result.PromptTPS},
			{candidate.Result.GenTPS, current.Result.GenTPS},
			{candidate.Result.MixedGenTPS, current.Result.MixedGenTPS},
		} {
			if phase[1] > 0 && phase[0] < phase[1]*floor {
				return false
			}
		}
	}
	if candidate.Result != nil && current.Result != nil && candidate.Result.AgentSamples > 0 {
		// The mean must win beyond the noise floor and the slower confirmation
		// sample must not regress. This prevents one lucky append from winning.
		return calibrationWorstWorkloadTime(candidate.Result) <= calibrationWorstWorkloadTime(current.Result)
	}
	return true
}

func boundedCalibrationTimeout(configured, remaining time.Duration) time.Duration {
	if configured <= 0 || configured > remaining {
		return remaining
	}
	return configured
}

// runCalibration measures each candidate placement/runtime pair with a live
// workload benchmark and returns the strategy to actually serve with. Automatic
// mode walks a bounded ladder even for very large models, but a
// start/failure/elapsed budget makes the controller finite. The optional support
// optimizer only sees typed, already-measured candidates after every main-model
// process has stopped.
//
// The decision is returned pending: the caller persists it only after the
// winner passes the normal functional/cache/profile lifecycle and becomes
// active. A benchmark alone must never become last-known-good state.
func runCalibration(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy, serverArgs []string, timeout time.Duration, p *server.Process, memoryRecovery *launchMemoryRecovery, resourceBaseline *detect.Capabilities, planned ...[]placement.CalibrationCandidate) (*server.Process, *placement.Strategy, []string, *placement.CalibrationDecision) {
	candidates := []placement.CalibrationCandidate(nil)
	if len(planned) > 0 {
		candidates = planned[0]
	} else {
		candidates = calibrationPlan(req, cfg, model, be, caps, strategy)
	}
	if len(candidates) < 2 {
		return p, strategy, serverArgs, nil
	}
	mode := effectiveCalibrationMode(req)
	budget := calibrationBudgetFor(mode)
	scopeKey := calibrationScopeKey(req, model, be, caps, strategy)
	startedAt := time.Now()
	if mode == calibrateAuto {
		fmt.Printf("[optimize] measuring the live baseline plus one successful calculated challenger (up to %d contained admissions, elapsed budget=%s)\n",
			max(1, len(candidates)-1), budget.MaxElapsed)
		printOptimizationSummary("optimize", candidates[0].Strategy, false)
		finalist := candidates[1]
		relative := 0.0
		if candidates[0].Estimate.AgentCost > 0 && finalist.Estimate.AgentCost > 0 &&
			!math.IsInf(candidates[0].Estimate.AgentCost, 0) && !math.IsInf(finalist.Estimate.AgentCost, 0) {
			relative = candidates[0].Estimate.AgentCost / finalist.Estimate.AgentCost
		}
		fmt.Printf("[optimize] calculated finalist %s: predicted relative %.3f, bottleneck %s, confidence %s; live agent workflows decide\n",
			finalist.Name, relative, finalist.Estimate.Bottleneck, finalist.Estimate.Confidence)
	} else {
		fmt.Printf("[calibrate] explicit maintenance sweep: measuring at most %d placements (failures=%d, elapsed=%s)\n",
			budget.MaxCandidates, budget.MaxFailures, budget.MaxElapsed)
	}

	baseURL := fmt.Sprintf("http://localhost:%d", req.Port)
	workloadLanes := requestWorkloadConcurrency(req)
	utilizationCaps, _ := runtimeGPUCapabilities(caps, req)
	agentPromptBytes := 0
	if mode == calibrateAuto {
		activeSlots := 1
		if strategy != nil && strategy.Parallel > 1 {
			activeSlots = strategy.Parallel
		}
		if activeSlots > workloadLanes {
			activeSlots = workloadLanes
		}
		// RunAgentParallel deliberately caps a bounded screen at four lanes.
		if activeSlots > 4 {
			activeSlots = 4
		}
		maxUBatch := 0
		for _, candidate := range candidates {
			if candidate.Strategy != nil && candidate.Strategy.UBatchSize > maxUBatch {
				maxUBatch = candidate.Strategy.UBatchSize
			}
		}
		probeTimeout := calibrationAutoProbeTimeout
		if remaining := budget.MaxElapsed - time.Since(startedAt); remaining/4 < probeTimeout {
			probeTimeout = remaining / 4
		}
		probeRunner := &benchmark.Runner{
			BaseURL: baseURL, Model: model.Basename, Timeout: probeTimeout, WorkloadID: scopeKey,
		}
		probe, probeErr := probeRunner.ProbeAgentPrefill()
		agentPromptBytes = benchmark.SuggestedAgentPromptBytes(probe, activeSlots, maxUBatch, calibrationAutoAgentWaveTarget)
		if probeErr != nil {
			fmt.Fprintf(os.Stderr, "[optimize] short prefill pilot failed (%v); using bounded %d-byte/lane fallback\n",
				probeErr, agentPromptBytes)
		} else {
			approxTokens := agentPromptBytes / 4
			if probe.PromptBytes > 0 && probe.PromptTokens > 0 {
				approxTokens = agentPromptBytes * probe.PromptTokens / probe.PromptBytes
			}
			fmt.Printf("[optimize] prefill pilot %.1f tok/s; bounded screen uses %d bytes (~%d tokens) per lane for both placements\n",
				probe.PromptTPS, agentPromptBytes, approxTokens)
		}
	}
	bench := func(active *placement.Strategy, activeProcess *server.Process) (*benchmark.Result, error) {
		remaining := budget.MaxElapsed - time.Since(startedAt)
		if remaining <= 3*time.Second {
			return nil, fmt.Errorf("calibration elapsed-time budget exhausted")
		}
		// Bound each request by a fraction of the remaining controller budget. The
		// agent runner issues several requests concurrently, so this is deliberately
		// a per-request guard rather than an estimate of total call count.
		requestTimeout := 5 * time.Minute
		if perRequest := remaining / 3; perRequest < requestTimeout {
			requestTimeout = perRequest
		}
		runner := &benchmark.Runner{
			BaseURL: baseURL, Model: model.Basename, Timeout: requestTimeout,
			WorkloadID: scopeKey, AgentPromptBytes: agentPromptBytes,
			SampleResources: calibrationProcessResourceSampler(activeProcess),
			SampleGPUUtilization: func() []benchmark.GPUUtilization {
				return sampleLaunchGPUUtilization(utilizationCaps)
			},
		}
		slots := 1
		if active != nil && active.Parallel > 1 {
			slots = active.Parallel
		}
		if slots > workloadLanes {
			slots = workloadLanes
		}
		res, err := runner.RunAgentParallel(slots)
		if err != nil {
			return nil, err
		}
		if !validCalibrationResult(res) {
			return nil, fmt.Errorf("benchmark returned incomplete metrics")
		}
		waves := (workloadLanes + slots - 1) / slots
		res.AgentWorkloadLanes = workloadLanes
		res.AgentWorkloadTimeS = res.AgentScenarioTimeS * float64(waves)
		res.AgentWorkloadMaxS = res.AgentScenarioMaxS * float64(waves)
		return res, nil
	}
	validateHotExpertRuntime := func(active *placement.Strategy, activeProcess *server.Process) (bool, error) {
		if active == nil || active.HotExpertCacheSlots <= 0 {
			return true, nil
		}
		remaining := budget.MaxElapsed - time.Since(startedAt)
		if remaining <= 3*time.Second {
			return false, fmt.Errorf("calibration elapsed-time budget exhausted before hot-expert telemetry canary")
		}
		requestTimeout := 5 * time.Minute
		if available := remaining - 3*time.Second; available < requestTimeout {
			requestTimeout = available
		}
		runner := &benchmark.Runner{
			BaseURL: baseURL, Model: model.Basename, Timeout: requestTimeout,
			WorkloadID: scopeKey,
		}
		generated, err := runner.RunHotExpertCacheTelemetryCanary()
		if err != nil {
			// A completed response that ignored the forced decode length is stable
			// incompatibility evidence. Transport/timeouts stay retryable.
			return errors.Is(err, benchmark.ErrHotExpertCanaryIncomplete), err
		}
		if activeProcess == nil || activeProcess.LogBuf == nil {
			return false, fmt.Errorf("hot-expert candidate has no captured backend log")
		}
		telemetry, err := placement.ValidateHotExpertCacheTelemetry(active, activeProcess.LogBuf.String())
		if err != nil {
			// A completed deterministic canary plus an invalid/missing backend report
			// is stable capability evidence for this exact backend/model scope.
			return true, err
		}
		active.HotExpertCacheSteps = telemetry.Steps
		active.HotExpertCacheHits = telemetry.Hits
		active.HotExpertCacheMisses = telemetry.Misses
		active.HotExpertCacheHitRate = telemetry.HitRate
		active.HotExpertCacheEvidence += "; measured aggregate runtime hit/miss telemetry"
		fmt.Printf("[hot-experts] runtime canary generated %d tokens; steps=%d hits=%d misses=%d hit-rate=%.1f%%\n",
			generated, telemetry.Steps, telemetry.Hits, telemetry.Misses, telemetry.HitRate)
		return true, nil
	}

	// The default is already running: measure it in place.
	defaultResult, err := bench(strategy, p)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[calibrate] baseline measurement failed (%v); serving default placement\n", err)
		return p, strategy, serverArgs, nil
	}
	measurements := []calibrationMeasurement{{
		Name: "default", Strategy: strategy, Args: serverArgs,
		Result: defaultResult, Score: 1,
	}}
	fmt.Printf("[calibrate] default: workload makespan %.2fs, decode %.1f tok/s, prefill %.1f tok/s\n",
		calibrationTurnTime(defaultResult), defaultResult.GenTPS, defaultResult.PromptTPS)
	baselineDiagnosis := recordAgentBottleneckDiagnosis("default", caps, model, strategy, defaultResult)
	if bottleneck := deviceBalanceBottleneckFromResult(strategy, defaultResult); bottleneck != "" {
		fmt.Printf("[optimize] device balance: %s\n", bottleneck)
		if strategy != nil && (strategy.OptimizationBottleneck == "" || strings.Contains(strategy.OptimizationBottleneck, "layer service")) {
			strategy.OptimizationBottleneck = bottleneck
		}
	}
	if defaultResult.AgentSamples > 0 {
		fmt.Printf("[calibrate] default agent evidence: %d active lanes for %d-agent target, %d samples, %d bytes/lane, reuse >=%d tokens/lane, slowest workflow %.2fs, mixed foreground %.1f tok/s\n",
			defaultResult.Parallel, defaultResult.AgentWorkloadLanes, defaultResult.AgentSamples, defaultResult.AgentPromptBytes,
			defaultResult.AgentCachedTokens, calibrationWorstWorkloadTime(defaultResult), defaultResult.MixedGenTPS)
	}
	if mode == calibrateAuto {
		signal := deviceBalanceSignalFromResult(strategy, defaultResult)
		if signal.Imbalanced {
			measuredBottleneck := placement.DeviceBalanceBottleneck(strategy, gpuUtilSamplesFromResult(defaultResult))
			if finalist, ok := telemetryDirectedCalibrationFinalist(req, cfg, model, be, caps, strategy, signal, memoryRecovery); ok {
				if keepHotExpertAutoFinalist(candidates, finalist) {
					fmt.Printf("[optimize] keeping hot-expert finalist %s instead of telemetry-directed %s\n",
						candidates[1].Name, finalist.Name)
				} else {
					if candidates[1].Name != finalist.Name {
						fmt.Printf("[optimize] replacing calculated finalist %s with telemetry-directed %s to move work off saturated GPU %d\n",
							candidates[1].Name, finalist.Name, signal.BusyGPU)
					}
					candidates = prioritizeCalibrationFinalist(candidates, finalist)
				}
			} else {
				fmt.Printf("[optimize] measured device imbalance, but no non-rejected same-workload topology can relieve GPU %d; keeping calculated finalist %s\n",
					signal.BusyGPU, candidates[1].Name)
			}
			if measuredBottleneck != "" {
				strategy.OptimizationBottleneck = measuredBottleneck
			}
		}
		if finalist, ok := bottleneckDirectedCalibrationFinalist(req, cfg, model, be, caps, strategy, baselineDiagnosis, memoryRecovery); ok {
			if keepHotExpertAutoFinalist(candidates, finalist) {
				fmt.Printf("[optimize] keeping hot-expert finalist %s instead of phase-directed %s for %s/%s\n",
					candidates[1].Name, finalist.Name, baselineDiagnosis.PrimaryPhase, baselineDiagnosis.Primary)
			} else {
				if candidates[1].Name != finalist.Name {
					fmt.Printf("[optimize] replacing calculated finalist %s with phase-directed %s for %s/%s\n",
						candidates[1].Name, finalist.Name, baselineDiagnosis.PrimaryPhase, baselineDiagnosis.Primary)
				}
				candidates = prioritizeCalibrationFinalist(candidates, finalist)
			}
		}
	}

	curP := p
	failures := 0
	stableAdmissionFailed := false
	stableFailureClass := ""
	stableFailureReason := ""
	admissionInconclusive := false
	exactCandidateStarted := false
	stablePostStartFailure := false
	for _, cand := range candidates[1:] {
		remaining := budget.MaxElapsed - time.Since(startedAt)
		if remaining <= 3*time.Second {
			fmt.Fprintln(os.Stderr, "[calibrate] elapsed-time budget reached; stopping candidate search")
			admissionInconclusive = true
			break
		}
		candArgs := buildLaunchServerArgs(req, cfg, be, caps, model, cand.Strategy)
		if fmt.Sprintf("%v", candArgs) == fmt.Sprintf("%v", serverArgs) {
			continue // candidate serializes identically to the default; nothing to measure
		}
		if memoryRecovery.isRejected(candArgs) {
			fmt.Printf("[calibrate] skipping %s: its exact argv already failed memory admission in this launch\n", cand.Name)
			stableAdmissionFailed = true
			if stableFailureClass == "" {
				stableFailureClass = "cached-memory-rejection"
				stableFailureReason = "the finalist exact argv already failed memory admission in this launch"
			}
			continue
		}
		fmt.Printf("[calibrate] measuring %s...\n", cand.Name)
		if curP != nil {
			if !stopCalibrationProcessAndWait(curP, "default before "+cand.Name, resourceBaseline, 30*time.Second) {
				return curP, strategy, serverArgs, nil
			}
			curP = nil
		}
		candidateTimeout := boundedCalibrationTimeout(timeout, remaining)
		cp, measuredStrategy, measuredArgs, serr := startLaunchExactAdmission(req, cfg, model, cand.Strategy, be, caps, candArgs, candidateTimeout, memoryRecovery)
		if serr != nil {
			fmt.Fprintf(os.Stderr, "[calibrate] %s failed to start (%v); skipping\n", cand.Name, serr)
			if isStableExactAdmissionFailure(serr) {
				stableAdmissionFailed = true
				if stableFailureClass == "" {
					stableFailureClass, stableFailureReason = exactAdmissionFailureEvidence(serr)
				}
			} else {
				admissionInconclusive = true
			}
			failures++
			if failures >= budget.MaxFailures {
				fmt.Fprintln(os.Stderr, "[calibrate] failure budget reached; stopping candidate search")
				break
			}
			continue
		}
		// A larger ubatch candidate is useful only when that exact candidate
		// passed admission. Recovery may make it start by moving experts or
		// derating ubatch; measuring that different argv and caching the original
		// candidate name would replay an unverified shape on the next launch.
		if !exactCalibrationCandidate(candArgs, measuredArgs) {
			fmt.Fprintf(os.Stderr, "[calibrate] %s needed memory recovery; skipping the altered candidate\n", cand.Name)
			stableAdmissionFailed = true
			if stableFailureClass == "" {
				stableFailureClass = "memory-recovery-rewrote-argv"
				stableFailureReason = "the finalist started only after memory recovery changed its exact argv"
			}
			if !stopCalibrationProcessAndWait(cp, cand.Name+" after memory recovery", resourceBaseline, 30*time.Second) {
				req.CalibrationScreened = true
				return cp, measuredStrategy, measuredArgs, nil
			}
			failures++
			if failures >= budget.MaxFailures {
				fmt.Fprintln(os.Stderr, "[calibrate] failure budget reached; stopping candidate search")
				break
			}
			continue
		}
		exactCandidateStarted = true
		result, berr := bench(measuredStrategy, cp)
		if berr != nil {
			fmt.Fprintf(os.Stderr, "[calibrate] %s measurement failed (%v); skipping\n", cand.Name, berr)
			if class, reason, oom := candidatePostStartCUDAOOM(cp, berr); oom {
				stableAdmissionFailed = true
				stablePostStartFailure = true
				if stableFailureClass == "" {
					stableFailureClass, stableFailureReason = class, reason
				}
				memoryRecovery.reject(candArgs)
			} else {
				admissionInconclusive = true
			}
			if !stopCalibrationProcessAndWait(cp, cand.Name+" after failed measurement", resourceBaseline, 30*time.Second) {
				req.CalibrationScreened = true
				return cp, measuredStrategy, measuredArgs, nil
			}
			failures++
			if failures >= budget.MaxFailures {
				fmt.Fprintln(os.Stderr, "[calibrate] failure budget reached; stopping candidate search")
				break
			}
			continue
		}
		if measuredStrategy != nil && measuredStrategy.HotExpertCacheSlots > 0 {
			stable, telemetryErr := validateHotExpertRuntime(measuredStrategy, cp)
			if telemetryErr != nil {
				fmt.Fprintf(os.Stderr, "[calibrate] %s hot-expert runtime validation failed (%v); skipping\n", cand.Name, telemetryErr)
				if stable {
					stableAdmissionFailed = true
					stablePostStartFailure = true
					if stableFailureClass == "" {
						stableFailureClass = "hot-expert-telemetry"
						stableFailureReason = telemetryErr.Error()
					}
				} else {
					admissionInconclusive = true
				}
				if !stopCalibrationProcessAndWait(cp, cand.Name+" after hot-expert runtime validation", resourceBaseline, 30*time.Second) {
					req.CalibrationScreened = true
					return cp, measuredStrategy, measuredArgs, nil
				}
				failures++
				if mode == calibrateAuto || failures >= budget.MaxFailures {
					break
				}
				continue
			}
		}
		score := calibrationScore(result, defaultResult)
		fmt.Printf("[calibrate] %s: workload makespan %.2fs, decode %.1f tok/s, prefill %.1f tok/s, relative %.3f\n",
			cand.Name, calibrationTurnTime(result), result.GenTPS, result.PromptTPS, score)
		recordAgentBottleneckDiagnosis(cand.Name, caps, model, measuredStrategy, result)
		if result.AgentSamples > 0 {
			fmt.Printf("[calibrate] %s agent evidence: %d active lanes for %d-agent target, %d samples, %d bytes/lane, reuse >=%d tokens/lane, slowest workflow %.2fs, mixed foreground %.1f tok/s\n",
				cand.Name, result.Parallel, result.AgentWorkloadLanes, result.AgentSamples, result.AgentPromptBytes,
				result.AgentCachedTokens, calibrationWorstWorkloadTime(result), result.MixedGenTPS)
		}
		measurements = append(measurements, calibrationMeasurement{
			Name: cand.Name, Strategy: measuredStrategy, Args: measuredArgs,
			Result: result, Score: score,
		})
		if !stopCalibrationProcessAndWait(cp, "measured candidate "+cand.Name, resourceBaseline, 30*time.Second) {
			req.CalibrationScreened = true
			return cp, measuredStrategy, measuredArgs, nil
		}
		if mode == calibrateAuto {
			// The extra candidates are admission fallbacks, not permission to make
			// ordinary launch reload a large model for a broad successful sweep.
			break
		}
	}

	// A failed finalist still leaves a complete baseline workload measurement.
	// Bring the known-good default back and return a default-won decision so the
	// exact scope does not repeat the same failed experiment next launch. The
	// caller still withholds persistence until the restored baseline passes the
	// ordinary functional/cache/lifecycle gates.
	if len(measurements) < 2 {
		if curP == nil {
			var restoredStrategy *placement.Strategy
			var restoredArgs []string
			curP, restoredStrategy, restoredArgs = restartPlacement(
				req, cfg, model, strategy, be, caps, serverArgs, timeout, memoryRecovery,
			)
			if curP != nil {
				// Negative finalist evidence belongs to the exact measured baseline.
				// If restoring it needed recovery, serve that safe result but leave the
				// experiment retryable under its new scope.
				if !exactCalibrationCandidate(serverArgs, restoredArgs) {
					return curP, restoredStrategy, restoredArgs, nil
				}
				strategy, serverArgs = restoredStrategy, restoredArgs
			}
		}
		if curP == nil {
			return nil, strategy, serverArgs, nil
		}
		// Only exact admission failures are stable negative evidence. A candidate
		// that started but whose benchmark timed out or returned incomplete data
		// must remain retryable on the next launch.
		if !stableAdmissionFailed || admissionInconclusive || (exactCandidateStarted && !stablePostStartFailure) {
			return curP, strategy, serverArgs, nil
		}
		pending := newCalibrationDecision(scopeKey, model, defaultResult, measurements[0])
		annotateOptimizationDecision(pending, candidates, measurements)
		pending.FinalistFailureClass = stableFailureClass
		pending.FinalistFailureReason = stableFailureReason
		if mode == calibrateAuto {
			fmt.Printf("[optimize] calculated finalist was unavailable; restored measured baseline and will record that bounded result after launch validation\n")
		} else {
			fmt.Printf("[calibrate] no challenger completed; restored measured baseline pending launch validation\n")
		}
		return curP, strategy, serverArgs, pending
	}

	best := measurements[0]
	for _, measured := range measurements[1:] {
		if calibrationCandidateBetter(measured, best) {
			best = measured
		}
	}
	selected, optimized := "", false
	if mode == calibrateOn && defaultResult.Parallel <= 1 {
		var optimizerErr error
		selected, optimized, optimizerErr = maybeOptimizeCalibration(req, cfg, model, be, caps, scopeKey, measurements)
		if optimizerErr != nil {
			fmt.Fprintf(os.Stderr, "[calibrate] refusing main-model restart: %v\n", optimizerErr)
			return nil, strategy, serverArgs, nil
		}
	}
	if optimized {
		for _, measured := range measurements {
			if measured.Name != selected {
				continue
			}
			floor := best.Score * (1 - calibrationAdvisorTiePct/100)
			if measured.Score >= floor {
				fmt.Printf("[calibrate] support optimizer selected measured near-tie %s (score %.3f)\n", measured.Name, measured.Score)
				best = measured
			} else {
				fmt.Fprintf(os.Stderr, "[calibrate] rejected advisor selection %s: score %.3f is below deterministic floor %.3f\n", measured.Name, measured.Score, floor)
			}
			break
		}
	}

	// The support helper (when enabled) has stopped and verified resource release.
	// Only now may the winning main-model process be started.
	servingStrategy, servingArgs := best.Strategy, best.Args
	curP, servingStrategy, servingArgs = restartPlacement(
		req, cfg, model, best.Strategy, be, caps, best.Args, timeout, memoryRecovery,
	)
	cleanRestartFailureClass, cleanRestartFailureReason := "", ""
	if curP != nil && !exactCalibrationCandidate(best.Args, servingArgs) {
		if classifyCalibrationRelaunch(best, measurements[0], servingArgs) == calibrationKeepRecoveredBaseline {
			fmt.Fprintln(os.Stderr, "[calibrate] restored baseline required recovery; keeping the healthy server without promoting a performance decision")
			return curP, servingStrategy, servingArgs, nil
		}
		// A winner is not reusable unless the clean relaunch reproduced its exact
		// argv. Automatic hot-cache activation may safely strip only the cache and
		// return the already-measured baseline; any other recovery is stopped and
		// followed by an explicit baseline restore.
		if classifyCalibrationRelaunch(best, measurements[0], servingArgs) == calibrationUseMeasuredBaseline {
			if best.Strategy != nil && best.Strategy.HotExpertCacheSlots > 0 &&
				servingStrategy != nil && servingStrategy.HotExpertCacheSlots == 0 {
				cleanRestartFailureClass = "hot-expert-clean-relaunch"
				cleanRestartFailureReason = "the measured hot-expert candidate did not reproduce its cache-on activation during clean relaunch"
			}
			fmt.Fprintf(os.Stderr, "[calibrate] winner did not reproduce its exact argv; serving the measured default\n")
			best = measurements[0]
		} else {
			if !stopCalibrationProcessAndWait(curP, "recovered winner before exact baseline restore", resourceBaseline, 30*time.Second) {
				req.CalibrationScreened = true
				return curP, servingStrategy, servingArgs, nil
			}
			curP = nil
		}
	}
	if curP == nil && best.Name != "default" {
		fmt.Fprintln(os.Stderr, "[calibrate] winner restart failed; restoring measured default")
		best = measurements[0]
		curP, servingStrategy, servingArgs = restartPlacement(
			req, cfg, model, best.Strategy, be, caps, best.Args, timeout, memoryRecovery,
		)
	}
	if curP == nil {
		return nil, servingStrategy, servingArgs, nil
	}
	if !exactCalibrationCandidate(best.Args, servingArgs) {
		// The safe recovery may continue serving, but it is a different placement
		// from every measured result and cannot produce a calibration decision.
		return curP, servingStrategy, servingArgs, nil
	}

	pending := newCalibrationDecision(scopeKey, model, defaultResult, best)
	annotateOptimizationDecision(pending, candidates, measurements)
	if pending != nil && cleanRestartFailureClass != "" {
		pending.FinalistOutcome = "unavailable"
		pending.FinalistFailureClass = cleanRestartFailureClass
		pending.FinalistFailureReason = cleanRestartFailureReason
	}
	if best.Name != "default" && mode == calibrateOn {
		req.CalibrationScreened = true
	}
	if mode == calibrateOn {
		fmt.Printf("[calibrate] screened winner %s (turn %.2fs, relative %.3f); explicit run only, awaiting lifecycle canaries and not eligible for automatic default\n",
			best.Name, calibrationTurnTime(best.Result), best.Score)
	} else {
		fmt.Printf("[optimize] candidate winner %s (turn %.2fs, relative %.3f); awaiting launch/cache/workload validation\n",
			best.Name, calibrationTurnTime(best.Result), best.Score)
	}
	return curP, servingStrategy, servingArgs, pending
}

func newCalibrationDecision(scopeKey string, model *placement.ModelProfile, defaultResult *benchmark.Result, best calibrationMeasurement) *placement.CalibrationDecision {
	if defaultResult == nil || best.Result == nil {
		return nil
	}
	modelBasename := ""
	if model != nil {
		modelBasename = placement.CalibrationModelBasename(model.Path)
	}
	return &placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: best.Name,
		ModelBasename:          modelBasename,
		ValidationLevel:        placement.CalibrationValidationScreened,
		DefaultTPS:             defaultResult.GenTPS,
		DefaultPromptTPS:       defaultResult.PromptTPS,
		DefaultMixedTPS:        defaultResult.MixedGenTPS,
		DefaultTurnTimeS:       calibrationTurnTime(defaultResult),
		DefaultTurnMaxS:        calibrationWorstWorkloadTime(defaultResult),
		DefaultAgentSamples:    defaultResult.AgentSamples,
		DefaultWorkloadLanes:   defaultResult.AgentWorkloadLanes,
		DefaultCachedTokens:    defaultResult.AgentCachedTokens,
		DefaultNewPromptTokens: defaultResult.AgentNewPromptTokens,
		AgentPromptBytes:       defaultResult.AgentPromptBytes,
		DefaultScore:           1,
		WinnerTPS:              best.Result.GenTPS,
		WinnerPromptTPS:        best.Result.PromptTPS,
		WinnerMixedTPS:         best.Result.MixedGenTPS,
		WinnerTurnTimeS:        calibrationTurnTime(best.Result),
		WinnerTurnMaxS:         calibrationWorstWorkloadTime(best.Result),
		WinnerAgentSamples:     best.Result.AgentSamples,
		WinnerWorkloadLanes:    best.Result.AgentWorkloadLanes,
		WinnerCachedTokens:     best.Result.AgentCachedTokens,
		WinnerNewPromptTokens:  best.Result.AgentNewPromptTokens,
		WinnerScore:            best.Score,
		Improvement:            (best.Score - 1) * 100,
	}
}

func annotateOptimizationDecision(decision *placement.CalibrationDecision, candidates []placement.CalibrationCandidate, measurements []calibrationMeasurement) {
	if decision == nil || len(candidates) < 2 || candidates[0].Strategy == nil {
		return
	}
	baseline := candidates[0].Strategy
	decision.BaselineResidency = baseline.Residency
	decision.BaselineBottleneck = baseline.OptimizationBottleneck
	decision.ExploredBoundary = baseline.OptimizationBoundary

	// If the primary estimate failed admission but a fallback completed, record
	// the candidate that was actually compared. With no measured challenger the
	// primary remains visible as unavailable and cannot pass the automatic
	// evidence gate.
	finalist := candidates[1]
	for _, measured := range measurements[1:] {
		for _, candidate := range candidates[1:] {
			if candidate.Name == measured.Name {
				finalist = candidate
				break
			}
		}
		break
	}
	decision.Finalist = finalist.Name
	decision.FinalistEstimatedCost = finalist.Estimate.AgentCost
	decision.FinalistConfidence = finalist.Estimate.Confidence
	decision.FinalistOutcome = "unavailable"
	for _, measured := range measurements {
		if measured.Name != finalist.Name {
			continue
		}
		if measured.Strategy != nil {
			decision.FinalistHotExpertSteps = measured.Strategy.HotExpertCacheSteps
			decision.FinalistHotExpertHits = measured.Strategy.HotExpertCacheHits
			decision.FinalistHotExpertMisses = measured.Strategy.HotExpertCacheMisses
			decision.FinalistHotExpertHitRate = measured.Strategy.HotExpertCacheHitRate
		}
		if decision.Winner == finalist.Name {
			decision.FinalistOutcome = "promoted"
		} else {
			decision.FinalistOutcome = "baseline-won"
		}
		return
	}
}

func prioritizeCalibrationFinalist(candidates []placement.CalibrationCandidate, finalist placement.CalibrationCandidate) []placement.CalibrationCandidate {
	if len(candidates) < 2 || finalist.Name == "" {
		return candidates
	}
	out := make([]placement.CalibrationCandidate, 0, len(candidates))
	out = append(out, candidates[0], finalist)
	for _, candidate := range candidates[1:] {
		if candidate.Name == finalist.Name {
			continue
		}
		out = append(out, candidate)
	}
	return out
}

func stopCalibrationProcess(process *server.Process, label string) bool {
	if process == nil {
		return true
	}
	err := process.Stop()
	if process.IsRunning() {
		fmt.Fprintf(os.Stderr, "[calibrate] %s did not stop; refusing overlapping candidate/helper start", label)
		if err != nil {
			fmt.Fprintf(os.Stderr, ": %v", err)
		}
		fmt.Fprintln(os.Stderr)
		return false
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "[calibrate] %s exited with stop warning: %v\n", label, err)
	}
	return true
}

func stopCalibrationProcessAndWait(process *server.Process, label string, baseline *detect.Capabilities, timeout time.Duration) bool {
	if !stopCalibrationProcess(process, label) {
		return false
	}
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	deadline := time.Now().Add(timeout)
	for {
		if launchResourcesAtBaseline(baseline) {
			return true
		}
		if time.Now().After(deadline) {
			fmt.Fprintf(os.Stderr, "[calibrate] %s released its process but RAM/VRAM did not return to the pre-main baseline within %s; refusing an overlapping start\n", label, timeout)
			return false
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func sampleLaunchGPUUtilization(caps *detect.Capabilities) []benchmark.GPUUtilization {
	if caps == nil {
		return nil
	}
	mapped := detect.MapUtilizationToIndexes(caps.GPUs, detect.SampleGPUUtilization())
	if len(mapped) == 0 {
		return nil
	}
	out := make([]benchmark.GPUUtilization, 0, len(mapped))
	for _, sample := range mapped {
		out = append(out, benchmark.GPUUtilization{
			GPU: sample.Index, SMPercent: sample.SMPercent, MemPercent: sample.MemPercent,
			PCIeRXMBps: sample.PCIeRXMBps, PCIeTXMBps: sample.PCIeTXMBps,
		})
	}
	return out
}

func calibrationProcessResourceSampler(p *server.Process) func() benchmark.ResourceSnapshot {
	if p == nil || p.Cmd == nil || p.Cmd.Process == nil || p.Cmd.Process.Pid <= 0 {
		return nil
	}
	return benchmark.NewProcessTreeResourceSampler(p.Cmd.Process.Pid)
}

func recordAgentBottleneckDiagnosis(label string, caps *detect.Capabilities, model *placement.ModelProfile, strategy *placement.Strategy, result *benchmark.Result) placement.WorkloadBottleneck {
	diagnosis := placement.DiagnoseAgentBottleneck(caps, model, strategy, result)
	if diagnosis.Primary == placement.BottleneckUnknown {
		fmt.Printf("[optimize] %s bottleneck: unknown (%s)\n", label, diagnosis.Summary)
		return diagnosis
	}
	fmt.Printf("[optimize] %s bottleneck [%s/%s]: %s\n",
		label, diagnosis.Primary, diagnosis.Confidence, diagnosis.Summary)
	for _, phase := range diagnosis.Phases {
		if phase.Kind == placement.BottleneckUnknown {
			continue
		}
		fmt.Printf("[optimize] %s phase %s [%s/%s]: %s\n",
			label, phase.Phase, phase.Kind, phase.Confidence, phase.Summary)
	}
	if len(diagnosis.Levers) > 0 {
		fmt.Printf("[optimize] %s safe levers: %s\n", label, strings.Join(diagnosis.Levers, "; "))
	}
	if strategy != nil {
		strategy.OptimizationBottleneck = diagnosis.Summary
	}
	return diagnosis
}

func deviceBalanceBottleneckFromResult(strategy *placement.Strategy, result *benchmark.Result) string {
	return placement.DeviceBalanceBottleneck(strategy, gpuUtilSamplesFromResult(result))
}

func deviceBalanceSignalFromResult(strategy *placement.Strategy, result *benchmark.Result) placement.DeviceBalanceSignal {
	return placement.AnalyzeDeviceBalance(strategy, gpuUtilSamplesFromResult(result))
}

func gpuUtilSamplesFromResult(result *benchmark.Result) []placement.GPUUtilSample {
	if result == nil || len(result.GPUUtilization) == 0 {
		return nil
	}
	samples := make([]placement.GPUUtilSample, 0, len(result.GPUUtilization))
	for _, sample := range result.GPUUtilization {
		samples = append(samples, placement.GPUUtilSample{
			GPU: sample.GPU, SMPercent: sample.SMPercent, MemPercent: sample.MemPercent,
		})
	}
	return samples
}

func telemetryDirectedCalibrationFinalist(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy, signal placement.DeviceBalanceSignal, memoryRecovery *launchMemoryRecovery) (placement.CalibrationCandidate, bool) {
	// Utilization can prove a performance defect but cannot prove memory
	// headroom. Tight launches may challenge only a same-workload topology that
	// does not increase CPU expert offload; SelectDeviceBalanceFinalist enforces
	// that rule. Exact baseline accounting plus contained candidate admission is
	// the safety authority, rather than the coarse roomy/tight label.
	if req == nil || cfg == nil || model == nil || be == nil || caps == nil || strategy == nil ||
		strategy.Residency == placement.ResidencyNonResident ||
		strategy.ResourceLedger == nil || !strategy.ResourceLedger.Exact || !strategy.ResourceLedger.Fits {
		return placement.CalibrationCandidate{}, false
	}
	frontier := analyzedMeasuredCalibrationFrontier(req, cfg, model, be, caps, strategy, memoryRecovery)
	return placement.SelectDeviceBalanceFinalist(frontier, signal)
}

func bottleneckDirectedCalibrationFinalist(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy, diagnosis placement.WorkloadBottleneck, memoryRecovery *launchMemoryRecovery) (placement.CalibrationCandidate, bool) {
	if strategy == nil || strategy.Residency != placement.ResidencyRoomy ||
		diagnosis.Primary == placement.BottleneckUnknown ||
		diagnosis.Primary == placement.BottleneckCapacity {
		return placement.CalibrationCandidate{}, false
	}
	frontier := analyzedMeasuredCalibrationFrontier(req, cfg, model, be, caps, strategy, memoryRecovery)
	return placement.SelectBottleneckFinalist(frontier, diagnosis)
}

func analyzedMeasuredCalibrationFrontier(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy, memoryRecovery *launchMemoryRecovery) []placement.CalibrationCandidate {
	frontier := calibrationCandidates(req, cfg, model, be, caps, strategy)
	if memoryRecovery != nil && memoryRecovery.hasRejections() {
		frontier = filterCalibrationCandidates(frontier, func(candidate *placement.Strategy) bool {
			args := buildLaunchServerArgs(req, cfg, be, caps, model, candidate)
			return memoryRecovery.isRejected(args)
		})
	}
	cacheDir := ""
	if cfg != nil {
		cacheDir = cfg.CacheDir
	}
	opts := placementOptionsFromRequest(req, model, be, cacheDir)
	return placement.AnalyzeCandidateFrontier(caps, model, opts, frontier)
}

func cmdStatus(args []string) {
	cfg := loadConfigOrExit()
	jsonOut := hasArg(args, "--json")
	if hasArg(args, "--capabilities") {
		printCapabilities(os.Stdout, collectCapabilities(cfg))
		return
	}
	modelArg := ""
	for _, arg := range args {
		if arg == "--json" {
			continue
		}
		if strings.HasPrefix(arg, "-") {
			fmt.Fprintf(os.Stderr, "unknown status option %q\n", arg)
			os.Exit(2)
		}
		if modelArg == "" {
			modelArg = arg
		}
	}

	type statusRecord struct {
		Model     string                          `json:"model,omitempty"`
		Status    string                          `json:"status"`
		Decision  *placement.CalibrationDecision  `json:"decision,omitempty"`
		Decisions []placement.CalibrationDecision `json:"decisions,omitempty"`
	}

	if modelArg != "" {
		decision, err := placement.LatestCalibrationDecisionForModel(cfg.CacheDir, filepath.Base(modelArg))
		if err != nil {
			if jsonOut {
				_ = json.NewEncoder(os.Stdout).Encode(statusRecord{
					Model: filepath.Base(modelArg), Status: placement.FormatCalibrationStatus(nil),
				})
				return
			}
			fmt.Printf("ggrun launch optimizer: %s\n", placement.FormatCalibrationStatus(nil))
			fmt.Printf("  model: %s\n", filepath.Base(modelArg))
			fmt.Printf("  detail: %v\n", err)
			return
		}
		if jsonOut {
			_ = json.NewEncoder(os.Stdout).Encode(statusRecord{
				Model: decision.ModelBasename, Status: placement.FormatCalibrationStatus(decision), Decision: decision,
			})
			return
		}
		printLaunchOptimizerStatus(decision)
		return
	}

	decisions := placement.ListCalibrationDecisions(cfg.CacheDir, 8)
	if jsonOut {
		status := "no measured winner yet (cold estimate on launch)"
		if len(decisions) > 0 {
			status = placement.FormatCalibrationStatus(&decisions[0])
		}
		_ = json.NewEncoder(os.Stdout).Encode(statusRecord{Status: status, Decisions: decisions})
		return
	}
	if len(decisions) == 0 {
		fmt.Println("ggrun launch optimizer: no measured winner yet (cold estimate on launch)")
		fmt.Println("  inspect a model with: ggrun status <model.gguf>")
		fmt.Println("  dry-run prints the same residency/boundary payload")
		return
	}
	fmt.Println("ggrun launch optimizer")
	for i, decision := range decisions {
		if i == 0 {
			printLaunchOptimizerStatus(&decision)
			continue
		}
		fmt.Printf("  also %s: %s\n", decision.ModelBasename, placement.FormatCalibrationStatus(&decision))
	}
}

func printLaunchOptimizerStatus(decision *placement.CalibrationDecision) {
	if decision == nil {
		fmt.Println("ggrun launch optimizer: no measured winner yet (cold estimate on launch)")
		return
	}
	fmt.Printf("ggrun launch optimizer: %s\n", placement.FormatCalibrationStatus(decision))
	if decision.ModelBasename != "" {
		fmt.Printf("  model:     %s\n", decision.ModelBasename)
	}
	fmt.Printf("  winner:    %s (%s)\n", decision.Winner, decision.ValidationLevel)
	if decision.Finalist != "" {
		fmt.Printf("  finalist:  %s (%s)\n", decision.Finalist, decision.FinalistOutcome)
	}
	if decision.ExploredBoundary != nil {
		b := decision.ExploredBoundary
		fmt.Printf("  explored:  %d candidates (%d feasible), batch %d..%d, ubatch %d..%d, parallel %d..%d\n",
			b.CandidateCount, b.FeasibleCount, b.MinBatch, b.MaxBatch, b.MinUBatch, b.MaxUBatch, b.MinParallel, b.MaxParallel)
	}
}

// maybeOptimizeCalibration invokes NanoBeige only when it is explicitly on or
// fully installed in auto mode. Its candidate IDs and metrics come exclusively
// from successful ggrun benchmark runs, and the caller retains a deterministic
// near-tie guard even after schema validation.
func maybeOptimizeCalibration(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, scopeKey string, measurements []calibrationMeasurement) (string, bool, error) {
	if cfg == nil || len(measurements) < 2 {
		return "", false, nil
	}
	mode, online := cfg.SupportExpert, cfg.SupportOnline
	if req != nil {
		if req.SupportExpert != "" {
			mode = req.SupportExpert
		}
		online = req.SupportOnline
	}
	if mode == "off" {
		return "", false, nil
	}
	if mode == "auto" && !currentSupportStatus(cfg, caps).Ready {
		return "", false, nil
	}
	incident := calibrationAdvisorIncident(req, model, be, caps, scopeKey, measurements)
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
		fmt.Fprintf(os.Stderr, "[support] could not persist optimizer analysis: %v\n", historyErr)
	}
	if runErr != nil {
		fmt.Fprintf(os.Stderr, "[support] optimizer unavailable; deterministic measured score remains authoritative: %v\n", runErr)
		if errors.Is(runErr, advisor.ErrResourceReleaseUnverified) {
			return "", false, runErr
		}
		return "", false, nil
	}
	fmt.Printf("[support] optimizer decision=%s confidence=%.2f; helper release verified; record=%s\n",
		decision.Action, decision.Confidence, history)
	if decision.Action != advisor.ActionSelectCandidate {
		return "", false, nil
	}
	return decision.CandidateID, true, nil
}

func calibrationAdvisorIncident(req *launchRequest, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, scopeKey string, measurements []calibrationMeasurement) advisor.Incident {
	arch, family, identity := "", "", ""
	if model != nil {
		arch = model.ModelArch
	}
	if be != nil {
		family, identity = backendDialect(be), evidenceBackendCacheTag(be)
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
	settings := map[string]string{}
	if req != nil {
		settings["context"] = req.CtxFlag
		settings["kv_quality"] = req.KVQuality
		settings["parallel"] = strconv.Itoa(req.Parallel)
	}
	candidates := make([]advisor.Candidate, 0, len(measurements))
	observations := make([]advisor.Observation, 0, len(measurements))
	for index, measured := range measurements {
		properties := map[string]string{}
		if measured.Strategy != nil {
			properties["placement_type"] = string(measured.Strategy.Type)
			properties["kv_placement"] = measured.Strategy.KVPlacement
			properties["cpu_moe_layers"] = strconv.Itoa(measured.Strategy.NCPUMoE)
			properties["main_gpu"] = strconv.Itoa(measured.Strategy.MainGPU)
			properties["mmap"] = strconv.FormatBool(measured.Strategy.MMap)
		}
		properties["agent_prompt_bytes"] = strconv.Itoa(measured.Result.AgentPromptBytes)
		candidates = append(candidates, advisor.Candidate{
			ID: measured.Name, Verified: true, Properties: properties,
			Metrics: map[string]float64{
				"decode_tps": measured.Result.GenTPS, "prefill_tps": measured.Result.PromptTPS,
				"decode_seconds": measured.Result.GenTimeS, "prefill_seconds": measured.Result.PromptTimeS,
				"agent_workflow_seconds": calibrationTurnTime(measured.Result),
				"agent_append_seconds":   measured.Result.AgentTurnTimeS,
				"cache_reused_tokens":    float64(measured.Result.AgentCachedTokens),
				"relative_turn_score":    measured.Score,
			},
		})
		observations = append(observations, advisor.Observation{
			Code: fmt.Sprintf("candidate_%d_measured", index), Component: "calibration",
			Value: calibrationTurnTime(measured.Result), Unit: "seconds", Source: "ggrun_agent_turn_screen", Confidence: "measured",
			Attributes: map[string]string{"candidate_id": measured.Name},
		})
	}
	sum := sha256.Sum256([]byte(scopeKey + "|optimizer"))
	incident := advisor.Incident{
		ID: "opt-" + hex.EncodeToString(sum[:8]), Mode: advisor.ModeOptimizer,
		Architecture: arch, BackendFamily: family, BackendIdentity: identity,
		Workload: requestWorkloadProfile(req, model), ProfileState: "performance_measured",
		Hardware: hardware, Settings: settings, Observations: observations,
		AllowedActions: []advisor.ActionID{advisor.ActionNoAction, advisor.ActionSelectCandidate},
		Candidates:     candidates,
	}
	_ = incident.Normalize()
	return incident
}

// restartPlacement brings the selected measured strategy back up after the
// losing candidate stopped it. It returns the actual strategy/argv because
// recovery may legally restore a cache-free baseline; callers must never label
// that process as the requested cache-on winner. A nil process means restart
// failed.
func restartPlacement(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, serverArgs []string, timeout time.Duration, memoryRecovery *launchMemoryRecovery) (*server.Process, *placement.Strategy, []string) {
	p, restoredStrategy, restoredArgs, err := restoreLaunchWithCUDAOOMRecoveryState(req, cfg, model, strategy, be, caps, serverArgs, timeout, memoryRecovery)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[calibrate] restart of best placement failed: %v\n", err)
		return nil, restoredStrategy, restoredArgs
	}
	return p, restoredStrategy, restoredArgs
}

type calibrationRelaunchDisposition uint8

const (
	calibrationExactRelaunch calibrationRelaunchDisposition = iota
	calibrationKeepRecoveredBaseline
	calibrationUseMeasuredBaseline
	calibrationRestoreMeasuredBaseline
)

// A healthy recovered baseline is already the safe serving fallback. Stopping
// it to restore the same baseline again can strand an otherwise successful
// launch. Its changed argv cannot carry the earlier performance decision.
func classifyCalibrationRelaunch(winner, baseline calibrationMeasurement, actualArgs []string) calibrationRelaunchDisposition {
	if exactCalibrationCandidate(winner.Args, actualArgs) {
		return calibrationExactRelaunch
	}
	if winner.Name == "default" {
		return calibrationKeepRecoveredBaseline
	}
	if exactCalibrationCandidate(baseline.Args, actualArgs) {
		return calibrationUseMeasuredBaseline
	}
	return calibrationRestoreMeasuredBaseline
}
