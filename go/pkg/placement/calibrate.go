package placement

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// CalibrationSchemaVersion bumps whenever the candidate set or scoring changes,
// so a decision measured under older semantics is never applied after an
// upgrade changes what "fastest" means.
//
// 24: `auto` hot experts no longer serve the cache-on placement as the first
// serve. Candidate 0 is the packed cache-free layout and the cache-on placement
// is the measured challenger, so a decision recorded under the pre-24 semantics
// (where the cache-on layout was the default) must not be replayed.
// 25: related-probe prediction inputs require exact artifact/backend/slot scope.
// Keep existing fit observations, but re-evaluate prior performance decisions.
// 26: retained observations keep their original source and GPU role.
// 27: cache-free allocation no longer proves a changed cache-on graph.
// 28: post-start candidate CUDA OOM is stable rejection; GLM recurrent cache policy.
// 29: allocation identity follows emitted float32 ratios with layer-mode guards.
// 30: oracle cache charges and router ownership corrected; cache/graph attribution separated.
// 31: explicit cache-slot demotion credits freed device bytes and charges host.
const CalibrationSchemaVersion = 35

var calibrationShardBasename = regexp.MustCompile(`(?i)^(.*)-00001-of-[0-9]{5}\.gguf$`)

// CalibrationModelBasename returns the one-model identity shown by the TUI.
// Split GGUF entrypoints are stored and launched through shard 00001, while the
// model picker deliberately collapses all shards to <name>.gguf. Persisting the
// same canonical name keeps status and cache cleanup truthful for both forms.
func CalibrationModelBasename(path string) string {
	base := filepath.Base(strings.TrimSpace(path))
	if match := calibrationShardBasename.FindStringSubmatch(base); len(match) == 2 {
		return match[1] + ".gguf"
	}
	return base
}

const (
	// CalibrationValidationScreened means the candidate passed the bounded live
	// comparison and ordinary launch canaries. It is useful evidence for an
	// explicit experiment, but it is not sufficient for an automatic default.
	CalibrationValidationScreened = "screened-v1"
	// CalibrationValidationWorkflow is reserved for the versioned agent screen
	// plus append/cache, contention, branch/replay, functional, and clean-relaunch
	// gates. Longer hardware acceptance belongs to the release matrix and does
	// not make one cold standard launch unbounded.
	CalibrationValidationWorkflow = "agent-workflow-v1"
	// CalibrationValidationAdmission means the baseline passed its full launch
	// lifecycle, but the bounded finalist was deterministically unavailable at
	// exact admission or a feature-specific post-start/lifecycle gate. It is
	// negative feasibility/compatibility evidence, not performance evidence:
	// automatic launch may avoid repeating the same destructive stop/reload loop,
	// but it must never advertise the default as a measured fastest configuration.
	CalibrationValidationAdmission = "admission-only-v1"
)

// CalibrationDecision records which candidate won a measured placement screen
// for one scope, with the numbers that decided it and the validation level that
// controls whether it may affect automatic serving. The winner is
// stored by name and re-derived from the deterministic candidate generator on
// later launches, so the full placement is reproduced exactly rather than
// partially deserialized.
type CalibrationDecision struct {
	SchemaVersion int    `json:"schema_version"`
	ScopeKey      string `json:"scope_key"`
	// ModelBasename is the basename of the model this decision measured, written
	// beside the scope hash so a "clear caches" action can find every decision a
	// model created. The scope key is an opaque hash of model+backend+hardware
	// that cannot be reversed into a model identity, so clearing by model needs
	// this explicit marker. Old decisions without the field stay valid for
	// loading (scope-key validation is unchanged); they just cannot be matched
	// for clearing.
	ModelBasename            string                 `json:"model_basename,omitempty"`
	Winner                   string                 `json:"winner"` // candidate Name, e.g. "default" or "kv-alternate"
	ValidationLevel          string                 `json:"validation_level"`
	DefaultTPS               float64                `json:"default_tps"`
	DefaultPromptTPS         float64                `json:"default_prompt_tps"`
	DefaultMixedTPS          float64                `json:"default_mixed_tps,omitempty"`
	DefaultTurnTimeS         float64                `json:"default_turn_time_s,omitempty"`
	DefaultTurnMaxS          float64                `json:"default_turn_max_s,omitempty"`
	DefaultAgentSamples      int                    `json:"default_agent_samples,omitempty"`
	DefaultWorkloadLanes     int                    `json:"default_workload_lanes,omitempty"`
	DefaultCachedTokens      int                    `json:"default_cached_tokens,omitempty"`
	DefaultNewPromptTokens   int                    `json:"default_new_prompt_tokens,omitempty"`
	AgentPromptBytes         int                    `json:"agent_prompt_bytes,omitempty"`
	DefaultScore             float64                `json:"default_score"`
	WinnerTPS                float64                `json:"winner_tps"`
	WinnerPromptTPS          float64                `json:"winner_prompt_tps"`
	WinnerMixedTPS           float64                `json:"winner_mixed_tps,omitempty"`
	WinnerTurnTimeS          float64                `json:"winner_turn_time_s,omitempty"`
	WinnerTurnMaxS           float64                `json:"winner_turn_max_s,omitempty"`
	WinnerAgentSamples       int                    `json:"winner_agent_samples,omitempty"`
	WinnerWorkloadLanes      int                    `json:"winner_workload_lanes,omitempty"`
	WinnerCachedTokens       int                    `json:"winner_cached_tokens,omitempty"`
	WinnerNewPromptTokens    int                    `json:"winner_new_prompt_tokens,omitempty"`
	WinnerScore              float64                `json:"winner_score"`
	Improvement              float64                `json:"improvement_pct"`
	BaselineResidency        ResidencyClass         `json:"baseline_residency,omitempty"`
	BaselineBottleneck       string                 `json:"baseline_bottleneck,omitempty"`
	Finalist                 string                 `json:"finalist,omitempty"`
	FinalistOutcome          string                 `json:"finalist_outcome,omitempty"`
	FinalistFailureClass     string                 `json:"finalist_failure_class,omitempty"`
	FinalistFailureReason    string                 `json:"finalist_failure_reason,omitempty"`
	FinalistEstimatedCost    float64                `json:"finalist_estimated_agent_cost,omitempty"`
	FinalistConfidence       string                 `json:"finalist_estimate_confidence,omitempty"`
	FinalistHotExpertSteps   uint64                 `json:"finalist_hot_expert_steps,omitempty"`
	FinalistHotExpertHits    uint64                 `json:"finalist_hot_expert_hits,omitempty"`
	FinalistHotExpertMisses  uint64                 `json:"finalist_hot_expert_misses,omitempty"`
	FinalistHotExpertHitRate float64                `json:"finalist_hot_expert_hit_rate,omitempty"`
	RejectedFinalists        []CalibrationRejection `json:"rejected_finalists,omitempty"`
	ExploredBoundary         *OptimizationBoundary  `json:"explored_boundary,omitempty"`
	MeasuredAt               string                 `json:"measured_at"`
}

// CalibrationRejection is durable, coordinate-local negative evidence. Keeping
// the bounded set beside the scope decision lets automatic optimization advance
// through the frontier without forgetting an earlier exact admission failure.
type CalibrationRejection struct {
	Finalist      string `json:"finalist"`
	FailureClass  string `json:"failure_class"`
	FailureReason string `json:"failure_reason"`
}

const maxCalibrationRejections = 32

// RecordAdmissionRejection merges one stable failure by candidate name. The
// newest explanation replaces an older one and the bounded tail prevents a
// malformed/foreign cache from growing without limit.
func (d *CalibrationDecision) RecordAdmissionRejection(finalist, failureClass, failureReason string) {
	if d == nil || finalist == "" || failureClass == "" || failureReason == "" {
		return
	}
	for i := range d.RejectedFinalists {
		if d.RejectedFinalists[i].Finalist == finalist {
			d.RejectedFinalists[i].FailureClass = failureClass
			d.RejectedFinalists[i].FailureReason = failureReason
			return
		}
	}
	d.RejectedFinalists = append(d.RejectedFinalists, CalibrationRejection{
		Finalist: finalist, FailureClass: failureClass, FailureReason: failureReason,
	})
	if len(d.RejectedFinalists) > maxCalibrationRejections {
		d.RejectedFinalists = append([]CalibrationRejection(nil), d.RejectedFinalists[len(d.RejectedFinalists)-maxCalibrationRejections:]...)
	}
}

// MergeAdmissionRejections preserves earlier scoped negatives when a later
// coordinate also fails. The current decision fields remain the latest event.
func (d *CalibrationDecision) MergeAdmissionRejections(previous *CalibrationDecision) {
	if d == nil || previous == nil {
		return
	}
	if previous.ValidationLevel == CalibrationValidationAdmission {
		d.RecordAdmissionRejection(previous.Finalist, previous.FinalistFailureClass, previous.FinalistFailureReason)
	}
	for _, rejection := range previous.RejectedFinalists {
		d.RecordAdmissionRejection(rejection.Finalist, rejection.FailureClass, rejection.FailureReason)
	}
}

// CalibrationPath returns the cache file for one calibration scope.
func CalibrationPath(cacheDir, scopeKey string) string {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	return filepath.Join(cacheDir, "calibration", "cal-"+scopeKey+".json")
}

// SaveCalibrationDecision persists a calibration result atomically.
func SaveCalibrationDecision(cacheDir string, d CalibrationDecision) (string, error) {
	if d.SchemaVersion == 0 {
		d.SchemaVersion = CalibrationSchemaVersion
	}
	if d.MeasuredAt == "" {
		d.MeasuredAt = time.Now().UTC().Format(time.RFC3339)
	}
	if d.ValidationLevel == "" {
		d.ValidationLevel = CalibrationValidationScreened
	}
	path := CalibrationPath(cacheDir, d.ScopeKey)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return "", err
	}
	data, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return "", err
	}
	release, err := acquirePlacementLock(path+".lock", 5*time.Second)
	if err != nil {
		return "", err
	}
	defer release()
	if err := atomicWriteFile(path, append(data, '\n'), 0o600); err != nil {
		return "", err
	}
	return path, nil
}

// AutomaticEligible reports whether this exact decision has enough evidence
// to affect the no-flag path. Screening-only explicit decisions intentionally
// return false; only the combined agent and lifecycle validation contract may
// opt in.
func (d *CalibrationDecision) AutomaticEligible() bool {
	return d != nil && d.ValidationLevel == CalibrationValidationWorkflow
}

// SuppressesAutomaticAdmissionRetry reports whether this decision proves that
// the exact deterministic finalist was unavailable. It deliberately does not
// make the decision AutomaticEligible: negative feasibility or compatibility
// evidence is not a performance winner. The calibration scope and schema bind
// the result to the same model/backend/hardware/workload policy; either changing
// retires it automatically.
func (d *CalibrationDecision) SuppressesAutomaticAdmissionRetry(finalist string) bool {
	if d == nil || d.ValidationLevel != CalibrationValidationAdmission || finalist == "" {
		return false
	}
	if d.Winner == "default" &&
		d.FinalistOutcome == "unavailable" &&
		d.FinalistFailureClass != "" &&
		d.FinalistFailureReason != "" &&
		d.Finalist != "" &&
		d.Finalist == finalist {
		return true
	}
	for _, rejection := range d.RejectedFinalists {
		if rejection.Finalist == finalist && rejection.FailureClass != "" && rejection.FailureReason != "" {
			return true
		}
	}
	return false
}

// LoadCalibrationDecision reads a prior calibration for the scope, rejecting
// stale-schema or mismatched keys so an old decision is never silently applied.
func LoadCalibrationDecision(cacheDir, scopeKey string) (*CalibrationDecision, error) {
	path := CalibrationPath(cacheDir, scopeKey)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var d CalibrationDecision
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, err
	}
	if d.SchemaVersion != CalibrationSchemaVersion || d.ScopeKey != scopeKey {
		return nil, fmt.Errorf("calibration decision scope mismatch")
	}
	return &d, nil
}

// DeleteCalibrationDecision removes a cached calibration result for one scope.
// A runtime OOM is evidence the declared winner is unsafe at runtime, so the
// decision must not re-declare it the winner on the next launch.
func DeleteCalibrationDecision(cacheDir, scopeKey string) error {
	path := CalibrationPath(cacheDir, scopeKey)
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// DeleteCalibrationDecisionsForModel removes every named calibration decision
// for a model. A promoted candidate can have a different placement/ubatch scope
// from the default key under which its decision was saved; model-scoped cleanup
// is the fail-closed runtime-OOM path that prevents that winner being replayed.
func DeleteCalibrationDecisionsForModel(cacheDir string, model *ModelProfile) error {
	if model == nil {
		return nil
	}
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	want := CalibrationModelBasename(model.Path)
	if want == "." || want == "" {
		want = CalibrationModelBasename(model.Basename)
	}
	if want == "." || want == "" {
		return nil
	}
	dir := filepath.Join(cacheDir, "calibration")
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		var decision CalibrationDecision
		if json.Unmarshal(data, &decision) != nil || CalibrationModelBasename(decision.ModelBasename) != want {
			continue
		}
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

// LatestCalibrationDecisionForModel returns the newest named decision for a
// model basename. It is inspect-only: callers still re-derive the winner
// strategy through the deterministic candidate generator.
func LatestCalibrationDecisionForModel(cacheDir, modelBasename string) (*CalibrationDecision, error) {
	want := CalibrationModelBasename(modelBasename)
	if want == "." || want == "" {
		return nil, fmt.Errorf("empty model identity")
	}
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	dir := filepath.Join(cacheDir, "calibration")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var best *CalibrationDecision
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		data, readErr := os.ReadFile(filepath.Join(dir, entry.Name()))
		if readErr != nil {
			continue
		}
		var decision CalibrationDecision
		if json.Unmarshal(data, &decision) != nil ||
			decision.SchemaVersion != CalibrationSchemaVersion ||
			decision.ScopeKey == "" || decision.Winner == "" ||
			CalibrationModelBasename(decision.ModelBasename) != want {
			continue
		}
		if best == nil || decision.MeasuredAt > best.MeasuredAt {
			copied := decision
			best = &copied
		}
	}
	if best == nil {
		return nil, fmt.Errorf("no calibration decision for %s", want)
	}
	return best, nil
}

// ListCalibrationDecisions returns cached decisions newest-first, capped at
// limit. A non-positive limit returns every readable record.
func ListCalibrationDecisions(cacheDir string, limit int) []CalibrationDecision {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	dir := filepath.Join(cacheDir, "calibration")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	out := make([]CalibrationDecision, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		data, readErr := os.ReadFile(filepath.Join(dir, entry.Name()))
		if readErr != nil {
			continue
		}
		var decision CalibrationDecision
		if json.Unmarshal(data, &decision) != nil ||
			decision.SchemaVersion != CalibrationSchemaVersion ||
			decision.ScopeKey == "" || decision.Winner == "" {
			continue
		}
		out = append(out, decision)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].MeasuredAt > out[j].MeasuredAt
	})
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

// FormatCalibrationStatus is the inspect line shared by launch, dry-run, TUI,
// and `ggrun status`.
func FormatCalibrationStatus(d *CalibrationDecision) string {
	if d == nil || d.Winner == "" {
		return "no measured winner yet (cold estimate on launch)"
	}
	residency := string(d.BaselineResidency)
	if residency == "" {
		residency = "measured"
	}
	outcome := d.FinalistOutcome
	if outcome == "" {
		if d.Winner == "default" {
			outcome = "baseline-won"
		} else {
			outcome = "promoted"
		}
	}
	line := ""
	if outcome == "unavailable" && d.Winner == "default" {
		line = fmt.Sprintf("%s, default retained", residency)
		if d.Finalist != "" {
			line += fmt.Sprintf("; finalist %s unavailable", d.Finalist)
		} else {
			line += "; finalist unavailable"
		}
	} else {
		line = fmt.Sprintf("%s, %s %s", residency, outcome, d.Winner)
	}
	if d.MeasuredAt != "" {
		line += ", measured " + d.MeasuredAt
	}
	if d.BaselineBottleneck != "" {
		line += "; bottleneck " + d.BaselineBottleneck
	}
	return line
}

// CalibrationCandidate is one alternative whole configuration to measure.
// The base strategy (index 0 in the slice returned by CalibrationCandidates) is
// the estimated default; the rest are deliberate, bounded variations. Placement
// alternatives are generated from the same Compute ledger; larger MoE ubatches
// must prove themselves through the launcher's contained allocation preflight.
// Context and KV quality remain fixed. Automatic batch/ubatch and slot count
// may move only through a fresh placement computation; explicit values stay
// constraints. Offloaded MoE plans may additionally challenge an automatic
// cold microbatch with bounded larger values.
type CalibrationCandidate struct {
	Name     string
	Strategy *Strategy
	Estimate CandidateEstimate
}

type batchPair struct {
	batch  int
	ubatch int
}

// CalibrationCandidates returns the estimated default plus the alternative
// placements worth measuring on this hardware. It returns just the default
// (length 1) — i.e. "nothing to calibrate" — whenever the alternatives collapse
// onto the default or the planner cannot prove they fit:
//
//   - CPU-only: GPU batch/placement calibration has no meaning
//   - no bounded batch, slot, KV, or topology neighbor survives the complete
//     placement ledger
//
// The launcher measures each candidate with the selected workload benchmark and
// keeps the fastest under the scope key; the default is always candidate 0 so a
// failed or inconclusive calibration degrades to today's behavior.
func CalibrationCandidates(caps *detect.Capabilities, model *ModelProfile, base *Strategy, opts Options) []CalibrationCandidate {
	if caps == nil || model == nil || base == nil {
		return nil
	}
	out := []CalibrationCandidate{{Name: "default", Strategy: base}}
	if base.Type == CPUOnly {
		return out
	}

	// A backend-provided hot-expert cache is a decode optimization and the
	// first challenger. Auto prefers leftover VRAM on the packed baseline; if
	// that cannot hold a useful cache, the challenger demotes GPU expert layers
	// until it can. The packed cache-free layout remains candidate 0. The live
	// agent workload still decides whether locality beats those extra CPU layers.
	// Compute already computed the challenger from the packed base and handed it
	// over as HotExpertCacheChallenger; regenerate it only when that hand-off is
	// absent (a cache-free base reached here without going through the auto
	// finalize path).
	hotCandidate := base.HotExpertCacheChallenger
	var hotErr error
	if hotCandidate == nil {
		hotCandidate, hotErr = hotExpertCacheCandidate(caps, model, base, opts)
	}
	if hotCandidate != nil {
		out = append(out, CalibrationCandidate{
			Name:     fmt.Sprintf("hot-experts-%d", hotCandidate.HotExpertCacheSlots),
			Strategy: hotCandidate,
		})
		if hotExpertRequestedSlots(opts) == 0 {
			if lower := hotExpertLowerSlotCandidate(caps, model, hotCandidate, opts); lower != nil {
				out = append(out, CalibrationCandidate{Name: fmt.Sprintf("hot-experts-%d", lower.HotExpertCacheSlots), Strategy: lower})
			}
		}
	} else if hotErr != nil && hotExpertOptimizerRequested(opts) {
		base.OptimizationExclusions = append(base.OptimizationExclusions, "hot-experts: "+hotErr.Error())
	}

	// Recurrent parallel-agent serving is governed by two independent limits:
	// n_batch controls how long one scheduler turn may ingest prompts, while
	// n_ubatch controls physical GPU work and graph allocation. Tune them as
	// valid pairs against the real mixed workload. The ascending ladder lets a
	// tight card fail closed before a larger graph is attempted, and every rung
	// still has to pass the launcher's exact allocation preflight and lifecycle.
	if base.HasSSM && base.Parallel > 1 &&
		strings.HasPrefix(opts.WorkloadProfile, "claude-agent-parallel-") &&
		!opts.BatchSizeExplicit && !opts.UBatchSizeExplicit {
		for _, pair := range []batchPair{
			{batch: 256, ubatch: 128},
			{batch: 256, ubatch: 256},
			{batch: 512, ubatch: 256},
			{batch: 512, ubatch: 512},
		} {
			if pair.batch == base.BatchSize && pair.ubatch == base.UBatchSize {
				continue
			}
			alt, err := recomputeBatchCandidate(caps, model, base, opts, pair.batch, pair.ubatch)
			if err != nil || alt == nil {
				continue
			}
			alt.BatchTuned = true
			name := fmt.Sprintf("batch-%d-ubatch-%d", pair.batch, pair.ubatch)
			if base.Type == MoEOffload && pair.batch == base.BatchSize && pair.ubatch > base.UBatchSize {
				name = fmt.Sprintf("ubatch-%d", pair.ubatch)
			}
			out = append(out, CalibrationCandidate{
				Name: name, Strategy: alt,
			})
		}
	}

	// Every GPU-backed standard launch owns a bounded coordinate search around
	// its stable batch pair. These are full placement recomputes, not argv
	// overlays: graph memory, checkpoints, expert residency, and context
	// allocation are re-priced together. A user-owned side stays fixed while an
	// automatic side may still move.
	batchCandidates := 0
	for _, pair := range calibrationBatchNeighbors(base, opts) {
		alt, err := recomputeBatchCandidate(caps, model, base, opts, pair.batch, pair.ubatch)
		if err != nil || alt == nil || !sameCalibrationResidency(base, alt) ||
			calibrationCandidateExists(out, alt) {
			continue
		}
		alt.BatchTuned = true
		name := fmt.Sprintf("batch-%d-ubatch-%d", pair.batch, pair.ubatch)
		if base.Type == MoEOffload && pair.batch == base.BatchSize && pair.ubatch > base.UBatchSize {
			name = fmt.Sprintf("ubatch-%d", pair.ubatch)
		}
		out = append(out, CalibrationCandidate{
			Name: name, Strategy: alt,
		})
		batchCandidates++
		if batchCandidates >= 24 {
			break
		}
	}

	// Slot count is a throughput/queueing dimension, not a free multiplier.
	// Challenge every bounded legal width that preserves useful per-agent context
	// and the current residency class. Explicit --parallel is a hard constraint
	// and never enters this search.
	if !opts.ParallelExplicit {
		for _, parallel := range calibrationParallelNeighbors(base, opts) {
			alt, err := recomputeParallelCandidate(caps, model, base, opts, parallel)
			if err != nil || alt == nil || !sameCalibrationResidency(base, alt) ||
				calibrationCandidateExists(out, alt) {
				continue
			}
			out = append(out, CalibrationCandidate{
				Name: fmt.Sprintf("parallel-%d", parallel), Strategy: alt,
			})
		}
	}

	switch base.Type {
	case MoEOffload:
		// A cold offloaded-MoE plan commonly lands on ubatch 512 because the
		// whole model cannot fit on one GPU, even when the distributed graph has
		// room for a much larger prefill microbatch. Challenge 1024 and 2048 and
		// measure them. This is never a blind default promotion: the launcher
		// still subjects each candidate to
		// allocation preflight, live benchmarking, lifecycle canaries, and the
		// cached-decision invalidation path.
		if !opts.UBatchSizeExplicit {
			for _, ubatch := range []int{512, 1024, 2048} {
				if ubatch <= base.UBatchSize || ubatch > base.BatchSize {
					continue
				}
				alt, err := recomputeBatchCandidate(caps, model, base, opts, base.BatchSize, ubatch)
				if err != nil || alt == nil || alt.Type != MoEOffload {
					continue
				}
				if calibrationCandidateExists(out, alt) {
					continue
				}
				out = append(out, CalibrationCandidate{Name: fmt.Sprintf("ubatch-%d", ubatch), Strategy: alt})
			}
		}

		if len(caps.GPUs) < 2 {
			return out
		}
		// KV-alternate: move the KV cache between GPU and CPU while keeping the
		// expert policy. Changing KV placement changes both VRAM expert capacity
		// and host RAM, so this must be a fresh Compute pass. Flipping the field on
		// a cloned strategy produced candidates that had never passed either
		// memory ledger.
		altKV := "cpu"
		if base.KVPlacement == "cpu" {
			altKV = "gpu"
		}
		altOpts := calibrationBaseOptions(opts, base)
		altOpts.KVPlacement = altKV
		if alt, err := Compute(caps, model, altOpts); err == nil && alt != nil && alt.Type == MoEOffload && alt.KVPlacement == altKV {
			out = append(out, CalibrationCandidate{Name: "kv-alternate", Strategy: alt})
		}
		for _, candidate := range moeTopologyCandidates(caps, model, base, opts) {
			if !calibrationCandidateExists(out, candidate.Strategy) {
				out = append(out, candidate)
			}
		}
	case MultiGPUDense:
		// Dense on multiple GPUs has exactly one real choice: which GPU owns the
		// output head and the largest split share. The default weights ownership
		// by bandwidth; try the VRAM-weighted inverse only when the fastest GPU is
		// not also the roomiest, which is the case where the estimate is most
		// likely to be wrong about end-to-end speed.
		if alt := invertDenseSplit(base); alt != nil {
			out = append(out, CalibrationCandidate{Name: "split-inverted", Strategy: alt})
		}
		for _, candidate := range denseTopologyCandidates(caps, model, base, opts) {
			if !calibrationCandidateExists(out, candidate.Strategy) {
				out = append(out, candidate)
			}
		}
	}
	return out
}

// hotExpertLowerSlotCandidate provides one deterministic auto fallback after a
// cache-on finalist fails post-start admission. It halves the original slots,
// retaining the complete topology and packed baseline snapshot.
func hotExpertLowerSlotCandidate(caps *detect.Capabilities, model *ModelProfile, original *Strategy, opts Options) *Strategy {
	if original == nil || original.HotExpertCacheSlots <= 0 || hotExpertRequestedSlots(opts) > 0 {
		return nil
	}
	slots := original.HotExpertCacheSlots / 2
	if minUseful := hotExpertMinUsefulSlots(model); slots < minUseful {
		slots = minUseful
	}
	if slots >= original.HotExpertCacheSlots || original.ResourceLedger == nil || !original.ResourceLedger.Fits || original.ResourceLedger.Host.SlackMB < 0 {
		return nil
	}
	shape, err := hotExpertCacheShapeFor(caps, model, original, opts)
	if err != nil {
		return nil
	}
	candidate := cloneStrategy(original)
	oldCharge := cloneIntMap(original.HotExpertCacheVRAMByGPU)
	originalLayout, err := hotExpertCacheLayout(shape, original.HotExpertCacheSlots)
	if err != nil || len(oldCharge) != len(originalLayout) {
		return nil
	}
	rowCount := make(map[int]int, len(original.ResourceLedger.Devices))
	for _, device := range original.ResourceLedger.Devices {
		rowCount[device.GPU]++
	}
	for gpu, charge := range originalLayout {
		if charge <= 0 || oldCharge[gpu] != charge || rowCount[gpu] != 1 {
			return nil
		}
	}
	for gpu, charge := range oldCharge {
		if charge <= 0 || originalLayout[gpu] != charge {
			return nil
		}
	}
	clearHotExpertCache(candidate)
	if err := assignHotExpertCache(candidate, shape, slots); err != nil {
		return nil
	}
	ledger := *original.ResourceLedger
	ledger.Devices = append([]DeviceResourceLedger(nil), original.ResourceLedger.Devices...)
	for i := range ledger.Devices {
		old := oldCharge[ledger.Devices[i].GPU]
		if ledger.Devices[i].HotExpertCacheMB > 0 && old <= 0 {
			return nil
		}
		if old > 0 && ledger.Devices[i].HotExpertCacheMB != old {
			return nil
		}
		if old > 0 {
			ledger.Devices[i].RequiredMB -= old
			ledger.Devices[i].SlackMB += old
		}
		ledger.Devices[i].HotExpertCacheMB = 0
		if ledger.Devices[i].RequiredMB < 0 || ledger.Devices[i].SlackMB < 0 || ledger.Devices[i].SlackMB > ledger.Devices[i].FreeMB {
			return nil
		}
	}
	ledger.Exact = false
	ledger.Fits = true
	applyHotExpertCacheLedger(&ledger, candidate, false)
	if !ledger.Fits {
		return nil
	}
	for gpu, charge := range candidate.HotExpertCacheVRAMByGPU {
		if charge <= 0 || charge > oldCharge[gpu] {
			return nil
		}
	}
	candidate.ResourceLedger = &ledger
	candidate.HotExpertCacheFreeBaseline = cloneStrategy(original.HotExpertCacheFreeBaseline)
	return candidate
}

func calibrationCandidateExists(candidates []CalibrationCandidate, strategy *Strategy) bool {
	if strategy == nil {
		return true
	}
	identity := calibrationStrategyIdentity(strategy)
	for _, candidate := range candidates {
		if calibrationStrategyIdentity(candidate.Strategy) == identity {
			return true
		}
	}
	return false
}

func calibrationStrategyIdentity(strategy *Strategy) string {
	if strategy == nil {
		return ""
	}
	return specHash(
		string(strategy.Type), fmt.Sprintf("%d", strategy.ContextSize),
		fmt.Sprintf("%d", strategy.Parallel), fmt.Sprintf("%d", strategy.BatchSize),
		fmt.Sprintf("%d", strategy.UBatchSize), strategy.KVPlacement, strategy.KVType,
		fmt.Sprintf("%t", strategy.MMapRequired), fmt.Sprintf("%d", strategy.NCPUMoE),
		fmt.Sprintf("%d", strategy.HotExpertCacheSlots),
		fmt.Sprintf("%d", strategy.HotExpertCacheInserts),
		splitCompactKey(strategy.TensorSplit), strategy.OTString, strategy.PlacementPolicy,
	)
}

func calibrationBatchNeighbors(base *Strategy, opts Options) []batchPair {
	if base == nil || (opts.BatchSizeExplicit && opts.UBatchSizeExplicit) {
		return nil
	}
	batch, ubatch := base.BatchSize, base.UBatchSize
	if batch <= 0 || ubatch <= 0 {
		return nil
	}
	out := make([]batchPair, 0, 32)
	seen := map[batchPair]bool{}
	add := func(nextBatch, nextUBatch int) {
		if opts.BatchSizeExplicit {
			nextBatch = batch
		}
		if opts.UBatchSizeExplicit {
			nextUBatch = ubatch
		}
		if nextBatch <= 0 || nextUBatch <= 0 || nextUBatch > nextBatch ||
			(nextBatch == batch && nextUBatch == ubatch) {
			return
		}
		pair := batchPair{batch: nextBatch, ubatch: nextUBatch}
		if !seen[pair] {
			seen[pair] = true
			out = append(out, pair)
		}
	}

	rungs := []int{32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}
	for _, rung := range rungs {
		// Search each coordinate independently as well as the common matched and
		// 2x/4x batch-to-ubatch shapes. Compute remains the memory-fit authority,
		// so unsupported or oversized pairs simply do not become candidates.
		add(batch, rung)
		add(rung, ubatch)
		add(rung, rung)
		if rung <= 4096 {
			add(rung*2, rung)
		}
		if rung <= 2048 {
			add(rung*4, rung)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		di := relativeIntegerDistance(out[i].batch, batch) +
			2*relativeIntegerDistance(out[i].ubatch, ubatch)
		dj := relativeIntegerDistance(out[j].batch, batch) +
			2*relativeIntegerDistance(out[j].ubatch, ubatch)
		if di != dj {
			return di < dj
		}
		if out[i].ubatch != out[j].ubatch {
			return out[i].ubatch > out[j].ubatch
		}
		return out[i].batch > out[j].batch
	})
	return out
}

func relativeIntegerDistance(value, baseline int) int {
	if baseline <= 0 {
		return value
	}
	return absInt(value-baseline) * 1024 / baseline
}

func adjacentBatchRung(value int, upward bool) int {
	rungs := []int{64, 128, 256, 512, 1024, 2048, 4096, 8192}
	if upward {
		for _, rung := range rungs {
			if rung > value {
				return rung
			}
		}
		return value
	}
	for i := len(rungs) - 1; i >= 0; i-- {
		if rungs[i] < value {
			return rungs[i]
		}
	}
	return value
}

func calibrationParallelNeighbors(base *Strategy, opts Options) []int {
	if base == nil || base.ContextSize <= 0 {
		return nil
	}
	current := base.Parallel
	if current <= 0 {
		current = 1
	}
	minPerSlot := opts.ParallelSlotTarget
	if minPerSlot <= 0 {
		minPerSlot = contextMinimum
	}
	maxParallel := base.ContextSize / minPerSlot
	if maxParallel > 8 {
		maxParallel = 8
	}
	if maxParallel < 1 {
		maxParallel = 1
	}
	out := make([]int, 0, maxParallel-1)
	for candidate := 1; candidate <= maxParallel; candidate++ {
		if candidate != current {
			out = append(out, candidate)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		di, dj := absInt(out[i]-current), absInt(out[j]-current)
		if di != dj {
			return di < dj
		}
		return out[i] > out[j]
	})
	return out
}

func sameCalibrationResidency(base, candidate *Strategy) bool {
	if base == nil || candidate == nil || base.ContextSize != candidate.ContextSize {
		return false
	}
	if !base.MMapRequired && candidate.MMapRequired {
		return false
	}
	baseGPUBacked := base.Type != DenseCPUOffload && base.Type != CPUOnly
	candidateGPUBacked := candidate.Type != DenseCPUOffload && candidate.Type != CPUOnly
	if baseGPUBacked && !candidateGPUBacked {
		return false
	}
	baseGPUResident := base.KVPlacement == "gpu" &&
		base.Type != DenseCPUOffload && base.Type != CPUOnly && !base.MMapRequired
	candidateGPUResident := candidate.KVPlacement == "gpu" &&
		candidate.Type != DenseCPUOffload && candidate.Type != CPUOnly && !candidate.MMapRequired
	return !baseGPUResident || candidateGPUResident
}

// recomputeBatchCandidate treats batch/ubatch as placement inputs, not late
// argv overlays. In particular, ubatch changes compute-buffer VRAM and can move
// experts between GPU and CPU. A clone of the old placement would therefore be
// an unproved configuration even if a later load happened to survive it.
func recomputeBatchCandidate(caps *detect.Capabilities, model *ModelProfile, base *Strategy, opts Options, batch, ubatch int) (*Strategy, error) {
	if batch <= 0 || ubatch <= 0 || ubatch > batch {
		return nil, fmt.Errorf("invalid batch candidate %d/%d", batch, ubatch)
	}
	altOpts := calibrationBaseOptions(opts, base)
	altOpts.BatchSize = batch
	altOpts.UBatchSize = ubatch
	// These bits are local to this Compute call: they force the exact candidate
	// through normalization without changing the user's intent bits in the
	// calibration scope.
	altOpts.BatchSizeExplicit = true
	altOpts.UBatchSizeExplicit = true
	// The candidate pair is the workload-policy challenge itself. Re-applying
	// the conservative cold-start RuntimePolicy inside this exact recompute would
	// collapse every candidate back to the baseline before it can be measured.
	altOpts.RuntimePolicy = nil
	alt, err := Compute(caps, model, altOpts)
	if err != nil || alt == nil {
		return nil, err
	}
	if alt.BatchSize != batch || alt.UBatchSize != ubatch {
		return nil, fmt.Errorf("batch candidate recomputed as %d/%d", alt.BatchSize, alt.UBatchSize)
	}
	return alt, nil
}

func recomputeParallelCandidate(caps *detect.Capabilities, model *ModelProfile, base *Strategy, opts Options, parallel int) (*Strategy, error) {
	if parallel <= 0 {
		return nil, fmt.Errorf("invalid parallel candidate %d", parallel)
	}
	altOpts := calibrationBaseOptions(opts, base)
	altOpts.Parallel = parallel
	altOpts.AutoParallel = false
	// A different scheduler width deserves its own automatic batch baseline.
	// Carrying a four-lane fairness cap into a one-lane candidate (or a large
	// serial batch into four lanes) would benchmark a coupled accident rather
	// than the candidate the standard launcher would actually choose.
	if !opts.BatchSizeExplicit {
		altOpts.BatchSize = 0
	}
	if !opts.UBatchSizeExplicit {
		altOpts.UBatchSize = 0
	}
	alt, err := Compute(caps, model, altOpts)
	if err != nil || alt == nil {
		return nil, err
	}
	if strategySlots(alt) != parallel {
		return nil, fmt.Errorf("parallel candidate recomputed as %d", strategySlots(alt))
	}
	return alt, nil
}

func calibrationBaseOptions(opts Options, base *Strategy) Options {
	alt := opts
	alt.SkipPlacementCache = true
	alt.CacheFile = ""
	alt.VerifiedConfigScopeKey = ""
	if base == nil {
		return alt
	}
	if base.ContextSize > 0 {
		alt.ContextSize = base.ContextSize
	}
	if base.Parallel > 0 {
		alt.Parallel = base.Parallel
	}
	if base.BatchSize > 0 {
		alt.BatchSize = base.BatchSize
	}
	if base.UBatchSize > 0 {
		alt.UBatchSize = base.UBatchSize
	}
	if base.KVQuality != "" {
		alt.KVQuality = base.KVQuality
	}
	return alt
}

// cloneStrategy deep-copies the placement-affecting fields of a strategy so a
// candidate can diverge without aliasing the base's slices.
func cloneStrategy(s *Strategy) *Strategy {
	if s == nil {
		return nil
	}
	c := *s
	if s.TensorSplit != nil {
		c.TensorSplit = append([]float64(nil), s.TensorSplit...)
	}
	if s.Draft != nil {
		d := *s.Draft
		c.Draft = &d
	}
	if s.CompanionPlacements != nil {
		c.CompanionPlacements = append([]CompanionPlacement(nil), s.CompanionPlacements...)
	}
	if s.VRAMLedger != nil {
		c.VRAMLedger = append([]GPULedgerEntry(nil), s.VRAMLedger...)
	}
	c.HotExpertCacheVRAMByGPU = cloneIntMap(s.HotExpertCacheVRAMByGPU)
	c.HotExpertCacheLayersByGPU = cloneIntMap(s.HotExpertCacheLayersByGPU)
	if s.HotExpertCacheFreeBaseline != nil {
		// The nested snapshot always has a nil HotExpertCacheFreeBaseline of its
		// own, so this recursion terminates after one level.
		c.HotExpertCacheFreeBaseline = cloneStrategy(s.HotExpertCacheFreeBaseline)
	}
	// The cache-on challenger is a transient candidate-generation artifact; a
	// clone re-derives it from the packed base when it is needed.
	c.HotExpertCacheChallenger = nil
	c.ResourceLedger = nil
	c.Residency = ""
	c.OptimizationBottleneck = ""
	c.EstimatedAgentCost = 0
	c.EstimateConfidence = ""
	c.OptimizationBoundary = nil
	return &c
}

// invertDenseSplit returns a copy of a multi-GPU dense strategy with the split
// ratio reversed across devices, or nil when there is nothing meaningful to
// invert (single share, or an already-symmetric split).
func invertDenseSplit(s *Strategy) *Strategy {
	if s == nil || len(s.TensorSplit) < 2 {
		return nil
	}
	reversed := make([]float64, len(s.TensorSplit))
	for i, v := range s.TensorSplit {
		reversed[len(s.TensorSplit)-1-i] = v
	}
	// An inversion that reproduces the same split is not a distinct candidate.
	same := true
	for i := range reversed {
		if reversed[i] != s.TensorSplit[i] {
			same = false
			break
		}
	}
	if same {
		return nil
	}
	c := cloneStrategy(s)
	c.TensorSplit = reversed
	// The output head follows the largest share, so the main GPU moves too.
	if len(reversed) > 0 {
		best := 0
		for i, v := range reversed {
			if v > reversed[best] {
				best = i
			}
		}
		c.MainGPU = best
	}
	return c
}

// denseTopologyCandidates enumerates legal single-device and bounded
// multi-device subsets without changing the backend-visible GPU namespace.
// Unused cards receive a zero tensor share, so the strategy remains a complete
// argv that the existing exact admission gate can validate directly.
func denseTopologyCandidates(caps *detect.Capabilities, model *ModelProfile, base *Strategy, opts Options) []CalibrationCandidate {
	if caps == nil || model == nil || base == nil || len(caps.GPUs) < 2 ||
		model.TotalSizeMB <= 0 || model.NumLayers <= 0 || base.ContextSize <= 0 {
		return nil
	}
	runtimeCaps, err := restrictGPUs(caps, opts.GPUs)
	if err != nil || runtimeCaps == nil || len(runtimeCaps.GPUs) < 2 {
		return nil
	}
	caps = runtimeCaps
	topologyOpts := opts
	topologyOpts.GPUs = nil
	n := len(caps.GPUs)
	if n > 8 {
		// Consumer/workstation hosts are normally far smaller. Avoid an
		// exponential cold-start calculation on a large cluster; its topology
		// belongs to an explicit maintenance sweep and a later cluster solver.
		n = 8
	}
	ledger := BuildResourceLedger(caps, model, base, topologyOpts)
	fixed := make([]int, len(caps.GPUs))
	free := make([]int, len(caps.GPUs))
	for i, gpu := range caps.GPUs {
		free[i] = gpu.VRAMFreeMB()
		if planned, ok := base.PlanFreeVRAM[gpu.Index]; ok {
			free[i] = planned
		}
		if i < len(ledger.Devices) {
			fixed[i] = ledger.Devices[i].GraphMB + ledger.Devices[i].RuntimeMB
		}
	}
	kvMB := computeKVTotalMB(model, base.ContextSize, base.KVType, base.SWAFull)
	if base.ContextAllocationMB > 0 {
		kvMB = base.ContextAllocationMB
	}
	combined := model.TotalSizeMB
	if base.KVPlacement != "cpu" {
		combined += kvMB
	}
	if combined <= 0 {
		return nil
	}

	out := make([]CalibrationCandidate, 0, 2*n)
	for i := 0; i < n; i++ {
		if free[i]-fixed[i] < combined {
			continue
		}
		candidate := cloneStrategy(base)
		candidate.Type = SingleGPU
		candidate.MainGPU = caps.GPUs[i].Index
		candidate.TensorSplit = nil
		candidate.SplitMode = ""
		candidate.PlacementPolicy = "single"
		candidate.NCPUMoE = 0
		candidate.OTString = ""
		candidate.VRAMLedger = nil
		candidate.MMap = false
		candidate.MMapRequired = false
		clearForeignAllocationEvidence(candidate)
		candidateLedger := BuildResourceLedger(caps, model, candidate, topologyOpts)
		if candidateLedger.Fits {
			out = append(out, CalibrationCandidate{
				Name: fmt.Sprintf("single-gpu-%d", caps.GPUs[i].Index), Strategy: candidate,
			})
		}
	}

	limit := 1 << n
	for mask := 1; mask < limit; mask++ {
		if bitsSet(mask) < 2 {
			continue
		}
		for _, policy := range []string{"critical", "capacity"} {
			split, ok := denseSubsetSplit(caps.GPUs[:n], free[:n], fixed[:n], combined, mask, policy)
			if !ok {
				continue
			}
			if len(caps.GPUs) > n {
				split = append(split, make([]float64, len(caps.GPUs)-n)...)
			}
			candidate := cloneStrategy(base)
			candidate.Type = MultiGPUDense
			candidate.TensorSplit = split
			candidate.SplitMode = "layer"
			candidate.PlacementPolicy = policy
			candidate.MainGPU = splitOutputOwner(split, model.NumLayers)
			clearForeignAllocationEvidence(candidate)
			candidateLedger := BuildResourceLedger(caps, model, candidate, topologyOpts)
			if !candidateLedger.Fits {
				continue
			}
			out = append(out, CalibrationCandidate{
				Name:     fmt.Sprintf("topology-%s-%s", policy, splitGPUSet(split)),
				Strategy: candidate,
			})
		}
	}
	return out
}

func denseSubsetSplit(gpus []detect.GPU, free, fixed []int, combined, mask int, policy string) ([]float64, bool) {
	n := len(gpus)
	split := make([]float64, n)
	capacity := make([]int, n)
	totalCapacity := 0
	for i := 0; i < n; i++ {
		if mask&(1<<i) == 0 {
			continue
		}
		capacity[i] = max(0, free[i]-fixed[i])
		totalCapacity += capacity[i]
	}
	if totalCapacity < combined {
		return nil, false
	}
	if policy == "capacity" {
		for i := range split {
			if capacity[i] > 0 {
				split[i] = float64(capacity[i]) / float64(totalCapacity)
			}
		}
		for i := 0; i < n; i++ {
			if mask&(1<<i) != 0 && split[i] <= 0 {
				return nil, false
			}
		}
		return normalizeSplit(split), true
	}
	order := make([]int, 0, n)
	for i := 0; i < n; i++ {
		if capacity[i] > 0 {
			order = append(order, i)
		}
	}
	sort.Slice(order, func(i, j int) bool {
		left, right := effectiveMemoryBandwidth(gpus[order[i]]), effectiveMemoryBandwidth(gpus[order[j]])
		if left != right {
			return left > right
		}
		if capacity[order[i]] != capacity[order[j]] {
			return capacity[order[i]] > capacity[order[j]]
		}
		return gpus[order[i]].Index < gpus[order[j]].Index
	})
	remaining := combined
	for _, i := range order {
		if remaining <= 0 {
			break
		}
		take := min(capacity[i], remaining)
		split[i] = float64(take) / float64(combined)
		remaining -= take
	}
	if remaining > 0 {
		return nil, false
	}
	for i := 0; i < n; i++ {
		if mask&(1<<i) != 0 && split[i] <= 0 {
			// If the faster card alone holds the model, this mask is not a real
			// multi-device topology; its single-GPU candidate already represents
			// the same placement without paying another device's entry overhead.
			return nil, false
		}
	}
	return normalizeSplit(split), true
}

func moeTopologyCandidates(caps *detect.Capabilities, model *ModelProfile, base *Strategy, opts Options) []CalibrationCandidate {
	if caps == nil || model == nil || base == nil || len(caps.GPUs) < 2 ||
		model.TotalSizeMB <= 0 || model.NumLayers <= 0 {
		return nil
	}
	out := make([]CalibrationCandidate, 0, 2+len(caps.GPUs))
	for _, policy := range []string{"memory", "capacity"} {
		altOpts := calibrationBaseOptions(opts, base)
		altOpts.PlacementPolicy = policy
		alt, err := Compute(caps, model, altOpts)
		if err != nil || alt == nil || alt.Type != MoEOffload ||
			!sameCalibrationResidency(base, alt) {
			continue
		}
		out = append(out, CalibrationCandidate{Name: "moe-topology-" + policy, Strategy: alt})
	}
	// A reweighted multi-owner layer split is not equivalent to moving the
	// serial transformer pipeline onto one device. In particular, a small first
	// card can own the leading dense blocks and saturate while larger cards wait.
	// Generate one bounded owner hypothesis per selected GPU through full
	// Compute, allowing the remaining cards to serve as complete-expert storage.
	// Contained allocation and the live workload still decide whether any of
	// these candidates is safe and faster.
	for _, gpu := range caps.GPUs {
		owner := gpu.Index
		altOpts := calibrationBaseOptions(opts, base)
		altOpts.PlacementPolicy = "link"
		altOpts.MoESplitOwnerGPU = &owner
		alt, err := Compute(caps, model, altOpts)
		// Say WHY a topology hypothesis was dropped. The optimizer previously
		// reported only "no non-rejected same-workload topology can relieve
		// GPU N" while every candidate vanished silently, which hid a real
		// defect for a long time: a de-owned device was still charged its old
		// split-owner compute buffer, so no owner hypothesis ever scored better
		// than the baseline it was meant to beat. A named rejection is the
		// difference between "this rig has no better topology" and "the model
		// cannot see the one it has".
		switch {
		case err != nil:
			moeTopologyRejected(owner, "placement did not resolve: "+err.Error())
			continue
		case alt == nil:
			moeTopologyRejected(owner, "placement returned no strategy")
			continue
		case alt.Type != MoEOffload:
			moeTopologyRejected(owner, fmt.Sprintf("resolved to %s, not an offloaded MoE plan", alt.Type))
			continue
		case !sameCalibrationResidency(base, alt):
			moeTopologyRejected(owner, fmt.Sprintf(
				"different residency than the baseline (ctx %d vs %d, mmap %t vs %t): not the same workload",
				alt.ContextSize, base.ContextSize, alt.MMapRequired, base.MMapRequired))
			continue
		}
		out = append(out, CalibrationCandidate{
			Name: fmt.Sprintf("moe-owner-%d", owner), Strategy: alt,
		})
	}
	return out
}

// moeTopologyRejected records why one owner hypothesis was not offered. It is
// deliberately visible on the ordinary launch path: a silently empty candidate
// set is indistinguishable from a rig that genuinely has no better topology.
func moeTopologyRejected(owner int, reason string) {
	fmt.Fprintf(os.Stderr, "[placement] topology hypothesis owner-%d rejected: %s\n", owner, reason)
}

func clearForeignAllocationEvidence(s *Strategy) {
	if s == nil {
		return
	}
	s.ContextAllocationMB = 0
	s.ContextAllocationEvidence = ""
	s.PlacementCacheHit = false
	s.VerifiedConfigReused = false
	s.PlacementCachePath = ""
	s.ResourceLedger = nil
}

func splitOutputOwner(split []float64, layers int) int {
	if layers > 0 {
		_, output := layerOwnership(split, layers)
		if output >= 0 {
			return output
		}
	}
	best := 0
	for i := range split {
		if split[i] > split[best] {
			best = i
		}
	}
	return best
}

func splitGPUSet(split []float64) string {
	parts := make([]string, 0, len(split))
	for i, share := range split {
		if share > 0 {
			parts = append(parts, fmt.Sprintf("%d", i))
		}
	}
	return strings.Join(parts, "-")
}

func bitsSet(value int) int {
	count := 0
	for value > 0 {
		count += value & 1
		value >>= 1
	}
	return count
}

// CalibrationScopeKey identifies the exact launch shape a calibration decision
// is valid for. Any field change — model, backend, hardware, workload, or the
// runtime knobs a candidate shares with the default — must produce a different
// key, or a stale decision could be applied to a launch it never measured.
type CalibrationScopeKey struct {
	ModelIdentity       string
	BackendIdentity     string
	HardwareID          string
	WorkloadProfile     string
	WorkloadConcurrency int
	ContextSize         int
	Parallel            int
	ParallelExplicit    bool
	BatchSize           int
	UBatchSize          int
	Threads             int
	ThreadsBatch        int
	CPURange            string
	CPUStrict           bool
	BatchExplicit       bool
	UBatchExplicit      bool
	KVQuality           string
	KVQualityV          string
	KVType              string
	GPUSet              string
	BasePlacement       string
	MemoryPolicy        string
	SamplingProfile     string
	CompanionPolicy     string
	SpecPolicy          string
	HotExpertPolicy     string
	BackendCapabilities string
	// SWAFull is part of the scope because it changes the KV allocation without
	// changing anything else the key already carries (see placement.go:6640-6643):
	// a config verified at the default window sizing is not a config for a
	// --swa-full launch, and reusing it hands the backend a layout that cannot
	// fit.
	SWAFull bool
	// ChatTemplate is part of the scope because a forced chat-template override
	// (catalog Entry.Name) is a different serving contract — the flags emitted
	// for --chat-template-file differ from an auto-matched template.
	ChatTemplate string
}

// NewCalibrationScopeKey builds the key from the same identity sources the
// speculative performance profile uses, so a calibration decision and a spec
// decision for the same launch can never disagree about what launch they
// describe.
func NewCalibrationScopeKey(model *ModelProfile, caps *detect.Capabilities, opts Options, base *Strategy) CalibrationScopeKey {
	contextSize := opts.ContextSize
	parallel := opts.Parallel
	batchSize := opts.BatchSize
	ubatchSize := opts.UBatchSize
	threads := opts.Threads
	threadsBatch := opts.Threads
	cpuRange := ""
	cpuStrict := false
	kvQuality := opts.KVQuality
	kvType := ""
	basePlacement := ""
	draftPolicy := ""
	if base != nil {
		if base.ContextSize > 0 {
			contextSize = base.ContextSize
		}
		if base.Parallel > 0 {
			parallel = base.Parallel
		}
		if base.BatchSize > 0 {
			batchSize = base.BatchSize
		}
		if base.UBatchSize > 0 {
			ubatchSize = base.UBatchSize
		}
		if base.Threads > 0 {
			threads = base.Threads
		}
		if base.ThreadsBatch > 0 {
			threadsBatch = base.ThreadsBatch
		}
		cpuRange = base.CPURange
		cpuStrict = base.CPUStrict
		if base.KVQuality != "" {
			kvQuality = base.KVQuality
		}
		kvType = base.KVType
		basePlacement = specHash(
			string(base.Type), base.KVPlacement, fmt.Sprintf("%t", base.MMap),
			fmt.Sprintf("%d", base.NCPUMoE), splitCompactKey(base.TensorSplit), base.OTString,
			fmt.Sprintf("%d", base.HotExpertCacheSlots),
			fmt.Sprintf("%d", base.HotExpertCacheInserts),
		)
		if base.Draft != nil {
			draftPolicy = specHash(
				string(base.Draft.Type), SpecCompanionIdentity(base.Draft.Path),
				base.Draft.KVTypeDraft, fmt.Sprintf("%d", base.Draft.CTXSizeDraft),
				fmt.Sprintf("%d", base.Draft.DraftMax), fmt.Sprintf("%d", base.Draft.VRAMMB),
				fmt.Sprintf("%d", base.Draft.DraftGPU),
			)
		}
	}
	return CalibrationScopeKey{
		ModelIdentity:       SpecTargetIdentity(model),
		BackendIdentity:     opts.BackendIdentity,
		HardwareID:          SpecHardwareIdentity(caps),
		WorkloadProfile:     opts.WorkloadProfile,
		WorkloadConcurrency: max(1, opts.WorkloadConcurrency),
		ContextSize:         contextSize,
		Parallel:            parallel,
		ParallelExplicit:    opts.ParallelExplicit,
		BatchSize:           batchSize,
		UBatchSize:          ubatchSize,
		Threads:             threads,
		ThreadsBatch:        threadsBatch,
		CPURange:            cpuRange,
		CPUStrict:           cpuStrict,
		BatchExplicit:       opts.BatchSizeExplicit,
		UBatchExplicit:      opts.UBatchSizeExplicit,
		KVQuality:           kvQuality,
		KVQualityV:          opts.KVQualityV,
		KVType:              kvType,
		GPUSet:              specGPUSet(opts.GPUs),
		BasePlacement:       basePlacement,
		MemoryPolicy: fmt.Sprintf("ram=%d,pct=%d,ram-head=%d,vram-head=%d,no-mmap=%t,force-mmap=%t,measured-buffers=%t,auto-ctx-max=%d,auto-parallel=%t,slot-target=%d",
			opts.RamBudgetMB, opts.RAMLimitPercent, opts.RAMHeadroomMB, opts.VRAMHeadroomMB,
			opts.NoMMap, opts.ForceMMap, opts.RequireMeasuredBuffers,
			opts.AutoContextMax, opts.AutoParallel, opts.ParallelSlotTarget),
		SamplingProfile: opts.SamplingProfile,
		CompanionPolicy: calibrationCompanionPolicy(opts.Companions),
		SpecPolicy:      specHash(opts.SpecMode, fmt.Sprintf("%t", opts.ForceSpecMoE), draftPolicy),
		HotExpertPolicy: specHash(
			strings.ToLower(strings.TrimSpace(opts.HotExperts)),
			fmt.Sprintf("%d", opts.HotExpertCacheSlots),
		),
		BackendCapabilities: specHash(
			opts.BackendHelp,
			string(effectiveCPUExpertMMapCapability(opts)),
			opts.CPUExpertMMapEvidence,
		),
		SWAFull:      opts.SWAFull,
		ChatTemplate: opts.ChatTemplate,
	}
}

func calibrationCompanionPolicy(companions []CompanionReservation) string {
	if len(companions) == 0 {
		return ""
	}
	parts := make([]string, 0, len(companions))
	for _, companion := range companions {
		preferences := make([]string, 0, len(companion.GPUPreference))
		for _, gpu := range companion.GPUPreference {
			preferences = append(preferences, fmt.Sprintf("%d", gpu))
		}
		parts = append(parts, specHash(
			companion.Name, fmt.Sprintf("%d", companion.VRAMMB),
			strings.Join(preferences, ","), fmt.Sprintf("%t", companion.AllowCPU),
		))
	}
	return specHash(parts...)
}

// String renders the key as a stable, opaque hash for use as a cache filename.
func (k CalibrationScopeKey) String() string {
	return specHash(
		k.ModelIdentity, k.BackendIdentity, k.HardwareID, k.WorkloadProfile,
		fmt.Sprintf("%d", max(1, k.WorkloadConcurrency)),
		fmt.Sprintf("%d", k.ContextSize), fmt.Sprintf("%d", k.Parallel),
		fmt.Sprintf("%d", k.BatchSize), fmt.Sprintf("%d", k.UBatchSize),
		fmt.Sprintf("%d", k.Threads), fmt.Sprintf("%d", k.ThreadsBatch),
		k.CPURange, fmt.Sprintf("%t", k.CPUStrict),
		fmt.Sprintf("%t", k.ParallelExplicit),
		fmt.Sprintf("%t", k.BatchExplicit), fmt.Sprintf("%t", k.UBatchExplicit),
		k.KVQuality, k.KVQualityV, k.KVType, k.GPUSet, k.BasePlacement, k.MemoryPolicy,
		k.SamplingProfile, k.CompanionPolicy, k.SpecPolicy, k.HotExpertPolicy, k.BackendCapabilities,
		fmt.Sprintf("%t", k.SWAFull), k.ChatTemplate,
	)
}
