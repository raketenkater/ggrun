package placement

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// CalibrationSchemaVersion bumps whenever the candidate set or scoring changes,
// so a decision measured under older semantics is never applied after an
// upgrade changes what "fastest" means.
const CalibrationSchemaVersion = 2

// CalibrationDecision records which candidate won a measured first-launch
// calibration for one scope, with the numbers that decided it. The winner is
// stored by name and re-derived from the deterministic candidate generator on
// later launches, so the full placement is reproduced exactly rather than
// partially deserialized.
type CalibrationDecision struct {
	SchemaVersion int     `json:"schema_version"`
	ScopeKey      string  `json:"scope_key"`
	Winner        string  `json:"winner"` // candidate Name, e.g. "default" or "kv-alternate"
	DefaultTPS    float64 `json:"default_tps"`
	WinnerTPS     float64 `json:"winner_tps"`
	Improvement   float64 `json:"improvement_pct"`
	MeasuredAt    string  `json:"measured_at"`
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
	path := CalibrationPath(cacheDir, d.ScopeKey)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return "", err
	}
	data, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return "", err
	}
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

// CalibrationCandidate is one alternative placement to measure at first launch.
// The base strategy (index 0 in the slice returned by CalibrationCandidates) is
// the estimated default; the rest are deliberate variations the planner believes
// also fit, generated from the same Compute ledger rather than hand-tuned per
// model. A candidate is only ever a different *choice* of where things live —
// never a different context size, slot count, or KV type, so a single scope key
// covers the whole set.
type CalibrationCandidate struct {
	Name     string
	Strategy *Strategy
}

// CalibrationCandidates returns the estimated default plus the alternative
// placements worth measuring on this hardware. It returns just the default
// (length 1) — i.e. "nothing to calibrate" — whenever the alternatives collapse
// onto the default or the planner cannot prove they fit:
//
//   - single GPU or CPU-only: expert/KV relocation has no meaning
//   - non-MoE single-GPU: there is only one place for the weights to go
//   - no alternative survives the same free-VRAM ledger the default passed
//
// The launcher measures each candidate with the same micro-probe and keeps the
// fastest under the scope key; the default is always candidate 0 so a failed or
// inconclusive calibration degrades to today's behavior.
func CalibrationCandidates(caps *detect.Capabilities, model *ModelProfile, base *Strategy, opts Options) []CalibrationCandidate {
	if caps == nil || model == nil || base == nil {
		return nil
	}
	out := []CalibrationCandidate{{Name: "default", Strategy: base}}
	if len(caps.GPUs) < 2 || base.Type == CPUOnly {
		return out
	}

	switch base.Type {
	case MoEOffload:
		// KV-alternate: move the KV cache between GPU and CPU while keeping the
		// expert policy. Changing KV placement changes both VRAM expert capacity
		// and host RAM, so this must be a fresh Compute pass. Flipping the field on
		// a cloned strategy produced candidates that had never passed either
		// memory ledger.
		altKV := "cpu"
		if base.KVPlacement == "cpu" {
			altKV = "gpu"
		}
		altOpts := opts
		altOpts.KVPlacement = altKV
		altOpts.SkipPlacementCache = true
		altOpts.CacheFile = ""
		if base.ContextSize > 0 {
			altOpts.ContextSize = base.ContextSize
		}
		if base.Parallel > 0 {
			altOpts.Parallel = base.Parallel
		}
		if base.BatchSize > 0 {
			altOpts.BatchSize = base.BatchSize
		}
		if base.UBatchSize > 0 {
			altOpts.UBatchSize = base.UBatchSize
		}
		if base.KVQuality != "" {
			altOpts.KVQuality = base.KVQuality
		}
		if alt, err := Compute(caps, model, altOpts); err == nil && alt != nil && alt.Type == MoEOffload && alt.KVPlacement == altKV {
			out = append(out, CalibrationCandidate{Name: "kv-alternate", Strategy: alt})
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
	}
	return out
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

// CalibrationScopeKey identifies the exact launch shape a calibration decision
// is valid for. Any field change — model, backend, hardware, workload, or the
// runtime knobs a candidate shares with the default — must produce a different
// key, or a stale decision could be applied to a launch it never measured.
type CalibrationScopeKey struct {
	ModelIdentity   string
	BackendIdentity string
	HardwareID      string
	WorkloadProfile string
	ContextSize     int
	Parallel        int
	BatchSize       int
	UBatchSize      int
	KVQuality       string
	KVType          string
	GPUSet          string
	BasePlacement   string
	MemoryPolicy    string
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
	kvQuality := opts.KVQuality
	kvType := ""
	basePlacement := ""
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
		if base.KVQuality != "" {
			kvQuality = base.KVQuality
		}
		kvType = base.KVType
		basePlacement = specHash(
			string(base.Type), base.KVPlacement, fmt.Sprintf("%t", base.MMap),
			fmt.Sprintf("%d", base.NCPUMoE), splitCompactKey(base.TensorSplit), base.OTString,
		)
	}
	return CalibrationScopeKey{
		ModelIdentity:   SpecTargetIdentity(model),
		BackendIdentity: opts.BackendIdentity,
		HardwareID:      SpecHardwareIdentity(caps),
		WorkloadProfile: opts.WorkloadProfile,
		ContextSize:     contextSize,
		Parallel:        parallel,
		BatchSize:       batchSize,
		UBatchSize:      ubatchSize,
		KVQuality:       kvQuality,
		KVType:          kvType,
		GPUSet:          specGPUSet(opts.GPUs),
		BasePlacement:   basePlacement,
		MemoryPolicy: fmt.Sprintf("ram=%d,pct=%d,ram-head=%d,vram-head=%d,no-mmap=%t,force-mmap=%t,measured-buffers=%t",
			opts.RamBudgetMB, opts.RAMLimitPercent, opts.RAMHeadroomMB, opts.VRAMHeadroomMB, opts.NoMMap, opts.ForceMMap, opts.RequireMeasuredBuffers),
	}
}

// String renders the key as a stable, opaque hash for use as a cache filename.
func (k CalibrationScopeKey) String() string {
	return specHash(
		k.ModelIdentity, k.BackendIdentity, k.HardwareID, k.WorkloadProfile,
		fmt.Sprintf("%d", k.ContextSize), fmt.Sprintf("%d", k.Parallel),
		fmt.Sprintf("%d", k.BatchSize), fmt.Sprintf("%d", k.UBatchSize),
		k.KVQuality, k.KVType, k.GPUSet, k.BasePlacement, k.MemoryPolicy,
	)
}
