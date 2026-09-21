package placement

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
)

const SpecProfileSchemaVersion = 1

// SpecProfileScope is the exact runtime shape for which a speculative result
// was measured. Changing any field invalidates the cached decision.
type SpecProfileScope struct {
	TargetIdentity    string `json:"target_identity"`
	CompanionIdentity string `json:"companion_identity"`
	BackendIdentity   string `json:"backend_identity"`
	HardwareIdentity  string `json:"hardware_identity"`
	GPUSet            string `json:"gpu_set"`
	Kind              string `json:"kind"`
	ContextSize       int    `json:"context_size"`
	Parallel          int    `json:"parallel"`
	SamplingProfile   string `json:"sampling_profile"`
	ReasoningOff      bool   `json:"reasoning_off"`
	SpecMode          string `json:"spec_mode"`      // NEW
	ForceSpecMoE      bool   `json:"force_spec_moe"` // NEW
	Threads           int    `json:"threads"`        // NEW
	CacheRAMMB        int    `json:"cache_ram_mb"`   // NEW
}

type SpecPerformanceProfile struct {
	SchemaVersion        int              `json:"schema_version"`
	Scope                SpecProfileScope `json:"scope"`
	ScopeKey             string           `json:"scope_key"`
	TargetSHA256         string           `json:"target_sha256,omitempty"`
	CompanionSHA256      string           `json:"companion_sha256,omitempty"`
	LaunchIdentity       string           `json:"launch_identity"`
	DraftMax             int              `json:"draft_max"`
	BaselineTPS          float64          `json:"baseline_tps"`
	SpeculativeTPS       float64          `json:"speculative_tps"`
	ImprovementPct       float64          `json:"improvement_pct"`
	WallImprovementPct   float64          `json:"wall_improvement_pct"`
	PromptRegressionPct  float64          `json:"prompt_regression_pct"`
	OutputLengthDeltaPct float64          `json:"output_length_delta_pct"`
	PromptCases          int              `json:"prompt_cases"`
	RepeatedRounds       int              `json:"repeated_rounds"`
	MaxPromptTokens      int              `json:"max_prompt_tokens"`
	CorrectnessPassed    bool             `json:"correctness_passed"`
	StabilityPassed      bool             `json:"stability_passed"`
	ParallelLoadPassed   bool             `json:"parallel_load_passed"`
	Complete             bool             `json:"complete"`
	MeasuredAt           string           `json:"measured_at"`
}

func specHash(parts ...string) string {
	h := sha256.New()
	for _, part := range parts {
		_, _ = h.Write([]byte(strconv.Itoa(len(part))))
		_, _ = h.Write([]byte{':'})
		_, _ = h.Write([]byte(part))
	}
	return hex.EncodeToString(h.Sum(nil))
}

func SpecTargetIdentity(model *ModelProfile) string {
	if model == nil {
		return ""
	}
	return specHash(
		filepath.Base(model.Path), strconv.FormatInt(model.SizeBytes, 10), strconv.Itoa(model.TotalSizeMB),
		strings.ToLower(model.ModelArch), strconv.Itoa(model.NumLayers), strconv.Itoa(model.EmbeddingLength),
		strconv.Itoa(model.VocabSize), strings.ToLower(model.TokenizerModel), strings.ToLower(model.TokenizerPre),
		strings.ToLower(model.TokenizerHash), strconv.Itoa(model.NextNPredictLayers), specArtifactStatIdentity(model.Path),
	)
}

func SpecCompanionIdentity(path string) string {
	if strings.TrimSpace(path) == "" {
		return "embedded"
	}
	fi, err := os.Stat(path)
	if err != nil {
		return ""
	}
	return specHash(filepath.Base(path), strconv.FormatInt(fi.Size(), 10), strconv.FormatInt(fi.ModTime().UnixNano(), 10))
}

var splitGGUFRE = regexp.MustCompile(`^(.*)-[0-9]{5}-of-([0-9]{5})\.gguf$`)

// specArtifactStatIdentity cheaply invalidates a proof when a local artifact is
// replaced. It includes every shard without reading hundreds of gigabytes of
// tensor data; reviewed downloads are additionally revision/checksum pinned.
func specArtifactStatIdentity(path string) string {
	paths := []string{path}
	base := filepath.Base(path)
	if match := splitGGUFRE.FindStringSubmatch(base); len(match) == 3 {
		pattern := filepath.Join(filepath.Dir(path), match[1]+"-?????-of-"+match[2]+".gguf")
		if matches, err := filepath.Glob(pattern); err == nil && len(matches) > 0 {
			paths = matches
		}
	}
	sort.Strings(paths)
	parts := make([]string, 0, len(paths)*3)
	for _, candidate := range paths {
		fi, err := os.Stat(candidate)
		if err != nil {
			parts = append(parts, filepath.Base(candidate), "missing")
			continue
		}
		parts = append(parts, filepath.Base(candidate), strconv.FormatInt(fi.Size(), 10), strconv.FormatInt(fi.ModTime().UnixNano(), 10))
	}
	return specHash(parts...)
}

// SpecHardwareIdentity identifies the PHYSICAL machine a cached record belongs to.
//
// It must be stable across re-measurement, because it scopes both the speculative
// profile and the calibration decision key (calibrate.go:1400). It previously
// included gpu.BandwidthMBps, a MEASURED value that drifts by single-digit MB/s
// run to run on fixed hardware; a +3 MB/s re-measure minted a new identity and
// orphaned the record. That is the same defect fixed in gpuIdentityHash, and it
// mattered more here: this key gates reuse of a whole admission-shaping argv, not
// just a stored speed figure.
//
// Identity-class facts only. Bandwidth is deliberately absent: it is an ordering
// input to planning, not a machine identity. (An earlier version of this comment
// justified that by pointing at a coarse bandwidth class in the placement key;
// that class was removed in 97404a3 because tensorSplit already carries what a
// bandwidth change moves, so the key separates plans without a bandwidth term.
// The conclusion is unchanged and now rests on the plan key carrying the split,
// not on a class that no longer exists.)
//
// g.Index is absent for the same reason as in gpuIdentityHash: it is a slot
// ordinal assigned by bus-id sort (detect.go:185-189), so a reseat or a driver
// reload re-orphaned the record for no information gained. PCIBusID is the
// card's stable address; PCIGen and PCILanes are kept because link shape is a
// real per-machine property that does not move when a measurement repeats.
func SpecHardwareIdentity(caps *detect.Capabilities) string {
	if caps == nil {
		return ""
	}
	parts := make([]string, 0, len(caps.GPUs))
	for _, gpu := range caps.GPUs {
		parts = append(parts, fmt.Sprintf("%s:%s:%d:%s:%s:%d:%d",
			gpu.PCIBusID, gpu.Name, gpu.VRAMTotalMB, gpu.Driver, gpu.ComputeCap,
			gpu.PCIGen, gpu.PCILanes))
	}
	sort.Strings(parts)
	parts = append(parts, caps.OS, caps.Arch, caps.CPU.Model, strconv.Itoa(caps.CPU.Cores), strconv.Itoa(caps.CPU.Threads), strconv.Itoa(caps.RAM.TotalMB))
	return specHash(parts...)
}

func specGPUSet(indices []int) string {
	if len(indices) == 0 {
		return "all"
	}
	indices = append([]int(nil), indices...)
	sort.Ints(indices)
	parts := make([]string, len(indices))
	for i, index := range indices {
		parts[i] = strconv.Itoa(index)
	}
	return strings.Join(parts, ",")
}

func NewSpecProfileScope(target *ModelProfile, caps *detect.Capabilities, opts Options, kind, companionPath string) SpecProfileScope {
	ctx := opts.ContextSize
	if ctx <= 0 && target != nil {
		ctx = target.ContextSize
	}
	parallel := opts.Parallel
	if parallel <= 0 {
		parallel = 1
	}
	backend := strings.TrimSpace(opts.BackendIdentity)
	if backend == "" {
		backend = backendCacheTag(opts)
	}
	sampling := strings.TrimSpace(opts.SamplingProfile)
	if sampling == "" {
		sampling = "default"
	}
	return SpecProfileScope{
		TargetIdentity: SpecTargetIdentity(target), CompanionIdentity: SpecCompanionIdentity(companionPath),
		BackendIdentity: backend, HardwareIdentity: SpecHardwareIdentity(caps), GPUSet: specGPUSet(opts.GPUs), Kind: kind,
		ContextSize: ctx, Parallel: parallel, SamplingProfile: sampling, ReasoningOff: opts.ReasoningOff,
		// SpecMode is the mode this profile is proof for: on the spec-test save
		// path kind is the tested --spec flag, and in Auto kind is the candidate
		// path being validated, so both describe the same launch. The literal
		// knob ("auto" vs "mtp") must NOT enter the key or Auto could never
		// consume a profile saved by spec-test.
		SpecMode: normalizeSpecMode(kind), ForceSpecMoE: opts.ForceSpecMoE,
		Threads: opts.Threads, CacheRAMMB: opts.CacheRAMMB,
	}
}

func (s SpecProfileScope) Key() string {
	data, _ := json.Marshal(s)
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:16])
}

func SpecProfilePath(cacheDir string, scope SpecProfileScope) string {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	return filepath.Join(cacheDir, "spec-profiles", "spec-"+scope.Key()+".json")
}

func SaveSpecPerformanceProfile(cacheDir string, profile SpecPerformanceProfile) (string, error) {
	if profile.SchemaVersion == 0 {
		profile.SchemaVersion = SpecProfileSchemaVersion
	}
	profile.ScopeKey = profile.Scope.Key()
	if profile.MeasuredAt == "" {
		profile.MeasuredAt = time.Now().UTC().Format(time.RFC3339)
	}
	path := SpecProfilePath(cacheDir, profile.Scope)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return "", err
	}
	data, err := json.MarshalIndent(profile, "", "  ")
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

func LoadSpecPerformanceProfile(cacheDir string, scope SpecProfileScope) (*SpecPerformanceProfile, error) {
	path := SpecProfilePath(cacheDir, scope)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var profile SpecPerformanceProfile
	if err := json.Unmarshal(data, &profile); err != nil {
		return nil, err
	}
	if profile.SchemaVersion != SpecProfileSchemaVersion || profile.Scope != scope || profile.ScopeKey != scope.Key() {
		return nil, fmt.Errorf("speculative profile scope mismatch")
	}
	return &profile, nil
}

func (p *SpecPerformanceProfile) AutoEligible() (bool, string) {
	if p == nil || !p.Complete {
		return false, "no complete matching performance profile"
	}
	if strings.TrimSpace(p.LaunchIdentity) == "" {
		return false, "profile lacks an exact launch identity"
	}
	if p.DraftMax < 1 || p.DraftMax > 4 {
		return false, "profile has an unsafe MTP ceiling"
	}
	if p.PromptCases < 9 || p.RepeatedRounds < 2 {
		return false, "profile lacks the nine-prompt repeated harness"
	}
	if !p.CorrectnessPassed || !p.StabilityPassed {
		return false, "profile did not pass correctness and stability"
	}
	if p.ImprovementPct < 2.0 || p.SpeculativeTPS <= p.BaselineTPS {
		return false, "profile did not beat baseline above noise"
	}
	if p.WallImprovementPct < 2.0 {
		return false, "profile did not improve end-to-end wall time above noise"
	}
	if p.PromptRegressionPct > 5.0 {
		return false, "profile regressed prompt processing by more than 5%"
	}
	if p.OutputLengthDeltaPct > 10.0 {
		return false, "profile changed mean output length by more than 10%"
	}
	parallel := p.Scope.Parallel
	if parallel < 1 {
		parallel = 1
	}
	if p.Scope.ContextSize/parallel >= 60000 && p.MaxPromptTokens < 60000 {
		return false, "profile lacks a 60k-context request"
	}
	if p.Scope.Parallel > 1 && !p.ParallelLoadPassed {
		return false, "profile lacks parallel load validation"
	}
	return true, ""
}
