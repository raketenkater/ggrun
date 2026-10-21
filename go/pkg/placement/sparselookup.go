package placement

import (
	"fmt"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Sparse lookup tables are a third host-memory tier, separate from resident
// execution weights and from non-reclaimable runtime (KV, scratch, CRAM,
// checkpoints, staging). Three facts are kept apart:
//
//  1. Tensor role. An architecture adapter in the GGUF parser marks a table as
//     a sparse row lookup from the architecture's own metadata (the first one
//     is DeepSeek-V4.1 Engram, `<arch>.engram.layer_ids`). Its bytes, kind and
//     per-layer split travel in ModelProfile whether or not any backend can
//     read it lazily. A tensor name or the word "ngram" is never proof.
//  2. Loader capability. Only a reviewed loader contract for the exact backend
//     build (see the command package) makes SparseLookupCapability
//     file-backed. A help line advertising a lazy flag is not such a contract.
//  3. Policy, chosen here from that capability and the hardware:
//     - layer: ordinary tensors that follow their layer onto its device. This
//     is how a loader without lazy reads treats them and the only policy
//     without a reviewed contract. The bytes are charged per layer.
//     - resident: lazily mapped, with the whole tier charged to host RAM, so
//     its page cache is never squeezed by the rest of the plan.
//     - paged: lazily mapped and outside the GPU and resident host ledgers;
//     the running scope gives its page cache a bounded allowance. Storage
//     latency then bounds lookup speed, so it is chosen only when the
//     resident tier does not fit or leaves less agent context.
//     Automatic expert paging beside a paged tier is refused: both would share
//     one reclaim pool, so lookup traffic could evict execution weights.

// SparseLookupCapability is the selected backend's verified loader behavior
// for this model's sparse lookup tables. Unknown is the conservative default.
type SparseLookupCapability string

const (
	SparseLookupUnknown    SparseLookupCapability = "unknown"
	SparseLookupFileBacked SparseLookupCapability = "file-backed"
	SparseLookupAnonymous  SparseLookupCapability = "anonymous"
)

// SparseLookupRange is one lookup table's byte range inside a shard file.
type SparseLookupRange struct {
	Shard  string `json:"shard"` // shard file basename
	Offset int64  `json:"offset"`
	Bytes  int64  `json:"bytes"`
}

// SparseLookupPolicy is how a plan holds the tier.
type SparseLookupPolicy string

const (
	SparseLookupLayer    SparseLookupPolicy = "layer"
	SparseLookupResident SparseLookupPolicy = "resident"
	SparseLookupPaged    SparseLookupPolicy = "paged"
)

// SparseLookupLazyFlag is the reviewed backend option that switches lazy
// lookup reads on or off.
const SparseLookupLazyFlag = "--lazy-mode"

// NormalizeSparseLookupCapability validates an external capability value.
func NormalizeSparseLookupCapability(value string) SparseLookupCapability {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case string(SparseLookupFileBacked):
		return SparseLookupFileBacked
	case string(SparseLookupAnonymous):
		return SparseLookupAnonymous
	default:
		return SparseLookupUnknown
	}
}

// NormalizeSparseLookupPolicy validates an external policy value; anything
// else is "" (no policy).
func NormalizeSparseLookupPolicy(value string) SparseLookupPolicy {
	switch p := SparseLookupPolicy(strings.ToLower(strings.TrimSpace(value))); p {
	case SparseLookupLayer, SparseLookupResident, SparseLookupPaged:
		return p
	}
	return ""
}

func sparseLookupKind(model *ModelProfile) string {
	if model != nil && model.SparseLookupKind != "" {
		return model.SparseLookupKind + " lookup"
	}
	return "sparse lookup"
}

func sparseLookupFlag(opts Options) string {
	if f := strings.TrimSpace(opts.SparseLookupFlag); f != "" {
		return f
	}
	return SparseLookupLazyFlag
}

func sparseLookupBackendName(opts Options) string {
	for _, v := range []string{opts.BackendIdentity, opts.BackendTag} {
		if v = strings.TrimSpace(v); v != "" {
			return v
		}
	}
	return "the selected backend"
}

// sparseLookupCandidates lists the policies Compute may try, in preference
// order, or an actionable error when the user's explicit lazy choice cannot be
// honored with verified accounting.
func sparseLookupCandidates(model *ModelProfile, opts Options) ([]SparseLookupPolicy, error) {
	lazy := NormalizeSparseLookupCapability(string(opts.SparseLookupCapability)) == SparseLookupFileBacked
	switch strings.ToLower(strings.TrimSpace(opts.SparseLookupLazy)) {
	case "off":
		return []SparseLookupPolicy{SparseLookupLayer}, nil
	case "on":
		if !lazy {
			evidence := strings.TrimSpace(opts.SparseLookupEvidence)
			if evidence != "" {
				evidence = " (" + evidence + ")"
			}
			return nil, fmt.Errorf(
				"%s on was requested, but %s has no reviewed lazy-loader contract for %s tables%s, so ggrun cannot account where their %d MiB live. Remove %s on to load them with their layers",
				sparseLookupFlag(opts), sparseLookupBackendName(opts), model.SparseLookupKind, evidence,
				bytesToMiBCeil(model.SparseLookupBytes), sparseLookupFlag(opts))
		}
	}
	if lazy {
		return []SparseLookupPolicy{SparseLookupResident, SparseLookupPaged}, nil
	}
	return []SparseLookupPolicy{SparseLookupLayer}, nil
}

// sparseLookupModelView returns the profile placement sizes for a policy.
// Under "layer" the tables are ordinary per-layer non-expert bytes; under the
// lazy policies they leave the ordinary weight totals and are accounted
// explicitly. Only the copy changes.
func sparseLookupModelView(model *ModelProfile, policy SparseLookupPolicy) *ModelProfile {
	view := *model
	if policy == SparseLookupLayer {
		view.NonExpertBytes += model.SparseLookupBytes
		if len(model.NonExpertLayerBytes) > 0 {
			layers := append([]int64(nil), model.NonExpertLayerBytes...)
			for i, b := range model.SparseLookupLayerBytes {
				if i < len(layers) {
					layers[i] += b
				}
			}
			view.NonExpertLayerBytes = layers
		}
		return &view
	}
	removed := model.SparseLookupBytes
	if removed > view.SizeBytes {
		removed = view.SizeBytes
	}
	view.SizeBytes -= removed
	view.sparseLookupViewBytes += removed
	if view.TotalSizeMB > 0 {
		if view.TotalSizeMB -= bytesToMiBFloor(removed); view.TotalSizeMB < 1 {
			view.TotalSizeMB = 1
		}
	}
	return &view
}

// strategyAccounting returns the model view and options a resolved strategy
// was planned with. Exported entry points that take a finished strategy
// (runtime refresh, resource ledger) receive the full profile and request
// options, while Compute planned a lookup-tier model on its policy's view and
// read policy-scoped evidence. Options that already carry a policy are
// Compute's own and are returned unchanged, so the view is never applied twice.
func strategyAccounting(model *ModelProfile, s *Strategy, opts Options) (*ModelProfile, Options) {
	if model == nil || s == nil || s.SparseLookupPolicy == "" || opts.sparseLookupPolicy != "" {
		return model, opts
	}
	opts.sparseLookupPolicy = s.SparseLookupPolicy
	if model.SparseLookupBytes > 0 {
		model = sparseLookupModelView(model, s.SparseLookupPolicy)
	}
	return model, opts
}

// ArtifactSizeBytes is the model file's size, unchanged by a sparse lookup
// accounting view. Evidence keys use it, so a measurement recorded against
// the full profile is found again by a policy's planning pass.
func (m *ModelProfile) ArtifactSizeBytes() int64 {
	return m.SizeBytes + m.sparseLookupViewBytes
}

// ArtifactTotalSizeMB is TotalSizeMB before any sparse lookup view.
func (m *ModelProfile) ArtifactTotalSizeMB() int {
	if m.sparseLookupViewBytes <= 0 || m.TotalSizeMB <= 0 {
		return m.TotalSizeMB
	}
	return m.TotalSizeMB + bytesToMiBFloor(m.sparseLookupViewBytes)
}

// SparseLookupScopedTag scopes evidence to the lookup-tier policy a plan
// uses. Each policy changes host allocation, so their probe, runtime and
// prompt-cache measurements are kept apart. Placement readers and command-side
// writers both call this, so the keys meet. Idempotent; "" leaves the tag
// unchanged (models without the tier keep their namespace).
func SparseLookupScopedTag(tag string, policy SparseLookupPolicy) string {
	if policy == "" {
		return tag
	}
	suffix := "|lookup=" + string(policy)
	if strings.HasSuffix(tag, suffix) {
		return tag
	}
	return tag + suffix
}

// syncMeasuredModelFields carries measurements a compute pass recorded on a
// view back to the caller's profile.
func syncMeasuredModelFields(dst, src *ModelProfile) {
	dst.MeasuredKVBytesPerTok = src.MeasuredKVBytesPerTok
	dst.MeasuredKVGeometry = src.MeasuredKVGeometry
	dst.MeasuredComputeCoefficient = src.MeasuredComputeCoefficient
}

// computeSparseLookup resolves the tier policy for a table-bearing model.
// Candidates are tried in preference order. A plan that also needs automatic
// expert paging wins only when no candidate keeps execution weights resident,
// and a resident tier yields to a paged one only for more agent context.
func computeSparseLookup(caps *detect.Capabilities, model *ModelProfile, opts Options) (*Strategy, error) {
	candidates, err := sparseLookupCandidates(model, opts)
	if err != nil {
		return nil, err
	}
	var best *Strategy
	var errs []string
	for _, policy := range candidates {
		view := sparseLookupModelView(model, policy)
		o := opts
		o.sparseLookupPolicy = policy
		s, err := Compute(caps, view, o)
		syncMeasuredModelFields(model, view)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s tier: %v", policy, err))
			continue
		}
		if best == nil || sparseLookupPlanBetter(s, best) {
			best = s
		}
	}
	if best != nil {
		return best, nil
	}
	hint := ""
	if len(candidates) == 1 && candidates[0] == SparseLookupLayer {
		why := strings.TrimSpace(opts.SparseLookupEvidence)
		if strings.EqualFold(strings.TrimSpace(opts.SparseLookupLazy), "off") {
			why = "lazy reads were turned off"
		}
		if why != "" {
			why = " (" + why + ")"
		}
		hint = fmt.Sprintf(". The tables load with their layers%s; a backend with a reviewed lazy-loader contract can hold them in host RAM or page them from storage", why)
	}
	return nil, fmt.Errorf("model has %d MiB of %s tables and no placement fits: %s%s",
		bytesToMiBCeil(model.SparseLookupBytes), sparseLookupKind(model), strings.TrimRight(strings.Join(errs, "; "), ". \n"), hint)
}

// sparseLookupPlanBetter ranks candidate a against the current best b:
// resident execution weights first, then useful context. Ties keep b, the
// earlier (preferred) candidate.
func sparseLookupPlanBetter(a, b *Strategy) bool {
	if a.MMapRequired != b.MMapRequired {
		return !a.MMapRequired
	}
	return a.ContextSize > b.ContextSize
}

// sparseLookupResidentCaps charges a resident lazy tier to host RAM before
// the rest of the plan sees the inventory.
func sparseLookupResidentCaps(caps *detect.Capabilities, model *ModelProfile, opts Options) (*detect.Capabilities, error) {
	if opts.sparseLookupPolicy != SparseLookupResident || model == nil || model.SparseLookupBytes <= 0 {
		return caps, nil
	}
	tierMB := bytesToMiBCeil(model.SparseLookupBytes)
	if caps == nil || caps.RAM.FreeMB <= tierMB {
		free := 0
		if caps != nil {
			free = caps.RAM.FreeMB
		}
		return nil, fmt.Errorf("a resident %s tier needs %d MiB of host RAM; %d MiB is available", sparseLookupKind(model), tierMB, free)
	}
	return detect.ApplyRAMHeadroom(caps, tierMB), nil
}

// applySparseLookupTier records the planned tier on a strategy. A resident
// tier joins the planned host footprint exactly once; a paged tier is never
// part of PlannedHostFootprintMB or a GPU ledger; a layer tier is already
// inside the per-layer charges.
func applySparseLookupTier(s *Strategy, model *ModelProfile, opts Options) {
	if s == nil || model == nil || model.SparseLookupBytes <= 0 || opts.sparseLookupPolicy == "" {
		return
	}
	tierMB := bytesToMiBCeil(model.SparseLookupBytes)
	already := s.SparseLookupPolicy == opts.sparseLookupPolicy && s.SparseLookupMB == tierMB
	s.SparseLookupMB = tierMB
	s.SparseLookupBytes = model.SparseLookupBytes
	s.SparseLookupPolicy = opts.sparseLookupPolicy
	s.SparseLookupCapability = NormalizeSparseLookupCapability(string(opts.SparseLookupCapability))
	s.SparseLookupFlag = strings.TrimSpace(opts.SparseLookupFlag)
	if s.SparseLookupFlag == "" && s.SparseLookupPolicy != SparseLookupLayer {
		// A lazy policy exists only under a reviewed contract, which names
		// the flag; never leave the backend's default in charge.
		s.SparseLookupFlag = SparseLookupLazyFlag
	}
	s.SparseLookupEvidence = opts.SparseLookupEvidence
	s.SparseLookupRanges = append([]SparseLookupRange(nil), model.SparseLookupRanges...)
	if !already && s.SparseLookupPolicy == SparseLookupResident && s.PlannedHostFootprintMB > 0 {
		// A zero footprint means "not derived" (the whole-host ceiling then
		// applies), so it is not turned into a tier-sized floor.
		s.PlannedHostFootprintMB += tierMB
	}
}

// StrategyUsesLazySparseLookup reports a plan whose lookup tables are lazily
// mapped from the model file (resident or paged).
func StrategyUsesLazySparseLookup(s *Strategy) bool {
	return s != nil && s.SparseLookupMB > 0 &&
		(s.SparseLookupPolicy == SparseLookupResident || s.SparseLookupPolicy == SparseLookupPaged)
}

// StrategyUsesFileBackedSparseLookup reports a plan that relies on a paged
// lookup tier outside its resident ledgers. It says nothing about CPU experts.
func StrategyUsesFileBackedSparseLookup(s *Strategy) bool {
	return s != nil && s.SparseLookupMB > 0 && s.SparseLookupPolicy == SparseLookupPaged
}

// sparseLookupArgs states the lazy-read choice explicitly: on for the lazy
// policies (which require the reviewed contract), off for the layer policy
// whenever the backend advertises the flag, so its own default cannot move
// tables the plan charged to a layer.
func sparseLookupArgs(s *Strategy) []string {
	if s == nil || s.SparseLookupMB <= 0 || s.SparseLookupFlag == "" {
		return nil
	}
	switch s.SparseLookupPolicy {
	case SparseLookupResident, SparseLookupPaged:
		return []string{s.SparseLookupFlag, "on"}
	case SparseLookupLayer:
		return []string{s.SparseLookupFlag, "off"}
	}
	return nil
}

// sparseLookupExpertPagingError refuses automatic ordinary expert paging
// beside a paged lookup tier. Both would share one reclaim pool, so lookup
// traffic could evict execution weights and turn every token into expert
// re-faults. Explicit --mmap remains the user's last-resort override.
func sparseLookupExpertPagingError(model *ModelProfile, opts Options, ramNeededMB, ramAvailMB int) error {
	if model == nil || model.SparseLookupBytes <= 0 || opts.ForceMMap || opts.sparseLookupPolicy != SparseLookupPaged {
		return nil
	}
	return fmt.Errorf(
		"execution weights need %d MiB resident RAM but %d MiB is available; the %d MiB %s tier already pages from disk, so ggrun will not also page experts automatically (that thrashes). Reduce context or use more GPU memory; --mmap accepts expert paging as a last resort",
		ramNeededMB, ramAvailMB, bytesToMiBCeil(model.SparseLookupBytes), sparseLookupKind(model))
}

// sparseLookupRecordMatches reports whether a saved verified config holds the
// lookup tier the way this candidate pass plans it. A record from another
// policy (or from before policies existed) is not evidence for this one.
func sparseLookupRecordMatches(restored *Strategy, model *ModelProfile, opts Options) bool {
	if restored == nil || model == nil || model.SparseLookupBytes <= 0 {
		return true
	}
	return restored.SparseLookupPolicy == opts.sparseLookupPolicy &&
		restored.SparseLookupMB == bytesToMiBCeil(model.SparseLookupBytes)
}
