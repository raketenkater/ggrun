package placement

import (
	"fmt"
	"strings"
)

// Sparse lookup tables are a third host-memory tier, separate from resident
// execution weights and from non-reclaimable runtime (KV, scratch, CRAM,
// checkpoints, staging).
//
// The first member is the DeepSeek-V4.1 Engram table: two ~97 GiB tensors from
// which the backend gathers a few hundred bytes per token. The reviewed loader
// (smalinin/llama.cpp 425badfe) creates them with TENSOR_READ_LAZY: with
// --lazy-mode on they live in a CPU buffer backed by the model-file mapping
// whatever --load-mode says, receive MADV_RANDOM, are excluded from
// MAP_POPULATE and from mlock, and never follow their layer onto a GPU. With
// lazy mode off they follow the layer like any other tensor.
//
// Placement therefore treats the tier as file-backed only when the exact
// backend advertises lazy reads; it then never charges those bytes as GPU
// backbone or as resident host RAM. Without that capability a plan is refused
// rather than silently charging ~195 GiB to a GPU or to anonymous memory.

// SparseLookupCapability is the selected backend's loader behavior for sparse
// lookup tables. Unknown is the fail-closed default.
type SparseLookupCapability string

const (
	SparseLookupUnknown    SparseLookupCapability = "unknown"
	SparseLookupFileBacked SparseLookupCapability = "file-backed"
	SparseLookupAnonymous  SparseLookupCapability = "anonymous"
)

// SparseLookupLazyFlag is the backend option that keeps marked lookup tables
// on demand. ggrun always states it explicitly so a changed backend default
// cannot move the tier.
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

// sparseLookupTierMB returns the model's sparse lookup tier in MiB, or an
// error when the model has one and the exact backend cannot keep it
// file-backed.
func sparseLookupTierMB(model *ModelProfile, opts Options) (int, error) {
	if model == nil || model.SparseLookupBytes <= 0 {
		return 0, nil
	}
	mb := bytesToMiBCeil(model.SparseLookupBytes)
	if NormalizeSparseLookupCapability(string(opts.SparseLookupCapability)) == SparseLookupFileBacked {
		return mb, nil
	}
	identity := strings.TrimSpace(opts.BackendIdentity)
	if identity == "" {
		identity = strings.TrimSpace(opts.BackendTag)
	}
	if identity == "" {
		identity = "the selected backend"
	}
	kind := model.SparseLookupKind
	if kind == "" {
		kind = "sparse lookup"
	}
	evidence := strings.TrimSpace(opts.SparseLookupEvidence)
	if evidence != "" {
		evidence = " (" + evidence + ")"
	}
	return 0, fmt.Errorf(
		"model has %d MiB of %s tables that must stay file-backed, but %s cannot read them lazily%s; loading them would need that much more resident RAM or VRAM. Use a backend that supports %s",
		mb, kind, identity, evidence, SparseLookupLazyFlag)
}

// applySparseLookupTier records the planned tier on a strategy. The bytes are
// never part of PlannedHostFootprintMB or a GPU ledger.
func applySparseLookupTier(s *Strategy, model *ModelProfile, opts Options) {
	if s == nil || model == nil || model.SparseLookupBytes <= 0 {
		return
	}
	s.SparseLookupMB = bytesToMiBCeil(model.SparseLookupBytes)
	s.SparseLookupCapability = SparseLookupFileBacked
	s.SparseLookupEvidence = opts.SparseLookupEvidence
}

// StrategyUsesFileBackedSparseLookup reports whether a plan relies on a
// file-backed sparse lookup tier. It says nothing about CPU experts.
func StrategyUsesFileBackedSparseLookup(s *Strategy) bool {
	return s != nil && s.SparseLookupMB > 0 &&
		NormalizeSparseLookupCapability(string(s.SparseLookupCapability)) == SparseLookupFileBacked
}

// sparseLookupExpertPagingError refuses automatic ordinary expert paging when
// a sparse lookup tier already pages from the same storage. Both would share
// one reclaim pool, so lookup traffic could evict execution weights and turn
// every token into expert re-faults. Explicit --mmap remains the user's
// last-resort override.
func sparseLookupExpertPagingError(model *ModelProfile, opts Options, ramNeededMB, ramAvailMB int) error {
	if model == nil || model.SparseLookupBytes <= 0 || opts.ForceMMap {
		return nil
	}
	return fmt.Errorf(
		"execution weights need %d MiB resident RAM but %d MiB is available; the %d MiB %s tier already pages from disk, so ggrun will not also page experts automatically (that thrashes). Reduce context or use more GPU memory; --mmap accepts expert paging as a last resort",
		ramNeededMB, ramAvailMB, bytesToMiBCeil(model.SparseLookupBytes), sparseLookupKind(model))
}

func sparseLookupKind(model *ModelProfile) string {
	if model != nil && model.SparseLookupKind != "" {
		return model.SparseLookupKind + " lookup"
	}
	return "sparse lookup"
}
