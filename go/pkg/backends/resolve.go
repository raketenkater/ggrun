package backends

import (
	"fmt"
	"sort"
	"strings"

	"github.com/raketenkater/ggrun/pkg/gguf"
)

// Requirement is what the catalog says a model needs from a backend. It has
// three states:
//   - requires something: Family is set or Recipe is non-nil;
//   - none needed: Family is empty, Recipe is nil and Unchecked is empty;
//   - unknown: Unchecked is non-empty, because a tensor-type entry for the
//     architecture could not be checked without the model's type histogram.
//
// Family and Recipe may both be set when the model needs both.
type Requirement struct {
	// Family is the generic backend family a name rule requires for the
	// architecture ("" when none); it is exactly RequiredBackendForArch.
	Family string
	// Recipe is the catalog entry whose required tensor encodings the model
	// uses (nil when none matched). Several entries may match; the first in
	// catalog order is returned.
	Recipe *Recipe
	// Unchecked names catalog entries for the architecture that require tensor
	// encodings which could not be checked because the model's type histogram
	// is unknown.
	Unchecked []string
	// Reason says which rule matched, or why nothing did.
	Reason string
}

// None reports that nothing is required and nothing is unknown. It is false
// while Unchecked is non-empty: an unchecked rule is not "no requirement".
func (r Requirement) None() bool { return r.Family == "" && r.Recipe == nil && len(r.Unchecked) == 0 }

// ResolveRequirement applies the catalog to a model's architecture and tensor
// type histogram (gguf.Info.TensorTypes or gguf.ReadTypeHistogramModel; nil
// means unknown).
//
// Rules, in order:
//  1. A name rule for the architecture sets Family. With no histogram this is
//     the whole answer and equals RequiredBackendForArch.
//  2. A catalog entry whose route_arch is the architecture and whose every
//     required tensor encoding is used by the model (the type id present, and
//     the entry's signature, if any, satisfied) sets Recipe. An entry without
//     required encodings is never a requirement: it is an option, not a need.
//
// Type ids are compared as numbers only; what an id means in a backend is the
// catalog entry's claim, checked per backend by the capability probe.
func ResolveRequirement(arch string, types []gguf.TypeUsage) Requirement {
	return loadedCatalog.resolve(arch, types)
}

func (c *parsedCatalog) resolve(arch string, types []gguf.TypeUsage) Requirement {
	a := strings.ToLower(strings.TrimSpace(arch))
	var req Requirement
	var why []string
	if c == nil {
		req.Reason = "no special requirement: the backend catalog did not load"
		return req
	}
	if fam := c.requiredBackend(a); fam != "" {
		req.Family = fam
		why = append(why, fmt.Sprintf("architecture %q matches a name rule that requires the %s family", a, fam))
	}
	for i, reqs := range c.encodings {
		if a == "" || len(reqs) == 0 || i >= len(c.recipes) {
			continue
		}
		rec := c.recipes[i]
		if !strings.EqualFold(strings.TrimSpace(rec.RouteArch), a) {
			continue
		}
		if len(types) == 0 {
			req.Unchecked = append(req.Unchecked, rec.Name)
			continue
		}
		if req.Recipe != nil {
			continue
		}
		if detail, ok := encodingsUsed(reqs, types); ok {
			picked := rec
			req.Recipe = &picked
			why = append(why, fmt.Sprintf("catalog entry %q for %q: %s", rec.Name, a, detail))
		}
	}
	switch {
	case len(why) > 0:
		req.Reason = strings.Join(why, "; ")
	case len(req.Unchecked) > 0:
		req.Reason = fmt.Sprintf("no special requirement found for %q, but catalog entries %s depend on tensor types and the model's are unknown",
			a, strings.Join(req.Unchecked, ", "))
	default:
		req.Reason = fmt.Sprintf("no special requirement: architecture %q matches no name rule or tensor-type entry", a)
	}
	return req
}

// encodingsUsed reports whether every requirement is met by the histogram.
func encodingsUsed(reqs []EncodingRequirement, types []gguf.TypeUsage) (string, bool) {
	var parts []string
	for _, r := range reqs {
		u, ok := gguf.TypeUsageMatching(types, r.ID)
		if !ok {
			return "", false
		}
		lo := r.TensorsMin
		if lo < 1 {
			lo = 1
		}
		if u.Tensors < lo || (r.TensorsMax > 0 && u.Tensors > r.TensorsMax) {
			return "", false
		}
		if r.NameContains != "" && !anyContains(u.Names, r.NameContains) {
			return "", false
		}
		parts = append(parts, fmt.Sprintf("tensor type %d (%s) is used by %d tensors", r.ID, r.Name, u.Tensors))
	}
	return strings.Join(parts, ", "), true
}

func anyContains(names []string, sub string) bool {
	for _, n := range names {
		if strings.Contains(n, sub) {
			return true
		}
	}
	return false
}

// EncodingNames lists the distinct encoding names the catalog's entries
// require, sorted: the literals ProbeBackendEncodings looks for.
func EncodingNames() []string { return loadedCatalog.encodingNames() }

func (c *parsedCatalog) encodingNames() []string {
	if c == nil {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, reqs := range c.encodings {
		for _, r := range reqs {
			if !seen[r.Name] {
				seen[r.Name] = true
				out = append(out, r.Name)
			}
		}
	}
	sort.Strings(out)
	return out
}
