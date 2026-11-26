package backends

import (
	"bytes"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"strings"
	"time"
)

// The catalog holds the reviewed fork knowledge as data: the recipes behind
// `ggrun support install --with-backend` and the generic backend-family rules.
// It is embedded, parsed and validated once at package initialisation.
//
// Failure behaviour: the embedded file is checked by TestEmbeddedCatalogLoads,
// so a load failure is a programming error that CI rejects before release. At
// runtime a failure never panics a user launch: the built-in recipe list and
// the name rules come up empty and the error stays in catalogLoadErr.

//go:embed catalog.json
var catalogJSON []byte

//go:embed patches/*/*.patch
var catalogPatchFS embed.FS

// catalogVersion is the only schema version this build understands.
const catalogVersion = 1

// Name-rule match kinds.
const matchArchPrefix = "arch_prefix"

// Recipe kinds a catalog entry may declare (optional).
var catalogRecipeKinds = map[string]bool{"fork": true, "pr": true, "mainline": true}

// catalogPatch references an embedded patch file by path and pins its bytes.
type catalogPatch struct {
	Name   string `json:"name"`   // identity recorded in backends.json (RecipePatch.Name)
	File   string `json:"file"`   // path under patches/
	SHA256 string `json:"sha256"` // hex digest of the file
}

// catalogRecipe is one recipe row. The first block maps one-to-one onto
// Recipe. The optional block is schema reserved for later entries and is not
// read by any selection logic yet.
type catalogRecipe struct {
	Name        string         `json:"name"`
	Note        string         `json:"note,omitempty"`
	Description string         `json:"description"`
	Tag         string         `json:"tag"`
	GitURL      string         `json:"git_url"`
	Branch      string         `json:"branch"`
	Commit      string         `json:"commit"`
	RouteArch   string         `json:"route_arch"`
	HelperOnly  bool           `json:"helper_only,omitempty"`
	Accel       string         `json:"accel,omitempty"`
	Patches     []catalogPatch `json:"patches,omitempty"`

	Kind                string   `json:"kind,omitempty"`                  // fork | pr | mainline
	RequiredTensorTypes []string `json:"required_tensor_types,omitempty"` // tensor type names the build must understand
	TensorCount         int      `json:"tensor_count,omitempty"`          // expected tensor count of the model it serves
	SupersededBy        string   `json:"superseded_by,omitempty"`         // e.g. a mainline revision that replaces it
	Evidence            []string `json:"evidence,omitempty"`              // links or ledger references
	Expires             string   `json:"expires,omitempty"`               // YYYY-MM-DD, review-by date
}

// catalogNameRule requires a generic backend family for architectures whose
// name matches. Rules apply in order; the first match wins.
type catalogNameRule struct {
	Match    string `json:"match"`    // match kind; only "arch_prefix"
	Value    string `json:"value"`    // lower-case, trimmed
	Requires string `json:"requires"` // backend family, e.g. "ik_llama"
	Note     string `json:"note,omitempty"`
}

type catalogFile struct {
	Version   int               `json:"version"`
	Comment   string            `json:"comment,omitempty"`
	Recipes   []catalogRecipe   `json:"recipes"`
	NameRules []catalogNameRule `json:"name_rules"`
}

// parsedCatalog is the validated, ready-to-use form.
type parsedCatalog struct {
	recipes   []Recipe
	nameRules []catalogNameRule
}

var (
	loadedCatalog, catalogLoadErr = loadCatalog(catalogJSON, catalogPatchFS)
	builtinRecipes                = catalogRecipes(loadedCatalog)
)

func catalogRecipes(c *parsedCatalog) []Recipe {
	if c == nil {
		return nil
	}
	return c.recipes
}

// loadCatalog parses and validates catalog data, reading patch bytes from
// patches. Every problem found is reported, joined into one error.
func loadCatalog(data []byte, patches fs.FS) (*parsedCatalog, error) {
	var file catalogFile
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&file); err != nil {
		return nil, fmt.Errorf("parse backend catalog: %w", err)
	}
	if file.Version != catalogVersion {
		return nil, fmt.Errorf("backend catalog version %d is not supported (want %d)", file.Version, catalogVersion)
	}

	var errs []error
	fail := func(format string, args ...any) { errs = append(errs, fmt.Errorf(format, args...)) }

	out := &parsedCatalog{}
	names := map[string]string{} // lower-case name or tag -> owning recipe name
	for i, r := range file.Recipes {
		label := fmt.Sprintf("recipe #%d (%q)", i+1, r.Name)
		for _, f := range []struct{ field, value string }{
			{"name", r.Name}, {"description", r.Description}, {"tag", r.Tag}, {"git_url", r.GitURL},
			{"branch", r.Branch}, {"commit", r.Commit}, {"route_arch", r.RouteArch},
		} {
			if strings.TrimSpace(f.value) == "" {
				fail("%s: missing %s", label, f.field)
			}
		}
		if r.Commit != "" && !isHexDigest(r.Commit, 40) {
			fail("%s: commit %q is not a 40-character hex id", label, r.Commit)
		}
		for _, key := range uniqueStrings(r.Name, r.Tag) {
			lower := strings.ToLower(strings.TrimSpace(key))
			if lower == "" {
				continue
			}
			if owner, dup := names[lower]; dup {
				fail("%s: duplicate id %q (already used by recipe %q)", label, key, owner)
			}
			names[lower] = r.Name
		}
		if r.Kind != "" && !catalogRecipeKinds[r.Kind] {
			fail("%s: unknown kind %q", label, r.Kind)
		}
		if r.Expires != "" {
			if _, err := time.Parse("2006-01-02", r.Expires); err != nil {
				fail("%s: expires %q is not YYYY-MM-DD", label, r.Expires)
			}
		}
		if r.TensorCount < 0 {
			fail("%s: negative tensor_count", label)
		}

		recipe := Recipe{
			Name: r.Name, Description: r.Description, Tag: r.Tag, GitURL: r.GitURL,
			Branch: r.Branch, Commit: r.Commit, RouteArch: r.RouteArch,
			HelperOnly: r.HelperOnly, Accel: r.Accel,
		}
		seenPatch := map[string]bool{}
		for _, p := range r.Patches {
			plabel := fmt.Sprintf("%s patch %q", label, p.Name)
			if p.Name == "" || p.File == "" {
				fail("%s: missing name or file", plabel)
				continue
			}
			if seenPatch[p.Name] {
				fail("%s: duplicate patch name", plabel)
			}
			seenPatch[p.Name] = true
			body, err := fs.ReadFile(patches, "patches/"+p.File)
			if err != nil {
				fail("%s: read %s: %v", plabel, p.File, err)
				continue
			}
			if len(body) == 0 {
				fail("%s: %s is empty", plabel, p.File)
				continue
			}
			sum := sha256.Sum256(body)
			if got := hex.EncodeToString(sum[:]); got != strings.ToLower(p.SHA256) {
				fail("%s: sha256 of %s is %s, catalog pins %q", plabel, p.File, got, p.SHA256)
				continue
			}
			recipe.Patches = append(recipe.Patches, RecipePatch{Name: p.Name, contents: body})
		}
		out.recipes = append(out.recipes, recipe)
	}

	seenRule := map[string]bool{}
	for i, rule := range file.NameRules {
		label := fmt.Sprintf("name rule #%d", i+1)
		switch rule.Match {
		case matchArchPrefix:
		default:
			fail("%s: unknown match kind %q", label, rule.Match)
			continue
		}
		if rule.Value == "" || rule.Value != strings.ToLower(strings.TrimSpace(rule.Value)) {
			fail("%s: value %q must be non-empty, lower-case and trimmed", label, rule.Value)
		}
		if strings.TrimSpace(rule.Requires) == "" {
			fail("%s: missing requires", label)
		}
		key := rule.Match + "\x00" + rule.Value
		if seenRule[key] {
			fail("%s: duplicate %s %q", label, rule.Match, rule.Value)
		}
		seenRule[key] = true
		out.nameRules = append(out.nameRules, rule)
	}

	if err := errors.Join(errs...); err != nil {
		return nil, fmt.Errorf("invalid backend catalog: %w", err)
	}
	return out, nil
}

func isHexDigest(s string, n int) bool {
	if len(s) != n {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

func uniqueStrings(a, b string) []string {
	if strings.EqualFold(a, b) {
		return []string{a}
	}
	return []string{a, b}
}

// requiredBackend applies the name rules in order.
func (c *parsedCatalog) requiredBackend(arch string) string {
	if c == nil {
		return ""
	}
	a := strings.ToLower(strings.TrimSpace(arch))
	for _, rule := range c.nameRules {
		switch rule.Match {
		case matchArchPrefix:
			if strings.HasPrefix(a, rule.Value) {
				return rule.Requires
			}
		}
	}
	return ""
}
