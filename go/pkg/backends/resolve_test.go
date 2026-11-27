package backends

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/gguf"
)

// encRecipe is a recipe row for route arch arch; extra is appended JSON fields.
func encRecipe(name, arch, extra string) string {
	return fmt.Sprintf(`{"name":%q,"description":"d","tag":%q,"git_url":"u","branch":"b","commit":%q,"route_arch":%q%s}`,
		name, name, testGoodCommit, arch, extra)
}

func mustCatalog(t *testing.T, recipes, rules string) *parsedCatalog {
	t.Helper()
	c, err := loadCatalog([]byte(testCatalogJSON(1, recipes, rules)), testPatchFS())
	if err != nil {
		t.Fatalf("catalog rejected: %v", err)
	}
	return c
}

func hist(entries ...gguf.TypeUsage) []gguf.TypeUsage { return entries }

// A synthetic entry for a type whose id another backend uses for something
// else: the id is claimed only for this architecture, with a signature.
const stqLike = `,"required_tensor_types":[{"id":44,"name":"STQ1_0","tensors_min":10,"tensors_max":100,"name_contains":"ffn_"}]`

func TestResolveRequirementTable(t *testing.T) {
	c := mustCatalog(t,
		encRecipe("needs-stq", "hy_v4", stqLike)+","+
			encRecipe("plain-option", "hy_v4", "")+","+
			encRecipe("second-match", "hy_v4", `,"required_tensor_types":[{"id":44,"name":"STQ1_0"}]`)+","+
			encRecipe("two-types", "multi", `,"required_tensor_types":[{"id":50,"name":"AAA"},{"id":51,"name":"BBB"}]`)+","+
			encRecipe("also-ruled", "minimax-m9", `,"required_tensor_types":[{"id":60,"name":"CCC"}]`),
		`{"match":"arch_prefix","value":"minimax-m","requires":"ik_llama"}`)

	stq58 := gguf.TypeUsage{Type: 44, Tensors: 58, Names: []string{"blk.*.ffn_up_exps.weight"}}
	cases := []struct {
		name      string
		arch      string
		types     []gguf.TypeUsage
		family    string
		recipe    string
		unchecked []string
		reason    string
	}{
		{"matches id and signature", "hy_v4", hist(stq58, gguf.TypeUsage{Type: 8, Tensors: 3}), "", "needs-stq", nil, "tensor type 44 (STQ1_0) is used by 58 tensors"},
		{"architecture is case and space insensitive", " HY_V4 ", hist(stq58), "", "needs-stq", nil, `catalog entry "needs-stq"`},
		{"id present, name signature fails", "hy_v4", hist(gguf.TypeUsage{Type: 44, Tensors: 58, Names: []string{"blk.*.attn_q.weight"}}), "", "second-match", nil, `"second-match"`},
		{"id present, too few tensors for the first entry", "hy_v4", hist(gguf.TypeUsage{Type: 44, Tensors: 3, Names: []string{"blk.*.ffn_up.weight"}}), "", "second-match", nil, `"second-match"`},
		{"id present, too many tensors for the first entry", "hy_v4", hist(gguf.TypeUsage{Type: 44, Tensors: 101, Names: []string{"blk.*.ffn_up.weight"}}), "", "second-match", nil, `"second-match"`},
		{"id absent", "hy_v4", hist(gguf.TypeUsage{Type: 8, Tensors: 3}), "", "", nil, "no special requirement"},
		{"another architecture never matches", "llama", hist(stq58), "", "", nil, "no special requirement"},
		{"unknown histogram is reported, not hidden", "hy_v4", nil, "", "", []string{"needs-stq", "second-match"}, "unknown"},
		{"empty histogram is unknown too", "hy_v4", []gguf.TypeUsage{}, "", "", []string{"needs-stq", "second-match"}, "unknown"},
		{"every listed type is required", "multi", hist(gguf.TypeUsage{Type: 50, Tensors: 1}), "", "", nil, "no special requirement"},
		{"all listed types present", "multi", hist(gguf.TypeUsage{Type: 50, Tensors: 1}, gguf.TypeUsage{Type: 51, Tensors: 2}), "", "two-types", nil, "tensor type 51 (BBB)"},
		{"name rule and tensor entry both apply", "minimax-m9", hist(gguf.TypeUsage{Type: 60, Tensors: 1}), "ik_llama", "also-ruled", nil, "requires the ik_llama family"},
		{"name rule alone without a histogram", "minimax-m9", nil, "ik_llama", "", []string{"also-ruled"}, "requires the ik_llama family"},
		{"empty architecture", "", hist(stq58), "", "", nil, "no special requirement"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := c.resolve(tc.arch, tc.types)
			recipe := ""
			if got.Recipe != nil {
				recipe = got.Recipe.Name
			}
			if got.Family != tc.family || recipe != tc.recipe || !reflect.DeepEqual(got.Unchecked, tc.unchecked) {
				t.Fatalf("family %q recipe %q unchecked %v; want %q %q %v (reason: %s)",
					got.Family, recipe, got.Unchecked, tc.family, tc.recipe, tc.unchecked, got.Reason)
			}
			if !strings.Contains(got.Reason, tc.reason) {
				t.Errorf("reason %q does not mention %q", got.Reason, tc.reason)
			}
			if got.None() != (tc.family == "" && tc.recipe == "") {
				t.Errorf("None() = %v for family %q recipe %q", got.None(), tc.family, tc.recipe)
			}
		})
	}
}

// Without type data the resolver is exactly the architecture table, for every
// architecture the shipped catalog knows and for variants of each.
func TestResolveRequirementWithoutTypesEqualsRequiredBackendForArch(t *testing.T) {
	archs := []string{"", "  ", "llama", "qwen4exp", "minimax", "minimaxm3", "minimax-m", "minimax-m2", "minimax-m31", "MiniMax-M3", " minimax-m4 "}
	for _, r := range Recipes() {
		archs = append(archs, r.RouteArch, strings.ToUpper(r.RouteArch), " "+r.RouteArch+" ")
	}
	for _, rule := range loadedCatalog.nameRules {
		archs = append(archs, rule.Value, rule.Value+"1", rule.Value+"-x", strings.ToUpper(rule.Value))
	}
	someTypes := hist(gguf.TypeUsage{Type: 8, Tensors: 5}, gguf.TypeUsage{Type: 44, Tensors: 58, Names: []string{"blk.*.ffn_up.weight"}})
	for _, arch := range archs {
		want := RequiredBackendForArch(arch)
		for label, types := range map[string][]gguf.TypeUsage{"nil": nil, "types": someTypes} {
			got := ResolveRequirement(arch, types)
			if got.Family != want {
				t.Errorf("ResolveRequirement(%q, %s).Family = %q, RequiredBackendForArch = %q", arch, label, got.Family, want)
			}
			// The shipped catalog states no tensor encodings, so no recipe is
			// a requirement and nothing is left unchecked.
			if got.Recipe != nil || len(got.Unchecked) != 0 {
				t.Errorf("ResolveRequirement(%q, %s) = recipe %+v unchecked %v, want none", arch, label, got.Recipe, got.Unchecked)
			}
			if (want == "") != got.None() || got.Reason == "" {
				t.Errorf("ResolveRequirement(%q, %s): None() = %v, reason %q", arch, label, got.None(), got.Reason)
			}
		}
	}
}

func TestNilCatalogResolvesWithoutPanicking(t *testing.T) {
	var c *parsedCatalog
	got := c.resolve("minimax-m3", hist(gguf.TypeUsage{Type: 44, Tensors: 1}))
	if !got.None() || got.Reason == "" {
		t.Fatalf("nil catalog: %+v", got)
	}
	if names := c.encodingNames(); names != nil {
		t.Fatalf("nil catalog names: %v", names)
	}
}

func TestEncodingNamesAreDistinctAndSorted(t *testing.T) {
	c := mustCatalog(t,
		encRecipe("a", "x", `,"required_tensor_types":[{"id":50,"name":"ZZZ"},{"id":51,"name":"AAA"}]`)+","+
			encRecipe("b", "y", `,"required_tensor_types":[{"id":52,"name":"AAA"}]`)+","+
			encRecipe("c", "z", ""),
		`{"match":"arch_prefix","value":"q","requires":"ik_llama"}`)
	if got, want := c.encodingNames(), []string{"AAA", "ZZZ"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("names %v, want %v", got, want)
	}
}

func TestLoadCatalogReadsEncodingRequirements(t *testing.T) {
	c := mustCatalog(t, encRecipe("a", "hy_v4", stqLike), `{"match":"arch_prefix","value":"q","requires":"ik_llama"}`)
	want := []EncodingRequirement{{ID: 44, Name: "STQ1_0", TensorsMin: 10, TensorsMax: 100, NameContains: "ffn_"}}
	if len(c.encodings) != 1 || !reflect.DeepEqual(c.encodings[0], want) {
		t.Fatalf("encodings %+v, want %+v", c.encodings, want)
	}
	// Ids below 40 are legal too: the file's id 0 is F32 everywhere.
	if _, err := loadCatalog([]byte(testCatalogJSON(1, encRecipe("a", "x", `,"required_tensor_types":[{"id":0,"name":"F32"}]`),
		`{"match":"arch_prefix","value":"q","requires":"ik_llama"}`)), testPatchFS()); err != nil {
		t.Fatalf("id 0 rejected: %v", err)
	}
}

func TestLoadCatalogRejectsBadEncodingRequirements(t *testing.T) {
	rule := `{"match":"arch_prefix","value":"q","requires":"ik_llama"}`
	enc := func(items string) string {
		return testCatalogJSON(1, encRecipe("a", "x", `,"required_tensor_types":[`+items+`]`), rule)
	}
	cases := []struct{ name, data, want string }{
		{"missing id", enc(`{"name":"X"}`), "missing id"},
		{"negative id", enc(`{"id":-1,"name":"X"}`), "not a tensor type id"},
		{"id beyond uint32", enc(`{"id":4294967296,"name":"X"}`), "not a tensor type id"},
		{"duplicate id", enc(`{"id":44,"name":"X"},{"id":44,"name":"Y"}`), "duplicate id 44"},
		{"missing name", enc(`{"id":44}`), "name"},
		{"name with space", enc(`{"id":44,"name":"STQ 1"}`), "letters, digits or underscores"},
		{"name too long", enc(`{"id":44,"name":"` + strings.Repeat("A", 49) + `"}`), "letters, digits or underscores"},
		{"negative minimum", enc(`{"id":44,"name":"X","tensors_min":-1}`), "not a valid range"},
		{"minimum above maximum", enc(`{"id":44,"name":"X","tensors_min":5,"tensors_max":4}`), "not a valid range"},
		{"non-ASCII name pattern", enc(`{"id":44,"name":"X","name_contains":"café"}`), "printable ASCII"},
		{"unknown field", enc(`{"id":44,"name":"X","bogus":1}`), "unknown field"},
		{"old list-of-names form", enc(`"stq1_0"`), "cannot unmarshal"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, err := loadCatalog([]byte(tc.data), testPatchFS())
			if err == nil {
				t.Fatalf("accepted a bad catalog: %+v", c)
			}
			if c != nil {
				t.Errorf("a failed load must not return a catalog")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not mention %q", err, tc.want)
			}
		})
	}
}

func TestEncodingNamesFollowTheShippedCatalog(t *testing.T) {
	if got, want := EncodingNames(), loadedCatalog.encodingNames(); !reflect.DeepEqual(got, want) {
		t.Fatalf("EncodingNames() = %v, catalog says %v", got, want)
	}
}
