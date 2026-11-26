package backends

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"testing/fstest"
)

// frozenRecipe is a copy of the Go literals that builtinRecipes held before
// the catalog moved to data, with each patch reduced to its name and the
// sha256 of its bytes. It must never be edited to follow the catalog: a
// catalog change that alters behaviour has to fail here and be made on purpose.
type frozenRecipe struct {
	recipe  Recipe
	patches []frozenPatch
}

type frozenPatch struct{ name, sha256 string }

var frozenRecipes = []frozenRecipe{
	{
		recipe: Recipe{
			Name:        "nanbeige42",
			Description: "Upstream llama.cpp with native Nanbeige4.2 looped-transformer support",
			Tag:         "nanbeige42",
			GitURL:      "https://github.com/ggml-org/llama.cpp.git",
			Branch:      "master",
			Commit:      "b77d646751d01c0962bc203b6809e9d94f7d50b7",
			RouteArch:   "nanbeige",
			HelperOnly:  true,
			Accel:       "",
		},
		patches: []frozenPatch{{"0001-accept-loop-count-gguf-schema.patch", "6aa0a0777d8c043040c5aa41d5f9d71880025b0e78fef6a84d04bd8da1466143"}},
	},
	{
		recipe: Recipe{
			Name:        "hy3",
			Description: "Tencent Hy3 / hy_v3 support with built-in MTP and its reviewed chat template",
			Tag:         "hy3",
			GitURL:      "https://github.com/noonr48/ik_llama-hy3.git",
			Branch:      "hy3-support",
			Commit:      "f46c95ee90d8c8200b0147c646b883405020b482",
			RouteArch:   "hy_v3",
			Accel:       "",
		},
		patches: []frozenPatch{{"hy3/0001-fix-router-tensor-name", "689947b8644078d6b911ad55833477c5a4c518e330c0330b4a945589c526781e"}},
	},
	{
		recipe: Recipe{
			Name:        "minimax-m3",
			Description: "Preliminary MiniMax-M3 support with native structured tool-call parsing",
			Tag:         "minimax-m3",
			GitURL:      "https://github.com/danielhanchen/llama.cpp.git",
			Branch:      "minimax-m3",
			Commit:      "66f43aa655a07999c7746fe9ff5ede94835e921e",
			RouteArch:   "minimax-m3",
			Accel:       "",
		},
	},
	{
		recipe: Recipe{
			Name:        "laguna",
			Description: "Poolside Laguna support including the DFlash drafter (poolside fork, not the upstream PR)",
			Tag:         "laguna",
			GitURL:      "https://github.com/poolsideai/llama.cpp.git",
			Branch:      "laguna",
			Commit:      "04b2b72cb54048ead292884adbe11f284e3ec950",
			RouteArch:   "laguna",
			Accel:       "",
		},
		patches: []frozenPatch{{"laguna/0001-include-cmath-for-isfinite", "d824bf82bdcdcb01126c83a232b0d0444a051b1164ebe5a8e8c4490ab6d46f3d"}},
	},
	{
		recipe: Recipe{
			Name:        "laguna-upstream-pr",
			Description: "Laguna target architecture from the open upstream llama.cpp PR (no DFlash drafter)",
			Tag:         "laguna-upstream-pr",
			GitURL:      "https://github.com/joerowell/llama.cpp.git",
			Branch:      "add-laguna",
			Commit:      "54f214a09b8c4e709357ae661a77925edb154f13",
			RouteArch:   "laguna",
			Accel:       "",
		},
	},
}

// The embedded file is validated here, not at launch: a load failure is a
// programming error and must stop CI, while a user launch never panics.
func TestEmbeddedCatalogLoads(t *testing.T) {
	if catalogLoadErr != nil {
		t.Fatalf("embedded backend catalog failed to load: %v", catalogLoadErr)
	}
	if loadedCatalog == nil {
		t.Fatal("embedded backend catalog is nil")
	}
}

func TestCatalogRecipesMatchFrozenLiterals(t *testing.T) {
	if len(builtinRecipes) != len(frozenRecipes) {
		t.Fatalf("catalog has %d recipes, frozen literals have %d", len(builtinRecipes), len(frozenRecipes))
	}
	for i, want := range frozenRecipes {
		got := builtinRecipes[i]
		if got.Name != want.recipe.Name {
			t.Fatalf("recipe #%d is %q, want %q (order matters)", i, got.Name, want.recipe.Name)
		}
		if len(got.Patches) != len(want.patches) {
			t.Fatalf("%s: %d patches, want %d", got.Name, len(got.Patches), len(want.patches))
		}
		for j, wp := range want.patches {
			sum := sha256.Sum256(got.Patches[j].contents)
			if got.Patches[j].Name != wp.name || hex.EncodeToString(sum[:]) != wp.sha256 {
				t.Errorf("%s patch #%d: name %q sha256 %x, want name %q sha256 %s",
					got.Name, j, got.Patches[j].Name, sum, wp.name, wp.sha256)
			}
		}
		// Compare every Recipe field with the patch bytes set aside; a field
		// added to Recipe later has to be accounted for in the frozen copy.
		stripped := got
		stripped.Patches = nil
		if !reflect.DeepEqual(stripped, want.recipe) {
			t.Errorf("%s differs from the frozen literal:\n got  %+v\n want %+v", got.Name, stripped, want.recipe)
		}
		if g, w := got.PatchNames(), patchNamesOf(want); !reflect.DeepEqual(g, w) {
			t.Errorf("%s PatchNames() = %v, want %v", got.Name, g, w)
		}
	}
}

func patchNamesOf(f frozenRecipe) []string {
	names := make([]string, 0, len(f.patches))
	for _, p := range f.patches {
		names = append(names, p.name)
	}
	return names
}

func TestRecipeLookupsFollowTheCatalog(t *testing.T) {
	if got := Recipes(); len(got) != len(frozenRecipes) {
		t.Fatalf("Recipes() returned %d, want %d", len(got), len(frozenRecipes))
	}
	for _, want := range frozenRecipes {
		for _, key := range []string{want.recipe.Name, strings.ToUpper(want.recipe.Tag), " " + want.recipe.Name + " "} {
			got := RecipeByName(key)
			if got == nil || got.Name != want.recipe.Name {
				t.Errorf("RecipeByName(%q) = %+v, want %s", key, got, want.recipe.Name)
			}
		}
	}
	if RecipeByName("no-such-recipe") != nil {
		t.Error("an unknown name must not resolve")
	}
}

func TestRequiredBackendForArchTable(t *testing.T) {
	cases := []struct{ arch, want string }{
		{"minimax-m2", "ik_llama"},
		{"minimax-m3", "ik_llama"},
		{"minimax-m31", "ik_llama"},
		{"minimax-m4", "ik_llama"},
		{" minimax-m4 ", "ik_llama"},
		{"MiniMax-M3", "ik_llama"},
		{"minimax-m", "ik_llama"},
		{"minimax", ""},
		{"minimaxm3", ""},
		{"", ""},
		{"   ", ""},
		{"qwen4exp", ""},
		{"laguna", ""},
		{"hy_v3", ""},
		{"nanbeige", ""},
		{"llama", ""},
	}
	for _, tc := range cases {
		if got := RequiredBackendForArch(tc.arch); got != tc.want {
			t.Errorf("RequiredBackendForArch(%q) = %q, want %q", tc.arch, got, tc.want)
		}
	}
}

func TestCatalogHoldsOnlyTheFrozenNameRule(t *testing.T) {
	rules := loadedCatalog.nameRules
	if len(rules) != 1 || rules[0].Match != "arch_prefix" || rules[0].Value != "minimax-m" || rules[0].Requires != "ik_llama" {
		t.Fatalf("name rules = %+v, want only arch_prefix minimax-m -> ik_llama", rules)
	}
}

// A nil catalog (a failed load) must degrade without panicking.
func TestNilCatalogDegradesToNoRules(t *testing.T) {
	var c *parsedCatalog
	if got := c.requiredBackend("minimax-m3"); got != "" {
		t.Errorf("nil catalog requiredBackend = %q, want empty", got)
	}
	if got := catalogRecipes(nil); got != nil {
		t.Errorf("nil catalog recipes = %v, want nil", got)
	}
}

const (
	testPatchBody  = "diff --git a/x b/x\n"
	testGoodCommit = "0123456789abcdef0123456789abcdef01234567"
)

func testPatchSum() string {
	sum := sha256.Sum256([]byte(testPatchBody))
	return hex.EncodeToString(sum[:])
}

func testPatchFS() fstest.MapFS {
	return fstest.MapFS{
		"patches/a/0001-x.patch":     {Data: []byte(testPatchBody)},
		"patches/a/0002-empty.patch": {Data: nil},
	}
}

func testRecipeJSON(name, tag, extra string) string {
	return fmt.Sprintf(`{"name":%q,"description":"d","tag":%q,"git_url":"u","branch":"b","commit":%q,"route_arch":"arch"%s}`,
		name, tag, testGoodCommit, extra)
}

func testCatalogJSON(version int, recipes, rules string) string {
	return fmt.Sprintf(`{"version":%d,"recipes":[%s],"name_rules":[%s]}`, version, recipes, rules)
}

func TestLoadCatalogAcceptsAValidCatalog(t *testing.T) {
	patch := fmt.Sprintf(`,"patches":[{"name":"p","file":"a/0001-x.patch","sha256":%q}]`, testPatchSum())
	data := testCatalogJSON(1, testRecipeJSON("one", "one", patch)+","+testRecipeJSON("two", "tag2", ""),
		`{"match":"arch_prefix","value":"foo-","requires":"ik_llama"}`)
	c, err := loadCatalog([]byte(data), testPatchFS())
	if err != nil {
		t.Fatalf("valid catalog rejected: %v", err)
	}
	if len(c.recipes) != 2 || len(c.recipes[0].Patches) != 1 || string(c.recipes[0].Patches[0].contents) != testPatchBody {
		t.Fatalf("unexpected parse result: %+v", c.recipes)
	}
	if got := c.requiredBackend("FOO-1"); got != "ik_llama" {
		t.Errorf("requiredBackend = %q", got)
	}
}

func TestLoadCatalogRejectsBadCatalogs(t *testing.T) {
	goodRule := `{"match":"arch_prefix","value":"x","requires":"ik_llama"}`
	badSum := fmt.Sprintf(`,"patches":[{"name":"p","file":"a/0001-x.patch","sha256":%q}]`, strings.Repeat("0", 64))
	cases := []struct {
		name, data, want string
	}{
		{"not json", `{`, "parse backend catalog"},
		{"unknown top-level field", `{"version":1,"recipes":[],"name_rules":[],"bogus":1}`, "unknown field"},
		{"unknown recipe field", testCatalogJSON(1, testRecipeJSON("a", "a", `,"bogus":1`), goodRule), "unknown field"},
		{"wrong version", testCatalogJSON(2, testRecipeJSON("a", "a", ""), goodRule), "version 2"},
		{"missing version", `{"recipes":[],"name_rules":[]}`, "version 0"},
		{"duplicate name", testCatalogJSON(1, testRecipeJSON("a", "a", "")+","+testRecipeJSON("a", "b", ""), goodRule), "duplicate id"},
		{"duplicate tag", testCatalogJSON(1, testRecipeJSON("a", "t", "")+","+testRecipeJSON("b", "t", ""), goodRule), "duplicate id"},
		{"duplicate id case-insensitive", testCatalogJSON(1, testRecipeJSON("a", "a", "")+","+testRecipeJSON("A", "z", ""), goodRule), "duplicate id"},
		{"name collides with another tag", testCatalogJSON(1, testRecipeJSON("a", "b", "")+","+testRecipeJSON("b", "c", ""), goodRule), "duplicate id"},
		{"missing name", testCatalogJSON(1, testRecipeJSON("", "t", ""), goodRule), "missing name"},
		{"missing git_url", testCatalogJSON(1, `{"name":"a","description":"d","tag":"a","branch":"b","commit":"`+testGoodCommit+`","route_arch":"x"}`, goodRule), "missing git_url"},
		{"missing route_arch", testCatalogJSON(1, `{"name":"a","description":"d","tag":"a","git_url":"u","branch":"b","commit":"`+testGoodCommit+`"}`, goodRule), "missing route_arch"},
		{"short commit", testCatalogJSON(1, strings.Replace(testRecipeJSON("a", "a", ""), testGoodCommit, "abc123", 1), goodRule), "40-character"},
		{"unknown recipe kind", testCatalogJSON(1, testRecipeJSON("a", "a", `,"kind":"wishful"`), goodRule), "unknown kind"},
		{"bad expiry", testCatalogJSON(1, testRecipeJSON("a", "a", `,"expires":"soon"`), goodRule), "YYYY-MM-DD"},
		{"patch file missing", testCatalogJSON(1, testRecipeJSON("a", "a", `,"patches":[{"name":"p","file":"a/nope.patch","sha256":"00"}]`), goodRule), "nope.patch"},
		{"patch empty", testCatalogJSON(1, testRecipeJSON("a", "a", `,"patches":[{"name":"p","file":"a/0002-empty.patch","sha256":"00"}]`), goodRule), "is empty"},
		{"patch sha256 mismatch", testCatalogJSON(1, testRecipeJSON("a", "a", badSum), goodRule), "sha256"},
		{"patch without name", testCatalogJSON(1, testRecipeJSON("a", "a", `,"patches":[{"file":"a/0001-x.patch","sha256":"00"}]`), goodRule), "missing name or file"},
		{"unknown rule kind", testCatalogJSON(1, testRecipeJSON("a", "a", ""), `{"match":"regex","value":"x","requires":"ik_llama"}`), "unknown match kind"},
		{"missing rule kind", testCatalogJSON(1, testRecipeJSON("a", "a", ""), `{"value":"x","requires":"ik_llama"}`), "unknown match kind"},
		{"rule value not lower-case", testCatalogJSON(1, testRecipeJSON("a", "a", ""), `{"match":"arch_prefix","value":"MiniMax-M","requires":"ik_llama"}`), "lower-case"},
		{"rule value empty", testCatalogJSON(1, testRecipeJSON("a", "a", ""), `{"match":"arch_prefix","value":"","requires":"ik_llama"}`), "non-empty"},
		{"rule missing requires", testCatalogJSON(1, testRecipeJSON("a", "a", ""), `{"match":"arch_prefix","value":"x"}`), "missing requires"},
		{"duplicate rule", testCatalogJSON(1, testRecipeJSON("a", "a", ""), goodRule+","+goodRule), "duplicate arch_prefix"},
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
