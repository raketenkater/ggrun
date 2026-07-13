package backends

import "testing"

func TestHY3RecipeIsPinnedAndRouted(t *testing.T) {
	recipe := RecipeByName("hy3")
	if recipe == nil {
		t.Fatal("HY3 recipe missing")
	}
	if recipe.RouteArch != "hy_v3" {
		t.Fatalf("HY3 route arch = %q, want hy_v3", recipe.RouteArch)
	}
	if recipe.Branch != "hy3-support" || len(recipe.Commit) != 40 {
		t.Fatalf("HY3 source is not reproducibly pinned: %#v", recipe)
	}
	if recipe.GitURL != "https://github.com/noonr48/ik_llama-hy3.git" {
		t.Fatalf("unexpected HY3 fork: %s", recipe.GitURL)
	}

	// Callers receive a copy and cannot mutate the built-in catalog.
	recipe.Commit = "changed"
	again := RecipeByName("HY3")
	if again == nil || again.Commit == "changed" {
		t.Fatal("recipe lookup leaked mutable catalog state")
	}
}
