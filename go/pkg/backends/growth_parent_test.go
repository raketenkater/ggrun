package backends

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHotExpertGrowthParentRequiresReviewedLineage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "server")
	if err := os.WriteFile(path, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	base := Backend{Tag: "base", Path: path, GitURL: "https://example.test/llama", Commit: strings.Repeat("a", 40)}
	recipe, err := ComposeRecipe(base, "hot-experts")
	if err != nil {
		t.Fatal(err)
	}
	original := Backend{Tag: recipe.Tag, BaseTag: base.Tag, GitURL: base.GitURL, Commit: base.Commit, Features: []string{"hot-experts"}, AppliedPatches: recipe.PatchNames()}
	cases := []struct {
		name   string
		mutate func(*Backend, *Backend)
		want   bool
	}{
		{"reviewed", func(o, b *Backend) {}, true},
		{"older reviewed overlay", func(o, b *Backend) { o.AppliedPatches = o.AppliedPatches[:1] }, true},
		{"different commit", func(o, b *Backend) { b.Commit = strings.Repeat("b", 40) }, false},
		{"different source", func(o, b *Backend) { b.GitURL = "https://other.test/repo" }, false},
		{"missing parent binary", func(o, b *Backend) { b.Path = path + "-missing" }, false},
		{"unknown patch", func(o, b *Backend) { o.AppliedPatches = []string{"custom"} }, false},
		{"no feature patch", func(o, b *Backend) { o.AppliedPatches = nil }, false},
		{"multiple features", func(o, b *Backend) { o.Features = []string{"hot-experts", "other"} }, false},
		{"unrelated parent", func(o, b *Backend) { o.BaseTag = "other" }, false},
		{"short commit", func(o, b *Backend) { o.Commit = "abc"; b.Commit = "abc" }, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			o, b := original, base
			tc.mutate(&o, &b)
			got := HotExpertGrowthParent(o, []Backend{b})
			if (got != nil) != tc.want {
				t.Fatalf("parent=%+v want=%v", got, tc.want)
			}
		})
	}
}
