package main

import (
	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/placement"
	"strings"
	"testing"
)

func TestRegisteredOverlayCarriesReserveParentWithoutAliasingFit(t *testing.T) {
	t.Setenv("LLM_APP_HOME", t.TempDir())
	base := backends.Backend{Tag: "base", Path: writeFakeBackend(t, "base-server", "echo base-version\n"), GitURL: "https://example.test/llama", Commit: strings.Repeat("a", 40)}
	recipe, err := backends.ComposeRecipe(base, "hot-experts")
	if err != nil {
		t.Fatal(err)
	}
	overlay := backends.Backend{Tag: recipe.Tag, Path: writeFakeBackend(t, "overlay-server", "echo '--moe-expert-cache N --moe-expert-cache-inserts N'\n"), BaseTag: base.Tag, GitURL: base.GitURL, Commit: base.Commit, Features: []string{"hot-experts"}, AppliedPatches: recipe.PatchNames()}
	if err := backends.Save([]backends.Backend{base, overlay}); err != nil {
		t.Fatal(err)
	}
	be := detectRegisteredBackend(&overlay)
	opts := placementOptionsFromRequestCaps(&launchRequest{HotExperts: "auto"}, &placement.ModelProfile{IsMoE: true}, be, t.TempDir(), nil)
	want := evidenceBackendCacheTag(detectRegisteredBackend(&base))
	if opts.RuntimeGrowthBaseBackendTag != want || opts.BackendCacheTag == want || opts.BackendCacheTag != evidenceBackendCacheTag(be) {
		t.Fatalf("reserve parent or exact scope lost: parent=%q target=%q wantParent=%q", opts.RuntimeGrowthBaseBackendTag, opts.BackendCacheTag, want)
	}
	overlay.Commit = strings.Repeat("b", 40)
	if got := detectRegisteredBackend(&overlay); got.RuntimeGrowthBaseBackendTag != "" {
		t.Fatal("different source revision inherited growth")
	}
}
