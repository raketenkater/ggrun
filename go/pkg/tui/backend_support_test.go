package tui

import (
	"reflect"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/recommend"
)

// One resolver answer, both surfaces: the local list row and config screen,
// and the recommendations screen, print the same state's wording.
func TestModelListAndRecommendationsWordTheSameState(t *testing.T) {
	lacks := []backends.InstalledBackend{{Name: "main", Path: "/b/main", Probed: true}}
	for _, in := range []backends.SupportInput{
		{Arch: "neverheard", Installed: lacks},
		{Arch: "hy_v3", Installed: lacks},
		{Arch: "hy_v3", Installed: lacks, BuildBlocked: "source builds are not available on Windows"},
	} {
		s := backends.ResolveSupport(in)
		item := ModelItem{Name: "m.gguf", Path: "/models/m.gguf", SizeGB: 4, Arch: in.Arch, Architecture: in.Arch, Support: s}
		if desc := mainModelDesc(item, nil); !strings.Contains(desc, " · "+s.Short()) {
			t.Fatalf("%s: list row %q lacks %q", s.State, desc, s.Short())
		}
		m := Model{models: []ModelItem{item}, selectedModel: 0, backend: "auto", kvPlacement: "auto", ctxMode: "fit"}
		if view := m.viewModelConfig(); !strings.Contains(view, s.Note()) {
			t.Fatalf("%s: config screen lacks %q", s.State, s.Note())
		}
		rec := recommend.Recommendation{Candidate: recommend.Candidate{Name: "Row", Arch: in.Arch}, QuantName: "Q4_K_M",
			BackendStatus: recommend.StatusForSupport(s.State), BackendSupport: s.State, BackendNote: s.Note()}
		rows := []recommend.Recommendation{rec}
		rm := Model{recommendationGroups: recommend.Categories{Balanced: rows}, recommendations: rows, selectedRecommendation: 0}
		if view := rm.viewRecommended(); !strings.Contains(view, s.Note()) {
			t.Fatalf("%s: recommendations screen lacks %q:\n%s", s.State, s.Note(), view)
		}
	}
}

// Support is display only: a recipe or route wording wins in the list, the
// zero value renders exactly as before, and the launch request is identical.
func TestBackendSupportNeverChangesTheLaunchRequest(t *testing.T) {
	s := backends.ResolveSupport(backends.SupportInput{Arch: "neverheard", Installed: []backends.InstalledBackend{{Name: "main", Probed: true}}})
	base := ModelItem{Name: "m.gguf", Path: "/models/m.gguf", SizeGB: 4, Arch: "neverheard", Architecture: "neverheard"}
	with := base
	with.Support = s
	if mainModelDesc(base, nil) == mainModelDesc(with, nil) {
		t.Fatal("the support note is not shown")
	}
	recipe := with
	recipe.BackendRecipe = "hy3"
	if strings.Contains(mainModelDesc(recipe, nil), s.Short()) {
		t.Fatal("a recipe row must keep its own wording")
	}
	for _, backend := range []string{"auto", "llama", "ik_llama"} {
		a := Model{models: []ModelItem{base}, selectedModel: 0, backend: backend, kvPlacement: "auto", ctxMode: "fit"}
		b := Model{models: []ModelItem{with}, selectedModel: 0, backend: backend, kvPlacement: "auto", ctxMode: "fit"}
		if ra, rb := a.buildLaunchRequest(), b.buildLaunchRequest(); !reflect.DeepEqual(ra, rb) {
			t.Fatalf("backend %s: launch request changed: %#v vs %#v", backend, ra, rb)
		}
	}
}
