package tui

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/gguf"
	"github.com/raketenkater/ggrun/pkg/modelusage"
	"github.com/raketenkater/ggrun/pkg/recommend"
)

// linesWith returns the lines of view that contain sub.
func linesWith(view, sub string) []string {
	var out []string
	for _, l := range strings.Split(view, "\n") {
		if strings.Contains(l, sub) {
			out = append(out, l)
		}
	}
	return out
}

// One resolver answer, both surfaces: the local list row and config screen,
// and the recommendations screen, print the same state's wording, each on one
// row of at most 80 columns at an 80-column terminal.
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
		m := Model{models: []ModelItem{item}, selectedModel: 0, backend: "auto", kvPlacement: "auto", ctxMode: "fit", width: 80}
		want := truncateCols(s.Note(), 80-31)
		got := linesWith(m.viewModelConfig(), want)
		if len(got) != 1 || lipgloss.Width(got[0]) > 80 {
			t.Fatalf("%s: config screen rows with the note: %q", s.State, got)
		}
		rec := recommend.Recommendation{Candidate: recommend.Candidate{Name: "Row", Arch: in.Arch}, QuantName: "Q4_K_M",
			BackendStatus: recommend.StatusForSupport(s.State), BackendSupport: s.State, BackendNote: s.Note()}
		rows := []recommend.Recommendation{rec}
		rm := Model{recommendationGroups: recommend.Categories{Balanced: rows}, recommendations: rows, selectedRecommendation: 0, width: 80}
		want = truncateCols(s.Note(), 80-4)
		got = linesWith(rm.viewRecommended(), want)
		if len(got) != 1 || lipgloss.Width(got[0]) > 80 {
			t.Fatalf("%s: recommendations rows with the note: %q", s.State, got)
		}
	}
}

// The list note comes after the fits and usage badges, so at 80 columns
// those keep the place they had without it.
func TestListNoteKeepsTheBadgesAt80Columns(t *testing.T) {
	s := backends.ResolveSupport(backends.SupportInput{Arch: "neverheard", Installed: []backends.InstalledBackend{{Name: "main", Probed: true}}})
	base := ModelItem{Name: "m.gguf", Path: "/models/sub/m.gguf", SizeGB: 24.5, Arch: "averylongarchname · MoE", Architecture: "averylongarchname", FitCtx: 131072}
	usage := &modelusage.Record{Launches: 12, LastUsedAt: time.Now().Add(-3 * time.Hour)}
	with := base
	with.Support = s
	plain, noted := mainModelDesc(base, usage), mainModelDesc(with, usage)
	cut := strings.Index(plain, " · sub")
	if cut < 0 || !strings.HasPrefix(noted, plain[:cut]) || !strings.Contains(noted, " · "+s.Short()) {
		t.Fatalf("badges moved: %q vs %q", plain, noted)
	}
	if lipgloss.Width(plain[:cut]) > 76 {
		t.Fatalf("fixture too wide for the check: %q", plain[:cut])
	}
}

// Support is display only: a recipe or route wording wins in the list, the
// zero value renders exactly as before, and the launch request is identical.
// (A tripwire: buildLaunchRequest does not read Support.)
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

// enrichModelItems with the hook installed keeps every launch input (the item
// list, AutoBackend, BackendRecipe) exactly as without it, and only fills
// Support from the file's own architecture.
func TestEnrichModelItemsHookOnlyFillsSupport(t *testing.T) {
	stubParser(t) // every header reports arch "fake"
	modelDir, cacheDir := t.TempDir(), t.TempDir()
	var items []ModelItem
	for _, name := range []string{"a.gguf", "b.gguf"} {
		path := filepath.Join(modelDir, name)
		if err := os.WriteFile(path, []byte("GGUF-"+name), 0o644); err != nil {
			t.Fatal(err)
		}
		items = append(items, ModelItem{Name: name, Path: path, Arch: "dense"})
	}
	clone := func() []ModelItem { return append([]ModelItem(nil), items...) }

	t.Cleanup(func() { SetBackendSupport(nil) })
	SetBackendSupport(nil)
	without := enrichModelItems(clone(), cacheDir, "auto", nil)

	var archs []string
	answer := backends.Support{State: backends.SupportUnknownArch, Reason: "note"}
	SetBackendSupport(func(_ *detect.Capabilities, arch string, _ []gguf.TypeUsage) backends.Support {
		archs = append(archs, arch)
		return answer
	})
	with := enrichModelItems(clone(), cacheDir, "auto", nil)

	if len(with) != len(without) || len(with) != 2 {
		t.Fatalf("item list changed: %d vs %d", len(with), len(without))
	}
	for i := range with {
		if with[i].AutoBackend != without[i].AutoBackend || with[i].BackendRecipe != without[i].BackendRecipe || with[i].Path != without[i].Path {
			t.Fatalf("launch inputs changed for %s: %+v vs %+v", with[i].Name, with[i], without[i])
		}
		if without[i].Support.State != "" || with[i].Support.State != answer.State {
			t.Fatalf("support for %s: without %q, with %q", with[i].Name, without[i].Support.State, with[i].Support.State)
		}
	}
	if !reflect.DeepEqual(archs, []string{"fake", "fake"}) {
		t.Fatalf("resolver asked about %q", archs)
	}
}
