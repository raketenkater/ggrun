package main

import (
	"fmt"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/recommend"
)

// legacyRowStatus is the status markBackendStatus gave a row before the shared
// resolver (its installed-arch and build-route branch, copied). The real old
// path is compared in TestSharedResolverMatchesTheOldHooksOnTheCatalog; this
// copy lets the wide flag enumeration below run without a catalog per case.
func legacyRowStatus(paths []string, arch string, probe backendArchProbe, mainline, pinned, buildable bool) recommend.BackendStatus {
	loads, known := archLoadedBy(paths, arch, probe)
	switch {
	case !known:
		return recommend.BackendUnknown
	case loads:
		return recommend.BackendReady
	case backendBuildRoute(arch, mainline, pinned, buildable):
		return recommend.BackendNeedsSetup
	}
	return recommend.BackendUnsupported
}

// Over the embedded catalog, which has no tensor-encoding entry, the shared
// resolver gives every row exactly the status the old hooks gave it, so
// ranking, `recommend --first` and the installer's first download are
// unchanged. Architectures cover each catalog shape: none, a reviewed recipe,
// a helper recipe with a pinned commit, a name rule, a name rule plus recipe.
func TestSharedResolverMatchesTheLegacyRowStatus(t *testing.T) {
	scenarios := map[string]map[string][2]bool{
		"none installed":     nil,
		"one unreadable":     {"/a": {false, false}},
		"loads":              {"/a": {true, true}},
		"does not load":      {"/a": {false, true}},
		"unreadable + loads": {"/a": {false, false}, "/b": {true, true}},
		"unreadable + lacks": {"/a": {false, false}, "/b": {false, true}},
	}
	archs := []string{"k2-horizon", "hy_v3", "laguna", "nanbeige", "minimax-m2", "minimax-m3", "inkling"}
	for name, answers := range scenarios {
		var paths []string
		for _, p := range []string{"/a", "/b"} {
			if _, ok := answers[p]; ok {
				paths = append(paths, p)
			}
		}
		probe := func(path, _ string) (bool, bool) { a := answers[path]; return a[0], a[1] }
		for _, arch := range archs {
			for _, goos := range []string{"linux", "windows"} {
				for bits := 0; bits < 32; bits++ {
					mainline, known, pinned, tools, nvidia := bits&1 != 0, bits&2 != 0, bits&4 != 0, bits&8 != 0, bits&16 != 0
					if mainline && !known {
						continue
					}
					accel := "cuda"
					if nvidia && bits%3 == 0 {
						accel = "vulkan"
					}
					onPath := tools && goos != "windows"
					want := legacyRowStatus(paths, arch, probe, mainline, pinned, buildCanStart(onPath, nvidia, accel))
					got := backends.ResolveSupport(backends.SupportInput{
						Arch:              arch,
						Installed:         installedBackendsFor(paths, arch, probe, nil, bits%2 == 0),
						MainlineRegisters: mainline,
						MainlineKnown:     known,
						BuildBlocked:      buildBlockedReason(goos, pinned, onPath, nvidia, accel),
					})
					if s := recommend.StatusForSupport(got.State); s != want {
						t.Fatalf("%s / %s / %s mainline=%v known=%v pinned=%v tools=%v nvidia=%v accel=%s: resolver %s (%s), legacy %s",
							name, arch, goos, mainline, known, pinned, onPath, nvidia, accel, got.State, got.Reason, want)
					}
				}
			}
		}
	}
}

func TestBuildBlockedReasonIsEmptyExactlyWhenABuildCanStart(t *testing.T) {
	for bits := 0; bits < 16; bits++ {
		pinned, tools, nvidia, cuda := bits&1 != 0, bits&2 != 0, bits&4 != 0, bits&8 != 0
		accel := "vulkan"
		if cuda {
			accel = "cuda"
		}
		can := !pinned && buildCanStart(tools, nvidia, accel)
		if got := buildBlockedReason("linux", pinned, tools, nvidia, accel); (got == "") != can {
			t.Fatalf("%s: reason %q, can start %v", fmt.Sprint(pinned, tools, nvidia, accel), got, can)
		}
		if got := buildBlockedReason("windows", pinned, tools, nvidia, accel); got == "" {
			t.Fatal("Windows has no source build")
		}
	}
}

// `ggrun recommend` prints the resolver's note under a flagged row, and only
// there: the same sentence the TUI shows.
func TestRecommendationNoteIsTheResolverWording(t *testing.T) {
	s := backends.ResolveSupport(backends.SupportInput{
		Arch:      "hy_v3",
		Installed: []backends.InstalledBackend{{Name: "main", Path: "/a", Probed: true}},
	})
	r := recommend.Recommendation{BackendStatus: recommend.StatusForSupport(s.State), BackendNote: s.Note()}
	if s.State != backends.SupportInstall || recommendationNote(r) != s.Note() {
		t.Fatalf("flagged row: %+v -> %q", s, recommendationNote(r))
	}
	r.BackendStatus = recommend.BackendReady
	if recommendationNote(r) != "" {
		t.Fatal("a ready row gets no note")
	}
}

// The real old code path against the real new one: TopCategories over the
// embedded model catalog, once with the old hooks (SetInstalledArchSupport +
// SetBackendRoute, the fallback branch of markBackendStatus) and once with the
// production resolver input (rowSupportInput). Every row's status and the
// ranked order must agree.
func TestSharedResolverMatchesTheOldHooksOnTheCatalog(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	t.Cleanup(func() {
		recommend.SetInstalledArchSupport(nil)
		recommend.SetBackendRoute(nil)
		recommend.SetBackendSupport(nil)
	})
	caps := &detect.Capabilities{OS: "linux", RAM: detect.RAMInfo{TotalMB: 262144, FreeMB: 250000}}
	// Arch-dependent answers so rows of both kinds occur.
	probes := map[string]backendArchProbe{
		"half load":  func(path, arch string) (bool, bool) { return len(arch)%2 == 0, true },
		"none loads": func(path, arch string) (bool, bool) { return false, true },
		"unreadable": func(path, arch string) (bool, bool) { return false, path != "/a" },
	}
	paths := []string{"/a", "/b"}
	key := func(c recommend.Categories) (map[string]recommend.BackendStatus, string) {
		m := map[string]recommend.BackendStatus{}
		var order []string
		for _, rows := range [][]recommend.Recommendation{c.Balanced, c.Smartest, c.Fastest} {
			for _, r := range rows {
				m[r.Repo+"|"+r.QuantName] = r.BackendStatus
				order = append(order, r.Repo+"|"+r.QuantName)
			}
			order = append(order, "/")
		}
		return m, strings.Join(order, ",")
	}
	for name, probe := range probes {
		for _, goos := range []string{"linux", "windows"} {
			for _, pinned := range []bool{false, true} {
				for _, tools := range []bool{false, true} {
					onPath := tools && goos != "windows"
					recommend.SetBackendSupport(nil)
					recommend.SetInstalledArchSupport(func(_ *detect.Capabilities, arch string) (bool, bool) {
						return archLoadedBy(paths, arch, probe)
					})
					recommend.SetBackendRoute(func(_ *detect.Capabilities, arch string, mainline bool) bool {
						return backendBuildRoute(arch, mainline, pinned, buildCanStart(onPath, false, "cuda"))
					})
					oldStatus, oldOrder := key(recommend.TopCategories(caps, 200))
					recommend.SetBackendSupport(func(_ *detect.Capabilities, r recommend.Recommendation) backends.Support {
						in := rowSupportInput(r, installedBackendsFor(paths, r.Arch, probe, nil, true), buildBlockedReason(goos, pinned, onPath, false, "cuda"))
						return backends.ResolveSupport(in)
					})
					newStatus, newOrder := key(recommend.TopCategories(caps, 200))
					if len(oldStatus) == 0 {
						t.Fatal("no rows")
					}
					for k, v := range oldStatus {
						if newStatus[k] != v {
							t.Fatalf("%s/%s pinned=%v tools=%v: %s old %s new %s", name, goos, pinned, tools, k, v, newStatus[k])
						}
					}
					if oldOrder != newOrder {
						t.Fatalf("%s/%s pinned=%v tools=%v: ranking changed", name, goos, pinned, tools)
					}
				}
			}
		}
	}
}

func TestRecommendationNoteLinesStayInTheTableWidth(t *testing.T) {
	note := strings.Repeat("word ", 60)
	lines := recommendationNoteLines(note)
	if len(lines) < 3 {
		t.Fatalf("not wrapped: %q", lines)
	}
	for _, l := range lines {
		if len(l) > recommendationNoteWidth || !strings.HasPrefix(l, "      ") {
			t.Fatalf("line %q", l)
		}
	}
	if recommendationNoteLines("") != nil {
		t.Fatal("empty note prints nothing")
	}
}

// A backend's name in a note is its tag or detected name, never its path.
func TestInstalledBackendsNeverNamedByPath(t *testing.T) {
	probe := func(path, _ string) (bool, bool) { return true, true }
	got := installedBackendsFor([]string{"/x/llama-server", "/y/llama-server"}, "llama", probe, map[string]string{"/y/llama-server": "fork"}, false)
	if len(got) != 2 || got[0].Name != "" || got[1].Name != "fork" {
		t.Fatalf("names: %+v", got)
	}
	if stopped := installedBackendsFor([]string{"/x", "/y"}, "llama", probe, nil, true); len(stopped) != 1 {
		t.Fatalf("did not stop at the first loader: %+v", stopped)
	}
	s := backends.ResolveSupport(backends.SupportInput{Arch: "llama", Installed: got[:1]})
	if strings.Contains(s.Note(), "/x/") {
		t.Fatalf("path in note: %q", s.Note())
	}
}
