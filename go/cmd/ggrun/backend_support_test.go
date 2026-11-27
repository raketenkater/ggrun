package main

import (
	"fmt"
	"testing"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/recommend"
)

// legacyRowStatus is the status markBackendStatus gave a row before the shared
// resolver (its installed-arch and build-route branch, verbatim).
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
						Installed:         installedBackendsFor(paths, arch, probe, nil),
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
