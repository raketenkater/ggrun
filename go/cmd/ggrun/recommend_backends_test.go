package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/recommend"
)

func TestArchLoadedByNeedsEveryBackendProbedToClaimABuild(t *testing.T) {
	probe := func(answers map[string][2]bool) backendArchProbe {
		return func(path, _ string) (bool, bool) { a := answers[path]; return a[0], a[1] }
	}
	cases := []struct {
		name         string
		answers      map[string][2]bool
		loads, known bool
	}{
		{"one backend loads it", map[string][2]bool{"a": {false, true}, "b": {true, true}}, true, true},
		{"all probed, none loads", map[string][2]bool{"a": {false, true}, "b": {false, true}}, false, true},
		{"one unprobeable", map[string][2]bool{"a": {false, true}, "b": {false, false}}, false, false},
	}
	for _, c := range cases {
		loads, known := archLoadedBy([]string{"a", "b"}, "k2-horizon", probe(c.answers))
		if loads != c.loads || known != c.known {
			t.Fatalf("%s: got loads=%v known=%v", c.name, loads, known)
		}
	}
	if _, known := archLoadedBy(nil, "k2-horizon", probe(nil)); known {
		t.Fatal("no installed backend is not evidence that a build is needed")
	}
}

// A row counts as needing setup only when a launch really offers a build for
// it; otherwise it is unsupported and ranks last.
func TestBackendBuildRouteFollowsTheLaunchOffers(t *testing.T) {
	cases := []struct {
		name                    string
		arch                    string
		mainline, pinned, tools bool
		want                    bool
	}{
		{"mainline arch builds current mainline", "k2-horizon", true, false, true, true},
		{"ik-only arch has no mainline build", "deepseek41", false, false, true, false},
		{"reviewed recipe", "hy_v3", false, false, true, true},
		{"reviewed helper recipe pins a commit", "nanbeige", false, false, true, true},
		{"fork-only arch without a recipe", "inkling", false, false, true, false},
		{"mainline cannot serve an ik-only arch", "minimax-m9", true, false, true, false},
		{"configured backend turns the offers off", "k2-horizon", true, true, true, false},
		{"no build tools", "hy_v3", true, false, false, false},
	}
	for _, c := range cases {
		if got := backendBuildRoute(c.arch, c.mainline, c.pinned, c.tools); got != c.want {
			t.Fatalf("%s: got %v", c.name, got)
		}
	}
}

func TestBuildToolsPresentNeedsGitCMakeAndACompiler(t *testing.T) {
	have := func(names ...string) func(string) bool {
		return func(name string) bool {
			for _, n := range names {
				if n == name {
					return true
				}
			}
			return false
		}
	}
	if !buildToolsPresent("linux", have("git", "cmake", "clang++")) {
		t.Fatal("git, cmake and clang++ are enough")
	}
	if buildToolsPresent("linux", have("git", "cmake")) || buildToolsPresent("linux", have("cmake", "g++")) {
		t.Fatal("a missing compiler or git counted as a build route")
	}
	if buildToolsPresent("windows", have("git", "cmake", "c++")) {
		t.Fatal("Windows source builds produce no bin/llama-server")
	}
}

// An NVIDIA host builds with CUDA, so without nvcc no build starts there.
func TestBuildCanStartNeedsCUDAOnAnNVIDIAHost(t *testing.T) {
	for _, c := range []struct {
		tools, nvidia bool
		accel         string
		want          bool
	}{
		{true, false, "cpu", true},
		{true, true, "cuda", true},
		{true, true, "cpu", false},
		{false, false, "cpu", false},
		{false, true, "cuda", false},
	} {
		if got := buildCanStart(c.tools, c.nvidia, c.accel); got != c.want {
			t.Fatalf("tools=%v nvidia=%v accel=%s: got %v", c.tools, c.nvidia, c.accel, got)
		}
	}
}

// The CLI marks each status that is not simply ready, and a probed status
// outranks the catalog's fork stamp.
func TestRecommendationMarkerPerBackendStatus(t *testing.T) {
	no := false
	fork := recommend.Candidate{Arch: "inkling", Runnable: &no}
	for _, c := range []struct {
		rec  recommend.Recommendation
		want string
	}{
		{recommend.Recommendation{BackendStatus: recommend.BackendReady, BackendHint: "CUDA / ik_llama"}, ""},
		{recommend.Recommendation{BackendStatus: recommend.BackendNeedsSetup}, " +"},
		{recommend.Recommendation{BackendStatus: recommend.BackendReady, BackendHint: "Vulkan", CUDABuildOffered: true}, " ^"},
		{recommend.Recommendation{BackendStatus: recommend.BackendReady, BackendHint: "Vulkan"}, " ^"},
		{recommend.Recommendation{Candidate: fork, BackendStatus: recommend.BackendUnsupported}, " !"},
		{recommend.Recommendation{Candidate: fork, BackendStatus: recommend.BackendUnknown}, " *"},
		{recommend.Recommendation{Candidate: fork, BackendStatus: recommend.BackendNeedsSetup}, " +"},
	} {
		got := recommendationMarker(c.rec)
		if got != c.want {
			t.Fatalf("%+v: marker %q, want %q", c.rec, got, c.want)
		}
		if got != "" && recommendationFootnotes[got] == "" {
			t.Fatalf("marker %q has no footnote", got)
		}
	}
}

// `recommend --cpu` sees no GPU, so it offers a build without nvcc; the --cpu
// launch on the same NVIDIA host must then build the CPU backend, not CUDA.
func TestCPUOnlyBuildRouteMatchesTheLaunchBuild(t *testing.T) {
	host := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "NVIDIA GeForce RTX 3090 Ti"}}}
	recommendCaps := &detect.Capabilities{}
	if !buildCanStart(true, hostHasNVIDIAGPU(recommendCaps), "cpu") {
		t.Fatal("recommend --cpu offers no build route without nvcc")
	}
	caps, accel := archBuildTarget(&launchRequest{CPUMode: true}, host, "cpu")
	if r := acceleratedArchBuildRecipe("k2-horizon", caps, accel); r == nil || r.Accel != "cpu" {
		t.Fatalf("--cpu launch builds %+v, want the CPU backend recommend offered", r)
	}
	if len(host.GPUs) != 1 {
		t.Fatal("archBuildTarget changed the detected capabilities")
	}
	// A GPU launch on the same host still builds CUDA.
	caps, accel = archBuildTarget(&launchRequest{}, host, "cpu")
	if r := acceleratedArchBuildRecipe("k2-horizon", caps, accel); r == nil || r.Accel != "cuda" {
		t.Fatalf("GPU launch builds %+v, want CUDA", r)
	}
}
