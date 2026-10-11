package main

import "testing"

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
		upstream, pinned, tools bool
		want                    bool
	}{
		{"upstream arch builds current mainline", "k2-horizon", true, false, true, true},
		{"reviewed recipe", "hy_v3", false, false, true, true},
		{"reviewed helper recipe pins a commit", "nanbeige", false, false, true, true},
		{"fork-only arch without a recipe", "inkling", false, false, true, false},
		{"mainline cannot serve an ik-only arch", "minimax-m9", true, false, true, false},
		{"configured backend turns the offers off", "k2-horizon", true, true, true, false},
		{"no build tools", "hy_v3", true, false, false, false},
	}
	for _, c := range cases {
		if got := backendBuildRoute(c.arch, c.upstream, c.pinned, c.tools); got != c.want {
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
