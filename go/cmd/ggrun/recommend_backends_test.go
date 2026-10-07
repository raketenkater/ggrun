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
