package main

import (
	"runtime"
	"testing"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestParseBackendAutoRestoresAutomaticSelection(t *testing.T) {
	isolateConfig(t)
	t.Setenv("LLM_BACKEND", "llama")
	for _, flags := range [][]string{{"--backend", "auto"}, {"--backend=auto"}, {"--backend", "llama", "--backend=auto"}} {
		req, err := parseLaunchArgs(append([]string{"model.gguf"}, flags...))
		if err != nil {
			t.Fatal(err)
		}
		if req.Backend != "auto" || req.BackendExplicit {
			t.Fatalf("%v must restore automatic architecture routing, got backend=%q explicit=%v", flags, req.Backend, req.BackendExplicit)
		}
	}
	for _, name := range []string{"llama", "ik_llama", "custom", "skip"} {
		for _, flags := range [][]string{{"--backend", name}, {"--backend=" + name}} {
			req, err := parseLaunchArgs(append([]string{"model.gguf"}, flags...))
			if err != nil {
				t.Fatal(err)
			}
			if !req.BackendExplicit {
				t.Fatalf("named selection %v must stay explicit", flags)
			}
		}
	}
}

// Exercise parsing followed by the same selection used by launch. Manually
// constructing an automatic request misses the explicit-auto parser regression.
func TestBackendAutoFlagSelectsRegisteredModelFork(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake backends use shell scripts")
	}
	isolateConfig(t)
	t.Setenv("LLM_APP_HOME", t.TempDir())
	t.Setenv("LLM_BACKEND", "llama")
	fork := writeFakeBackend(t, "model-fork", "echo 'llama server help'\n")
	pinned := writeFakeBackend(t, "pinned-server", "echo 'llama server help'\n")
	if err := backends.Save([]backends.Backend{
		{Tag: "model-fork", Path: fork, RouteArch: "test_fork_arch"},
		{Tag: "pinned", Path: pinned},
	}); err != nil {
		t.Fatal(err)
	}
	model := &placement.ModelProfile{ModelArch: "test_fork_arch"}
	for _, tc := range []struct {
		flags []string
		want  string
	}{
		{[]string{"--backend", "auto"}, fork},
		{[]string{"--backend=auto"}, fork},
		{[]string{"--backend", "pinned"}, pinned},
		{[]string{"--backend=auto", "--server-bin", pinned}, pinned},
	} {
		req, err := parseLaunchArgs(append([]string{"model.gguf"}, tc.flags...))
		if err != nil {
			t.Fatal(err)
		}
		got := selectBackendForModel(&detect.Capabilities{}, req, model)
		if got == nil || got.Path != tc.want {
			t.Fatalf("%v selected %#v, want %s", tc.flags, got, tc.want)
		}
	}
}
