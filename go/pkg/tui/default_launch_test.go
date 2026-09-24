package tui

import (
	"reflect"
	"testing"
)

// TUIDefaultLaunchArgs is the argv a fresh TUI emits for a model with every
// setting left at its default. cmd/ggrun proves it resolves to exactly the same
// launch as plain `ggrun <model>`; keep the two tests in step.
var TUIDefaultLaunchArgsTail = []string{"--port", "8081", "--ctx-size", "fit", "--kv-placement", "auto", "--support-expert", "auto"}

func TestFreshTUIDefaultLaunchArgs(t *testing.T) {
	modelPath := isolatedModelDir(t)
	m := InitialModel()
	m.models = []ModelItem{{Name: "tiny", Path: modelPath}}
	m.selectedModel = 0
	req := m.buildLaunchRequest()
	if req == nil {
		t.Fatal("no launch request")
	}
	want := append([]string{modelPath}, TUIDefaultLaunchArgsTail...)
	if got := req.LaunchArgs(); !reflect.DeepEqual(got, want) {
		t.Fatalf("fresh TUI default argv changed:\n got  %q\n want %q\nupdate the CLI equivalence test in cmd/ggrun as well", got, want)
	}
}
