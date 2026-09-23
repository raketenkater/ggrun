package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/tui"
)

// The TUI's default launch goes through cmdLaunch with its own argv. It must
// parse to the same request as `ggrun <model>`, or the TUI user gets different
// fallbacks, recovery and cached tuning than the CLI default.
func TestTUIDefaultLaunchParsesLikeCLIDefault(t *testing.T) {
	home := t.TempDir()
	t.Setenv("LLM_APP_HOME", home)
	model := filepath.Join(home, "m.gguf")
	if err := os.WriteFile(model, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	cli, err := parseLaunchArgs([]string{model})
	if err != nil {
		t.Fatal(err)
	}
	// What the TUI builds with default settings: port, fit context and KV
	// placement from config, backend "auto", support expert from config.
	tuiReq := &tui.LaunchRequest{ModelPath: model, Port: cli.Port, CtxFlag: "fit", KVPlacement: cli.KVPlacement,
		Backend: "auto", FlashAttn: true, Parallel: 1, SupportExpert: cli.SupportExpert}
	fromTUI, err := parseLaunchArgs(tuiReq.LaunchArgs())
	if err != nil {
		t.Fatal(err)
	}
	if a, b := requestedLaunchPolicyIdentity(cli, nil), requestedLaunchPolicyIdentity(fromTUI, nil); a != b {
		t.Fatalf("TUI default keys tuning differently:\n cli %s\n tui %s", a, b)
	}
	if backendChoiceExplicit(fromTUI) != backendChoiceExplicit(cli) || fromTUI.SupportOnlineSet != cli.SupportOnlineSet ||
		fromTUI.SupportOnline != cli.SupportOnline || fromTUI.CtxFlag != cli.CtxFlag || fromTUI.KVPlacement != cli.KVPlacement {
		t.Fatalf("TUI default differs from CLI default:\n cli %+v\n tui %+v", cli, fromTUI)
	}
}
