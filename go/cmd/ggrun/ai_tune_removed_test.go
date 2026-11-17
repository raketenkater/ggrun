package main

import (
	"strings"
	"testing"
)

// D6: AI Tune was removed; its flags stop with that reason instead of being
// passed to the backend or silently ignored. --rounds stays: spec-test uses it.
func TestAITuneFlagsAreRefusedWithTheReason(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	for _, args := range [][]string{
		{"m.gguf", "--ai-tune"}, {"m.gguf", "--retune"}, {"m.gguf", "--show-configs"},
		{"m.gguf", "--tune-cache", "x.json"}, {"m.gguf", "--tune-cache=x.json"},
	} {
		if _, err := parseLaunchArgs(args); err == nil || !strings.Contains(err.Error(), "AI Tune was removed") {
			t.Errorf("%v: %v", args, err)
		}
	}
	req, err := parseLaunchArgs([]string{"m.gguf", "--rounds", "3"})
	if err != nil {
		t.Fatalf("--rounds refused: %v", err)
	}
	for _, a := range req.ExtraArgs {
		if a == "--rounds" || a == "3" {
			t.Fatalf("--rounds reached the backend: %v", req.ExtraArgs)
		}
	}
	if n, err := roundsFromArgs([]string{"--rounds=4"}, 2); err != nil || n != 4 {
		t.Fatalf("roundsFromArgs = %d %v", n, err)
	}
}
