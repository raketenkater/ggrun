package tui

import (
	"reflect"
	"strings"
	"testing"
)

func TestClaudeModeCyclesAndKeepsOldLocalMeaning(t *testing.T) {
	m := Model{}
	m.cycleClaudeMode(1)
	if !m.claudeCode || m.claudeHeadless {
		t.Fatal("first x must preserve local Claude mode")
	}
	m.resumeSession = "local-session"
	m.cycleClaudeMode(1)
	if !m.claudeCode || !m.claudeHeadless || m.resumeSession != "" {
		t.Fatal("remote mode must not resume a server-local session")
	}
	m.cycleClaudeMode(1)
	if m.claudeCode || m.claudeHeadless {
		t.Fatal("third x must return to Off")
	}
	m.cycleClaudeMode(-1)
	if !m.claudeCode || !m.claudeHeadless {
		t.Fatal("reverse cycle inconsistent")
	}
}

func TestRemoteTUILaunchRoundTrip(t *testing.T) {
	dir := t.TempDir()
	r := &LaunchRequest{ModelPath: "model.gguf", ClaudeCode: true, ClaudeHeadless: true, GatewayHost: "127.0.0.1", GatewayPort: 18080, GatewayURL: "https://models.example/prefix", GatewayTokenFile: "private-key"}
	if err := SaveLatestLaunch(dir, r); err != nil {
		t.Fatal(err)
	}
	got, _, err := LoadLatestLaunch(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(r.LaunchArgs(), got.LaunchArgs()) {
		t.Fatalf("saved args changed: %v vs %v", r.LaunchArgs(), got.LaunchArgs())
	}
	args := strings.Join(got.LaunchArgs(), " ")
	for _, want := range []string{"--claude-code", "--headless", "--gateway-port 18080", "--gateway-url https://models.example/prefix"} {
		if !strings.Contains(args, want) {
			t.Errorf("missing %s", want)
		}
	}
	old := &LaunchRequest{ModelPath: "model.gguf", ClaudeCode: true}
	if err := SaveLatestLaunch(dir, old); err != nil {
		t.Fatal(err)
	}
	got, _, err = LoadLatestLaunch(dir)
	if err != nil || got.ClaudeHeadless {
		t.Fatal("legacy local launch restored as remote")
	}
}
