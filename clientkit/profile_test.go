package clientkit

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestVersionedContractFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/profile-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var p Profile
	if err = json.Unmarshal(data, &p); err != nil {
		t.Fatal(err)
	}
	if err = p.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestBaseURLTransportIndependent(t *testing.T) {
	for _, raw := range []string{"https://models.example.com/inference/", "http://127.0.0.1:8090", "http://[::1]:8090"} {
		got, err := BaseURL(raw, false)
		if err != nil || got != strings.TrimRight(raw, "/") {
			t.Fatalf("%s: %s %v", raw, got, err)
		}
	}
	for _, raw := range []string{"http://100.97.62.30:8080", "https://user:pass@host", "https://host/v1", "https://host/?key=x", "http://0.0.0.0:80", "https://host/#fragment", "file:///tmp/a"} {
		if _, err := BaseURL(raw, false); err == nil {
			t.Errorf("accepted %s", raw)
		}
	}
	if _, err := BaseURL("http://192.168.1.7:8080/prefix", true); err != nil {
		t.Fatal(err)
	}
}

func TestProfileAndEnvironment(t *testing.T) {
	p := Profile{ProtocolVersion: 1, ModelID: "opaque-model", MainAlias: "local", UtilityAlias: "local-fast", ContextPerSlot: 65536, Slots: 1, Policy: Policy{Version: 1, CompactWindow: 65536, CompactPercent: 75, Effort: "xhigh", TimeoutMS: MaxTimeoutMS}}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	env := p.Environment("https://models.example/prefix", "secret", []string{"PATH=/usr/bin", "ANTHROPIC_API_KEY=cloud-key", "ANTHROPIC_MODEL=opus", "CLAUDE_CODE_AUTO_COMPACT_WINDOW=1000000", "CUSTOM_SETTING=yes"})
	all := strings.Join(env, "\n")
	for _, want := range []string{"ANTHROPIC_BASE_URL=https://models.example/prefix", "ANTHROPIC_MODEL=local", "ANTHROPIC_DEFAULT_HAIKU_MODEL=local-fast", "CLAUDE_CODE_AUTO_COMPACT_WINDOW=65536", "CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536", "CUSTOM_SETTING=yes"} {
		if !strings.Contains(all, want) {
			t.Errorf("missing %s", want)
		}
	}
	if strings.Contains(all, "cloud-key") {
		t.Fatal("inherited cloud credential retained")
	}
	chosen := strings.Join(p.Environment("http://127.0.0.1", "secret", []string{"CLAUDE_CODE_SUBAGENT_MODEL=haiku"}), "\n")
	if !strings.Contains(chosen, "CLAUDE_CODE_SUBAGENT_MODEL=haiku") {
		t.Fatal("explicit served subagent tier was lost")
	}
	p.Policy.CompactWindow = 65537
	if p.Validate() == nil {
		t.Fatal("accepted compaction beyond the actual context")
	}
}
