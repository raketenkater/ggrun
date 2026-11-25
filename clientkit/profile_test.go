package clientkit

import (
	"encoding/json"
	"os"
	"strconv"
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

func TestOutputBudget(t *testing.T) {
	for _, c := range []struct{ ctx, want int }{
		{-1, 0}, {0, 0}, {2048, 4096}, {40960, 5120}, {77824, 9728}, {99999, 12288}, {100000, 0}, {262144, 0},
	} {
		if got := OutputBudget(c.ctx); got != c.want {
			t.Errorf("OutputBudget(%d) = %d, want %d", c.ctx, got, c.want)
		}
	}
}

func TestEnvironmentOutputBudget(t *testing.T) {
	const key = "CLAUDE_CODE_MAX_OUTPUT_TOKENS"
	mk := func(ctx int) Profile {
		return Profile{MainAlias: "local", UtilityAlias: "local-fast", ContextPerSlot: ctx, Slots: 1, Policy: Policy{Version: 1, CompactWindow: ctx, CompactPercent: 75, Effort: "xhigh", TimeoutMS: 1000}}
	}
	get := func(env []string) string {
		for _, kv := range env {
			if v, ok := strings.CutPrefix(kv, key+"="); ok {
				return v
			}
		}
		return "<absent>"
	}
	for _, c := range []struct {
		name string
		ctx  int
		in   []string
		want string
	}{
		{"small slot", 40960, nil, "5120"},
		{"large slot absent", 262144, nil, "<absent>"},
		{"inherited valid kept", 40960, []string{key + "=8000"}, "8000"},
		{"inherited text", 40960, []string{key + "=abc"}, "5120"},
		{"inherited zero", 40960, []string{key + "=0"}, "5120"},
		{"inherited at context", 40960, []string{key + "=40960"}, "5120"},
		{"inherited above context", 40960, []string{key + "=50000"}, "5120"},
		{"large slot passthrough", 262144, []string{key + "=9999"}, "9999"},
	} {
		if got := get(mk(c.ctx).Environment("http://x", "t", c.in)); got != c.want {
			t.Errorf("%s: %s = %s, want %s", c.name, key, got, c.want)
		}
	}
}

func TestCompactionEstimate(t *testing.T) {
	for _, c := range []struct {
		name          string
		ctx, out      int
		pct           string
		wantAt, wantW int
	}{
		{"40960 default", 40960, 0, "75", 7960, 40960},
		{"40960 out4096", 40960, 4096, "75", 23864, 40960},
		{"77824 default", 77824, 0, "75", 43368, 77824},
		{"77824 out4096", 77824, 4096, "75", 55296, 77824},
		{"77824 70", 77824, 0, "70", 40476, 77824},
		{"77824 no pct", 77824, 0, "", 44824, 77824},
		{"40960 out5120", 40960, 5120, "75", 22840, 40960},
		{"tiny floors at 0", 10000, 0, "", 0, 10000},
	} {
		env := []string{"CLAUDE_CODE_MAX_CONTEXT_TOKENS=" + itoa(c.ctx)}
		if c.out > 0 {
			env = append(env, "CLAUDE_CODE_MAX_OUTPUT_TOKENS="+itoa(c.out))
		}
		if c.pct != "" {
			env = append(env, "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE="+c.pct)
		}
		if at, w := CompactionEstimate(env); at != c.wantAt || w != c.wantW {
			t.Errorf("%s: got (%d, %d), want (%d, %d)", c.name, at, w, c.wantAt, c.wantW)
		}
	}
	if at, w := CompactionEstimate([]string{"PATH=/bin"}); at != 0 || w != 0 {
		t.Errorf("no window: got (%d, %d)", at, w)
	}
	// The compaction window overrides the context variable.
	if _, w := CompactionEstimate([]string{"CLAUDE_CODE_MAX_CONTEXT_TOKENS=100000", "CLAUDE_CODE_AUTO_COMPACT_WINDOW=50000"}); w != 50000 {
		t.Errorf("window = %d, want 50000", w)
	}
}

func itoa(n int) string { return strconv.Itoa(n) }
