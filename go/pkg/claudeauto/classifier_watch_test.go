package claudeauto

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func bashTurn(n int) []byte {
	var calls []string
	for i := 0; i < n; i++ {
		calls = append(calls, `{"type":"tool_use","id":"t`+string(rune('a'+i))+`","name":"Bash","input":{"command":"ls"}}`)
	}
	return []byte(`{"messages":[{"role":"user","content":"go"},{"role":"assistant","content":[` + strings.Join(calls, ",") +
		`]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"ta","content":"ok"}]}]}`)
}

// I-04: Auto mode with Bash calls but no recognised review is reported once.
func TestClassifierWatchWarnsOnceWhenReviewsAreNeverSeen(t *testing.T) {
	var out bytes.Buffer
	w := &classifierWatch{out: &out}
	w.expect.Store(true)
	w.sawMain(bashTurn(1))
	w.sawMain(bashTurn(1))
	if out.Len() != 0 {
		t.Fatalf("warned after 2 calls: %s", out.String())
	}
	w.sawMain(bashTurn(1))
	w.sawMain(bashTurn(2))
	if strings.Count(out.String(), "warning: Auto mode is on") != 1 {
		t.Fatalf("want exactly one warning, got %q", out.String())
	}
}

func TestClassifierWatchQuietWhenReviewsArriveOrNotAuto(t *testing.T) {
	var out bytes.Buffer
	w := &classifierWatch{out: &out}
	w.expect.Store(true)
	w.sawClassifier()
	for i := 0; i < 5; i++ {
		w.sawMain(bashTurn(1))
	}
	off := &classifierWatch{out: &out}
	for i := 0; i < 5; i++ {
		off.sawMain(bashTurn(1))
	}
	if out.Len() != 0 {
		t.Fatalf("warned: %s", out.String())
	}
	if n := lastTurnToolUses([]byte(`{"messages":[{"role":"assistant","content":"text only"}]}`), "Bash"); n != 0 {
		t.Fatalf("string content counted %d", n)
	}
}

// Contract with the installed Claude Code: its binary still carries the
// marker IsClassifierRequest looks for. Skipped where Claude Code is absent
// (CI); run on a machine with the tested version before qualifying it.
func TestInstalledClaudeCarriesTheClassifierMarker(t *testing.T) {
	path, err := exec.LookPath("claude")
	if err != nil {
		t.Skip("claude not installed")
	}
	if real, err := filepath.EvalSymlinks(path); err == nil {
		path = real
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skip(err)
	}
	version, _ := exec.Command(path, "--version").Output()
	if !bytes.Contains(data, []byte(ClassifierMarker)) {
		t.Fatalf("Claude Code %s no longer contains the classifier marker %q; Auto reviews would be routed as main-model work", strings.TrimSpace(string(version)), ClassifierMarker)
	}
	t.Logf("marker present in Claude Code %s", strings.TrimSpace(string(version)))
}
