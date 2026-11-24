package claudeauto

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// Claude Code 2.1.331 sends two classifier prompt variants: the default asks
// for <block> verdicts, and with telemetry/nonessential traffic disabled it
// asks for <severity>N</severity> (captured 2026-11-24). Both put the marker in
// the second system block, after the billing header block.
func variantClassifierBody(stop string) string {
	return `{"model":"m","stop_sequences":["` + stop + `"],"system":[` +
		`{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.331.94f; cc_entrypoint=sdk-cli;"},` +
		`{"type":"text","text":"` + ClassifierMarker + `\n\n## Context\n\nThe agent is autonomous."}],` +
		`"messages":[{"role":"user","content":"rm -rf /"}]}`
}

func TestBothClassifierPromptVariantsAreRecognised(t *testing.T) {
	for _, stop := range []string{"</block>", "</severity>"} {
		if !IsClassifierRequest([]byte(variantClassifierBody(stop))) {
			t.Errorf("classifier request with stop %s not recognised", stop)
		}
	}
	// The main-model request carries no marker.
	if IsClassifierRequest([]byte(`{"system":[{"type":"text","text":"You are Claude Code."}],"messages":[]}`)) {
		t.Fatal("main request treated as a classifier")
	}
}

// Contract with every Claude Code version installed under the standard
// install directory: its binary still carries the marker and both verdict
// formats. Skipped where none are installed (CI).
func TestInstalledClaudeVersionsCarryBothClassifierVariants(t *testing.T) {
	dir := os.Getenv("CLAUDE_VERSIONS_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".local", "share", "claude", "versions")
	}
	bins, _ := filepath.Glob(filepath.Join(dir, "*"))
	if p, err := exec.LookPath("claude"); err == nil {
		if real, err := filepath.EvalSymlinks(p); err == nil {
			bins = append(bins, real)
		}
	}
	seen := map[string]bool{}
	checked := 0
	for _, bin := range bins {
		if seen[bin] {
			continue
		}
		seen[bin] = true
		info, err := os.Stat(bin)
		if err != nil || info.IsDir() || info.Size() < 1<<20 {
			continue
		}
		data, err := os.ReadFile(bin)
		if err != nil {
			continue
		}
		checked++
		name := filepath.Base(bin)
		if !bytes.Contains(data, []byte(ClassifierMarker)) {
			t.Errorf("Claude Code %s lost the classifier marker %q; Auto reviews would be routed as main-model work", name, ClassifierMarker)
		}
		for _, tag := range []string{"</block>", "</severity>"} {
			if !bytes.Contains(data, []byte(tag)) {
				t.Errorf("Claude Code %s no longer mentions the %s verdict format; recheck the classifier stop-sequence contract", name, tag)
			}
		}
	}
	if checked == 0 {
		t.Skip("no installed Claude Code binary found")
	}
	t.Logf("checked %d installed Claude Code binaries", checked)
}
