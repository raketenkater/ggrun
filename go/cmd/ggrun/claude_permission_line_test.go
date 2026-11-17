package main

import (
	"strings"
	"testing"
)

// S-01: the launch states which permission mode Claude Code runs in.
func TestClaudePermissionModeLine(t *testing.T) {
	if got := claudePermissionModeLine([]string{"--permission-mode", "auto", "--x"}); !strings.Contains(got, "permission mode: auto") {
		t.Fatalf("auto: %q", got)
	}
	if got := claudePermissionModeLine([]string{"--permission-mode=acceptEdits"}); !strings.HasSuffix(got, "permission mode: acceptEdits") {
		t.Fatalf("equals form: %q", got)
	}
	if got := claudePermissionModeLine(nil); !strings.Contains(got, "from your Claude settings") {
		t.Fatalf("inherited: %q", got)
	}
}
