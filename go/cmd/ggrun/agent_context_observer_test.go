package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

// TestObserverCountsCachedPrefixAsContext is the measurement this whole path
// depends on. An agent turn sends a few hundred new tokens on top of a cached
// prefix of tens of thousands; the cached prefix occupies KV exactly like fresh
// tokens. Counting only input_tokens would under-measure real agent context by
// two orders of magnitude and size the deployment far too small.
func TestObserverCountsCachedPrefixAsContext(t *testing.T) {
	dir := t.TempDir()
	line := `{"usage":{"input_tokens":312,"cache_read_input_tokens":74000,"cache_creation_input_tokens":4646}}`
	if err := os.WriteFile(filepath.Join(dir, "ggrun-claude-requests-8081.jsonl"),
		[]byte(line+"\n"), 0o644); err != nil {
		t.Fatalf("seed log: %v", err)
	}
	d, err := observeAgentContextDemand(dir)
	if err != nil {
		t.Fatalf("observe: %v", err)
	}
	if want := 312 + 74000 + 4646; d.MaxTokens != want {
		t.Fatalf("context must include the cached prefix: got %d, want %d", d.MaxTokens, want)
	}
}

// TestObserverSurvivesUnparsableLines keeps a launch from failing on a log that
// is mid-write, truncated, or carries a line this build does not understand.
func TestObserverSurvivesUnparsableLines(t *testing.T) {
	dir := t.TempDir()
	body := "not json at all\n" +
		`{"usage":{"input_tokens":1000}}` + "\n" +
		`{"no_usage":true}` + "\n" +
		`{"usage":{"input_tokens":5000,"cache_read_input_tokens":1}}` + "\n" +
		`{"usage":{"input_tokens":` // deliberately truncated final line
	if err := os.WriteFile(filepath.Join(dir, "ggrun-claude-requests-9.jsonl"), []byte(body), 0o644); err != nil {
		t.Fatalf("seed log: %v", err)
	}
	d, err := observeAgentContextDemand(dir)
	if err != nil {
		t.Fatalf("a damaged log must not fail the launch: %v", err)
	}
	if d.Samples != 2 {
		t.Fatalf("expected the 2 parsable rows, got %d", d.Samples)
	}
	if d.MaxTokens != 5001 {
		t.Fatalf("expected max 5001, got %d", d.MaxTokens)
	}
}

// TestObserverIgnoresEmptyAndMissingDirs pins the no-history case: a fresh
// install must observe nothing and leave automatic sizing exactly as it was.
func TestObserverIgnoresEmptyAndMissingDirs(t *testing.T) {
	for _, dir := range []string{"", t.TempDir(), filepath.Join(t.TempDir(), "nope")} {
		d, err := observeAgentContextDemand(dir)
		if err != nil {
			t.Fatalf("dir %q: %v", dir, err)
		}
		if d.Samples != 0 {
			t.Fatalf("dir %q should yield no samples, got %d", dir, d.Samples)
		}
		if got := placement.AgentContextCeiling(d, 1); got != 0 {
			t.Fatalf("dir %q must offer no ceiling, got %d", dir, got)
		}
	}
}
