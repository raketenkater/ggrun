package detect

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// resetBandwidthScriptCache restores the extraction singleton and cache
// location so each test observes a cold start.
func resetBandwidthScriptCache(t *testing.T, dir string) {
	t.Helper()
	prev := bandwidthScriptCacheDir
	bandwidthScriptCacheDir = func() (string, error) { return dir, nil }
	extractOnce = sync.Once{}
	extractPath, extractErr = "", nil
	t.Cleanup(func() {
		bandwidthScriptCacheDir = prev
		extractOnce = sync.Once{}
		extractPath, extractErr = "", nil
	})
}

// clearScriptEnv removes every repository-dependent lookup root so the test
// sees what a bare `go install` binary sees.
func clearScriptEnv(t *testing.T) {
	t.Helper()
	for _, key := range []string{"LLM_SCRIPT_DIR", "LLM_SERVER_HOME", "LLM_APP_HOME"} {
		t.Setenv(key, "")
	}
}

// The invariant this whole component exists for: `ggrun detect --bandwidth`
// must resolve its probe with no repository, no installer and no env vars.
// Before the embed, this returned "" on every `go install` binary, which is
// why measured bandwidth evidence never existed in the field.
func TestBandwidthScriptResolvesWithNoRepositoryPresent(t *testing.T) {
	dir := t.TempDir()
	resetBandwidthScriptCache(t, dir)
	clearScriptEnv(t)

	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if err := os.Chdir(t.TempDir()); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(wd) })

	path := findBandwidthScript()
	if path == "" {
		t.Fatal("findBandwidthScript returned empty with no repo present; the shipped command cannot work on a bare install")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("resolved script %s is not readable: %v", path, err)
	}
}

func TestEmbeddedScriptCarriesTheProbeContract(t *testing.T) {
	if strings.TrimSpace(measureBandwidthSource) == "" {
		t.Fatal("embedded measure_bandwidth.py is empty; the build is broken")
	}
	// MeasureBandwidth passes these flags; if the embedded copy ever drifts
	// away from that contract the command fails at runtime, not at build time.
	for _, flag := range []string{"--bytes", "--min-iterations", "--host-workers", "--gpu-bus-id"} {
		if !strings.Contains(measureBandwidthSource, flag) {
			t.Errorf("embedded script does not accept %s, but MeasureBandwidth passes it", flag)
		}
	}
}

func TestExtractedScriptMatchesEmbeddedBytes(t *testing.T) {
	dir := t.TempDir()
	resetBandwidthScriptCache(t, dir)

	path, err := extractBandwidthScript()
	if err != nil {
		t.Fatalf("extract: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read extracted script: %v", err)
	}
	if string(got) != measureBandwidthSource {
		t.Fatal("extracted script differs from embedded source")
	}
}

func TestExtractionIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	resetBandwidthScriptCache(t, dir)

	first, err := writeEmbeddedBandwidthScript()
	if err != nil {
		t.Fatalf("first extract: %v", err)
	}
	second, err := writeEmbeddedBandwidthScript()
	if err != nil {
		t.Fatalf("second extract: %v", err)
	}
	if first != second {
		t.Fatalf("extraction path moved between calls: %s vs %s", first, second)
	}
	// No stray .partial files may survive a successful write.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	for _, e := range entries {
		if strings.Contains(e.Name(), ".partial") {
			t.Errorf("leftover staging file %s", e.Name())
		}
	}
}

// Failure path: a truncated cache entry, e.g. from an interrupted write, must
// be repaired rather than executed as if it were the real probe.
func TestTruncatedCacheEntryIsRepaired(t *testing.T) {
	dir := t.TempDir()
	resetBandwidthScriptCache(t, dir)

	path, err := writeEmbeddedBandwidthScript()
	if err != nil {
		t.Fatalf("seed extract: %v", err)
	}
	if err := os.WriteFile(path, []byte("# truncated\n"), 0o755); err != nil {
		t.Fatalf("corrupt: %v", err)
	}

	repaired, err := writeEmbeddedBandwidthScript()
	if err != nil {
		t.Fatalf("repair: %v", err)
	}
	got, err := os.ReadFile(repaired)
	if err != nil {
		t.Fatalf("read repaired: %v", err)
	}
	if string(got) != measureBandwidthSource {
		t.Fatal("truncated cache entry was reused instead of repaired")
	}
}

// An on-disk copy must continue to win, so editing the repo's script still
// takes effect for developers and packaged installs override the embed.
func TestOnDiskCopyWinsOverEmbedded(t *testing.T) {
	dir := t.TempDir()
	resetBandwidthScriptCache(t, dir)
	clearScriptEnv(t)

	override := t.TempDir()
	want := filepath.Join(override, "measure_bandwidth.py")
	if err := os.WriteFile(want, []byte("# local override\n"), 0o755); err != nil {
		t.Fatalf("write override: %v", err)
	}
	t.Setenv("LLM_SCRIPT_DIR", override)

	if got := findBandwidthScript(); got != want {
		t.Fatalf("on-disk copy did not win: got %s, want %s", got, want)
	}
}

// Failure path: when the cache cannot be created the user must see the real
// cause, not a generic "not found" implying a broken install.
func TestUnwritableCacheReportsRealCause(t *testing.T) {
	parent := t.TempDir()
	blocked := filepath.Join(parent, "blocked")
	if err := os.WriteFile(blocked, []byte("not a directory"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	resetBandwidthScriptCache(t, filepath.Join(blocked, "scripts"))

	if _, err := writeEmbeddedBandwidthScript(); err == nil {
		t.Fatal("expected an error when the script cache cannot be created")
	} else if !strings.Contains(err.Error(), "script cache") {
		t.Fatalf("error does not name the real cause: %v", err)
	}
}

// The digest must track content, so an upgraded ggrun never runs a stale
// extraction left behind by an older build.
func TestDigestIsContentAddressed(t *testing.T) {
	first := embeddedBandwidthScriptDigest()
	if len(first) != 16 {
		t.Fatalf("digest length %d, want 16", len(first))
	}

	prev := measureBandwidthSource
	measureBandwidthSource = prev + "\n# a later release\n"
	t.Cleanup(func() { measureBandwidthSource = prev })

	if second := embeddedBandwidthScriptDigest(); second == first {
		t.Fatal("digest did not change when the embedded script changed; a stale extraction would be reused")
	}
}
