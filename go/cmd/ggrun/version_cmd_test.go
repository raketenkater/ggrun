package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The plain form is parsed by installers and update checks; it must not change.
func TestVersionPlainFormatUnchanged(t *testing.T) {
	read := captureStdout(t)
	cmdVersion(nil)
	if got := read(); got != "ggrun "+version+"\n" {
		t.Fatalf("plain version output changed: %q", got)
	}
}

func TestVersionJSONCarriesIdentity(t *testing.T) {
	read := captureStdout(t)
	cmdVersion([]string{"--json"})
	got := read()
	var id buildIdentity
	if err := json.Unmarshal([]byte(strings.TrimSpace(got)), &id); err != nil {
		t.Fatalf("version --json is not JSON: %q: %v", got, err)
	}
	if id.Version != version || id.GoVersion == "" || id.Platform == "" {
		t.Fatalf("identity incomplete: %+v", id)
	}
	// Existing keys stay; executable_resolved names the bytes that answered.
	var keys map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(got)), &keys); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"version", "go_version", "platform", "executable", "executable_resolved"} {
		if _, ok := keys[key]; !ok {
			t.Fatalf("version --json lost %q: %s", key, got)
		}
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	want, err := filepath.EvalSymlinks(exe)
	if err != nil {
		t.Fatal(err)
	}
	if id.ExecutableResolved != want || !filepath.IsAbs(id.ExecutableResolved) {
		t.Fatalf("executable_resolved = %q, want %q", id.ExecutableResolved, want)
	}
}

// An entry point that is a symlink must resolve to the file whose bytes run:
// that file, not the link or a wrapper, is what an install check hashes.
func TestResolvedExecutableFollowsSymlinks(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "ggrun-build")
	if err := os.WriteFile(target, []byte("build"), 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "ggrun")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	want, err := filepath.EvalSymlinks(target)
	if err != nil {
		t.Fatal(err)
	}
	if got := resolvedExecutable(link); got != want {
		t.Fatalf("resolvedExecutable(%q) = %q, want %q", link, got, want)
	}
	if got := resolvedExecutable(filepath.Join(dir, "missing")); got != "" {
		t.Fatalf("an unresolvable executable must report no path, got %q", got)
	}
}
