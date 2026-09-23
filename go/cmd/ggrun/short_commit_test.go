package main

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// A short pin must resolve to the full commit, including one behind the tip.
func TestShortCommitPinResolvesToTheFullID(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	dir := t.TempDir()
	git := func(d string, args ...string) string {
		cmd := exec.Command("git", append([]string{"-c", "user.email=t@t", "-c", "user.name=t"}, args...)...)
		cmd.Dir = d
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	work := filepath.Join(dir, "work")
	git(dir, "init", "-q", "-b", "master", work)
	git(work, "commit", "-q", "--allow-empty", "-m", "one")
	pinned := git(work, "rev-parse", "HEAD")
	git(work, "commit", "-q", "--allow-empty", "-m", "two")
	bare := filepath.Join(dir, "origin.git")
	git(dir, "clone", "-q", "--bare", work, bare)
	src := filepath.Join(dir, "src")
	git(dir, "clone", "-q", "--depth", "1", "file://"+bare, src)

	full, err := resolveShortCommit(src, "master", pinned[:7])
	if err != nil || full != pinned {
		t.Fatalf("short pin %s resolved to %q (err %v), want %s", pinned[:7], full, err, pinned)
	}
	if _, err := resolveShortCommit(src, "master", "0000000"); err == nil {
		t.Fatal("an unknown short commit resolved")
	}
}
