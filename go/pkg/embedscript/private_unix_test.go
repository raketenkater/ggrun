//go:build !windows

package embedscript

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A cache directory other users can write is refused, not used: they could
// replace the extracted script before it runs.
func TestExtractRefusesASharedCacheDir(t *testing.T) {
	shared := filepath.Join(t.TempDir(), "scripts")
	if err := os.Mkdir(shared, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(shared, 0o777); err != nil {
		t.Fatal(err)
	}
	old := CacheDir
	CacheDir = func() (string, error) { return shared, nil }
	t.Cleanup(func() { CacheDir = old })
	_, err := Extract("helper.py", "print('a')\n")
	if err == nil || !strings.Contains(err.Error(), shared) || !strings.Contains(err.Error(), "writable by other users") {
		t.Fatalf("shared cache dir accepted or unclear error: %v", err)
	}
}

func TestExtractRefusesASymlinkedCacheDir(t *testing.T) {
	base := t.TempDir()
	target := filepath.Join(base, "real")
	if err := os.Mkdir(target, 0o700); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(base, "link")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	old := CacheDir
	CacheDir = func() (string, error) { return link, nil }
	t.Cleanup(func() { CacheDir = old })
	if _, err := Extract("helper.py", "print('a')\n"); err == nil || !strings.Contains(err.Error(), "not a directory") {
		t.Fatalf("symlinked cache dir accepted: %v", err)
	}
}

// Without a user cache directory the fallback is a per-user 0700 directory,
// not a shared name in the temp directory.
func TestExtractFallbackIsPerUserAndPrivate(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("TMPDIR", tmp)
	old := CacheDir
	CacheDir = func() (string, error) { return "", os.ErrNotExist }
	t.Cleanup(func() { CacheDir = old })
	p, err := Extract("helper.py", "print('a')\n")
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Dir(p)
	if filepath.Dir(dir) != tmp || filepath.Base(dir) == "ggrun-scripts" {
		t.Fatalf("fallback %s is not a per-user directory under %s", dir, tmp)
	}
	info, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o700 {
		t.Fatalf("fallback mode %o, want 0700", info.Mode().Perm())
	}
}
