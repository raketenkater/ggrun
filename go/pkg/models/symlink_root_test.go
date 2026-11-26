package models

import (
	"os"
	"path/filepath"
	"testing"
)

// A models directory that is itself a symlink (to a larger disk) is listed;
// a symlinked directory below it is followed too.
func TestListFollowsSymlinkedRootAndNestedFolder(t *testing.T) {
	base := t.TempDir()
	real := filepath.Join(base, "disk", "models")
	other := filepath.Join(base, "other")
	for _, d := range []string{real, other} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(real, "a.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(other, "b.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(other, filepath.Join(real, "nested-link")); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(base, "home-models")
	if err := os.Symlink(real, link); err != nil {
		t.Fatal(err)
	}
	got, err := List(link)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Name != "a.gguf" || got[1].Name != "nested-link/b.gguf" {
		t.Fatalf("listed %+v, want a.gguf and nested-link/b.gguf", got)
	}
}

func TestListSymlinkLoopTerminatesWithoutDuplicates(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "models")
	sub := filepath.Join(root, "sub")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sub, "m.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	// sub/up -> root (loop) and root/again -> sub (duplicate route).
	if err := os.Symlink(root, filepath.Join(sub, "up")); err != nil {
		t.Skip("symlinks unavailable:", err)
	}
	if err := os.Symlink(sub, filepath.Join(root, "again")); err != nil {
		t.Fatal(err)
	}
	got, err := List(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("listed %+v, want exactly one model", got)
	}
}
