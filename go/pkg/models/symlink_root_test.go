package models

import (
	"os"
	"path/filepath"
	"testing"
)

// A models directory that is itself a symlink (to a larger disk) is listed;
// a symlinked directory below it is still not followed.
func TestListFollowsASymlinkedRootOnly(t *testing.T) {
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
	if len(got) != 1 || got[0].Name != "a.gguf" || got[0].Files[0] != "a.gguf" {
		t.Fatalf("listed %+v, want only a.gguf from the linked root", got)
	}
}
