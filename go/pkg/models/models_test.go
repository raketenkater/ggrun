package models

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func writeModel(t *testing.T, root, name string, size int) {
	t.Helper()
	path := filepath.Join(root, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestListGroupsShardsAndSorts(t *testing.T) {
	root := t.TempDir()
	writeModel(t, root, "zeta.gguf", 11)
	writeModel(t, root, "nested/alpha-00001-of-00002.gguf", 7)
	writeModel(t, root, "nested/alpha-00002-of-00002.gguf", 13)
	writeModel(t, root, "nested/notes.txt", 100)

	got, err := List(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("models = %#v, want two", got)
	}
	if got[0].Name != filepath.Join("nested", "alpha.gguf") || got[0].Bytes != 20 || len(got[0].Files) != 2 {
		t.Fatalf("grouped shards = %#v", got[0])
	}
	if got[1].Name != "zeta.gguf" || got[1].Bytes != 11 {
		t.Fatalf("single model = %#v", got[1])
	}
}

func TestRemoveDeletesOnlySelectedShardedModel(t *testing.T) {
	root := t.TempDir()
	writeModel(t, root, "bundle-00001-of-00002.gguf", 7)
	writeModel(t, root, "bundle-00002-of-00002.gguf", 13)
	writeModel(t, root, "keep.gguf", 5)

	removed, err := Remove(root, "bundle.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if removed.Bytes != 20 || len(removed.Files) != 2 {
		t.Fatalf("removed = %#v", removed)
	}
	for _, shard := range removed.Files {
		if _, err := os.Lstat(filepath.Join(root, shard)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("shard %s still exists: %v", shard, err)
		}
	}
	if _, err := os.Stat(filepath.Join(root, "keep.gguf")); err != nil {
		t.Fatalf("unrelated model was removed: %v", err)
	}
}

func TestRemoveRejectsTraversalAndUnknownModels(t *testing.T) {
	root := t.TempDir()
	writeModel(t, root, "keep.gguf", 5)

	for _, name := range []string{"../keep.gguf", "not-a-model", filepath.Join(root, "keep.gguf")} {
		if _, err := Remove(root, name); err == nil {
			t.Fatalf("Remove(%q) succeeded", name)
		}
	}
	if _, err := Remove(root, "missing.gguf"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing model error = %v, want ErrNotFound", err)
	}
}

func TestRemoveRejectsFilesThroughSymlinkedDirectory(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink permissions are not portable on Windows CI")
	}
	root := t.TempDir()
	outside := t.TempDir()
	writeModel(t, outside, "outside.gguf", 5)
	if err := os.Symlink(outside, filepath.Join(root, "outside")); err != nil {
		t.Fatal(err)
	}

	if _, err := Remove(root, filepath.Join("outside", "outside.gguf")); err == nil {
		t.Fatal("Remove followed a symlinked directory outside root")
	}
	if _, err := os.Stat(filepath.Join(outside, "outside.gguf")); err != nil {
		t.Fatalf("outside model was removed: %v", err)
	}
}
