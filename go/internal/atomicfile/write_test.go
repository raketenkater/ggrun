package atomicfile

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestFailedEncodingPreservesStateAndCleansStage(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "state")
	if err := os.WriteFile(path, []byte("old complete state"), 0600); err != nil {
		t.Fatal(err)
	}
	failure := errors.New("disk write failed")
	err := Write(path, 0644, func(w io.Writer) error {
		if _, err := io.WriteString(w, "partial replacement"); err != nil {
			t.Fatal(err)
		}
		return failure
	})
	if !errors.Is(err, failure) {
		t.Fatalf("lost write error: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil || string(got) != "old complete state" {
		t.Fatalf("old state lost: %q %v", got, err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 1 {
		t.Fatalf("staged file leaked: %v %v", entries, err)
	}
}

func TestWritePreservesPermissionsAndSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "state")
	if err := os.WriteFile(target, []byte("old"), 0600); err != nil {
		t.Fatal(err)
	}
	path := target
	if runtime.GOOS != "windows" {
		path = filepath.Join(dir, "linked")
		if err := os.Symlink("state", path); err != nil {
			t.Fatal(err)
		}
	}
	if err := Write(path, 0644, func(w io.Writer) error { _, err := io.WriteString(w, "complete new state"); return err }); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(target)
	if err != nil || string(got) != "complete new state" {
		t.Fatalf("target: %q %v", got, err)
	}
	if runtime.GOOS != "windows" {
		info, err := os.Stat(target)
		if err != nil || info.Mode().Perm() != 0600 {
			t.Fatalf("permissions changed: %v %v", info, err)
		}
		if _, err := os.Readlink(path); err != nil {
			t.Fatalf("symlink replaced: %v", err)
		}
	}
}

func TestCommitFailureCleansStage(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "state")
	err := Write(path, 0600, func(w io.Writer) error {
		// A competing filesystem mutation prevents promotion after staging.
		if err := os.Mkdir(path, 0700); err != nil {
			return err
		}
		_, err := io.WriteString(w, "new")
		return err
	})
	if err == nil {
		t.Fatal("rename failure reported success")
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 1 || !entries[0].IsDir() {
		t.Fatalf("stage cleanup failed: %v %v", entries, err)
	}
}
