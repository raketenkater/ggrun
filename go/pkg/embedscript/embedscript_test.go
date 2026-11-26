package embedscript

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func redirect(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	old := CacheDir
	CacheDir = func() (string, error) { return dir, nil }
	t.Cleanup(func() { CacheDir = old })
	return dir
}

func TestExtractIsContentAddressedAndRepairsTheCache(t *testing.T) {
	dir := redirect(t)
	p, err := Extract("helper.py", "print('a')\n")
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Dir(p) != dir || !strings.HasPrefix(filepath.Base(p), "helper-") || filepath.Ext(p) != ".py" {
		t.Fatalf("unexpected path %s", p)
	}
	if err := os.WriteFile(p, []byte("print('tampered')\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	again, err := Extract("helper.py", "print('a')\n")
	if err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(again); string(data) != "print('a')\n" {
		t.Fatalf("edited cache entry was reused: %q", data)
	}
	other, err := Extract("helper.py", "print('b')\n")
	if err != nil {
		t.Fatal(err)
	}
	if other == again {
		t.Fatal("different content shares a cache path")
	}
}

func TestExtractRejectsEmptySource(t *testing.T) {
	redirect(t)
	if _, err := Extract("helper.py", ""); err == nil {
		t.Fatal("empty source extracted")
	}
}
