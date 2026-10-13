package download

import (
	"os"
	"path/filepath"
	"testing"
)

// A stale downloader in the working directory (a model directory) must not
// shadow the installed one.
func TestFindScriptPrefersTheInstalledCopy(t *testing.T) {
	t.Setenv("LLM_SERVER_HOME", "")
	work, app := t.TempDir(), t.TempDir()
	if err := os.WriteFile(filepath.Join(work, "download_any_gguf.py"), []byte("# old"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(app, ".bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	installed := filepath.Join(app, ".bin", "download_any_gguf.py")
	if err := os.WriteFile(installed, []byte("# bundled"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Chdir(work)
	if got := findScript(app); got != installed {
		t.Fatalf("found %s, want the installed %s", got, installed)
	}
}
