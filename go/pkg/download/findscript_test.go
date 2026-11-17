package download

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/embedscript"
)

func redirectScriptCache(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	old := embedscript.CacheDir
	embedscript.CacheDir = func() (string, error) { return dir, nil }
	t.Cleanup(func() { embedscript.CacheDir = old })
	return dir
}

func clearScriptEnv(t *testing.T) {
	t.Setenv("LLM_SCRIPT_DIR", "")
	t.Setenv("LLM_SERVER_HOME", "")
	t.Setenv("LLM_APP_HOME", "")
}

// A downloader in the working directory (a model directory) or on PATH must
// not be run (S-10); the installed copy wins.
func TestFindScriptPrefersTheInstalledCopy(t *testing.T) {
	redirectScriptCache(t)
	clearScriptEnv(t)
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

// With nothing installed, a planted copy in the working directory or on PATH
// is ignored and the embedded downloader is used.
func TestFindScriptIgnoresWorkingDirectoryAndPath(t *testing.T) {
	cache := redirectScriptCache(t)
	clearScriptEnv(t)
	planted := t.TempDir()
	for _, rel := range []string{"download_any_gguf.py", "tools/download/download_any_gguf.py"} {
		p := filepath.Join(planted, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("# planted"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PATH", planted)
	t.Chdir(planted)
	got := findScript("")
	if got == "" || strings.HasPrefix(got, planted) || !filepath.IsAbs(got) {
		t.Fatalf("planted downloader selected or nothing found: %q", got)
	}
	if filepath.Dir(got) != cache {
		t.Fatalf("got %s, want the embedded copy under %s", got, cache)
	}
	if data, err := os.ReadFile(got); err != nil || string(data) != downloaderSource {
		t.Fatalf("resolved downloader is not the embedded one (err %v)", err)
	}
}

// `go generate ./pkg/download` refreshes the embedded copy.
func TestEmbeddedDownloaderMatchesTools(t *testing.T) {
	tools, err := os.ReadFile(filepath.Join("..", "..", "..", "tools", "download", "download_any_gguf.py"))
	if err != nil {
		t.Fatal(err)
	}
	if string(tools) != downloaderSource {
		t.Fatal("go/pkg/download/download_any_gguf.py differs from tools/download/download_any_gguf.py; run `go generate ./pkg/download`")
	}
}

// For ~/.local/bin/ggrun nothing directly in ~ or ~/tools is a candidate.
func TestExeScriptCandidatesStayInTheInstall(t *testing.T) {
	home := t.TempDir()
	for _, exeDir := range []string{filepath.Join(home, ".local", "bin"), filepath.Join(home, "app", ".bin"), filepath.Join(home, "repo", "go")} {
		root := exeDir
		if b := filepath.Base(exeDir); b == "bin" || b == ".bin" {
			root = filepath.Dir(exeDir)
		}
		for _, c := range exeScriptCandidates(exeDir) {
			if !strings.HasPrefix(filepath.Clean(c), root+string(filepath.Separator)) {
				t.Errorf("exe %s: candidate %s leaves %s", exeDir, c, root)
			}
		}
	}
}
