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
	t.Setenv(downloaderEnv, "")
}

func writeHelper(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
}

// Bug #12: a helper left by an older install rejected --non-interactive. A
// stale copy in LLM_SCRIPT_DIR, LLM_SERVER_HOME or the app home no longer
// wins over the embedded downloader that matches this binary.
func TestFindScriptRunsTheEmbeddedCopyOverStaleInstalls(t *testing.T) {
	cache := redirectScriptCache(t)
	clearScriptEnv(t)
	scripts, server, app := t.TempDir(), t.TempDir(), t.TempDir()
	writeHelper(t, filepath.Join(scripts, "download_any_gguf.py"), "# old helper without --non-interactive")
	writeHelper(t, filepath.Join(server, "tools", "download", "download_any_gguf.py"), "# old")
	writeHelper(t, filepath.Join(app, ".bin", "download_any_gguf.py"), "# old")
	t.Setenv("LLM_SCRIPT_DIR", scripts)
	t.Setenv("LLM_SERVER_HOME", server)
	t.Setenv("LLM_APP_HOME", app)
	got := findScript(app)
	if filepath.Dir(got) != cache {
		t.Fatalf("got %s, want the embedded copy under %s", got, cache)
	}
	if data, err := os.ReadFile(got); err != nil || string(data) != downloaderSource {
		t.Fatalf("resolved downloader is not the embedded one (err %v)", err)
	}
}

// GGRUN_DOWNLOADER is the explicit choice and is run as given, even when it
// differs from the embedded copy.
func TestFindScriptHonoursTheConfiguredDownloader(t *testing.T) {
	redirectScriptCache(t)
	clearScriptEnv(t)
	own := filepath.Join(t.TempDir(), "my_downloader.py")
	writeHelper(t, own, "# work in progress")
	t.Setenv("LLM_SCRIPT_DIR", t.TempDir())
	t.Setenv(downloaderEnv, own)
	if got := findScript(""); got != own {
		t.Fatalf("found %s, want the configured %s", got, own)
	}
	d := &Downloader{ScriptPath: own, ModelDir: "/m", CacheDir: "/c"}
	if args := d.scriptArgs("org/repo", "", nil, true); args[0] != own {
		t.Fatalf("argv runs %s, want %s", args[0], own)
	}
}

// When the cache cannot be written, only an installed copy with the embedded
// bytes is used; a differing one is never a fallback.
func TestFindScriptFallsBackOnlyToAnIdenticalInstall(t *testing.T) {
	clearScriptEnv(t)
	blocker := filepath.Join(t.TempDir(), "file")
	writeHelper(t, blocker, "not a directory")
	old := embedscript.CacheDir
	embedscript.CacheDir = func() (string, error) { return filepath.Join(blocker, "scripts"), nil }
	t.Cleanup(func() { embedscript.CacheDir = old })

	stale, same := t.TempDir(), t.TempDir()
	writeHelper(t, filepath.Join(stale, "download_any_gguf.py"), "# old")
	t.Setenv("LLM_SCRIPT_DIR", stale)
	if got := findScript(""); got != "" {
		t.Fatalf("a differing installed copy was selected: %s", got)
	}
	identical := filepath.Join(same, ".bin", "download_any_gguf.py")
	writeHelper(t, identical, downloaderSource)
	if got := findScript(same); got != identical {
		t.Fatalf("found %q, want the identical install %s", got, identical)
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
