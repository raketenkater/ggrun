package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// stubParser installs a parse_gguf.py that logs each call to a file and
// returns fixed metadata, so a test can count how often headers are parsed.
func stubParser(t *testing.T) (logPath string) {
	t.Helper()
	dir := t.TempDir()
	script := "META = {'tensor_accounting_schema': 3}\n" +
		"import json, os, sys\n" +
		"with open(os.environ['GGRUN_TEST_PARSE_LOG'], 'a') as f:\n" +
		"    f.write(os.path.basename(sys.argv[1]) + '\\n')\n" +
		"print(json.dumps({'tensor_accounting_schema': 3, 'arch': 'fake', 'layers': 4, 'ctx_train': 4096}))\n"
	if err := os.WriteFile(filepath.Join(dir, "parse_gguf.py"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	logPath = filepath.Join(t.TempDir(), "parse.log")
	t.Setenv("LLM_SCRIPT_DIR", dir)
	t.Setenv("GGRUN_TEST_PARSE_LOG", logPath)
	t.Setenv("LLM_APP_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	return logPath
}

func parseLogLines(t *testing.T, logPath string) []string {
	t.Helper()
	data, err := os.ReadFile(logPath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	return strings.Fields(string(data))
}

func TestEnrichModelItemsReusesCachedHeaders(t *testing.T) {
	logPath := stubParser(t)
	modelDir := t.TempDir()
	cacheDir := t.TempDir()
	names := []string{"a.gguf", "b-draft.gguf", "c.gguf", "d.gguf", "e.gguf"}
	base := time.Now().Add(-time.Hour)
	items := func() []ModelItem {
		out := make([]ModelItem, 0, len(names))
		for _, name := range names {
			out = append(out, ModelItem{Name: name, Path: filepath.Join(modelDir, name), Arch: "dense"})
		}
		return out
	}
	for _, name := range names {
		path := filepath.Join(modelDir, name)
		if err := os.WriteFile(path, []byte("GGUF-"+name), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, base, base); err != nil {
			t.Fatal(err)
		}
	}

	first := enrichModelItems(items(), cacheDir, "auto", nil)
	if got := len(parseLogLines(t, logPath)); got != len(names) {
		t.Fatalf("first start parsed %d headers, want %d", got, len(names))
	}
	want := []string{"a.gguf", "c.gguf", "d.gguf", "e.gguf"}
	if len(first) != len(want) {
		t.Fatalf("enriched %d models, want %d (auxiliary draft model must be dropped)", len(first), len(want))
	}
	for i, name := range want {
		if first[i].Name != name || first[i].Architecture != "fake" || first[i].MaxCtx != 4096 {
			t.Fatalf("model %d = %+v, want %s enriched with the parsed header, in input order", i, first[i], name)
		}
	}

	second := enrichModelItems(items(), cacheDir, "auto", nil)
	if got := len(parseLogLines(t, logPath)); got != len(names) {
		t.Fatalf("second start parsed %d more headers, want none", got-len(names))
	}
	if len(second) != len(first) {
		t.Fatalf("cached start listed %d models, first listed %d", len(second), len(first))
	}
	for i := range first {
		if second[i].Name != first[i].Name || second[i].Arch != first[i].Arch ||
			second[i].MaxCtx != first[i].MaxCtx || second[i].ChatTemplate != first[i].ChatTemplate ||
			second[i].IsMoE != first[i].IsMoE {
			t.Fatalf("cached model %d differs: %+v vs %+v", i, second[i], first[i])
		}
	}

	// A file that changed on disk is parsed again; the others stay cached.
	changed := filepath.Join(modelDir, "c.gguf")
	if err := os.WriteFile(changed, []byte("GGUF-c.gguf-replaced"), 0o644); err != nil {
		t.Fatal(err)
	}
	enrichModelItems(items(), cacheDir, "auto", nil)
	lines := parseLogLines(t, logPath)
	if len(lines) != len(names)+1 || lines[len(lines)-1] != "c.gguf" {
		t.Fatalf("after changing c.gguf the parse log is %v, want exactly one more entry for c.gguf", lines)
	}
}
