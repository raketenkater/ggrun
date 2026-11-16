package claudesession

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// I-06 contract with the installed Claude Code: for real transcripts on this
// machine, the project directory ggrun derives from the session's working
// directory is the one Claude Code wrote. Skipped where no transcripts exist
// (CI).
func TestDerivedProjectDirMatchesInstalledClaudeTranscripts(t *testing.T) {
	projects := defaultProjectsDir()
	entries, err := os.ReadDir(projects)
	if err != nil || len(entries) == 0 {
		t.Skip("no Claude Code transcripts on this machine")
	}
	checked := 0
	for _, e := range entries {
		if !e.IsDir() || checked >= 20 {
			continue
		}
		files, _ := filepath.Glob(filepath.Join(projects, e.Name(), "*.jsonl"))
		for _, f := range files {
			cwd := transcriptCwd(f)
			if cwd == "" {
				continue
			}
			id := strings.TrimSuffix(filepath.Base(f), ".jsonl")
			if !TranscriptPresent(projects, cwd, id) {
				t.Errorf("transcript %s (cwd %s) is not where ggrun derives it (%s)", f, cwd, ProjectKey(cwd))
			}
			checked++
			break
		}
	}
	if checked == 0 {
		t.Skip("no transcript recorded a working directory")
	}
	t.Logf("checked %d projects", checked)
}

func transcriptCwd(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	for i := 0; sc.Scan() && i < 200; i++ {
		var line struct {
			Cwd string `json:"cwd"`
		}
		if json.Unmarshal(sc.Bytes(), &line) == nil && line.Cwd != "" {
			return line.Cwd
		}
	}
	return ""
}

func TestTranscriptPresent(t *testing.T) {
	projects := t.TempDir()
	work := "/home/someone/project"
	dir := filepath.Join(projects, ProjectKey(work))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "abc.jsonl"), []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !TranscriptPresent(projects, work, "abc") || TranscriptPresent(projects, work, "missing") || TranscriptPresent("", work, "abc") {
		t.Fatal("presence wrong")
	}
}
