package claudesession

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeSessionFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestDiagnoseQuietForTheExpectedLayout(t *testing.T) {
	projects, work, id := t.TempDir(), "/w/p", "s1"
	dir := filepath.Join(projects, ProjectKey(work))
	writeSessionFile(t, filepath.Join(dir, id+".jsonl"),
		`{"message":{"usage":{"input_tokens":5,"cache_read_input_tokens":10}}}`+"\n"+strings.Repeat(" ", 70<<10))
	writeSessionFile(t, filepath.Join(dir, id, "workflows", "wf_1.json"), `{"runId":"wf_1"}`)
	writeSessionFile(t, filepath.Join(dir, id, "subagents", "workflows", "wf_1", "journal.jsonl"),
		`{"type":"launched"}`+"\n"+`{"type":"result"}`+"\n")
	if got := Diagnose(projects, work, id, "wf_1"); len(got) != 0 {
		t.Fatalf("warned on a healthy layout: %v", got)
	}
	if got := Diagnose("", work, id, ""); got != nil {
		t.Fatalf("no projects dir must be silent: %v", got)
	}
}

func TestDiagnoseReportsMovedFilesAndRenamedFields(t *testing.T) {
	projects, work, id := t.TempDir(), "/w/p", "s2"
	dir := filepath.Join(projects, ProjectKey(work))
	// Usage fields renamed: a large transcript yields no measurement.
	writeSessionFile(t, filepath.Join(dir, id+".jsonl"),
		`{"message":{"usage":{"inputTokens":5}}}`+"\n"+strings.Repeat(" ", 70<<10))
	// Run definition present, journals moved elsewhere.
	writeSessionFile(t, filepath.Join(dir, id, "workflows", "wf_9.json"), `{"runId":"wf_9"}`)
	got := strings.Join(Diagnose(projects, work, id, "wf_9"), "\n")
	for _, want := range []string{"token-usage rows", "has no journal at", "moved its workflow journals"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing warning %q in:\n%s", want, got)
		}
	}
	writeSessionFile(t, filepath.Join(dir, id, "subagents", "workflows", "wf_9", "journal.jsonl"), `{"kind":"x"}`+"\n")
	if got := strings.Join(Diagnose(projects, work, id, "wf_9"), "\n"); !strings.Contains(got, `no rows with a "type" field`) {
		t.Errorf("journal format change not reported: %s", got)
	}
}

// I-06 contract with the installed Claude Code's own session files: usage rows
// parse, and every workflow journal it wrote is found by LatestRun/CachedAgents
// with its definition file beside it. Skipped where no such session exists.
func TestInstalledClaudeSessionFilesMatchWhatResumeReads(t *testing.T) {
	projects := defaultProjectsDir()
	transcripts, _ := filepath.Glob(filepath.Join(projects, "*", "*.jsonl"))
	if len(transcripts) == 0 {
		t.Skip("no Claude Code transcripts on this machine")
	}
	usageChecked, journals := 0, 0
	for _, f := range transcripts {
		if info, err := os.Stat(f); err != nil || info.Size() < 4<<10 {
			continue
		}
		cwd := transcriptCwd(f)
		if cwd == "" {
			continue
		}
		id := strings.TrimSuffix(filepath.Base(f), ".jsonl")
		if rawUsageTotal(f) > 0 {
			if _, ok := CurrentContextTokens(projects, cwd, id); !ok && usageChecked < 50 {
				t.Errorf("transcript %s carries non-zero usage rows but CurrentContextTokens found none", f)
			}
			usageChecked++
		}
		sessionDir := filepath.Join(filepath.Dir(f), id)
		matches, _ := filepath.Glob(filepath.Join(sessionDir, "subagents", "workflows", "*", "journal.jsonl"))
		for _, j := range matches {
			journals++
			run := filepath.Base(filepath.Dir(j))
			if got := JournalPath(projects, cwd, id, run); got != j {
				t.Errorf("JournalPath = %s, Claude Code wrote %s", got, j)
			}
			if !JournalShapeOK(j) {
				t.Errorf("journal %s has no typed rows", j)
			}
			if CachedAgents(j) > 0 {
				if wf, n := LatestRun(projects, cwd, id); wf == nil || n == 0 {
					t.Errorf("LatestRun found nothing for %s which has cached results", j)
				}
			}
		}
		if usageChecked >= 50 && journals >= 20 {
			break
		}
	}
	t.Logf("checked %d transcripts with usage, %d workflow journals", usageChecked, journals)
}

// rawUsageTotal sums the three input-token fields of every transcript row by
// an independent generic parse, so it does not share CurrentContextTokens' types.
func rawUsageTotal(path string) int {
	f, err := os.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	total := 0
	for sc.Scan() {
		var row map[string]any
		if json.Unmarshal(sc.Bytes(), &row) != nil {
			continue
		}
		msg, _ := row["message"].(map[string]any)
		usage, _ := msg["usage"].(map[string]any)
		for _, k := range []string{"input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"} {
			if v, ok := usage[k].(float64); ok {
				total += int(v)
			}
		}
	}
	return total
}
