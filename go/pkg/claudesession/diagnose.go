package claudesession

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Diagnose reports drift between Claude Code's private session files and what
// resume reads from them (review I-06), beyond TranscriptPresent: a transcript
// with no token-usage rows, a recorded workflow run whose journal is not at the
// expected path, run definitions with no journal beside them, and a journal
// whose rows carry no recognised "result" entry. Each string is one warning;
// none means the layout looks as expected (or there is nothing to check).
func Diagnose(projectsDir, workDir, sessionID, runID string) []string {
	if projectsDir == "" || sessionID == "" {
		return nil
	}
	dir := projectDir(projectsDir, workDir, sessionID)
	var out []string
	transcript := filepath.Join(dir, sessionID+".jsonl")
	if info, err := os.Stat(transcript); err == nil && info.Size() > 64<<10 {
		if _, ok := CurrentContextTokens(projectsDir, workDir, sessionID); !ok {
			out = append(out, fmt.Sprintf("transcript %s has no assistant token-usage rows; Claude Code may have renamed its usage fields, so this resume cannot be sized from it", transcript))
		}
	}
	sessionDir := filepath.Join(dir, sessionID)
	if runID != "" {
		journal := JournalPath(projectsDir, workDir, sessionID, runID)
		if _, err := os.Stat(journal); err != nil {
			out = append(out, fmt.Sprintf("recorded workflow run %s has no journal at %s; cached agent results cannot be counted", runID, journal))
		} else if !JournalShapeOK(journal) {
			out = append(out, fmt.Sprintf("workflow journal %s has no rows with a \"type\" field; Claude Code may have changed its journal format", journal))
		}
	}
	defs, _ := filepath.Glob(filepath.Join(sessionDir, "workflows", "*.json"))
	if len(defs) > 0 {
		journals, _ := filepath.Glob(filepath.Join(sessionDir, "subagents", "workflows", "*", "journal.jsonl"))
		if len(journals) == 0 {
			out = append(out, fmt.Sprintf("session has %d workflow run definition(s) under %s but no journal under subagents/workflows; Claude Code may have moved its workflow journals", len(defs), filepath.Join(sessionDir, "workflows")))
		}
	}
	return out
}

// JournalShapeOK reports whether a non-empty journal contains rows ggrun can
// classify: every parsed line has a "type", and a journal that started agents
// carries "result" rows only if some agent finished. It returns false only
// when rows exist and none has a "type" field at all.
func JournalShapeOK(journalPath string) bool {
	body, err := os.ReadFile(journalPath)
	if err != nil {
		return true
	}
	rows := 0
	for _, line := range strings.Split(string(body), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		rows++
		if strings.Contains(line, `"type"`) {
			return true
		}
	}
	return rows == 0
}
