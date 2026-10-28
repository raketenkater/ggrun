package claudesession

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestNewSessionIDIsAValidUUIDAndUnique(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 50; i++ {
		id, err := NewSessionID()
		if err != nil {
			t.Fatalf("NewSessionID: %v", err)
		}
		if !ValidSessionID(id) {
			t.Fatalf("generated id %q is not a valid session id", id)
		}
		// Claude Code requires a v4 UUID; a non-conforming id is rejected at
		// launch, which would silently lose the resume handle.
		if id[14] != '4' {
			t.Fatalf("id %q is not version 4", id)
		}
		if variant := id[19]; variant != '8' && variant != '9' && variant != 'a' && variant != 'b' {
			t.Fatalf("id %q has wrong variant nibble %q", id, variant)
		}
		if seen[id] {
			t.Fatalf("duplicate session id %q", id)
		}
		seen[id] = true
	}
}

func TestValidSessionIDRejectsPathEscapes(t *testing.T) {
	for _, id := range []string{
		"", "not-a-uuid", "../../etc/passwd",
		"../../../home/mik/.claude/creds.json",
		"12345678-1234-1234-1234-12345678901", // too short
		"12345678-1234-1234-1234-1234567890123",
		"12345678/1234-1234-1234-123456789012",
		"gggggggg-1234-1234-1234-123456789012",
	} {
		if ValidSessionID(id) {
			t.Errorf("ValidSessionID(%q) = true, want false", id)
		}
	}
	if !ValidSessionID("072e63a1-819a-4682-a742-559695c3cd76") {
		t.Error("rejected a real session id")
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	rec := Record{
		SessionID:  "072e63a1-819a-4682-a742-559695c3cd76",
		WorkDir:    "/home/mik/ggrun-project/ggrun",
		ModelPath:  "/models/Laguna.gguf",
		Port:       8081,
		ServerArgs: []string{"--ctx-size", "1048576", "--parallel", "4"},
		Workflow:   &Workflow{RunID: "wf_894b5285-5d3", Name: "deep-research"},
	}
	if err := Save(dir, rec); err != nil {
		t.Fatalf("Save: %v", err)
	}
	got, err := Load(dir, rec.SessionID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.ModelPath != rec.ModelPath || got.Port != rec.Port {
		t.Errorf("round trip lost fields: %+v", got)
	}
	if got.Workflow == nil || got.Workflow.RunID != "wf_894b5285-5d3" {
		t.Errorf("workflow not preserved: %+v", got.Workflow)
	}
	if got.Recorded.IsZero() {
		t.Error("Recorded not stamped on save")
	}
	// A partial write must not survive as a loadable record.
	if entries, _ := os.ReadDir(Dir(dir)); len(entries) != 1 {
		t.Errorf("want 1 record file, got %d", len(entries))
	}
}

func TestSaveRejectsInvalidSessionID(t *testing.T) {
	dir := t.TempDir()
	if err := Save(dir, Record{SessionID: "../escape"}); err == nil {
		t.Fatal("Save accepted an invalid session id")
	}
	if _, err := os.Stat(Dir(dir)); err == nil {
		t.Error("Save created the records directory for an invalid id")
	}
}

func TestListAndLatestAreScopedToWorkDirAndOrdered(t *testing.T) {
	dir := t.TempDir()
	older := Record{
		SessionID: "11111111-1111-4111-8111-111111111111",
		WorkDir:   "/project/a", Recorded: time.Now().Add(-2 * time.Hour),
	}
	newer := Record{
		SessionID: "22222222-2222-4222-8222-222222222222",
		WorkDir:   "/project/a", Recorded: time.Now().Add(-1 * time.Hour),
	}
	other := Record{
		SessionID: "33333333-3333-4333-8333-333333333333",
		WorkDir:   "/project/b", Recorded: time.Now(),
	}
	for _, rec := range []Record{older, newer, other} {
		if err := Save(dir, rec); err != nil {
			t.Fatalf("Save: %v", err)
		}
	}
	list, err := List(dir, "/project/a")
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(list) != 2 {
		t.Fatalf("want 2 records for /project/a, got %d", len(list))
	}
	if list[0].SessionID != newer.SessionID {
		t.Errorf("List not newest-first: got %s", list[0].SessionID)
	}
	latest, err := Latest(dir, "/project/a")
	if err != nil {
		t.Fatalf("Latest: %v", err)
	}
	if latest.SessionID != newer.SessionID {
		t.Errorf("Latest = %s, want %s", latest.SessionID, newer.SessionID)
	}
	// A different project must not resume into this one's session.
	if _, err := Latest(dir, "/project/c"); err == nil {
		t.Error("Latest returned a record for an unrelated work dir")
	}
}

func TestListOnMissingDirectoryIsEmptyNotAnError(t *testing.T) {
	list, err := List(t.TempDir(), "")
	if err != nil {
		t.Fatalf("List on empty cache: %v", err)
	}
	if len(list) != 0 {
		t.Errorf("want no records, got %d", len(list))
	}
}

// Recoverable must be true for a session with a recorded workflow pointer even
// when no projects directory exists yet, and false for a fresh live session.
func TestRecoverableSkipsALiveLaunchWithNoTranscript(t *testing.T) {
	withWF := Record{
		SessionID: "11111111-2222-4333-8444-555555555555",
		WorkDir:   "/project/a",
		Workflow:  &Workflow{RunID: "wf_894b5285-5d3"},
	}
	if !withWF.Recoverable() {
		t.Error("a session with a recorded workflow pointer must be recoverable")
	}
	fresh := Record{SessionID: "22222222-2222-4222-8222-222222222222", WorkDir: "/project/a"}
	if fresh.Recoverable() {
		t.Error("a fresh live session with no transcript/workflow must not be recoverable")
	}
}

func TestLatestRecoverableSkipsTheNewestEmptyRecord(t *testing.T) {
	dir := t.TempDir()
	fresh := Record{
		SessionID: "11111111-2222-4333-8444-555555555555",
		WorkDir:   "/project/a", Recorded: time.Now().Add(-time.Hour),
	}
	withWF := Record{
		SessionID: "22222222-2222-4222-8222-222222222222",
		WorkDir:   "/project/a", Recorded: time.Now().Add(-2 * time.Hour),
		Workflow: &Workflow{RunID: "wf_894b5285-5d3"},
	}
	if err := Save(dir, fresh); err != nil {
		t.Fatal(err)
	}
	if err := Save(dir, withWF); err != nil {
		t.Fatal(err)
	}
	got, err := LatestRecoverable(dir, "/project/a")
	if err != nil {
		t.Fatalf("LatestRecoverable: %v", err)
	}
	if got.SessionID != withWF.SessionID {
		t.Errorf("LatestRecoverable picked the empty live session %s, want %s",
			got.SessionID, withWF.SessionID)
	}
	if _, err := LatestRecoverable(dir, "/project/none"); err == nil {
		t.Error("LatestRecoverable returned a record for a workdir with nothing recoverable")
	}
}

func TestModelPathExistsReportsAMissingModel(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(good, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if !(Record{ModelPath: good}).ModelPathExists() {
		t.Error("an existing model path must report present")
	}
	if (Record{ModelPath: filepath.Join(dir, "gone.gguf")}).ModelPathExists() {
		t.Error("a missing model path must report absent")
	}
	// Nothing recorded means nothing to check; the launch path reports on its own.
	if !(Record{}).ModelPathExists() {
		t.Error("an empty recorded model path must not be reported missing")
	}
}

func TestProjectKeyMatchesClaudeCodeLayout(t *testing.T) {
	cases := map[string]string{
		"/home/mik/ggrun-project/ggrun":   "-home-mik-ggrun-project-ggrun",
		"/home/mik/ggrun/.src/llm-server": "-home-mik-ggrun--src-llm-server",
		"/home/mik/ultra-zen":             "-home-mik-ultra-zen",
		// Observed on disk: Claude Code recorded cwd /tmp/ggrun-cell-calc-_uy_2_42
		// under projects/-tmp-ggrun-cell-calc--uy-2-42.
		"/tmp/ggrun-cell-calc-_uy_2_42": "-tmp-ggrun-cell-calc--uy-2-42",
		`C:\Users\a b\my_app`:           "C--Users-a-b-my-app",
		`\\server\share\proj (v2)`:      "--server-share-proj--v2-",
		"/home/me/café/🚀":               "-home-me-caf----",
	}
	for in, want := range cases {
		if got := ProjectKey(in); got != want {
			t.Errorf("ProjectKey(%q) = %q, want %q", in, got, want)
		}
	}
}

// Long keys carry Claude Code's own hash suffix. The expected values were
// computed with the client's function copied from Claude Code 2.1.307
// (slice(0, 200) + "-" + Math.abs(hash).toString(36)) under node.
func TestProjectKeyHashesLongPathsLikeClaudeCode(t *testing.T) {
	deep := "/" + strings.Repeat("deep/", 50)
	cases := map[string]string{
		deep + "project-a": "aobq66",
		deep + "project-b": "aobq67",
		`C:\Users\me\` + strings.Repeat("x_y ", 60) + "\U0001F600end": "aw86qt",
	}
	for path, hash := range cases {
		key := ProjectKey(path)
		want := sanitizedPrefix(path) + "-" + hash
		if key != want {
			t.Errorf("ProjectKey(%.40q...) = ...%q, want ...%q", path, key[claudeMaxProjectKey:], want[claudeMaxProjectKey:])
		}
	}
	// Review 2026-10-28: two sibling projects under one long prefix must not
	// share a transcript directory.
	projects := t.TempDir()
	a, b := projectDir(projects, deep+"project-a"), projectDir(projects, deep+"project-b")
	if a == b {
		t.Fatalf("two distinct projects select the same transcript directory: %s", a)
	}
	if short := projectDir(projects, "/home/mik/repo"); short != filepath.Join(projects, "-home-mik-repo") {
		t.Fatalf("short key changed: %q", short)
	}
}

// sanitizedPrefix is the first claudeMaxProjectKey characters of the dashed key.
func sanitizedPrefix(path string) string {
	var b strings.Builder
	for _, r := range path {
		switch {
		case 'a' <= r && r <= 'z', 'A' <= r && r <= 'Z', '0' <= r && r <= '9':
			b.WriteRune(r)
		case r > 0xFFFF:
			b.WriteString("--")
		default:
			b.WriteByte('-')
		}
	}
	return b.String()[:claudeMaxProjectKey]
}

func TestJournalPathAndCachedAgents(t *testing.T) {
	projects := t.TempDir()
	workDir := "/home/mik/ggrun-project/ggrun"
	session := "072e63a1-819a-4682-a742-559695c3cd76"
	run := "wf_894b5285-5d3"
	path := JournalPath(projects, workDir, session, run)
	want := filepath.Join(projects, "-home-mik-ggrun-project-ggrun", session,
		"subagents", "workflows", run, "journal.jsonl")
	if path != want {
		t.Fatalf("JournalPath = %q, want %q", path, want)
	}

	if got := CachedAgents(path); got != 0 {
		t.Errorf("missing journal reported %d cached agents, want 0", got)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	journal := `{"type":"started","key":"a","agentId":"1"}
{"type":"result","key":"a","agentId":"1","result":{}}
{"type":"started","key":"b","agentId":"2"}
{"type":"result","key":"b","agentId":"2","result":{}}
{"type":"started","key":"c","agentId":"3"}
`
	if err := os.WriteFile(path, []byte(journal), 0o600); err != nil {
		t.Fatal(err)
	}
	// Two results and three starts: only the finished agents replay.
	if got := CachedAgents(path); got != 2 {
		t.Errorf("CachedAgents = %d, want 2", got)
	}
}

func valueOf(args []string, key string) string {
	for i, a := range args {
		if a == key && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

// Going from --parallel 1 to 2 at the same --ctx-size halves the slot, which
// the flag comparison alone reads as "cannot hold the session". Measured on a
// live run the conversation was 82,846 tokens against a new slot of 131,072, so
// the refusal cost a resume that would have worked with a 37% margin.
func TestShapeMismatchesAllowsAShrinkTheConversationStillFitsIn(t *testing.T) {
	rec := Record{
		SessionID:  "11111111-2222-4333-8444-555555555555",
		ServerArgs: []string{"--ctx-size", "262144", "--parallel", "1"},
	}
	now := []string{"--ctx-size", "262144", "--parallel", "2"}
	measured := func(_, _, _ string) (int, bool) { return 82846, true }
	if got := rec.shapeMismatches(now, measured); len(got) != 0 {
		t.Fatalf("refused a resume that fits: %v", got)
	}
}

func TestShapeMismatchesStillRefusesAConversationThatNoLongerFits(t *testing.T) {
	rec := Record{
		SessionID:  "11111111-2222-4333-8444-555555555555",
		ServerArgs: []string{"--ctx-size", "262144", "--parallel", "1"},
	}
	now := []string{"--ctx-size", "262144", "--parallel", "2"}
	// Inside the slot but past the compaction margin: the next reply overflows.
	measured := func(_, _, _ string) (int, bool) { return 120000, true }
	got := rec.shapeMismatches(now, measured)
	if len(got) != 1 {
		t.Fatalf("expected one mismatch, got %v", got)
	}
	if !strings.Contains(got[0].Recorded, "120000") {
		t.Errorf("the refusal must show the measured size, got %q", got[0].Recorded)
	}
}

// With no readable transcript there is nothing to measure, and assuming the
// session is small would resume straight into a mid-run truncation.
func TestShapeMismatchesFallsBackToFlagsWithoutATranscript(t *testing.T) {
	rec := Record{
		SessionID:  "11111111-2222-4333-8444-555555555555",
		ServerArgs: []string{"--ctx-size", "262144", "--parallel", "1"},
	}
	now := []string{"--ctx-size", "262144", "--parallel", "2"}
	unmeasurable := func(_, _, _ string) (int, bool) { return 0, false }
	if got := rec.shapeMismatches(now, unmeasurable); len(got) != 1 {
		t.Fatalf("expected the flag comparison to stand in, got %v", got)
	}
}

// A grown slot is never a mismatch, measured or not.
func TestShapeMismatchesIgnoresAGrownSlot(t *testing.T) {
	rec := Record{ServerArgs: []string{"--ctx-size", "131072", "--parallel", "1"}}
	now := []string{"--ctx-size", "262144", "--parallel", "1"}
	if got := rec.shapeMismatches(now, nil); len(got) != 0 {
		t.Fatalf("a larger slot must never refuse: %v", got)
	}
}

func TestCurrentContextTokensReadsTheLatestTurnNotThePeak(t *testing.T) {
	projects := t.TempDir()
	workDir := "/tmp/some-project"
	dir := filepath.Join(projects, ProjectKey(workDir))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	session := "11111111-2222-4333-8444-555555555555"
	// Peak first, then a compaction shrinks it. The peak says nothing about what
	// the next turn has to fit.
	lines := `{"message":{"usage":{"input_tokens":1000,"cache_read_input_tokens":130402}}}
{"message":{"usage":{"input_tokens":500,"cache_read_input_tokens":82346}}}
{"type":"user","message":{"role":"user","content":"no usage here"}}
`
	if err := os.WriteFile(filepath.Join(dir, session+".jsonl"), []byte(lines), 0o600); err != nil {
		t.Fatal(err)
	}
	got, ok := CurrentContextTokens(projects, workDir, session)
	if !ok {
		t.Fatal("a readable transcript must report a measurement")
	}
	if got != 82846 {
		t.Errorf("got %d, want the latest turn 82846 rather than the 131402 peak", got)
	}
}

func TestCurrentContextTokensReportsNothingToMeasure(t *testing.T) {
	if _, ok := CurrentContextTokens(t.TempDir(), "/nope", "11111111-2222-4333-8444-555555555555"); ok {
		t.Error("a missing transcript must not report a measurement")
	}
}
