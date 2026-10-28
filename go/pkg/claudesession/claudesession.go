// Package claudesession records which Claude Code session a local launch
// produced, together with the exact backend shape it ran under.
//
// Claude Code's workflow resume cache lives at
//
//	<sessionProjectDir>/<sessionID>/subagents/workflows/<runID>/journal.jsonl
//
// so the session ID is the handle for everything already computed. Resuming
// into a different context, quant, KV type or placement does not fail loudly:
// it produces wrong output from a cache built under other settings. ggrun
// therefore stores the launch shape next to the session ID and refuses a resume
// whose shape no longer matches.
package claudesession

import (
	"bufio"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf16"
)

// DirName is the subdirectory of the ggrun cache holding session records.
const DirName = "claude-sessions"

// Workflow identifies a workflow run inside a session so a resume can continue
// it instead of only reopening the conversation.
type Workflow struct {
	RunID      string `json:"run_id,omitempty"`
	Name       string `json:"name,omitempty"`
	ScriptPath string `json:"script_path,omitempty"`
}

// Record is one resumable Claude Code session.
type Record struct {
	SessionID  string    `json:"session_id"`
	Recorded   time.Time `json:"recorded"`
	WorkDir    string    `json:"work_dir"`
	ModelPath  string    `json:"model_path"`
	Backend    string    `json:"backend,omitempty"`
	Port       int       `json:"port,omitempty"`
	LaunchArgs []string  `json:"launch_args,omitempty"`
	ServerArgs []string  `json:"server_args,omitempty"`
	Workflow   *Workflow `json:"workflow,omitempty"`
}

// NewSessionID returns a random RFC 4122 version 4 UUID. Claude Code's
// --session-id requires a valid UUID, so ggrun mints one at launch instead of
// discovering it afterwards by scanning transcripts.
func NewSessionID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("generate session id: %w", err)
	}
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16]), nil
}

// ValidSessionID reports whether id is a UUID in the form Claude Code accepts.
// It also keeps a hostile id from escaping the records directory.
func ValidSessionID(id string) bool {
	if len(id) != 36 {
		return false
	}
	for i, c := range id {
		switch i {
		case 8, 13, 18, 23:
			if c != '-' {
				return false
			}
		default:
			isHex := (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
			if !isHex {
				return false
			}
		}
	}
	return true
}

// Dir returns the records directory inside the given ggrun cache directory.
func Dir(cacheDir string) string {
	return filepath.Join(cacheDir, DirName)
}

// Save writes one record, replacing any earlier record for the same session.
func Save(cacheDir string, rec Record) error {
	if !ValidSessionID(rec.SessionID) {
		return fmt.Errorf("invalid session id %q", rec.SessionID)
	}
	dir := Dir(cacheDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create %s: %w", dir, err)
	}
	if rec.Recorded.IsZero() {
		rec.Recorded = time.Now().UTC()
	}
	body, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return fmt.Errorf("encode session record: %w", err)
	}
	path := filepath.Join(dir, rec.SessionID+".json")
	// Write and rename so an interrupted save cannot leave a half record that
	// would later be resumed into.
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(body, '\n'), 0o600); err != nil {
		return fmt.Errorf("write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("rename %s: %w", path, err)
	}
	return nil
}

// Load reads one session record by ID.
func Load(cacheDir, sessionID string) (Record, error) {
	var rec Record
	if !ValidSessionID(sessionID) {
		return rec, fmt.Errorf("invalid session id %q", sessionID)
	}
	path := filepath.Join(Dir(cacheDir), sessionID+".json")
	body, err := os.ReadFile(path)
	if err != nil {
		return rec, fmt.Errorf("no recorded session %s: %w", sessionID, err)
	}
	if err := json.Unmarshal(body, &rec); err != nil {
		return rec, fmt.Errorf("decode %s: %w", path, err)
	}
	return rec, nil
}

// List returns records for workDir, newest first. An empty workDir lists all.
func List(cacheDir, workDir string) ([]Record, error) {
	entries, err := os.ReadDir(Dir(cacheDir))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []Record
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".json") {
			continue
		}
		rec, err := Load(cacheDir, strings.TrimSuffix(name, ".json"))
		if err != nil {
			continue
		}
		if workDir != "" && rec.WorkDir != workDir {
			continue
		}
		out = append(out, rec)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Recorded.After(out[j].Recorded) })
	return out, nil
}

// Latest returns the newest record for workDir.
func Latest(cacheDir, workDir string) (Record, error) {
	records, err := List(cacheDir, workDir)
	if err != nil {
		return Record{}, err
	}
	if len(records) == 0 {
		return Record{}, fmt.Errorf("no recorded Claude Code session for %s", workDir)
	}
	return records[0], nil
}

// LatestRecoverable returns the newest record for workDir that has something to
// resume: a transcript, a workflow journal, or a recorded workflow pointer. A
// session that was launched live but never ran a turn records no transcript, so
// resuming it would reopen an empty conversation. "latest" must skip those.
func LatestRecoverable(cacheDir, workDir string) (Record, error) {
	records, err := List(cacheDir, workDir)
	if err != nil {
		return Record{}, err
	}
	for _, rec := range records {
		if rec.Recoverable() {
			return rec, nil
		}
	}
	return Record{}, fmt.Errorf("no recorded Claude Code session for %s has anything to resume", workDir)
}

// A resume restores a conversation and a workflow journal, both of which are
// plain recorded data. No backend tensor state survives a restart, so nothing
// in the launch configuration can reinterpret them.
//
// The guard therefore checks capacity, not identity. The one way a resume can
// genuinely fail is a slot too small to hold the conversation it must replay.
// An earlier version compared placement -- --n-cpu-moe, -ot, --tensor-split --
// and refused whenever they moved. They move legitimately: ggrun recomputes
// placement from live VRAM, and a companion model holding a couple of
// gigabytes shifts it by a few expert layers. That guard blocked three
// consecutive resumes over a difference that could not affect correctness.

// perSlotContext returns the context each slot gets, which is what a resumed
// conversation actually has to fit into.
func perSlotContext(serverArgs []string) int {
	ctx, parallel := 0, 1
	for i := 0; i < len(serverArgs)-1; i++ {
		switch serverArgs[i] {
		case "--ctx-size", "-c":
			if v, err := strconv.Atoi(serverArgs[i+1]); err == nil {
				ctx = v
			}
		case "--parallel", "-np":
			if v, err := strconv.Atoi(serverArgs[i+1]); err == nil && v > 0 {
				parallel = v
			}
		}
	}
	if ctx <= 0 {
		return 0
	}
	return ctx / parallel
}

// Mismatch describes one setting that differs between a recorded session and a
// proposed relaunch.
type Mismatch struct {
	Key      string
	Recorded string
	Proposed string
}

func (m Mismatch) String() string {
	recorded, proposed := m.Recorded, m.Proposed
	if recorded == "" {
		recorded = "(unset)"
	}
	if proposed == "" {
		proposed = "(unset)"
	}
	return fmt.Sprintf("%s: recorded %s, now %s", m.Key, recorded, proposed)
}

// ShapeMismatches reports a proposed relaunch that cannot hold the recorded
// session. Only a shrunk per-slot context qualifies: a conversation that fit
// before may not fit now, and the failure would surface mid-run as a truncated
// or rejected request rather than at launch.
//
// Growing the slot is fine, and everything else -- placement, KV type, batch
// sizes, rope settings -- is deliberately not checked. Those change how the
// model runs, not whether recorded data can be replayed.
//
// The shrink alone is not the question, though: what matters is whether the
// conversation being resumed still fits, and that is measurable rather than
// inferable from flags. Going from --parallel 1 to 2 at the same --ctx-size
// halves the slot and refused a session whose current context was 82,846
// tokens against a new slot of 131,072 -- a 37% margin. Its historical peak had
// touched 131,402, but that was seven turns out of 332 and days before the
// compaction that shrank it. So consult the transcript when one is available
// and compare against what the next turn will actually send; fall back to the
// flag comparison only when there is nothing to measure.
func (r Record) ShapeMismatches(serverArgs []string) []Mismatch {
	return r.shapeMismatches(serverArgs, CurrentContextTokens)
}

// shapeMismatches takes the measurement function so the decision can be tested
// without a transcript on disk.
func (r Record) shapeMismatches(serverArgs []string, currentTokens func(projectsDir, workDir, sessionID string) (int, bool)) []Mismatch {
	was, now := perSlotContext(r.ServerArgs), perSlotContext(serverArgs)
	if was <= 0 || now <= 0 || now >= was {
		return nil
	}
	if currentTokens != nil {
		if tokens, ok := currentTokens(defaultProjectsDir(), r.WorkDir, r.SessionID); ok {
			// Leave the same headroom the client does before it compacts, so a
			// resume that fits today does not overflow on the very next reply.
			if tokens > 0 && tokens <= now*3/4 {
				return nil
			}
			return []Mismatch{{
				Key:      "context per slot",
				Recorded: strconv.Itoa(was) + " (conversation is " + strconv.Itoa(tokens) + " tokens)",
				Proposed: strconv.Itoa(now),
			}}
		}
	}
	return []Mismatch{{
		Key:      "context per slot",
		Recorded: strconv.Itoa(was),
		Proposed: strconv.Itoa(now),
	}}
}

// JournalPath returns where Claude Code keeps a session's workflow resume
// cache, so a caller can report whether there is anything to recover before
// paying for a relaunch. It mirrors the client's own path construction:
// <projects>/<projectKey>/<sessionID>/subagents/workflows/<runID>/journal.jsonl
func JournalPath(projectsDir, workDir, sessionID, runID string) string {
	return filepath.Join(projectDir(projectsDir, workDir), sessionID,
		"subagents", "workflows", runID, "journal.jsonl")
}

// LatestRun finds the newest workflow run in a session and how much of it is
// recoverable. The run ID is assigned inside Claude Code, so ggrun discovers it
// from the session directory rather than trying to observe it at launch.
//
// The returned Workflow is nil when the session ran no workflow.
func LatestRun(projectsDir, workDir, sessionID string) (*Workflow, int) {
	sessionDir := filepath.Join(projectDir(projectsDir, workDir), sessionID)
	entries, err := os.ReadDir(filepath.Join(sessionDir, "subagents", "workflows"))
	if err != nil {
		return nil, 0
	}
	var (
		bestRun  string
		bestMod  time.Time
		bestSize int
	)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		journal := filepath.Join(sessionDir, "subagents", "workflows", entry.Name(), "journal.jsonl")
		info, err := os.Stat(journal)
		if err != nil {
			continue
		}
		// Prefer the most recently written run; a run with no cached result is
		// not worth resuming into.
		cached := CachedAgents(journal)
		if cached == 0 {
			continue
		}
		if bestRun == "" || info.ModTime().After(bestMod) {
			bestRun, bestMod, bestSize = entry.Name(), info.ModTime(), cached
		}
	}
	if bestRun == "" {
		return nil, 0
	}
	wf := &Workflow{RunID: bestRun}
	// The run's definition carries the script path and name. It is written when
	// a run finishes, so treat it as optional rather than required.
	if body, err := os.ReadFile(filepath.Join(sessionDir, "workflows", bestRun+".json")); err == nil {
		var def struct {
			ScriptPath   string `json:"scriptPath"`
			WorkflowName string `json:"workflowName"`
		}
		if json.Unmarshal(body, &def) == nil {
			wf.ScriptPath, wf.Name = def.ScriptPath, def.WorkflowName
		}
	}
	return wf, bestSize
}

// Recoverable reports whether resuming this session would reopen anything: the
// recorded workflow pointer, a recoverable workflow run, or the session
// transcript. A fresh live launch records none of these until the first turn, so
// an empty resume must not be offered for it.
func (r Record) Recoverable() bool {
	if r.Workflow != nil && r.Workflow.RunID != "" {
		return true
	}
	if wf, _ := LatestRun(defaultProjectsDir(), r.WorkDir, r.SessionID); wf != nil {
		return true
	}
	if _, err := os.Stat(filepath.Join(projectDir(defaultProjectsDir(), r.WorkDir), r.SessionID+".jsonl")); err == nil {
		return true
	}
	return false
}

// ModelPathExists reports whether the model a session was recorded under is
// still on disk. The recorded path is the absolute path the backend loaded, so a
// reinstall or a renamed model directory makes the record unresumable. The
// launch path would fail deep inside the loader with an opaque error; naming the
// stale path up front tells the user the record is stale rather than that their
// command was wrong.
func (r Record) ModelPathExists() bool {
	if r.ModelPath == "" {
		return true // nothing recorded to check; the launch path reports on its own
	}
	_, err := os.Stat(r.ModelPath)
	return err == nil
}

// CachedAgents counts completed agent results in a journal. Resuming replays
// exactly these; anything that was still in flight re-runs from zero.
func CachedAgents(journalPath string) int {
	body, err := os.ReadFile(journalPath)
	if err != nil {
		return 0
	}
	count := 0
	for _, line := range strings.Split(string(body), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var entry struct {
			Type string `json:"type"`
		}
		if json.Unmarshal([]byte(line), &entry) == nil && entry.Type == "result" {
			count++
		}
	}
	return count
}

// claudeMaxProjectKey is Claude Code's cap on a project key; a longer key is
// cut to this length and suffixed with "-" and a hash of the original path.
const claudeMaxProjectKey = 200

// ProjectKey mirrors Claude Code's per-project directory naming: every
// character of the absolute path that is not an ASCII letter or digit becomes
// a dash, so C:\Users\me\my_app keys as C--Users-me-my-app. Claude Code
// replaces UTF-16 code units, so a character outside the BMP becomes two
// dashes. A key over claudeMaxProjectKey is cut and suffixed with the hash of
// the path, as the client does (Claude Code 2.1.307: slice(0, 200) + "-" +
// Math.abs(hash).toString(36)).
func ProjectKey(workDir string) string {
	var b strings.Builder
	for _, r := range workDir {
		switch {
		case 'a' <= r && r <= 'z', 'A' <= r && r <= 'Z', '0' <= r && r <= '9':
			b.WriteRune(r)
		case r > 0xFFFF:
			b.WriteString("--")
		default:
			b.WriteByte('-')
		}
	}
	key := b.String()
	if len(key) <= claudeMaxProjectKey {
		return key
	}
	return key[:claudeMaxProjectKey] + "-" + claudePathHash(workDir)
}

// claudePathHash is Claude Code's 32-bit string hash over the path's UTF-16
// code units (h = h*31 + unit, wrapping), printed as base-36 of its absolute
// value. Two sibling projects under one long prefix therefore get distinct
// directories, so no prefix guessing is needed.
func claudePathHash(path string) string {
	var h int32
	for _, unit := range utf16.Encode([]rune(path)) {
		h = (h << 5) - h + int32(unit)
	}
	v := int64(h)
	if v < 0 {
		v = -v
	}
	return strconv.FormatInt(v, 36)
}

// projectDir returns workDir's Claude Code transcript directory.
func projectDir(projectsDir, workDir string) string {
	return filepath.Join(projectsDir, ProjectKey(workDir))
}

// defaultProjectsDir locates Claude Code's transcript root the same way the
// client does. An empty result simply means "nothing to measure".
func defaultProjectsDir() string {
	if dir := os.Getenv("CLAUDE_CONFIG_DIR"); dir != "" {
		return filepath.Join(dir, "projects")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".claude", "projects")
}

// CurrentContextTokens reports how large the session's conversation currently
// is, which is what a resume will actually send. It reads the last assistant
// turn's usage out of the transcript rather than the largest ever seen: a
// compaction earlier in the session means the peak says nothing about what has
// to fit now.
//
// Returns ok=false when there is no readable transcript, so the caller can fall
// back to comparing flags instead of assuming a session is small.
func CurrentContextTokens(projectsDir, workDir, sessionID string) (int, bool) {
	if projectsDir == "" || sessionID == "" {
		return 0, false
	}
	f, err := os.Open(filepath.Join(projectDir(projectsDir, workDir), sessionID+".jsonl"))
	if err != nil {
		return 0, false
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// Transcript lines carry whole assistant messages and can be far longer than
	// the default 64 KiB limit; a truncated line just fails to parse, which would
	// silently look like "no usage recorded".
	scanner.Buffer(make([]byte, 0, 1<<20), 64<<20)
	latest, found := 0, false
	for scanner.Scan() {
		var row struct {
			Message struct {
				Usage struct {
					InputTokens         int `json:"input_tokens"`
					CacheReadTokens     int `json:"cache_read_input_tokens"`
					CacheCreationTokens int `json:"cache_creation_input_tokens"`
				} `json:"usage"`
			} `json:"message"`
		}
		if json.Unmarshal(scanner.Bytes(), &row) != nil {
			continue
		}
		u := row.Message.Usage
		if total := u.InputTokens + u.CacheReadTokens + u.CacheCreationTokens; total > 0 {
			latest, found = total, true
		}
	}
	if scanner.Err() != nil && !found {
		return 0, false
	}
	return latest, found
}
