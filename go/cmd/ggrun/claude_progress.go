package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// Two seconds keeps the UI responsive without injecting two high-priority
	// monitoring tasks into llama-server every second during slow multi-slot decode.
	claudeProgressPollInterval = 2 * time.Second
	claudeProgressStaleAfter   = 10 * time.Second
)

type claudeProgressLog interface {
	Tail(max int) string
}

type claudeSlotProgress struct {
	ID                     int  `json:"id"`
	IDTask                 int  `json:"id_task"`
	IsProcessing           bool `json:"is_processing"`
	NContext               int  `json:"n_ctx"`
	NPromptTokens          int  `json:"n_prompt_tokens"`
	NPromptTokensProcessed int  `json:"n_prompt_tokens_processed"`
	NextToken              struct {
		HasNextToken bool `json:"has_next_token"`
		NRemain      int  `json:"n_remain"`
		NDecoded     int  `json:"n_decoded"`
	} `json:"next_token"`
}

type claudeRequestProgress struct {
	Slot            int     `json:"slot"`
	Task            int     `json:"task"`
	Stage           string  `json:"stage"`
	PromptProcessed int     `json:"prompt_processed,omitempty"`
	PromptTotal     int     `json:"prompt_total,omitempty"`
	PromptFraction  float64 `json:"prompt_fraction,omitempty"`
	Generated       int     `json:"generated,omitempty"`
	TokensPerSecond float64 `json:"tokens_per_second,omitempty"`
	ElapsedSeconds  int     `json:"elapsed_seconds,omitempty"`
}

type claudeProgressState struct {
	UpdatedAt  time.Time               `json:"updated_at"`
	TotalSlots int                     `json:"total_slots"`
	Active     int                     `json:"active"`
	Queued     int                     `json:"queued"`
	Requests   []claudeRequestProgress `json:"requests,omitempty"`
	Event      string                  `json:"event,omitempty"`
	Error      string                  `json:"error,omitempty"`
}

type claudeProgressTracker struct {
	started     map[int]time.Time
	previous    map[int]bool
	lastEvent   string
	lastEventAt time.Time
	now         func() time.Time
}

type promptLogProgress struct {
	Slot      int
	Task      int
	Processed int
	Fraction  float64
	Rate      float64
}

var promptProgressRE = regexp.MustCompile(`slot print_timing: id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+prompt processing, n_tokens =\s*(\d+), progress =\s*([0-9.]+), t =\s*[0-9.]+\s*s /\s*([0-9.]+) tokens per second`)

var metricLineRE = regexp.MustCompile(`(?m)^llamacpp:(requests_deferred|prompt_tokens_seconds|predicted_tokens_seconds)\s+([0-9.eE+-]+)\s*$`)

func claudeCodeProgressServerArgs(args []string, enabled bool, backendHelp string) []string {
	if !enabled || hasArg(args, "--metrics") || !strings.Contains(backendHelp, "--metrics") {
		return args
	}
	return append(args, "--metrics")
}

func claudeCodeProgressClientArgs(extraArgs []string, port int) ([]string, bool) {
	if progressDisabled() || hasSettingsArg(extraArgs) || claudeHasCustomStatusLine() {
		return append([]string(nil), extraArgs...), false
	}
	settings := map[string]interface{}{
		"statusLine": map[string]interface{}{
			"type":            "command",
			"command":         fmt.Sprintf("%s claude-status --port %d", claudeProgressCommand(), port),
			"refreshInterval": 2,
			"padding":         0,
		},
	}
	data, err := json.Marshal(settings)
	if err != nil {
		return append([]string(nil), extraArgs...), false
	}
	out := []string{"--settings", string(data)}
	out = append(out, extraArgs...)
	return out, true
}

func claudeProgressCommand() string {
	exe, err := os.Executable()
	if err != nil || exe == "" {
		return "ggrun"
	}
	if installed, lookErr := exec.LookPath("ggrun"); lookErr == nil {
		if exeInfo, statErr := os.Stat(exe); statErr == nil {
			if installedInfo, installedErr := os.Stat(installed); installedErr == nil && os.SameFile(exeInfo, installedInfo) {
				return "ggrun"
			}
		}
	}
	if runtime.GOOS == "windows" {
		return `"` + strings.ReplaceAll(exe, `"`, `\"`) + `"`
	}
	return "'" + strings.ReplaceAll(exe, "'", "'\\''") + "'"
}

func progressDisabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("GGRUN_CLAUDE_PROGRESS"))) {
	case "0", "off", "false", "no":
		return true
	default:
		return false
	}
}

func hasSettingsArg(args []string) bool {
	for _, arg := range args {
		if arg == "--settings" || strings.HasPrefix(arg, "--settings=") {
			return true
		}
	}
	return false
}

func claudeHasCustomStatusLine() bool {
	var paths []string
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		paths = append(paths, filepath.Join(home, ".claude", "settings.json"))
	}
	if cwd, err := os.Getwd(); err == nil && cwd != "" {
		paths = append(paths,
			filepath.Join(cwd, ".claude", "settings.json"),
			filepath.Join(cwd, ".claude", "settings.local.json"),
		)
	}
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var settings map[string]json.RawMessage
		if json.Unmarshal(data, &settings) == nil {
			if raw, ok := settings["statusLine"]; ok && string(raw) != "null" {
				return true
			}
		}
	}
	return false
}

func normalizeClaudeProgressHost(host string) string {
	if host == "" || host == "0.0.0.0" || host == "::" {
		return "127.0.0.1"
	}
	return strings.Trim(host, "[]")
}

func claudeProgressStatePath(port int) string {
	base, err := os.UserCacheDir()
	if err != nil || base == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, "ggrun", fmt.Sprintf("claude-progress-%d.json", port))
}

func startClaudeProgressMonitor(host string, port int, log claudeProgressLog, terminalTitle bool) func() {
	ctxDone := make(chan struct{})
	done := make(chan struct{})
	var once sync.Once
	stop := func() {
		once.Do(func() {
			close(ctxDone)
			<-done
			_ = os.Remove(claudeProgressStatePath(port))
		})
	}
	go func() {
		defer close(done)
		tracker := newClaudeProgressTracker()
		// A four-way CPU-offloaded MoE decode can hold the backend scheduler for
		// longer than one second. Let the high-priority slots/metrics task reach a
		// scheduling boundary instead of flashing "unavailable" on every token.
		client := &http.Client{Timeout: 3 * time.Second}
		ticker := time.NewTicker(claudeProgressPollInterval)
		defer ticker.Stop()
		for {
			state := pollClaudeProgress(client, host, port, log)
			tracker.enrich(&state)
			_ = writeClaudeProgressState(port, state)
			if terminalTitle {
				fmt.Fprintf(os.Stderr, "\033]0;%s\007", formatClaudeProgress(state))
			}
			select {
			case <-ctxDone:
				return
			case <-ticker.C:
			}
		}
	}()
	return stop
}

func newClaudeProgressTracker() *claudeProgressTracker {
	return &claudeProgressTracker{
		started:  map[int]time.Time{},
		previous: map[int]bool{},
		now:      time.Now,
	}
}

func (t *claudeProgressTracker) enrich(state *claudeProgressState) {
	now := t.now()
	state.Event = ""
	current := make(map[int]bool, len(state.Requests))
	for i := range state.Requests {
		task := state.Requests[i].Task
		current[task] = true
		started, ok := t.started[task]
		if !ok {
			started = now
			t.started[task] = started
		}
		elapsed := now.Sub(started)
		if elapsed > 0 {
			state.Requests[i].ElapsedSeconds = int(elapsed.Seconds())
		}
	}
	for task := range t.previous {
		if current[task] {
			continue
		}
		delete(t.started, task)
		if state.Error != "" {
			t.lastEvent = "request failed"
		} else {
			t.lastEvent = "request completed"
		}
		t.lastEventAt = now
	}
	if t.lastEvent != "" && now.Sub(t.lastEventAt) < 5*time.Second {
		state.Event = t.lastEvent
	}
	t.previous = current
}

func pollClaudeProgress(client *http.Client, host string, port int, log claudeProgressLog) claudeProgressState {
	state := claudeProgressState{UpdatedAt: time.Now()}
	baseURL := "http://" + net.JoinHostPort(normalizeClaudeProgressHost(host), strconv.Itoa(port))
	resp, err := client.Get(baseURL + "/slots")
	if err != nil {
		state.Error = err.Error()
		return state
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		state.Error = fmt.Sprintf("slots endpoint returned %s", resp.Status)
		return state
	}
	var slots []claudeSlotProgress
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&slots); err != nil {
		state.Error = err.Error()
		return state
	}
	state.TotalSlots = len(slots)

	logged := map[int]promptLogProgress{}
	if log != nil {
		for _, item := range parsePromptLogProgress(log.Tail(256 << 10)) {
			logged[item.Task] = item
		}
	}
	promptRate, generationRate, queued := pollClaudeMetrics(client, baseURL)
	state.Queued = queued
	for _, slot := range slots {
		if !slot.IsProcessing {
			continue
		}
		state.Active++
		req := claudeRequestProgress{
			Slot:            slot.ID,
			Task:            slot.IDTask,
			Stage:           "working",
			PromptProcessed: slot.NPromptTokensProcessed,
			Generated:       slot.NextToken.NDecoded,
		}
		if item, ok := logged[slot.IDTask]; ok && slot.NextToken.NDecoded == 0 {
			req.Stage = "prefill"
			req.PromptProcessed = item.Processed
			req.PromptFraction = item.Fraction
			req.TokensPerSecond = item.Rate
			if item.Fraction > 0 {
				req.PromptTotal = int(float64(item.Processed)/item.Fraction + 0.5)
			}
		} else if slot.NextToken.NDecoded > 0 || slot.NextToken.HasNextToken {
			req.Stage = "generating"
			req.TokensPerSecond = generationRate
		} else if slot.NPromptTokensProcessed > 0 {
			req.Stage = "prefill"
			req.TokensPerSecond = promptRate
		}
		state.Requests = append(state.Requests, req)
	}
	return state
}

func parsePromptLogProgress(text string) []promptLogProgress {
	latest := map[int]promptLogProgress{}
	for _, match := range promptProgressRE.FindAllStringSubmatch(text, -1) {
		if len(match) != 6 {
			continue
		}
		slot, errSlot := strconv.Atoi(match[1])
		task, errTask := strconv.Atoi(match[2])
		processed, errProcessed := strconv.Atoi(match[3])
		fraction, errFraction := strconv.ParseFloat(match[4], 64)
		rate, errRate := strconv.ParseFloat(match[5], 64)
		if errSlot != nil || errTask != nil || errProcessed != nil || errFraction != nil || errRate != nil {
			continue
		}
		latest[task] = promptLogProgress{Slot: slot, Task: task, Processed: processed, Fraction: fraction, Rate: rate}
	}
	out := make([]promptLogProgress, 0, len(latest))
	for _, item := range latest {
		out = append(out, item)
	}
	return out
}

func pollClaudeMetrics(client *http.Client, baseURL string) (promptRate, generationRate float64, queued int) {
	resp, err := client.Get(baseURL + "/metrics")
	if err != nil {
		return 0, 0, 0
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, 0, 0
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return 0, 0, 0
	}
	for _, match := range metricLineRE.FindAllStringSubmatch(string(data), -1) {
		value, err := strconv.ParseFloat(match[2], 64)
		if err != nil {
			continue
		}
		switch match[1] {
		case "requests_deferred":
			queued = int(value)
		case "prompt_tokens_seconds":
			promptRate = value
		case "predicted_tokens_seconds":
			generationRate = value
		}
	}
	return promptRate, generationRate, queued
}

func writeClaudeProgressState(port int, state claudeProgressState) error {
	path := claudeProgressStatePath(port)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	data, err := json.Marshal(state)
	if err != nil {
		return err
	}
	tmp := path + fmt.Sprintf(".%d.tmp", os.Getpid())
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func readClaudeProgressState(port int) (claudeProgressState, error) {
	data, err := os.ReadFile(claudeProgressStatePath(port))
	if err != nil {
		return claudeProgressState{}, err
	}
	var state claudeProgressState
	if err := json.Unmarshal(data, &state); err != nil {
		return claudeProgressState{}, err
	}
	if state.UpdatedAt.IsZero() || time.Since(state.UpdatedAt) > claudeProgressStaleAfter {
		return state, errors.New("progress state is stale")
	}
	return state, nil
}

func cmdClaudeStatus(args []string) {
	fs := flag.NewFlagSet("claude-status", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	port := fs.Int("port", 8081, "llama-server port")
	if err := fs.Parse(args); err != nil {
		fmt.Print("ggrun · progress unavailable")
		return
	}
	state, err := readClaudeProgressState(*port)
	if err != nil {
		fmt.Print("ggrun · connecting")
		return
	}
	fmt.Print(formatClaudeProgress(state))
}

func formatClaudeProgress(state claudeProgressState) string {
	if state.Event != "" && state.Active == 0 {
		return "ggrun · " + state.Event
	}
	if state.Error != "" {
		return "ggrun · backend unavailable"
	}
	if state.Active == 0 {
		if state.Queued > 0 {
			return fmt.Sprintf("ggrun · %d queued", state.Queued)
		}
		if state.TotalSlots > 0 {
			return fmt.Sprintf("ggrun · local ready · %d slots", state.TotalSlots)
		}
		return "ggrun · connecting"
	}
	primary := state.Requests[0]
	for _, req := range state.Requests[1:] {
		if req.PromptTotal > primary.PromptTotal || req.PromptProcessed > primary.PromptProcessed {
			primary = req
		}
	}
	parts := []string{"ggrun"}
	if primary.Stage == "prefill" {
		if primary.PromptFraction > 0 {
			pct := int(primary.PromptFraction*100 + 0.5)
			parts = append(parts, progressBar(primary.PromptFraction, 10), fmt.Sprintf("%d%%", pct))
		}
		count := formatCount(primary.PromptProcessed)
		if primary.PromptTotal > 0 {
			count += "/~" + formatCount(primary.PromptTotal)
		}
		parts = append(parts, fmt.Sprintf("S%d prefill %s", primary.Slot, count))
	} else if primary.Stage == "generating" {
		parts = append(parts, fmt.Sprintf("S%d generating %d tok", primary.Slot, primary.Generated))
	} else {
		parts = append(parts, fmt.Sprintf("S%d working", primary.Slot))
	}
	if primary.TokensPerSecond > 0 {
		parts = append(parts, fmt.Sprintf("%.1f tok/s", primary.TokensPerSecond))
	}
	if primary.ElapsedSeconds > 0 {
		parts = append(parts, formatElapsed(time.Duration(primary.ElapsedSeconds)*time.Second))
	}
	if state.Active > 1 {
		parts = append(parts, fmt.Sprintf("%d active", state.Active))
	}
	if state.Queued > 0 {
		parts = append(parts, fmt.Sprintf("%d queued", state.Queued))
	}
	return strings.Join(parts, " · ")
}

func formatElapsed(elapsed time.Duration) string {
	if elapsed < time.Minute {
		return fmt.Sprintf("%ds", int(elapsed.Seconds()))
	}
	if elapsed < time.Hour {
		return fmt.Sprintf("%dm%02ds", int(elapsed.Minutes()), int(elapsed.Seconds())%60)
	}
	return fmt.Sprintf("%dh%02dm", int(elapsed.Hours()), int(elapsed.Minutes())%60)
}

func progressBar(fraction float64, width int) string {
	if fraction < 0 {
		fraction = 0
	}
	if fraction > 1 {
		fraction = 1
	}
	filled := int(fraction*float64(width) + 0.5)
	return strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
}

func formatCount(value int) string {
	s := strconv.Itoa(value)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}
