package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

type testProgressLog string

func (l testProgressLog) Tail(int) string { return string(l) }

func TestParsePromptLogProgressKeepsLatestPerTask(t *testing.T) {
	log := `
slot print_timing: id  0 | task 4 | prompt processing, n_tokens =  2049, progress = 0.03, t = 125.00 s / 16.39 tokens per second
slot print_timing: id  1 | task 7 | prompt processing, n_tokens =  1025, progress = 0.25, t = 50.00 s / 20.50 tokens per second
slot print_timing: id  0 | task 4 | prompt processing, n_tokens = 32769, progress = 0.55, t = 1113.88 s / 29.42 tokens per second
`
	items := parsePromptLogProgress(log)
	if len(items) != 2 {
		t.Fatalf("got %d tasks, want 2: %+v", len(items), items)
	}
	var main promptLogProgress
	for _, item := range items {
		if item.Task == 4 {
			main = item
		}
	}
	if main.Processed != 32769 || main.Fraction != 0.55 || main.Rate != 29.42 {
		t.Fatalf("latest task progress not retained: %+v", main)
	}
}

func TestPollAndFormatClaudeProgress(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/slots", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[
  {"id":0,"id_task":4,"is_processing":true,"n_ctx":262144,"n_prompt_tokens":32769,"n_prompt_tokens_processed":32769,"next_token":{"has_next_token":false,"n_remain":16,"n_decoded":0}},
  {"id":1,"is_processing":false,"n_ctx":262144}
]`))
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("llamacpp:prompt_tokens_seconds 29.42\nllamacpp:predicted_tokens_seconds 5.88\nllamacpp:requests_deferred 3\n"))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	log := testProgressLog("slot print_timing: id  0 | task 4 | prompt processing, n_tokens = 32769, progress = 0.55, t = 1113.88 s / 29.42 tokens per second\n")
	state := pollClaudeProgress(srv.Client(), u.Hostname(), port, log)
	if state.Active != 1 || state.Queued != 3 || state.TotalSlots != 2 || len(state.Requests) != 1 {
		t.Fatalf("unexpected state: %+v", state)
	}
	got := formatClaudeProgress(state)
	for _, want := range []string{"██████░░░░", "55%", "S0 prefill 32,769/~59,580", "29.4 tok/s", "3 queued"} {
		if !strings.Contains(got, want) {
			t.Fatalf("status %q missing %q", got, want)
		}
	}
}

func TestFormatClaudeProgressGenerationAndReady(t *testing.T) {
	generated := formatClaudeProgress(claudeProgressState{
		TotalSlots: 4,
		Active:     2,
		Queued:     1,
		Requests: []claudeRequestProgress{{
			Slot: 2, Stage: "generating", Generated: 17, TokensPerSecond: 5.88, ElapsedSeconds: 134,
		}},
	})
	for _, want := range []string{"S2 generating 17 tok", "5.9 tok/s", "2m14s", "2 active", "1 queued"} {
		if !strings.Contains(generated, want) {
			t.Fatalf("generation status %q missing %q", generated, want)
		}
	}
	if got := formatClaudeProgress(claudeProgressState{TotalSlots: 4}); got != "ggrun · local ready · 4 slots" {
		t.Fatalf("ready status = %q", got)
	}
	if got := formatClaudeProgress(claudeProgressState{Event: "request completed"}); got != "ggrun · request completed" {
		t.Fatalf("completion status = %q", got)
	}
}

func TestClaudeProgressTrackerElapsedAndLifecycle(t *testing.T) {
	now := time.Unix(1000, 0)
	tracker := newClaudeProgressTracker()
	tracker.now = func() time.Time { return now }
	active := claudeProgressState{Active: 1, Requests: []claudeRequestProgress{{Task: 4, Stage: "prefill"}}}
	tracker.enrich(&active)
	now = now.Add(74 * time.Second)
	tracker.enrich(&active)
	if active.Requests[0].ElapsedSeconds != 74 {
		t.Fatalf("elapsed=%d, want 74", active.Requests[0].ElapsedSeconds)
	}
	completed := claudeProgressState{TotalSlots: 4}
	tracker.enrich(&completed)
	if completed.Event != "request completed" {
		t.Fatalf("completion event=%q", completed.Event)
	}

	now = now.Add(6 * time.Second)
	tracker.enrich(&completed)
	if completed.Event != "" {
		t.Fatalf("stale completion event still visible: %q", completed.Event)
	}

	failedActive := claudeProgressState{Active: 1, Requests: []claudeRequestProgress{{Task: 5, Stage: "generating"}}}
	tracker.enrich(&failedActive)
	failed := claudeProgressState{Error: "connection refused"}
	tracker.enrich(&failed)
	if failed.Event != "request failed" || formatClaudeProgress(failed) != "ggrun · request failed" {
		t.Fatalf("failed request transition missing: %+v", failed)
	}
}

func TestClaudeCodeProgressArgs(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("GGRUN_CLAUDE_PROGRESS", "")
	args, ok := claudeCodeProgressClientArgs([]string{"--model", "local"}, 8123)
	if !ok || len(args) < 3 || args[0] != "--settings" {
		t.Fatalf("status line was not injected: ok=%v args=%v", ok, args)
	}
	var settings map[string]map[string]interface{}
	if err := json.Unmarshal([]byte(args[1]), &settings); err != nil {
		t.Fatal(err)
	}
	status := settings["statusLine"]
	command, _ := status["command"].(string)
	if !strings.Contains(command, "claude-status --port 8123") || status["refreshInterval"] != float64(2) {
		t.Fatalf("unexpected status settings: %v", status)
	}

	args, ok = claudeCodeProgressClientArgs([]string{"--settings", "mine.json"}, 8123)
	if ok || len(args) != 2 {
		t.Fatalf("user settings must win: ok=%v args=%v", ok, args)
	}

	withMetrics := claudeCodeProgressServerArgs([]string{"llama-server"}, true, "--slots --metrics")
	if !hasArg(withMetrics, "--metrics") {
		t.Fatalf("metrics flag missing: %v", withMetrics)
	}
	withoutSupport := claudeCodeProgressServerArgs([]string{"llama-server"}, true, "--slots")
	if hasArg(withoutSupport, "--metrics") {
		t.Fatalf("unsupported metrics flag added: %v", withoutSupport)
	}

	t.Setenv("GGRUN_CLAUDE_PROGRESS", "off")
	if _, ok := claudeCodeProgressClientArgs(nil, 8123); ok {
		t.Fatal("progress status line should not be injected when explicitly disabled")
	}
}

func TestClaudeProgressStateRoundTripAndStaleness(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	port := 18765
	want := claudeProgressState{UpdatedAt: time.Now(), TotalSlots: 4, Active: 1}
	if err := writeClaudeProgressState(port, want); err != nil {
		t.Fatal(err)
	}
	got, err := readClaudeProgressState(port)
	if err != nil || got.TotalSlots != 4 || got.Active != 1 {
		t.Fatalf("round trip: got=%+v err=%v", got, err)
	}
	want.UpdatedAt = time.Now().Add(-claudeProgressStaleAfter - time.Second)
	if err := writeClaudeProgressState(port, want); err != nil {
		t.Fatal(err)
	}
	if _, err := readClaudeProgressState(port); err == nil {
		t.Fatal("expected stale state to be rejected")
	}
	_ = os.Remove(claudeProgressStatePath(port))
}

func TestClaudeProgressMonitorPublishesAndCleansState(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	mux := http.NewServeMux()
	mux.HandleFunc("/slots", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`[{"id":0,"id_task":9,"is_processing":true,"n_prompt_tokens_processed":2049,"next_token":{"n_decoded":0}}]`))
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("llamacpp:prompt_tokens_seconds 20\nllamacpp:requests_deferred 0\n"))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	log := testProgressLog("slot print_timing: id  0 | task 9 | prompt processing, n_tokens =  2049, progress = 0.25, t = 100.00 s / 20.49 tokens per second\n")
	stop := startClaudeProgressMonitor(u.Hostname(), port, log, false)
	deadline := time.Now().Add(2 * time.Second)
	for {
		state, readErr := readClaudeProgressState(port)
		if readErr == nil && state.Active == 1 && len(state.Requests) == 1 {
			break
		}
		if time.Now().After(deadline) {
			stop()
			t.Fatalf("monitor did not publish active progress: state=%+v err=%v", state, readErr)
		}
		time.Sleep(10 * time.Millisecond)
	}
	stop()
	if _, err := os.Stat(claudeProgressStatePath(port)); !os.IsNotExist(err) {
		t.Fatalf("progress state was not cleaned up: %v", err)
	}
}
