package claudeauto

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// sseBody is the shape Claude Code actually receives: input counts arrive in
// message_start and the output count grows across message_delta events.
const sseBody = `event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":61000,"cache_read_input_tokens":24000,"cache_creation_input_tokens":1200,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","usage":{"output_tokens":57}}

event: message_delta
data: {"type":"message_delta","usage":{"output_tokens":211}}

`

// newTestRouter builds a router whose reviewer shares the main backend, which
// is the no-room seating: with nowhere separate to send a review, the main model
// classifies its own requests. Use newTestRouterPair for a seated reviewer.
func newTestRouter(t *testing.T, backend string, maxMainActive int) (*Router, string) {
	t.Helper()
	return newTestRouterPair(t, backend, backend, maxMainActive)
}

// newTestRouterPair builds a router over distinct main and reviewer backends,
// which is what "a separate reviewer is seated" means to the router: it compares
// the two base URLs.
func newTestRouterPair(t *testing.T, mainBackend, reviewerBackend string, maxMainActive int) (*Router, string) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "requests.jsonl")
	router, err := StartRouter(mainBackend, reviewerBackend, true, maxMainActive)
	if err != nil {
		t.Fatalf("StartRouter: %v", err)
	}
	t.Cleanup(func() { _ = router.Close() })
	if err := router.EnableMetrics(path); err != nil {
		t.Fatalf("EnableMetrics: %v", err)
	}
	return router, path
}

func readRecords(t *testing.T, path string) []RequestRecord {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open metrics: %v", err)
	}
	defer f.Close()
	var out []RequestRecord
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for scanner.Scan() {
		var rec RequestRecord
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			t.Fatalf("decode record %q: %v", scanner.Text(), err)
		}
		out = append(out, rec)
	}
	return out
}

// waitForRecords polls because a record is written after the proxy finishes,
// which is strictly after the client has seen the last byte.
func waitForRecords(t *testing.T, path string, want int) []RequestRecord {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	var records []RequestRecord
	for time.Now().Before(deadline) {
		if records = readRecords(t, path); len(records) >= want {
			return records
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("want %d records, got %d", want, len(records))
	return nil
}

// waitForQueued polls the router's queued counter until it reaches want. A fixed
// sleep here raced admission under load: too short and the second request had
// not yet blocked on the slot when the backend was released, so no queue wait
// was ever recorded and the assertion below failed spuriously.
func waitForQueued(t *testing.T, router *Router, want int64) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if n := router.mainQueued.Load(); n >= want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("router never reported %d queued request(s)", want)
}

// postErr sends one routed request and drains the response, so the proxy has
// finished before the caller inspects metrics. It returns an error rather than
// failing the test so it is also safe to call from a goroutine.
func postErr(router *Router, body string) error {
	resp, err := http.Post(router.URL()+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, err = io.Copy(io.Discard, resp.Body)
	return err
}

func post(t *testing.T, router *Router, body string) {
	t.Helper()
	if err := postErr(router, body); err != nil {
		t.Fatalf("post: %v", err)
	}
}

func TestRouterRecordsStreamingUsageAndTiming(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("backend writer is not a Flusher")
		}
		for _, chunk := range strings.SplitAfter(sseBody, "\n\n") {
			_, _ = w.Write([]byte(chunk))
			if ok {
				flusher.Flush()
			}
		}
	}))
	defer backend.Close()

	router, path := newTestRouter(t, backend.URL, 1)
	post(t, router, `{"model":"local","stream":true,"system":"coding agent","messages":[{"role":"user","content":"hi"}]}`)

	rec := waitForRecords(t, path, 1)[0]
	if rec.Route != routeMain {
		t.Errorf("route = %q, want %q", rec.Route, routeMain)
	}
	if !rec.Stream {
		t.Error("stream request not recorded as streaming")
	}
	if rec.Aborted {
		t.Error("completed request recorded as aborted")
	}
	if rec.Status != http.StatusOK {
		t.Errorf("status = %d, want 200", rec.Status)
	}
	if rec.Usage.InputTokens != 61000 {
		t.Errorf("input tokens = %d, want 61000", rec.Usage.InputTokens)
	}
	// The cache keys must not be swallowed by the shorter "input_tokens" match.
	if rec.Usage.CacheReadTokens != 24000 {
		t.Errorf("cache read tokens = %d, want 24000", rec.Usage.CacheReadTokens)
	}
	if rec.Usage.CacheCreationTokens != 1200 {
		t.Errorf("cache creation tokens = %d, want 1200", rec.Usage.CacheCreationTokens)
	}
	// The final message_delta wins, not the placeholder in message_start.
	if rec.Usage.OutputTokens != 211 {
		t.Errorf("output tokens = %d, want 211", rec.Usage.OutputTokens)
	}
	if rec.Usage.PromptTokens() != 62200 {
		t.Errorf("prompt tokens = %d, want 62200", rec.Usage.PromptTokens())
	}
	if rec.Conversation == "" {
		t.Error("conversation key not recorded")
	}
	if rec.ResponseBytes == 0 {
		t.Error("response bytes not recorded")
	}
}

// The metered writer wraps the proxy's ResponseWriter. If it failed to forward
// Flush, every SSE token would buffer until the response ended and Claude Code
// would look frozen, so assert the client sees a chunk while the backend is
// still writing.
func TestRouterForwardsFlushWhileStreaming(t *testing.T) {
	seen := make(chan struct{})
	finish := make(chan struct{})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: message_start\ndata: {}\n\n"))
		w.(http.Flusher).Flush()
		select {
		case <-seen:
		case <-time.After(3 * time.Second):
			t.Error("client never received the flushed chunk")
		}
		_, _ = w.Write([]byte(`data: {"usage":{"output_tokens":5}}` + "\n\n"))
		close(finish)
	}))
	defer backend.Close()

	router, _ := newTestRouter(t, backend.URL, 1)
	resp, err := http.Post(router.URL()+"/v1/messages", "application/json",
		strings.NewReader(`{"model":"local","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	buf := make([]byte, 16)
	if _, err := resp.Body.Read(buf); err != nil {
		t.Fatalf("read first chunk: %v", err)
	}
	close(seen)
	select {
	case <-finish:
	case <-time.After(3 * time.Second):
		t.Fatal("backend did not finish streaming")
	}
	_, _ = io.Copy(io.Discard, resp.Body)
}

func TestMeteredWriterMarksDecodeOnlyOnGeneratedDelta(t *testing.T) {
	recorder := httptest.NewRecorder()
	marked := 0
	w := &meteredWriter{ResponseWriter: recorder, start: time.Now(), onDecode: func() { marked++ }}
	_, _ = w.Write([]byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"output_tokens\":0}}}\n\n"))
	_, _ = w.Write([]byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"thinking\"}}\n\n"))
	if marked != 0 {
		t.Fatalf("HTTP envelope marked decode %d time(s)", marked)
	}
	_, _ = w.Write([]byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"x\"}}\n\n"))
	_, _ = w.Write([]byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"y\"}}\n\n"))
	if marked != 1 {
		t.Fatalf("generated deltas marked decode %d time(s), want once", marked)
	}
}

func TestGeneratedDeltaDetectionSurvivesAWriteBoundary(t *testing.T) {
	recorder := httptest.NewRecorder()
	marked := 0
	w := &meteredWriter{ResponseWriter: recorder, start: time.Now(), onDecode: func() { marked++ }}
	_, _ = w.Write([]byte(`data: {"delta":{"type":"think`))
	_, _ = w.Write([]byte(`ing_delta","thinking":"x"}}`))
	if marked != 1 {
		t.Fatalf("split generated delta marked decode %d time(s), want once", marked)
	}
}

func TestStreamingTimingUsesGeneratedDeltaNotHTTPEnvelope(t *testing.T) {
	rec := RequestRecord{
		Stream:        true,
		QueueMS:       10,
		TTFBMS:        20,
		DecodeStartMS: 120,
		TotalMS:       320,
	}
	if got := rec.PrefillMS(); got != 110 {
		t.Errorf("prefill = %dms, want 110ms to generated delta", got)
	}
	if got := rec.DecodeMS(); got != 200 {
		t.Errorf("decode = %dms, want 200ms after generated delta", got)
	}
	// A stream aborted after an SSE envelope but before generation has neither
	// measured prefill completion nor decode work.
	aborted := RequestRecord{Stream: true, TTFBMS: 20, TotalMS: 60, Aborted: true}
	if aborted.PrefillMS() != 0 || aborted.DecodeMS() != 0 {
		t.Errorf("envelope-only abort reported prefill/decode: %+v", aborted)
	}
}

func TestRouterRecordsQueueWaitSeparatelyFromPrefill(t *testing.T) {
	release := make(chan struct{})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		<-release
		_, _ = w.Write([]byte(`{"usage":{"input_tokens":10,"output_tokens":2}}`))
	}))
	defer backend.Close()

	// One admission slot, so the second request must wait for the first.
	router, path := newTestRouter(t, backend.URL, 1)
	body := `{"model":"local","messages":[{"role":"user","content":"hi"}]}`
	done := make(chan error, 2)
	for i := 0; i < 2; i++ {
		go func() { done <- postErr(router, body) }()
	}
	waitForQueued(t, router, 1)
	close(release)
	for i := 0; i < 2; i++ {
		if err := <-done; err != nil {
			t.Fatalf("post: %v", err)
		}
	}

	records := waitForRecords(t, path, 2)
	var queued int
	for _, rec := range records {
		if rec.QueueMS > 0 {
			queued++
		}
		// Queue wait must never be counted as backend prefill.
		if rec.PrefillMS() > rec.TTFBMS {
			t.Errorf("prefill %dms exceeds ttfb %dms", rec.PrefillMS(), rec.TTFBMS)
		}
	}
	if queued == 0 {
		t.Error("no request recorded a queue wait despite a saturated admission limit")
	}
}

func TestRouterRecordsReviewerRouteWithoutAdmissionControl(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"usage":{"input_tokens":25000,"output_tokens":3}}`))
	}))
	defer backend.Close()
	// A reviewer on its own backend, so the review really does take the reviewer
	// route rather than falling back to the main model.
	reviewer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"<block>no</block>"}],"usage":{"input_tokens":25000,"output_tokens":3}}`))
	}))
	defer reviewer.Close()

	router, path := newTestRouterPair(t, backend.URL, reviewer.URL, 1)
	body := `{"model":"local","system":[{"type":"text","text":"` + ClassifierMarker + ` Review it."}],` +
		`"messages":[{"role":"user","content":"rm -rf /"}]}`
	post(t, router, body)

	records := waitForRecords(t, path, 1)
	if records[0].Route != routeReviewer {
		t.Errorf("route = %q, want %q", records[0].Route, routeReviewer)
	}
	// Safety must not consume a main admission slot.
	if records[0].QueueMS != 0 {
		t.Errorf("reviewer queued %dms, want 0", records[0].QueueMS)
	}
	if records[0].Usage.InputTokens != 25000 {
		t.Errorf("input tokens = %d, want 25000", records[0].Usage.InputTokens)
	}
}

func TestMetricsSummaryAppearsOnRouterEndpoint(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"usage":{"input_tokens":100,"output_tokens":20}}`))
	}))
	defer backend.Close()

	router, _ := newTestRouter(t, backend.URL, 2)
	post(t, router, `{"model":"local","messages":[{"role":"user","content":"hi"}]}`)

	statusResp, err := http.Get(router.URL() + "/ggrun/router")
	if err != nil {
		t.Fatalf("get status: %v", err)
	}
	defer statusResp.Body.Close()
	var status struct {
		Limit   int64          `json:"limit"`
		Metrics map[string]any `json:"metrics"`
	}
	if err := json.NewDecoder(statusResp.Body).Decode(&status); err != nil {
		t.Fatalf("decode status: %v", err)
	}
	if status.Limit != 2 {
		t.Errorf("limit = %d, want 2", status.Limit)
	}
	if status.Metrics == nil {
		t.Fatal("metrics summary missing from /ggrun/router")
	}
	if got, ok := status.Metrics["requests"].(float64); !ok || got != 1 {
		t.Errorf("requests = %v, want 1", status.Metrics["requests"])
	}
	if got, ok := status.Metrics["prompt_tokens"].(float64); !ok || got != 100 {
		t.Errorf("prompt_tokens = %v, want 100", status.Metrics["prompt_tokens"])
	}
}

func TestMetricsSummarySeparatesAbortsAndFailuresFromModelWork(t *testing.T) {
	sink := &metricsSink{}
	sink.record(RequestRecord{
		Status:  http.StatusOK,
		QueueMS: 2,
		TTFBMS:  12,
		TotalMS: 112,
		Usage:   Usage{InputTokens: 100, OutputTokens: 10},
	})
	// This is the exact shape of a client that gives up while waiting in the
	// admission queue: no response writer, first byte, status, or output.
	sink.record(RequestRecord{
		Aborted: true,
		QueueMS: 50,
		TotalMS: 75,
	})
	sink.record(RequestRecord{
		Status:  http.StatusBadRequest,
		QueueMS: 3,
		TTFBMS:  4,
		TotalMS: 5,
	})

	summary := sink.summary()
	for key, want := range map[string]int64{
		"requests":                 3,
		"successful":               1,
		"failed":                   1,
		"aborted":                  1,
		"aborted_ms_total":         75,
		"aborted_queue_ms_total":   50,
		"aborted_service_ms_total": 25,
		"queue_ms_total":           55,
		"prefill_ms_total":         10,
		"decode_ms_total":          100,
		"prompt_tokens":            100,
		"output_tokens":            10,
	} {
		if got := summary[key]; got != want {
			t.Errorf("%s = %v, want %d", key, got, want)
		}
	}
	if got := (RequestRecord{Aborted: true, TotalMS: 75}).DecodeMS(); got != 0 {
		t.Errorf("aborted decode time = %dms, want 0", got)
	}
	if got := summary["decode_tokens_per_s"]; got != float64(100) {
		t.Errorf("decode throughput = %v, want 100", got)
	}
}

func TestRouterWithoutMetricsStaysSilent(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer backend.Close()

	router, err := StartRouter(backend.URL, backend.URL, true, 1)
	if err != nil {
		t.Fatalf("StartRouter: %v", err)
	}
	defer router.Close()
	post(t, router, `{"model":"local","messages":[{"role":"user","content":"hi"}]}`)
	if summary := router.MetricsSummary(); summary != nil {
		t.Errorf("summary = %v, want nil when recording is disabled", summary)
	}
}

func TestScanUsageAcrossChunkBoundary(t *testing.T) {
	full := `{"usage":{"input_tokens":123456,"cache_read_input_tokens":789,"output_tokens":42}}`
	for split := 1; split < len(full); split++ {
		w := &meteredWriter{start: time.Now()}
		w.scan([]byte(full[:split]))
		w.scan([]byte(full[split:]))
		if w.usage.InputTokens != 123456 {
			t.Fatalf("split %d: input tokens = %d, want 123456", split, w.usage.InputTokens)
		}
		if w.usage.CacheReadTokens != 789 {
			t.Fatalf("split %d: cache read tokens = %d, want 789", split, w.usage.CacheReadTokens)
		}
		if w.usage.OutputTokens != 42 {
			t.Fatalf("split %d: output tokens = %d, want 42", split, w.usage.OutputTokens)
		}
	}
}

func TestConversationKeyIsStablePerAgentAndDistinctAcrossAgents(t *testing.T) {
	first := []byte(`{"system":"agent A","messages":[{"role":"user","content":"start"}]}`)
	later := []byte(`{"system":"agent A","messages":[{"role":"user","content":"start"},{"role":"assistant","content":"ok"}]}`)
	other := []byte(`{"system":"agent B","messages":[{"role":"user","content":"start"}]}`)
	if conversationKey(first) != conversationKey(later) {
		t.Error("conversation key changed across turns of the same agent")
	}
	if conversationKey(first) == conversationKey(other) {
		t.Error("different agents share a conversation key")
	}
	// user_id still separates two sessions sharing one server...
	sessionA := []byte(`{"metadata":{"user_id":"{\"session_id\":\"a\"}"},"system":"agent A","messages":[{"role":"user","content":"start"}]}`)
	sessionB := []byte(`{"metadata":{"user_id":"{\"session_id\":\"b\"}"},"system":"agent A","messages":[{"role":"user","content":"start"}]}`)
	if conversationKey(sessionA) == conversationKey(sessionB) {
		t.Error("two sessions collapsed onto one conversation key")
	}
	// ...but it must not erase the agent distinction within a session, which is
	// what Claude Code's per-install user_id blob used to do to every request.
	blob := `{\"device_id\":\"b524\",\"account_uuid\":\"\",\"session_id\":\"072e\"}`
	agentA := []byte(`{"metadata":{"user_id":"` + blob + `"},"system":"agent A","messages":[{"role":"user","content":"start"}]}`)
	agentB := []byte(`{"metadata":{"user_id":"` + blob + `"},"system":"agent B","messages":[{"role":"user","content":"start"}]}`)
	if conversationKey(agentA) == conversationKey(agentB) {
		t.Error("a shared per-install user_id collapsed two agents onto one key")
	}
	agentALater := []byte(`{"metadata":{"user_id":"` + blob + `"},"system":"agent A","messages":[{"role":"user","content":"start"},{"role":"assistant","content":"ok"}]}`)
	if conversationKey(agentA) != conversationKey(agentALater) {
		t.Error("conversation key changed across turns of the same agent")
	}
	if conversationKey(agentA) == "" {
		t.Fatal("the per-install user_id blob must still parse; an empty key disables grouping entirely")
	}
}

// A stream that breaks after the headers went out makes ReverseProxy panic with
// http.ErrAbortHandler. The request must still be recorded, once, as aborted in
// service, and the status the client already saw must be kept.
func TestRouterRecordsStreamBrokenMidBody(t *testing.T) {
	backendDone := make(chan struct{})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(backendDone)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7}}}\n\n"))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer backend.Close()

	router, path := newTestRouter(t, backend.URL, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, router.URL()+"/v1/messages",
		strings.NewReader(`{"model":"local","stream":true,"system":"coding agent","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if _, err := bufio.NewReader(resp.Body).ReadString('\n'); err != nil {
		t.Fatalf("read first event line: %v", err)
	}
	cancel()

	records := waitForRecords(t, path, 1)
	select {
	case <-backendDone:
	case <-time.After(5 * time.Second):
		t.Fatal("backend request was never cancelled")
	}
	// Give a spurious second record the chance to appear before counting.
	time.Sleep(100 * time.Millisecond)
	if records = readRecords(t, path); len(records) != 1 {
		t.Fatalf("records = %d, want exactly 1: %+v", len(records), records)
	}
	rec := records[0]
	if !rec.Aborted || rec.CancelPhase != "service" {
		t.Errorf("aborted=%v cancel_phase=%q, want true/service", rec.Aborted, rec.CancelPhase)
	}
	if rec.Status != http.StatusOK {
		t.Errorf("status = %d, want 200", rec.Status)
	}
	if rec.Usage.InputTokens != 7 {
		t.Errorf("partial usage lost: input tokens = %d, want 7", rec.Usage.InputTokens)
	}
	s := router.MetricsSummary()
	if s["requests"] != int64(1) && s["requests"] != 1 {
		t.Errorf("summary requests = %v, want 1", s["requests"])
	}
	if s["aborted"] != int64(1) && s["aborted"] != 1 {
		t.Errorf("summary aborted = %v, want 1", s["aborted"])
	}
	if s["aborted_in_service"] != int64(1) && s["aborted_in_service"] != 1 {
		t.Errorf("summary aborted_in_service = %v, want 1", s["aborted_in_service"])
	}
}
