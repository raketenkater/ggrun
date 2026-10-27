package remoteclaude

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/clientkit"
	"github.com/raketenkater/ggrun/pkg/claudeauto"
)

const testKey = "0123456789abcdef0123456789abcdef"

func testProfile() clientkit.Profile {
	return clientkit.Profile{ProtocolVersion: 1, ModelID: "test-shape", ModelName: "test", MainAlias: "local", UtilityAlias: "local-fast", ContextPerSlot: 65536, Slots: 1, Policy: clientkit.Policy{Version: 1, CompactWindow: 65536, CompactPercent: 75, Effort: "xhigh", TimeoutMS: clientkit.MaxTimeoutMS}}
}

func TestProtocolRoundTripPreservesThinkingToolsUsageAndHeaders(t *testing.T) {
	body := `{"id":"message","type":"message","role":"assistant","content":[{"type":"thinking","thinking":"checking","signature":"opaque"},{"type":"tool_use","id":"tool-1","name":"Read","input":{"file_path":"client.txt"}}],"stop_reason":"tool_use","usage":{"input_tokens":30,"output_tokens":5,"cache_read_input_tokens":20}}`
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Anthropic-Version") != "2023-06-01" || r.Header.Get("Anthropic-Beta") != "future-beta" {
			t.Error("protocol header lost")
		}
		w.Header().Set("Request-Id", "fixture-request")
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, body)
	}))
	defer up.Close()
	h, err := NewHandler(up.URL, testKey, testProfile())
	if err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{"model":"local-fast","messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"previous","content":"ok"}]}]}`))
	r.Header.Set("Authorization", "Bearer "+testKey)
	r.Header.Set("Anthropic-Version", "2023-06-01")
	r.Header.Set("Anthropic-Beta", "future-beta")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 200 || w.Body.String() != body || w.Header().Get("Request-Id") != "fixture-request" {
		t.Fatal("protocol response changed", w.Code, w.Body.String())
	}
}

func TestChangedCapacityRejectsOldClientBeforeInference(t *testing.T) {
	calls := 0
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++ }))
	defer up.Close()
	h, err := NewHandler(up.URL, testKey, testProfile())
	if err != nil {
		t.Fatal(err)
	}
	for _, headers := range []map[string]string{{"X-Ggrun-Model-ID": "old-model"}, {"X-Ggrun-Context": "131072"}} {
		r := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{"model":"local","messages":[]}`))
		r.Header.Set("Authorization", "Bearer "+testKey)
		for k, v := range headers {
			r.Header.Set(k, v)
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != 400 {
			t.Fatal("stale session admitted", w.Code)
		}
	}
	if calls != 0 {
		t.Fatal("stale session consumed inference")
	}
}

func TestQueuedDisconnectNeverStartsAnExtraInference(t *testing.T) {
	var calls atomic.Int64
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: ping\ndata: {}\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer up.Close()
	router, err := claudeauto.StartRouter(up.URL, up.URL, false, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()
	h, err := NewHandler(router.URL(), testKey, testProfile())
	if err != nil {
		t.Fatal(err)
	}
	gateway := httptest.NewServer(h)
	defer gateway.Close()
	client := &http.Client{Timeout: 5 * time.Second}
	send := func(ctx context.Context) (*http.Response, error) {
		r, _ := http.NewRequestWithContext(ctx, "POST", gateway.URL+"/v1/messages", strings.NewReader(`{"model":"local","messages":[{"role":"user","content":"ordinary coding turn"}],"stream":true}`))
		r.Header.Set("Authorization", "Bearer "+testKey)
		return client.Do(r)
	}
	first, err := send(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer first.Body.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		resp, err := send(ctx)
		if resp != nil {
			resp.Body.Close()
		}
		done <- err
	}()
	waitQueued := func(want int) {
		t.Helper()
		deadline := time.Now().Add(3 * time.Second)
		for time.Now().Before(deadline) {
			r, _ := http.NewRequest("GET", gateway.URL+"/ggrun/router", nil)
			r.Header.Set("Authorization", "Bearer "+testKey)
			resp, err := client.Do(r)
			if err != nil {
				t.Fatal(err)
			}
			var s struct{ Queued int }
			json.NewDecoder(resp.Body).Decode(&s)
			resp.Body.Close()
			if s.Queued == want {
				return
			}
			time.Sleep(10 * time.Millisecond)
		}
		t.Fatalf("queue did not reach %d", want)
	}
	waitQueued(1)
	cancel()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("queued call completed instead of cancelling")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("queued disconnect stuck")
	}
	waitQueued(0)
	if calls.Load() != 1 {
		t.Fatal("cancelled queued request consumed inference", calls.Load())
	}
	first.Body.Close()
	third, err := send(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	third.Body.Close()
	if calls.Load() != 2 {
		t.Fatal("admission did not recover after disconnect", calls.Load())
	}
}

func TestAuthenticatedSurfacePreservesPrivateBackend(t *testing.T) {
	calls := 0
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Header.Get("Authorization") != "" || r.Header.Get("X-Api-Key") != "" {
			t.Error("gateway credentials leaked upstream")
		}
		b, _ := io.ReadAll(r.Body)
		_, _ = w.Write(b)
	}))
	defer up.Close()
	h, err := NewHandler(up.URL, testKey, testProfile())
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/v1/messages", "/ggrun/client", "/ggrun/router", "/v1/models", "/health"} {
		r := httptest.NewRequest("GET", path, nil)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != 401 {
			t.Errorf("%s unauthenticated=%d", path, w.Code)
		}
	}
	if calls != 0 {
		t.Fatal("unauthenticated call reached model")
	}
	for _, path := range []string{"/slots", "/props", "/completion", "/v1/chat/completions"} {
		r := httptest.NewRequest("POST", path, nil)
		r.Header.Set("Authorization", "Bearer "+testKey)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != 404 {
			t.Errorf("exposed %s", path)
		}
	}
	body := `{"model":"local","messages":[],"new_future_field":{"keep":true}}`
	r := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	r.Header.Set("Authorization", "Bearer "+testKey)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 200 || w.Body.String() != body {
		t.Fatalf("body not preserved: %d %s", w.Code, w.Body)
	}
	r = httptest.NewRequest("GET", "/ggrun/client", nil)
	r.Header.Set("X-Api-Key", testKey)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, r)
	var p clientkit.Profile
	if json.Unmarshal(w.Body.Bytes(), &p) != nil || p.Validate() != nil {
		t.Fatalf("bad profile %s", w.Body)
	}
	r = httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{"model":"cloud","messages":[]}`))
	r.Header.Set("Authorization", "Bearer "+testKey)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 400 {
		t.Fatal("unsupported model accepted")
	}
}

func TestGatewayStreamsAndPropagatesDisconnect(t *testing.T) {
	cancelled := make(chan struct{})
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: ping\ndata: {}\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(cancelled)
	}))
	defer up.Close()
	h, err := NewHandler(up.URL, testKey, testProfile())
	if err != nil {
		t.Fatal(err)
	}
	gateway := httptest.NewServer(h)
	defer gateway.Close()
	r, _ := http.NewRequest("POST", gateway.URL+"/v1/messages", strings.NewReader(`{"model":"local","messages":[]}`))
	r.Header.Set("Authorization", "Bearer "+testKey)
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Do(r)
	if err != nil {
		t.Fatal(err)
	}
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	if err != nil || line != "event: ping\n" {
		t.Fatalf("stream buffered: %q %v", line, err)
	}
	_ = resp.Body.Close()
	select {
	case <-cancelled:
	case <-time.After(2 * time.Second):
		t.Fatal("disconnect not propagated")
	}
}
