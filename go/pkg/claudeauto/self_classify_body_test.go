package claudeauto

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

// capturingBackend answers every request with a valid allow verdict and keeps
// the bodies it received, so a test can check what reached each backend.
func capturingBackend(t *testing.T) (*httptest.Server, func() []map[string]any) {
	t.Helper()
	var mu sync.Mutex
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"type":"message","role":"assistant","content":[{"type":"text","text":"<block>no</block>"}],"usage":{"input_tokens":11,"output_tokens":2}}`))
	}))
	t.Cleanup(srv.Close)
	return srv, func() []map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return append([]map[string]any(nil), bodies...)
	}
}

func thinkingSwitch(body map[string]any) (any, bool) {
	kwargs, ok := body["chat_template_kwargs"].(map[string]any)
	if !ok {
		return nil, false
	}
	value, ok := kwargs["enable_thinking"]
	return value, ok
}

func TestSelfClassifierBodyMatchesTheReviewerShape(t *testing.T) {
	decode := func(raw []byte) map[string]any {
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatalf("invalid output %s", raw)
		}
		return body
	}
	out := decode(selfClassifierBody([]byte(`{"model":"claude-x","max_tokens":8,"messages":[]}`)))
	if value, ok := thinkingSwitch(out); !ok || value != false {
		t.Fatalf("reasoning not turned off: %v", out)
	}
	if out["model"] != MainAlias || out["max_tokens"] != float64(32) {
		t.Fatalf("model/max_tokens = %v/%v, want %s/32", out["model"], out["max_tokens"], MainAlias)
	}
	if out := decode(selfClassifierBody([]byte(`{"max_tokens":4096}`))); out["max_tokens"] != float64(4096) {
		t.Fatalf("a larger output budget was lowered: %v", out["max_tokens"])
	}
	// The client's explicit choice and other kwargs survive.
	out = decode(selfClassifierBody([]byte(`{"chat_template_kwargs":{"enable_thinking":true,"x":1}}`)))
	if value, _ := thinkingSwitch(out); value != true {
		t.Fatalf("explicit enable_thinking overridden: %v", out)
	}
	out = decode(selfClassifierBody([]byte(`{"chat_template_kwargs":{"reasoning_effort":"low"}}`)))
	if value, _ := thinkingSwitch(out); value != false || out["chat_template_kwargs"].(map[string]any)["reasoning_effort"] != "low" {
		t.Fatalf("kwargs not merged: %v", out)
	}
	if got := string(selfClassifierBody([]byte(`not json`))); got != `not json` {
		t.Fatalf("non-JSON body changed: %s", got)
	}
}

func TestSelfClassifiedReviewRunsWithReasoningOff(t *testing.T) {
	backend, received := capturingBackend(t)
	router, path := newTestRouter(t, backend.URL, 1)
	post(t, router, classifierBody(16))
	post(t, router, `{"model":"local","messages":[{"role":"user","content":"write the code"}]}`)
	records := waitForRecords(t, path, 2)
	if records[0].Route != routeMain || records[0].Kind != kindClassifier {
		t.Fatalf("first record = %s/%s, want a main-route classifier", records[0].Route, records[0].Kind)
	}
	bodies := received()
	if len(bodies) != 2 {
		t.Fatalf("backend received %d requests, want 2", len(bodies))
	}
	if value, ok := thinkingSwitch(bodies[0]); !ok || value != false {
		t.Errorf("self-classified review reached the main model with reasoning on: %v", bodies[0]["chat_template_kwargs"])
	}
	if _, ok := thinkingSwitch(bodies[1]); ok {
		t.Errorf("ordinary main-model work gained a thinking switch: %v", bodies[1]["chat_template_kwargs"])
	}
}

func TestReviewerRouteBodyKeepsItsLaunchSettings(t *testing.T) {
	mainBackend, mainReceived := capturingBackend(t)
	reviewerBackend, reviewerReceived := capturingBackend(t)
	router, path := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	router.SetReviewerContext(65536)
	post(t, router, classifierBody(64))
	waitForRecords(t, path, 1)
	if got := reviewerReceived(); len(got) != 1 {
		t.Fatalf("reviewer received %d requests, want 1", len(got))
	} else if _, ok := thinkingSwitch(got[0]); ok {
		t.Errorf("reviewer body gained a thinking switch; the reviewer runs with --reasoning off at launch")
	}
	if got := mainReceived(); len(got) != 0 {
		t.Fatalf("main received %d requests, want 0", len(got))
	}
}

func TestOverflowedReviewOnMainRunsWithReasoningOff(t *testing.T) {
	mainBackend, mainReceived := capturingBackend(t)
	reviewerBackend, _ := capturingBackend(t)
	router, path := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	router.SetReviewerContext(64)
	post(t, router, classifierBody(4096))
	records := waitForRecords(t, path, 1)
	if records[0].Route != routeMain {
		t.Fatalf("route = %q, want %q", records[0].Route, routeMain)
	}
	got := mainReceived()
	if len(got) != 1 {
		t.Fatalf("main received %d requests, want 1", len(got))
	}
	if value, ok := thinkingSwitch(got[0]); !ok || value != false {
		t.Errorf("overflowed review reached the main model with reasoning on: %v", got[0]["chat_template_kwargs"])
	}
}
