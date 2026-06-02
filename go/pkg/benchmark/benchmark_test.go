package benchmark

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEstimateTokens(t *testing.T) {
	if got := estimateTokens("hello world"); got != 2 {
		t.Fatalf("expected ~2 tokens for 11 chars, got %d", got)
	}
}

func TestResultJSON(t *testing.T) {
	r := &Result{
		Model:     "test",
		GenTPS:    15.5,
		Timestamp: 1234567890,
	}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if len(data) == 0 {
		t.Fatalf("empty json")
	}
}

func TestChatParsesUsageAndTimings(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"choices":[{"message":{"content":"hello world"}}],
			"usage":{"prompt_tokens":12,"completion_tokens":5},
			"timings":{"prompt_per_second":123.5,"predicted_per_second":45.5,"draft_n":40,"draft_n_accepted":18}
		}`))
	}))
	defer srv.Close()

	r := &Runner{BaseURL: srv.URL, Model: "test"}
	res, err := r.chat("hello", 8)
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	if res.PromptTokens != 12 || res.CompletionTokens != 5 {
		t.Fatalf("usage mismatch: %#v", res)
	}
	if res.PromptTPS != 123.5 || res.GenTPS != 45.5 {
		t.Fatalf("timings mismatch: %#v", res)
	}
	if res.DraftTokens != 40 || res.DraftAccepted != 18 {
		t.Fatalf("draft timings mismatch: %#v", res)
	}
}

func TestDraftAcceptRate(t *testing.T) {
	if got := draftAcceptRate(40, 10); got != 0.25 {
		t.Fatalf("expected 0.25, got %f", got)
	}
	if got := draftAcceptRate(0, 10); got != 0 {
		t.Fatalf("expected 0 for no draft tokens, got %f", got)
	}
}
