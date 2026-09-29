package benchmark

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// fakeReasoningRouter answers like a small reasoning model: below the larger
// cap it spends every token thinking and returns no text.
func fakeReasoningRouter(t *testing.T, caps *[]int, answer string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			MaxTokens int `json:"max_tokens"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		*caps = append(*caps, body.MaxTokens)
		if body.MaxTokens < canaryReasoningMaxTokens {
			_, _ = w.Write([]byte(`{"content":[{"type":"thinking","thinking":"Let me think"}],"stop_reason":"max_tokens"}`))
			return
		}
		_, _ = w.Write([]byte(`{"content":[{"type":"thinking","thinking":"ok"},{"type":"text","text":"` + answer + `"}],"stop_reason":"end_turn"}`))
	}))
}

func TestRouterCanaryRetriesOnceWhenReasoningExhaustsTheCap(t *testing.T) {
	var caps []int
	server := fakeReasoningRouter(t, &caps, "GGRUN_OK")
	defer server.Close()
	if err := (&Runner{BaseURL: server.URL, Model: "local"}).RunClaudeRouterCanary(); err != nil {
		t.Fatalf("reasoning model failed the canary: %v", err)
	}
	if len(caps) != 2 || caps[0] != 96 || caps[1] != canaryReasoningMaxTokens {
		t.Fatalf("output caps = %v, want one bounded retry", caps)
	}
}

func TestReviewerCanaryRetriesOnceWhenReasoningExhaustsTheCap(t *testing.T) {
	var caps []int
	server := fakeReasoningRouter(t, &caps, "<block>no</block>")
	defer server.Close()
	if err := (&Runner{BaseURL: server.URL, Model: "local"}).RunClaudeReviewerCanary(); err != nil {
		t.Fatalf("reasoning reviewer failed the canary: %v", err)
	}
	if len(caps) != 2 {
		t.Fatalf("output caps = %v", caps)
	}
}

// An empty answer that the model itself ended is a real failure, not a
// truncation, and must not buy a larger budget.
func TestEmptyAnswerWithoutTruncationIsNotRetried(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		_, _ = w.Write([]byte(`{"content":[],"stop_reason":"end_turn"}`))
	}))
	defer server.Close()
	if err := (&Runner{BaseURL: server.URL, Model: "local"}).RunClaudeRouterCanary(); err == nil {
		t.Fatal("empty answer accepted")
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want no retry", calls)
	}
}

func canaryServer(t *testing.T, response string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(response))
	}))
}

// Qwen3.5-4B through the live router: stop_reason tool_use with two calls to
// the declared no-op tool and no text. Transport worked; the model disobeyed.
func TestRouterCanaryAcceptsWellFormedCallsToTheDeclaredTool(t *testing.T) {
	server := canaryServer(t, `{"content":[{"type":"thinking","thinking":"hm"},{"type":"tool_use","id":"a","name":"ggrun_canary_noop","input":{}},{"type":"tool_use","id":"b","name":"ggrun_canary_noop","input":{}}],"stop_reason":"tool_use"}`)
	defer server.Close()
	if err := (&Runner{BaseURL: server.URL, Model: "local"}).RunClaudeRouterCanary(); err != nil {
		t.Fatalf("well-formed declared tool call rejected: %v", err)
	}
}

func TestRouterCanaryRejectsUndeclaredOrMalformedToolCalls(t *testing.T) {
	for name, body := range map[string]string{
		"undeclared": `{"content":[{"type":"tool_use","id":"a","name":"Bash","input":{"command":"ls"}}],"stop_reason":"tool_use"}`,
		"malformed":  `{"content":[{"type":"tool_use","id":"a","name":"ggrun_canary_noop","input":"{"}],"stop_reason":"tool_use"}`,
		"no-reason":  `{"content":[{"type":"tool_use","id":"a","name":"ggrun_canary_noop","input":{}}],"stop_reason":"end_turn"}`,
	} {
		server := canaryServer(t, body)
		err := (&Runner{BaseURL: server.URL, Model: "local"}).RunClaudeRouterCanary()
		server.Close()
		if err == nil {
			t.Fatalf("%s tool response accepted", name)
		}
	}
}
