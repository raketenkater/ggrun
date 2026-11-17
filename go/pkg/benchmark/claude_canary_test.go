package benchmark

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/claudeauto"
)

func TestClaudeRouterCanaryExercisesAnthropicEndpointAndTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Fatalf("canary path = %s", r.URL.Path)
		}
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if _, ok := body["tools"]; !ok {
			t.Fatal("Claude canary omitted tool schema")
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"content": []map[string]string{{"type": "text", "text": "GGRUN_OK"}},
		})
	}))
	defer server.Close()
	if err := (&Runner{BaseURL: server.URL, Model: "local"}).RunClaudeRouterCanary(); err != nil {
		t.Fatal(err)
	}
}

// The Anthropic path shares validCanaryOutput, so it inherits the same bar: a
// thinking model that answers in prose has routed correctly through /v1/messages
// and must not be reported as a broken gateway.
func TestClaudeRouterCanaryAcceptsBoundedThinkingPreamble(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"content": []map[string]string{{"type": "text", "text": "The user asks for a deterministic verification token."}},
		})
	}))
	defer server.Close()
	if err := (&Runner{BaseURL: server.URL, Model: "local"}).RunClaudeRouterCanary(); err != nil {
		t.Fatalf("bounded thinking-model answer rejected: %v", err)
	}
}

func TestClaudeRouterCanaryRejectsEmptyOrUnboundedOutput(t *testing.T) {
	for _, tc := range []struct {
		name string
		text string
	}{
		{"empty", ""},
		{"unbounded", strings.TrimSpace(strings.Repeat("rambling ", maxFunctionalCanaryFields+1))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewEncoder(w).Encode(map[string]interface{}{
					"content": []map[string]string{{"type": "text", "text": tc.text}},
				})
			}))
			defer server.Close()
			if err := (&Runner{BaseURL: server.URL, Model: "local"}).RunClaudeRouterCanary(); err == nil {
				t.Fatalf("%s Anthropic completion activated Claude workload profile", tc.name)
			}
		})
	}
}

func TestClaudeReviewerCanaryExercisesClassifierRouteWithoutTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			System []struct {
				Text string `json:"text"`
			} `json:"system"`
			Tools json.RawMessage `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if len(body.System) == 0 || !strings.Contains(body.System[0].Text, claudeauto.ClassifierMarker) {
			t.Fatal("reviewer canary omitted Claude Code's classifier marker")
		}
		if len(body.Tools) != 0 {
			t.Fatal("reviewer canary presented a synthetic tool to the safety model")
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"content": []map[string]string{{"type": "text", "text": "<block>no</block>"}},
		})
	}))
	defer server.Close()
	if err := (&Runner{BaseURL: server.URL, Model: "local"}).RunClaudeReviewerCanary(); err != nil {
		t.Fatal(err)
	}
}

func TestClaudeReviewerCanaryRejectsToolOnlyOrMalformedVerdict(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content []map[string]interface{}
	}{
		{"tool-only", []map[string]interface{}{{"type": "tool_use", "name": "ggrun_canary_noop", "input": map[string]interface{}{}}}},
		{"prose", []map[string]interface{}{{"type": "text", "text": "This action looks safe."}}},
		{"wrong-verdict", []map[string]interface{}{{"type": "text", "text": "<block>yes</block>"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_ = json.NewEncoder(w).Encode(map[string]interface{}{"content": tc.content})
			}))
			defer server.Close()
			if err := (&Runner{BaseURL: server.URL, Model: "local"}).RunClaudeReviewerCanary(); err == nil {
				t.Fatal("unusable reviewer verdict activated Claude workload profile")
			}
		})
	}
}

// S-01: the must-block canary passes only on a block verdict. It is a
// warning in the launch, so a weak reviewer degrades Auto mode visibly
// instead of stopping the launch.
func TestReviewerBlockCanaryNeedsABlockVerdict(t *testing.T) {
	for answer, ok := range map[string]bool{
		"<block>yes</block>":                   true,
		"<block>yes":                           true,
		"<block>no</block>":                    false,
		"<block>no":                            false,
		"Looks fine to me.":                    false,
		"<block>yes</block> <block>no</block>": false,
	} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var body map[string]interface{}
			_ = json.NewDecoder(r.Body).Decode(&body)
			raw, _ := json.Marshal(body["system"])
			if !strings.Contains(string(raw), claudeauto.ClassifierMarker) {
				t.Errorf("must-block canary is not a classifier request: %s", raw)
			}
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"content": []map[string]string{{"type": "text", "text": answer}},
			})
		}))
		err := (&Runner{BaseURL: server.URL, Model: "local"}).RunClaudeReviewerBlockCanary()
		server.Close()
		if ok != (err == nil) {
			t.Errorf("%q: err %v, want ok=%t", answer, err, ok)
		}
	}
}
