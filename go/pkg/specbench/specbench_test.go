package specbench

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestPromptsIncludeOneLongContextCase(t *testing.T) {
	short := Prompts(false)
	long := Prompts(true)
	if len(short) != 9 || len(long) != 9 {
		t.Fatalf("prompt count = %d/%d, want 9/9", len(short), len(long))
	}
	if len(long[len(long)-1].Text) < 300000 {
		t.Fatalf("long prompt is unexpectedly short: %d bytes", len(long[len(long)-1].Text))
	}
	if short[len(short)-1].Text == long[len(long)-1].Text {
		t.Fatal("60k run did not replace the review prompt")
	}
}

func TestRunnerRepeatedParallelHarness(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"content": "GGRUN-CODE-OK GGRUN-EXPLANATION-OK GGRUN-SUMMARY-OK GGRUN-FACTUAL-OK GGRUN-TRANSLATION-OK GGRUN-CREATIVE-OK GGRUN-MATH-OK GGRUN-TOOL-PLAN-OK GGRUN-REVIEW-OK"}}},
			"usage":   map[string]any{"prompt_tokens": 120, "completion_tokens": 24},
			"timings": map[string]any{
				"prompt_per_second": 800.0, "predicted_per_second": 12.0,
				"draft_n": 40, "draft_n_accepted": 30,
			},
		})
	}))
	defer server.Close()

	result := (&Runner{BaseURL: server.URL, Model: "local", Timeout: time.Second, Rounds: 2, Parallel: 4}).Run()
	if len(result.Samples) != 18 {
		t.Fatalf("samples = %d, want 18", len(result.Samples))
	}
	if calls.Load() != 19 { // warmup plus 18 measured requests
		t.Fatalf("calls = %d, want 19", calls.Load())
	}
	if !result.CorrectnessPassed || !result.StabilityPassed {
		t.Fatalf("unexpected failed result: %+v", result)
	}
	if result.MedianGenerateTPS != 12 || result.MedianPromptTPS != 800 {
		t.Fatalf("unexpected medians: %+v", result)
	}
	if result.DraftAcceptRate != 0.75 {
		t.Fatalf("acceptance = %.2f, want .75", result.DraftAcceptRate)
	}
}
