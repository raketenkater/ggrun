package benchmark

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// speedServer serves a large context at a fixed prompt speed (one token per
// character) and records every chat prompt's size. Calls after the first
// real canary prompt report prefix reuse so a full canary can pass.
func speedServer(t *testing.T, promptTPS float64) (*httptest.Server, func() []int) {
	t.Helper()
	var mu sync.Mutex
	var prompts []int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/props") {
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"default_generation_settings": map[string]interface{}{"n_ctx": 262144},
			})
			return
		}
		if strings.HasSuffix(r.URL.Path, "/tokenize") {
			var body struct {
				Content string `json:"content"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"tokens": make([]int, len(body.Content))})
			return
		}
		var body struct {
			Messages []canaryMessage `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		prompt := 0
		for _, m := range body.Messages {
			prompt += len(m.Content)
		}
		mu.Lock()
		prompts = append(prompts, prompt)
		big := 0
		for _, p := range prompts[:len(prompts)-1] {
			if p > 100 {
				big++
			}
		}
		mu.Unlock()
		cached := 0
		if big > 0 {
			cached = prompt * 4 / 5
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": "GGRUN_OK"}}},
			"usage":   map[string]interface{}{"prompt_tokens": prompt},
			"timings": map[string]interface{}{"cache_n": cached, "prompt_per_second": promptTPS},
		})
	}))
	return server, func() []int {
		mu.Lock()
		defer mu.Unlock()
		return append([]int(nil), prompts...)
	}
}

func largest(values []int) int {
	m := 0
	for _, v := range values {
		if v > m {
			m = v
		}
	}
	return m
}

func fullCanaryPrompt(t *testing.T, tps float64) int {
	t.Helper()
	server, prompts := speedServer(t, tps)
	defer server.Close()
	if _, err := (&Runner{BaseURL: server.URL, Model: "local"}).RunCacheCanary(); err != nil {
		t.Fatal(err)
	}
	return largest(prompts())
}

// A two-thread CPU runner prefilled at 2.6 tok/s; the ~7k-token canary held
// the launch for about 43 minutes before it could report ready. Under a budget
// the check must stay bounded, verify the endpoint, and refuse to claim reuse.
func TestCacheCanaryTooSlowToProveReuseVerifiesEndpointOnly(t *testing.T) {
	server, prompts := speedServer(t, 2.6)
	defer server.Close()
	result, err := (&Runner{BaseURL: server.URL, Model: "local", PrefillBudget: 5 * time.Minute}).RunCacheCanary()
	if err != nil {
		t.Fatal(err)
	}
	if !result.Functional || result.Passed {
		t.Fatalf("want functional, unproven reuse: %+v", result)
	}
	if !strings.Contains(result.Reason, "startup budget") {
		t.Fatalf("reason must say why reuse was not proven: %q", result.Reason)
	}
	if got := largest(prompts()); float64(got) > 2.6*300 {
		t.Fatalf("sent a %d-token prompt; at 2.6 tok/s that exceeds the 5 minute budget", got)
	}
}

// Between the two limits the canary shrinks to the budget but still proves
// reuse; it must never send more than the budget allows.
func TestCacheCanaryShrinksToTheBudget(t *testing.T) {
	full := fullCanaryPrompt(t, 10)
	server, prompts := speedServer(t, 10)
	defer server.Close()
	result, err := (&Runner{BaseURL: server.URL, Model: "local", PrefillBudget: 5 * time.Minute}).RunCacheCanary()
	if err != nil {
		t.Fatal(err)
	}
	got := largest(prompts())
	if got >= full {
		t.Skipf("full canary (%d tokens) already fits 3000 tokens", full)
	}
	if float64(got) > 10*300*1.15 {
		t.Fatalf("shrunk canary sent %d tokens; budget allows about 3000", got)
	}
	if got < canaryCacheProofTokens {
		t.Fatalf("shrunk canary (%d tokens) is too short to prove reuse", got)
	}
	if !result.Functional || !result.Passed {
		t.Fatalf("a shrunk canary that fits must still verify reuse: %+v", result)
	}
}

// Machines that prefill the full canary within the budget, or whose speed is
// unknown, run exactly the check they ran before.
func TestCacheCanaryUnchangedWhenFastOrUnknown(t *testing.T) {
	for _, tps := range []float64{2000, 0} {
		full := fullCanaryPrompt(t, tps)
		server, prompts := speedServer(t, tps)
		result, err := (&Runner{BaseURL: server.URL, Model: "local", PrefillBudget: 5 * time.Minute}).RunCacheCanary()
		server.Close()
		if err != nil {
			t.Fatal(err)
		}
		if got := largest(prompts()); got != full {
			t.Fatalf("tps %.0f: canary prompt %d, want the unbudgeted %d", tps, got, full)
		}
		if !result.Passed {
			t.Fatalf("tps %.0f: full canary should pass on this fixture: %+v", tps, result)
		}
	}
}
