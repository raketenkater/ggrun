package benchmark

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// The canary asks /props for the context a slot actually got and /tokenize for
// the model's real expansion. These fixtures answer chat completions and count
// calls, so an unhandled side request would shift their call numbering. Decline
// both explicitly; the code under test falls back when they are unavailable.
func declineSideRequests(w http.ResponseWriter, r *http.Request) bool {
	switch {
	case strings.HasSuffix(r.URL.Path, "/props"), strings.HasSuffix(r.URL.Path, "/tokenize"):
		w.WriteHeader(http.StatusNotFound)
		return true
	}
	return false
}

func TestCacheCanaryPassesAppendBranchAndReplay(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if declineSideRequests(w, r) {
			return
		}
		call := calls.Add(1)
		cached := 0
		prompt := 1800
		switch call {
		case 2:
			cached = 1500
			prompt = 1900
		case 3:
			cached = 900
			prompt = 1100
		case 4:
			cached = 1050
			prompt = 1100
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": "GGRUN_OK"}}},
			"usage":   map[string]interface{}{"prompt_tokens": prompt},
			"timings": map[string]interface{}{"cache_n": cached, "prompt_per_second": 42.0},
		})
	}))
	defer server.Close()

	result, err := (&Runner{BaseURL: server.URL, Model: "local"}).RunCacheCanary()
	if err != nil {
		t.Fatal(err)
	}
	if !result.Passed || !result.Supported || !result.Functional || !result.Deterministic {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestCacheCanaryRejectsOneCheckpointBranchMiss(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if declineSideRequests(w, r) {
			return
		}
		call := calls.Add(1)
		cached := 0
		if call == 2 || call == 4 {
			cached = 1600
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": "GGRUN_OK"}}},
			"usage":   map[string]interface{}{"prompt_tokens": 1800},
			"timings": map[string]interface{}{"cache_n": cached},
		})
	}))
	defer server.Close()

	result, err := (&Runner{BaseURL: server.URL, Model: "local"}).RunCacheCanary()
	if err != nil {
		t.Fatal(err)
	}
	if result.Passed || result.BranchCachedTokens != 0 {
		t.Fatalf("branch miss was accepted: %+v", result)
	}
}

func TestCacheCanaryReportsUnsupportedTelemetry(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if declineSideRequests(w, r) {
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": "GGRUN_OK"}}},
			"usage":   map[string]interface{}{"prompt_tokens": 100},
		})
	}))
	defer server.Close()
	result, err := (&Runner{BaseURL: server.URL, Model: "local"}).RunCacheCanary()
	if err != nil {
		t.Fatal(err)
	}
	if result.Supported || result.Passed || !result.Functional {
		t.Fatalf("unsupported telemetry misclassified: %+v", result)
	}
}

// A thinking model (Nanbeige, DeepSeek-R1 style) writes an analytic preamble and
// never emits the bare token, yet it is answering. Requiring literal GGRUN_OK
// rejected that working backend on 2026-08-03, so "functional" means the
// endpoint returned a bounded non-empty answer. Determinism and cache reuse stay
// strict, and this fixture still has to reach Passed through them.
func TestCacheCanaryAcceptsBoundedThinkingPreamble(t *testing.T) {
	const preamble = "The user is providing a series of evidence entries; this is a deterministic verification prompt."
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if declineSideRequests(w, r) {
			return
		}
		call := calls.Add(1)
		cached := 0
		prompt := 1800
		switch call {
		case 2:
			cached = 1500
			prompt = 1900
		case 3:
			cached = 900
			prompt = 1100
		case 4:
			cached = 1050
			prompt = 1100
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": preamble}}},
			"usage":   map[string]interface{}{"prompt_tokens": prompt},
			"timings": map[string]interface{}{"cache_n": cached, "prompt_per_second": 42.0},
		})
	}))
	defer server.Close()

	result, err := (&Runner{BaseURL: server.URL, Model: "local"}).RunCacheCanary()
	if err != nil {
		t.Fatal(err)
	}
	if !result.Functional {
		t.Fatalf("bounded thinking-model answer rejected as non-functional: %+v", result)
	}
	if !result.Passed || !result.Deterministic || !result.Supported {
		t.Fatalf("thinking-model canary did not pass on good cache evidence: %+v", result)
	}
}

// What "functional" still rejects: silence, and a backend that ignores the token
// cap. Neither is an answer, so neither may activate a profile.
func TestCacheCanaryRejectsEmptyOrUnboundedOutput(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content string
	}{
		{"empty", ""},
		{"unbounded", strings.TrimSpace(strings.Repeat("rambling ", maxFunctionalCanaryFields+1))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if declineSideRequests(w, r) {
					return
				}
				_ = json.NewEncoder(w).Encode(map[string]interface{}{
					"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": tc.content}}},
					"usage":   map[string]interface{}{"prompt_tokens": 1800},
					"timings": map[string]interface{}{"cache_n": 1600, "prompt_per_second": 42.0},
				})
			}))
			defer server.Close()

			result, err := (&Runner{BaseURL: server.URL, Model: "local"}).RunCacheCanary()
			if err != nil {
				t.Fatal(err)
			}
			if result.Functional || result.Passed {
				t.Fatalf("%s completion activated profile: %+v", tc.name, result)
			}
		})
	}
}

// Relaxing "functional" moved the burden of rejecting a misbehaving model onto
// determinism: an identical branch replay that answers differently must still
// fail, even though every individual answer is now accepted as functional.
func TestCacheCanaryRejectsNonDeterministicReplay(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if declineSideRequests(w, r) {
			return
		}
		call := calls.Add(1)
		content := "a bounded deterministic answer"
		if call == 4 {
			content = "a bounded but different answer"
		}
		cached := 0
		prompt := 1800
		switch call {
		case 2:
			cached = 1500
			prompt = 1900
		case 3:
			cached = 900
			prompt = 1100
		case 4:
			cached = 1050
			prompt = 1100
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": content}}},
			"usage":   map[string]interface{}{"prompt_tokens": prompt},
			"timings": map[string]interface{}{"cache_n": cached, "prompt_per_second": 42.0},
		})
	}))
	defer server.Close()

	result, err := (&Runner{BaseURL: server.URL, Model: "local"}).RunCacheCanary()
	if err != nil {
		t.Fatal(err)
	}
	if !result.Functional {
		t.Fatalf("fixture should stay functional so determinism is the rejecting gate: %+v", result)
	}
	if result.Deterministic || result.Passed {
		t.Fatalf("divergent replay activated profile: %+v", result)
	}
}

// contextBoundServer mimics llama-server closely enough to matter: it tokenizes
// with a fixed expansion and rejects any prompt that does not fit n_ctx, which
// is what a real backend does ("request (15873 tokens) exceeds the available
// context size (2048 tokens)") and what the fixed-size canary always tripped.
func contextBoundServer(t *testing.T, nCtx, tokensPerChar int) *httptest.Server {
	t.Helper()
	var calls atomic.Int32
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// This fixture models a real server, so it answers both side requests
		// and counts neither: /props with the context a slot actually got, and
		// /tokenize with the real expansion.
		if strings.HasSuffix(r.URL.Path, "/props") {
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"default_generation_settings": map[string]interface{}{"n_ctx": nCtx},
			})
			return
		}
		if strings.HasSuffix(r.URL.Path, "/tokenize") {
			var body struct {
				Content string `json:"content"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"tokens": make([]int, len(body.Content)*tokensPerChar),
			})
			return
		}
		var body struct {
			Messages []canaryMessage `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		prompt := 0
		for _, m := range body.Messages {
			prompt += len(m.Content) * tokensPerChar
		}
		if prompt > nCtx {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"request exceeds the available context size","type":"exceed_context_size_error"}}`))
			return
		}
		call := calls.Add(1)
		cached := 0
		switch call {
		case 2:
			cached = prompt * 4 / 5
		case 3, 4:
			cached = prompt * 4 / 5
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": "GGRUN_OK"}}},
			"usage":   map[string]interface{}{"prompt_tokens": prompt},
			"timings": map[string]interface{}{"cache_n": cached, "prompt_per_second": 42.0},
		})
	}))
}

func TestCacheCanaryFitsTheServedContext(t *testing.T) {
	server := contextBoundServer(t, 2048, 1)
	defer server.Close()

	// Even with no planned context the canary now asks /props and sizes itself,
	// so it fits without being told. That is stronger than the old behaviour,
	// which overflowed unless the caller passed ContextTokens.
	if _, err := (&Runner{BaseURL: server.URL, Model: "local"}).RunCacheCanary(); err != nil {
		t.Fatalf("canary should size itself from /props with no planned context: %v", err)
	}

	result, err := (&Runner{BaseURL: server.URL, Model: "local", ContextTokens: 2048}).RunCacheCanary()
	if err != nil {
		t.Fatalf("context-sized canary must fit the server it was launched against: %v", err)
	}
	if !result.Functional {
		t.Fatalf("context-sized canary did not verify the endpoint: %+v", result)
	}
}

func TestCacheCanaryReportsUnprovableReuseInsteadOfFailing(t *testing.T) {
	server := contextBoundServer(t, 600, 1)
	defer server.Close()

	result, err := (&Runner{BaseURL: server.URL, Model: "local", ContextTokens: 600}).RunCacheCanary()
	if err != nil {
		t.Fatalf("a small context must not fail a healthy server: %v", err)
	}
	if !result.Functional {
		t.Fatalf("endpoint verification should still happen: %+v", result)
	}
	if result.Passed {
		t.Fatalf("a context too small for two checkpoints cannot claim verified reuse: %+v", result)
	}
	if !strings.Contains(result.Reason, "too small to prove prefix reuse") {
		t.Fatalf("reason must say what was not proven, got %q", result.Reason)
	}
}

// The planned context is a request, not a grant. llama.cpp clamps a slot to the
// model's training context: a release build asked for 2048 against stories15M
// (n_ctx_train 128), got a 128-token slot, and the canary sized from the plan
// sent 1779 tokens into it. Ask the server what it actually served.
func TestCacheCanaryPrefersTheServedContextOverThePlanned(t *testing.T) {
	// nCtx 128 is what /props reports; 2048 is what the caller planned.
	server := contextBoundServer(t, 128, 1)
	defer server.Close()

	result, err := (&Runner{BaseURL: server.URL, Model: "local", ContextTokens: 2048}).RunCacheCanary()
	if err != nil {
		t.Fatalf("canary must fit the context the server actually served: %v", err)
	}
	if !result.Functional {
		t.Fatalf("endpoint verification should still happen: %+v", result)
	}
	if result.Passed {
		t.Fatalf("128 tokens cannot hold two checkpoints, so reuse is unprovable: %+v", result)
	}
}
