package claudeauto

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHonorDisabledThinking(t *testing.T) {
	kw := func(body []byte) (map[string]any, bool) {
		var m map[string]any
		if json.Unmarshal(body, &m) != nil {
			t.Fatalf("invalid output %s", body)
		}
		k, ok := m["chat_template_kwargs"].(map[string]any)
		return k, ok
	}
	// Disabled thinking gains the template switch the backend honors.
	out := honorDisabledThinking([]byte(`{"model":"local","thinking":{"type":"disabled"},"messages":[{"role":"user","content":"hi"}]}`))
	if k, ok := kw(out); !ok || k["enable_thinking"] != false {
		t.Fatalf("not mapped: %s", out)
	}
	// Existing kwargs are kept and extended.
	out = honorDisabledThinking([]byte(`{"thinking":{"type":"disabled"},"chat_template_kwargs":{"reasoning_effort":"low"}}`))
	if k, _ := kw(out); k["enable_thinking"] != false || k["reasoning_effort"] != "low" {
		t.Fatalf("kwargs not merged: %s", out)
	}
	// A JSON null is an omitted optional object, not a writable Go map.
	out = honorDisabledThinking([]byte(`{"thinking":{"type":"disabled"},"chat_template_kwargs":null}`))
	if k, ok := kw(out); !ok || k["enable_thinking"] != false {
		t.Fatalf("null kwargs not mapped: %s", out)
	}
	// Unchanged byte-for-byte: explicit client choice, thinking enabled, absent, or not JSON.
	for _, in := range []string{
		`{"thinking":{"type":"disabled"},"chat_template_kwargs":{"enable_thinking":true}}`,
		`{"thinking":{"type":"enabled","budget_tokens":1024}}`,
		`{"thinking":{"type":"adaptive"}}`,
		`{"model":"local","messages":[]}`,
		`{"thinking":{"type":"disabled"},"chat_template_kwargs":"odd"}`,
		`not json`,
	} {
		if got := string(honorDisabledThinking([]byte(in))); got != in {
			t.Errorf("changed %s -> %s", in, got)
		}
	}
}

func TestRouterHonorsThinkingWhenMappingKeepsBodyLength(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(w, r.Body)
	}))
	defer backend.Close()
	// Preserve request whitespace, as the vision route does. Adding kwargs and
	// compacting JSON can leave the length unchanged; length is not identity.
	router, err := StartRouter(backend.URL, backend.URL, true, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()
	body := `{"model":"local","thinking":{"type":"disabled"},"messages":[]}`
	mapped := honorDisabledThinking([]byte(body))
	padding := len(mapped) - len(body)
	if padding <= 0 {
		t.Fatal("fixture must add the missing template switch")
	}
	body += strings.Repeat(" ", padding)
	if len(body) != len(honorDisabledThinking([]byte(body))) {
		t.Fatal("fixture must exercise an equal-length transformation")
	}
	resp, err := http.Post(router.URL()+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK || string(got) != string(mapped) {
		t.Fatalf("disabled thinking not forwarded: status=%d body=%s", resp.StatusCode, got)
	}
}
