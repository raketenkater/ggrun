package claudeauto

import (
	"encoding/json"
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
