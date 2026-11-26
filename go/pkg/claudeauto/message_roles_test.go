package claudeauto

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestUnsupportedSystemRoleRejectedBeforeInference(t *testing.T) {
	calls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++; io.Copy(w, r.Body) }))
	defer backend.Close()
	router, err := StartRouter(backend.URL, backend.URL, false, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()
	for _, model := range []string{"local", "local-fast"} {
		body := `{"model":"` + model + `","messages":[{"role":"user","content":"inspect"},{"role":"assistant","content":"delegated"},{"role":"system","content":[{"type":"text","text":"worker report: four tests passed"}],"cache_control":{"type":"ephemeral"}}]}`
		response, err := http.Post(router.URL()+"/v1/messages", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		raw, _ := io.ReadAll(response.Body)
		response.Body.Close()
		if response.StatusCode != 400 || !strings.Contains(string(raw), `Unexpected role \"system\"`) {
			t.Fatalf("capability rejection not recognizable: %d %s", response.StatusCode, raw)
		}
		var result struct {
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(raw, &result) != nil || result.Error.Type != "invalid_request_error" {
			t.Fatal("not an Anthropic error")
		}
		// These are the observed Claude 2.1.307 fallback matcher fragments.
		if !strings.Contains(result.Error.Message, "Unexpected role") || !strings.Contains(result.Error.Message, "input message role") {
			t.Fatalf("Claude capability retry would not match: %s", result.Error.Message)
		}
	}
	if calls != 0 {
		t.Fatal("unsupported role consumed inference", calls)
	}
	// Claude's documented retry carries the report as user content. Preserve all
	// bytes and future fields, along with ordinary top-level system instructions.
	body := `{"model":"local","system":[{"type":"text","text":"trusted instructions"}],"future_field":true,"messages":[{"role":"user","content":"worker report: four tests passed"}]}`
	response, err := http.Post(router.URL()+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(response.Body)
	response.Body.Close()
	if response.StatusCode != 200 || string(raw) != body {
		t.Fatalf("valid retry changed: %d %s", response.StatusCode, raw)
	}
	if calls != 1 {
		t.Fatal("retry did not reach backend exactly once", calls)
	}
}
