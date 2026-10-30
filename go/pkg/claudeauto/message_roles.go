package claudeauto

import (
	"encoding/json"
	"net/http"
)

// RejectUnsupportedSystemMessages declares the message bridge's verified role
// contract before a request reaches any selected backend/template. Older Jinja
// templates may accept late system entries but silently omit their content.
// Claude's documented capability retry moves that content into user messages.
// Keep both Unexpected role and input message role in the error: Claude
// 2.1.307 requires both fragments to recognize this rejection.
// This preserves agent handbacks without rewriting prompts or impersonating a
// Claude model ID. Top-level system instructions are supported and unchanged.
// Native mid-conversation system roles need an explicit backend+template
// capability proof before this bridge can opt in to them.
func RejectUnsupportedSystemMessages(w http.ResponseWriter, body []byte) bool {
	var envelope struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	if json.Unmarshal(body, &envelope) != nil {
		return false
	}
	for _, message := range envelope.Messages {
		if message.Role != "system" {
			continue
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{"type": "error", "error": map[string]string{"type": "invalid_request_error", "message": `Unexpected role "system": input message role system is not supported by this message bridge. Retry with its text in user messages; top-level system instructions remain supported.`}})
		return true
	}
	return false
}
