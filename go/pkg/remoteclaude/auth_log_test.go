package remoteclaude

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// S-05: each refused request leaves one log line, never the credential.
func TestRefusedAuthenticationIsLoggedOncePerRequest(t *testing.T) {
	var log bytes.Buffer
	old := authFailureLog
	authFailureLog = &log
	t.Cleanup(func() { authFailureLog = old })
	h := Authenticate("right-token-0123456789abcdef0123", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	for _, bad := range []string{"Bearer wrong-guess-1", "Bearer wrong-guess-2"} {
		req := httptest.NewRequest("POST", "/v1/messages", nil)
		req.Header.Set("Authorization", bad)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != 401 {
			t.Fatalf("status %d", rec.Code)
		}
	}
	ok := httptest.NewRequest("GET", "/health", nil)
	ok.Header.Set("Authorization", "Bearer right-token-0123456789abcdef0123")
	h.ServeHTTP(httptest.NewRecorder(), ok)
	got := log.String()
	if strings.Count(got, "\n") != 2 || !strings.Contains(got, "refused POST /v1/messages from") {
		t.Fatalf("log = %q", got)
	}
	if strings.Contains(got, "wrong-guess") || strings.Contains(got, "right-token") {
		t.Fatalf("credential in log: %q", got)
	}
}
