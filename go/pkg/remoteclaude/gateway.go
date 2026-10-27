// Package remoteclaude exposes the existing private Claude router through a
// narrow authenticated API. Placement and routing remain owned by ggrun.
package remoteclaude

import (
	"bytes"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/clientkit"
)

type Gateway struct {
	listener net.Listener
	server   *http.Server
}

func NewHandler(upstream, token string, profile clientkit.Profile) (http.Handler, error) {
	if len(token) < 32 || strings.ContainsAny(token, "\r\n\x00") {
		return nil, fmt.Errorf("gateway token must contain at least 32 characters without line breaks")
	}
	if err := profile.Validate(); err != nil {
		return nil, err
	}
	u, err := url.Parse(upstream)
	if err != nil || u.Host == "" || u.Scheme != "http" || u.User != nil {
		return nil, fmt.Errorf("invalid private Claude router URL")
	}
	proxy := httputil.NewSingleHostReverseProxy(u)
	proxy.FlushInterval = -1
	proxy.ModifyResponse = func(resp *http.Response) error {
		if resp.StatusCode >= 300 && resp.StatusCode < 400 {
			return fmt.Errorf("private backend returned a redirect")
		}
		return nil
	}
	director := proxy.Director
	proxy.Director = func(r *http.Request) { director(r); r.Header.Del("Authorization"); r.Header.Del("X-Api-Key") }
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		apiError(w, http.StatusBadGateway, "api_error", "model gateway is unavailable; check server status before retrying")
	}
	return Authenticate(token, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/ggrun/client":
			if r.Method != "GET" {
				apiError(w, 405, "invalid_request_error", "use GET")
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(profile)
		case "/v1/models":
			if r.Method != "GET" {
				apiError(w, 405, "invalid_request_error", "use GET")
				return
			}
			models := []map[string]any{{"id": profile.MainAlias, "type": "model", "display_name": profile.ModelName}}
			if profile.UtilityAlias != profile.MainAlias {
				models = append(models, map[string]any{"id": profile.UtilityAlias, "type": "model", "display_name": "ggrun utility"})
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"data": models, "has_more": false, "first_id": profile.MainAlias, "last_id": models[len(models)-1]["id"]})
		case "/v1/messages":
			if r.Method != "POST" {
				apiError(w, 405, "invalid_request_error", "use POST")
				return
			}
			if model := r.Header.Get("X-Ggrun-Model-ID"); model != "" && model != profile.ModelID {
				apiError(w, 400, "invalid_request_error", "server model/backend changed; exit Claude Code and start a new verified client session")
				return
			}
			if capacity := r.Header.Get("X-Ggrun-Context"); capacity != "" {
				n, err := strconv.Atoi(capacity)
				if err != nil || n < 1 || n > profile.ContextPerSlot {
					apiError(w, 400, "invalid_request_error", "server context changed; exit Claude Code and run ggrun-client doctor before starting a new session")
					return
				}
			}
			// Validate only the model field. Preserve unknown future fields and
			// the exact body bytes for the existing router and its transforms.
			const maxBody = 32 << 20
			raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBody))
			if err != nil {
				apiError(w, 400, "invalid_request_error", "invalid or oversized message body")
				return
			}
			var envelope struct {
				Model string `json:"model"`
			}
			if err := json.Unmarshal(raw, &envelope); err != nil || (envelope.Model != profile.MainAlias && envelope.Model != profile.UtilityAlias) {
				apiError(w, 400, "invalid_request_error", "model is not served by this gateway; use /v1/models")
				return
			}
			r.Body = io.NopCloser(bytes.NewReader(raw))
			r.ContentLength = int64(len(raw))
			proxy.ServeHTTP(w, r)
		case "/v1/messages/count_tokens":
			apiError(w, 501, "not_supported_error", "exact token counting is unavailable; use client estimation")
		case "/health":
			if r.Method != "GET" {
				apiError(w, 405, "invalid_request_error", "use GET")
				return
			}
			// This handler is installed only after verified model admission and
			// removed during recovery. Health must not enqueue a backend task.
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"status":"ok"}`)
		case "/ggrun/router":
			if r.Method != "GET" {
				apiError(w, 405, "invalid_request_error", "use GET")
				return
			}
			proxy.ServeHTTP(w, r)
		default:
			apiError(w, 404, "not_found_error", "endpoint is not exposed by the remote gateway")
		}
	})), nil
}

func Start(address, upstream, token string, profile clientkit.Profile) (*Gateway, error) {
	h, err := NewHandler(upstream, token, profile)
	if err != nil {
		return nil, err
	}
	ln, err := net.Listen("tcp", address)
	if err != nil {
		return nil, fmt.Errorf("listen for remote Claude gateway: %w", err)
	}
	s := &http.Server{Handler: h, ReadHeaderTimeout: 10 * time.Second}
	g := &Gateway{listener: ln, server: s}
	go func() { _ = s.Serve(ln) }()
	return g, nil
}
func (g *Gateway) Address() string { return g.listener.Addr().String() }
func (g *Gateway) Close() error    { return g.server.Close() }

func apiError(w http.ResponseWriter, status int, kind, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"type": "error", "error": map[string]string{"type": kind, "message": message}})
}

// Authenticate also protects loading/recovery responses before a model is ready.
func Authenticate(token string, next http.Handler) http.Handler {
	want := sha256.Sum256([]byte(token))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		auth := strings.TrimSpace(r.Header.Get("Authorization"))
		key := r.Header.Get("X-Api-Key")
		provided := ""
		if strings.HasPrefix(auth, "Bearer ") {
			provided = strings.TrimPrefix(auth, "Bearer ")
		}
		if provided == "" && auth == "" {
			provided = key
		}
		got := sha256.Sum256([]byte(provided))
		if token == "" || subtle.ConstantTimeCompare(got[:], want[:]) != 1 || (key != "" && key != provided) {
			apiError(w, 401, "authentication_error", "invalid gateway credential")
			return
		}
		next.ServeHTTP(w, r)
	})
}
