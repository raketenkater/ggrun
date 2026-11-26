package update

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestCheckReleaseFallsBackOnAPILimit(t *testing.T) {
	for _, status := range []int{http.StatusForbidden, http.StatusTooManyRequests} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			const tagPath = "/raketenkater/ggrun/releases/tag/v3.2.13"
			var probes atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api":
					w.WriteHeader(status)
				case "/raketenkater/ggrun/releases/latest":
					probes.Add(1)
					if r.Method != http.MethodHead {
						t.Errorf("latest page requested with %s", r.Method)
					}
					w.Header().Set("Location", tagPath)
					w.WriteHeader(http.StatusFound)
				default:
					t.Errorf("unexpected request to %s", r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer server.Close()
			got, err := checkRelease(context.Background(), server.Client(), server.URL+"/api",
				server.URL+"/raketenkater/ggrun/releases/latest")
			if err != nil || got.Latest != "v3.2.13" || got.URL != server.URL+tagPath || probes.Load() != 1 {
				t.Fatalf("result=%+v, err=%v, latest page requests=%d", got, err, probes.Load())
			}
		})
	}
}

func TestCheckReleaseUsesAPINormally(t *testing.T) {
	for _, status := range []int{http.StatusOK, http.StatusNotFound} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/api" {
					t.Errorf("unexpected request to %s", r.URL.Path)
				}
				w.WriteHeader(status)
				fmt.Fprint(w, `{"tag_name":"v3.2.13","html_url":"https://github.com/raketenkater/ggrun/releases/tag/v3.2.13"}`)
			}))
			defer server.Close()
			got, err := checkRelease(context.Background(), server.Client(), server.URL+"/api",
				server.URL+"/raketenkater/ggrun/releases/latest")
			if status == http.StatusOK {
				if err != nil || got.Latest != "v3.2.13" || !strings.HasSuffix(got.URL, "/v3.2.13") {
					t.Fatalf("result=%+v, err=%v", got, err)
				}
			} else if err == nil || !strings.Contains(err.Error(), "404") || got != nil {
				t.Fatalf("result=%+v, err=%v", got, err)
			}
		})
	}
}

func TestCheckReleaseRejectsInvalidRedirect(t *testing.T) {
	for _, location := range []string{
		"https://elsewhere.invalid/raketenkater/ggrun/releases/tag/v3.2.13",
		"/another/ggrun/releases/tag/v3.2.13",
		"/raketenkater/ggrun/releases/tag/main",
		"/raketenkater/ggrun/releases/tag/v3.2.13%2fmore",
		"/raketenkater/ggrun/releases/tag/v3.2.13?extra=1",
		"",
	} {
		t.Run(location, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/api" {
					w.WriteHeader(http.StatusForbidden)
					return
				}
				if r.URL.Path != "/raketenkater/ggrun/releases/latest" {
					t.Errorf("followed unexpected destination %s", r.URL.Path)
				}
				w.Header().Set("Location", location)
				w.WriteHeader(http.StatusFound)
			}))
			defer server.Close()
			got, err := checkRelease(context.Background(), server.Client(), server.URL+"/api",
				server.URL+"/raketenkater/ggrun/releases/latest")
			if got != nil || err == nil || !strings.Contains(err.Error(), "403") {
				t.Fatalf("result=%+v, err=%v", got, err)
			}
		})
	}
}
