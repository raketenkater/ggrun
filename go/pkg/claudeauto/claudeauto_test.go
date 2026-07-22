package claudeauto

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestIsClassifierRequestIsNarrow(t *testing.T) {
	if !IsClassifierRequest([]byte(`{"system":[{"text":"` + ClassifierMarker + `"}]}`)) {
		t.Fatal("exact Auto classifier marker was not detected")
	}
	if IsClassifierRequest([]byte(`{"messages":[{"text":"please review security"}]}`)) {
		t.Fatal("ordinary security request must stay on the main model")
	}
	if IsClassifierRequest([]byte(`{"messages":[{"text":"` + ClassifierMarker + `"}],"system":[{"text":"normal"}]}`)) {
		t.Fatal("a user message containing the marker must stay on the main model")
	}
}

func TestRouterSeparatesClassifierAndMainTraffic(t *testing.T) {
	backend := func(name string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			w.Header().Set("X-Backend", name)
			fmt.Fprintf(w, "%s:%s", name, body)
		}))
	}
	main := backend("main")
	defer main.Close()
	reviewer := backend("reviewer")
	defer reviewer.Close()

	router, err := StartRouter(main.URL, reviewer.URL, false, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()

	for _, tc := range []struct {
		path, body, want string
	}{
		{"/v1/messages", `{"messages":[{"text":"code"}]}`, "main"},
		{"/v1/messages", `{"system":[{"text":"` + ClassifierMarker + `"}]}`, "reviewer"},
		{"/health", "", "main"},
	} {
		resp, err := http.Post(router.URL()+tc.path, "application/json", strings.NewReader(tc.body))
		if err != nil {
			t.Fatal(err)
		}
		got, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.Header.Get("X-Backend") != tc.want || !strings.HasPrefix(string(got), tc.want+":") {
			t.Fatalf("%s routed to %q body=%q, want %q", tc.path, resp.Header.Get("X-Backend"), got, tc.want)
		}
	}
}

func TestRouterSanitizesImagesForTextOnlyModel(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_, _ = w.Write(body)
	}))
	defer backend.Close()

	router, err := StartRouter(backend.URL, backend.URL, false, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()

	body := `{"messages":[{"role":"user","content":[{"type":"text","text":"keep me"},{"type":"image","source":{"type":"base64","data":"DIRECT_SECRET"}},{"type":"tool_result","tool_use_id":"read-1","content":[{"type":"image","source":{"type":"base64","data":"TOOL_SECRET"}},{"type":"text","text":"also keep me"}]}]}],"tools":[{"input_schema":{"type":"image"}}]}`
	resp, err := http.Post(router.URL()+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	got, _ := io.ReadAll(resp.Body)
	resp.Body.Close()

	text := string(got)
	if strings.Contains(text, "DIRECT_SECRET") || strings.Contains(text, "TOOL_SECRET") {
		t.Fatalf("text-only route forwarded image data: %s", text)
	}
	if strings.Count(text, textOnlyImagePlaceholder) != 2 {
		t.Fatalf("got %d image notices, want 2: %s", strings.Count(text, textOnlyImagePlaceholder), text)
	}
	for _, want := range []string{"keep me", "also keep me", `"input_schema":{"type":"image"}`} {
		if !strings.Contains(text, want) {
			t.Fatalf("sanitizer removed unrelated content %q: %s", want, text)
		}
	}
}

func TestRouterPreservesImagesForVisionModel(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_, _ = w.Write(body)
	}))
	defer backend.Close()

	router, err := StartRouter(backend.URL, backend.URL, true, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()

	body := `{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","data":"VISION_DATA"}}]}]}`
	resp, err := http.Post(router.URL()+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	got, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if string(got) != body {
		t.Fatalf("vision route changed request body: got %s, want %s", got, body)
	}
}

func TestRouterLimitsConcurrentMainRequests(t *testing.T) {
	release := make(chan struct{})
	started := make(chan struct{}, 2)
	var active atomic.Int64
	var peak atomic.Int64
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		current := active.Add(1)
		defer active.Add(-1)
		if current > peak.Load() {
			peak.Store(current)
		}
		started <- struct{}{}
		<-release
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer backend.Close()
	router, err := StartRouter(backend.URL, backend.URL, false, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()

	done := make(chan error, 2)
	request := func() {
		resp, err := http.Post(router.URL()+"/v1/messages", "application/json", strings.NewReader(`{"messages":[]}`))
		if err == nil {
			resp.Body.Close()
		}
		done <- err
	}
	go request()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("first request did not reach backend")
	}
	go request()
	time.Sleep(50 * time.Millisecond)
	resp, err := http.Get(router.URL() + "/ggrun/router")
	if err != nil {
		t.Fatal(err)
	}
	var status struct{ Active, Queued, Limit int }
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if status.Active != 1 || status.Queued != 1 || status.Limit != 1 {
		t.Fatalf("unexpected router status: %+v", status)
	}
	close(release)
	for range 2 {
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	}
	if peak.Load() != 1 {
		t.Fatalf("peak backend concurrency=%d, want 1", peak.Load())
	}
}

func TestDownloadModelVerifiesArtifact(t *testing.T) {
	payload := append([]byte("GGUF"), []byte(" reviewer")...)
	sum := sha256.Sum256(payload)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(payload)
	}))
	defer srv.Close()
	spec := ModelSpec{URL: srv.URL, Name: "test.gguf", Size: int64(len(payload)), SHA256: hex.EncodeToString(sum[:])}
	dest := filepath.Join(t.TempDir(), spec.Name)
	if err := downloadModel(context.Background(), srv.Client(), spec, dest, io.Discard); err != nil {
		t.Fatal(err)
	}
	if err := validateGGUF(dest, spec.Size); err != nil {
		t.Fatal(err)
	}
	if err := validatePinnedGGUF(dest, spec); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(dest)
	if string(data) != string(payload) {
		t.Fatalf("downloaded data mismatch: %q", data)
	}
}

func TestDownloadModelRejectsChecksumMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("GGUF bad"))
	}))
	defer srv.Close()
	dest := filepath.Join(t.TempDir(), "bad.gguf")
	spec := ModelSpec{URL: srv.URL, Size: 8, SHA256: strings.Repeat("0", 64)}
	if err := downloadModel(context.Background(), srv.Client(), spec, dest, io.Discard); err == nil {
		t.Fatal("expected checksum failure")
	}
	if _, err := os.Stat(dest); !os.IsNotExist(err) {
		t.Fatalf("bad artifact should not be installed: %v", err)
	}
}
