package claudeauto

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
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

	router, err := StartRouter(main.URL, reviewer.URL)
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
