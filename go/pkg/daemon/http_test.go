package daemon

import (
	"encoding/json"
	"errors"
	"github.com/raketenkater/ggrun/pkg/server"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestReloadRejectsTrailingAndOversizedDocumentsBeforeMutation(t *testing.T) {
	for _, body := range []string{
		`{"model_path":"new.gguf"} {}`,
		`{"model_path":"new.gguf"} garbage`,
		`{"model_path":"new.gguf"}` + strings.Repeat(" ", maxControlBodyBytes),
		`{"model_path":"` + strings.Repeat("x", maxControlBodyBytes) + `"}`,
	} {
		d := New(Config{ModelPath: "old.gguf", ControlToken: "test-token", ComputeArgs: func(string, int) ([]string, error) { t.Fatal("invalid request reached planner"); return nil, nil }})
		d.startServer = func([]string, int, time.Duration) (*server.Process, error) {
			t.Fatal("invalid request started backend")
			return nil, nil
		}
		rr := httptest.NewRecorder()
		d.handleReload(rr, authedRequest(http.MethodPost, "/reload", strings.NewReader(body)))
		if rr.Code != http.StatusBadRequest || d.config.ModelPath != "old.gguf" {
			t.Fatalf("invalid request mutated state: %d %+v", rr.Code, d.config)
		}
		if !json.Valid(rr.Body.Bytes()) {
			t.Fatalf("invalid error JSON: %s", rr.Body.String())
		}
	}
}

func TestReloadErrorEscapesBackendMessage(t *testing.T) {
	message := "backend \"quoted\"\nC:\\model failed"
	d := New(Config{ControlToken: "test-token"})
	d.startServer = func([]string, int, time.Duration) (*server.Process, error) { return nil, errors.New(message) }
	rr := httptest.NewRecorder()
	d.handleReload(rr, authedRequest(http.MethodPost, "/reload", strings.NewReader("{} \n")))
	var response map[string]string
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if rr.Code != http.StatusInternalServerError || response["error"] != message || rr.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("bad error response: %d %s", rr.Code, rr.Body.String())
	}
}

// A slow authenticated request must not hold the lifecycle mutex while decoding.
func TestReloadBodyDoesNotBlockStatus(t *testing.T) {
	reading, release := make(chan struct{}), make(chan struct{})
	d := New(Config{ControlToken: "test-token"})
	d.startServer = func([]string, int, time.Duration) (*server.Process, error) { return &server.Process{}, nil }
	body := &delayedBody{reading: reading, release: release}
	request := httptest.NewRequest(http.MethodPost, "/reload", body)
	request.Header.Set("Authorization", "Bearer test-token")
	done := make(chan struct{})
	go func() { defer close(done); d.handleReload(httptest.NewRecorder(), request) }()
	defer func() { close(release); <-done }()
	select {
	case <-reading:
	case <-time.After(2 * time.Second):
		t.Fatal("reload never read body")
	}
	statusDone := make(chan struct{})
	go func() {
		defer close(statusDone)
		d.handleStatus(httptest.NewRecorder(), authedRequest(http.MethodGet, "/status", nil))
	}()
	select {
	case <-statusDone:
	case <-time.After(2 * time.Second):
		t.Fatal("body decoding held lifecycle lock")
	}
}

type delayedBody struct {
	reading, release chan struct{}
	read             bool
}

func (b *delayedBody) Read(p []byte) (int, error) {
	if b.read {
		return 0, io.EOF
	}
	b.read = true
	close(b.reading)
	<-b.release
	return copy(p, "{}"), nil
}
