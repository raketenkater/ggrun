package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/claudeauto"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/remoteclaude"
)

type remoteStatusWriterFunc func([]byte) (int, error)

func (f remoteStatusWriterFunc) Write(p []byte) (int, error) { return f(p) }

func TestRemoteServingStatusChangesAndStops(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ticks := make(chan time.Time, 2)
	lines := make(chan string, 8)
	sampled := make(chan struct{}, 8)
	var state atomic.Value
	state.Store(claudeauto.RouterAdmissionStatus{Limit: 2})
	out := remoteStatusWriterFunc(func(p []byte) (int, error) {
		lines <- string(p)
		return len(p), nil
	})
	done := make(chan struct{})
	go func() {
		defer close(done)
		runRemoteClaudeStatus(ctx, ticks, out, func() claudeauto.RouterAdmissionStatus {
			value := state.Load().(claudeauto.RouterAdmissionStatus)
			sampled <- struct{}{}
			return value
		})
	}()
	readLine := func() string {
		t.Helper()
		select {
		case line := <-lines:
			return line
		case <-time.After(time.Second):
			t.Fatal("serving display did not update")
			return ""
		}
	}
	readSample := func() {
		t.Helper()
		select {
		case <-sampled:
		case <-time.After(time.Second):
			t.Fatal("serving display stopped sampling")
		}
	}
	if line := readLine(); !strings.Contains(line, "active 0 | queued 0") || !strings.Contains(line, "configured main limit 2") || !strings.Contains(line, "Ctrl+C") {
		t.Fatal(line)
	}
	readSample()
	ticks <- time.Now()
	readSample() // An unchanged idle sample must not fill logs with duplicate lines.
	state.Store(claudeauto.RouterAdmissionStatus{Active: 1, Queued: 3, Limit: 2})
	ticks <- time.Now()
	if line := readLine(); !strings.Contains(line, "active 1 | queued 3") {
		t.Fatal(line)
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("display did not release its observer after cancellation")
	}
	if len(lines) != 0 {
		t.Fatal("unchanged samples produced duplicate output")
	}
}

func TestRemoteServingStatusStopsOnClosedOutput(t *testing.T) {
	done := make(chan struct{})
	go func() {
		defer close(done)
		runRemoteClaudeStatus(context.Background(), nil,
			remoteStatusWriterFunc(func([]byte) (int, error) { return 0, io.ErrClosedPipe }),
			func() claudeauto.RouterAdmissionStatus { return claudeauto.RouterAdmissionStatus{} })
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("closed output left a status observer running")
	}
}

func TestRemoteStoppingRevokesReadinessAndCancelsStatus(t *testing.T) {
	stopped, upstream := 0, 0
	key := strings.Repeat("test", 8)
	s := &remoteClaudeService{stopStatus: func() { stopped++ }}
	s.handler.Store(handlerBox{http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstream++
		w.WriteHeader(http.StatusOK)
	})})
	handler := remoteclaude.Authenticate(key, s)
	request := func(token string) *httptest.ResponseRecorder {
		r := httptest.NewRequest("GET", "/health", nil)
		if token != "" {
			r.Header.Set("Authorization", "Bearer "+token)
		}
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, r)
		return w
	}
	if got := request(key).Code; got != 200 {
		t.Fatal(got)
	}
	s.stopping()
	s.stopping()
	if stopped != 1 {
		t.Fatalf("status observer cleanup called %d times", stopped)
	}
	if got := request("").Code; got != 401 {
		t.Fatalf("shutdown readiness lost authentication: %d", got)
	}
	w := request(key)
	if w.Code != 503 || w.Header().Get("Retry-After") != "5" || !strings.Contains(w.Body.String(), "stopping") {
		t.Fatalf("shutdown falsely ready: %d %s", w.Code, w.Body.String())
	}
	if upstream != 1 {
		t.Fatal("shutdown status reached the old backend handler")
	}
}

// Exercise the actual activation wiring. This is gateway lifecycle coverage;
// the fake backend does not establish model placement or hardware performance.
func TestRemoteActivationOwnsItsServingDisplay(t *testing.T) {
	var inference atomic.Int64
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		inference.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer backend.Close()
	router, err := claudeauto.StartRouter(backend.URL, backend.URL, false, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()
	req := &launchRequest{
		ModelPath:    filepath.Join(t.TempDir(), "fixture.gguf"),
		ClaudeCode:   true,
		RemoteClaude: remoteClaudeOptions{Headless: true, Host: "127.0.0.1", TokenFile: filepath.Join(t.TempDir(), "key")},
	}
	s, err := prepareRemoteClaude(req, &config.Config{})
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	strategy := &placement.Strategy{ContextSize: 65536, Parallel: 1}
	if err := s.activate(req, &placement.ModelProfile{Name: "fixture"}, strategy, &backendInfo{Tag: "llama"},
		&claudeAutoRuntime{router: router}, []string{"--ctx-size", "65536", "--parallel", "1"}); err != nil {
		t.Fatal(err)
	}
	if s.stopStatus == nil {
		t.Fatal("ready service did not start its serving display")
	}
	r := httptest.NewRequest("GET", "/ggrun/client", nil)
	r.Header.Set("Authorization", "Bearer "+s.key)
	w := httptest.NewRecorder()
	s.server.Handler.ServeHTTP(w, r)
	if w.Code != 200 || !strings.Contains(w.Body.String(), "fixture") {
		t.Fatalf("activation did not publish the verified profile: %d %s", w.Code, w.Body.String())
	}
	s.stopping()
	if s.stopStatus != nil {
		t.Fatal("shutdown retained the serving observer")
	}
	w = httptest.NewRecorder()
	s.server.Handler.ServeHTTP(w, r)
	if w.Code != 503 {
		t.Fatalf("stopping service advertises ready: %d", w.Code)
	}
	if inference.Load() != 0 {
		t.Fatalf("status/metadata consumed %d inference calls", inference.Load())
	}
}
