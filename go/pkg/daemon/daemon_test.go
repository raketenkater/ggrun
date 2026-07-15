package daemon

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/server"
)

func TestControlAPIIsLoopbackOnly(t *testing.T) {
	d := New(Config{ControlPort: 19090})
	if d.addr != "127.0.0.1:19090" {
		t.Fatalf("control address = %q, want loopback only", d.addr)
	}
}

func TestConcurrentStatusAndReload(t *testing.T) {
	d := New(Config{ModelPath: "first.gguf", Port: 8081})
	// This test exercises the daemon lock only; reload now starts even while
	// idle, so avoid spawning a real backend for each concurrent request.
	d.startServer = func([]string, int, time.Duration) (*server.Process, error) {
		return &server.Process{}, nil
	}
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			r := httptest.NewRequest("GET", "/status", nil)
			d.handleStatus(httptest.NewRecorder(), r)
		}()
		go func() {
			defer wg.Done()
			r := httptest.NewRequest("POST", "/reload", strings.NewReader("{\"model_path\":\"next.gguf\"}"))
			d.handleReload(httptest.NewRecorder(), r)
		}()
	}
	wg.Wait()
}

func TestMutatingEndpointsRequirePost(t *testing.T) {
	d := New(Config{})
	for _, handler := range []func(http.ResponseWriter, *http.Request){d.handleStart, d.handleStop, d.handleReload} {
		recorder := httptest.NewRecorder()
		handler(recorder, httptest.NewRequest(http.MethodGet, "/", nil))
		if recorder.Code != http.StatusMethodNotAllowed {
			t.Fatalf("GET mutation returned %d, want 405", recorder.Code)
		}
	}
}

func TestCloseWithoutStartedProcess(t *testing.T) {
	d := New(Config{})
	if err := d.Close(); err != nil {
		t.Fatalf("Close() = %v, want nil", err)
	}
}

func TestReloadComputeFailureDoesNotPartiallyChangeConfig(t *testing.T) {
	d := New(Config{
		ModelPath: "old.gguf",
		Port:      8081,
		ComputeArgs: func(string, int) ([]string, error) {
			return nil, errors.New("unsupported model")
		},
	})
	recorder := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/reload", strings.NewReader(`{"model_path":"new.gguf","port":8082}`))
	d.handleReload(recorder, r)
	if recorder.Code != http.StatusInternalServerError {
		t.Fatalf("reload status = %d, want 500", recorder.Code)
	}
	if d.config.ModelPath != "old.gguf" || d.config.Port != 8081 {
		t.Fatalf("failed reload partially changed config: %#v", d.config)
	}
}

func TestReloadStartsWhenIdle(t *testing.T) {
	d := New(Config{ModelPath: "old.gguf", ServerArgs: []string{"old-server"}, Port: 8081})
	var gotArgs []string
	var gotPort int
	d.startServer = func(args []string, port int, _ time.Duration) (*server.Process, error) {
		gotArgs = append([]string(nil), args...)
		gotPort = port
		return &server.Process{}, nil
	}
	recorder := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/reload", strings.NewReader(`{"model_path":"new.gguf","server_args":["new-server"],"port":8082}`))
	d.handleReload(recorder, r)
	if recorder.Code != http.StatusOK {
		t.Fatalf("idle reload status = %d, want 200: %s", recorder.Code, recorder.Body.String())
	}
	if gotPort != 8082 || len(gotArgs) != 1 || gotArgs[0] != "new-server" {
		t.Fatalf("idle reload start = %v on %d, want new-server on 8082", gotArgs, gotPort)
	}
	if d.process == nil || d.config.ModelPath != "new.gguf" || d.config.Port != 8082 {
		t.Fatalf("idle reload did not apply and start new config: %#v", d.config)
	}
}
