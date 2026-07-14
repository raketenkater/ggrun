package daemon

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func TestControlAPIIsLoopbackOnly(t *testing.T) {
	d := New(Config{ControlPort: 19090})
	if d.addr != "127.0.0.1:19090" {
		t.Fatalf("control address = %q, want loopback only", d.addr)
	}
}

func TestConcurrentStatusAndReload(t *testing.T) {
	d := New(Config{ModelPath: "first.gguf", Port: 8081})
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
