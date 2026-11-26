package daemon

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
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
	d := New(Config{ModelPath: "first.gguf", Port: 8081, ControlToken: "test-token"})
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
			r := authedRequest("GET", "/status", nil)
			d.handleStatus(httptest.NewRecorder(), r)
		}()
		go func() {
			defer wg.Done()
			r := authedRequest("POST", "/reload", strings.NewReader("{\"model_path\":\"next.gguf\"}"))
			d.handleReload(httptest.NewRecorder(), r)
		}()
	}
	wg.Wait()
}

func TestControlAPIRequiresToken(t *testing.T) {
	d := New(Config{ControlToken: "test-token"})
	recorder := httptest.NewRecorder()
	d.handleStatus(recorder, httptest.NewRequest(http.MethodGet, "/status", nil))
	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated status = %d, want 401", recorder.Code)
	}

	recorder = httptest.NewRecorder()
	r := authedRequest(http.MethodGet, "/status", nil)
	r.Header.Del("X-GGRUN-Daemon-Token")
	r.Header.Set("Authorization", "Bearer test-token")
	d.handleStatus(recorder, r)
	if recorder.Code != http.StatusOK {
		t.Fatalf("bearer-auth status = %d, want 200", recorder.Code)
	}
}

func TestStatusRedactsRawServerArgs(t *testing.T) {
	d := New(Config{ModelPath: "m.gguf", ServerArgs: []string{"llama-server", "--secret"}, Port: 8081, ControlToken: "test-token"})
	recorder := httptest.NewRecorder()
	d.handleStatus(recorder, authedRequest(http.MethodGet, "/status", nil))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", recorder.Code)
	}
	var body map[string]interface{}
	if err := json.Unmarshal(recorder.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode status: %v", err)
	}
	cfg, ok := body["config"].(map[string]interface{})
	if !ok {
		t.Fatalf("missing config in status: %#v", body)
	}
	if _, ok := cfg["server_args"]; ok {
		t.Fatalf("status leaked server_args: %#v", cfg)
	}
}

func TestMutatingEndpointsRequirePost(t *testing.T) {
	d := New(Config{ControlToken: "test-token"})
	for _, handler := range []func(http.ResponseWriter, *http.Request){d.handleStart, d.handleStop, d.handleReload} {
		recorder := httptest.NewRecorder()
		handler(recorder, authedRequest(http.MethodGet, "/", nil))
		if recorder.Code != http.StatusMethodNotAllowed {
			t.Fatalf("GET mutation returned %d, want 405", recorder.Code)
		}
	}
}

func TestCloseWithoutStartedProcess(t *testing.T) {
	d := New(Config{ControlToken: "test-token"})
	if err := d.Close(); err != nil {
		t.Fatalf("Close() = %v, want nil", err)
	}
}

func TestReloadComputeFailureDoesNotPartiallyChangeConfig(t *testing.T) {
	d := New(Config{
		ModelPath:    "old.gguf",
		Port:         8081,
		ControlToken: "test-token",
		ComputeArgs: func(string, int) ([]string, error) {
			return nil, errors.New("unsupported model")
		},
	})
	recorder := httptest.NewRecorder()
	r := authedRequest(http.MethodPost, "/reload", strings.NewReader(`{"model_path":"new.gguf","port":8082}`))
	d.handleReload(recorder, r)
	if recorder.Code != http.StatusInternalServerError {
		t.Fatalf("reload status = %d, want 500", recorder.Code)
	}
	if d.config.ModelPath != "old.gguf" || d.config.Port != 8081 {
		t.Fatalf("failed reload partially changed config: %#v", d.config)
	}
}

func TestReloadStartsWhenIdle(t *testing.T) {
	d := New(Config{
		ModelPath:               "old.gguf",
		ServerArgs:              []string{"old-server"},
		Port:                    8081,
		ControlToken:            "test-token",
		AllowExplicitServerArgs: true,
	})
	var gotArgs []string
	var gotPort int
	d.startServer = func(args []string, port int, _ time.Duration) (*server.Process, error) {
		gotArgs = append([]string(nil), args...)
		gotPort = port
		return &server.Process{}, nil
	}
	recorder := httptest.NewRecorder()
	r := authedRequest(http.MethodPost, "/reload", strings.NewReader(`{"model_path":"new.gguf","server_args":["new-server"],"port":8082}`))
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

func TestReloadRecomputesMemoryPolicyForNewArgv(t *testing.T) {
	d := New(Config{
		ModelPath:    "old.gguf",
		ServerArgs:   []string{"old-server", "--no-mmap"},
		Port:         8081,
		ControlToken: "test-token",
		ComputeArgs: func(string, int) ([]string, error) {
			return []string{"new-server", "--n-cpu-moe", "40"}, nil
		},
		ComputeMemory: func(args []string) (int, int, error) {
			if len(args) == 0 || args[0] != "new-server" {
				t.Fatalf("memory policy saw stale argv: %v", args)
			}
			return 90000, 114000, nil
		},
	})
	d.startServer = func([]string, int, time.Duration) (*server.Process, error) {
		return &server.Process{}, nil
	}
	recorder := httptest.NewRecorder()
	d.handleReload(recorder, authedRequest(http.MethodPost, "/reload", strings.NewReader(`{"model_path":"new.gguf"}`)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("reload status = %d: %s", recorder.Code, recorder.Body.String())
	}
	if d.config.MemoryHighMB != 90000 || d.config.MemoryMaxMB != 114000 {
		t.Fatalf("reload retained old-model memory policy: %#v", d.config)
	}
}

func TestReloadRejectsExplicitServerArgsByDefault(t *testing.T) {
	d := New(Config{ModelPath: "old.gguf", ServerArgs: []string{"old-server"}, Port: 8081, ControlToken: "test-token"})
	d.startServer = func([]string, int, time.Duration) (*server.Process, error) {
		t.Fatal("explicit server_args must be rejected before start")
		return nil, nil
	}
	recorder := httptest.NewRecorder()
	r := authedRequest(http.MethodPost, "/reload", strings.NewReader(`{"server_args":["/tmp/evil"]}`))
	d.handleReload(recorder, r)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("explicit server_args reload status = %d, want 400", recorder.Code)
	}
}

func TestReloadCapsRequestBody(t *testing.T) {
	d := New(Config{ControlToken: "test-token"})
	recorder := httptest.NewRecorder()
	r := authedRequest(http.MethodPost, "/reload", strings.NewReader(strings.Repeat("x", maxControlBodyBytes+1)))
	d.handleReload(recorder, r)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("oversized reload status = %d, want 400", recorder.Code)
	}
}

func authedRequest(method, target string, body *strings.Reader) *http.Request {
	var r *http.Request
	if body == nil {
		r = httptest.NewRequest(method, target, nil)
	} else {
		r = httptest.NewRequest(method, target, body)
	}
	r.Header.Set("X-GGRUN-Daemon-Token", "test-token")
	return r
}

func TestDaemonCarriesMmapReclaimBand(t *testing.T) {
	// Config with a band must retain it through the scope mapping: an explicit
	// MemoryHighMB below MemoryMaxMB is the mmap reclaim band.
	high, max := (Config{MemoryMaxMB: 114000, MemoryHighMB: 100000}).memoryHighMaxMB(nil)
	if high != 100000 || max != 114000 {
		t.Fatalf("daemon must retain the band: high=%d max=%d", high, max)
	}

	// Without an explicit band, the daemon derives the soft threshold from the
	// max (the pre-existing single-cap behavior) — never loosens containment.
	high, max = (Config{MemoryMaxMB: 114000}).memoryHighMaxMB(nil)
	if high != 114000 || max != 114000 {
		t.Fatalf("default daemon scope must be a single hard cap: high=%d max=%d", high, max)
	}

	// An mmap-backed plan must carry a reclaim band (high < max) so the kernel
	// can evict clean page cache before OOM-killing. This is the assertion that
	// fails on the pre-fix code (no MemoryHighMB field at all).
	high, max = (Config{MemoryMaxMB: 114000, MemoryHighMB: 100000}).memoryHighMaxMB([]string{"llama-server", "--n-cpu-moe", "48"})
	if !(high > 0 && high < max) {
		t.Fatalf("mmap plan must carry a reclaim band, got high=%d max=%d", high, max)
	}
}

// S-06: the control token goes to an owner-only file, not the terminal.
func TestControlTokenIsWrittenOwnerOnly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "daemon.token")
	if err := os.WriteFile(filepath.Join(filepath.Dir(filepath.Dir(path)), "unrelated"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	d := New(Config{ControlToken: "secret-token", TokenFile: path})
	got, err := d.writeControlToken("secret-token")
	if err != nil || got != path {
		t.Fatalf("write: %v %s", err, got)
	}
	b, err := os.ReadFile(path)
	if err != nil || strings.TrimSpace(string(b)) != "secret-token" {
		t.Fatalf("token file %q %v", b, err)
	}
	if runtime.GOOS != "windows" {
		if st, _ := os.Stat(path); st.Mode().Perm() != 0o600 {
			t.Fatalf("token file mode %v", st.Mode().Perm())
		}
		// A pre-existing world-readable file is replaced, not reused.
		if err := os.Chmod(path, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := d.writeControlToken("next"); err != nil {
			t.Fatal(err)
		}
		if st, _ := os.Stat(path); st.Mode().Perm() != 0o600 {
			t.Fatalf("rewritten token file mode %v", st.Mode().Perm())
		}
	}
}

func TestDefaultTokenPathIsPerControlPort(t *testing.T) {
	d := New(Config{ControlPort: 9191})
	p, err := d.controlTokenPath()
	if err != nil {
		t.Skip(err)
	}
	if filepath.Base(p) != "daemon-9191.token" || filepath.Base(filepath.Dir(p)) != "ggrun" {
		t.Fatalf("path %s", p)
	}
}
