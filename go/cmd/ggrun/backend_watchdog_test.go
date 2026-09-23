package main

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeBackend stands in for a launched backend. A step blocked on it returns
// only when it is stopped, like a canary request to a wedged server.
type fakeBackend struct {
	mu      sync.Mutex
	log     string
	running bool
	ticks   uint64
	burning bool // CPU time advances on every read
	stopped chan struct{}
	once    sync.Once
}

func newFakeBackend() *fakeBackend {
	return &fakeBackend{running: true, stopped: make(chan struct{})}
}

func (f *fakeBackend) IsRunning() bool { f.mu.Lock(); defer f.mu.Unlock(); return f.running }
func (f *fakeBackend) Log() string     { f.mu.Lock(); defer f.mu.Unlock(); return f.log }
func (f *fakeBackend) Stop() error {
	f.once.Do(func() { f.mu.Lock(); f.running = false; f.mu.Unlock(); close(f.stopped) })
	return nil
}
func (f *fakeBackend) CPUTicks() (uint64, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.burning {
		f.ticks++
	}
	return f.ticks, true
}
func (f *fakeBackend) appendLog(s string) { f.mu.Lock(); f.log += s; f.mu.Unlock() }

var fastWatch = backendWatch{
	interval: 10 * time.Millisecond, probeTimeout: 50 * time.Millisecond,
	wedgedAfter: 200 * time.Millisecond, stopGrace: time.Second,
}

// A health endpoint that never answers, like a backend stuck in its abort handler.
func silentHealth(t *testing.T) string {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(func() { close(release); srv.Close() })
	return srv.URL + "/health"
}

func blockedUntilStopped(b *fakeBackend) func() error {
	return func() error {
		<-b.stopped
		return errors.New("connection reset by peer")
	}
}

func runBounded(t *testing.T, b watchedBackend, url string, w backendWatch, step func() error) error {
	t.Helper()
	result := make(chan error, 1)
	go func() { result <- runWatchingBackend(b, url, w, step) }()
	select {
	case err := <-result:
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("the watchdog did not end the step: a dead backend would block the launch")
		return nil
	}
}

// The observed failure: a CUDA abort printed, the process stayed alive and
// silent, and the launch waited out the canary's 20-minute timeout.
func TestWatchdogStopsABackendThatPrintedAFatalError(t *testing.T) {
	b := newFakeBackend()
	b.appendLog("loading...\n")
	url := silentHealth(t)
	go func() {
		time.Sleep(30 * time.Millisecond)
		b.appendLog("CUDA error: an unsupported value or parameter was passed to the function\n")
	}()
	err := runBounded(t, b, url, backendWatch{interval: 10 * time.Millisecond, probeTimeout: 50 * time.Millisecond,
		wedgedAfter: time.Hour, stopGrace: time.Second}, blockedUntilStopped(b))
	if err == nil || !strings.Contains(err.Error(), "CUDA error") {
		t.Fatalf("want a fatal-error stop naming the CUDA error, got %v", err)
	}
	if b.IsRunning() {
		t.Fatal("the backend was left running after a fatal error")
	}
}

// A fatal line printed before the step started still means this process is
// dead: in the real MiniMax run the abort happened during calibration and the
// canary started afterwards against the wedged process.
func TestWatchdogStopsOnAFatalLinePrintedBeforeTheStep(t *testing.T) {
	b := newFakeBackend()
	b.appendLog("CUDA error: an unsupported value or parameter was passed to the function\n")
	err := runBounded(t, b, silentHealth(t), backendWatch{interval: 10 * time.Millisecond, probeTimeout: 50 * time.Millisecond,
		wedgedAfter: time.Hour, stopGrace: time.Second}, blockedUntilStopped(b))
	if err == nil || !strings.Contains(err.Error(), "CUDA error") {
		t.Fatalf("an earlier abort in the same process was ignored: %v", err)
	}
}

func TestWatchdogStopsWhenTheBackendExits(t *testing.T) {
	b := newFakeBackend()
	url := silentHealth(t)
	go func() { time.Sleep(30 * time.Millisecond); b.mu.Lock(); b.running = false; b.mu.Unlock() }()
	err := runBounded(t, b, url, backendWatch{interval: 10 * time.Millisecond, probeTimeout: 50 * time.Millisecond,
		wedgedAfter: time.Hour, stopGrace: time.Second}, blockedUntilStopped(b))
	if err == nil || !strings.Contains(err.Error(), "exited") {
		t.Fatalf("want an exited-backend stop, got %v", err)
	}
}

// Silent /health and no CPU progress: wedged, with no log line to go on.
func TestWatchdogStopsAWedgedBackend(t *testing.T) {
	b := newFakeBackend()
	err := runBounded(t, b, silentHealth(t), fastWatch, blockedUntilStopped(b))
	if err == nil || !strings.Contains(err.Error(), "no CPU") {
		t.Fatalf("want a wedged-backend stop, got %v", err)
	}
}

// The case that must NOT trigger: ik's /health waits on the main task queue,
// so a long llama_decode silences it while the backend is working hard.
func TestWatchdogLeavesASlowButWorkingBackendAlone(t *testing.T) {
	b := newFakeBackend()
	b.burning = true
	err := runBounded(t, b, silentHealth(t), fastWatch, func() error {
		time.Sleep(600 * time.Millisecond) // three times wedgedAfter
		return nil
	})
	if err != nil || !b.IsRunning() {
		t.Fatalf("a busy backend was stopped: err=%v running=%v", err, b.IsRunning())
	}
}

func TestFatalBackendLineRecognizesGGMLAborts(t *testing.T) {
	// First lines of real aborts, as ggml prints them.
	for _, line := range []string{
		"CUDA error: an unsupported value or parameter was passed to the function",
		"/src/ggml/src/ggml.c:1234: GGML_ASSERT(ne00 == ne10) failed",
		"terminate called after throwing an instance of 'std::runtime_error'",
	} {
		if fatalBackendLine("ok\n"+line+"\n") == "" {
			t.Errorf("fatal line not recognized: %q", line)
		}
	}
	if got := fatalBackendLine("prompt processing progress\nslot released\n"); got != "" {
		t.Errorf("ordinary log flagged as fatal: %q", got)
	}
}

func TestProcessTreeCPUTicksReadsThisProcess(t *testing.T) {
	if runtime.GOOS == "linux" {
		if _, ok := processTreeCPUTicks(os.Getpid()); !ok {
			t.Error("this test process's CPU time could not be read")
		}
	}
	if _, ok := processTreeCPUTicks(1 << 30); ok {
		t.Error("a pid that does not exist must report unreadable")
	}
}
