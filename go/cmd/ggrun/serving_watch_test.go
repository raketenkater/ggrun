package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"
)

func (f *fakeBackend) Kill() { _ = f.Stop() }
func (f *fakeBackend) LogSince(offset int) (string, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if offset < 0 || offset > len(f.log) {
		offset = 0
	}
	return f.log[offset:], len(f.log)
}

var fastServingWatch = servingWatch{interval: 10 * time.Millisecond, probeTimeout: 50 * time.Millisecond, confirmAfter: 150 * time.Millisecond}

func answeringHealth(t *testing.T) string {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) }))
	t.Cleanup(srv.Close)
	return srv.URL + "/health"
}

func waitServing(t *testing.T, b servingBackend, sig chan os.Signal, url string, within time.Duration) (crashed, returned bool) {
	t.Helper()
	done := make(chan bool, 1)
	go func() { done <- waitForShutdownCrashOrWedge(b, sig, url, fastServingWatch) }()
	select {
	case c := <-done:
		return c, true
	case <-time.After(within):
		return false, false
	}
}

const ikCublasAbort = "CUDA error: an unsupported value or parameter was passed to the function\n" +
	"  current device: 0, in function ggml_cuda_op_mul_mat_cublas at /src/ggml/src/ggml-cuda.cu:1878\n" +
	"/src/ggml/src/ggml-cuda.cu:139: CUDA error\n"

// The B2 acceptance cell: ik aborted on a cuBLAS error, its backtrace helper
// deadlocked, and the process lived on answering nothing. Serving must end.
func TestServingWatchStopsABackendHungAfterAFatalError(t *testing.T) {
	b := newFakeBackend()
	b.appendLog("srv  update_slots: all slots are idle\n")
	sig := make(chan os.Signal, 1)
	url := silentHealth(t)
	go func() { time.Sleep(50 * time.Millisecond); b.appendLog(ikCublasAbort) }()
	crashed, returned := waitServing(t, b, sig, url, 5*time.Second)
	if !returned || !crashed {
		t.Fatalf("hung backend after a fatal error must end serving as a crash (returned=%v crashed=%v)", returned, crashed)
	}
	if b.IsRunning() {
		t.Fatal("the hung backend must be killed")
	}
}

// Mainline prefixes its own log lines; ggml's abort line carries a source location.
func TestServingFatalLineFormats(t *testing.T) {
	for _, line := range []string{
		"CUDA error: out of memory",
		"0.12.345.678 E CUDA error: an illegal memory access was encountered",
		"/src/ggml/src/ggml-cuda.cu:139: CUDA error",
		"/src/ggml/src/ggml.c:5432: GGML_ASSERT(ne0 > 0) failed",
		"terminate called after throwing an instance of 'std::runtime_error'",
	} {
		if servingFatalLine("ok\n"+line+"\n") == "" {
			t.Fatalf("fatal line not recognized: %q", line)
		}
	}
	for _, line := range []string{
		`VERB [ log_server_request] request | request="{\"messages\":[{\"content\":\"CUDA error: out of memory\"}]}"`,
		"The build failed with: CUDA error: out of memory",
		"INFO [ print] the user mentions a Segmentation fault",
	} {
		if got := servingFatalLine(line + "\n"); got != "" {
			t.Fatalf("text in a logged prompt or answer counted as fatal: %q", got)
		}
	}
}

// A fatal-looking line on its own line in generated text, while the server
// keeps answering, must never stop it.
func TestServingWatchKeepsAnAnsweringServer(t *testing.T) {
	b := newFakeBackend()
	b.appendLog("CUDA error: out of memory\n")
	sig := make(chan os.Signal, 1)
	if _, returned := waitServing(t, b, sig, answeringHealth(t), 600*time.Millisecond); returned {
		t.Fatal("a server that still answers /health was treated as dead")
	}
	if !b.IsRunning() {
		t.Fatal("an answering server was killed")
	}
}

// A silent /health with CPU still advancing is a long decode, not a hang.
func TestServingWatchKeepsABusyServer(t *testing.T) {
	b := newFakeBackend()
	b.burning = true
	b.appendLog("CUDA error: out of memory\n")
	sig := make(chan os.Signal, 1)
	if _, returned := waitServing(t, b, sig, silentHealth(t), 600*time.Millisecond); returned {
		t.Fatal("a backend still using CPU was treated as hung")
	}
}

func TestServingWatchStillHonoursShutdownAndExit(t *testing.T) {
	b := newFakeBackend()
	sig := make(chan os.Signal, 1)
	sig <- os.Interrupt
	if crashed, returned := waitServing(t, b, sig, answeringHealth(t), 2*time.Second); !returned || crashed {
		t.Fatal("a shutdown signal must end serving as a clean stop")
	}
	b2 := newFakeBackend()
	_ = b2.Stop()
	if crashed, returned := waitServing(t, b2, make(chan os.Signal, 1), answeringHealth(t), 2*time.Second); !returned || !crashed {
		t.Fatal("an exited backend must end serving as a crash")
	}
}
