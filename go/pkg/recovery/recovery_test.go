package recovery

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeLog(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "launch.log")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestParseLoadFailureUsesOwnLog(t *testing.T) {
	l := &Launcher{}
	l.lastLogPath = writeLog(t, "llama_model_load: error loading model\nCUDA out of memory\n")
	ft, _ := l.parseLoadFailure()
	if ft != FailureOOM {
		t.Fatalf("expected oom, got %s", ft)
	}
}

func TestParseLoadFailureNoLogPath(t *testing.T) {
	l := &Launcher{}
	if ft, _ := l.parseLoadFailure(); ft != FailureUnknown {
		t.Fatalf("expected unknown without log path, got %s", ft)
	}
}

func TestParseLoadFailureBloomIsNotOOM(t *testing.T) {
	// "Bloom" and "room" contain the substring "oom" but are not OOM markers.
	l := &Launcher{}
	l.lastLogPath = writeLog(t, "loading model /models/Bloom-7B.gguf\nno room for more tensors in this print statement\nsegfault\n")
	ft, _ := l.parseLoadFailure()
	if ft == FailureOOM || ft == FailureRAMOOM {
		t.Fatalf("Bloom/room must not classify as OOM, got %s", ft)
	}
}

func TestParseLoadFailureBufferType(t *testing.T) {
	// CPU-only llama-server told to offload to CUDA0: deterministic capability
	// mismatch that must not loop, and must surface the real error line.
	l := &Launcher{}
	l.lastLogPath = writeLog(t, "load_backend: loaded CPU backend\n"+
		"error while handling argument \"-ot\": unknown buffer type\n"+
		"Available buffer types:\n  CPU\n")
	ft, msg := l.parseLoadFailure()
	if ft != FailureBackendCapability {
		t.Fatalf("expected backend_capability, got %s", ft)
	}
	if !ft.deterministic() {
		t.Fatal("buffer-type failure must be deterministic (no restart loop)")
	}
	if !strings.Contains(strings.ToLower(msg), "unknown buffer type") {
		t.Fatalf("expected the actionable error line in msg, got %q", msg)
	}
}

func TestParseLoadFailureSurfacesUnknownStderr(t *testing.T) {
	// An unclassified failure must still surface the backend's real output
	// instead of the old empty "failure: unknown:" message.
	l := &Launcher{}
	l.lastLogPath = writeLog(t, "starting server\nterminate called after throwing an instance of std::runtime_error\n")
	ft, msg := l.parseLoadFailure()
	if ft != FailureUnknown {
		t.Fatalf("expected unknown, got %s", ft)
	}
	if msg == "" {
		t.Fatal("FailureUnknown must surface the real stderr, got empty msg")
	}
}

func TestParseLoadFailureRealOOMVariants(t *testing.T) {
	cases := map[string]FailureType{
		"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1024 MiB failed: out of memory":                             FailureOOM,
		"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 11875.43 MiB on device 1: cudaMalloc failed: out of memory": FailureCUDAOOM,
		"kernel: Out of memory: Killed process 1234 (llama-server)":                                                         FailureOOM,
		"CUDA error: out of memory":                             FailureOOM,
		"RAM OOM detected while loading experts":                FailureRAMOOM,
		"unknown model architecture: 'qwen9'":                   FailureUnknownModel,
		"pinned memory capacity exceeded while loading tensors": FailurePinnedCap,
	}
	for line, want := range cases {
		l := &Launcher{}
		l.lastLogPath = writeLog(t, line+"\n")
		if ft, _ := l.parseLoadFailure(); ft != want {
			t.Fatalf("line %q: expected %s, got %s", line, want, ft)
		}
	}
}

func TestParseCUDAOOMDetails(t *testing.T) {
	device, allocMB, ok := ParseCUDAOOM("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 11875.43 MiB on device 1: cudaMalloc failed: out of memory")
	if !ok || device != 1 || allocMB != 11876 {
		t.Fatalf("unexpected CUDA OOM parse: device=%d alloc=%d ok=%v", device, allocMB, ok)
	}
}

func TestParseCUDADeviceFromVMMOOM(t *testing.T) {
	device, ok := ParseCUDADevice("120.54 E   current device: 0, in function alloc at ggml-cuda.cu:529")
	if !ok || device != 0 {
		t.Fatalf("unexpected CUDA device parse: device=%d ok=%v", device, ok)
	}
	if _, ok := ParseCUDADevice("CUDA error: out of memory"); ok {
		t.Fatal("OOM marker without a current-device diagnostic must not invent a device")
	}
}
