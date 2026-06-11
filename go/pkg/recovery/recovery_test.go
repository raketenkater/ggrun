package recovery

import (
	"os"
	"path/filepath"
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

func TestParseLoadFailureRealOOMVariants(t *testing.T) {
	cases := map[string]FailureType{
		"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1024 MiB failed: out of memory": FailureOOM,
		"kernel: Out of memory: Killed process 1234 (llama-server)":                             FailureOOM,
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
