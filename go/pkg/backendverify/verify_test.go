//go:build !windows

package backendverify

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const goodLog = `load_tensors: offloading 0 repeating layers to GPU
load_tensors: offloaded 0/25 layers to GPU
load_tensors:          CPU model buffer size =   402.17 MiB
llama_context:        CPU  output buffer size =     0.58 MiB
llama_kv_cache:        CPU KV buffer size =     6.00 MiB
llama_context:        CPU compute buffer size =   298.75 MiB
main: model loaded
main: server is listening on http://127.0.0.2:40000
`

// fakeBackend prints log and then sleeps, like a loaded server.
func fakeBackend(t *testing.T, log string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "llama-server")
	script := "#!/bin/sh\nif [ \"$1\" = --help ]; then echo '--no-mmap  -lv, --log-verbosity N'; exit 0; fi\n" +
		"case \"$*\" in *--no-mmap*-lv?4*) ;; *) echo \"unexpected argv: $*\"; exit 2;; esac\n" +
		"cat <<'LOG'\n" + log + "LOG\nsleep 30\n"
	if err := os.WriteFile(p, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestContractCheckPassesOnTheCurrentLogWording(t *testing.T) {
	r, err := Run(fakeBackend(t, goodLog), "m.gguf", 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if r.HostBuffersMB <= 0 || r.KVTotalMB != 6 || r.TotalLayers != 25 || r.OffloadedLayers != 0 {
		t.Fatalf("result %+v", r)
	}
}

// I-05 proof: a backend that changes one log line fails the check.
func TestChangedLogLineFailsTheCheck(t *testing.T) {
	for name, change := range map[string][2]string{
		"kv wording":      {"CPU KV buffer size", "CPU KV cache bytes"},
		"buffer wording":  {"model buffer size", "weights size"},
		"buffer on a GPU": {"llama_kv_cache:        CPU KV buffer size", "load_tensors: CUDA0 model buffer size =   100.00 MiB\nllama_kv_cache:        CPU KV buffer size"},
		"gpu layers":      {"offloaded 0/25", "offloaded 4/25"},
	} {
		log := strings.Replace(goodLog, change[0], change[1], 1)
		if _, err := Run(fakeBackend(t, log), "m.gguf", 10*time.Second); err == nil {
			t.Errorf("%s: changed log passed the contract check", name)
		}
	}
}

func TestBackendThatExitsEarlyFails(t *testing.T) {
	p := filepath.Join(t.TempDir(), "llama-server")
	if err := os.WriteFile(p, []byte("#!/bin/sh\necho 'error: unknown argument'\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := Run(p, "m.gguf", 10*time.Second); err == nil || !strings.Contains(err.Error(), "exited before loading") {
		t.Fatalf("err %v", err)
	}
}
