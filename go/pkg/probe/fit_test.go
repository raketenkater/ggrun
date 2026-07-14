package probe

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/gguf"
)

func TestEstimateFitCtxForInfoUsesExistingMetadata(t *testing.T) {
	cacheDir := t.TempDir()
	probeDir := filepath.Join(cacheDir, "probes")
	if err := os.MkdirAll(probeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	valid := "MODEL=Model.gguf\nctx=65536\nPROBED_KV_PER_LAYER_MB=10\n"
	corrupt := "MODEL=Model.gguf\nctx=131072\nPROBED_KV_PER_LAYER_MB=10000\n"
	if err := os.WriteFile(filepath.Join(probeDir, "valid.probe"), []byte(valid), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(probeDir, "corrupt.probe"), []byte(corrupt), 0o644); err != nil {
		t.Fatal(err)
	}

	info := &gguf.Info{BlockCount: 40, ContextLength: 100000}
	got := EstimateFitCtxForInfo("/models/Model-00001-of-00002.gguf", cacheDir, info, 128000)
	if got != 65536 {
		t.Fatalf("fit context = %d, want valid cached 65536", got)
	}
}
