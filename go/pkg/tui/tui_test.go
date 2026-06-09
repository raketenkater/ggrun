package tui

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDiscoverModels(t *testing.T) {
	// Test with a temp dir
	models := discoverModels("/tmp/nonexistent-dir-12345")
	if len(models) != 0 {
		t.Fatalf("expected no models for nonexistent dir")
	}
}

func TestBoolLabel(t *testing.T) {
	if boolLabel(true) != "on" {
		t.Fatalf("expected 'on' for true")
	}
	if boolLabel(false) != "off" {
		t.Fatalf("expected 'off' for false")
	}
}

func TestHWSummary(t *testing.T) {
	// Test with nil
	s := hwSummary(nil)
	if s != "detecting..." {
		t.Fatalf("expected 'detecting...' for nil caps")
	}
}

func TestInitialModelUsesConfigPaths(t *testing.T) {
	appHome := filepath.Join(t.TempDir(), "llm-server")
	cfgDir := filepath.Join(appHome, ".config")
	modelDir := filepath.Join(appHome, "models")
	cacheDir := filepath.Join(appHome, ".cache")
	if err := os.MkdirAll(cfgDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatal(err)
	}
	cfgPath := filepath.Join(cfgDir, "config")
	doc := "MODEL_DIR=\"" + modelDir + "\"\nCACHE_DIR=\"" + cacheDir + "\"\nBACKEND=\"vulkan\"\nKV_PLACEMENT=\"gpu\"\nTUNE_ROUNDS=\"9\"\n"
	if err := os.WriteFile(cfgPath, []byte(doc), 0644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLM_CONFIG", cfgPath)
	t.Setenv("LLM_APP_HOME", appHome)
	m := InitialModel()
	if m.modelDir != modelDir || m.cacheDir != cacheDir || m.backend != "vulkan" || m.kvPlacement != "gpu" || m.aituneRounds != 9 {
		t.Fatalf("initial model did not use config: %#v", m)
	}
}
