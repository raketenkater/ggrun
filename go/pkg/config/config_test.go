package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaults(t *testing.T) {
	cfg := Defaults()
	if cfg.Port != 8081 {
		t.Fatalf("expected port 8081, got %d", cfg.Port)
	}
	if cfg.ModelDir == "" {
		t.Fatalf("model dir should not be empty")
	}
}

func TestLoadFile(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "config")
	content := `PORT=9090
CTX_SIZE=8192
MODEL_DIR="/models"
BACKEND=ik_llama
KV_PLACEMENT=gpu
VISION=true
`
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}

	cfg := Defaults()
	if err := loadFile(path, cfg); err != nil {
		t.Fatalf("load: %v", err)
	}

	if cfg.Port != 9090 {
		t.Fatalf("expected port 9090, got %d", cfg.Port)
	}
	if cfg.CtxSize != 8192 {
		t.Fatalf("expected ctx 8192, got %d", cfg.CtxSize)
	}
	if cfg.ModelDir != "/models" {
		t.Fatalf("expected /models, got %s", cfg.ModelDir)
	}
	if cfg.Backend != "ik_llama" {
		t.Fatalf("expected ik_llama, got %s", cfg.Backend)
	}
	if cfg.KVPlacement != "gpu" {
		t.Fatalf("expected gpu, got %s", cfg.KVPlacement)
	}
	if !cfg.Vision {
		t.Fatalf("expected vision true")
	}
}

func TestSaveAndLoad(t *testing.T) {
	tmpDir := t.TempDir()
	origHome := os.Getenv("HOME")
	os.Setenv("HOME", tmpDir)
	defer os.Setenv("HOME", origHome)

	cfg := &Config{
		Port:        9090,
		CtxSize:     4096,
		ModelDir:    "/test/models",
		CacheDir:    "/test/cache",
		Backend:     "llama",
		KVPlacement: "cpu",
		KVQuality:   "high",
		TuneRounds:  3,
		Vision:      true,
		Parallel:    2,
		KeepAlive:   30,
		Host:        "0.0.0.0",
	}

	if err := cfg.Save(); err != nil {
		t.Fatalf("save: %v", err)
	}

	loaded, err := Load()
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if loaded.Port != 9090 {
		t.Fatalf("port mismatch: %d", loaded.Port)
	}
	if loaded.ModelDir != "/test/models" {
		t.Fatalf("model dir mismatch: %s", loaded.ModelDir)
	}
	if !loaded.Vision {
		t.Fatalf("vision mismatch")
	}
}
