package config

import (
	"os"
	"path/filepath"
	"strings"
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
	if cfg.Host != "127.0.0.1" {
		t.Fatalf("expected safe loopback host, got %q", cfg.Host)
	}
	if cfg.RAMLimitPercent != 95 {
		t.Fatalf("expected default RAM limit 95%%, got %d", cfg.RAMLimitPercent)
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
	if cfg.CtxValue() != "8192" {
		t.Fatalf("expected ctx 8192, got %s", cfg.CtxValue())
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
		Port:            9090,
		Ctx:             "4096",
		ModelDir:        "/test/models",
		CacheDir:        "/test/cache",
		Backend:         "llama",
		KVPlacement:     "cpu",
		KVQuality:       "high",
		TuneRounds:      3,
		Vision:          true,
		Parallel:        2,
		KeepAlive:       30,
		Host:            "0.0.0.0",
		Spec:            "ngram",
		RAMLimitPercent: 87,
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
	if loaded.CtxValue() != "4096" {
		t.Fatalf("ctx mismatch: %s", loaded.CtxValue())
	}
	if loaded.Spec != "ngram" {
		t.Fatalf("spec mismatch: %s", loaded.Spec)
	}
	if loaded.RAMLimitPercent != 87 {
		t.Fatalf("RAM limit percent mismatch: %d", loaded.RAMLimitPercent)
	}

	data, err := os.ReadFile(Path())
	if err != nil {
		t.Fatalf("read saved config: %v", err)
	}
	for _, want := range []string{"LLM_PORT=", "LLM_CTX_SIZE=", "LLM_KV_QUALITY=", "LLM_RAM_LIMIT_PERCENT=87", "LLM_SPEC="} {
		if !strings.Contains(string(data), want) {
			t.Fatalf("saved config missing %s:\n%s", want, string(data))
		}
	}
}

func TestLoadFileCanonicalKeysAndContextModes(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "config")
	content := `LLM_PORT=9091
LLM_CTX_SIZE=fit
LLM_MODEL_DIR="/models-v3"
LLM_KV_PLACEMENT=cpu
LLM_KV_QUALITY=low
LLM_SPEC=ngram
LLM_TUNE_ROUNDS=7
LLM_HOST="127.0.0.1"
`
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}

	cfg := Defaults()
	if err := loadFile(path, cfg); err != nil {
		t.Fatalf("load: %v", err)
	}
	if cfg.Port != 9091 {
		t.Fatalf("expected port 9091, got %d", cfg.Port)
	}
	if cfg.CtxValue() != "fit" || cfg.Ctx != "fit" {
		t.Fatalf("expected fit context, got %q", cfg.Ctx)
	}
	if cfg.ModelDir != "/models-v3" {
		t.Fatalf("expected canonical model dir, got %s", cfg.ModelDir)
	}
	if cfg.KVPlacement != "cpu" || cfg.KVQuality != "low" {
		t.Fatalf("kv config mismatch: %s/%s", cfg.KVPlacement, cfg.KVQuality)
	}
	if cfg.Spec != "ngram" || cfg.TuneRounds != 7 || cfg.Host != "127.0.0.1" {
		t.Fatalf("new config keys mismatch: spec=%s rounds=%d host=%s", cfg.Spec, cfg.TuneRounds, cfg.Host)
	}
}

func TestApplyCtxValueMax(t *testing.T) {
	cfg := Defaults()
	if err := cfg.SetCtxValue("max"); err != nil {
		t.Fatal(err)
	}
	if cfg.CtxValue() != "max" || cfg.Ctx != "max" {
		t.Fatalf("expected max context, got %q", cfg.Ctx)
	}
}

func TestLoadFileRejectsInvalidSafetyValues(t *testing.T) {
	for name, content := range map[string]string{
		"port":        "LLM_PORT=abc\n",
		"context":     "LLM_CTX_SIZE=lots\n",
		"headroom":    "LLM_VRAM_HEADROOM=two gigabytes\n",
		"parallel":    "LLM_PARALLEL=-1\n",
		"keep_alive":  "LLM_KEEP_ALIVE=never\n",
		"ram_percent": "LLM_RAM_LIMIT_PERCENT=101\n",
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config")
			if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := loadFile(path, Defaults()); err == nil {
				t.Fatalf("loadFile(%q) accepted invalid value", content)
			}
		})
	}
}

func TestShowReportsEachSettingSource(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config")
	if err := os.WriteFile(path, []byte("LLM_PORT=9090\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLM_CONFIG", path)
	t.Setenv("LLM_MODEL_DIR", "/from-env")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.sources["PORT"]; got != "file" {
		t.Fatalf("PORT source = %q, want file", got)
	}
	if got := cfg.sources["MODEL_DIR"]; got != "env" {
		t.Fatalf("MODEL_DIR source = %q, want env", got)
	}
	if got := cfg.sources["CACHE_DIR"]; got != "default" {
		t.Fatalf("CACHE_DIR source = %q, want default", got)
	}
	show := cfg.Show()
	for _, want := range []string{"9090                 (file)", "/from-env            (env)", "(default)"} {
		if !strings.Contains(show, want) {
			t.Fatalf("show missing %q:\n%s", want, show)
		}
	}
}

func TestBudgetParserRejectsMalformedValues(t *testing.T) {
	for _, raw := range []string{"-1G", "twoG", "2.5G"} {
		if _, err := ParseBudgetMBStrict(raw); err == nil {
			t.Fatalf("ParseBudgetMBStrict(%q) succeeded", raw)
		}
	}
	if got, err := ParseBudgetMBStrict("2G"); err != nil || got != 2048 {
		t.Fatalf("ParseBudgetMBStrict(2G) = %d, %v; want 2048, nil", got, err)
	}
}
