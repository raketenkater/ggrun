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
	if cfg.SWAFull {
		t.Fatal("full SWA cache should default off")
	}
	if cfg.CgroupHeadroomMB != 4096 {
		t.Fatalf("expected default cgroup headroom 4096 MiB, got %d", cfg.CgroupHeadroomMB)
	}
	if cfg.AllowLiveMemoryProbe {
		t.Fatal("live memory probe approval must default off")
	}
	if cfg.HotExperts != "auto" {
		t.Fatalf("hot experts should default to measured auto mode, got %q", cfg.HotExperts)
	}
}

func TestNormalizeHotExperts(t *testing.T) {
	for input, want := range map[string]string{
		"": "auto", " AUTO ": "auto", "On": "on", "Off": "off", "01": "1", "64": "64",
	} {
		got, err := NormalizeHotExperts(input)
		if err != nil || got != want {
			t.Fatalf("NormalizeHotExperts(%q) = %q, %v; want %q", input, got, err, want)
		}
	}
	for _, input := range []string{"0", "-1", "1.5", "unbounded"} {
		if got, err := NormalizeHotExperts(input); err == nil || got != "" {
			t.Fatalf("NormalizeHotExperts(%q) accepted as %q", input, got)
		}
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
SWA_FULL=true
ALLOW_LIVE_MEMORY_PROBE=true
VISION=true
LLM_HOT_EXPERTS=7
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
	if !cfg.SWAFull {
		t.Fatal("expected full SWA cache true")
	}
	if !cfg.AllowLiveMemoryProbe {
		t.Fatal("expected remembered live memory probe approval")
	}
	if !cfg.Vision {
		t.Fatalf("expected vision true")
	}
	if cfg.HotExperts != "7" {
		t.Fatalf("expected 7 hot-expert slots, got %q", cfg.HotExperts)
	}
}

func TestSaveAndLoad(t *testing.T) {
	tmpDir := t.TempDir()
	origHome := os.Getenv("HOME")
	os.Setenv("HOME", tmpDir)
	defer os.Setenv("HOME", origHome)

	cfg := &Config{
		Port:                 9090,
		Ctx:                  "4096",
		ModelDir:             "/test/models",
		CacheDir:             "/test/cache",
		Backend:              "llama",
		KVPlacement:          "cpu",
		KVQuality:            "high",
		SWAFull:              true,
		AllowLiveMemoryProbe: true,
		TuneRounds:           3,
		Vision:               true,
		Parallel:             2,
		KeepAlive:            30,
		Host:                 "0.0.0.0",
		Spec:                 "ngram",
		HotExperts:           "6",
		RAMLimitPercent:      87,
		CgroupHeadroomMB:     2048,
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
	if loaded.CgroupHeadroomMB != 2048 {
		t.Fatalf("cgroup headroom mismatch: %d", loaded.CgroupHeadroomMB)
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
	if !loaded.SWAFull {
		t.Fatal("full SWA cache mismatch")
	}
	if !loaded.AllowLiveMemoryProbe {
		t.Fatal("live memory probe approval mismatch")
	}
	if loaded.HotExperts != "6" {
		t.Fatalf("hot-expert policy mismatch: %q", loaded.HotExperts)
	}

	data, err := os.ReadFile(Path())
	if err != nil {
		t.Fatalf("read saved config: %v", err)
	}
	for _, want := range []string{"LLM_PORT=", "LLM_CTX_SIZE=", "LLM_KV_QUALITY=", "LLM_SWA_FULL=true", "LLM_ALLOW_LIVE_MEMORY_PROBE=true", "LLM_RAM_LIMIT_PERCENT=87", "LLM_SPEC=", "LLM_HOT_EXPERTS=\"6\""} {
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
LLM_SWA_FULL=1
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
	if !cfg.SWAFull {
		t.Fatal("full SWA cache was not loaded")
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
	t.Setenv("LLM_SWA_FULL", "true")
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
	if !cfg.SWAFull || !cfg.IsExplicit("SWA_FULL") || !cfg.IsExplicit("LLM_SWA_FULL") || !cfg.IsExplicit("port") || cfg.IsExplicit("CACHE_DIR") {
		t.Fatalf("explicit source reporting is wrong: swa=%v sources=%v", cfg.SWAFull, cfg.sources)
	}
	show := cfg.Show()
	for _, want := range []string{"9090                 (file)", "/from-env            (env)", "true                 (env)", "(default)"} {
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

// A binary installed at ~/.local/bin/ggrun makes AppHome() resolve to ~/.local.
// Treating that as an app home hid the user's real $HOME/.config/ggrun/config:
// saved settings silently reverted to defaults on the next launch.
func TestPathIgnoresABinDirectoryThatIsNotAGgrunInstall(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("LLM_CONFIG", "")
	// Pretend the executable lives in <home>/.local/bin, so AppHome() reports
	// <home>/.local -- a directory holding no ggrun state at all.
	t.Setenv("LLM_APP_HOME", filepath.Join(home, ".local"))

	want := filepath.Join(home, ".config", "ggrun", "config")
	if got := Path(); got != want {
		t.Errorf("Path() = %q, want the user's own config at %q", got, want)
	}
}

// An existing $HOME config must win over an app home that has none, so an
// install location change never orphans settings that are already on disk.
func TestPathPrefersAnExistingUserConfigOverAnEmptyAppHome(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("LLM_CONFIG", "")
	t.Setenv("LLM_APP_HOME", filepath.Join(home, ".local"))

	userCfg := filepath.Join(home, ".config", "ggrun", "config")
	if err := os.MkdirAll(filepath.Dir(userCfg), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(userCfg, []byte("LLM_MODEL_DIR=\"/models\"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := Path(); got != userCfg {
		t.Errorf("Path() = %q, want the existing user config %q", got, userCfg)
	}
}

// A genuine self-contained install keeps its config beside backends.json.
func TestPathHonoursARealAppHome(t *testing.T) {
	home := t.TempDir()
	appHome := filepath.Join(home, "ggrun")
	t.Setenv("HOME", home)
	t.Setenv("LLM_CONFIG", "")
	t.Setenv("LLM_APP_HOME", appHome)

	if err := os.MkdirAll(filepath.Join(appHome, ".config"), 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := filepath.Join(appHome, ".config", "backends.json")
	if err := os.WriteFile(manifest, []byte("[]"), 0o600); err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(appHome, ".config", "ggrun", "config")
	if got := Path(); got != want {
		t.Errorf("Path() = %q, want the install-local config %q", got, want)
	}
}

// LLM_CONFIG stays an absolute override.
func TestPathRespectsExplicitOverride(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("LLM_CONFIG", "/tmp/explicit-ggrun.conf")
	if got := Path(); got != "/tmp/explicit-ggrun.conf" {
		t.Errorf("Path() = %q, want the explicit override", got)
	}
}

// Under a test runner stdin is a pipe (or /dev/null), never a terminal. Edit
// must refuse before launching $EDITOR: nano against a pipe sprays curses
// output and then dies, and the user is left without the config path.
func TestEditRefusesNonInteractiveTerminal(t *testing.T) {
	t.Setenv("EDITOR", "/bin/true") // would "succeed" instantly if launched
	if err := Edit(); err == nil {
		t.Fatal("Edit succeeded without an interactive terminal; editor guard missing")
	} else {
		msg := err.Error()
		if !strings.Contains(msg, "interactive terminal") {
			t.Errorf("error %q does not mention the interactive-terminal requirement", msg)
		}
		if !strings.Contains(msg, Path()) {
			t.Errorf("error %q does not name the config path %q", msg, Path())
		}
	}
}
