package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/raketenkater/ggrun/pkg/config"
)

func TestActionMenuArrowNav(t *testing.T) {
	// First-run menu: Down advances the cursor.
	fr := Model{screen: ScreenFirstRun, modelDir: "/tmp"}
	fr.input = textinput.New()
	nm, _ := fr.Update(tea.KeyMsg{Type: tea.KeyDown})
	fr = nm.(Model)
	if fr.menuCursor != 1 {
		t.Fatalf("firstrun down: expected menuCursor 1, got %d", fr.menuCursor)
	}
}

func TestModelConfigArrowNav(t *testing.T) {
	m := Model{
		screen:      ScreenModelConfig,
		models:      []ModelItem{{Name: "test.gguf"}},
		kvPlacement: "auto",
		ctxMode:     "fit",
		ctxSize:     "fit",
	}
	m.input = textinput.New()

	// Down through the real Update() path should advance the cursor.
	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = nm.(Model)
	if m.cfgCursor != 1 {
		t.Fatalf("down: expected cfgCursor 1, got %d", m.cfgCursor)
	}
	nm, _ = m.Update(tea.KeyMsg{Type: tea.KeyUp})
	m = nm.(Model)
	if m.cfgCursor != 0 {
		t.Fatalf("up: expected cfgCursor 0, got %d", m.cfgCursor)
	}
	// Right on the context row (cursor 0) cycles fit -> max.
	nm, _ = m.Update(tea.KeyMsg{Type: tea.KeyRight})
	m = nm.(Model)
	if m.ctxMode != "max" {
		t.Fatalf("right on context: expected ctxMode max, got %q", m.ctxMode)
	}
}

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

func TestBuildLaunchRequestCarriesSelectedBackend(t *testing.T) {
	m := Model{
		models:        []ModelItem{{Name: "DeepSeek", Path: "/models/deepseek.gguf"}},
		selectedModel: 0,
		backend:       "llama",
		kvPlacement:   "auto",
		ctxMode:       "fit",
	}
	req := m.buildLaunchRequest()
	if req == nil {
		t.Fatal("expected launch request")
	}
	if req.Backend != "llama" {
		t.Fatalf("expected selected backend to be carried into launch request, got %q", req.Backend)
	}
}

func TestBuildArgsUsesPlannerDryRunCommand(t *testing.T) {
	m := Model{
		models:        []ModelItem{{Name: "DeepSeek", Path: "/models/deepseek.gguf"}},
		selectedModel: 0,
		backend:       "llama",
		kvPlacement:   "cpu",
		kvQuality:     "high",
		ctxMode:       "fit",
	}
	args := m.buildArgs()
	joined := strings.Join(args, " ")
	if len(args) < 2 || args[0] != "ggrun" || args[1] != "dry-run" {
		t.Fatalf("TUI dry run should call the real planner, got %v", args)
	}
	if !strings.Contains(joined, "--backend llama") {
		t.Fatalf("selected backend must stay explicit, got %q", joined)
	}
	if strings.Contains(joined, " -ngl ") {
		t.Fatalf("TUI dry run must not emit fake low-level placement flags, got %q", joined)
	}
}

func TestPrelaunchViewShowsSelectedBackend(t *testing.T) {
	m := Model{
		models:        []ModelItem{{Name: "DeepSeek", Path: "/models/deepseek.gguf"}},
		selectedModel: 0,
		backend:       "llama",
		kvPlacement:   "auto",
		ctxMode:       "fit",
	}
	view := m.viewPrelaunch()
	if !strings.Contains(view, "Backend:") || !strings.Contains(view, "llama") {
		t.Fatalf("prelaunch view should show selected backend, got %q", view)
	}
}

func TestBackendTagUsesRegisteredBackendTag(t *testing.T) {
	if got := (Model{backend: "custom"}).backendTag(); got != "custom" {
		t.Fatalf("expected custom tune tag, got %q", got)
	}
	if got := (Model{backend: "ik_llama"}).backendTag(); got != "ik" {
		t.Fatalf("expected ik tune tag, got %q", got)
	}
	if got := (Model{}).backendTag(); got != "llama" {
		t.Fatalf("empty backend should use llama tune tag, got %q", got)
	}
}

func TestInitialModelUsesConfigPaths(t *testing.T) {
	appHome := filepath.Join(t.TempDir(), "ggrun")
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

// The launch request must carry the configured KV quality. A hardcoded "mid"
// here once emitted an explicit --kv-quality mid on every TUI launch, silently
// overriding the user's saved setting (the Settings screen appeared to save
// but launches always went out with q8_0 KV).
func TestBuildLaunchRequestCarriesConfiguredKVQuality(t *testing.T) {
	m := Model{
		models:        []ModelItem{{Name: "DeepSeek", Path: "/models/deepseek.gguf"}},
		selectedModel: 0,
		backend:       "llama",
		kvPlacement:   "auto",
		kvQuality:     "high",
		ctxMode:       "fit",
	}
	req := m.buildLaunchRequest()
	if req == nil {
		t.Fatal("expected launch request")
	}
	if req.KVQuality != "high" {
		t.Fatalf("configured KV quality must reach the launch request, got %q", req.KVQuality)
	}
}

// Changing KV settings in the Settings screen must apply to the live session,
// not only to the next TUI start.
func TestApplySettingSyncsKVIntoLiveSession(t *testing.T) {
	t.Setenv("LLM_APP_HOME", t.TempDir())
	m := Model{settingsCfg: config.Defaults(), kvQuality: "mid", kvPlacement: "auto"}
	var qRow, pRow settingRow
	for _, r := range settingRows() {
		switch r.label {
		case "KV quality":
			qRow = r
		case "KV placement":
			pRow = r
		}
	}
	if qRow.label == "" || pRow.label == "" {
		t.Fatal("KV settings rows not found")
	}
	m.applySetting(qRow, "high")
	if m.kvQuality != "high" {
		t.Fatalf("KV quality setting must sync into the session, got %q", m.kvQuality)
	}
	m.applySetting(pRow, "cpu")
	if m.kvPlacement != "cpu" {
		t.Fatalf("KV placement setting must sync into the session, got %q", m.kvPlacement)
	}
}
