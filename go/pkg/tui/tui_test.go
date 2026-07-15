package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/recommend"
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

func TestFirstRunUpdateIsExplicitAndNonBlocking(t *testing.T) {
	m := Model{screen: ScreenFirstRun, modelDir: "/tmp"}
	m.input = textinput.New()
	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'u'}})
	m = nm.(Model)
	if m.launchRequest == nil || !m.launchRequest.Update {
		t.Fatal("first-run update shortcut must return an explicit update request")
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

func TestDiscoverModelsKeepsSameBasenameInDifferentDirectories(t *testing.T) {
	dir := t.TempDir()
	for _, sub := range []string{"repo-a", "repo-b"} {
		modelDir := filepath.Join(dir, sub)
		if err := os.MkdirAll(modelDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(modelDir, "model-Q4.gguf"), []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if got := len(discoverModels(dir)); got != 2 {
		t.Fatalf("same basename in two repositories should produce two choices, got %d", got)
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

func TestRecommendedViewLabelsPredictedSpeedAsEstimate(t *testing.T) {
	rec := recommend.Recommendation{
		Candidate:    recommend.Candidate{Name: "Test model"},
		Fit:          "single GPU",
		QuantName:    "Q4_K_M",
		QuantSizeGB:  4,
		PredictedTPS: 6,
	}
	m := Model{
		recommendationGroups: recommend.Categories{Balanced: []recommend.Recommendation{rec}},
		recommendations:      []recommend.Recommendation{rec},
	}
	view := m.viewRecommended()
	if !strings.Contains(view, "~6 t/s") || !strings.Contains(view, "Speeds are estimates") {
		t.Fatalf("recommended speed must be clearly estimated, got %q", view)
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

func TestLaunchArgsAutoBackendDoesNotDisableArchitectureRouting(t *testing.T) {
	req := &LaunchRequest{ModelPath: "/models/hy3.gguf", Backend: "auto", CtxFlag: "fit"}
	joined := strings.Join(req.LaunchArgs(), " ")
	if strings.Contains(joined, "--backend") {
		t.Fatalf("automatic backend must remain implicit so architecture routes can select a fork: %q", joined)
	}
}

func TestLaunchArgsCarriesAITuneRounds(t *testing.T) {
	req := &LaunchRequest{ModelPath: "/models/test.gguf", AITune: true, AITuneRounds: 11}
	joined := strings.Join(req.LaunchArgs(), " ")
	if !strings.Contains(joined, "--rounds 11") {
		t.Fatalf("TUI AI tune rounds must reach cmdTune: %q", joined)
	}
}

func TestRunModesAreMutuallyExclusive(t *testing.T) {
	m := Model{screen: ScreenModelConfig, models: []ModelItem{{Name: "test.gguf"}}, kvPlacement: "auto", ctxMode: "fit", ctxSize: "fit"}
	m.input = textinput.New()
	m.cfgCursor = 4 // AI tune
	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)
	if !m.aitune || m.benchmark {
		t.Fatal("AI tune should enable itself and disable benchmark")
	}
	m.cfgCursor = 8 // benchmark after AI tune adds the rounds row
	nm, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)
	if !m.benchmark || m.aitune {
		t.Fatal("benchmark should enable itself and disable AI tune")
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
	doc := "MODEL_DIR=\"" + modelDir + "\"\nCACHE_DIR=\"" + cacheDir + "\"\nBACKEND=\"vulkan\"\nKV_PLACEMENT=\"gpu\"\nCTX_SIZE=\"max\"\nPARALLEL=\"3\"\nPORT=\"9091\"\nVISION=\"1\"\nTUNE_ROUNDS=\"9\"\n"
	if err := os.WriteFile(cfgPath, []byte(doc), 0644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLM_CONFIG", cfgPath)
	t.Setenv("LLM_APP_HOME", appHome)
	m := InitialModel()
	if m.modelDir != modelDir || m.cacheDir != cacheDir || m.backend != "vulkan" || m.kvPlacement != "gpu" || m.aituneRounds != 9 || m.ctxMode != "max" || m.parallel != "3" || m.port != 9091 || !m.vision {
		t.Fatalf("config not restored: modelDir=%q cacheDir=%q backend=%q kv=%q rounds=%d ctx=%q parallel=%q port=%d vision=%v", m.modelDir, m.cacheDir, m.backend, m.kvPlacement, m.aituneRounds, m.ctxMode, m.parallel, m.port, m.vision)
	}
	if m.parallelSet {
		t.Fatal("a config parallel value is policy input, not an explicit per-launch override")
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

func TestApplySettingSyncsLaunchCriticalValues(t *testing.T) {
	t.Setenv("LLM_APP_HOME", t.TempDir())
	m := Model{settingsCfg: config.Defaults(), ctxMode: "fit", ctxSize: "fit", port: 8081, parallel: "1", aituneRounds: 8}
	rows := map[string]settingRow{}
	for _, row := range settingRows() {
		rows[row.label] = row
	}
	for _, label := range []string{"Context", "Vision", "Port", "Parallel", "AI-tune rounds"} {
		if rows[label].label == "" {
			t.Fatalf("settings row %q not found", label)
		}
	}
	m.applySetting(rows["Context"], "max")
	m.applySetting(rows["Vision"], "on")
	m.applySetting(rows["Port"], "9099")
	m.applySetting(rows["Parallel"], "3")
	m.applySetting(rows["AI-tune rounds"], "12")
	if m.ctxMode != "max" || !m.vision || m.port != 9099 || m.parallel != "3" || m.aituneRounds != 12 {
		t.Fatalf("launch settings not synced: ctx=%q vision=%v port=%d parallel=%q rounds=%d", m.ctxMode, m.vision, m.port, m.parallel, m.aituneRounds)
	}
	if m.parallelSet {
		t.Fatal("Settings parallel value must still allow Claude mode to apply its minimum slot policy")
	}
}

func TestApplySettingRejectsInvalidSafetyValues(t *testing.T) {
	m := Model{settingsCfg: config.Defaults(), port: 8081, parallel: "1"}
	rows := map[string]settingRow{}
	for _, row := range settingRows() {
		rows[row.label] = row
	}
	m.applySetting(rows["Port"], "not-a-port")
	if m.settingsCfg.Port != 8081 || m.port != 8081 || m.messageType != "warning" {
		t.Fatalf("invalid port changed settings: cfg=%d live=%d message=%q", m.settingsCfg.Port, m.port, m.message)
	}
	m.applySetting(rows["VRAM headroom"], "two gigabytes")
	if m.settingsCfg.VRAMHeadroom != "" || m.messageType != "warning" {
		t.Fatalf("invalid headroom changed settings: %q", m.settingsCfg.VRAMHeadroom)
	}
}

func TestPerModelParallelEntryIsExplicit(t *testing.T) {
	m := Model{
		screen:    ScreenModelConfig,
		models:    []ModelItem{{Name: "test.gguf", Path: "/models/test.gguf"}},
		inputMode: "parallel",
	}
	m.input = textinput.New()
	m.input.SetValue("1")
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = next.(Model)
	if m.parallel != "1" || !m.parallelSet {
		t.Fatalf("per-model parallel must be explicit: value=%q set=%v", m.parallel, m.parallelSet)
	}
	req := m.buildLaunchRequest()
	if req == nil || !req.ParallelSet || req.Parallel != 1 {
		t.Fatalf("explicit parallel did not reach launch request: %#v", req)
	}
}

func TestPerModelParallelEntryRejectsInvalidValue(t *testing.T) {
	m := Model{
		screen:    ScreenModelConfig,
		models:    []ModelItem{{Name: "test.gguf", Path: "/models/test.gguf"}},
		inputMode: "parallel",
		parallel:  "4",
	}
	m.input = textinput.New()
	m.input.SetValue("many")
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = next.(Model)
	if m.parallel != "4" || m.parallelSet || m.messageType != "warning" {
		t.Fatalf("invalid parallel changed launch settings: value=%q explicit=%v message=%q", m.parallel, m.parallelSet, m.message)
	}
}
