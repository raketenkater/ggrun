package tui

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
)

func TestLatestLaunchRoundTripPreservesRunnableConfiguration(t *testing.T) {
	cacheDir := t.TempDir()
	req := &LaunchRequest{
		Update:        true,
		BackendArgs:   []string{"remove", "wrong"},
		DownloadRepo:  "wrong/repo",
		DownloadQuant: "Q4_K_M",
		ModelPath:     "/models/minimax.gguf",
		Port:          9091,
		CtxFlag:       "131072",
		KVPlacement:   "gpu",
		KVQuality:     "high",
		SWAFull:       true,
		SWAFullSet:    true,
		FlashAttn:     true,
		Parallel:      4,
		ParallelSet:   true,
		Vision:        true,
		Backend:       "ik_llama",
		ClaudeCode:    true,
		ClaudeProfile: "agent-parallel",
		ResumeSession: "session-123",
		SupportExpert: "on",
		SupportOnline: true,
		SupportSet:    true,
	}
	if err := SaveLatestLaunch(cacheDir, req); err != nil {
		t.Fatal(err)
	}
	loaded, savedAt, err := LoadLatestLaunch(cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	if savedAt.IsZero() {
		t.Fatal("latest launch has no timestamp")
	}
	if loaded.Update || len(loaded.BackendArgs) != 0 || loaded.DownloadRepo != "" || loaded.DownloadQuant != "" {
		t.Fatalf("non-launch actions leaked into replay: %#v", loaded)
	}
	want := *req
	want.Update = false
	want.BackendArgs = nil
	want.DownloadRepo = ""
	want.DownloadQuant = ""
	if !reflect.DeepEqual(*loaded, want) {
		t.Fatalf("loaded request = %#v, want %#v", *loaded, want)
	}
	if runtime.GOOS != "windows" {
		info, err := os.Stat(filepath.Join(cacheDir, latestLaunchFile))
		if err != nil {
			t.Fatal(err)
		}
		if got := info.Mode().Perm(); got != 0o600 {
			t.Fatalf("latest launch permissions = %o, want 600", got)
		}
	}

	// Replacing a prior record is part of normal use, not a one-shot path.
	req.Backend = "llama"
	if err := SaveLatestLaunch(cacheDir, req); err != nil {
		t.Fatalf("replace latest launch: %v", err)
	}
	loaded, _, err = LoadLatestLaunch(cacheDir)
	if err != nil || loaded.Backend != "llama" {
		t.Fatalf("replaced latest launch = %#v, err=%v", loaded, err)
	}
}

func TestLoadLatestLaunchDistinguishesMissingAndCorrupt(t *testing.T) {
	cacheDir := t.TempDir()
	if _, _, err := LoadLatestLaunch(cacheDir); !errors.Is(err, ErrNoLatestLaunch) {
		t.Fatalf("missing record error = %v, want ErrNoLatestLaunch", err)
	}
	if err := os.WriteFile(filepath.Join(cacheDir, latestLaunchFile), []byte("not-json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadLatestLaunch(cacheDir); err == nil || errors.Is(err, ErrNoLatestLaunch) {
		t.Fatalf("corrupt record should be reported distinctly, got %v", err)
	}
}

func TestLatestLaunchActionVisibleOnMainAndFirstRun(t *testing.T) {
	found := false
	for _, raw := range newMainList(nil).Items() {
		if item, ok := raw.(mainItem); ok && item.action == "latest" {
			found = strings.Contains(item.title, "Run latest")
		}
	}
	if !found {
		t.Fatal("Main menu does not expose Run latest configuration")
	}
	found = false
	for _, action := range firstRunActions() {
		if action == "latest" {
			found = true
		}
	}
	if !found {
		t.Fatal("first-run menu does not expose Run latest configuration")
	}
}

func TestLatestLaunchOpensReviewAndReplaysExactRequest(t *testing.T) {
	cacheDir := t.TempDir()
	modelDir := t.TempDir()
	externalDir := t.TempDir()
	modelPath := filepath.Join(externalDir, "MiniMax-M3-IQ3.gguf")
	if err := os.WriteFile(modelPath, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	want := &LaunchRequest{
		ModelPath:     modelPath,
		Port:          8088,
		CtxFlag:       "131072",
		KVPlacement:   "gpu",
		KVQuality:     "high",
		SWAFull:       true,
		SWAFullSet:    true,
		Parallel:      4,
		ParallelSet:   true,
		Backend:       "ik_llama",
		ClaudeCode:    true,
		ClaudeProfile: "agent-parallel",
		SupportExpert: "on",
		SupportOnline: true,
		SupportSet:    true,
	}
	if err := SaveLatestLaunch(cacheDir, want); err != nil {
		t.Fatal(err)
	}
	m := Model{
		screen:      ScreenMain,
		modelDir:    modelDir,
		cacheDir:    cacheDir,
		backend:     "auto",
		kvPlacement: "auto",
		kvQuality:   "auto",
		mainList:    newMainList(nil),
	}
	m.input = textinput.New()

	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'l'}})
	m = next.(Model)
	if cmd != nil {
		t.Fatal("opening latest configuration must not launch immediately")
	}
	if m.screen != ScreenPrelaunch || m.replayRequest == nil {
		t.Fatalf("latest action did not open replay review: screen=%v replay=%#v", m.screen, m.replayRequest)
	}
	if len(m.models) != 1 || m.models[m.selectedModel].Path != modelPath || !m.models[m.selectedModel].External {
		t.Fatalf("external latest model was not restored: %#v", m.models)
	}
	if m.backend != "ik_llama" || m.ctxSize != "131072" || m.kvPlacement != "gpu" || m.kvQuality != "high" || !m.swaFull || m.parallel != "4" || !m.parallelSet || m.supportExpert != "on" || !m.supportOnline {
		t.Fatalf("review fields do not reflect saved request: %#v", m)
	}
	if view := m.viewPrelaunch(); !strings.Contains(view, "Latest saved TUI configuration") || !strings.Contains(view, "Backend:        ik_llama") || !strings.Contains(view, "Full SWA cache: on") || !strings.Contains(view, "Support expert: on (required, ephemeral)") || !strings.Contains(view, "Online research: on") {
		t.Fatalf("pre-launch replay review is incomplete:\n%s", view)
	}

	next, cmd = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = next.(Model)
	if cmd == nil || m.launchRequest == nil {
		t.Fatal("confirming latest configuration did not return a launch request")
	}
	if !reflect.DeepEqual(m.launchRequest.LaunchArgs(), want.LaunchArgs()) {
		t.Fatalf("replayed args = %v, want %v", m.launchRequest.LaunchArgs(), want.LaunchArgs())
	}
}

func TestLatestLaunchEscapeCancelsReplayIntoEditableConfig(t *testing.T) {
	cacheDir := t.TempDir()
	modelDir := t.TempDir()
	modelPath := filepath.Join(modelDir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := SaveLatestLaunch(cacheDir, &LaunchRequest{ModelPath: modelPath, Backend: "ik_llama"}); err != nil {
		t.Fatal(err)
	}
	models := discoverModels(modelDir)
	m := Model{screen: ScreenMain, modelDir: modelDir, cacheDir: cacheDir, models: models, mainList: newMainList(models)}
	m.input = textinput.New()
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'l'}})
	m = next.(Model)
	next, _ = m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = next.(Model)
	if m.screen != ScreenModelConfig || m.replayRequest != nil || !m.replaySavedAt.IsZero() {
		t.Fatalf("Esc did not safely cancel replay into editable config: screen=%v replay=%#v", m.screen, m.replayRequest)
	}
}

func TestLatestLaunchMissingModelWarnsWithoutLeavingMenu(t *testing.T) {
	cacheDir := t.TempDir()
	if err := SaveLatestLaunch(cacheDir, &LaunchRequest{ModelPath: filepath.Join(t.TempDir(), "gone.gguf")}); err != nil {
		t.Fatal(err)
	}
	m := Model{screen: ScreenFirstRun, cacheDir: cacheDir, mainList: newMainList(nil)}
	m.input = textinput.New()
	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'l'}})
	m = next.(Model)
	if cmd != nil || m.screen != ScreenFirstRun || m.messageType != "warning" || !strings.Contains(m.message, "missing") {
		t.Fatalf("missing latest model handling: screen=%v type=%q msg=%q", m.screen, m.messageType, m.message)
	}
}
