package tui

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	modelstore "github.com/raketenkater/ggrun/pkg/models"
	"github.com/raketenkater/ggrun/pkg/modelusage"
	"github.com/raketenkater/ggrun/pkg/placement"
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

func TestBackendManagerListsRecipesAndInstalledForks(t *testing.T) {
	appHome := t.TempDir()
	t.Setenv("LLM_APP_HOME", appHome)
	if err := backends.Save([]backends.Backend{{Tag: "laguna", Path: "/tmp/laguna-server"}}); err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(backendManagerOptions(), "\n")
	for _, want := range []string{
		"Install reviewed: hy3",
		"Install reviewed: minimax-m3",
		"Install reviewed: laguna",
		"Add custom fork from Git URL",
		"Register an existing llama-server binary",
		"Remove installed: laguna",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("backend manager missing %q in %q", want, joined)
		}
	}
}

func TestBackendManagerRecipeReturnsCLIRequest(t *testing.T) {
	t.Setenv("LLM_APP_HOME", t.TempDir())
	m := Model{}
	m.input = textinput.New()
	m.openBackendManager(ScreenMain)
	for i, option := range m.choiceOptions {
		if option == "Install reviewed: laguna" {
			m.choiceCursor = i
			break
		}
	}
	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = next.(Model)
	if m.launchRequest == nil || strings.Join(m.launchRequest.BackendArgs, " ") != "install laguna" {
		t.Fatalf("unexpected backend request: %#v", m.launchRequest)
	}
	if cmd == nil {
		t.Fatal("backend action must quit the TUI so the CLI handler can run")
	}
}

func TestFirstRunExposesBackendManager(t *testing.T) {
	t.Setenv("LLM_APP_HOME", t.TempDir())
	m := Model{screen: ScreenFirstRun}
	m.input = textinput.New()
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'f'}})
	m = next.(Model)
	if m.screen != ScreenChoice || m.choiceTitle != "Backend forks" {
		t.Fatalf("first-run backend shortcut did not open manager: screen=%v title=%q", m.screen, m.choiceTitle)
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

func TestModelScanActionIsVisibleOnMainAndFirstRun(t *testing.T) {
	main := newMainList(nil)
	found := false
	for _, raw := range main.Items() {
		item, ok := raw.(mainItem)
		if ok && item.action == "scan" {
			found = true
			if !strings.Contains(item.title, "Scan computer") {
				t.Fatalf("scan action title = %q", item.title)
			}
		}
	}
	if !found {
		t.Fatal("main menu does not expose the computer model scan")
	}

	found = false
	for _, action := range firstRunActions() {
		if action == "scan" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("first-run menu does not expose the computer model scan")
	}
}

func TestModelScanCompletionMergesExternalModelAndLeavesFirstRun(t *testing.T) {
	primary := t.TempDir()
	externalDir := t.TempDir()
	externalPath := filepath.Join(externalDir, "found-Q4_K_M.gguf")
	if err := os.WriteFile(externalPath, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	m := Model{
		screen:         ScreenFirstRun,
		modelDir:       primary,
		cacheDir:       t.TempDir(),
		backend:        "auto",
		mainList:       newMainList(nil),
		scanningModels: true,
	}
	m.input = textinput.New()

	next, cmd := m.Update(modelScanFinishedMsg{result: modelstore.ScanResult{
		Paths:    []string{externalPath},
		Duration: 1250 * time.Millisecond,
	}})
	m = next.(Model)
	if cmd != nil {
		t.Fatal("scan completion should not quit or launch another command")
	}
	if m.scanningModels {
		t.Fatal("scan completion did not clear scanning state")
	}
	if m.screen != ScreenMain {
		t.Fatalf("screen = %v, want Main after finding a model", m.screen)
	}
	if len(m.models) != 1 || !m.models[0].External || m.models[0].Path != externalPath {
		t.Fatalf("recognized models = %#v", m.models)
	}
	if !strings.Contains(m.message, "1 outside") {
		t.Fatalf("completion message = %q", m.message)
	}
}

func TestModelScanCompletionPreservesActiveModelByPath(t *testing.T) {
	primary := t.TempDir()
	externalDir := t.TempDir()
	activePath := filepath.Join(primary, "zeta.gguf")
	newPath := filepath.Join(externalDir, "alpha.gguf")
	for _, path := range []string{activePath, newPath} {
		if err := os.WriteFile(path, []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	models := loadModels(primary, t.TempDir(), "auto", nil)
	m := Model{
		screen:        ScreenModelConfig,
		modelDir:      primary,
		cacheDir:      t.TempDir(),
		backend:       "auto",
		models:        models,
		selectedModel: 0,
		mainList:      newMainList(models),
	}
	m.mainList.Select(3)

	next, _ := m.Update(modelScanFinishedMsg{result: modelstore.ScanResult{Paths: []string{newPath}}})
	m = next.(Model)
	if got := m.models[m.selectedModel].Path; got != activePath {
		t.Fatalf("active model changed under an open config screen: got %q, want %q", got, activePath)
	}
	selected, ok := m.mainList.SelectedItem().(mainItem)
	if !ok || !selected.isModel || m.models[selected.index].Path != activePath {
		t.Fatalf("Main-list focus was not preserved on %q: %#v", activePath, selected)
	}
}

func TestLoadRecognizedModelsDeduplicatesPrimaryAndCachedPaths(t *testing.T) {
	primary := t.TempDir()
	externalDir := t.TempDir()
	cacheDir := t.TempDir()
	primaryPath := filepath.Join(primary, "primary.gguf")
	externalPath := filepath.Join(externalDir, "elsewhere.gguf")
	for _, path := range []string{primaryPath, externalPath} {
		if err := os.WriteFile(path, []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := modelstore.SaveDiscoveredPaths(cacheDir, []string{primaryPath, externalPath}); err != nil {
		t.Fatal(err)
	}

	models := loadRecognizedModels(primary, cacheDir, "auto", nil)
	if len(models) != 2 {
		t.Fatalf("recognized models = %#v, want two unique paths", models)
	}
	byPath := make(map[string]ModelItem)
	for _, model := range models {
		byPath[model.Path] = model
	}
	if byPath[primaryPath].External {
		t.Fatal("a cached path inside the primary directory must remain a primary model")
	}
	if !byPath[externalPath].External {
		t.Fatal("cached path outside the primary directory must be marked discovered")
	}
}

func TestExternalDiscoveredModelCannotBeDeletedWithoutTypedConfirmation(t *testing.T) {
	externalDir := t.TempDir()
	path := filepath.Join(externalDir, "external.gguf")
	if err := os.WriteFile(path, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	item, _, ok := modelItemFromPath(path, info, true)
	if !ok {
		t.Fatal("test external GGUF was not recognized")
	}
	dir := t.TempDir()
	m := Model{screen: ScreenMain, modelDir: dir, models: []ModelItem{item}, mainList: newMainList([]ModelItem{item})}
	m.input = textinput.New()

	// x on an external row must open the arrow-select offer (not a dead-end
	// warning), defaulting to Cancel.
	m.mainList.Select(3) // rows 0-2 are Recommended, Latest, and Scan computer
	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'x'}})
	m = nm.(Model)
	if m.screen != ScreenChoice {
		t.Fatalf("x on an external model should open the remove offer, screen=%v", m.screen)
	}
	if m.choiceOptions[m.choiceCursor] != "Cancel" {
		t.Fatalf("external remove offer must default to Cancel, got %q", m.choiceOptions[m.choiceCursor])
	}
	if !strings.Contains(m.choiceTitle, "outside the primary directory") {
		t.Fatalf("external remove offer must warn about the primary directory, got %q", m.choiceTitle)
	}

	// Advancing to "Remove file <name>" and confirming opens the typed prompt.
	for i, opt := range m.choiceOptions {
		if strings.HasPrefix(opt, "Remove file ") {
			m.choiceCursor = i
			break
		}
	}
	nm, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)
	if m.screen != ScreenDownload || m.inputMode != "external-remove-confirm" {
		t.Fatalf("confirming the offer must open the typed yes screen, screen=%v inputMode=%q", m.screen, m.inputMode)
	}
	if m.pendingExternalRemoveIdx != 0 {
		t.Fatalf("pending external remove index not recorded, got %d", m.pendingExternalRemoveIdx)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("the typed screen alone must not touch the file: %v", err)
	}

	// Entering anything other than "yes" on the typed screen cancels safely.
	m.input.SetValue("n")
	nm, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)
	if m.screen != ScreenMain || m.inputMode != "" {
		t.Fatalf("non-yes answer must return to Main with no pending removal, screen=%v inputMode=%q", m.screen, m.inputMode)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("non-yes answer must not remove the file: %v", err)
	}

	// The lower-level path is guarded too: removeModelAt refuses any external
	// row whose index was not handed down by the typed screen.
	m = Model{screen: ScreenMain, modelDir: dir, models: []ModelItem{item}}
	m.pendingExternalRemoveIdx = -1
	m.removeModelAt(0)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("ungated removeModelAt must not touch the file: %v", err)
	}
	if m.messageType != "warning" {
		t.Fatalf("ungated removeModelAt must warn, got type=%q msg=%q", m.messageType, m.message)
	}
}

func TestExternalDiscoveredModelDeletedAfterTypedConfirmation(t *testing.T) {
	externalDir := t.TempDir()
	path := filepath.Join(externalDir, "external.gguf")
	if err := os.WriteFile(path, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	item, _, ok := modelItemFromPath(path, info, true)
	if !ok {
		t.Fatal("test external GGUF was not recognized")
	}
	dir := t.TempDir()
	m := Model{screen: ScreenMain, modelDir: dir, models: []ModelItem{item}}
	m.input = textinput.New()

	m.openExternalRemoveConfirm(0)
	m.input.SetValue("yes")
	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)

	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("typed yes must remove the external file, stat err=%v", err)
	}
	if m.screen != ScreenMain || m.inputMode != "" {
		t.Fatalf("after external removal the TUI must be back on Main, screen=%v inputMode=%q", m.screen, m.inputMode)
	}
	if m.messageType != "info" || !strings.Contains(m.message, "Removed") {
		t.Fatalf("expected an info removal message, got type=%q msg=%q", m.messageType, m.message)
	}
	if m.pendingExternalRemoveIdx != -1 {
		t.Fatalf("pending external remove index must be cleared after removal, got %d", m.pendingExternalRemoveIdx)
	}
}

func TestExternalRemoveConfirmEscapedSymlinkDeletesRealTargetOnlyAfterConfirmation(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink permissions are not portable on Windows CI")
	}
	outsideDir := t.TempDir()
	realTarget := filepath.Join(outsideDir, "real-target.gguf")
	if err := os.WriteFile(realTarget, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	linkDir := t.TempDir()
	link := filepath.Join(linkDir, "link.gguf")
	if err := os.Symlink(realTarget, link); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(link)
	if err != nil {
		t.Fatal(err)
	}
	item, _, ok := modelItemFromPath(link, info, true)
	if !ok {
		t.Fatal("test symlinked GGUF was not recognized")
	}
	dir := t.TempDir()
	m := Model{screen: ScreenMain, modelDir: dir, models: []ModelItem{item}}
	m.input = textinput.New()
	m.pendingExternalRemoveIdx = -1

	// Without confirmation the symlink and its target must be untouched.
	m.removeModelAt(0)
	if _, err := os.Stat(realTarget); err != nil {
		t.Fatalf("ungated remove must not touch the symlink target: %v", err)
	}
	if _, err := os.Lstat(link); err != nil {
		t.Fatalf("ungated remove must not touch the symlink: %v", err)
	}

	// With typed confirmation the real target is deleted even though the row's
	// path is a symlink pointing outside the primary directory.
	m.pendingExternalRemoveIdx = 0
	m.openExternalRemoveConfirm(0)
	m.input.SetValue("yes")
	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)
	if _, err := os.Stat(realTarget); !os.IsNotExist(err) {
		t.Fatalf("typed yes must delete the symlink's real target, stat err=%v", err)
	}
	if _, err := os.Lstat(link); !os.IsNotExist(err) {
		t.Fatalf("typed yes must also remove the symlink itself, err=%v", err)
	}
	if m.messageType != "info" {
		t.Fatalf("expected an info removal message, got type=%q msg=%q", m.messageType, m.message)
	}
}

func TestDeleteModelKeyOpensConfirmDefaultedToCancel(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "test-Q4_K_M.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	models := discoverModels(dir)
	m := Model{screen: ScreenMain, modelDir: dir, models: models, mainList: newMainList(models)}
	m.input = textinput.New()
	m.mainList.Select(3) // rows 0-2 are Recommended, Latest, and Scan computer

	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'x'}})
	m = nm.(Model)
	if m.screen != ScreenChoice {
		t.Fatalf("expected the delete shortcut to open the confirm screen, got screen=%v", m.screen)
	}
	if !strings.Contains(m.choiceTitle, models[0].Name) {
		t.Fatalf("confirm title should name the model, got %q", m.choiceTitle)
	}
	if m.choiceOptions[m.choiceCursor] != "Cancel" {
		t.Fatalf("confirm cursor must default to Cancel, got %q", m.choiceOptions[m.choiceCursor])
	}
}

func TestDeleteModelConfirmRemovesFileWithoutQuitting(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test-Q4_K_M.gguf")
	if err := os.WriteFile(path, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	models := discoverModels(dir)
	m := Model{screen: ScreenMain, modelDir: dir, models: models, mainList: newMainList(models)}
	m.input = textinput.New()
	m.mainList.Select(3)

	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'x'}})
	m = nm.(Model)
	for i, opt := range m.choiceOptions {
		if strings.HasPrefix(opt, "Confirm: delete ") {
			m.choiceCursor = i
			break
		}
	}
	nm, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)

	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("expected the model file to be removed from disk, stat err=%v", err)
	}
	if len(m.models) != 0 {
		t.Fatalf("expected the Main model list to refresh after removal, got %d models", len(m.models))
	}
	if m.screen != ScreenMain {
		t.Fatalf("expected to return to the Main screen after removal, got %v", m.screen)
	}
	if m.messageType != "info" || !strings.Contains(m.message, "Removed") {
		t.Fatalf("expected an info message confirming removal, got type=%q msg=%q", m.messageType, m.message)
	}
	if cmd != nil {
		t.Fatal("model removal must not quit the TUI — unlike backend removal it needs no external process")
	}
}

func TestDeleteModelCancelLeavesFileInPlace(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test-Q4_K_M.gguf")
	if err := os.WriteFile(path, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	models := discoverModels(dir)
	m := Model{screen: ScreenMain, modelDir: dir, models: models, mainList: newMainList(models)}
	m.input = textinput.New()
	m.mainList.Select(3)

	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'x'}})
	m = nm.(Model)
	// Cursor already defaults to "Cancel".
	nm, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)

	if _, err := os.Stat(path); err != nil {
		t.Fatalf("Cancel must leave the model file in place, stat err=%v", err)
	}
	if len(m.models) != 1 {
		t.Fatalf("Cancel must not touch the Main model list, got %d models", len(m.models))
	}
}

func TestRemoveModelAtDisambiguatesSameBasenameByPath(t *testing.T) {
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
	models := discoverModels(dir)
	if len(models) != 2 {
		t.Fatalf("setup: expected 2 same-basename models, got %d", len(models))
	}

	m := Model{modelDir: dir, models: models}
	m.removeModelAt(0)

	if _, err := os.Stat(models[0].Path); !os.IsNotExist(err) {
		t.Fatalf("expected the selected model's file to be removed, stat err=%v", err)
	}
	if _, err := os.Stat(models[1].Path); err != nil {
		t.Fatalf("expected the other same-basename model to survive, stat err=%v", err)
	}
}

// TestRebuildMainListPreservesWindowSize guards against the list snapping
// back to newMainList's 40x20 placeholder size (and splitting the screen)
// whenever the model set changes mid-session — e.g. after a delete, a model
// directory change, or a tuned-count refresh, all of which rebuild the list
// well after the initial WindowSizeMsg has already sized it.
func TestRebuildMainListPreservesWindowSize(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "test-Q4_K_M.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	models := discoverModels(dir)
	m := &Model{modelDir: dir, models: models, mainList: newMainList(models), width: 120, height: 40}

	m.rebuildMainList()

	wantW, wantH := 120-4, 40-12
	if got := m.mainList.Width(); got != wantW {
		t.Fatalf("mainList.Width() = %d, want %d (fell back to the 40x20 placeholder?)", got, wantW)
	}
	if got := m.mainList.Height(); got != wantH {
		t.Fatalf("mainList.Height() = %d, want %d (fell back to the 40x20 placeholder?)", got, wantH)
	}
}

// TestRebuildMainListClampsIndexAgainstFilteredNotRawCount guards against a
// crash found by the second audit pass: rebuildMainList used to clamp the
// restored cursor against the RAW (unfiltered) item count. If a filter was
// active, scrolled deep, and the rebuild's filtered match count shrinks a lot
// while the raw count stays large enough to dodge that clamp, list.Select()
// lands the paginator past the real number of filtered pages and the next
// View() panics with a slice-bounds-out-of-range, killing the whole TUI.
func TestRebuildMainListClampsIndexAgainstFilteredNotRawCount(t *testing.T) {
	dir := t.TempDir()
	for i := 0; i < 20; i++ {
		name := fmt.Sprintf("match-%02d-Q4_K_M.gguf", i)
		if err := os.WriteFile(filepath.Join(dir, name), []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	models := discoverModels(dir)
	if len(models) != 20 {
		t.Fatalf("setup: expected 20 models, got %d", len(models))
	}

	m := &Model{models: models, mainList: newMainList(models), width: 120, height: 40}
	m.mainList.SetFilterText("match") // matches all 20
	m.mainList.Select(15)             // deep into a filtered page

	// New model set: raw count stays large (20, dodging a raw-count clamp),
	// but only 2 of them still match "match" - the filtered count crashes
	// down to 2 while the raw count doesn't.
	newDir := t.TempDir()
	for i := 0; i < 2; i++ {
		name := fmt.Sprintf("match-%02d-Q4_K_M.gguf", i)
		if err := os.WriteFile(filepath.Join(newDir, name), []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 18; i++ {
		name := fmt.Sprintf("other-%02d-Q4_K_M.gguf", i)
		if err := os.WriteFile(filepath.Join(newDir, name), []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	m.models = discoverModels(newDir)
	if len(m.models) != 20 {
		t.Fatalf("setup: expected 20 new models, got %d", len(m.models))
	}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("rebuildMainList + View panicked (index clamped against the wrong count): %v", r)
		}
	}()
	m.rebuildMainList()
	_ = m.mainList.View()

	if got := len(m.mainList.VisibleItems()); got != 2 {
		t.Fatalf("expected the reapplied filter to match 2 items, got %d", got)
	}
	if idx := m.mainList.Index(); idx >= 2 {
		t.Fatalf("Index() = %d, want < 2 (clamped against the filtered count)", idx)
	}
}

func TestEscCancelsInputModeWithoutLeavingScreen(t *testing.T) {
	m := Model{screen: ScreenModelConfig, inputMode: "download"}
	m.input = textinput.New()
	m.input.Focus()

	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = nm.(Model)

	if m.inputMode != "" {
		t.Fatalf("expected esc to clear inputMode, got %q", m.inputMode)
	}
	if m.screen != ScreenModelConfig {
		t.Fatalf("expected esc to cancel the input in place, not navigate; got screen=%v", m.screen)
	}
	if m.input.Focused() {
		t.Fatal("expected the stale text input to be blurred after esc")
	}
}

// TestEscOnDownloadScreenReturnsToMainInOnePress guards against a regression
// found by the second audit pass: ScreenDownload/ScreenBackend have no local
// "esc" case of their own (unlike ModelConfig/Settings, which have a row menu
// to fall back into on the same screen), so clearing inputMode alone left the
// user stuck on a blurred, unresponsive "Input" screen until a second Esc.
func TestEscOnDownloadScreenReturnsToMainInOnePress(t *testing.T) {
	m := Model{screen: ScreenDownload, inputMode: "download"}
	m.input = textinput.New()
	m.input.Focus()

	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = nm.(Model)

	if m.inputMode != "" {
		t.Fatalf("expected esc to clear inputMode, got %q", m.inputMode)
	}
	if m.screen != ScreenMain {
		t.Fatalf("expected a single esc to leave ScreenDownload (no on-screen fallback menu), got screen=%v", m.screen)
	}
}

func TestEscOnPrelaunchReturnsToModelConfigNotMain(t *testing.T) {
	m := Model{screen: ScreenPrelaunch, models: []ModelItem{{Name: "test.gguf"}}}
	m.input = textinput.New()

	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = nm.(Model)

	if m.screen != ScreenModelConfig {
		t.Fatalf("expected esc on Prelaunch to go back to ModelConfig (its own case), got screen=%v", m.screen)
	}
}

func TestEscOnRecommendedClearsHeadroomFocusWithoutLeaving(t *testing.T) {
	m := Model{screen: ScreenRecommended, recHeadroomFocus: "vram", message: "Saved: VRAM reserve = 24G"}
	m.input = textinput.New()

	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = nm.(Model)

	if m.recHeadroomFocus != "" {
		t.Fatalf("expected esc to defocus the headroom control, got recHeadroomFocus=%q", m.recHeadroomFocus)
	}
	if m.screen != ScreenRecommended {
		t.Fatalf("expected the first esc to stay on Recommended (defocus only), got screen=%v", m.screen)
	}
}

func TestOpenTunedPickerRestoresIndexForActiveConfig(t *testing.T) {
	dir := t.TempDir()
	cacheDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "test-Q4_K_M.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	models := discoverModels(dir)

	// ListTunedConfigs (pkg/tune) globs cacheDir for tune_<modelName>_*.json,
	// requires doc.Model == modelName and a positive best_config.gen_tps, and
	// infers backend "llama" / non-vision from the absence of "_ik"/"_vulkan"/
	// "_v_" in the filename — backendTag() defaults to "llama" too, so these
	// two fixture files match what Model{backend: ""} will actually look up.
	writeTunedConfig := func(suffix string, genTPS float64) string {
		path := filepath.Join(cacheDir, fmt.Sprintf("tune_%s_%s.json", models[0].Name, suffix))
		doc := fmt.Sprintf(`{"model":%q,"baseline_gen_tps":10,"best_config":{"gen_tps":%f,"flags":{}},"rounds":5,"tuned_at":"2026-07-01T00:00:00Z"}`, models[0].Name, genTPS)
		if err := os.WriteFile(path, []byte(doc), 0o644); err != nil {
			t.Fatal(err)
		}
		return path
	}
	writeTunedConfig("variant-a", 20)
	activePath := writeTunedConfig("variant-b", 30)

	m := &Model{cacheDir: cacheDir, models: models, selectedModel: 0, tunePath: activePath}
	m.openTunedPicker()

	if m.screen != ScreenTunedPicker {
		t.Fatalf("expected openTunedPicker to switch to ScreenTunedPicker, got %v", m.screen)
	}
	if m.tunedIndex < 0 || m.tunedIndex >= len(m.tunedConfigs) {
		t.Fatalf("expected tunedIndex to point at the active config, got %d (len=%d)", m.tunedIndex, len(m.tunedConfigs))
	}
	if got := m.tunedConfigs[m.tunedIndex].Path; got != activePath {
		t.Fatalf("expected the restored index to match the active tuned config; got %q, want %q", got, activePath)
	}
}

// TestOpenTunedPickerFallsBackToAutoForStaleTunePath covers the case the
// happy-path test above doesn't: m.tunePath pointing at a config that no
// longer exists in the freshly-rebuilt list (e.g. its file was deleted out
// from under it). openTunedPicker must fall back to -1 ("Auto"), not leave a
// stale non-matching index in place.
func TestOpenTunedPickerFallsBackToAutoForStaleTunePath(t *testing.T) {
	dir := t.TempDir()
	cacheDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "test-Q4_K_M.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	models := discoverModels(dir)

	m := &Model{
		cacheDir:      cacheDir,
		models:        models,
		selectedModel: 0,
		tunePath:      filepath.Join(cacheDir, "tune_"+models[0].Name+"_deleted-config.json"),
	}
	m.openTunedPicker()

	if m.tunedIndex != -1 {
		t.Fatalf("expected tunedIndex to fall back to -1 (Auto) for a tunePath with no matching config, got %d", m.tunedIndex)
	}
}

func TestDiscoverModelsHidesAuxiliaryArtifacts(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{
		"target-Q4_K_M.gguf",
		"target-mmproj.gguf",
		"target-DFlash-BF16.gguf",
		"target-MTP-Q8_0.gguf",
		"target-draft.gguf",
		"target-speculator.gguf",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	models := discoverModels(dir)
	if len(models) != 1 || models[0].Name != "target-Q4_K_M.gguf" {
		t.Fatalf("expected only target model, got %#v", models)
	}
}

func TestDiscoverModelsHidesIncompleteShardSet(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{
		"complete.gguf",
		"partial-00001-of-00003.gguf",
		"partial-00002-of-00003.gguf",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	models := discoverModels(dir)
	if len(models) != 1 || models[0].Name != "complete.gguf" {
		t.Fatalf("incomplete sharded download was presented as runnable: %#v", models)
	}
}

func TestAuxiliaryArchitectureIsHiddenWithNeutralFilename(t *testing.T) {
	if !isAuxiliaryModel("small-helper.gguf", "dflash") {
		t.Fatal("dflash architecture must be hidden even without a filename marker")
	}
	if isAuxiliaryModel("target-Q4_K_M.gguf", "laguna") {
		t.Fatal("target model must stay visible")
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

func TestBuildLaunchRequestUsesStandardFitPolicy(t *testing.T) {
	m := Model{
		models:        []ModelItem{{Name: "Laguna", Path: "/models/laguna.gguf"}},
		selectedModel: 0,
		backend:       "auto",
		ctxMode:       "fit",
		ctxSize:       "fit",
		kvPlacement:   "gpu",
		kvQuality:     "q4_0",
	}
	req := m.buildLaunchRequest()
	if req == nil || req.CtxFlag != "fit" || req.CtxSize != 0 {
		t.Fatalf("fit mode must reach the standard placement ladder unchanged: %#v", req)
	}
	joined := strings.Join(req.LaunchArgs(), " ")
	if !strings.Contains(joined, "--ctx-size fit") || strings.Contains(joined, "132222") {
		t.Fatalf("TUI emitted an ad-hoc context instead of fit: %q", joined)
	}
}

func TestBuildLaunchRequestAutoSelectsInstalledModelRoute(t *testing.T) {
	m := Model{
		models: []ModelItem{{
			Name:         "MiniMax-M3.gguf",
			Path:         "/models/minimax.gguf",
			Architecture: "minimax-m3",
			AutoBackend:  "minimax-m3",
		}},
		selectedModel: 0,
		backend:       "ik_llama",
		kvPlacement:   "auto",
		ctxMode:       "fit",
	}
	req := m.buildLaunchRequest()
	if req == nil || req.Backend != "minimax-m3" {
		t.Fatalf("installed architecture route was not selected over the default backend: %#v", req)
	}
	if view := m.viewModelConfig(); !strings.Contains(view, "auto-selected for minimax-m3") {
		t.Fatalf("model config does not disclose its effective backend: %q", view)
	}
}

func TestMissingModelBackendRecipeConfirmsInstallAndCarriesLaunch(t *testing.T) {
	m := Model{
		screen: ScreenModelConfig,
		models: []ModelItem{{
			Name:          "MiniMax-M3.gguf",
			Path:          "/models/minimax.gguf",
			Architecture:  "minimax-m3",
			Arch:          "minimax-m3 · MoE",
			BackendRecipe: "minimax-m3",
		}},
		selectedModel: 0,
		backend:       "ik_llama",
		kvPlacement:   "auto",
		kvQuality:     "high",
		ctxMode:       "fit",
	}
	m.input = textinput.New()

	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'L'}})
	m = next.(Model)
	if cmd != nil || m.screen != ScreenChoice || m.choiceCursor != 0 {
		t.Fatalf("launch must stop at a cancel-first install confirmation: screen=%v cursor=%d cmd=%v", m.screen, m.choiceCursor, cmd)
	}
	joined := strings.Join(m.choiceOptions, "\n")
	if !strings.Contains(joined, "Install minimax-m3 and select it") || !strings.Contains(joined, "Continue once with ik_llama") {
		t.Fatalf("backend confirmation is incomplete: %q", joined)
	}

	m.choiceCursor = 1
	next, cmd = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = next.(Model)
	if cmd == nil {
		t.Fatal("confirmed backend installation must quit to the CLI installer")
	}
	if m.launchRequest == nil || strings.Join(m.launchRequest.BackendArgs, " ") != "install minimax-m3" {
		t.Fatalf("unexpected backend install request: %#v", m.launchRequest)
	}
	if m.launchRequest.ModelPath != "/models/minimax.gguf" || m.launchRequest.Backend != "minimax-m3" || m.launchRequest.KVQuality != "high" {
		t.Fatalf("post-install review did not retain model settings and selected recipe: %#v", m.launchRequest)
	}
}

func TestMissingModelBackendRecipeCanBeBypassedOnce(t *testing.T) {
	m := Model{
		screen: ScreenModelConfig,
		models: []ModelItem{{
			Name:          "MiniMax-M3.gguf",
			Path:          "/models/minimax.gguf",
			Architecture:  "minimax-m3",
			BackendRecipe: "minimax-m3",
		}},
		selectedModel: 0,
		backend:       "ik_llama",
		ctxMode:       "fit",
	}
	m.input = textinput.New()
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'i'}})
	m = next.(Model)
	m.choiceCursor = 2
	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = next.(Model)
	if cmd != nil || m.screen != ScreenPrelaunch || !m.backendRouteBypass {
		t.Fatalf("continue-once did not reach pre-launch: screen=%v bypass=%v cmd=%v", m.screen, m.backendRouteBypass, cmd)
	}
	if m.effectiveBackend() != "ik_llama" || m.selectedBackendRecipe() != nil {
		t.Fatalf("continue-once did not preserve the selected backend for this attempt: backend=%q recipe=%#v", m.effectiveBackend(), m.selectedBackendRecipe())
	}
	if view := m.viewPrelaunch(); !strings.Contains(view, "model may fail") {
		t.Fatalf("pre-launch did not disclose the unsupported-backend bypass: %q", view)
	}
}

func TestMissingIKOnlyRecipeBypassOverridesAutoBackend(t *testing.T) {
	m := Model{
		screen: ScreenModelConfig,
		models: []ModelItem{{
			Name:          "MiniMax-M3.gguf",
			Path:          "/models/minimax.gguf",
			Architecture:  "minimax-m3",
			Arch:          "minimax-m3 · MoE",
			BackendRecipe: "minimax-m3",
		}},
		selectedModel: 0,
		backend:       "auto",
		ctxMode:       "fit",
	}
	m.input = textinput.New()
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'i'}})
	m = next.(Model)
	if got := strings.Join(m.choiceOptions, "\n"); !strings.Contains(got, "Continue once with ik_llama") {
		t.Fatalf("IK-only architecture offered an incompatible auto bypass: %q", got)
	}
	m.choiceCursor = 2
	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = next.(Model)
	if cmd != nil || m.effectiveBackend() != "ik_llama" {
		t.Fatalf("IK-only bypass backend = %q, cmd=%v; want ik_llama", m.effectiveBackend(), cmd)
	}
	if req := m.buildLaunchRequest(); req == nil || req.Backend != "ik_llama" {
		t.Fatalf("IK-only bypass launch request = %#v, want explicit ik_llama", req)
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

func TestLaunchArgsCarriesExplicitSWAChoice(t *testing.T) {
	on := &LaunchRequest{ModelPath: "/models/test.gguf", SWAFull: true, SWAFullSet: true}
	if joined := strings.Join(on.LaunchArgs(), " "); !strings.Contains(joined, "--swa-full") || strings.Contains(joined, "--no-swa-full") {
		t.Fatalf("enabled full SWA choice was not emitted correctly: %q", joined)
	}
	off := &LaunchRequest{ModelPath: "/models/test.gguf", SWAFullSet: true}
	if joined := strings.Join(off.LaunchArgs(), " "); !strings.Contains(joined, "--no-swa-full") || strings.Contains(joined, " --swa-full") {
		t.Fatalf("disabled full SWA choice was not emitted correctly: %q", joined)
	}
	inherit := &LaunchRequest{ModelPath: "/models/test.gguf"}
	if joined := strings.Join(inherit.LaunchArgs(), " "); strings.Contains(joined, "swa-full") {
		t.Fatalf("unset legacy request unexpectedly overrode SWA policy: %q", joined)
	}
}

func TestLaunchArgsCarriesExactSupportPolicy(t *testing.T) {
	req := &LaunchRequest{
		ModelPath:     "/models/test.gguf",
		SupportExpert: "auto",
		SupportOnline: true,
		SupportSet:    true,
	}
	joined := strings.Join(req.LaunchArgs(), " ")
	if !strings.Contains(joined, "--support-expert auto") || !strings.Contains(joined, "--support-online") || strings.Contains(joined, "--no-support-online") {
		t.Fatalf("support policy was not emitted exactly: %q", joined)
	}
	req.SupportOnline = false
	joined = strings.Join(req.LaunchArgs(), " ")
	if !strings.Contains(joined, "--no-support-online") || strings.Contains(joined, " --support-online") {
		t.Fatalf("disabled online policy was not emitted exactly: %q", joined)
	}
}

func TestModelConfigTogglesAndDisplaysFullSWA(t *testing.T) {
	m := Model{
		models:        []ModelItem{{Name: "Laguna", Path: "/models/laguna.gguf"}},
		selectedModel: 0,
		backend:       "auto",
		ctxMode:       "manual",
		ctxSize:       "131072",
		kvPlacement:   "gpu",
		kvQuality:     "q4_0",
	}
	if view := m.viewModelConfig(); !strings.Contains(view, "Full SWA cache") || !strings.Contains(view, "off (smaller KV cache)") {
		t.Fatalf("disabled full SWA state is missing from model config: %q", view)
	}
	m.cycleCfgRow("swa", 1)
	if view := m.viewModelConfig(); !strings.Contains(view, "on (more cache hits; memory estimate after dry run)") {
		t.Fatalf("enabled full SWA state is missing from model config: %q", view)
	}
	if req := m.buildLaunchRequest(); req == nil || !req.SWAFull || !req.SWAFullSet {
		t.Fatalf("full SWA toggle did not reach launch request: %#v", req)
	}
}

func TestModelConfigShowsFullSWAKVMemoryDelta(t *testing.T) {
	m := Model{
		models: []ModelItem{{
			Name: "windowed.gguf",
			KVProfile: &placement.ModelProfile{
				NumLayers:     48,
				HeadCountKV:   8,
				KeyLength:     128,
				ValueLength:   128,
				SlidingWindow: 4096,
				ModelArch:     "laguna",
			},
		}},
		ctxSize:   "131072",
		ctxMode:   "manual",
		kvQuality: "q4_0",
	}
	if delta := m.swaExtraKVMB(m.models[0]); delta <= 0 {
		t.Fatalf("Full SWA delta = %d MiB, want a positive estimate", delta)
	}
	view := m.viewModelConfig()
	if !strings.Contains(view, "Full SWA cache") || !strings.Contains(view, "+") || !strings.Contains(view, "GiB KV") {
		t.Fatalf("Full SWA row does not show its KV cost: %s", view)
	}
	m.swaFull = true
	if view := m.viewPrelaunch(); !strings.Contains(view, "more cache hits") || !strings.Contains(view, "GiB KV") {
		t.Fatalf("pre-launch summary lost Full SWA cost: %s", view)
	}
}

// TestConfigMemoCachedMatchesUncachedAndInvalidates verifies the config-screen
// memo: the memoized path (used by every render once the memo is seeded) returns
// exactly the same Chat-template label and Full-SWA KV extra-MiB as the direct,
// uncached path for the same input, that an unchanged-config render reuses the
// cache instead of recomputing, and that setCtx / cycleCfgRow invalidate the
// cache so the next render recomputes with the new value.
func TestConfigMemoCachedMatchesUncachedAndInvalidates(t *testing.T) {
	model := ModelItem{
		Name:         "nanbeige.gguf",
		Path:         "/models/nanbeige.gguf",
		Architecture: "nanbeige",
		// A known-broken embedded template (carries the raise_exception guard) so
		// the catalog auto-fix resolves to a real entry rather than just "auto".
		ChatTemplate: "{% if raise_exception %}{% endif %}",
		KVProfile: &placement.ModelProfile{
			NumLayers:     48,
			HeadCountKV:   8,
			KeyLength:     128,
			ValueLength:   128,
			SlidingWindow: 4096,
			ModelArch:     "laguna",
		},
	}
	base := func(memo *configMemo) Model {
		return Model{
			cfgMemo:       memo,
			models:        []ModelItem{model},
			selectedModel: 0,
			ctxMode:       "manual",
			ctxSize:       "131072",
			kvQuality:     "q4_0",
		}
	}
	cached := base(&configMemo{})
	uncached := base(nil)

	// 1) Cached path returns the same values as the uncached path.
	if got, want := cached.chatTemplateLabel(model), uncached.chatTemplateLabel(model); got != want {
		t.Fatalf("memoized template %q != direct %q", got, want)
	}
	if got, want := cached.swaExtraKVMB(model), uncached.swaExtraKVMB(model); got != want {
		t.Fatalf("memoized KV %d != direct %d", got, want)
	}
	if cached.swaExtraKVMB(model) <= 0 {
		t.Fatalf("Full SWA delta should be positive for a windowed model, got %d", cached.swaExtraKVMB(model))
	}
	if !strings.Contains(cached.chatTemplateLabel(model), "auto (") {
		t.Fatalf("auto chat template was not resolved: %q", cached.chatTemplateLabel(model))
	}

	// 2) An unchanged-config render reuses the memo (O(1)); values stay stable.
	firstTemplate, firstKV := cached.chatTemplateLabel(model), cached.swaExtraKVMB(model)
	if cached.cfgMemo.key == "" {
		t.Fatalf("memo was not populated on first render")
	}
	if got, want := cached.chatTemplateLabel(model), firstTemplate; got != want {
		t.Fatalf("unchanged-template render changed: %q vs %q", got, want)
	}
	if got, want := cached.swaExtraKVMB(model), firstKV; got != want {
		t.Fatalf("unchanged-config render recomputed KV: %d vs %d", got, want)
	}

	// 3) setCtx invalidates the memo and recomputes with the new context.
	cached.setCtx("65536")
	if cached.cfgMemo.key != "" {
		t.Fatalf("setCtx did not invalidate the config memo")
	}
	uncached.ctxSize = "65536"
	if got, want := cached.swaExtraKVMB(model), uncached.swaExtraKVMB(model); got != want {
		t.Fatalf("after setCtx, memoized KV %d != direct %d", got, want)
	}
	if cached.cfgMemo.key == "" {
		t.Fatalf("render after setCtx did not repopulate the memo")
	}

	// 4) cycleCfgRow invalidates the memo and recomputes with the new KV quality.
	cached.cycleCfgRow("kvq", 1)
	if cached.cfgMemo.key != "" {
		t.Fatalf("cycleCfgRow did not invalidate the config memo")
	}
	if cached.kvQuality == "q4_0" {
		t.Fatalf("cycleCfgRow did not change kvQuality")
	}
	uncached.kvQuality = cached.kvQuality
	if got, want := cached.swaExtraKVMB(model), uncached.swaExtraKVMB(model); got != want {
		t.Fatalf("after cycleCfgRow, memoized KV %d != direct %d", got, want)
	}
}

func TestClaudeProfileSelectorCyclesPerLaunchOptions(t *testing.T) {
	m := Model{}
	if indexOf(m.cfgRows(), "claudeprofile") >= 0 {
		t.Fatal("Claude profile must stay hidden until Claude Code is enabled")
	}

	m.claudeCode = true
	if indexOf(m.cfgRows(), "claudeprofile") < 0 {
		t.Fatal("Claude profile selector missing while Claude Code is enabled")
	}
	for _, want := range []string{"agent-interactive", "agent-parallel", ""} {
		m.cycleCfgRow("claudeprofile", 1)
		if m.claudeProfile != want {
			t.Fatalf("profile=%q, want %q", m.claudeProfile, want)
		}
	}
}

// denseTestModel builds a dense (non-MoE) model whose KV cache grows ~8.5 GiB at
// 131072 context on q8_0: 32 layers, 8 KV heads, 128+128 dims →
// 32*8*256*1.0625 B/token. At 250000 that is ~16 GiB of KV, enough to spill off
// a single 30 GiB-free GPU once the ~20 GiB weights are counted.
func denseTestModel() ModelItem {
	return ModelItem{
		Name:   "Dense.gguf",
		Path:   "/models/dense.gguf",
		MaxCtx: 262144,
		KVProfile: &placement.ModelProfile{
			TotalSizeMB: 20000,
			SizeBytes:   20000 * 1048576,
			NumLayers:   32,
			HeadCountKV: 8,
			KeyLength:   128,
			ValueLength: 128,
			ModelArch:   "llama3",
		},
	}
}

func TestPrelaunchShowsKVContextSuggestionForDenseModel(t *testing.T) {
	m := Model{
		screen:        ScreenPrelaunch,
		models:        []ModelItem{denseTestModel()},
		selectedModel: 0,
		kvPlacement:   "gpu",
		kvQuality:     "q8_0",
		ctxMode:       "manual",
		ctxSize:       "250000",
		backend:       "llama",
		caps: &detect.Capabilities{
			GPUs: []detect.GPU{{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 30000}},
			RAM:  detect.RAMInfo{TotalMB: 128000, FreeMB: 120000},
			CPU:  detect.CPUInfo{Cores: 8},
		},
		cacheDir: "", // no system probe → zero CUDA overhead
	}
	view := m.viewPrelaunch()
	for _, want := range []string{
		"would spill",
		"ctx 131072 → ~",
		"ctx 65536 → ~",
		"ctx 32768 → ~",
		"[c] use ctx 131072",
		"fits all GPU",
	} {
		if !strings.Contains(view, want) {
			t.Fatalf("pre-launch KV/ctx suggestion missing %q in:\n%s", want, view)
		}
	}
	// The suggestion must be actionable: pressing c applies the suggested
	// context as a manual context size.
	hint := m.denseKVHint(m.models[0])
	if hint == nil || hint.suggestCtx != 131072 {
		t.Fatalf("dense KV hint should suggest ctx 131072, got %#v", hint)
	}
	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'c'}})
	m = nm.(Model)
	if m.ctxMode != "manual" || m.ctxSize != "131072" {
		t.Fatalf("applying the suggested context: ctx=%q/%q, want manual/131072", m.ctxMode, m.ctxSize)
	}
}

func TestPrelaunchHidesKVContextSuggestionWhenDenseFitsAllGPU(t *testing.T) {
	m := Model{
		models:        []ModelItem{denseTestModel()},
		selectedModel: 0,
		kvPlacement:   "gpu",
		kvQuality:     "q8_0",
		ctxMode:       "manual",
		ctxSize:       "131072", // fits: 20000 + 1024 + ~8.5 GiB <= 30000
		backend:       "llama",
		caps: &detect.Capabilities{
			GPUs: []detect.GPU{{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 30000}},
			RAM:  detect.RAMInfo{TotalMB: 128000, FreeMB: 120000},
			CPU:  detect.CPUInfo{Cores: 8},
		},
		cacheDir: "",
	}
	if hint := m.denseKVHint(m.models[0]); hint != nil {
		t.Fatalf("dense model that fits all-GPU must not get a suggestion, got %#v", hint)
	}
	view := m.viewPrelaunch()
	if strings.Contains(view, "would spill") {
		t.Fatalf("pre-launch must not show a suggestion when the model fits all-GPU:\n%s", view)
	}
}

func TestClaudeReviewerLabelPresentsProfilesByTradeoff(t *testing.T) {
	for reviewer, wantParts := range map[string][]string{
		"nanbeige": {"big/fast worker", "Nanbeige4.2", "reviews + works"},
		"qwen":     {"small/light", "Qwen3.5-4B", "worker+reviewer"},
		"auto":     {"auto", "automatic"},
	} {
		label := claudeReviewerLabel(reviewer)
		for _, part := range wantParts {
			if !strings.Contains(label, part) {
				t.Fatalf("reviewer %q label %q missing %q", reviewer, label, part)
			}
		}
		// The engine resolves both "nanbeige" and the historical "qwen" to
		// worker/reviewer models (main.go, claudeauto.go); only qwen2b is
		// review-only, so neither dual-role profile may carry that label.
		if (reviewer == "nanbeige" || reviewer == "qwen") && strings.Contains(label, "review-only") {
			t.Fatalf("%q profile serves worker+reviewer and must not be labelled review-only: %q", reviewer, label)
		}
	}
}

// A bare "ggrun" under cron/CI has no controlling terminal. The caller only
// prints "Error: %v", so the message itself must carry the subcommand guidance
// instead of bubbletea's raw /dev/tty open failure.
func TestRunWithoutTerminalReturnsGuidanceNotRawTTYError(t *testing.T) {
	if terminalAvailable() {
		t.Skip("test environment has a controlling terminal; the no-TTY path is unreachable")
	}
	req, err := Run()
	if req != nil {
		t.Fatalf("no-TTY run returned a launch request %+v", req)
	}
	if err == nil {
		t.Fatal("no-TTY run returned nil error")
	}
	msg := err.Error()
	if strings.Contains(msg, "/dev/tty") {
		t.Errorf("error leaks the raw TTY open failure: %q", msg)
	}
	for _, want := range []string{"no terminal available", "ggrun recommend", "ggrun models list", "ggrun <model.gguf>"} {
		if !strings.Contains(msg, want) {
			t.Errorf("no-TTY error %q missing guidance %q", msg, want)
		}
	}
}

func TestClaudeReviewerSelectorStillCyclesAndReachesLaunch(t *testing.T) {
	m := Model{
		models:        []ModelItem{{Name: "DeepSeek", Path: "/models/deepseek.gguf"}},
		selectedModel: 0,
		kvPlacement:   "auto",
		ctxMode:       "fit",
		ctxSize:       "fit",
		claudeCode:    true,
	}
	// Cycle from the empty (auto) default: auto -> qwen -> qwen2b -> nanbeige -> auto.
	for _, want := range []string{"auto", "qwen", "qwen2b", "nanbeige"} {
		m.cycleCfgRow("claudereviewer", 1)
		if m.claudeReviewer != want {
			t.Fatalf("reviewer=%q, want %q", m.claudeReviewer, want)
		}
	}
	// The explicit pick reaches the launch request unchanged (qwen/nanbeige/auto).
	for _, want := range []string{"qwen", "nanbeige", "auto"} {
		m.claudeReviewer = want
		req := m.buildLaunchRequest()
		if req == nil || req.ClaudeReviewerOverride != want {
			t.Fatalf("reviewer %q did not reach launch request: %#v", want, req)
		}
	}
}

func TestKVQualityRowCyclesAndMarksTouched(t *testing.T) {
	m := Model{
		models:        []ModelItem{{Name: "Laguna", Path: "/models/laguna.gguf"}},
		selectedModel: 0,
		backend:       "auto",
		kvPlacement:   "auto",
		kvQuality:     "auto",
	}
	if indexOf(m.cfgRows(), "kvq") < 0 {
		t.Fatal("KV quality row missing from the Advanced config screen")
	}

	// Cycling the row advances through the option list and is a deliberate
	// per-launch choice, so the launch request must emit it explicitly.
	m.cycleCfgRow("kvq", 1)
	if m.kvQuality != "high" || !m.kvQualityTouched {
		t.Fatalf("next KV quality = %q touched=%v, want high/true", m.kvQuality, m.kvQualityTouched)
	}
	req := m.buildLaunchRequest()
	if req == nil || !req.KVQualitySet || req.KVQuality != "high" {
		t.Fatalf("cycled KV quality did not reach launch request: %#v", req)
	}
	joined := strings.Join(req.LaunchArgs(), " ")
	if !strings.Contains(joined, "--kv-quality high") {
		t.Fatalf("touched KV quality must be an explicit launch flag: %q", joined)
	}

	// Cycling left (dir -1) walks back toward auto.
	m.cycleCfgRow("kvq", -1)
	if m.kvQuality != "auto" {
		t.Fatalf("prev KV quality = %q, want auto", m.kvQuality)
	}
}

func TestLaunchArgsCarriesInheritedKVQualityWithoutFlag(t *testing.T) {
	// A setting inherited from config must still reach the request value, but an
	// untouched launch must not type --kv-quality: ggrun treats a typed flag as
	// inviolable and would lock the recovery ladder out of withdrawing a quality
	// that no longer fits. Mirror the --swa-full provenance rule.
	m := Model{
		models:        []ModelItem{{Name: "Laguna", Path: "/models/laguna.gguf"}},
		selectedModel: 0,
		backend:       "auto",
		kvPlacement:   "auto",
		kvQuality:     "q4_0",
	}
	req := m.buildLaunchRequest()
	if req == nil || req.KVQuality != "q4_0" || req.KVQualitySet {
		t.Fatalf("inherited KV quality must reach the value without being marked explicit: %#v", req)
	}
	joined := strings.Join(req.LaunchArgs(), " ")
	if strings.Contains(joined, "kv-quality") {
		t.Fatalf("untouched KV quality must stay in config, not argv: %q", joined)
	}
}

func TestLaunchArgsCarriesClaudeProfileOnlyForClaudeCode(t *testing.T) {
	interactive := &LaunchRequest{
		ModelPath:     "/models/test.gguf",
		ClaudeCode:    true,
		ClaudeProfile: "agent-interactive",
	}
	joined := strings.Join(interactive.LaunchArgs(), " ")
	if !strings.Contains(joined, "--claude-code --claude-profile agent-interactive") {
		t.Fatalf("selected Claude profile must reach the CLI with Claude Code: %q", joined)
	}

	defaultProfile := &LaunchRequest{ModelPath: "/models/test.gguf", ClaudeCode: true}
	if joined := strings.Join(defaultProfile.LaunchArgs(), " "); strings.Contains(joined, "--claude-profile") {
		t.Fatalf("empty selector must preserve the CLI default, got %q", joined)
	}

	nonClaude := &LaunchRequest{ModelPath: "/models/test.gguf", ClaudeProfile: "agent-interactive"}
	if joined := strings.Join(nonClaude.LaunchArgs(), " "); strings.Contains(joined, "--claude-profile") {
		t.Fatalf("profile cannot be emitted without --claude-code, got %q", joined)
	}
}

func TestRunModesAreMutuallyExclusive(t *testing.T) {
	m := Model{screen: ScreenModelConfig, models: []ModelItem{{Name: "test.gguf"}}, kvPlacement: "auto", ctxMode: "fit", ctxSize: "fit"}
	m.input = textinput.New()
	m.cfgCursor = indexOf(m.cfgRows(), "aitune")
	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)
	if !m.aitune || m.benchmark {
		t.Fatal("AI tune should enable itself and disable benchmark")
	}
	m.cfgCursor = indexOf(m.cfgRows(), "benchmark")
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

func TestPrelaunchClaudeSlotWordingMatchesProfilePolicy(t *testing.T) {
	base := Model{
		models:        []ModelItem{{Name: "DeepSeek", Path: "/models/deepseek.gguf"}},
		selectedModel: 0,
		backend:       "llama",
		kvPlacement:   "auto",
		ctxMode:       "fit",
		claudeCode:    true,
		parallel:      "8", // a stale configured value, not an explicit launch override
	}

	interactive := base
	interactive.claudeProfile = "agent-interactive"
	view := interactive.viewPrelaunch()
	if !strings.Contains(view, "Parallel:       agent-interactive (1 foreground slot)") ||
		!strings.Contains(view, "Claude profile: agent-interactive (1 foreground agent)") {
		t.Fatalf("interactive prelaunch policy missing or inaccurate: %q", view)
	}
	if strings.Contains(view, "2 for Claude Code") {
		t.Fatalf("prelaunch still shows the obsolete two-slot Claude wording: %q", view)
	}

	parallel := base
	parallel.claudeProfile = "agent-parallel"
	view = parallel.viewPrelaunch()
	if !strings.Contains(view, "Parallel:       agent-parallel (4 workflow slots)") {
		t.Fatalf("parallel prelaunch policy missing or inaccurate: %q", view)
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
	doc := "MODEL_DIR=\"" + modelDir + "\"\nCACHE_DIR=\"" + cacheDir + "\"\nBACKEND=\"vulkan\"\nKV_PLACEMENT=\"gpu\"\nKV_QUALITY=\"q4_0\"\nSWA_FULL=\"1\"\nCTX_SIZE=\"131072\"\nPARALLEL=\"3\"\nPORT=\"9091\"\nVISION=\"1\"\nTUNE_ROUNDS=\"9\"\n"
	if err := os.WriteFile(cfgPath, []byte(doc), 0644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLM_CONFIG", cfgPath)
	t.Setenv("LLM_APP_HOME", appHome)
	m := InitialModel()
	if m.modelDir != modelDir || m.cacheDir != cacheDir || m.backend != "vulkan" || m.kvPlacement != "gpu" || m.kvQuality != "q4_0" || !m.swaFull || m.aituneRounds != 9 || m.ctxMode != "manual" || m.ctxSize != "131072" || m.parallel != "3" || m.port != 9091 || !m.vision {
		t.Fatalf("config not restored: modelDir=%q cacheDir=%q backend=%q kv=%q/%q swa=%v rounds=%d ctx=%q/%q parallel=%q port=%d vision=%v", m.modelDir, m.cacheDir, m.backend, m.kvPlacement, m.kvQuality, m.swaFull, m.aituneRounds, m.ctxMode, m.ctxSize, m.parallel, m.port, m.vision)
	}
	if !m.parallelSet {
		t.Fatal("an explicitly configured parallel value must remain authoritative for the launch")
	}
	m.models = []ModelItem{{Name: "Laguna", Path: filepath.Join(modelDir, "laguna.gguf")}}
	m.selectedModel = 0
	req := m.buildLaunchRequest()
	// SWAFullSet is deliberately NOT asserted here. A saved setting reaches the
	// launch through config, not as a command-line override: emitting it as a
	// typed flag made it inviolable (userExplicitBackendFlag reads OriginalArgs),
	// which locked the recovery ladder and the support expert out of withdrawing
	// a --swa-full that was costing 5.3 GiB of KV and buying no prefix reuse.
	// The value must still arrive; only its provenance changed.
	if req == nil || req.CtxFlag != "131072" || !req.ParallelSet || !req.SWAFull {
		t.Fatalf("standard optimal config did not reach launch request: %#v", req)
	}
	if req.SWAFullSet {
		t.Fatalf("an inherited setting must not be emitted as an explicit flag: %#v", req)
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

func TestLoadingScreenShowsUntilStartupReady(t *testing.T) {
	t.Setenv("LLM_CONFIG", filepath.Join(t.TempDir(), "config"))
	t.Setenv("LLM_APP_HOME", t.TempDir())
	m := loadingModel()
	if m.screen != ScreenLoading {
		t.Fatalf("startup screen=%v, want loading", m.screen)
	}
	cmd := m.Init()
	if cmd == nil {
		t.Fatal("loading screen did not start the spinner and hardware scan")
	}
	view := m.View()
	if !strings.Contains(view, "ggrun") || !strings.Contains(view, "Starting up") {
		t.Fatalf("loading view missing animation copy: %q", view)
	}
	got, _ := m.Update(startupReadyMsg{
		models: []ModelItem{{Name: "model.gguf", Path: "/models/model.gguf"}},
	})
	ready, ok := got.(Model)
	if !ok || ready.screen != ScreenMain || len(ready.models) != 1 {
		t.Fatalf("startup did not enter the main list: screen=%v models=%d ok=%t", ready.screen, len(ready.models), ok)
	}
	empty, _ := loadingModel().Update(startupReadyMsg{})
	firstRun, ok := empty.(Model)
	if !ok || firstRun.screen != ScreenFirstRun {
		t.Fatalf("empty catalog did not open first-run: %+v ok=%t", firstRun.screen, ok)
	}
}

func TestInitialModelDefaultsKVQualityToAuto(t *testing.T) {
	t.Setenv("LLM_CONFIG", filepath.Join(t.TempDir(), "config"))
	t.Setenv("LLM_APP_HOME", t.TempDir())
	m := InitialModel()
	if m.kvQuality != "auto" {
		t.Fatalf("TUI default KV quality must be model-aware auto, got %q", m.kvQuality)
	}
	if m.supportExpert != "auto" || m.supportOnline {
		t.Fatalf("TUI support defaults = %q online=%v, want auto/off", m.supportExpert, m.supportOnline)
	}
}

// Changing KV settings in the Settings screen must apply to the live session,
// not only to the next TUI start.
func TestApplySettingSyncsKVIntoLiveSession(t *testing.T) {
	t.Setenv("LLM_CONFIG", filepath.Join(t.TempDir(), "config"))
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
	t.Setenv("LLM_CONFIG", filepath.Join(t.TempDir(), "config"))
	t.Setenv("LLM_APP_HOME", t.TempDir())
	m := Model{settingsCfg: config.Defaults(), ctxMode: "fit", ctxSize: "fit", port: 8081, parallel: "1", aituneRounds: 8}
	rows := map[string]settingRow{}
	for _, row := range settingRows() {
		rows[row.label] = row
	}
	for _, label := range []string{"Context", "Full SWA cache", "Support expert / optimizer", "Support online research", "Vision", "Port", "Parallel", "AI-tune rounds"} {
		if rows[label].label == "" {
			t.Fatalf("settings row %q not found", label)
		}
	}
	m.applySetting(rows["Context"], "max")
	m.applySetting(rows["Full SWA cache"], "on")
	m.applySetting(rows["Support expert / optimizer"], "on")
	m.applySetting(rows["Support online research"], "on")
	m.applySetting(rows["Vision"], "on")
	m.applySetting(rows["Port"], "9099")
	m.applySetting(rows["Parallel"], "3")
	m.applySetting(rows["AI-tune rounds"], "12")
	if m.ctxMode != "max" || !m.swaFull || m.supportExpert != "on" || !m.supportOnline || !m.vision || m.port != 9099 || m.parallel != "3" || m.aituneRounds != 12 {
		t.Fatalf("launch settings not synced: ctx=%q swa=%v support=%q online=%v vision=%v port=%d parallel=%q rounds=%d", m.ctxMode, m.swaFull, m.supportExpert, m.supportOnline, m.vision, m.port, m.parallel, m.aituneRounds)
	}
	if !m.parallelSet {
		t.Fatal("a parallel value explicitly entered in Settings must remain authoritative")
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
	m.applySetting(rows["RAM limit percent"], "101")
	if m.settingsCfg.RAMLimitPercent != 95 || m.messageType != "warning" {
		t.Fatalf("invalid RAM limit percent changed settings: %d", m.settingsCfg.RAMLimitPercent)
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

func TestLaunchArgsEmitsClaudeResume(t *testing.T) {
	req := &LaunchRequest{
		ModelPath: "model.gguf", Port: 8081, CtxFlag: "fit",
		ClaudeCode: true, ResumeSession: "072e63a1-819a-4682-a742-559695c3cd76",
	}
	args := strings.Join(req.LaunchArgs(), " ")
	if !strings.Contains(args, "--claude-resume 072e63a1-819a-4682-a742-559695c3cd76") {
		t.Errorf("resume flag missing from launch args: %s", args)
	}
	// The TUI and the CLI must reach the same launch path.
	if !strings.Contains(args, "--claude-code") {
		t.Errorf("resume emitted without claude-code mode: %s", args)
	}
}

func TestLaunchArgsOmitsResumeWhenNotResuming(t *testing.T) {
	req := &LaunchRequest{ModelPath: "model.gguf", CtxFlag: "fit", ClaudeCode: true}
	if args := strings.Join(req.LaunchArgs(), " "); strings.Contains(args, "--claude-resume") {
		t.Errorf("fresh launch emitted a resume flag: %s", args)
	}
}

func TestShortSessionIDKeepsPrelaunchReadable(t *testing.T) {
	if got := shortSessionID("072e63a1-819a-4682-a742-559695c3cd76"); got != "072e63a1-819a…" {
		t.Errorf("shortSessionID = %q", got)
	}
	if got := shortSessionID("short"); got != "short" {
		t.Errorf("short id was truncated: %q", got)
	}
}

// A download must be able to land on a different disk than the configured model
// directory without the user having to repoint ModelDir and put it back. The
// repo prompt therefore hands off to a destination prompt rather than quitting,
// and only emits the request once both halves are known.
func TestDownloadPromptsForDestination(t *testing.T) {
	m := Model{screen: ScreenFirstRun, modelDir: "/home/u/models"}
	m.input = textinput.New()

	nm, _ := m.doFirstRunAction("download")
	m = nm.(Model)
	m.input.SetValue("unsloth/Inkling-Small-GGUF")
	nm, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)

	if m.launchRequest != nil {
		t.Fatal("entering the repo must not emit the download yet")
	}
	if m.inputMode != "downloaddir" {
		t.Fatalf("expected the destination prompt, got inputMode=%q", m.inputMode)
	}
	if got := m.input.Value(); got != "/home/u/models" {
		t.Fatalf("destination should be pre-filled with the model dir, got %q", got)
	}

	m.input.SetValue("/mnt/2tb/AI_Models")
	nm, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)
	if m.launchRequest == nil {
		t.Fatal("confirming the destination must emit the download")
	}
	if m.launchRequest.DownloadRepo != "unsloth/Inkling-Small-GGUF" {
		t.Fatalf("repo lost across the destination prompt: %q", m.launchRequest.DownloadRepo)
	}
	if m.launchRequest.DownloadDir != "/mnt/2tb/AI_Models" {
		t.Fatalf("destination not carried, got %q", m.launchRequest.DownloadDir)
	}
}

// Pressing Enter on the pre-filled default must behave exactly as before this
// prompt existed: download into the configured model directory, with no
// per-download override recorded.
func TestDownloadDestinationDefaultsToModelDir(t *testing.T) {
	m := Model{screen: ScreenFirstRun, modelDir: "/home/u/models"}
	m.input = textinput.New()
	nm, _ := m.promptDownloadDir(&LaunchRequest{DownloadRepo: "org/repo", DownloadQuant: "Q3_K_XL"})
	m = nm.(Model)
	nm, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)

	if m.launchRequest == nil {
		t.Fatal("expected the download to be emitted")
	}
	if m.launchRequest.DownloadDir != "" {
		t.Fatalf("accepting the default must not record an override, got %q", m.launchRequest.DownloadDir)
	}
	if m.launchRequest.DownloadQuant != "Q3_K_XL" {
		t.Fatalf("quant lost across the destination prompt: %q", m.launchRequest.DownloadQuant)
	}
}

// The quant picker reached from the recommendation screen is the other way into
// a download, and needs the same destination step.
func TestRecommendedDownloadPromptsForDestination(t *testing.T) {
	m := Model{
		screen:   ScreenRecommended,
		modelDir: "/home/u/models",
		recommendations: []recommend.Recommendation{{
			Candidate: recommend.Candidate{Repo: "unsloth/Inkling-Small-GGUF"},
			QuantName: "UD-Q3_K_XL",
		}},
		selectedRecommendation: 0,
	}
	m.input = textinput.New()
	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)
	if m.launchRequest != nil {
		t.Fatal("the quant picker must ask for a destination before downloading")
	}
	if m.inputMode != "downloaddir" || m.pendingDownload == nil {
		t.Fatalf("expected the destination prompt, got inputMode=%q", m.inputMode)
	}
	if m.pendingDownload.DownloadQuant != "UD-Q3_K_XL" {
		t.Fatalf("selected quant not carried, got %q", m.pendingDownload.DownloadQuant)
	}
}

// The model config screen's "y" shortcut must open the confirm-first clear
// screen defaulting to Cancel, and confirming must remove only that model's
// cached configs while keeping the GGUF and staying on the config screen.
func TestClearCachesKeyOpensConfirmDefaultedToCancelAndClearsCaches(t *testing.T) {
	cacheDir := t.TempDir()
	modelDir := t.TempDir()
	ggufPath := filepath.Join(modelDir, "cache-test-Q4_K_M.gguf")
	if err := os.WriteFile(ggufPath, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	// A probe cache and a placement cache for this model, plus one for another.
	// WriteProbeCacheForModel lands at the exact hashed path loadProbeCache reads,
	// so the clear helper's header match is exercised against a real file.
	profile := &placement.ModelProfile{Path: ggufPath, Basename: "cache-test-Q4_K_M", SizeBytes: 4, TotalSizeMB: 1,
		NumLayers: 12, NumExperts: 16, EmbeddingLength: 2048, FeedForwardLength: 4096}
	if err := placement.WriteProbeCacheForModel(cacheDir, profile, 4096, 512, "mid", "gpu", "llama", nil, map[int]int{0: 2048}, 128); err != nil {
		t.Fatal(err)
	}
	other := &placement.ModelProfile{Path: "/models/Other.gguf", Basename: "Other", SizeBytes: 4, TotalSizeMB: 1,
		NumLayers: 12, NumExperts: 16, EmbeddingLength: 2048, FeedForwardLength: 4096}
	if err := placement.WriteProbeCacheForModel(cacheDir, other, 4096, 512, "mid", "gpu", "llama", nil, map[int]int{0: 2048}, 128); err != nil {
		t.Fatal(err)
	}
	// A placement cache tagged with this model's basename (the same form
	// Compute's StrategyToCacheEntry writes: filepath.Base(model.Path)).
	place := placement.PlacementCachePathFor(cacheDir, profile, 4096, 512, "mid", "gpu", "llama", nil, 1, "", false)
	if err := placement.SavePlacementCache(place, &placement.CacheEntry{
		OTString: "exps=CPU", TensorSplit: []float64{1}, SplitMode: "layer",
		NCPUMoE: 30, BatchSize: 2048, UBatchSize: 512, Parallel: 1, MMap: true, ModelBasename: filepath.Base(ggufPath),
	}); err != nil {
		t.Fatal(err)
	}
	// Find the exact file paths the writers produced.
	var probe, otherProbe string
	entries, _ := os.ReadDir(cacheDir)
	for _, ent := range entries {
		if ent.IsDir() || !strings.HasSuffix(ent.Name(), ".probe") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(cacheDir, ent.Name()))
		if err != nil {
			continue
		}
		if strings.Contains(string(data), "cache-test-Q4_K_M") {
			probe = filepath.Join(cacheDir, ent.Name())
		} else if strings.Contains(string(data), "Other.gguf") {
			otherProbe = filepath.Join(cacheDir, ent.Name())
		}
	}
	if probe == "" || otherProbe == "" {
		t.Fatalf("probe caches not written as expected: probe=%q other=%q", probe, otherProbe)
	}

	m := Model{
		screen:        ScreenModelConfig,
		modelDir:      modelDir,
		cacheDir:      cacheDir,
		models:        []ModelItem{{Name: "cache-test-Q4_K_M.gguf", Path: ggufPath}},
		selectedModel: 0,
	}
	m.input = textinput.New()

	// 'y' opens the confirm screen, cursor on Cancel.
	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
	m = nm.(Model)
	if m.screen != ScreenChoice {
		t.Fatalf("expected the 'y' shortcut to open the confirm screen, got screen=%v", m.screen)
	}
	if !strings.Contains(m.choiceTitle, "cache-test-Q4_K_M") {
		t.Fatalf("confirm title should name the model, got %q", m.choiceTitle)
	}
	if m.choiceOptions[m.choiceCursor] != "Cancel" {
		t.Fatalf("confirm cursor must default to Cancel, got %q", m.choiceOptions[m.choiceCursor])
	}

	// Cancel leaves everything in place.
	nm, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)
	if _, err := os.Stat(probe); err != nil {
		t.Fatalf("Cancel must not clear caches: %v", err)
	}

	// Reopen and confirm.
	nm, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
	m = nm.(Model)
	for i, opt := range m.choiceOptions {
		if strings.HasPrefix(opt, "Confirm: clear caches for ") {
			m.choiceCursor = i
			break
		}
	}
	nm, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = nm.(Model)
	if cmd != nil {
		t.Fatal("clearing caches must not quit the TUI")
	}
	if m.screen != ScreenModelConfig {
		t.Fatalf("expected to return to the model config screen, got %v", m.screen)
	}
	if _, err := os.Stat(probe); !os.IsNotExist(err) {
		t.Errorf("model probe cache was not cleared (stat err=%v)", err)
	}
	if _, err := os.Stat(place); !os.IsNotExist(err) {
		t.Errorf("model placement cache was not cleared (stat err=%v)", err)
	}
	if _, err := os.Stat(otherProbe); err != nil {
		t.Errorf("other model's probe cache must survive: %v", err)
	}
	if _, err := os.Stat(ggufPath); err != nil {
		t.Errorf("the GGUF must be kept: %v", err)
	}
	if m.messageType != "info" || !strings.Contains(m.message, "Cleared") {
		t.Fatalf("expected an info message about clearing caches, got type=%q msg=%q", m.messageType, m.message)
	}
}

// The lower-level clear path is guarded against an out-of-range index.
func TestClearModelCachesAtOutOfRangeIsNoop(t *testing.T) {
	m := Model{cacheDir: t.TempDir(), models: []ModelItem{{Name: "test.gguf"}}}
	m.clearModelCachesAt(5)
	if m.messageType == "error" {
		t.Fatalf("out-of-range clear must be a silent no-op, got %q", m.message)
	}
}

// The "g" shortcut on the model config screen toggles "launch without cached
// config", which must reach the LaunchRequest as --no-cached-config.
func TestNoCachedConfigToggleEmitsFlag(t *testing.T) {
	m := Model{
		screen:        ScreenModelConfig,
		models:        []ModelItem{{Name: "test.gguf", Path: "/models/test.gguf"}},
		selectedModel: 0,
		kvPlacement:   "auto",
		ctxMode:       "fit",
		ctxSize:       "fit",
	}
	m.input = textinput.New()

	nm, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'g'}})
	m = nm.(Model)
	if !m.noCachedConfig {
		t.Fatal("'g' should enable launch-without-cached-config")
	}
	req := m.buildLaunchRequest()
	if req == nil || !req.NoCachedConfig {
		t.Fatalf("launch request must carry NoCachedConfig, got %#v", req)
	}
	args := req.LaunchArgs()
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--no-cached-config") {
		t.Fatalf("expected --no-cached-config in launch args, got %q", joined)
	}

	// Toggling again restores the default.
	nm, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'g'}})
	m = nm.(Model)
	if m.noCachedConfig {
		t.Fatal("'g' again should disable launch-without-cached-config")
	}
	req = m.buildLaunchRequest()
	if req == nil || req.NoCachedConfig {
		t.Fatalf("launch request must drop NoCachedConfig after toggling off, got %#v", req)
	}
}

// "Launch without cached config" must round-trip through a replayed launch
// request (Run latest configuration).
func TestNoCachedConfigSurvivesReplay(t *testing.T) {
	req := &LaunchRequest{ModelPath: "/models/test.gguf", NoCachedConfig: true}
	m := Model{}
	m.applyLaunchRequestFields(req)
	if !m.noCachedConfig {
		t.Fatal("replayed request must restore the no-cached-config toggle")
	}
}

// The model config screen must advertise both cache-management actions: the
// destructive "[y] Clear caches" and the per-launch "[g] Launch without cached
// config" toggle, with the toggle reflecting its off state.
func TestModelConfigViewAdvertisesCacheActions(t *testing.T) {
	m := Model{
		screen:        ScreenModelConfig,
		models:        []ModelItem{{Name: "test.gguf", Path: "/models/test.gguf"}},
		selectedModel: 0,
		kvPlacement:   "auto",
		kvQuality:     "mid",
		ctxMode:       "fit",
		ctxSize:       "fit",
	}
	v := m.viewModelConfig()
	for _, want := range []string{"[y] Clear caches", "[g] Launch without cached config", "reuse cached placement/probes"} {
		if !strings.Contains(v, want) {
			t.Fatalf("model config view missing %q", want)
		}
	}
	// The on state renders too.
	m.noCachedConfig = true
	v = m.viewModelConfig()
	if !strings.Contains(v, "derive fresh, ignore cached config") {
		t.Fatalf("model config view missing the no-cached-config on state")
	}
}

func TestSortModelsByUsageDesc(t *testing.T) {
	items := []ModelItem{
		{Name: "zeta.gguf", Path: "/models/zeta.gguf"},
		{Name: "alpha.gguf", Path: "/models/alpha.gguf"},
		{Name: "mid.gguf", Path: "/models/mid.gguf"},
	}
	usage := map[string]modelusage.Record{
		"/models/alpha.gguf": {Launches: 5, LastUsedAt: time.Now().Add(-2 * time.Hour)},
		"/models/mid.gguf":   {Launches: 3, LastUsedAt: time.Now().Add(-1 * time.Hour)},
		"/models/zeta.gguf":  {Launches: 5, LastUsedAt: time.Now().Add(-3 * time.Hour)},
	}
	sortModels(items, usage)
	got := []string{items[0].Name, items[1].Name, items[2].Name}
	want := []string{"alpha.gguf", "zeta.gguf", "mid.gguf"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("sortModels[%d] = %q, want %q (full order %v)", i, got[i], want[i], got)
		}
	}
}

func TestSortModelsFallsBackToNameWhenNoUsage(t *testing.T) {
	items := []ModelItem{
		{Name: "zeta.gguf", Path: "/models/zeta.gguf"},
		{Name: "alpha.gguf", Path: "/models/alpha.gguf"},
		{Name: "mid.gguf", Path: "/models/mid.gguf"},
	}
	sortModels(items, nil)
	got := []string{items[0].Name, items[1].Name, items[2].Name}
	want := []string{"alpha.gguf", "mid.gguf", "zeta.gguf"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("sortModels no-usage[%d] = %q, want %q (full order %v)", i, got[i], want[i], got)
		}
	}
}

func TestChatTemplateRowCyclesAndEmitsFlag(t *testing.T) {
	m := Model{
		models:        []ModelItem{{Name: "nanbeige.gguf", Path: "/models/nanbeige.gguf", Architecture: "nanbeige"}},
		selectedModel: 0,
		kvPlacement:   "auto",
		ctxMode:       "fit",
	}
	if indexOf(m.cfgRows(), "chattemplate") < 0 {
		t.Fatal("Chat template row missing from the Advanced config screen")
	}
	// auto -> first catalog entry (nanbeige4.2-3b is first in catalog.json).
	m.cycleCfgRow("chattemplate", 1)
	if m.chatTemplate != "nanbeige4.2-3b" {
		t.Fatalf("cycle chat template: got %q, want nanbeige4.2-3b", m.chatTemplate)
	}
	req := m.buildLaunchRequest()
	if req == nil || req.ChatTemplate != "nanbeige4.2-3b" {
		t.Fatalf("chat template must reach the launch request: %#v", req)
	}
	joined := strings.Join(req.LaunchArgs(), " ")
	if !strings.Contains(joined, "--chat-template nanbeige4.2-3b") {
		t.Fatalf("forced chat template must be an explicit launch flag: %q", joined)
	}
	// Cycling back returns to auto (empty).
	m.cycleCfgRow("chattemplate", -1)
	if m.chatTemplate != "" {
		t.Fatalf("cycle chat template back: got %q, want empty auto", m.chatTemplate)
	}
	req = m.buildLaunchRequest()
	if req == nil || req.ChatTemplate != "" {
		t.Fatalf("auto chat template must not emit a flag: %#v", req)
	}
	joined = strings.Join(req.LaunchArgs(), " ")
	if strings.Contains(joined, "--chat-template") {
		t.Fatalf("auto chat template must stay in catalog matching, not argv: %q", joined)
	}
}

func TestChatTemplateSurvivesReplay(t *testing.T) {
	req := &LaunchRequest{ModelPath: "/models/test.gguf", ChatTemplate: "qwen3.8-27b"}
	m := Model{}
	m.applyLaunchRequestFields(req)
	if m.chatTemplate != "qwen3.8-27b" {
		t.Fatalf("replayed request must restore the chat template, got %q", m.chatTemplate)
	}
}

func TestChatTemplateLabelShowsForcedAndAuto(t *testing.T) {
	forced := Model{chatTemplate: "nanbeige4.2-3b"}
	if label := forced.chatTemplateLabel(ModelItem{Architecture: "nanbeige"}); !strings.Contains(label, "forced: nanbeige4.2-3b") {
		t.Fatalf("forced chat template label = %q", label)
	}
	// A model whose arch matches a catalog entry with a broken embedded template
	// shows the auto fix even before the user touches the row.
	auto := Model{}
	if entry, ok := auto.autoChatTemplate(ModelItem{Name: "nanbeige.gguf", Architecture: "nanbeige"}); !ok || entry.Name != "nanbeige4.2-3b" {
		t.Fatalf("auto chat template not detected for nanbeige arch: entry=%#v ok=%v", entry, ok)
	}
	// A model with no catalog match has no auto override and the row says auto.
	plain := Model{}
	if label := plain.chatTemplateLabel(ModelItem{Name: "plain.gguf", Architecture: "llama3"}); label != "auto" {
		t.Fatalf("plain model chat template label = %q, want auto", label)
	}
}

func TestKVQualityOptionsDedupeAliases(t *testing.T) {
	opts := kvQualityOptions()
	for _, dup := range []string{"low", "f16"} {
		if indexOf(opts, dup) >= 0 {
			t.Fatalf("kvQualityOptions still contains the alias %q: %v", dup, opts)
		}
	}
	if opts[0] != "auto" {
		t.Fatalf("kvQualityOptions must start with auto, got %q", opts[0])
	}
	if !strings.Contains(strings.Join(opts, ","), "q4_0") {
		t.Fatalf("kvQualityOptions must keep the concrete q4_0 type: %v", opts)
	}
}

func TestKVQualityLabelFriendlyOnBothScreens(t *testing.T) {
	if got := kvQualityLabel("high"); got != "high (f16)" {
		t.Fatalf("kvQualityLabel(high) = %q, want high (f16)", got)
	}
	if got := kvQualityLabel("low"); got != "low (q4_0)" {
		t.Fatalf("kvQualityLabel(low) = %q, want low (q4_0)", got)
	}
}

func TestMainModelDescShowsUsageAndLocation(t *testing.T) {
	rec := &modelusage.Record{Launches: 12, LastUsedAt: time.Now().Add(-3 * 24 * time.Hour)}
	desc := mainModelDesc(ModelItem{
		Name: "model-Q4.gguf", Path: "/models/repo-a/model-Q4.gguf",
		SizeGB: 7.6, Arch: "llama3", FitCtx: 131072,
	}, rec)
	for _, want := range []string{"7.6GB", "llama3", "fits ~131k", "12x", "3d ago", "repo-a"} {
		if !strings.Contains(desc, want) {
			t.Fatalf("main model desc missing %q in %q", want, desc)
		}
	}
	// External models get the "discovered" marker instead of a bare dir.
	ext := mainModelDesc(ModelItem{Name: "found.gguf", Path: "/elsewhere/found.gguf", SizeGB: 1.2, Arch: "qwen", External: true}, nil)
	if !strings.Contains(ext, "discovered") {
		t.Fatalf("external model desc must say discovered: %q", ext)
	}
	if strings.Contains(ext, "elsewhere") {
		t.Fatalf("external model desc must not leak the full dir: %q", ext)
	}
}

func TestClipToFocusedKeepsCursorRowVisible(t *testing.T) {
	lines := []string{}
	for i := 0; i < 12; i++ {
		lines = append(lines, fmt.Sprintf("  row %02d", i))
	}
	lines[6] = "  > row 06"
	body := strings.Join(lines, "\n")
	clipped, ok := clipToFocused(body, 5)
	if !ok {
		t.Fatal("clip must trigger when body exceeds maxLines")
	}
	if !strings.Contains(clipped, "> row 06") {
		t.Fatalf("clip must keep the focused row, got:\n%s", clipped)
	}
	if len(strings.Split(clipped, "\n")) != 5 {
		t.Fatalf("clip must return exactly maxLines rows, got %d", len(strings.Split(clipped, "\n")))
	}
	// Tail-first for screens without a cursor marker: keep the launch actions.
	plain := []string{}
	for i := 0; i < 12; i++ {
		plain = append(plain, fmt.Sprintf("  row %02d", i))
	}
	clippedTail, _ := clipToFocused(strings.Join(plain, "\n"), 5)
	if !strings.Contains(clippedTail, "row 11") {
		t.Fatalf("tail-first clip must keep the last row, got:\n%s", clippedTail)
	}
}

func TestHasAnyUsage(t *testing.T) {
	if hasAnyUsage(map[string]modelusage.Record{}) {
		t.Fatal("empty usage must report false")
	}
	if hasAnyUsage(map[string]modelusage.Record{"/m": {Launches: 0}}) {
		t.Fatal("zero-launch records must report false")
	}
	if !hasAnyUsage(map[string]modelusage.Record{"/m": {Launches: 1}}) {
		t.Fatal("a launched record must report true")
	}
}

func TestInitialParallelPolicyReachesLaunch(t *testing.T) {
	for _, tc := range []struct {
		name, configValue, envValue string
		wantValue                   int
		wantExplicit                bool
	}{
		{"saved default", "1", "", 1, false},
		{"configured wider layout", "2", "", 2, true},
		{"environment pins one", "2", "1", 1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			app := t.TempDir()
			cfgPath := filepath.Join(app, "config")
			if err := os.WriteFile(cfgPath, []byte("LLM_PARALLEL="+tc.configValue+"\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			t.Setenv("LLM_APP_HOME", app)
			t.Setenv("LLM_CONFIG", cfgPath)
			t.Setenv("LLM_PARALLEL", tc.envValue)
			m := InitialModel()
			m.models = []ModelItem{{Name: "test.gguf", Path: "/models/test.gguf"}}
			m.selectedModel = 0
			m.claudeCode = true
			req := m.buildLaunchRequest()
			if req == nil || req.Parallel != tc.wantValue || req.ParallelSet != tc.wantExplicit {
				t.Fatalf("launch=%#v, want parallel=%d explicit=%v", req, tc.wantValue, tc.wantExplicit)
			}
			args := req.LaunchArgs()
			pinned := false
			for _, arg := range args {
				if arg == "--parallel" {
					pinned = true
				}
			}
			if pinned != tc.wantExplicit {
				t.Fatalf("args=%v, want explicit parallel=%v", args, tc.wantExplicit)
			}
		})
	}
}

// The terminal check must be a property of STDIN, not of a Unix device path.
//
// It previously opened the literal "/dev/tty", which cannot exist on Windows, so
// a bare `ggrun` at a real Windows console always printed the no-terminal
// guidance and the TUI was unreachable there (issue #63). This test pins the
// portable contract on every platform: give the process a known stdin and the
// answer must follow.
//
// HONEST LIMIT, recorded rather than hidden: on Linux this test does not fail
// against the old /dev/tty implementation, because /dev/tty genuinely exists in
// a developer shell and the old probe succeeded regardless of stdin. The
// platform-independent assertion that WOULD catch the regression is the Windows
// build, where the old code could never open the path — see
// TestTerminalCheckDoesNotDependOnAUnixDevicePath below, which fails at compile
// time on Windows if the check reintroduces a Unix-only path.
func TestTerminalAvailableFollowsStdinNotADevicePath(t *testing.T) {
	saved := os.Stdin
	t.Cleanup(func() { os.Stdin = saved })

	// A pipe is not a terminal, so the UI must decline it regardless of whether
	// this process happens to own a controlling terminal.
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	defer func() { _ = r.Close(); _ = w.Close() }()
	os.Stdin = r
	if terminalAvailable() {
		t.Error("a pipe stdin was reported as an interactive terminal; " +
			"a full-screen UI would be started in a pipeline or CI job")
	}

	// /dev/null is a character device but is not a terminal. The old /dev/tty
	// probe could not distinguish this case either.
	devNull, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatalf("open %s: %v", os.DevNull, err)
	}
	defer func() { _ = devNull.Close() }()
	os.Stdin = devNull
	if terminalAvailable() {
		t.Error("stdin on the null device was reported as an interactive terminal")
	}

	// A regular file is not a terminal.
	f, err := os.CreateTemp(t.TempDir(), "stdin")
	if err != nil {
		t.Fatalf("tempfile: %v", err)
	}
	defer func() { _ = f.Close() }()
	os.Stdin = f
	if terminalAvailable() {
		t.Error("a regular-file stdin was reported as an interactive terminal")
	}
}

// The platform-independent half of the #63 regression, expressible without a
// Windows console: the terminal check must not be derived from the Unix device
// path. A source-level assertion, because the behavioural one above cannot fail
// on Linux (where /dev/tty exists) and no CI job runs Go tests on Windows.
//
// If the check is reimplemented by opening "/dev/tty", this fails. If it is
// reimplemented portably, it passes — the intent is pinned even though the
// mechanism may legitimately change.
func TestTerminalCheckDoesNotDependOnAUnixDevicePath(t *testing.T) {
	src, err := os.ReadFile("tui.go")
	if err != nil {
		t.Fatalf("read tui.go: %v", err)
	}
	body := string(src)
	// Locate the terminalAvailable definition and inspect only its own body, so
	// an unrelated mention of the path elsewhere cannot satisfy this.
	start := strings.Index(body, "func terminalAvailable() bool {")
	if start < 0 {
		t.Fatal("terminalAvailable is gone; update this guard with it")
	}
	end := strings.Index(body[start:], "\n}")
	if end < 0 {
		t.Fatal("could not delimit terminalAvailable")
	}
	fn := body[start : start+end]

	if strings.Contains(fn, "/dev/tty") {
		t.Error("terminalAvailable opens the Unix-only /dev/tty again: on Windows " +
			"that path never exists, so a bare ggrun would report no terminal at a " +
			"real console and the TUI would be unreachable (issue #63)")
	}
	if !strings.Contains(fn, "IsTerminal") {
		t.Error("terminalAvailable no longer uses a platform-aware terminal check; " +
			"the portable predicate is term.IsTerminal(fd)")
	}
}
