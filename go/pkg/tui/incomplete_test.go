package tui

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/raketenkater/ggrun/pkg/gguf"
)

// writeF32GGUF writes a one-tensor GGUF (elems F32 values, alignment 32) whose
// data section is dataBytes long, so dataBytes < elems*4 is a short file.
func writeF32GGUF(t *testing.T, path string, elems uint64, dataBytes int) {
	t.Helper()
	buf := new(bytes.Buffer)
	buf.WriteString("GGUF")
	binary.Write(buf, binary.LittleEndian, uint32(3))
	binary.Write(buf, binary.LittleEndian, uint64(1)) // tensors
	binary.Write(buf, binary.LittleEndian, uint64(0)) // kv
	name := "w"
	binary.Write(buf, binary.LittleEndian, uint64(len(name)))
	buf.WriteString(name)
	binary.Write(buf, binary.LittleEndian, uint32(1))
	binary.Write(buf, binary.LittleEndian, elems)
	binary.Write(buf, binary.LittleEndian, uint32(0)) // F32
	binary.Write(buf, binary.LittleEndian, uint64(0))
	end := buf.Len()
	buf.Write(make([]byte, (end+31)/32*32-end+dataBytes))
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestIncompleteDownloadNote(t *testing.T) {
	dir := t.TempDir()
	full := filepath.Join(dir, "full.gguf")
	writeF32GGUF(t, full, 1024, 4096)
	if got := incompleteDownloadNote(gguf.OpenHeaderCache(""), full); got != "" {
		t.Fatalf("full file flagged: %q", got)
	}
	short := filepath.Join(dir, "short.gguf")
	writeF32GGUF(t, short, 1024, 4095)
	if got := incompleteDownloadNote(gguf.OpenHeaderCache(""), short); !strings.HasPrefix(got, "incomplete download: 0.0 of 0.0 GB") {
		t.Fatalf("short file note = %q", got)
	}

	// Split model: first shard complete, second short.
	for i, data := range []int{4096, 100} {
		writeF32GGUF(t, filepath.Join(dir, fmt.Sprintf("big-%05d-of-00002.gguf", i+1)), 1024, data)
	}
	if got := incompleteDownloadNote(gguf.OpenHeaderCache(""), filepath.Join(dir, "big-00001-of-00002.gguf")); got == "" {
		t.Fatal("split model with a short shard not flagged")
	}
	writeF32GGUF(t, filepath.Join(dir, "big-00002-of-00002.gguf"), 1024, 4096)
	if got := incompleteDownloadNote(gguf.OpenHeaderCache(""), filepath.Join(dir, "big-00001-of-00002.gguf")); got != "" {
		t.Fatalf("complete split model flagged: %q", got)
	}
}

func TestIncompleteModelIsMarkedAndNotLaunchable(t *testing.T) {
	dir := t.TempDir()
	short := filepath.Join(dir, "short.gguf")
	writeF32GGUF(t, short, 1024, 10)
	models := enrichModelItems([]ModelItem{{Name: "short.gguf", Path: short, Arch: "dense"}}, t.TempDir(), "auto", nil)
	if len(models) != 1 || !strings.Contains(models[0].Incomplete, "incomplete download") {
		t.Fatalf("model not marked: %#v", models)
	}
	if desc := mainModelDesc(models[0], nil); !strings.Contains(desc, "incomplete download") {
		t.Fatalf("desc = %q", desc)
	}

	m := Model{screen: ScreenMain, models: models, mainList: newMainList(models)}
	m.input = textinput.New()
	m.mainList.Select(3) // first model row, after the three leading actions
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	got := next.(Model)
	if got.screen != ScreenMain || got.launchRequest != nil {
		t.Fatalf("incomplete model opened screen %v", got.screen)
	}
	if !strings.Contains(got.message, "incomplete download") || got.messageType != "error" {
		t.Fatalf("message = %q (%s)", got.message, got.messageType)
	}
}

func TestIncompleteVerdictUsesHeaderCache(t *testing.T) {
	calls := 0
	orig := expectedSize
	expectedSize = func(path string) (int64, bool) { calls++; return orig(path) }
	t.Cleanup(func() { expectedSize = orig })

	dir := t.TempDir()
	cacheDir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	writeF32GGUF(t, path, 1024, 4095)
	note := func() string {
		cache := gguf.OpenHeaderCache(cacheDir)
		got := incompleteDownloadNote(cache, path)
		_ = cache.Save()
		return got
	}
	if got := note(); got == "" || calls != 1 {
		t.Fatalf("first start: note %q, calls %d", got, calls)
	}
	if got := note(); got == "" || calls != 1 {
		t.Fatalf("second start re-read the header: note %q, calls %d", got, calls)
	}

	// The download finishes: size changes, the header is read again and the
	// verdict flips.
	writeF32GGUF(t, path, 1024, 4096)
	if got := note(); got != "" || calls != 2 {
		t.Fatalf("after size change: note %q, calls %d", got, calls)
	}
	// Same size, new mtime also invalidates.
	later := time.Now().Add(time.Hour)
	if err := os.Chtimes(path, later, later); err != nil {
		t.Fatal(err)
	}
	if got := note(); got != "" || calls != 3 {
		t.Fatalf("after mtime change: note %q, calls %d", got, calls)
	}
	if got := note(); got != "" || calls != 3 {
		t.Fatalf("unchanged file re-read: note %q, calls %d", got, calls)
	}
}

func TestKeyPressRechecksFinishedDownload(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	writeF32GGUF(t, path, 1024, 10)
	cacheDir := t.TempDir()
	models := enrichModelItems([]ModelItem{{Name: "m.gguf", Path: path, Arch: "dense"}}, cacheDir, "auto", nil)
	if len(models) != 1 || models[0].Incomplete == "" {
		t.Fatalf("model not marked: %#v", models)
	}
	m := Model{screen: ScreenMain, cacheDir: cacheDir, models: models, mainList: newMainList(models)}
	m.input = textinput.New()
	m.mainList.Select(3)

	// Still short: refused, as before.
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if got := next.(Model); got.screen != ScreenMain || got.messageType != "error" {
		t.Fatalf("short model opened screen %v", got.screen)
	}

	// The download finishes while the TUI is open.
	writeF32GGUF(t, path, 1024, 4096)
	next, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	got := next.(Model)
	if got.screen == ScreenMain || got.models[0].Incomplete != "" {
		t.Fatalf("finished download still blocked: screen %v, note %q", got.screen, got.models[0].Incomplete)
	}
	if item, ok := got.mainList.Items()[3].(mainItem); !ok || strings.Contains(item.desc, "incomplete download") {
		t.Fatalf("list row still marked: %#v", item)
	}
}

// A header-only file that declares 4 GiB of F32 data: the note must carry the
// real sizes, not "0.0 of 0.0 GB".
func TestIncompleteNoteShowsRealGigabytes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "big.gguf")
	writeF32GGUF(t, path, 1<<30, 0)
	got := incompleteDownloadNote(gguf.OpenHeaderCache(""), path)
	if got != "incomplete download: 0.0 of 4.0 GB" {
		t.Fatalf("note = %q", got)
	}
}

func incompleteMainModel(t *testing.T) Model {
	t.Helper()
	path := filepath.Join(t.TempDir(), "short.gguf")
	writeF32GGUF(t, path, 1<<30, 10)
	cacheDir := t.TempDir()
	models := enrichModelItems([]ModelItem{{Name: "short.gguf", Path: path, Arch: "dense"}}, cacheDir, "auto", nil)
	if len(models) != 1 || models[0].Incomplete == "" {
		t.Fatalf("model not marked: %#v", models)
	}
	m := Model{screen: ScreenMain, cacheDir: cacheDir, models: models, mainList: newMainList(models)}
	m.input = textinput.New()
	m.mainList.Select(3)
	return m
}

func requireIncompleteRefusal(t *testing.T, got Model) {
	t.Helper()
	if got.screen != ScreenMain || got.launchRequest != nil || got.messageType != "error" {
		t.Fatalf("incomplete model was not refused: screen %v, type %q, msg %q", got.screen, got.messageType, got.message)
	}
	if !strings.Contains(got.message, "incomplete download: 0.0 of 4.0 GB") || !strings.Contains(got.message, "ggrun launch <path>") {
		t.Fatalf("refusal = %q", got.message)
	}
}

func TestConfigureKeyRefusesIncompleteModel(t *testing.T) {
	m := incompleteMainModel(t)
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'c'}})
	requireIncompleteRefusal(t, next.(Model))
}

func TestLatestLaunchRefusesIncompleteModel(t *testing.T) {
	m := incompleteMainModel(t)
	if err := SaveLatestLaunch(m.cacheDir, &LaunchRequest{ModelPath: m.models[0].Path}); err != nil {
		t.Fatal(err)
	}
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'l'}})
	got := next.(Model)
	requireIncompleteRefusal(t, got)
	if got.replayRequest != nil {
		t.Fatalf("refused replay left a request behind: %#v", got.replayRequest)
	}
}

func TestBackendInstallReopenRefusesIncompleteModel(t *testing.T) {
	modelPath := isolatedModelDir(t)
	writeF32GGUF(t, modelPath, 1<<30, 10)
	m := afterBackendInstallModel(&LaunchRequest{ModelPath: modelPath, Backend: "some-recipe"})
	requireIncompleteRefusal(t, m)
}
