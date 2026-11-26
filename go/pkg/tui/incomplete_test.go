package tui

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
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
	if got := incompleteDownloadNote(full); got != "" {
		t.Fatalf("full file flagged: %q", got)
	}
	short := filepath.Join(dir, "short.gguf")
	writeF32GGUF(t, short, 1024, 4095)
	if got := incompleteDownloadNote(short); !strings.HasPrefix(got, "incomplete download: 0.0 of 0.0 GB") {
		t.Fatalf("short file note = %q", got)
	}

	// Split model: first shard complete, second short.
	for i, data := range []int{4096, 100} {
		writeF32GGUF(t, filepath.Join(dir, fmt.Sprintf("big-%05d-of-00002.gguf", i+1)), 1024, data)
	}
	if got := incompleteDownloadNote(filepath.Join(dir, "big-00001-of-00002.gguf")); got == "" {
		t.Fatal("split model with a short shard not flagged")
	}
	writeF32GGUF(t, filepath.Join(dir, "big-00002-of-00002.gguf"), 1024, 4096)
	if got := incompleteDownloadNote(filepath.Join(dir, "big-00001-of-00002.gguf")); got != "" {
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
