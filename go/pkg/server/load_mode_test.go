package server

import (
	"path/filepath"
	"reflect"
	"testing"
)

// A backend built from current llama.cpp (the discovered qwen4exp PR fork,
// 2026-09-23) rejected "--no-mmap" before model load: upstream folded
// --mmap/--no-mmap/--mlock into --load-mode. Dropping the flag would silently
// page CPU experts that placement and host admission assumed resident, so the
// translation must preserve the loading semantics exactly.
func TestLoadModeTranslationPreservesResidency(t *testing.T) {
	for _, tc := range []struct {
		in   []string
		want []string
	}{
		{[]string{"s", "-m", "x", "--no-mmap"}, []string{"s", "-m", "x", "--load-mode", "none"}},
		{[]string{"s", "--no-mmap", "--mlock"}, []string{"s", "--load-mode", "mlock"}},
		{[]string{"s", "--mlock"}, []string{"s", "--load-mode", "mmap+mlock"}},
		{[]string{"s", "--mmap", "--mlock"}, []string{"s", "--load-mode", "mmap+mlock"}},
		{[]string{"s", "--mmap"}, []string{"s", "--load-mode", "mmap"}},
		// Last spelling wins, as in llama.cpp's own parser.
		{[]string{"s", "--mmap", "--no-mmap"}, []string{"s", "--load-mode", "none"}},
		// No loading flag: the backend default (auto = mmap) is the classic default.
		{[]string{"s", "-m", "x", "-c", "4096"}, []string{"s", "-m", "x", "-c", "4096"}},
	} {
		if got := translateLoadMode(tc.in); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("translateLoadMode(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

// Only a backend whose own --help proved the new dialect is rewritten: ik_llama
// and older mainline builds still require --no-mmap, and a sibling tool of a
// registered build (llama-fit-params) parses the same options.
func TestLoadModeArgsOnlyRewritesRegisteredBackends(t *testing.T) {
	newDir, oldDir := t.TempDir(), t.TempDir()
	RegisterLoadModeBackend(filepath.Join(newDir, "llama-server"))

	classic := []string{filepath.Join(oldDir, "llama-server"), "--no-mmap"}
	if got := LoadModeArgs(classic); !reflect.DeepEqual(got, classic) {
		t.Fatalf("unregistered backend rewritten: %v", got)
	}
	for _, bin := range []string{"llama-server", "llama-fit-params"} {
		got := LoadModeArgs([]string{filepath.Join(newDir, bin), "--no-mmap"})
		want := []string{filepath.Join(newDir, bin), "--load-mode", "none"}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%s: got %v, want %v", bin, got, want)
		}
	}
}
