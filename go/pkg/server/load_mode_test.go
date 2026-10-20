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
		// No loading flag means mapped to every ggrun planner. The backend's own
		// default is "auto", which disables mapping when any device lacks mmap
		// support, so the mapped choice is stated rather than inherited.
		{[]string{"s", "-m", "x", "-c", "4096"}, []string{"s", "-m", "x", "-c", "4096", "--load-mode", "mmap"}},
		// Lazy lookup reads are a separate option and pass through unchanged.
		{[]string{"s", "--no-mmap", "--lazy-mode", "on"}, []string{"s", "--lazy-mode", "on", "--load-mode", "none"}},
	} {
		if got := translateLoadMode(tc.in); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("translateLoadMode(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

// An explicit native choice is a user constraint. Passthrough flags follow the
// generated ones, so a mode derived from classic flags alone must never be
// appended after it (PR #93 review: "--load-mode none" became "... mmap").
func TestLoadModeExplicitNativeChoiceIsPreserved(t *testing.T) {
	for _, args := range [][]string{
		{"s", "--load-mode", "none"},
		{"s", "-lm", "mlock"},
		{"s", "--load-mode=none"},
		{"s", "--load-mode", "dio"},
		{"s", "--load-mode", "auto"},
		// The backend's own last-wins rule already decides between two natives.
		{"s", "--load-mode", "none", "--load-mode", "mmap"},
	} {
		if got := translateLoadMode(args); !reflect.DeepEqual(got, args) {
			t.Errorf("explicit load mode rewritten: %v -> %v", args, got)
		}
	}
}

// Generated classic flags and a later explicit native choice fold in argv
// order: the later explicit choice wins and exactly one mode is emitted.
func TestLoadModeFoldsClassicAndNativeInOrder(t *testing.T) {
	for _, tc := range []struct {
		in   []string
		want []string
	}{
		{[]string{"s", "--no-mmap", "--load-mode", "mmap"}, []string{"s", "--load-mode", "mmap"}},
		{[]string{"s", "--no-mmap", "-lm", "dio"}, []string{"s", "--load-mode", "dio"}},
		{[]string{"s", "--load-mode=mmap", "--no-mmap"}, []string{"s", "--load-mode", "none"}},
		{[]string{"s", "--load-mode", "none", "--mlock"}, []string{"s", "--load-mode", "mlock"}},
		{[]string{"s", "--load-mode", "mlock", "--mmap"}, []string{"s", "--load-mode", "mmap+mlock"}},
		{[]string{"s", "--load-mode", "auto", "--mlock"}, []string{"s", "--load-mode", "mmap+mlock"}},
	} {
		if got := translateLoadMode(tc.in); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("translateLoadMode(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

// Translation is applied at every process start (server, fit pass, recovery
// relaunch); applying it to its own output must change nothing.
func TestLoadModeTranslationIsIdempotent(t *testing.T) {
	for _, in := range [][]string{
		{"s", "-m", "x"},
		{"s", "--no-mmap"},
		{"s", "--mlock"},
		{"s", "--no-mmap", "--mlock", "--lazy-mode", "on"},
		{"s", "--load-mode", "none", "--mmap"},
		{"s", "-lm", "dio"},
	} {
		once := translateLoadMode(in)
		if twice := translateLoadMode(once); !reflect.DeepEqual(once, twice) {
			t.Errorf("not idempotent: %v -> %v -> %v", in, once, twice)
		}
	}
}

func TestParseLoadModeStatesMapping(t *testing.T) {
	for _, tc := range []struct {
		args          []string
		mapped, known bool
	}{
		{[]string{"s"}, true, true},
		{[]string{"s", "--no-mmap"}, false, true},
		{[]string{"s", "--load-mode", "none"}, false, true},
		{[]string{"s", "-lm=dio"}, false, true},
		{[]string{"s", "--load-mode", "mmap+mlock"}, true, true},
		{[]string{"s", "--load-mode", "auto"}, false, false},
	} {
		mode, err := ParseLoadMode(tc.args)
		if err != nil {
			t.Fatalf("%v: %v", tc.args, err)
		}
		if mapped, known := mode.Mapped(); mapped != tc.mapped || known != tc.known {
			t.Errorf("%v: mapped=%v known=%v, want %v %v", tc.args, mapped, known, tc.mapped, tc.known)
		}
	}
	if _, err := ParseLoadMode([]string{"s", "--load-mode", "fast"}); err == nil {
		t.Fatal("unknown native value accepted")
	}
	// An unknown value is left for the backend to reject, unchanged.
	in := []string{"s", "--no-mmap", "--load-mode", "fast"}
	if got := translateLoadMode(in); !reflect.DeepEqual(got, in) {
		t.Fatalf("unknown value rewritten: %v", got)
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
