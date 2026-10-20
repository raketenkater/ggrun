package server

import (
	"fmt"
	"path/filepath"
	"strings"
	"sync"
)

// Upstream llama.cpp replaced --mmap / --no-mmap / --mlock with one
// --load-mode MODE option. A backend built from a newer source (a discovered
// PR fork, a mainline update) then rejects --no-mmap before model load, and
// ggrun cannot serve on it at all. --no-mmap is not optional: it is how a
// placement keeps CPU experts resident, and host admission is computed for it,
// so it must be translated rather than dropped.
//
// Every planner, admission and recovery path reads the classic spelling from
// the generated argv, so ggrun keeps speaking it internally and translates only
// what the backend process receives. The dialect is recorded per backend
// directory, which also covers sibling tools of the same build (llama-fit-params).
var loadModeDirs sync.Map

// RegisterLoadModeBackend records that binaries in path's directory parse
// --load-mode instead of the classic mmap flags.
func RegisterLoadModeBackend(path string) {
	if path != "" {
		loadModeDirs.Store(filepath.Dir(path), true)
	}
}

func usesLoadMode(binary string) bool {
	_, ok := loadModeDirs.Load(filepath.Dir(binary))
	return ok
}

// LoadModeArgs rewrites a backend argv (binary at index 0) for a --load-mode
// backend. Other backends get args unchanged.
func LoadModeArgs(args []string) []string {
	if len(args) == 0 || !usesLoadMode(args[0]) {
		return args
	}
	return translateLoadMode(args)
}

// Native --load-mode values. "auto" maps unless a device lacks mmap support;
// "dio" reads with direct I/O and does not map.
var loadModeValues = map[string]bool{
	"auto": true, "none": true, "mmap": true, "mlock": true, "mmap+mlock": true, "dio": true,
}

// LoadMode is the loading state an argv selects, folded in argv order over
// both spellings. Classic flags compose (--no-mmap then --mlock is "mlock");
// a native --load-mode value replaces the whole state. The last spelling wins,
// as in the backend's own parser.
type LoadMode struct {
	Base   string // "", "auto", "none", "mmap" or "dio"; "" means no choice was stated
	MLock  bool
	Native int // --load-mode / -lm occurrences
	Legacy int // --mmap / --no-mmap / --mlock occurrences
}

// ParseLoadMode folds every loading flag in args. An unknown native value is
// an error; the caller decides whether to refuse or pass it to the backend.
func ParseLoadMode(args []string) (LoadMode, error) {
	var m LoadMode
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--mmap":
			m.Base, m.Legacy = "mmap", m.Legacy+1
			continue
		case "--no-mmap":
			m.Base, m.Legacy = "none", m.Legacy+1
			continue
		case "--mlock":
			m.MLock, m.Legacy = true, m.Legacy+1
			continue
		}
		value, ok := nativeLoadModeValue(args, i)
		if !ok {
			continue
		}
		if !strings.Contains(args[i], "=") {
			i++
		}
		m.Native++
		if !loadModeValues[value] {
			return m, fmt.Errorf("unknown --load-mode %q (want auto, none, mmap, mlock, mmap+mlock or dio)", value)
		}
		m.Base, m.MLock = value, false
		switch value {
		case "mlock":
			m.Base, m.MLock = "none", true
		case "mmap+mlock":
			m.Base, m.MLock = "mmap", true
		}
	}
	return m, nil
}

// nativeLoadModeValue returns the value of a --load-mode / -lm token at i.
func nativeLoadModeValue(args []string, i int) (string, bool) {
	flag, value, inline := strings.Cut(args[i], "=")
	if flag != "--load-mode" && flag != "-lm" {
		return "", false
	}
	if inline {
		return value, true
	}
	if i+1 < len(args) {
		return args[i+1], true
	}
	return "", true
}

// Stated reports whether args choose a loading mode at all.
func (m LoadMode) Stated() bool { return m.Base != "" || m.MLock }

// Spelling is the native value for the folded state. With no stated mapping
// choice the classic default (mapped) applies, as every ggrun planner assumes.
func (m LoadMode) Spelling() string {
	base := m.Base
	if base == "" {
		base = "mmap"
	}
	if !m.MLock {
		return base
	}
	switch base {
	case "none":
		return "mlock"
	case "mmap", "auto":
		// auto resolves to mmap on every device that supports it, which is the
		// only state in which locking mapped pages means anything.
		return "mmap+mlock"
	}
	return base
}

// Mapped reports whether the folded state maps the model file. "auto" is
// mapped unless a device lacks mmap support, so it is not proof either way.
func (m LoadMode) Mapped() (mapped, known bool) {
	switch m.Base {
	case "", "mmap":
		return true, true
	case "none", "dio":
		return false, true
	}
	return false, false
}

// translateLoadMode maps the classic flags onto llama.cpp's load modes:
// mmap unless --no-mmap is given, and --mlock is its own mode when mmap is off
// ("mlock") or combined with it ("mmap+mlock").
//
// Native and classic spellings are folded together in argv order, so a user's
// explicit --load-mode (passed through after the generated flags) is never
// overridden by a mode derived only from the classic ones. An argv that
// already states its mode natively and carries no classic flag is returned
// unchanged, which makes the translation idempotent.
//
// The absent-flag case is translated too. ggrun's planners read "no --no-mmap"
// as file-backed, while these backends default to "auto", which turns mapping
// off when any device lacks mmap support. Leaving the default in place let a
// plan priced as reclaimable load as anonymous memory.
func translateLoadMode(args []string) []string {
	mode, err := ParseLoadMode(args)
	if err != nil || (mode.Legacy == 0 && mode.Native > 0) {
		// An unknown native value reaches the backend, which reports it.
		return args
	}
	out := make([]string, 0, len(args)+2)
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--mmap", "--no-mmap", "--mlock":
			continue
		}
		if _, ok := nativeLoadModeValue(args, i); ok {
			if !strings.Contains(args[i], "=") {
				i++
			}
			continue
		}
		out = append(out, args[i])
	}
	return append(out, "--load-mode", mode.Spelling())
}
