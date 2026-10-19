package server

import (
	"path/filepath"
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

// translateLoadMode maps the classic flags onto llama.cpp's load modes:
// mmap unless --no-mmap is given, and --mlock is its own mode when mmap is off
// ("mlock") or combined with it ("mmap+mlock").
//
// The absent-flag case is translated too. ggrun's planners read "no --no-mmap"
// as file-backed, while these backends default to "auto", which turns mapping
// off when any device lacks mmap support. Leaving the default in place let a
// plan priced as reclaimable load as anonymous memory.
func translateLoadMode(args []string) []string {
	mmap, mlock := true, false
	out := make([]string, 0, len(args)+1)
	for _, a := range args {
		switch a {
		case "--no-mmap":
			mmap = false
		case "--mmap":
			mmap = true
		case "--mlock":
			mlock = true
		default:
			out = append(out, a)
			continue
		}
	}
	mode := "mmap"
	switch {
	case mlock && mmap:
		mode = "mmap+mlock"
	case mlock:
		mode = "mlock"
	case !mmap:
		mode = "none"
	}
	return append(out, "--load-mode", mode)
}
