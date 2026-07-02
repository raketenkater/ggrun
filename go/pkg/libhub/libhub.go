package libhub

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// isLib reports whether name is a shared library ggrun should hub
// (.so/.so.N on Linux, .dylib on macOS).
func isLib(name string) bool {
	return strings.HasSuffix(name, ".so") ||
		strings.Contains(name, ".so.") ||
		strings.HasSuffix(name, ".dylib")
}

// dirHasLibs reports whether dir directly contains any shared library.
func dirHasLibs(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if !e.IsDir() && isLib(e.Name()) {
			return true
		}
	}
	return false
}

// Setup builds a temporary directory of symlinks to the backend's shared
// libraries and returns its path (for LD_LIBRARY_PATH). Modern llama.cpp/
// ik_llama.cpp builds ship a thin server binary plus libggml*/libllama/libmtmd
// shared objects, and the binary's baked RUNPATH points at the *build machine's*
// path — absent on a user's box. Without this, an installed release fails to
// load its own libraries ("cannot open shared object file").
//
// It resolves symlinks first (ggrun's backend is usually .bin/llama-server-cuda
// pointing into a build tree), then looks for libraries in, in order: the
// binary's own directory (flat release bundle, or a co-located build/bin), then
// the enclosing build/ tree (dev checkouts where libraries are scattered across
// build subdirectories). Returns "", false when the binary sits in a system path
// or no libraries are found (a static binary needs no hub).
func Setup(binaryPath string) (string, bool, error) {
	// Resolve symlinks so a .bin/ symlink resolves to the real build tree.
	resolved := binaryPath
	if r, err := filepath.EvalSymlinks(binaryPath); err == nil {
		resolved = r
	}
	binDir := filepath.Dir(resolved)

	// Skip system paths — the dynamic loader already finds system libraries.
	for _, sp := range []string{"/usr/bin", "/usr/local/bin", "/bin", "/sbin", "/usr/sbin", "/usr/lib", "/usr/lib64"} {
		if binDir == sp || strings.HasPrefix(binDir, sp+"/") {
			return "", false, nil
		}
	}

	// Pick the directory tree to harvest libraries from.
	var searchRoot string
	switch {
	case dirHasLibs(binDir):
		// Flat release bundle, or a build whose libs sit beside the binary
		// (e.g. build/bin). This is the common installed-user case.
		searchRoot = binDir
	default:
		// Dev checkout: libraries scattered across the build/ tree. Walk up to
		// the build directory and search the whole tree.
		buildDir := filepath.Dir(binDir) // e.g. build/ when binary is build/bin/x
		if filepath.Base(buildDir) != "build" {
			if parent := filepath.Dir(buildDir); filepath.Base(parent) == "build" {
				buildDir = parent
			} else {
				return "", false, nil
			}
		}
		searchRoot = buildDir
	}

	hubDir, err := os.MkdirTemp("", "ggrun-lib-hub-*")
	if err != nil {
		return "", false, fmt.Errorf("create lib hub: %w", err)
	}

	// Symlink every library found under searchRoot into the hub. Symlinking (not
	// copying) keeps it cheap; the hub is a throwaway temp dir so Cleanup can
	// safely remove it without touching real build artifacts. First name wins so
	// a shallow copy beside the binary shadows a duplicate deeper in the tree.
	var symlinked int
	seen := make(map[string]bool)
	filepath.Walk(searchRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !isLib(info.Name()) {
			return nil
		}
		base := info.Name()
		if seen[base] {
			return nil
		}
		if err := os.Symlink(path, filepath.Join(hubDir, base)); err == nil {
			seen[base] = true
			symlinked++
		}
		return nil
	})

	if symlinked == 0 {
		os.RemoveAll(hubDir)
		return "", false, nil
	}
	return hubDir, true, nil
}

// Cleanup removes the lib hub directory. Safe because Setup only ever returns a
// temp directory of symlinks, never a real library directory.
func Cleanup(hubDir string) {
	if hubDir != "" {
		os.RemoveAll(hubDir)
	}
}

// Env returns LD_LIBRARY_PATH with the hub prepended (process environment form).
func Env(hubDir string) string {
	if hubDir == "" {
		return os.Getenv("LD_LIBRARY_PATH")
	}
	old := os.Getenv("LD_LIBRARY_PATH")
	if old == "" {
		return hubDir
	}
	return hubDir + ":" + old
}

// ApplyToChildEnv prepends the configured lib hub (from the LLM_SERVER_LIB_HUB
// env var that Setup's caller exports) to LD_LIBRARY_PATH within a child-process
// environment slice. Both backend launch paths (server and recovery) call this so
// a shared-library backend build finds its co-located libraries regardless of the
// binary's baked RUNPATH. No-op when no hub is configured.
func ApplyToChildEnv(env []string) []string {
	hub := os.Getenv("LLM_SERVER_LIB_HUB")
	if hub == "" {
		return env
	}
	for i, e := range env {
		if strings.HasPrefix(e, "LD_LIBRARY_PATH=") {
			old := strings.TrimPrefix(e, "LD_LIBRARY_PATH=")
			if old == "" {
				env[i] = "LD_LIBRARY_PATH=" + hub
			} else {
				env[i] = "LD_LIBRARY_PATH=" + hub + ":" + old
			}
			return env
		}
	}
	return append(env, "LD_LIBRARY_PATH="+hub)
}
