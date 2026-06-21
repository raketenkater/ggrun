package libhub

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Setup creates a temporary directory, symlinks all .so files from the
// backend binary's build tree, and returns the hub path for LD_LIBRARY_PATH.
// If the binary is in a system path (/usr/bin, /usr/local/bin, /bin),
// it returns "", false since system binaries don't need a lib hub.
func Setup(binaryPath string) (string, bool, error) {
	// Skip system paths
	systemPaths := []string{"/usr/bin", "/usr/local/bin", "/bin", "/sbin", "/usr/sbin"}
	binDir := filepath.Dir(binaryPath)
	for _, sp := range systemPaths {
		if strings.HasPrefix(binDir, sp) {
			return "", false, nil
		}
	}

	// Find the build tree (binary is in build/bin/llama-server)
	buildDir := filepath.Dir(binDir) // build/
	if !strings.HasSuffix(buildDir, "build") {
		// Try walking up
		parent := filepath.Dir(buildDir)
		if filepath.Base(parent) == "build" {
			buildDir = parent
		} else {
			// Can't find build tree, skip lib hub
			return "", false, nil
		}
	}

	// Create temp hub directory
	hubDir, err := os.MkdirTemp("", "ggrun-lib-hub-*")
	if err != nil {
		return "", false, fmt.Errorf("create lib hub: %w", err)
	}

	// Walk build tree for .so files
	var symlinked int
	filepath.Walk(buildDir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		if strings.HasSuffix(path, ".so") || strings.Contains(path, ".so.") {
			base := filepath.Base(path)
			dest := filepath.Join(hubDir, base)
			// Remove versioned symlinks and point to actual file
			os.Remove(dest)
			if err := os.Symlink(path, dest); err == nil {
				symlinked++
			}
		}
		return nil
	})

	if symlinked == 0 {
		os.RemoveAll(hubDir)
		return "", false, nil
	}

	return hubDir, true, nil
}

// Cleanup removes the lib hub directory.
func Cleanup(hubDir string) {
	if hubDir != "" {
		os.RemoveAll(hubDir)
	}
}

// Env returns LD_LIBRARY_PATH with the hub prepended.
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
