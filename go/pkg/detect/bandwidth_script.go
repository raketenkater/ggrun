package detect

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// measureBandwidthSource is the canonical bandwidth probe. It is embedded so
// that `ggrun detect --bandwidth` works from a bare `go install` binary, with
// no repository, no installer, and no LLM_SCRIPT_DIR. Before this was embedded
// the command failed on every install, which is why measured bandwidth
// evidence was absent in the field and the planner silently fell back to
// theoretical PCIe ceilings.
//
//go:embed scripts/measure_bandwidth.py
var measureBandwidthSource string

var (
	extractOnce sync.Once
	extractPath string
	extractErr  error
)

// embeddedBandwidthScriptDigest identifies the embedded script by content, so
// an upgraded ggrun never executes a stale extraction left by an older build.
func embeddedBandwidthScriptDigest() string {
	sum := sha256.Sum256([]byte(measureBandwidthSource))
	return hex.EncodeToString(sum[:])[:16]
}

// bandwidthScriptCacheDir is the directory extractions live in. It is a
// variable so tests can redirect it without touching the user's real cache.
var bandwidthScriptCacheDir = func() (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "ggrun", "scripts"), nil
}

// extractBandwidthScript materialises the embedded probe on disk and returns
// its path. Extraction is content-addressed and idempotent: a cached copy is
// reused only when its bytes match the embedded source exactly, so a truncated
// or hand-edited cache entry is repaired rather than executed.
func extractBandwidthScript() (string, error) {
	extractOnce.Do(func() {
		extractPath, extractErr = writeEmbeddedBandwidthScript()
	})
	return extractPath, extractErr
}

// errBandwidthScriptUnavailable reports why extraction failed, so the user sees
// the real cause (an unwritable cache, say) instead of a generic "not found"
// that wrongly implies a broken install.
func errBandwidthScriptUnavailable() error {
	if _, err := extractBandwidthScript(); err != nil {
		return err
	}
	return fmt.Errorf("script cache returned no path")
}

func writeEmbeddedBandwidthScript() (string, error) {
	if measureBandwidthSource == "" {
		return "", fmt.Errorf("embedded measure_bandwidth.py is empty; this build is broken")
	}
	dir, err := bandwidthScriptCacheDir()
	if err != nil {
		dir = filepath.Join(os.TempDir(), "ggrun-scripts")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("create script cache %s: %w", dir, err)
	}
	target := filepath.Join(dir, fmt.Sprintf("measure_bandwidth-%s.py", embeddedBandwidthScriptDigest()))

	// Reuse only a byte-identical copy. A partial write from an interrupted
	// run must not be executed as if it were the real probe.
	if existing, err := os.ReadFile(target); err == nil && string(existing) == measureBandwidthSource {
		return target, nil
	}

	// Write through a temp file in the same directory so a concurrent ggrun
	// never observes a half-written script.
	tmp, err := os.CreateTemp(dir, "measure_bandwidth-*.py.partial")
	if err != nil {
		return "", fmt.Errorf("stage script in %s: %w", dir, err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.WriteString(measureBandwidthSource); err != nil {
		tmp.Close()
		return "", fmt.Errorf("write script: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return "", fmt.Errorf("close script: %w", err)
	}
	if err := os.Chmod(tmpName, 0o755); err != nil {
		return "", fmt.Errorf("chmod script: %w", err)
	}
	if err := os.Rename(tmpName, target); err != nil {
		return "", fmt.Errorf("install script at %s: %w", target, err)
	}
	return target, nil
}
