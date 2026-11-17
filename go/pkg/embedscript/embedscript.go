// Package embedscript materialises helper scripts that are compiled into the
// ggrun binary, so a bare binary can run its Python helpers without any file
// beside it and without trusting whatever sits in the working directory.
package embedscript

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// CacheDir is where extractions live. Tests redirect it.
var CacheDir = func() (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "ggrun", "scripts"), nil
}

var mu sync.Mutex

// Digest identifies source by content.
func Digest(source string) string {
	sum := sha256.Sum256([]byte(source))
	return hex.EncodeToString(sum[:])[:16]
}

// Extract writes source to the cache as <stem>-<digest><ext> and returns its
// path. A cached copy is reused only when its bytes match source exactly, so a
// truncated or edited cache entry is replaced rather than executed.
func Extract(name, source string) (string, error) {
	if source == "" {
		return "", fmt.Errorf("embedded %s is empty; this build is broken", name)
	}
	mu.Lock()
	defer mu.Unlock()
	dir, err := CacheDir()
	if err != nil {
		dir = filepath.Join(os.TempDir(), "ggrun-scripts")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("create script cache %s: %w", dir, err)
	}
	ext := filepath.Ext(name)
	stem := strings.TrimSuffix(name, ext)
	target := filepath.Join(dir, fmt.Sprintf("%s-%s%s", stem, Digest(source), ext))
	if data, err := os.ReadFile(target); err == nil && string(data) == source {
		return target, nil
	}
	tmp, err := os.CreateTemp(dir, stem+"-*"+ext+".partial")
	if err != nil {
		return "", fmt.Errorf("stage %s in %s: %w", name, dir, err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.WriteString(source); err != nil {
		tmp.Close()
		return "", fmt.Errorf("write %s: %w", name, err)
	}
	if err := tmp.Close(); err != nil {
		return "", fmt.Errorf("close %s: %w", name, err)
	}
	if err := os.Chmod(tmpName, 0o755); err != nil {
		return "", fmt.Errorf("chmod %s: %w", name, err)
	}
	if err := os.Rename(tmpName, target); err != nil {
		return "", fmt.Errorf("install %s at %s: %w", name, target, err)
	}
	return target, nil
}
