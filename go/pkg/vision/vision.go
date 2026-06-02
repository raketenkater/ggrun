package vision

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// FindLocal searches for a compatible mmproj file near the model.
func FindLocal(modelPath string) (string, error) {
	dir := filepath.Dir(modelPath)
	modelName := strings.ToLower(filepath.Base(modelPath))

	// Preferred order: F16, BF16, F32
	candidates := []string{
		filepath.Join(dir, "mmproj-F16.gguf"),
		filepath.Join(dir, "mmproj-BF16.gguf"),
		filepath.Join(dir, "mmproj-F32.gguf"),
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			if matchesModel(c, modelName) {
				return c, nil
			}
		}
	}

	// Glob for any mmproj in same dir
	matches, _ := filepath.Glob(filepath.Join(dir, "*mmproj*.gguf"))
	for _, m := range matches {
		if matchesModel(m, modelName) {
			return m, nil
		}
	}

	return "", fmt.Errorf("no mmproj found for %s", modelPath)
}

// matchesModel checks if mmproj filename matches the model name.
func matchesModel(mmprojPath, modelName string) bool {
	name := strings.ToLower(filepath.Base(mmprojPath))
	base := modelBase(modelName)

	// Generic mmproj in same directory is always considered compatible
	// (users typically keep matching mmproj with model)
	if name == "mmproj-f16.gguf" || name == "mmproj-bf16.gguf" || name == "mmproj-f32.gguf" {
		return true
	}

	// Otherwise check if names share a common base
	mmprojBase := modelBase(name)
	return strings.Contains(mmprojBase, base) || strings.Contains(base, mmprojBase) ||
		strings.Contains(name, base) || strings.Contains(base, name)
}

func modelBase(name string) string {
	// Remove common quant suffixes
	name = strings.ToLower(name)
	for _, suffix := range []string{"-q4_k_m", "-q5_k_m", "-q6_k", "-q8_0", "-f16", "-f32", "-bf16", "-q4_0", "-q5_0", ".gguf"} {
		name = strings.TrimSuffix(name, suffix)
	}
	return name
}

// Label returns a human-readable label for the mmproj file.
func Label(path string) string {
	name := strings.ToLower(filepath.Base(path))
	if strings.Contains(name, "bf16") {
		return "bf16"
	}
	if strings.Contains(name, "f16") {
		return "f16"
	}
	if strings.Contains(name, "f32") {
		return "f32"
	}
	return "unknown"
}
