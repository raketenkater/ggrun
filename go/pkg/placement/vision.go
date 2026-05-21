package placement

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// findOrDownloadMMProj finds a vision projector locally, or downloads from HuggingFace.
// Tries standard mmproj filename patterns in the model directory, then falls back
// to download_any_gguf.py for automated HF download.
func findOrDownloadMMProj(modelPath, cacheDir string) (string, error) {
	modelDir := filepath.Dir(modelPath)

	// 1. Check local standard locations
	candidates := []string{
		filepath.Join(modelDir, "mmproj-F16.gguf"),
		filepath.Join(modelDir, "mmproj-BF16.gguf"),
		filepath.Join(modelDir, "mmproj-F32.gguf"),
		filepath.Join(modelDir, "mmproj.gguf"),
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}

	// 2. Glob for any mmproj/projector files
	entries, err := os.ReadDir(modelDir)
	if err == nil {
		for _, e := range entries {
			name := strings.ToLower(e.Name())
			if !e.IsDir() && strings.HasSuffix(name, ".gguf") &&
				(strings.Contains(name, "mmproj") || strings.Contains(name, "projector")) {
				return filepath.Join(modelDir, e.Name()), nil
			}
		}
	}

	// 3. Try download from HuggingFace
	script := findDownloadScript()
	if script != "" {
		fmt.Fprintf(os.Stderr, "[vision] Downloading mmproj from HuggingFace...\n")
		cmd := exec.Command("python3", script,
			"--repo", "auto",
			"--dir", modelDir,
			"--cache-dir", cacheDir,
			"--mmproj-only",
		)
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err == nil {
			// Re-check after download
			for _, c := range candidates {
				if _, err := os.Stat(c); err == nil {
					return c, nil
				}
			}
		}
	}

	return "", fmt.Errorf("no mmproj found — place mmproj-F16.gguf in %s or use --mmproj <path>", modelDir)
}

func findDownloadScript() string {
	candidates := []string{
		"download_any_gguf.py",
		filepath.Join("..", "download_any_gguf.py"),
	}
	if home := os.Getenv("LLM_SERVER_HOME"); home != "" {
		candidates = append(candidates, filepath.Join(home, "download_any_gguf.py"))
	}
	if exe, err := os.Executable(); err == nil {
		d := filepath.Dir(exe)
		candidates = append(candidates,
			filepath.Join(d, "download_any_gguf.py"),
			filepath.Join(d, "..", "download_any_gguf.py"),
		)
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}
