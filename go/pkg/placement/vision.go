package placement

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/raketenkater/llm-server/pkg/gguf"
)

// findOrDownloadMMProj finds a vision projector locally, or downloads from HuggingFace.
// Validates GGUF metadata for compatibility with the text model.
func findOrDownloadMMProj(modelPath, cacheDir string) (string, error) {
	modelDir := filepath.Dir(modelPath)
	baseName := strings.TrimSuffix(filepath.Base(modelPath), ".gguf")
	// Strip common quant suffixes (Q4_K_M, Q5_K_M, Q8_0, F16, etc.)
	baseNoQuant := baseName
	for _, suffix := range []string{"Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q4_K_L", "Q5_K_S", "Q5_K_M", "Q5_K_L", "Q6_K", "Q8_0", "F16", "F32", "BF16", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL"} {
		if strings.HasSuffix(baseNoQuant, "-"+suffix) {
			baseNoQuant = strings.TrimSuffix(baseNoQuant, "-"+suffix)
			break
		}
	}

	// 1. Check model-specific mmproj first
	for _, ftype := range []string{"F16", "BF16", "F32"} {
		for _, name := range []string{baseName, baseNoQuant} {
			c := filepath.Join(modelDir, fmt.Sprintf("mmproj-%s-%s.gguf", name, ftype))
			if _, err := os.Stat(c); err == nil {
				if err := validateMMProj(c); err == nil {
					return c, nil
				}
			}
		}
	}

	// 2. Check generic mmproj files
	candidates := []string{
		filepath.Join(modelDir, "mmproj-F16.gguf"),
		filepath.Join(modelDir, "mmproj-BF16.gguf"),
		filepath.Join(modelDir, "mmproj-F32.gguf"),
		filepath.Join(modelDir, "mmproj.gguf"),
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			if err := validateMMProj(c); err != nil {
				fmt.Fprintf(os.Stderr, "[vision] rejecting %s: %v\n", filepath.Base(c), err)
				continue
			}
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
				c := filepath.Join(modelDir, e.Name())
				if err := validateMMProj(c); err != nil {
					fmt.Fprintf(os.Stderr, "[vision] rejecting %s: %v\n", e.Name(), err)
					continue
				}
				return c, nil
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
			for _, c := range candidates {
				if _, err := os.Stat(c); err == nil {
					return c, nil
				}
			}
		}
	}

	return "", fmt.Errorf("no mmproj found — place mmproj-F16.gguf in %s or use --mmproj <path>", modelDir)
}

// validateMMProj checks that an mmproj GGUF file is valid (not corrupt, has expected metadata).
func validateMMProj(path string) error {
	info, err := gguf.Parse(path)
	if err != nil {
		return fmt.Errorf("parse failed: %w", err)
	}
	if info == nil {
		return fmt.Errorf("empty metadata")
	}
	if info.Architecture == "" || info.Architecture == "unknown" {
		return fmt.Errorf("unknown architecture")
	}
	// Reject files that look like full models, not projectors
	if info.BlockCount > 32 {
		return fmt.Errorf("looks like a full model (%d layers), not a projector", info.BlockCount)
	}
	return nil
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
