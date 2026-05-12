package download

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"

	"github.com/raketenkater/llm-server/pkg/detect"
)

// Downloader wraps the download_any_gguf.py script.
type Downloader struct {
	ScriptPath string
	ModelDir   string
	CacheDir   string
}

// New creates a Downloader with auto-discovered script path.
func New(modelDir, cacheDir string) *Downloader {
	return &Downloader{
		ScriptPath: findScript(),
		ModelDir:   modelDir,
		CacheDir:   cacheDir,
	}
}

func findScript() string {
	candidates := []string{
		"download_any_gguf.py",
		filepath.Join("..", "download_any_gguf.py"),
		filepath.Join("..", "..", "download_any_gguf.py"),
	}
	// Try relative to binary
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates,
			filepath.Join(filepath.Dir(exe), "..", "..", "download_any_gguf.py"),
			filepath.Join(filepath.Dir(exe), "download_any_gguf.py"),
		)
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return "download_any_gguf.py"
}

// Run executes the downloader for the given repo.
func (d *Downloader) Run(repo string, caps *detect.Capabilities) error {
	if _, err := os.Stat(d.ScriptPath); os.IsNotExist(err) {
		return fmt.Errorf("downloader script not found: %s", d.ScriptPath)
	}

	vramMB := 0
	if caps != nil {
		vramMB = caps.TotalVRAM()
	}
	ramMB := 0
	if caps != nil {
		ramMB = caps.RAM.FreeMB
	}

	cmd := exec.Command("python3", d.ScriptPath,
		"--repo", repo,
		"--dir", d.ModelDir,
		"--cache-dir", d.CacheDir,
		"--vram", strconv.Itoa(vramMB),
		"--ram", strconv.Itoa(ramMB),
	)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
