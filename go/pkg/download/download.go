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
// appHome is the configured APP_HOME or repo root (may be empty).
func New(modelDir, cacheDir, appHome string) *Downloader {
	return &Downloader{
		ScriptPath: findScript(appHome),
		ModelDir:   modelDir,
		CacheDir:   cacheDir,
	}
}

func findScript(appHome string) string {
	candidates := []string{
		"download_any_gguf.py",
		filepath.Join("tools", "download", "download_any_gguf.py"),
		filepath.Join("..", "download_any_gguf.py"),
		filepath.Join("..", "tools", "download", "download_any_gguf.py"),
		filepath.Join("..", "..", "download_any_gguf.py"),
		filepath.Join("..", "..", "tools", "download", "download_any_gguf.py"),
	}
	// Check LLM_SERVER_HOME env var (repo root)
	if home := os.Getenv("LLM_SERVER_HOME"); home != "" {
		candidates = append(candidates,
			filepath.Join(home, "download_any_gguf.py"),
			filepath.Join(home, "tools", "download", "download_any_gguf.py"),
		)
	}
	// Check configured app home
	if appHome != "" {
		candidates = append(candidates,
			filepath.Join(appHome, ".bin", "download_any_gguf.py"),
			filepath.Join(appHome, "bin", "download_any_gguf.py"),
			filepath.Join(appHome, "download_any_gguf.py"),
		)
	}
	// Try relative to binary (installed alongside llm-server)
	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		candidates = append(candidates,
			filepath.Join(exeDir, "download_any_gguf.py"),
			filepath.Join(exeDir, "..", "download_any_gguf.py"),
			filepath.Join(exeDir, "..", "tools", "download", "download_any_gguf.py"),
			filepath.Join(exeDir, "..", "..", "download_any_gguf.py"),
			filepath.Join(exeDir, "..", "..", "tools", "download", "download_any_gguf.py"),
			filepath.Join(exeDir, "..", "..", "..", "download_any_gguf.py"),
		)
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

// Run executes the downloader for the given repo.
func (d *Downloader) Run(repo string, caps *detect.Capabilities) error {
	return d.RunQuant(repo, "", caps)
}

// RunQuant executes the downloader with an optional preselected quant.
func (d *Downloader) RunQuant(repo string, quant string, caps *detect.Capabilities) error {
	if d.ScriptPath == "" {
		return fmt.Errorf("download_any_gguf.py not found; set LLM_SERVER_HOME to the repo root or install the bundled tools")
	}
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

	args := []string{
		d.ScriptPath,
		"--repo", repo,
		"--dir", d.ModelDir,
		"--cache-dir", d.CacheDir,
		"--vram", strconv.Itoa(vramMB),
		"--ram", strconv.Itoa(ramMB),
	}
	if quant != "" && quant != "auto" && quant != "catalog" {
		args = append(args, "--quant", quant)
	}
	cmd := exec.Command("python3", args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
