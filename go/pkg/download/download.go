package download

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
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
	// The installed copy wins over whatever sits in the working directory: a
	// model directory can hold an older downloader that ignores --repo and
	// waits for input (October 8, `ggrun download` run from AI_Models).
	var candidates []string
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
	// Try relative to binary (installed alongside ggrun)
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
	// A source checkout run from the repo or go/ directory.
	candidates = append(candidates,
		filepath.Join("tools", "download", "download_any_gguf.py"),
		filepath.Join("..", "tools", "download", "download_any_gguf.py"),
		filepath.Join("..", "..", "tools", "download", "download_any_gguf.py"),
		"download_any_gguf.py",
		filepath.Join("..", "download_any_gguf.py"),
		filepath.Join("..", "..", "download_any_gguf.py"),
	)
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
	// The downloader ends with a "Start download? (y/n)" prompt. With no terminal
	// attached that raises EOFError and kills the run after the repo and file
	// list have already been resolved -- which is exactly what happens to a
	// scripted or queued download. Confirming is only meaningful when there is
	// someone there to confirm, so skip it precisely when stdin is not a TTY and
	// leave interactive behaviour untouched.
	if !stdinIsTerminal() {
		args = append(args, "--yes")
	}
	py, ok := pythonCommand()
	if !ok {
		return fmt.Errorf("Python 3 is required to download models, but no python interpreter was found on PATH.\n%s", pythonInstallHint())
	}
	cmd := exec.Command(py, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// pythonCommand returns the path to a usable Python 3 interpreter and whether
// one was found. On Windows the launcher "py" resolves the newest Python 3.
func pythonCommand() (string, bool) {
	for _, name := range []string{"python3", "python", "py"} {
		path, err := exec.LookPath(name)
		if err != nil {
			continue
		}
		// Validate it actually runs as Python 3, skipping Windows Store execution
		// aliases and dead shims that resolve on PATH but don't run. This mirrors
		// install.ps1 Resolve-Python so the interpreter that received
		// huggingface_hub is the one we invoke download_any_gguf.py with.
		check := exec.Command(path, "-c", "import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)")
		if check.Run() == nil {
			return path, true
		}
	}
	return "", false
}

// pythonInstallHint returns an OS-appropriate one-liner for installing Python 3.
func pythonInstallHint() string {
	switch runtime.GOOS {
	case "windows":
		return "Install it with:  winget install -e --id Python.Python.3.12\n" +
			"  or from https://www.python.org/downloads/ (tick \"Add python.exe to PATH\"), then reopen the terminal."
	case "darwin":
		return "Install it with:  brew install python   (or from https://www.python.org/downloads/)."
	default:
		return "Install it with your package manager, e.g.:  sudo apt install python3 python3-pip"
	}
}

// stdinIsTerminal reports whether standard input is an interactive terminal,
// i.e. whether a confirmation prompt could actually be answered.
//
// The obvious ModeCharDevice test is not enough: /dev/null is a character
// device too, and redirecting from /dev/null is precisely what a scripted run
// does. Treating that as a terminal is what let the Inkling download die on
// EOFError after it had already resolved the repo and file list. On Linux the
// fd's target names the device, which separates a real tty from /dev/null;
// elsewhere fall back to the mode test.
func stdinIsTerminal() bool {
	info, err := os.Stdin.Stat()
	if err != nil {
		return false
	}
	if info.Mode()&os.ModeCharDevice == 0 {
		return false // pipe or regular file: certainly not interactive
	}
	if target, err := os.Readlink("/proc/self/fd/0"); err == nil {
		return strings.HasPrefix(target, "/dev/pts/") ||
			strings.HasPrefix(target, "/dev/tty") ||
			strings.HasPrefix(target, "/dev/console")
	}
	return true
}
