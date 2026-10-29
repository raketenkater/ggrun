// Package python finds the Python 3 interpreter that ggrun's helper scripts
// (parse_gguf.py, download_any_gguf.py) run with.
package python

import (
	"os"
	"os/exec"
	"runtime"
	"sync"
)

var (
	mu    sync.Mutex
	cache = map[string]string{} // PATH -> verified interpreter
)

// Command returns a Python 3 interpreter that actually runs, and whether one
// was found. A name that resolves on PATH is not enough: on Windows python3.exe
// is often the Microsoft Store execution alias, which exits 9009 instead of
// running the script. The order matches install.ps1 Resolve-Python so the
// interpreter that received huggingface_hub is the one invoked.
func Command() (string, bool) {
	key := os.Getenv("PATH")
	mu.Lock()
	defer mu.Unlock()
	if path, ok := cache[key]; ok {
		return path, true
	}
	// Only a hit is cached: a long-running ggrun sees Python installed later.
	for _, name := range []string{"python3", "python", "py"} {
		path, err := exec.LookPath(name)
		if err != nil {
			continue
		}
		check := exec.Command(path, "-c", "import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)")
		if check.Run() == nil {
			cache[key] = path
			return path, true
		}
	}
	return "", false
}

// InstallHint returns an OS-appropriate one-liner for installing Python 3.
func InstallHint() string {
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
