// Package python finds the Python 3 interpreter that ggrun's helper scripts
// (parse_gguf.py, download_any_gguf.py) run with.
package python

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/raketenkater/ggrun/pkg/boundexec"
)

var (
	mu    sync.Mutex
	cache = map[string]string{} // PATH + app interpreter -> verified interpreter

	// executablePath is os.Executable, replaceable in tests.
	executablePath = os.Executable
)

const probeScript = "import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)"

// appInterpreter returns the app-owned venv interpreter that setup installs
// as <dir>/.ggrun-python/bin/python3 next to the ggrun binary, or "". It is
// found from the binary's location rather than PATH so the helper packages
// stay off PATH and child processes (agent sessions, user commands) keep
// seeing the user's own Python.
func appInterpreter() string {
	if runtime.GOOS == "windows" {
		return ""
	}
	exe, err := executablePath()
	if err != nil || exe == "" {
		return ""
	}
	dirs := []string{filepath.Dir(exe)}
	if real, err := filepath.EvalSymlinks(exe); err == nil && filepath.Dir(real) != dirs[0] {
		dirs = append(dirs, filepath.Dir(real))
	}
	for _, dir := range dirs {
		p := filepath.Join(dir, ".ggrun-python", "bin", "python3")
		if info, err := os.Stat(p); err == nil && info.Mode().IsRegular() && info.Mode()&0o111 != 0 {
			return p
		}
	}
	return ""
}

// Command returns a Python 3 interpreter that actually runs, and whether one
// was found. A name that resolves on PATH is not enough: on Windows python3.exe
// is often the Microsoft Store execution alias, which exits 9009 instead of
// running the script. The order matches install.ps1 Resolve-Python so the
// interpreter that received huggingface_hub is the one invoked.
func Command() (string, bool) {
	app := appInterpreter()
	key := os.Getenv("PATH") + "\x00" + app
	mu.Lock()
	defer mu.Unlock()
	if path, ok := cache[key]; ok {
		return path, true
	}
	if app != "" {
		check, finish := boundexec.Command(boundexec.ProbeTimeout, app, "-c", probeScript)
		err := finish(check.Run())
		if err == nil {
			cache[key] = app
			return app, true
		}
		boundexec.Report(err, "falling back to PATH")
	}
	// Only a hit is cached: a long-running ggrun sees Python installed later.
	for _, name := range []string{"python3", "python", "py"} {
		path, err := exec.LookPath(name)
		if err != nil {
			continue
		}
		check, finish := boundexec.Command(boundexec.ProbeTimeout, path, "-c", probeScript)
		err = finish(check.Run())
		if err == nil {
			cache[key] = path
			return path, true
		}
		boundexec.Report(err, "trying the next interpreter")
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
