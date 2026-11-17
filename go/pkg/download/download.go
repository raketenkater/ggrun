package download

import (
	_ "embed"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/embedscript"
	"github.com/raketenkater/ggrun/pkg/python"
	"github.com/raketenkater/ggrun/pkg/recommend"
)

// downloaderSource is tools/download/download_any_gguf.py, embedded so a bare
// binary can download. TestEmbeddedDownloaderMatchesTools keeps them identical.
//
//go:generate cp ../../../tools/download/download_any_gguf.py download_any_gguf.py
//go:embed download_any_gguf.py
var downloaderSource string

// Downloader wraps the download_any_gguf.py script.
type Downloader struct {
	ScriptPath string
	ModelDir   string
	CacheDir   string
	// AssumeYes takes the recommended quant and skips the confirmation.
	AssumeYes bool
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
	// Only locations the user, the installer or the binary's own install
	// name are searched. The working directory and PATH are not: a model
	// directory can hold an older downloader that ignores --repo and waits for
	// input (October 8, `ggrun download` run from AI_Models), or a planted one.
	var candidates []string
	if dir := os.Getenv("LLM_SCRIPT_DIR"); dir != "" {
		candidates = append(candidates,
			filepath.Join(dir, "download_any_gguf.py"),
			filepath.Join(dir, "tools", "download", "download_any_gguf.py"),
		)
	}
	if home := os.Getenv("LLM_SERVER_HOME"); home != "" {
		candidates = append(candidates,
			filepath.Join(home, "download_any_gguf.py"),
			filepath.Join(home, "tools", "download", "download_any_gguf.py"),
		)
	}
	for _, home := range []string{os.Getenv("LLM_APP_HOME"), appHome} {
		if home == "" {
			continue
		}
		candidates = append(candidates,
			filepath.Join(home, ".bin", "download_any_gguf.py"),
			filepath.Join(home, "bin", "download_any_gguf.py"),
			filepath.Join(home, "download_any_gguf.py"),
		)
	}
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, exeScriptCandidates(filepath.Dir(exe))...)
	}
	for _, c := range candidates {
		if info, err := os.Stat(c); err == nil && !info.IsDir() {
			return c
		}
	}
	// Nothing on disk: the copy compiled into the binary.
	if p, err := embedscript.Extract("download_any_gguf.py", downloaderSource); err == nil {
		return p
	}
	return ""
}

// exeScriptCandidates are the downloader locations of the binary's own
// install: beside it, and for a binary in <app-home>/bin or .bin the source
// checkout in that app home. No walk climbs further, so a binary in
// ~/.local/bin never reaches files in the home directory.
func exeScriptCandidates(exeDir string) []string {
	candidates := []string{filepath.Join(exeDir, "download_any_gguf.py")}
	if base := filepath.Base(exeDir); base == ".bin" || base == "bin" {
		candidates = append(candidates,
			filepath.Join(filepath.Dir(exeDir), ".src", "ggrun", "tools", "download", "download_any_gguf.py"))
	}
	return candidates
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

	py, ok := python.Command()
	if !ok {
		return fmt.Errorf("Python 3 is required to download models, but no python interpreter was found on PATH.\n%s", python.InstallHint())
	}
	cmd := exec.Command(py, d.scriptArgs(repo, quant, caps, stdinIsTerminal())...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// scriptArgs builds the downloader argv. caps should be the planning view
// (recommend.PlanningCapabilities): its RAM is the installed ceiling, so the
// default quant matches what the recommender would pick on an idle machine.
func (d *Downloader) scriptArgs(repo, quant string, caps *detect.Capabilities, interactive bool) []string {
	vramMB, ramMB := 0, 0
	if caps != nil {
		vramMB = caps.TotalVRAM()
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
	if active := catalogActiveParamsB(repo); active > 0 {
		args = append(args, "--active-params-b", strconv.FormatFloat(active, 'f', -1, 64))
	}
	if d.AssumeYes {
		args = append(args, "--yes")
	}
	// With no terminal attached a prompt dies on EOFError after the repo and
	// file list have been resolved -- exactly what happens to a scripted or
	// queued download. The script then proceeds only on an explicit --quant
	// (or --yes) and otherwise fails listing the choices.
	if !interactive {
		args = append(args, "--non-interactive")
	}
	return args
}

// catalogActiveParamsB is the catalog's active parameter count for a MoE repo,
// so the downloader does not size an unnamed MoE (no A##B in the repo name)
// as dense. 0 when the repo is not a catalog MoE.
func catalogActiveParamsB(repo string) float64 {
	for _, c := range recommend.Shortlist() {
		if c.MoE && strings.EqualFold(c.Repo, repo) {
			return c.ActiveParamsB
		}
	}
	return 0
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
