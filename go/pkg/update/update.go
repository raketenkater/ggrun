package update

import (
	"bufio"
	"context"
	"crypto/md5"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/backends"

	"github.com/charmbracelet/x/term"
)

const (
	githubRepo            = "raketenkater/ggrun"
	githubAPIURL          = "https://api.github.com/repos/%s/releases/latest"
	rawInstallURL         = "https://raw.githubusercontent.com/%s/%s/install.sh"
	rawInstallPSURL       = "https://raw.githubusercontent.com/%s/%s/install.ps1"
	githubReleaseAssetURL = "https://github.com/%s/releases/download/%s/%s"
	updateDismissDays     = 7
	maxInstallerBytes     = 2 << 20
	maxChecksumBytes      = 64 << 10
	maxDownloadBytes      = 32 << 20
)

// currentVersion is the single source of truth for the binary version.
// Release builds override it: go build -ldflags "-X github.com/raketenkater/ggrun/pkg/update.currentVersion=vX.Y.Z"
var currentVersion = "v3.2.0-go"

// PromptOnStartup checks local repos for updates and asks interactive users
// whether to run the updater. It intentionally skips non-interactive shells so
// scripts and CI never block on network or stdin.
func PromptOnStartup() {
	PromptOnStartupWithBackendUpdater(nil)
}

// PromptOnStartupWithBackendUpdater lets the command package extend the update
// transaction with registered forks while this package keeps terminal/dismiss
// policy in one place. A nil callback preserves the generic-backend behavior.
func PromptOnStartupWithBackendUpdater(updateBackends func() error) {
	if updateBackends == nil {
		updateBackends = UpdateBackends
	}
	if os.Getenv("LLM_SERVER_UPDATE_CHECKED") != "" || os.Getenv("LLM_SERVER_NO_UPDATE_CHECK") != "" {
		return
	}
	if !isTerminal(os.Stdin) && !isTerminal(os.Stdout) {
		return
	}
	cacheDir := updateCacheDir()
	if !shouldCheckStartupUpdates(cacheDir, time.Now()) {
		return
	}
	_ = os.Setenv("LLM_SERVER_UPDATE_CHECKED", "1")

	updates := CheckStartupUpdates()
	if len(updates) == 0 {
		return
	}

	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		return
	}
	defer tty.Close()

	fmt.Fprintf(tty, "\nUpdates available: %s\n", strings.Join(updates, ", "))
	fmt.Fprintf(tty, "Update now? [y/N/d=dismiss %d days] ", updateDismissDays)
	answer := strings.ToLower(strings.TrimSpace(readAnswerWithTimeout(tty, 20*time.Second)))
	switch answer {
	case "y", "yes":
		fmt.Fprintln(tty, "Running --update...")
		if err := SelfUpdate(); err != nil {
			fmt.Fprintf(os.Stderr, "Self-update: %v\n", err)
		}
		if runtime.GOOS != "windows" {
			if err := updateBackends(); err != nil {
				fmt.Fprintf(os.Stderr, "Backend update: %v\n", err)
			}
		}
	case "d", "dismiss":
		if err := dismissStartupUpdates(cacheDir, time.Now()); err != nil {
			fmt.Fprintf(os.Stderr, "Update dismiss: %v\n", err)
			return
		}
		fmt.Fprintf(tty, "Dismissed for %d days.\n", updateDismissDays)
	default:
		fmt.Fprintln(tty, "Skipped.")
	}
}

// CheckStartupUpdates returns both source-checkout and latest-release updates.
func CheckStartupUpdates() []string {
	updates := CheckRepoUpdates()
	if hasUpdateLabel(updates, "ggrun") {
		return updates
	}
	res, err := Check()
	if err == nil && res.HasUpdate {
		updates = append(updates, "ggrun "+res.Latest)
	}
	return updates
}

func hasUpdateLabel(updates []string, label string) bool {
	for _, u := range updates {
		if u == label || strings.HasPrefix(u, label+" ") {
			return true
		}
	}
	return false
}

// CheckRepoUpdates returns local git repos that are behind their upstreams.
func CheckRepoUpdates() []string {
	updates := []string{}
	seen := map[string]bool{}
	for _, repo := range updateRepoCandidates() {
		if seen[repo.Label] {
			continue // same backend can be checked in several dirs; report it once
		}
		if repoBehind(repo.Dir) {
			updates = append(updates, repo.Label)
			seen[repo.Label] = true
		}
	}
	return updates
}

type repoCandidate struct {
	Label string
	Dir   string
}

func homeDir() string {
	if home := strings.TrimSpace(os.Getenv("HOME")); home != "" {
		return home
	}
	home, _ := os.UserHomeDir()
	return home
}

func updateRepoCandidates() []repoCandidate {
	home := homeDir()
	seen := map[string]bool{}
	candidates := []repoCandidate{}
	add := func(label, dir string) {
		dir = strings.TrimSpace(dir)
		if dir == "" || seen[label+"\x00"+dir] {
			return
		}
		seen[label+"\x00"+dir] = true
		candidates = append(candidates, repoCandidate{Label: label, Dir: dir})
	}

	repoDir := os.Getenv("LLM_SERVER_REPO")
	if repoDir == "" && home != "" {
		repoDir = filepath.Join(home, "ggrun")
	}
	add("ggrun", repoDir)

	if server := os.Getenv("LLAMA_SERVER"); server != "" {
		root := filepath.Dir(filepath.Dir(filepath.Dir(server)))
		base := filepath.Base(root)
		if strings.Contains(base, "ik_llama") {
			add("ik_llama.cpp", root)
		} else if strings.Contains(base, "llama.cpp") {
			add("llama.cpp", root)
		}
	}
	if appHome := os.Getenv("LLM_APP_HOME"); appHome != "" {
		add("ggrun", filepath.Join(appHome, ".src", "ggrun"))
		add("ik_llama.cpp", filepath.Join(appHome, ".src", "ik_llama.cpp"))
		add("llama.cpp", filepath.Join(appHome, ".src", "llama.cpp"))
		for _, target := range BackendBuildTargetsAt(appHome) {
			add(target.Label, target.RepoDir)
		}
	}
	// The app home is normally tied to a source checkout the update layer can
	// pull directly (a release-bundle app home holds no .git and is skipped).
	// backends.AppHome() resolves the same pointer/discovery chain cmdUpdate
	// already trusts for backend updates, so ggrun self-update stops missing a
	// source tree that lives outside ~/ggrun (e.g. ~/ggrun-project/ggrun).
	if appHome := backends.AppHome(); appHome != "" {
		if repo := repoFromAppHome(appHome); repo != "" {
			add("ggrun", repo)
		} else if repoDir := filepath.Join(appHome, ".src", "ggrun"); repoDir != "" {
			add("ggrun", repoDir)
		}
	}
	if home != "" {
		add("ik_llama.cpp", filepath.Join(home, "ik_llama.cpp"))
		add("llama.cpp", filepath.Join(home, "llama.cpp"))
	}
	return candidates
}

func repoBehind(repoDir string) bool {
	if _, err := os.Stat(filepath.Join(repoDir, ".git")); err != nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := exec.CommandContext(ctx, "git", "-C", repoDir, "remote", "update", "--prune").Run(); err != nil {
		return false
	}
	localHead, err := gitRevParse(repoDir, "HEAD")
	if err != nil || localHead == "" {
		return false
	}
	remoteHead, err := gitRevParse(repoDir, "@{u}")
	if err != nil || remoteHead == "" {
		return false
	}
	return localHead != remoteHead
}

func readAnswerWithTimeout(tty *os.File, timeout time.Duration) string {
	answers := make(chan string, 1)
	go func() {
		line, _ := bufio.NewReader(tty).ReadString('\n')
		answers <- line
	}()
	select {
	case answer := <-answers:
		return answer
	case <-time.After(timeout):
		fmt.Fprintln(tty)
		return ""
	}
}

func shouldCheckStartupUpdates(cacheDir string, now time.Time) bool {
	data, err := os.ReadFile(updateDismissPath(cacheDir))
	if err != nil {
		return true
	}
	dismissedAt, err := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		return true
	}
	return now.Sub(time.Unix(dismissedAt, 0)) >= time.Duration(updateDismissDays)*24*time.Hour
}

func dismissStartupUpdates(cacheDir string, now time.Time) error {
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return err
	}
	return os.WriteFile(updateDismissPath(cacheDir), []byte(strconv.FormatInt(now.Unix(), 10)+"\n"), 0644)
}

func updateDismissPath(cacheDir string) string {
	return filepath.Join(cacheDir, "update_dismissed")
}

func updateCacheDir() string {
	if dir := os.Getenv("LLM_CACHE_DIR"); dir != "" {
		return dir
	}
	if home := homeDir(); home != "" {
		return filepath.Join(home, ".cache", "ggrun")
	}
	return filepath.Join(os.TempDir(), "ggrun")
}

// isTerminal is a real terminal check. A ModeCharDevice test also accepts
// /dev/null, which is what a scripted run redirects stdin from.
func isTerminal(f *os.File) bool {
	if f == nil {
		return false
	}
	return term.IsTerminal(f.Fd())
}

// Release holds GitHub release info.
type Release struct {
	TagName string `json:"tag_name"`
	HTMLURL string `json:"html_url"`
}

// Result holds the outcome of an update check.
type Result struct {
	Current   string
	Latest    string
	HasUpdate bool
	URL       string
}

// Check queries GitHub for the latest release and compares it to current.
func Check() (*Result, error) {
	current := Version()
	url := fmt.Sprintf(githubAPIURL, githubRepo)
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("fetch release: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("github api %s", resp.Status)
	}
	var rel Release
	if err := json.NewDecoder(resp.Body).Decode(&rel); err != nil {
		return nil, fmt.Errorf("decode release: %w", err)
	}
	latest := rel.TagName
	// Simple semver comparison: strip v prefix, split by dots, compare numeric parts
	hasUpdate := compareVersions(current, latest) < 0
	return &Result{
		Current:   current,
		Latest:    latest,
		HasUpdate: hasUpdate,
		URL:       rel.HTMLURL,
	}, nil
}

// Version returns the current version string.
func Version() string {
	if v := os.Getenv("LLM_SERVER_VERSION"); v != "" {
		return v
	}
	return currentVersion
}

// SelfUpdate pulls the latest ggrun from git and re-runs install.sh.
func SelfUpdate() error {
	if runtime.GOOS == "windows" {
		if appHome := strings.TrimSpace(os.Getenv("LLM_APP_HOME")); appHome != "" {
			return SelfUpdateAppHomeInstaller(appHome)
		}
		return SelfUpdateFromReleaseInstaller()
	}
	repoDir := installedSourceRepoDir()
	gitDir := filepath.Join(repoDir, ".git")
	if _, err := os.Stat(gitDir); err != nil {
		if appHome := os.Getenv("LLM_APP_HOME"); appHome != "" {
			fmt.Printf("ggrun repo not found at %s; refreshing app home from main.\n", repoDir)
			return SelfUpdateAppHomeInstaller(appHome)
		}
		fmt.Printf("ggrun repo not found at %s; using latest release installer.\n", repoDir)
		return SelfUpdateFromReleaseInstaller()
	}
	if dirty, _ := gitStatusPorcelain(repoDir); dirty != "" {
		return fmt.Errorf("source checkout %s has tracked changes; commit or stash them before self-update", repoDir)
	}

	fmt.Println("═══ Updating ggrun ═══")
	oldHash, err := gitRevParse(repoDir, "HEAD")
	if err != nil {
		oldHash = "unknown"
	}

	scriptPath := installedLLMServerPath()
	var backupPath string
	if _, err := os.Stat(scriptPath); err == nil {
		backupPath = scriptPath + ".bak"
		cp(scriptPath, backupPath)
	}

	if out, err := gitPullFFOnly(repoDir); err != nil {
		if backupPath != "" {
			os.Remove(backupPath)
		}
		return fmt.Errorf("git pull --ff-only failed: %v\n%s\nhint: update or reconcile local commits in %s, then retry",
			err, strings.TrimSpace(out), repoDir)
	} else {
		fmt.Println(strings.TrimSpace(out))
	}

	newHash, _ := gitRevParse(repoDir, "HEAD")
	if oldHash == newHash {
		// The checkout may have been pulled separately, or a previous build
		// may have failed. An unchanged HEAD says nothing about the binary.
		fmt.Println("  Source already up to date; rebuilding the installed binary.")
	} else {
		commits, _ := gitLogOneline(repoDir, oldHash+".."+newHash)
		fmt.Printf("  Updated: %d new commits\n", len(commits))
		for _, c := range commits {
			if len(c) > 60 {
				c = c[:60] + "..."
			}
			fmt.Printf("    %s\n", c)
		}
	}

	if err := rebuildSelfUpdateBinary(repoDir, scriptPath); err != nil {
		fmt.Println("  Error: Build/install failed. Rolling back...")
		if oldHash != newHash {
			gitCheckout(repoDir, oldHash)
		}
		if backupPath != "" {
			cp(backupPath, scriptPath)
		}
		os.Remove(backupPath)
		return err
	}

	// Self-check: can the new script run --version?
	if scriptPath != "" {
		cmd := exec.Command(scriptPath, "--version")
		if err := cmd.Run(); err != nil {
			fmt.Println("  Error: New version failed self-check. Rolling back...")
			if oldHash != newHash {
				gitCheckout(repoDir, oldHash)
			}
			if backupPath != "" {
				cp(backupPath, scriptPath)
			}
			os.Remove(backupPath)
			return fmt.Errorf("self-check failed")
		}
	}

	if backupPath != "" {
		os.Remove(backupPath)
	}
	fmt.Println("  ✓ ggrun updated and verified. Restart to use the new version.")
	return nil
}

func rebuildSelfUpdateBinary(repoDir, scriptPath string) error {
	if strings.TrimSpace(scriptPath) == "" {
		return fmt.Errorf("cannot resolve the active ggrun binary path")
	}
	goDir := filepath.Join(repoDir, "go")
	if _, err := os.Stat(filepath.Join(goDir, "go.mod")); err != nil {
		goDir = repoDir
		if _, err := os.Stat(filepath.Join(goDir, "go.mod")); err != nil {
			return fmt.Errorf("no Go module found in %s", repoDir)
		}
	}
	staging := scriptPath + ".next"
	_ = os.Remove(staging)
	// Stamp the rebuilt binary with the version the repo is at now. Without a
	// stamp it keeps the in-source default (v3.2.0-go), which then permanently
	// reports "a newer version is available" after every source update.
	version := gitDescribeVersion(repoDir)
	if version == "" {
		version = Version()
	}
	ldflags := "-s -w -X github.com/raketenkater/ggrun/pkg/update.currentVersion=" + version
	cmd := exec.Command("go", "build", "-trimpath", "-ldflags="+ldflags, "-o", staging, "./cmd/ggrun")
	cmd.Dir = goDir
	if out, err := cmd.CombinedOutput(); err != nil {
		_ = os.Remove(staging)
		return fmt.Errorf("build ggrun: %w\n%s", err, strings.TrimSpace(string(out)))
	}
	if err := exec.Command(staging, "version").Run(); err != nil {
		_ = os.Remove(staging)
		return fmt.Errorf("staged ggrun self-check: %w", err)
	}
	if err := os.Rename(staging, scriptPath); err != nil {
		_ = os.Remove(staging)
		return fmt.Errorf("activate rebuilt ggrun: %w", err)
	}
	return nil
}

// SelfUpdateFromReleaseInstaller updates release-bundle installs that do not have a
// local ggrun git checkout. It downloads the latest tagged install.sh and lets
// the installer select the right platform/backend bundle or source fallback.
func SelfUpdateFromReleaseInstaller() error {
	if runtime.GOOS == "windows" {
		return selfUpdateWindowsInstaller(strings.TrimSpace(os.Getenv("LLM_APP_HOME")))
	}
	fmt.Println("═══ Updating ggrun from latest release installer ═══")
	scriptPath := installedLLMServerPath()
	backupPath := ""
	if scriptPath != "" {
		if _, err := os.Stat(scriptPath); err == nil {
			backupPath = scriptPath + ".bak"
			_ = cp(scriptPath, backupPath)
		}
	}

	res, err := Check()
	if err != nil || strings.TrimSpace(res.Latest) == "" {
		if backupPath != "" {
			_ = os.Remove(backupPath)
		}
		if err != nil {
			return fmt.Errorf("resolve latest release: %w", err)
		}
		return fmt.Errorf("resolve latest release: empty tag")
	}
	tmpDir, err := os.MkdirTemp("", "ggrun-update-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)
	installerPath := filepath.Join(tmpDir, "install.sh")
	if err := downloadVerifiedInstaller(res.Latest, "install.sh", installerPath, 0755); err != nil {
		if backupPath != "" {
			_ = os.Remove(backupPath)
		}
		return fmt.Errorf("download installer: %w", err)
	}

	cmd := exec.Command("bash", installerPath)
	cmd.Env = selfUpdateInstallEnv(os.Getenv("LLM_APP_HOME"))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		if backupPath != "" && scriptPath != "" {
			_ = cp(backupPath, scriptPath)
			_ = os.Remove(backupPath)
		}
		return fmt.Errorf("release installer failed: %w", err)
	}

	if scriptPath != "" {
		if err := exec.Command(scriptPath, "--version").Run(); err != nil {
			if backupPath != "" {
				_ = cp(backupPath, scriptPath)
				_ = os.Remove(backupPath)
			}
			return fmt.Errorf("self-check failed after release installer")
		}
	}
	if backupPath != "" {
		_ = os.Remove(backupPath)
	}
	fmt.Println("  ✓ ggrun release installer completed and verified. Restart to use the new version.")
	return nil
}

func rawInstallerURL(ref string) string {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		ref = "main"
	}
	return fmt.Sprintf(rawInstallURL, githubRepo, ref)
}

func rawInstallerPSURLForRef(ref string) string {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		ref = "main"
	}
	return fmt.Sprintf(rawInstallPSURL, githubRepo, ref)
}

func selfUpdateWindowsInstaller(appHome string) error {
	fmt.Println("═══ Updating ggrun from latest Windows installer ═══")
	scriptPath := installedLLMServerPath()
	backupPath := ""
	if scriptPath != "" {
		if _, err := os.Stat(scriptPath); err == nil {
			backupPath = scriptPath + ".bak"
			_ = cp(scriptPath, backupPath)
		}
	}

	res, err := Check()
	if err != nil || strings.TrimSpace(res.Latest) == "" {
		if backupPath != "" {
			_ = os.Remove(backupPath)
		}
		if err != nil {
			return fmt.Errorf("resolve latest release: %w", err)
		}
		return fmt.Errorf("resolve latest release: empty tag")
	}
	tmpDir, err := os.MkdirTemp("", "ggrun-update-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)
	installerPath := filepath.Join(tmpDir, "install.ps1")
	if err := downloadVerifiedInstaller(res.Latest, "install.ps1", installerPath, 0644); err != nil {
		if backupPath != "" {
			_ = os.Remove(backupPath)
		}
		return fmt.Errorf("download Windows installer: %w", err)
	}

	cmd, err := powershellInstallCommand(installerPath, appHome)
	if err != nil {
		return err
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		if backupPath != "" && scriptPath != "" {
			_ = cp(backupPath, scriptPath)
			_ = os.Remove(backupPath)
		}
		return fmt.Errorf("Windows installer failed: %w", err)
	}

	if scriptPath != "" {
		if err := exec.Command(scriptPath, "--version").Run(); err != nil {
			if backupPath != "" {
				_ = cp(backupPath, scriptPath)
				_ = os.Remove(backupPath)
			}
			return fmt.Errorf("self-check failed after Windows installer")
		}
	}
	if backupPath != "" {
		_ = os.Remove(backupPath)
	}
	fmt.Println("  ✓ ggrun Windows installer completed and verified. Restart to use the new version.")
	return nil
}

func powershellInstallCommand(installerPath, appHome string) (*exec.Cmd, error) {
	shell := ""
	for _, candidate := range []string{"pwsh", "powershell.exe", "powershell"} {
		if path, err := exec.LookPath(candidate); err == nil {
			shell = path
			break
		}
	}
	if shell == "" {
		return nil, fmt.Errorf("PowerShell not found")
	}
	args := []string{"-NoProfile", "-ExecutionPolicy", "Bypass", "-File", installerPath, "-NoPath"}
	if appHome != "" {
		args = append(args, "-InstallDir", appHome)
	}
	return exec.Command(shell, args...), nil
}

func selfUpdateInstallEnv(appHome string) []string {
	env := append([]string{}, os.Environ()...)
	env = append(env, "LLM_INSTALL_NONINTERACTIVE=1", "LLM_INSTALL_MAIN=go")
	appHome = strings.TrimSpace(appHome)
	if appHome == "" {
		return env
	}
	return append(env,
		"LLM_APP_HOME="+appHome,
		"LLM_INSTALL_PREFIX="+filepath.Join(appHome, ".bin"),
		"LLM_INSTALL_MODEL_DIR="+filepath.Join(appHome, "models"),
		"LLM_INSTALL_BACKEND_ROOT="+filepath.Join(appHome, ".src"),
		"LLM_INSTALL_REPO_DIR="+filepath.Join(appHome, ".src", "ggrun"),
		"LLM_INSTALL_REF=main",
		"LLM_INSTALL_BACKEND=skip",
		"LLM_INSTALL_MODE=build",
	)
}

// SelfUpdateAppHomeInstaller refreshes app-home installs from the latest main
// installer while preserving the existing app-home layout. This updates the Go
// binary and embedded catalog without depending on a local git checkout.
func SelfUpdateAppHomeInstaller(appHome string) error {
	appHome = strings.TrimSpace(appHome)
	if appHome == "" {
		return fmt.Errorf("LLM_APP_HOME is not set")
	}
	if runtime.GOOS == "windows" {
		return selfUpdateWindowsInstaller(appHome)
	}
	fmt.Println("═══ Updating ggrun app home from latest release ═══")
	scriptPath := installedLLMServerPath()
	backupPath := ""
	if scriptPath != "" {
		if _, err := os.Stat(scriptPath); err == nil {
			backupPath = scriptPath + ".bak"
			_ = cp(scriptPath, backupPath)
		}
	}

	res, err := Check()
	if err != nil || strings.TrimSpace(res.Latest) == "" {
		if backupPath != "" {
			_ = os.Remove(backupPath)
		}
		if err != nil {
			return fmt.Errorf("resolve latest release: %w", err)
		}
		return fmt.Errorf("resolve latest release: empty tag")
	}
	tmpDir, err := os.MkdirTemp("", "ggrun-update-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)
	installerPath := filepath.Join(tmpDir, "install.sh")
	if err := downloadVerifiedInstaller(res.Latest, "install.sh", installerPath, 0755); err != nil {
		if backupPath != "" {
			_ = os.Remove(backupPath)
		}
		return fmt.Errorf("download installer: %w", err)
	}

	cmd := exec.Command("bash", installerPath)
	cmd.Env = selfUpdateInstallEnv(appHome)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		if backupPath != "" && scriptPath != "" {
			_ = cp(backupPath, scriptPath)
			_ = os.Remove(backupPath)
		}
		return fmt.Errorf("app-home installer failed: %w", err)
	}

	if scriptPath != "" {
		if err := exec.Command(scriptPath, "--version").Run(); err != nil {
			if backupPath != "" {
				_ = cp(backupPath, scriptPath)
				_ = os.Remove(backupPath)
			}
			return fmt.Errorf("self-check failed after app-home installer")
		}
	}
	if backupPath != "" {
		_ = os.Remove(backupPath)
	}
	fmt.Println("  ✓ ggrun app home updated and verified. Restart to use the new version.")
	return nil
}

func installedSourceRepoDir() string {
	if repoDir := strings.TrimSpace(os.Getenv("LLM_SERVER_REPO")); repoDir != "" {
		return repoDir
	}
	if exe, err := os.Executable(); err == nil {
		if repoDir := sourceRepoFromExecutable(exe); repoDir != "" {
			return repoDir
		}
	}
	if appHome := strings.TrimSpace(os.Getenv("LLM_APP_HOME")); appHome != "" {
		if repoDir := repoFromAppHome(appHome); repoDir != "" {
			return repoDir
		}
	}
	if appHome := backends.AppHome(); appHome != "" && appHome != os.Getenv("LLM_APP_HOME") {
		if repoDir := repoFromAppHome(appHome); repoDir != "" {
			return repoDir
		}
	}
	if home := homeDir(); home != "" {
		return filepath.Join(home, "ggrun")
	}
	return ""
}

// repoFromAppHome returns the ggrun git checkout inside an app home, or "" when
// that directory holds none. A source install's app home IS the repo root, so
// the directory itself is checked first; a release-bundle app home (which holds
// no .git) falls through to the conventional nested locations.
func repoFromAppHome(appHome string) string {
	if appHome == "" {
		return ""
	}
	for _, cand := range []string{appHome, filepath.Join(appHome, "ggrun"), filepath.Join(appHome, ".src", "ggrun")} {
		if _, err := os.Stat(filepath.Join(cand, ".git")); err == nil {
			return cand
		}
	}
	return ""
}

func sourceRepoFromExecutable(exe string) string {
	if resolved, err := filepath.EvalSymlinks(exe); err == nil && resolved != "" {
		exe = resolved
	}
	dir := filepath.Dir(exe)
	for _, candidate := range []string{dir, filepath.Dir(dir)} {
		if _, err := os.Stat(filepath.Join(candidate, ".git")); err == nil {
			return candidate
		}
	}
	return ""
}

func installedLLMServerPath() string {
	if appHome := os.Getenv("LLM_APP_HOME"); appHome != "" {
		for _, candidate := range []string{
			filepath.Join(appHome, ".bin", "ggrun"),
			filepath.Join(appHome, ".bin", "ggrun.exe"),
			filepath.Join(appHome, "bin", "ggrun"),
			filepath.Join(appHome, "bin", "ggrun.exe"),
			filepath.Join(appHome, "ggrun"),
			filepath.Join(appHome, "ggrun.cmd"),
		} {
			if _, err := os.Stat(candidate); err == nil {
				// Resolve a linked app-home entry to the real binary. Renaming a
				// rebuilt binary over the link would replace the link with a
				// second, independent copy that PATH never runs.
				if resolved, err := filepath.EvalSymlinks(candidate); err == nil && resolved != "" {
					return resolved
				}
				return candidate
			}
		}
	}
	if path, _ := exec.LookPath("ggrun"); path != "" {
		if resolved, err := filepath.EvalSymlinks(path); err == nil && resolved != "" {
			return resolved
		}
		return path
	}
	if home := homeDir(); home != "" {
		if runtime.GOOS == "windows" {
			return filepath.Join(home, "ggrun", ".bin", "ggrun.exe")
		}
		return filepath.Join(home, ".local", "bin", "ggrun")
	}
	return ""
}

func downloadFile(url, dst string, mode os.FileMode) error {
	data, err := downloadBytes(url, maxDownloadBytes)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, mode)
}

func releaseAssetURL(tag, name string) string {
	return fmt.Sprintf(githubReleaseAssetURL, githubRepo, tag, name)
}

func downloadBytes(url string, limit int64) ([]byte, error) {
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("%s", resp.Status)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("download exceeded %d bytes", limit)
	}
	return data, nil
}

func parseSHA256SUMS(data []byte) map[string]string {
	out := make(map[string]string)
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		sum := strings.ToLower(fields[0])
		name := strings.TrimPrefix(filepath.Base(fields[len(fields)-1]), "*")
		if len(sum) != 64 {
			continue
		}
		out[name] = sum
	}
	return out
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func downloadVerifiedInstaller(tag, name, dst string, mode os.FileMode) error {
	tag = strings.TrimSpace(tag)
	name = filepath.Base(strings.TrimSpace(name))
	if tag == "" || tag == "main" || tag == "master" {
		return fmt.Errorf("refusing installer download without a release tag")
	}
	if name != "install.sh" && name != "install.ps1" {
		return fmt.Errorf("unsupported installer %q", name)
	}
	sums, err := downloadBytes(releaseAssetURL(tag, "SHA256SUMS"), maxChecksumBytes)
	if err != nil {
		return fmt.Errorf("download SHA256SUMS: %w", err)
	}
	want, ok := parseSHA256SUMS(sums)[name]
	if !ok {
		return fmt.Errorf("SHA256SUMS has no %s; refusing unsigned installer", name)
	}
	body, err := downloadBytes(releaseAssetURL(tag, name), maxInstallerBytes)
	if err != nil {
		body, err = downloadBytes(rawInstallerURLForName(tag, name), maxInstallerBytes)
		if err != nil {
			return err
		}
	}
	if got := sha256Hex(body); got != want {
		return fmt.Errorf("%s checksum mismatch", name)
	}
	return os.WriteFile(dst, body, mode)
}

func rawInstallerURLForName(tag, name string) string {
	if name == "install.ps1" {
		return rawInstallerPSURLForRef(tag)
	}
	return rawInstallerURL(tag)
}

// UpdateBackend updates a backend repo (ik_llama.cpp or llama.cpp).
func UpdateBackend(name, repoDir string, walkback int) error {
	buildDir := filepath.Join(repoDir, "build")
	binary := filepath.Join(buildDir, "bin", "llama-server")
	fallbackDir := filepath.Join(homeDir(), ".cache", "ggrun", "update-fallbacks")
	os.MkdirAll(fallbackDir, 0755)

	fmt.Printf("\n═══ Updating %s ═══\n", name)
	if _, err := os.Stat(filepath.Join(repoDir, ".git")); err != nil {
		fmt.Printf("  Skip: %s is not a git repo\n", repoDir)
		return fmt.Errorf("not a git repo")
	}

	oldCommit, _ := gitRevParse(repoDir, "HEAD")
	branch, _ := gitSymbolicRef(repoDir)
	if branch == "" {
		// Detached HEAD (e.g. a checkout of a pinned commit): `git pull` refuses
		// to run, so the backend silently stayed stale. Re-attach to the default
		// branch first; fall back to master/main if origin/HEAD isn't set.
		branch = "master"
		if out, err := exec.Command("git", "-C", repoDir, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD").Output(); err == nil {
			if b := strings.TrimPrefix(strings.TrimSpace(string(out)), "origin/"); b != "" {
				branch = b
			}
		}
		fmt.Printf("  Detached HEAD — checking out %s before pull\n", branch)
		if out, err := exec.Command("git", "-C", repoDir, "checkout", branch).CombinedOutput(); err != nil {
			fmt.Printf("  FAILED: git checkout %s: %v\n%s\n", branch, err, strings.TrimSpace(string(out)))
			return fmt.Errorf("git checkout %s failed: %v", branch, err)
		}
	}

	oldHash := ""
	if _, err := os.Stat(binary); err == nil {
		oldHash = md5sum(binary)
	}
	fmt.Printf("  Current: %s\n", oldCommit)

	// Backup working binary outside build dir
	binaryBackup := filepath.Join(repoDir, ".ggrun.llama-server.backup")
	if _, err := os.Stat(binary); err == nil {
		cp(binary, binaryBackup)
		fmt.Println("  Backed up working binary")
	}

	// Check for dirty tree
	if dirty, _ := gitStatusPorcelain(repoDir); dirty != "" {
		fmt.Println("  Skip: working tree has uncommitted changes:")
		lines := strings.Split(dirty, "\n")
		for i, l := range lines {
			if i >= 5 {
				break
			}
			fmt.Printf("    %s\n", l)
		}
		fmt.Printf("    (commit or stash them in %s, then re-run --update)\n", repoDir)
		os.Remove(binaryBackup)
		return fmt.Errorf("dirty tree")
	}

	if out, err := gitPullFFOnly(repoDir); err != nil {
		fmt.Println("  Warning: fast-forward pull failed, trying rebase...")
		if out2, err2 := gitPullRebase(repoDir); err2 != nil {
			fmt.Printf("  FAILED: git pull failed — skipping %s\n", name)
			os.Remove(binaryBackup)
			return fmt.Errorf("git pull failed: %v | %v", err, err2)
		} else {
			fmt.Println(strings.TrimSpace(out2))
		}
	} else {
		fmt.Println(strings.TrimSpace(out))
	}

	newCommit, _ := gitRevParse(repoDir, "HEAD")
	if oldCommit == newCommit {
		if validateErr := smokeBackendConfigured(binary, collectCMakeFlags(buildDir)); validateErr == nil {
			fmt.Println("  Already up to date and active backend passes conformance.")
			os.Remove(binaryBackup)
			return nil
		} else {
			fmt.Printf("  Source is current but active backend failed conformance (%v); rebuilding.\n", validateErr)
		}
	}
	fmt.Printf("  Updated: %s\n", newCommit)

	// Walk-back: if HEAD fails to build/test, try up to N-1 parent commits
	if walkback <= 0 {
		walkback = 3
	}
	var successCommit string
	for attempt := 0; attempt < walkback; attempt++ {
		targetCommit := newCommit
		if attempt > 0 {
			targetCommit, _ = gitRevParse(repoDir, newCommit+"~"+strconv.Itoa(attempt))
			if targetCommit == "" {
				break
			}
			fmt.Printf("\n  ── Attempt %d/%d: walking back to %s ──\n", attempt+1, walkback, targetCommit)
			gitCheckoutQuiet(repoDir, targetCommit)
		}
		if buildAndTest(repoDir, buildDir) {
			successCommit = targetCommit
			break
		}
	}

	if successCommit == "" {
		fmt.Printf("\n  All %d attempts failed — rolling back to previous version...\n", walkback)
		gitCheckout(repoDir, oldCommit)
		if _, err := os.Stat(binaryBackup); err == nil {
			cp(binaryBackup, binary)
		}
		fmt.Printf("  Rolled back to %s\n", oldCommit)
		os.Remove(binaryBackup)
		return fmt.Errorf("all build attempts failed")
	}

	if successCommit != newCommit {
		marker := filepath.Join(fallbackDir, strings.ReplaceAll(name, "/", "_")+".env")
		f, _ := os.Create(marker)
		if f != nil {
			fmt.Fprintf(f, "repo_dir=%q\n", repoDir)
			fmt.Fprintf(f, "branch=%q\n", branch)
			fmt.Fprintf(f, "head_commit=%q\n", newCommit)
			fmt.Fprintf(f, "fallback_commit=%q\n", successCommit)
			fmt.Fprintf(f, "recorded_at=%q\n", time.Now().UTC().Format(time.RFC3339))
			f.Close()
		}
		fmt.Printf("  Walk-back succeeded at %s\n", successCommit)
		fmt.Printf("  Reattaching repo to branch '%s' while keeping built fallback binary.\n", branch)
		gitCheckoutQuiet(repoDir, branch)
	}

	newHash := ""
	if _, err := os.Stat(binary); err == nil {
		newHash = md5sum(binary)
	}
	if oldHash == newHash {
		fmt.Println("  Binary unchanged (no relevant code changes)")
	} else {
		fmt.Println("  New binary built successfully ✓")
	}

	os.Remove(binaryBackup)
	fmt.Printf("  %s updated: %s\n", name, successCommit)
	return nil
}

// BackendBuildTarget is one active, source-built generic backend. BuildDir is
// explicit because canonical installs commonly keep CUDA and Vulkan builds in
// build-cuda/build-vulkan rather than the legacy build directory.
type BackendBuildTarget struct {
	Label    string
	RepoDir  string
	BuildDir string
}

// BackendUpdateResult reports one independently preserved backend build.
type BackendUpdateResult struct {
	Target BackendBuildTarget
	Status string // current, updated, fallback, failed
	Err    error
}

// BackendBuildTargetsAt discovers the backends the canonical app home actually
// launches by following its .bin symlinks. This avoids updating an unrelated
// ~/ik_llama.cpp checkout while production points at another source tree.
func BackendBuildTargetsAt(appHome string) []BackendBuildTarget {
	appHome = strings.TrimSpace(appHome)
	seen := make(map[string]bool)
	var targets []BackendBuildTarget
	add := func(label, binary string) {
		binary = strings.TrimSpace(binary)
		if binary == "" {
			return
		}
		real, err := filepath.EvalSymlinks(binary)
		if err != nil {
			return
		}
		binDir := filepath.Dir(real)
		if filepath.Base(binDir) != "bin" {
			return
		}
		buildDir := filepath.Dir(binDir)
		repoDir := filepath.Dir(buildDir)
		if _, err := os.Stat(filepath.Join(repoDir, ".git")); err != nil {
			return
		}
		key := repoDir + "\x00" + buildDir
		if seen[key] {
			return
		}
		seen[key] = true
		if label == "" {
			family := "llama.cpp"
			if strings.Contains(strings.ToLower(filepath.Base(repoDir)), "ik_llama") {
				family = "ik_llama.cpp"
			}
			variant := strings.TrimPrefix(filepath.Base(buildDir), "build-")
			if variant == "build" || variant == "" {
				label = family
			} else {
				label = fmt.Sprintf("%s (%s)", family, variant)
			}
		}
		targets = append(targets, BackendBuildTarget{Label: label, RepoDir: repoDir, BuildDir: buildDir})
	}

	if appHome != "" {
		for _, candidate := range []struct {
			name  string
			label string
		}{
			{name: "llama-server-cuda", label: "llama.cpp (cuda)"},
			{name: "llama-server-vulkan", label: "llama.cpp (vulkan)"},
			{name: "ik_llama-server-cuda", label: "ik_llama.cpp (cuda)"},
			{name: "llama-server", label: ""},
		} {
			add(candidate.label, filepath.Join(appHome, ".bin", candidate.name))
			add(candidate.label, filepath.Join(appHome, "bin", candidate.name))
		}
		// Source-contained installs may not use links. Include each active build
		// variant, but only when it has both a configured cache and a server.
		for _, family := range []string{"llama.cpp", "ik_llama.cpp"} {
			repoDir := filepath.Join(appHome, ".src", family)
			for _, variant := range []string{"build-cuda", "build-vulkan", "build"} {
				buildDir := filepath.Join(repoDir, variant)
				if _, err := os.Stat(filepath.Join(buildDir, "CMakeCache.txt")); err != nil {
					continue
				}
				label := family
				if suffix := strings.TrimPrefix(variant, "build-"); suffix != "build" && suffix != "" {
					label += " (" + suffix + ")"
				}
				add(label, filepath.Join(buildDir, "bin", "llama-server"))
			}
		}
		return targets
	}

	// Legacy installs with no resolved app home retain the old bounded source
	// discovery, using whichever configured build directory is present.
	for _, repo := range backendUpdateCandidates() {
		for _, variant := range []string{"build-cuda", "build-vulkan", "build"} {
			buildDir := filepath.Join(repo.Dir, variant)
			if _, err := os.Stat(filepath.Join(buildDir, "CMakeCache.txt")); err == nil {
				add(repo.Label, filepath.Join(buildDir, "bin", "llama-server"))
			}
		}
	}
	return targets
}

// UpdateBackends updates the active generic backends for a legacy/environment
// resolved install. Command code should prefer UpdateBackendsAtAppHome so the
// canonical production boundary is explicit.
func UpdateBackends() error {
	return UpdateBackendsAtAppHome(strings.TrimSpace(os.Getenv("LLM_APP_HOME")))
}

func UpdateBackendsAtAppHome(appHome string) error {
	targets := BackendBuildTargetsAt(appHome)
	if len(targets) == 0 {
		fmt.Println("No active source-built generic backends found — skipping")
		return nil
	}
	results := updateBackendBuildTargets(targets, 3)
	var updateErrs []error
	fmt.Println("\nBackend update summary:")
	for _, result := range results {
		if result.Err != nil {
			fmt.Printf("  %-28s failed (kept previous build): %v\n", result.Target.Label, result.Err)
			updateErrs = append(updateErrs, fmt.Errorf("%s: %w", result.Target.Label, result.Err))
			continue
		}
		fmt.Printf("  %-28s %s\n", result.Target.Label, result.Status)
	}
	return errors.Join(updateErrs...)
}

// UpdateMainlineBackendAtAppHome advances only the mainline llama.cpp backend —
// the .src/llama.cpp checkout and the build variants ggrun actually launches —
// to the latest commit, preserving each build it replaces on failure. It is the
// same safe staged update as UpdateBackendsAtAppHome but scoped to the mainline
// family, so an unsupported-architecture offer cannot drag the ik_llama.cpp
// family or unrelated forks into a rebuild.
func UpdateMainlineBackendAtAppHome(appHome string) error {
	var targets []BackendBuildTarget
	for _, target := range BackendBuildTargetsAt(appHome) {
		if strings.EqualFold(filepath.Base(target.RepoDir), "llama.cpp") {
			targets = append(targets, target)
		}
	}
	if len(targets) == 0 {
		fmt.Println("No active mainline llama.cpp source build found — nothing to update.")
		return nil
	}
	results := updateBackendBuildTargets(targets, 3)
	var updateErrs []error
	fmt.Println("\nMainline llama.cpp update summary:")
	for _, result := range results {
		if result.Err != nil {
			fmt.Printf("  %-28s failed (kept previous build): %v\n", result.Target.Label, result.Err)
			updateErrs = append(updateErrs, fmt.Errorf("%s: %w", result.Target.Label, result.Err))
			continue
		}
		fmt.Printf("  %-28s %s\n", result.Target.Label, result.Status)
	}
	return errors.Join(updateErrs...)
}

func updateBackendBuildTargets(targets []BackendBuildTarget, walkback int) []BackendUpdateResult {
	return updateBackendBuildTargetsWith(targets, walkback, updateBackendBuildGroup)
}

func updateBackendBuildTargetsWith(
	targets []BackendBuildTarget,
	walkback int,
	runGroup func(string, []BackendBuildTarget, int) []BackendUpdateResult,
) []BackendUpdateResult {
	type group struct {
		repo    string
		targets []BackendBuildTarget
	}
	var groups []group
	groupIndex := make(map[string]int)
	for _, target := range targets {
		idx, ok := groupIndex[target.RepoDir]
		if !ok {
			idx = len(groups)
			groupIndex[target.RepoDir] = idx
			groups = append(groups, group{repo: target.RepoDir})
		}
		groups[idx].targets = append(groups[idx].targets, target)
	}
	var results []BackendUpdateResult
	for _, group := range groups {
		results = append(results, runGroup(group.repo, group.targets, walkback)...)
	}
	return results
}

func failedBackendResults(targets []BackendBuildTarget, err error) []BackendUpdateResult {
	results := make([]BackendUpdateResult, 0, len(targets))
	for _, target := range targets {
		results = append(results, BackendUpdateResult{Target: target, Status: "failed", Err: err})
	}
	return results
}

func updateBackendBuildGroup(repoDir string, targets []BackendBuildTarget, walkback int) []BackendUpdateResult {
	fmt.Printf("\n═══ Updating backend source %s ═══\n", repoDir)
	if _, err := os.Stat(filepath.Join(repoDir, ".git")); err != nil {
		return failedBackendResults(targets, fmt.Errorf("not a git repo"))
	}
	if dirty, _ := gitStatusPorcelain(repoDir); dirty != "" {
		return failedBackendResults(targets, fmt.Errorf("tracked working-tree changes; commit or stash them first"))
	}

	oldCommit, _ := gitRevParse(repoDir, "HEAD")
	branch, _ := gitSymbolicRef(repoDir)
	if branch == "" {
		branch = "master"
		if out, err := exec.Command("git", "-C", repoDir, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD").Output(); err == nil {
			if value := strings.TrimPrefix(strings.TrimSpace(string(out)), "origin/"); value != "" {
				branch = value
			}
		}
		if out, err := exec.Command("git", "-C", repoDir, "checkout", branch).CombinedOutput(); err != nil {
			return failedBackendResults(targets, fmt.Errorf("attach detached checkout to %s: %s", branch, strings.TrimSpace(string(out))))
		}
	}

	if out, err := gitPullFFOnly(repoDir); err != nil {
		if out2, err2 := gitPullRebase(repoDir); err2 != nil {
			return failedBackendResults(targets, fmt.Errorf("git pull failed: %v | %v", err, err2))
		} else if strings.TrimSpace(out2) != "" {
			fmt.Println(strings.TrimSpace(out2))
		}
	} else if strings.TrimSpace(out) != "" {
		fmt.Println(strings.TrimSpace(out))
	}
	newCommit, _ := gitRevParse(repoDir, "HEAD")
	if oldCommit == newCommit {
		results := make([]BackendUpdateResult, 0, len(targets))
		allValid := true
		for _, target := range targets {
			binary := filepath.Join(target.BuildDir, "bin", "llama-server")
			if err := smokeBackendConfigured(binary, collectCMakeFlags(target.BuildDir)); err != nil {
				fmt.Printf("  %s source is current but active build failed conformance (%v); rebuilding.\n", target.Label, err)
				allValid = false
				continue
			}
			results = append(results, BackendUpdateResult{Target: target, Status: "current"})
		}
		if allValid {
			return results
		}
		// Rebuild the group at the unchanged source revision. The normal staged
		// activation below leaves every valid active binary untouched until its
		// replacement passes the same conformance gate.
		results = results[:0]
	}

	if walkback <= 0 {
		walkback = 3
	}
	fallbackDir := filepath.Join(homeDir(), ".cache", "ggrun", "update-fallbacks")
	_ = os.MkdirAll(fallbackDir, 0o755)
	results := make([]BackendUpdateResult, 0, len(targets))
	anySuccess := false
	for _, target := range targets {
		oldHash := md5sum(filepath.Join(target.BuildDir, "bin", "llama-server"))
		successCommit := ""
		for attempt := 0; attempt < walkback; attempt++ {
			targetCommit := newCommit
			if attempt > 0 {
				targetCommit, _ = gitRevParse(repoDir, newCommit+"~"+strconv.Itoa(attempt))
				if targetCommit == "" {
					break
				}
			}
			if err := gitCheckoutQuiet(repoDir, targetCommit); err != nil {
				continue
			}
			fmt.Printf("  Building %s at %s (attempt %d/%d)\n", target.Label, targetCommit, attempt+1, walkback)
			if buildAndTest(repoDir, target.BuildDir) {
				successCommit = targetCommit
				break
			}
		}
		if successCommit == "" {
			results = append(results, BackendUpdateResult{
				Target: target,
				Status: "failed",
				Err:    fmt.Errorf("all %d build attempts failed", walkback),
			})
			continue
		}
		anySuccess = true
		status := "updated"
		if successCommit != newCommit {
			status = "fallback " + successCommit
			marker := filepath.Join(fallbackDir, strings.NewReplacer("/", "_", " ", "_").Replace(target.Label)+".env")
			body := fmt.Sprintf("repo_dir=%q\nbranch=%q\nhead_commit=%q\nfallback_commit=%q\nrecorded_at=%q\n",
				repoDir, branch, newCommit, successCommit, time.Now().UTC().Format(time.RFC3339))
			_ = os.WriteFile(marker, []byte(body), 0o644)
		}
		newHash := md5sum(filepath.Join(target.BuildDir, "bin", "llama-server"))
		if oldHash == newHash {
			status += " (binary unchanged)"
		}
		results = append(results, BackendUpdateResult{Target: target, Status: status})
	}

	if anySuccess {
		_ = gitCheckoutQuiet(repoDir, branch)
	} else if oldCommit != "" {
		_ = gitCheckoutQuiet(repoDir, oldCommit)
	}
	return results
}

func backendUpdateCandidates() []repoCandidate {
	seen := map[string]bool{}
	candidates := []repoCandidate{}
	add := func(label, dir string) {
		dir = strings.TrimSpace(dir)
		if dir == "" || seen[label+"\x00"+dir] {
			return
		}
		seen[label+"\x00"+dir] = true
		candidates = append(candidates, repoCandidate{Label: label, Dir: dir})
	}

	if server := os.Getenv("LLAMA_SERVER"); server != "" {
		root := filepath.Dir(filepath.Dir(filepath.Dir(server)))
		base := filepath.Base(root)
		if strings.Contains(base, "ik_llama") {
			add("ik_llama.cpp", root)
		} else if strings.Contains(base, "llama.cpp") {
			add("llama.cpp", root)
		}
	}
	if appHome := os.Getenv("LLM_APP_HOME"); appHome != "" {
		add("ik_llama.cpp", filepath.Join(appHome, ".src", "ik_llama.cpp"))
		add("llama.cpp", filepath.Join(appHome, ".src", "llama.cpp"))
	}
	if home := homeDir(); home != "" {
		add("ik_llama.cpp", filepath.Join(home, "ik_llama.cpp"))
		add("llama.cpp", filepath.Join(home, "llama.cpp"))
	}
	return candidates
}

func buildAndTest(repoDir, buildDir string) bool {
	nproc := 8
	if out, err := exec.Command("nproc").Output(); err == nil {
		nproc, _ = strconv.Atoi(strings.TrimSpace(string(out)))
		if nproc < 1 {
			nproc = 1
		} else if nproc > 8 {
			nproc = 8
		}
	}

	stagingDir := buildDir + ".ggrun-update"
	if err := os.RemoveAll(stagingDir); err != nil {
		fmt.Printf("  Cannot clean staging build: %v\n", err)
		return false
	}
	defer os.RemoveAll(stagingDir)

	fmt.Println("  Configuring isolated update build...")
	if _, err := os.Stat(filepath.Join(buildDir, "CMakeCache.txt")); err != nil {
		fmt.Printf("  Refusing to guess missing backend build configuration: %s\n", buildDir)
		return false
	}
	cmakeFlags := collectCMakeFlags(buildDir)
	configure := exec.Command("cmake", cmakeConfigureArgs(repoDir, stagingDir, cmakeFlags)...)
	if out, err := configure.CombinedOutput(); err != nil {
		fmt.Printf("  Configure failed: %s\n", tailLines(string(out), 8))
		return false
	}

	fmt.Println("  Building isolated update...")
	build := exec.Command("cmake", "--build", stagingDir, "--config", "Release", "--parallel", strconv.Itoa(nproc), "--target", "llama-server")
	if out, err := build.CombinedOutput(); err != nil {
		fmt.Printf("  Build failed at this commit: %s\n", tailLines(string(out), 8))
		return false
	}

	stagingBinary := filepath.Join(stagingDir, "bin", "llama-server")
	if err := smokeBackendConfigured(stagingBinary, cmakeFlags); err != nil {
		fmt.Printf("  Backend conformance failed at this commit: %v\n", err)
		return false
	}
	validateActive := func(activeDir string) error {
		return smokeBackendConfigured(filepath.Join(activeDir, "bin", "llama-server"), cmakeFlags)
	}
	if err := promoteBackendBuild(buildDir, stagingDir, validateActive); err != nil {
		fmt.Printf("  Could not activate validated build: %v\n", err)
		return false
	}
	fmt.Println("  Isolated build succeeded and was activated")
	return true
}

func cmakeConfigureArgs(repoDir, buildDir string, flags []string) []string {
	args := []string{"-S", repoDir, "-B", buildDir, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"}
	return append(args, flags...)
}

func smokeBackend(binary string) error {
	if info, err := os.Stat(binary); err != nil || info.IsDir() {
		return fmt.Errorf("backend binary missing: %s", binary)
	}
	run := func(flag string) (string, error) {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		out, err := exec.CommandContext(ctx, binary, flag).CombinedOutput()
		if ctx.Err() != nil {
			return "", fmt.Errorf("%s timed out", flag)
		}
		if err != nil {
			return string(out), fmt.Errorf("%s: %w", flag, err)
		}
		return string(out), nil
	}
	if _, err := run("--version"); err != nil {
		return fmt.Errorf("%s: %w", binary, err)
	}
	help, err := run("--help")
	if err != nil {
		return fmt.Errorf("%s: %w", binary, err)
	}
	for _, required := range []string{"--model", "--ctx-size", "--host", "--port"} {
		if !strings.Contains(help, required) {
			return fmt.Errorf("%s --help lacks %s", binary, required)
		}
	}
	return nil
}

func smokeBackendConfigured(binary string, cmakeFlags []string) error {
	if err := smokeBackend(binary); err != nil {
		return err
	}
	expected := ""
	for _, flag := range cmakeFlags {
		switch flag {
		case "-DGGML_CUDA=ON":
			expected = "cuda"
		case "-DGGML_VULKAN=ON":
			expected = "vulkan"
		}
	}
	if expected == "" {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, binary, "--list-devices").CombinedOutput()
	if err != nil && len(out) == 0 {
		return fmt.Errorf("%s device probe failed: %w", expected, err)
	}
	text := strings.ToLower(string(out))
	if !strings.Contains(text, "available devices") || !strings.Contains(text, expected) {
		return fmt.Errorf("requested %s build does not report a %s device", expected, expected)
	}
	return nil
}

func promoteBackendBuild(buildDir, stagingDir string, validate func(string) error) error {
	backupDir := buildDir + ".ggrun-backup"
	if _, err := os.Stat(backupDir); err == nil {
		if _, currentErr := os.Stat(buildDir); os.IsNotExist(currentErr) {
			if err := os.Rename(backupDir, buildDir); err != nil {
				return fmt.Errorf("recover interrupted promotion: %w", err)
			}
		} else {
			return fmt.Errorf("stale backup requires inspection: %s", backupDir)
		}
	}

	hadCurrent := false
	if _, err := os.Stat(buildDir); err == nil {
		hadCurrent = true
		if err := os.Rename(buildDir, backupDir); err != nil {
			return fmt.Errorf("preserve current build: %w", err)
		}
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("inspect current build: %w", err)
	}

	if err := os.Rename(stagingDir, buildDir); err != nil {
		if hadCurrent {
			_ = os.Rename(backupDir, buildDir)
		}
		return fmt.Errorf("activate staging build: %w", err)
	}
	if validate != nil {
		if err := validate(buildDir); err != nil {
			_ = os.Rename(buildDir, stagingDir)
			if hadCurrent {
				_ = os.Rename(backupDir, buildDir)
			}
			return fmt.Errorf("validate activated build: %w", err)
		}
	}
	if hadCurrent {
		if err := os.RemoveAll(backupDir); err != nil {
			return fmt.Errorf("remove previous build backup: %w", err)
		}
	}
	return nil
}

func tailLines(value string, count int) string {
	lines := strings.Split(strings.TrimSpace(value), "\n")
	if len(lines) > count {
		lines = lines[len(lines)-count:]
	}
	return strings.Join(lines, "\n")
}

func collectCMakeFlags(buildDir string) []string {
	var flags []string
	cache := filepath.Join(buildDir, "CMakeCache.txt")
	data, err := os.ReadFile(cache)
	if err != nil {
		return flags
	}
	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "GGML_CUDA:BOOL=ON") {
			flags = append(flags, "-DGGML_CUDA=ON")
		}
		if strings.HasPrefix(line, "GGML_CUDA_FA_ALL_QUANTS:BOOL=ON") {
			flags = append(flags, "-DGGML_CUDA_FA_ALL_QUANTS=ON")
		}
		if strings.HasPrefix(line, "GGML_CUDA_NCCL:BOOL=ON") {
			flags = append(flags, "-DGGML_CUDA_NCCL=ON")
		}
		if strings.HasPrefix(line, "GGML_VULKAN:BOOL=ON") {
			flags = append(flags, "-DGGML_VULKAN=ON")
		}
		if strings.HasPrefix(line, "GGML_METAL:BOOL=ON") {
			flags = append(flags, "-DGGML_METAL=ON")
		}
		if strings.HasPrefix(line, "CMAKE_CUDA_ARCHITECTURES:") {
			parts := strings.SplitN(line, "=", 2)
			if len(parts) == 2 {
				flags = append(flags, "-DCMAKE_CUDA_ARCHITECTURES="+strings.TrimSpace(parts[1]))
			}
		}
		if strings.HasPrefix(line, "CMAKE_CUDA_COMPILER:") {
			parts := strings.SplitN(line, "=", 2)
			if len(parts) == 2 {
				val := strings.TrimSpace(parts[1])
				if val != "" && !strings.Contains(val, "NOTFOUND") {
					flags = append(flags, "-DCMAKE_CUDA_COMPILER="+val)
				}
			}
		}
	}
	if cudacxx := os.Getenv("CUDACXX"); cudacxx != "" {
		flags = append(flags, "-DCMAKE_CUDA_COMPILER="+cudacxx)
	}
	return flags
}

// versionPartParts splits one dotted version component into its integer base and
// whether it carries a git-describe "commits ahead" suffix. A build stamped
// v3.2.8-4-g44f99a0 is 4 commits ahead of tag v3.2.8, so it must compare NEWER
// than the bare tag, never as an unparseable 0 (which made it look older and
// triggered a false "newer version available" for a build already past the tag).
func versionPartParts(s string) (base int, ahead bool, ok bool) {
	num := s
	if idx := strings.IndexByte(s, '-'); idx >= 0 {
		num = s[:idx]
		ahead = true
	}
	n, err := strconv.Atoi(num)
	if err != nil {
		return 0, false, false
	}
	return n, ahead, true
}

func compareVersions(a, b string) int {
	a = strings.TrimPrefix(strings.TrimSpace(a), "v")
	b = strings.TrimPrefix(strings.TrimSpace(b), "v")
	aParts := strings.Split(a, ".")
	bParts := strings.Split(b, ".")
	for i := 0; i < len(aParts) && i < len(bParts); i++ {
		ai, aAhead, aOK := versionPartParts(aParts[i])
		bi, bAhead, bOK := versionPartParts(bParts[i])
		switch {
		case !aOK && !bOK:
			continue
		case !aOK:
			return -1
		case !bOK:
			return 1
		case ai < bi:
			return -1
		case ai > bi:
			return 1
		case ai == bi && aAhead != bAhead:
			if aAhead {
				return 1
			}
			return -1
		}
	}
	if len(aParts) < len(bParts) {
		return -1
	}
	if len(aParts) > len(bParts) {
		return 1
	}
	return 0
}

// git helpers
func gitRevParse(dir, rev string) (string, error) {
	out, err := exec.Command("git", "-C", dir, "rev-parse", rev).Output()
	return strings.TrimSpace(string(out)), err
}

func gitSymbolicRef(dir string) (string, error) {
	out, err := exec.Command("git", "-C", dir, "symbolic-ref", "--quiet", "--short", "HEAD").Output()
	return strings.TrimSpace(string(out)), err
}

// gitDescribeVersion reports the version to stamp a self-rebuilt binary with:
// the repo's current git describe (e.g. v3.2.8 or v3.2.8-4-g44f99a0), or "" if it
// cannot be determined.
func gitDescribeVersion(dir string) string {
	out, err := exec.Command("git", "-C", dir, "describe", "--tags", "--always").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func gitPullFFOnly(dir string) (string, error) {
	cmd := exec.Command("git", "-C", dir, "pull", "--ff-only")
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func gitPullRebase(dir string) (string, error) {
	cmd := exec.Command("git", "-C", dir, "pull", "--rebase")
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func gitStatusPorcelain(dir string) (string, error) {
	cmd := exec.Command("git", "-C", dir, "status", "--porcelain", "--untracked-files=no")
	out, err := cmd.Output()
	return strings.TrimSpace(string(out)), err
}

func gitCheckout(dir, rev string) error {
	return exec.Command("git", "-C", dir, "checkout", rev).Run()
}

func gitCheckoutQuiet(dir, rev string) error {
	return exec.Command("git", "-C", dir, "checkout", "--quiet", rev).Run()
}

func gitLogOneline(dir, rangeSpec string) ([]string, error) {
	out, err := exec.Command("git", "-C", dir, "log", "--oneline", rangeSpec).Output()
	if err != nil {
		return nil, err
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 1 && lines[0] == "" {
		return nil, nil
	}
	return lines, nil
}

func md5sum(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	h := md5.New()
	if _, err := io.Copy(h, f); err != nil {
		return ""
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func cp(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0755)
}
