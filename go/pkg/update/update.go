package update

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	githubRepo        = "raketenkater/llm-server"
	githubAPIURL      = "https://api.github.com/repos/%s/releases/latest"
	rawInstallURL     = "https://raw.githubusercontent.com/%s/%s/install.sh"
	updateDismissDays = 7
)

// currentVersion is the single source of truth for the binary version.
// Release builds override it: go build -ldflags "-X github.com/raketenkater/llm-server/pkg/update.currentVersion=vX.Y.Z"
var currentVersion = "v3.0.0-go"

// PromptOnStartup checks local repos for updates and asks interactive users
// whether to run the updater. It intentionally skips non-interactive shells so
// scripts and CI never block on network or stdin.
func PromptOnStartup() {
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
		if err := UpdateBackends(); err != nil {
			fmt.Fprintf(os.Stderr, "Backend update: %v\n", err)
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
	if hasUpdateLabel(updates, "llm-server") {
		return updates
	}
	res, err := Check()
	if err == nil && res.HasUpdate {
		updates = append(updates, "llm-server "+res.Latest)
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
	for _, repo := range updateRepoCandidates() {
		if repoBehind(repo.Dir) {
			updates = append(updates, repo.Label)
		}
	}
	return updates
}

type repoCandidate struct {
	Label string
	Dir   string
}

func updateRepoCandidates() []repoCandidate {
	home := os.Getenv("HOME")
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
		repoDir = filepath.Join(home, "llm-server")
	}
	add("llm-server", repoDir)

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
		add("llm-server", filepath.Join(appHome, ".src", "llm-server"))
		add("ik_llama.cpp", filepath.Join(appHome, ".src", "ik_llama.cpp"))
		add("llama.cpp", filepath.Join(appHome, ".src", "llama.cpp"))
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
	if home := os.Getenv("HOME"); home != "" {
		return filepath.Join(home, ".cache", "llm-server")
	}
	return filepath.Join(os.TempDir(), "llm-server")
}

func isTerminal(f *os.File) bool {
	if f == nil {
		return false
	}
	info, err := f.Stat()
	return err == nil && (info.Mode()&os.ModeCharDevice) != 0
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

// SelfUpdate pulls the latest llm-server from git and re-runs install.sh.
func SelfUpdate() error {
	repoDir := installedSourceRepoDir()
	gitDir := filepath.Join(repoDir, ".git")
	if _, err := os.Stat(gitDir); err != nil {
		if appHome := os.Getenv("LLM_APP_HOME"); appHome != "" {
			fmt.Printf("llm-server repo not found at %s; refreshing app home from main.\n", repoDir)
			return SelfUpdateAppHomeInstaller(appHome)
		}
		fmt.Printf("llm-server repo not found at %s; using latest release installer.\n", repoDir)
		return SelfUpdateFromReleaseInstaller()
	}

	fmt.Println("═══ Updating llm-server ═══")
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
		return fmt.Errorf("git pull failed: %v\n%s", err, strings.TrimSpace(out))
	} else {
		fmt.Println(strings.TrimSpace(out))
	}

	newHash, _ := gitRevParse(repoDir, "HEAD")
	if oldHash == newHash {
		fmt.Println("  Already up to date.")
		if backupPath != "" {
			os.Remove(backupPath)
		}
		return nil
	}

	commits, _ := gitLogOneline(repoDir, oldHash+".."+newHash)
	fmt.Printf("  Updated: %d new commits\n", len(commits))
	for _, c := range commits {
		if len(c) > 60 {
			c = c[:60] + "..."
		}
		fmt.Printf("    %s\n", c)
	}

	installScript := filepath.Join(repoDir, "install.sh")
	if _, err := os.Stat(installScript); err == nil {
		fmt.Println("  Re-installing...")
		cmd := exec.Command("bash", installScript)
		cmd.Dir = repoDir
		cmd.Env = selfUpdateInstallEnv(os.Getenv("LLM_APP_HOME"))
		out, err := cmd.CombinedOutput()
		if err != nil {
			fmt.Println("  Error: Install failed. Rolling back...")
			gitCheckout(repoDir, oldHash)
			if backupPath != "" {
				cp(backupPath, scriptPath)
			}
			os.Remove(backupPath)
			return fmt.Errorf("install failed: %w", err)
		}
		_ = out
	}

	// Self-check: can the new script run --version?
	if scriptPath != "" {
		cmd := exec.Command(scriptPath, "--version")
		if err := cmd.Run(); err != nil {
			fmt.Println("  Error: New version failed self-check. Rolling back...")
			gitCheckout(repoDir, oldHash)
			if backupPath != "" {
				cp(backupPath, scriptPath)
			}
			if _, err := os.Stat(installScript); err == nil {
				cmd := exec.Command("bash", installScript)
				cmd.Env = selfUpdateInstallEnv(os.Getenv("LLM_APP_HOME"))
				_ = cmd.Run()
			}
			os.Remove(backupPath)
			return fmt.Errorf("self-check failed")
		}
	}

	if backupPath != "" {
		os.Remove(backupPath)
	}
	fmt.Println("  ✓ llm-server updated and verified. Restart to use the new version.")
	return nil
}

// SelfUpdateFromReleaseInstaller updates release-bundle installs that do not have a
// local llm-server git checkout. It downloads the latest tagged install.sh and lets
// the installer select the right platform/backend bundle or source fallback.
func SelfUpdateFromReleaseInstaller() error {
	fmt.Println("═══ Updating llm-server from latest release installer ═══")
	scriptPath := installedLLMServerPath()
	backupPath := ""
	if scriptPath != "" {
		if _, err := os.Stat(scriptPath); err == nil {
			backupPath = scriptPath + ".bak"
			_ = cp(scriptPath, backupPath)
		}
	}

	installerURL := rawInstallerURL("main")
	if res, err := Check(); err == nil && res.Latest != "" {
		installerURL = rawInstallerURL(res.Latest)
	}
	tmpDir, err := os.MkdirTemp("", "llm-server-update-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)
	installerPath := filepath.Join(tmpDir, "install.sh")
	if err := downloadFile(installerURL, installerPath, 0755); err != nil {
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
	fmt.Println("  ✓ llm-server release installer completed and verified. Restart to use the new version.")
	return nil
}

func rawInstallerURL(ref string) string {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		ref = "main"
	}
	return fmt.Sprintf(rawInstallURL, githubRepo, ref)
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
		"LLM_INSTALL_REPO_DIR="+filepath.Join(appHome, ".src", "llm-server"),
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
	fmt.Println("═══ Updating llm-server app home from main ═══")
	scriptPath := installedLLMServerPath()
	backupPath := ""
	if scriptPath != "" {
		if _, err := os.Stat(scriptPath); err == nil {
			backupPath = scriptPath + ".bak"
			_ = cp(scriptPath, backupPath)
		}
	}

	tmpDir, err := os.MkdirTemp("", "llm-server-update-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)
	installerPath := filepath.Join(tmpDir, "install.sh")
	if err := downloadFile(rawInstallerURL("main"), installerPath, 0755); err != nil {
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
	fmt.Println("  ✓ llm-server app home updated and verified. Restart to use the new version.")
	return nil
}

func installedSourceRepoDir() string {
	if repoDir := strings.TrimSpace(os.Getenv("LLM_SERVER_REPO")); repoDir != "" {
		return repoDir
	}
	if appHome := strings.TrimSpace(os.Getenv("LLM_APP_HOME")); appHome != "" {
		repoDir := filepath.Join(appHome, ".src", "llm-server")
		if _, err := os.Stat(filepath.Join(repoDir, ".git")); err == nil {
			return repoDir
		}
	}
	if home := strings.TrimSpace(os.Getenv("HOME")); home != "" {
		return filepath.Join(home, "llm-server")
	}
	return ""
}

func installedLLMServerPath() string {
	if appHome := os.Getenv("LLM_APP_HOME"); appHome != "" {
		for _, candidate := range []string{
			filepath.Join(appHome, ".bin", "llm-server"),
			filepath.Join(appHome, "llm-server"),
		} {
			if _, err := os.Stat(candidate); err == nil {
				return candidate
			}
		}
	}
	if path, _ := exec.LookPath("llm-server"); path != "" {
		return path
	}
	if home := os.Getenv("HOME"); home != "" {
		return filepath.Join(home, ".local", "bin", "llm-server")
	}
	return ""
}

func downloadFile(url, dst string, mode os.FileMode) error {
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("%s", resp.Status)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, mode)
}

// UpdateBackend updates a backend repo (ik_llama.cpp or llama.cpp).
func UpdateBackend(name, repoDir string, walkback int) error {
	buildDir := filepath.Join(repoDir, "build")
	binary := filepath.Join(buildDir, "bin", "llama-server")
	fallbackDir := filepath.Join(os.Getenv("HOME"), ".cache", "llm-server", "update-fallbacks")
	os.MkdirAll(fallbackDir, 0755)

	fmt.Printf("\n═══ Updating %s ═══\n", name)
	if _, err := os.Stat(filepath.Join(repoDir, ".git")); err != nil {
		fmt.Printf("  Skip: %s is not a git repo\n", repoDir)
		return fmt.Errorf("not a git repo")
	}

	oldCommit, _ := gitRevParse(repoDir, "HEAD")
	branch, _ := gitSymbolicRef(repoDir)
	if branch == "" {
		branch = "master"
	}

	oldHash := ""
	if _, err := os.Stat(binary); err == nil {
		oldHash = md5sum(binary)
	}
	fmt.Printf("  Current: %s\n", oldCommit)

	// Backup working binary outside build dir
	binaryBackup := filepath.Join(repoDir, ".llm-server.llama-server.backup")
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
		fmt.Println("  Already up to date.")
		os.Remove(binaryBackup)
		return nil
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
		if buildAndTest(buildDir, binary) {
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

// UpdateBackends updates both ik_llama.cpp and llama.cpp if present.
func UpdateBackends() error {
	found := map[string]bool{}
	for _, repo := range backendUpdateCandidates() {
		if _, err := os.Stat(repo.Dir); err != nil {
			continue
		}
		found[repo.Label] = true
		if err := UpdateBackend(repo.Label, repo.Dir, 3); err != nil {
			fmt.Printf("  %s update failed: %v\n", repo.Label, err)
		}
	}
	if !found["ik_llama.cpp"] {
		fmt.Println("ik_llama.cpp not found — skipping")
	}
	if !found["llama.cpp"] {
		fmt.Println("llama.cpp not found — skipping")
	}
	return nil
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
	if home := os.Getenv("HOME"); home != "" {
		add("ik_llama.cpp", filepath.Join(home, "ik_llama.cpp"))
		add("llama.cpp", filepath.Join(home, "llama.cpp"))
	}
	return candidates
}

func buildAndTest(buildDir, binary string) bool {
	nproc := 8
	if out, err := exec.Command("nproc").Output(); err == nil {
		nproc, _ = strconv.Atoi(strings.TrimSpace(string(out)))
		if nproc < 1 {
			nproc = 8
		}
	}

	fmt.Println("  Building...")
	cmd := exec.Command("cmake", "--build", buildDir, "--config", "Release", "-j", strconv.Itoa(nproc))
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println("  Build failed — trying clean reconfigure...")
		cmakeFlags := collectCMakeFlags(buildDir)
		os.RemoveAll(buildDir)
		cmd1 := exec.Command("cmake", append([]string{"-B", buildDir, "-DCMAKE_BUILD_TYPE=Release"}, cmakeFlags...)...)
		out1, err1 := cmd1.CombinedOutput()
		if err1 != nil {
			fmt.Printf("  Configure failed: %s\n", string(out1))
			return false
		}
		cmd2 := exec.Command("cmake", "--build", buildDir, "--config", "Release", "-j", strconv.Itoa(nproc))
		out2, err2 := cmd2.CombinedOutput()
		if err2 != nil {
			fmt.Printf("  Build failed at this commit.\n")
			_ = out2
			return false
		}
		_ = out
		fmt.Println("  Clean rebuild succeeded")
	} else {
		lines := strings.Split(string(out), "\n")
		for i := len(lines) - 5; i < len(lines); i++ {
			if i >= 0 {
				fmt.Println("  " + lines[i])
			}
		}
	}

	if _, err := os.Stat(binary); err != nil {
		fmt.Println("  Binary missing after build.")
		return false
	}

	// Shallow smoke: --version exits 0
	cmd = exec.Command(binary, "--version")
	if err := cmd.Run(); err != nil {
		fmt.Println("  Binary crashes on --version at this commit.")
		return false
	}
	return true
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

func compareVersions(a, b string) int {
	a = strings.TrimPrefix(a, "v")
	b = strings.TrimPrefix(b, "v")
	aParts := strings.Split(a, ".")
	bParts := strings.Split(b, ".")
	for i := 0; i < len(aParts) && i < len(bParts); i++ {
		ai, _ := strconv.Atoi(aParts[i])
		bi, _ := strconv.Atoi(bParts[i])
		if ai < bi {
			return -1
		}
		if ai > bi {
			return 1
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
	out, err := exec.Command("md5sum", path).Output()
	if err != nil {
		return ""
	}
	parts := strings.Fields(string(out))
	if len(parts) > 0 {
		return parts[0]
	}
	return ""
}

func cp(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0755)
}
