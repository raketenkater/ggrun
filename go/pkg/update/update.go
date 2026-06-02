package update

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	githubRepo     = "raketenkater/llm-server"
	githubAPIURL   = "https://api.github.com/repos/%s/releases/latest"
	defaultVersion = "v3.0.0-go"
)

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
	return defaultVersion
}

// SelfUpdate pulls the latest llm-server from git and re-runs install.sh.
func SelfUpdate() error {
	repoDir := os.Getenv("LLM_SERVER_REPO")
	if repoDir == "" {
		repoDir = filepath.Join(os.Getenv("HOME"), "llm-server")
	}
	gitDir := filepath.Join(repoDir, ".git")
	if _, err := os.Stat(gitDir); err != nil {
		return fmt.Errorf("llm-server repo not found at %s — skipping self-update", repoDir)
	}

	fmt.Println("═══ Updating llm-server ═══")
	oldHash, err := gitRevParse(repoDir, "HEAD")
	if err != nil {
		oldHash = "unknown"
	}

	scriptPath, _ := exec.LookPath("llm-server")
	if scriptPath == "" {
		scriptPath = filepath.Join(os.Getenv("HOME"), ".local", "bin", "llm-server")
	}
	var backupPath string
	if _, err := os.Stat(scriptPath); err == nil {
		backupPath = scriptPath + ".bak"
		cp(scriptPath, backupPath)
	}

	if out, err := gitPullFFOnly(repoDir); err != nil {
		fmt.Printf("  Warning: git pull failed: %v\n", err)
		if backupPath != "" {
			os.Remove(backupPath)
		}
		return nil
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
				exec.Command("bash", installScript).Run()
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
	home := os.Getenv("HOME")
	ikDir := filepath.Join(home, "ik_llama.cpp")
	mainDir := filepath.Join(home, "llama.cpp")

	if _, err := os.Stat(ikDir); err == nil {
		if err := UpdateBackend("ik_llama.cpp", ikDir, 3); err != nil {
			fmt.Printf("  ik_llama.cpp update failed: %v\n", err)
		}
	} else {
		fmt.Printf("ik_llama.cpp not found at %s — skipping\n", ikDir)
	}

	if _, err := os.Stat(mainDir); err == nil {
		if err := UpdateBackend("llama.cpp", mainDir, 3); err != nil {
			fmt.Printf("  llama.cpp update failed: %v\n", err)
		}
	} else {
		fmt.Printf("llama.cpp not found at %s — skipping\n", mainDir)
	}
	return nil
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
