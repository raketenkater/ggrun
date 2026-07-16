package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/backends"
)

func fmtCustomBackend(b backends.Backend) string {
	route := b.RouteArch
	if route == "" {
		route = "(none)"
	}
	status := "ok"
	if _, err := os.Stat(b.Path); err != nil {
		status = "MISSING BINARY"
	}
	return fmt.Sprintf("  %-12s [%s] %s\n    route-arch: %s   src: %s @ %s", b.Tag, status, b.Path, route, b.GitURL, b.Branch)
}

func backendUsage() {
	fmt.Fprint(os.Stderr, `Usage: ggrun backend <subcommand>

  list                              List registered custom backends
  recipes                           List reviewed reproducible fork recipes
  install <recipe> [flags]          Build/register a reviewed recipe (for example hy3)
  add <git-url> [flags]             Clone, build, and register a custom llama.cpp backend
  register [flags]                  Register an already-built binary
  remove <tag>                      Unregister a backend

add/register flags:
  --tag <name>          Selection name (default: derived from URL)
  --route-arch <arch>   Auto-select this backend for models of this architecture
                        (e.g. custommoe), so it "just works" with no --backend
  --branch <branch>     Git branch to clone (add only; default: default branch)
  --commit <sha>        Pin the exact source commit (add only; recipes always pin)
  --accel cuda|vulkan|cpu   Build accelerator (add only; default: cuda if nvcc present)
  --cuda-arch <list>    CUDA archs, e.g. "86;89" (add only; default: native)
  --path <binary>       Path to a prebuilt llama-server (register only)

Examples:
  ggrun backend add https://github.com/your-org/llama.cpp \
    --branch feature/custom-arch --tag custom --route-arch custommoe --cuda-arch "86;89"
  ggrun backend install hy3 --cuda-arch "86;89"
  ggrun backend list
`)
}

func cmdBackend(args []string) {
	if len(args) == 0 {
		backendUsage()
		return
	}
	switch args[0] {
	case "list", "ls":
		cmdBackendList()
	case "recipes":
		cmdBackendRecipes()
	case "install":
		cmdBackendInstall(args[1:])
	case "add":
		cmdBackendAdd(args[1:])
	case "register":
		cmdBackendRegister(args[1:])
	case "remove", "rm":
		cmdBackendRemove(args[1:])
	default:
		backendUsage()
		os.Exit(2)
	}
}

func cmdBackendRecipes() {
	for _, recipe := range backends.Recipes() {
		fmt.Printf("  %-10s arch=%-12s %s\n    %s @ %s (%s)\n", recipe.Name, recipe.RouteArch, recipe.Description, recipe.GitURL, recipe.Commit, recipe.Branch)
	}
}

func cmdBackendInstall(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "install needs a recipe name; available recipes:")
		cmdBackendRecipes()
		os.Exit(2)
	}
	recipe := backends.RecipeByName(args[0])
	if recipe == nil {
		fmt.Fprintf(os.Stderr, "unknown backend recipe %q; available recipes:\n", args[0])
		cmdBackendRecipes()
		os.Exit(2)
	}
	generated := []string{
		recipe.GitURL,
		"--tag", recipe.Tag,
		"--branch", recipe.Branch,
		"--commit", recipe.Commit,
		"--route-arch", recipe.RouteArch,
	}
	if recipe.Accel != "" {
		generated = append(generated, "--accel", recipe.Accel)
	}
	// User build-only overrides such as --cuda-arch come last.
	generated = append(generated, args[1:]...)
	cmdBackendAddRecipe(generated, recipe)
}

func cmdBackendList() {
	list := backends.Load()
	if len(list) == 0 {
		fmt.Println("No custom backends registered. Add one with: ggrun backend add <git-url>")
		return
	}
	fmt.Printf("Registered custom backends (%s):\n", backends.ManifestPath())
	for _, b := range list {
		fmt.Println(fmtCustomBackend(b))
	}
}

// parseBackendFlags reads --key value / --key=value pairs and returns the map
// plus any leading positional (e.g. the git URL).
func parseBackendFlags(args []string) (positional string, flags map[string]string) {
	flags = map[string]string{}
	for i := 0; i < len(args); i++ {
		a := args[i]
		if !strings.HasPrefix(a, "--") {
			if positional == "" {
				positional = a
			}
			continue
		}
		key := strings.TrimPrefix(a, "--")
		if eq := strings.IndexByte(key, '='); eq >= 0 {
			flags[key[:eq]] = key[eq+1:]
			continue
		}
		if i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
			flags[key] = args[i+1]
			i++
		} else {
			flags[key] = "true"
		}
	}
	return positional, flags
}

func cmdBackendRegister(args []string) {
	_, f := parseBackendFlags(args)
	be := backends.Backend{
		Tag:       f["tag"],
		Path:      f["path"],
		RouteArch: f["route-arch"],
		GitURL:    f["git-url"],
		Branch:    f["branch"],
	}
	if be.Tag == "" || be.Path == "" {
		fmt.Fprintln(os.Stderr, "register needs --tag and --path")
		os.Exit(2)
	}
	if _, err := os.Stat(be.Path); err != nil {
		fmt.Fprintf(os.Stderr, "binary not found: %s\n", be.Path)
		os.Exit(1)
	}
	if err := backends.Upsert(be); err != nil {
		fmt.Fprintf(os.Stderr, "register failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Registered backend %q → %s (route-arch: %s)\n", be.Tag, be.Path, be.RouteArch)
}

func cmdBackendRemove(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "remove needs a tag")
		os.Exit(2)
	}
	found, err := backends.Remove(args[0])
	if err != nil {
		fmt.Fprintf(os.Stderr, "remove failed: %v\n", err)
		os.Exit(1)
	}
	if !found {
		fmt.Fprintf(os.Stderr, "no backend tagged %q\n", args[0])
		os.Exit(1)
	}
	fmt.Printf("Removed backend %q\n", args[0])
}

func cmdBackendAdd(args []string) {
	cmdBackendAddRecipe(args, nil)
}

func cmdBackendAddRecipe(args []string, recipe *backends.Recipe) {
	url, f := parseBackendFlags(args)
	if url == "" {
		url = f["url"]
	}
	if url == "" {
		fmt.Fprintln(os.Stderr, "add needs a git URL")
		backendUsage()
		os.Exit(2)
	}
	branch := f["branch"]
	commit := f["commit"]
	tag := f["tag"]
	if tag == "" {
		tag = deriveBackendTag(url, branch)
	}
	accel := f["accel"]
	if accel == "" {
		accel = defaultAccel()
	}

	// Clone into .src/fork-<repo>-<branch> so it never clobbers the mainline checkout.
	name := deriveBackendTag(url, branch)
	srcDir := filepath.Join(backends.AppHome(), ".src", "fork-"+name)
	if _, err := os.Stat(filepath.Join(srcDir, ".git")); err != nil {
		fmt.Printf("[backend] cloning %s%s → %s\n", url, branchNote(branch), srcDir)
		cloneArgs := []string{"clone", "--depth", "1"}
		if branch != "" {
			cloneArgs = append(cloneArgs, "-b", branch)
		}
		cloneArgs = append(cloneArgs, url, srcDir)
		if err := runStreamed("", "git", cloneArgs...); err != nil {
			fmt.Fprintf(os.Stderr, "clone failed: %v\n", err)
			os.Exit(1)
		}
	} else {
		fmt.Printf("[backend] reusing existing checkout %s\n", srcDir)
	}
	if err := prepareForkCheckoutRecipe(srcDir, branch, commit, recipe); err != nil {
		fmt.Fprintf(os.Stderr, "source checkout failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("[backend] building (%s)… this can take 30–60 min\n", accel)
	bin, err := buildLlamaFork(srcDir, accel, f["cuda-arch"])
	if err != nil {
		fmt.Fprintf(os.Stderr, "build failed: %v\n", err)
		os.Exit(1)
	}

	actualCommit, _ := gitOutput(srcDir, "rev-parse", "HEAD")
	be := backends.Backend{Tag: tag, Path: bin, RouteArch: f["route-arch"], GitURL: url, Branch: branch, Commit: strings.TrimSpace(actualCommit)}
	if recipe != nil {
		be.AppliedPatches = recipe.PatchNames()
	}
	if err := backends.Upsert(be); err != nil {
		fmt.Fprintf(os.Stderr, "register failed: %v\n", err)
		os.Exit(1)
	}
	// Symlink into .bin so it also shows up in the normal backend search paths.
	link := filepath.Join(backends.AppHome(), ".bin", tag+"-server-"+accel)
	_ = os.Remove(link)
	if err := os.Symlink(bin, link); err == nil {
		fmt.Printf("[backend] linked %s\n", link)
	}
	fmt.Printf("Registered backend %q → %s\n", tag, bin)
	if be.RouteArch != "" {
		fmt.Printf("Models with arch %q will now use this backend automatically.\n", be.RouteArch)
	}
}

// prepareForkCheckout updates a ggrun-owned fork checkout without trampling
// local edits. A commit pin wins; otherwise the requested branch (or remote
// default HEAD) advances to its latest fetched revision.
func prepareForkCheckout(srcDir, branch, commit string) error {
	return prepareForkCheckoutRecipe(srcDir, branch, commit, nil)
}

func prepareForkCheckoutRecipe(srcDir, branch, commit string, recipe *backends.Recipe) error {
	dirty, err := gitOutput(srcDir, "status", "--porcelain")
	if err != nil {
		return err
	}
	if strings.TrimSpace(dirty) != "" {
		if recipe != nil {
			if err := recipe.RevertPatches(srcDir); err != nil {
				return fmt.Errorf("%s has local changes (%w)", srcDir, err)
			}
			dirty, err = gitOutput(srcDir, "status", "--porcelain")
			if err != nil {
				return err
			}
		}
	}
	if strings.TrimSpace(dirty) != "" {
		return fmt.Errorf("%s has local changes; commit/stash them or use a different recipe tag", srcDir)
	}
	ref := commit
	if ref == "" {
		ref = "HEAD"
		if branch != "" {
			ref = branch
		}
	}
	fetchArgs := []string{"fetch", "--depth", "1", "origin", ref}
	if err := runStreamed(srcDir, "git", fetchArgs...); err != nil {
		return fmt.Errorf("git fetch %s: %w", ref, err)
	}
	checkout := "FETCH_HEAD"
	if commit != "" {
		checkout = commit
	}
	if err := runStreamed(srcDir, "git", "checkout", "--detach", checkout); err != nil {
		return fmt.Errorf("git checkout %s: %w", checkout, err)
	}
	if commit != "" {
		actual, err := gitOutput(srcDir, "rev-parse", "HEAD")
		if err != nil {
			return err
		}
		if !strings.EqualFold(strings.TrimSpace(actual), strings.TrimSpace(commit)) {
			return fmt.Errorf("commit verification failed: got %s, want %s", strings.TrimSpace(actual), commit)
		}
	}
	if recipe != nil {
		if err := recipe.ApplyPatches(srcDir); err != nil {
			return err
		}
	}
	return nil
}

func gitOutput(dir string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("git %s: %s: %w", strings.Join(args, " "), strings.TrimSpace(string(out)), err)
	}
	return string(out), nil
}

// deriveBackendTag builds a stable name from a git URL (+branch).
func deriveBackendTag(url, branch string) string {
	base := url
	if i := strings.LastIndexByte(base, '/'); i >= 0 {
		base = base[i+1:]
	}
	base = strings.TrimSuffix(base, ".git")
	if branch != "" {
		b := branch
		if i := strings.LastIndexByte(b, '/'); i >= 0 {
			b = b[i+1:]
		}
		base = base + "-" + b
	}
	base = strings.ToLower(base)
	base = strings.NewReplacer("/", "-", " ", "-", "_", "-").Replace(base)
	if base == "" {
		base = "fork"
	}
	return base
}

func branchNote(branch string) string {
	if branch == "" {
		return ""
	}
	return " (" + branch + ")"
}

// defaultAccel picks cuda when the CUDA toolkit is present, else cpu.
func defaultAccel() string {
	if _, err := exec.LookPath("nvcc"); err == nil {
		return "cuda"
	}
	for _, p := range []string{"/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.2/bin/nvcc"} {
		if _, err := os.Stat(p); err == nil {
			return "cuda"
		}
	}
	return "cpu"
}

// buildLlamaFork cmake-configures and builds llama-server for the given
// accelerator. Returns the built binary path.
func buildLlamaFork(srcDir, accel, cudaArch string) (string, error) {
	buildDir := filepath.Join(srcDir, "build-"+accel)
	cfg := []string{"-B", buildDir, "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"}
	switch accel {
	case "cuda":
		if cudaArch == "" {
			cudaArch = "native"
		}
		cfg = append(cfg, "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES="+cudaArch)
	case "vulkan":
		cfg = append(cfg, "-DGGML_VULKAN=ON")
	case "cpu":
		// default CPU build
	default:
		return "", fmt.Errorf("unknown --accel %q (use cuda|vulkan|cpu)", accel)
	}
	if err := runStreamed(srcDir, "cmake", cfg...); err != nil {
		return "", fmt.Errorf("cmake configure: %w", err)
	}
	jobs := backendBuildJobs(accel, runtime.NumCPU())
	if err := runStreamed(srcDir, "cmake", "--build", buildDir, "--config", "Release", "--parallel", strconv.Itoa(jobs), "--target", "llama-server"); err != nil {
		return "", fmt.Errorf("cmake build: %w", err)
	}
	bin := filepath.Join(buildDir, "bin", "llama-server")
	if _, err := os.Stat(bin); err != nil {
		return "", fmt.Errorf("build produced no binary at %s", bin)
	}
	return bin, nil
}

func backendBuildJobs(accel string, cpuCount int) int {
	if cpuCount < 1 {
		cpuCount = 1
	}
	limit := 16
	if accel == "cuda" {
		// NVCC fans each translation unit into several memory-heavy processes;
		// an unbounded `cmake -j` can create hundreds of them on large hosts.
		limit = 8
	}
	if cpuCount > limit {
		return limit
	}
	return cpuCount
}

// runStreamed runs a command in dir with CUDA on PATH, streaming its output.
func runStreamed(dir, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	if dir != "" {
		cmd.Dir = dir
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	env := os.Environ()
	if _, err := exec.LookPath("nvcc"); err != nil {
		for _, c := range []string{"/usr/local/cuda/bin", "/usr/local/cuda-13.2/bin"} {
			if _, err := os.Stat(filepath.Join(c, "nvcc")); err == nil {
				env = append(env, "PATH="+c+":"+os.Getenv("PATH"))
				break
			}
		}
	}
	cmd.Env = env
	return cmd.Run()
}
