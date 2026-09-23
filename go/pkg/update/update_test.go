package update

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestShouldCheckStartupUpdatesDismissal(t *testing.T) {
	dir := t.TempDir()
	now := time.Unix(1_700_000_000, 0)
	if !shouldCheckStartupUpdates(dir, now) {
		t.Fatal("expected missing dismiss file to allow update check")
	}
	if err := dismissStartupUpdates(dir, now); err != nil {
		t.Fatalf("dismiss: %v", err)
	}
	if shouldCheckStartupUpdates(dir, now.Add(24*time.Hour)) {
		t.Fatal("expected recent dismiss file to suppress update check")
	}
	if !shouldCheckStartupUpdates(dir, now.Add(time.Duration(updateDismissDays)*24*time.Hour)) {
		t.Fatal("expected expired dismiss file to allow update check")
	}
}

func TestUpdateCacheDirUsesEnv(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "cache")
	t.Setenv("LLM_CACHE_DIR", dir)
	if got := updateCacheDir(); got != dir {
		t.Fatalf("cache dir mismatch: %s", got)
	}
}

func TestVersionUsesEnv(t *testing.T) {
	t.Setenv("LLM_SERVER_VERSION", "v9.9.9")
	if got := Version(); got != "v9.9.9" {
		t.Fatalf("version env mismatch: %s", got)
	}
}

func TestCompareVersionsWithSuffix(t *testing.T) {
	if compareVersions("v3.0.0-go", "v3.0.1") >= 0 {
		t.Fatal("expected v3.0.1 to be newer than v3.0.0-go")
	}
	if compareVersions("v3.1.0", "v3.0.9") <= 0 {
		t.Fatal("expected v3.1.0 to compare newer than v3.0.9")
	}
	// A build carrying git-describe commits-ahead is not older than the bare tag.
	if compareVersions("v3.2.8-4-g44f99a0", "v3.2.8") <= 0 {
		t.Fatal("expected a build 4 commits ahead of v3.2.8 to compare newer than v3.2.8")
	}
	if compareVersions("v3.2.8", "v3.2.8-4-g44f99a0") >= 0 {
		t.Fatal("expected bare v3.2.8 to compare older than the ahead build")
	}
	if compareVersions("v3.2.8-4-g44f99a0", "v3.2.9") >= 0 {
		t.Fatal("expected v3.2.9 to still be newer than a build ahead of v3.2.8")
	}
}

func TestSmokeBackendRequiresServerSurfaceAndConfiguredDevice(t *testing.T) {
	write := func(name, help, devices string) string {
		path := filepath.Join(t.TempDir(), name)
		body := "#!/bin/sh\ncase \"$1\" in\n" +
			"  --version) echo 'llama-server test' ;;\n" +
			"  --help) echo '" + help + "' ;;\n" +
			"  --list-devices) echo 'Available devices:'; echo '" + devices + "' ;;\n" +
			"  *) exit 1 ;;\nesac\n"
		if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
			t.Fatal(err)
		}
		return path
	}
	good := write("good", "--model --ctx-size --host --port", "CUDA0: test")
	if err := smokeBackendConfigured(good, []string{"-DGGML_CUDA=ON"}); err != nil {
		t.Fatalf("valid configured backend rejected: %v", err)
	}
	bad := write("bad", "--model --host --port", "CUDA0: test")
	if err := smokeBackend(bad); err == nil || !strings.Contains(err.Error(), "--ctx-size") {
		t.Fatalf("incomplete server surface accepted: %v", err)
	}
	wrongDevice := write("wrong-device", "--model --ctx-size --host --port", "Vulkan0: test")
	if err := smokeBackendConfigured(wrongDevice, []string{"-DGGML_CUDA=ON"}); err == nil || !strings.Contains(err.Error(), "cuda") {
		t.Fatalf("wrong accelerator accepted: %v", err)
	}
}

func TestUpdateDismissPath(t *testing.T) {
	dir := t.TempDir()
	want := filepath.Join(dir, "update_dismissed")
	got := updateDismissPath(dir)
	if got != want {
		t.Fatalf("path mismatch: %s", got)
	}
	if err := os.WriteFile(got, []byte("0\n"), 0644); err != nil {
		t.Fatalf("write dismiss path: %v", err)
	}
}

func TestParseSHA256SUMSAndVerifyInstallerName(t *testing.T) {
	sums := parseSHA256SUMS([]byte("" +
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  install.sh\n" +
		"fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210 *install.ps1\n"))
	if sums["install.sh"] != "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" {
		t.Fatalf("install.sh sum = %q", sums["install.sh"])
	}
	if sums["install.ps1"] != "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210" {
		t.Fatalf("install.ps1 sum = %q", sums["install.ps1"])
	}
}

func TestDownloadVerifiedInstallerRefusesUnsignedRefs(t *testing.T) {
	dst := filepath.Join(t.TempDir(), "install.sh")
	if err := downloadVerifiedInstaller("main", "install.sh", dst, 0755); err == nil {
		t.Fatal("expected main installer download to fail")
	}
	if err := downloadVerifiedInstaller("", "install.sh", dst, 0755); err == nil {
		t.Fatal("expected empty tag to fail")
	}
	if err := downloadVerifiedInstaller("v3.2.0", "setup.sh", dst, 0755); err == nil {
		t.Fatal("expected unsupported installer name to fail")
	}
}

func TestRawInstallerURL(t *testing.T) {
	want := "https://raw.githubusercontent.com/raketenkater/ggrun/v3.0.1/install.sh"
	if got := rawInstallerURL("v3.0.1"); got != want {
		t.Fatalf("installer URL mismatch: %s", got)
	}
	want = "https://raw.githubusercontent.com/raketenkater/ggrun/main/install.sh"
	if got := rawInstallerURL(""); got != want {
		t.Fatalf("default installer URL mismatch: %s", got)
	}
}

func TestHasUpdateLabel(t *testing.T) {
	if !hasUpdateLabel([]string{"ggrun v3.0.1"}, "ggrun") {
		t.Fatal("expected prefixed ggrun release label to match")
	}
	if hasUpdateLabel([]string{"llama.cpp"}, "ggrun") {
		t.Fatal("unexpected ggrun match")
	}
}

func envHas(env []string, want string) bool {
	for _, item := range env {
		if item == want {
			return true
		}
	}
	return false
}

func TestSelfUpdateInstallEnvPreservesAppHome(t *testing.T) {
	appHome := filepath.Join(t.TempDir(), "ggrun")
	env := selfUpdateInstallEnv(appHome)
	checks := []string{
		"LLM_APP_HOME=" + appHome,
		"LLM_INSTALL_PREFIX=" + filepath.Join(appHome, ".bin"),
		"LLM_INSTALL_MODEL_DIR=" + filepath.Join(appHome, "models"),
		"LLM_INSTALL_BACKEND_ROOT=" + filepath.Join(appHome, ".src"),
		"LLM_INSTALL_REPO_DIR=" + filepath.Join(appHome, ".src", "ggrun"),
		"LLM_INSTALL_REF=main",
		"LLM_INSTALL_BACKEND=skip",
		"LLM_INSTALL_MODE=build",
		"LLM_INSTALL_MAIN=go",
		"LLM_INSTALL_NONINTERACTIVE=1",
	}
	for _, want := range checks {
		if !envHas(env, want) {
			t.Fatalf("missing env %q in %#v", want, env)
		}
	}
}

func TestInstalledPathPrefersAppHomeBinary(t *testing.T) {
	appHome := t.TempDir()
	binDir := filepath.Join(appHome, ".bin")
	if err := os.MkdirAll(binDir, 0755); err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(binDir, "ggrun")
	if err := os.WriteFile(want, []byte("#!/bin/sh\n"), 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLM_APP_HOME", appHome)
	if got := installedLLMServerPath(); got != want {
		t.Fatalf("installed path mismatch: got %s want %s", got, want)
	}
}

func TestSourceRepoFromExecutableResolvesCanonicalSymlink(t *testing.T) {
	repo := filepath.Join(t.TempDir(), "ggrun")
	if err := os.MkdirAll(filepath.Join(repo, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(repo, "ggrun")
	if err := os.WriteFile(binary, []byte("binary"), 0o755); err != nil {
		t.Fatal(err)
	}
	linkDir := t.TempDir()
	link := filepath.Join(linkDir, "ggrun")
	if err := os.Symlink(binary, link); err != nil {
		t.Fatal(err)
	}
	if got := sourceRepoFromExecutable(link); got != repo {
		t.Fatalf("source repo from symlink = %q, want %q", got, repo)
	}
}

func TestInstalledPathResolvesPATHSymlink(t *testing.T) {
	t.Setenv("LLM_APP_HOME", "")
	dir := t.TempDir()
	canonical := filepath.Join(t.TempDir(), "ggrun")
	if err := os.WriteFile(canonical, []byte("binary"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(canonical, filepath.Join(dir, "ggrun")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
	if got := installedLLMServerPath(); got != canonical {
		t.Fatalf("installed path = %q, want resolved canonical %q", got, canonical)
	}
}

func TestSelfUpdateRefusesDirtySourceBeforeNetworkOrBuild(t *testing.T) {
	repo := t.TempDir()
	run := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	run("init", "-q")
	tracked := filepath.Join(repo, "tracked.txt")
	if err := os.WriteFile(tracked, []byte("one\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	run("add", "tracked.txt")
	run("-c", "user.name=ggrun-test", "-c", "user.email=ggrun@example.invalid", "commit", "-qm", "initial")
	if err := os.WriteFile(tracked, []byte("two\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLM_SERVER_REPO", repo)
	if err := SelfUpdate(); err == nil || !strings.Contains(err.Error(), "tracked changes") {
		t.Fatalf("dirty self-update error = %v, want tracked-changes refusal", err)
	}
}

func TestBackendUpdateCandidatesIncludeAppHomeSource(t *testing.T) {
	appHome := filepath.Join(t.TempDir(), "ggrun")
	t.Setenv("LLM_APP_HOME", appHome)
	t.Setenv("HOME", filepath.Join(t.TempDir(), "home"))
	rows := backendUpdateCandidates()
	want := map[string]string{
		"ik_llama.cpp": filepath.Join(appHome, ".src", "ik_llama.cpp"),
		"llama.cpp":    filepath.Join(appHome, ".src", "llama.cpp"),
	}
	for label, dir := range want {
		found := false
		for _, row := range rows {
			if row.Label == label && row.Dir == dir {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing backend candidate %s %s in %#v", label, dir, rows)
		}
	}
}

func TestBackendBuildTargetsFollowCanonicalAppHomeLinks(t *testing.T) {
	appHome := t.TempDir()
	binDir := filepath.Join(appHome, ".bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	makeBuild := func(repoName, variant string) string {
		repo := filepath.Join(t.TempDir(), repoName)
		if err := os.MkdirAll(filepath.Join(repo, ".git"), 0o755); err != nil {
			t.Fatal(err)
		}
		binary := filepath.Join(repo, variant, "bin", "llama-server")
		if err := os.MkdirAll(filepath.Dir(binary), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(binary, []byte("server"), 0o755); err != nil {
			t.Fatal(err)
		}
		return binary
	}
	mainline := makeBuild("llama.cpp", "build-cuda")
	ik := makeBuild("ik_llama.cpp", "build")
	if err := os.Symlink(mainline, filepath.Join(binDir, "llama-server-cuda")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(ik, filepath.Join(binDir, "ik_llama-server-cuda")); err != nil {
		t.Fatal(err)
	}
	// The generic link resolves to the same ik build and must be deduplicated.
	if err := os.Symlink(ik, filepath.Join(binDir, "llama-server")); err != nil {
		t.Fatal(err)
	}

	targets := BackendBuildTargetsAt(appHome)
	if len(targets) != 2 {
		t.Fatalf("canonical targets = %#v, want exactly mainline + ik", targets)
	}
	got := map[string]string{}
	for _, target := range targets {
		got[target.Label] = target.BuildDir
	}
	if got["llama.cpp (cuda)"] != filepath.Dir(filepath.Dir(mainline)) {
		t.Fatalf("mainline target mismatch: %#v", got)
	}
	if got["ik_llama.cpp (cuda)"] != filepath.Dir(filepath.Dir(ik)) {
		t.Fatalf("ik target mismatch: %#v", got)
	}
}

func TestUpdateMainlineBackendAtAppHomeFiltersToMainline(t *testing.T) {
	appHome := t.TempDir()
	binDir := filepath.Join(appHome, ".bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	makeBuild := func(repoName, variant string) string {
		repo := filepath.Join(t.TempDir(), repoName)
		if err := os.MkdirAll(filepath.Join(repo, ".git"), 0o755); err != nil {
			t.Fatal(err)
		}
		binary := filepath.Join(repo, variant, "bin", "llama-server")
		if err := os.MkdirAll(filepath.Dir(binary), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(binary, []byte("server"), 0o755); err != nil {
			t.Fatal(err)
		}
		return binary
	}
	mainline := makeBuild("llama.cpp", "build-cuda")
	ik := makeBuild("ik_llama.cpp", "build")
	if err := os.Symlink(mainline, filepath.Join(binDir, "llama-server-cuda")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(ik, filepath.Join(binDir, "ik_llama-server-cuda")); err != nil {
		t.Fatal(err)
	}

	// The scoped update must only see the mainline llama.cpp build, never the
	// ik_llama.cpp family. Reproduce the filter UpdateMainlineBackendAtAppHome
	// applies to BackendBuildTargetsAt so the scope is asserted without running a
	// real pull/build.
	filtered := make([]BackendBuildTarget, 0)
	for _, target := range BackendBuildTargetsAt(appHome) {
		if strings.EqualFold(filepath.Base(target.RepoDir), "llama.cpp") {
			filtered = append(filtered, target)
		}
	}
	if len(filtered) != 1 || filepath.Base(filtered[0].RepoDir) != "llama.cpp" {
		t.Fatalf("mainline filter = %#v, want only llama.cpp", filtered)
	}
	// No active mainline build (empty filtered set) is a no-op, not an error.
	emptyHome := t.TempDir()
	if err := UpdateMainlineBackendAtAppHome(emptyHome); err != nil {
		t.Fatalf("empty app home should no-op, got: %v", err)
	}
}

func TestBackendUpdateGroupsContinueAfterIndependentFailure(t *testing.T) {
	targets := []BackendBuildTarget{
		{Label: "mainline cuda", RepoDir: "/repo/main", BuildDir: "/repo/main/build-cuda"},
		{Label: "mainline vulkan", RepoDir: "/repo/main", BuildDir: "/repo/main/build-vulkan"},
		{Label: "ik", RepoDir: "/repo/ik", BuildDir: "/repo/ik/build"},
	}
	var called []string
	runner := func(repo string, group []BackendBuildTarget, _ int) []BackendUpdateResult {
		called = append(called, repo)
		results := make([]BackendUpdateResult, 0, len(group))
		for _, target := range group {
			result := BackendUpdateResult{Target: target, Status: "updated"}
			if repo == "/repo/main" {
				result.Status = "failed"
				result.Err = errors.New("compiler failed")
			}
			results = append(results, result)
		}
		return results
	}
	results := updateBackendBuildTargetsWith(targets, 3, runner)
	if len(called) != 2 || called[0] != "/repo/main" || called[1] != "/repo/ik" {
		t.Fatalf("repo groups called = %#v, want main then ik", called)
	}
	if len(results) != 3 || results[0].Err == nil || results[1].Err == nil || results[2].Err != nil {
		t.Fatalf("independent results = %#v", results)
	}
}

func TestUpdateRepoCandidatesIncludeAppHomeSource(t *testing.T) {
	appHome := filepath.Join(t.TempDir(), "ggrun")
	t.Setenv("LLM_APP_HOME", appHome)
	t.Setenv("HOME", filepath.Join(t.TempDir(), "home"))
	rows := updateRepoCandidates()
	want := repoCandidate{Label: "ggrun", Dir: filepath.Join(appHome, ".src", "ggrun")}
	for _, row := range rows {
		if row == want {
			return
		}
	}
	t.Fatalf("missing app-home repo candidate %#v in %#v", want, rows)
}

func TestInstalledSourceRepoDirPrefersAppHomeCheckout(t *testing.T) {
	appHome := filepath.Join(t.TempDir(), "ggrun")
	repoDir := filepath.Join(appHome, ".src", "ggrun")
	if err := os.MkdirAll(filepath.Join(repoDir, ".git"), 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLM_APP_HOME", appHome)
	t.Setenv("HOME", filepath.Join(t.TempDir(), "home"))
	if got := installedSourceRepoDir(); got != repoDir {
		t.Fatalf("source repo mismatch: got %s want %s", got, repoDir)
	}
}

func TestInstalledSourceRepoDirEnvOverride(t *testing.T) {
	want := filepath.Join(t.TempDir(), "repo")
	t.Setenv("LLM_SERVER_REPO", want)
	t.Setenv("LLM_APP_HOME", filepath.Join(t.TempDir(), "app"))
	if got := installedSourceRepoDir(); got != want {
		t.Fatalf("source repo override mismatch: got %s want %s", got, want)
	}
}

// TestInstalledSourceRepoDirFindsNonDefaultLayout covers the ~/ggrun-project/ggrun
// layout (any source checkout not at $HOME/ggrun and unrelated to the binary's
// own directory): the app-home resolver backends use must surface it even when
// neither LLM_SERVER_REPO nor LLM_APP_HOME is set.
func TestInstalledSourceRepoDirFindsNonDefaultLayout(t *testing.T) {
	home := t.TempDir()
	repoDir := filepath.Join(home, "ggrun-project", "ggrun")
	// A git checkout …
	if err := os.MkdirAll(filepath.Join(repoDir, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	// … that is also a ggrun app home (holds state).
	if err := os.MkdirAll(filepath.Join(repoDir, ".config"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repoDir, ".config", "backends.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", home)
	t.Setenv("LLM_SERVER_REPO", "")
	t.Setenv("LLM_APP_HOME", "")
	if got := installedSourceRepoDir(); got != repoDir {
		t.Fatalf("source repo from non-default layout = %q, want %q", got, repoDir)
	}
}

func TestCMakeConfigureArgsPinsSourceAndStagingBuild(t *testing.T) {
	got := cmakeConfigureArgs("/backend/repo", "/backend/build.ggrun-update", []string{"-DGGML_CUDA=ON"})
	want := []string{"-S", "/backend/repo", "-B", "/backend/build.ggrun-update", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON", "-DGGML_CUDA=ON"}
	if len(got) != len(want) {
		t.Fatalf("configure args = %#v, want %#v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("configure arg %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestCollectCMakeFlagsPreservesAccelerator(t *testing.T) {
	buildDir := t.TempDir()
	cache := "GGML_CUDA:BOOL=ON\nGGML_CUDA_NCCL:BOOL=ON\nGGML_VULKAN:BOOL=ON\nGGML_METAL:BOOL=ON\n"
	if err := os.WriteFile(filepath.Join(buildDir, "CMakeCache.txt"), []byte(cache), 0644); err != nil {
		t.Fatal(err)
	}
	got := collectCMakeFlags(buildDir)
	for _, want := range []string{"-DGGML_CUDA=ON", "-DGGML_CUDA_NCCL=ON", "-DGGML_VULKAN=ON", "-DGGML_METAL=ON"} {
		found := false
		for _, flag := range got {
			if flag == want {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing %q in %#v", want, got)
		}
	}
}

func TestPromoteBackendBuildReplacesWholeDirectory(t *testing.T) {
	root := t.TempDir()
	buildDir := filepath.Join(root, "build")
	stagingDir := buildDir + ".ggrun-update"
	if err := os.MkdirAll(filepath.Join(buildDir, "bin"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(buildDir, "bin", "llama-server"), []byte("old"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(stagingDir, "bin"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stagingDir, "bin", "llama-server"), []byte("new"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := promoteBackendBuild(buildDir, stagingDir, nil); err != nil {
		t.Fatalf("promote: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(buildDir, "bin", "llama-server"))
	if err != nil || string(data) != "new" {
		t.Fatalf("active binary = %q, err=%v", data, err)
	}
	if _, err := os.Stat(buildDir + ".ggrun-backup"); !os.IsNotExist(err) {
		t.Fatalf("backup was not cleaned: %v", err)
	}
}

func TestPromoteBackendBuildRecoversInterruptedBackup(t *testing.T) {
	root := t.TempDir()
	buildDir := filepath.Join(root, "build")
	backupDir := buildDir + ".ggrun-backup"
	stagingDir := buildDir + ".ggrun-update"
	for _, dir := range []string{backupDir, stagingDir} {
		if err := os.MkdirAll(filepath.Join(dir, "bin"), 0755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(backupDir, "bin", "llama-server"), []byte("old"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stagingDir, "bin", "llama-server"), []byte("new"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := promoteBackendBuild(buildDir, stagingDir, nil); err != nil {
		t.Fatalf("recover and promote: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(buildDir, "bin", "llama-server"))
	if err != nil || string(data) != "new" {
		t.Fatalf("active binary = %q, err=%v", data, err)
	}
}

func TestPromoteBackendBuildRollsBackFailedValidation(t *testing.T) {
	root := t.TempDir()
	buildDir := filepath.Join(root, "build")
	stagingDir := buildDir + ".ggrun-update"
	for _, dir := range []string{buildDir, stagingDir} {
		if err := os.MkdirAll(filepath.Join(dir, "bin"), 0755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(buildDir, "bin", "llama-server"), []byte("old"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stagingDir, "bin", "llama-server"), []byte("new"), 0755); err != nil {
		t.Fatal(err)
	}
	wantErr := errors.New("invalid backend")
	if err := promoteBackendBuild(buildDir, stagingDir, func(string) error { return wantErr }); !errors.Is(err, wantErr) {
		t.Fatalf("promote error = %v, want %v", err, wantErr)
	}
	data, err := os.ReadFile(filepath.Join(buildDir, "bin", "llama-server"))
	if err != nil || string(data) != "old" {
		t.Fatalf("rollback binary = %q, err=%v", data, err)
	}
}

func TestUpdateRepoCandidatesIncludeAppHomeItself(t *testing.T) {
	appHome := t.TempDir()
	if err := os.Mkdir(filepath.Join(appHome, ".git"), 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLM_APP_HOME", appHome)
	t.Setenv("LLM_SERVER_REPO", "")
	for _, row := range updateRepoCandidates() {
		if row.Label == "ggrun" && row.Dir == appHome {
			return
		}
	}
	t.Fatal("startup update check omitted the source checkout used by SelfUpdate")
}

func TestSelfUpdateRebuildsWhenSourceAlreadyCurrent(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix source updater")
	}
	repo, remote, appHome, fakeBin := t.TempDir(), t.TempDir(), t.TempDir(), t.TempDir()
	run := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", args...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v %s", args, err, out)
		}
	}
	write := func(path, content string, mode os.FileMode) {
		t.Helper()
		if err := os.WriteFile(path, []byte(content), mode); err != nil {
			t.Fatal(err)
		}
	}
	run("init", "--bare", "-q", remote)
	run("init", "-q", repo)
	write(filepath.Join(repo, "go.mod"), "module fixture\n", 0644)
	run("-C", repo, "add", "go.mod")
	run("-C", repo, "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "initial")
	run("-C", repo, "tag", "v3.2.9")
	run("-C", repo, "remote", "add", "origin", remote)
	run("-C", repo, "push", "-qu", "origin", "HEAD")
	if err := os.Mkdir(filepath.Join(appHome, ".bin"), 0755); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(appHome, ".bin", "ggrun")
	write(binary, "#!/bin/sh\necho stale\n", 0755)
	// Exercise the update transaction without compiling or using the network.
	write(filepath.Join(fakeBin, "go"), `#!/bin/sh
while [ "$#" -gt 0 ]; do
 if [ "$1" = "-o" ]; then shift; output="$1"; fi
 shift
done
printf '#!/bin/sh\necho rebuilt\n' > "$output"
chmod +x "$output"
`, 0755)
	t.Setenv("LLM_SERVER_REPO", repo)
	t.Setenv("LLM_APP_HOME", appHome)
	t.Setenv("PATH", fakeBin+string(os.PathListSeparator)+os.Getenv("PATH"))
	if err := SelfUpdate(); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(binary, "--version").CombinedOutput()
	if err != nil || strings.TrimSpace(string(out)) != "rebuilt" {
		t.Fatalf("stale binary survived update: %q %v", out, err)
	}
	if _, err := os.Stat(binary + ".bak"); !os.IsNotExist(err) {
		t.Fatalf("backup not cleaned: %v", err)
	}
	// Retrying a failed build on an unchanged checkout must preserve both the
	// working installation and the checked-out branch (no detached HEAD).
	branch, err := exec.Command("git", "-C", repo, "symbolic-ref", "HEAD").Output()
	if err != nil {
		t.Fatal(err)
	}
	write(filepath.Join(fakeBin, "go"), "#!/bin/sh\nexit 1\n", 0755)
	if err := SelfUpdate(); err == nil {
		t.Fatal("failed build reported success")
	}
	after, err := exec.Command("git", "-C", repo, "symbolic-ref", "HEAD").Output()
	if err != nil || string(after) != string(branch) {
		t.Fatalf("failed retry changed branch: %q %v", after, err)
	}
	out, err = exec.Command(binary, "--version").CombinedOutput()
	if err != nil || strings.TrimSpace(string(out)) != "rebuilt" {
		t.Fatalf("failed retry damaged binary: %q %v", out, err)
	}
}
