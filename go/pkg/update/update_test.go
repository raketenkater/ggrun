package update

import (
	"errors"
	"os"
	"path/filepath"
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
