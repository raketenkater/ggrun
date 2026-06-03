package update

import (
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
	want := "https://raw.githubusercontent.com/raketenkater/llm-server/v3.0.1/install.sh"
	if got := rawInstallerURL("v3.0.1"); got != want {
		t.Fatalf("installer URL mismatch: %s", got)
	}
	want = "https://raw.githubusercontent.com/raketenkater/llm-server/main/install.sh"
	if got := rawInstallerURL(""); got != want {
		t.Fatalf("default installer URL mismatch: %s", got)
	}
}

func TestHasUpdateLabel(t *testing.T) {
	if !hasUpdateLabel([]string{"llm-server v3.0.1"}, "llm-server") {
		t.Fatal("expected prefixed llm-server release label to match")
	}
	if hasUpdateLabel([]string{"llama.cpp"}, "llm-server") {
		t.Fatal("unexpected llm-server match")
	}
}
