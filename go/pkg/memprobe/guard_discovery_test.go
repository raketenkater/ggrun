package memprobe

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func writeGuard(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("elf"), 0o644); err != nil {
		t.Fatal(err)
	}
}

// ggrun installed by `go install` runs from ~/go/bin while the source installer
// put the guard in <app-home>/.bin. Launching from an unrelated directory must
// still find it; the directory itself must never supply a preloaded library.
func TestGuardFoundInAppHomeFromUnrelatedDirectory(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("memguard is Linux-only")
	}
	appHome, unrelated := t.TempDir(), t.TempDir()
	installed := filepath.Join(appHome, ".bin", GuardLibraryName)
	writeGuard(t, installed)
	writeGuard(t, filepath.Join(unrelated, "native", "memguard", GuardLibraryName))
	wd, _ := os.Getwd()
	if err := os.Chdir(unrelated); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(wd)
	t.Setenv("GGRUN_MEMGUARD_LIBRARY", "")

	if got := FindGuardLibrary(appHome); got != installed {
		t.Fatalf("guard = %q, want installed %q", got, installed)
	}
	if got := FindGuardLibrary(""); got != "" {
		t.Fatalf("working directory supplied a preload library: %q", got)
	}
}

func TestExplicitGuardOverrideIsAuthoritative(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("memguard is Linux-only")
	}
	appHome := t.TempDir()
	writeGuard(t, filepath.Join(appHome, ".bin", GuardLibraryName))
	explicit := filepath.Join(t.TempDir(), "custom-guard.so")
	writeGuard(t, explicit)

	t.Setenv("GGRUN_MEMGUARD_LIBRARY", explicit)
	if got := FindGuardLibrary(appHome); got != explicit {
		t.Fatalf("explicit guard = %q, want %q", got, explicit)
	}
	// A configured path that is wrong disables the guard; it is not silently
	// replaced by a different library the user did not choose.
	t.Setenv("GGRUN_MEMGUARD_LIBRARY", filepath.Join(t.TempDir(), "missing.so"))
	if got := FindGuardLibrary(appHome); got != "" {
		t.Fatalf("broken override fell back to %q", got)
	}
}
