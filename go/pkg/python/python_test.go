package python

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// A python3 that resolves on PATH but does not run (the Windows Store alias
// exits 9009) must not be chosen over a working python.
func TestCommandSkipsInterpreterThatDoesNotRun(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses shell-script stand-ins")
	}
	dir := t.TempDir()
	write := func(name, body string) string {
		t.Helper()
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte("#!/bin/sh\n"+body), 0o755); err != nil {
			t.Fatal(err)
		}
		return p
	}
	t.Setenv("PATH", dir)
	write("python3", "exit 9009\n")
	if got, ok := Command(); ok {
		t.Fatalf("Command() = %q, want none when only a broken stub exists", got)
	}
	want := write("python", "exit 0\n")
	if got, ok := Command(); !ok || got != want {
		t.Fatalf("Command() = %q %v, want %q", got, ok, want)
	}
}

func appVenvFixture(t *testing.T) (appDir, exe string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("uses shell-script stand-ins")
	}
	appDir = t.TempDir()
	bin := filepath.Join(appDir, ".ggrun-python", "bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	exe = filepath.Join(appDir, "ggrun")
	if err := os.WriteFile(exe, []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { executablePath = os.Executable })
	return appDir, exe
}

func writeFakePython(t *testing.T, path string) {
	t.Helper()
	if err := os.WriteFile(path, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
}

func TestCommandPrefersAppVenvNextToExecutable(t *testing.T) {
	appDir, exe := appVenvFixture(t)
	app := filepath.Join(appDir, ".ggrun-python", "bin", "python3")
	writeFakePython(t, app)
	pathDir := t.TempDir()
	writeFakePython(t, filepath.Join(pathDir, "python3"))
	t.Setenv("PATH", pathDir)
	executablePath = func() (string, error) { return exe, nil }
	if got, ok := Command(); !ok || got != app {
		t.Fatalf("Command() = %q, %v; want app venv %q", got, ok, app)
	}
}

func TestCommandFallsBackToPathWithoutAppVenv(t *testing.T) {
	appDir, exe := appVenvFixture(t)
	if err := os.RemoveAll(filepath.Join(appDir, ".ggrun-python")); err != nil {
		t.Fatal(err)
	}
	pathDir := t.TempDir()
	want := filepath.Join(pathDir, "python3")
	writeFakePython(t, want)
	t.Setenv("PATH", pathDir)
	executablePath = func() (string, error) { return exe, nil }
	if got, ok := Command(); !ok || got != want {
		t.Fatalf("Command() = %q, %v; want PATH python %q", got, ok, want)
	}
}

func TestCommandFindsAppVenvThroughExecutableSymlink(t *testing.T) {
	appDir, _ := appVenvFixture(t)
	app := filepath.Join(appDir, ".ggrun-python", "bin", "python3")
	writeFakePython(t, app)
	link := filepath.Join(t.TempDir(), "ggrun")
	if err := os.Symlink(filepath.Join(appDir, "ggrun"), link); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", t.TempDir())
	executablePath = func() (string, error) { return link, nil }
	if got, ok := Command(); !ok || got != app {
		t.Fatalf("Command() = %q, %v; want app venv %q", got, ok, app)
	}
}
