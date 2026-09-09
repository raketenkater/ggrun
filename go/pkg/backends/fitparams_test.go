package backends

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// An in-place backend rebuild must refresh its oracle too. File existence
// cannot establish that both tools contain the same allocation code.
func TestBuildFitParamsRefreshesExistingOracle(t *testing.T) {
	buildDir := fakeFitParamsBuild(t, `printf '%s' current > "$2/bin/llama-fit-params"`)
	path := FitParamsPath(buildDir)
	if err := os.WriteFile(path, []byte("stale"), 0755); err != nil {
		t.Fatal(err)
	}
	got, err := BuildFitParamsOracle("", buildDir, 2)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(got)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "current" {
		t.Fatalf("oracle still uses old backend code: %q", data)
	}
}

// A backend can continue serving without an optional oracle, but a failed
// rebuild must not leave an old oracle available to predict its allocations.
func TestBuildFitParamsFailureInvalidatesOldOracle(t *testing.T) {
	buildDir := fakeFitParamsBuild(t, "exit 1")
	path := FitParamsPath(buildDir)
	if err := os.WriteFile(path, []byte("stale"), 0755); err != nil {
		t.Fatal(err)
	}
	if _, err := BuildFitParamsOracle("", buildDir, 1); err == nil {
		t.Fatal("expected build failure")
	}
	if HasFitParams(buildDir) {
		t.Fatal("failed build left a stale oracle usable")
	}
}

func TestBuildFitParamsRequiresBuildOutput(t *testing.T) {
	buildDir := fakeFitParamsBuild(t, "exit 0")
	if _, err := BuildFitParamsOracle("", buildDir, 1); err == nil || !strings.Contains(err.Error(), "produced no") {
		t.Fatalf("successful command without an oracle must be rejected: %v", err)
	}
}

func fakeFitParamsBuild(t *testing.T, body string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake CMake uses a POSIX shell")
	}
	root := t.TempDir()
	buildDir := filepath.Join(root, "build")
	if err := os.MkdirAll(filepath.Join(buildDir, "bin"), 0755); err != nil {
		t.Fatal(err)
	}
	commandDir := filepath.Join(root, "commands")
	if err := os.Mkdir(commandDir, 0755); err != nil {
		t.Fatal(err)
	}
	script := "#!/bin/sh\nset -eu\n" + body + "\n"
	if err := os.WriteFile(filepath.Join(commandDir, "cmake"), []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", commandDir)
	return buildDir
}
