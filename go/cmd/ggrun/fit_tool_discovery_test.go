package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/backends"
)

func writeExecutable(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
}

func symlink(t *testing.T, target, link string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
}

// The app-home layout the MiMo launch ran from: .bin/llama-server is ik_llama,
// .bin/llama-server-cuda is mainline, and one .bin/llama-fit-params link points
// into mainline's build. The oracle may only price its own build's server.
func TestFitParamsOracleComesFromTheServersOwnBuild(t *testing.T) {
	root := t.TempDir()
	mainline := filepath.Join(root, ".src", "llama.cpp", "build-cuda", "bin")
	ik := filepath.Join(root, ".src", "ik_llama.cpp", "build", "bin")
	appBin := filepath.Join(root, ".bin")
	writeExecutable(t, filepath.Join(mainline, "llama-server"), "#!/bin/sh\n")
	writeExecutable(t, filepath.Join(mainline, "llama-fit-params"), "#!/bin/sh\n")
	writeExecutable(t, filepath.Join(ik, "llama-server"), "#!/bin/sh\n")
	symlink(t, filepath.Join(mainline, "llama-server"), filepath.Join(appBin, "llama-server-cuda"))
	symlink(t, filepath.Join(ik, "llama-server"), filepath.Join(appBin, "llama-server"))
	symlink(t, filepath.Join(mainline, "llama-fit-params"), filepath.Join(appBin, "llama-fit-params"))

	if got, want := findFitParamsBin(filepath.Join(appBin, "llama-server-cuda"), ""), filepath.Join(mainline, "llama-fit-params"); got != want {
		t.Fatalf("mainline server oracle = %q, want %q", got, want)
	}
	if got := findFitParamsBin(filepath.Join(appBin, "llama-server"), ""); got != "" {
		t.Fatalf("ik_llama server was paired with mainline's oracle %q", got)
	}
}

// A build that lost its oracle leaves a dangling app-home link. That must read
// as "no oracle", never as an error or a different build's tool.
func TestFitParamsDanglingLinkMeansNoOracle(t *testing.T) {
	root := t.TempDir()
	build := filepath.Join(root, "build-cuda", "bin")
	writeExecutable(t, filepath.Join(build, "llama-server"), "#!/bin/sh\n")
	symlink(t, filepath.Join(build, "llama-server"), filepath.Join(root, ".bin", "llama-server-cuda"))
	symlink(t, filepath.Join(build, "llama-fit-params"), filepath.Join(root, ".bin", "llama-fit-params"))
	if got := findFitParamsBin(filepath.Join(root, ".bin", "llama-server-cuda"), ""); got != "" {
		t.Fatalf("missing oracle resolved to %q", got)
	}
}

func TestFitParamsBareServerNameResolvesThroughPATH(t *testing.T) {
	root := t.TempDir()
	build := filepath.Join(root, "Cellar", "llama.cpp", "bin")
	pathDir := filepath.Join(root, "bin")
	writeExecutable(t, filepath.Join(build, "llama-server"), "#!/bin/sh\n")
	writeExecutable(t, filepath.Join(build, "llama-fit-params"), "#!/bin/sh\n")
	symlink(t, filepath.Join(build, "llama-server"), filepath.Join(pathDir, "llama-server"))
	t.Setenv("PATH", pathDir)
	if got, want := findFitParamsBin("llama-server", ""), filepath.Join(build, "llama-fit-params"); got != want {
		t.Fatalf("PATH server oracle = %q, want %q", got, want)
	}
	// A different install's oracle earlier on PATH is not this server's.
	other := filepath.Join(root, "other")
	writeExecutable(t, filepath.Join(other, "llama-fit-params"), "#!/bin/sh\n")
	if err := os.Remove(filepath.Join(build, "llama-fit-params")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", other+string(os.PathListSeparator)+pathDir)
	if got := findFitParamsBin("llama-server", ""); got != "" {
		t.Fatalf("PATH server paired with another install's oracle %q", got)
	}
}

func TestOptionalBackendToolFailureIsRecordedNotFatal(t *testing.T) {
	buildDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(buildDir, "tools", "fit-params", "CMakeFiles", "llama-fit-params.dir"), 0o755); err != nil {
		t.Fatal(err)
	}
	tool := filepath.Join(buildDir, "bin", "llama-fit-params")

	// Builds, but cannot start (e.g. a missing shared library): removed.
	buildOptionalBackendTools(buildDir, func(target string) error {
		writeExecutable(t, tool, "#!/bin/sh\nexit 127\n")
		return nil
	})
	if _, err := os.Stat(tool); !os.IsNotExist(err) {
		t.Fatalf("broken oracle left selectable: %v", err)
	}
	if missing := backends.MissingBuildTools(buildDir); len(missing) != 0 {
		t.Fatalf("recorded failure reported as a repairable gap: %v", missing)
	}

	// A fresh build tree retries and keeps a working oracle.
	fresh := t.TempDir()
	if err := os.MkdirAll(filepath.Join(fresh, "tools", "fit-params", "CMakeFiles", "llama-fit-params.dir"), 0o755); err != nil {
		t.Fatal(err)
	}
	buildOptionalBackendTools(fresh, func(target string) error {
		writeExecutable(t, filepath.Join(fresh, "bin", target), "#!/bin/sh\necho '--fit-print'\n")
		return nil
	})
	if missing := backends.MissingBuildTools(fresh); len(missing) != 0 {
		t.Fatalf("working oracle reported missing: %v", missing)
	}
}
