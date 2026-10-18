package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/backends"
)

func TestBackendLayoutFor(t *testing.T) {
	ok, err := backendLayoutFor(backends.Backend{
		Tag:  "laguna",
		Path: "/src/fork-llama.cpp-laguna/build-cuda/bin/llama-server",
	})
	if err != nil {
		t.Fatalf("recognised layout rejected: %v", err)
	}
	if ok.srcDir != "/src/fork-llama.cpp-laguna" || ok.buildDir != "/src/fork-llama.cpp-laguna/build-cuda" || ok.accel != "cuda" {
		t.Errorf("layout = %+v", ok)
	}

	// A hand-registered binary has no build tree, and updating it would rebuild
	// something ggrun never built.
	for _, bad := range []string{
		"/usr/local/bin/llama-server",
		"/src/fork/build-cuda/llama-server",
		"/src/fork/out/bin/llama-server",
		"",
	} {
		if _, err := backendLayoutFor(backends.Backend{Tag: "x", Path: bad}); err == nil {
			t.Errorf("path %q accepted as a ggrun build layout", bad)
		}
	}
}

func TestEnsureRecipeOriginMigratesLegacyFork(t *testing.T) {
	repo := t.TempDir()
	if _, err := gitOutput(repo, "init"); err != nil {
		t.Fatal(err)
	}
	if _, err := gitOutput(repo, "remote", "add", "origin", "https://github.com/legacy/fork.git"); err != nil {
		t.Fatal(err)
	}
	recipe := &backends.Recipe{GitURL: "https://github.com/ggml-org/llama.cpp.git"}
	if err := ensureRecipeOrigin(repo, recipe); err != nil {
		t.Fatal(err)
	}
	got, err := gitOutput(repo, "remote", "get-url", "origin")
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(got) != recipe.GitURL {
		t.Fatalf("origin=%q, want %q", strings.TrimSpace(got), recipe.GitURL)
	}
}

func TestArchiveBuildDirPreservesAndDoesNotCollide(t *testing.T) {
	root := t.TempDir()
	build := filepath.Join(root, "build-cuda")
	if err := os.MkdirAll(filepath.Join(build, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	marker := filepath.Join(build, "bin", "llama-server")
	if err := os.WriteFile(marker, []byte("first"), 0o755); err != nil {
		t.Fatal(err)
	}

	first, err := archiveBuildDir(build, "04b2b72cb54048ead292884adbe11f284e3ec950")
	if err != nil {
		t.Fatalf("archive: %v", err)
	}
	if !strings.HasSuffix(first, "build-cuda-prev-04b2b72cb540") {
		t.Errorf("archive name = %q", first)
	}
	// The point of archiving is that the old binary survives intact.
	if got, _ := os.ReadFile(filepath.Join(first, "bin", "llama-server")); string(got) != "first" {
		t.Errorf("archived binary content = %q, want %q", got, "first")
	}
	if _, err := os.Stat(build); !os.IsNotExist(err) {
		t.Error("build dir should have been moved aside, not copied")
	}

	// A second archive at the same commit must not overwrite the first: that
	// would destroy the only build known to work.
	if err := os.MkdirAll(filepath.Join(build, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(marker, []byte("second"), 0o755); err != nil {
		t.Fatal(err)
	}
	second, err := archiveBuildDir(build, "04b2b72cb54048ead292884adbe11f284e3ec950")
	if err != nil {
		t.Fatalf("second archive: %v", err)
	}
	if second == first {
		t.Fatal("second archive reused the first path")
	}
	if got, _ := os.ReadFile(filepath.Join(first, "bin", "llama-server")); string(got) != "first" {
		t.Error("first archive was clobbered by the second")
	}
}

func TestArchiveBuildDirUnknownCommit(t *testing.T) {
	root := t.TempDir()
	build := filepath.Join(root, "build-cuda")
	if err := os.MkdirAll(build, 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := archiveBuildDir(build, "")
	if err != nil {
		t.Fatalf("archive: %v", err)
	}
	if !strings.HasSuffix(got, "build-cuda-prev-unknown") {
		t.Errorf("archive name = %q", got)
	}
}

func TestPruneBuildDirRefusesNonArchives(t *testing.T) {
	root := t.TempDir()
	// Guard against deleting a live build tree, or anything else, by name.
	for _, name := range []string{"build-cuda", "src", ""} {
		p := filepath.Join(root, name)
		if name != "" {
			if err := os.MkdirAll(p, 0o755); err != nil {
				t.Fatal(err)
			}
		} else {
			p = ""
		}
		if err := pruneBuildDir(p); err == nil {
			t.Errorf("pruneBuildDir(%q) was allowed", p)
		}
		if name != "" {
			if _, err := os.Stat(p); err != nil {
				t.Errorf("%q was removed despite the refusal", name)
			}
		}
	}

	archive := filepath.Join(root, "build-cuda-prev-abc123")
	if err := os.MkdirAll(archive, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := pruneBuildDir(archive); err != nil {
		t.Fatalf("pruning a real archive failed: %v", err)
	}
	if _, err := os.Stat(archive); !os.IsNotExist(err) {
		t.Error("archive still present after prune")
	}
}

func TestPruneCandidateBuildDirIsScopedBesideActive(t *testing.T) {
	root := t.TempDir()
	active := filepath.Join(root, "build-cuda")
	candidate := active + "-candidate-abc-1"
	if err := os.MkdirAll(candidate, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := pruneCandidateBuildDir(candidate, active); err != nil {
		t.Fatalf("valid candidate cleanup: %v", err)
	}
	if _, err := os.Stat(candidate); !os.IsNotExist(err) {
		t.Fatal("candidate was not removed")
	}
	if err := os.MkdirAll(active, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := pruneCandidateBuildDir(active, active); err == nil {
		t.Fatal("active build was accepted as candidate cleanup target")
	}
	if _, err := os.Stat(active); err != nil {
		t.Fatal("active build was removed")
	}
}

func TestValidateBackendCandidateChecksServerSurfaceAndAccelerator(t *testing.T) {
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
	good := write("good-server", "--model --ctx-size --host --port", "CUDA0: test")
	if err := validateBackendCandidate(good, "", "cuda"); err != nil {
		t.Fatalf("valid staged backend rejected: %v", err)
	}
	missing := write("missing-server", "--model --host --port", "CUDA0: test")
	if err := validateBackendCandidate(missing, "", "cpu"); err == nil || !strings.Contains(err.Error(), "--ctx-size") {
		t.Fatalf("missing launch option accepted: %v", err)
	}
	cpuOnly := write("cpu-server", "--model --ctx-size --host --port", "CPU")
	if err := validateBackendCandidate(cpuOnly, "", "cuda"); err == nil || !strings.Contains(err.Error(), "no supported GPU") {
		t.Fatalf("CPU-only binary accepted for CUDA activation: %v", err)
	}
}

func TestValidateBackendCandidateHelpUsageExitDoesNotBypassConformance(t *testing.T) {
	const surface = "--model --ctx-size --host --port"
	for _, tc := range []struct {
		name, help, devices, arch string
		helpExit, versionExit     int
		wantError                 string
	}{
		{name: "ik_usage_exit", help: "usage: llama-server [options]\n" + surface, helpExit: 1, devices: "CUDA0: test"},
		{name: "ordinary_success", help: surface, devices: "CUDA0: test"},
		{name: "no_usage_header", help: surface, helpExit: 1, devices: "CUDA0: test", wantError: "--help failed"},
		{name: "missing_context", help: "usage: llama-server [options]\n--model --host --port", helpExit: 1, devices: "CUDA0: test", wantError: "--ctx-size"},
		{name: "usage_error_status", help: "usage: llama-server [options]\n" + surface, helpExit: 2, devices: "CUDA0: test", wantError: "--help failed"},
		{name: "loader_error_status", help: "usage: llama-server [options]\n" + surface, helpExit: 127, devices: "CUDA0: test", wantError: "--help failed"},
		{name: "crash_status", help: "usage: llama-server [options]\n" + surface, helpExit: 139, devices: "CUDA0: test", wantError: "--help failed"},
		{name: "failed_version", help: "usage: llama-server [options]\n" + surface, helpExit: 1, versionExit: 1, devices: "CUDA0: test", wantError: "--version failed"},
		{name: "cpu_only", help: "usage: llama-server [options]\n" + surface, helpExit: 1, devices: "CPU", wantError: "no supported GPU"},
		{name: "unverified_arch", help: "usage: llama-server [options]\n" + surface, helpExit: 1, devices: "CUDA0: test", arch: "certainly-not-a-real-architecture", wantError: "architecture"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "llama-server")
			body := fmt.Sprintf("#!/bin/sh\ncase \"$1\" in\n--version) echo 'llama-server test'; exit %d ;;\n--help) echo '%s'; exit %d ;;\n--list-devices) echo 'Available devices:'; echo '%s' ;;\n*) exit 1 ;;\nesac\n", tc.versionExit, tc.help, tc.helpExit, tc.devices)
			if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
				t.Fatal(err)
			}
			err := validateBackendCandidate(path, tc.arch, "cuda")
			if tc.wantError == "" {
				if err != nil {
					t.Fatalf("valid backend rejected: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.wantError) {
				t.Fatalf("got %v; want rejection containing %q", err, tc.wantError)
			}
		})
	}
}

func TestRollbackKeepsCanonicalLayoutForNextUpdate(t *testing.T) {
	appHome := t.TempDir()
	t.Setenv("LLM_APP_HOME", appHome)
	src := filepath.Join(appHome, ".src", "fork-test")
	activeDir := filepath.Join(src, "build-cpu")
	previousDir := activeDir + "-prev-old"
	writeServer := func(dir, version string) string {
		if err := os.MkdirAll(filepath.Join(dir, "bin"), 0o755); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(dir, "bin", "llama-server")
		body := "#!/bin/sh\ncase \"$1\" in\n" +
			"  --version) echo '" + version + "' ;;\n" +
			"  --help) echo '--model --ctx-size --host --port' ;;\n" +
			"  *) exit 1 ;;\nesac\n"
		if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
			t.Fatal(err)
		}
		return path
	}
	active := writeServer(activeDir, "current")
	previous := writeServer(previousDir, "previous")
	if err := backends.Save([]backends.Backend{{
		Tag: "test", Path: active, Commit: "new",
		Previous: &backends.BackendVersion{Path: previous, Commit: "old"},
	}}); err != nil {
		t.Fatal(err)
	}

	restored, err := rollbackRegisteredBackend("test")
	if err != nil {
		t.Fatal(err)
	}
	if restored.Path != active {
		t.Fatalf("rollback pointed manifest into archive: %s", restored.Path)
	}
	if layout, err := backendLayoutFor(restored); err != nil || layout.accel != "cpu" || layout.buildDir != activeDir {
		t.Fatalf("rolled-back backend is not updateable: layout=%+v err=%v", layout, err)
	}
	data, _ := os.ReadFile(active)
	if !strings.Contains(string(data), "previous") {
		t.Fatalf("canonical tree did not receive previous build: %s", data)
	}

	// Rollback is symmetric and must remain canonical when used as undo.
	undone, err := rollbackRegisteredBackend("test")
	if err != nil {
		t.Fatal(err)
	}
	if undone.Path != active {
		t.Fatalf("rollback undo left canonical path: %s", undone.Path)
	}
}

func TestShortCommit(t *testing.T) {
	if got := shortCommit("04b2b72cb54048ead292884adbe11f284e3ec950"); got != "04b2b72cb540" {
		t.Errorf("shortCommit = %q", got)
	}
	if got := shortCommit(""); got != "(unknown)" {
		t.Errorf("empty commit = %q", got)
	}
	if got := shortCommit("abc123"); got != "abc123" {
		t.Errorf("short input = %q", got)
	}
}
