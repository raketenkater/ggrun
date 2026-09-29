package update

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fakeBackendBuild wires a git source checkout, an active CUDA build tree and a
// fake cmake that records every build target, so the staged update transaction
// runs for real without a compiler.
type fakeBackendBuild struct {
	repo, buildDir, buildLog string
}

func newFakeBackendBuild(t *testing.T, definesFit, fitBuildFails bool) fakeBackendBuild {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("Unix build fixture")
	}
	root := t.TempDir()
	remote, repo, tools, fakeBin := filepath.Join(root, "remote"), filepath.Join(root, "llama.cpp"), filepath.Join(root, "tools"), filepath.Join(root, "fakebin")
	for _, dir := range []string{tools, fakeBin} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	git := func(args ...string) {
		t.Helper()
		if out, err := exec.Command("git", args...).CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v %s", args, err, out)
		}
	}
	git("init", "--bare", "-q", remote)
	git("init", "-q", "-b", "master", repo)
	if err := os.WriteFile(filepath.Join(repo, "CMakeLists.txt"), []byte("project(fake)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("-C", repo, "add", ".")
	git("-C", repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "src")
	git("-C", repo, "remote", "add", "origin", remote)
	git("-C", repo, "push", "-qu", "origin", "master")

	write := func(path, body string, mode os.FileMode) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), mode); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(tools, "llama-server"), "#!/bin/sh\ncase \"$1\" in\n"+
		"  --version) echo 'version: 1 (abc)' ;;\n"+
		"  --help) echo '--model --ctx-size --host --port' ;;\n"+
		"  --list-devices) echo 'Available devices:'; echo 'CUDA0: fake' ;;\n"+
		"  *) exit 1 ;;\nesac\n", 0o755)
	write(filepath.Join(tools, "llama-fit-params"), "#!/bin/sh\necho '-fitp, --fit-print [on|off]'\n", 0o755)
	write(filepath.Join(tools, "CMakeCache.txt"), "GGML_CUDA:BOOL=ON\n", 0o644)

	build := fakeBackendBuild{repo: repo, buildDir: filepath.Join(repo, "build-cuda"), buildLog: filepath.Join(root, "builds.log")}
	defines, fails := "", ""
	if definesFit {
		defines = "1"
	}
	if fitBuildFails {
		fails = "1"
	}
	write(filepath.Join(fakeBin, "cmake"), `#!/bin/sh
if [ "$1" = "-S" ]; then
  b="$4"; mkdir -p "$b/bin"; cp "`+tools+`/CMakeCache.txt" "$b/CMakeCache.txt"
  if [ -n "`+defines+`" ]; then mkdir -p "$b/tools/fit-params/CMakeFiles/llama-fit-params.dir"; fi
  exit 0
fi
dir="$2"; for last; do :; done
echo "$last" >> "`+build.buildLog+`"
if [ "$last" = llama-fit-params ] && [ -n "`+fails+`" ]; then echo 'fit-params.cpp: error' >&2; exit 2; fi
cp "`+tools+`/$last" "$dir/bin/$last"
`, 0o755)
	t.Setenv("PATH", fakeBin+string(os.PathListSeparator)+os.Getenv("PATH"))

	// The active build serves but predates the oracle: exactly the installed
	// state the MiMo launch found (server present, fit tool missing).
	write(filepath.Join(build.buildDir, "CMakeCache.txt"), "GGML_CUDA:BOOL=ON\n", 0o644)
	write(filepath.Join(build.buildDir, "bin", "llama-server"), mustRead(t, filepath.Join(tools, "llama-server")), 0o755)
	if definesFit {
		if err := os.MkdirAll(filepath.Join(build.buildDir, "tools", "fit-params", "CMakeFiles", "llama-fit-params.dir"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	return build
}

func mustRead(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func (b fakeBackendBuild) update(t *testing.T) (string, []string) {
	t.Helper()
	_ = os.Remove(b.buildLog)
	results := updateBackendBuildGroup(b.repo, []BackendBuildTarget{{Label: "llama.cpp (cuda)", RepoDir: b.repo, BuildDir: b.buildDir}}, 1)
	if len(results) != 1 {
		t.Fatalf("results = %#v", results)
	}
	if results[0].Err != nil {
		t.Fatalf("update failed: %v", results[0].Err)
	}
	data, _ := os.ReadFile(b.buildLog)
	return results[0].Status, strings.Fields(string(data))
}

func TestUnchangedSourceRebuildsABuildThatLostItsFitTool(t *testing.T) {
	b := newFakeBackendBuild(t, true, false)
	head, _ := gitRevParse(b.repo, "HEAD")

	status, builds := b.update(t)
	if !strings.HasPrefix(status, "updated") {
		t.Fatalf("incomplete build reported %q; it must be rebuilt", status)
	}
	if strings.Join(builds, ",") != "llama-server,llama-fit-params" {
		t.Fatalf("staged build targets = %v", builds)
	}
	if after, _ := gitRevParse(b.repo, "HEAD"); after != head {
		t.Fatalf("repairing the tool changed the backend revision: %s -> %s", head, after)
	}
	if err := activeBuildComplete(b.buildDir); err != nil {
		t.Fatalf("promoted build still incomplete: %v", err)
	}

	status, builds = b.update(t)
	if status != "current" || len(builds) != 0 {
		t.Fatalf("complete build rebuilt again: status=%q builds=%v", status, builds)
	}
}

func TestFitToolBuildFailureKeepsServerAndDoesNotLoop(t *testing.T) {
	b := newFakeBackendBuild(t, true, true)

	status, builds := b.update(t)
	if !strings.HasPrefix(status, "updated") || strings.Join(builds, ",") != "llama-server,llama-fit-params" {
		t.Fatalf("status=%q builds=%v", status, builds)
	}
	if _, err := os.Stat(filepath.Join(b.buildDir, "bin", "llama-fit-params")); !os.IsNotExist(err) {
		t.Fatalf("failed optional tool left in the active build: %v", err)
	}
	if err := smokeBackendConfigured(filepath.Join(b.buildDir, "bin", "llama-server"), collectCMakeFlags(b.buildDir)); err != nil {
		t.Fatalf("optional tool failure cost the server: %v", err)
	}

	// Same source, same failure: a second update must not rebuild forever.
	status, builds = b.update(t)
	if status != "current" || len(builds) != 0 {
		t.Fatalf("known-unavailable tool triggered another rebuild: status=%q builds=%v", status, builds)
	}
}

func TestForkWithoutFitTargetIsCurrent(t *testing.T) {
	b := newFakeBackendBuild(t, false, false)
	status, builds := b.update(t)
	if status != "current" || len(builds) != 0 {
		t.Fatalf("fork lacking the target was rebuilt: status=%q builds=%v", status, builds)
	}
}
