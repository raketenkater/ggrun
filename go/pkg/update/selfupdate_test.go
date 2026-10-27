package update

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func writeScript(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+body), 0o755); err != nil {
		t.Fatal(err)
	}
}

func TestGoVersionAtLeast(t *testing.T) {
	for _, c := range []struct {
		have, need string
		want       bool
	}{
		{"go1.24.13", "1.24.13", true},
		{"go1.25.0", "1.24.13", true},
		{"go1.24.2", "1.24.13", false},
		{"go1.24", "1.24.0", true},
		{"go1.26.1 X:nocoverageredesign", "1.24.13", true},
		{"", "1.24.13", false},
		{"go1.22.0", "", true},
	} {
		if got := goVersionAtLeast(c.have, c.need); got != c.want {
			t.Errorf("goVersionAtLeast(%q, %q) = %v, want %v", c.have, c.need, got, c.want)
		}
	}
}

// Most installs have no system Go: install.sh downloads one into
// <backend root>/.llm-server-go, and a self-update must build with it.
func TestGoToolchainUsesInstallerDownloadedGo(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix source updater")
	}
	appHome, empty := t.TempDir(), t.TempDir()
	repo := filepath.Join(appHome, ".src", "ggrun")
	goDir := filepath.Join(repo, "go")
	if err := os.MkdirAll(goDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(goDir, "go.mod"), []byte("module x\n\ngo 1.24.13\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(appHome, ".src", ".llm-server-go")
	old := filepath.Join(root, "go1.22.0.linux-amd64", "bin", "go")
	want := filepath.Join(root, "go1.24.13.linux-amd64", "bin", "go")
	writeScript(t, old, "echo go1.22.0\n")
	writeScript(t, want, "echo go1.24.13\n")
	t.Setenv("PATH", empty)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("LLM_APP_HOME", appHome)
	t.Setenv("LLM_INSTALL_GO_ROOT", "")
	if got := goToolchain(goDir, repo); got != want {
		t.Fatalf("goToolchain = %q, want installer Go %q", got, want)
	}

	// A system Go that is new enough wins.
	system := filepath.Join(empty, "go")
	writeScript(t, system, "echo go1.25.0\n")
	if got := goToolchain(goDir, repo); got != system {
		t.Fatalf("goToolchain = %q, want system Go %q", got, system)
	}
	// One that is too old loses to the downloaded toolchain.
	writeScript(t, system, "echo go1.21.0\n")
	if got := goToolchain(goDir, repo); got != want {
		t.Fatalf("goToolchain = %q, want installer Go over old system Go", got)
	}
}

// A failed build after a pull rolls the source back on its branch. Checking
// out the old hash detached HEAD, and every later update then failed to pull.
func TestSelfUpdateRollbackKeepsBranchAttached(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix source updater")
	}
	remote, repo, other, appHome, fakeBin := t.TempDir(), t.TempDir(), t.TempDir(), t.TempDir(), t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		out, err := exec.Command("git", append([]string{"-c", "user.name=t", "-c", "user.email=t@example.invalid"}, args...)...).CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "--bare", "-q", remote)
	git("init", "-q", repo)
	if err := os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module fixture\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("-C", repo, "add", "go.mod")
	git("-C", repo, "commit", "-qm", "one")
	git("-C", repo, "remote", "add", "origin", remote)
	git("-C", repo, "push", "-qu", "origin", "HEAD")
	oldHead := git("-C", repo, "rev-parse", "HEAD")
	branch := git("-C", repo, "symbolic-ref", "HEAD")
	git("clone", "-q", remote, other)
	if err := os.WriteFile(filepath.Join(other, "two"), []byte("two"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("-C", other, "add", "two")
	git("-C", other, "commit", "-qm", "two")
	git("-C", other, "push", "-q")
	newHead := git("-C", other, "rev-parse", "HEAD")

	binary := filepath.Join(appHome, ".bin", "ggrun")
	writeScript(t, binary, "echo stale\n")
	writeScript(t, filepath.Join(fakeBin, "go"), "[ \"$1\" = env ] && echo go1.99.0\nexit 1\n")
	t.Setenv("LLM_SERVER_REPO", repo)
	t.Setenv("LLM_APP_HOME", appHome)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("PATH", fakeBin+string(os.PathListSeparator)+os.Getenv("PATH"))
	if err := SelfUpdate(); err == nil {
		t.Fatal("failed build reported success")
	}
	if got := git("-C", repo, "symbolic-ref", "HEAD"); got != branch {
		t.Fatalf("rollback left HEAD on %q, want branch %q", got, branch)
	}
	if got := git("-C", repo, "rev-parse", "HEAD"); got != oldHead {
		t.Fatalf("rollback HEAD = %s, want %s", got, oldHead)
	}
	if out, _ := exec.Command(binary).Output(); strings.TrimSpace(string(out)) != "stale" {
		t.Fatalf("rollback damaged the installed binary: %q", out)
	}

	// The next update pulls again and succeeds.
	writeScript(t, filepath.Join(fakeBin, "go"), `[ "$1" = env ] && { echo go1.99.0; exit 0; }
while [ "$#" -gt 0 ]; do
 if [ "$1" = "-o" ]; then shift; output="$1"; fi
 shift
done
printf '#!/bin/sh\necho rebuilt\n' > "$output"
chmod +x "$output"
`)
	if err := SelfUpdate(); err != nil {
		t.Fatalf("update after rollback: %v", err)
	}
	if got := git("-C", repo, "rev-parse", "HEAD"); got != newHead {
		t.Fatalf("update after rollback HEAD = %s, want %s", got, newHead)
	}
}

// Windows cannot overwrite a running ggrun.exe; the update moves it aside and
// restores it when the installer fails.
func TestRunInstallerOverRunningBinary(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses shell-script stand-ins for ggrun.exe")
	}
	read := func(path string) string {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	dir := t.TempDir()
	exe := filepath.Join(dir, "ggrun")
	oldBody := "#!/bin/sh\necho old\n"

	t.Run("success leaves the old binary for the next start", func(t *testing.T) {
		writeScript(t, exe, "echo old\n")
		writeScript(t, exe+".old", "echo stale\n")
		install := func() error {
			if _, err := os.Stat(exe); !os.IsNotExist(err) {
				t.Fatalf("installer ran while the running binary was still in place: %v", err)
			}
			writeScript(t, exe, "echo new\n")
			return nil
		}
		if err := runInstallerOverRunningBinary(exe, install); err != nil {
			t.Fatal(err)
		}
		if got := read(exe); !strings.Contains(got, "new") {
			t.Fatalf("binary after update = %q", got)
		}
		if got := read(exe + ".old"); got != oldBody {
			t.Fatalf("moved-aside binary = %q, want the running one", got)
		}
	})

	for name, install := range map[string]func() error{
		"installer fails after writing": func() error {
			writeScript(t, exe, "echo partial\n")
			return errors.New("copy failed")
		},
		"installer writes nothing": func() error { return nil },
		"new binary fails self-check": func() error {
			writeScript(t, exe, "exit 3\n")
			return nil
		},
	} {
		t.Run(name, func(t *testing.T) {
			writeScript(t, exe, "echo old\n")
			_ = os.Remove(exe + ".old")
			if err := runInstallerOverRunningBinary(exe, install); err == nil {
				t.Fatal("failed update reported success")
			}
			if got := read(exe); got != oldBody {
				t.Fatalf("binary after failed update = %q, want restored original", got)
			}
			if _, err := os.Stat(exe + ".old"); !os.IsNotExist(err) {
				t.Fatalf("moved-aside binary left behind: %v", err)
			}
		})
	}
}
