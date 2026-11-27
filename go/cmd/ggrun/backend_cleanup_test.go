package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/backends"
)

// cleanupFixture is a temporary app home with two registered forks and an
// orphan, wired to the real manifest through LLM_APP_HOME.
type cleanupFixture struct {
	t      *testing.T
	home   string
	procs  []backends.ProcInfo
	noProc bool
}

const testForkSHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func newCleanupFixture(t *testing.T) *cleanupFixture {
	t.Helper()
	isolateConfig(t)
	home, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLM_APP_HOME", home)
	// Every command in these tests deletes through the real AppHome(): prove it
	// is this test's temporary directory before anything can run.
	if got := backends.AppHome(); got != home {
		t.Fatalf("app home is %q, not the temporary %q; refusing to run deletion tests", got, home)
	}
	f := &cleanupFixture{t: t, home: home}
	for _, name := range []string{"fork-a", "fork-b", "fork-orphan"} {
		f.file(".src/"+name+"/src/main.cpp", "int main(){}")
		f.file(".src/"+name+"/.git/HEAD", testForkSHA+"\n")
		f.file(".src/"+name+"/.git/refs/remotes/origin/main", testForkSHA+"\n")
		f.file(".src/"+name+"/.git/index", "index")
		f.file(".src/"+name+"/build-cuda/bin/llama-server", "binary")
	}
	f.settle()
	if err := backends.Save([]backends.Backend{
		{Tag: "a", Path: f.path(".src/fork-a/build-cuda/bin/llama-server")},
		{Tag: "b", Path: f.path(".src/fork-b/build-cuda/bin/llama-server")},
	}); err != nil {
		t.Fatal(err)
	}
	return f
}

// settle makes everything under .src 72 hours old.
func (f *cleanupFixture) settle() {
	old := time.Now().Add(-72 * time.Hour)
	_ = filepath.WalkDir(f.path(".src"), func(p string, _ os.DirEntry, err error) error {
		if err == nil {
			_ = os.Chtimes(p, old, old)
		}
		return nil
	})
}

func (f *cleanupFixture) path(rel string) string {
	return filepath.Join(f.home, filepath.FromSlash(rel))
}

func (f *cleanupFixture) file(rel, body string) {
	f.t.Helper()
	p := f.path(rel)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		f.t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o755); err != nil {
		f.t.Fatal(err)
	}
}

func (f *cleanupFixture) exists(rel string) bool {
	_, err := os.Lstat(f.path(rel))
	return err == nil
}

func (f *cleanupFixture) env(o cleanupOpts) backends.CleanupEnv {
	e := defaultCleanupEnv(o)
	if e.AppHome != f.home {
		f.t.Fatalf("cleanup env targets %q, not the temporary app home %q", e.AppHome, f.home)
	}
	e.Processes = func() ([]backends.ProcInfo, error) { return f.procs, nil }
	if f.noProc {
		e.Processes = nil
	}
	return e
}

func (f *cleanupFixture) tags() string {
	var tags []string
	for _, b := range backends.Load() {
		tags = append(tags, b.Tag)
	}
	return strings.Join(tags, ",")
}

func tio(terminal bool, input string) (cleanupIO, *bytes.Buffer) {
	var out bytes.Buffer
	return cleanupIO{In: strings.NewReader(input), Out: &out, Terminal: terminal}, &out
}

func TestBackendRemoveWithoutPurgeOnlyUnregisters(t *testing.T) {
	f := newCleanupFixture(t)
	c, out := tio(false, "")
	if err := runBackendRemove([]string{"a"}, c, f.env, backends.Remove); err != nil {
		t.Fatal(err)
	}
	if f.tags() != "b" || !f.exists(".src/fork-a/src/main.cpp") {
		t.Fatalf("tags %q, tree kept=%v", f.tags(), f.exists(".src/fork-a/src/main.cpp"))
	}
	if !strings.Contains(out.String(), `Removed backend "a"`) {
		t.Fatalf("output %q", out.String())
	}
	if err := runBackendRemove([]string{"a"}, c, f.env, backends.Remove); err == nil {
		t.Fatal("removing an unknown tag must fail")
	}
}

func TestBackendRemovePurgeRefusalsChangeNothing(t *testing.T) {
	cases := []struct {
		name     string
		args     []string
		terminal bool
		input    string
		setup    func(f *cleanupFixture)
		wantErr  string
	}{
		{name: "no terminal and no --yes", args: []string{"a", "--purge"}, wantErr: "without a terminal"},
		{name: "terminal answers no", args: []string{"a", "--purge"}, terminal: true, input: "n\n", wantErr: "not confirmed"},
		{name: "terminal closed input", args: []string{"a", "--purge"}, terminal: true, input: "", wantErr: "not confirmed"},
		{name: "process runs from the tree", args: []string{"a", "--purge", "--yes"}, wantErr: "process is running",
			setup: func(f *cleanupFixture) {
				f.procs = []backends.ProcInfo{{PID: 3, Exe: f.path(".src/fork-a/build-cuda/bin/llama-server")}}
			}},
		{name: "no process check and no force flag", args: []string{"a", "--purge", "--yes"}, wantErr: "cannot check for running processes",
			setup: func(f *cleanupFixture) { f.noProc = true }},
		{name: "shared tree", args: []string{"a", "--purge", "--yes"}, wantErr: "shared",
			setup: func(f *cleanupFixture) {
				f.file(".src/fork-a/build-vulkan/bin/llama-server", "x")
				list := backends.Load()
				list = append(list, backends.Backend{Tag: "a-vk", Path: f.path(".src/fork-a/build-vulkan/bin/llama-server")})
				if err := backends.Save(list); err != nil {
					t.Fatal(err)
				}
			}},
		{name: "mainline checkout", args: []string{"main", "--purge", "--yes"}, wantErr: "not an isolated",
			setup: func(f *cleanupFixture) {
				f.file(".src/llama.cpp/build-cuda/bin/llama-server", "x")
				if err := backends.Upsert(backends.Backend{Tag: "main", Path: f.path(".src/llama.cpp/build-cuda/bin/llama-server")}); err != nil {
					t.Fatal(err)
				}
			}},
		{name: "unknown flag", args: []string{"a", "--purgee"}, wantErr: "unknown flag"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newCleanupFixture(t)
			if tc.setup != nil {
				tc.setup(f)
			}
			before := f.tags()
			c, _ := tio(tc.terminal, tc.input)
			err := runBackendRemove(tc.args, c, f.env, backends.Remove)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("got %v, want %q", err, tc.wantErr)
			}
			if f.tags() != before {
				t.Fatalf("registration changed: %q -> %q", before, f.tags())
			}
			for _, rel := range []string{".src/fork-a/src/main.cpp", ".src/fork-b/src/main.cpp", ".src/fork-orphan/src/main.cpp"} {
				if !f.exists(rel) {
					t.Fatalf("%s was deleted", rel)
				}
			}
		})
	}
}

func TestBackendRemovePurgeDryRunListsOnly(t *testing.T) {
	f := newCleanupFixture(t)
	c, out := tio(true, "y\n")
	if err := runBackendRemove([]string{"a", "--purge", "--dry-run"}, c, f.env, backends.Remove); err != nil {
		t.Fatal(err)
	}
	if f.tags() != "a,b" || !f.exists(".src/fork-a/src/main.cpp") {
		t.Fatal("dry run changed something")
	}
	for _, want := range []string{"fork-a", "source tree", "Dry run", "local work: none detected (staged changes and edits older than the git index cannot be seen without git)"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
}

func TestBackendRemovePurgeDeletesOnlyThatBackendsTree(t *testing.T) {
	for _, tc := range []struct {
		name     string
		args     []string
		terminal bool
		input    string
	}{
		{"terminal confirms", []string{"a", "--purge"}, true, "y\n"},
		{"--yes without a terminal", []string{"a", "--purge", "--yes"}, false, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newCleanupFixture(t)
			c, out := tio(tc.terminal, tc.input)
			if err := runBackendRemove(tc.args, c, f.env, backends.Remove); err != nil {
				t.Fatal(err)
			}
			if f.tags() != "b" || f.exists(".src/fork-a") {
				t.Fatalf("tags %q, fork-a exists=%v", f.tags(), f.exists(".src/fork-a"))
			}
			for _, rel := range []string{".src/fork-b/src/main.cpp", ".src/fork-orphan/src/main.cpp"} {
				if !f.exists(rel) {
					t.Fatalf("%s was deleted", rel)
				}
			}
			if !strings.Contains(out.String(), "Removed "+f.path(".src/fork-a")) {
				t.Fatalf("output must name what was removed:\n%s", out.String())
			}
		})
	}
}

func TestBackendGCListsByDefaultAndDeletesWithConsent(t *testing.T) {
	f := newCleanupFixture(t)
	c, out := tio(true, "")
	if err := runBackendGC(nil, c, f.env); err != nil {
		t.Fatal(err)
	}
	if !f.exists(".src/fork-orphan/src/main.cpp") || !strings.Contains(out.String(), "fork-orphan") || strings.Contains(out.String(), "fork-a\n") {
		t.Fatalf("default must list the orphan and delete nothing:\n%s", out.String())
	}
	if err := runBackendGC([]string{"--delete"}, mustIO(false), f.env); err == nil || !strings.Contains(err.Error(), "without a terminal") {
		t.Fatalf("--delete without a terminal: %v", err)
	}
	c, _ = tio(true, "no\n")
	if err := runBackendGC([]string{"--delete"}, c, f.env); err == nil || !f.exists(".src/fork-orphan/src/main.cpp") {
		t.Fatalf("declined delete removed something or succeeded: %v", err)
	}
	f.procs = []backends.ProcInfo{{PID: 4, Cwd: f.path(".src/fork-orphan")}}
	c, out = tio(false, "")
	if err := runBackendGC([]string{"--yes"}, c, f.env); err != nil || !f.exists(".src/fork-orphan/src/main.cpp") {
		t.Fatalf("a running process must block the orphan: %v\n%s", err, out.String())
	}
	f.procs = nil
	c, out = tio(false, "")
	if err := runBackendGC([]string{"--yes"}, c, f.env); err != nil {
		t.Fatal(err)
	}
	if f.exists(".src/fork-orphan") || !f.exists(".src/fork-a/src/main.cpp") || !f.exists(".src/fork-b/src/main.cpp") || f.tags() != "a,b" {
		t.Fatal("gc must remove exactly the orphan")
	}
	if !strings.Contains(out.String(), "Removed "+f.path(".src/fork-orphan")) {
		t.Fatalf("output must name what was removed:\n%s", out.String())
	}
	c, out = tio(false, "")
	if err := runBackendGC(nil, c, f.env); err != nil || !strings.Contains(out.String(), "No orphaned") {
		t.Fatalf("%v %q", err, out.String())
	}
}

func TestBackendCleanupRefusesWhenManifestIsCorrupt(t *testing.T) {
	f := newCleanupFixture(t)
	if err := os.WriteFile(backends.ManifestPath(), []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	c, _ := tio(false, "")
	if err := runBackendGC([]string{"--yes"}, c, f.env); err == nil || !strings.Contains(err.Error(), "manifest unavailable") {
		t.Fatalf("gc --yes on a corrupt manifest: %v", err)
	}
	if err := runBackendRemove([]string{"a", "--purge", "--yes"}, c, f.env, backends.Remove); err == nil {
		t.Fatal("purge on a corrupt manifest must refuse")
	}
	for _, rel := range []string{".src/fork-a/src/main.cpp", ".src/fork-b/src/main.cpp", ".src/fork-orphan/src/main.cpp"} {
		if !f.exists(rel) {
			t.Fatalf("%s was deleted", rel)
		}
	}
}

func TestBackendCleanupPlansNothingWithoutAManifest(t *testing.T) {
	f := newCleanupFixture(t)
	if err := os.Remove(backends.ManifestPath()); err != nil {
		t.Fatal(err)
	}
	if env := f.env(cleanupOpts{}); !errors.Is(env.ManifestErr, backends.ErrManifestMissing) {
		t.Fatalf("a missing manifest must be reported, not read as empty: %+v", env.ManifestErr)
	}
	c, _ := tio(false, "")
	if err := runBackendGC([]string{"--yes"}, c, f.env); err == nil || !strings.Contains(err.Error(), "does not exist") {
		t.Fatalf("gc --yes without a manifest: %v", err)
	}
	for _, rel := range []string{".src/fork-a/src/main.cpp", ".src/fork-b/src/main.cpp", ".src/fork-orphan/src/main.cpp"} {
		if !f.exists(rel) {
			t.Fatalf("%s was deleted", rel)
		}
	}
}

func TestBackendPurgeNeedsBothFlagsForLocalWork(t *testing.T) {
	for _, tc := range []struct {
		name    string
		args    []string
		term    bool
		input   string
		wantErr string
	}{
		{"typed y is not enough", []string{"a", "--purge"}, true, "y\n", "local work"},
		{"--yes alone is not enough", []string{"a", "--purge", "--yes"}, false, "", "local work"},
		{"the discard flag alone is not enough", []string{"a", "--purge", "--discard-local-work"}, true, "y\n", "local work"},
		{"both flags", []string{"a", "--purge", "--yes", "--discard-local-work"}, false, "", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newCleanupFixture(t)
			f.file(".src/fork-a/.git/refs/stash", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n")
			f.settle()
			c, out := tio(tc.term, tc.input)
			err := runBackendRemove(tc.args, c, f.env, backends.Remove)
			if tc.wantErr == "" {
				if err != nil || f.exists(".src/fork-a") || f.tags() != "b" {
					t.Fatalf("%v tags=%q", err, f.tags())
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) || !f.exists(".src/fork-a/src/main.cpp") || f.tags() != "a,b" {
				t.Fatalf("got %v tags=%q\n%s", err, f.tags(), out.String())
			}
			if !strings.Contains(out.String(), "stash entries") {
				t.Fatalf("the plan must show what was found:\n%s", out.String())
			}
		})
	}
}

func TestBackendPurgeRefusesWorktreesRecentChangeAndSymlinkedRoot(t *testing.T) {
	t.Run("linked worktrees", func(t *testing.T) {
		f := newCleanupFixture(t)
		f.file(".src/fork-a/.git/worktrees/wt/gitdir", "/elsewhere/.git\n")
		f.settle()
		c, _ := tio(false, "")
		if err := runBackendRemove([]string{"a", "--purge", "--yes"}, c, f.env, backends.Remove); err == nil || !strings.Contains(err.Error(), "worktree") || f.tags() != "a,b" {
			t.Fatalf("%v", err)
		}
	})
	t.Run("changed in the last 30 minutes", func(t *testing.T) {
		f := newCleanupFixture(t)
		f.file(".src/fork-a/build-cuda/new.o", "fresh")
		c, _ := tio(false, "")
		if err := runBackendRemove([]string{"a", "--purge", "--yes"}, c, f.env, backends.Remove); err == nil || !strings.Contains(err.Error(), "modified recently") || !f.exists(".src/fork-a/src/main.cpp") {
			t.Fatalf("%v", err)
		}
	})
}

func TestBackendCleanupRechecksAfterConfirmation(t *testing.T) {
	t.Run("gc: a registration appears while waiting", func(t *testing.T) {
		f := newCleanupFixture(t)
		calls := 0
		envFor := func(o cleanupOpts) backends.CleanupEnv {
			calls++
			if calls == 2 {
				if err := backends.Upsert(backends.Backend{Tag: "late", Path: f.path(".src/fork-orphan/build-cuda/bin/llama-server")}); err != nil {
					t.Fatal(err)
				}
			}
			return f.env(o)
		}
		c, out := tio(false, "")
		err := runBackendGC([]string{"--yes"}, c, envFor)
		if err == nil || !f.exists(".src/fork-orphan/src/main.cpp") || !strings.Contains(out.String(), "changed while waiting") {
			t.Fatalf("a tree registered after the listing must survive: %v\n%s", err, out.String())
		}
	})
	t.Run("purge: the tree is shared while waiting", func(t *testing.T) {
		f := newCleanupFixture(t)
		calls := 0
		envFor := func(o cleanupOpts) backends.CleanupEnv {
			calls++
			if calls == 2 {
				f.file(".src/fork-a/build-vulkan/bin/llama-server", "x")
				f.settle()
				if err := backends.Upsert(backends.Backend{Tag: "a-vk", Path: f.path(".src/fork-a/build-vulkan/bin/llama-server")}); err != nil {
					t.Fatal(err)
				}
			}
			return f.env(o)
		}
		c, _ := tio(false, "")
		err := runBackendRemove([]string{"a", "--purge", "--yes"}, c, envFor, backends.Remove)
		if err == nil || !strings.Contains(err.Error(), "after confirmation") || !f.exists(".src/fork-a/src/main.cpp") || f.tags() != "a,b,a-vk" {
			t.Fatalf("%v tags=%q", err, f.tags())
		}
	})
}

func mustIO(terminal bool) cleanupIO {
	c, _ := tio(terminal, "")
	return c
}

func TestBackendStatusReportsSupersededEntry(t *testing.T) {
	oldLookup, oldHead := statusRecipeLookup, statusCheckoutHead
	t.Cleanup(func() { statusRecipeLookup, statusCheckoutHead = oldLookup, oldHead })
	pin := "1111111111111111111111111111111111111111"
	statusRecipeLookup = func(tag string) *backends.Recipe {
		if tag != "forky" {
			return nil
		}
		return &backends.Recipe{Name: "forky", Commit: pin, Kind: "fork",
			SupersededBy: "mainline llama.cpp b9000", Evidence: []string{"https://example.invalid/pr/1"}, Expires: "2026-01-31"}
	}
	statusCheckoutHead = func(string) string { return "2222222222222222222222222222222222222222" }

	f := newCleanupFixture(t)
	now := time.Now()
	list := []backends.Backend{
		{Tag: "forky", Path: f.path(".src/fork-a/build-cuda/bin/llama-server"), Commit: pin,
			Build: &backends.BuildRecord{Seconds: 2460, TreeBytes: 3 << 30, At: now.Add(-50 * time.Hour).UTC().Format(time.RFC3339)}},
		{Tag: "mine", Path: f.path(".src/fork-b/build-cuda/bin/llama-server"), Commit: "abc"},
	}
	var out bytes.Buffer
	printBackendStatus(&out, list, now)
	got := out.String()
	for _, want := range []string{
		"forky", "build:    2 days old", "SUPERSEDED UPSTREAM: mainline llama.cpp b9000",
		"evidence: https://example.invalid/pr/1", "catalog kind: fork", "last build took 41 min, tree 3.0 GB",
		"at 222222222222, not the pin 111111111111", "review date 2026-01-31 has passed",
		"mine", "none (hand-added",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("status lacks %q:\n%s", want, got)
		}
	}
	if strings.Count(got, "SUPERSEDED") != 1 {
		t.Fatalf("only the marked entry may be flagged:\n%s", got)
	}
	if note := supersededNote("forky", statusRecipeLookup("forky")); !strings.Contains(note, "superseded upstream (mainline llama.cpp b9000)") || strings.Count(note, "\n") != 1 {
		t.Fatalf("update note must be one line: %q", note)
	}
	if supersededNote("mine", nil) != "" {
		t.Fatal("no note for an entry that is not superseded")
	}
}

func TestBuildEstimateFallsBackToConstant(t *testing.T) {
	const constant = "this can take 30–60 min"
	for name, be := range map[string]*backends.Backend{
		"unregistered":     nil,
		"no record":        {Tag: "x"},
		"zero seconds":     {Tag: "x", Build: &backends.BuildRecord{}},
		"negative seconds": {Tag: "x", Build: &backends.BuildRecord{Seconds: -4}},
	} {
		if got := buildEstimate(be); got != constant {
			t.Errorf("%s: %q", name, got)
		}
	}
	got := buildEstimate(&backends.Backend{Tag: "x", Build: &backends.BuildRecord{Seconds: 1800, TreeBytes: 2 << 30}})
	if got != "last build here took 30 min and left a 2.0 GB tree" {
		t.Fatalf("%q", got)
	}
	if got := buildEstimate(&backends.Backend{Build: &backends.BuildRecord{Seconds: 40}}); got != "last build here took 40 s" {
		t.Fatalf("%q", got)
	}
}

func TestParseCleanupArgs(t *testing.T) {
	o, err := parseCleanupArgs([]string{"--purge", "tag", "--yes", "--force-not-running", "--discard-local-work"}, true)
	if err != nil || o.Tag != "tag" || !o.Purge || !o.Yes || !o.ForceNotRunning || !o.DiscardLocalWork {
		t.Fatalf("%+v %v", o, err)
	}
	for _, bad := range [][]string{{}, {"a", "b"}, {"--bogus", "a"}} {
		if _, err := parseCleanupArgs(bad, true); err == nil {
			t.Errorf("%v accepted", bad)
		} else if _, usage := err.(backendUsageError); !usage {
			t.Errorf("%v: not a usage error: %v", bad, errors.Unwrap(err))
		}
	}
	if _, err := parseCleanupArgs([]string{"stray"}, false); err == nil {
		t.Fatal("gc takes no tag")
	}
}
