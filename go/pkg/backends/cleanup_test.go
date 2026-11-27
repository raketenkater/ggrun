package backends

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// cleanupHome builds a temporary app home. Every path is resolved so the
// simulated process table can name the same files the code sees.
type cleanupHome struct {
	t    *testing.T
	Home string
}

func newCleanupHome(t *testing.T) *cleanupHome {
	t.Helper()
	home, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	h := &cleanupHome{t: t, Home: home}
	h.mkdir(".src")
	h.mkdir(".bin")
	return h
}

func (h *cleanupHome) mkdir(rel string) string {
	h.t.Helper()
	p := filepath.Join(h.Home, filepath.FromSlash(rel))
	if err := os.MkdirAll(p, 0o755); err != nil {
		h.t.Fatal(err)
	}
	return p
}

func (h *cleanupHome) write(rel, body string) string {
	h.t.Helper()
	p := filepath.Join(h.Home, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		h.t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o755); err != nil {
		h.t.Fatal(err)
	}
	return p
}

// fork makes .src/<name>/build-cuda/bin/llama-server plus a source file and
// returns the binary path.
func (h *cleanupHome) fork(name string) string {
	h.write(".src/"+name+"/src/main.cpp", "int main(){}")
	// A clean checkout as far as the .git files can tell: HEAD is a remote tip.
	h.write(".src/"+name+"/.git/HEAD", testSHA+"\n")
	h.write(".src/"+name+"/.git/refs/remotes/origin/main", testSHA+"\n")
	h.write(".src/"+name+"/.git/index", "index")
	return h.write(".src/"+name+"/build-cuda/bin/llama-server", "binary")
}

const testSHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

// settle makes everything under .src old enough to pass the quiet window.
func (h *cleanupHome) settle() { h.age(".src", 48*time.Hour) }

func (h *cleanupHome) symlink(target, rel string) {
	h.t.Helper()
	if err := os.Symlink(target, filepath.Join(h.Home, filepath.FromSlash(rel))); err != nil {
		h.t.Skipf("symlinks unavailable: %v", err)
	}
}

func (h *cleanupHome) exists(rel string) bool {
	_, err := os.Lstat(filepath.Join(h.Home, filepath.FromSlash(rel)))
	return err == nil
}

// age sets every entry under rel to the given age.
func (h *cleanupHome) age(rel string, d time.Duration) {
	h.t.Helper()
	when := time.Now().Add(-d)
	var all []string
	_ = filepath.WalkDir(filepath.Join(h.Home, filepath.FromSlash(rel)), func(p string, _ os.DirEntry, err error) error {
		if err == nil {
			all = append(all, p)
		}
		return nil
	})
	for i := len(all) - 1; i >= 0; i-- {
		_ = os.Chtimes(all[i], when, when)
	}
}

func (h *cleanupHome) env(list []Backend, procs []ProcInfo) CleanupEnv {
	return CleanupEnv{
		AppHome:   h.Home,
		Backends:  list,
		Processes: func() ([]ProcInfo, error) { return procs, nil },
	}
}

func TestPlanPurgeRefusals(t *testing.T) {
	cases := []struct {
		name  string
		setup func(h *cleanupHome) (list []Backend, procs []ProcInfo, mutate func(*CleanupEnv))
		tag   string
		want  error
		keep  []string // paths that must still exist
	}{
		{
			name: "binary outside the source root",
			setup: func(h *cleanupHome) ([]Backend, []ProcInfo, func(*CleanupEnv)) {
				other := t.TempDir()
				bin := filepath.Join(other, "build-cuda", "bin", "llama-server")
				_ = os.MkdirAll(filepath.Dir(bin), 0o755)
				_ = os.WriteFile(bin, []byte("x"), 0o755)
				return []Backend{{Tag: "ext", Path: bin}}, nil, nil
			},
			tag: "ext", want: ErrNoTree,
		},
		{
			name: "mainline checkout is never purged",
			setup: func(h *cleanupHome) ([]Backend, []ProcInfo, func(*CleanupEnv)) {
				bin := h.fork("llama.cpp")
				return []Backend{{Tag: "mainline", Path: bin}}, nil, nil
			},
			tag: "mainline", want: ErrNotForkTree, keep: []string{".src/llama.cpp/src/main.cpp"},
		},
		{
			name: "ik_llama checkout is never purged",
			setup: func(h *cleanupHome) ([]Backend, []ProcInfo, func(*CleanupEnv)) {
				bin := h.fork("ik_llama.cpp")
				return []Backend{{Tag: "ik", Path: bin}}, nil, nil
			},
			tag: "ik", want: ErrNotForkTree, keep: []string{".src/ik_llama.cpp/src/main.cpp"},
		},
		{
			name: "tree is a symlink out of the root",
			setup: func(h *cleanupHome) ([]Backend, []ProcInfo, func(*CleanupEnv)) {
				outside := t.TempDir()
				bin := filepath.Join(outside, "build-cuda", "bin", "llama-server")
				_ = os.MkdirAll(filepath.Dir(bin), 0o755)
				_ = os.WriteFile(bin, []byte("x"), 0o755)
				h.symlink(outside, ".src/fork-escape")
				return []Backend{{Tag: "esc", Path: filepath.Join(h.Home, ".src", "fork-escape", "build-cuda", "bin", "llama-server")}}, nil, nil
			},
			tag: "esc", want: ErrSymlink,
		},
		{
			name: "tree is a symlink to another tree inside the root",
			setup: func(h *cleanupHome) ([]Backend, []ProcInfo, func(*CleanupEnv)) {
				h.fork("fork-real")
				h.symlink(filepath.Join(h.Home, ".src", "fork-real"), ".src/fork-alias")
				return []Backend{{Tag: "alias", Path: filepath.Join(h.Home, ".src", "fork-alias", "build-cuda", "bin", "llama-server")}}, nil, nil
			},
			tag: "alias", want: ErrSymlink, keep: []string{".src/fork-real/src/main.cpp"},
		},
		{
			name: "two registrations share the tree",
			setup: func(h *cleanupHome) ([]Backend, []ProcInfo, func(*CleanupEnv)) {
				bin := h.fork("fork-shared")
				vk := h.write(".src/fork-shared/build-vulkan/bin/llama-server", "x")
				return []Backend{{Tag: "cuda", Path: bin}, {Tag: "vulkan", Path: vk}}, nil, nil
			},
			tag: "cuda", want: ErrSharedTree, keep: []string{".src/fork-shared/src/main.cpp"},
		},
		{
			name: "another registration keeps its previous build in the tree",
			setup: func(h *cleanupHome) ([]Backend, []ProcInfo, func(*CleanupEnv)) {
				bin := h.fork("fork-prev")
				other := h.fork("fork-other")
				prev := h.write(".src/fork-prev/build-cuda-prev-abc/bin/llama-server", "x")
				return []Backend{{Tag: "a", Path: bin}, {Tag: "b", Path: other, Previous: &BackendVersion{Path: prev}}}, nil, nil
			},
			tag: "a", want: ErrSharedTree,
		},
		{
			name: "a process runs the binary",
			setup: func(h *cleanupHome) ([]Backend, []ProcInfo, func(*CleanupEnv)) {
				bin := h.fork("fork-run")
				return []Backend{{Tag: "run", Path: bin}}, []ProcInfo{{PID: 42, Exe: bin}}, nil
			},
			tag: "run", want: ErrRunning, keep: []string{".src/fork-run/src/main.cpp"},
		},
		{
			name: "a build runs with its working directory in the tree",
			setup: func(h *cleanupHome) ([]Backend, []ProcInfo, func(*CleanupEnv)) {
				bin := h.fork("fork-cwd")
				return []Backend{{Tag: "cwd", Path: bin}}, []ProcInfo{{PID: 7, Exe: "/usr/bin/cmake", Cwd: filepath.Join(h.Home, ".src", "fork-cwd")}}, nil
			},
			tag: "cwd", want: ErrRunning,
		},
		{
			name: "force flag never overrides a process that is seen",
			setup: func(h *cleanupHome) ([]Backend, []ProcInfo, func(*CleanupEnv)) {
				bin := h.fork("fork-forced-run")
				return []Backend{{Tag: "fr", Path: bin}}, []ProcInfo{{PID: 11, Exe: bin}}, func(e *CleanupEnv) { e.ForceNotRunning = true }
			},
			tag: "fr", want: ErrRunning, keep: []string{".src/fork-forced-run/src/main.cpp"},
		},
		{
			name: "manifest unreadable",
			setup: func(h *cleanupHome) ([]Backend, []ProcInfo, func(*CleanupEnv)) {
				bin := h.fork("fork-manifest")
				return []Backend{{Tag: "m", Path: bin}}, nil, func(e *CleanupEnv) { e.ManifestErr = errors.New("decode backend manifest: invalid character") }
			},
			tag: "m", keep: []string{".src/fork-manifest/src/main.cpp"},
		},
		{
			name: "no process check on this platform",
			setup: func(h *cleanupHome) ([]Backend, []ProcInfo, func(*CleanupEnv)) {
				bin := h.fork("fork-nocheck")
				return []Backend{{Tag: "nc", Path: bin}}, nil, func(e *CleanupEnv) { e.Processes = nil }
			},
			tag: "nc", want: ErrNoProcessCheck, keep: []string{".src/fork-nocheck/src/main.cpp"},
		},
		{
			name: "process table cannot be read",
			setup: func(h *cleanupHome) ([]Backend, []ProcInfo, func(*CleanupEnv)) {
				bin := h.fork("fork-err")
				return []Backend{{Tag: "er", Path: bin}}, nil, func(e *CleanupEnv) {
					e.Processes = func() ([]ProcInfo, error) { return nil, errors.New("denied") }
				}
			},
			tag: "er", want: ErrNoProcessCheck,
		},
		{
			name: "unknown tag",
			setup: func(h *cleanupHome) ([]Backend, []ProcInfo, func(*CleanupEnv)) {
				return nil, nil, nil
			},
			tag: "ghost",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newCleanupHome(t)
			list, procs, mutate := tc.setup(h)
			env := h.env(list, procs)
			if mutate != nil {
				mutate(&env)
			}
			_, err := PlanPurge(env, tc.tag)
			if err == nil {
				t.Fatal("expected a refusal")
			}
			if tc.want != nil && !errors.Is(err, tc.want) {
				t.Fatalf("got %v, want %v", err, tc.want)
			}
			for _, k := range tc.keep {
				if !h.exists(k) {
					t.Fatalf("%s was removed", k)
				}
			}
		})
	}
}

func TestSymlinkEscapeNamesBothReasonsAndLeavesTarget(t *testing.T) {
	h := newCleanupHome(t)
	outside := t.TempDir()
	keep := filepath.Join(outside, "precious.txt")
	if err := os.WriteFile(keep, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	h.symlink(outside, ".src/fork-escape")
	env := h.env(nil, nil)
	_, err := resolveUnderRoot(SourceRoot(env.AppHome), filepath.Join(h.Home, ".src", "fork-escape"))
	if !errors.Is(err, ErrSymlink) || !errors.Is(err, ErrOutsideRoot) {
		t.Fatalf("want symlink and outside-root, got %v", err)
	}
	plans, err := PlanGC(env)
	if err != nil || len(plans) != 1 || plans[0].Blocked == "" {
		t.Fatalf("gc must list the symlink as blocked: %v %+v", err, plans)
	}
	if _, err := env.Delete(plans[0]); err == nil {
		t.Fatal("Delete accepted a symlinked tree")
	}
	if _, err := os.Stat(keep); err != nil {
		t.Fatal("symlink target was touched")
	}
}

func TestResolveUnderRootRejectsPathsOutside(t *testing.T) {
	h := newCleanupHome(t)
	h.fork("fork-a")
	root := SourceRoot(h.Home)
	for name, p := range map[string]string{
		"parent of the root": h.Home,
		"the root itself":    root,
		"dot-dot traversal":  filepath.Join(root, "fork-a", "..", "..", ".bin"),
		"sibling directory":  filepath.Join(h.Home, ".bin"),
		"other temp dir":     t.TempDir(),
	} {
		if _, err := resolveUnderRoot(root, p); !errors.Is(err, ErrOutsideRoot) {
			t.Errorf("%s: got %v, want ErrOutsideRoot", name, err)
		}
	}
	if _, err := resolveUnderRoot(root, filepath.Join(root, "fork-a")); err != nil {
		t.Fatalf("a real tree inside the root must resolve: %v", err)
	}
}

func TestPurgeForceNotRunningAllowsPlatformsWithoutProcessCheck(t *testing.T) {
	h := newCleanupHome(t)
	bin := h.fork("fork-forced")
	h.settle()
	env := h.env([]Backend{{Tag: "f", Path: bin}}, nil)
	env.Processes = nil
	env.ForceNotRunning = true
	plan, err := PlanPurge(env, "f")
	if err != nil {
		t.Fatalf("--force-not-running must allow it: %v", err)
	}
	if _, err := env.Delete(plan); err != nil {
		t.Fatal(err)
	}
	if h.exists(".src/fork-forced") {
		t.Fatal("tree not removed")
	}
}

func TestPurgeDeletesExactlyTheTree(t *testing.T) {
	h := newCleanupHome(t)
	binA := h.fork("fork-a")
	h.fork("fork-b")
	h.fork("llama.cpp")
	h.write(".src/fork-a/build-cuda-prev-1/bin/llama-server", "old")
	// A symlink inside the tree pointing out must be unlinked, not followed.
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "keep.txt"), []byte("k"), 0o644); err != nil {
		t.Fatal(err)
	}
	h.symlink(outside, ".src/fork-a/outside-link")
	// .bin links: one into fork-a (goes), one into fork-b and a plain file (stay).
	h.symlink(binA, ".bin/a-server-cuda")
	h.symlink(filepath.Join(h.Home, ".src", "fork-b", "build-cuda", "bin", "llama-server"), ".bin/b-server-cuda")
	h.write(".bin/llama-server", "mainline")

	h.settle()
	list := []Backend{
		{Tag: "a", Path: binA},
		{Tag: "b", Path: filepath.Join(h.Home, ".src", "fork-b", "build-cuda", "bin", "llama-server")},
	}
	env := h.env(list, []ProcInfo{{PID: 1, Exe: filepath.Join(h.Home, ".src", "fork-b", "build-cuda", "bin", "llama-server")}})
	// The inner symlink's own mtime cannot be backdated portably: look from later.
	env.Now = time.Now().Add(72 * time.Hour)
	plan, err := PlanPurge(env, "a")
	if err != nil {
		t.Fatalf("a process in another tree must not block: %v", err)
	}
	if plan.Bytes == 0 || plan.Files < 3 || len(plan.Links) != 1 {
		t.Fatalf("plan not measured: %+v", plan)
	}
	removed, err := env.Delete(plan)
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 2 {
		t.Fatalf("removed %v", removed)
	}
	for _, gone := range []string{".src/fork-a", ".bin/a-server-cuda"} {
		if h.exists(gone) {
			t.Errorf("%s should be gone", gone)
		}
	}
	for _, kept := range []string{".src/fork-b/src/main.cpp", ".src/llama.cpp/src/main.cpp", ".bin/b-server-cuda", ".bin/llama-server"} {
		if !h.exists(kept) {
			t.Errorf("%s must survive", kept)
		}
	}
	if _, err := os.Stat(filepath.Join(outside, "keep.txt")); err != nil {
		t.Fatal("the symlink inside the tree was followed")
	}
}

func TestDeleteRechecksRunningProcess(t *testing.T) {
	h := newCleanupHome(t)
	bin := h.fork("fork-late")
	h.settle()
	var procs []ProcInfo
	env := h.env([]Backend{{Tag: "late", Path: bin}}, nil)
	env.Processes = func() ([]ProcInfo, error) { return procs, nil }
	plan, err := PlanPurge(env, "late")
	if err != nil {
		t.Fatal(err)
	}
	procs = []ProcInfo{{PID: 9, Exe: bin}} // starts between plan and delete
	if _, err := env.Delete(plan); !errors.Is(err, ErrRunning) {
		t.Fatalf("got %v", err)
	}
	if !h.exists(".src/fork-late/src/main.cpp") {
		t.Fatal("tree removed although a process started")
	}
}

func TestPlanGCListsOnlyUnreferencedTrees(t *testing.T) {
	h := newCleanupHome(t)
	active := h.fork("fork-live")
	prev := h.write(".src/fork-live/build-cuda-prev-keep/bin/llama-server", "prev")
	h.write(".src/fork-live/build-cuda-prev-old/bin/llama-server", "stale")
	h.write(".src/fork-live/build-cuda-candidate-777/bin/llama-server", "half built")
	h.write(".src/fork-live/build-vulkan/bin/llama-server", "hand build, not ggrun's leftover pattern")
	h.fork("fork-orphan")
	h.fork("fork-busy")
	h.fork("llama.cpp")
	h.fork("ik_llama.cpp-v1-legacy")
	h.mkdir(".src/ggrun")
	outside := t.TempDir()
	h.symlink(outside, ".src/fork-linked")
	// An unregistered tree that a .bin link still points at is not an orphan.
	h.fork("fork-linkref")
	h.symlink(filepath.Join(h.Home, ".src", "fork-linkref", "build-cuda", "bin", "llama-server"), ".bin/llama-server-cuda")

	for _, rel := range []string{".src/fork-live", ".src/fork-orphan", ".src/fork-busy", ".src/fork-linkref", ".src/llama.cpp", ".src/ik_llama.cpp-v1-legacy"} {
		h.age(rel, 48*time.Hour)
	}
	// The candidate directory is being written right now.
	h.age(".src/fork-live/build-cuda-candidate-777", 2*time.Minute)

	busyBin := filepath.Join(h.Home, ".src", "fork-busy", "build-cuda", "bin", "llama-server")
	env := h.env([]Backend{{Tag: "live", Path: active, Previous: &BackendVersion{Path: prev}}}, []ProcInfo{{PID: 5, Exe: busyBin}})
	plans, err := PlanGC(env)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]TreePlan{}
	for _, p := range plans {
		got[strings.TrimPrefix(p.Path, h.Home+string(filepath.Separator))] = p
	}
	rel := func(parts ...string) string { return filepath.Join(parts...) }
	wantRemovable := []string{rel(".src", "fork-orphan"), rel(".src", "fork-live", "build-cuda-prev-old")}
	wantBlocked := map[string]error{
		rel(".src", "fork-busy"):                             ErrRunning,
		rel(".src", "fork-linkref"):                          errors.New("referenced from .bin by " + filepath.Join(h.Home, ".bin", "llama-server-cuda")),
		rel(".src", "fork-live", "build-cuda-candidate-777"): ErrRecentlyModified,
		rel(".src", "fork-linked"):                           ErrSymlink,
	}
	if len(got) != len(wantRemovable)+len(wantBlocked) {
		t.Fatalf("listed %v", keys(got))
	}
	for _, w := range wantRemovable {
		if p, ok := got[w]; !ok || p.Blocked != "" {
			t.Errorf("%s should be removable: %+v", w, p)
		}
	}
	for w, why := range wantBlocked {
		p, ok := got[w]
		if !ok || p.Blocked == "" || !strings.Contains(p.Blocked, why.Error()) {
			t.Errorf("%s should be blocked by %v: %+v", w, why, p)
		}
	}
	if p := got[rel(".src", "fork-orphan")]; p.Bytes == 0 || p.LocalWork != "" {
		t.Errorf("orphan plan lacks size or reports local work: %+v", p)
	}

	for _, p := range plans {
		if p.Blocked != "" {
			// The CLI never deletes a blocked entry; Delete itself still
			// refuses the ones blocked by a safety rule (not by recency).
			if !strings.Contains(p.Blocked, ErrRecentlyModified.Error()) && !strings.Contains(p.Blocked, "referenced from .bin") {
				if _, err := env.Delete(p); err == nil {
					t.Errorf("Delete accepted blocked %s", p.Path)
				}
			}
			continue
		}
		if _, err := env.Delete(p); err != nil {
			t.Fatal(err)
		}
	}
	for _, gone := range []string{".src/fork-orphan", ".src/fork-live/build-cuda-prev-old"} {
		if h.exists(gone) {
			t.Errorf("%s should be gone", gone)
		}
	}
	for _, kept := range []string{
		".src/fork-live/build-cuda/bin/llama-server", ".src/fork-live/build-cuda-prev-keep/bin/llama-server",
		".src/fork-live/build-vulkan/bin/llama-server", ".src/fork-busy/src/main.cpp", ".src/llama.cpp/src/main.cpp",
		".src/ik_llama.cpp-v1-legacy/src/main.cpp", ".src", ".src/ggrun", ".src/fork-linked", ".src/fork-linkref/src/main.cpp", ".bin/llama-server-cuda",
	} {
		if !h.exists(kept) {
			t.Errorf("%s must survive", kept)
		}
	}
	if !h.exists(".src/fork-live/build-cuda-candidate-777/bin/llama-server") {
		t.Error("recently written candidate build must survive")
	}
}

func keys(m map[string]TreePlan) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	return out
}

func TestPlanGCWithoutProcessCheckBlocksEverything(t *testing.T) {
	h := newCleanupHome(t)
	h.fork("fork-orphan")
	h.age(".src/fork-orphan", 48*time.Hour)
	env := h.env(nil, nil)
	env.Processes = nil
	plans, err := PlanGC(env)
	if err != nil || len(plans) != 1 {
		t.Fatalf("%v %v", err, plans)
	}
	if !strings.Contains(plans[0].Blocked, ErrNoProcessCheck.Error()) {
		t.Fatalf("listing must say why it cannot delete: %q", plans[0].Blocked)
	}
	if _, err := env.Delete(plans[0]); !errors.Is(err, ErrNoProcessCheck) {
		t.Fatalf("got %v", err)
	}
	env.ForceNotRunning = true
	plans, _ = PlanGC(env)
	if plans[0].Blocked != "" {
		t.Fatalf("forced run still blocked: %q", plans[0].Blocked)
	}
}

func TestPlanGCRefusesWithUnreadableManifest(t *testing.T) {
	h := newCleanupHome(t)
	h.fork("fork-registered-but-manifest-unreadable")
	env := h.env(nil, nil)
	env.ManifestErr = errors.New("decode backend manifest: invalid character")
	if plans, err := PlanGC(env); err == nil || len(plans) != 0 {
		t.Fatalf("gc must plan nothing: %v %v", plans, err)
	}
}

func TestPlanGCMissingSourceRoot(t *testing.T) {
	plans, err := PlanGC(CleanupEnv{AppHome: t.TempDir()})
	if err != nil || len(plans) != 0 {
		t.Fatalf("%v %v", err, plans)
	}
}

func TestLinkedWorktreesBlockPurgeAndGC(t *testing.T) {
	h := newCleanupHome(t)
	bin := h.fork("fork-main")
	h.write(".src/fork-main/.git/worktrees/wt1/gitdir", "/elsewhere/hotx/llama/.git\n")
	h.fork("fork-orphan-main")
	h.write(".src/fork-orphan-main/.git/worktrees/wt2/gitdir", "/elsewhere/other/.git\n")
	h.fork("fork-prunable")
	h.mkdir(".src/fork-prunable/.git/worktrees") // empty: nothing depends on it
	h.settle()
	env := h.env([]Backend{{Tag: "main", Path: bin}}, nil)

	_, err := PlanPurge(env, "main")
	if !errors.Is(err, ErrLinkedWorktrees) || !strings.Contains(err.Error(), "wt1 (/elsewhere/hotx/llama/.git)") || !strings.Contains(err.Error(), "git worktree") {
		t.Fatalf("purge must refuse and name the worktree: %v", err)
	}
	plans, err := PlanGC(env)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]TreePlan{}
	for _, p := range plans {
		got[filepath.Base(p.Path)] = p
	}
	if p := got["fork-orphan-main"]; !strings.Contains(p.Blocked, "worktree") || !strings.Contains(p.Blocked, "wt2") {
		t.Fatalf("gc must block a tree with worktrees: %+v", p)
	}
	if p := got["fork-prunable"]; p.Blocked != "" {
		t.Fatalf("an empty worktrees directory must not block: %+v", p)
	}
	if _, err := env.Delete(got["fork-orphan-main"]); !errors.Is(err, ErrLinkedWorktrees) {
		t.Fatalf("Delete must refuse too: %v", err)
	}
	for _, rel := range []string{".src/fork-main/.git/worktrees/wt1/gitdir", ".src/fork-orphan-main/src/main.cpp"} {
		if !h.exists(rel) {
			t.Fatalf("%s was removed", rel)
		}
	}
}

func TestGitFileTreeIsALinkedWorktreeAndCountsAsUnknownLocalWork(t *testing.T) {
	h := newCleanupHome(t)
	h.write(".src/fork-wt/src/main.cpp", "x")
	h.write(".src/fork-wt/.git", "gitdir: /elsewhere/repo/.git/worktrees/wt\n")
	h.settle()
	env := h.env(nil, nil)
	if lw := env.LocalWork(filepath.Join(h.Home, ".src", "fork-wt")); !strings.HasPrefix(lw, "unknown: a linked worktree") || !strings.Contains(lw, "/elsewhere/repo") {
		t.Fatalf("got %q", lw)
	}
	plans, _ := PlanGC(env)
	if len(plans) != 1 || !strings.Contains(plans[0].Blocked, "--discard-local-work") {
		t.Fatalf("a linked worktree needs the explicit flag: %+v", plans)
	}
	env.DiscardLocalWork = true
	plans, _ = PlanGC(env)
	if plans[0].Blocked != "" {
		t.Fatalf("flag must allow it: %+v", plans[0])
	}
	if _, err := env.Delete(plans[0]); err != nil {
		t.Fatal(err)
	}
}

func TestSymlinkedSourceRootBlocksEverythingButListing(t *testing.T) {
	owner := newCleanupHome(t) // the app home that owns the trees
	bin := owner.fork("fork-theirs")
	owner.settle()
	other := newCleanupHome(t)
	if err := os.Remove(filepath.Join(other.Home, ".src")); err != nil {
		t.Fatal(err)
	}
	other.symlink(filepath.Join(owner.Home, ".src"), ".src")

	// This home's manifest does not know the tree; another home's does.
	env := other.env(nil, nil)
	plans, err := PlanGC(env)
	if err != nil || len(plans) != 1 {
		t.Fatalf("listing must still work: %v %v", err, plans)
	}
	if !strings.Contains(plans[0].Blocked, "symlink") || !strings.Contains(plans[0].Blocked, "not deletable") {
		t.Fatalf("must say .src is a symlink: %q", plans[0].Blocked)
	}
	if _, err := env.Delete(plans[0]); !errors.Is(err, ErrSourceRootSymlink) {
		t.Fatalf("Delete must refuse: %v", err)
	}
	// Even a registration made here, naming the shared tree, cannot purge it.
	env = other.env([]Backend{{Tag: "mine", Path: bin}}, nil)
	if _, err := PlanPurge(env, "mine"); !errors.Is(err, ErrSourceRootSymlink) {
		t.Fatalf("purge must refuse: %v", err)
	}
	if !owner.exists(".src/fork-theirs/src/main.cpp") {
		t.Fatal("the owner's tree was touched")
	}
}

func TestSymlinkedAppHomeIsAllowed(t *testing.T) {
	h := newCleanupHome(t)
	h.fork("fork-orphan")
	h.settle()
	link := filepath.Join(t.TempDir(), "homelink")
	if err := os.Symlink(h.Home, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	env := h.env(nil, nil)
	env.AppHome = link
	plans, err := PlanGC(env)
	if err != nil || len(plans) != 1 || plans[0].Blocked != "" {
		t.Fatalf("a symlink to the whole app home is the supported way to move .src: %v %+v", err, plans)
	}
}

func TestDeleteRemovesOnlyTheCheckedPath(t *testing.T) {
	h := newCleanupHome(t)
	h.fork("fork-a")
	h.fork("fork-b")
	h.settle()
	root := filepath.Join(h.Home, ".src")
	h.symlink(filepath.Join(root, "fork-b"), ".src/fork-a/sub")
	alias := filepath.Join(t.TempDir(), "alias")
	if err := os.Symlink(root, alias); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	env := h.env(nil, nil)
	sep := string(filepath.Separator)
	for name, path := range map[string]string{
		"dot-dot after a symlink":    root + sep + "fork-a" + sep + "sub" + sep + "..",
		"dot-dot inside the root":    root + sep + "fork-a" + sep + ".." + sep + "fork-b",
		"through a symlinked parent": filepath.Join(alias, "fork-a"),
		"the app home":               h.Home,
		"the source root":            root,
		"relative path":              filepath.Join(".src", "fork-a"),
		"empty":                      "",
		"trailing separator":         filepath.Join(root, "fork-a") + sep,
	} {
		if _, err := env.Delete(TreePlan{Path: path}); err == nil {
			t.Errorf("%s: Delete accepted %q", name, path)
		}
	}
	for _, rel := range []string{".src/fork-a/src/main.cpp", ".src/fork-b/src/main.cpp", ".bin", ".src"} {
		if !h.exists(rel) {
			t.Fatalf("%s was removed", rel)
		}
	}
	// The exact, clean, resolved path still works.
	if _, err := env.Delete(TreePlan{Path: filepath.Join(root, "fork-b")}); err != nil {
		t.Fatal(err)
	}
	if h.exists(".src/fork-b") || !h.exists(".src/fork-a/src/main.cpp") {
		t.Fatal("wrong tree removed")
	}
}

func TestPurgeAndDeleteRefuseARecentlyChangedTree(t *testing.T) {
	h := newCleanupHome(t)
	bin := h.fork("fork-busy-update")
	env := h.env([]Backend{{Tag: "u", Path: bin}}, nil)
	if _, err := PlanPurge(env, "u"); !errors.Is(err, ErrRecentlyModified) {
		t.Fatalf("a tree changed seconds ago must be refused: %v", err)
	}
	if _, err := env.Delete(TreePlan{Path: filepath.Join(h.Home, ".src", "fork-busy-update")}); !errors.Is(err, ErrRecentlyModified) {
		t.Fatalf("Delete must re-check: %v", err)
	}
	h.settle()
	if _, err := PlanPurge(env, "u"); err != nil {
		t.Fatalf("after the window it is allowed: %v", err)
	}
	env.QuietWindow = 100 * 24 * time.Hour
	if _, err := PlanPurge(env, "u"); !errors.Is(err, ErrRecentlyModified) {
		t.Fatalf("window must be honoured: %v", err)
	}
}

func TestLocalWorkFromGitFiles(t *testing.T) {
	other := "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	cases := []struct {
		name  string
		setup func(h *cleanupHome, tree string)
		want  string // substring; "" means no local work
	}{
		{name: "clean checkout", setup: func(h *cleanupHome, tree string) {}},
		{name: "no .git", want: "unknown: no .git", setup: func(h *cleanupHome, tree string) {
			_ = os.RemoveAll(filepath.Join(tree, ".git"))
		}},
		{name: "stash", want: "stash entries", setup: func(h *cleanupHome, tree string) {
			h.write(".src/fork-x/.git/refs/stash", other+"\n")
		}},
		{name: "detached HEAD at an unpublished commit", want: "commits not on any remote branch", setup: func(h *cleanupHome, tree string) {
			h.write(".src/fork-x/.git/HEAD", other+"\n")
		}},
		{name: "local branch ahead of the remote", want: "refs/heads/work", setup: func(h *cleanupHome, tree string) {
			h.write(".src/fork-x/.git/refs/heads/work", other+"\n")
		}},
		{name: "HEAD at a reviewed pin counts as published", setup: func(h *cleanupHome, tree string) {
			h.write(".src/fork-x/.git/HEAD", "f46c95ee90d8c8200b0147c646b883405020b482\n")
			_ = os.Remove(filepath.Join(tree, ".git", "refs", "remotes", "origin", "main"))
		}},
		{name: "HEAD in packed-refs of a remote", setup: func(h *cleanupHome, tree string) {
			h.write(".src/fork-x/.git/HEAD", other+"\n")
			h.write(".src/fork-x/.git/packed-refs", "# pack-refs\n"+other+" refs/remotes/origin/dev\n")
		}},
		{name: "file newer than the index", want: "newer than the git index", setup: func(h *cleanupHome, tree string) {
			old := time.Now().Add(-3 * time.Hour)
			_ = os.Chtimes(filepath.Join(tree, ".git", "index"), old, old)
			h.write(".src/fork-x/src/edited.cpp", "changed")
		}},
		{name: "build output newer than the index is ignored", setup: func(h *cleanupHome, tree string) {
			old := time.Now().Add(-3 * time.Hour)
			_ = os.Chtimes(filepath.Join(tree, ".git", "index"), old, old)
			_ = os.Chtimes(filepath.Join(tree, "src", "main.cpp"), old, old)
			h.write(".src/fork-x/build-cuda/bin/new.o", "object")
		}},
		{name: "no index", want: "no git index", setup: func(h *cleanupHome, tree string) {
			_ = os.Remove(filepath.Join(tree, ".git", "index"))
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newCleanupHome(t)
			h.fork("fork-x")
			tree := filepath.Join(h.Home, ".src", "fork-x")
			tc.setup(h, tree)
			got := h.env(nil, nil).LocalWork(tree)
			if tc.want == "" && got != "" || tc.want != "" && !strings.Contains(got, tc.want) {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestDeleteNeedsTheDiscardFlagForLocalWork(t *testing.T) {
	h := newCleanupHome(t)
	h.fork("fork-poc")
	h.write(".src/fork-poc/.git/refs/stash", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n")
	h.settle()
	env := h.env(nil, nil)
	plan := TreePlan{Path: filepath.Join(h.Home, ".src", "fork-poc")}
	if _, err := env.Delete(plan); !errors.Is(err, ErrLocalWork) {
		t.Fatalf("got %v", err)
	}
	if !h.exists(".src/fork-poc/src/main.cpp") {
		t.Fatal("removed without the flag")
	}
	env.DiscardLocalWork = true
	if _, err := env.Delete(plan); err != nil {
		t.Fatal(err)
	}
}

func TestRegistrationCountsEvenWhenBinaryIsGoneAndSpelledThroughASymlink(t *testing.T) {
	h := newCleanupHome(t)
	bin := h.fork("fork-failed-build")
	h.fork("fork-orphan")
	h.settle()
	if err := os.RemoveAll(filepath.Join(h.Home, ".src", "fork-failed-build", "build-cuda")); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(t.TempDir(), "homelink")
	if err := os.Symlink(h.Home, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	viaLink := filepath.Join(link, ".src", "fork-failed-build", "build-cuda", "bin", "llama-server")
	for name, path := range map[string]string{"resolved spelling, binary missing": bin, "symlink spelling, binary missing": viaLink} {
		env := h.env([]Backend{{Tag: "failed", Path: path}}, nil)
		plans, err := PlanGC(env)
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range plans {
			if filepath.Base(p.Path) == "fork-failed-build" {
				t.Errorf("%s: a registered tree was listed as an orphan", name)
			}
		}
		if len(plans) != 1 {
			t.Errorf("%s: want only the real orphan, got %d", name, len(plans))
		}
	}
	// And the same registration, with the app home given as the link.
	env := h.env([]Backend{{Tag: "failed", Path: bin}}, nil)
	env.AppHome = link
	plans, _ := PlanGC(env)
	if len(plans) != 1 {
		t.Errorf("app home as symlink: %d plans", len(plans))
	}
}
