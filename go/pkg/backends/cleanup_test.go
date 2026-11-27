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
	return h.write(".src/"+name+"/build-cuda/bin/llama-server", "binary")
}

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

	list := []Backend{
		{Tag: "a", Path: binA},
		{Tag: "b", Path: filepath.Join(h.Home, ".src", "fork-b", "build-cuda", "bin", "llama-server")},
	}
	env := h.env(list, []ProcInfo{{PID: 1, Exe: filepath.Join(h.Home, ".src", "fork-b", "build-cuda", "bin", "llama-server")}})
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
	h.write(".bin/orphan-server-cuda", "")
	_ = os.Remove(filepath.Join(h.Home, ".bin", "orphan-server-cuda"))
	h.symlink(filepath.Join(h.Home, ".src", "fork-orphan", "build-cuda", "bin", "llama-server"), ".bin/orphan-server-cuda")

	for _, rel := range []string{".src/fork-live", ".src/fork-orphan", ".src/fork-busy", ".src/llama.cpp", ".src/ik_llama.cpp-v1-legacy"} {
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
	if p := got[rel(".src", "fork-orphan")]; len(p.Links) != 1 || p.Bytes == 0 {
		t.Errorf("orphan plan lacks size or link: %+v", p)
	}

	for _, p := range plans {
		if p.Blocked != "" {
			// The CLI never deletes a blocked entry; Delete itself still
			// refuses the ones blocked by a safety rule (not by recency).
			if !strings.Contains(p.Blocked, ErrRecentlyModified.Error()) {
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
	for _, gone := range []string{".src/fork-orphan", ".bin/orphan-server-cuda", ".src/fork-live/build-cuda-prev-old"} {
		if h.exists(gone) {
			t.Errorf("%s should be gone", gone)
		}
	}
	for _, kept := range []string{
		".src/fork-live/build-cuda/bin/llama-server", ".src/fork-live/build-cuda-prev-keep/bin/llama-server",
		".src/fork-live/build-vulkan/bin/llama-server", ".src/fork-busy/src/main.cpp", ".src/llama.cpp/src/main.cpp",
		".src/ik_llama.cpp-v1-legacy/src/main.cpp", ".src", ".src/ggrun", ".src/fork-linked",
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
