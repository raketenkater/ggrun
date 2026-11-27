package backends

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

// Cleanup of backend source and build trees. Everything here is about files
// under <app home>/.src: it never selects or launches a backend. Deleting a
// tree is the one irreversible thing ggrun does to a user's disk on its own
// behalf, so every plan is checked twice (when planned and again right before
// removal) and each refusal has its own error value.

var (
	// ErrNoTree: the backend's binary is not inside the source root, so there is
	// no tree ggrun owns to delete.
	ErrNoTree = errors.New("backend has no source tree under the app home's .src")
	// ErrOutsideRoot: the path is not strictly inside the source root.
	ErrOutsideRoot = errors.New("path is outside the backend source root")
	// ErrSymlink: the path, or a directory above it inside the root, is a symlink.
	ErrSymlink = errors.New("path goes through a symlink")
	// ErrNotForkTree: only isolated .src/fork-* checkouts are ever deleted; the
	// mainline, ik_llama and ggrun checkouts belong to the updater.
	ErrNotForkTree = errors.New("not an isolated .src/fork-* source tree")
	// ErrSharedTree: another registration lives in the same tree.
	ErrSharedTree = errors.New("tree is shared with another registered backend")
	// ErrRunning: a process runs from the tree.
	ErrRunning = errors.New("a process is running from the tree")
	// ErrNoProcessCheck: running processes cannot be listed here, so nothing can
	// be proven idle.
	ErrNoProcessCheck = errors.New("cannot check for running processes")
	// ErrRecentlyModified: files in the tree changed lately, so a build may be
	// writing to it.
	ErrRecentlyModified = errors.New("tree was modified recently; a build may be running")
	// ErrSourceRootSymlink: .src is a symlink, so the trees belong to another
	// app home whose registrations this manifest does not know.
	ErrSourceRootSymlink = errors.New("not deletable: .src is a symlink to another location; run cleanup from the app home that owns it")
	// ErrLinkedWorktrees: git worktrees elsewhere keep their objects in the tree.
	ErrLinkedWorktrees = errors.New("tree is the main checkout of linked git worktrees")
	// ErrLocalWork: the tree may hold work that exists nowhere else.
	ErrLocalWork = errors.New("tree may hold local work that exists nowhere else; pass --yes --discard-local-work to delete it anyway")
	// ErrManifestMissing: there is no backends.json, so registrations are unknown.
	ErrManifestMissing = errors.New("backend manifest does not exist")
)

// defaultQuietWindow is how long a tree must be untouched before gc offers it.
const defaultQuietWindow = 30 * time.Minute

// ProcInfo is one running process as far as cleanup cares: where its program
// file and its working directory are.
type ProcInfo struct {
	PID int
	Exe string
	Cwd string
}

// CleanupEnv is everything cleanup reads from the outside, so tests can run it
// against a temporary app home with a simulated process table.
type CleanupEnv struct {
	AppHome  string
	Backends []Backend
	// Processes lists running processes. Nil means this platform has no
	// implementation; deleting then needs ForceNotRunning.
	Processes func() ([]ProcInfo, error)
	// ForceNotRunning (--force-not-running) lets deletion proceed when the
	// process table is unavailable. It never overrides a process that is seen.
	ForceNotRunning bool
	// DiscardLocalWork (--yes --discard-local-work) allows deleting a tree whose
	// LocalWork is non-empty (detected or unknown).
	DiscardLocalWork bool
	// ManifestErr is set when backends.json exists but could not be read. Cleanup
	// then plans nothing: an unreadable manifest would make every registered
	// tree look unreferenced.
	ManifestErr error
	Now         time.Time
	// QuietWindow overrides defaultQuietWindow for gc.
	QuietWindow time.Duration
}

// TreePlan is one directory cleanup would delete.
type TreePlan struct {
	Path   string    // resolved absolute path
	Kind   string    // "source tree", "orphan source tree" or "stale build directory"
	Bytes  int64     // regular-file bytes inside
	Files  int       // regular files inside
	Newest time.Time // newest modification time inside
	Links  []string  // .bin symlinks that point into the tree and go with it
	Tags   []string  // registrations that point into the tree
	// LocalWork is what might exist only in this tree ("" when none was found
	// and everything could be read); see CleanupEnv.LocalWork. Whole trees only.
	LocalWork string
	// Blocked is why gc will not delete it ("" when it may).
	Blocked string
}

// SourceRoot is the only directory cleanup deletes under.
func SourceRoot(appHome string) string { return filepath.Join(appHome, ".src") }

func (env CleanupEnv) now() time.Time {
	if env.Now.IsZero() {
		return time.Now()
	}
	return env.Now
}

// caseFold reports platforms whose file systems are normally case-insensitive.
func caseFold() bool { return runtime.GOOS == "windows" || runtime.GOOS == "darwin" }

func samePath(a, b string) bool {
	if caseFold() {
		return strings.EqualFold(a, b)
	}
	return a == b
}

// relUnder returns p relative to root and whether p is strictly inside it.
func relUnder(root, p string) (string, bool) {
	rel, err := filepath.Rel(root, p)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return "", false
	}
	return rel, true
}

// pathWithin reports whether p equals dir or lies inside it.
func pathWithin(dir, p string) bool {
	if samePath(dir, p) {
		return true
	}
	if caseFold() {
		dir, p = strings.ToLower(dir), strings.ToLower(p)
	}
	return strings.HasPrefix(p, dir+string(filepath.Separator))
}

// rootRel returns p relative to the source root when p lies inside it. Only
// names are compared, so it works for paths that no longer exist; when the two
// are spelled through different symlinks it retries with both resolved.
func rootRel(srcRoot, p string) (string, bool) {
	if strings.TrimSpace(p) == "" {
		return "", false
	}
	absRoot, err := filepath.Abs(srcRoot)
	if err != nil {
		return "", false
	}
	absP, err := filepath.Abs(p)
	if err != nil {
		return "", false
	}
	if rel, ok := relUnder(absRoot, absP); ok {
		return rel, true
	}
	realRoot, err := filepath.EvalSymlinks(absRoot)
	if err != nil {
		return "", false
	}
	// The resolved spellings, without needing the file to exist: a registration
	// whose binary is gone still refers to its tree.
	if rel, ok := relUnder(realRoot, absP); ok {
		return rel, true
	}
	return relUnder(realRoot, evalLoose(absP))
}

// evalLoose resolves symlinks in the longest existing prefix of p and appends
// the rest unchanged.
func evalLoose(p string) string {
	tail := ""
	for cur := p; ; {
		if real, err := filepath.EvalSymlinks(cur); err == nil {
			return filepath.Join(real, tail)
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			return p
		}
		tail = filepath.Join(filepath.Base(cur), tail)
		cur = parent
	}
}

// treeOf returns the first-level directory of the source root that holds p, or
// "" when p is not inside the root.
func treeOf(srcRoot, p string) string {
	rel, ok := rootRel(srcRoot, p)
	if !ok {
		return ""
	}
	absRoot, _ := filepath.Abs(srcRoot)
	return filepath.Join(absRoot, strings.SplitN(rel, string(filepath.Separator), 2)[0])
}

// resolveUnderRoot proves p is an existing directory strictly inside root with
// no symlink on the way, and returns its resolved path.
func resolveUnderRoot(root, p string) (string, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	absP, err := filepath.Abs(p)
	if err != nil {
		return "", err
	}
	rel, ok := relUnder(absRoot, absP)
	if !ok {
		return "", fmt.Errorf("%w: %s is not inside %s", ErrOutsideRoot, absP, absRoot)
	}
	realRoot, err := filepath.EvalSymlinks(absRoot)
	if err != nil {
		return "", fmt.Errorf("source root %s: %w", absRoot, err)
	}
	info, err := os.Lstat(absP)
	if err != nil {
		return "", err
	}
	realP, err := filepath.EvalSymlinks(absP)
	if err != nil {
		return "", err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		if !pathWithin(realRoot, realP) {
			return "", fmt.Errorf("%w: %s is a symlink to %s (%w)", ErrSymlink, absP, realP, ErrOutsideRoot)
		}
		return "", fmt.Errorf("%w: %s is a symlink to %s", ErrSymlink, absP, realP)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("%s is not a directory", absP)
	}
	if !samePath(realP, filepath.Join(realRoot, rel)) {
		if !pathWithin(realRoot, realP) {
			return "", fmt.Errorf("%w: %s resolves to %s (%w)", ErrSymlink, absP, realP, ErrOutsideRoot)
		}
		return "", fmt.Errorf("%w: %s resolves to %s", ErrSymlink, absP, realP)
	}
	return realP, nil
}

// measureTree sums regular files and finds the newest modification time. It
// never follows symlinks.
func measureTree(dir string) (bytes int64, files int, newest time.Time) {
	_ = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		if info.ModTime().After(newest) {
			newest = info.ModTime()
		}
		if info.Mode().IsRegular() {
			bytes += info.Size()
			files++
		}
		return nil
	})
	return bytes, files, newest
}

// linksInto lists symlinks directly inside <app home>/.bin whose target lies in
// tree (the links `backend add` makes beside the other backend binaries). A
// target spelled through the unresolved app home counts too.
func linksInto(appHome, tree string) []string {
	dirs := []string{tree}
	srcRoot := SourceRoot(appHome)
	if absRoot, err := filepath.Abs(srcRoot); err == nil {
		if realRoot, err := filepath.EvalSymlinks(absRoot); err == nil {
			if rel, ok := relUnder(realRoot, tree); ok {
				dirs = append(dirs, filepath.Join(absRoot, rel))
			}
		}
	}
	binDir := filepath.Join(appHome, ".bin")
	entries, err := os.ReadDir(binDir)
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range entries {
		if e.Type()&os.ModeSymlink == 0 {
			continue
		}
		link := filepath.Join(binDir, e.Name())
		target, err := os.Readlink(link)
		if err != nil {
			continue
		}
		if !filepath.IsAbs(target) {
			target = filepath.Join(binDir, target)
		}
		target = filepath.Clean(target)
		for _, d := range dirs {
			if pathWithin(d, target) {
				out = append(out, link)
				break
			}
		}
	}
	return out
}

// registeredTrees maps each first-level source directory to the registrations
// whose binary or retained previous build lives in it.
func registeredTrees(env CleanupEnv) map[string][]string {
	srcRoot := SourceRoot(env.AppHome)
	out := map[string][]string{}
	add := func(tree, tag string) {
		if tree == "" {
			return
		}
		for _, t := range out[tree] {
			if t == tag {
				return
			}
		}
		out[tree] = append(out[tree], tag)
	}
	for _, b := range env.Backends {
		add(treeOf(srcRoot, b.Path), b.Tag)
		if b.Previous != nil {
			add(treeOf(srcRoot, b.Previous.Path), b.Tag)
		}
	}
	return out
}

// treeKey finds the map key equal to tree under this platform's path rules.
func treeKey(m map[string][]string, tree string) []string {
	for k, v := range m {
		if samePath(k, tree) {
			return v
		}
	}
	return nil
}

// runningFrom lists processes whose program file or working directory is inside
// dir. It fails closed: when the process table cannot be read, deleting needs
// ForceNotRunning. That flag only stands in for a missing or failing table; a
// process the table does show in the tree always blocks.
func (env CleanupEnv) runningFrom(dir string) ([]ProcInfo, error) {
	var procs []ProcInfo
	var err error
	if env.Processes == nil {
		err = fmt.Errorf("%w on %s; pass --force-not-running once you have checked yourself", ErrNoProcessCheck, runtime.GOOS)
	} else if procs, err = env.Processes(); err != nil {
		err = fmt.Errorf("%w (%v); pass --force-not-running once you have checked yourself", ErrNoProcessCheck, err)
	}
	if err != nil {
		if env.ForceNotRunning {
			return nil, nil
		}
		return nil, err
	}
	var hit []ProcInfo
	for _, p := range procs {
		if (p.Exe != "" && pathWithin(dir, filepath.Clean(p.Exe))) || (p.Cwd != "" && pathWithin(dir, filepath.Clean(p.Cwd))) {
			hit = append(hit, p)
		}
	}
	return hit, nil
}

func describeProcs(procs []ProcInfo) string {
	parts := make([]string, 0, len(procs))
	for i, p := range procs {
		if i == 3 {
			parts = append(parts, fmt.Sprintf("and %d more", len(procs)-3))
			break
		}
		what := p.Exe
		if what == "" {
			what = p.Cwd
		}
		parts = append(parts, fmt.Sprintf("pid %d %s", p.PID, what))
	}
	return strings.Join(parts, ", ")
}

// realSourceRoot resolves <app home>/.src and refuses when .src itself is a
// symlink: its trees then belong to another app home.
func (env CleanupEnv) realSourceRoot() (string, error) {
	absHome, err := filepath.Abs(env.AppHome)
	if err != nil {
		return "", err
	}
	realHome, err := filepath.EvalSymlinks(absHome)
	if err != nil {
		return "", fmt.Errorf("app home %s: %w", absHome, err)
	}
	absRoot := filepath.Join(absHome, ".src")
	realRoot, err := filepath.EvalSymlinks(absRoot)
	if err != nil {
		return "", fmt.Errorf("source root %s: %w", absRoot, err)
	}
	if !samePath(realRoot, filepath.Join(realHome, ".src")) {
		return "", fmt.Errorf("%w (%s -> %s)", ErrSourceRootSymlink, absRoot, realRoot)
	}
	return realRoot, nil
}

func (env CleanupEnv) window() time.Duration {
	if env.QuietWindow > 0 {
		return env.QuietWindow
	}
	return defaultQuietWindow
}

// quietErr refuses a tree whose newest file changed within the quiet window.
func (env CleanupEnv) quietErr(newest time.Time) error {
	if newest.IsZero() {
		return nil
	}
	if age := env.now().Sub(newest); age < env.window() {
		return fmt.Errorf("%w (changed %s ago)", ErrRecentlyModified, age.Round(time.Minute))
	}
	return nil
}

// checkPlan applies every path and process rule to a plan, when planning and
// again immediately before deleting. It returns the resolved path it
// validated: that, and nothing else, is what may be removed.
func (env CleanupEnv) checkPlan(p TreePlan) (string, error) {
	realRoot, err := env.realSourceRoot()
	if err != nil {
		return "", err
	}
	real, err := resolveUnderRoot(realRoot, p.Path)
	if err != nil {
		return "", err
	}
	top := treeOf(realRoot, real)
	if top == "" || !IsIsolatedForkSourceDir(top) {
		return "", fmt.Errorf("%w: %s", ErrNotForkTree, real)
	}
	if samePath(real, top) {
		if err := checkWorktrees(real); err != nil {
			return "", err
		}
	}
	procs, err := env.runningFrom(real)
	if err != nil {
		return "", err
	}
	if len(procs) > 0 {
		return "", fmt.Errorf("%w %s: %s", ErrRunning, real, describeProcs(procs))
	}
	return real, nil
}

// wholeTree reports whether real is a first-level fork tree (not a build
// directory inside one).
func (env CleanupEnv) wholeTree(real string) bool {
	realRoot, err := env.realSourceRoot()
	return err == nil && samePath(real, treeOf(realRoot, real))
}

// PlanPurge plans removing one backend's source tree (the build directories are
// inside it). It returns a refusal as an error and deletes nothing.
func PlanPurge(env CleanupEnv, tag string) (TreePlan, error) {
	if env.ManifestErr != nil {
		return TreePlan{}, fmt.Errorf("backend manifest unavailable, so nothing is planned: %w", env.ManifestErr)
	}
	var be *Backend
	for i := range env.Backends {
		if strings.EqualFold(env.Backends[i].Tag, tag) {
			be = &env.Backends[i]
			break
		}
	}
	if be == nil {
		return TreePlan{}, fmt.Errorf("no backend tagged %q", tag)
	}
	srcRoot := SourceRoot(env.AppHome)
	tree := treeOf(srcRoot, be.Path)
	if tree == "" {
		return TreePlan{}, fmt.Errorf("%w: %s", ErrNoTree, be.Path)
	}
	real, err := resolveUnderRoot(srcRoot, tree)
	if err != nil {
		return TreePlan{}, err
	}
	var shared []string
	for _, t := range treeKey(registeredTrees(env), tree) {
		if !strings.EqualFold(t, be.Tag) {
			shared = append(shared, t)
		}
	}
	if len(shared) > 0 {
		sort.Strings(shared)
		return TreePlan{}, fmt.Errorf("%w: %s also serves %s", ErrSharedTree, filepath.Base(tree), strings.Join(shared, ", "))
	}
	plan := TreePlan{Path: real, Kind: "source tree", Tags: []string{be.Tag}}
	if _, err := env.checkPlan(plan); err != nil {
		return TreePlan{}, err
	}
	plan.Bytes, plan.Files, plan.Newest = measureTree(real)
	if err := env.quietErr(plan.Newest); err != nil {
		return TreePlan{}, err
	}
	plan.Links = linksInto(env.AppHome, real)
	plan.LocalWork = env.LocalWork(real)
	return plan, nil
}

// staleBuildMarkers are the directory names ggrun's own update and rollback
// code leaves behind. Other build-* directories are never offered.
var staleBuildMarkers = []string{"-prev-", "-candidate-", "-unregistered-", "-rollback-swap-"}

func isStaleBuildName(name string) bool {
	if !strings.HasPrefix(name, "build-") {
		return false
	}
	for _, m := range staleBuildMarkers {
		if strings.Contains(name, m) {
			return true
		}
	}
	return false
}

// PlanGC lists what no registration references: whole .src/fork-* trees, and
// inside referenced trees the leftover build directories of failed or replaced
// updates that no registration (including its retained previous build) uses.
// Entries that cannot be deleted right now carry Blocked.
func PlanGC(env CleanupEnv) ([]TreePlan, error) {
	if env.ManifestErr != nil {
		return nil, fmt.Errorf("backend manifest unavailable, so nothing can be called unreferenced: %w", env.ManifestErr)
	}
	srcRoot := SourceRoot(env.AppHome)
	entries, err := os.ReadDir(srcRoot)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	absRoot, _ := filepath.Abs(srcRoot)
	trees := registeredTrees(env)
	usedBuilds := map[string]bool{}
	for _, b := range env.Backends {
		paths := []string{b.Path}
		if b.Previous != nil {
			paths = append(paths, b.Previous.Path)
		}
		for _, p := range paths {
			if rel, ok := rootRel(srcRoot, p); ok {
				parts := strings.Split(rel, string(filepath.Separator))
				if len(parts) >= 2 {
					usedBuilds[strings.ToLower(parts[0]+"/"+parts[1])] = true
				}
			}
		}
	}
	finish := func(p TreePlan) TreePlan {
		p.Bytes, p.Files, p.Newest = measureTree(p.Path)
		p.Links = linksInto(env.AppHome, p.Path)
		_, err := env.checkPlan(p)
		if err == nil {
			err = env.quietErr(p.Newest)
		}
		if err == nil && env.wholeTree(p.Path) {
			if p.LocalWork = env.LocalWork(p.Path); p.LocalWork != "" && !env.DiscardLocalWork {
				err = fmt.Errorf("local work: %s; delete only with --yes --discard-local-work", p.LocalWork)
			}
		}
		if err == nil && p.Kind == "orphan source tree" && len(p.Links) > 0 {
			err = fmt.Errorf("referenced from .bin by %s; remove the link if it is unused", strings.Join(p.Links, ", "))
		}
		if err != nil {
			p.Blocked = err.Error()
		}
		return p
	}
	var out []TreePlan
	for _, e := range entries {
		if !strings.HasPrefix(e.Name(), "fork-") {
			continue
		}
		dir := filepath.Join(absRoot, e.Name())
		if e.Type()&os.ModeSymlink != 0 || !e.IsDir() {
			if e.Type()&os.ModeSymlink != 0 {
				out = append(out, TreePlan{Path: dir, Kind: "orphan source tree", Blocked: ErrSymlink.Error() + ": left alone"})
			}
			continue
		}
		real, err := resolveUnderRoot(srcRoot, dir)
		if err != nil {
			out = append(out, TreePlan{Path: dir, Kind: "orphan source tree", Blocked: err.Error()})
			continue
		}
		tags := treeKey(trees, dir)
		if len(tags) == 0 {
			out = append(out, finish(TreePlan{Path: real, Kind: "orphan source tree"}))
			continue
		}
		subs, err := os.ReadDir(real)
		if err != nil {
			continue
		}
		for _, s := range subs {
			if !s.IsDir() || s.Type()&os.ModeSymlink != 0 || !isStaleBuildName(s.Name()) {
				continue
			}
			if usedBuilds[strings.ToLower(e.Name()+"/"+s.Name())] {
				continue
			}
			out = append(out, finish(TreePlan{Path: filepath.Join(real, s.Name()), Kind: "stale build directory", Tags: tags}))
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, nil
}

// Delete removes one planned directory and the .bin links that point into it.
// It re-checks every rule first and removes the resolved path it checked, never
// the plan's own string: a plan whose path is not absolute and clean, or does
// not resolve to itself, is refused. RemoveAll unlinks a symlink inside the
// tree rather than following it. It returns the paths removed.
func (env CleanupEnv) Delete(p TreePlan) ([]string, error) {
	if !filepath.IsAbs(p.Path) || p.Path != filepath.Clean(p.Path) {
		return nil, fmt.Errorf("%w: plan path %q is not absolute and clean", ErrOutsideRoot, p.Path)
	}
	real, err := env.checkPlan(p)
	if err != nil {
		return nil, err
	}
	if !samePath(real, p.Path) {
		return nil, fmt.Errorf("%w: plan path %s resolves to %s", ErrSymlink, p.Path, real)
	}
	_, _, newest := measureTree(real)
	if err := env.quietErr(newest); err != nil {
		return nil, err
	}
	if env.wholeTree(real) && !env.DiscardLocalWork {
		if lw := env.LocalWork(real); lw != "" {
			return nil, fmt.Errorf("%w: %s", ErrLocalWork, lw)
		}
	}
	var removed []string
	if err := os.RemoveAll(real); err != nil {
		return removed, fmt.Errorf("remove %s: %w", real, err)
	}
	removed = append(removed, real)
	for _, l := range linksInto(env.AppHome, real) {
		if info, err := os.Lstat(l); err == nil && info.Mode()&os.ModeSymlink != 0 {
			if err := os.Remove(l); err == nil {
				removed = append(removed, l)
			}
		}
	}
	return removed, nil
}

// PlatformProcessLister returns the running-process lister for this platform,
// or nil where none is implemented.
func PlatformProcessLister() func() ([]ProcInfo, error) { return platformProcessLister() }
