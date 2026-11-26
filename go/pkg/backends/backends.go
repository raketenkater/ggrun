// Package backends manages user-registered llama.cpp fork backends. New model
// architectures usually land in forks before mainline, so ggrun lets the user
// add one (`ggrun backend add <git-url>`) and optionally route a model
// architecture to it automatically. The manifest is shared by the CLI (backend
// selection/routing) and the TUI (backend picker).
package backends

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// Backend is a registered fork backend.
type Backend struct {
	Tag       string `json:"tag"`                  // selection name (--backend <tag>)
	Path      string `json:"path"`                 // path to the built llama-server binary
	RouteArch string `json:"route_arch,omitempty"` // auto-select for models of this arch
	// HelperOnly retains architecture conformance metadata without globally
	// routing main-model launches onto a deliberately CPU-only support build.
	HelperOnly bool   `json:"helper_only,omitempty"`
	GitURL     string `json:"git_url,omitempty"`
	Branch     string `json:"branch,omitempty"`
	Commit     string `json:"commit,omitempty"`
	// AppliedPatches identifies reviewed source fixes applied while building the
	// backend. It makes a recipe-built binary auditable from backends.json.
	AppliedPatches []string `json:"applied_patches,omitempty"`
	// Previous is the build this backend replaced, kept so a bad update can be
	// undone. One level is deliberate: the value of a rollback is getting back
	// to the build that was working an hour ago, and each retained build costs a
	// full compiled tree on disk.
	Previous *BackendVersion `json:"previous,omitempty"`
}

// BackendVersion is a superseded build, retained only if its binary still
// exists. Recording a path whose tree had been rebuilt over would make rollback
// claim a recovery it cannot perform.
type BackendVersion struct {
	Path           string   `json:"path"`
	Commit         string   `json:"commit,omitempty"`
	AppliedPatches []string `json:"applied_patches,omitempty"`
	ReplacedAt     string   `json:"replaced_at,omitempty"`
}

// RecipePatch is a narrow, reviewed source correction applied to a pinned fork.
// Its contents are embedded in ggrun so an install is deterministic and does not
// depend on an unrecorded edit in the user's fork checkout.
type RecipePatch struct {
	Name     string
	contents []byte
}

// Recipe is a reviewed, reproducible fork integration. Recipes keep new-model
// support declarative: the CLI owns clone/build/register/routing once, while a
// model-specific entry supplies only source identity and architecture.
type Recipe struct {
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Tag         string        `json:"tag"`
	GitURL      string        `json:"git_url"`
	Branch      string        `json:"branch"`
	Commit      string        `json:"commit"`
	RouteArch   string        `json:"route_arch"`
	HelperOnly  bool          `json:"helper_only,omitempty"`
	Accel       string        `json:"accel,omitempty"`
	Patches     []RecipePatch `json:"-"`
}

// Recipes returns a copy of the reviewed built-in recipe catalog.
func Recipes() []Recipe {
	return append([]Recipe(nil), builtinRecipes...)
}

// RecipeByName resolves a recipe by name or tag, case-insensitively.
func RecipeByName(name string) *Recipe {
	name = strings.ToLower(strings.TrimSpace(name))
	for _, recipe := range builtinRecipes {
		if strings.ToLower(recipe.Name) == name || strings.ToLower(recipe.Tag) == name {
			copy := recipe
			return &copy
		}
	}
	return nil
}

// PatchNames returns stable IDs for the reviewed patches included by a recipe.
func (r Recipe) PatchNames() []string {
	names := make([]string, 0, len(r.Patches))
	for _, patch := range r.Patches {
		names = append(names, patch.Name)
	}
	return names
}

// ApplyPatches applies a recipe's reviewed source corrections. It is
// idempotent: a patch that is already present is left untouched, while an
// unexpected source mismatch fails loudly instead of silently building an
// unverified backend.
func (r Recipe) ApplyPatches(srcDir string) error {
	for _, patch := range r.Patches {
		if len(patch.contents) == 0 {
			return fmt.Errorf("recipe patch %q is empty", patch.Name)
		}
		if err := gitApply(srcDir, patch.contents, false, true); err == nil {
			if err := gitApply(srcDir, patch.contents, false, false); err != nil {
				return fmt.Errorf("apply recipe patch %q: %w", patch.Name, err)
			}
			continue
		}
		if err := gitApply(srcDir, patch.contents, true, true); err == nil {
			continue // exact patch already present
		}
		return fmt.Errorf("recipe patch %q does not apply to this checkout", patch.Name)
	}
	return nil
}

// RevertPatches removes only the exact reviewed patches before a pinned
// checkout is refreshed. Any other local edits remain and are rejected by the
// caller, preserving the user's work.
func (r Recipe) RevertPatches(srcDir string) error {
	for i := len(r.Patches) - 1; i >= 0; i-- {
		patch := r.Patches[i]
		if err := gitApply(srcDir, patch.contents, true, true); err != nil {
			return fmt.Errorf("local edits do not match managed recipe patch %q", patch.Name)
		}
		if err := gitApply(srcDir, patch.contents, true, false); err != nil {
			return fmt.Errorf("revert recipe patch %q: %w", patch.Name, err)
		}
	}
	return nil
}

func gitApply(srcDir string, patch []byte, reverse, check bool) error {
	args := []string{"apply"}
	if reverse {
		args = append(args, "--reverse")
	}
	if check {
		args = append(args, "--check")
	}
	cmd := exec.Command("git", args...)
	cmd.Dir = srcDir
	cmd.Stdin = bytes.NewReader(patch)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("git %s: %s", strings.Join(args, " "), strings.TrimSpace(string(out)))
	}
	return nil
}

// PointerFile records the app home for binaries that cannot derive it. The
// installer's default target is ~/.local/bin, whose parent ~/.local holds no
// ggrun state, so a second copy of ggrun installed there would otherwise see no
// models and no registered backends while the real install sat untouched
// elsewhere. Silently falling back to defaults is worse than any error: it
// looks like the install lost its configuration.
const PointerFile = "apphome"

// HasState reports whether dir is a real ggrun app home rather than a directory
// that merely happens to contain the binary.
func HasState(dir string) bool {
	if dir == "" {
		return false
	}
	for _, marker := range []string{
		filepath.Join(dir, ".config", "backends.json"),
		filepath.Join(dir, ".config", "config"),
		filepath.Join(dir, ".config", "ggrun", "config"),
	} {
		if st, err := os.Stat(marker); err == nil && !st.IsDir() {
			return true
		}
	}
	return false
}

// RequiredBackendForArch returns the generic backend family known to be
// mandatory for an architecture. An empty result means either family may be
// attempted. Keep this shared by CLI preflight and the TUI so an "auto"
// continue-once choice cannot select a backend the CLI will immediately reject.
func RequiredBackendForArch(arch string) string {
	return loadedCatalog.requiredBackend(arch)
}

// pointerPath is where the app home pointer lives, always under the user's own
// config directory so every binary can find it regardless of install location.
func pointerPath() string {
	home := os.Getenv("HOME")
	if home == "" {
		return ""
	}
	return filepath.Join(home, ".config", "ggrun", PointerFile)
}

// RecordAppHome saves the app home so any other ggrun binary resolves the same
// install. Best effort: failing to write it only costs discoverability.
func RecordAppHome(dir string) {
	p := pointerPath()
	if p == "" || dir == "" || dir == os.Getenv("HOME") || !HasState(dir) {
		return
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return
	}
	_ = os.WriteFile(p, []byte(dir+"\n"), 0o644)
}

// pointedAppHome returns the recorded app home if it still holds ggrun state.
func pointedAppHome() string {
	p := pointerPath()
	if p == "" {
		return ""
	}
	body, err := os.ReadFile(p)
	if err != nil {
		return ""
	}
	dir := strings.TrimSpace(string(body))
	if HasState(dir) {
		return dir
	}
	return ""
}

// maxDiscoveryDirs bounds the search so startup cost stays predictable on a
// home directory with many entries.
const maxDiscoveryDirs = 400

// AppHomeFromExe derives the app home from the running binary's location, or
// returns "" when that location says nothing about it.
//
// A direct binary at <app-home>/ggrun identifies that directory when it holds
// ggrun state. The parent of a .bin/bin directory is handled the same way.
// Requiring state is important: the installer's default target is
// ~/.local/bin, and treating a stateless ~/.local as an app home made a second
// copy of ggrun report no models and no registered backends while the real
// install sat untouched elsewhere.
//
// Split out from AppHome so it is testable: os.Executable() inside a test
// returns the test binary, which can never exercise this path.
func AppHomeFromExe(exe string) string {
	if exe == "" {
		return ""
	}
	exeDir := filepath.Dir(exe)
	if HasState(exeDir) {
		return exeDir
	}
	switch filepath.Base(exeDir) {
	case ".bin", "bin":
		if parent := filepath.Dir(exeDir); HasState(parent) {
			return parent
		}
	}
	return ""
}

// DiscoverAppHome looks for a ggrun install tree when nothing else identifies
// one. It checks the user's home directory and one level below it, which covers
// both ~/ggrun and the common ~/<project>/ggrun layout, and stops at the first
// directory holding real ggrun state.
//
// This is what keeps a fresh shell, a second install, or a machine that lost an
// exported LLM_APP_HOME from presenting an empty ggrun with no models and no
// registered backends.
func DiscoverAppHome() string {
	home := os.Getenv("HOME")
	if home == "" {
		return ""
	}
	// Named layouts first: cheap, and correct in the overwhelming majority of
	// installs.
	for _, name := range []string{"ggrun", ".ggrun", filepath.Join(".local", "share", "ggrun")} {
		if dir := filepath.Join(home, name); HasState(dir) {
			return dir
		}
	}
	// Then the home directory, then the usual places an install can be moved
	// to, including mounted volumes. The budget is shared across every root so
	// total work stays bounded no matter how many roots exist.
	budget := maxDiscoveryDirs
	roots := append([]string{home}, systemSearchRoots()...)
	for _, root := range roots {
		if dir := scanForState(root, 2, &budget); dir != "" {
			return dir
		}
	}
	return ""
}

// systemSearchRoots are the non-home locations an install may have been moved
// to. Deliberately a short list rather than a whole-filesystem walk: an
// unbounded scan would be slow on every startup and could wander into network
// or removable mounts.
func systemSearchRoots() []string {
	return []string{"/opt", "/srv", "/mnt", "/media", "/usr/local/share"}
}

// scanForState looks for a ggrun app home under root, up to depth levels deep,
// consuming from a shared budget.
func scanForState(root string, depth int, budget *int) string {
	if depth <= 0 || *budget <= 0 {
		return ""
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return ""
	}
	var subdirs []string
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		// Skip dot directories: ggrun's own state lives in a .config inside a
		// normal directory, never in a hidden top-level one, and descending
		// into caches and VCS metadata wastes the budget.
		if strings.HasPrefix(name, ".") {
			continue
		}
		if *budget--; *budget <= 0 {
			return ""
		}
		dir := filepath.Join(root, name)
		if HasState(dir) {
			return dir
		}
		subdirs = append(subdirs, dir)
	}
	for _, dir := range subdirs {
		if found := scanForState(dir, depth-1, budget); found != "" {
			return found
		}
	}
	return ""
}

// AppHome resolves the ggrun app home (holds .bin, .src, .config).
//
// Order: LLM_APP_HOME, then the parent of the .bin/bin directory the running
// binary lives in but only when that parent actually holds ggrun state, then
// the recorded pointer, then a bounded search of the home directory.
//
// A binary sitting in a plain install directory such as ~/.local/bin -- the
// installer's own default -- does not get to claim an app home. Before this
// ordering it did, which made a second copy of ggrun report no models and no
// backends while the real install sat untouched, indistinguishable from a lost
// configuration.
func AppHome() string {
	if h := os.Getenv("LLM_APP_HOME"); h != "" {
		return h
	}
	if exe, err := os.Executable(); err == nil {
		if home := AppHomeFromExe(exe); home != "" {
			return home
		}
	}
	if p := pointedAppHome(); p != "" {
		return p
	}
	if d := DiscoverAppHome(); d != "" {
		// Remember it so the next run skips the search entirely.
		RecordAppHome(d)
		return d
	}
	if h := os.Getenv("HOME"); h != "" {
		return h
	}
	return "."
}

// ManifestPath is the on-disk location of the backends manifest.
func ManifestPath() string {
	return filepath.Join(AppHome(), ".config", "backends.json")
}

// Load returns the registered fork backends (empty if none/unreadable).
func Load() []Backend {
	list, err := loadManifest(ManifestPath())
	if err != nil {
		return nil
	}
	for i := range list {
		list[i] = ApplyBuiltinPolicy(list[i])
	}
	return list
}

// ApplyBuiltinPolicy makes safety attributes in the reviewed catalog
// authoritative over legacy manifests. Older backends.json files predate
// helper_only; interpreting an omitted false literally would route a
// deliberately isolated helper backend as a generic main-model backend.
func ApplyBuiltinPolicy(backend Backend) Backend {
	if recipe := RecipeByName(backend.Tag); recipe != nil && recipe.HelperOnly {
		backend.HelperOnly = true
	}
	return backend
}

func IsHelperOnly(backend Backend) bool {
	return ApplyBuiltinPolicy(backend).HelperOnly
}

// Save writes the manifest.
func Save(list []Backend) error {
	p := ManifestPath()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	release, err := acquireManifestLock(p + ".lock")
	if err != nil {
		return err
	}
	defer release()
	return saveManifestLocked(p, list)
}

// ByTag returns the registered backend with this tag (case-insensitive), or nil.
func ByTag(tag string) *Backend {
	tag = strings.TrimSpace(strings.ToLower(tag))
	if tag == "" {
		return nil
	}
	list := Load()
	for i := range list {
		if strings.ToLower(list[i].Tag) == tag {
			return &list[i]
		}
	}
	return nil
}

// ForArch returns a registered backend that routes this model architecture,
// if its binary exists on disk. Case-insensitive on arch.
func ForArch(arch string) *Backend {
	arch = strings.TrimSpace(strings.ToLower(arch))
	if arch == "" {
		return nil
	}
	list := Load()
	for i := range list {
		if IsHelperOnly(list[i]) {
			continue
		}
		if strings.ToLower(list[i].RouteArch) != arch {
			continue
		}
		if _, err := os.Stat(list[i].Path); err == nil {
			return &list[i]
		}
	}
	return nil
}

// Upsert adds or replaces a backend by tag.
func Upsert(be Backend) error {
	p := ManifestPath()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	release, err := acquireManifestLock(p + ".lock")
	if err != nil {
		return err
	}
	defer release()
	list, err := loadManifestForMutation(p)
	if err != nil {
		return err
	}
	for i := range list {
		if strings.EqualFold(list[i].Tag, be.Tag) {
			list[i] = be
			return saveManifestLocked(p, list)
		}
	}
	return saveManifestLocked(p, append(list, be))
}

// Remove drops a backend by tag; returns false if not found.
func Remove(tag string) (bool, error) {
	p := ManifestPath()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return false, err
	}
	release, err := acquireManifestLock(p + ".lock")
	if err != nil {
		return false, err
	}
	defer release()
	list, err := loadManifestForMutation(p)
	if err != nil {
		return false, err
	}
	out := list[:0:0]
	found := false
	for _, b := range list {
		if strings.EqualFold(b.Tag, tag) {
			found = true
			continue
		}
		out = append(out, b)
	}
	if !found {
		return false, nil
	}
	return found, saveManifestLocked(p, out)
}

func loadManifest(path string) ([]Backend, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var list []Backend
	if err := json.Unmarshal(data, &list); err != nil {
		return nil, fmt.Errorf("decode backend manifest: %w", err)
	}
	return list, nil
}

func loadManifestForMutation(path string) ([]Backend, error) {
	list, err := loadManifest(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	return list, err
}

func saveManifestLocked(path string, list []Backend) error {
	data, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".backends-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o644); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	if dirHandle, err := os.Open(dir); err == nil {
		err = dirHandle.Sync()
		_ = dirHandle.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

func acquireManifestLock(path string) (func(), error) {
	deadline := time.Now().Add(5 * time.Second)
	for {
		file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err == nil {
			_, _ = fmt.Fprintf(file, "%d\n", os.Getpid())
			_ = file.Sync()
			_ = file.Close()
			return func() { _ = os.Remove(path) }, nil
		}
		if !os.IsExist(err) {
			return nil, err
		}
		if info, statErr := os.Stat(path); statErr == nil && time.Since(info.ModTime()) > 2*time.Minute {
			_ = os.Remove(path)
			continue
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("timed out waiting for backend manifest lock")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// Tags returns the registered backend tags (for pickers).
func Tags() []string {
	list := Load()
	tags := make([]string, 0, len(list))
	for _, b := range list {
		tags = append(tags, b.Tag)
	}
	return tags
}
