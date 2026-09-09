// Package backends manages user-registered llama.cpp fork backends. New model
// architectures usually land in forks before mainline, so ggrun lets the user
// add one (`ggrun backend add <git-url>`) and optionally route a model
// architecture to it automatically. The manifest is shared by the CLI (backend
// selection/routing) and the TUI (backend picker).
package backends

import (
	"bytes"
	"crypto/sha256"
	_ "embed"
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
	// BaseTag and Features identify a composed backend. The base backend remains
	// registered and untouched; this record points at a separate checkout/build
	// containing the ordered, reviewed source-feature overlays.
	BaseTag  string   `json:"base_tag,omitempty"`
	Features []string `json:"features,omitempty"`
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
	BaseTag        string   `json:"base_tag,omitempty"`
	Features       []string `json:"features,omitempty"`
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
	Name        string `json:"name"`
	Description string `json:"description"`
	Tag         string `json:"tag"`
	GitURL      string `json:"git_url"`
	Branch      string `json:"branch"`
	Commit      string `json:"commit"`
	RouteArch   string `json:"route_arch"`
	HelperOnly  bool   `json:"helper_only,omitempty"`
	Accel       string `json:"accel,omitempty"`
	// BaseTag and Features are set for a composed recipe. They are persisted in
	// the backend manifest so update/rollback can reconstruct the same ordered
	// source transformation rather than rebuilding an unpatched base fork.
	BaseTag  string   `json:"base_tag,omitempty"`
	Features []string `json:"features,omitempty"`
	// RequiredFlags are the exact server options which make this recipe useful.
	// Architecture conformance alone cannot prove an optimization backend still
	// exposes its advertised capability after a source update.
	RequiredFlags []string      `json:"-"`
	Patches       []RecipePatch `json:"-"`
}

// Feature is a reviewed source overlay which can be composed onto a pinned
// llama.cpp-derived backend. A feature is not an architecture route by itself:
// the composed backend inherits its base's loader, route and exact commit.
type Feature struct {
	Name          string
	Description   string
	Accel         string
	RequiredFlags []string
	Patches       []RecipePatch
}

//go:embed patches/hy3/0001-fix-router-tensor-name.patch
var hy3RouterTensorNamePatch []byte

// The reviewed Nanbeige4.2 GGUF predates upstream's final metadata spelling:
// it uses loop_count plus a logical block count, while upstream b77d646 expects
// num_loops plus a physical block count. The loader supports both unambiguously.
//
//go:embed patches/nanbeige42/0001-accept-loop-count-gguf-schema.patch
var nanbeige42LoopCountPatch []byte

// common/speculative.cpp calls std::isfinite without including <cmath>. The
// declaration reaches poolside's compiler transitively and ours not at all, so
// the fork builds for them and fails here at 95% with "'isfinite' is not a
// member of 'std'". Adding the include is the whole fix and changes no
// behaviour.
//
//go:embed patches/laguna/0001-include-cmath-for-isfinite.patch
var lagunaCmathPatch []byte

// Hot experts is maintained upstream as a one-commit draft. Embedding the
// exact reviewed commit as a patch lets ggrun apply it to a compatible pinned
// architecture fork (for example glm5next) without replacing that fork with a
// different loader. The original author/commit metadata is retained inside the
// format-patch artifact.
//
//go:embed patches/features/hot-experts/0001-MoE-expert-cache-GPU-resident-LRU-cache-for-host-off.patch
var hotExpertsFeaturePatch []byte

// Aggregate counters are required by cache admission. Keep their bounded
// 512-step cadence visible at normal verbosity without per-token debug logs.
//
//go:embed patches/features/hot-experts/0002-bounded-runtime-telemetry.patch
var hotExpertsTelemetryPatch []byte

// Keep an expert unpublished until its one outstanding upload completes.
//
//go:embed patches/features/hot-experts/0003-deduplicate-inflight-experts.patch
var hotExpertsInflightPatch []byte

var builtinFeatures = []Feature{
	{
		Name:          "hot-experts",
		Description:   "GPU LRU cache for host-resident separate gate/up/down MoE experts",
		Accel:         "cuda",
		RequiredFlags: []string{"--moe-expert-cache", "--moe-expert-cache-inserts"},
		Patches: []RecipePatch{{
			Name:     "features/hot-experts/bccbacdb8945",
			contents: hotExpertsFeaturePatch,
		}, {
			Name:     "features/hot-experts/telemetry-info-v1",
			contents: hotExpertsTelemetryPatch,
		}, {
			Name:     "features/hot-experts/inflight-dedup-v1",
			contents: hotExpertsInflightPatch,
		}},
	},
}

var builtinRecipes = []Recipe{
	{
		// Experimental backend capability, deliberately not architecture-routed.
		// The optimizer may use it for any compatible separate-gate/up routed MoE,
		// but only after the user selects/installs this isolated backend and a live
		// cache-on candidate beats the cache-free baseline. Pin the exact reviewed
		// draft commit; moving a performance patch while retaining measurements
		// would make those measurements describe a different implementation.
		Name:          "hot-experts",
		Description:   "Experimental llama.cpp GPU LRU cache for host-resident routed experts",
		Tag:           "hot-experts",
		GitURL:        "https://github.com/csantiago78/llama.cpp.git",
		Branch:        "moe-expert-cache",
		Commit:        "bccbacdb8945680f1cfc7e6bffd1e59014705750",
		Patches:       []RecipePatch{{Name: "features/hot-experts/telemetry-info-v1", contents: hotExpertsTelemetryPatch}, {Name: "features/hot-experts/inflight-dedup-v1", contents: hotExpertsInflightPatch}},
		RouteArch:     "",
		Accel:         "cuda",
		RequiredFlags: []string{"--moe-expert-cache", "--moe-expert-cache-inserts"},
	},
	{
		// NanoBeige support is upstream llama.cpp, not a permanent fork. Pin the
		// first reviewed merge commit so `ggrun support install --with-backend`
		// can build a known architecture-capable helper backend even when the
		// machine's normal production backend intentionally stays older.
		Name:        "nanbeige42",
		Description: "Upstream llama.cpp with native Nanbeige4.2 looped-transformer support",
		Tag:         "nanbeige42",
		GitURL:      "https://github.com/ggml-org/llama.cpp.git",
		Branch:      "master",
		Commit:      "b77d646751d01c0962bc203b6809e9d94f7d50b7",
		RouteArch:   "nanbeige",
		HelperOnly:  true,
		Accel:       "",
		Patches: []RecipePatch{{
			Name:     "0001-accept-loop-count-gguf-schema.patch",
			contents: nanbeige42LoopCountPatch,
		}},
	},
	{
		Name:        "hy3",
		Description: "Tencent Hy3 / hy_v3 support with built-in MTP and its reviewed chat template",
		Tag:         "hy3",
		GitURL:      "https://github.com/noonr48/ik_llama-hy3.git",
		Branch:      "hy3-support",
		Commit:      "f46c95ee90d8c8200b0147c646b883405020b482",
		RouteArch:   "hy_v3",
		Accel:       "",
		Patches: []RecipePatch{{
			Name:     "hy3/0001-fix-router-tensor-name",
			contents: hy3RouterTensorNamePatch,
		}},
	},
	{
		Name:        "minimax-m3",
		Description: "Preliminary MiniMax-M3 support with native structured tool-call parsing",
		Tag:         "minimax-m3",
		GitURL:      "https://github.com/danielhanchen/llama.cpp.git",
		Branch:      "minimax-m3",
		Commit:      "66f43aa655a07999c7746fe9ff5ede94835e921e",
		RouteArch:   "minimax-m3",
		Accel:       "",
	},
	{
		// Poolside's own fork, not the upstream pull request.
		//
		// The upstream PR (joerowell/llama.cpp @ add-laguna) serves Laguna
		// correctly but implements only the target architecture. Its loader
		// builds 69 of the 76 tensors the published DFlash drafter contains, so
		// any speculative launch dies during model load:
		//
		//   done_getting_tensors: wrong number of tensors; expected 76, got 69
		//
		// Poolside state it directly: "Upstream llama.cpp ships the generic
		// DFlash framework but not the Laguna decoder contract this draft model
		// needs." Speculation is the largest measured lever on a RAM-resident
		// MoE -- a forward pass costs nearly the same carrying one token or a
		// micro-batch -- so a recipe that cannot load the drafter forfeits it.
		Name:        "laguna",
		Description: "Poolside Laguna support including the DFlash drafter (poolside fork, not the upstream PR)",
		Tag:         "laguna",
		GitURL:      "https://github.com/poolsideai/llama.cpp.git",
		Branch:      "laguna",
		Commit:      "04b2b72cb54048ead292884adbe11f284e3ec950",
		RouteArch:   "laguna",
		Accel:       "",
		Patches: []RecipePatch{{
			Name:     "laguna/0001-include-cmath-for-isfinite",
			contents: lagunaCmathPatch,
		}},
	},
	{
		// Kept selectable: it is the branch heading for upstream, so it is the
		// one to test against when the PR merges, and it remains a working
		// target-only fallback if the poolside fork regresses.
		Name:        "laguna-upstream-pr",
		Description: "Laguna target architecture from the open upstream llama.cpp PR (no DFlash drafter)",
		Tag:         "laguna-upstream-pr",
		GitURL:      "https://github.com/joerowell/llama.cpp.git",
		Branch:      "add-laguna",
		Commit:      "54f214a09b8c4e709357ae661a77925edb154f13",
		RouteArch:   "laguna",
		Accel:       "",
	},
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

// Features returns the reviewed source-feature catalog. The nested slices are
// copied so callers cannot mutate the process-wide definitions.
func Features() []Feature {
	out := make([]Feature, len(builtinFeatures))
	for i := range builtinFeatures {
		out[i] = builtinFeatures[i]
		out[i].RequiredFlags = append([]string(nil), builtinFeatures[i].RequiredFlags...)
		out[i].Patches = append([]RecipePatch(nil), builtinFeatures[i].Patches...)
	}
	return out
}

// FeatureByName resolves a reviewed source feature case-insensitively.
func FeatureByName(name string) *Feature {
	name = strings.ToLower(strings.TrimSpace(name))
	for _, feature := range builtinFeatures {
		if strings.ToLower(feature.Name) != name {
			continue
		}
		copy := feature
		copy.RequiredFlags = append([]string(nil), feature.RequiredFlags...)
		copy.Patches = append([]RecipePatch(nil), feature.Patches...)
		return &copy
	}
	return nil
}

func normalizedFeatureNames(names []string) ([]string, error) {
	out := make([]string, 0, len(names))
	seen := make(map[string]bool, len(names))
	for _, raw := range names {
		name := strings.ToLower(strings.TrimSpace(raw))
		if name == "" || seen[name] {
			continue
		}
		if FeatureByName(name) == nil {
			return nil, fmt.Errorf("unknown backend feature %q", raw)
		}
		seen[name] = true
		out = append(out, name)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("at least one backend feature is required")
	}
	return out, nil
}

// BackendHasFeature reports manifest-proven composition. Runtime capability is
// still checked from the binary's exact --help before the build is registered.
func BackendHasFeature(backend Backend, feature string) bool {
	feature = strings.ToLower(strings.TrimSpace(feature))
	for _, installed := range backend.Features {
		if strings.EqualFold(strings.TrimSpace(installed), feature) {
			return true
		}
	}
	return false
}

// CompositeTag is stable across launches and updates of the same base tag. A
// short digest prevents two unusually long base tags from collapsing onto the
// same truncated filesystem/manifest name.
func CompositeTag(baseTag string, featureNames ...string) string {
	parts := make([]string, 0, len(featureNames)+1)
	parts = append(parts, strings.ToLower(strings.TrimSpace(baseTag)))
	for _, feature := range featureNames {
		if feature = strings.ToLower(strings.TrimSpace(feature)); feature != "" {
			parts = append(parts, feature)
		}
	}
	raw := strings.Join(parts, "-")
	clean := strings.NewReplacer("/", "-", "\\", "-", " ", "-", "_", "-", "+", "-").Replace(raw)
	for strings.Contains(clean, "--") {
		clean = strings.ReplaceAll(clean, "--", "-")
	}
	clean = strings.Trim(clean, "-")
	if len(clean) <= 56 {
		return clean
	}
	sum := sha256.Sum256([]byte(raw))
	return strings.Trim(clean[:47], "-") + fmt.Sprintf("-%x", sum[:4])
}

// ComposeRecipe creates a separate, reproducible build recipe for a base fork
// plus ordered reviewed feature overlays. It never edits or replaces the base
// backend. Unknown custom patches fail closed because their bytes are not
// available to reproduce in the new checkout.
func ComposeRecipe(base Backend, featureNames ...string) (*Recipe, error) {
	if strings.TrimSpace(base.Tag) == "" {
		return nil, fmt.Errorf("base backend has no tag")
	}
	if strings.TrimSpace(base.GitURL) == "" || strings.TrimSpace(base.Commit) == "" {
		return nil, fmt.Errorf("backend %q has no reproducible source URL and commit", base.Tag)
	}
	names, err := normalizedFeatureNames(featureNames)
	if err != nil {
		return nil, err
	}
	for _, name := range names {
		if BackendHasFeature(base, name) {
			return nil, fmt.Errorf("backend %q already includes feature %q", base.Tag, name)
		}
	}

	var patches []RecipePatch
	var required []string
	helperOnly := true // composites are selected by feature policy, never global arch routing
	if baseRecipe := RecipeByName(base.Tag); baseRecipe != nil {
		patches = append(patches, baseRecipe.Patches...)
		required = append(required, baseRecipe.RequiredFlags...)
	} else if len(base.AppliedPatches) > 0 {
		return nil, fmt.Errorf("backend %q carries unregistered source patches and cannot be reproduced as a feature base", base.Tag)
	}

	accel := ""
	for _, name := range names {
		feature := FeatureByName(name)
		if feature == nil {
			return nil, fmt.Errorf("unknown backend feature %q", name)
		}
		if feature.Accel != "" {
			if accel != "" && !strings.EqualFold(accel, feature.Accel) {
				return nil, fmt.Errorf("backend features require conflicting accelerators %q and %q", accel, feature.Accel)
			}
			accel = feature.Accel
		}
		patches = append(patches, feature.Patches...)
		required = append(required, feature.RequiredFlags...)
	}

	tag := CompositeTag(base.Tag, names...)
	return &Recipe{
		Name:          tag,
		Description:   fmt.Sprintf("%s with %s", base.Tag, strings.Join(names, "+")),
		Tag:           tag,
		GitURL:        base.GitURL,
		Branch:        base.Branch,
		Commit:        base.Commit,
		RouteArch:     base.RouteArch,
		HelperOnly:    helperOnly,
		Accel:         accel,
		BaseTag:       base.Tag,
		Features:      names,
		RequiredFlags: required,
		Patches:       patches,
	}, nil
}

// RecipeForBackend returns the catalog recipe or reconstructs a composed
// backend recipe from manifest provenance. Updates therefore reapply the exact
// feature patch instead of silently rebuilding a cache-free base fork.
func RecipeForBackend(backend Backend) (*Recipe, error) {
	if len(backend.Features) == 0 {
		return RecipeByName(backend.Tag), nil
	}
	base := Backend{
		Tag:       backend.BaseTag,
		GitURL:    backend.GitURL,
		Branch:    backend.Branch,
		Commit:    backend.Commit,
		RouteArch: backend.RouteArch,
	}
	if strings.TrimSpace(base.Tag) == "" {
		return nil, fmt.Errorf("composed backend %q has no base tag", backend.Tag)
	}
	if installed := ByTag(base.Tag); installed != nil {
		base.AppliedPatches = append([]string(nil), installed.AppliedPatches...)
	}
	recipe, err := ComposeRecipe(base, backend.Features...)
	if err != nil {
		return nil, err
	}
	recipe.Tag = backend.Tag
	recipe.Name = backend.Tag
	if len(backend.AppliedPatches) > 0 && !sameStringSetOrdered(recipe.PatchNames(), backend.AppliedPatches) {
		return nil, fmt.Errorf("composed backend %q patch manifest differs from reviewed recipe", backend.Tag)
	}
	return recipe, nil
}

func sameStringSetOrdered(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
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
	a := strings.ToLower(strings.TrimSpace(arch))
	if strings.HasPrefix(a, "minimax-m") { // minimax-m2, minimax-m3, ...
		return "ik_llama"
	}
	return ""
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

// FeatureBackend returns the installed composite for one exact base tag and
// feature. It deliberately ignores architecture-only matches: two forks can
// load the same architecture while carrying different graph semantics, and a
// feature validated for one must never shadow the other.
func FeatureBackend(baseTag, feature string) *Backend {
	baseTag = strings.TrimSpace(baseTag)
	feature = strings.TrimSpace(feature)
	if baseTag == "" || feature == "" {
		return nil
	}
	for _, backend := range Load() {
		if !strings.EqualFold(strings.TrimSpace(backend.BaseTag), baseTag) || !BackendHasFeature(backend, feature) {
			continue
		}
		if _, err := os.Stat(backend.Path); err == nil {
			copy := backend
			return &copy
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
