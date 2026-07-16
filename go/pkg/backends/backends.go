// Package backends manages user-registered llama.cpp fork backends. New model
// architectures usually land in forks before mainline, so ggrun lets the user
// add one (`ggrun backend add <git-url>`) and optionally route a model
// architecture to it automatically. The manifest is shared by the CLI (backend
// selection/routing) and the TUI (backend picker).
package backends

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Backend is a registered fork backend.
type Backend struct {
	Tag       string `json:"tag"`                  // selection name (--backend <tag>)
	Path      string `json:"path"`                 // path to the built llama-server binary
	RouteArch string `json:"route_arch,omitempty"` // auto-select for models of this arch
	GitURL    string `json:"git_url,omitempty"`
	Branch    string `json:"branch,omitempty"`
	Commit    string `json:"commit,omitempty"`
	// AppliedPatches identifies reviewed source fixes applied while building the
	// backend. It makes a recipe-built binary auditable from backends.json.
	AppliedPatches []string `json:"applied_patches,omitempty"`
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
	Accel       string        `json:"accel,omitempty"`
	Patches     []RecipePatch `json:"-"`
}

//go:embed patches/hy3/0001-fix-router-tensor-name.patch
var hy3RouterTensorNamePatch []byte

var builtinRecipes = []Recipe{
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

// AppHome resolves the ggrun app home (holds .bin, .src, .config): LLM_APP_HOME
// wins, else the parent of the .bin/bin dir the running binary lives in.
func AppHome() string {
	if h := os.Getenv("LLM_APP_HOME"); h != "" {
		return h
	}
	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		switch filepath.Base(exeDir) {
		case ".bin", "bin":
			return filepath.Dir(exeDir)
		}
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
	data, err := os.ReadFile(ManifestPath())
	if err != nil {
		return nil
	}
	var list []Backend
	if json.Unmarshal(data, &list) != nil {
		return nil
	}
	return list
}

// Save writes the manifest.
func Save(list []Backend) error {
	p := ManifestPath()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(p, data, 0o644)
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
	list := Load()
	for i := range list {
		if strings.EqualFold(list[i].Tag, be.Tag) {
			list[i] = be
			return Save(list)
		}
	}
	return Save(append(list, be))
}

// Remove drops a backend by tag; returns false if not found.
func Remove(tag string) (bool, error) {
	list := Load()
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
	return found, Save(out)
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
