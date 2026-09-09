package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/update"
)

// Updating a backend used to mean reinstalling it, which rebuilt over the only
// working binary. When the new revision failed to build -- the poolside Laguna
// fork did exactly that, dying at 95% on a missing include -- there was nothing
// to go back to, and the machine was left with no backend for the architecture
// at all.
//
// So an update preserves the build it replaces instead of overwriting it. The
// candidate is compiled and conformance-tested in a sibling directory while
// the active tree remains untouched. Only then does a same-filesystem rename
// promote it and retain the previous tree for rollback.
//
// Nothing here is specific to a model or a fork: the source directory,
// accelerator and pin all come from the backend's own record, so any registered
// backend built from source can be updated and rolled back the same way.

// backendBuildLayout recovers where a backend was built from its binary path,
// which is the only thing backends.json records. The install path builds into
// <srcDir>/build-<accel>/bin/llama-server, so the layout is recoverable, and a
// path not matching that shape means the backend was registered by hand and
// cannot be rebuilt.
type backendBuildLayout struct {
	srcDir   string
	buildDir string
	accel    string
}

func validateBackendCandidateForRecipe(binary, routeArch, accel string, recipe *backends.Recipe) error {
	if err := validateBackendCandidate(binary, routeArch, accel); err != nil {
		return err
	}
	return validateBackendRecipeCandidate(binary, recipe)
}

// desiredBackendRecipe returns the reviewed recipe an update should build. A
// composed backend follows its installed base backend's current source pin, so
// updating the base and then the overlay advances both without ever attempting
// an unsafe merge into the active checkout.
func desiredBackendRecipe(be backends.Backend) (*backends.Recipe, error) {
	if len(be.Features) == 0 {
		return backends.RecipeForBackend(be)
	}
	base := backends.ByTag(be.BaseTag)
	if base == nil {
		return nil, fmt.Errorf("composed backend %q is missing its base backend %q", be.Tag, be.BaseTag)
	}
	recipe, err := backends.ComposeRecipe(*base, be.Features...)
	if err != nil {
		return nil, err
	}
	// The installed composite owns a stable public tag even if the tag-shaping
	// scheme changes in a later ggrun version.
	recipe.Tag = be.Tag
	recipe.Name = be.Tag
	return recipe, nil
}

func backendLayoutFor(be backends.Backend) (backendBuildLayout, error) {
	if be.Path == "" {
		return backendBuildLayout{}, fmt.Errorf("backend %q has no binary path", be.Tag)
	}
	binDir := filepath.Dir(be.Path)
	if filepath.Base(binDir) != "bin" {
		return backendBuildLayout{}, fmt.Errorf("backend %q was not built by ggrun (binary is not in a build-*/bin directory)", be.Tag)
	}
	buildDir := filepath.Dir(binDir)
	name := filepath.Base(buildDir)
	if !strings.HasPrefix(name, "build-") {
		return backendBuildLayout{}, fmt.Errorf("backend %q was not built by ggrun (found %q, expected build-<accel>)", be.Tag, name)
	}
	return backendBuildLayout{
		srcDir:   filepath.Dir(buildDir),
		buildDir: buildDir,
		accel:    strings.TrimPrefix(name, "build-"),
	}, nil
}

// archiveBuildDir moves a build aside so the new one can take its place. The
// commit is part of the name so an archived tree is identifiable, and a
// collision is resolved rather than overwritten -- silently discarding the one
// build that still works would defeat the entire point.
func archiveBuildDir(buildDir, commit string) (string, error) {
	short := strings.TrimSpace(commit)
	if len(short) > 12 {
		short = short[:12]
	}
	if short == "" {
		short = "unknown"
	}
	base := buildDir + "-prev-" + short
	dest := base
	for i := 1; ; i++ {
		if _, err := os.Stat(dest); os.IsNotExist(err) {
			break
		}
		dest = fmt.Sprintf("%s.%d", base, i)
		if i > 50 {
			return "", fmt.Errorf("too many archived builds beside %s", buildDir)
		}
	}
	if err := os.Rename(buildDir, dest); err != nil {
		return "", fmt.Errorf("archive previous build: %w", err)
	}
	return dest, nil
}

// pruneBuildDir removes a superseded build tree. It refuses anything that is
// not one of our own archives, because this deletes a directory outright.
func pruneBuildDir(path string) error {
	if path == "" || !strings.Contains(filepath.Base(path), "-prev-") {
		return fmt.Errorf("refusing to remove %q: not a ggrun archived build", path)
	}
	return os.RemoveAll(path)
}

func pruneCandidateBuildDir(path, activeBuildDir string) error {
	if path == "" || activeBuildDir == "" || filepath.Dir(path) != filepath.Dir(activeBuildDir) ||
		!strings.HasPrefix(filepath.Base(path), filepath.Base(activeBuildDir)+"-candidate-") {
		return fmt.Errorf("refusing to remove %q: not a staged candidate beside %s", path, activeBuildDir)
	}
	return os.RemoveAll(path)
}

// cmdBackendUpdate rebuilds one named fork. The reusable core returns errors so
// ggrun update can continue through every registered backend instead of the
// first failure terminating the entire process via os.Exit.
func cmdBackendUpdate(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "update needs a backend tag; see: ggrun backend list")
		os.Exit(2)
	}
	if err := updateRegisteredBackend(args[0]); err != nil {
		fmt.Fprintf(os.Stderr, "cannot update backend %q: %v\n", args[0], err)
		os.Exit(1)
	}
}

// updateRegisteredBackend updates one manifest backend while preserving its
// prior build. It never exits the process, which lets update-all isolate errors
// and produce a complete per-fork summary.
func updateRegisteredBackend(tag string) error {
	be := backends.ByTag(tag)
	if be == nil {
		return fmt.Errorf("unknown backend; see: ggrun backend list")
	}
	layout, err := backendLayoutFor(*be)
	if err != nil {
		return err
	}
	if _, err := os.Stat(layout.srcDir); err != nil {
		return fmt.Errorf("source checkout %s is gone; reinstall instead", layout.srcDir)
	}

	// A recipe is the authority on where a backend should be: it carries the
	// reviewed commit pin and the patches. Falling back to the record's own
	// branch keeps hand-added forks updatable too.
	recipe, err := desiredBackendRecipe(*be)
	if err != nil {
		return fmt.Errorf("resolve reviewed recipe: %w", err)
	}
	branch, commit := be.Branch, be.Commit
	forceRebuild := false
	if recipe != nil {
		branch, commit = recipe.Branch, recipe.Commit
		if commit != "" && strings.EqualFold(commit, be.Commit) &&
			sameStrings(recipe.PatchNames(), be.AppliedPatches) {
			if validateErr := validateBackendCandidateForRecipe(be.Path, be.RouteArch, layout.accel, recipe); validateErr == nil {
				// Persist catalog policy even when no source rebuild is needed.
				// This migrates manifests written before helper_only existed.
				*be = backends.ApplyBuiltinPolicy(*be)
				if err := backends.Upsert(*be); err != nil {
					return fmt.Errorf("persist reviewed backend policy: %w", err)
				}
				fmt.Printf("[backend] %s is already at the reviewed commit %s and passes conformance; nothing to update.\n", tag, shortCommit(commit))
				return nil
			} else {
				forceRebuild = true
				fmt.Printf("[backend] %s is current but failed conformance (%v); rebuilding the pinned source.\n", tag, validateErr)
			}
		}
	} else if commit != "" {
		// A hand-pinned backend has nothing to advance to. Tracking the branch
		// instead would silently discard the pin the user chose. It may still
		// need a same-commit rebuild when its active binary is damaged.
		if validateErr := validateBackendCandidateForRecipe(be.Path, be.RouteArch, layout.accel, recipe); validateErr == nil {
			fmt.Printf("[backend] %s is pinned to commit %s with no recipe to advance it and passes conformance.\n", tag, shortCommit(commit))
			fmt.Println("[backend] re-add it with a new --commit to move the pin.")
			return nil
		} else {
			forceRebuild = true
			fmt.Printf("[backend] %s pinned binary failed conformance (%v); rebuilding the same commit.\n", tag, validateErr)
		}
	}

	fmt.Printf("[backend] updating %s (%s)\n", tag, layout.srcDir)
	oldCommit, _ := gitOutput(layout.srcDir, "rev-parse", "HEAD")
	oldCommit = strings.TrimSpace(oldCommit)
	installedRecipe, err := installedBackendPatchRecipe(*be, recipe)
	if err != nil {
		return err
	}
	if err := prepareForkCheckoutRecipe(layout.srcDir, branch, commit, recipe, installedRecipe); err != nil {
		return fmt.Errorf("source checkout failed: %w", err)
	}
	newCommit, _ := gitOutput(layout.srcDir, "rev-parse", "HEAD")
	newCommit = strings.TrimSpace(newCommit)
	patchesCurrent := recipe == nil || sameStrings(recipe.PatchNames(), be.AppliedPatches)
	if newCommit != "" && strings.EqualFold(newCommit, be.Commit) && patchesCurrent && !forceRebuild {
		fmt.Printf("[backend] %s already at %s; nothing to rebuild.\n", tag, shortCommit(newCommit))
		return nil
	}

	candidateBuildDir := fmt.Sprintf("%s-candidate-%s-%d", layout.buildDir, shortCommit(newCommit), time.Now().UnixNano())
	promoted := false
	restoreSource := func(reason string) {
		// A failed build leaves the fork detached at the new (unbuildable)
		// commit with recipe patches applied. Return the checkout to the
		// commit we started from so the source state survives a failed
		// update the same way the generic backend path restores it.
		if oldCommit != "" {
			if _, err := gitOutput(layout.srcDir, "checkout", oldCommit); err != nil {
				fmt.Fprintf(os.Stderr, "[backend] %s: after %s, restore checkout %s failed: %v\n", tag, reason, oldCommit, err)
			} else {
				fmt.Printf("[backend] %s: %s; restored source checkout to %s\n", tag, reason, shortCommit(oldCommit))
			}
		}
	}
	defer func() {
		if !promoted {
			_ = pruneCandidateBuildDir(candidateBuildDir, layout.buildDir)
		}
	}()
	fmt.Printf("[backend] building isolated candidate (%s)… this can take 30–60 min\n", layout.accel)
	candidateBin, err := buildLlamaForkAt(layout.srcDir, candidateBuildDir, layout.accel, "")
	if err != nil {
		restoreSource("build failed")
		return fmt.Errorf("build failed: %w", err)
	}
	if err := validateBackendCandidateForRecipe(candidateBin, be.RouteArch, layout.accel, recipe); err != nil {
		restoreSource("candidate conformance failed")
		return fmt.Errorf("candidate conformance failed; active backend unchanged: %w", err)
	}
	fmt.Printf("[backend] candidate passed server/architecture/accelerator conformance: %s\n", candidateBin)

	archived := ""
	if _, err := os.Stat(layout.buildDir); err == nil {
		archived, err = archiveBuildDir(layout.buildDir, be.Commit)
		if err != nil {
			return err
		}
		fmt.Printf("[backend] previous build kept at %s\n", archived)
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("inspect active build: %w", err)
	}
	if err := os.Rename(candidateBuildDir, layout.buildDir); err != nil {
		if archived != "" {
			_ = os.Rename(archived, layout.buildDir)
		}
		return fmt.Errorf("activate conformance-tested candidate: %w", err)
	}
	promoted = true
	bin := filepath.Join(layout.buildDir, "bin", filepath.Base(be.Path))
	if err := validateBackendCandidateForRecipe(bin, be.RouteArch, layout.accel, recipe); err != nil {
		failed := layout.buildDir + "-candidate-invalid-" + shortCommit(newCommit)
		_ = os.Rename(layout.buildDir, failed)
		if archived != "" {
			_ = os.Rename(archived, layout.buildDir)
		}
		return fmt.Errorf("promoted candidate changed behavior; previous build restored, candidate at %s: %w", failed, err)
	}

	prev := &backends.BackendVersion{
		Path:           filepath.Join(archived, "bin", filepath.Base(be.Path)),
		Commit:         be.Commit,
		BaseTag:        be.BaseTag,
		Features:       append([]string(nil), be.Features...),
		AppliedPatches: be.AppliedPatches,
		ReplacedAt:     time.Now().UTC().Format(time.RFC3339),
	}
	if archived == "" {
		prev = nil
	}
	updated := *be
	updated.Path = bin
	updated.Commit = newCommit
	updated.Previous = prev
	if recipe != nil {
		updated.BaseTag = recipe.BaseTag
		updated.Features = append([]string(nil), recipe.Features...)
		updated.AppliedPatches = recipe.PatchNames()
		updated.HelperOnly = recipe.HelperOnly
		updated.RouteArch = recipe.RouteArch
		if recipe.GitURL != "" {
			updated.GitURL = recipe.GitURL
		}
		updated.Branch = recipe.Branch
	}
	if err := backends.Upsert(updated); err != nil {
		if archived != "" {
			unregistered := layout.buildDir + "-unregistered-" + shortCommit(newCommit)
			if renameErr := os.Rename(layout.buildDir, unregistered); renameErr == nil {
				if restoreErr := os.Rename(archived, layout.buildDir); restoreErr == nil {
					return fmt.Errorf("register failed: %w (previous build restored; new build preserved at %s)", err, unregistered)
				}
			}
		}
		return fmt.Errorf("register failed: %w", err)
	}
	// Retention is one level, so the tree this update pushed out of history is
	// now unreachable and would otherwise accumulate a compiled checkout per
	// update. It is only removed after the new build succeeded and registered.
	if be.Previous != nil && be.Previous.Path != "" {
		stale := filepath.Dir(filepath.Dir(be.Previous.Path))
		if err := pruneBuildDir(stale); err == nil {
			fmt.Printf("[backend] removed superseded build %s\n", stale)
		}
	}
	fmt.Printf("Updated backend %q → %s\n", tag, shortCommit(newCommit))
	if prev != nil {
		fmt.Printf("Roll back with: ggrun backend rollback %s\n", tag)
	}
	return nil
}

// updateMainlineBackend advances the mainline llama.cpp backend to its latest
// commit. It is safe by construction: the update package compiles each build
// variant in an isolated staging directory, smoke-tests it, and only then swaps
// it over the active build, retaining the previous one. A failed build or pull
// leaves the active backend untouched.
func updateMainlineBackend() error {
	return update.UpdateMainlineBackendAtAppHome(backends.AppHome())
}

func sameStrings(left, right []string) bool {
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

// cmdBackendRollback re-points a backend at the build it replaced.
func cmdBackendRollback(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "rollback needs a backend tag; see: ggrun backend list")
		os.Exit(2)
	}
	tag := args[0]
	restored, err := rollbackRegisteredBackend(tag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "rollback backend %q: %v\n", tag, err)
		os.Exit(1)
	}
	fmt.Printf("Rolled back backend %q → %s\n", tag, restored.Path)
	if restored.Commit != "" {
		fmt.Printf("  commit: %s\n", shortCommit(restored.Commit))
	}
	fmt.Printf("Undo with: ggrun backend rollback %s\n", tag)
}

// rollbackRegisteredBackend physically swaps the active and archived build
// trees while keeping the manifest path canonical (build-<accel>/bin). Merely
// pointing Path into build-cuda-prev-* makes the next update parse the
// accelerator as "cuda-prev-*" and leaves the backend permanently unupdatable.
func rollbackRegisteredBackend(tag string) (backends.Backend, error) {
	be := backends.ByTag(tag)
	if be == nil {
		return backends.Backend{}, fmt.Errorf("unknown backend; see: ggrun backend list")
	}
	if be.Previous == nil || be.Previous.Path == "" {
		return backends.Backend{}, fmt.Errorf("no previous build recorded")
	}
	if _, err := os.Stat(be.Previous.Path); err != nil {
		return backends.Backend{}, fmt.Errorf("previous build is gone (%s); reinstall instead", be.Previous.Path)
	}
	layout, err := backendLayoutFor(*be)
	if err != nil {
		return backends.Backend{}, err
	}
	previousBuildDir := filepath.Dir(filepath.Dir(be.Previous.Path))
	if filepath.Dir(previousBuildDir) != filepath.Dir(layout.buildDir) ||
		!strings.HasPrefix(filepath.Base(previousBuildDir), filepath.Base(layout.buildDir)+"-prev-") {
		return backends.Backend{}, fmt.Errorf("previous build path %s is not a managed archive beside %s", previousBuildDir, layout.buildDir)
	}
	if err := swapBackendBuildDirs(layout.buildDir, previousBuildDir); err != nil {
		return backends.Backend{}, err
	}
	revert := func() { _ = swapBackendBuildDirs(layout.buildDir, previousBuildDir) }

	canonicalBin := filepath.Join(layout.buildDir, "bin", filepath.Base(be.Path))
	archivedBin := filepath.Join(previousBuildDir, "bin", filepath.Base(be.Path))
	previousIdentity := *be
	previousIdentity.Commit = be.Previous.Commit
	previousIdentity.BaseTag = be.Previous.BaseTag
	previousIdentity.Features = append([]string(nil), be.Previous.Features...)
	previousIdentity.AppliedPatches = append([]string(nil), be.Previous.AppliedPatches...)
	previousRecipe, recipeErr := backends.RecipeForBackend(previousIdentity)
	if recipeErr != nil {
		revert()
		return backends.Backend{}, fmt.Errorf("restore recipe cannot be reconstructed; active build put back: %w", recipeErr)
	}
	if err := validateBackendCandidateForRecipe(canonicalBin, be.RouteArch, layout.accel, previousRecipe); err != nil {
		revert()
		return backends.Backend{}, fmt.Errorf("restored build failed conformance; active build put back: %w", err)
	}
	current := &backends.BackendVersion{
		Path:           archivedBin,
		Commit:         be.Commit,
		BaseTag:        be.BaseTag,
		Features:       append([]string(nil), be.Features...),
		AppliedPatches: be.AppliedPatches,
		ReplacedAt:     time.Now().UTC().Format(time.RFC3339),
	}
	restored := *be
	restored.Path = canonicalBin
	restored.Commit = be.Previous.Commit
	restored.BaseTag = be.Previous.BaseTag
	restored.Features = append([]string(nil), be.Previous.Features...)
	restored.AppliedPatches = be.Previous.AppliedPatches
	restored.Previous = current
	if err := backends.Upsert(restored); err != nil {
		revert()
		return backends.Backend{}, fmt.Errorf("register rollback failed; active build put back: %w", err)
	}
	return restored, nil
}

func swapBackendBuildDirs(left, right string) error {
	if left == "" || right == "" || left == right || filepath.Dir(left) != filepath.Dir(right) {
		return fmt.Errorf("refusing unsafe backend build swap %q <-> %q", left, right)
	}
	tmp := left + fmt.Sprintf("-rollback-swap-%d", time.Now().UnixNano())
	if err := os.Rename(left, tmp); err != nil {
		return fmt.Errorf("stage current build for swap: %w", err)
	}
	if err := os.Rename(right, left); err != nil {
		_ = os.Rename(tmp, left)
		return fmt.Errorf("move previous build to canonical path: %w", err)
	}
	if err := os.Rename(tmp, right); err != nil {
		_ = os.Rename(left, right)
		_ = os.Rename(tmp, left)
		return fmt.Errorf("archive replaced build after swap: %w", err)
	}
	return nil
}

func shortCommit(c string) string {
	c = strings.TrimSpace(c)
	if len(c) > 12 {
		return c[:12]
	}
	if c == "" {
		return "(unknown)"
	}
	return c
}

// Remove only patches recorded for the installed build. The desired recipe can
// add overlays; trying to reverse those before they exist blocks every upgrade.
// Stable patch IDs must resolve in order, otherwise preserve the checkout.
func installedBackendPatchRecipe(be backends.Backend, desired *backends.Recipe) (*backends.Recipe, error) {
	if desired == nil {
		return nil, nil
	}
	installed := *desired
	installed.Patches = nil
	next := 0
	for _, name := range be.AppliedPatches {
		found := false
		for next < len(desired.Patches) {
			patch := desired.Patches[next]
			next++
			if patch.Name == name {
				installed.Patches = append(installed.Patches, patch)
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("installed backend patch %q is not available in the reviewed recipe order; preserving source checkout", name)
		}
	}
	return &installed, nil
}
