package backends

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// What a source tree may hold that exists nowhere else, read from the files
// under .git only. ggrun never runs git to decide whether to delete.

// checkWorktrees refuses a tree that is the main checkout of git worktrees:
// they live elsewhere and keep their objects and commits in this tree's .git.
func checkWorktrees(tree string) error {
	gitDir := filepath.Join(tree, ".git")
	info, err := os.Lstat(gitDir)
	if err != nil || !info.IsDir() {
		return nil // no .git, or a .git file (this tree is itself a linked worktree)
	}
	dir := filepath.Join(gitDir, "worktrees")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("%w: cannot read %s: %v", ErrLinkedWorktrees, dir, err)
	}
	if len(entries) == 0 {
		return nil
	}
	var names []string
	for _, e := range entries {
		name := e.Name()
		if b, err := os.ReadFile(filepath.Join(dir, name, "gitdir")); err == nil {
			name += " (" + strings.TrimSpace(string(b)) + ")"
		}
		names = append(names, name)
	}
	sort.Strings(names)
	return fmt.Errorf("%w: %s is the main checkout of %d git worktree(s): %s; run `git worktree remove <path>` in it for each (or `git worktree prune` if they are gone) first",
		ErrLinkedWorktrees, tree, len(names), strings.Join(names, ", "))
}

func isHex40(s string) bool { return isHexDigest(strings.ToLower(s), 40) }

// readGitRefs returns every ref name under refs/ (loose files override
// packed-refs) with its commit id.
func readGitRefs(gitDir string) map[string]string {
	refs := map[string]string{}
	if b, err := os.ReadFile(filepath.Join(gitDir, "packed-refs")); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			line = strings.TrimSpace(line)
			if line == "" || line[0] == '#' || line[0] == '^' {
				continue
			}
			if f := strings.Fields(line); len(f) == 2 && isHex40(f[0]) {
				refs[f[1]] = strings.ToLower(f[0])
			}
		}
	}
	root := filepath.Join(gitDir, "refs")
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !d.Type().IsRegular() {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return nil
		}
		if sha := strings.TrimSpace(string(b)); isHex40(sha) {
			rel, _ := filepath.Rel(gitDir, p)
			refs[filepath.ToSlash(rel)] = strings.ToLower(sha)
		}
		return nil
	})
	return refs
}

// publishedCommits are commits known to exist upstream without asking git: the
// reviewed pins in the catalog and the commits recorded for registrations.
func (env CleanupEnv) publishedCommits() map[string]bool {
	out := map[string]bool{}
	add := func(c string) {
		if c = strings.ToLower(strings.TrimSpace(c)); isHex40(c) {
			out[c] = true
		}
	}
	for _, r := range builtinRecipes {
		add(r.Commit)
	}
	for _, b := range env.Backends {
		add(b.Commit)
		if b.Previous != nil {
			add(b.Previous.Commit)
		}
	}
	return out
}

// LocalWork says what in a whole source tree might exist nowhere else, or ""
// when nothing was found and everything could be read. It looks only at files:
// a stash ref; HEAD or a local branch at a commit that is neither a remote
// branch tip nor a reviewed pin; files newer than the git index. A tree with
// no readable .git, or that is a linked worktree, is "unknown". Recipe patches
// make files newer than the index, so a patched tree reports possible changes.
func (env CleanupEnv) LocalWork(tree string) string {
	gitDir := filepath.Join(tree, ".git")
	info, err := os.Lstat(gitDir)
	if err != nil {
		return "unknown: no .git directory, so unsaved work cannot be told"
	}
	if !info.IsDir() {
		target := ""
		if b, err := os.ReadFile(gitDir); err == nil {
			target = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(string(b)), "gitdir:"))
		}
		return fmt.Sprintf("unknown: a linked worktree or submodule of %s; uncommitted changes cannot be read without running git", target)
	}
	var found, unknown []string
	refs := readGitRefs(gitDir)
	if _, ok := refs["refs/stash"]; ok {
		found = append(found, "stash entries")
	}
	published := env.publishedCommits()
	for name, sha := range refs {
		if strings.HasPrefix(name, "refs/remotes/") {
			published[sha] = true
		}
	}
	var unpublished []string
	if b, err := os.ReadFile(filepath.Join(gitDir, "HEAD")); err != nil {
		unknown = append(unknown, "HEAD unreadable")
	} else if head := strings.TrimSpace(string(b)); strings.HasPrefix(head, "ref:") {
		ref := strings.TrimSpace(strings.TrimPrefix(head, "ref:"))
		if sha, ok := refs[ref]; !ok {
			unknown = append(unknown, "HEAD ref "+ref+" unreadable")
		} else if !published[sha] {
			unpublished = append(unpublished, ref)
		}
	} else if isHex40(head) {
		if !published[strings.ToLower(head)] {
			unpublished = append(unpublished, "detached HEAD")
		}
	} else {
		unknown = append(unknown, "HEAD unreadable")
	}
	for name, sha := range refs {
		if strings.HasPrefix(name, "refs/heads/") && !published[sha] {
			unpublished = append(unpublished, name)
		}
	}
	sort.Strings(unpublished)
	if len(unpublished) > 0 {
		found = append(found, "commits not on any remote branch or reviewed pin ("+strings.Join(uniqueSorted(unpublished), ", ")+")")
	}
	idx, err := os.Stat(filepath.Join(gitDir, "index"))
	if err != nil {
		unknown = append(unknown, "no git index")
	} else {
		limit := idx.ModTime().Add(2 * time.Second)
		n, example := 0, ""
		_ = filepath.WalkDir(tree, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				if p != tree && (d.Name() == ".git" || strings.HasPrefix(d.Name(), "build")) {
					return filepath.SkipDir
				}
				return nil
			}
			if !d.Type().IsRegular() {
				return nil
			}
			if fi, err := d.Info(); err == nil && fi.ModTime().After(limit) {
				if n == 0 {
					example, _ = filepath.Rel(tree, p)
				}
				n++
			}
			return nil
		})
		if n > 0 {
			found = append(found, fmt.Sprintf("%d file(s) newer than the git index, possible uncommitted changes (e.g. %s)", n, filepath.ToSlash(example)))
		}
	}
	switch {
	case len(found) > 0 && len(unknown) > 0:
		return strings.Join(found, "; ") + "; also unknown: " + strings.Join(unknown, ", ")
	case len(found) > 0:
		return strings.Join(found, "; ")
	case len(unknown) > 0:
		return "unknown: " + strings.Join(unknown, ", ")
	}
	return ""
}

func uniqueSorted(in []string) []string {
	var out []string
	for i, s := range in {
		if i == 0 || s != in[i-1] {
			out = append(out, s)
		}
	}
	return out
}
