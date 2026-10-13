// Package models manages GGUF artifacts in ggrun's configured model directory.
package models

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// ErrNotFound reports a model name that is not present in the configured model
// directory.
var ErrNotFound = errors.New("model not found")

// Model is one launchable GGUF artifact. A sharded GGUF is represented as one
// model with all of its shard paths in Files.
type Model struct {
	Name  string
	Files []string
	Bytes int64
}

var shardName = regexp.MustCompile(`(?i)^(.*)-[0-9]{5}-of-[0-9]{5}\.gguf$`)
var shardPartsName = regexp.MustCompile(`(?i)^(.*)-([0-9]{5})-of-([0-9]{5})\.gguf$`)

const maxGGUFShards = 10_000

// ResolveGGUFShardFiles returns the exact files belonging to a GGUF entrypoint.
// A split model is launchable only from shard 00001 and only after every shard
// named by its -of-N suffix exists. Silently summing whichever shards happen to
// be present makes an interrupted download look smaller than it is, which can
// turn a placement plan into a host/GPU OOM.
func ResolveGGUFShardFiles(path string) (files []string, sharded bool, err error) {
	path = filepath.Clean(path)
	match := shardPartsName.FindStringSubmatch(filepath.Base(path))
	if len(match) != 4 {
		info, statErr := os.Stat(path)
		if statErr != nil {
			return nil, false, statErr
		}
		if info.IsDir() {
			return nil, false, fmt.Errorf("GGUF path is a directory: %s", path)
		}
		return []string{path}, false, nil
	}

	index, indexErr := strconv.Atoi(match[2])
	total, totalErr := strconv.Atoi(match[3])
	if indexErr != nil || totalErr != nil || index != 1 || total < 1 || total > maxGGUFShards {
		return nil, true, fmt.Errorf("invalid sharded GGUF entrypoint %q (use shard 00001; supported shard count 1-%d)", filepath.Base(path), maxGGUFShards)
	}

	dir := filepath.Dir(path)
	files = make([]string, 0, total)
	for shard := 1; shard <= total; shard++ {
		name := fmt.Sprintf("%s-%05d-of-%05d.gguf", match[1], shard, total)
		candidate := filepath.Join(dir, name)
		info, statErr := os.Stat(candidate)
		if statErr != nil || info.IsDir() {
			return nil, true, fmt.Errorf("incomplete sharded GGUF %q: missing shard %05d of %05d", match[1], shard, total)
		}
		files = append(files, candidate)
	}
	return files, true, nil
}

// List finds GGUF files below root. It follows file symlinks for their sizes
// but never follows symlinked directories, so a model directory cannot cause a
// recursive walk of another filesystem. Split GGUFs are grouped into one model.
func List(root string) ([]Model, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return nil, fmt.Errorf("model directory is empty")
	}
	info, err := os.Stat(root)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("model directory is not a directory: %s", root)
	}

	grouped := map[string]*Model{}
	err = filepath.WalkDir(WalkRoot(root), func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() || !strings.EqualFold(filepath.Ext(entry.Name()), ".gguf") {
			return nil
		}
		// Stat, rather than DirEntry.Info, reports the target size for a model
		// file symlinked from a larger model disk.
		fileInfo, err := os.Stat(path)
		if err != nil || fileInfo.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.Clean(rel)
		name := logicalName(rel)
		model := grouped[name]
		if model == nil {
			model = &Model{Name: name}
			grouped[name] = model
		}
		model.Files = append(model.Files, rel)
		model.Bytes += fileInfo.Size()
		return nil
	})
	if err != nil {
		return nil, err
	}

	out := make([]Model, 0, len(grouped))
	for _, model := range grouped {
		sort.Strings(model.Files)
		out = append(out, *model)
	}
	sort.Slice(out, func(i, j int) bool {
		return strings.ToLower(out[i].Name) < strings.ToLower(out[j].Name)
	})
	return out, nil
}

// WalkRoot is root with a trailing separator when root itself is a symlink to
// a directory: filepath.Walk and WalkDir do not descend into a symlinked root,
// so a models directory linked to a larger disk listed nothing. Only the root
// is resolved; links to directories below it are still not followed.
func WalkRoot(root string) string {
	if fi, err := os.Lstat(root); err == nil && fi.Mode()&os.ModeSymlink != 0 && !strings.HasSuffix(root, string(os.PathSeparator)) {
		return root + string(os.PathSeparator)
	}
	return root
}

// Remove deletes one listed model. It accepts only a model name emitted by
// List, including its logical name for a sharded GGUF, and never follows a
// symlinked directory outside root.
func Remove(root, name string) (Model, error) {
	name, err := cleanName(name)
	if err != nil {
		return Model{}, err
	}
	all, err := List(root)
	if err != nil {
		return Model{}, err
	}
	var target *Model
	for i := range all {
		if all[i].Name == name {
			target = &all[i]
			break
		}
	}
	if target == nil {
		return Model{}, fmt.Errorf("%w: %s", ErrNotFound, name)
	}

	for _, rel := range target.Files {
		path, err := removablePath(root, rel)
		if err != nil {
			return Model{}, err
		}
		if err := os.Remove(path); err != nil {
			return Model{}, fmt.Errorf("remove %s: %w", rel, err)
		}
	}
	return *target, nil
}

// RemoveExternal deletes a model file that lives outside the configured model
// directory. Unlike Remove it takes an explicit path, not a model name: the
// caller is the TUI, which has already shown a strong typed confirmation that
// the file sits outside the primary model directory and will be permanently
// deleted. Safety boundary: the path must be the model's own resolved identity
// (never arbitrary user input), directories are refused, and a path that
// resolves back inside the primary directory is refused so this API can never
// bypass Remove's in-dir checks. Symlinks are followed to their real target,
// which is what gets deleted — the strong confirmation is the guard for that.
func RemoveExternal(root, path string) (Model, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return Model{}, errors.New("model directory is empty")
	}
	if path == "" {
		return Model{}, errors.New("remove external model: empty path")
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return Model{}, fmt.Errorf("remove external model %s: %w", path, err)
	}
	abs = filepath.Clean(abs)

	// A sharded GGUF is one model; delete the whole set. An incomplete set (or
	// a missing shard) falls back to the single entrypoint the caller knows
	// about rather than failing silently.
	files, _, shardErr := ResolveGGUFShardFiles(abs)
	if shardErr != nil {
		files = []string{abs}
	}

	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return Model{}, fmt.Errorf("resolve model directory: %w", err)
	}

	var removed Model
	removed.Files = make([]string, 0, len(files))
	for _, f := range files {
		realPath, err := filepath.EvalSymlinks(f)
		if err != nil {
			return Model{}, fmt.Errorf("resolve external model %s: %w", f, err)
		}
		realPath = filepath.Clean(realPath)
		// A file that resolves inside the primary directory is protected: the
		// normal Remove path manages those and this API must not bypass its
		// cleanName/symlinked-directory checks. External rows only exist for
		// paths outside the primary dir, so reaching here means the file's
		// identity changed underneath us — refuse rather than guess.
		rel, relErr := filepath.Rel(resolvedRoot, realPath)
		if relErr == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return Model{}, fmt.Errorf("refusing to remove %s through the external path: it resolves inside the model directory", f)
		}
		info, err := os.Lstat(realPath)
		if err != nil {
			return Model{}, fmt.Errorf("remove external model %s: %w", realPath, err)
		}
		if info.IsDir() {
			return Model{}, fmt.Errorf("refusing to remove a directory: %s", realPath)
		}
		if err := os.Remove(realPath); err != nil {
			return Model{}, fmt.Errorf("remove external model %s: %w", realPath, err)
		}
		removed.Bytes += info.Size()
		removed.Files = append(removed.Files, realPath)
		// The confirmed path may itself have been a symlink to that real target.
		// Remove the entry too so the row cannot linger as a broken link.
		if f != realPath {
			if _, lerr := os.Lstat(f); lerr == nil {
				if err := os.Remove(f); err != nil {
					return Model{}, fmt.Errorf("remove external model symlink %s: %w", f, err)
				}
			}
		}
	}
	if len(removed.Files) == 0 {
		return Model{}, errors.New("remove external model: no files removed")
	}
	removed.Name = filepath.Base(removed.Files[0])
	return removed, nil
}

func logicalName(rel string) string {
	base := filepath.Base(rel)
	match := shardName.FindStringSubmatch(base)
	if len(match) != 2 {
		return rel
	}
	return filepath.Join(filepath.Dir(rel), match[1]+".gguf")
}

func cleanName(name string) (string, error) {
	name = filepath.Clean(strings.TrimSpace(name))
	if name == "" || name == "." || filepath.IsAbs(name) || name == ".." ||
		strings.HasPrefix(name, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("model name must be a relative GGUF path inside the model directory")
	}
	if !strings.EqualFold(filepath.Ext(name), ".gguf") {
		return "", fmt.Errorf("model name must end in .gguf")
	}
	return name, nil
}

func removablePath(root, rel string) (string, error) {
	path := filepath.Join(root, rel)
	parent, err := filepath.EvalSymlinks(filepath.Dir(path))
	if err != nil {
		return "", fmt.Errorf("resolve model directory for %s: %w", rel, err)
	}
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return "", fmt.Errorf("resolve model directory: %w", err)
	}
	inside, err := filepath.Rel(resolvedRoot, parent)
	if err != nil || inside == ".." || strings.HasPrefix(inside, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("refusing to remove a file outside the model directory: %s", rel)
	}
	info, err := os.Lstat(path)
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return "", fmt.Errorf("refusing to remove a directory: %s", rel)
	}
	return path, nil
}
