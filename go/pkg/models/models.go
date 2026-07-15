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
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
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
