// Package atomicfile stages complete files before replacing persisted state.
package atomicfile

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// Write preserves the destination until encode, sync and close succeed. Existing
// regular-file permissions and valid symlinks are preserved. New files use perm.
// The parent directory must exist. Writers are last-successful-rename wins;
// callers performing read-modify-write must provide their own serialization.
// Replacement uses os.Rename (atomic on Unix within the same filesystem); this
// is not a directory-fsync or cross-platform power-loss durability guarantee.
func Write(path string, perm os.FileMode, encode func(io.Writer) error) error {
	if info, err := os.Lstat(path); err == nil && info.Mode()&os.ModeSymlink != 0 {
		resolved, err := filepath.EvalSymlinks(path)
		if err != nil {
			return fmt.Errorf("resolve destination: %w", err)
		}
		path = resolved
	} else if err != nil && !os.IsNotExist(err) {
		return err
	}
	if info, err := os.Stat(path); err == nil {
		if !info.Mode().IsRegular() {
			return fmt.Errorf("destination is not a regular file: %s", path)
		}
		perm = info.Mode().Perm()
	} else if !os.IsNotExist(err) {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".ggrun-state-*.tmp")
	if err != nil {
		return err
	}
	name := f.Name()
	defer os.Remove(name)
	defer f.Close()
	if err := encode(f); err != nil {
		return fmt.Errorf("write staged state: %w", err)
	}
	// Keep the incomplete file private even when the final destination is public.
	if err := f.Chmod(perm); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(name, path); err != nil {
		return fmt.Errorf("replace state: %w", err)
	}
	return nil
}
