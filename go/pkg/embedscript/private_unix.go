//go:build !windows

package embedscript

import (
	"fmt"
	"os"
	"syscall"
)

// checkPrivateDir refuses a cache directory that is a symlink, belongs to
// another user, or is writable by group or others.
func checkPrivateDir(dir string) error {
	info, err := os.Lstat(dir)
	if err != nil {
		return fmt.Errorf("check script cache %s: %w", dir, err)
	}
	if !info.IsDir() {
		return fmt.Errorf("script cache %s is not a directory (symlink or file); remove it and retry", dir)
	}
	if st, ok := info.Sys().(*syscall.Stat_t); ok && int(st.Uid) != os.Geteuid() {
		return fmt.Errorf("script cache %s belongs to uid %d, not to you (uid %d); remove it or set XDG_CACHE_HOME", dir, st.Uid, os.Geteuid())
	}
	if perm := info.Mode().Perm(); perm&0o022 != 0 {
		return fmt.Errorf("script cache %s is writable by other users (mode %o); run `chmod go-w %s` or remove it", dir, perm, dir)
	}
	return nil
}
