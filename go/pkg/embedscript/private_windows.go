//go:build windows

package embedscript

// checkPrivateDir is a no-op on Windows: the user cache and %TEMP% are
// per-user profile directories.
func checkPrivateDir(string) error { return nil }
