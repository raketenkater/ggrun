//go:build !linux

package backends

// platformProcessLister: no process table reader is implemented here, so
// deleting a source tree needs --force-not-running.
func platformProcessLister() func() ([]ProcInfo, error) { return nil }
