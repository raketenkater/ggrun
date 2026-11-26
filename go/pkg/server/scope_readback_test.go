//go:build linux

package server

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// C-10: the hard memory limit is read back from the scope's cgroup; a launch
// must not proceed when it is not in force.
func TestScopeMemoryLimitReadback(t *testing.T) {
	root := t.TempDir()
	cg := "/user.slice/ggrun-test.scope"
	if err := os.MkdirAll(filepath.Join(root, cg), 0o755); err != nil {
		t.Fatal(err)
	}
	oldRoot, oldFn, oldWait := cgroupFSRoot, scopeControlGroupFn, scopeLimitWait
	cgroupFSRoot, scopeLimitWait = root, 300*time.Millisecond
	scopeControlGroupFn = func(string, scopeMode) (string, error) { return cg, nil }
	t.Cleanup(func() { cgroupFSRoot, scopeControlGroupFn, scopeLimitWait = oldRoot, oldFn, oldWait })
	write := func(v string) {
		if err := os.WriteFile(filepath.Join(root, cg, "memory.max"), []byte(v+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for value, ok := range map[string]bool{
		"8589934592": true,  // exactly 8192 MiB
		"8589930496": true,  // rounded down to a page
		"4294967296": true,  // stricter than asked
		"max":        false, // controller not delegated
		"9663676416": false, // 9216 MiB: looser than asked
		"garbage":    false,
	} {
		write(value)
		err := verifyScopeMemoryLimit("ggrun-test.scope", scopeUser, 8192)
		if ok && err != nil {
			t.Errorf("%s refused: %v", value, err)
		}
		if !ok && !errors.Is(err, errScopeLimitNotInForce) {
			t.Errorf("%s accepted (err %v)", value, err)
		}
	}
	// No scope or no limit: nothing to verify.
	if err := verifyScopeMemoryLimit("", scopeUser, 8192); err != nil {
		t.Fatal(err)
	}
	if err := verifyScopeMemoryLimit("u", scopeUser, 0); err != nil {
		t.Fatal(err)
	}
	// A scope that never appears is left to the readiness path.
	scopeControlGroupFn = func(string, scopeMode) (string, error) { return "", errors.New("gone") }
	start := time.Now()
	if err := verifyScopeMemoryLimit("u", scopeUser, 8192); err != nil || time.Since(start) > 3*time.Second {
		t.Fatalf("missing scope: %v after %s", err, time.Since(start))
	}
}
