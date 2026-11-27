//go:build linux

package server

import (
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// A stand-in backend that ignores SIGTERM, so only SIGKILL ends it. It writes
// its pid (= its process group, setsid'd by the wrapper) before it sleeps.
const termIgnoringChild = `trap '' TERM; echo $$ > "$1"; exec sleep 60`

// alive treats a zombie as dead: it holds no resources and waits for a reaper.
func alive(pid int) bool {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return false
	}
	s := string(data)
	i := strings.LastIndexByte(s, ')')
	return i < 0 || i+2 >= len(s) || s[i+2] != 'Z'
}

func groupGone(pgid int) bool {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return false
	}
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		got, err := syscall.Getpgid(pid)
		if err == nil && got == pgid && alive(pid) {
			return false
		}
	}
	return true
}

func readPIDFile(path string, timeout time.Duration) (int, error) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if data, err := os.ReadFile(path); err == nil {
			if pid, err := strconv.Atoi(strings.TrimSpace(string(data))); err == nil && pid > 0 {
				return pid, nil
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	return 0, fmt.Errorf("no pid in %s after %s", path, timeout)
}

func waitPIDFile(t *testing.T, path string, timeout time.Duration) int {
	t.Helper()
	pid, err := readPIDFile(path, timeout)
	if err != nil {
		t.Fatal(err)
	}
	return pid
}

func scopeWrapperCmd(parent int, pidFile string) *exec.Cmd {
	cmd := exec.Command("/bin/sh", "-c", scopedParentWatchScript, "ggrun-scope-watch", strconv.Itoa(parent),
		"/bin/sh", "-c", termIgnoringChild, "child", pidFile)
	// Like the production wrapper: its own process group.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	return cmd
}

func spin(d time.Duration) {
	for end := time.Now().Add(d); time.Now().Before(end); {
	}
}

// ggrun's parent-death SIGTERM arrives once per exiting ggrun thread. Every
// TERM after the first must not interrupt the wrapper's teardown: it must
// still reach SIGKILL for the backend group and exit 143. A 1-2% race shows
// up in 200 tries.
func TestScopeWrapperTeardownSurvivesRepeatedTERM(t *testing.T) {
	if _, err := exec.LookPath("setsid"); err != nil {
		t.Skip("setsid not installed")
	}
	const tries, batch = 200, 40
	dir := t.TempDir()
	var mu sync.Mutex
	var failures []string
	for start := 0; start < tries; start += batch {
		var wg sync.WaitGroup
		for i := start; i < start+batch && i < tries; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				fail := func(format string, args ...any) {
					mu.Lock()
					failures = append(failures, fmt.Sprintf("try %d: ", i)+fmt.Sprintf(format, args...))
					mu.Unlock()
				}
				pidFile := filepath.Join(dir, fmt.Sprintf("child-%d.pid", i))
				w := scopeWrapperCmd(os.Getpid(), pidFile)
				if err := w.Start(); err != nil {
					fail("start wrapper: %v", err)
					return
				}
				pgid, err := readPIDFile(pidFile, 10*time.Second)
				if err != nil {
					_ = w.Process.Kill()
					_ = w.Wait()
					fail("%v", err)
					return
				}
				defer syscall.Kill(-pgid, syscall.SIGKILL)
				first := time.Now()
				_ = w.Process.Signal(syscall.SIGTERM)
				spin(200 * time.Microsecond)
				_ = w.Process.Signal(syscall.SIGTERM)
				spin(300 * time.Microsecond)
				_ = w.Process.Signal(syscall.SIGTERM)
				err = w.Wait()
				var exitErr *exec.ExitError
				if !errors.As(err, &exitErr) || exitErr.ExitCode() != 143 {
					fail("wrapper ended with %v, want exit status 143", err)
				}
				for !groupGone(pgid) {
					if time.Since(first) > 3*time.Second {
						fail("backend group %d still alive 3s after the first TERM", pgid)
						return
					}
					time.Sleep(20 * time.Millisecond)
				}
			}(i)
		}
		wg.Wait()
	}
	if len(failures) > 0 {
		t.Fatalf("%d of %d tries failed:\n%s", len(failures), tries, strings.Join(failures, "\n"))
	}
}

// Without any signal to the wrapper, its watcher must end the backend group
// once ggrun is gone: one 1 s poll, TERM, 2 s, KILL.
func TestScopeWrapperWatcherKillsChildAfterParentDies(t *testing.T) {
	if _, err := exec.LookPath("setsid"); err != nil {
		t.Skip("setsid not installed")
	}
	parent := exec.Command("sleep", "2")
	if err := parent.Start(); err != nil {
		t.Fatal(err)
	}
	pidFile := filepath.Join(t.TempDir(), "child.pid")
	w := scopeWrapperCmd(parent.Process.Pid, pidFile)
	if err := w.Start(); err != nil {
		t.Fatal(err)
	}
	pgid := waitPIDFile(t, pidFile, 10*time.Second)
	defer syscall.Kill(-pgid, syscall.SIGKILL)
	_ = parent.Wait() // reaped: kill -0 on it now fails
	gone := time.Now()
	done := make(chan error, 1)
	go func() { done <- w.Wait() }()
	for !groupGone(pgid) {
		if time.Since(gone) > 5*time.Second {
			t.Fatalf("backend group %d still alive 5s after its parent died", pgid)
		}
		time.Sleep(50 * time.Millisecond)
	}
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("wrapper did not exit after its backend group ended")
	}
}

// The script is handed to systemd-run, which expands "${...}" itself
// (--expand-environment, systemd 254+); the shell must see its own text.
func TestScopeWrapperHasNoSystemdExpansion(t *testing.T) {
	if strings.Contains(scopedParentWatchScript, "${") {
		t.Fatal(`scopedParentWatchScript contains "${", which systemd-run expands before the shell runs`)
	}
}

// A SIGTERM while ggrun waits for the backend to become ready used to kill
// ggrun by default and leave the backend to the scope wrapper alone. The
// child here ignores TERM, so the parent-death signal cannot end it: only
// Process.Stop (TERM, 2 s, KILL to its group) can.
func TestShutdownSignalDuringLoadWaitStopsBackend(t *testing.T) {
	if os.Getenv("GGRUN_GUARD_HELPER") == "1" {
		args := []string{"/bin/sh", "-c", termIgnoringChild, "child", os.Getenv("GGRUN_GUARD_PID_FILE")}
		port, _ := strconv.Atoi(os.Getenv("GGRUN_GUARD_PORT"))
		_, _ = StartWithTimeoutToOptions(args, port, time.Minute, io.Discard, io.Discard, StartOptions{})
		os.Exit(3) // readiness can only end by the signal
	}
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	_ = l.Close() // nothing answers: the helper stays in its load-wait
	pidFile := filepath.Join(t.TempDir(), "child.pid")
	helper := exec.Command(os.Args[0], "-test.run=^TestShutdownSignalDuringLoadWaitStopsBackend$")
	helper.Env = append(os.Environ(),
		"GGRUN_GUARD_HELPER=1",
		"GGRUN_GUARD_PID_FILE="+pidFile,
		"GGRUN_GUARD_PORT="+strconv.Itoa(port),
	)
	if err := helper.Start(); err != nil {
		t.Fatal(err)
	}
	pid := waitPIDFile(t, pidFile, 10*time.Second)
	defer syscall.Kill(-pid, syscall.SIGKILL)
	time.Sleep(200 * time.Millisecond) // well inside the readiness poll
	if err := helper.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	err = helper.Wait()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("helper ended with %v, want death by SIGTERM", err)
	}
	if ws, ok := exitErr.Sys().(syscall.WaitStatus); !ok || !ws.Signaled() || ws.Signal() != syscall.SIGTERM {
		t.Fatalf("helper ended with %v, want death by SIGTERM", err)
	}
	if alive(pid) {
		t.Fatalf("backend pid %d outlived ggrun's SIGTERM during the load-wait", pid)
	}
}
