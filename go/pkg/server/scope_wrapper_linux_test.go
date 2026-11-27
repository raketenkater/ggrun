//go:build linux

package server

import (
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
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
					// The script sleeps 2 s before its KILL; the rest is margin
					// for a loaded runner. A passing try ends at about 2 s.
					if time.Since(first) > 8*time.Second {
						fail("backend group %d still alive 8s after the first TERM", pgid)
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

// The guard tests run ggrun's side in a helper process: the guard is global
// to a process and ends it with the signal.
//
// GGRUN_GUARD_MODE:
//
//	loadwait  start a TERM-ignoring backend and sit in the readiness wait
//	deadline  the same with a 300 ms guard deadline
//	handover  the same in the background; then the caller's own Notify for
//	          SIGTERM and OwnShutdownSignals; exit 0 on SIGTERM
func TestGuardHelperProcess(t *testing.T) {
	mode := os.Getenv("GGRUN_GUARD_MODE")
	if mode == "" {
		t.Skip("helper process only")
	}
	args := []string{"/bin/sh", "-c", termIgnoringChild, "child", os.Getenv("GGRUN_GUARD_PID_FILE")}
	port, _ := strconv.Atoi(os.Getenv("GGRUN_GUARD_PORT"))
	start := func() {
		_, _ = StartWithTimeoutToOptions(args, port, time.Minute, io.Discard, io.Discard, StartOptions{})
	}
	switch mode {
	case "deadline":
		guardStopDeadline = 300 * time.Millisecond
		start()
	case "handover":
		go start()
		if _, err := readPIDFile(os.Getenv("GGRUN_GUARD_PID_FILE"), 10*time.Second); err != nil {
			os.Exit(4)
		}
		own := make(chan os.Signal, 1)
		signal.Notify(own, syscall.SIGTERM)
		OwnShutdownSignals()
		if err := os.WriteFile(os.Getenv("GGRUN_GUARD_READY_FILE"), []byte("1"), 0o600); err != nil {
			os.Exit(5)
		}
		<-own
		os.Exit(0)
	default:
		start()
	}
	os.Exit(3) // readiness can only end by the signal
}

type guardHelper struct {
	cmd     *exec.Cmd
	backend int
	ready   string
	done    chan error
}

// startGuardHelper starts the helper in mode; ignoreHUP starts it the way
// `nohup` does, with SIGHUP ignored.
func startGuardHelper(t *testing.T, mode string, ignoreHUP bool) *guardHelper {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	_ = l.Close() // nothing answers: the helper stays in its load-wait
	dir := t.TempDir()
	h := &guardHelper{ready: filepath.Join(dir, "ready"), done: make(chan error, 1)}
	pidFile := filepath.Join(dir, "child.pid")
	run := []string{os.Args[0], "-test.run=^TestGuardHelperProcess$"}
	if ignoreHUP {
		// An ignored disposition survives exec, as under nohup.
		run = append([]string{"/bin/sh", "-c", `trap '' HUP; exec "$0" "$@"`}, run...)
	}
	h.cmd = exec.Command(run[0], run[1:]...)
	h.cmd.Env = append(os.Environ(),
		"GGRUN_GUARD_MODE="+mode,
		"GGRUN_GUARD_PID_FILE="+pidFile,
		"GGRUN_GUARD_READY_FILE="+h.ready,
		"GGRUN_GUARD_PORT="+strconv.Itoa(port),
	)
	if err := h.cmd.Start(); err != nil {
		t.Fatal(err)
	}
	go func() { h.done <- h.cmd.Wait() }()
	h.backend = waitPIDFile(t, pidFile, 10*time.Second)
	t.Cleanup(func() {
		_ = syscall.Kill(-h.backend, syscall.SIGKILL)
		_ = h.cmd.Process.Kill()
	})
	time.Sleep(200 * time.Millisecond) // well inside the readiness poll
	return h
}

func (h *guardHelper) signal(t *testing.T, sig syscall.Signal) {
	t.Helper()
	if err := h.cmd.Process.Signal(sig); err != nil {
		t.Fatal(err)
	}
}

// wait returns how the helper ended, failing after timeout.
func (h *guardHelper) wait(t *testing.T, timeout time.Duration) error {
	t.Helper()
	select {
	case err := <-h.done:
		return err
	case <-time.After(timeout):
		t.Fatalf("helper still running %s after the signal", timeout)
		return nil
	}
}

func diedOf(err error, sig syscall.Signal) bool {
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		return false
	}
	ws, ok := exitErr.Sys().(syscall.WaitStatus)
	return ok && ws.Signaled() && ws.Signal() == sig
}

// A SIGTERM while ggrun waits for the backend to become ready used to kill
// ggrun by default and leave the backend to the scope wrapper alone. The
// child here ignores TERM, so the parent-death signal cannot end it: only
// Process.Stop (TERM, 2 s, KILL to its group) can.
func TestShutdownSignalDuringLoadWaitStopsBackend(t *testing.T) {
	h := startGuardHelper(t, "loadwait", false)
	h.signal(t, syscall.SIGTERM)
	if err := h.wait(t, 30*time.Second); !diedOf(err, syscall.SIGTERM) {
		t.Fatalf("helper ended with %v, want death by SIGTERM", err)
	}
	if alive(h.backend) {
		t.Fatalf("backend pid %d outlived ggrun's SIGTERM during the load-wait", h.backend)
	}
}

// `nohup ggrun ...` starts with SIGHUP ignored and must keep ignoring it: the
// guard may not install a handler over an inherited ignore. SIGTERM is still
// guarded.
func TestGuardKeepsInheritedIgnoredSIGHUP(t *testing.T) {
	h := startGuardHelper(t, "loadwait", true)
	h.signal(t, syscall.SIGHUP)
	select {
	case err := <-h.done:
		t.Fatalf("a nohup'd launch ended on SIGHUP: %v", err)
	case <-time.After(time.Second):
	}
	if !alive(h.backend) {
		t.Fatal("SIGHUP stopped the backend of a nohup'd launch")
	}
	h.signal(t, syscall.SIGTERM)
	if err := h.wait(t, 30*time.Second); !diedOf(err, syscall.SIGTERM) {
		t.Fatalf("helper ended with %v, want death by SIGTERM", err)
	}
	if alive(h.backend) {
		t.Fatalf("backend pid %d outlived SIGTERM", h.backend)
	}
}

// An unscoped Process.Stop takes at least 2 s (TERM, 2 s, KILL). With a
// 300 ms deadline the guard must end ggrun long before that.
func TestGuardStopIsBounded(t *testing.T) {
	h := startGuardHelper(t, "deadline", false)
	sent := time.Now()
	h.signal(t, syscall.SIGTERM)
	if err := h.wait(t, 30*time.Second); !diedOf(err, syscall.SIGTERM) {
		t.Fatalf("helper ended with %v, want death by SIGTERM", err)
	}
	if took := time.Since(sent); took > 1800*time.Millisecond {
		t.Fatalf("guard took %s to give up, want about its 300 ms deadline", took)
	}
}

// A second signal during the guard's stop must end ggrun at once.
func TestGuardSecondSignalEndsAtOnce(t *testing.T) {
	h := startGuardHelper(t, "loadwait", false)
	sent := time.Now()
	h.signal(t, syscall.SIGTERM)
	time.Sleep(300 * time.Millisecond)
	h.signal(t, syscall.SIGTERM)
	if err := h.wait(t, 30*time.Second); !diedOf(err, syscall.SIGTERM) {
		t.Fatalf("helper ended with %v, want death by SIGTERM", err)
	}
	if took := time.Since(sent); took > 1800*time.Millisecond {
		t.Fatalf("second SIGTERM ended ggrun after %s, want at once (the stop alone takes 2 s)", took)
	}
}

// After OwnShutdownSignals the guard no longer acts: SIGTERM reaches only the
// caller's own handler, and SIGHUP has its default action again. In both
// cases the TERM-ignoring backend is left alone (a guard stop would kill it).
func TestGuardHandOver(t *testing.T) {
	for _, sig := range []syscall.Signal{syscall.SIGTERM, syscall.SIGHUP} {
		t.Run(sig.String(), func(t *testing.T) {
			h := startGuardHelper(t, "handover", false)
			if _, err := readPIDFile(h.ready, 10*time.Second); err != nil {
				t.Fatal(err)
			}
			h.signal(t, sig)
			err := h.wait(t, 30*time.Second)
			switch sig {
			case syscall.SIGTERM:
				if err != nil {
					t.Fatalf("helper ended with %v, want its own handler's exit 0", err)
				}
			case syscall.SIGHUP:
				if !diedOf(err, syscall.SIGHUP) {
					t.Fatalf("helper ended with %v, want death by SIGHUP (default action)", err)
				}
			}
			if !alive(h.backend) {
				t.Fatalf("the guard stopped the backend after the hand-over (%s)", sig)
			}
		})
	}
}

func liveCount() int {
	guard.Lock()
	defer guard.Unlock()
	return len(guard.live)
}

// Every way a start ends must leave guard.live as it was, or a later guard
// stop would act on a reaped pid.
func TestGuardLiveTrackingIsCleaned(t *testing.T) {
	healthy, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer healthy.Close()
	go func() {
		_ = http.Serve(healthy, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
	}()
	up := healthy.Addr().(*net.TCPAddr).Port
	closed, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	down := closed.Addr().(*net.TCPAddr).Port
	_ = closed.Close()
	sleeper := []string{"/bin/sh", "-c", "exec sleep 60", "sh", "--host", "127.0.0.1"}
	base := liveCount()
	check := func(what string) {
		t.Helper()
		if got := liveCount(); got != base {
			t.Fatalf("%s: %d tracked backends, want %d", what, got, base)
		}
	}
	if _, err := StartWithTimeoutToOptions([]string{"/nonexistent/ggrun-no-such-backend"}, down, time.Second, io.Discard, io.Discard, StartOptions{}); err == nil {
		t.Fatal("start of a missing binary succeeded")
	}
	check("start failed")
	if _, err := StartWithTimeoutToOptions([]string{"/bin/sh", "-c", "exit 3"}, down, 10*time.Second, io.Discard, io.Discard, StartOptions{}); err == nil {
		t.Fatal("a backend that exits during startup was reported ready")
	}
	check("process exited by itself")
	if _, err := StartWithTimeoutToOptions(sleeper, down, 300*time.Millisecond, io.Discard, io.Discard, StartOptions{}); err == nil {
		t.Fatal("a backend that never answers was reported ready")
	}
	check("load timeout")
	p, err := StartWithTimeoutToOptions(sleeper, up, 10*time.Second, io.Discard, io.Discard, StartOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if got := liveCount(); got != base+1 {
		t.Fatalf("running backend: %d tracked, want %d", got, base+1)
	}
	_ = p.Stop()
	check("Stop")
	p, err = StartWithTimeoutToOptions(sleeper, up, 10*time.Second, io.Discard, io.Discard, StartOptions{})
	if err != nil {
		t.Fatal(err)
	}
	p.Kill()
	check("Kill")
}
