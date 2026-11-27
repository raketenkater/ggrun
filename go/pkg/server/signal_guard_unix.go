//go:build !windows

package server

import (
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

// Until a caller's own shutdown loop takes over, a SIGTERM or SIGHUP used to
// kill ggrun with Go's default action: no Process.Stop, no scope stop, and the
// backend's survival rested on the scope wrapper alone. The guard is installed
// with the first backend and stops every live backend before ggrun dies of the
// same signal. SIGINT is left alone: Ctrl+C has its own owners (claude_resume).
var guard struct {
	sync.Mutex
	ch        chan os.Signal
	installed bool
	claimed   bool
	stopping  bool
	live      map[*Process]struct{}
}

// guardStopDeadline bounds the guard's stop before ggrun dies of the signal
// anyway. An unscoped Process.Stop takes at most about 19 s (TERM, 2 s, KILL;
// a 15 s exit wait; a second 2 s TERM/KILL); a scoped one is bounded only by
// systemd. 30 s covers the unscoped worst case with margin and still ends
// well inside the shipped unit's TimeoutStopSec=60; the scope wrapper and the
// parent-death signal remain the backstop for whatever is left.
var guardStopDeadline = 30 * time.Second

// guardedSignals leaves an inherited "ignored" disposition alone: signal.Notify
// would replace it, and `nohup ggrun ...` must keep surviving a closed
// terminal.
func guardedSignals() []os.Signal {
	var out []os.Signal
	for _, s := range []os.Signal{syscall.SIGTERM, syscall.SIGHUP} {
		if !signal.Ignored(s) {
			out = append(out, s)
		}
	}
	return out
}

func installSignalGuard() {
	guard.Lock()
	defer guard.Unlock()
	if guard.claimed || guard.installed {
		return
	}
	guard.installed = true
	sigs := guardedSignals()
	if len(sigs) == 0 {
		return
	}
	guard.ch = make(chan os.Signal, 2)
	signal.Notify(guard.ch, sigs...)
	go runSignalGuard(guard.ch)
}

func runSignalGuard(ch chan os.Signal) {
	sig, ok := (<-ch).(syscall.Signal)
	if !ok {
		return
	}
	guard.Lock()
	if guard.claimed {
		// The caller's own loop received this signal too and owns shutdown.
		guard.Unlock()
		return
	}
	guard.stopping = true
	live := make([]*Process, 0, len(guard.live))
	for p := range guard.live {
		if p.scopeUnit == "" && p.exited() {
			continue // reaped: its pid may already belong to someone else
		}
		live = append(live, p)
	}
	guard.Unlock()
	stopped := make(chan struct{})
	go func() {
		var wg sync.WaitGroup
		for _, p := range live {
			wg.Add(1)
			go func(p *Process) {
				defer wg.Done()
				_ = p.Stop()
			}(p)
		}
		wg.Wait()
		close(stopped)
	}()
	select {
	case <-stopped:
	case <-time.After(guardStopDeadline):
	case again := <-ch:
		// The user's second signal always ends ggrun at once.
		if s, ok := again.(syscall.Signal); ok {
			sig = s
		}
	}
	dieOf(sig)
}

// dieOf ends ggrun by sig with the default action, as before the guard.
func dieOf(sig syscall.Signal) {
	signal.Reset(sig)
	_ = syscall.Kill(os.Getpid(), sig)
	time.Sleep(time.Second)
	os.Exit(128 + int(sig))
}

// OwnShutdownSignals hands SIGTERM and SIGHUP back to the caller, which must
// already have its own signal.Notify for the signals it handles. SIGHUP regains
// its default action. Call it where a serving loop starts owning shutdown.
func OwnShutdownSignals() {
	guard.Lock()
	defer guard.Unlock()
	guard.claimed = true
	if guard.ch != nil {
		signal.Stop(guard.ch)
	}
}

// blockIfStopping parks a caller once the guard is stopping backends for a
// signal: no new backend may start, and a launch whose readiness failed
// because the guard stopped it must not race on to a retry or its own exit
// code. The guard ends the process with the signal.
func blockIfStopping() {
	guard.Lock()
	stopping := guard.stopping
	guard.Unlock()
	if stopping {
		select {}
	}
}

func (p *Process) exited() bool {
	select {
	case <-p.done:
		return true
	default:
		return false
	}
}

func trackLive(p *Process) {
	guard.Lock()
	defer guard.Unlock()
	if guard.live == nil {
		guard.live = map[*Process]struct{}{}
	}
	guard.live[p] = struct{}{}
}

func untrackLive(p *Process) {
	guard.Lock()
	defer guard.Unlock()
	delete(guard.live, p)
}
