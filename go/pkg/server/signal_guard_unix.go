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
	ch       chan os.Signal
	claimed  bool
	stopping bool
	live     map[*Process]struct{}
}

func installSignalGuard() {
	guard.Lock()
	defer guard.Unlock()
	if guard.claimed || guard.ch != nil {
		return
	}
	guard.ch = make(chan os.Signal, 1)
	signal.Notify(guard.ch, syscall.SIGTERM, syscall.SIGHUP)
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
		live = append(live, p)
	}
	guard.Unlock()
	var wg sync.WaitGroup
	for _, p := range live {
		wg.Add(1)
		go func(p *Process) {
			defer wg.Done()
			_ = p.Stop()
		}(p)
	}
	wg.Wait()
	// Die of the signal that was asked for, as before the guard existed.
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
