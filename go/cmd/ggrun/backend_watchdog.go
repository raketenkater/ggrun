package main

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/server"
)

// watchedBackend is what the launch watchdog needs from a running backend.
type watchedBackend interface {
	IsRunning() bool
	Stop() error
	Log() string
	// CPUTicks is the backend process tree's cumulative CPU time; ok is false
	// when it cannot be read on this platform.
	CPUTicks() (ticks uint64, ok bool)
}

// backendWatch bounds how long a launch step may wait on a backend that has
// stopped working.
type backendWatch struct {
	interval     time.Duration // how often to check
	probeTimeout time.Duration // one /health request
	// wedgedAfter is how long /health may stay silent with NO CPU progress
	// before the backend counts as wedged. /health alone is not enough: ik's
	// handler waits on the main task queue, so one long llama_decode (about
	// 100 s for a 2048-token batch on CPU experts) legitimately stalls it.
	wedgedAfter time.Duration
	// stopGrace bounds the wait for the step to return once the backend has
	// been stopped.
	stopGrace time.Duration
}

var defaultBackendWatch = backendWatch{
	interval:     5 * time.Second,
	probeTimeout: 10 * time.Second,
	wedgedAfter:  3 * time.Minute,
	stopGrace:    time.Minute,
}

// fatalBackendMarkers are lines ggml prints immediately before aborting. The
// abort itself can then hang in ggml's backtrace handler (it forks a child and
// waits), leaving a live process that answers nothing: MiniMax-M3 on the
// bundled ik CUDA build did exactly that, and the launch waited out a
// 20-minute canary timeout.
var fatalBackendMarkers = []string{
	"CUDA error:",
	"GGML_ASSERT(",
	"GGML_ABORT",
	"ggml_abort",
	"Segmentation fault",
	"terminate called",
}

func fatalBackendLine(log string) string {
	for _, line := range strings.Split(log, "\n") {
		for _, marker := range fatalBackendMarkers {
			if strings.Contains(line, marker) {
				return strings.TrimSpace(line)
			}
		}
	}
	return ""
}

// runWatchingBackend runs step (a canary or verification that talks to the
// backend) and stops the backend as soon as it has exited, printed a fatal
// error, or is wedged. Stopping the backend closes its sockets, which is what
// unblocks step's pending request. A backend that is merely slow keeps running.
//
// The whole log is checked, not only lines written during step: the buffer
// belongs to this one backend process, and ggml aborts right after printing
// these lines, so a fatal line printed earlier (the MiniMax abort happened
// during calibration, before the canary) still means the process is dead.
func runWatchingBackend(b watchedBackend, healthURL string, w backendWatch, step func() error) error {
	done := make(chan error, 1)
	go func() { done <- step() }()

	client := &http.Client{Timeout: w.probeTimeout}
	ticker := time.NewTicker(w.interval)
	defer ticker.Stop()
	var silentSince time.Time
	lastTicks, ticksOK := b.CPUTicks()
	for {
		select {
		case err := <-done:
			return err
		case <-ticker.C:
		}
		cause := ""
		if !b.IsRunning() {
			cause = "backend process exited"
		} else if line := fatalBackendLine(b.Log()); line != "" {
			cause = "backend reported a fatal error: " + line
		} else if backendAnswers(client, healthURL) {
			silentSince = time.Time{}
		} else {
			ticks, ok := b.CPUTicks()
			progressing := !ok || !ticksOK || ticks != lastTicks
			lastTicks, ticksOK = ticks, ok
			switch {
			case progressing:
				silentSince = time.Time{}
			case silentSince.IsZero():
				silentSince = time.Now()
			case time.Since(silentSince) >= w.wedgedAfter:
				cause = fmt.Sprintf("backend stopped answering /health and used no CPU for %s", w.wedgedAfter)
			}
		}
		if cause == "" {
			continue
		}
		_ = b.Stop()
		select {
		case err := <-done:
			if err != nil {
				return fmt.Errorf("%s; stopped it (%v)", cause, err)
			}
			return fmt.Errorf("%s; stopped it", cause)
		case <-time.After(w.stopGrace):
			return fmt.Errorf("%s; stopped it, and the pending request did not return", cause)
		}
	}
}

// backendAnswers reports whether the backend's HTTP server answered at all. Any
// status counts: 503 while loading or with no free slot is still an answer.
func backendAnswers(client *http.Client, url string) bool {
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	_ = resp.Body.Close()
	return true
}

// processWatch adapts a launched backend to watchedBackend.
type processWatch struct{ p *server.Process }

func (w processWatch) IsRunning() bool { return w.p.IsRunning() }
func (w processWatch) Stop() error     { return w.p.Stop() }

func (w processWatch) Log() string {
	if w.p.LogBuf == nil {
		return ""
	}
	return w.p.LogBuf.String()
}

func (w processWatch) CPUTicks() (uint64, bool) {
	if w.p.Cmd == nil || w.p.Cmd.Process == nil {
		return 0, false
	}
	return processTreeCPUTicks(w.p.Cmd.Process.Pid)
}

// processTreeCPUTicks sums utime+stime over root and its descendants. The
// launched pid is a scope wrapper; the backend is its child. Linux only.
func processTreeCPUTicks(root int) (uint64, bool) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return 0, false
	}
	parent := map[int]int{}
	ticks := map[int]uint64{}
	for _, entry := range entries {
		pid, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}
		data, err := os.ReadFile(filepath.Join("/proc", entry.Name(), "stat"))
		if err != nil {
			continue
		}
		// The command name is parenthesised and may contain spaces.
		stat := string(data)
		end := strings.LastIndexByte(stat, ')')
		if end < 0 {
			continue
		}
		fields := strings.Fields(stat[end+1:])
		if len(fields) < 13 {
			continue
		}
		ppid, _ := strconv.Atoi(fields[1])
		utime, _ := strconv.ParseUint(fields[11], 10, 64)
		stime, _ := strconv.ParseUint(fields[12], 10, 64)
		parent[pid] = ppid
		ticks[pid] = utime + stime
	}
	if _, ok := ticks[root]; !ok {
		return 0, false
	}
	var total uint64
	for pid, t := range ticks {
		for cur, hops := pid, 0; cur > 0 && hops < 64; cur, hops = parent[cur], hops+1 {
			if cur == root {
				total += t
				break
			}
		}
	}
	return total, true
}
