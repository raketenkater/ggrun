package main

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
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

// servingFatalLine is fatalBackendLine for the serving phase. At verbose log
// levels the backend logs request bodies and generated text, so a marker in a
// user's prompt or a model's answer must not count: only lines that begin like
// ggml's own abort output ("CUDA error: ...", "/src/ggml-cuda.cu:139: CUDA
// error", "terminate called ...") do.
func servingFatalLine(text string) string {
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if servingFatalRe.MatchString(line) {
			return line
		}
	}
	return ""
}

// servingFatalRe anchors the fatal markers at the start of a line, after at
// most a llama.cpp log prefix ("0.12.345.678 E ") and a "file.cu:139: " source
// location, which is how ggml writes them. A logged request body or generated
// text begins with something else.
var servingFatalRe = regexp.MustCompile(`^(?:\d+\.\d+\.\d+\.\d+ [A-Z] )?(?:\S+:\d+: )?(?:CUDA error|GGML_ASSERT\(|GGML_ABORT|ggml_abort|terminate called|Segmentation fault)`)

// servingBackend is what the serving watch needs from a running backend.
type servingBackend interface {
	IsRunning() bool
	Kill()
	LogSince(offset int) (string, int)
	CPUTicks() (uint64, bool)
}

// servingWatch bounds how long a backend may sit dead after printing a fatal
// error while serving.
type servingWatch struct {
	interval     time.Duration
	probeTimeout time.Duration
	// confirmAfter is how long, after a fatal line, /health must stay silent
	// with no CPU progress before the backend is stopped.
	confirmAfter time.Duration
}

var defaultServingWatch = servingWatch{interval: 2 * time.Second, probeTimeout: 5 * time.Second, confirmAfter: 20 * time.Second}

// waitForShutdownCrashOrWedge returns false on a shutdown signal and true when
// the backend has died. Dead includes a backend that printed a fatal ggml error
// and then hung: ggml's abort forks a backtrace helper, which can deadlock, and
// the process then never exits. Qwen3.6-35B-A3B on the bundled ik CUDA build
// did that after "CUDA error: an unsupported value or parameter was passed to
// the function" and Claude Code waited on it indefinitely. A fatal line alone is
// not enough; /health must also stop answering and the process tree stop using
// CPU, so text in a logged prompt never stops a working server.
func waitForShutdownCrashOrWedge(b servingBackend, sigCh <-chan os.Signal, healthURL string, w servingWatch) bool {
	ticker := time.NewTicker(w.interval)
	defer ticker.Stop()
	client := &http.Client{Timeout: w.probeTimeout}
	offset, carry, fatal := 0, "", ""
	var fatalSince time.Time
	var lastTicks uint64
	for {
		select {
		case <-sigCh:
			return false
		case <-ticker.C:
		}
		if !b.IsRunning() {
			return true
		}
		chunk, next := b.LogSince(offset)
		offset = next
		text := carry + chunk
		if i := strings.LastIndexByte(text, '\n'); i >= 0 {
			carry, text = text[i+1:], text[:i+1]
		} else {
			carry, text = text, ""
		}
		if len(carry) > 4096 {
			carry = carry[len(carry)-4096:]
		}
		if fatal == "" {
			if fatal = servingFatalLine(text); fatal == "" {
				continue
			}
			fatalSince = time.Now()
			lastTicks, _ = b.CPUTicks()
		}
		if backendAnswers(client, healthURL) {
			fatal = ""
			continue
		}
		if ticks, ok := b.CPUTicks(); !ok || ticks != lastTicks {
			lastTicks, fatalSince = ticks, time.Now()
			continue
		}
		if time.Since(fatalSince) >= w.confirmAfter {
			fmt.Fprintf(os.Stderr, "[launch] backend failed and stopped responding: %s\n", fatal)
			b.Kill()
			return true
		}
	}
}

// processWatch adapts a launched backend to watchedBackend.
type processWatch struct{ p *server.Process }

func (w processWatch) IsRunning() bool { return w.p.IsRunning() }
func (w processWatch) Stop() error     { return w.p.Stop() }

func (w processWatch) Kill() { w.p.Kill() }

func (w processWatch) LogSince(offset int) (string, int) {
	if w.p.LogBuf == nil {
		return "", 0
	}
	return w.p.LogBuf.Since(offset)
}

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

// procEntry is one /proc/<pid>/stat row reduced to what the watchdog needs.
type procEntry struct {
	ppid  int
	comm  string
	ticks uint64
}

// processTreeCPUTicks sums utime+stime of the backend processes under root.
// The launched pid is a scope wrapper; the backend is its child. Linux only;
// elsewhere ok=false and the watchdog never treats silence as a wedge.
func processTreeCPUTicks(root int) (uint64, bool) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return 0, false
	}
	procs := map[int]procEntry{}
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
		start, end := strings.IndexByte(stat, '('), strings.LastIndexByte(stat, ')')
		if start < 0 || end < start {
			continue
		}
		fields := strings.Fields(stat[end+1:])
		if len(fields) < 13 {
			continue
		}
		ppid, _ := strconv.Atoi(fields[1])
		utime, _ := strconv.ParseUint(fields[11], 10, 64)
		stime, _ := strconv.ParseUint(fields[12], 10, 64)
		procs[pid] = procEntry{ppid: ppid, comm: stat[start+1 : end], ticks: utime + stime}
	}
	return backendTreeTicks(procs, root)
}

// Launch helpers that live as long as the backend. The memory-scope wrapper's
// watcher subshell polls with `sleep 1` for the backend's whole life; its fork
// ticks made a frozen backend (SIGSTOP, 10 minutes, live) look busy, so the
// wedge rule never fired.
var watchdogHelperComms = map[string]bool{"sh": true, "bash": true, "dash": true, "sleep": true, "setsid": true, "systemd-run": true}

func backendTreeTicks(procs map[int]procEntry, root int) (uint64, bool) {
	if _, ok := procs[root]; !ok {
		return 0, false
	}
	var total uint64
	for pid, p := range procs {
		if watchdogHelperComms[p.comm] {
			continue
		}
		for cur, hops := pid, 0; cur > 0 && hops < 64; cur, hops = procs[cur].ppid, hops+1 {
			if cur == root {
				total += p.ticks
				break
			}
		}
	}
	return total, true
}
