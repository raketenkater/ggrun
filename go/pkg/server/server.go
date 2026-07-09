package server

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/raketenkater/ggrun/pkg/libhub"
)

// Process wraps a running llama-server subprocess.
type Process struct {
	Cmd    *exec.Cmd
	Port   int
	cancel context.CancelFunc
	LogBuf *threadSafeBuffer // captured stderr for post-launch probe
	waitCh chan error        // receives cmd.Wait() exactly once
}

// threadSafeBuffer is a bytes.Buffer protected by a mutex for concurrent writes.
type threadSafeBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *threadSafeBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *threadSafeBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func (b *threadSafeBuffer) Tail(max int) string {
	b.mu.Lock()
	defer b.mu.Unlock()
	data := b.buf.Bytes()
	if len(data) > max {
		data = data[len(data)-max:]
	}
	return string(data)
}

// Start launches llama-server with the given args and waits for it to be ready.
func Start(args []string, port int) (*Process, error) {
	return StartWithTimeout(args, port, 60*time.Second)
}

// StartWithTimeout launches llama-server with a custom readiness timeout. The
// backend's ongoing logs stream to the process's own stdout/stderr.
func StartWithTimeout(args []string, port int, timeout time.Duration) (*Process, error) {
	return StartWithTimeoutTo(args, port, timeout, os.Stdout, os.Stderr)
}

// StartWithTimeoutTo is StartWithTimeout but streams the backend's ongoing logs
// to termOut/termErr instead of os.Stdout/os.Stderr. claude-code mode passes a
// log file here so the backend's per-request log spam doesn't bleed into Claude
// Code's terminal UI once ggrun hands the terminal to the `claude` client.
func StartWithTimeoutTo(args []string, port int, timeout time.Duration, termOut, termErr io.Writer) (*Process, error) {
	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	setSysProcAttr(cmd)
	// Capture stdout/stderr to a buffer for the post-launch probe. On a TTY we
	// keep the screen clean during model load only when writing directly to the
	// terminal; a caller-provided log file should receive startup output too.
	logBuf := &threadSafeBuffer{}
	tty := stdoutIsTTY()
	streamFromStart := streamLogsFromStart(tty, termOut, termErr)
	logStartupEvents := dedicatedLogWriter(termOut, termErr)
	live := &atomic.Bool{}
	live.Store(streamFromStart)
	cmd.Stdout = &gatedWriter{buf: logBuf, term: termOut, live: live}
	cmd.Stderr = &gatedWriter{buf: logBuf, term: termErr, live: live}

	// Ensure CUDA device enumeration matches nvidia-smi PCI bus order.
	// Without this, llama-server may enumerate GPUs differently from our
	// detection, causing -ot / --tensor-split flags to target wrong devices.
	//
	// We filter out any existing CUDA_DEVICE_ORDER before adding ours,
	// because appending a duplicate key has undefined behaviour in CUDA.
	env := os.Environ()
	filtered := make([]string, 0, len(env)+1)
	for _, e := range env {
		if !strings.HasPrefix(e, "CUDA_DEVICE_ORDER=") {
			filtered = append(filtered, e)
		}
	}
	filtered = append(filtered, "CUDA_DEVICE_ORDER=PCI_BUS_ID")
	// A shared-library backend build finds its co-located libs via the hub, not
	// its build-machine RUNPATH.
	cmd.Env = libhub.ApplyToChildEnv(filtered)

	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("start server: %w", err)
	}

	p := &Process{Cmd: cmd, Port: port, cancel: cancel, LogBuf: logBuf, waitCh: make(chan error, 1)}
	go func() { p.waitCh <- cmd.Wait() }()

	start := time.Now()
	logStartupEvent(logStartupEvents, termErr, "[launch] health check: polling http://127.0.0.1:%d/health then /v1/models (timeout %s)", port, timeout)
	var stopSpin chan struct{}
	if tty {
		stopSpin = make(chan struct{})
		go spinUntilReady(stopSpin, logBuf, start, timeout, cmd.Process.Pid, args)
	}
	err := p.waitReady(timeout)
	if tty {
		close(stopSpin)
		fmt.Fprint(os.Stderr, "\r\033[K") // clear the spinner line
	}
	if err != nil {
		logStartupEvent(logStartupEvents, termErr, "[launch] health check failed after %s: %v", time.Since(start).Round(time.Second), err)
		if tty {
			fmt.Fprintln(os.Stderr, "[launch] backend failed to start; last output:")
			fmt.Fprintln(os.Stderr, tailLines(logBuf.String(), 20))
		}
		p.Stop()
		return p, fmt.Errorf("server not ready: %w", err)
	}
	live.Store(true) // backend is up — stream its logs from here on
	logStartupEvent(logStartupEvents, termErr, "[launch] health check OK after %s", time.Since(start).Round(time.Second))
	if tty {
		fmt.Fprintf(os.Stderr, "[launch] model loaded - server ready in %s\n", time.Since(start).Round(time.Second))
	}
	return p, nil
}

func (p *Process) waitReady(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	url := fmt.Sprintf("http://localhost:%d/health", p.Port)
	for time.Now().Before(deadline) {
		// Fail fast when the process dies during startup instead of polling
		// the health endpoint until the full (model-size-scaled) timeout.
		select {
		case err := <-p.waitCh:
			p.waitCh <- err // keep available for Stop()
			if err != nil {
				return fmt.Errorf("server process exited during startup: %v", err)
			}
			return fmt.Errorf("server process exited during startup")
		default:
		}
		resp, err := http.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		// Fallback: try /v1/models as well
		resp, err = http.Get(fmt.Sprintf("http://localhost:%d/v1/models", p.Port))
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("timeout waiting for server on port %d", p.Port)
}

// Stop terminates the server process gracefully then forcefully.
func (p *Process) Stop() error {
	p.cancel()
	if p.Cmd.Process != nil {
		killProcessTree(p.Cmd.Process.Pid)
	}
	// Wait with timeout — don't block forever if process hangs during cleanup.
	select {
	case err := <-p.waitCh:
		p.waitCh <- err // allow repeated Stop() calls
		return err
	case <-time.After(15 * time.Second):
		return fmt.Errorf("process did not exit within 15s")
	}
}

// IsRunning returns true if the process is still alive.
func (p *Process) IsRunning() bool {
	if p == nil || p.Cmd == nil || p.Cmd.Process == nil {
		return false
	}
	return isProcessAlive(p.Cmd.Process.Pid) && p.Cmd.ProcessState == nil
}

// QueryModels returns the models endpoint response.
func (p *Process) QueryModels() ([]byte, error) {
	url := fmt.Sprintf("http://localhost:%d/v1/models", p.Port)
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func stdoutIsTTY() bool {
	fi, err := os.Stdout.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

func streamLogsFromStart(tty bool, termOut, termErr io.Writer) bool {
	if !tty {
		return true
	}
	return dedicatedLogWriter(termOut, termErr)
}

func dedicatedLogWriter(termOut, termErr io.Writer) bool {
	return !sameFileWriter(termOut, os.Stdout) || !sameFileWriter(termErr, os.Stderr)
}

func logStartupEvent(enabled bool, w io.Writer, format string, args ...any) {
	if !enabled || w == nil {
		return
	}
	fmt.Fprintf(w, format+"\n", args...)
}

func sameFileWriter(w io.Writer, f *os.File) bool {
	of, ok := w.(*os.File)
	return ok && of == f
}

// gatedWriter always captures to buf; it forwards to term only once live is set,
// so the backend's load-time log spam stays hidden behind the startup spinner
// until the server is ready (after which subsequent output streams through).
type gatedWriter struct {
	buf  *threadSafeBuffer
	term io.Writer
	live *atomic.Bool
}

func (g *gatedWriter) Write(p []byte) (int, error) {
	if g.buf != nil {
		g.buf.Write(p)
	}
	if g.live.Load() {
		return g.term.Write(p)
	}
	return len(p), nil
}

var spinnerFrames = []string{"|", "/", "-", "\\"}

// spinUntilReady animates a single status line on stderr while the backend loads.
// It combines backend log phase text with /proc fd offsets, which gives useful
// progress even when llama.cpp itself does not print a byte counter.
func spinUntilReady(stop <-chan struct{}, log *threadSafeBuffer, start time.Time, timeout time.Duration, pid int, args []string) {
	t := time.NewTicker(time.Second)
	defer t.Stop()
	progress := newLoadProgressTracker(pid, args)
	lastLine := ""
	for i := 0; ; i++ {
		select {
		case <-stop:
			return
		case <-t.C:
			logTail := log.Tail(64 * 1024)
			line := fitStatusLine(fmt.Sprintf("%s  %s",
				spinnerFrames[i%len(spinnerFrames)],
				startupStatus(logTail, time.Since(start), timeout, progress.Snapshot())))
			if line != lastLine {
				fmt.Fprintf(os.Stderr, "\r\033[K%s", line)
				lastLine = line
			}
		}
	}
}

func fitStatusLine(line string) string {
	line = strings.Join(strings.Fields(line), " ")
	cols := terminalColumns()
	if cols <= 1 {
		return ""
	}
	max := cols - 1
	runes := []rune(line)
	if len(runes) <= max {
		return line
	}
	if max <= 3 {
		return string(runes[:max])
	}
	return string(runes[:max-3]) + "..."
}

func terminalColumns() int {
	if cols := terminalColumnsOS(); cols > 0 {
		return cols
	}
	if v, err := strconv.Atoi(os.Getenv("COLUMNS")); err == nil && v > 0 {
		return v
	}
	return 100
}

type loadProgress struct {
	Done  int64
	Total int64
}

func startupStatus(logText string, elapsed, timeout time.Duration, progress loadProgress) string {
	parts := make([]string, 0, 5)
	if progress.Total > 0 {
		pct := progressPercent(progress)
		parts = append(parts, fmt.Sprintf("%s %3d%%", progressBar(pct, 20), pct))
	}
	parts = append(parts, startupPhase(logText))
	if timeout > 0 {
		parts = append(parts, fmt.Sprintf("%s/%s", elapsed.Round(time.Second), timeout.Round(time.Second)))
	} else {
		parts = append(parts, elapsed.Round(time.Second).String())
	}
	if progress.Total > 0 {
		parts = append(parts, fmt.Sprintf("read %s/%s", formatGiB(progress.Done), formatGiB(progress.Total)))
	}
	if line := latestBackendLine(logText); line != "" {
		parts = append(parts, truncateStatus(line, 90))
	}
	return strings.Join(parts, " | ")
}

func progressPercent(p loadProgress) int {
	if p.Total <= 0 || p.Done <= 0 {
		return 0
	}
	if p.Done >= p.Total {
		return 100
	}
	return int((p.Done * 100) / p.Total)
}

func progressBar(percent, width int) string {
	if width <= 0 {
		return "[]"
	}
	if percent < 0 {
		percent = 0
	}
	if percent > 100 {
		percent = 100
	}
	filled := (percent*width + 50) / 100
	if filled > width {
		filled = width
	}
	return "[" + strings.Repeat("#", filled) + strings.Repeat("-", width-filled) + "]"
}

func formatGiB(n int64) string {
	if n < 0 {
		n = 0
	}
	return fmt.Sprintf("%.1fGiB", float64(n)/(1024*1024*1024))
}

func latestBackendLine(logText string) string {
	lines := strings.Split(strings.TrimSpace(logText), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}
		line = strings.Join(strings.Fields(line), " ")
		return line
	}
	return ""
}

func truncateStatus(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	if max <= 3 {
		return s[:max]
	}
	return s[:max-3] + "..."
}

// startupPhase turns the backend's recent log output into a short status label.
func startupPhase(logText string) string {
	l := strings.ToLower(logText)
	switch {
	case strings.Contains(l, "server is listening"), strings.Contains(l, "model loaded"):
		return "finishing startup"
	case strings.Contains(l, "warming up"):
		return "warming up the model"
	case strings.Contains(l, "load_tensors"), strings.Contains(l, "loading model"):
		return "loading model weights"
	case strings.Contains(l, "pinned host memory"), strings.Contains(l, "allocating"):
		return "pinning host memory (large MoE; can take a few minutes)"
	default:
		return "starting backend"
	}
}

// tailLines returns the last n lines of s (for dumping backend output on failure).
func tailLines(s string, n int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

type loadProgressTracker struct {
	pid   int
	paths map[string]int64
	total int64
}

func newLoadProgressTracker(pid int, args []string) *loadProgressTracker {
	paths, total := modelShardPaths(modelPathFromArgs(args))
	return &loadProgressTracker{pid: pid, paths: paths, total: total}
}

func (t *loadProgressTracker) Snapshot() loadProgress {
	if t == nil || t.pid <= 0 || t.total <= 0 {
		return loadProgress{}
	}
	done := t.fdPositions()
	if done == 0 {
		done = procRChar(t.pid)
	}
	if done > t.total {
		done = t.total
	}
	return loadProgress{Done: done, Total: t.total}
}

func (t *loadProgressTracker) fdPositions() int64 {
	dir := fmt.Sprintf("/proc/%d/fd", t.pid)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	byPath := map[string]int64{}
	for _, entry := range entries {
		fd := entry.Name()
		target, err := os.Readlink(filepath.Join(dir, fd))
		if err != nil {
			continue
		}
		size, ok := t.paths[target]
		if !ok {
			continue
		}
		pos := fdPosition(t.pid, fd)
		if pos > size {
			pos = size
		}
		if pos > byPath[target] {
			byPath[target] = pos
		}
	}
	var done int64
	for _, pos := range byPath {
		done += pos
	}
	return done
}

func fdPosition(pid int, fd string) int64 {
	f, err := os.Open(fmt.Sprintf("/proc/%d/fdinfo/%s", pid, fd))
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "pos:") {
			continue
		}
		v := strings.TrimSpace(strings.TrimPrefix(line, "pos:"))
		n, err := strconv.ParseInt(v, 10, 64)
		if err == nil {
			return n
		}
	}
	return 0
}

func procRChar(pid int) int64 {
	f, err := os.Open(fmt.Sprintf("/proc/%d/io", pid))
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "rchar:") {
			continue
		}
		v := strings.TrimSpace(strings.TrimPrefix(line, "rchar:"))
		n, err := strconv.ParseInt(v, 10, 64)
		if err == nil {
			return n
		}
	}
	return 0
}

func modelPathFromArgs(args []string) string {
	for i, arg := range args {
		if arg == "-m" || arg == "--model" {
			if i+1 < len(args) {
				return args[i+1]
			}
			return ""
		}
		if strings.HasPrefix(arg, "-m=") {
			return strings.TrimPrefix(arg, "-m=")
		}
		if strings.HasPrefix(arg, "--model=") {
			return strings.TrimPrefix(arg, "--model=")
		}
	}
	return ""
}

var splitGGUFName = regexp.MustCompile(`(?i)^(.*)-([0-9]+)-of-([0-9]+)\.gguf$`)

func modelShardPaths(modelPath string) (map[string]int64, int64) {
	paths := map[string]int64{}
	if modelPath == "" {
		return paths, 0
	}
	dir := filepath.Dir(modelPath)
	base := filepath.Base(modelPath)
	match := splitGGUFName.FindStringSubmatch(base)
	if match != nil {
		totalParts, err := strconv.Atoi(match[3])
		if err == nil && totalParts > 0 {
			var total int64
			for i := 1; i <= totalParts; i++ {
				name := fmt.Sprintf("%s-%0*d-of-%s.gguf", match[1], len(match[2]), i, match[3])
				path := filepath.Join(dir, name)
				if info, err := os.Stat(path); err == nil && !info.IsDir() {
					paths[path] = info.Size()
					total += info.Size()
				}
			}
			if total > 0 {
				return paths, total
			}
		}
	}
	if info, err := os.Stat(modelPath); err == nil && !info.IsDir() {
		paths[modelPath] = info.Size()
		return paths, info.Size()
	}
	return paths, 0
}
