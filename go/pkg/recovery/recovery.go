package recovery

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
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

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
)

// FailureType classifies how a server load failed.
type FailureType string

var errPlacementPromotion = errors.New("placement promotion restart")

const (
	FailureOOM          FailureType = "oom"
	FailurePinnedFail   FailureType = "pinned_fail"
	FailurePinnedCap    FailureType = "pinned_cap_exceeded"
	FailurePinnedHang   FailureType = "pinned_hang"
	FailureCUDAOOM      FailureType = "cuda_oom"
	FailureRAMOOM       FailureType = "ram_oom"
	FailureUnknownModel FailureType = "unknown_model"
	// FailureBackendCapability is a deterministic mismatch between the launch
	// args and what the active backend build supports — e.g. a CPU-only
	// llama-server told to offload to CUDA0 ("unknown buffer type"). Restarting
	// the identical command can never succeed, so the launcher fails fast.
	FailureBackendCapability FailureType = "backend_capability"
	FailureUnknown           FailureType = "unknown"
)

// deterministic reports whether a failure will recur identically on restart, so
// the launcher should fail fast instead of burning MaxRestarts with backoff.
// FailureUnknownModel is deterministic only once any ik->mainline fallback has
// already been attempted (handled in Run before this check).
func (ft FailureType) deterministic() bool {
	return ft == FailureBackendCapability || ft == FailureUnknownModel
}

// Launcher wraps server startup with crash recovery and fallback.
type Launcher struct {
	BinaryPath    string
	Args          []string
	FallbackPath  string // mainline llama-server if ik_llama fails
	MaxRestarts   int
	BackoffBase   time.Duration
	HealthTimeout time.Duration
	KeepAlive     bool
	OnLog         func(string)
	OnFailure     func(FailureType, string)
	OnRestart     func(int, time.Duration)
	OnFallback    func(string)
	OnCUDAOOM     func(device int, allocMB int, args []string) ([]string, *placement.CacheEntry, bool)
	OnPromote     func(logPath string, args []string) ([]string, bool)

	// Quiet keeps the backend's stdout out of the terminal (it still goes to the
	// per-run log file). Used by Claude Code mode, where ggrun hands the terminal
	// to the `claude` client and backend logs must not bleed into its UI.
	Quiet bool

	// LogPath, if set, is where the backend log is written (a stable, discoverable
	// location like .logs/ggrun-server-<port>.log). Empty falls back to a temp file.
	LogPath string

	PlacementCachePath string
	// SuccessEntry, if set, is the computed placement to persist to
	// PlacementCachePath the first time the server loads healthy — so a launch
	// that lands right is reused verbatim next time (OOM-recovery overwrites it
	// with the corrected placement if it has to intervene).
	SuccessEntry     *placement.CacheEntry
	ProbeCacheDir    string
	ProbeModel       *placement.ModelProfile
	ProbeCtxSize     int
	ProbeUBatchSize  int
	ProbeKVQuality   string
	ProbeKVPlacement string
	ProbeBackendTag  string
	ProbeGPUs        []detect.GPU

	lastLogPath string // log written by the most recent runOnce
}

// DefaultLauncher returns a launcher with sensible defaults.
func DefaultLauncher(binaryPath string, args []string) *Launcher {
	return &Launcher{
		BinaryPath:    binaryPath,
		Args:          args,
		MaxRestarts:   5,
		BackoffBase:   2 * time.Second,
		HealthTimeout: 60 * time.Second,
	}
}

// Run starts the server with crash recovery. Blocks until the process exits.
func (l *Launcher) Run(ctx context.Context) error {
	restartCount := 0
	cudaOOMRetries := 0
	backoff := l.BackoffBase
	binaryPath := l.BinaryPath

	for {
		if err := l.runOnce(ctx, binaryPath, restartCount); err != nil {
			if errors.Is(err, errPlacementPromotion) {
				restartCount = 0
				backoff = l.BackoffBase
				continue
			}

			// Shutdown requested: the child was killed by context cancellation,
			// not a crash. Exit immediately without fallback/restart churn.
			if ctx.Err() != nil {
				return ctx.Err()
			}

			// Check for known failure types from stderr log
			ft, msg := l.parseLoadFailure()
			if l.OnFailure != nil {
				l.OnFailure(ft, msg)
			}

			if ft == FailureCUDAOOM && cudaOOMRetries < 2 && l.OnCUDAOOM != nil {
				device, allocMB, ok := ParseCUDAOOM(msg)
				if ok {
					if newArgs, entry, retry := l.OnCUDAOOM(device, allocMB, append([]string(nil), l.Args...)); retry {
						l.Args = newArgs
						// Don't persist the derated placement yet — it has never
						// loaded. Make it the success candidate; handleHealthy
						// persists it only once the relaunch proves itself.
						if entry != nil {
							l.SuccessEntry = entry
						}
						cudaOOMRetries++
						restartCount++
						continue
					}
				}
			}

			// Try ik_llama -> mainline fallback for unknown model
			if ft == FailureUnknownModel && l.FallbackPath != "" && binaryPath == l.BinaryPath {
				if l.OnFallback != nil {
					l.OnFallback(l.FallbackPath)
				}
				binaryPath = l.FallbackPath
				restartCount = 0
				backoff = l.BackoffBase
				continue
			}

			// Deterministic failures recur identically on every restart. Once any
			// ik->mainline fallback has been exhausted, fail fast with the real
			// message instead of burning MaxRestarts with exponential backoff.
			if ft.deterministic() {
				return fmt.Errorf("%s: %s", ft, msg)
			}

			// Check if we should restart
			if restartCount >= l.MaxRestarts {
				return fmt.Errorf("max restarts (%d) exceeded: %s", l.MaxRestarts, msg)
			}

			if !l.KeepAlive {
				return fmt.Errorf("server failed: %s", msg)
			}

			// Backoff and restart
			restartCount++
			if l.OnRestart != nil {
				l.OnRestart(restartCount, backoff)
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
			// Cap backoff at 30s
			backoff *= 2
			if backoff > 30*time.Second {
				backoff = 30 * time.Second
			}
			continue
		}
		// Process exited normally
		return nil
	}
}

func (l *Launcher) runOnce(ctx context.Context, binaryPath string, restartCount int) error {
	var logFile *os.File
	var err error
	if l.LogPath != "" {
		logFile, err = os.Create(l.LogPath)
	} else {
		logFile, err = os.CreateTemp("", "ggrun-launch-*.log")
	}
	if err != nil {
		return err
	}
	defer logFile.Close()
	logPath := logFile.Name()
	// Remember our own log so failure parsing never reads a log written by a
	// concurrently running instance.
	l.lastLogPath = logPath

	cmd := exec.CommandContext(ctx, binaryPath, l.Args...)
	cmd.SysProcAttr = setProcessGroupAttr()

	// llama.cpp writes ~all of its logs (load progress, errors) to STDERR. During
	// interactive startup, keep the terminal to a single progress line while still
	// writing the full backend log to disk. Once healthy, raw backend logs stream
	// again. Non-TTY runs keep the old tee behavior for scripts and benchmarks.
	var startupLog *startupLogCapture
	if l.Quiet {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	} else if stderrIsTTY() {
		startupLog = newStartupLogCapture(logFile)
		cmd.Stdout = startupLog.writer(os.Stdout)
		cmd.Stderr = startupLog.writer(os.Stderr)
	} else {
		cmd.Stdout = io.MultiWriter(os.Stdout, logFile)
		cmd.Stderr = io.MultiWriter(os.Stderr, logFile)
	}

	cmd.Env = server.ChildEnv(os.Environ(), l.Args)

	if err := cmd.Start(); err != nil {
		return err
	}

	stopProgress := func() {}
	if startupLog != nil {
		stop := make(chan struct{})
		go spinStartupProgress(stop, startupLog, time.Now(), l.HealthTimeout, cmd.Process.Pid, l.Args)
		stopProgress = func() {
			close(stop)
			fmt.Fprint(os.Stderr, "\r\033[K")
		}
	}
	defer stopProgress()

	probeWritten := false
	writeFailureProbe := func() {
		if probeWritten {
			return
		}
		probeWritten = true
		_ = logFile.Sync()
		l.writeProbeCache(logPath)
	}

	// Ensure the full process group is killed on any exit path
	// (context cancellation, health timeout, crash, etc.).
	defer func() {
		if cmd.Process != nil {
			killProcGroup(cmd.Process.Pid)
			cmd.Wait()
		}
	}()

	// Wait for health check or process death
	port := l.extractPort()
	healthURL := fmt.Sprintf("http://127.0.0.1:%s/health", port)
	modelsURL := fmt.Sprintf("http://127.0.0.1:%s/v1/models", port)

	deadline := time.Now().Add(l.HealthTimeout)
	for time.Now().Before(deadline) {
		// Honor shutdown promptly, even while the model is still loading.
		select {
		case <-ctx.Done():
			if cmd.Process != nil {
				killProcGroup(cmd.Process.Pid)
			}
			writeFailureProbe()
			return ctx.Err()
		default:
		}

		// Check if process died
		if cmd.Process != nil {
			if !procAlive(cmd.Process.Pid) {
				// Process died before health check
				writeFailureProbe()
				return fmt.Errorf("process died during startup")
			}
		}

		// Try health endpoint
		if resp, err := doHTTPGet(healthURL); err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				stopProgress()
				stopProgress = func() {}
				if startupLog != nil {
					startupLog.setLive(true)
					fmt.Fprintf(os.Stderr, "[launch] model loaded - server ready in %s\n", time.Since(startupLog.start).Round(time.Second))
				}
				// Server is healthy. Write probe cache, then optionally restart once
				// with a placement promoted from the measured runtime values.
				if l.handleHealthy(logPath) {
					return errPlacementPromotion
				}
				return cmd.Wait()
			}
		}
		if resp, err := doHTTPGet(modelsURL); err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				stopProgress()
				stopProgress = func() {}
				if startupLog != nil {
					startupLog.setLive(true)
					fmt.Fprintf(os.Stderr, "[launch] model loaded - server ready in %s\n", time.Since(startupLog.start).Round(time.Second))
				}
				if l.handleHealthy(logPath) {
					return errPlacementPromotion
				}
				return cmd.Wait()
			}
		}
		time.Sleep(500 * time.Millisecond)
	}

	// Health timeout — kill process
	if cmd.Process != nil {
		killProcGroup(cmd.Process.Pid)
	}
	writeFailureProbe()
	return fmt.Errorf("health timeout")
}

func (l *Launcher) handleHealthy(logPath string) bool {
	l.writeProbeCache(logPath)
	// Persist the placement that is actually serving. Overwrite: SuccessEntry
	// tracks OOM-recovery derates, and after one the on-disk cache still holds
	// the plan that failed — reusing it would re-crash the next launch.
	if l.PlacementCachePath != "" && l.SuccessEntry != nil {
		_ = placement.SavePlacementCache(l.PlacementCachePath, l.SuccessEntry)
	}
	if l.OnPromote == nil {
		return false
	}
	newArgs, ok := l.OnPromote(logPath, append([]string(nil), l.Args...))
	if !ok || len(newArgs) == 0 {
		return false
	}
	l.Args = append([]string(nil), newArgs...)
	return true
}

type startupLogCapture struct {
	mu    sync.Mutex
	file  io.Writer
	live  atomic.Bool
	buf   []byte
	start time.Time
}

type startupStreamWriter struct {
	capture *startupLogCapture
	term    io.Writer
}

func newStartupLogCapture(file io.Writer) *startupLogCapture {
	return &startupLogCapture{file: file, start: time.Now()}
}

func (c *startupLogCapture) writer(term io.Writer) io.Writer {
	return startupStreamWriter{capture: c, term: term}
}

func (c *startupLogCapture) setLive(v bool) {
	c.live.Store(v)
}

func (c *startupLogCapture) tail(max int) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	data := c.buf
	if len(data) > max {
		data = data[len(data)-max:]
	}
	return string(data)
}

func (w startupStreamWriter) Write(p []byte) (int, error) {
	if w.capture == nil {
		if w.term != nil {
			_, _ = w.term.Write(p)
		}
		return len(p), nil
	}
	w.capture.mu.Lock()
	if w.capture.file != nil {
		_, _ = w.capture.file.Write(p)
	}
	w.capture.buf = append(w.capture.buf, p...)
	const maxTail = 64 * 1024
	if len(w.capture.buf) > maxTail {
		w.capture.buf = append([]byte(nil), w.capture.buf[len(w.capture.buf)-maxTail:]...)
	}
	w.capture.mu.Unlock()
	if w.capture.live.Load() && w.term != nil {
		_, _ = w.term.Write(p)
	}
	return len(p), nil
}

var startupSpinnerFrames = []string{"|", "/", "-", "\\"}

func spinStartupProgress(stop <-chan struct{}, log *startupLogCapture, start time.Time, timeout time.Duration, pid int, args []string) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	progress := newStartupProgressTracker(pid, args)
	lastLine := ""
	for i := 0; ; i++ {
		select {
		case <-stop:
			return
		case <-ticker.C:
			line := fitStartupStatusLine(fmt.Sprintf("%s  %s",
				startupSpinnerFrames[i%len(startupSpinnerFrames)],
				startupProgressStatus(log.tail(64*1024), time.Since(start), timeout, progress.snapshot())))
			if line != lastLine {
				fmt.Fprintf(os.Stderr, "\r\033[K%s", line)
				lastLine = line
			}
		}
	}
}

func fitStartupStatusLine(line string) string {
	line = strings.Join(strings.Fields(line), " ")
	cols := startupTerminalColumns()
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

func startupTerminalColumns() int {
	if cols := startupTerminalColumnsOS(); cols > 0 {
		return cols
	}
	if v, err := strconv.Atoi(os.Getenv("COLUMNS")); err == nil && v > 0 {
		return v
	}
	return 100
}

type startupProgress struct {
	done  int64
	total int64
}

func startupProgressStatus(logText string, elapsed, timeout time.Duration, progress startupProgress) string {
	parts := make([]string, 0, 5)
	if progress.total > 0 {
		pct := startupProgressPercent(progress)
		parts = append(parts, fmt.Sprintf("%s %3d%%", startupProgressBar(pct, 20), pct))
	}
	parts = append(parts, startupPhase(logText))
	if timeout > 0 {
		parts = append(parts, fmt.Sprintf("%s/%s", elapsed.Round(time.Second), timeout.Round(time.Second)))
	} else {
		parts = append(parts, elapsed.Round(time.Second).String())
	}
	if progress.total > 0 {
		parts = append(parts, fmt.Sprintf("read %s/%s", formatGiB(progress.done), formatGiB(progress.total)))
	}
	if line := latestBackendLine(logText); line != "" {
		parts = append(parts, truncateStatus(line, 90))
	}
	return strings.Join(parts, " | ")
}

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
		return "pinning host memory"
	default:
		return "starting backend"
	}
}

func startupProgressPercent(p startupProgress) int {
	if p.total <= 0 || p.done <= 0 {
		return 0
	}
	if p.done >= p.total {
		return 100
	}
	return int((p.done * 100) / p.total)
}

func startupProgressBar(percent, width int) string {
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
		return strings.Join(strings.Fields(line), " ")
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

type startupProgressTracker struct {
	pid   int
	paths map[string]int64
	total int64
}

func newStartupProgressTracker(pid int, args []string) *startupProgressTracker {
	paths, total := startupModelShardPaths(startupModelPathFromArgs(args))
	return &startupProgressTracker{pid: pid, paths: paths, total: total}
}

func (t *startupProgressTracker) snapshot() startupProgress {
	if t == nil || t.pid <= 0 || t.total <= 0 {
		return startupProgress{}
	}
	done := t.fdPositions()
	if done == 0 {
		done = procRChar(t.pid)
	}
	if done > t.total {
		done = t.total
	}
	return startupProgress{done: done, total: t.total}
}

func (t *startupProgressTracker) fdPositions() int64 {
	entries, err := os.ReadDir(fmt.Sprintf("/proc/%d/fd", t.pid))
	if err != nil {
		return 0
	}
	byPath := map[string]int64{}
	for _, entry := range entries {
		fd := entry.Name()
		target, err := os.Readlink(filepath.Join(fmt.Sprintf("/proc/%d/fd", t.pid), fd))
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
		n, err := strconv.ParseInt(strings.TrimSpace(strings.TrimPrefix(line, "pos:")), 10, 64)
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
		n, err := strconv.ParseInt(strings.TrimSpace(strings.TrimPrefix(line, "rchar:")), 10, 64)
		if err == nil {
			return n
		}
	}
	return 0
}

func startupModelPathFromArgs(args []string) string {
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

var startupSplitGGUFName = regexp.MustCompile(`(?i)^(.*)-([0-9]+)-of-([0-9]+)\.gguf$`)

func startupModelShardPaths(modelPath string) (map[string]int64, int64) {
	paths := map[string]int64{}
	if modelPath == "" {
		return paths, 0
	}
	dir := filepath.Dir(modelPath)
	base := filepath.Base(modelPath)
	match := startupSplitGGUFName.FindStringSubmatch(base)
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

func stderrIsTTY() bool {
	fi, err := os.Stderr.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

// extractPort finds --port from args.
func (l *Launcher) extractPort() string {
	for i, a := range l.Args {
		if a == "--port" && i+1 < len(l.Args) {
			return l.Args[i+1]
		}
	}
	return "8081"
}

// oomPattern matches OOM markers on word boundaries so model names like
// "Bloom" or words like "room" in log output don't classify as OOM.
var (
	oomPattern        = regexp.MustCompile(`(?i)\b(oom|out of memory)\b`)
	ramOOMPattern     = regexp.MustCompile(`(?i)\bram\b.*\boom\b|\boom\b.*\bram\b`)
	cudaOOMPattern    = regexp.MustCompile(`(?i)allocating\s+([0-9]+(?:\.[0-9]+)?)\s+MiB\s+on device\s+(\d+):\s+cudaMalloc failed: out of memory`)
	cudaDevicePattern = regexp.MustCompile(`(?i)\bcurrent device:\s*(\d+)\b`)
)

// ParseCUDAOOM extracts the CUDA device and allocation size from a llama.cpp
// cudaMalloc OOM line.
func ParseCUDAOOM(line string) (device int, allocMB int, ok bool) {
	m := cudaOOMPattern.FindStringSubmatch(line)
	if m == nil {
		return 0, 0, false
	}
	alloc, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return 0, 0, false
	}
	device, err = strconv.Atoi(m[2])
	if err != nil {
		return 0, 0, false
	}
	return device, int(math.Ceil(alloc)), true
}

// ParseCUDADevice extracts the device from the diagnostic emitted by newer
// llama.cpp CUDA VMM allocation failures. Those failures abort in cuMemCreate
// and report "current device" on a separate line, but omit the requested byte
// count, so ParseCUDAOOM cannot recognize them.
func ParseCUDADevice(line string) (device int, ok bool) {
	m := cudaDevicePattern.FindStringSubmatch(line)
	if m == nil {
		return 0, false
	}
	device, err := strconv.Atoi(m[1])
	if err != nil {
		return 0, false
	}
	return device, true
}

// parseLoadFailure reads this launcher's own log for known error patterns.
func (l *Launcher) parseLoadFailure() (FailureType, string) {
	if l.lastLogPath == "" {
		return FailureUnknown, ""
	}
	f, err := os.Open(l.lastLogPath)
	if err != nil {
		return FailureUnknown, ""
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	var lines []string
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	// Check from the end for error patterns. Track the most informative line so
	// an unclassified failure still surfaces the backend's real stderr instead of
	// an empty "failure: unknown:" message (the buffer-type crash was fully
	// diagnosable from llama.cpp's own output, but the launcher used to hide it).
	var lastErrLine, lastNonEmpty string
	for i := len(lines) - 1; i >= 0 && i > len(lines)-50; i-- {
		line := lines[i]
		low := strings.ToLower(line)
		if lastNonEmpty == "" && strings.TrimSpace(line) != "" {
			lastNonEmpty = line
		}
		if lastErrLine == "" && (strings.Contains(low, "error") ||
			strings.Contains(low, "failed") || strings.Contains(low, "abort")) {
			lastErrLine = line
		}

		// CPU-only build asked to place tensors on a GPU buffer it wasn't built
		// with: `error while handling argument "-ot": unknown buffer type`
		// (followed by `Available buffer types: CPU`). Match the error line, not
		// the header, so the surfaced message is the actionable one. Deterministic.
		if strings.Contains(low, "buffer type") &&
			(strings.Contains(low, "unknown") || strings.Contains(low, "unsupported") ||
				strings.Contains(low, "invalid") || strings.Contains(low, "no such")) {
			return FailureBackendCapability, line
		}
		if strings.Contains(low, "unknown model architecture") ||
			strings.Contains(low, "unable to load model") {
			return FailureUnknownModel, line
		}
		if strings.Contains(low, "pinned memory") && strings.Contains(low, "fail") {
			return FailurePinnedFail, line
		}
		if strings.Contains(low, "pinned memory") && strings.Contains(low, "capacity") {
			return FailurePinnedCap, line
		}
		if strings.Contains(low, "pinned memory") && strings.Contains(low, "hang") {
			return FailurePinnedHang, line
		}
		if _, _, ok := ParseCUDAOOM(line); ok {
			return FailureCUDAOOM, line
		}
		if ramOOMPattern.MatchString(line) {
			return FailureRAMOOM, line
		}
		if oomPattern.MatchString(line) || strings.Contains(low, "cuda error") {
			return FailureOOM, line
		}
	}

	// Unclassified: surface the backend's real stderr rather than nothing.
	if lastErrLine != "" {
		return FailureUnknown, lastErrLine
	}
	return FailureUnknown, lastNonEmpty
}

// writeProbeCache parses the launch log and writes measured probe values.
func (l *Launcher) writeProbeCache(logPath string) {
	data, err := os.ReadFile(logPath)
	if err != nil {
		return
	}
	computeBuf, kvPerLayer := placement.ParseLogForProbe(string(data))
	if computeBuf <= 0 && kvPerLayer <= 0 {
		return
	}
	if l.ProbeModel != nil {
		computeByGPU := placement.ParseComputeBuffersByGPU(string(data))
		_ = placement.WriteProbeCacheForModel(l.ProbeCacheDir, l.ProbeModel, l.ProbeCtxSize, l.ProbeUBatchSize, l.ProbeKVQuality, l.ProbeKVPlacement, l.ProbeBackendTag, l.ProbeGPUs, computeByGPU, kvPerLayer)
		return
	}
	modelName := l.extractModelName()
	if modelName == "" {
		modelName = "unknown"
	}
	if err := placement.WriteProbeCache("", modelName, computeBuf, kvPerLayer); err != nil {
		// Silently ignore — probe cache is best-effort
		return
	}
}

func (l *Launcher) extractModelName() string {
	for i, a := range l.Args {
		if a == "-m" && i+1 < len(l.Args) {
			return filepath.Base(l.Args[i+1])
		}
	}
	return ""
}

func doHTTPGet(url string) (*http.Response, error) {
	client := &http.Client{Timeout: 2 * time.Second}
	return client.Get(url)
}
