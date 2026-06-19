package server

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"time"
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

// Start launches llama-server with the given args and waits for it to be ready.
func Start(args []string, port int) (*Process, error) {
	return StartWithTimeout(args, port, 60*time.Second)
}

// StartWithTimeout launches llama-server with a custom readiness timeout.
func StartWithTimeout(args []string, port int, timeout time.Duration) (*Process, error) {
	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	setSysProcAttr(cmd)
	// Capture stdout/stderr to a buffer for the post-launch probe. On a TTY we
	// keep the screen clean during model load (a spinner owns it) and only begin
	// streaming the backend's own logs once it is ready; off a TTY we stream from
	// the start, exactly as before (so piped/benchmark runs are unchanged).
	logBuf := &threadSafeBuffer{}
	tty := stdoutIsTTY()
	live := &atomic.Bool{}
	live.Store(!tty)
	cmd.Stdout = &gatedWriter{buf: logBuf, term: os.Stdout, live: live}
	cmd.Stderr = &gatedWriter{buf: logBuf, term: os.Stderr, live: live}

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
	cmd.Env = filtered

	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("start server: %w", err)
	}

	p := &Process{Cmd: cmd, Port: port, cancel: cancel, LogBuf: logBuf, waitCh: make(chan error, 1)}
	go func() { p.waitCh <- cmd.Wait() }()

	start := time.Now()
	var stopSpin chan struct{}
	if tty {
		stopSpin = make(chan struct{})
		go spinUntilReady(stopSpin, logBuf, start)
	}
	err := p.waitReady(timeout)
	if tty {
		close(stopSpin)
		fmt.Fprint(os.Stderr, "\r\033[K") // clear the spinner line
	}
	if err != nil {
		if tty {
			fmt.Fprintln(os.Stderr, "[launch] backend failed to start; last output:")
			fmt.Fprintln(os.Stderr, tailLines(logBuf.String(), 20))
		}
		p.Stop()
		return nil, fmt.Errorf("server not ready: %w", err)
	}
	live.Store(true) // backend is up — stream its logs from here on
	if tty {
		fmt.Fprintf(os.Stderr, "✓ model loaded — server ready in %s\n", time.Since(start).Round(time.Second))
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

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// spinUntilReady animates a single status line on stderr while the backend loads,
// labelling the current phase from the backend's captured log output.
func spinUntilReady(stop <-chan struct{}, log *threadSafeBuffer, start time.Time) {
	t := time.NewTicker(120 * time.Millisecond)
	defer t.Stop()
	for i := 0; ; i++ {
		select {
		case <-stop:
			return
		case <-t.C:
			fmt.Fprintf(os.Stderr, "\r\033[K%s  %s · %s",
				spinnerFrames[i%len(spinnerFrames)],
				startupPhase(log.String()),
				time.Since(start).Round(time.Second))
		}
	}
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
		return "pinning host memory (large MoE — can take a few minutes)"
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
