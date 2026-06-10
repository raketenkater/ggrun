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
	cmd.Stdout = os.Stdout

	// Tee stderr: output to terminal AND capture in buffer for post-launch probe.
	logBuf := &threadSafeBuffer{}
	cmd.Stderr = io.MultiWriter(os.Stderr, logBuf)

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
	if err := p.waitReady(timeout); err != nil {
		p.Stop()
		return nil, fmt.Errorf("server not ready: %w", err)
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
