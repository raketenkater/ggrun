package server

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"syscall"
	"time"
)

// Process wraps a running llama-server subprocess.
type Process struct {
	Cmd    *exec.Cmd
	Port   int
	cancel context.CancelFunc
}

// Start launches llama-server with the given args and waits for it to be ready.
func Start(args []string, port int) (*Process, error) {
	return StartWithTimeout(args, port, 60*time.Second)
}

// StartWithTimeout launches llama-server with a custom readiness timeout.
func StartWithTimeout(args []string, port int, timeout time.Duration) (*Process, error) {
	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	// Ensure CUDA device enumeration matches nvidia-smi PCI bus order.
	// Without this, llama-server may enumerate GPUs differently from our
	// detection, causing -ot override-tensor flags to target wrong devices.
	cmd.Env = append(os.Environ(), "CUDA_DEVICE_ORDER=PCI_BUS_ID")

	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("start server: %w", err)
	}

	p := &Process{Cmd: cmd, Port: port, cancel: cancel}
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
		// Try graceful kill of process group
		_ = syscall.Kill(-p.Cmd.Process.Pid, syscall.SIGTERM)
		time.Sleep(1 * time.Second)
		_ = syscall.Kill(-p.Cmd.Process.Pid, syscall.SIGKILL)
	}
	return p.Cmd.Wait()
}

// IsRunning returns true if the process is still alive.
func (p *Process) IsRunning() bool {
	if p == nil || p.Cmd == nil || p.Cmd.Process == nil {
		return false
	}
	return p.Cmd.Process.Signal(syscall.Signal(0)) == nil
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
