package recovery

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/raketenkater/llm-server/pkg/placement"
)

// FailureType classifies how a server load failed.
type FailureType string

const (
	FailureOOM           FailureType = "oom"
	FailurePinnedFail    FailureType = "pinned_fail"
	FailurePinnedCap     FailureType = "pinned_cap_exceeded"
	FailurePinnedHang    FailureType = "pinned_hang"
	FailureRAMOOM        FailureType = "ram_oom"
	FailureUnknownModel  FailureType = "unknown_model"
	FailureUnknown       FailureType = "unknown"
)

// Launcher wraps server startup with crash recovery and fallback.
type Launcher struct {
	BinaryPath      string
	Args            []string
	FallbackPath    string // mainline llama-server if ik_llama fails
	MaxRestarts     int
	BackoffBase     time.Duration
	HealthTimeout   time.Duration
	KeepAlive       bool
	OnLog           func(string)
	OnFailure       func(FailureType, string)
	OnRestart       func(int, time.Duration)
	OnFallback      func(string)
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
	backoff := l.BackoffBase
	binaryPath := l.BinaryPath

	for {
		if err := l.runOnce(ctx, binaryPath, restartCount); err != nil {
			// Check for known failure types from stderr log
			ft, msg := l.parseLoadFailure()
			if l.OnFailure != nil {
				l.OnFailure(ft, msg)
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
	logPath := fmt.Sprintf("/tmp/llm-server-launch-%d.log", time.Now().Unix())
	logFile, err := os.Create(logPath)
	if err != nil {
		return err
	}
	defer logFile.Close()

	cmd := exec.CommandContext(ctx, binaryPath, l.Args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	// Tee stdout/stderr to both terminal and log file
	cmd.Stdout = os.Stdout
	cmd.Stderr = logFile

	// Build a clean environment with our required CUDA ordering.
	// Filter out any existing CUDA_DEVICE_ORDER before adding ours,
	// because duplicates have undefined behaviour in the CUDA runtime.
	env := os.Environ()
	filtered := make([]string, 0, len(env)+2)
	for _, e := range env {
		if !strings.HasPrefix(e, "CUDA_DEVICE_ORDER=") {
			filtered = append(filtered, e)
		}
	}
	filtered = append(filtered, "CUDA_DEVICE_ORDER=PCI_BUS_ID")

	// Prepend lib hub to LD_LIBRARY_PATH if available
	if hub := os.Getenv("LLM_SERVER_LIB_HUB"); hub != "" {
		old := os.Getenv("LD_LIBRARY_PATH")
		for i, e := range filtered {
			if strings.HasPrefix(e, "LD_LIBRARY_PATH=") {
				if old == "" {
					filtered[i] = "LD_LIBRARY_PATH=" + hub
				} else {
					filtered[i] = "LD_LIBRARY_PATH=" + hub + ":" + old
				}
				old = "" // marker: already handled
				break
			}
		}
		if old != "" {
			// No LD_LIBRARY_PATH in env yet — add it
			filtered = append(filtered, "LD_LIBRARY_PATH="+hub)
		}
	}
	cmd.Env = filtered

	if err := cmd.Start(); err != nil {
		return err
	}

	// Wait for health check or process death
	port := l.extractPort()
	healthURL := fmt.Sprintf("http://127.0.0.1:%s/health", port)
	modelsURL := fmt.Sprintf("http://127.0.0.1:%s/v1/models", port)

	deadline := time.Now().Add(l.HealthTimeout)
	for time.Now().Before(deadline) {
		// Check if process died
		if cmd.Process != nil {
			if err := cmd.Process.Signal(syscall.Signal(0)); err != nil {
				// Process died before health check
				return fmt.Errorf("process died during startup")
			}
		}

		// Try health endpoint
		if resp, err := doHTTPGet(healthURL); err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				// Server is healthy! Write probe cache then wait for exit.
				l.writeProbeCache(logPath)
				return cmd.Wait()
			}
		}
		if resp, err := doHTTPGet(modelsURL); err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				l.writeProbeCache(logPath)
				return cmd.Wait()
			}
		}
		time.Sleep(500 * time.Millisecond)
	}

	// Health timeout — kill process
	if cmd.Process != nil {
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
		time.Sleep(1 * time.Second)
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	return fmt.Errorf("health timeout")
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

// parseLoadFailure reads the latest log for known error patterns.
func (l *Launcher) parseLoadFailure() (FailureType, string) {
	// Find most recent log file
	entries, err := os.ReadDir("/tmp")
	if err != nil {
		return FailureUnknown, ""
	}
	var latest string
	var latestTime time.Time
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "llm-server-launch-") && strings.HasSuffix(e.Name(), ".log") {
			info, _ := e.Info()
			if info.ModTime().After(latestTime) {
				latestTime = info.ModTime()
				latest = "/tmp/" + e.Name()
			}
		}
	}
	if latest == "" {
		return FailureUnknown, ""
	}

	f, err := os.Open(latest)
	if err != nil {
		return FailureUnknown, ""
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	var lines []string
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	// Check from the end for error patterns
	for i := len(lines) - 1; i >= 0 && i > len(lines)-50; i-- {
		line := lines[i]
		low := strings.ToLower(line)

		if strings.Contains(low, "unknown model architecture") ||
			strings.Contains(low, "unable to load model") {
			return FailureUnknownModel, line
		}
		if strings.Contains(low, "out of memory") || strings.Contains(low, "oom") {
			return FailureOOM, line
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
		if strings.Contains(low, "ram") && strings.Contains(low, "oom") {
			return FailureRAMOOM, line
		}
		if strings.Contains(low, "cuda error") || strings.Contains(low, "cuda out of memory") {
			return FailureOOM, line
		}
	}

	return FailureUnknown, ""
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
