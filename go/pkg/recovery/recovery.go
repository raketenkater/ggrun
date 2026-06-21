package recovery

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/placement"
)

// FailureType classifies how a server load failed.
type FailureType string

const (
	FailureOOM          FailureType = "oom"
	FailurePinnedFail   FailureType = "pinned_fail"
	FailurePinnedCap    FailureType = "pinned_cap_exceeded"
	FailurePinnedHang   FailureType = "pinned_hang"
	FailureCUDAOOM      FailureType = "cuda_oom"
	FailureRAMOOM       FailureType = "ram_oom"
	FailureUnknownModel FailureType = "unknown_model"
	FailureUnknown      FailureType = "unknown"
)

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

	PlacementCachePath string

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
				device, allocMB, ok := parseCUDAOOM(msg)
				if ok {
					if newArgs, entry, retry := l.OnCUDAOOM(device, allocMB, append([]string(nil), l.Args...)); retry {
						l.Args = newArgs
						if l.PlacementCachePath != "" && entry != nil {
							_ = placement.SavePlacementCache(l.PlacementCachePath, entry)
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
	logFile, err := os.CreateTemp("", "ggrun-launch-*.log")
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
			return ctx.Err()
		default:
		}

		// Check if process died
		if cmd.Process != nil {
			if !procAlive(cmd.Process.Pid) {
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
		killProcGroup(cmd.Process.Pid)
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

// oomPattern matches OOM markers on word boundaries so model names like
// "Bloom" or words like "room" in log output don't classify as OOM.
var (
	oomPattern     = regexp.MustCompile(`(?i)\b(oom|out of memory)\b`)
	ramOOMPattern  = regexp.MustCompile(`(?i)\bram\b.*\boom\b|\boom\b.*\bram\b`)
	cudaOOMPattern = regexp.MustCompile(`(?i)allocating\s+([0-9]+(?:\.[0-9]+)?)\s+MiB\s+on device\s+(\d+):\s+cudaMalloc failed: out of memory`)
)

func parseCUDAOOM(line string) (device int, allocMB int, ok bool) {
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

	// Check from the end for error patterns
	for i := len(lines) - 1; i >= 0 && i > len(lines)-50; i-- {
		line := lines[i]
		low := strings.ToLower(line)

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
		if _, _, ok := parseCUDAOOM(line); ok {
			return FailureCUDAOOM, line
		}
		if ramOOMPattern.MatchString(line) {
			return FailureRAMOOM, line
		}
		if oomPattern.MatchString(line) || strings.Contains(low, "cuda error") {
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
