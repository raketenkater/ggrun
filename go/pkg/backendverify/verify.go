// Package backendverify is ggrun's contract check for a backend build: load a
// small model with a fixed argv and confirm the planner's log parsers still
// read what they depend on (review I-05). A backend whose log wording changed
// passes --help and --version but leaves placement measuring nothing.
package backendverify

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/raketenkater/ggrun/pkg/placement"
)

// Result is what the check read from the backend's load log.
type Result struct {
	placement.BackendLogFacts
	OffloadedLayers int
	TotalLayers     int
}

var (
	offloadedRE = regexp.MustCompile(`offloaded (\d+)/(\d+) layers to GPU`)
	gpuBufferRE = regexp.MustCompile(`(CUDA\d+|Vulkan\d+|ROCm\d+|Metal)\s+(model |KV |compute )?buffer size =`)
)

// readyMarkers end the wait: the backend finished loading.
var readyMarkers = []string{"server is listening", "listening on http", "model loaded", "all slots are idle"}

// Check reads a load log and fails when a field the planner needs is missing.
// The argv runs on CPU (-ngl 0), so the planner's prediction is zero layers
// on a GPU; the log must say the same.
func Check(log string) (Result, error) {
	r := Result{BackendLogFacts: placement.ParseBackendLogFacts(log)}
	var problems []string
	if r.HostModelMB <= 0 {
		problems = append(problems, "no model buffer size (\"CPU model buffer size = ... MiB\")")
	}
	if r.KVTotalMB <= 0 {
		problems = append(problems, "no KV size (\"KV buffer size\" or \"KV self size\")")
	}
	// GPUs are hidden from the check, so a build may not print the offload
	// line at all; then no buffer may sit on a GPU. Either way the planner's
	// prediction for -ngl 0 is zero GPU layers.
	if m := offloadedRE.FindStringSubmatch(log); m != nil {
		r.OffloadedLayers, _ = strconv.Atoi(m[1])
		r.TotalLayers, _ = strconv.Atoi(m[2])
		if r.OffloadedLayers != 0 {
			problems = append(problems, fmt.Sprintf("%d layers on a GPU with -ngl 0; the planner predicts 0", r.OffloadedLayers))
		}
	}
	if gpuBufferRE.MatchString(log) {
		problems = append(problems, "a buffer was placed on a GPU with -ngl 0 and hidden GPUs; the planner predicts none")
	}
	if len(problems) > 0 {
		return r, fmt.Errorf("backend log no longer matches the planner's parsers: %s", strings.Join(problems, "; "))
	}
	return r, nil
}

type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// freePort finds a free port on 127.0.0.2, so the check never collides with
// a server on 127.0.0.1.
func freePort(host string) (int, error) {
	ln, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
	if err != nil {
		return 0, err
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port, nil
}

// Run loads model in binary with a fixed CPU argv, waits for the load to
// finish (or the process to exit, or timeout), stops it and checks the log.
func Run(binary, model string, timeout time.Duration) (Result, error) {
	host := "127.0.0.2"
	port, err := freePort(host)
	if err != nil {
		host = "127.0.0.1"
		if port, err = freePort(host); err != nil {
			return Result{}, fmt.Errorf("find a free port: %w", err)
		}
	}
	// Probe the dialect before the load deadline starts.
	loadArgs := residentLoadArgs(binary)
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	argv := []string{"-m", model, "--host", host, "--port", strconv.Itoa(port),
		"-c", "512", "-ngl", "0", "-t", "2", "--parallel", "1"}
	// Resident weights: the planner counts only non-reclaimable host buffers,
	// so a mapped (CPU_Mapped) model buffer would not be read.
	argv = append(argv, loadArgs...)
	cmd := exec.CommandContext(ctx, binary, argv...)
	cmd.SysProcAttr = sysProcAttr()
	// The check is CPU-only and must not take memory on a GPU that may be
	// serving: hide every GPU from the backend.
	cmd.Env = append(os.Environ(), "CUDA_VISIBLE_DEVICES=", "HIP_VISIBLE_DEVICES=", "GGML_VK_VISIBLE_DEVICES=")
	var log lockedBuffer
	cmd.Stdout, cmd.Stderr = &log, &log
	cmd.WaitDelay = 2 * time.Second
	if err := cmd.Start(); err != nil {
		return Result{}, fmt.Errorf("start %s: %w", binary, err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	ready := false
	tick := time.NewTicker(100 * time.Millisecond)
	defer tick.Stop()
wait:
	for {
		select {
		case <-done:
			break wait
		case <-ctx.Done():
			break wait
		case <-tick.C:
			text := log.String()
			for _, m := range readyMarkers {
				if strings.Contains(text, m) {
					ready = true
					break wait
				}
			}
		}
	}
	stopProcess(cmd)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
	}
	text := log.String()
	r, checkErr := Check(text)
	if !ready && checkErr != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return r, fmt.Errorf("backend did not finish loading within %s: %w\n%s", timeout, checkErr, tail(text, 12))
		}
		return r, fmt.Errorf("backend exited before loading: %w\n%s", checkErr, tail(text, 12))
	}
	return r, checkErr
}

// residentLoadArgs asks for resident (not mapped) weights in the backend's own
// dialect: builds with --load-mode dropped --no-mmap.
// It also asks for the log verbosity ggrun's launches use: builds with -lv
// print buffer sizes only at level 4.
func residentLoadArgs(binary string) []string {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, _ := exec.CommandContext(ctx, binary, "--help").CombinedOutput()
	help := string(out)
	var args []string
	if strings.Contains(help, "--load-mode") {
		args = append(args, "--load-mode", "none")
	} else {
		args = append(args, "--no-mmap")
	}
	if strings.Contains(help, "--log-verbosity") || strings.Contains(help, "-lv,") {
		args = append(args, "-lv", "4")
	}
	return args
}

func tail(s string, lines int) string {
	parts := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(parts) > lines {
		parts = parts[len(parts)-lines:]
	}
	return strings.Join(parts, "\n")
}
