package main

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/claudeauto"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/server"
)

type claudeAutoRuntime struct {
	reviewer     *server.Process
	reviewerLog  io.Closer
	reviewerPort int
	reviewerGPU  int
	router       *claudeauto.Router
}

func claudeAutoReviewerNeeded(extraArgs []string) bool {
	if disabledEnv("GGRUN_CLAUDE_AUTO_REVIEWER") {
		return false
	}
	permissionArgs := claudeCodePermissionArgs(extraArgs)
	// "inherit" can still resolve to Auto in settings.json. Starting the small
	// reviewer is harmless when no classifier calls arrive and keeps inheritance
	// functional when the user's configured default is Auto.
	return permissionArgs == nil || (len(permissionArgs) == 2 && permissionArgs[1] == "auto")
}

func disabledEnv(key string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "0", "false", "no", "off", "disabled":
		return true
	default:
		return false
	}
}

func startClaudeAutoReviewer(req *launchRequest, cfg *config.Config, caps *detect.Capabilities) (*claudeAutoRuntime, error) {
	if req == nil || !req.ClaudeCode || !claudeAutoReviewerNeeded(nil) {
		return nil, nil
	}
	appHome := ""
	if cfg != nil {
		appHome = strings.TrimSpace(cfg.AppHome)
	}
	if appHome == "" {
		appHome = backends.AppHome()
	}
	modelPath, err := claudeauto.EnsureReviewerModel(context.Background(), appHome, os.Stdout)
	if err != nil {
		return nil, fmt.Errorf("prepare local Auto reviewer: %w", err)
	}
	be := findClaudeReviewerBackend(caps)
	if be == nil {
		return nil, fmt.Errorf("local Auto needs a current mainline llama-server (Qwen3.5 support); none was found")
	}
	port, err := freeLoopbackPort()
	if err != nil {
		return nil, err
	}
	logWriter, logCloser := claudeReviewerLog(cfg, port)

	var lastErr error
	for _, gpu := range claudeReviewerGPUCandidates(caps, req) {
		args := claudeReviewerArgs(be.Path, modelPath, port, gpu, be.Help)
		p, err := server.StartWithTimeoutTo(args, port, 5*time.Minute, logWriter, logWriter)
		if err == nil {
			fmt.Printf("[claude-code] Auto reviewer ready on GPU %d (PID %d, Qwen3.5-2B, ctx 64k)\n", gpu, p.Cmd.Process.Pid)
			return &claudeAutoRuntime{reviewer: p, reviewerLog: logCloser, reviewerPort: port, reviewerGPU: gpu}, nil
		}
		lastErr = err
		fmt.Fprintf(os.Stderr, "[claude-code] Auto reviewer did not fit GPU %d; trying the next device.\n", gpu)
	}

	// CPU is slower, but it preserves autonomous/fail-closed behavior on systems
	// whose GPUs are already full. It is also the normal path on CPU-only hosts.
	args := claudeReviewerArgs(be.Path, modelPath, port, -1, be.Help)
	p, err := server.StartWithTimeoutTo(args, port, 5*time.Minute, logWriter, logWriter)
	if err != nil {
		if logCloser != nil {
			_ = logCloser.Close()
		}
		if lastErr != nil {
			return nil, fmt.Errorf("start local Auto reviewer (GPU: %v; CPU: %w)", lastErr, err)
		}
		return nil, fmt.Errorf("start local Auto reviewer: %w", err)
	}
	fmt.Printf("[claude-code] Auto reviewer ready on CPU (PID %d, Qwen3.5-2B, ctx 64k)\n", p.Cmd.Process.Pid)
	return &claudeAutoRuntime{reviewer: p, reviewerLog: logCloser, reviewerPort: port, reviewerGPU: -1}, nil
}

func claudeReviewerArgs(binary, modelPath string, port, gpu int, help string) []string {
	args := []string{
		binary, "-m", modelPath,
		"--host", "127.0.0.1", "--port", strconv.Itoa(port),
		"--ctx-size", "65536", "--parallel", "1",
		"--alias", "local", "--jinja",
		"--temp", "0", "--presence-penalty", "0", "--repeat-penalty", "1",
	}
	if strings.Contains(help, "--reasoning") {
		args = append(args, "--reasoning", "off")
	}
	if gpu >= 0 {
		// --device exposes one device to this model, renumbered locally to 0.
		args = append(args, "--device", fmt.Sprintf("CUDA%d", gpu), "--split-mode", "none", "-ngl", "999", "-mg", "0", "--fit", "off")
	} else {
		args = append(args, "-ngl", "0")
	}
	return args
}

func findClaudeReviewerBackend(caps *detect.Capabilities) *backendInfo {
	seen := map[string]bool{}
	var candidates []string
	// Prefer ggrun's maintained mainline binary over an arbitrary LLAMA_SERVER
	// or architecture fork selected for the main model.
	appHome := backends.AppHome()
	candidates = append(candidates,
		filepath.Join(appHome, ".bin", "llama-server-cuda"),
		filepath.Join(appHome, ".bin", "llama-server-cuda.exe"),
		filepath.Join(appHome, ".bin", "llama-server"),
		filepath.Join(appHome, ".bin", "llama-server.exe"),
	)
	if caps != nil {
		for _, be := range caps.Backends {
			candidates = append(candidates, be.Path)
		}
	}
	candidates = append(candidates, backendSearchPaths()...)
	for _, path := range candidates {
		if path == "" || seen[path] {
			continue
		}
		seen[path] = true
		if _, err := os.Stat(path); err != nil {
			continue
		}
		be := detectBackend(path)
		if !be.IsIK && strings.Contains(be.Help, "--reasoning") {
			return be
		}
	}
	return nil
}

// claudeReviewerGPUCandidates preserves the largest GPU for the main model,
// then prefers bandwidth among equal-sized devices. A failed real load moves
// to the next candidate; no estimated memory cushion is used.
func claudeReviewerGPUCandidates(caps *detect.Capabilities, req *launchRequest) []int {
	if caps == nil || len(caps.GPUs) == 0 || (req != nil && req.CPUMode) {
		return nil
	}
	if req != nil && strings.TrimSpace(req.GPUsFlag) != "" {
		// CUDA_VISIBLE_DEVICES renumbers the selected subset. Try each visible
		// device in that local order; placement later maps actual physical usage.
		parts := strings.Split(req.GPUsFlag, ",")
		physical := map[int]bool{}
		for _, part := range parts {
			if idx, err := strconv.Atoi(strings.TrimSpace(part)); err == nil && idx >= 0 {
				physical[idx] = true
			}
		}
		out := make([]int, 0, len(physical))
		for i := 0; i < len(physical); i++ {
			out = append(out, i)
		}
		return out
	}
	gpus := append([]detect.GPU(nil), caps.GPUs...)
	sort.SliceStable(gpus, func(i, j int) bool {
		if gpus[i].VRAMTotalMB != gpus[j].VRAMTotalMB {
			return gpus[i].VRAMTotalMB < gpus[j].VRAMTotalMB
		}
		if gpus[i].BandwidthMBps != gpus[j].BandwidthMBps {
			return gpus[i].BandwidthMBps > gpus[j].BandwidthMBps
		}
		return gpus[i].Index < gpus[j].Index
	})
	out := make([]int, 0, len(gpus))
	for _, gpu := range gpus {
		out = append(out, gpu.Index)
	}
	return out
}

func freeLoopbackPort() (int, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, fmt.Errorf("allocate local Auto reviewer port: %w", err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	if err := ln.Close(); err != nil {
		return 0, err
	}
	return port, nil
}

func claudeReviewerLog(cfg *config.Config, port int) (io.Writer, io.Closer) {
	dir := os.TempDir()
	if cfg != nil && cfg.LogDir != "" {
		dir = cfg.LogDir
	}
	path := filepath.Join(dir, fmt.Sprintf("ggrun-claude-reviewer-%d.log", port))
	f, err := os.Create(path)
	if err != nil {
		return io.Discard, nil
	}
	fmt.Printf("[claude-code] Auto reviewer logs -> %s\n", path)
	return f, f
}

func (r *claudeAutoRuntime) startRouter(mainHost string, mainPort int) error {
	if r == nil {
		return nil
	}
	host := mainHost
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	router, err := claudeauto.StartRouter(
		fmt.Sprintf("http://%s:%d", host, mainPort),
		fmt.Sprintf("http://127.0.0.1:%d", r.reviewerPort),
	)
	if err != nil {
		return err
	}
	r.router = router
	fmt.Printf("[claude-code] Auto router ready on %s (coding -> main model, safety -> local reviewer)\n", router.URL())
	return nil
}

func (r *claudeAutoRuntime) clientPort(fallback int) int {
	if r != nil && r.router != nil && r.router.Port() > 0 {
		return r.router.Port()
	}
	return fallback
}

func (r *claudeAutoRuntime) isRunning() bool {
	return r != nil && r.reviewer != nil && r.reviewer.IsRunning()
}

func (r *claudeAutoRuntime) stop() {
	if r == nil {
		return
	}
	if r.router != nil {
		_ = r.router.Close()
		r.router = nil
	}
	if r.reviewer != nil {
		_ = r.reviewer.Stop()
		r.reviewer = nil
	}
	if r.reviewerLog != nil {
		_ = r.reviewerLog.Close()
		r.reviewerLog = nil
	}
}
