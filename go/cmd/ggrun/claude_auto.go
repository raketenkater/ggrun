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
	"github.com/raketenkater/ggrun/pkg/libhub"
	"github.com/raketenkater/ggrun/pkg/placement"
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

// claudeReviewerCompanionName is the placement-ledger name of the Auto reviewer
// reservation and its resolved seat.
const claudeReviewerCompanionName = "claude-auto-reviewer"

// claudeReviewerReservationVRAMMB is the reviewer's on-device footprint reserved
// in the placement ledger: ~1.4 GB Q4_K_M weights + 64k Q8 KV + CUDA context and
// compute. The planner only needs a conservative bound — after a real launch the
// reviewer's actual usage is visible in the normal probe paths, and the seat is
// re-planned on every launch anyway.
const claudeReviewerReservationVRAMMB = 2600

// claudeReviewerReservation builds the placement companion for the Auto reviewer
// when the launch needs one. The GPU preference mirrors the legacy walk —
// least-valuable (slowest-link, smallest) first, main GPU last — but expressed
// as data so the planner owns the final seat and the main model packs around it.
// CPU fallback stays allowed: a full-GPU host must keep fail-closed Auto working.
func claudeReviewerReservation(req *launchRequest, caps *detect.Capabilities) *placement.CompanionReservation {
	if req == nil || !req.ClaudeCode || !claudeAutoReviewerNeeded(nil) || req.CPUMode {
		return nil
	}
	if caps == nil || len(caps.GPUs) == 0 {
		return nil
	}
	return &placement.CompanionReservation{
		Name:          claudeReviewerCompanionName,
		VRAMMB:        claudeReviewerReservationVRAMMB,
		GPUPreference: claudeReviewerGPUCandidates(caps, req),
		AllowCPU:      true,
	}
}

// startClaudeAutoReviewer launches the pinned Auto reviewer on the seat the
// placement planner returned in companionPlacements. When the planner ran, its
// single decision is authoritative: a GPU seat is tried once (no fallback walk
// that could land the reviewer on the GPU the plan deliberately kept free), a
// CPU seat (-1) goes straight to the CPU path. Without a plan (reviewer needed
// but no companion reservation supplied) it falls back to the legacy
// least-valuable-GPU-first walk.
func startClaudeAutoReviewer(req *launchRequest, cfg *config.Config, caps *detect.Capabilities, companionPlacements []placement.CompanionPlacement) (*claudeAutoRuntime, error) {
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
		return nil, fmt.Errorf("local Auto needs a current mainline llama-server; none was found")
	}
	port, err := freeLoopbackPort()
	if err != nil {
		return nil, err
	}
	logWriter, logCloser := claudeReviewerLog(cfg, port)

	candidates := claudeReviewerGPUCandidates(caps, req)
	planned := false
	for _, cp := range companionPlacements {
		if cp.Name == claudeReviewerCompanionName {
			planned = true
			if cp.GPU >= 0 {
				candidates = []int{cp.GPU}
			} else {
				candidates = nil
			}
		}
	}

	var lastErr error
	for _, gpu := range candidates {
		// CUDA_VISIBLE_DEVICES is required in addition to --device. Without it,
		// llama.cpp initializes contexts on every GPU even though all reviewer
		// tensors live on the selected device (observed: +262 MiB on the main
		// CUDA0 during a DeepSeek-V4 run). The isolated physical GPU is CUDA0
		// inside the child process.
		args := claudeReviewerArgs(be.Path, modelPath, port, 0, be.Help)
		env := claudeReviewerBackendEnv(be.Path, []string{fmt.Sprintf("CUDA_VISIBLE_DEVICES=%d", gpu)})
		p, err := server.StartWithTimeoutToEnv(args, port, 5*time.Minute, logWriter, logWriter, env)
		if err == nil {
			fmt.Printf("[claude-code] Auto reviewer ready on GPU %d (PID %d, %s, ctx 64k)\n", gpu, p.Cmd.Process.Pid, claudeauto.DefaultReviewerDisplayName)
			return &claudeAutoRuntime{reviewer: p, reviewerLog: logCloser, reviewerPort: port, reviewerGPU: gpu}, nil
		}
		lastErr = err
		if planned {
			// The planner already accounted for this GPU's free VRAM; a failed
			// load means the plan's data was stale, so report it instead of
			// silently moving onto a GPU the plan reserved for the model.
			if logCloser != nil {
				_ = logCloser.Close()
			}
			return nil, fmt.Errorf("start local Auto reviewer on planned GPU %d: %w", gpu, err)
		}
		fmt.Fprintf(os.Stderr, "[claude-code] Auto reviewer did not fit GPU %d; trying the next device.\n", gpu)
	}

	// CPU is slower, but it preserves autonomous/fail-closed behavior on systems
	// whose GPUs are already full. It is also the normal path on CPU-only hosts.
	args := claudeReviewerArgs(be.Path, modelPath, port, -1, be.Help)
	p, err := server.StartWithTimeoutToEnv(args, port, 5*time.Minute, logWriter, logWriter, claudeReviewerBackendEnv(be.Path, claudeReviewerCPUEnv()))
	if err != nil {
		if logCloser != nil {
			_ = logCloser.Close()
		}
		if lastErr != nil {
			return nil, fmt.Errorf("start local Auto reviewer (GPU: %v; CPU: %w)", lastErr, err)
		}
		return nil, fmt.Errorf("start local Auto reviewer: %w", err)
	}
	fmt.Printf("[claude-code] Auto reviewer ready on CPU (PID %d, %s, ctx 64k)\n", p.Cmd.Process.Pid, claudeauto.DefaultReviewerDisplayName)
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
	// The classifier carries a large policy prompt, so its KV cache is a
	// meaningful part of the reviewer footprint. Q8 halves that cache versus
	// F16 while retaining substantially more precision than Q4. Keep the
	// compatibility guard for older user-provided llama-server binaries.
	if strings.Contains(help, "--cache-type-k") && strings.Contains(help, "--cache-type-v") {
		args = append(args, "--cache-type-k", "q8_0", "--cache-type-v", "q8_0")
	}
	if gpu >= 0 {
		// --device exposes one device to this model, renumbered locally to 0.
		args = append(args, "--device", fmt.Sprintf("CUDA%d", gpu), "--split-mode", "none", "-ngl", "999", "-mg", "0")
	} else {
		args = append(args, "-ngl", "0")
	}
	return args
}

func claudeReviewerCPUEnv() []string {
	// GPU-enabled llama-server binaries initialize their accelerator backend even
	// with -ngl 0. Hide accelerators so a CPU fallback still starts when every
	// device is full, which is precisely when this path is needed.
	return []string{
		"CUDA_VISIBLE_DEVICES=-1",
		"HIP_VISIBLE_DEVICES=-1",
		"ROCR_VISIBLE_DEVICES=-1",
	}
}

func claudeReviewerBackendEnv(binary string, env []string) []string {
	if libPath, ok := libhub.StableLibraryPath(binary); ok {
		return libhub.ApplyHubToChildEnv(env, libPath)
	}
	return env
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

// claudeReviewerGPUCandidates preserves the largest GPU for the main model and
// tries the least valuable remaining accelerator first. The reviewer is small
// and bursty, while a large model benefits continuously from keeping its faster
// secondary GPU available for expert or dense tensors. A failed real load moves
// to the next candidate; no estimated memory cushion is used.
func claudeReviewerGPUCandidates(caps *detect.Capabilities, req *launchRequest) []int {
	if caps == nil || len(caps.GPUs) == 0 || (req != nil && req.CPUMode) {
		return nil
	}
	if req != nil && strings.TrimSpace(req.GPUsFlag) != "" {
		// CUDA_VISIBLE_DEVICES takes physical device IDs here. Preserve the user's
		// selected order, reject nonexistent devices, and never silently substitute
		// CUDA0 for a sparse selection such as --gpus 1,2.
		parts := strings.Split(req.GPUsFlag, ",")
		available := map[int]bool{}
		for _, gpu := range caps.GPUs {
			available[gpu.Index] = true
		}
		seen := map[int]bool{}
		out := make([]int, 0, len(parts))
		for _, part := range parts {
			if idx, err := strconv.Atoi(strings.TrimSpace(part)); err == nil && available[idx] && !seen[idx] {
				seen[idx] = true
				out = append(out, idx)
			}
		}
		return out
	}
	gpus := append([]detect.GPU(nil), caps.GPUs...)
	reserved := 0
	for i := 1; i < len(gpus); i++ {
		if gpus[i].VRAMTotalMB > gpus[reserved].VRAMTotalMB ||
			(gpus[i].VRAMTotalMB == gpus[reserved].VRAMTotalMB && gpus[i].BandwidthMBps > gpus[reserved].BandwidthMBps) {
			reserved = i
		}
	}
	mainGPU := gpus[reserved]
	gpus = append(gpus[:reserved], gpus[reserved+1:]...)
	sort.SliceStable(gpus, func(i, j int) bool {
		if gpus[i].BandwidthMBps != gpus[j].BandwidthMBps {
			return gpus[i].BandwidthMBps < gpus[j].BandwidthMBps
		}
		if gpus[i].VRAMTotalMB != gpus[j].VRAMTotalMB {
			return gpus[i].VRAMTotalMB < gpus[j].VRAMTotalMB
		}
		return gpus[i].Index < gpus[j].Index
	})
	gpus = append(gpus, mainGPU)
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
