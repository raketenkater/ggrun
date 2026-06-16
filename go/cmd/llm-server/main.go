package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/raketenkater/llm-server/pkg/benchmark"
	"github.com/raketenkater/llm-server/pkg/config"
	"github.com/raketenkater/llm-server/pkg/daemon"
	"github.com/raketenkater/llm-server/pkg/detect"
	"github.com/raketenkater/llm-server/pkg/download"
	"github.com/raketenkater/llm-server/pkg/gguf"
	"github.com/raketenkater/llm-server/pkg/libhub"
	"github.com/raketenkater/llm-server/pkg/placement"
	"github.com/raketenkater/llm-server/pkg/probe"
	"github.com/raketenkater/llm-server/pkg/recovery"
	"github.com/raketenkater/llm-server/pkg/server"
	"github.com/raketenkater/llm-server/pkg/tui"
	"github.com/raketenkater/llm-server/pkg/tune"
	"github.com/raketenkater/llm-server/pkg/update"
)

// version comes from pkg/update so the binary and the update checker can never
// disagree; releases override it via -ldflags (see .github/workflows/release.yml).
var version = update.Version()

func main() {
	if len(os.Args) < 2 {
		update.PromptOnStartup()
		cmdGUI()
		return
	}

	args := os.Args[1:]
	if dispatchCompat(args) {
		return
	}

	switch args[0] {
	case "version", "--version", "-v":
		fmt.Println("llm-server", version)
	case "detect":
		cmdDetect()
	case "launch":
		cmdLaunch(args[1:])
	case "benchmark":
		cmdBenchmark(args[1:])
	case "daemon":
		cmdDaemon(args[1:])
	case "dry-run":
		cmdDryRun(args[1:])
	case "probe":
		cmdProbe()
	case "download":
		cmdDownload(args[1:])
	case "tune":
		cmdTune(args[1:])
	case "gui", "tui":
		update.PromptOnStartup()
		cmdGUI()
	case "config":
		cmdConfig(args[1:])
	case "update", "--update":
		cmdUpdate()
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `Usage: llm-server [command] [args]

With no command, launches the interactive TUI (same as llm-server-gui).

Commands:
  version              Show version
  detect               Detect hardware capabilities
  probe                Check free GPU/RAM memory
  launch <model.gguf>  Launch model with auto-placement
  benchmark <model>    Benchmark a running server
  daemon               Start persistent daemon
  dry-run <model.gguf> Print computed flags without launching
  download <repo/name> Download from HuggingFace
  tune <model.gguf>    AI-tune model for best performance
  config [show|edit|path|reset]  Manage settings
  update, --update     Update llm-server and backends
  gui, tui             Interactive TUI (model picker, settings, launch)

Launch flags:
  -port int            Server port (default 8081)
  -ctx int             Context size (default auto)
  -kv string           KV placement: auto|gpu|cpu (default auto)
  -kv-quality string   KV quality: high|mid|low (default mid)
  -cpu                 Force CPU-only mode
  -gpus string         Comma-separated GPU indices
  -vision              Enable vision (auto-detect mmproj)
  --spec string       Speculative decoding: off|auto|mtp|eagle3|draft|ngram|ngram-mod|ngram-k4v
`)
}

func knownCommand(cmd string) bool {
	switch cmd {
	case "version", "--version", "-v", "detect", "launch", "benchmark", "daemon", "dry-run", "probe", "download", "tune", "gui", "tui", "config", "update", "--update":
		return true
	default:
		return false
	}
}

func dispatchCompat(args []string) bool {
	if len(args) == 0 || knownCommand(args[0]) {
		return false
	}
	if hasArg(args, "--show-configs") {
		cmdShowConfigs(args)
		return true
	}
	if hasArg(args, "--download") {
		model := firstPositional(args)
		if model == "" {
			fmt.Fprintln(os.Stderr, "Usage: llm-server <repo/name> --download")
			os.Exit(2)
		}
		cmdDownload([]string{model})
		return true
	}
	if hasArg(args, "--ai-tune") {
		cmdTune(args)
		return true
	}
	if hasArg(args, "--benchmark") {
		if firstPositional(args) != "" {
			cmdLaunch(args)
		} else {
			cmdBenchmark(benchmarkCompatArgs(args))
		}
		return true
	}
	if hasArg(args, "--dry-run") {
		cmdDryRun(args)
		return true
	}
	if strings.HasPrefix(args[0], "-") && firstPositional(args) == "" {
		return false
	}
	cmdLaunch(args)
	return true
}

func formatCommand(args []string) string {
	quoted := make([]string, len(args))
	for i, arg := range args {
		quoted[i] = shellQuote(arg)
	}
	return strings.Join(quoted, " ")
}

func shellQuote(arg string) string {
	if arg == "" {
		return "''"
	}
	safe := true
	for _, r := range arg {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			continue
		}
		switch r {
		case '@', '%', '_', '+', '=', ':', ',', '.', '/', '-':
			continue
		default:
			safe = false
		}
	}
	if safe {
		return arg
	}
	return "'" + strings.ReplaceAll(arg, "'", "'\\''") + "'"
}

func hasArg(args []string, want string) bool {
	for _, a := range args {
		if a == want {
			return true
		}
	}
	return false
}

func benchmarkCompatArgs(args []string) []string {
	out := []string{}
	if model := firstPositional(args); model != "" {
		out = append(out, "--model", filepath.Base(model))
	}
	for i := 0; i < len(args); i++ {
		if args[i] == "--model" || args[i] == "-m" {
			if i+1 < len(args) {
				out = append(out, "--model", filepath.Base(args[i+1]))
				i++
			}
			continue
		}
		if args[i] == "--port" || args[i] == "-port" {
			if i+1 < len(args) {
				out = append(out, "--port", args[i+1])
				i++
			}
			continue
		}
		if key, val, ok := strings.Cut(args[i], "="); ok {
			switch key {
			case "--port", "-port":
				out = append(out, "--port", val)
			case "--model", "-m":
				out = append(out, "--model", filepath.Base(val))
			}
		}
	}
	return out
}

func firstPositional(args []string) string {
	skip := false
	for _, a := range args {
		if skip {
			skip = false
			continue
		}
		if a == "--" {
			return ""
		}
		if strings.HasPrefix(a, "-") {
			// Must stay in sync with the value-taking flags in parseLaunchArgs.
			switch a {
			case "--model", "-m", "--port", "-port", "--ctx", "-ctx", "--ctx-size", "-c", "--kv", "-kv", "--kv-placement", "--kv-quality", "--gpus", "--host", "--server-bin", "--mmproj", "--backend", "--tune-cache", "--rounds", "--ram-budget", "--spec", "--parallel", "--lib-path", "--threads", "-t", "--batch-size", "-b", "--ubatch-size", "-ub":
				skip = true
			}
			continue
		}
		return a
	}
	return ""
}

func cmdDetect() {
	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	data, err := caps.JSON()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(string(data))
}

type launchRequest struct {
	ModelPath         string
	Port              int
	CtxFlag           string
	KVPlacement       string
	KVQuality         string
	CPUMode           bool
	GPUsFlag          string
	Host              string
	VisionAuto        bool
	MMProjPath        string
	ServerBin         string
	ServerBinExplicit bool
	Backend           string
	BackendExplicit   bool
	TuneCache         string
	SpecMode          string
	ForceSpecMoE      bool
	RamBudgetMB       int
	Parallel          int
	Benchmark         bool
	ExtraArgs         []string
}

func parseLaunchArgs(args []string) (*launchRequest, error) {
	cfg := config.Defaults()
	if c, err := config.Load(); err == nil {
		cfg = c
	}
	req := &launchRequest{
		Port:        cfg.Port,
		CtxFlag:     cfg.CtxValue(),
		KVPlacement: cfg.KVPlacement,
		KVQuality:   cfg.KVQuality,
		Host:        cfg.Host,
		VisionAuto:  cfg.Vision,
		ServerBin:   cfg.LlamaServer,
		Backend:     cfg.Backend,
		SpecMode:    cfg.Spec,
		Parallel:    cfg.Parallel,
	}
	if req.Port == 0 {
		req.Port = 8081
	}
	if req.KVPlacement == "" {
		req.KVPlacement = "auto"
	}
	if req.KVQuality == "" {
		req.KVQuality = "low"
	}
	if req.Host == "" {
		req.Host = "0.0.0.0"
	}
	if req.SpecMode == "" {
		req.SpecMode = "off"
	}

	for i := 0; i < len(args); i++ {
		a := args[i]
		if key, val, ok := strings.Cut(a, "="); ok && strings.HasPrefix(key, "-") {
			switch key {
			case "--model", "-m":
				req.ModelPath = val
				continue
			case "--port", "-port":
				req.Port, _ = strconv.Atoi(val)
				continue
			case "--ctx", "-ctx", "--ctx-size", "-c":
				req.CtxFlag = val
				continue
			case "--kv", "-kv", "--kv-placement":
				req.KVPlacement = val
				continue
			case "--kv-quality":
				req.KVQuality = val
				continue
			case "--gpus":
				req.GPUsFlag = val
				continue
			case "--host":
				req.Host = val
				continue
			case "--mmproj":
				if val == "auto" {
					req.VisionAuto = true
				} else {
					req.MMProjPath = val
				}
				continue
			case "--server-bin":
				req.ServerBin = val
				req.ServerBinExplicit = true
				continue
			case "--backend":
				req.Backend = val
				req.BackendExplicit = true
				continue
			case "--tune-cache":
				req.TuneCache = val
				continue
			case "--rounds":
				continue
			case "--ram-budget":
				req.RamBudgetMB = parseBudgetMB(val)
				continue
			case "--spec":
				req.SpecMode = val
				continue
			case "--parallel":
				req.Parallel, _ = strconv.Atoi(val)
				continue
			}
		}
		next := func() (string, error) {
			if i+1 >= len(args) {
				return "", fmt.Errorf("%s requires a value", a)
			}
			i++
			return args[i], nil
		}
		switch a {
		case "--benchmark":
			req.Benchmark = true
			continue
		case "--dry-run", "--ai-tune", "--retune", "--download", "--show-configs", "--keep-alive":
			continue
		case "--model", "-m":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.ModelPath = v
		case "--port", "-port":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.Port, _ = strconv.Atoi(v)
		case "--ctx", "-ctx", "--ctx-size", "-c":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.CtxFlag = v
		case "--kv", "-kv", "--kv-placement":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.KVPlacement = v
		case "--kv-quality":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.KVQuality = v
		case "--cpu":
			req.CPUMode = true
		case "--gpus":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.GPUsFlag = v
		case "--host":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.Host = v
		case "--vision":
			req.VisionAuto = true
		case "--mmproj":
			v, err := next()
			if err != nil {
				return nil, err
			}
			if v == "auto" {
				req.VisionAuto = true
			} else {
				req.MMProjPath = v
			}
		case "--server-bin":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.ServerBin = v
			req.ServerBinExplicit = true
		case "--backend":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.Backend = v
			req.BackendExplicit = true
		case "--tune-cache":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.TuneCache = v
		case "--rounds":
			_, err := next()
			if err != nil {
				return nil, err
			}
		case "--ram-budget":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.RamBudgetMB = parseBudgetMB(v)
		case "--spec":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.SpecMode = v
		case "--parallel":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.Parallel, _ = strconv.Atoi(v)
		case "--force-spec-moe":
			req.ForceSpecMoE = true
		case "--":
			req.ExtraArgs = append(req.ExtraArgs, args[i+1:]...)
			i = len(args)
		default:
			if strings.HasPrefix(a, "-") {
				req.ExtraArgs = append(req.ExtraArgs, a)
				if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
					i++
					req.ExtraArgs = append(req.ExtraArgs, args[i])
				}
				continue
			}
			if req.ModelPath == "" {
				req.ModelPath = a
			} else {
				req.ExtraArgs = append(req.ExtraArgs, a)
			}
		}
	}
	return req, nil
}

// applyGPUVisibility restricts which devices the backend can enumerate so the
// computed placement (tensor splits, -ot device names, renumbered indices)
// matches reality. Returns the env assignment for display, or "" when --gpus
// was not given.
func applyGPUVisibility(req *launchRequest, backendTag string) string {
	if req == nil || req.GPUsFlag == "" {
		return ""
	}
	seen := map[int]bool{}
	indices := []int{}
	for _, s := range strings.Split(req.GPUsFlag, ",") {
		idx, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || idx < 0 || seen[idx] {
			continue
		}
		seen[idx] = true
		indices = append(indices, idx)
	}
	if len(indices) == 0 {
		return ""
	}
	// Keep PCI ordering so renumbered placement indices line up with the
	// backend's enumeration of the visible subset.
	sort.Ints(indices)
	parts := make([]string, len(indices))
	for i, idx := range indices {
		parts[i] = strconv.Itoa(idx)
	}
	list := strings.Join(parts, ",")
	envKey := "CUDA_VISIBLE_DEVICES"
	if strings.EqualFold(backendTag, "vulkan") {
		envKey = "GGML_VK_VISIBLE_DEVICES"
	}
	os.Setenv(envKey, list)
	return envKey + "=" + list
}

func resolveModelPath(path, modelDir string) string {
	if path == "" || filepath.IsAbs(path) {
		return path
	}
	if _, err := os.Stat(path); err == nil {
		return path
	}
	if modelDir == "" {
		return path
	}
	candidate := filepath.Join(modelDir, path)
	if _, err := os.Stat(candidate); err == nil {
		return candidate
	}
	return path
}

func parseBudgetMB(s string) int {
	s = strings.TrimSpace(strings.ToUpper(s))
	mult := 1
	if strings.HasSuffix(s, "G") || strings.HasSuffix(s, "GB") {
		mult = 1024
		s = strings.TrimSuffix(strings.TrimSuffix(s, "GB"), "G")
	} else if strings.HasSuffix(s, "M") || strings.HasSuffix(s, "MB") {
		s = strings.TrimSuffix(strings.TrimSuffix(s, "MB"), "M")
	}
	n, _ := strconv.Atoi(strings.TrimSpace(s))
	return n * mult
}

func selectBackend(caps *detect.Capabilities, req *launchRequest) *backendInfo {
	want := strings.TrimSpace(req.Backend)
	useExplicitServerBin := req.ServerBin != "" && (req.ServerBinExplicit || !req.BackendExplicit || want == "" || want == "auto")
	if useExplicitServerBin {
		if _, err := os.Stat(req.ServerBin); err == nil {
			return detectBackend(req.ServerBin)
		}
		fmt.Fprintf(os.Stderr, "Warning: server binary not found: %s\n", req.ServerBin)
	}
	if want != "" && want != "auto" {
		for _, b := range caps.Backends {
			info := detectBackend(b.Path)
			if backendMatches(info, b.Name, want) {
				return info
			}
		}
		for _, p := range backendSearchPaths() {
			if p == "" {
				continue
			}
			if _, err := os.Stat(p); err == nil {
				info := detectBackend(p)
				if backendMatches(info, filepath.Base(p), want) {
					return info
				}
			}
		}
	}
	if req.ServerBin != "" && !useExplicitServerBin {
		if _, err := os.Stat(req.ServerBin); err == nil {
			return detectBackend(req.ServerBin)
		}
		fmt.Fprintf(os.Stderr, "Warning: server binary not found: %s\n", req.ServerBin)
	}
	return findBackend(caps)
}

func backendMatches(info *backendInfo, name, want string) bool {
	want = strings.TrimSpace(strings.ToLower(want))
	if want == "" || want == "auto" {
		return true
	}
	name = strings.ToLower(name)
	tag := strings.ToLower(info.Tag)
	return tag == want || name == want ||
		(want == "ik" && tag == "ik_llama") ||
		(want == "llama" && tag == "llama") ||
		(want == "vulkan" && (tag == "vulkan" || strings.Contains(strings.ToLower(info.Path), "vulkan"))) ||
		(want == "llama-vk" && tag == "vulkan")
}

func placementOptionsFromRequest(req *launchRequest, model *placement.ModelProfile, be *backendInfo, cacheDir string) placement.Options {
	opts := placement.Options{
		ContextSize:  resolveCtxFlag(req.CtxFlag, model.CTXTrain),
		KVPlacement:  req.KVPlacement,
		KVQuality:    req.KVQuality,
		CPUMode:      req.CPUMode,
		RamBudgetMB:  req.RamBudgetMB,
		CacheDir:     cacheDir,
		Host:         req.Host,
		BackendTag:   be.Tag,
		VisionAuto:   req.VisionAuto,
		MMProjPath:   req.MMProjPath,
		SpecMode:     req.SpecMode,
		ForceSpecMoE: req.ForceSpecMoE,
		BackendHelp:  be.Help,
		CacheFile:    req.TuneCache,
		Parallel:     req.Parallel,
	}
	if req.GPUsFlag != "" {
		for _, s := range strings.Split(req.GPUsFlag, ",") {
			idx, _ := strconv.Atoi(strings.TrimSpace(s))
			opts.GPUs = append(opts.GPUs, idx)
		}
	}
	return opts
}

func cmdLaunch(args []string) {
	req, err := parseLaunchArgs(args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(2)
	}
	if req.ModelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: llm-server launch <model.gguf>")
		os.Exit(2)
	}

	cfg := config.Defaults()
	if c, err := config.Load(); err == nil {
		cfg = c
	}

	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}

	req.ModelPath = resolveModelPath(req.ModelPath, cfg.ModelDir)

	model, err := parseModel(req.ModelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing model: %v\n", err)
		os.Exit(1)
	}
	warnModelCompatibility(model)

	be := selectBackend(caps, req)
	if be == nil {
		fmt.Fprintln(os.Stderr, "Error: no llama-server binary found")
		os.Exit(1)
	}
	if env := applyGPUVisibility(req, be.Tag); env != "" {
		fmt.Printf("[launch] GPU restriction: %s\n", env)
	}
	if err := guardPortFree(req.Port, "launch"); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	strategy, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %v\n", err)
		os.Exit(1)
	}

	totalSizeMB := float64(model.SizeBytes) / (1024 * 1024)
	if len(caps.GPUs) > 0 {
		totalVRAM := int64(0)
		for _, g := range caps.GPUs {
			totalVRAM += int64(g.VRAMTotalMB) * 1024 * 1024
		}
		if model.SizeBytes > totalVRAM {
			fmt.Fprintf(os.Stderr, "[warning] Model (%.1f GB) exceeds total GPU VRAM (%.1f GB). Expect partial offload or CPU fallback.\n",
				float64(model.SizeBytes)/(1024*1024*1024), float64(totalVRAM)/(1024*1024*1024))
		}
	}

	serverArgs := append([]string{be.Path}, strategy.Args(req.ModelPath, req.Port)...)
	serverArgs = append(serverArgs, req.ExtraArgs...)
	serverArgs = applyTuneCache(req, serverArgs, cfg.CacheDir, be.Tag, strategy.MMProjPath != "", caps)
	fmt.Printf("[launch] %s\n", formatCommand(serverArgs))
	if s := placement.DraftSummary(strategy.Draft); s != "" {
		fmt.Printf("[spec]   %s\n", s)
	}

	hubDir, ok, err := libhub.Setup(be.Path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[warning] lib hub: %v\n", err)
	}
	if ok {
		os.Setenv("LLM_SERVER_LIB_HUB", hubDir)
		defer libhub.Cleanup(hubDir)
	}

	timeoutSec := 240.0 + totalSizeMB/1700.0
	if timeoutSec < 60 {
		timeoutSec = 60
	}
	if model.IsMoE && totalSizeMB > 100*1024 {
		timeoutSec = 900
	}

	p, err := server.StartWithTimeout(serverArgs, req.Port, time.Duration(timeoutSec)*time.Second)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error starting server: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("[launch] Server running on port %d (PID %d)\n", req.Port, p.Cmd.Process.Pid)
	if model.IsMoE && len(caps.GPUs) > 0 && p.LogBuf != nil {
		go placement.RunPostLaunchProbe(cfg.CacheDir, caps.GPUs, p.LogBuf.String())
	}

	if req.Benchmark {
		runOneShotBenchmark(req.Port, filepath.Base(req.ModelPath))
		if err := p.Stop(); err != nil {
			fmt.Fprintf(os.Stderr, "[launch] stop after benchmark: %v\n", err)
		}
		return
	}

	fmt.Println("[launch] Press Ctrl+C to stop")
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, shutdownSignals()...)
	<-sigCh
	fmt.Fprintln(os.Stderr, "\n[launch] Shutting down...")

	done := make(chan struct{})
	go func() {
		p.Stop()
		close(done)
	}()
	select {
	case <-done:
	case <-sigCh:
		fmt.Fprintln(os.Stderr, "[launch] Force quitting...")
		if p.Cmd.Process != nil {
			p.Cmd.Process.Kill()
		}
	case <-time.After(30 * time.Second):
		fmt.Fprintln(os.Stderr, "[launch] Timeout — forcing shutdown...")
		if p.Cmd.Process != nil {
			p.Cmd.Process.Kill()
		}
	}
}

func cmdGUI() {
	req, err := tui.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if req == nil {
		return
	}

	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}

	// Load config for cache directory
	cfg, _ := config.Load()
	if req.DownloadRepo != "" {
		d := download.New(cfg.ModelDir, cfg.CacheDir, cfg.AppHome)
		if err := d.RunQuant(req.DownloadRepo, req.DownloadQuant, caps); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	backendName := req.Backend
	if backendName == "" {
		backendName = cfg.Backend
	}
	be := selectBackend(caps, &launchRequest{ServerBin: cfg.LlamaServer, Backend: backendName})
	if be == nil {
		fmt.Fprintln(os.Stderr, "Error: no llama-server binary found")
		os.Exit(1)
	}

	model, err := parseModel(req.ModelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing model: %v\n", err)
		os.Exit(1)
	}
	warnModelCompatibility(model)

	opts := placement.Options{
		ContextSize: req.CtxSize,
		KVPlacement: req.KVPlacement,
		KVQuality:   req.KVQuality,
		BackendTag:  be.Tag,
		Parallel:    req.Parallel,
		CacheDir:    cfg.CacheDir,
		Host:        cfg.Host,
		VisionAuto:  req.Vision,
		SpecMode:    cfg.Spec,
		BackendHelp: be.Help,
	}
	if req.TuneCache != "" {
		opts.CacheFile = req.TuneCache
	}
	strategy, err := placement.Compute(caps, model, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %v\n", err)
		os.Exit(1)
	}

	if err := guardPortFree(req.Port, "launch"); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	serverArgs := append([]string{be.Path}, strategy.Args(req.ModelPath, req.Port)...)
	fmt.Printf("[launch] %s\n", formatCommand(serverArgs))
	if s := placement.DraftSummary(strategy.Draft); s != "" {
		fmt.Printf("[spec]   %s\n", s)
	}

	// Setup lib hub for non-system binaries
	hubDir, ok, err := libhub.Setup(be.Path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[warning] lib hub: %v\n", err)
	}
	if ok {
		os.Setenv("LLM_SERVER_LIB_HUB", hubDir)
		defer libhub.Cleanup(hubDir)
	}

	// Dynamic health timeout: 240 + size/1700 seconds
	totalSizeMB := float64(model.SizeBytes) / (1024 * 1024)
	timeoutSec := 240.0 + totalSizeMB/1700.0
	if timeoutSec < 60 {
		timeoutSec = 60
	}
	if model.IsMoE && totalSizeMB > 100*1024 {
		timeoutSec = 900
	}

	// Use recovery launcher with keep-alive
	launcher := recovery.DefaultLauncher(be.Path, serverArgs[1:])
	launcher.HealthTimeout = time.Duration(timeoutSec) * time.Second
	launcher.KeepAlive = true
	launcher.PlacementCachePath = req.TuneCache
	launcher.OnFailure = func(ft recovery.FailureType, msg string) {
		fmt.Fprintf(os.Stderr, "[launch] failure: %s: %s\n", ft, msg)
	}
	launcher.OnCUDAOOM = func(device int, allocMB int, args []string) ([]string, *placement.CacheEntry, bool) {
		newArgs, entry, ok := placement.DerateCUDAOOMArgs(args, model, caps, device, allocMB)
		if ok {
			fmt.Fprintf(os.Stderr, "[launch] CUDA OOM on device %d allocating %d MiB; moving expert layer(s) to CPU and retrying\n", device, allocMB)
		}
		return newArgs, entry, ok
	}
	launcher.OnRestart = func(n int, backoff time.Duration) {
		fmt.Printf("[launch] restart %d in %v...\n", n, backoff)
	}
	launcher.OnFallback = func(path string) {
		fmt.Printf("[launch] falling back to mainline: %s\n", path)
	}

	// Find fallback binary (mainline llama-server)
	if be.IsIK {
		for _, b := range caps.Backends {
			if b.Name == "llama-server" && b.Path != be.Path {
				launcher.FallbackPath = b.Path
				break
			}
		}
	}

	fmt.Printf("[launch] Server starting on port %d (health timeout %.0fs)\n", req.Port, timeoutSec)
	fmt.Println("[launch] Press Ctrl+C to stop")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Run the launcher in the background so the main goroutine can react to a
	// second Ctrl+C (force quit) if graceful shutdown stalls.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, shutdownSignals()...)
	runErr := make(chan error, 1)
	go func() { runErr <- launcher.Run(ctx) }()

	for {
		select {
		case err := <-runErr:
			// A context-cancellation error is the expected result of shutdown.
			if err != nil && ctx.Err() == nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			return
		case <-sigCh:
			fmt.Fprintln(os.Stderr, "\n[launch] Shutting down...")
			cancel()
			// Escape hatch: don't get stuck if the child won't exit.
			select {
			case <-runErr:
				return
			case <-sigCh:
				fmt.Fprintln(os.Stderr, "[launch] Force quitting...")
				os.Exit(1)
			case <-time.After(20 * time.Second):
				fmt.Fprintln(os.Stderr, "[launch] Timeout — forcing shutdown...")
				os.Exit(1)
			}
		}
	}
}

func applyTuneCache(req *launchRequest, serverArgs []string, cacheDir, backendTag string, vision bool, caps *detect.Capabilities) []string {
	if req == nil {
		return serverArgs
	}
	if req.TuneCache != "" {
		return applySelectedTuneCache(req, serverArgs, caps)
	}
	path := bestTuneCachePath(cacheDir, filepath.Base(req.ModelPath), backendTag, vision, tuneHardwareHash(caps))
	if path == "" {
		// No local tune for this model+hardware+backend: try the community
		// pool. Downloads are sanitized to the tune-flag allow-list and both
		// hits and misses are cached on disk, so launches stay offline-safe.
		path = tune.FetchCommunityTune(cacheDir, req.ModelPath, gpuNamesFromCaps(caps), vision, backendTag)
		if path == "" {
			return serverArgs
		}
		fmt.Printf("[tune] Using community-shared config: %s (LLM_COMMUNITY_TUNES=off to disable)\n", filepath.Base(path))
	} else {
		fmt.Printf("[tune] Auto-selected cached config: %s\n", filepath.Base(path))
	}
	autoReq := *req
	autoReq.TuneCache = path
	return applySelectedTuneCache(&autoReq, serverArgs, caps)
}

func gpuNamesFromCaps(caps *detect.Capabilities) []string {
	if caps == nil {
		return nil
	}
	names := make([]string, 0, len(caps.GPUs))
	for _, gpu := range caps.GPUs {
		names = append(names, gpu.Name)
	}
	return names
}

func bestTuneCachePath(cacheDir, modelName, backendTag string, vision bool, hardwareHash string) string {
	if cacheDir == "" || modelName == "" {
		return ""
	}
	rows := tune.ListTunedConfigs(cacheDir, modelName, tuneCacheBackendTag(backendTag), vision)
	for _, row := range rows {
		if hardwareHash == "" || strings.Contains(filepath.Base(row.Path), "_hw"+hardwareHash+"_") {
			return row.Path
		}
	}
	return ""
}

func tuneHardwareHash(caps *detect.Capabilities) string {
	if caps == nil {
		return ""
	}
	names := make([]string, 0, len(caps.GPUs))
	for _, gpu := range caps.GPUs {
		names = append(names, gpu.Name)
	}
	if len(names) == 0 {
		return ""
	}
	return tune.BashHardwareHash(names)
}

func tuneCacheBackendTag(backendTag string) string {
	b := strings.ToLower(strings.TrimSpace(backendTag))
	switch {
	case strings.Contains(b, "vulkan"):
		return "vulkan"
	case strings.Contains(b, "ik"):
		return "ik"
	default:
		return "llama"
	}
}

func applySelectedTuneCache(req *launchRequest, serverArgs []string, caps *detect.Capabilities) []string {
	if req == nil || req.TuneCache == "" {
		return serverArgs
	}
	summary, err := tune.LoadTuneFile(req.TuneCache, filepath.Base(req.ModelPath))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: invalid --tune-cache: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("[tune] Using selected AI-tuned config: %s\n", filepath.Base(req.TuneCache))
	if summary.BaselineWins || len(summary.Flags) == 0 {
		fmt.Println("[tune] Baseline was best; no override flags applied")
		return serverArgs
	}
	if reason := tuneCacheVRAMGuard(serverArgs, summary.Flags, caps); reason != "" {
		fmt.Printf("[tune] Skipping cached config %s: %s\n", summary.Name, reason)
		return serverArgs
	}
	serverArgs = tune.ApplyOverrides(serverArgs, summary.Flags, tune.DefaultProtectedFlags())
	fmt.Printf("[tune] Config: %s (expected %.2f tok/s)\n", summary.Name, summary.GenTPS)
	return serverArgs
}

func canonicalLaunchFlagName(flag string) string {
	if idx := strings.Index(flag, "="); idx > 0 {
		flag = flag[:idx]
	}
	switch flag {
	case "-b", "--batch-size":
		return "-b"
	case "-ub", "--ubatch-size":
		return "-ub"
	case "-np", "--parallel":
		return "--parallel"
	case "-fa", "--flash-attn":
		return "--flash-attn"
	case "--mg", "--main-gpu":
		return "-mg"
	case "-ot", "--override-tensor":
		return "-ot"
	case "--dev", "-dev", "--device":
		return "--device"
	default:
		return flag
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func tuneCacheVRAMGuard(baseArgs []string, overrides map[string]interface{}, caps *detect.Capabilities) string {
	if caps == nil || len(caps.GPUs) == 0 || !tuneOverridesIncreaseVRAM(baseArgs, overrides) {
		return ""
	}
	selected := tuneSelectedGPUIndices(baseArgs, caps)
	if len(selected) == 0 {
		return ""
	}
	minFree := 0
	minTotal := 0
	for i, idx := range selected {
		if idx < 0 || idx >= len(caps.GPUs) {
			continue
		}
		gpu := caps.GPUs[idx]
		free := gpu.VRAMFreeMB()
		if i == 0 || free < minFree {
			minFree = free
			minTotal = gpu.VRAMTotalMB
		}
	}
	if minFree <= 0 || minTotal <= 0 {
		return ""
	}
	needed := tuneRuntimeHeadroomMB(minTotal)
	if minFree < needed {
		return fmt.Sprintf("runtime VRAM headroom is low on selected GPU(s): min free %d MiB < guard %d MiB for memory-expanding flags", minFree, needed)
	}
	return ""
}

func tuneRuntimeHeadroomMB(gpuTotalMB int) int {
	guard := gpuTotalMB / 5
	if guard < 4096 {
		guard = 4096
	}
	if guard > 8192 {
		guard = 8192
	}
	return guard
}

func tuneOverridesIncreaseVRAM(baseArgs []string, overrides map[string]interface{}) bool {
	base := argMap(baseArgs)
	if tuneIntOverrideGreater(overrides, base, "-b", 2048) || tuneIntOverrideGreater(overrides, base, "-ub", 512) || tuneIntOverrideGreater(overrides, base, "--parallel", 1) {
		return true
	}
	for _, key := range []string{"--cache-type-k", "--cache-type-v"} {
		if val, ok := tuneOverrideString(overrides, key); ok && kvCacheRank(val) > kvCacheRank(base[key]) {
			return true
		}
	}
	if val, ok := tuneOverrideString(overrides, "--flash-attn"); ok && strings.EqualFold(val, "off") && !strings.EqualFold(base["--flash-attn"], "off") {
		return true
	}
	if _, ok := tuneOverrideString(overrides, "--spec-type"); ok && base["--spec-type"] == "" {
		return true
	}
	for _, key := range []string{"--spec-draft-n-max", "--draft-max", "--spec-ngram-mod-n-max"} {
		if tuneIntOverrideGreater(overrides, base, key, 0) {
			return true
		}
	}
	return false
}

func tuneIntOverrideGreater(overrides map[string]interface{}, base map[string]string, key string, fallback int) bool {
	val, ok := tuneOverrideString(overrides, key)
	if !ok {
		return false
	}
	next, err := strconv.Atoi(strings.TrimSpace(val))
	if err != nil {
		return false
	}
	cur := fallback
	if raw := strings.TrimSpace(base[key]); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil {
			cur = n
		}
	}
	return next > cur
}

func tuneOverrideString(overrides map[string]interface{}, key string) (string, bool) {
	for k, v := range overrides {
		if canonicalLaunchFlagName(k) == key {
			return strings.TrimSpace(fmt.Sprint(v)), true
		}
	}
	return "", false
}

func kvCacheRank(kind string) int {
	s := strings.ToLower(strings.TrimSpace(kind))
	s = strings.TrimPrefix(s, "ggml_")
	switch s {
	case "", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1":
		return 1
	case "q8_0", "q8_1", "bf16":
		return 2
	case "f16", "fp16", "f32", "fp32":
		return 3
	default:
		return 1
	}
}

func argMap(args []string) map[string]string {
	out := map[string]string{}
	for i := 0; i < len(args); i++ {
		key := canonicalLaunchFlagName(args[i])
		if key == "" || !strings.HasPrefix(key, "-") {
			continue
		}
		if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
			out[key] = args[i+1]
			i++
		} else {
			out[key] = "true"
		}
	}
	return out
}

func tuneSelectedGPUIndices(args []string, caps *detect.Capabilities) []int {
	seen := map[int]bool{}
	add := func(idx int) {
		if idx >= 0 && idx < len(caps.GPUs) {
			seen[idx] = true
		}
	}
	values := argMap(args)
	for _, key := range []string{"--device", "-dev", "--dev"} {
		for _, idx := range indicesFromDeviceList(values[key]) {
			add(idx)
		}
	}
	for _, idx := range indicesFromTensorSplit(values["--tensor-split"]) {
		add(idx)
	}
	for _, idx := range indicesFromDeviceList(values["-ot"]) {
		add(idx)
	}
	if len(seen) == 0 {
		for _, key := range []string{"-mg", "--main-gpu"} {
			if n, err := strconv.Atoi(strings.TrimSpace(values[key])); err == nil {
				add(n)
			}
		}
	}
	if len(seen) == 0 {
		for i := range caps.GPUs {
			add(i)
		}
	}
	out := make([]int, 0, len(seen))
	for idx := range seen {
		out = append(out, idx)
	}
	sort.Ints(out)
	return out
}

func indicesFromTensorSplit(value string) []int {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	out := []int{}
	for i, part := range parts {
		if f, err := strconv.ParseFloat(strings.TrimSpace(part), 64); err == nil && f > 0 {
			out = append(out, i)
		}
	}
	return out
}

func indicesFromDeviceList(value string) []int {
	out := []int{}
	for i := 0; i < len(value); i++ {
		if !unicode.IsDigit(rune(value[i])) {
			continue
		}
		j := i + 1
		for j < len(value) && unicode.IsDigit(rune(value[j])) {
			j++
		}
		prefix := strings.ToLower(value[maxInt(0, i-8):i])
		if strings.Contains(prefix, "cuda") || strings.Contains(prefix, "vulkan") || strings.Contains(prefix, "gpu") {
			if n, err := strconv.Atoi(value[i:j]); err == nil {
				out = append(out, n)
			}
		}
		i = j - 1
	}
	return out
}

func cmdDryRun(args []string) {
	req, err := parseLaunchArgs(args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(2)
	}
	if req.ModelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: llm-server dry-run <model.gguf>")
		os.Exit(2)
	}

	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}

	cfg := config.Defaults()
	if c, err := config.Load(); err == nil {
		cfg = c
	}
	req.ModelPath = resolveModelPath(req.ModelPath, cfg.ModelDir)

	model, err := parseModel(req.ModelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing model: %v\n", err)
		os.Exit(1)
	}
	warnModelCompatibility(model)

	be := selectBackend(caps, req)
	backendTag := "llama"
	binPath := "llama-server"
	if be != nil {
		binPath = be.Path
		backendTag = be.Tag
	} else {
		be = &backendInfo{Path: binPath, Tag: backendTag}
	}

	strategy, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %v\n", err)
		os.Exit(1)
	}

	serverArgs := append([]string{binPath}, strategy.Args(req.ModelPath, req.Port)...)
	serverArgs = append(serverArgs, req.ExtraArgs...)
	serverArgs = applyTuneCache(req, serverArgs, cfg.CacheDir, be.Tag, strategy.MMProjPath != "", caps)
	if envPrefix := applyGPUVisibility(req, be.Tag); envPrefix != "" {
		fmt.Print(envPrefix + " ")
	}
	fmt.Println(formatCommand(serverArgs))
	if s := placement.DraftSummary(strategy.Draft); s != "" {
		fmt.Printf("[spec] %s\n", s)
	}
}

func cmdShowConfigs(args []string) {
	cfg := config.Defaults()
	if c, err := config.Load(); err == nil {
		cfg = c
	}
	modelName := ""
	for _, a := range args {
		if a == "--show-configs" || strings.HasPrefix(a, "-") {
			continue
		}
		modelName = filepath.Base(a)
		break
	}
	if modelName != "" {
		var rows []tune.ConfigEntry
		for _, backend := range []string{"llama", "ik", "ik_llama", "vulkan"} {
			rows = append(rows, tune.ListTunedConfigs(cfg.CacheDir, modelName, backend, false)...)
			rows = append(rows, tune.ListTunedConfigs(cfg.CacheDir, modelName, backend, true)...)
		}
		sort.Slice(rows, func(i, j int) bool { return rows[i].GenTPS > rows[j].GenTPS })
		if len(rows) == 0 {
			fmt.Printf("No tuned configs found for %s in %s\n", modelName, cfg.CacheDir)
			return
		}
		for _, row := range rows {
			fmt.Printf("%s\n  %s\n", row.Label, row.Path)
		}
		return
	}

	matches, _ := filepath.Glob(filepath.Join(cfg.CacheDir, "tune_*.json"))
	sort.Strings(matches)
	if len(matches) == 0 {
		fmt.Printf("No tuned configs found in %s\n", cfg.CacheDir)
		return
	}
	for _, path := range matches {
		fmt.Println(path)
	}
}

func cmdProbe() {
	mem, err := probe.Probe()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(mem.String())
}

func cmdDownload(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "Usage: llm-server download <repo/name>")
		os.Exit(2)
	}
	repo := args[0]

	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}

	cfg := config.Defaults()
	if f, err := config.Load(); err == nil {
		cfg = f
	}

	d := download.New(cfg.ModelDir, cfg.CacheDir, cfg.AppHome)
	if err := d.Run(repo, caps); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func tuneRoundsFromArgs(args []string, fallback int) int {
	if fallback <= 0 {
		fallback = 3
	}
	for i := 0; i < len(args); i++ {
		if key, val, ok := strings.Cut(args[i], "="); ok && (key == "--rounds" || key == "-rounds") {
			if n, err := strconv.Atoi(val); err == nil && n > 0 {
				return n
			}
		}
		if args[i] == "--rounds" || args[i] == "-rounds" {
			if i+1 < len(args) {
				if n, err := strconv.Atoi(args[i+1]); err == nil && n > 0 {
					return n
				}
			}
		}
	}
	return fallback
}

func cmdTune(args []string) {
	req, err := parseLaunchArgs(args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(2)
	}
	if req.ModelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: llm-server tune <model.gguf>")
		os.Exit(2)
	}

	cfg := config.Defaults()
	if f, err := config.Load(); err == nil {
		cfg = f
	}
	rounds := tuneRoundsFromArgs(args, cfg.TuneRounds)

	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}

	req.ModelPath = resolveModelPath(req.ModelPath, cfg.ModelDir)

	model, err := parseModel(req.ModelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing model: %v\n", err)
		os.Exit(1)
	}
	warnModelCompatibility(model)

	be := selectBackend(caps, req)
	if be == nil {
		fmt.Fprintln(os.Stderr, "Error: no llama-server binary found")
		os.Exit(1)
	}
	if env := applyGPUVisibility(req, be.Tag); env != "" {
		fmt.Printf("[tune] GPU restriction: %s\n", env)
	}

	strategy, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %v\n", err)
		os.Exit(1)
	}
	strategy.BackendTag = be.Tag

	// A completed tune for this model/hardware/backend is reused unless the
	// user explicitly asks for a fresh run with --retune.
	if !hasArg(args, "--retune") {
		cachePath := tune.TuneCachePath(cfg.CacheDir, req.ModelPath, gpuNamesFromCaps(caps), strategy.MMProjPath != "", be.Tag)
		if cachePath != "" && tune.TuneFileComplete(cachePath) {
			fmt.Printf("[tune] Completed tune cache found: %s\n", cachePath)
			fmt.Println("[tune] It is applied automatically on launch. Re-run with --retune to tune again.")
			return
		}
	}

	serverArgs := append([]string{be.Path}, strategy.Args(req.ModelPath, req.Port)...)
	serverArgs = append(serverArgs, req.ExtraArgs...)
	if err := guardPortFree(req.Port, "AI Tune"); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	totalSizeMB := float64(model.SizeBytes) / (1024 * 1024)
	timeoutSec := 240.0 + totalSizeMB/1700.0
	if timeoutSec < 60 {
		timeoutSec = 60
	}
	if model.IsMoE && totalSizeMB > 100*1024 {
		timeoutSec = 900
	}
	benchTimeout := 2 * time.Minute
	if strategy.Type == placement.CPUOnly {
		benchTimeout = 5 * time.Minute
	} else if strategy.Type == placement.MoEOffload {
		benchTimeout = 90 * time.Second
	}

	cache := tune.NewCache(cfg.CacheDir)
	engine := &tune.Engine{
		BaseURL:          fmt.Sprintf("http://localhost:%d", req.Port),
		Model:            filepath.Base(req.ModelPath),
		Rounds:           rounds,
		Cache:            cache,
		Caps:             caps,
		Backend:          be.Tag,
		Vision:           strategy.MMProjPath != "",
		BenchmarkTimeout: benchTimeout,
		BackendHelp:      be.Help,
		OnProgress: func(msg string) {
			fmt.Println("[tune]", msg)
		},
		StartServer: func(flags []string) (func(), error) {
			p, err := server.StartWithTimeout(flags, req.Port, time.Duration(timeoutSec)*time.Second)
			if err != nil {
				return nil, err
			}
			return func() { _ = p.Stop() }, nil
		},
	}

	entry, err := engine.Run(req.ModelPath, serverArgs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("[tune] Best config: %.1f tok/s\n", entry.Result.GenTPS)
	tunePath := tune.TuneCachePath(cfg.CacheDir, req.ModelPath, gpuNamesFromCaps(caps), strategy.MMProjPath != "", be.Tag)
	if hint := tune.ShareHint(tunePath); hint != "" {
		fmt.Println(hint)
	}
}

// guardPortFree refuses to start when something is already listening on the
// port. Without this, the health check can hit the EXISTING server and report
// a dead child process as "running".
func guardPortFree(port int, context string) error {
	addr := fmt.Sprintf("127.0.0.1:%d", port)
	conn, err := net.DialTimeout("tcp", addr, 200*time.Millisecond)
	if err != nil {
		return nil
	}
	_ = conn.Close()
	return fmt.Errorf("port %d is already in use; choose a free --port for %s", port, context)
}

func cmdBenchmark(args []string) {
	fs := flag.NewFlagSet("benchmark", flag.ExitOnError)
	port := fs.Int("port", 8081, "Server port")
	model := fs.String("model", "default", "Model name")
	fs.Parse(args)
	runOneShotBenchmark(*port, *model)
}

func runOneShotBenchmark(port int, model string) {
	runner := &benchmark.Runner{
		BaseURL: fmt.Sprintf("http://localhost:%d", port),
		Model:   model,
	}
	res, err := runner.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	data, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println(string(data))
}

// computeServerArgs runs hardware detection + placement for a model and
// returns the full llama-server argv (backend path first). This is the
// single source of truth for "how should this model be launched on this
// box" — used for both the daemon's initial model and any /reload swap.
func computeServerArgs(modelPath string, port int) ([]string, error) {
	caps, err := detect.Detect()
	if err != nil {
		return nil, fmt.Errorf("detect hardware: %w", err)
	}
	model, err := parseModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("parse model: %w", err)
	}
	cfg := config.Defaults()
	if c, err := config.Load(); err == nil {
		cfg = c
	}
	// Find the backend FIRST so its tag feeds placement — otherwise the
	// split-mode/flag selection can't tell ik_llama from mainline and emits
	// flags the backend rejects (e.g. `--split-mode row`, unsupported by ik).
	be := selectBackend(caps, &launchRequest{ServerBin: cfg.LlamaServer, Backend: cfg.Backend})
	if be == nil {
		return nil, fmt.Errorf("no llama-server binary found")
	}
	opts := placement.Options{
		ContextSize: resolveCtxFlag(cfg.CtxValue(), model.CTXTrain),
		KVPlacement: cfg.KVPlacement,
		KVQuality:   cfg.KVQuality,
		CacheDir:    cfg.CacheDir,
		Host:        cfg.Host,
		BackendTag:  be.Tag,
		VisionAuto:  cfg.Vision,
		SpecMode:    cfg.Spec,
	}
	strategy, err := placement.Compute(caps, model, opts)
	if err != nil {
		return nil, fmt.Errorf("compute placement: %w", err)
	}
	strategy.BackendTag = be.Tag
	return append([]string{be.Path}, strategy.Args(modelPath, port)...), nil
}

func cmdDaemon(args []string) {
	fs := flag.NewFlagSet("daemon", flag.ExitOnError)
	modelPath := fs.String("model", "", "Model path")
	port := fs.Int("port", 8081, "Server port")
	controlPort := fs.Int("control-port", 9090, "Control API port")
	startupTimeoutSecs := fs.Int("startup-timeout-secs", 300, "Max seconds to wait for llama-server to become healthy after start/reload")
	fs.Parse(args)

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: llm-server daemon --model <model.gguf>")
		os.Exit(2)
	}

	serverArgs, err := computeServerArgs(*modelPath, *port)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	d := daemon.New(daemon.Config{
		ModelPath:          *modelPath,
		ServerArgs:         serverArgs,
		Port:               *port,
		ControlPort:        *controlPort,
		StartupTimeoutSecs: *startupTimeoutSecs,
		// Let /reload recompute placement when handed a bare model path,
		// so model swaps get the same auto-placement as the initial launch.
		ComputeArgs: computeServerArgs,
	})
	if err := d.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func cmdConfig(args []string) {
	sub := "show"
	if len(args) > 0 {
		sub = args[0]
	}
	switch sub {
	case "show", "":
		cfg, err := config.Load()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(cfg.Show())
	case "path":
		fmt.Println(config.Path())
	case "edit":
		if err := config.Edit(); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("Saved.")
	case "reset":
		if err := config.Reset(); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("Config reset. Built-in defaults will be used.")
	default:
		fmt.Fprintln(os.Stderr, "Usage: llm-server config [show|edit|path|reset]")
		os.Exit(2)
	}
}

func cmdUpdate() {
	// Self-update llm-server
	if err := update.SelfUpdate(); err != nil {
		fmt.Fprintf(os.Stderr, "Self-update: %v\n", err)
	}
	if runtime.GOOS == "windows" {
		fmt.Println("Backend updates are handled by the native Windows release bundle.")
	} else {
		// Update source-built backends
		if err := update.UpdateBackends(); err != nil {
			fmt.Fprintf(os.Stderr, "Backend update: %v\n", err)
		}
	}

	// Check for newer version on GitHub
	res, err := update.Check()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Version check: %v\n", err)
		return
	}
	if res.HasUpdate {
		fmt.Printf("\nA newer version is available: %s (current: %s)\n", res.Latest, res.Current)
		fmt.Printf("Release page: %s\n", res.URL)
	} else {
		fmt.Printf("\nYou are on the latest version: %s\n", res.Current)
	}
}

func warnModelCompatibility(model *placement.ModelProfile) {
	if isDeepSeekV4FlashMistag(model) {
		fmt.Fprintln(os.Stderr, "[warning] DeepSeek V4 Flash mistagged as deepseek2. Stock llama.cpp builds may reject this GGUF; use antirez/llama.cpp-deepseek-v4-flash or a build with PR #22378 support.")
	}
}

func isDeepSeekV4FlashMistag(model *placement.ModelProfile) bool {
	if model == nil {
		return false
	}
	name := strings.ToLower(model.Name + " " + model.Basename + " " + filepath.Base(model.Path))
	if !strings.Contains(name, "deepseek") || !strings.Contains(name, "v4") || !strings.Contains(name, "flash") {
		return false
	}
	if strings.ToLower(model.ModelArch) != "deepseek2" {
		return false
	}
	return model.KeyLengthMLA > 0 && model.RopeDim > 0 && model.KeyLengthMLA <= model.RopeDim
}

// infoToProfile converts gguf.Info to placement.ModelProfile.
func infoToProfile(info *gguf.Info, path string) *placement.ModelProfile {
	numExperts := info.Experts
	if numExperts == 0 {
		numExperts = info.Fused
	}

	// Compute attention head count: embd / key_length
	// (GGUF only exposes KV head count; total heads = embd / head_dim where head_dim = kl)
	headCount := 0
	if info.KeyLength > 0 {
		headCount = info.EmbeddingLength / info.KeyLength
	}

	totalBytes := info.NonExpertBytes + info.ExpertBytes
	totalSizeMB := int(totalBytes / 1024 / 1024)

	return &placement.ModelProfile{
		Path:               path,
		Name:               info.Name,
		Basename:           info.Basename,
		QuantizedBy:        info.QuantizedBy,
		SizeBytes:          totalBytes,
		TotalSizeMB:        totalSizeMB,
		NumLayers:          info.BlockCount,
		NumParams:          info.EstimateParams(),
		IsMoE:              info.IsMoE,
		NumExperts:         numExperts,
		ContextSize:        info.ContextLength,
		HiddenSize:         info.EmbeddingLength,
		HeadCount:          headCount,
		HeadCountKV:        info.HeadCountKV,
		KeyLength:          info.KeyLength,
		ValueLength:        info.ValueLength,
		VocabSize:          info.VocabSize,
		QuantType:          "", // not parsed from gguf.py output
		ExpertBytes:        info.ExpertBytes,
		NonExpertBytes:     info.NonExpertBytes,
		Fused:              info.Fused,
		EmbeddingLength:    info.EmbeddingLength,
		FeedForwardLength:  info.FeedForwardLength,
		ExpertUsedCount:    info.ExpertUsed,
		ExpertFF:           info.ExpFF,
		ExpertSharedFF:     info.ExpSharedFF,
		LeadingDense:       info.LeadingDense,
		RopeDim:            info.NRot,
		HasSSM:             info.SSM,
		FullAttnInterval:   info.FullAttnInterval,
		SlidingWindow:      info.SlidingWindow,
		HasShexp:           info.HasShexp,
		KVLoraRank:         info.KVLoraRank,
		QLoraRank:          info.QLoraRank,
		KeyLengthMLA:       info.KeyLengthMLA,
		ValueLengthMLA:     info.ValueLengthMLA,
		CTXTrain:           info.ContextLength,
		ModelArch:          info.Architecture,
		NextNPredictLayers: info.NextNPredictLayers,
	}
}

// parseModel calls parse_gguf.py to extract real model metadata.
// For multi-part models, it sums all shard files for total size.
func parseModel(path string) (*placement.ModelProfile, error) {
	info, err := gguf.Parse(path)
	if err != nil {
		return nil, fmt.Errorf("parse_gguf.py failed: %w", err)
	}

	profile := infoToProfile(info, path)

	// Handle multi-part models: sum all shard files
	profile.SizeBytes = totalModelSize(path)

	return profile, nil
}

// totalModelSize returns the total bytes of a model, including all shards.
func totalModelSize(path string) int64 {
	dir := filepath.Dir(path)
	base := filepath.Base(path)

	// Check if this is a shard (e.g., model-00001-of-00003.gguf)
	if !strings.Contains(base, "-of-") {
		info, err := os.Stat(path)
		if err == nil {
			return info.Size()
		}
		return 0
	}

	// Find the prefix before the shard number
	// e.g., "model-00001-of-00003.gguf" -> prefix "model-"
	idx := strings.Index(base, "-000")
	if idx < 0 {
		info, err := os.Stat(path)
		if err == nil {
			return info.Size()
		}
		return 0
	}
	prefix := base[:idx]
	ext := filepath.Ext(base)

	var total int64
	entries, err := os.ReadDir(dir)
	if err != nil {
		info, err := os.Stat(path)
		if err == nil {
			return info.Size()
		}
		return 0
	}
	for _, entry := range entries {
		name := entry.Name()
		if strings.HasPrefix(name, prefix) && strings.HasSuffix(name, ext) && strings.Contains(name, "-of-") {
			fi, err := entry.Info()
			if err == nil {
				total += fi.Size()
			}
		}
	}
	if total == 0 {
		info, err := os.Stat(path)
		if err == nil {
			return info.Size()
		}
	}
	return total
}

type backendInfo struct {
	Path              string
	IsIK              bool
	SupportsReasoning bool
	Tag               string
	Help              string
}

// resolveCtxFlag converts --ctx flag to int: ""/"fit"=0, "max"=native, else number.
func resolveCtxFlag(s string, nativeCtx int) int {
	s = strings.TrimSpace(s)
	if s == "" || s == "fit" || s == "auto" {
		return 0
	}
	if s == "max" || s == "native" {
		if nativeCtx > 0 {
			return nativeCtx
		}
		return 65536
	}
	if n, err := strconv.Atoi(s); err == nil && n > 0 {
		return n
	}
	return 0
}

func findBackend(caps *detect.Capabilities) *backendInfo {
	// Try detected backends first
	for _, b := range caps.Backends {
		if b.Name == "llama-server" || b.Name == "ik_llama" || b.Name == "ik_llama-server" {
			return detectBackend(b.Path)
		}
	}
	for _, p := range backendSearchPaths() {
		if p != "" {
			if _, err := os.Stat(p); err == nil {
				return detectBackend(p)
			}
		}
	}
	return nil
}

func backendSearchPaths() []string {
	home := os.Getenv("HOME")
	if home == "" {
		home, _ = os.UserHomeDir()
	}
	appHome := os.Getenv("LLM_APP_HOME")
	if appHome == "" {
		if exe, err := os.Executable(); err == nil {
			exeDir := filepath.Dir(exe)
			switch filepath.Base(exeDir) {
			case ".bin", "bin":
				appHome = filepath.Dir(exeDir)
			}
		}
	}
	return []string{
		os.Getenv("LLAMA_SERVER"),
		filepath.Join(appHome, ".bin", "llama-server-cuda"),
		filepath.Join(appHome, ".bin", "llama-server-cuda.exe"),
		filepath.Join(appHome, ".bin", "ik_llama-server-cuda"),
		filepath.Join(appHome, ".bin", "ik_llama-server-cuda.exe"),
		filepath.Join(appHome, ".bin", "llama-server-vulkan"),
		filepath.Join(appHome, ".bin", "llama-server-vulkan.exe"),
		filepath.Join(appHome, ".bin", "llama-server"),
		filepath.Join(appHome, ".bin", "llama-server.exe"),
		filepath.Join(appHome, "bin", "llama-server"),
		filepath.Join(appHome, "bin", "llama-server.exe"),
		filepath.Join(appHome, ".src", "llama.cpp", "build-cuda", "bin", "llama-server"),
		filepath.Join(appHome, ".src", "llama.cpp", "build-cuda", "bin", "llama-server.exe"),
		filepath.Join(appHome, ".src", "ik_llama.cpp", "build", "bin", "llama-server"),
		filepath.Join(appHome, ".src", "ik_llama.cpp", "build", "bin", "llama-server.exe"),
		filepath.Join(appHome, ".src", "llama.cpp", "build-vulkan", "bin", "llama-server"),
		filepath.Join(appHome, ".src", "llama.cpp", "build-vulkan", "bin", "llama-server.exe"),
		filepath.Join(appHome, ".src", "llama.cpp", "build", "bin", "llama-server"),
		filepath.Join(appHome, ".src", "llama.cpp", "build", "bin", "llama-server.exe"),
		filepath.Join(home, "ik_llama.cpp", "build", "bin", "llama-server"),
		filepath.Join(home, "ik_llama.cpp", "build", "bin", "llama-server.exe"),
		filepath.Join(home, "llama.cpp", "build-cuda", "bin", "llama-server"),
		filepath.Join(home, "llama.cpp", "build-cuda", "bin", "llama-server.exe"),
		filepath.Join(home, "llama.cpp", "build-vulkan", "bin", "llama-server"),
		filepath.Join(home, "llama.cpp", "build-vulkan", "bin", "llama-server.exe"),
		filepath.Join(home, "llama.cpp", "build", "bin", "llama-server"),
		filepath.Join(home, "llama.cpp", "build", "bin", "llama-server.exe"),
		"/usr/local/bin/llama-server",
		"/usr/bin/llama-server",
	}
}

// detectBackend runs --help to determine if this is ik_llama.cpp fork.
// llama-server --help returns exit code 1, so we check the output regardless of error.
func detectBackend(path string) *backendInfo {
	info := &backendInfo{Path: path, Tag: "llama"}
	out, _ := exec.Command(path, "--help").CombinedOutput()
	help := string(out)
	info.Help = help
	lowerHelp := strings.ToLower(help)
	lowerPath := strings.ToLower(path)
	if strings.Contains(help, "ikawrakow") || strings.Contains(help, "split-mode-graph") {
		info.IsIK = true
		info.Tag = "ik_llama"
	} else if strings.Contains(lowerHelp, "vulkan") || strings.Contains(lowerPath, "build-vulkan") || strings.Contains(lowerPath, "vulkan") {
		info.Tag = "vulkan"
	} else if runtime.GOOS == "darwin" {
		// macOS llama.cpp builds default to Metal; placement must not emit
		// CUDA/Vulkan device-routing flags for them.
		info.Tag = "metal"
	}
	if strings.Contains(help, "--reasoning") {
		info.SupportsReasoning = true
	}
	return info
}
