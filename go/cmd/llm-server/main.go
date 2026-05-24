package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

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
	"github.com/raketenkater/llm-server/pkg/tune"
	"github.com/raketenkater/llm-server/pkg/tui"
	"github.com/raketenkater/llm-server/pkg/update"
)

const version = "v3.0.0-go"

func main() {
	if len(os.Args) < 2 {
		cmdGUI()
		return
	}

	switch os.Args[1] {
	case "version", "--version", "-v":
		fmt.Println("llm-server", version)
	case "detect":
		cmdDetect()
	case "launch":
		cmdLaunch(os.Args[2:])
	case "benchmark":
		cmdBenchmark(os.Args[2:])
	case "daemon":
		cmdDaemon(os.Args[2:])
	case "dry-run":
		cmdDryRun(os.Args[2:])
	case "probe":
		cmdProbe()
	case "download":
		cmdDownload(os.Args[2:])
	case "tune":
		cmdTune(os.Args[2:])
	case "gui", "tui":
		cmdGUI()
	case "config":
		cmdConfig(os.Args[2:])
	case "update":
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
  update               Update llm-server and backends
  gui, tui             Interactive TUI (model picker, settings, launch)

Launch flags:
  -port int            Server port (default 8081)
  -ctx int             Context size (default auto)
  -kv string           KV placement: auto|gpu|cpu (default auto)
  -kv-quality string   KV quality: high|mid|low (default mid)
  -cpu                 Force CPU-only mode
  -gpus string         Comma-separated GPU indices
  -vision              Enable vision (auto-detect mmproj)
`)
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

func cmdLaunch(args []string) {
	fs := flag.NewFlagSet("launch", flag.ExitOnError)
	port := fs.Int("port", 8081, "Server port")
	ctxFlag := fs.String("ctx", "", "Context size: fit, max, or number")
	kvPlacement := fs.String("kv", "auto", "KV placement")
	kvQuality := fs.String("kv-quality", "mid", "KV quality")
	cpuMode := fs.Bool("cpu", false, "Force CPU-only")
	gpusFlag := fs.String("gpus", "", "Comma-separated GPU indices")
	hostFlag := fs.String("host", "", "Listen address (default from config or 0.0.0.0)")
	visionAuto := fs.Bool("vision", false, "Enable vision (auto-detect/download mmproj)")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "Usage: llm-server launch <model.gguf>")
		os.Exit(2)
	}
	modelPath := fs.Arg(0)

	// Load config for CacheDir, Host, etc.
	cfg := config.Defaults()
	if c, err := config.Load(); err == nil {
		cfg = c
	}

	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}

	// Parse model profile from GGUF (use parse_gguf.py for now)
	model, err := parseModel(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing model: %v\n", err)
		os.Exit(1)
	}

	host := cfg.Host
	if *hostFlag != "" {
		host = *hostFlag
	}

	// Find backend binary and detect type BEFORE computing placement,
	// so split-mode and ik_llama flags are decided correctly.
	be := findBackend(caps)
	if be == nil {
		fmt.Fprintln(os.Stderr, "Error: no llama-server binary found")
		os.Exit(1)
	}

	opts := placement.Options{
		ContextSize: resolveCtxFlag(*ctxFlag, model.CTXTrain),
		KVPlacement: *kvPlacement,
		KVQuality:   *kvQuality,
		CPUMode:     *cpuMode,
		CacheDir:    cfg.CacheDir,
		Host:        host,
		BackendTag:  be.Tag,
		VisionAuto:  *visionAuto,
	}
	if *gpusFlag != "" {
		for _, s := range strings.Split(*gpusFlag, ",") {
			idx, _ := strconv.Atoi(strings.TrimSpace(s))
			opts.GPUs = append(opts.GPUs, idx)
		}
	}

	strategy, err := placement.Compute(caps, model, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %v\n", err)
		os.Exit(1)
	}

	// Memory warning before launch
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

	serverArgs := append([]string{be.Path}, strategy.Args(modelPath, *port)...)
	fmt.Printf("[launch] %s\n", strings.Join(serverArgs, " "))
	if s := placement.DraftSummary(strategy.Draft); s != "" {
		fmt.Printf("[spec]   %s\n", s)
	}

	// Dynamic health timeout: 240 + size/1700 seconds, min 60s
	timeoutSec := 240.0 + totalSizeMB/1700.0
	if timeoutSec < 60 {
		timeoutSec = 60
	}
	if model.IsMoE && totalSizeMB > 100*1024 {
		timeoutSec = 900 // Large MoE needs more time
	}

	p, err := server.StartWithTimeout(serverArgs, *port, time.Duration(timeoutSec)*time.Second)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error starting server: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("[launch] Server running on port %d (PID %d)\n", *port, p.Cmd.Process.Pid)

	// Post-launch system probe: measure CUDA overhead from actual VRAM usage
	// and buffer sizes reported by llama-server. Caches result for next launch.
	if model.IsMoE && len(caps.GPUs) > 0 && p.LogBuf != nil {
		go placement.RunPostLaunchProbe(opts.CacheDir, caps.GPUs, p.LogBuf.String())
	}

	fmt.Println("[launch] Press Ctrl+C to stop")

	// Wait for shutdown signal, then clean up the child process group.
	// Without this, Ctrl+C kills only the Go binary — llama-server
	// becomes an orphan, leaking VRAM and leaving zombie processes.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	fmt.Fprintln(os.Stderr, "\n[launch] Shutting down...")

	// Stop with a hard deadline — second Ctrl+C will force-exit.
	done := make(chan struct{})
	go func() {
		p.Stop()
		close(done)
	}()
	select {
	case <-done:
	case <-sigCh:
		fmt.Fprintln(os.Stderr, "[launch] Force quitting...")
		_ = syscall.Kill(-p.Cmd.Process.Pid, syscall.SIGKILL)
	case <-time.After(30 * time.Second):
		fmt.Fprintln(os.Stderr, "[launch] Timeout — forcing shutdown...")
		_ = syscall.Kill(-p.Cmd.Process.Pid, syscall.SIGKILL)
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

	be := findBackend(caps)
	if be == nil {
		fmt.Fprintln(os.Stderr, "Error: no llama-server binary found")
		os.Exit(1)
	}

	model, err := parseModel(req.ModelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing model: %v\n", err)
		os.Exit(1)
	}

	opts := placement.Options{
		ContextSize: req.CtxSize,
		KVPlacement: req.KVPlacement,
		KVQuality:   req.KVQuality,
		BackendTag:  be.Tag,
		Parallel:    req.Parallel,
		CacheDir:    cfg.CacheDir,
		Host:        cfg.Host,
		VisionAuto:  req.Vision,
	}
	if req.TuneCache != "" {
		opts.CacheFile = req.TuneCache
	}
	strategy, err := placement.Compute(caps, model, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %v\n", err)
		os.Exit(1)
	}

	serverArgs := append([]string{be.Path}, strategy.Args(req.ModelPath, req.Port)...)
	fmt.Printf("[launch] %s\n", strings.Join(serverArgs, " "))
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
	launcher.OnFailure = func(ft recovery.FailureType, msg string) {
		fmt.Fprintf(os.Stderr, "[launch] failure: %s: %s\n", ft, msg)
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

	// Cancel context on SIGINT/SIGTERM so the launcher cleans up the child
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Fprintln(os.Stderr, "\n[launch] Shutting down...")
		cancel()
	}()

	if err := launcher.Run(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func cmdDryRun(args []string) {
	fs := flag.NewFlagSet("dry-run", flag.ExitOnError)
	port := fs.Int("port", 8081, "Server port")
	ctxFlag := fs.String("ctx", "", "Context size: fit, max, or number")
	kvPlacement := fs.String("kv", "auto", "KV placement")
	kvQuality := fs.String("kv-quality", "mid", "KV quality")
	cpuMode := fs.Bool("cpu", false, "Force CPU-only")
	visionAuto := fs.Bool("vision", false, "Enable vision (auto-detect/download mmproj)")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "Usage: llm-server dry-run <model.gguf>")
		os.Exit(2)
	}
	modelPath := fs.Arg(0)

	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}

	model, err := parseModel(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing model: %v\n", err)
		os.Exit(1)
	}

	// Detect backend BEFORE computing placement (affects split-mode, MoE flags, etc.)
	be := findBackend(caps)
	backendTag := "llama"
	binPath := "llama-server"
	if be != nil {
		binPath = be.Path
		backendTag = be.Tag
	}

	cfg := config.Defaults()
	if c, err := config.Load(); err == nil {
		cfg = c
	}

	strategy, err := placement.Compute(caps, model, placement.Options{
		ContextSize: resolveCtxFlag(*ctxFlag, model.CTXTrain),
		KVPlacement: *kvPlacement,
		KVQuality:   *kvQuality,
		CPUMode:     *cpuMode,
		CacheDir:    cfg.CacheDir,
		Host:        cfg.Host,
		BackendTag:  backendTag,
		VisionAuto:  *visionAuto,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %v\n", err)
		os.Exit(1)
	}

	serverArgs := append([]string{binPath}, strategy.Args(modelPath, *port)...)
	fmt.Println(strings.Join(serverArgs, " "))
	if s := placement.DraftSummary(strategy.Draft); s != "" {
		fmt.Printf("[spec] %s\n", s)
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

	d := download.New(cfg.ModelDir, cfg.CacheDir)
	if err := d.Run(repo, caps); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func cmdTune(args []string) {
	fs := flag.NewFlagSet("tune", flag.ExitOnError)
	port := fs.Int("port", 8081, "Server port")
	rounds := fs.Int("rounds", 3, "AI-tune rounds")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "Usage: llm-server tune <model.gguf>")
		os.Exit(2)
	}
	modelPath := fs.Arg(0)

	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}

	info, err := gguf.Parse(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing model: %v\n", err)
		os.Exit(1)
	}

	model := infoToProfile(info, modelPath)
	strategy, err := placement.Compute(caps, model, placement.Options{})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %v\n", err)
		os.Exit(1)
	}

	be := findBackend(caps)
	if be == nil {
		fmt.Fprintln(os.Stderr, "Error: no llama-server binary found")
		os.Exit(1)
	}
	strategy.BackendTag = be.Tag

	serverArgs := append([]string{be.Path}, strategy.Args(modelPath, *port)...)

	cfg := config.Defaults()
	if f, err := config.Load(); err == nil {
		cfg = f
	}

	cache := tune.NewCache(cfg.CacheDir)

	engine := &tune.Engine{
		BaseURL: fmt.Sprintf("http://localhost:%d", *port),
		Model:   filepath.Base(modelPath),
		Rounds:  *rounds,
		Cache:   cache,
		Caps:    caps,
		OnProgress: func(msg string) {
			fmt.Println("[tune]", msg)
		},
	}

	entry, err := engine.Run(modelPath, serverArgs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("[tune] Best config: %.1f tok/s\n", entry.Result.GenTPS)
}

func cmdBenchmark(args []string) {
	fs := flag.NewFlagSet("benchmark", flag.ExitOnError)
	port := fs.Int("port", 8081, "Server port")
	model := fs.String("model", "default", "Model name")
	fs.Parse(args)

	runner := &benchmark.Runner{
		BaseURL: fmt.Sprintf("http://localhost:%d", *port),
		Model:   *model,
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
	// Find the backend FIRST so its tag feeds placement — otherwise the
	// split-mode/flag selection can't tell ik_llama from mainline and emits
	// flags the backend rejects (e.g. `--split-mode row`, unsupported by ik).
	be := findBackend(caps)
	if be == nil {
		return nil, fmt.Errorf("no llama-server binary found")
	}
	cfg := config.Defaults()
	if c, err := config.Load(); err == nil {
		cfg = c
	}
	opts := placement.Options{
		KVPlacement: "auto",
		KVQuality:   "mid",
		CacheDir:    cfg.CacheDir,
		Host:        cfg.Host,
		BackendTag:  be.Tag,
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
	// Update backends
	if err := update.UpdateBackends(); err != nil {
		fmt.Fprintf(os.Stderr, "Backend update: %v\n", err)
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
	Path    string
	IsIK    bool
	SupportsReasoning bool
	Tag     string
}

// resolveCtxFlag converts --ctx flag to int: ""/"fit"=0, "max"=native, else number.
func resolveCtxFlag(s string, nativeCtx int) int {
	s = strings.TrimSpace(s)
	if s == "" || s == "fit" || s == "auto" { return 0 }
	if s == "max" || s == "native" {
		if nativeCtx > 0 { return nativeCtx }
		return 65536
	}
	if n, err := strconv.Atoi(s); err == nil && n > 0 { return n }
	return 0
}

func findBackend(caps *detect.Capabilities) *backendInfo {
	// Try detected backends first
	for _, b := range caps.Backends {
		if b.Name == "llama-server" || b.Name == "ik_llama" || b.Name == "ik_llama-server" {
			return detectBackend(b.Path)
		}
	}
	// Fallback: search common build paths
	home := os.Getenv("HOME")
	paths := []string{
		os.Getenv("LLAMA_SERVER"),
		filepath.Join(home, "ik_llama.cpp", "build", "bin", "llama-server"),
		filepath.Join(home, "llama.cpp", "build", "bin", "llama-server"),
		"/usr/local/bin/llama-server",
		"/usr/bin/llama-server",
	}
	for _, p := range paths {
		if p != "" {
			if _, err := os.Stat(p); err == nil {
				return detectBackend(p)
			}
		}
	}
	return nil
}

// detectBackend runs --help to determine if this is ik_llama.cpp fork.
// llama-server --help returns exit code 1, so we check the output regardless of error.
func detectBackend(path string) *backendInfo {
	info := &backendInfo{Path: path, Tag: "llama"}
	out, _ := exec.Command(path, "--help").Output()
	help := string(out)
	if strings.Contains(help, "ikawrakow") || strings.Contains(help, "split-mode-graph") {
		info.IsIK = true
		info.Tag = "ik_llama"
	}
	if strings.Contains(help, "--reasoning") {
		info.SupportsReasoning = true
	}
	return info
}
