package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"net/http"
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

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/benchmark"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/daemon"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/download"
	"github.com/raketenkater/ggrun/pkg/gguf"
	"github.com/raketenkater/ggrun/pkg/libhub"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/probe"
	"github.com/raketenkater/ggrun/pkg/recommend"
	"github.com/raketenkater/ggrun/pkg/recovery"
	"github.com/raketenkater/ggrun/pkg/server"
	"github.com/raketenkater/ggrun/pkg/tui"
	"github.com/raketenkater/ggrun/pkg/tune"
	"github.com/raketenkater/ggrun/pkg/update"
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
		fmt.Println("ggrun", version)
	case "detect":
		cmdDetect()
	case "launch":
		// `launch --dry-run` must never start a server. Without this reroute the
		// flag was silently swallowed by parseLaunchArgs and the "dry run" did
		// real launch attempts (and wrote OOM-replan placement caches).
		if hasArg(args[1:], "--dry-run") {
			cmdDryRun(args[1:])
		} else {
			cmdLaunch(args[1:])
		}
	case "benchmark":
		cmdBenchmark(args[1:])
	case "daemon":
		cmdDaemon(args[1:])
	case "dry-run":
		cmdDryRun(args[1:])
	case "probe":
		cmdProbe()
	case "kv-probe":
		cmdKVProbe(args[1:])
	case "download":
		cmdDownload(args[1:])
	case "tune":
		cmdTune(args[1:])
	case "recommend":
		cmdRecommend(args[1:])
	case "gui", "tui":
		update.PromptOnStartup()
		cmdGUI()
	case "config":
		cmdConfig(args[1:])
	case "backend", "backends":
		cmdBackend(args[1:])
	case "update", "--update":
		cmdUpdate()
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `Usage: ggrun [command] [args]

With no command, launches the interactive TUI (same as ggrun gui).

Commands:
  version              Show version
  detect               Detect hardware capabilities
  probe                Check free GPU/RAM memory
  kv-probe <model>     Measure real KV cache size (2 short launches) and cache it,
                       so context sizing is exact for compressed-attention models
  launch <model.gguf>  Launch model with auto-placement
  benchmark <model>    Benchmark a running server
  daemon               Start persistent daemon
  dry-run <model.gguf> Print computed flags without launching
  download <repo/name> Download from HuggingFace
  tune <model.gguf>    AI-tune model for best performance
  recommend [-n N]     Rank models that fit this machine (intelligence x speed)
  config [show|edit|path|reset]  Manage settings
  backend [list|add|register|remove]  Manage llama.cpp fork backends (add a fork,
                       auto-route a model arch to it, e.g. DeepSeek V4 → V4 fork)
  update, --update     Update ggrun and backends
  gui, tui             Interactive TUI (model picker, settings, launch)

Launch flags:
  -port int            Server port (default 8081)
  -ctx int             Context size (default auto)
  -kv string           KV placement: auto|gpu|cpu (default auto)
  -kv-quality string   KV quality: high|mid|low (default low)
  -cpu                 Force CPU-only mode
  -gpus string         Comma-separated GPU indices
  --vram-headroom str  Reserve VRAM the recommender/placement won't use, e.g. 2G
  --ram-headroom str   Reserve system RAM the recommender/placement won't use, e.g. 8G
  -vision              Enable vision (auto-detect mmproj)
  --spec string       Speculative decoding: off|auto|mtp|eagle3|draft|ngram|ngram-mod|ngram-k4v
`)
}

func knownCommand(cmd string) bool {
	switch cmd {
	case "version", "--version", "-v", "detect", "launch", "benchmark", "daemon", "dry-run", "probe", "kv-probe", "download", "tune", "recommend", "gui", "tui", "config", "backend", "backends", "update", "--update":
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
			fmt.Fprintln(os.Stderr, "Usage: ggrun <repo/name> --download")
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

func autoStartupTimeout(model *placement.ModelProfile) time.Duration {
	if model == nil {
		return 2 * time.Minute
	}
	totalSizeMB := float64(model.SizeBytes) / (1024 * 1024)
	timeoutSec := 240.0 + totalSizeMB/1700.0
	if timeoutSec < 60 {
		timeoutSec = 60
	}
	if model.IsMoE && totalSizeMB > 100*1024 {
		timeoutSec = 900
	}
	return time.Duration(timeoutSec*2) * time.Second
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
			case "--model", "-m", "--port", "-port", "--ctx", "-ctx", "--ctx-size", "-c", "--kv", "-kv", "--kv-placement", "--kv-quality", "--gpus", "--host", "--server-bin", "--mmproj", "--backend", "--tune-cache", "--rounds", "--ram-budget", "--vram-headroom", "--spec", "--parallel", "--lib-path", "--threads", "-t", "--batch-size", "-b", "--ubatch-size", "-ub":
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
	VRAMHeadroomMB    int
	RAMHeadroomMB     int
	NoMMap            bool
	Parallel          int
	ParallelSet       bool // --parallel given explicitly; claude-code mode must not override it
	Benchmark         bool
	ClaudeCode        bool
	ExtraArgs         []string
}

func parseLaunchArgs(args []string) (*launchRequest, error) {
	cfg := config.Defaults()
	if c, err := config.Load(); err == nil {
		cfg = c
	}
	backendExplicit := configuredBackendExplicit(cfg.Backend)
	req := &launchRequest{
		Port:            cfg.Port,
		CtxFlag:         cfg.CtxValue(),
		KVPlacement:     cfg.KVPlacement,
		KVQuality:       cfg.KVQuality,
		Host:            cfg.Host,
		VisionAuto:      cfg.Vision,
		ServerBin:       cfg.LlamaServer,
		Backend:         cfg.Backend,
		BackendExplicit: backendExplicit,
		SpecMode:        cfg.Spec,
		Parallel:        cfg.Parallel,
		VRAMHeadroomMB:  parseBudgetMB(cfg.VRAMHeadroom),
		RAMHeadroomMB:   parseBudgetMB(cfg.RAMHeadroom),
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
		req.Host = "127.0.0.1"
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
			case "--vram-headroom":
				req.VRAMHeadroomMB = parseBudgetMB(val)
				continue
			case "--ram-headroom":
				req.RAMHeadroomMB = parseBudgetMB(val)
				continue
			case "--no-mmap":
				req.NoMMap = val == "" || parseBoolFlag(val)
				continue
			case "--spec":
				req.SpecMode = val
				continue
			case "--parallel":
				req.Parallel, _ = strconv.Atoi(val)
				req.ParallelSet = true
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
		case "--claude-code":
			req.ClaudeCode = true
		case "--no-mmap":
			req.NoMMap = true
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
		case "--vram-headroom":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.VRAMHeadroomMB = parseBudgetMB(v)
		case "--ram-headroom":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.RAMHeadroomMB = parseBudgetMB(v)
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
			req.ParallelSet = true
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
	req.ExtraArgs = normalizePlacementAwareExtraArgs(req, req.ExtraArgs)
	return req, nil
}

func parseBoolFlag(raw string) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

func normalizePlacementAwareExtraArgs(req *launchRequest, args []string) []string {
	if req == nil || len(args) == 0 {
		return args
	}
	out := args[:0]
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--no-mmap" {
			req.NoMMap = true
			continue
		}
		if key, val, ok := strings.Cut(a, "="); ok && key == "--no-mmap" {
			req.NoMMap = val == "" || parseBoolFlag(val)
			continue
		}
		out = append(out, a)
	}
	return out
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

func parseBudgetMB(s string) int { return config.ParseBudgetMB(s) }

func configuredBackendExplicit(backend string) bool {
	backend = strings.TrimSpace(backend)
	return backend != "" && !strings.EqualFold(backend, "auto")
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
		// A registered fork backend selected by its manifest tag (--backend <tag>).
		if cb := backends.ByTag(want); cb != nil {
			if _, err := os.Stat(cb.Path); err == nil {
				return detectRegisteredBackend(cb)
			}
			fmt.Fprintf(os.Stderr, "Warning: registered backend %q binary not found: %s\n", cb.Tag, cb.Path)
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

// routeArchBackend redirects to a registered fork backend when the model's
// architecture is registered with a route-arch and the backend is still
// implicit/auto. A configured or CLI-selected backend must keep its actual
// backend instead of being hijacked by a fork route.
func routeArchBackend(be *backendInfo, model *placement.ModelProfile, req *launchRequest) *backendInfo {
	if req.BackendExplicit || model == nil {
		return be
	}
	if cb := backends.ForArch(model.ModelArch); cb != nil {
		fmt.Printf("[launch] %s runs on fork backend %q — routing to %s\n", model.ModelArch, cb.Tag, cb.Path)
		return detectRegisteredBackend(cb)
	}
	return be
}

func detectRegisteredBackend(cb *backends.Backend) *backendInfo {
	if cb == nil {
		return nil
	}
	info := detectBackend(cb.Path)
	if tag := strings.TrimSpace(cb.Tag); tag != "" {
		info.Tag = tag
	}
	return info
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
		ContextSize:    resolveCtxFlag(req.CtxFlag, model.CTXTrain),
		KVPlacement:    req.KVPlacement,
		KVQuality:      req.KVQuality,
		CPUMode:        req.CPUMode,
		RamBudgetMB:    req.RamBudgetMB,
		VRAMHeadroomMB: req.VRAMHeadroomMB,
		RAMHeadroomMB:  req.RAMHeadroomMB,
		NoMMap:         req.NoMMap,
		CacheDir:       cacheDir,
		Host:           req.Host,
		BackendTag:     be.Tag,
		VisionAuto:     req.VisionAuto,
		MMProjPath:     req.MMProjPath,
		SpecMode:       req.SpecMode,
		ForceSpecMoE:   req.ForceSpecMoE,
		BackendHelp:    be.Help,
		CacheFile:      req.TuneCache,
		Parallel:       req.Parallel,
		// Disable the model's thinking only when measuring (`--benchmark`); a
		// normal launch keeps reasoning on so tools like Claude Code can think.
		ReasoningOff: req.Benchmark,
	}
	if req.GPUsFlag != "" {
		for _, s := range strings.Split(req.GPUsFlag, ",") {
			idx, _ := strconv.Atoi(strings.TrimSpace(s))
			opts.GPUs = append(opts.GPUs, idx)
		}
	}
	opts.Parallel = claudeCodeParallel(opts.Parallel, req.ClaudeCode, req.ParallelSet)
	if req.ClaudeCode && !req.ParallelSet && opts.Parallel > 1 && deepseek4V4GraphQuirk(model, opts.BackendTag) {
		// Multi-slot (n_seq > 1) graph reservation is part of the same fork
		// abort; a single slot keeps the model's full context window anyway.
		fmt.Println("[claude-code] deepseek4 on the v4 fork aborts multi-slot graph reservation — using --parallel 1 (the single slot keeps the full context)")
		opts.Parallel = 1
	}
	return opts
}

// claudeCodeParallel floors --parallel at 4 in Claude Code mode. There are two
// opposing pressures and 4 is the balance point:
//   - Too few slots (1): the main turn, the command-safety classifier and
//     background/subagent calls thrash a single slot's KV cache → requests get
//     cancelled and every turn re-processes the whole prompt.
//   - Too many slots: --ctx-size is split evenly across slots, so each slot's
//     context shrinks (262144/8 = 32k). A real Claude Code conversation easily
//     exceeds that, and the backend then FAILS the request ("context shift is
//     disabled"). 4 slots keep ~65k per slot, which fits normal sessions.
//
// Wide fan-out is handled by a long API_TIMEOUT_MS (queued agents wait for one
// of the 4 slots and complete) rather than by more slots — a single GPU
// serializes the work either way. Total KV is unchanged (fixed ctx, just split).
// An explicitly passed --parallel always wins: big-MoE tuning (e.g. 2 slots so a
// background call can't evict the main conversation's expensive prompt cache)
// needs the user's value to survive claude-code mode.
func claudeCodeParallel(parallel int, claudeCode, explicit bool) int {
	if claudeCode && !explicit && parallel < 4 {
		return 4
	}
	return parallel
}

// claudeSlotTarget is the per-slot context Claude Code comfortably works in.
// claudeSlotMin is the floor below which a session can't even hold the system
// prompt (~15-20k tokens) and requests truncate or fail outright.
const (
	claudeSlotTarget = 65536
	claudeSlotMin    = 24576
)

// claudeCodeSlotAdjust caps the computed --parallel so each slot keeps a workable
// context window. claudeCodeParallel floors parallel at 4 BEFORE placement, which
// is right for large contexts (262144/4 = 65k slots) — but "fit" mode can pick a
// small total context (e.g. 32768 for a huge MoE on tight VRAM), and 32768/4 = 8k
// slots can't even hold Claude Code's system prompt. Fewer, bigger slots beat
// more, broken ones: concurrent requests then queue (API_TIMEOUT_MS covers the
// wait) and may re-process the prompt on interleave — slow, but functional.
// Runs after placement.Compute and before Strategy.Args, so the emitted
// --parallel and the derived CLAUDE_AUTOCOMPACT_PCT_OVERRIDE stay consistent.
func claudeCodeSlotAdjust(strategy *placement.Strategy, claudeCode, parallelExplicit bool) {
	if !claudeCode || strategy == nil || strategy.ContextSize <= 0 || strategy.Parallel <= 1 {
		return
	}
	// A user-chosen --parallel is a deliberate slot layout — keep it, warn below if tight.
	if !parallelExplicit {
		p := strategy.ContextSize / claudeSlotTarget
		if p < 1 {
			p = 1
		}
		if p < strategy.Parallel {
			fmt.Printf("[claude-code] context %d is too small for %d slots — lowering --parallel to %d (~%dk per slot)\n",
				strategy.ContextSize, strategy.Parallel, p, strategy.ContextSize/p/1000)
			strategy.Parallel = p
		}
	}
	if slot := strategy.ContextSize / strategy.Parallel; slot < claudeSlotMin {
		fmt.Printf("[claude-code] warning: only ~%dk context per slot — Claude Code needs ~24k+ just for its system prompt. Use a larger --ctx-size or a smaller model.\n", slot/1000)
	}
}

func buildLaunchServerArgs(req *launchRequest, cfg *config.Config, be *backendInfo, caps *detect.Capabilities, model *placement.ModelProfile, strategy *placement.Strategy) []string {
	serverArgs := append([]string{be.Path}, strategy.Args(req.ModelPath, req.Port)...)
	serverArgs = append(serverArgs, req.ExtraArgs...)
	serverArgs = applyTuneCache(req, serverArgs, cfg.CacheDir, be.Tag, strategy.MMProjPath != "", caps)
	serverArgs = claudeCodeAliasArgs(serverArgs, req.ClaudeCode)
	serverArgs = claudeCodeSamplingArgs(serverArgs, req.ClaudeCode, model)
	return serverArgs
}

func startLaunchProcess(req *launchRequest, cfg *config.Config, serverArgs []string, timeout time.Duration) (*server.Process, error) {
	if req.ClaudeCode {
		// In Claude Code mode ggrun hands the terminal to the `claude` client, so
		// the backend's ongoing per-request logs must go to a file instead of
		// bleeding into Claude Code's UI.
		logDir := cfg.LogDir
		if logDir == "" {
			logDir = os.TempDir()
		}
		logPath := filepath.Join(logDir, fmt.Sprintf("ggrun-claude-server-%d.log", req.Port))
		if lf, ferr := os.Create(logPath); ferr == nil {
			fmt.Printf("[claude-code] backend logs -> %s\n", logPath)
			return server.StartWithTimeoutTo(serverArgs, req.Port, timeout, lf, lf)
		}
	}
	return server.StartWithTimeout(serverArgs, req.Port, timeout)
}

func recordMeasuredLaunchProbes(cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, serverLog string, baselineVRAMByGPU map[int]int) {
	baselineLen := len(baselineVRAMByGPU)
	fmt.Fprintf(os.Stderr, "[launch] recordMeasuredLaunchProbes: log=%d bytes, baseline=%d gpus, cfg=%v model=%v strategy=%v be=%v\n",
		len(serverLog), baselineLen, cfg != nil, model != nil, strategy != nil, be != nil)
	if cfg == nil || model == nil || strategy == nil || be == nil || serverLog == "" {
		return
	}
	var gpus []detect.GPU
	if caps != nil {
		gpus = caps.GPUs
	}
	if model.IsMoE && len(gpus) > 0 {
		placement.RunPostLaunchProbe(cfg.CacheDir, gpus, serverLog)
		if len(baselineVRAMByGPU) > 0 {
			fmt.Fprintf(os.Stderr, "[launch] VRAM baseline: CUDA0=%d CUDA1=%d CUDA2=%d (log buf %d bytes)\n",
				baselineVRAMByGPU[0], baselineVRAMByGPU[1], baselineVRAMByGPU[2], len(serverLog))
		}
		ok := placement.RunPostLaunchModelProbeVRAMDelta(cfg.CacheDir, model, strategy, be.Tag, gpus, baselineVRAMByGPU)
		if !ok {
			fmt.Fprintf(os.Stderr, "[launch] VRAM probe: no cache written (needs baseline & post-load VRAM > baseline)\n")
		}
	}
	placement.RunPostLaunchModelProbe(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, be.Tag, gpus, serverLog)
	placement.RunPostLaunchKVProbe(cfg.CacheDir, model, strategy.ContextSize, strategy.KVType, serverLog)
}

func maybePromoteMeasuredPlacement(req *launchRequest, cfg *config.Config, be *backendInfo, caps *detect.Capabilities, model *placement.ModelProfile, current *placement.Strategy, currentArgs []string) (*placement.Strategy, []string, bool) {
	if req == nil || cfg == nil || be == nil || caps == nil || model == nil || current == nil || !model.IsMoE || len(caps.GPUs) == 0 {
		return nil, nil, false
	}
	// A measured KV probe may have been written after the first load. Force the
	// recompute to reload it instead of reusing the pre-launch model struct state.
	model.MeasuredKVBytesPerTok = nil
	next, err := claudeCodeComputeStrategy(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir), req.ClaudeCode, req.CtxFlag == "max")
	if err != nil {
		fmt.Fprintf(os.Stderr, "[launch] calibration: measured placement recompute failed: %v\n", err)
		return nil, nil, false
	}
	claudeCodeSlotAdjust(next, req.ClaudeCode, req.ParallelSet)
	if !shouldPromoteMoEPlacement(current, next) {
		return nil, nil, false
	}
	nextArgs := buildLaunchServerArgs(req, cfg, be, caps, model, next)
	if formatCommand(nextArgs) == formatCommand(currentArgs) {
		return nil, nil, false
	}
	return next, nextArgs, true
}

func startLaunchWithCUDAOOMRecovery(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, serverArgs []string, timeout time.Duration) (*server.Process, *placement.Strategy, []string, error) {
	const maxRetries = 2
	retries := 0
	preflightReplans := 0
	oomPenalty := map[int]int{}
	for {
		// Ask the backend's no-alloc accounting whether this placement can even
		// load (~1s) before committing to a real load (15+ min for a big MoE).
		// A measured deficit re-plans exactly like a startup CUDA OOM would —
		// without paying for the load to learn it. Re-planned args loop back
		// here, so every retry is re-gated too.
		if preflightReplans < 3 && strategy != nil {
			if dev, deficit, bad := preflightPlacement(be, &configForPreflight{CacheDir: cfg.CacheDir}, caps, model, strategy, serverArgs); bad {
				preflightReplans++
				oomPenalty[dev] += deficit
				if s, rerr := placement.ReplanAfterOOM(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir), oomPenalty); rerr == nil && s != nil && s.OTString != "" {
					fmt.Fprintf(os.Stderr, "[launch] preflight re-plan (n-cpu-moe=%d)\n", s.NCPUMoE)
					serverArgs = patchPlacementArgs(serverArgs, s)
					strategy = s
					continue
				}
				// No re-plan available (dense model, or packer can't shift
				// further): fall through to the real launch — the preflight
				// gate must never block a launch outright, and startup OOM
				// recovery below still applies.
			}
		}
		p, err := startLaunchProcess(req, cfg, serverArgs, timeout)
		if err == nil {
			// Persist the placement that actually loaded and passed the health
			// check — and only that. Overwrite unconditionally: after an OOM
			// re-plan the file on disk still holds the plan that just failed.
			if strategy != nil && strategy.Type == placement.MoEOffload && strategy.PlacementCachePath != "" {
				_ = placement.SavePlacementCache(strategy.PlacementCachePath, placement.StrategyToCacheEntry(strategy))
			}
			return p, strategy, serverArgs, nil
		}

		logData := ""
		if p != nil && p.LogBuf != nil {
			logData = p.LogBuf.String()
			recordMeasuredLaunchProbes(cfg, model, strategy, be, caps, logData, nil)
		}
		// Diagnose before checking the retry budget: a clean, parseable OOM on
		// the very last allowed attempt still deserves its real cause recorded
		// and reported, instead of surfacing only the process's raw exit error
		// (e.g. a bare "signal: segmentation fault" with no VRAM context).
		device, allocMB, isComputeBuffer, ok := startupLogCUDAOOMDetailed(logData)
		if ok && strategy != nil && caps != nil && be != nil {
			_ = placement.RecordRuntimeGraphGrowthFromOOM(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, be.Tag, caps.GPUs, device, allocMB)
		}
		if retries >= maxRetries {
			if ok {
				return p, strategy, serverArgs, fmt.Errorf("CUDA OOM on device %d allocating %d MiB (retry budget exhausted after %d attempts): %w", device, allocMB, retries, err)
			}
			return p, strategy, serverArgs, err
		}
		if !ok {
			return p, strategy, serverArgs, err
		}

		// Re-plan with the failed card penalized by its overshoot: the real packer
		// refits it with partial gate+up chunks and reclaims stranded VRAM on the
		// other cards via the sub-pin squeeze (experts move off system RAM),
		// instead of a blind whole-layer drop that over-corrects and erases the
		// squeeze. Falls back to the whole-layer derate if a re-plan can't fit.
		// Do NOT persist the re-planned/derated placement here: it has never
		// loaded. Caches written mid-retry poisoned later launches with plans
		// that were themselves OOM guesses (e.g. "all experts on one GPU").
		// The success branch above persists whatever finally worked.
		oomPenalty[device] += oomOvershoot(caps, device, allocMB)
		if s, rerr := placement.ReplanAfterOOM(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir), oomPenalty); rerr == nil && s != nil && s.OTString != "" {
			fmt.Fprintf(os.Stderr, "[launch] CUDA OOM on device %d (%d MiB, over ~%d MiB); re-planned (n-cpu-moe=%d) and retrying\n", device, allocMB, oomPenalty[device], s.NCPUMoE)
			serverArgs = patchPlacementArgs(serverArgs, s)
			strategy = s
		} else {
			nextArgs, entry, derated := placement.DerateCUDAOOMArgs(serverArgs, model, caps, device, allocMB, isComputeBuffer)
			if !derated {
				return p, strategy, serverArgs, err
			}
			fmt.Fprintf(os.Stderr, "[launch] CUDA OOM on device %d allocating %d MiB; moving expert layer(s) to CPU and retrying\n", device, allocMB)
			applyDeratedPlacementEntry(strategy, entry)
			serverArgs = nextArgs
		}
		retries++
		fmt.Printf("[launch] %s\n", formatCommand(serverArgs))
	}
}

// oomOvershoot is how much a failed cudaMalloc exceeded the device's free VRAM
// (min 512 MiB), used to penalize that card on a corrective re-plan.
func oomOvershoot(caps *detect.Capabilities, device, allocMB int) int {
	over := allocMB
	if caps != nil {
		for _, g := range caps.GPUs {
			if g.Index == device {
				if free := g.VRAMFreeMB(); allocMB > free {
					over = allocMB - free
				}
				break
			}
		}
	}
	if over <= 0 {
		over = 512
	}
	return over
}

func startupLogCUDAOOM(logData string) (device int, allocMB int, ok bool) {
	device, allocMB, _, ok = startupLogCUDAOOMDetailed(logData)
	return device, allocMB, ok
}

// startupLogCUDAOOMDetailed additionally reports whether the failed
// allocation was the compute graph (gallocr/graph_reserve — scales with
// ubatch) rather than a model-weight tensor (scales with which expert layers
// are GPU-resident). The two need different derate levers: shrinking ubatch
// fixes the former, moving expert layers to CPU fixes the latter.
func startupLogCUDAOOMDetailed(logData string) (device int, allocMB int, isComputeBuffer bool, ok bool) {
	lines := strings.Split(logData, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if device, allocMB, ok := recovery.ParseCUDAOOM(lines[i]); ok {
			isComputeBuffer := false
			for j := i + 1; j < len(lines) && j <= i+3; j++ {
				if strings.Contains(lines[j], "gallocr") || strings.Contains(lines[j], "graph_reserve") {
					isComputeBuffer = true
					break
				}
			}
			return device, allocMB, isComputeBuffer, true
		}
	}
	return 0, 0, false, false
}

func applyDeratedPlacementEntry(strategy *placement.Strategy, entry *placement.CacheEntry) {
	if strategy == nil || entry == nil {
		return
	}
	// Keep OTString in sync: the success-path cache save serializes the
	// strategy, and a stale -ot with a derated split is a poisoned cache.
	if entry.OTString != "" {
		strategy.OTString = entry.OTString
	}
	if entry.NCPUMoE > 0 {
		strategy.NCPUMoE = entry.NCPUMoE
	}
	if len(entry.TensorSplit) > 0 {
		strategy.TensorSplit = append([]float64(nil), entry.TensorSplit...)
	}
	if entry.SplitMode != "" {
		strategy.SplitMode = entry.SplitMode
	}
	if entry.BatchSize > 0 {
		strategy.BatchSize = entry.BatchSize
	}
	if entry.UBatchSize > 0 {
		strategy.UBatchSize = entry.UBatchSize
	}
	if entry.Parallel > 0 {
		strategy.Parallel = entry.Parallel
	}
	strategy.MMap = entry.MMap
}

func shouldPromoteMoEPlacement(current, next *placement.Strategy) bool {
	if current == nil || next == nil || current.Type != placement.MoEOffload || next.Type != placement.MoEOffload {
		return false
	}
	if current.NCPUMoE > 0 && next.NCPUMoE < current.NCPUMoE {
		return true
	}
	// VERIFICATION: measured cold-launch calibration can improve stable-max fill
	// by adding gate/up subpins while the CPU MoE layer count stays unchanged.
	// Promote that too; otherwise the automatic second pass misses the squeeze.
	return next.NCPUMoE == current.NCPUMoE && next.OTString != "" && next.OTString != current.OTString
}

// resolveLaunchBackend selects the backend, applies fork-arch routing (e.g.
// deepseek4 -> the v4 fork), and preflights the arch. This is the exact step
// that must be identical across every launch path (CLI, TUI, dry-run) — the TUI
// once skipped the fork routing and couldn't load V4 at all. Returns nil if no
// backend binary is available.
func resolveLaunchBackend(req *launchRequest, model *placement.ModelProfile, caps *detect.Capabilities) *backendInfo {
	be := selectBackend(caps, req)
	if be == nil {
		return nil
	}
	be = routeArchBackend(be, model, req)
	preflightBackendArch(model, be, caps)
	return be
}

func cmdLaunch(args []string) {
	req, err := parseLaunchArgs(args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(2)
	}
	if req.ModelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: ggrun launch <model.gguf>")
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

	be := resolveLaunchBackend(req, model, caps)
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

	strategy, err := claudeCodeComputeStrategy(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir), req.ClaudeCode, req.CtxFlag == "max")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %v\n", err)
		os.Exit(1)
	}
	claudeCodeSlotAdjust(strategy, req.ClaudeCode, req.ParallelSet)

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

	serverArgs := buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
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

	timeout := autoStartupTimeout(model)

	// Capture VRAM baseline before the server starts so the post-launch probe
	// can measure the real compute-buffer allocation.
	baselineVRAM := map[int]int{}
	if model.IsMoE && len(caps.GPUs) > 0 {
		for _, g := range caps.GPUs {
			baselineVRAM[g.Index] = placement.QueryVRAMUsed(g.Index)
		}
	}

	p, strategy, serverArgs, err := startLaunchWithCUDAOOMRecovery(req, cfg, model, strategy, be, caps, serverArgs, timeout)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error starting server: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("[launch] Server running on port %d (PID %d)\n", req.Port, p.Cmd.Process.Pid)
	if p.LogBuf != nil {
		recordMeasuredLaunchProbes(cfg, model, strategy, be, caps, p.LogBuf.String(), baselineVRAM)
		if nextStrategy, nextArgs, ok := maybePromoteMeasuredPlacement(req, cfg, be, caps, model, strategy, serverArgs); ok {
			fmt.Printf("[launch] calibration: measured placement fits more GPU experts (%d CPU MoE -> %d); restarting once\n", strategy.NCPUMoE, nextStrategy.NCPUMoE)
			_ = p.Stop()
			fmt.Printf("[launch] %s\n", formatCommand(nextArgs))
			p, nextStrategy, nextArgs, err = startLaunchWithCUDAOOMRecovery(req, cfg, model, nextStrategy, be, caps, nextArgs, timeout)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error starting promoted server: %v\n", err)
				os.Exit(1)
			}
			strategy = nextStrategy
			serverArgs = nextArgs
			fmt.Printf("[launch] Server running on port %d (PID %d)\n", req.Port, p.Cmd.Process.Pid)
			if p.LogBuf != nil {
				go recordMeasuredLaunchProbes(cfg, model, strategy, be, caps, p.LogBuf.String(), baselineVRAM)
			}
		}
	}
	if req.ClaudeCode {
		// Smooth path: one command brings up the model AND drops the user into
		// Claude Code wired to it. When claude exits, stop the server too.
		//
		// Run a health monitor alongside Claude so a mid-session backend crash
		// is recorded immediately — otherwise Claude Code times out silently
		// and the OOM data is lost until the user notices (audit cross-check #4).
		healthCtx, healthCancel := context.WithCancel(context.Background())
		defer healthCancel()
		go func() {
			ticker := time.NewTicker(30 * time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-healthCtx.Done():
					return
				case <-ticker.C:
				}
				if !isServerRunning(req.Host, req.Port) {
					fmt.Fprintf(os.Stderr, "[launch] backend died mid-session — recording OOM for next launch\n")
					healthCancel()
					return
				}
			}
		}()

		if code := runClaudeCodeClient(req.Host, req.Port, serverArgs, nil); code >= 0 {
			healthCancel()
			// The terminal was handed to `claude`, so a mid-session backend
			// crash isn't something this process can retry live — but it can
			// still be recorded before Stop(), so the NEXT `--claude-code`
			// launch of this exact model/context reserves the measured
			// deficit instead of repeating the same crash blind.
			if !p.IsRunning() && p.LogBuf != nil {
				if device, allocMB, ok := startupLogCUDAOOM(p.LogBuf.String()); ok {
					_ = placement.RecordRuntimeGraphGrowthFromOOM(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, be.Tag, caps.GPUs, device, allocMB)
					fmt.Fprintf(os.Stderr, "[launch] backend crashed during this session (CUDA OOM on device %d, %d MiB) — recorded, next launch of this model/context will reserve for it.\n", device, allocMB)
				}
			}
			if err := p.Stop(); err != nil {
				fmt.Fprintf(os.Stderr, "[launch] stop after claude: %v\n", err)
			}
			os.Exit(code)
		}
		// `claude` isn't installed — fall back to the copy-paste recipe.
		printClaudeCodeRecipe(req.Host, req.Port, serverArgs)
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

	// The loop below owns the entire remaining lifecycle: it blocks until
	// either the user asks to stop, or the backend dies on its own. A crash
	// AFTER health check (not covered by startLaunchWithCUDAOOMRecovery,
	// which only wraps startup) used to leave this process silently blocked
	// forever on <-sigCh with no idea its child had already exited — "Press
	// Ctrl+C to stop" claiming to serve a model that was actually gone.
	// Reproduced 2026-07-08/09: DeepSeek-V4 crashed with a real CUDA OOM
	// during a long request (see maximizeMoEGPUFitByUBatch's runtime-growth
	// comment in placement.go) well after loading clean and passing health.
	const maxRuntimeOOMRetries = 2
	runtimeOOMRetries := 0
	for {
		crashed := waitForShutdownOrCrash(p, sigCh)
		if !crashed {
			fmt.Fprintln(os.Stderr, "\n[launch] Shutting down...")
			break
		}

		logData := ""
		if p.LogBuf != nil {
			logData = p.LogBuf.String()
		}
		device, allocMB, ok := startupLogCUDAOOM(logData)
		if !ok || runtimeOOMRetries >= maxRuntimeOOMRetries {
			if ok {
				fmt.Fprintf(os.Stderr, "[launch] server crashed (CUDA OOM on device %d, %d MiB) after %d recovery attempt(s) — giving up. Try again; the deficit already recorded should reduce it next time.\n", device, allocMB, runtimeOOMRetries)
			} else {
				fmt.Fprintln(os.Stderr, "[launch] server exited unexpectedly (not a recognized CUDA OOM) — see the log for details.")
			}
			os.Exit(1)
		}

		runtimeOOMRetries++
		_ = placement.RecordRuntimeGraphGrowthFromOOM(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, be.Tag, caps.GPUs, device, allocMB)
		fmt.Fprintf(os.Stderr, "[launch] server crashed after health check: CUDA OOM on device %d needing %d MiB more mid-request — recorded, re-planning and relaunching (attempt %d/%d)...\n",
			device, allocMB, runtimeOOMRetries, maxRuntimeOOMRetries)

		replanOpts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
		// Without this, Compute() prefers the .place cache written when the
		// prior instance loaded cleanly and passed health — which is exactly
		// the placement that just OOM'd mid-request. Skipping it forces a
		// fresh derivation that actually consults the growth deficit just
		// recorded above via RecordRuntimeGraphGrowthFromOOM.
		replanOpts.SkipPlacementCache = true
		nextStrategy, err := claudeCodeComputeStrategy(caps, model, replanOpts, req.ClaudeCode, req.CtxFlag == "max")
		if err != nil {
			fmt.Fprintf(os.Stderr, "[launch] re-plan after runtime OOM failed: %v\n", err)
			os.Exit(1)
		}
		claudeCodeSlotAdjust(nextStrategy, req.ClaudeCode, req.ParallelSet)
		nextArgs := buildLaunchServerArgs(req, cfg, be, caps, model, nextStrategy)
		fmt.Printf("[launch] %s\n", formatCommand(nextArgs))
		newP, newStrategy, newArgs, err := startLaunchWithCUDAOOMRecovery(req, cfg, model, nextStrategy, be, caps, nextArgs, timeout)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[launch] relaunch after runtime OOM failed: %v\n", err)
			os.Exit(1)
		}
		p, strategy, serverArgs = newP, newStrategy, newArgs
		fmt.Printf("[launch] Server running on port %d (PID %d)\n", req.Port, p.Cmd.Process.Pid)
		fmt.Println("[launch] Press Ctrl+C to stop")
	}

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

// waitForShutdownOrCrash blocks until either a shutdown signal arrives
// (returns false) or the backend process exits on its own (returns true).
// Polls IsRunning rather than needing a dedicated exit channel from the
// server package, since a crash mid-request is not otherwise observable from
// here — cmd.Wait() already returned inside server.Process's own goroutine.
func waitForShutdownOrCrash(p *server.Process, sigCh <-chan os.Signal) bool {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-sigCh:
			return false
		case <-ticker.C:
			if !p.IsRunning() {
				return true
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
	serverArgs = tune.ApplyOverrides(serverArgs, summary.Flags, tune.QualityProtectedFlags())
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

// cmdKVProbe measures a model's real KV cache size by launching it twice at
// different contexts and attributing the VRAM difference to KV (see
// placement.ProbeKVViaVRAMDelta). It caches the result so later launches size the
// context from measured truth instead of the per-arch formula — the reliable path
// for compressed-attention models (DeepSeek V4, MiniMax-M3) and for backend builds
// that don't log their KV size.
func cmdKVProbe(args []string) {
	req, err := parseLaunchArgs(args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(2)
	}
	if req.ModelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: ggrun kv-probe <model.gguf>")
		os.Exit(2)
	}
	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}
	if len(caps.GPUs) == 0 {
		fmt.Fprintln(os.Stderr, "kv-probe needs at least one GPU (it measures KV via VRAM delta)")
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
	be := selectBackend(caps, req)
	binPath := "llama-server"
	if be != nil {
		binPath = be.Path
	} else {
		be = &backendInfo{Path: binPath, Tag: "llama"}
	}
	strategy, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %v\n", err)
		os.Exit(1)
	}
	serverArgs := append([]string{be.Path}, strategy.Args(req.ModelPath, req.Port)...)
	serverArgs = append(serverArgs, req.ExtraArgs...)

	fmt.Printf("[kv-probe] Measuring KV for %s at cache-type %s — two short launches; a big model takes a few minutes each.\n", model.Basename, strategy.KVType)
	if placement.ProbeKVViaVRAMDelta(be.Path, serverArgs[1:], caps.GPUs, cfg.CacheDir, model, strategy.KVType) {
		fmt.Println("[kv-probe] Done. Future launches size context from the measured KV (frees VRAM the formula over-reserved).")
	} else {
		fmt.Fprintln(os.Stderr, "[kv-probe] Could not measure (a load didn't finish, or the VRAM delta was unusable). Launches keep using the formula.")
		os.Exit(1)
	}
}

func tuiLaunchArgs(req *tui.LaunchRequest, cfg *config.Config) []string {
	if req == nil {
		return nil
	}
	_ = cfg
	return req.LaunchArgs()
}

func cmdGUI() {
	go recommend.MaybeRefresh() // refresh catalog in the background; TUI uses cache-or-embedded
	req, err := tui.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if req == nil {
		return
	}

	cfg := config.Defaults()
	if c, err := config.Load(); err == nil {
		cfg = c
	}

	if req.DownloadRepo != "" {
		caps, err := detect.Detect()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
			os.Exit(1)
		}
		d := download.New(cfg.ModelDir, cfg.CacheDir, cfg.AppHome)
		if err := d.RunQuant(req.DownloadRepo, req.DownloadQuant, caps); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	cmdLaunch(tuiLaunchArgs(req, cfg))
}

func cmdDryRun(args []string) {
	req, err := parseLaunchArgs(args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(2)
	}
	if req.ModelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: ggrun dry-run <model.gguf>")
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
	be = routeArchBackend(be, model, req)
	backendTag := "llama"
	binPath := "llama-server"
	if be != nil {
		binPath = be.Path
		backendTag = be.Tag
	} else {
		be = &backendInfo{Path: binPath, Tag: backendTag}
	}

	strategy, err := claudeCodeComputeStrategy(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir), req.ClaudeCode, req.CtxFlag == "max")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %v\n", err)
		os.Exit(1)
	}
	claudeCodeSlotAdjust(strategy, req.ClaudeCode, req.ParallelSet)

	serverArgs := append([]string{binPath}, strategy.Args(req.ModelPath, req.Port)...)
	serverArgs = append(serverArgs, req.ExtraArgs...)
	serverArgs = applyTuneCache(req, serverArgs, cfg.CacheDir, be.Tag, strategy.MMProjPath != "", caps)
	serverArgs = claudeCodeAliasArgs(serverArgs, req.ClaudeCode)
	serverArgs = claudeCodeSamplingArgs(serverArgs, req.ClaudeCode, model)
	if envPrefix := applyGPUVisibility(req, be.Tag); envPrefix != "" {
		fmt.Print(envPrefix + " ")
	}
	fmt.Println(formatCommand(serverArgs))
	if s := placement.DraftSummary(strategy.Draft); s != "" {
		fmt.Printf("[spec] %s\n", s)
	}
	if req.ClaudeCode {
		printClaudeCodeRecipe(req.Host, req.Port, serverArgs)
	}
}

// printClaudeCodeRecipe prints the exact env to point Claude Code at this
// locally-served model. ggrun serves llama.cpp's native Anthropic /v1/messages
// endpoint with --jinja already on, so no proxy is needed.
func printClaudeCodeRecipe(host string, port int, serverArgs []string) {
	clientHost := host
	if clientHost == "" || clientHost == "0.0.0.0" || clientHost == "::" {
		clientHost = "127.0.0.1"
	}
	pct := claudeCodeAutocompactPct(serverArgs)
	slot := ""
	if ctx := argIntValue(serverArgs, "--ctx-size", "-c", "--ctx"); ctx > 0 {
		par := argIntValue(serverArgs, "--parallel", "-np")
		if par < 1 {
			par = 1
		}
		slot = fmt.Sprintf(" (~%dk per slot at --parallel %d)", ctx/par/1000, par)
	}
	// Every tier maps to local so background work and the command-safety
	// classifier hit the local server too, not api.anthropic.com.
	fmt.Println()
	fmt.Println("[claude-code] In another terminal:")
	// Match claudeCodeEnv: drop any real key so the dummy token + local base URL win,
	// otherwise Claude Code prefers the real key and routes to api.anthropic.com.
	fmt.Println("  unset ANTHROPIC_API_KEY")
	fmt.Printf("  export ANTHROPIC_BASE_URL=http://%s:%d ANTHROPIC_AUTH_TOKEN=ggrun\n", clientHost, port)
	fmt.Println("  export ANTHROPIC_MODEL=local ANTHROPIC_SMALL_FAST_MODEL=local")
	fmt.Println("  export ANTHROPIC_DEFAULT_HAIKU_MODEL=local ANTHROPIC_DEFAULT_SONNET_MODEL=local ANTHROPIC_DEFAULT_OPUS_MODEL=local")
	fmt.Println("  export API_TIMEOUT_MS=1800000   # let queued fan-out/subagent requests finish, not cancel")
	fmt.Println("  export API_FORCE_IDLE_TIMEOUT=0  # local models can spend >5 min prompt-processing before streaming")
	fmt.Printf("  export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=%d  # compact early to fit the real slot%s\n", pct, slot)
	if _, err := exec.LookPath("uvx"); err == nil {
		fmt.Println(`  claude --disallowedTools WebSearch --mcp-config '{"mcpServers":{"ddg-search":{"command":"uvx","args":["duckduckgo-mcp-server"]}}}'`)
	} else {
		fmt.Println("  claude --disallowedTools WebSearch   # add a search MCP (e.g. uvx duckduckgo-mcp-server) for web research")
	}
}

// claudeCodeEnv returns the child environment that points Claude Code at the
// locally-served model. Every model tier maps to "local" so background work and
// the command-safety classifier hit the local server too; ANTHROPIC_API_KEY is
// dropped so the dummy auth token + base URL take effect.
func claudeCodeEnv(host string, port int, serverArgs []string) []string {
	clientHost := host
	if clientHost == "" || clientHost == "0.0.0.0" || clientHost == "::" {
		clientHost = "127.0.0.1"
	}
	var env []string
	for _, kv := range os.Environ() {
		if strings.HasPrefix(kv, "ANTHROPIC_API_KEY=") {
			continue
		}
		env = append(env, kv)
	}
	return append(env,
		fmt.Sprintf("ANTHROPIC_BASE_URL=http://%s:%d", clientHost, port),
		"ANTHROPIC_AUTH_TOKEN=ggrun",
		"ANTHROPIC_MODEL=local",
		"ANTHROPIC_SMALL_FAST_MODEL=local",
		"ANTHROPIC_DEFAULT_HAIKU_MODEL=local",
		"ANTHROPIC_DEFAULT_SONNET_MODEL=local",
		"ANTHROPIC_DEFAULT_OPUS_MODEL=local",
		// A wide fan-out (subagents / ultracode) queues behind the GPU; without a
		// long timeout Claude Code cancels the queued requests. 30 min lets them
		// wait for a slot and complete instead. User-set value wins.
		"API_TIMEOUT_MS="+envOr("API_TIMEOUT_MS", "1800000"),
		// The request timeout above is not enough for local giant models: Claude Code
		// also has a stream-idle watchdog, and llama.cpp may spend >5 minutes in
		// prompt processing before emitting the first byte. Disable that watchdog for
		// local backend launches unless the user explicitly overrides it.
		"API_FORCE_IDLE_TIMEOUT="+envOr("API_FORCE_IDLE_TIMEOUT", "0"),
		// Behind a custom base URL Claude Code assumes a 200k window and won't
		// auto-compact until ~92% of it (~184k tokens) — but each slot only has ctx/parallel,
		// so the conversation overflows the slot and the backend fails the request
		// ("context shift is disabled"). Compact early instead, at a percentage
		// derived from the real slot size so it adapts to --parallel automatically.
		// A user-set value still wins.
		"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE="+envOr("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", strconv.Itoa(claudeCodeAutocompactPct(serverArgs))),
	)
}

// envOr returns the current environment value for key, or def if unset/empty.
func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// argIntValue returns the integer value following the LAST parseable occurrence of
// any of names in args (e.g. argIntValue(args, "--ctx-size", "-c")). Last-wins mirrors
// llama.cpp/ik_llama, which honor the final value when a flag is repeated — so when a
// user appends their own --ctx-size/--parallel after ggrun's computed ones (serverArgs
// has strategy flags first, then req.ExtraArgs), this reads the value the backend
// actually uses. Returns -1 if no matching flag has a parseable value.
func argIntValue(args []string, names ...string) int {
	result := -1
	for i := 0; i < len(args)-1; i++ {
		for _, name := range names {
			if args[i] == name {
				if n, err := strconv.Atoi(args[i+1]); err == nil {
					result = n
				}
			}
		}
	}
	return result
}

// claudeCodeAutocompactPct derives the CLAUDE_AUTOCOMPACT_PCT_OVERRIDE value from
// the backend's real per-slot context budget. llama.cpp/ik_llama split --ctx-size
// across --parallel sequence slots, so each request's usable window is
// ctx-size/parallel — far smaller than the 200k Claude Code assumes behind a custom
// base URL. Without an override Claude Code won't auto-compact until ~92% of that
// imagined 200k, long after the real slot has overflowed (and with --no-context-shift
// the backend then hard-fails the request). We compute the percentage of the assumed
// 200k window that lands at 75% of the real slot, leaving a quarter of the slot as
// headroom for the in-flight reply, tool results, and jinja/template overhead the
// proxied token count can't see. The same env var is inherited by subagents and
// workflow agents, so their conversations get the same slot-safe trigger.
//
// Examples (ctx 262144): --parallel 4 → 65536/slot → 24; --parallel 8 → 32768 → 12.
const claudeAssumedWindow = 200000

func claudeCodeAutocompactPct(serverArgs []string) int {
	ctx := argIntValue(serverArgs, "--ctx-size", "-c", "--ctx")
	if ctx <= 0 {
		return 25 // unknown ctx: keep the historical default
	}
	parallel := argIntValue(serverArgs, "--parallel", "-np")
	if parallel < 1 {
		parallel = 1
	}
	slot := ctx / parallel
	// Upstream pads per-sequence context to 256-token alignment; align
	// down so auto-compact triggers before the actual slot overflows
	// (Codex audit #7).
	slot = slot & ^255
	if slot < 2048 {
		slot = 2048
	}
	safe := int(float64(slot) * 0.75)
	pct := safe * 100 / claudeAssumedWindow
	// Floor so compaction still leaves working room; cap under Claude Code's own
	// ~92% native trigger so the override never relaxes the default.
	if pct < 5 {
		pct = 5
	}
	if pct > 90 {
		pct = 90
	}
	return pct
}

// claudeCodeSearchMCPArgs returns --mcp-config args that wire a no-key DuckDuckGo
// search MCP into Claude Code, replacing the Anthropic-only WebSearch tool that
// can't run against a local endpoint. Returns nil if the user already passed their
// own --mcp-config or no MCP runner (uvx) is installed. The exposed tool surfaces to
// agents and workflows as mcp__ddg-search__search.
func claudeCodeSearchMCPArgs(extraArgs []string) []string {
	if hasArg(extraArgs, "--mcp-config") {
		return nil
	}
	// The canonical duckduckgo-mcp-server is a Python package; uvx runs it with no
	// install step and no API key. Only wire it up when uvx is actually present.
	if _, err := exec.LookPath("uvx"); err != nil {
		return nil
	}
	cfg := `{"mcpServers":{"ddg-search":{"command":"uvx","args":["duckduckgo-mcp-server"]}}}`
	return []string{"--mcp-config", cfg}
}

// claudeCodeCtxLadder returns descending context-size targets for context-first
// placement: the target, then halved until 32768 (Claude Code's practical floor).
func claudeCodeCtxLadder(target int) []int {
	if target <= 0 {
		return nil
	}
	if target < 32768 {
		return []int{target}
	}
	var out []int
	for c := target; c >= 32768; c /= 2 {
		out = append(out, c)
	}
	return out
}

// claudeCodeComputeStrategy implements context-first placement for Claude Code
// mode. Default "fit" placement optimizes model layout and settles for a flat
// 32768 context (defaultContextSize) — but for Claude Code, context is the hard
// failure dimension (slots too small → truncated system prompt, compaction
// thrash), while a few expert layers moving to CPU is only a soft slowdown. So
// here the priority inverts: reserve KV on GPU for the largest workable context
// and let the model layout adjust around it. The ladder starts at the model's
// trained context and halves only when placement cannot fit with KV on GPU; dense
// models must additionally keep their weights fully offloaded — spilling dense
// weights for KV would cost every token, which is worse than a smaller context.
// An explicit numeric --ctx-size bypasses the ladder entirely.
// patchPlacementArgs replaces only the placement flags (-ot, --n-cpu-moe,
// --tensor-split, --split-mode) in an existing argv with a re-planned strategy's
// values, preserving every other flag (extras, warmup, backend dialect, etc.).
func patchPlacementArgs(args []string, s *placement.Strategy) []string {
	out := append([]string(nil), args...)
	set := func(name, val string) {
		if val == "" {
			return
		}
		for i := 0; i+1 < len(out); i++ {
			if out[i] == name {
				out[i+1] = val
				return
			}
		}
		out = append(out, name, val)
	}
	set("-ot", s.OTString)
	if s.ContextSize > 0 {
		set("--ctx-size", strconv.Itoa(s.ContextSize))
	}
	if s.NCPUMoE > 0 {
		set("--n-cpu-moe", strconv.Itoa(s.NCPUMoE))
	}
	if len(s.TensorSplit) > 0 {
		parts := make([]string, 0, len(s.TensorSplit))
		for _, v := range s.TensorSplit {
			parts = append(parts, fmt.Sprintf("%.2f", v))
		}
		set("--tensor-split", strings.Join(parts, ","))
	}
	set("--split-mode", s.SplitMode)
	// Re-patch the (u)batch sizes on every call — including the OOM-derate
	// re-plan path. Without this the launcher keeps launching the original
	// -ub 512 even after placement derated to a smaller ubatch, so the graph
	// reserve still OOMs and the server segfaults in a restart loop.
	if s.UBatchSize > 0 {
		set("-ub", strconv.Itoa(s.UBatchSize))
	}
	if s.BatchSize > 0 {
		set("-b", strconv.Itoa(s.BatchSize))
	}
	return out
}

// deepseek4V4GraphQuirk reports whether this model/backend can only reserve
// single-slot, KV-on-CPU graphs. The v4 fork's deepseek4 build_arch_graph
// aborts with GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2) (ggml.c:3660)
// when the hybrid-iswa KV cache lives on the GPU (ctx 262144 AND 1048576, FA
// on AND off, patched AND clean fork builds) and equally with --parallel 4
// even with KV on CPU — reproduced across five 15-minute loads on 2026-07-07.
// The ONLY configuration that has ever served this arch here is KV on CPU
// with a single sequence slot (verified end-to-end at ctx 1048576). Old
// kv=gpu probe caches are no evidence to the contrary: their compute values
// came from the fork's fit-table printout, and .place files are also written
// during OOM re-plans, not only on success.
func deepseek4V4GraphQuirk(model *placement.ModelProfile, backendTag string) bool {
	return model != nil && strings.EqualFold(model.ModelArch, "deepseek4") && strings.EqualFold(backendTag, "v4")
}

func claudeCodeComputeStrategy(caps *detect.Capabilities, model *placement.ModelProfile, opts placement.Options, claudeCode, ctxMax bool) (*placement.Strategy, error) {
	fmt.Fprintf(os.Stderr, "[launch] claudeCodeComputeStrategy: claudeCode=%v ctxMax=%v ctx=%d\n", claudeCode, ctxMax, opts.ContextSize)
	if !claudeCode || (opts.ContextSize > 0 && !ctxMax) {
		fmt.Fprintf(os.Stderr, "[launch] claudeCodeComputeStrategy: direct path (no ladder)\n")
		return placement.Compute(caps, model, opts)
	}
	if deepseek4V4GraphQuirk(model, opts.BackendTag) {
		// The KV-on-GPU ladder would pick a placement the fork cannot build a
		// graph for. The default placement (KV on CPU, ctx fit → the trained
		// context) is the verified path; with claude-code's --parallel 4 the
		// per-slot window is still ctx/4.
		fmt.Println("[claude-code] deepseek4 on the v4 fork cannot reserve KV-on-GPU graphs; using the verified KV-on-CPU placement")
		return placement.Compute(caps, model, opts)
	}
	target := model.CTXTrain
	if target <= 0 {
		target = 262144
	}
	for _, ctx := range claudeCodeCtxLadder(target) {
		o := opts
		o.ContextSize = ctx
		s, err := placement.Compute(caps, model, o)
		if err != nil {
			continue
		}
		if s.KVPlacement == "cpu" || (!model.IsMoE && s.Type == placement.CPUOnly) {
			continue
		}
		if ctx == target {
			fmt.Printf("[claude-code] context-first placement: ctx %d, KV on GPU (model layout adjusts around it)\n", ctx)
		} else {
			fmt.Printf("[claude-code] context-first placement: %d does not fit, using ctx %d (KV on GPU)\n", target, ctx)
		}
		return s, nil
	}
	// No ladder rung fit with KV on GPU — fall back to the default placement.
	return placement.Compute(caps, model, opts)
}

// claudeCodeSamplingArgs appends anti-loop sampling defaults in Claude Code mode.
// The Anthropic /v1/messages conversion only forwards temperature/top_p/top_k from
// the client (server-chat.cpp), and the Anthropic API has no penalty fields at all —
// so repetition control MUST come from server-side defaults, and ik_llama ships with
// every penalty disabled (repeat 1.0, presence 0.0). Quantized thinking models
// (Qwen3.x model card explicitly warns) fall into endless repetition without them:
// the user-visible symptom is repeated phrases and the model re-issuing the same
// tool call, since the tool-call grammar shapes degenerate output into valid JSON.
// Values: presence-penalty 1.0 (Qwen recommends up to 2 against repetition; 1.0 is
// mild enough for code), repeat-penalty 1.05 over the last 512 tokens (targets tight
// local loops, small enough to leave code idioms alone), top-k 20 / top-p 0.95 /
// min-p 0 (Qwen thinking-mode recommendation; also softens the client's greedy
// temperature-0 classifier calls, where penalties still apply to the argmax).
// Any flag the user already passed (ExtraArgs) wins — we skip it here.
func claudeCodeSamplingArgs(args []string, claudeCode bool, model *placement.ModelProfile) []string {
	if !claudeCode {
		return args
	}
	defaults := [][2]string{
		{"--presence-penalty", "1.0"},
		{"--repeat-penalty", "1.05"},
		{"--repeat-last-n", "512"},
		{"--top-k", "20"},
		{"--top-p", "0.95"},
		{"--min-p", "0.0"},
	}
	if model != nil && strings.EqualFold(model.ModelArch, "deepseek4") {
		// V4 template starts assistant turns inside <think>. The validated
		// server recipe closes that immediately; leaving the budget unlimited
		// made Claude Code requests wander in malformed thinking output and the
		// Anthropic parser returned 500s before any useful tool call/content.
		defaults = [][2]string{
			{"--presence-penalty", "1.0"},
			{"--repeat-penalty", "1.05"},
			{"--repeat-last-n", "512"},
			{"--temp", "0.7"},
			{"--top-k", "40"},
			{"--top-p", "0.95"},
			{"--min-p", "0.05"},
			{"--reasoning-budget", "0"},
		}
	}
	for _, d := range defaults {
		if !hasArg(args, d[0]) {
			args = append(args, d[0], d[1])
		}
	}
	return args
}

// claudeCodeAliasArgs appends `--alias local` so the backend's /v1/models advertises
// "local", matching the ANTHROPIC_MODEL=local the client uses. Without it llama.cpp/
// ik_llama advertise the gguf file path as the model id, and Claude Code's interactive
// model check rejects "local" ("the selected model (local) ... may not exist"). Both
// backends honor --alias (verified). No-op outside claude-code mode, or if the user
// already passed an alias.
func claudeCodeAliasArgs(args []string, claudeCode bool) []string {
	if !claudeCode || hasArg(args, "--alias") || hasArg(args, "-a") {
		return args
	}
	return append(args, "--alias", "local")
}

// runClaudeCodeClient launches Claude Code in the foreground wired to the local
// server, inheriting the terminal. It returns claude's exit code, or -1 if the
// `claude` CLI isn't installed (so the caller can fall back to the recipe).
func runClaudeCodeClient(host string, port int, serverArgs, extraArgs []string) int {
	claudePath, err := exec.LookPath("claude")
	if err != nil {
		return -1
	}
	clientHost := host
	if clientHost == "" || clientHost == "0.0.0.0" || clientHost == "::" {
		clientHost = "127.0.0.1"
	}
	fmt.Printf("[claude-code] Claude Code → http://%s:%d\n", clientHost, port)
	args := extraArgs
	// Built-in WebSearch is an Anthropic server-side tool; on a local endpoint it
	// can't run, and the model loops on it while the auto-permission classifier
	// fails. Disable it and wire a no-key DuckDuckGo MCP in its place so agents and
	// workflows can still do web research. Skip either if the user passed their own.
	if !hasArg(extraArgs, "--disallowedTools") {
		args = append([]string{"--disallowedTools", "WebSearch"}, args...)
	}
	if mcp := claudeCodeSearchMCPArgs(extraArgs); mcp != nil {
		args = append(mcp, args...)
		fmt.Println("[claude-code] WebSearch disabled (Anthropic-only); DuckDuckGo search MCP wired in (mcp__ddg-search__search).")
	} else {
		fmt.Println("[claude-code] WebSearch disabled (Anthropic-only); install uvx or add a search MCP for web research.")
	}
	cmd := exec.Command(claudePath, args...)
	cmd.Env = claudeCodeEnv(host, port, serverArgs)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintf(os.Stderr, "[claude-code] failed to run claude: %v\n", err)
		return 1
	}
	return 0
}

// waitForHealth polls the server's /health (then /v1/models) until it answers or
// the timeout elapses. Used by the TUI path, where the backend starts in a
// background goroutine and there's no synchronous readiness signal.
func waitForHealth(host string, port int, timeout time.Duration) bool {
	clientHost := host
	if clientHost == "" || clientHost == "0.0.0.0" || clientHost == "::" {
		clientHost = "127.0.0.1"
	}
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		for _, path := range []string{"/health", "/v1/models"} {
			resp, err := client.Get(fmt.Sprintf("http://%s:%d%s", clientHost, port, path))
			if err == nil {
				resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					return true
				}
			}
		}
		time.Sleep(time.Second)
	}
	return false
}

// isServerRunning returns true if the server at host:port responds to /health
// with 200 OK within a short timeout.
func isServerRunning(host string, port int) bool {
	clientHost := host
	if clientHost == "" || clientHost == "0.0.0.0" || clientHost == "::" {
		clientHost = "127.0.0.1"
	}
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(fmt.Sprintf("http://%s:%d/health", clientHost, port))
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
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
		fmt.Fprintln(os.Stderr, "Usage: ggrun download <repo/name>")
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
		fallback = 8
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

func cmdRecommend(args []string) {
	limit := 5
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-n", "--limit":
			if i+1 < len(args) {
				if n, err := strconv.Atoi(args[i+1]); err == nil && n > 0 {
					limit = n
				}
				i++
			}
		default:
			if n, err := strconv.Atoi(strings.TrimPrefix(args[i], "-n")); err == nil && n > 0 {
				limit = n
			}
		}
	}

	recommend.MaybeRefresh() // pull the latest published catalog (TTL-gated, best-effort)

	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}

	gpu := "CPU only"
	if len(caps.GPUs) > 0 {
		names := make([]string, 0, len(caps.GPUs))
		for _, g := range caps.GPUs {
			names = append(names, fmt.Sprintf("%s %dGB", g.Name, g.VRAMTotalMB/1024))
		}
		gpu = strings.Join(names, " + ")
	}
	fmt.Printf("Hardware: %s | RAM %dGB\n", gpu, caps.RAM.TotalMB/1024)

	if cfg, err := config.Load(); err == nil {
		if headroomMB := parseBudgetMB(cfg.VRAMHeadroom); headroomMB > 0 {
			fmt.Printf("VRAM headroom: %d MB reserved (set via Settings or --vram-headroom)\n", headroomMB)
			caps = detect.ApplyVRAMHeadroom(caps, headroomMB)
		}
		if headroomMB := parseBudgetMB(cfg.RAMHeadroom); headroomMB > 0 {
			fmt.Printf("RAM headroom: %d MB reserved (set via Settings or --ram-headroom)\n", headroomMB)
			caps = detect.ApplyRAMHeadroom(caps, headroomMB)
		}
	}

	cats := recommend.TopCategories(caps, limit)
	if len(cats.Balanced) == 0 {
		fmt.Println("No models in the catalog fit this machine.")
		return
	}
	printRecGroup := func(title string, rows []recommend.Recommendation) {
		if len(rows) == 0 {
			return
		}
		fmt.Printf("\n%s\n", title)
		fmt.Printf("  %-36s %-10s %-8s %6s %5s %8s\n", "Model", "Fit", "Quant", "Size", "Qual", "Speed")
		for _, r := range rows {
			name := r.Name
			if len(name) > 36 {
				name = name[:35] + "…"
			}
			tps := "—"
			if r.PredictedTPS > 0 {
				tps = fmt.Sprintf("%.0f t/s", r.PredictedTPS)
			}
			fmt.Printf("  %-36s %-10s %-8s %5.1fG %4.0f%% %8s\n",
				name, recommend.DisplayFit(r.Fit), r.QuantName, r.QuantSizeGB, r.QualityRetained*100, tps)
		}
	}
	printRecGroup("Best overall — balanced quality, speed and fit", cats.Balanced)
	printRecGroup("Smartest — highest intelligence that fits", cats.Smartest)
	printRecGroup("Fastest — quickest while still capable", cats.Fastest)
	fmt.Printf("\n%s\n", recommend.CatalogAttribution())
}

func cmdTune(args []string) {
	req, err := parseLaunchArgs(args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(2)
	}
	if req.ModelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: ggrun tune <model.gguf>")
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

	tuneOpts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	tuneOpts.ReasoningOff = true // tuning measures throughput, so think-free like benchmarks
	strategy, err := placement.Compute(caps, model, tuneOpts)
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

	timeout := autoStartupTimeout(model)
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
			p, err := server.StartWithTimeout(flags, req.Port, timeout)
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
		fmt.Fprintln(os.Stderr, "Usage: ggrun daemon --model <model.gguf>")
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
		fmt.Fprintln(os.Stderr, "Usage: ggrun config [show|edit|path|reset]")
		os.Exit(2)
	}
}

func cmdUpdate() {
	// Self-update ggrun
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

// isIKOnlyArch reports whether a model architecture can only be loaded by
// ik_llama.cpp; mainline llama.cpp rejects these with "unknown model architecture".
func isIKOnlyArch(arch string) bool {
	a := strings.ToLower(strings.TrimSpace(arch))
	return strings.HasPrefix(a, "minimax-m") // minimax-m2, minimax-m3, ...
}

// availableIKBinary returns the path of a detected ik_llama.cpp server binary, if any.
func availableIKBinary(caps *detect.Capabilities) string {
	seen := map[string]bool{}
	cands := make([]string, 0, len(caps.Backends)+4)
	for _, b := range caps.Backends {
		cands = append(cands, b.Path)
	}
	cands = append(cands, backendSearchPaths()...)
	for _, p := range cands {
		if p == "" || seen[p] {
			continue
		}
		seen[p] = true
		if _, err := os.Stat(p); err != nil {
			continue
		}
		if detectBackend(p).IsIK {
			return p
		}
	}
	return ""
}

// preflightBackendArch fails fast with an actionable message when the model needs
// ik_llama.cpp but the resolved backend is mainline llama.cpp, instead of letting
// the backend die later with a cryptic "unknown model architecture" load error.
func preflightBackendArch(model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities) {
	if model == nil || be == nil || be.IsIK || !isIKOnlyArch(model.ModelArch) {
		return
	}
	fmt.Fprintf(os.Stderr,
		"Error: model architecture %q needs the ik_llama.cpp backend, but the selected backend is mainline llama.cpp.\n"+
			"  backend binary: %s\n", model.ModelArch, be.Path)
	if ik := availableIKBinary(caps); ik != "" {
		fmt.Fprintf(os.Stderr,
			"  fix: set LLAMA_SERVER=%q in your ggrun config (.config/config),\n"+
				"       or unset LLAMA_SERVER and keep LLM_BACKEND=ik_llama.\n", ik)
	} else {
		fmt.Fprintln(os.Stderr,
			"  fix: no ik_llama.cpp binary found. Build/install ik_llama.cpp and point LLAMA_SERVER at its llama-server.")
	}
	os.Exit(1)
}

// gateBackendGPU guards against the decoupling of hardware detection and backend
// capability: ggrun may detect NVIDIA GPUs while the active llama-server is a
// CPU-only build (e.g. the default Windows bundle), in which case placement
// would emit -ngl / -ot ...=CUDA0 flags the binary cannot honor — it aborts with
// "unknown buffer type" and the launcher used to crash-loop on it. When the
// active backend cannot see any GPU, run CPU-clean and tell the user how to get
// GPU acceleration. If the backend cannot be probed, caps is left untouched so
// behavior is unchanged elsewhere (recovery's FailureBackendCapability fast-fail
// still catches a real mismatch without an infinite restart loop).
func gateBackendGPU(be *backendInfo, caps *detect.Capabilities) *detect.Capabilities {
	if caps == nil || be == nil || len(caps.GPUs) == 0 {
		return caps
	}
	capable, probed := backendGPUCapable(be.Path)
	if !probed || capable {
		return caps
	}
	fmt.Fprintf(os.Stderr, "[launch] notice: %d GPU(s) detected but backend %s is a CPU-only build — running on CPU.\n", len(caps.GPUs), be.Path)
	fmt.Fprintln(os.Stderr, "[launch] for GPU acceleration reinstall the GPU backend (Windows: install.ps1 -Backend cuda) or set LLAMA_SERVER to a CUDA-capable llama-server.")
	cpuCaps := *caps
	cpuCaps.GPUs = nil
	return &cpuCaps
}

// backendGPUCapable probes whether the backend binary can see any GPU device by
// running `llama-server --list-devices` (supported by both mainline llama.cpp
// and ik_llama.cpp, and independent of whether the GPU backend is statically
// linked or a dynamic ggml-*.{dll,so}). probed is false when the probe could not
// run or its output was unrecognized, so the caller falls back to prior behavior.
func backendGPUCapable(binPath string) (capable, probed bool) {
	if binPath == "" {
		return false, false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, binPath, "--list-devices").CombinedOutput()
	if err != nil && len(out) == 0 {
		return false, false
	}
	text := strings.ToLower(string(out))
	idx := strings.Index(text, "available devices")
	if idx < 0 {
		return false, false
	}
	// ggrun's placement supports CUDA, Vulkan, and Metal; AMD/Intel GPUs run
	// through Vulkan. ROCm/HIP/SYCL backends aren't supported, so they're not
	// probed here.
	for _, kw := range []string{"cuda", "vulkan", "metal"} {
		if strings.Contains(text[idx:], kw) {
			return true, true
		}
	}
	return false, true
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
		TokenEmbdBytes:     info.TokenEmbdBytes,
		OutputBytes:        info.OutputBytes,
		ShexpBytes:         info.ShexpBytes,
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
		return nil, err
	}

	profile := infoToProfile(info, path)

	// Handle multi-part models: sum all shard files
	profile.SizeBytes = totalModelSize(path)

	// Fallback safety net only. parse_gguf.py now sizes every tensor from its real
	// on-disk byte span (offset deltas), so expert/non-expert already sum to the
	// file size and this rescale is a no-op (scale ~= 1.0). It still guards the rare
	// case where a GGUF's offsets are unusable and the parser falls back to the
	// per-ggml-type size table (which mis-sizes new quants like MXFP4): then the
	// sum drifts from the file and we rescale, keeping the expert:non-expert ratio.
	if tableTotal := profile.ExpertBytes + profile.NonExpertBytes; tableTotal > 0 && profile.SizeBytes > 0 {
		if scale := float64(profile.SizeBytes) / float64(tableTotal); scale < 0.95 || scale > 1.05 {
			profile.ExpertBytes = int64(float64(profile.ExpertBytes) * scale)
			profile.NonExpertBytes = int64(float64(profile.NonExpertBytes) * scale)
			profile.TokenEmbdBytes = int64(float64(profile.TokenEmbdBytes) * scale)
			profile.OutputBytes = int64(float64(profile.OutputBytes) * scale)
			profile.ShexpBytes = int64(float64(profile.ShexpBytes) * scale)
		}
	}

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
			// os.Stat, not entry.Info(): shards may be symlinks (models dir
			// symlinked to another disk), and entry.Info() returns the link's
			// own ~73-byte size. That once shrank a 146GB model to 365 bytes,
			// and the parseModel drift-rescale then crushed ExpertBytes with
			// it — placement pinned all 43 expert layers onto one GPU.
			fi, err := os.Stat(filepath.Join(dir, name))
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
	lowerBase := strings.ToLower(filepath.Base(path))
	lowerDir := strings.ToLower(filepath.Dir(path))
	if strings.Contains(help, "ikawrakow") || strings.Contains(help, "split-mode-graph") {
		info.IsIK = true
		info.Tag = "ik_llama"
	} else if strings.Contains(lowerBase, "vulkan") || strings.Contains(lowerDir, "build-vulkan") {
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
