package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/server"
)

const freeTokenUsage = `Usage: ggrun freetoken <checkpoint-dir-or-hf-id> [options] [-- <ft serve args>]

Experimental single-GPU adapter for FreeToken's native serving engine.
FreeToken must be installed separately; ggrun does not translate GGUF placement
or llama.cpp tuning flags into this path.

Options:
  --gpu N                 Physical NVIDIA GPU index (required on multi-GPU hosts)
  --host HOST             Bind address (default 127.0.0.1)
  --port N                API port (default 1919)
  --ctx N                 FreeToken maximum sequence length override
  --parallel N            Maximum running requests (default 1)
  --moe-backend MODE      auto|fused|offload|cpu|hybrid (default auto)
  --startup-timeout N     Model-load timeout in seconds (default 900)
  --ft-bin PATH           FreeToken CLI (default FREETOKEN_BIN or ft on PATH)
  --dry-run               Print the isolated environment and argv; start nothing

FreeToken-specific options such as --memory-ratio or --moe-cache-auto must go
after --. Adapter-owned model/host/port/context/parallel/backend and tensor-
parallel flags are rejected there so the single-GPU boundary stays enforceable.
`

type freeTokenRequest struct {
	Model              string
	GPU                int
	GPUSet             bool
	Host               string
	Port               int
	ContextSize        int
	Parallel           int
	MoEBackend         string
	StartupTimeoutSecs int
	Bin                string
	DryRun             bool
	Help               bool
	ExtraArgs          []string
}

func defaultFreeTokenRequest() *freeTokenRequest {
	return &freeTokenRequest{
		GPU:                -1,
		Host:               "127.0.0.1",
		Port:               1919,
		Parallel:           1,
		MoEBackend:         "auto",
		StartupTimeoutSecs: 900,
	}
}

func parseFreeTokenArgs(args []string) (*freeTokenRequest, error) {
	req := defaultFreeTokenRequest()
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--" {
			req.ExtraArgs = append(req.ExtraArgs, args[i+1:]...)
			break
		}
		key, inline, hasInline := strings.Cut(arg, "=")
		nextValue := func() (string, error) {
			if hasInline {
				if inline == "" {
					return "", fmt.Errorf("%s requires a value", key)
				}
				return inline, nil
			}
			if i+1 >= len(args) {
				return "", fmt.Errorf("%s requires a value", key)
			}
			i++
			return args[i], nil
		}

		switch key {
		case "-h", "--help":
			if hasInline {
				return nil, fmt.Errorf("%s does not take a value", key)
			}
			req.Help = true
			return req, nil
		case "--dry-run":
			if hasInline {
				return nil, fmt.Errorf("--dry-run does not take a value")
			}
			req.DryRun = true
		case "--model", "-m":
			value, err := nextValue()
			if err != nil {
				return nil, err
			}
			if req.Model != "" {
				return nil, fmt.Errorf("FreeToken accepts exactly one model")
			}
			req.Model = value
		case "--gpu":
			value, err := nextValue()
			if err != nil {
				return nil, err
			}
			gpu, err := strconv.Atoi(value)
			if err != nil || gpu < 0 {
				return nil, fmt.Errorf("--gpu must be a non-negative physical GPU index")
			}
			req.GPU, req.GPUSet = gpu, true
		case "--host":
			value, err := nextValue()
			if err != nil {
				return nil, err
			}
			if strings.TrimSpace(value) == "" {
				return nil, fmt.Errorf("--host cannot be empty")
			}
			req.Host = value
		case "--port":
			value, err := nextValue()
			if err != nil {
				return nil, err
			}
			port, err := config.ParsePort(value)
			if err != nil {
				return nil, fmt.Errorf("--port: %w", err)
			}
			req.Port = port
		case "--ctx", "--ctx-size":
			value, err := nextValue()
			if err != nil {
				return nil, err
			}
			ctx, err := strconv.Atoi(value)
			if err != nil || ctx <= 0 {
				return nil, fmt.Errorf("%s must be a positive integer", key)
			}
			req.ContextSize = ctx
		case "--parallel":
			value, err := nextValue()
			if err != nil {
				return nil, err
			}
			parallel, err := strconv.Atoi(value)
			if err != nil || parallel <= 0 || parallel > 64 {
				return nil, fmt.Errorf("--parallel must be between 1 and 64")
			}
			req.Parallel = parallel
		case "--moe-backend":
			value, err := nextValue()
			if err != nil {
				return nil, err
			}
			value = strings.ToLower(strings.TrimSpace(value))
			switch value {
			case "auto", "fused", "offload", "cpu", "hybrid":
				req.MoEBackend = value
			default:
				return nil, fmt.Errorf("--moe-backend must be auto, fused, offload, cpu, or hybrid")
			}
		case "--startup-timeout", "--startup-timeout-secs":
			value, err := nextValue()
			if err != nil {
				return nil, err
			}
			seconds, err := strconv.Atoi(value)
			if err != nil || seconds <= 0 || seconds > 86400 {
				return nil, fmt.Errorf("%s must be between 1 and 86400 seconds", key)
			}
			req.StartupTimeoutSecs = seconds
		case "--ft-bin":
			value, err := nextValue()
			if err != nil {
				return nil, err
			}
			if strings.TrimSpace(value) == "" {
				return nil, fmt.Errorf("--ft-bin cannot be empty")
			}
			req.Bin = value
		default:
			if strings.HasPrefix(arg, "-") {
				return nil, fmt.Errorf("unknown adapter option %q; put native ft serve options after --", arg)
			}
			if req.Model != "" {
				return nil, fmt.Errorf("FreeToken accepts exactly one model; got %q and %q", req.Model, arg)
			}
			req.Model = arg
		}
	}
	if req.Help {
		return req, nil
	}
	if strings.TrimSpace(req.Model) == "" {
		return nil, fmt.Errorf("a checkpoint directory or Hugging Face model ID is required")
	}
	if err := validateFreeTokenPassthrough(req.ExtraArgs); err != nil {
		return nil, err
	}
	return req, nil
}

func validateFreeTokenPassthrough(args []string) error {
	owned := map[string]bool{
		"--model":                true,
		"--model-path":           true,
		"--host":                 true,
		"--port":                 true,
		"--max-seq-len-override": true,
		"--max-running-requests": true,
		"--moe-backend":          true,
		"--tensor-parallel-size": true,
		"--tp-size":              true,
		"--shell-mode":           true,
	}
	for _, arg := range args {
		key := arg
		if before, _, ok := strings.Cut(arg, "="); ok {
			key = before
		}
		if owned[key] {
			return fmt.Errorf("FreeToken passthrough flag %s is adapter-owned; set it before -- using the matching ggrun option", key)
		}
	}
	return nil
}

func resolveFreeTokenBinary(explicit string) (string, string, error) {
	candidate := strings.TrimSpace(explicit)
	if candidate == "" {
		candidate = strings.TrimSpace(os.Getenv("FREETOKEN_BIN"))
	}
	if candidate == "" {
		candidate = "ft"
	}
	path, err := exec.LookPath(candidate)
	if err != nil {
		return "", "", fmt.Errorf("FreeToken CLI %q not found; install it separately with `uv pip install \"freetoken[accel]\"` or pass --ft-bin", candidate)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, runErr := exec.CommandContext(ctx, path, "--version").CombinedOutput()
	if ctx.Err() != nil {
		return "", "", fmt.Errorf("FreeToken version probe timed out: %s", path)
	}
	version := strings.TrimSpace(string(out))
	if runErr != nil {
		return "", "", fmt.Errorf("FreeToken CLI cannot start: %s: %s", path, firstNonEmptyLine(version))
	}
	if !strings.Contains(strings.ToLower(version), "freetoken version") {
		return "", "", fmt.Errorf("%s is not a recognized FreeToken CLI (--version returned %q)", path, firstNonEmptyLine(version))
	}
	return path, firstNonEmptyLine(version), nil
}

func selectFreeTokenGPU(caps *detect.Capabilities, requested int, requestedSet bool) (detect.GPU, error) {
	var nvidia []detect.GPU
	if caps != nil {
		for _, gpu := range caps.GPUs {
			if strings.Contains(strings.ToLower(gpu.Name), "nvidia") {
				nvidia = append(nvidia, gpu)
			}
		}
	}
	if len(nvidia) == 0 {
		return detect.GPU{}, fmt.Errorf("the FreeToken adapter requires an NVIDIA CUDA GPU")
	}
	if requestedSet {
		for _, gpu := range nvidia {
			if gpu.Index == requested {
				if strings.TrimSpace(gpu.PCIBusID) == "" {
					return detect.GPU{}, fmt.Errorf("GPU %d has no PCI bus ID; refusing ambiguous CUDA isolation", requested)
				}
				return gpu, nil
			}
		}
		return detect.GPU{}, fmt.Errorf("--gpu %d is not a detected NVIDIA GPU; available: %s", requested, freeTokenGPUList(nvidia))
	}
	if len(nvidia) != 1 {
		return detect.GPU{}, fmt.Errorf("this host has %d NVIDIA GPUs; choose exactly one with --gpu (%s)", len(nvidia), freeTokenGPUList(nvidia))
	}
	if strings.TrimSpace(nvidia[0].PCIBusID) == "" {
		return detect.GPU{}, fmt.Errorf("the detected NVIDIA GPU has no PCI bus ID; refusing ambiguous CUDA isolation")
	}
	return nvidia[0], nil
}

func freeTokenGPUList(gpus []detect.GPU) string {
	parts := make([]string, 0, len(gpus))
	for _, gpu := range gpus {
		parts = append(parts, fmt.Sprintf("%d=%s", gpu.Index, gpu.Name))
	}
	sort.Strings(parts)
	return strings.Join(parts, ", ")
}

func buildFreeTokenCommand(req *freeTokenRequest, bin string) []string {
	args := []string{
		bin, "serve",
		"--model", req.Model,
		"--host", req.Host,
		"--port", strconv.Itoa(req.Port),
		"--max-running-requests", strconv.Itoa(req.Parallel),
		"--moe-backend", req.MoEBackend,
	}
	if req.ContextSize > 0 {
		args = append(args, "--max-seq-len-override", strconv.Itoa(req.ContextSize))
	}
	return append(args, req.ExtraArgs...)
}

func cmdFreeToken(args []string) {
	req, err := parseFreeTokenArgs(args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n\n%s", err, freeTokenUsage)
		os.Exit(2)
	}
	if req.Help {
		fmt.Print(freeTokenUsage)
		return
	}
	bin, ftVersion, err := resolveFreeTokenBinary(req.Bin)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}
	gpu, err := selectFreeTokenGPU(caps, req.GPU, req.GPUSet)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	serverArgs := buildFreeTokenCommand(req, bin)
	env := []string{fmt.Sprintf("CUDA_VISIBLE_DEVICES=%d", gpu.Index)}
	if req.DryRun {
		fmt.Println(displayCommandWithEnv([]string{"CUDA_DEVICE_ORDER=PCI_BUS_ID", env[0]}, serverArgs))
		return
	}
	if err := guardPortFree(req.Port, "FreeToken"); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "[freetoken] experimental adapter: %s\n", ftVersion)
	fmt.Fprintf(os.Stderr, "[freetoken] isolated to physical GPU %d (%s, %d MiB free); ggrun GGUF placement and tune caches are not applied\n",
		gpu.Index, gpu.Name, gpu.VRAMFreeMB())
	process, err := server.StartWithTimeoutToEnv(
		serverArgs,
		req.Port,
		time.Duration(req.StartupTimeoutSecs)*time.Second,
		os.Stdout,
		os.Stderr,
		env,
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error starting FreeToken: %v\n", err)
		os.Exit(1)
	}
	if _, err := process.QueryModels(); err != nil {
		_ = process.Stop()
		fmt.Fprintf(os.Stderr, "Error: FreeToken passed health but /v1/models failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("[freetoken] ready at http://%s:%d (OpenAI /v1/* and Anthropic /v1/messages)\n", req.Host, req.Port)
	fmt.Printf("[freetoken] for Claude Code, use FreeToken's native launcher in another terminal: ft launch claude --server http://127.0.0.1:%d\n", req.Port)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, shutdownSignals()...)
	crashed := waitForShutdownOrCrash(process, sigCh)
	signal.Stop(sigCh)
	if crashed {
		_ = process.Stop()
		fmt.Fprintln(os.Stderr, "Error: FreeToken server exited unexpectedly")
		os.Exit(1)
	}
	if err := process.Stop(); err != nil {
		fmt.Fprintf(os.Stderr, "Error stopping FreeToken: %v\n", err)
		os.Exit(1)
	}
}
