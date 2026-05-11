package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/raketenkater/llm-server/pkg/benchmark"
	"github.com/raketenkater/llm-server/pkg/daemon"
	"github.com/raketenkater/llm-server/pkg/detect"
	"github.com/raketenkater/llm-server/pkg/placement"
	"github.com/raketenkater/llm-server/pkg/server"
	"github.com/raketenkater/llm-server/pkg/tui"
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
	case "gui", "tui":
		cmdGUI()
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
  launch <model.gguf>  Launch model with auto-placement
  benchmark <model>    Benchmark a running server
  daemon               Start persistent daemon
  dry-run <model.gguf> Print computed flags without launching
  gui, tui             Interactive TUI (model picker, settings, launch)

Launch flags:
  -port int            Server port (default 8081)
  -ctx int             Context size (default auto)
  -kv string           KV placement: auto|gpu|cpu (default auto)
  -kv-quality string   KV quality: high|mid|low (default mid)
  -cpu                 Force CPU-only mode
  -gpus string         Comma-separated GPU indices
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
	ctxSize := fs.Int("ctx", 0, "Context size")
	kvPlacement := fs.String("kv", "auto", "KV placement")
	kvQuality := fs.String("kv-quality", "mid", "KV quality")
	cpuMode := fs.Bool("cpu", false, "Force CPU-only")
	gpusFlag := fs.String("gpus", "", "Comma-separated GPU indices")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "Usage: llm-server launch <model.gguf>")
		os.Exit(2)
	}
	modelPath := fs.Arg(0)

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

	opts := placement.Options{
		ContextSize: *ctxSize,
		KVPlacement: *kvPlacement,
		KVQuality:   *kvQuality,
		CPUMode:     *cpuMode,
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

	// Find backend binary
	binPath := findBackend(caps)
	if binPath == "" {
		fmt.Fprintln(os.Stderr, "Error: no llama-server binary found")
		os.Exit(1)
	}

	serverArgs := append([]string{binPath}, strategy.Args(modelPath, *port)...)
	fmt.Printf("[launch] %s\n", strings.Join(serverArgs, " "))

	p, err := server.Start(serverArgs, *port)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error starting server: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("[launch] Server running on port %d (PID %d)\n", *port, p.Cmd.Process.Pid)
	fmt.Println("[launch] Press Ctrl+C to stop")

	// Block until interrupted
	select {}
}

func cmdGUI() {
	if err := tui.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func cmdDryRun(args []string) {
	fs := flag.NewFlagSet("dry-run", flag.ExitOnError)
	port := fs.Int("port", 8081, "Server port")
	ctxSize := fs.Int("ctx", 0, "Context size")
	kvPlacement := fs.String("kv", "auto", "KV placement")
	kvQuality := fs.String("kv-quality", "mid", "KV quality")
	cpuMode := fs.Bool("cpu", false, "Force CPU-only")
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

	strategy, err := placement.Compute(caps, model, placement.Options{
		ContextSize: *ctxSize,
		KVPlacement: *kvPlacement,
		KVQuality:   *kvQuality,
		CPUMode:     *cpuMode,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %v\n", err)
		os.Exit(1)
	}

	binPath := findBackend(caps)
	if binPath == "" {
		binPath = "llama-server"
	}

	serverArgs := append([]string{binPath}, strategy.Args(modelPath, *port)...)
	fmt.Println(strings.Join(serverArgs, " "))
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

func cmdDaemon(args []string) {
	fs := flag.NewFlagSet("daemon", flag.ExitOnError)
	modelPath := fs.String("model", "", "Model path")
	port := fs.Int("port", 8081, "Server port")
	controlPort := fs.Int("control-port", 9090, "Control API port")
	fs.Parse(args)

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: llm-server daemon --model <model.gguf>")
		os.Exit(2)
	}

	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}

	model, err := parseModel(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing model: %v\n", err)
		os.Exit(1)
	}

	strategy, err := placement.Compute(caps, model, placement.Options{})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %v\n", err)
		os.Exit(1)
	}

	binPath := findBackend(caps)
	if binPath == "" {
		fmt.Fprintln(os.Stderr, "Error: no llama-server binary found")
		os.Exit(1)
	}

	serverArgs := append([]string{binPath}, strategy.Args(*modelPath, *port)...)

	d := daemon.New(daemon.Config{
		ModelPath:   *modelPath,
		ServerArgs:  serverArgs,
		Port:        *port,
		ControlPort: *controlPort,
	})
	if err := d.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// parseModel calls parse_gguf.py to extract model metadata.
func parseModel(path string) (*placement.ModelProfile, error) {
	scriptDir := os.Getenv("LLM_SCRIPT_DIR")
	if scriptDir == "" {
		// Try to find parse_gguf.py relative to binary or in PATH
		exe, _ := os.Executable()
		scriptDir = filepath.Join(filepath.Dir(exe), "..", "..")
	}
	parseScript := filepath.Join(scriptDir, "parse_gguf.py")
	if _, err := os.Stat(parseScript); os.IsNotExist(err) {
		// Fallback: try repo root next to go/
		cwd, _ := os.Getwd()
		parseScript = filepath.Join(cwd, "..", "parse_gguf.py")
	}
	if _, err := os.Stat(parseScript); os.IsNotExist(err) {
		// Last resort: look in PATH
		parseScript = "parse_gguf.py"
	}

	out, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	_ = out // We could call parse_gguf.py here, but for now use heuristics

	// Fallback: derive from file size and name
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}

	profile := &placement.ModelProfile{
		Path:      path,
		SizeBytes: info.Size(),
	}

	// Heuristic from filename
	name := strings.ToLower(filepath.Base(path))
	if strings.Contains(name, "moe") || strings.Contains(name, "mixtral") || strings.Contains(name, "minimax") || strings.Contains(name, "qwen3.6") {
		profile.IsMoE = true
		profile.NumExperts = 64
	}
	if strings.Contains(name, "70b") || strings.Contains(name, "72b") {
		profile.NumParams = 70_000_000_000
		profile.NumLayers = 80
		profile.HiddenSize = 8192
	} else if strings.Contains(name, "32b") || strings.Contains(name, "30b") {
		profile.NumParams = 32_000_000_000
		profile.NumLayers = 64
		profile.HiddenSize = 5120
	} else if strings.Contains(name, "14b") || strings.Contains(name, "15b") {
		profile.NumParams = 14_000_000_000
		profile.NumLayers = 48
		profile.HiddenSize = 5120
	} else if strings.Contains(name, "8b") || strings.Contains(name, "7b") {
		profile.NumParams = 8_000_000_000
		profile.NumLayers = 32
		profile.HiddenSize = 4096
	} else {
		// Default
		profile.NumParams = 7_000_000_000
		profile.NumLayers = 32
		profile.HiddenSize = 4096
	}

	// Size-based override for MoE
	if info.Size() > 50*1024*1024*1024 {
		profile.IsMoE = true
		profile.NumExperts = 64
		profile.NumLayers = 64
		profile.HiddenSize = 6144
	}

	return profile, nil
}

func findBackend(caps *detect.Capabilities) string {
	for _, b := range caps.Backends {
		if b.Name == "llama-server" || b.Name == "ik_llama" || b.Name == "ik_llama-server" {
			return b.Path
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
				return p
			}
		}
	}
	return ""
}
