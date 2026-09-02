package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
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
	"sync"
	"time"
	"unicode"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/benchmark"
	"github.com/raketenkater/ggrun/pkg/chattemplate"
	"github.com/raketenkater/ggrun/pkg/claudeauto"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/controller"
	"github.com/raketenkater/ggrun/pkg/daemon"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/download"
	"github.com/raketenkater/ggrun/pkg/gguf"
	"github.com/raketenkater/ggrun/pkg/libhub"
	modelstore "github.com/raketenkater/ggrun/pkg/models"
	"github.com/raketenkater/ggrun/pkg/modelusage"
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
		cmdGUI()
		return
	}

	args := os.Args[1:]
	if dispatchCompat(args) {
		return
	}

	// Offer updates before a long-running command starts. The machinery for
	// this already existed -- upstream-behind detection, release check, and a
	// dismiss-for-N-days window -- and had no caller, so backends drifted
	// silently. That is not cosmetic: a checkpoint-creation fix landed upstream
	// in May 2026 that directly governs whether prefix reuse works at all on
	// sliding-window models, and nothing here would have said so.
	//
	// PromptOnStartup already declines on non-interactive shells, on repeat
	// runs, and under LLM_SERVER_NO_UPDATE_CHECK, so scripts and CI never block.
	if promptsForUpdates(args) {
		update.PromptOnStartupWithBackendUpdater(updateAllBackends)
	}

	switch args[0] {
	case "help", "--help", "-h":
		usage()
	case "version", "--version", "-v":
		fmt.Println("ggrun", version)
	case "detect":
		cmdDetect(args[1:])
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
	case "freetoken":
		cmdFreeToken(args[1:])
	case "daemon":
		cmdDaemon(args[1:])
	case "status":
		cmdStatus(args[1:])
	case "claude-status":
		cmdClaudeStatus(args[1:])
	case "claude-workflow-hook":
		cmdClaudeWorkflowHook(args[1:])
	case "dry-run":
		cmdDryRun(args[1:])
	case "probe":
		cmdProbe()
	case "memory-probe":
		cmdMemoryProbe(args[1:])
	case "kv-probe":
		cmdKVProbe(args[1:])
	case "probe-reset":
		cmdProbeReset(args[1:])
	case "download":
		cmdDownload(args[1:])
	case "tune":
		cmdTune(args[1:])
	case "spec-test":
		cmdSpecTest(args[1:])
	case "recommend":
		cmdRecommend(args[1:])
	case "support", "advisor":
		cmdSupport(args[1:])
	case "models":
		cmdModels(args[1:])
	case "claude":
		cmdClaude(args[1:])
	case "gui", "tui":
		cmdGUI()
	case "config":
		cmdConfig(args[1:])
	case "backend", "backends":
		cmdBackend(args[1:])
	case "update", "--update":
		cmdUpdate(args[1:])
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, usageText())
}

// usageText returns the full help text so tests can assert on it without
// capturing stderr.
func usageText() string {
	return `Usage: ggrun [command] [args]

With no command, launches the interactive TUI (same as ggrun gui).

Commands:
  version              Show version
  detect               Detect hardware capabilities
  detect --bandwidth   Measure and cache host/CUDA transfer bandwidth
  launch <model.gguf>  Launch model with auto-placement
  benchmark <model>    Benchmark a running server
  freetoken <model>    Experimental single-GPU FreeToken server adapter
  daemon               Start persistent daemon
  dry-run <model.gguf> Print computed flags without launching
                       (--emit-server-argv-json supported)
  status [model.gguf]  Show the cached launch-optimizer winner/boundary
                       (--json supported)
  download <repo/name> Download from HuggingFace
                       (--dir <path> to use another disk, --quant <name> to
                       preselect instead of the interactive picker)
  tune <model.gguf>    AI-tune model for best performance
  recommend [-n N]     Rank models that fit this machine (intelligence x speed)
  recommend --first    Print the top Hugging Face repo only
  support              Native optional support expert / optimizer (status, install, doctor)
  models [list|browse|path|rm] List, browse, locate, or safely remove GGUF models
  config [show|edit|path|reset]  Manage settings
  backend [list|install|feature|add|register|remove]  Manage isolated llama.cpp
                       backends, routes, and reviewed source-feature overlays
  update, --update     Update ggrun and backends
  claude [list|resume] List recorded Claude Code sessions, or relaunch the recorded
                       backend shape and resume one (default: newest in this directory)
  gui, tui             Interactive TUI (model picker, settings, launch)

Diagnostics (advanced):
  probe                Check free GPU/RAM memory (useful when a launch's capacity numbers look wrong)
  memory-probe <model> Preview the computed backend memory plan without launching (--json supported)
  probe-reset <model>  Clear a learned VRAM reserve that's making launches overly conservative
                       (e.g. after an unrelated OOM); measured compute/KV probes are kept
  kv-probe <model>     Measure exact KV cache size for compressed-attention models; only needed
                       if launches undersize context
  spec-test <model>    Verify speculative-decoding (MTP) ceilings 1-4 against a target-only
                       baseline; only relevant with --spec mtp

Examples:
  ggrun model.gguf
  ggrun model.gguf --claude-code
  ggrun recommend
  ggrun model.gguf --dry-run

Launch flags:
  -port int            Server port (default 8081)
  -ctx string          Context size: fit|max|token count (default fit)
  -kv string           KV placement: auto|gpu|cpu (default auto)
  --kv-quality, -kv-quality string  KV quality: auto|high|mid|low or an exact llama.cpp type such as q5_1 (default auto)
  --cpu, -cpu          Force CPU-only mode
  --gpus, -gpus string  Comma-separated GPU indices
  --backend string     auto|llama|ik_llama|registered backend tag
  --parallel int       Concurrent sequence slots
  --threads int, -t    CPU threads (default: physical cores)
  --cache-ram, -cram int  Host prompt-cache budget in MiB; 0 derives it
  --claude-max-active int  Main-model requests admitted at once in --claude-code mode;
                       0 removes the limit, and it is capped at --parallel
  --mmap               Explicitly approve file-backed mmap when placement needs it
  --no-mmap            Require fully resident model weights
  --swa-full           Keep the full sliding-window KV cache for more prompt-cache hits
  --no-swa-full        Disable full SWA cache even when enabled in config
  --ram-limit-percent int  Maximum whole-host RAM utilisation (default 95)
  --vram-headroom str  Reserve VRAM the recommender/placement won't use, e.g. 2G
  --ram-headroom str   Reserve system RAM the recommender/placement won't use, e.g. 8G
  --allow-live-memory-probe  Approve a contained full-load probe for this launch when no complete dry-run is available
                       Set LLM_ALLOW_LIVE_MEMORY_PROBE=true to remember approval
  --vision, -vision    Enable vision (auto-detect mmproj)
  --claude-code        Serve locally and launch Claude Code with workflows/research
  --claude-profile str Claude Code scheduling (requires --claude-code): agent-interactive|agent-parallel
  --claude-reviewer str  Local reviewer/worker for Claude Code (requires --claude-code):
                       auto|qwen|qwen2b|nanbeige|off (default auto). auto and qwen
                       run the Qwen3.5-4B worker+reviewer; qwen2b the light
                       Qwen3.5-2B review-only lane; nanbeige the Nanbeige4.2
                       big-MoE worker; off seats no reviewer and the main model
                       self-reviews
  --chat-template str   Force a corrected chat template from the data-driven catalog
                       by entry name (e.g. nanbeige4.2-3b, qwen3.8-27b) for models whose
                       embedded template llama.cpp's minja engine cannot parse; a value
                       that is not a catalog entry is passed through as a backend flag
  --claude-resume str  Reopen a recorded Claude Code session (id or "latest") and resume
                       its interrupted workflow from the cached journal
  --claude-resume-force  Resume even though the backend shape changed (unsafe)
  --calibrate str      Core launch optimization: auto|on|off (default auto; auto runs/replays
                       a bounded agent-workload search, on widens an explicit screen)
  --hot-experts str    GPU cache for host-resident MoE experts: off|auto|on|slot-count
                       (auto may fall back cache-free; on requires optimizer-sized
                       cache-on; TUI can provision a compatible backend overlay)
  --worker-benchmark   Load once, measure throughput plus typed support/reviewer decisions,
                       print JSON, and stop
  --support-expert str Optional native expert/optimizer: off|auto|on (default auto)
  --support-online     Allow typed official llama.cpp research for support incidents
  --spec string        Speculative decoding: off|auto|mtp|dflash|eagle3|draft|ngram|ngram-mod|ngram-k4v
`
}

func knownCommand(cmd string) bool {
	switch cmd {
	case "help", "--help", "-h", "version", "--version", "-v", "detect", "launch", "benchmark", "freetoken", "daemon", "status", "claude-status", "claude-workflow-hook", "dry-run", "probe", "probe-reset", "memory-probe", "kv-probe", "download", "tune", "spec-test", "recommend", "support", "advisor", "models", "gui", "tui", "config", "backend", "backends", "claude", "update", "--update":
		return true
	default:
		return false
	}
}

// firstWordCouldBeAModel distinguishes the legacy `ggrun model.gguf` launch
// form from a misspelled command. A missing .gguf path is still treated as a
// model attempt so cmdLaunch can report the useful file error; existing paths
// and names found in the configured model directory preserve the other legacy
// launch forms.
func firstWordCouldBeAModel(word string) bool {
	if strings.EqualFold(filepath.Ext(word), ".gguf") {
		return true
	}
	if _, err := os.Stat(word); err == nil {
		return true
	}
	cfg, err := config.Load()
	if err != nil || cfg.ModelDir == "" {
		return false
	}
	_, err = os.Stat(filepath.Join(cfg.ModelDir, word))
	return err == nil
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
	modelWord := args[0]
	if strings.HasPrefix(modelWord, "-") {
		modelWord = firstPositional(args)
	}
	if modelWord == "" || !firstWordCouldBeAModel(modelWord) {
		fmt.Fprintf(os.Stderr, "unknown command %q\n", args[0])
		usage()
		os.Exit(2)
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

const longModelLoadWarningThresholdBytes int64 = 64 * 1024 * 1024 * 1024

// longModelLoadWarning is deliberately informational, not another consent
// prompt. Large resident/offloaded checkpoints can spend minutes in a contained
// allocation probe and may need one corrected retry; say that before the first
// backend load so an otherwise quiet terminal is not mistaken for a hang.
func longModelLoadWarning(model *placement.ModelProfile, timeout time.Duration) string {
	if model == nil {
		return ""
	}
	sizeBytes := model.SizeBytes
	if sizeBytes <= 0 && model.TotalSizeMB > 0 {
		sizeBytes = int64(model.TotalSizeMB) * 1024 * 1024
	}
	if sizeBytes < longModelLoadWarningThresholdBytes {
		return ""
	}
	kind := "model"
	if model.IsMoE {
		kind = "MoE model"
	}
	return fmt.Sprintf(
		"[launch] Long first load ahead: this %s is %s. Allocation measurement and a bounded recovery may load it more than once; each attempt can take several minutes (startup limit %s). Progress will continue below.",
		kind, formatModelBytes(sizeBytes), timeout,
	)
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

// userExplicitBackendFlag distinguishes command-line intent from a generated
// or config-default optimization already materialized in ExtraArgs. Recovery
// may remove the latter after measured rejection, but must fail closed rather
// than silently changing the former.
func userExplicitBackendFlag(req *launchRequest, flag string) bool {
	if req == nil || strings.TrimSpace(flag) == "" {
		return false
	}
	for _, arg := range req.OriginalArgs {
		if arg == flag || strings.HasPrefix(arg, flag+"=") {
			return true
		}
		if flag == "--swa-full" && (arg == "--no-swa-full" || strings.HasPrefix(arg, "--no-swa-full=")) {
			return true
		}
	}
	return false
}

func loadConfigOrExit() *config.Config {
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
		os.Exit(2)
	}
	return cfg
}

func placementErrorMessage(err error) string {
	if err == nil {
		return ""
	}
	msg := err.Error()
	if !strings.Contains(msg, "Model does not fit") {
		return msg
	}
	if hint := activeLlamaServerMemoryHint(); hint != "" {
		msg += "\n\n" + hint
	}
	return msg
}

type activeLlamaServerProcess struct {
	pid   int
	rssMB int
	cmd   string
}

func activeLlamaServerMemoryHint() string {
	if runtime.GOOS != "linux" {
		return ""
	}
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return ""
	}
	self := os.Getpid()
	procs := make([]activeLlamaServerProcess, 0, 4)
	for _, entry := range entries {
		pid, err := strconv.Atoi(entry.Name())
		if err != nil || pid == self {
			continue
		}
		cmdline, err := os.ReadFile(filepath.Join("/proc", entry.Name(), "cmdline"))
		if err != nil || len(cmdline) == 0 {
			continue
		}
		exe, _ := os.Readlink(filepath.Join("/proc", entry.Name(), "exe"))
		argv0 := string(cmdline)
		if end := strings.IndexByte(argv0, 0); end >= 0 {
			argv0 = argv0[:end]
		}
		if !isLlamaServerExecutable(exe) && !isLlamaServerExecutable(argv0) {
			continue
		}
		cmd := strings.TrimSpace(strings.ReplaceAll(string(cmdline), "\x00", " "))
		if cmd == "" {
			continue
		}
		procs = append(procs, activeLlamaServerProcess{pid: pid, rssMB: procRSSMB(pid), cmd: compactProcessCommand(cmd, 180)})
	}
	if len(procs) == 0 {
		return ""
	}
	sort.Slice(procs, func(i, j int) bool { return procs[i].rssMB > procs[j].rssMB })
	if len(procs) > 3 {
		procs = procs[:3]
	}
	var b strings.Builder
	b.WriteString("Active llama-server process(es) are currently consuming memory; if you are switching models, stop the current server and retry:\n")
	for _, p := range procs {
		if p.rssMB > 0 {
			fmt.Fprintf(&b, "  PID %d: %d MiB RSS — %s\n", p.pid, p.rssMB, p.cmd)
		} else {
			fmt.Fprintf(&b, "  PID %d: %s\n", p.pid, p.cmd)
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

func isLlamaServerExecutable(path string) bool {
	name := strings.ToLower(filepath.Base(strings.TrimSpace(path)))
	return strings.Contains(name, "llama-server") || strings.Contains(name, "ik_llama-server")
}

func procRSSMB(pid int) int {
	data, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "status"))
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "VmRSS:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		kb, err := strconv.Atoi(fields[1])
		if err != nil {
			return 0
		}
		return (kb + 1023) / 1024
	}
	return 0
}

func compactProcessCommand(cmd string, limit int) string {
	cmd = strings.Join(strings.Fields(cmd), " ")
	if limit <= 0 || len(cmd) <= limit {
		return cmd
	}
	if limit <= 3 {
		return cmd[:limit]
	}
	return cmd[:limit-3] + "..."
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
			case "--model", "-m", "--port", "-port", "--ctx", "-ctx", "--ctx-size", "-c", "--kv", "-kv", "--kv-placement", "--kv-quality", "--gpus", "--host", "--server-bin", "--mmproj", "--backend", "--tune-cache", "--rounds", "--ram-budget", "--ram-limit-percent", "--vram-headroom", "--ram-headroom", "--spec", "--parallel", "--claude-profile", "--lib-path", "--threads", "-t", "--cache-ram", "-cram", "--batch-size", "-b", "--ubatch-size", "-ub", "--support-expert":
				skip = true
			}
			continue
		}
		return a
	}
	return ""
}

func cmdDetect(args []string) {
	if len(args) == 1 && args[0] == "bandwidth" {
		args = []string{"--bandwidth"}
	}
	flags := flag.NewFlagSet("detect", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	flags.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: ggrun detect [--bandwidth]")
		flags.PrintDefaults()
	}
	measureBandwidth := flags.Bool("bandwidth", false, "measure and cache host/CUDA transfer bandwidth")
	if err := flags.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return
		}
		os.Exit(2)
	}
	if flags.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "Usage: ggrun detect [--bandwidth]")
		os.Exit(2)
	}
	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if *measureBandwidth {
		fmt.Fprintln(os.Stderr, "[detect] measuring host memcpy and pinned CUDA transfer bandwidth...")
		profile, path, measureErr := detect.MeasureAndSaveBandwidth(caps)
		if measureErr != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", measureErr)
			os.Exit(1)
		}
		if !detect.ApplyBandwidthProfile(caps, profile) {
			fmt.Fprintln(os.Stderr, "Error: measured bandwidth profile did not match current hardware")
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "[detect] saved hardware-matched bandwidth profile: %s\n", path)
	}
	data, err := caps.JSON()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(string(data))
}

type launchRequest struct {
	ModelPath   string
	Port        int
	CtxFlag     string
	KVPlacement string
	KVQuality   string
	// KVQualityV forces the V-cache leg (--cache-type-v) to a fixed type while
	// KVQuality still sizes the unified plan. Set by the backend-compatibility
	// retry when a backend rejects a quantized V cache for a model whose K leg
	// may stay compressed; persisted on the request so every later re-plan
	// (OOM recovery, measured-evidence recompute) keeps the override.
	KVQualityV        string
	KVTypeK           string // explicit llama.cpp --cache-type-k override
	KVTypeV           string // explicit llama.cpp --cache-type-v override
	CPUMode           bool
	GPUsFlag          string
	Host              string
	VisionAuto        bool
	MMProjPath        string
	ServerBin         string
	ServerBinExplicit bool
	AppHome           string
	Backend           string
	BackendExplicit   bool
	TuneCache         string
	SpecMode          string
	HotExperts        string // off, auto, on, or a positive cache-slot count per eligible layer
	HotExpertsSet     bool   // --hot-experts was supplied explicitly
	// HotExpertsRuntimeDisabled is set only after a requested cache failed its
	// post-load capability check. Re-plans in the same lifecycle must return to
	// the cache-free baseline instead of reapplying a stale measured winner.
	HotExpertsRuntimeDisabled bool
	// HotExpertPendingNegative carries deterministic cache-on failure evidence
	// until the restored cache-free baseline reaches StateActive. It is runtime
	// controller state only and never changes the requested launch identity.
	HotExpertPendingNegative *placement.CalibrationDecision
	ForceSpecMoE             bool
	RamBudgetMB              int
	RAMLimitPercent          int
	VRAMHeadroomMB           int
	RAMHeadroomMB            int
	// CgroupHeadroomMB is the headroom the post-launch measured-footprint sizing
	// keeps between the backend's measured non-reclaimable footprint and its hard
	// MemoryMax. 0 keeps the pre-launch plan-derived ceiling (auto re-size off).
	// -1 means unset; the config default applies.
	CgroupHeadroomMB     int
	AllowLiveMemoryProbe bool
	NoMMap               bool
	ForceMMap            bool
	Parallel             int
	ParallelSet          bool // --parallel given explicitly; claude-code mode must not override it
	Threads              int  // --threads; 0 keeps the physical-core default
	CacheRAMMB           int  // --cache-ram; 0 keeps the derived prompt-cache budget
	ClaudeMaxActive      int  // --claude-max-active; 0 means no admission limit
	ClaudeMaxActiveSet   bool
	BatchSize            int
	BatchSizeSet         bool
	UBatchSize           int
	UBatchSizeSet        bool
	Benchmark            bool
	WorkerBenchmark      bool // task-specific support/reviewer quality plus throughput
	ClaudeCode           bool
	// ClaudeReviewerOverride selects the local reviewer/worker model when
	// Claude Code mode starts its Auto companion: "auto" (and the historical
	// "qwen") resolve to the Qwen3.5-4B worker/reviewer, "qwen2b" forces the
	// small/light Qwen3.5-2B review-only profile, and "nanbeige" forces the
	// Nanbeige4.2 big-MoE worker. Empty means auto. Set via --claude-reviewer.
	ClaudeReviewerOverride string
	// ClaudeReviewerDisabled routes Auto reviews through the main model
	// (self-review) instead of seating a separate reviewer. It is set either
	// after the user accepts the resident fallback at launch, or up front by
	// --claude-reviewer off, which is the user asking for exactly that.
	ClaudeReviewerDisabled bool
	ClaudeProfile          string   // agent-interactive avoids the automatic parallel-4 floor
	ClaudeResume           string   // session id or "latest": reopen a recorded Claude session
	ClaudeResumeForce      bool     // accept a resume whose backend shape no longer matches
	OriginalArgs           []string // launch argv as given, so a resume can reproduce it exactly
	Calibrate              string   // "auto" (workflow-validated replay only), "on" (explicit bounded screen), "off"
	// CalibrationScreened marks a non-default configuration selected only by the
	// bounded screen. It may serve this explicit run, but must not leak into the
	// automatic verified-config or MoE placement caches.
	CalibrationScreened bool
	// CalibrationPending prevents verifyAndActivateLaunch from persisting a
	// tuned-looking verified config before the pending optimizer decision has
	// itself passed promotion and been written. Runtime-only.
	CalibrationPending bool
	// AppliedCalibration is the reusable cached decision that this launch or
	// dry-run actually replayed. It is inspect-only: argv still comes from the
	// re-derived winner strategy, never from deserialized flags.
	AppliedCalibration *placement.CalibrationDecision
	NoCachedConfig     bool   // --no-cached-config: derive placement fresh, ignoring cached measurements
	SupportExpert      string // off, auto (installed-only), on
	SupportOnline      bool   // typed official llama.cpp research only
	// SupportOnlineSet is true when the user named --support-online or
	// --no-support-online on the command line. The default (and the config) leave
	// it false, which lets an escalated launch force online research — the advisor
	// is at its best when it can cite official sources for a novel failure. An
	// explicit --no-support-online is a user instruction and is honored even on
	// escalation.
	SupportOnlineSet bool
	// ProfilePolicyIdentity is captured after the GGUF is parsed but before
	// backend/advisor recovery mutates the request. It groups alternative
	// measured argv under one last-known-good family while keeping genuinely
	// different user workload/quality policies isolated.
	ProfilePolicyIdentity string
	EmitServerArgvJSON    bool // dry-run machine interface for reproducible benchmark harnesses
	SpecDraftMax          int  // internal spec-test ceiling; not a public launch override
	ExtraArgs             []string
	// ChatTemplateOverride names a chat-template catalog entry (pkg/chattemplate)
	// the user explicitly selected with --chat-template. It forces that entry's
	// corrected template regardless of the model's arch/basename, and overrides
	// any automatic catalog match.
	ChatTemplateOverride string
	// DisabledBackendFlags records generated optimizations that this exact
	// model/backend pairing is known (or measured) not to support. It is launch
	// state rather than configuration: applying it after every argument rebuild
	// keeps placement and calibration retries from reintroducing a bad flag.
	DisabledBackendFlags map[string]string
	// DisabledBackendFlagValues marks rejected generated options that consume
	// the following argv token, so filtering cannot leave an orphan value.
	DisabledBackendFlagValues map[string]bool
	// ReviewerReservation holds the Claude Auto reviewer's placement companion
	// for the whole launch. placementOptionsFromRequest attaches it to every
	// Compute — including OOM/preflight/spec re-plans — so the reviewer's VRAM
	// stays reserved no matter which path recomputes the strategy.
	ReviewerReservation *placement.CompanionReservation
	// DenseOffloadPrompt, when set, is attached to every Compute via
	// placementOptionsFromRequest so placement can ASK the user before silently
	// offloading a dense model to system RAM. The launcher installs it once in
	// cmdLaunch; assumeYes / non-terminal callers leave it nil and keep the
	// historical silent host-offload behavior.
	DenseOffloadPrompt placement.DenseCPUOffloadPromptFunc
	// ReviewerProfile freezes the verified worker/model/backend choice before
	// placement so downloads or backend updates cannot change seats mid-launch.
	ReviewerProfile *claudeCompanionProfile
	// AdvisorVRAMPenaltyMB shrinks a device's usable VRAM for the next re-plan.
	// It is how a bounded move_expert_layer decision reaches the packer: the
	// advisor names a device and a layer count, and the deterministic planner
	// re-packs every GPU around the reduced budget. No model-produced value ever
	// becomes argv — only this integer, and only after ValidateDecision.
	AdvisorVRAMPenaltyMB map[int]int
	// BackendUnavailableReason carries why auto backend selection refused a
	// launch, so the nil-backend callers surface the real cause instead of the
	// generic "no llama-server binary found". Set when FAIL-CLOSED auto
	// selection rejects every candidate for a model architecture.
	BackendUnavailableReason string
}

const (
	claudeProfileInteractive = "agent-interactive"
	claudeProfileParallel    = "agent-parallel"
)

func parseLaunchArgs(args []string) (*launchRequest, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	backendExplicit := configuredBackendExplicit(cfg.Backend)
	originalArgs := append([]string(nil), args...)
	req := &launchRequest{
		Port:                 cfg.Port,
		CtxFlag:              cfg.CtxValue(),
		KVPlacement:          cfg.KVPlacement,
		KVQuality:            cfg.KVQuality,
		Host:                 cfg.Host,
		VisionAuto:           cfg.Vision,
		ServerBin:            cfg.LlamaServer,
		AppHome:              cfg.AppHome,
		Backend:              cfg.Backend,
		BackendExplicit:      backendExplicit,
		SpecMode:             cfg.Spec,
		HotExperts:           cfg.HotExperts,
		Parallel:             cfg.Parallel,
		RamBudgetMB:          parseBudgetMB(cfg.RamBudget),
		RAMLimitPercent:      cfg.RAMLimitPercent,
		VRAMHeadroomMB:       parseBudgetMB(cfg.VRAMHeadroom),
		RAMHeadroomMB:        parseBudgetMB(cfg.RAMHeadroom),
		CgroupHeadroomMB:     cfg.CgroupHeadroomMB,
		AllowLiveMemoryProbe: cfg.AllowLiveMemoryProbe,
		OriginalArgs:         originalArgs,
		SupportExpert:        cfg.SupportExpert,
		SupportOnline:        cfg.SupportOnline,
	}
	if cfg.SWAFull {
		req.ExtraArgs = append(req.ExtraArgs, "--swa-full")
	}
	if req.Port == 0 {
		req.Port = 8081
	}
	if req.KVPlacement == "" {
		req.KVPlacement = "auto"
	}
	if req.KVQuality == "" {
		req.KVQuality = "auto"
	}
	if req.Host == "" {
		req.Host = "127.0.0.1"
	}
	if req.SpecMode == "" {
		req.SpecMode = "off"
	}
	if req.HotExperts == "" {
		req.HotExperts = "auto"
	}

	for i := 0; i < len(args); i++ {
		a := args[i]
		if key, val, ok := strings.Cut(a, "="); ok && strings.HasPrefix(key, "-") {
			switch key {
			case "--allow-live-memory-probe":
				req.AllowLiveMemoryProbe = val == "" || parseBoolFlag(val)
				continue
			case "--support-expert":
				mode, err := config.NormalizeSupportExpert(val)
				if err != nil {
					return nil, fmt.Errorf("%s: %w", key, err)
				}
				req.SupportExpert = mode
				continue
			case "--hot-experts", "--moe-expert-cache":
				mode, err := config.NormalizeHotExperts(val)
				if err != nil {
					return nil, fmt.Errorf("%s: %w", key, err)
				}
				req.HotExperts, req.HotExpertsSet = mode, true
				continue
			case "--moe-expert-cache-inserts":
				if strings.TrimSpace(val) != "2" {
					return nil, fmt.Errorf("%s is managed by ggrun and currently fixed at 2 for one-dimensional calibration", key)
				}
				continue
			case "--support-online":
				req.SupportOnline = val == "" || parseBoolFlag(val)
				req.SupportOnlineSet = true
				continue
			case "--no-support-online":
				req.SupportOnline = !(val == "" || parseBoolFlag(val))
				req.SupportOnlineSet = true
				continue
			case "--model", "-m":
				req.ModelPath = val
				continue
			case "--port", "-port":
				port, err := config.ParsePort(val)
				if err != nil {
					return nil, fmt.Errorf("%s: %w", key, err)
				}
				req.Port = port
				continue
			case "--ctx", "-ctx", "--ctx-size", "-c":
				req.CtxFlag = val
				continue
			case "--kv", "-kv", "--kv-placement":
				req.KVPlacement = val
				continue
			case "--kv-quality", "-kv-quality":
				req.KVQuality = val
				continue
			case "--cache-type-k", "-ctk":
				req.KVTypeK = val
				continue
			case "--cache-type-v", "-ctv":
				req.KVTypeV = val
				continue
			case "--gpus", "-gpus":
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
				budget, err := parseBudgetFlag(key, val)
				if err != nil {
					return nil, err
				}
				req.RamBudgetMB = budget
				continue
			case "--ram-limit-percent":
				percent, err := config.ParseRAMLimitPercent(val)
				if err != nil {
					return nil, fmt.Errorf("%s: %w", key, err)
				}
				req.RAMLimitPercent = percent
				continue
			case "--vram-headroom":
				budget, err := parseBudgetFlag(key, val)
				if err != nil {
					return nil, err
				}
				req.VRAMHeadroomMB = budget
				continue
			case "--ram-headroom":
				budget, err := parseBudgetFlag(key, val)
				if err != nil {
					return nil, err
				}
				req.RAMHeadroomMB = budget
				continue
			case "--no-mmap":
				req.NoMMap = val == "" || parseBoolFlag(val)
				if req.NoMMap {
					req.ForceMMap = false
				}
				continue
			case "--swa-full":
				req.ExtraArgs = setPassthroughBoolFlag(req.ExtraArgs, "--swa-full", val == "" || parseBoolFlag(val))
				continue
			case "--no-swa-full":
				disabled := val == "" || parseBoolFlag(val)
				req.ExtraArgs = setPassthroughBoolFlag(req.ExtraArgs, "--swa-full", !disabled)
				continue
			case "--mmap":
				req.ForceMMap = val == "" || parseBoolFlag(val)
				if req.ForceMMap {
					req.NoMMap = false
				}
				continue
			case "--spec":
				req.SpecMode = val
				continue
			case "--parallel":
				parallel, err := parsePositiveFlag(key, val)
				if err != nil {
					return nil, err
				}
				req.Parallel = parallel
				req.ParallelSet = true
				continue
			case "--threads", "-t":
				threads, err := parsePositiveFlag(key, val)
				if err != nil {
					return nil, err
				}
				req.Threads = threads
				continue
			case "--cache-ram", "-cram":
				cram, err := parsePositiveFlag(key, val)
				if err != nil {
					return nil, err
				}
				req.CacheRAMMB = cram
				continue
			case "--claude-max-active":
				limit, err := parseNonNegativeFlag(key, val)
				if err != nil {
					return nil, err
				}
				req.ClaudeMaxActive = limit
				req.ClaudeMaxActiveSet = true
				continue
			case "--batch-size", "-b":
				batch, err := parsePositiveFlag(key, val)
				if err != nil {
					return nil, err
				}
				req.BatchSize, req.BatchSizeSet = batch, true
				continue
			case "--ubatch-size", "-ub":
				ubatch, err := parsePositiveFlag(key, val)
				if err != nil {
					return nil, err
				}
				req.UBatchSize, req.UBatchSizeSet = ubatch, true
				continue
			case "--claude-profile":
				profile, err := parseClaudeProfile(key, val)
				if err != nil {
					return nil, err
				}
				req.ClaudeProfile = profile
				continue
			case "--claude-reviewer":
				reviewer, err := parseClaudeReviewer(key, val)
				if err != nil {
					return nil, err
				}
				req.ClaudeReviewerOverride = reviewer
				continue
			case "--claude-resume":
				req.ClaudeResume, req.ClaudeCode = val, true
				continue
			case "--chat-template":
				// --chat-template doubles as llama.cpp's built-in template selector
				// (e.g. "--chat-template chatml"). When the value names a
				// chat-template catalog entry, it forces that corrected template;
				// otherwise it stays a passthrough backend flag like today.
				if entry, ok := chattemplate.ResolveOverride(val); ok {
					req.ChatTemplateOverride = entry.Name
					continue
				}
				req.ExtraArgs = append(req.ExtraArgs, a)
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
		case "--allow-live-memory-probe":
			req.AllowLiveMemoryProbe = true
			continue
		case "--support-online":
			req.SupportOnline = true
			req.SupportOnlineSet = true
			continue
		case "--no-support-online":
			req.SupportOnline = false
			req.SupportOnlineSet = true
			continue
		case "--support-expert":
			v, err := next()
			if err != nil {
				return nil, err
			}
			mode, err := config.NormalizeSupportExpert(v)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", a, err)
			}
			req.SupportExpert = mode
			continue
		case "--benchmark":
			req.Benchmark = true
			continue
		case "--worker-benchmark":
			req.WorkerBenchmark = true
			continue
		case "--calibrate":
			v, err := next()
			if err != nil {
				return nil, err
			}
			mode, err := parseCalibrateMode(v)
			if err != nil {
				return nil, err
			}
			req.Calibrate = mode
			continue
		case "--hot-experts", "--moe-expert-cache":
			v, err := next()
			if err != nil {
				return nil, err
			}
			mode, err := config.NormalizeHotExperts(v)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", a, err)
			}
			req.HotExperts, req.HotExpertsSet = mode, true
			continue
		case "--moe-expert-cache-inserts":
			v, err := next()
			if err != nil {
				return nil, err
			}
			if strings.TrimSpace(v) != "2" {
				return nil, fmt.Errorf("%s is managed by ggrun and currently fixed at 2 for one-dimensional calibration", a)
			}
			continue
		case "--dry-run", "--emit-server-argv-json", "--ai-tune", "--retune", "--download", "--show-configs", "--keep-alive":
			if a == "--emit-server-argv-json" {
				req.EmitServerArgvJSON = true
			}
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
			port, err := config.ParsePort(v)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", a, err)
			}
			req.Port = port
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
		case "--kv-quality", "-kv-quality":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.KVQuality = v
		case "--cache-type-k", "-ctk":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.KVTypeK = v
		case "--cache-type-v", "-ctv":
			v, err := next()
			if err != nil {
				return nil, err
			}
			req.KVTypeV = v
		case "--cpu", "-cpu":
			req.CPUMode = true
		case "--gpus", "-gpus":
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
		case "--vision", "-vision":
			req.VisionAuto = true
		case "--claude-code":
			req.ClaudeCode = true
		case "--claude-reviewer":
			v, err := next()
			if err != nil {
				return nil, err
			}
			reviewer, err := parseClaudeReviewer(a, v)
			if err != nil {
				return nil, err
			}
			req.ClaudeReviewerOverride = reviewer
		case "--chat-template":
			v, err := next()
			if err != nil {
				return nil, err
			}
			if entry, ok := chattemplate.ResolveOverride(v); ok {
				req.ChatTemplateOverride = entry.Name
				break
			}
			req.ExtraArgs = append(req.ExtraArgs, a, v)
		case "--claude-resume":
			v, err := next()
			if err != nil {
				return nil, err
			}
			// Resuming a Claude session is only meaningful in Claude Code mode.
			req.ClaudeResume, req.ClaudeCode = v, true
		case "--claude-resume-force":
			req.ClaudeResumeForce = true
		case "--no-cached-config":
			req.NoCachedConfig = true
		case "--claude-profile":
			v, err := next()
			if err != nil {
				return nil, err
			}
			profile, err := parseClaudeProfile(a, v)
			if err != nil {
				return nil, err
			}
			req.ClaudeProfile = profile
		case "--no-mmap":
			req.NoMMap = true
			req.ForceMMap = false
		case "--swa-full":
			req.ExtraArgs = setPassthroughBoolFlag(req.ExtraArgs, "--swa-full", true)
		case "--no-swa-full":
			req.ExtraArgs = setPassthroughBoolFlag(req.ExtraArgs, "--swa-full", false)
		case "--mmap":
			req.ForceMMap = true
			req.NoMMap = false
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
			budget, err := parseBudgetFlag(a, v)
			if err != nil {
				return nil, err
			}
			req.RamBudgetMB = budget
		case "--ram-limit-percent":
			v, err := next()
			if err != nil {
				return nil, err
			}
			percent, err := config.ParseRAMLimitPercent(v)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", a, err)
			}
			req.RAMLimitPercent = percent
		case "--vram-headroom":
			v, err := next()
			if err != nil {
				return nil, err
			}
			budget, err := parseBudgetFlag(a, v)
			if err != nil {
				return nil, err
			}
			req.VRAMHeadroomMB = budget
		case "--ram-headroom":
			v, err := next()
			if err != nil {
				return nil, err
			}
			budget, err := parseBudgetFlag(a, v)
			if err != nil {
				return nil, err
			}
			req.RAMHeadroomMB = budget
		case "--cgroup-headroom":
			v, err := next()
			if err != nil {
				return nil, err
			}
			n, err := strconv.Atoi(v)
			if err != nil || n < 0 {
				return nil, fmt.Errorf("%s: must be a non-negative MiB amount (0 disables the post-launch measured re-size)", a)
			}
			req.CgroupHeadroomMB = n
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
			parallel, err := parsePositiveFlag(a, v)
			if err != nil {
				return nil, err
			}
			req.Parallel = parallel
			req.ParallelSet = true
		case "--threads", "-t":
			v, err := next()
			if err != nil {
				return nil, err
			}
			threads, err := parsePositiveFlag(a, v)
			if err != nil {
				return nil, err
			}
			req.Threads = threads
		case "--cache-ram", "-cram":
			v, err := next()
			if err != nil {
				return nil, err
			}
			cram, err := parsePositiveFlag(a, v)
			if err != nil {
				return nil, err
			}
			req.CacheRAMMB = cram
		case "--claude-max-active":
			v, err := next()
			if err != nil {
				return nil, err
			}
			limit, err := parseNonNegativeFlag(a, v)
			if err != nil {
				return nil, err
			}
			req.ClaudeMaxActive = limit
			req.ClaudeMaxActiveSet = true
		case "--batch-size", "-b":
			v, err := next()
			if err != nil {
				return nil, err
			}
			batch, err := parsePositiveFlag(a, v)
			if err != nil {
				return nil, err
			}
			req.BatchSize, req.BatchSizeSet = batch, true
		case "--ubatch-size", "-ub":
			v, err := next()
			if err != nil {
				return nil, err
			}
			ubatch, err := parsePositiveFlag(a, v)
			if err != nil {
				return nil, err
			}
			req.UBatchSize, req.UBatchSizeSet = ubatch, true
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
	if _, err := parseGPUIndices(req.GPUsFlag); err != nil {
		return nil, fmt.Errorf("--gpus: %w", err)
	}
	if req.ClaudeProfile != "" && !req.ClaudeCode {
		return nil, fmt.Errorf("--claude-profile requires --claude-code")
	}
	// Every --claude-reviewer value is gated the same way. "auto" was once
	// exempt because it merely restated the default, but it is a real choice
	// now (it selects the Qwen3.5-4B companion), and letting it pass without
	// --claude-code would silently accept a flag that can never take effect.
	if req.ClaudeReviewerOverride != "" && !req.ClaudeCode {
		return nil, fmt.Errorf("--claude-reviewer requires --claude-code")
	}
	// "off" is a placement decision, not a model choice: seat no reviewer at
	// all. Setting it here means every downstream gate that already honours
	// ClaudeReviewerDisabled (the reservation, the proactive VRAM gate, the
	// reviewer launch) needs no new special case, and the router sees no
	// separate reviewer and self-classifies with its usual notice.
	if req.ClaudeReviewerOverride == claudeReviewerOff {
		req.ClaudeReviewerDisabled = true
	}
	if err := resolveKVCacheTypeFlags(req); err != nil {
		return nil, err
	}
	if req.BatchSizeSet && req.UBatchSizeSet && req.BatchSize < req.UBatchSize {
		return nil, fmt.Errorf("--batch-size (%d) must be at least --ubatch-size (%d)", req.BatchSize, req.UBatchSize)
	}
	for _, arg := range req.ExtraArgs {
		if arg == "--moe-expert-cache" || strings.HasPrefix(arg, "--moe-expert-cache=") ||
			arg == "--moe-expert-cache-inserts" || strings.HasPrefix(arg, "--moe-expert-cache-inserts=") {
			return nil, fmt.Errorf("raw %s after -- bypasses ggrun's VRAM ledger; use --hot-experts instead", strings.SplitN(arg, "=", 2)[0])
		}
	}
	req.ExtraArgs = normalizePlacementAwareExtraArgs(req, req.ExtraArgs)
	return req, nil
}

func parseClaudeProfile(flag, value string) (string, error) {
	profile := strings.ToLower(strings.TrimSpace(value))
	switch profile {
	case claudeProfileInteractive, claudeProfileParallel:
		return profile, nil
	default:
		return "", fmt.Errorf("%s must be %q or %q, got %q", flag, claudeProfileInteractive, claudeProfileParallel, value)
	}
}

const (
	claudeReviewerAuto     = "auto"
	claudeReviewerQwen     = "qwen"
	claudeReviewerQwen2B   = "qwen2b"
	claudeReviewerNanbeige = "nanbeige"
	// claudeReviewerOff seats no separate reviewer at all. It is the user
	// asking for self-review outright, which is why it is the one value that
	// sets ClaudeReviewerDisabled straight from the command line.
	claudeReviewerOff = "off"
)

func parseClaudeReviewer(flag, value string) (string, error) {
	reviewer := strings.ToLower(strings.TrimSpace(value))
	switch reviewer {
	case claudeReviewerAuto, claudeReviewerQwen, claudeReviewerQwen2B, claudeReviewerNanbeige, claudeReviewerOff:
		return reviewer, nil
	default:
		return "", fmt.Errorf("%s must be %q, %q, %q, %q or %q, got %q", flag, claudeReviewerAuto, claudeReviewerQwen, claudeReviewerQwen2B, claudeReviewerNanbeige, claudeReviewerOff, value)
	}
}

const (
	calibrateAuto = "auto"
	calibrateOn   = "on"
	calibrateOff  = "off"
)

func parseCalibrateMode(value string) (string, error) {
	mode := strings.ToLower(strings.TrimSpace(value))
	switch mode {
	case calibrateAuto, calibrateOn, calibrateOff:
		return mode, nil
	default:
		return "", fmt.Errorf("--calibrate must be %q, %q, or %q, got %q", calibrateAuto, calibrateOn, calibrateOff, value)
	}
}

// effectiveClaudeProfile turns the omitted profile into an explicit runtime
// policy. Keeping this separate from parsing makes the default and an explicit
// --claude-profile agent-parallel share placement/probe evidence: their
// scheduling behavior is identical.
func effectiveClaudeProfile(req *launchRequest) string {
	if req == nil || !req.ClaudeCode {
		return ""
	}
	if req.ClaudeProfile == claudeProfileInteractive {
		return claudeProfileInteractive
	}
	return claudeProfileParallel
}

// requestWorkloadProfile scopes placement/probe evidence to a versioned Claude
// scheduling profile. A non-empty value deliberately does not match legacy
// generic cache entries, which were never validation for an agent workload.
func requestWorkloadProfile(req *launchRequest, model *placement.ModelProfile) string {
	profile := effectiveClaudeProfile(req)
	if profile == "" && (req == nil || (!req.BatchSizeSet && !req.UBatchSizeSet)) {
		return ""
	}
	if profile == "" {
		profile = "explicit-batch"
	}
	return fmt.Sprintf("claude-%s-v4:%s", profile, requestSamplingProfile(req, model))
}

func requestWorkloadConcurrency(req *launchRequest) int {
	if req == nil {
		return 1
	}
	if req.ClaudeCode && effectiveClaudeProfile(req) == claudeProfileInteractive {
		if req.ParallelSet && req.Parallel > 0 {
			return min(8, req.Parallel)
		}
		return 1
	}
	if req.ClaudeCode && effectiveClaudeProfile(req) == claudeProfileParallel {
		if req.ClaudeMaxActive > 0 {
			return min(8, req.ClaudeMaxActive)
		}
		if req.Parallel > 0 {
			// LLM_PARALLEL is also the ordinary server default, which is one on a
			// fresh install. The agent-parallel profile itself declares at least
			// two independently runnable turns; treating that inherited one as the
			// workload demand made both the cost model and live A/B measure p1 and
			// therefore incapable of learning whether p2 reduced workflow time.
			// Larger configured values remain useful declared demand, while the
			// automatic slot count is still only a capacity ceiling.
			return min(8, max(2, req.Parallel))
		}
		return 2
	}
	if req.Parallel > 0 {
		return min(8, req.Parallel)
	}
	return 1
}

// evidenceBackendCacheTag gives placement/probe evidence an exact backend-build
// namespace. Backend tags identify a flag dialect, not a binary: a rebuilt
// mainline server or a fork under the same "llama" tag can have different graph
// allocation behavior and must never inherit old fit/OOM evidence.
func evidenceBackendCacheTag(be *backendInfo) string {
	tag := "llama"
	if be != nil && strings.TrimSpace(be.Tag) != "" {
		tag = strings.TrimSpace(be.Tag)
	}
	if be != nil && strings.TrimSpace(be.Identity) != "" {
		return tag + "@" + strings.TrimSpace(be.Identity)
	}
	return tag
}

func scopedProbeBackendTag(req *launchRequest, model *placement.ModelProfile, be *backendInfo) string {
	tag := placement.ScopedBackendCacheTag(evidenceBackendCacheTag(be), requestWorkloadProfile(req, model))
	hotSlots := 0
	if req != nil && !req.HotExpertsRuntimeDisabled {
		hotSlots, _ = strconv.Atoi(strings.TrimSpace(req.HotExperts))
	}
	return placement.ScopedBackendRuntimeFeatureTag(
		tag, req != nil && hasArg(req.ExtraArgs, "--swa-full"), hotSlots, 2,
	)
}

func scopedProbeBackendTagForStrategy(req *launchRequest, model *placement.ModelProfile, be *backendInfo, strategy *placement.Strategy) string {
	workload := requestWorkloadProfile(req, model)
	if strategy != nil {
		workload = placement.SpecWorkloadProfile(workload, strategy.Draft)
	}
	tag := placement.ScopedBackendCacheTag(evidenceBackendCacheTag(be), workload)
	swaFull := req != nil && hasArg(req.ExtraArgs, "--swa-full")
	if strategy != nil {
		swaFull = strategy.SWAFull
	}
	hotSlots, hotInserts := 0, 0
	if strategy != nil {
		hotSlots = strategy.HotExpertCacheSlots
		hotInserts = strategy.HotExpertCacheInserts
	}
	return placement.ScopedBackendRuntimeFeatureTag(tag, swaFull, hotSlots, hotInserts)
}

// resolveKVCacheTypeFlags turns llama.cpp's direct K/V flags into one planned
// cache type. ggrun currently owns K and V as a pair, which means it can size
// the cache, preserve the selected type through context fitting, and emit the
// flags exactly once. A mixed K/V pair remains an upstream-only setting until
// placement can estimate each side independently.
func resolveKVCacheTypeFlags(req *launchRequest) error {
	if req == nil {
		return nil
	}
	if req.KVTypeK != "" || req.KVTypeV != "" {
		if req.KVTypeK == "" || req.KVTypeV == "" {
			return fmt.Errorf("set both --cache-type-k and --cache-type-v, or use --kv-quality <type> for a matching K/V cache")
		}
		keyType, err := placement.NormalizeKVType(req.KVTypeK)
		if err != nil {
			return fmt.Errorf("--cache-type-k: %w", err)
		}
		valueType, err := placement.NormalizeKVType(req.KVTypeV)
		if err != nil {
			return fmt.Errorf("--cache-type-v: %w", err)
		}
		if keyType != valueType {
			return fmt.Errorf("mixed --cache-type-k/--cache-type-v values are not planned safely yet; use the same type for both or --kv-quality <type>")
		}
		req.KVQuality = keyType
	}
	if _, err := placement.NormalizeKVType(req.KVQuality); err != nil {
		return fmt.Errorf("--kv-quality: %w", err)
	}
	return nil
}

func parsePositiveFlag(name, value string) (int, error) {
	n, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || n < 1 {
		return 0, fmt.Errorf("%s: must be a positive integer", name)
	}
	return n, nil
}

// parseNonNegativeFlag accepts zero, which several knobs use to mean "no limit"
// rather than "off".
func parseNonNegativeFlag(name, value string) (int, error) {
	n, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || n < 0 {
		return 0, fmt.Errorf("%s: must be a non-negative integer", name)
	}
	return n, nil
}

func parseBudgetFlag(name, value string) (int, error) {
	mb, err := config.ParseBudgetMBStrict(value)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", name, err)
	}
	return mb, nil
}

func parseBoolFlag(raw string) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

// setPassthroughBoolFlag keeps a single canonical positive backend flag. The
// CLI accepts positive and negative overrides, but llama-server only sees the
// positive form when enabled.
func setPassthroughBoolFlag(args []string, flag string, enabled bool) []string {
	out := make([]string, 0, len(args)+1)
	for _, arg := range args {
		if arg == flag || strings.HasPrefix(arg, flag+"=") {
			continue
		}
		out = append(out, arg)
	}
	if enabled {
		out = append(out, flag)
	}
	return out
}

func normalizePlacementAwareExtraArgs(req *launchRequest, args []string) []string {
	if req == nil || len(args) == 0 {
		return args
	}
	out := args[:0]
	swaSet := false
	swaFull := false
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--swa-full" {
			swaSet, swaFull = true, true
			continue
		}
		if key, val, ok := strings.Cut(a, "="); ok && key == "--swa-full" {
			swaSet, swaFull = true, val == "" || parseBoolFlag(val)
			continue
		}
		if a == "--no-swa-full" {
			swaSet, swaFull = true, false
			continue
		}
		if key, val, ok := strings.Cut(a, "="); ok && key == "--no-swa-full" {
			swaSet, swaFull = true, !(val == "" || parseBoolFlag(val))
			continue
		}
		if a == "--no-mmap" {
			req.NoMMap = true
			req.ForceMMap = false
			continue
		}
		if key, val, ok := strings.Cut(a, "="); ok && key == "--no-mmap" {
			req.NoMMap = val == "" || parseBoolFlag(val)
			continue
		}
		if a == "--mmap" {
			req.ForceMMap = true
			req.NoMMap = false
			continue
		}
		if key, val, ok := strings.Cut(a, "="); ok && key == "--mmap" {
			req.ForceMMap = val == "" || parseBoolFlag(val)
			if req.ForceMMap {
				req.NoMMap = false
			}
			continue
		}
		out = append(out, a)
	}
	if swaSet && swaFull {
		out = append(out, "--swa-full")
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
	indices, err := parseGPUIndices(req.GPUsFlag)
	if err != nil {
		return ""
	}
	if len(indices) == 0 {
		return ""
	}
	// Keep PCI ordering so renumbered placement indices line up with the
	// backend's enumeration of the visible subset.
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

// parseGPUIndices is shared by parsing, placement and visibility setup so an
// invalid token can never be converted by strconv.Atoi's zero value into GPU 0.
func parseGPUIndices(raw string) ([]int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	seen := map[int]bool{}
	indices := make([]int, 0, strings.Count(raw, ",")+1)
	for _, token := range strings.Split(raw, ",") {
		token = strings.TrimSpace(token)
		idx, err := strconv.Atoi(token)
		if err != nil || idx < 0 {
			return nil, fmt.Errorf("%q is not a non-negative GPU index", token)
		}
		if seen[idx] {
			return nil, fmt.Errorf("GPU %d is listed more than once", idx)
		}
		seen[idx] = true
		indices = append(indices, idx)
	}
	sort.Ints(indices)
	return indices, nil
}

// validateRequestedGPUs rejects a --gpus selection naming hardware this machine
// does not have. parseGPUIndices only checks the syntax; existence can only be
// checked once the hardware is known, and it has to be an error rather than a
// silent narrowing -- reserving a GPU for other work is exactly the case where
// quietly using it anyway is worst.
func validateRequestedGPUs(caps *detect.Capabilities, req *launchRequest) error {
	if req == nil || strings.TrimSpace(req.GPUsFlag) == "" {
		return nil
	}
	requested, err := parseGPUIndices(req.GPUsFlag)
	if err != nil {
		return fmt.Errorf("--gpus: %w", err)
	}
	if len(requested) == 0 {
		return nil
	}
	available := map[int]bool{}
	present := []string{}
	if caps != nil {
		for _, gpu := range caps.GPUs {
			available[gpu.Index] = true
			present = append(present, fmt.Sprintf("%d (%s)", gpu.Index, gpu.Name))
		}
	}
	missing := []string{}
	for _, idx := range requested {
		if !available[idx] {
			missing = append(missing, strconv.Itoa(idx))
		}
	}
	if len(missing) == 0 {
		return nil
	}
	if len(present) == 0 {
		return fmt.Errorf("--gpus %s: no GPUs were detected on this machine", req.GPUsFlag)
	}
	return fmt.Errorf("--gpus %s: no GPU with index %s; this machine has %s",
		req.GPUsFlag, strings.Join(missing, ", "), strings.Join(present, ", "))
}

// runtimeGPUCapabilities mirrors the device renumbering performed by
// CUDA_VISIBLE_DEVICES/GGML_VK_VISIBLE_DEVICES. Placement.Compute accepts the
// physical --gpus indices and restricts internally, but launch-time preflight,
// probe recording, and OOM recovery observe the backend's visible CUDA indices.
// Keeping this mapping explicit prevents a visible CUDA0 (for --gpus 2) from
// being charged against physical GPU0's memory budget.
func runtimeGPUCapabilities(caps *detect.Capabilities, req *launchRequest) (*detect.Capabilities, map[int]int) {
	visibleToPhysical := map[int]int{}
	if caps == nil {
		return caps, visibleToPhysical
	}
	if req == nil || strings.TrimSpace(req.GPUsFlag) == "" {
		for _, gpu := range caps.GPUs {
			visibleToPhysical[gpu.Index] = gpu.Index
		}
		return caps, visibleToPhysical
	}

	available := map[int]detect.GPU{}
	for _, gpu := range caps.GPUs {
		available[gpu.Index] = gpu
	}
	requested, err := parseGPUIndices(req.GPUsFlag)
	if err != nil {
		return caps, visibleToPhysical
	}
	physical := []int{}
	for _, idx := range requested {
		if _, ok := available[idx]; !ok {
			continue
		}
		physical = append(physical, idx)
	}
	if len(physical) == 0 {
		return caps, visibleToPhysical
	}

	filtered := *caps
	filtered.GPUs = make([]detect.GPU, 0, len(physical))
	for visible, idx := range physical {
		gpu := available[idx]
		gpu.Index = visible
		filtered.GPUs = append(filtered.GPUs, gpu)
		visibleToPhysical[visible] = idx
	}
	return &filtered, visibleToPhysical
}

// runtimeVRAMUsedMB is indirected so the launch admission boundary can be
// tested without consulting the developer machine's GPUs.
var runtimeVRAMUsedMB = placement.QueryVRAMUsed

// runtimeGPUCapabilitiesForLaunch returns the capacity the main backend can use
// after a separately launched companion has occupied its planned seat. The
// original hardware snapshot predates companion startup; using it directly in
// backend preflight can approve an allocation that no longer fits. Keep the
// stricter of the deterministic reservation and a fresh whole-device reading,
// which also catches unrelated workloads that appeared after placement.
func runtimeGPUCapabilitiesForLaunch(caps *detect.Capabilities, req *launchRequest, strategy *placement.Strategy) (*detect.Capabilities, map[int]int) {
	runtimeCaps, visibleToPhysical := runtimeGPUCapabilities(caps, req)
	if runtimeCaps == nil {
		return nil, visibleToPhysical
	}
	adjusted := *runtimeCaps
	adjusted.GPUs = append([]detect.GPU(nil), runtimeCaps.GPUs...)

	plannedByPhysical := map[int]int{}
	if req != nil && req.ReviewerReservation != nil && strategy != nil {
		for _, companion := range strategy.CompanionPlacements {
			if companion.Name == req.ReviewerReservation.Name && companion.GPU >= 0 {
				plannedByPhysical[companion.GPU] += req.ReviewerReservation.VRAMMB
			}
		}
	}

	for i := range adjusted.GPUs {
		physical := physicalGPUIndex(adjusted.GPUs[i].Index, visibleToPhysical)
		usedFloor := adjusted.GPUs[i].VRAMUsedMB + plannedByPhysical[physical]
		if liveUsed := runtimeVRAMUsedMB(physical); liveUsed > usedFloor {
			usedFloor = liveUsed
		}
		adjusted.GPUs[i].VRAMUsedMB = usedFloor
	}
	return &adjusted, visibleToPhysical
}

func physicalGPUIndex(visible int, visibleToPhysical map[int]int) int {
	if physical, ok := visibleToPhysical[visible]; ok {
		return physical
	}
	return visible
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
	// "skip" is an installer-only choice used by older launcher-only app homes;
	// it never named a runnable backend. Treat those persisted configs as auto,
	// while a literal CLI --backend skip remains explicit and fails clearly.
	return backend != "" && !strings.EqualFold(backend, "auto") && !strings.EqualFold(backend, "skip")
}

func requestedBackendName(req *launchRequest) string {
	if req == nil {
		return ""
	}
	want := strings.TrimSpace(req.Backend)
	if !req.BackendExplicit && strings.EqualFold(want, "skip") {
		return "auto"
	}
	return want
}

func selectBackend(caps *detect.Capabilities, req *launchRequest) *backendInfo {
	want := requestedBackendName(req)
	// Only --server-bin is an unconditional binary selection. LLAMA_SERVER from
	// config is an auto-selection candidate: treating it as authoritative made a
	// stale ~/.local/bin symlink beat the canonical app-home backend even after
	// the GGUF had identified an architecture that another installed build
	// supported better.
	useExplicitServerBin := req.ServerBin != "" && req.ServerBinExplicit
	if useExplicitServerBin {
		if _, err := os.Stat(req.ServerBin); err == nil {
			return detectBackend(req.ServerBin)
		}
		fmt.Fprintf(os.Stderr, "Warning: server binary not found: %s\n", req.ServerBin)
	}
	if want != "" && want != "auto" {
		seen := make(map[string]bool)
		try := func(path, name string) *backendInfo {
			path = strings.TrimSpace(path)
			if path == "" || seen[path] {
				return nil
			}
			seen[path] = true
			if _, err := os.Stat(path); err != nil {
				return nil
			}
			info := detectBackend(path)
			if backendMatches(info, name, want) {
				return info
			}
			return nil
		}
		// A configured LLAMA_SERVER may live outside every standard app-home
		// path. It is eligible only when it actually matches the named backend;
		// a stale mainline path must never override an explicit ik_llama choice.
		if !useExplicitServerBin {
			if info := try(req.ServerBin, filepath.Base(req.ServerBin)); info != nil {
				return info
			}
		}
		// A registered fork backend selected by its manifest tag (--backend <tag>).
		if cb := backends.ByTag(want); cb != nil {
			if _, err := os.Stat(cb.Path); err == nil {
				return detectRegisteredBackend(cb)
			}
			fmt.Fprintf(os.Stderr, "Warning: registered backend %q binary not found: %s\n", cb.Tag, cb.Path)
		}
		// A self-contained APP_HOME is the user's chosen installation. Prefer its
		// matching backend over a same-dialect binary discovered globally (for
		// example ~/.local/bin/llama-server pointing at another IK checkout).
		for _, p := range backendSearchPaths(req.AppHome) {
			if info := try(p, filepath.Base(p)); info != nil {
				return info
			}
		}
		if caps != nil {
			for _, b := range caps.Backends {
				if info := try(b.Path, b.Name); info != nil {
					return info
				}
			}
		}
		// A named backend is an explicit compatibility requirement. Falling back
		// to some other binary makes the TUI claim it honored the selection and
		// then produces a misleading architecture error later.
		return nil
	}
	if req.ServerBin != "" && !useExplicitServerBin {
		if _, err := os.Stat(req.ServerBin); err == nil {
			if info := detectUsableBackend(req.ServerBin); info != nil {
				return info
			}
			fmt.Fprintf(os.Stderr, "Warning: configured backend %s did not start; trying another installed backend\n", req.ServerBin)
		} else {
			fmt.Fprintf(os.Stderr, "Warning: server binary not found: %s\n", req.ServerBin)
		}
	}
	return findBackend(caps, req.AppHome)
}

type autoBackendCandidate struct {
	info         *backendInfo
	canonical    bool
	incompatible bool
}

// autoBackendCandidates enumerates every generic backend available to this
// installation. Canonical means the path itself lives under APP_HOME; a
// symlink there remains canonical even when its build tree is stored elsewhere.
// The distinction is a tie-breaker only: proven architecture support always
// beats installation locality.
func autoBackendCandidates(caps *detect.Capabilities, req *launchRequest) []autoBackendCandidate {
	if req == nil {
		return nil
	}
	appHome := strings.TrimSpace(req.AppHome)
	if appHome == "" {
		appHome = backends.AppHome()
	}
	seen := make(map[string]bool)
	var out []autoBackendCandidate
	add := func(path string) {
		path = strings.TrimSpace(path)
		if path == "" || seen[path] {
			return
		}
		seen[path] = true
		if st, err := os.Stat(path); err != nil || st.IsDir() {
			return
		}
		info := detectBackend(path)
		if backendLoaderFailed(info.Help) {
			return
		}
		out = append(out, autoBackendCandidate{
			info:      info,
			canonical: pathInsideDir(path, appHome),
		})
	}

	// APP_HOME paths come first for deterministic tie-breaking. ServerBin and
	// detection still participate, but neither can short-circuit the comparison.
	for _, path := range backendSearchPaths(appHome) {
		add(path)
	}
	add(req.ServerBin)
	if caps != nil {
		for _, backend := range caps.Backends {
			add(backend.Path)
		}
	}
	return out
}

func pathInsideDir(path, dir string) bool {
	if strings.TrimSpace(path) == "" || strings.TrimSpace(dir) == "" {
		return false
	}
	absPath, errPath := filepath.Abs(filepath.Clean(path))
	absDir, errDir := filepath.Abs(filepath.Clean(dir))
	if errPath != nil || errDir != nil {
		return false
	}
	rel, err := filepath.Rel(absDir, absPath)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator))
}

type backendArchProbe func(binaryPath, arch string) (supported, probed bool)

// autoBackendLargeMoEMinMB is the smallest MoE model for which the file-backed
// CPU-expert tie-break in chooseAutoBackend engages. Below this an MoE either
// fits on GPU or its CPU-expert footprint is small enough that anonymous versus
// file-backed allocation does not decide survival under the cgroup.
const autoBackendLargeMoEMinMB = 48 * 1024 // 48 GiB

// backendCPUExpertMMapCapability returns the explicit capability attached to
// the exact probed backend. The fallback exists for old synthetic callers and
// tests; detectBackend always populates the tri-state field, including an
// explicit unknown, so production never authorizes reclaim from a tag alone.
func backendCPUExpertMMapCapability(be *backendInfo) placement.CPUExpertMMapCapability {
	if be == nil {
		return placement.CPUExpertMMapUnknown
	}
	if be.CPUExpertMMapCapability != "" {
		return placement.NormalizeCPUExpertMMapCapability(string(be.CPUExpertMMapCapability))
	}
	if placement.BackendUsesAnonymousCPUExperts(backendDialect(be)) || be.IsIK {
		return placement.CPUExpertMMapAnonymous
	}
	return placement.CPUExpertMMapFileBacked
}

// cpuExpertsFileBacked reports whether a backend leaves CPU-offloaded expert
// tensors file-backed (reclaimable) rather than in anonymous CUDA-host memory.
func cpuExpertsFileBacked(be *backendInfo) bool {
	return backendCPUExpertMMapCapability(be) == placement.CPUExpertMMapFileBacked
}

// largeCPUMoEPrefersFileBacked reports whether this model is a large-CPU-expert
// MoE for which the backend's memory behavior decides survival: a ~94GB
// CPU-expert DeepSeek-V4 launch under ik_llama (anonymous CUDA-host experts,
// unpagable under cgroup pressure) was OOM-killed, while mainline's file-backed
// experts survive via the mmap reclaim band.
func largeCPUMoEPrefersFileBacked(model *placement.ModelProfile) bool {
	return model != nil && model.IsMoE && model.TotalSizeMB >= autoBackendLargeMoEMinMB
}

// chooseAutoBackend ranks candidates using facts from the parsed GGUF. A
// proven-supporting backend wins over an unprobeable one, which wins over a
// proven-unsupported backend. Within the same support class the memory behavior
// of the loader wins for large-CPU-expert MoE models — file-backed experts stay
// reclaimable under cgroup pressure, anonymous CUDA-host ones have OOM-killed
// launches — then the canonical production path wins, then discovery order keeps
// the result stable.
//
// Selection FAILS CLOSED: a candidate whose loader the probe PROVED lacks the
// model's architecture (supported=false, probed=true) is scored 0 and is never
// launchable-by-default. When every viable candidate lands at 0 for that reason,
// the second return names the best one and the first is nil, so the caller can
// refuse loudly instead of letting the canonical tie-break launch a backend that
// dies with "unknown model architecture" at model load. An unprobeable candidate
// (probed=false) keeps support 1 and stays launchable: refusing on a failed
// probe would be worse than the loader error it replaces.
func chooseAutoBackend(candidates []autoBackendCandidate, arch string, probe backendArchProbe, model *placement.ModelProfile) (*backendInfo, *backendInfo) {
	arch = strings.ToLower(strings.TrimSpace(arch))
	required := backends.RequiredBackendForArch(arch)
	recipeBacked := len(backends.RecipesForArch(arch)) > 0
	tieBreakFileBacked := largeCPUMoEPrefersFileBacked(model)
	bestIndex, bestSupport, bestCanonical, bestFileBacked := -1, -1, false, false
	bestProbeUnsupported := false
	for i, candidate := range candidates {
		if candidate.info == nil {
			continue
		}
		support := 1 // unknown/unprobeable remains launchable
		probeUnsupported := false
		if candidate.incompatible {
			support = -1
		} else if required == "ik_llama" && !candidate.info.IsIK {
			support = 0
		} else if arch != "" && probe != nil {
			if supported, probed := probe(candidate.info.Path, arch); probed {
				if supported {
					// Generic ik binaries can contain architecture names only as
					// tokenizer/template literals (observed with Laguna). A reviewed
					// registered route is handled before this generic ranking; do not
					// treat an unregistered IK string hit as loader conformance.
					if candidate.info.IsIK && recipeBacked {
						support = 1
					} else {
						support = 2
					}
				} else {
					// The probe read the backend's loader and the architecture
					// literal is absent: this backend cannot load the model.
					support = 0
					probeUnsupported = true
				}
			}
		}
		candFileBacked := !candidate.incompatible && cpuExpertsFileBacked(candidate.info)
		if bestIndex < 0 || support > bestSupport {
			bestIndex, bestSupport, bestCanonical, bestFileBacked = i, support, candidate.canonical, candFileBacked
			bestProbeUnsupported = probeUnsupported
			continue
		}
		if support != bestSupport {
			continue
		}
		// Same support class. For a large-CPU-expert MoE the loader's memory
		// behavior outranks canonical locality: file-backed experts stay
		// reclaimable via the mmap reclaim band, while anonymous CUDA-host
		// experts cannot be paged out and have OOM-killed launches. When the
		// tie-break is not engaged, or both sides agree, fall through to the
		// existing canonical-path preference.
		if tieBreakFileBacked {
			if candFileBacked && !bestFileBacked {
				bestIndex, bestCanonical, bestFileBacked = i, candidate.canonical, true
				bestProbeUnsupported = probeUnsupported
				continue
			}
			if !candFileBacked && bestFileBacked {
				continue
			}
		}
		if candidate.canonical && !bestCanonical {
			bestIndex, bestCanonical = i, true
			bestProbeUnsupported = probeUnsupported
		}
	}
	if bestIndex < 0 {
		return nil, nil
	}
	// FAIL-CLOSED: never launch a backend the probe proved cannot serve this
	// architecture. An all-unsupported set must refuse loudly instead of letting
	// the canonical tie-break pick a backend that dies at model load. The
	// mandated-ik branch (required == "ik_llama" && !IsIK) is deliberately not
	// covered here: that case is resolved later by preflightIKOnlyArch with a
	// more specific fix instruction.
	if arch != "" && bestSupport < 1 && bestProbeUnsupported {
		return nil, candidates[bestIndex].info
	}
	return candidates[bestIndex].info, nil
}

func reviewedRecipeRequiredForMain(arch string, be *backendInfo) *backends.Recipe {
	recipes := backends.RecipesForArch(arch)
	if len(recipes) == 0 {
		return nil
	}
	if be != nil && be.Path != "" {
		supported, probed := backends.BackendSupportsArch(be.Path, arch)
		// Being an ik build is not evidence of support on its own: the fork
		// carries architectures mainline lacks and lacks some mainline has, so a
		// reviewed recipe normally wins over it. But when ik is the family this
		// architecture *requires*, refusing it for being ik is unsatisfiable --
		// ggrun resolves the backend it mandated and then rejects it. A probed
		// arch literal is the same proof there as it is for mainline.
		//
		// minimax-m3 hit exactly that: RequiredBackendForArch returns "ik_llama",
		// the installed ik libllama.so carries the "minimax-m3" literal, and the
		// launch still failed with "no proven main-model backend".
		requiredIsIK := strings.EqualFold(backends.RequiredBackendForArch(arch), "ik_llama")
		if probed && supported && (!be.IsIK || requiredIsIK) {
			return nil
		}
		if !probed && !be.IsIK {
			// Static probing is intentionally conservative on platforms whose
			// architecture table lives in a DLL. Preserve the explicit warning
			// path rather than forcing a fork over an unknown canonical backend.
			return nil
		}
	}
	return &recipes[0]
}

func confirmReviewedBackendInstall(recipe *backends.Recipe, arch string, assumeYes bool, in io.Reader, out io.Writer, terminal bool) bool {
	if recipe == nil {
		return false
	}
	if assumeYes {
		return true
	}
	if !terminal {
		return false
	}
	fmt.Fprintf(out, "Model architecture %q needs reviewed backend %q (%s). Install/build it now? [y/N] ", arch, recipe.Name, recipe.Description)
	line, _ := bufio.NewReader(in).ReadString('\n')
	answer := strings.ToLower(strings.TrimSpace(line))
	return answer == "y" || answer == "yes"
}

// confirmDenseCPUOffloadWith returns the placement.DenseCPUOffloadPrompt hook
// the launcher installs so placement can ASK before silently offloading a dense
// model to system RAM. When even a minimum context cannot keep the model on
// GPU, placement invokes this with the model's total size and the GPUs' free
// VRAM in MB; it returns whether to accept host offload. assumeYes and
// non-terminal callers preserve today's silent behavior (accept), matching
// confirmReviewedBackendInstall.
//
// The reader/writer/terminal shape mirrors confirmReviewedBackendInstall so the
// decision can be unit-tested without a real terminal.
func confirmDenseCPUOffloadWith(assumeYes bool, in io.Reader, out io.Writer, terminal bool) placement.DenseCPUOffloadPromptFunc {
	return func(totalSizeMB, freeGPUVRAMMB int) bool {
		if assumeYes || !terminal {
			return true
		}
		fmt.Fprintf(out, "The dense model needs %d MB but the GPUs have only %d MB free even at minimum context. Reduce context to fit on GPU, or accept offloading to system RAM? [y/N] ", totalSizeMB, freeGPUVRAMMB)
		line, _ := bufio.NewReader(in).ReadString('\n')
		answer := strings.ToLower(strings.TrimSpace(line))
		return answer == "y" || answer == "yes"
	}
}

// confirmDenseCPUOffload returns the DenseCPUOffloadPrompt hook wired to
// os.Stdin/os.Stderr and the real terminal state.
func confirmDenseCPUOffload(assumeYes bool) placement.DenseCPUOffloadPromptFunc {
	return confirmDenseCPUOffloadWith(assumeYes, os.Stdin, os.Stderr, stdinIsTerminal())
}

func selectBackendForModel(caps *detect.Capabilities, req *launchRequest, model *placement.ModelProfile) *backendInfo {
	if req == nil {
		return nil
	}
	// Explicit CLI choices and named configured backends are compatibility
	// contracts, not hints. Auto-routing must never replace them.
	if req.ServerBinExplicit || req.BackendExplicit {
		return selectBackend(caps, req)
	}
	arch := ""
	if model != nil {
		arch = model.ModelArch
		// A registered route is more specific than any generic architecture
		// probe. It represents a reviewed or user-installed fork for this exact
		// GGUF architecture.
		if routed := backends.ForArch(arch); routed != nil {
			fmt.Printf("[launch] %s runs on fork backend %q — routing to %s\n", arch, routed.Tag, routed.Path)
			return detectRegisteredBackend(routed)
		}
		// ForArch skips helper-only forks: they retain architecture metadata
		// without globally routing main-model launches. But a model whose arch
		// is served ONLY by a helper-only fork (Nanbeige's "nanbeige" arch is
		// the one case) has no canonical backend that can load it — mainline
		// aborts with "unknown model architecture". When the helper fork is the
		// sole registered backend for this exact arch, routing to it is the only
		// way the model can run at all, so prefer it over mainline.
		if soleHelper := backends.SoleHelperForArch(arch); soleHelper != nil {
			fmt.Printf("[launch] %s runs on sole helper backend %q — routing to %s\n", arch, soleHelper.Tag, soleHelper.Path)
			return detectRegisteredBackend(soleHelper)
		}
	}
	// Backend ranking happens before the chosen backend's feature normalization.
	// DeepSeek4's best compressed common denominator is q8_0: ik supports it,
	// while mainline families are then correctly rejected by their stricter f16
	// rule. Leaving q4 here marks the valid ik candidate incompatible and can
	// accidentally route a CUDA launch to an unrelated canonical backend.
	normalizeArchKVRequest(req, model)
	candidates := autoBackendCandidates(caps, req)
	if model != nil {
		for i := range candidates {
			if candidates[i].info == nil {
				continue
			}
			_, err := placement.ResolveKVQuality(model, req.KVQuality, backendDialect(candidates[i].info))
			candidates[i].incompatible = err != nil
		}
	}
	chosen, unsupported := chooseAutoBackend(candidates, arch, backends.BackendSupportsArch, model)
	if chosen == nil {
		// FAIL-CLOSED: every candidate's loader was probed and lacks this
		// architecture. Launching any of them would die at model load with
		// "unknown model architecture"; refuse with the fix instead. The reason
		// rides the request so every nil-backend call site (launch, dry-run,
		// tune, kv-probe, daemon) surfaces it instead of "no llama-server found".
		if unsupported != nil && arch != "" {
			req.BackendUnavailableReason = backendUnavailableReason(arch, unsupported.Path)
		}
		return nil
	}
	hasCanonical, chosenCanonical := false, false
	for _, candidate := range candidates {
		if candidate.canonical {
			hasCanonical = true
			if candidate.info != nil && candidate.info.Path == chosen.Path {
				chosenCanonical = true
			}
		}
	}
	if hasCanonical && !chosenCanonical && arch != "" {
		fmt.Printf("[launch] auto selected %s outside APP_HOME because canonical backends cannot satisfy architecture/profile %s with KV %s.\n",
			chosen.Path, arch, req.KVQuality)
	}
	return chosen
}

// normalizeArchKVRequest applies the architecture's KV correctness rule before a
// backend is even chosen, so selection and placement both size the cache the
// model actually needs. The rule comes from the GGUF architecture rather than
// from what any backend says it accepts -- see pkg/backends/archconstraints.go
// for why those are different questions.
func normalizeArchKVRequest(req *launchRequest, model *placement.ModelProfile) {
	if req == nil || model == nil {
		return
	}
	rule, ok := backends.KVRuleForArch(model.ModelArch)
	if !ok {
		return
	}
	kvType, err := placement.NormalizeKVType(req.KVQuality)
	if err != nil || rule.Permits(kvType) {
		return
	}
	fmt.Printf("[launch] %s needs a %s K-cache: %s. Using %s for backend selection and placement.\n",
		model.ModelArch, rule.Target(), rule.Reason, rule.Target())
	req.KVQuality = rule.Target()
	req.KVTypeK, req.KVTypeV = "", ""
}

// backendUnavailableReason names the actionable fix for a FAIL-CLOSED auto
// backend refusal. A reviewed recipe for the architecture means the user can
// install exactly that fork; a novel architecture with no reviewed recipe has no
// such install line, so the actionable path is advancing the mainline llama.cpp
// checkout (which upstream tracks new architectures into before forks diverge)
// or installing any fork that adds the architecture.
func backendUnavailableReason(arch, backendPath string) string {
	actionable := fmt.Sprintf("Update the mainline llama.cpp backend or install a fork that adds %s (ggrun backend install <recipe>).", arch)
	if len(backends.RecipesForArch(arch)) == 0 {
		actionable = "It requires a newer llama.cpp mainline or a fork that adds the architecture. ggrun can search open llama.cpp PRs for a supporting fork, or update the mainline backend."
	}
	return fmt.Sprintf(
		"No installed backend supports the %s architecture. The %s backend does not support it.\n  %s",
		arch, backendPath, actionable)
}

// offerMainlineBackendUpdate asks the user whether to advance the mainline
// llama.cpp backend when the FAIL-CLOSED refusal is an unsupported architecture
// with NO reviewed recipe (a novel arch): there is no fork install line to
// offer, so the actionable path is updating mainline. It returns true only when
// the user accepts and the update ran; a declined or non-terminal call returns
// false so the caller keeps the dead-end error path.
func offerMainlineBackendUpdate(req *launchRequest, model *placement.ModelProfile, assumeYes bool) bool {
	return offerMainlineBackendUpdateWith(req, model, assumeYes, os.Stdin, os.Stderr, stdinIsTerminal(), updateMainlineBackend)
}

// offerMainlineBackendUpdateWith is the injectable core: the confirmation uses
// the same reader/writer/terminal shape as confirmReviewedBackendInstall, and
// the update is a seam so tests can observe the decision without rebuilding a
// llama.cpp checkout.
func offerMainlineBackendUpdateWith(req *launchRequest, model *placement.ModelProfile, assumeYes bool, in io.Reader, out io.Writer, terminal bool, update func() error) bool {
	if req == nil || model == nil {
		return false
	}
	arch := strings.TrimSpace(model.ModelArch)
	if arch == "" {
		return false
	}
	// Only a FAIL-CLOSED unsupported-architecture refusal is eligible, and only
	// when no reviewed recipe exists to offer instead. A recipe-backed arch is
	// handled by the reviewedRecipeRequiredForMain flow before this point.
	if strings.TrimSpace(req.BackendUnavailableReason) == "" || len(backends.RecipesForArch(arch)) > 0 {
		return false
	}
	if assumeYes {
		return update() == nil
	}
	if !terminal {
		return false
	}
	fmt.Fprintf(out, "No installed backend can load this model architecture. Update the mainline llama.cpp backend to add support for %s? [y/N] ", arch)
	line, _ := bufio.NewReader(in).ReadString('\n')
	answer := strings.ToLower(strings.TrimSpace(line))
	if answer != "y" && answer != "yes" {
		return false
	}
	return update() == nil
}

// searchArchForkPRs is the live GitHub + Hugging Face lookup for an
// architecture with no reviewed recipe. Tests replace it so launch prompting
// does not hit the network.
var searchArchForkPRs = func(ctx context.Context, model *placement.ModelProfile) ([]backends.ArchForkPR, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	query := backends.ArchForkSearch{}
	if model != nil {
		query.Arch = model.ModelArch
		query.QuantizedBy = model.QuantizedBy
		query.Name = model.Name
		query.Basename = model.Basename
	}
	return backends.SearchArchForks(ctx, query, backends.ForkSearchClient{})
}

func installDiscoveredArchFork(recipe backends.Recipe) error {
	cmdBackendAddRecipe([]string{
		recipe.GitURL,
		"--tag", recipe.Tag,
		"--checkout-name", recipe.Tag,
		"--branch", recipe.Branch,
		"--commit", recipe.Commit,
		"--route-arch", recipe.RouteArch,
	}, &recipe)
	return nil
}

// offerDiscoveredArchFork searches the official llama.cpp PR index and the
// GGUF publisher's Hugging Face card for a fork that adds this architecture,
// then offers to clone/build it. Reviewed recipes stay first; this is the
// generic path for a novel arch the catalog does not yet name. A miss or
// decline falls through to the mainline-update offer.
func offerDiscoveredArchFork(req *launchRequest, model *placement.ModelProfile, assumeYes bool) bool {
	return offerDiscoveredArchForkWith(req, model, assumeYes, os.Stdin, os.Stderr, stdinIsTerminal(), searchArchForkPRs, installDiscoveredArchFork)
}

func offerDiscoveredArchForkWith(
	req *launchRequest,
	model *placement.ModelProfile,
	assumeYes bool,
	in io.Reader,
	out io.Writer,
	terminal bool,
	search func(context.Context, *placement.ModelProfile) ([]backends.ArchForkPR, error),
	install func(backends.Recipe) error,
) bool {
	if req == nil || model == nil || search == nil || install == nil {
		return false
	}
	arch := strings.TrimSpace(model.ModelArch)
	if arch == "" || strings.TrimSpace(req.BackendUnavailableReason) == "" {
		return false
	}
	if len(backends.RecipesForArch(arch)) > 0 {
		return false
	}
	prs, err := search(context.Background(), model)
	if err != nil {
		fmt.Fprintf(out, "[launch] could not search llama.cpp PRs for architecture %s: %v\n", arch, err)
		return false
	}
	var candidate backends.ArchForkPR
	for _, pr := range prs {
		if pr.GitURL != "" && pr.Branch != "" && pr.Commit != "" && pr.Number > 0 && !pr.Merged {
			candidate = pr
			break
		}
	}
	if candidate.Number == 0 {
		return false
	}
	recipe := candidate.RecipeForModel(discoveredForkModelName(model))
	source := fmt.Sprintf("Open llama.cpp PR #%d (%s)", candidate.Number, candidate.URL)
	if candidate.Cited {
		source = fmt.Sprintf("Hugging Face model card cites open llama.cpp PR #%d (%s)", candidate.Number, candidate.URL)
	}
	if !assumeYes {
		if !terminal {
			fmt.Fprintf(out, "[launch] %s adds %s. Install with: ggrun backend add %s --branch %s --commit %s --tag %s --checkout-name %s --route-arch %s\n",
				source, arch, recipe.GitURL, recipe.Branch, recipe.Commit, recipe.Tag, recipe.Tag, recipe.RouteArch)
			return false
		}
		fmt.Fprintf(out, "No installed backend can load architecture %s for %s. %s adds it as a separate fork named %q under .src/fork-* (mainline llama.cpp is not modified). Install that fork now? [y/N] ",
			arch, recipe.Tag, source, recipe.Tag)
		line, _ := bufio.NewReader(in).ReadString('\n')
		answer := strings.ToLower(strings.TrimSpace(line))
		if answer != "y" && answer != "yes" {
			return false
		}
	}
	fmt.Fprintf(out, "[launch] installing isolated fork %q from llama.cpp PR #%d into %s (mainline %s untouched)\n",
		recipe.Tag, candidate.Number, backends.IsolatedForkSourceDir("", recipe.GitURL, recipe.Branch, recipe.Tag), backends.MainlineSourceDir(""))
	return install(recipe) == nil
}

func discoveredForkModelName(model *placement.ModelProfile) string {
	if model == nil {
		return ""
	}
	if name := strings.TrimSpace(model.Name); name != "" {
		return name
	}
	if name := strings.TrimSpace(model.Basename); name != "" {
		return name
	}
	return placement.CalibrationModelBasename(model.Path)
}

func backendUnavailableMessage(req *launchRequest) string {
	if req != nil {
		// FAIL-CLOSED auto selection refused: the reason is specific (the chosen
		// backend cannot load the model's architecture), so surface it verbatim
		// instead of the generic "no binary found" fallback.
		if reason := strings.TrimSpace(req.BackendUnavailableReason); reason != "" {
			return reason
		}
		want := requestedBackendName(req)
		if want != "" && !strings.EqualFold(want, "auto") {
			where := strings.TrimSpace(req.AppHome)
			if where == "" {
				where = backends.AppHome()
			}
			return fmt.Sprintf("selected backend %q was not found under APP_HOME %q or the registered backend paths; install/build it or choose backend auto", want, where)
		}
	}
	return "no llama-server binary found. Install one with: ggrun backend install <recipe>  (see: ggrun backend recipes)"
}

// routeArchBackend redirects to a registered fork backend when the model's
// architecture is registered with a route-arch and the backend is still
// implicit/auto. A configured or CLI-selected backend must keep its actual
// backend instead of being hijacked by a fork route.
func routeArchBackend(be *backendInfo, model *placement.ModelProfile, req *launchRequest) *backendInfo {
	if req.BackendExplicit || req.ServerBinExplicit || model == nil {
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
	// Keep recipe identity for selection/tune-cache isolation while retaining
	// the probed flag dialect separately. A recipe name such as "hy3" must not
	// make an IK fork receive mainline split/spec flags.
	info.Tag = cb.Tag
	return info
}

func backendDialect(be *backendInfo) string {
	if be == nil {
		return "llama"
	}
	if be.Dialect != "" {
		return be.Dialect
	}
	return be.Tag
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
	return placementOptionsFromRequestCaps(req, model, be, cacheDir, nil)
}

// placementOptionsFromRequestCaps is placementOptionsFromRequest plus an
// optional caps argument used to derive the verified-config scope key. When
// caps is nil the scope key is left unset (callers that do not want verified
// config reuse, or cannot derive hardware, get today's behavior).
// warnIfContextForcesOffload tells the user when the context they named will push
// the KV cache onto the host. An explicit context is never lowered -- it is their
// call -- but a silent collapse in generation speed is not a good way to find out.
//
// The comparison asks the planner the same question it is about to answer, with
// the same options and only ContextSize cleared, rather than re-deriving a fit
// separately. An independent estimate answered a subtly different question and
// named a fallback context that did not match what the planner would actually
// choose -- in both directions.
func warnIfContextForcesOffload(caps *detect.Capabilities, model *placement.ModelProfile, req *launchRequest, opts placement.Options, userSetCtx bool) {
	if !userSetCtx || caps == nil || model == nil || opts.ContextSize <= 0 || len(caps.GPUs) == 0 {
		return
	}
	fitOpts := opts
	fitOpts.ContextSize = 0
	// A prompt hook would block on a background comparison, and this must never
	// ask the user anything.
	fitOpts.DenseCPUOffloadPrompt = nil
	fitPlan, err := placement.Compute(caps, model, fitOpts)
	if err != nil || fitPlan == nil || fitPlan.ContextSize <= 0 {
		return
	}
	if opts.ContextSize <= fitPlan.ContextSize {
		return
	}
	ctxOffloadWarning.Do(func() {
		fmt.Fprintf(os.Stderr,
			"[launch] warning: --ctx %d exceeds what fits in VRAM (%d); the KV cache will be offloaded to host RAM and generation will be markedly slower. Use --ctx fit, or --ctx %d, to keep it resident.\n",
			opts.ContextSize, fitPlan.ContextSize, fitPlan.ContextSize)
	})
}

// ctxOffloadWarning keeps the "your context will offload" notice to one line:
// placement is recomputed several times per launch (measured re-plan, recovery),
// and the notice is about the user's choice, not about any individual plan.
var ctxOffloadWarning sync.Once

func placementOptionsFromRequestCaps(req *launchRequest, model *placement.ModelProfile, be *backendInfo, cacheDir string, caps *detect.Capabilities) placement.Options {
	ctxSize := resolveCtxFlag(req.CtxFlag, model.CTXTrain)
	// A resolved size here means the user named one (a number, or "max"); "fit"
	// and the empty default both resolve to 0.
	userSetCtx := ctxSize > 0
	autoContextMax := 0
	if req.ClaudeCode && ctxSize <= 0 {
		// This is a workload ceiling, not a pre-resolved context. The placement
		// package still searches the exact full-plan boundary below it, so Claude,
		// ordinary CLI, TUI, recovery, and dry-run all use one fit resolver.
		autoContextMax = model.CTXTrain
		if autoContextMax > 1048576 {
			autoContextMax = 1048576
		} else if autoContextMax <= 0 {
			autoContextMax = 131072
		}
	}
	samplingProfile := requestSamplingProfile(req, model)
	hotExperts := req.HotExperts
	if req.HotExpertsRuntimeDisabled {
		hotExperts = "off"
	}
	hotSlots := 0
	if parsed, err := strconv.Atoi(strings.TrimSpace(hotExperts)); err == nil && parsed > 0 {
		hotSlots = parsed
	}
	if req.DisabledBackendFlags != nil {
		if _, disabled := req.DisabledBackendFlags["--moe-expert-cache"]; disabled {
			hotExperts, hotSlots = "off", 0
		}
	}
	opts := placement.Options{
		ContextSize:             ctxSize,
		AutoContextMax:          autoContextMax,
		KVPlacement:             req.KVPlacement,
		KVQuality:               req.KVQuality,
		KVQualityV:              req.KVQualityV,
		CPUMode:                 req.CPUMode,
		RamBudgetMB:             req.RamBudgetMB,
		RAMLimitPercent:         req.RAMLimitPercent,
		VRAMHeadroomMB:          req.VRAMHeadroomMB,
		RAMHeadroomMB:           req.RAMHeadroomMB,
		RequireMeasuredBuffers:  true,
		NoMMap:                  req.NoMMap,
		ForceMMap:               req.ForceMMap,
		CacheDir:                cacheDir,
		Host:                    req.Host,
		BackendTag:              backendDialect(be),
		BackendCacheTag:         evidenceBackendCacheTag(be),
		BackendIdentity:         be.Identity,
		CPUExpertMMapCapability: backendCPUExpertMMapCapability(be),
		CPUExpertMMapEvidence:   be.CPUExpertMMapEvidence,
		SamplingProfile:         samplingProfile,
		WorkloadProfile:         requestWorkloadProfile(req, model),
		WorkloadConcurrency:     requestWorkloadConcurrency(req),
		VisionAuto:              req.VisionAuto,
		MMProjPath:              req.MMProjPath,
		SpecMode:                req.SpecMode,
		ForceSpecMoE:            req.ForceSpecMoE,
		BackendHelp:             be.Help,
		HotExperts:              hotExperts,
		HotExpertCacheSlots:     hotSlots,
		SpecCandidateValidator:  backendSpecCandidateValidator(be, model.ModelArch),
		CacheFile:               req.TuneCache,
		Parallel:                req.Parallel,
		ParallelExplicit:        req.ParallelSet,
		Threads:                 req.Threads,
		CacheRAMMB:              req.CacheRAMMB,
		// --swa-full is a passthrough flag, but placement cannot treat it as
		// one: it decides whether sliding-window layers hold the whole context,
		// which on Laguna is the difference between 13.8 GB and 54.0 GB of KV
		// and therefore between fitting and not fitting.
		SWAFull:            hasArg(req.ExtraArgs, "--swa-full"),
		BatchSize:          req.BatchSize,
		UBatchSize:         req.UBatchSize,
		BatchSizeExplicit:  req.BatchSizeSet,
		UBatchSizeExplicit: req.UBatchSizeSet,
		// Disable the model's thinking only when measuring (`--benchmark`); a
		// normal launch keeps reasoning on so tools like Claude Code can think.
		ReasoningOff: req.Benchmark || req.WorkerBenchmark,
		// --no-cached-config derives placement fresh: skip the keyed .place cache
		// and every probe/KV/prompt measurement (SkipCachedConfig implies
		// SkipPlacementCache inside Compute).
		SkipCachedConfig: req.NoCachedConfig,
		// ChatTemplate scopes placement/cache keys to the forced template entry,
		// so a verified config measured under one template is never reused under
		// another (different serving contract, different emitted flags).
		ChatTemplate: req.ChatTemplateOverride,
	}
	if req.GPUsFlag != "" {
		if indices, err := parseGPUIndices(req.GPUsFlag); err == nil {
			opts.GPUs = indices
		}
	}
	// Attach the reviewer companion on every Compute path — first plan and every
	// re-plan alike — so a corrective recompute never forgets the reviewer's VRAM.
	if req.ReviewerReservation != nil {
		opts.Companions = []placement.CompanionReservation{*req.ReviewerReservation}
	}
	// Attach the dense-offload prompt on every Compute path so a re-plan that
	// could still land on DenseCPUOffload asks the user instead of silently
	// spilling the model into system RAM.
	opts.DenseCPUOffloadPrompt = req.DenseOffloadPrompt
	requestedParallel := claudeCodeParallel(opts.Parallel, req.ClaudeCode, req.ParallelSet, req.ClaudeProfile)
	opts.Parallel = claudeCodeSlotsForPlacement(
		requestedParallel,
		opts.ContextSize, req.ClaudeCode, req.ParallelSet,
	)
	opts.AutoParallel = req.ClaudeCode && !req.ParallelSet && opts.ContextSize <= 0 && requestedParallel > 1
	if opts.AutoParallel {
		opts.ParallelSlotTarget = claudeSlotTarget
	}
	if req.ClaudeCode {
		opts.RuntimePolicy = func(strategy *placement.Strategy) {
			applyClaudeCodeRuntimePolicy(strategy, model, req.BatchSizeSet, req.UBatchSizeSet, false)
		}
	}
	warnIfContextForcesOffload(caps, model, req, opts, userSetCtx)
	// Derive the strategy-free verified-config scope key from the finalized opts.
	// It is what the reuse lookup in placement.Compute hashes against, and the
	// save path uses the same computation, so save and load can never disagree.
	if caps != nil && !req.NoCachedConfig {
		opts.VerifiedConfigScopeKey = placement.NewCalibrationScopeKey(model, caps, opts, nil).String() + "-plan" + planLogicVersion
	}
	return opts
}

// claudeCodeSlotsForPlacement applies the same context-driven slot clamp that
// claudeCodeSlotAdjust applies to the finished strategy, so the plan is computed
// for the slot count the server will actually run with.
//
// Without it the two disagreed, and the disagreement was silent and total:
// placement asked for 4 slots while the launch ran 2, and every probe/placement
// key embeds the slot count (probeParallelKey). Measured 2026-08-03 on
// DeepSeek-V4-Flash UD-Q3_K_XL — the planner looked up evidence at parallel=4,
// the preflight recorded it at parallel<=2, so a measured 9501 MiB compute
// buffer sat on disk and was never read. The planner charged 0 MiB for compute,
// over-packed CUDA0, and the load died asking for 9307 MiB. Evidence that is
// written under one key and read under another is evidence that does not exist.
func claudeCodeSlotsForPlacement(parallel, contextSize int, claudeCode, parallelExplicit bool) int {
	if !claudeCode || parallelExplicit || contextSize <= 0 || parallel <= 1 {
		return parallel
	}
	slots := contextSize / claudeSlotTarget
	if slots < 1 {
		slots = 1
	}
	if slots < parallel {
		return slots
	}
	return parallel
}

func requestSamplingProfile(req *launchRequest, model *placement.ModelProfile) string {
	if req == nil {
		return "default"
	}
	// Include every explicit backend override: unknown fork flags can affect
	// sampling or throughput too. Then add ggrun's effective Claude defaults so
	// a Claude profile cannot be reused by an ordinary OpenAI-compatible launch.
	values := append([]string(nil), req.ExtraArgs...)
	if req.BatchSizeSet {
		values = append(values, fmt.Sprintf("batch-size=%d", req.BatchSize))
	}
	if req.UBatchSizeSet {
		values = append(values, fmt.Sprintf("ubatch-size=%d", req.UBatchSize))
	}
	values = claudeCodeSamplingArgs(values, req.ClaudeCode, model)
	if len(values) == 0 && !req.ClaudeCode {
		return "default"
	}
	values = append(values, fmt.Sprintf("claude-code=%t", req.ClaudeCode))
	if profile := effectiveClaudeProfile(req); profile != "" {
		// The omitted profile and explicit agent-parallel have the same behavior;
		// identify them identically while keeping interactive separate.
		values = append(values, "claude-profile="+profile)
	}
	sum := sha256.Sum256([]byte(strings.Join(values, "\x00")))
	return fmt.Sprintf("custom-%x", sum[:8])
}

// claudeCodeParallel requests four sequence slots in Claude Code mode so the main
// turn and a small Workflow fan-out can make progress concurrently. Four is a
// concurrency default, not a claim of 4x inference throughput: active requests on
// a bandwidth-bound big MoE still share the same memory bandwidth. The explicit
// agent-interactive profile keeps the configured single foreground slot instead.
//
// claudeCodeSlotAdjust runs after placement and lowers this automatic value when
// the selected total context cannot preserve a useful per-slot window. An explicit
// --parallel always wins, including --parallel 2 for a tighter big-MoE setup or 8
// for hardware that has been proven stable under wider fan-out.
func claudeCodeParallel(parallel int, claudeCode, explicit bool, profile string) int {
	if !claudeCode || explicit {
		return parallel
	}
	switch profile {
	case claudeProfileInteractive:
		// A selected interactive profile is a preset, not merely a refusal to
		// raise the configured count. This keeps a stale LLM_PARALLEL=4 from
		// silently defeating the promised one foreground-agent lane.
		return 1
	case claudeProfileParallel:
		return 4
	default:
		if parallel < 4 {
			return 4
		}
	}
	return parallel
}

// claudeSlotTarget is the per-slot context Claude Code comfortably works in.
// claudeSlotMin is the floor below which a session can't even hold the system
// prompt (~15-20k tokens) and requests truncate or fail outright.
const (
	claudeSlotTarget = 65536
	claudeSlotMin    = 24576
	// Live parallel-2 testing on an offloaded DeepSeek V4 showed a 512-token
	// prompt chunk holding the scheduler for about 22 seconds and reducing a
	// concurrent worker to 0.05-0.15 tok/s. Keep concurrent hybrid workloads at
	// 128 so another Claude slot gets a scheduling opportunity roughly four times
	// as often. A single foreground slot has no fairness contention, so it keeps
	// the placement-selected batch size for efficient MoE prefill. Explicit extra
	// backend arguments still override this value.
	claudeHybridBatch = 128
)

// claudeCodeSlotAdjust caps the computed --parallel so each slot keeps a workable
// context window. claudeCodeParallel floors parallel at 4 BEFORE placement, which
// is right for large contexts, but 131072/4 is only 32k and "fit" mode can select
// even less (e.g. 32768/4 = 8k). Fewer, bigger slots beat undersized slots:
// more, broken ones: concurrent requests then queue (API_TIMEOUT_MS covers the
// wait) and may re-process the prompt on interleave — slow, but functional.
// Runs after placement.Compute and before Strategy.Args, so the emitted
// --parallel and the derived CLAUDE_AUTOCOMPACT_PCT_OVERRIDE stay consistent.
func claudeCodeSlotAdjust(strategy *placement.Strategy, model *placement.ModelProfile, claudeCode, parallelExplicit, batchExplicit, ubatchExplicit bool) {
	if !claudeCode || strategy == nil {
		return
	}
	if strategy.ContextSize > 0 && strategy.Parallel > 1 {
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
	applyClaudeCodeRuntimePolicy(strategy, model, batchExplicit, ubatchExplicit, true)
}

// applyClaudeCodeRuntimePolicy resolves the batch graph that placement must
// account for after automatic context fit chooses a fixed slot count. The quiet
// form runs inside placement.Compute before draft/companion/cache/placement
// ledgers; the announcing form runs at finalization as a no-op safety check.
func applyClaudeCodeRuntimePolicy(strategy *placement.Strategy, model *placement.ModelProfile, batchExplicit, ubatchExplicit, announce bool) {
	if strategy == nil {
		return
	}
	// A full verified-config replay is allowed to bypass placement.Compute. Keep
	// the final serving policy model-derived as well as strategy-derived so a
	// cached record can never erase recurrent semantics, context-shift safety, or
	// the parallel-agent fairness baseline.
	modelHasSSM := model != nil && (model.HasSSM != 0 || strings.EqualFold(model.ModelArch, "deepseek4"))
	if modelHasSSM {
		strategy.HasSSM = true
	}
	// Normalize the slot count before applying the fairness policy: an automatic
	// 4-slot request can legitimately become one slot at a smaller context size.
	// Capping that final single foreground slot to 128 would sacrifice long-prompt
	// MoE efficiency without protecting any competing decode.
	if strategy.HasSSM && strategy.Parallel > 1 && strategy.BatchSize > claudeHybridBatch && !batchExplicit && !strategy.BatchTuned {
		target := claudeHybridBatch
		if ubatchExplicit && strategy.UBatchSize > target {
			// An explicit physical microbatch is user-owned. Keep the logical
			// batch large enough to make it real instead of advertising a value
			// llama.cpp will silently clamp.
			target = strategy.UBatchSize
		}
		if announce {
			fmt.Printf("[claude-code] hybrid recurrent model: lowering --batch-size from %d to %d so prompt prefill does not starve another active slot\n",
				strategy.BatchSize, target)
		}
		strategy.BatchSize = target
	}
	placement.NormalizeBatchSizes(strategy, model, batchExplicit, ubatchExplicit)
}

func buildLaunchServerArgs(req *launchRequest, cfg *config.Config, be *backendInfo, caps *detect.Capabilities, model *placement.ModelProfile, strategy *placement.Strategy) []string {
	if req.SpecDraftMax > 0 && strategy != nil && strategy.Draft != nil && strategy.Draft.Type != placement.DraftNone {
		strategy.Draft.DraftMax = req.SpecDraftMax
	}
	oldBatch, oldUBatch := strategy.BatchSize, strategy.UBatchSize
	if placement.NormalizeBatchSizes(strategy, model, req.BatchSizeSet, req.UBatchSizeSet) {
		fmt.Printf("[launch] normalized batch/ubatch %d/%d -> %d/%d (llama.cpp requires ubatch <= batch)\n",
			oldBatch, oldUBatch, strategy.BatchSize, strategy.UBatchSize)
	}
	serverArgs := append([]string{be.Path}, strategy.Args(req.ModelPath, req.Port)...)
	serverArgs = append(serverArgs, hy3CompatibilityArgs(req.ExtraArgs, model, be)...)
	serverArgs = append(serverArgs, hy3TemplateArgs(req.ExtraArgs, be)...)
	serverArgs = append(serverArgs, catalogTemplateArgs(req, cfg, model)...)
	serverArgs = append(serverArgs, req.ExtraArgs...)
	serverArgs = applyTuneCache(req, serverArgs, cfg.CacheDir, be.Tag, strategy.MMProjPath != "", caps)
	serverArgs = claudeCodeAliasArgs(serverArgs, req.ClaudeCode)
	serverArgs = claudeCodeSamplingArgs(serverArgs, req.ClaudeCode, model)
	serverArgs = claudeCodeCacheArgs(serverArgs, req.ClaudeCode, be.Help, claudeCodeShiftableContext(model, strategy))
	serverArgs = claudeCodeProgressServerArgs(serverArgs, req.ClaudeCode, be.Help)
	serverArgs = backendVerbosityArgs(serverArgs, be.Help)
	return applyRequestDisabledBackendFlags(serverArgs, req)
}

func disableBackendFlag(req *launchRequest, flag, reason string) bool {
	return disableBackendFlagWithArity(req, flag, reason, false)
}

func disableBackendFlagWithArity(req *launchRequest, flag, reason string, hasValue bool) bool {
	if req == nil || strings.TrimSpace(flag) == "" {
		return false
	}
	if req.DisabledBackendFlags == nil {
		req.DisabledBackendFlags = make(map[string]string)
	}
	if _, exists := req.DisabledBackendFlags[flag]; exists {
		return false
	}
	req.DisabledBackendFlags[flag] = reason
	if hasValue {
		if req.DisabledBackendFlagValues == nil {
			req.DisabledBackendFlagValues = make(map[string]bool)
		}
		req.DisabledBackendFlagValues[flag] = true
	}
	return true
}

func applyDisabledBackendFlags(args []string, disabled map[string]string) []string {
	if len(args) == 0 || len(disabled) == 0 {
		return args
	}
	out := make([]string, 0, len(args))
	for index := 0; index < len(args); index++ {
		arg := args[index]
		if _, remove := disabled[arg]; remove {
			// The arity map belongs to the request, but this function is retained as
			// the boolean-only compatibility helper for existing call sites.
			continue
		}
		out = append(out, arg)
	}
	return out
}

func applyRequestDisabledBackendFlags(args []string, req *launchRequest) []string {
	if req == nil || len(req.DisabledBackendFlags) == 0 {
		return args
	}
	out := make([]string, 0, len(args))
	for index := 0; index < len(args); index++ {
		arg := args[index]
		if _, remove := req.DisabledBackendFlags[arg]; remove {
			if req.DisabledBackendFlagValues[arg] && index+1 < len(args) {
				index++
			}
			continue
		}
		out = append(out, arg)
	}
	return out
}

// backendVerbosityArgs raises the backend's log level to trace, because ggrun's
// whole approach is to read measurements out of the backend rather than model
// them -- and the measurements that matter are not printed at the default level.
//
// llama.cpp defaults to verbosity 3 (info). At that level ggrun cannot see:
//
//   - "forcing full prompt re-processing due to lack of cache data" and
//     "restored context checkpoint", which are the only evidence of whether a
//     sliding-window model reused any prefix at all. This project ran for weeks
//     re-prefilling every turn -- 0 reused tokens out of 1.16 million measured
//     after the fact -- with nothing in the logs to say so.
//   - "cache state: N prompts, X MiB (limits: ...)", the host prompt cache's own
//     accounting, which is what CRAM should be sized from.
//   - the per-checkpoint search decisions that explain a reuse miss.
//
// Trace, not debug: level 5 adds per-token output that makes a multi-hour agent
// log unusable, while level 4 adds only these decision points. An explicit
// backend-supported verbosity flag, or LLAMA_ARG_LOG_VERBOSITY in the
// environment, still wins.
func backendVerbosityArgs(args []string, backendHelp string) []string {
	if os.Getenv("LLAMA_ARG_LOG_VERBOSITY") != "" {
		return args
	}
	for _, a := range args {
		if a == "-lv" || a == "--verbosity" || a == "--log-verbosity" ||
			strings.HasPrefix(a, "-lv=") || strings.HasPrefix(a, "--verbosity=") ||
			strings.HasPrefix(a, "--log-verbosity=") {
			return args
		}
	}
	// Backend forks do not share one spelling. Recent ik_llama advertises only
	// --verbosity, while mainline/older forks may advertise -lv as an alias.
	// Merely finding the long spelling and then emitting the short one caused the
	// entire V4 command to abort in argument parsing before model load. Select an
	// exact advertised spelling and make an unavailable help surface mean "do not
	// assume" rather than guessing a flag.
	flag := ""
	switch {
	case helpHasExactFlag(backendHelp, "-lv"):
		flag = "-lv"
	case helpHasExactFlag(backendHelp, "--verbosity"):
		flag = "--verbosity"
	case helpHasExactFlag(backendHelp, "--log-verbosity"):
		flag = "--log-verbosity"
	}
	if flag == "" {
		return args
	}
	return append(args, flag, strconv.Itoa(backendTraceVerbosity))
}

// validateBackendLaunchArgs makes backend compatibility a launch invariant,
// not something discovered after a helper or a 100+ GB model has started. Both
// maintained llama-server families parse every preceding option before handling
// a trailing --version. That gives us an exact, no-weight-load parser probe of
// the final argv, including tune-cache and backend-dialect additions.
func validateBackendLaunchArgs(be *backendInfo, args []string) error {
	if be == nil || len(args) == 0 || strings.TrimSpace(args[0]) == "" {
		return fmt.Errorf("cannot validate an empty backend launch command")
	}
	if backendLoaderFailed(be.Help) {
		detail := strings.TrimSpace(firstNonEmptyLine(be.Help))
		if detail == "" {
			detail = "shared library or loader error"
		}
		return fmt.Errorf("backend %s cannot start: %s", be.Path, detail)
	}
	probeFlag := ""
	switch {
	case helpHasExactFlag(be.Help, "--version"):
		probeFlag = "--version"
	case helpHasExactFlag(be.Help, "--help"):
		probeFlag = "--help"
	default:
		return fmt.Errorf("backend %s exposes neither --version nor --help, so ggrun cannot validate its launch dialect safely", be.Path)
	}

	probeArgs := append([]string(nil), args[1:]...)
	probeArgs = append(probeArgs, probeFlag)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, args[0], probeArgs...)
	cmd.Env = server.ChildEnv(os.Environ(), args)
	if hubDir, ok, err := libhub.Setup(be.Path); err != nil {
		return fmt.Errorf("prepare backend libraries for argument validation: %w", err)
	} else if ok {
		defer libhub.Cleanup(hubDir)
		cmd.Env = libhub.ApplyHubToChildEnv(cmd.Env, hubDir)
	}
	out, runErr := cmd.CombinedOutput()
	if ctx.Err() != nil {
		return fmt.Errorf("backend %s argument validation timed out before any model load", be.Path)
	}
	if diagnostic := backendArgumentDiagnostic(string(out)); diagnostic != "" {
		return &backendArgValidationError{Backend: be.Path, Flag: rejectedBackendFlag(diagnostic), Diagnostic: diagnostic}
	}
	if runErr == nil {
		return nil
	}
	// Some older llama-server builds deliberately return non-zero for --help.
	// A real usage page proves the parser reached the requested help action; a
	// loader crash or missing shared library does not.
	if probeFlag == "--help" && strings.Contains(strings.ToLower(string(out)), "usage:") {
		return nil
	}
	detail := strings.TrimSpace(firstNonEmptyLine(string(out)))
	if detail == "" {
		detail = runErr.Error()
	}
	return fmt.Errorf("backend %s could not validate the generated launch command before model load: %s", be.Path, detail)
}

type backendArgValidationError struct {
	Backend    string
	Flag       string
	Diagnostic string
}

func (e *backendArgValidationError) Error() string {
	return fmt.Sprintf("backend %s rejected the generated launch command before model load: %s", e.Backend, e.Diagnostic)
}

func rejectedBackendFlag(diagnostic string) string {
	for _, field := range strings.Fields(diagnostic) {
		candidate := strings.Trim(field, "'\"`:,;()[]{}")
		if strings.HasPrefix(candidate, "--") && len(candidate) > 2 {
			return candidate
		}
		if strings.HasPrefix(candidate, "-") && len(candidate) > 1 {
			if _, err := strconv.ParseFloat(candidate, 64); err != nil {
				return candidate
			}
		}
	}
	return ""
}

func backendArgumentDiagnostic(output string) string {
	for _, line := range strings.Split(output, "\n") {
		trimmed := strings.TrimSpace(line)
		lower := strings.ToLower(trimmed)
		if strings.Contains(lower, "unknown argument") ||
			strings.Contains(lower, "unrecognized argument") ||
			strings.Contains(lower, "invalid argument") ||
			strings.Contains(lower, "unrecognized option") ||
			strings.Contains(lower, "invalid option") {
			return trimmed
		}
	}
	return ""
}

func firstNonEmptyLine(output string) string {
	for _, line := range strings.Split(output, "\n") {
		if trimmed := strings.TrimSpace(line); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

// backendTraceVerbosity is llama.cpp's LOG_LEVEL_TRACE.
const backendTraceVerbosity = 4

// hy3CompatibilityArgs supplies only the metadata omitted by the known HY3
// GGUF layout. The values are derived from tensors by parse_gguf.py, restricted
// to ggrun's reviewed HY3 recipe, and appear before user extra arguments so a
// deliberate override remains authoritative.
func hy3CompatibilityArgs(extra []string, model *placement.ModelProfile, be *backendInfo) []string {
	if model == nil || be == nil || !strings.EqualFold(model.ModelArch, "hy_v3") || !strings.EqualFold(be.Tag, "hy3") {
		return nil
	}
	args := make([]string, 0, 4)
	if model.ExpertSharedCountInferred && model.ExpertSharedCount > 0 && !hasKVOverride(extra, "hy_v3.expert_shared_count") {
		args = append(args, "--override-kv", fmt.Sprintf("hy_v3.expert_shared_count=int:%d", model.ExpertSharedCount))
	}
	if model.LeadingDenseInferred && model.LeadingDense >= 0 && !hasKVOverride(extra, "hy_v3.leading_dense_block_count") {
		args = append(args, "--override-kv", fmt.Sprintf("hy_v3.leading_dense_block_count=int:%d", model.LeadingDense))
	}
	return args
}

func hasKVOverride(args []string, key string) bool {
	for i := 0; i < len(args); i++ {
		value := ""
		if args[i] == "--override-kv" && i+1 < len(args) {
			value = args[i+1]
			i++
		} else if strings.HasPrefix(args[i], "--override-kv=") {
			value = strings.TrimPrefix(args[i], "--override-kv=")
		}
		if strings.HasPrefix(value, key+"=") {
			return true
		}
	}
	return false
}

// hy3TemplateArgs replaces the GGUF's Python-specific .format() template with
// the equivalent minja-compatible template shipped by the reviewed HY3 fork.
// It is deliberately recipe-scoped and never overrides a user's explicit chat
// template choice.
func hy3TemplateArgs(extra []string, be *backendInfo) []string {
	if be == nil || !strings.EqualFold(be.Tag, "hy3") || hasChatTemplateOverride(extra) {
		return nil
	}
	root := filepath.Dir(filepath.Dir(filepath.Dir(be.Path)))
	template := filepath.Join(root, "models", "templates", "Hy3.jinja")
	if info, err := os.Stat(template); err != nil || info.IsDir() {
		return nil
	}
	return []string{"--chat-template-file", template}
}

func hasChatTemplateOverride(args []string) bool {
	for _, arg := range args {
		if arg == "--chat-template" || arg == "--chat-template-file" ||
			strings.HasPrefix(arg, "--chat-template=") || strings.HasPrefix(arg, "--chat-template-file=") {
			return true
		}
	}
	return false
}

// catalogTemplateArgs overrides the GGUF's embedded chat template with a
// corrected copy from the data-driven chat-template catalog (pkg/chattemplate).
// Some models ship GGUFs whose embedded template contains a Jinja
// `raise_exception('System message must be at the beginning.')` guard that
// fires when a system message is not the first message (llama.cpp appends its
// own tool-instruction system message during tool-call parser generation),
// 400ing every such request under --jinja with "Unable to generate parser for
// this template." Tool calls REQUIRE --jinja, so the fix is to keep --jinja and
// serve a corrected template instead of dropping the flag.
//
// Which model gets which corrected template is pure catalog data: adding an
// entry to pkg/chattemplate/catalog.json fixes any future broken model with no
// code change. The corrected template is materialized into the ggrun cache dir
// so the backend receives a real --chat-template-file path. A user's explicit
// chat template choice (--chat-template or --chat-template-file) always wins.
func catalogTemplateArgs(req *launchRequest, cfg *config.Config, model *placement.ModelProfile) []string {
	if req == nil || cfg == nil || model == nil {
		return nil
	}
	// A passthrough --chat-template/--chat-template-file (e.g. after a --) is a
	// direct backend flag: it wins over both the catalog auto-match and the
	// --chat-template name override.
	if hasChatTemplateOverride(req.ExtraArgs) {
		return nil
	}
	var entry chattemplate.Entry
	var ok bool
	if override := strings.TrimSpace(req.ChatTemplateOverride); override != "" {
		// Explicit user selection by catalog name forces that entry regardless
		// of arch/basename and regardless of whether the embedded template is
		// broken.
		entry, ok = chattemplate.ResolveOverride(override)
		if !ok {
			return nil
		}
	} else {
		arch := model.ModelArch
		basename := model.Basename
		if basename == "" {
			basename = filepath.Base(model.Path)
		}
		// Detect the raise_exception guard in the model's own embedded template.
		embedded := gguf.ChatTemplate(model.Path)
		entry, ok = chattemplate.Resolve(arch, basename, embedded, true)
		if !ok {
			return nil
		}
	}
	path, err := chattemplate.Materialize(cfg.CacheDir, entry)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[launch] warning: chat-template override for %s unavailable: %v\n", entry.Name, err)
		return nil
	}
	return []string{"--chat-template-file", path}
}

// specLaunchIdentity fingerprints the final runtime argv after tune caches,
// automatic Claude sampling and recovery placement have all been applied. Port
// and bind host are excluded because they do not affect model performance.
func specLaunchIdentity(args []string) string {
	canonical := make([]string, 0, len(args))
	for i := 1; i < len(args); i++ { // backend binary has its own scope identity
		arg := args[i]
		if arg == "--port" || arg == "--host" {
			i++
			continue
		}
		canonical = append(canonical, arg)
	}
	data, _ := json.Marshal(canonical)
	sum := sha256.Sum256(data)
	return fmt.Sprintf("%x", sum[:16])
}

// planDerivedLaunchFlags are the flags placement computes rather than the user
// choosing. Each carries a value, and all of them are excluded from the identity
// a crash log is filed under.
//
// They have to be: recovery exists precisely to change these. Keying the log on
// them files every attempt under a different name, so the next launch reads a
// path that was never written and re-plans the placement that just aborted. That
// is not hypothetical -- it is what happened here once `-cram` became a derived
// value. Three consecutive launches crashed at `--n-cpu-moe 23`; their argv
// differed only in `-cram` (9753, 9752, 9742, tracking free-RAM drift), which
// was enough to give each its own scope, and no OOM was ever recorded.
var planDerivedLaunchFlags = map[string]bool{
	"--port": true, "--host": true,
	"-ot": true, "--override-tensor": true,
	"--n-cpu-moe": true, "--tensor-split": true, "--split-mode": true,
	"-ngl": true, "--n-gpu-layers": true, "--gpu-layers": true,
	"-cram": true, "--cache-ram": true, "--ctx-checkpoints": true,
	"--moe-expert-cache": true, "--moe-expert-cache-inserts": true,
}

// recoveryLaunchIdentity identifies the launch *shape* -- model, context, slots,
// KV, batch, sampling, spec -- with the placement decisions removed. Two runs
// that differ only in how placement split the model share an identity, which is
// what lets the second one read the first one's crash.
//
// Kept separate from specLaunchIdentity on purpose: that one verifies a
// speculative result was produced by an exact command line and must stay exact.
func recoveryLaunchIdentity(args []string) string {
	canonical := make([]string, 0, len(args))
	for i := 1; i < len(args); i++ { // backend binary has its own scope identity
		if planDerivedLaunchFlags[args[i]] {
			i++
			continue
		}
		canonical = append(canonical, args[i])
	}
	data, _ := json.Marshal(canonical)
	sum := sha256.Sum256(data)
	return fmt.Sprintf("%x", sum[:16])
}

// claudeLaunchLogScope ties a recoverable Claude server log to the effective
// workload profile, exact backend build, and the launch shape. A port-only
// filename let an old interactive/parallel run donate (or suppress) OOM evidence
// for a different runtime shape.
func claudeLaunchLogScope(req *launchRequest, model *placement.ModelProfile, be *backendInfo, serverArgs []string) string {
	material := strings.Join([]string{
		"claude-server-log-v2",
		requestWorkloadProfile(req, model),
		evidenceBackendCacheTag(be),
		recoveryLaunchIdentity(serverArgs),
	}, "\x00")
	sum := sha256.Sum256([]byte(material))
	return fmt.Sprintf("%x", sum[:12])
}

func backendMemoryMaxMB(req *launchRequest, caps *detect.Capabilities) int {
	if req == nil || caps == nil {
		return 0
	}
	if runtime.GOOS != "linux" {
		return 0
	}
	limit := caps.RAM.FreeMB
	if req.RamBudgetMB > 0 && (limit <= 0 || req.RamBudgetMB < limit) {
		limit = req.RamBudgetMB
	} else if req.RamBudgetMB <= 0 && req.RAMLimitPercent > 0 {
		if percentLimit := detect.RAMLimitMB(caps.RAM, req.RAMLimitPercent); percentLimit < limit {
			limit = percentLimit
		}
	}
	if req.RAMHeadroomMB > 0 {
		limit -= req.RAMHeadroomMB
	}
	if limit <= 0 {
		return 0
	}
	return limit
}

func validateHostMemoryContainment(req *launchRequest, caps *detect.Capabilities, strategy *placement.Strategy) error {
	if runtime.GOOS != "linux" || req == nil || caps == nil || strategy == nil {
		return nil
	}
	hostMemoryPlacement := strategy.Type == placement.CPUOnly ||
		strategy.Type == placement.DenseCPUOffload ||
		strategy.Type == placement.MoEOffload
	if !hostMemoryPlacement {
		return nil
	}
	if req.RamBudgetMB <= 0 && req.RAMHeadroomMB <= 0 && req.RAMLimitPercent <= 0 {
		return fmt.Errorf(
			"%s placement uses host RAM and requires --ram-limit-percent, --ram-budget, or --ram-headroom",
			strategy.Type,
		)
	}
	if backendMemoryMaxMB(req, caps) <= 0 {
		return fmt.Errorf("host-memory containment limit is not positive")
	}
	// Fail-closed pre-launch gate (Fix B.4): refuse to start when the plan's own
	// resident footprint plus its larger future reserve (measured-footprint
	// headroom or CRAM) would exceed the whole-host ceiling. The plan's expert
	// bytes are exact, so a plan that can
	// only run by blowing past the ceiling is a plan that would OOM its own
	// scope at runtime (the 41-layer V4 crash: planned ~116 GB vs a ~111 GB
	// ceiling, --ctx-checkpoints 16). Refusing here forces a Fix-A re-plan (more
	// GPU layers, fewer CPU layers) instead of launching into a guaranteed death.
	// A 0 headroom disables the gate (--cgroup-headroom 0).
	//
	// Scoped to --no-mmap (anonymous) plans: file-backed expert pages are
	// reclaimable page cache that can legitimately overshoot the resident
	// footprint, so the mmap reclaim band absorbs it (see backendStartOptions).
	if req.CgroupHeadroomMB > 0 && strategy.PlannedHostFootprintMB > 0 && !strategyUsesReclaimableMMap(strategy) {
		ceiling := backendMemoryMaxMB(req, caps)
		plannedReserve := req.CgroupHeadroomMB
		if strategy.CRAM > plannedReserve {
			plannedReserve = strategy.CRAM
		}
		if strategy.PlannedHostFootprintMB+plannedReserve > ceiling {
			return fmt.Errorf(
				"planned host footprint %d MiB + required reserve %d MiB (cgroup headroom %d MiB, CRAM %d MiB) exceeds the %d MiB whole-host ceiling; refusing to launch a server that cannot be contained (re-plan with fewer CPU expert layers)",
				strategy.PlannedHostFootprintMB, plannedReserve, req.CgroupHeadroomMB, strategy.CRAM, ceiling,
			)
		}
	}
	return nil
}

var errMMapDeclined = errors.New("mmap declined; use --no-mmap with a placement that fits resident RAM")

func confirmRequiredMMap(req *launchRequest, strategy *placement.Strategy, input io.Reader, output io.Writer, interactive bool) error {
	if req == nil || strategy == nil || !strategy.MMapRequired || req.ForceMMap {
		return nil
	}
	if !interactive {
		return fmt.Errorf("placement requires file-backed mmap; rerun with --mmap to approve it explicitly")
	}
	fmt.Fprint(output, "Placement requires file-backed mmap and may page model weights from disk. Use mmap? [y/N] ")
	answer, err := bufio.NewReader(input).ReadString('\n')
	if err != nil && len(answer) == 0 {
		return fmt.Errorf("read mmap confirmation: %w", err)
	}
	switch strings.ToLower(strings.TrimSpace(answer)) {
	case "y", "yes":
		req.ForceMMap = true
		req.NoMMap = false
		return nil
	default:
		// A negative answer is a policy choice, not a terminal placement error.
		// The launcher catches the sentinel and recomputes through the normal
		// placement engine with strict resident accounting.
		req.NoMMap = true
		req.ForceMMap = false
		return errMMapDeclined
	}
}

func confirmMainModelReviewerFallback(input io.Reader, output io.Writer, interactive bool) error {
	if !interactive {
		return fmt.Errorf("resident placement fits only without the separate Claude Auto reviewer; set GGRUN_CLAUDE_AUTO_REVIEWER=off to choose that mode explicitly")
	}
	fmt.Fprint(output, "Resident placement fits without the separate Claude Auto reviewer. Continue with Auto reviews routed through the main model? [y/N] ")
	answer, err := bufio.NewReader(input).ReadString('\n')
	if err != nil && len(answer) == 0 {
		return fmt.Errorf("read reviewer fallback confirmation: %w", err)
	}
	switch strings.ToLower(strings.TrimSpace(answer)) {
	case "y", "yes":
		return nil
	default:
		return fmt.Errorf("main-model reviewer fallback declined")
	}
}

// tryResidentWithoutClaudeReviewer is the last safe resident fallback for a
// Claude Code launch. It never weakens the placement engine's RAM checks: it
// asks that same engine whether removing only the optional helper reservation
// creates a valid --no-mmap plan, then asks before adopting it. The gateway
// remains active and routes Auto review requests through the main model.
func tryResidentWithoutClaudeReviewer(
	req *launchRequest,
	originalErr error,
	input io.Reader,
	output io.Writer,
	interactive bool,
	compute func(*launchRequest) (*placement.Strategy, error),
) (*placement.Strategy, error) {
	if req == nil || !req.ClaudeCode || !req.NoMMap || req.ClaudeReviewerDisabled || req.ReviewerReservation == nil || compute == nil {
		return nil, originalErr
	}
	candidateReq := *req
	candidateReq.ReviewerReservation = nil
	candidateReq.ClaudeReviewerDisabled = true
	candidate, err := compute(&candidateReq)
	if err != nil || candidate == nil || candidate.MMapRequired {
		return nil, originalErr
	}
	if err := confirmMainModelReviewerFallback(input, output, interactive); err != nil {
		if !interactive {
			return nil, fmt.Errorf("%w\n%v", originalErr, err)
		}
		return nil, originalErr
	}
	req.ReviewerReservation = nil
	req.ClaudeReviewerDisabled = true
	fmt.Fprintln(output, "[launch] separate Auto reviewer disabled; Auto reviews will use the main model")
	return candidate, nil
}

// proactivelyDropReviewerForVRAMModel is the proactive counterpart to
// tryResidentWithoutClaudeReviewer. Where the fallback reacts to a placement that
// FAILS with the reviewer seated, this gate acts up front on the first plan that
// fits with the reviewer seated. Its one job is to shed a separate Auto reviewer
// that genuinely cannot be co-resident with the main model — and, crucially, to
// NEVER silently route Auto review through the main model.
//
// A reviewer exists to be a SEPARATE model judging the main model's output;
// dropping it makes the main model review itself, which is both unwanted and
// (for a safety/security classifier) insecure. So the gate keeps the reviewer
// whenever the recomputed main-model-only plan leaves enough spare VRAM for the
// configured reviewer to sit alongside the main model. Only when the main model
// consumes the device with no room for a separate reviewer is the drop even
// considered — and that self-review is permitted only with the user's explicit
// consent, mirroring confirmMainModelReviewerFallback. Returns the strategy to
// launch with: the recomputed main-model-only plan when the reviewer was dropped
// (self-review, consented), or the original strategy otherwise.
func proactivelyDropReviewerForVRAMModel(
	req *launchRequest,
	caps *detect.Capabilities,
	model *placement.ModelProfile,
	strategy *placement.Strategy,
	compute func(*launchRequest) (*placement.Strategy, error),
	input io.Reader,
	output io.Writer,
	interactive bool,
	cacheDir string,
) *placement.Strategy {
	if req == nil || strategy == nil || compute == nil {
		return strategy
	}
	// The request must actually have a GPU reviewer seat to reclaim.
	if !req.ClaudeCode || req.ClaudeReviewerDisabled || req.ReviewerReservation == nil {
		return strategy
	}
	// The gate only ever considers a fully-resident dense plan; MoE/offload
	// strategies keep their separate reviewer and their own reactive fallback.
	if strategy.MMapRequired || strategy.IsMoE {
		return strategy
	}
	if strategy.Type != placement.SingleGPU && strategy.Type != placement.MultiGPUDense {
		return strategy
	}
	candidateReq := *req
	candidateReq.ReviewerReservation = nil
	candidateReq.ClaudeReviewerDisabled = true
	candidate, err := compute(&candidateReq)
	if err != nil || candidate == nil || candidate.MMapRequired {
		return strategy
	}
	// The recomputed plan must still be a fully-resident dense plan; a recompute
	// that degraded to offload means dropping the reviewer costs the model its
	// residency, which is exactly the OOM risk the gate must not take.
	if candidate.IsMoE ||
		(candidate.Type != placement.SingleGPU && candidate.Type != placement.MultiGPUDense) {
		return strategy
	}
	headroom := placement.StrategyVRAMHeadroomMB(caps, model, candidate, cacheDir)
	// A separate reviewer is only ever worth shedding if it cannot already sit
	// alongside the main model. When the main model's plan leaves room for the
	// configured reviewer's own footprint in the spare VRAM, the reviewer stays
	// co-resident so a separate model keeps reviewing — dropping it here would
	// silently hand the review to the main model (the self-review regression).
	if headroom >= req.ReviewerReservation.VRAMMB {
		return strategy
	}
	// The main model alone leaves no room for a separate reviewer: the only way to
	// shed the reviewer is to have the main model review itself. That is never
	// silent — it requires the user's explicit approval, or fail-closed.
	if err := confirmProactiveSelfReview(input, output, interactive); err != nil {
		return strategy
	}
	req.ReviewerReservation = nil
	req.ClaudeReviewerDisabled = true
	fmt.Fprintf(output, "[launch] no VRAM room for a separate Auto reviewer next to the main model; Auto reviews will use the main model (self-review)\n")
	return candidate
}

// confirmProactiveSelfReview asks before letting the main model review its own
// output. Self-review is insecure for a safety classifier, so it is fail-closed:
// non-interactive launches need the explicit GGRUN_CLAUDE_AUTO_SELF_REVIEW=on
// opt-in, and an interactive decline keeps the separate reviewer. This mirrors
// the safety posture of confirmMainModelReviewerFallback.
func confirmProactiveSelfReview(input io.Reader, output io.Writer, interactive bool) error {
	if !interactive {
		if strings.EqualFold(strings.TrimSpace(os.Getenv("GGRUN_CLAUDE_AUTO_SELF_REVIEW")), "on") {
			return nil
		}
		return fmt.Errorf("no VRAM room for a separate Auto reviewer next to the main model; the main model would review its own output (insecure). set GGRUN_CLAUDE_AUTO_SELF_REVIEW=on to allow self-review explicitly")
	}
	fmt.Fprint(output, "No VRAM room for a separate Auto reviewer next to the main model. Let the main model review its own output (insecure)? [y/N] ")
	answer, err := bufio.NewReader(input).ReadString('\n')
	if err != nil && len(answer) == 0 {
		return fmt.Errorf("read self-review confirmation: %w", err)
	}
	switch strings.ToLower(strings.TrimSpace(answer)) {
	case "y", "yes":
		return nil
	default:
		return fmt.Errorf("main-model self-review declined")
	}
}

func confirmLiveMemoryProbe(req *launchRequest, reason string, input io.Reader, output io.Writer, interactive bool) error {
	if req == nil {
		return fmt.Errorf("live memory probe approval requires a launch request")
	}
	if req.AllowLiveMemoryProbe {
		return nil
	}
	if !interactive {
		return fmt.Errorf("%s; rerun with --allow-live-memory-probe", reason)
	}
	fmt.Fprintf(output, "%s. Run one contained live memory probe? This can take as long as loading the model. [y/N] ", reason)
	answer, err := bufio.NewReader(input).ReadString('\n')
	if err != nil && len(answer) == 0 {
		return fmt.Errorf("read live memory probe confirmation: %w", err)
	}
	switch strings.ToLower(strings.TrimSpace(answer)) {
	case "y", "yes":
		req.AllowLiveMemoryProbe = true
		return nil
	default:
		return fmt.Errorf("live memory probe declined")
	}
}

// rememberLiveMemoryProbeConsent persists an approval made at the interactive
// prompt. A command-line --allow-live-memory-probe remains a one-launch
// override because this helper is called only after the prompt path. Failure to
// remember the preference must not invalidate consent for the current launch.
func rememberLiveMemoryProbeConsent(cfg *config.Config, output io.Writer) {
	if cfg == nil || cfg.AllowLiveMemoryProbe {
		return
	}
	cfg.AllowLiveMemoryProbe = true
	if err := cfg.Save(); err != nil {
		fmt.Fprintf(output, "[config] warning: live memory probe approved for this launch, but the preference could not be saved: %v\n", err)
		return
	}
	fmt.Fprintln(output, "[config] live memory probe approval saved; future launches will not ask again")
}

func stdinIsTerminal() bool {
	info, err := os.Stdin.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

// hostExpertPinningEnv disables ik_llama's pinned host buffer when the plan
// parks expert tensors on the CPU.
//
// ik's CUDA host buffer is not a plain allocation. On Linux
// (ggml/src/ggml-cuda.cu ggml_cuda_host_malloc) it mmaps the whole request as
// MAP_ANONYMOUS, prefaults every single page (MADV_POPULATE_WRITE, or a manual
// byte-touch loop when that is unavailable), then cudaHostRegister()s the
// result. For the DeepSeek-V4 plan on this class of machine that is 84 GiB:
// ~22 million pages faulted in and then page-locked before the first weight is
// read. Observed 2026-08-04 as a launch that sat at a flat 88 GB RSS with the
// progress bar frozen -- not deadlocked, just paying that bill.
//
// GGML_CUDA_NO_PINNED makes ggml_cuda_host_malloc return nullptr immediately,
// and the caller has an explicit "fallback to cpu buffer" branch
// (ggml_backend_cuda_host_buffer_type_alloc_buffer), so the experts land in an
// ordinary CPU buffer instead. Resident loading is preserved -- which is the
// behaviour worth keeping, since a long one-time load beats permanently paging
// experts off SSD -- and the placement math above it stays correct, because it
// already budgets those bytes as resident.
//
// The cost is real and one-directional: pageable memory cannot use async DMA,
// so prompt processing is slower than it would be with pinning. A server that
// cannot finish loading has no throughput at all, so this is the right trade.
func hostExpertPinningEnv(be *backendInfo, serverArgs []string) []string {
	if be == nil || !be.IsIK {
		return nil
	}
	if !argsOffloadExpertsToCPU(serverArgs) {
		return nil
	}
	return []string{"GGML_CUDA_NO_PINNED=1"}
}

// argsOffloadExpertsToCPU reports whether the generated command line actually
// parks expert tensors in host memory. Keyed off the emitted argv rather than
// the strategy so it cannot drift from what the backend is really told.
func argsOffloadExpertsToCPU(serverArgs []string) bool {
	for i, a := range serverArgs {
		if a == "--n-cpu-moe" && i+1 < len(serverArgs) && serverArgs[i+1] != "0" {
			return true
		}
		if a == "-ot" && i+1 < len(serverArgs) && strings.Contains(strings.ToUpper(serverArgs[i+1]), "=CPU") {
			return true
		}
	}
	return false
}

// argsMMapBackedExperts reports whether the emitted command leaves CPU-side
// expert tensors file-backed rather than copied into host buffers. Keyed off
// argv for the same reason as argsOffloadExpertsToCPU: it cannot drift from
// what the backend is actually told.
func argsMMapBackedExperts(be *backendInfo, serverArgs []string) bool {
	if !cpuExpertsFileBacked(be) {
		return false
	}
	if !argsOffloadExpertsToCPU(serverArgs) {
		return false
	}
	for _, a := range serverArgs {
		if a == "--no-mmap" {
			return false
		}
	}
	return true
}

func strategyUsesReclaimableMMap(strategy *placement.Strategy) bool {
	if strategy == nil || !strategy.MMap || strategy.ReclaimableHostWeightsMB <= 0 {
		return false
	}
	capability := strategy.CPUExpertMMapCapability
	if capability == "" {
		// Compatibility for old in-memory strategies. Standard launches always
		// carry the explicit exact-backend capability.
		return !placement.BackendUsesAnonymousCPUExperts(strategy.BackendTag)
	}
	return placement.NormalizeCPUExpertMMapCapability(string(capability)) == placement.CPUExpertMMapFileBacked
}

func mmapPageabilityContradicted(strategy *placement.Strategy, nonReclaimableMB int) (bool, int) {
	if !strategyUsesReclaimableMMap(strategy) || nonReclaimableMB <= 0 {
		return false, 0
	}
	expected := strategy.ReclaimableHostWeightsMB
	if expected <= 0 {
		return false, 0
	}
	// CRAM is future anonymous capacity and the planner's host footprint is the
	// non-reclaimable working-set estimate. Permit allocator noise equal to the
	// larger of 1 GiB or one eighth of the expected mapped expert bytes. A real
	// anonymous-copy loader exceeds both the working-set envelope and half the
	// exact expert footprint (the observed ik case was ~113 GiB vs <1 GiB).
	limit := strategy.PlannedHostFootprintMB + strategy.CRAM + maxInt(1024, expected/8)
	return nonReclaimableMB > limit && nonReclaimableMB > expected/2, limit
}

func validateObservedMMapPageability(cfg *config.Config, model *placement.ModelProfile, be *backendInfo, strategy *placement.Strategy, process *server.Process) error {
	if cfg == nil || model == nil || be == nil || process == nil || !strategyUsesReclaimableMMap(strategy) {
		return nil
	}
	measuredMB, err := process.ScopeNonReclaimableMB()
	if err != nil || measuredMB <= 0 {
		// The audited exact-backend capability remains the admission fact when the
		// host cannot expose cgroup memory.stat (non-Linux or non-systemd).
		return nil
	}
	contradicted, limitMB := mmapPageabilityContradicted(strategy, measuredMB)
	if !contradicted {
		fmt.Fprintf(os.Stderr, "[mmap] live pageability check: %d MiB non-reclaimable for %d MiB mapped CPU experts\n",
			measuredMB, strategy.ReclaimableHostWeightsMB)
		return nil
	}
	evidence := fmt.Sprintf(
		"live cgroup contradiction: %d MiB non-reclaimable exceeds %d MiB envelope with %d MiB CPU experts expected file-backed",
		measuredMB, limitMB, strategy.ReclaimableHostWeightsMB,
	)
	be.CPUExpertMMapCapability = placement.CPUExpertMMapAnonymous
	be.CPUExpertMMapEvidence = evidence
	if persistErr := persistBackendCPUExpertMMapCapability(cfg.CacheDir, model, be, placement.CPUExpertMMapAnonymous, evidence); persistErr != nil {
		fmt.Fprintf(os.Stderr, "[mmap] warning: could not persist pageability contradiction: %v\n", persistErr)
	}
	return fmt.Errorf("backend copied CPU experts into anonymous host memory despite mmap (%s)", evidence)
}

// backendStartOptions builds the memory scope for the backend.
//
// MemoryHigh and MemoryMax were the same number, which leaves the kernel no
// band to work in: the reclaim threshold and the kill threshold arrive
// together. That is correct for resident weights, where the bytes are anonymous
// and reclaiming is not an option -- a hard cap is the whole point.
//
// Under mmap it is wrong, and it is what stops MiniMax-M3 from running here.
// Its CPU-side experts are file-backed, so the plan correctly reports only a
// 458 MiB host footprint -- those pages are reclaimable. But page cache is
// still *charged* to the cgroup, so ~107 GiB of experts under a 114 GiB cap
// walked memory.current straight into memory.max and the OOM killer fired
// (measured: cgroup peak 114558 MiB, oom_kill=1) when the kernel could simply
// have dropped clean pages. DeepSeek-V4 survives the same path only because its
// ~80 GiB of experts happen to leave headroom.
//
// So for an mmap-backed plan the plan's own budget becomes the reclaim
// threshold and the hard ceiling moves up to the whole-host utilisation limit
// the user already configures with --ram-limit-percent. Containment is kept --
// there is still a hard ceiling, and MemorySwapMax=0 still holds -- but the
// kernel is allowed to evict page cache before it kills the process.
func backendStartOptions(req *launchRequest, caps *detect.Capabilities, be *backendInfo, envOverrides []string, serverArgs []string) server.StartOptions {
	budgetMB := backendMemoryMaxMB(req, caps)
	highMB, maxMB := budgetMB, budgetMB
	if budgetMB > 0 && argsMMapBackedExperts(be, serverArgs) {
		if hostCeiling := hostReclaimCeilingMB(req, caps); hostCeiling > budgetMB {
			maxMB = hostCeiling
		}
	}
	return server.StartOptions{EnvOverrides: envOverrides, MemoryHighMB: highMB, MemoryMaxMB: maxMB}
}

// resizeScopeToMeasuredFootprint re-sizes the running backend scope's hard
// memory ceiling to the measured non-reclaimable footprint plus headroom
// (Fix B). The pre-launch MemoryMax is a plan estimate; after health + canary
// the backend's real anon+shmem+slab is known, so the ceiling is tightened to
// measured+headroom. A runaway consumer (or a wrong plan) then dies inside its
// own per-instance scope, never the shared server or the machine.
//
// The value is clamped to the whole-host ceiling the user configured (the same
// backendMemoryMaxMB the pre-launch scope used), so this can only tighten
// containment, never loosen it past the safety limit. When --ram-budget is set
// the user named an explicit ceiling and the auto re-size is skipped.
func resizeScopeToMeasuredFootprint(req *launchRequest, caps *detect.Capabilities, strategy *placement.Strategy, p *server.Process) {
	if req == nil || caps == nil || p == nil || runtime.GOOS != "linux" {
		return
	}
	if req.CgroupHeadroomMB <= 0 || req.RamBudgetMB > 0 {
		return
	}
	ceiling := backendMemoryMaxMB(req, caps)
	if ceiling <= 0 {
		return
	}
	measured, err := p.ScopeNonReclaimableMB()
	if err != nil || measured <= 0 {
		if err != nil {
			fmt.Fprintf(os.Stderr, "[launch] measured-footprint cgroup re-size skipped: %v\n", err)
		}
		return
	}
	plannedFloor := measuredFootprintPlannedFloor(strategy, measured, ceiling)
	newMax := measuredFootprintCgroupMaxMB(measured, req.CgroupHeadroomMB, plannedFloor, ceiling)
	if strategyUsesReclaimableMMap(strategy) {
		// Never collapse a file-backed plan's hard ceiling to its anonymous
		// working set. memory.max must retain the reclaim band for clean expert
		// pages; only memory.high is tightened to measured state + headroom.
		if err := p.SetMemoryHighMB(newMax); err != nil {
			fmt.Fprintf(os.Stderr, "[launch] measured-footprint mmap reclaim threshold re-size to %d MiB failed: %v\n", newMax, err)
			return
		}
		fmt.Fprintf(os.Stderr, "[launch] mmap reclaim threshold re-sized to %d MiB (measured %d MiB + headroom, planned floor %d MiB, hard ceiling preserved)\n",
			newMax, measured, plannedFloor)
		return
	}
	if err := p.SetMemoryMaxMB(newMax); err != nil {
		fmt.Fprintf(os.Stderr, "[launch] measured-footprint cgroup re-size to %d MiB failed: %v\n", newMax, err)
		return
	}
	fmt.Fprintf(os.Stderr, "[launch] memory scope re-sized to %d MiB (measured %d MiB + headroom, planned floor %d MiB, ceiling %d MiB)\n",
		newMax, measured, plannedFloor, ceiling)
}

// measuredFootprintPlannedFloor is the plan's host-footprint floor the
// measured-footprint resize preserves (Fix B). The canary has not populated the
// prompt cache yet, so the ceiling must keep room for the plan's priced host
// footprint plus the CRAM capacity it left available for later conversations.
//
// When the plan derived no host footprint (PlannedHostFootprintMB == 0 -- the
// fully-resident placements priced no host cost until the dense builders started
// deriving one; DenseCPUOffload / CPUOnly still do not), the floor must NOT
// collapse to bare CRAM: sizing the scope to the prompt-cache budget clamps the
// cgroup DOWN below the backend's real resident footprint and the server OOMs
// against its own scope on a long prompt (the Qwen3.8 27B crash: Fix-B wrote
// ~11 GB = CRAM as the ceiling, then the server was killed at 10.9 GB on a
// 36k-token prompt). The whole-host ceiling is the only honest fallback -- the
// resize then only ever tightens to measured+headroom, never below the
// pre-launch plan ceiling, and the ceiling clamp keeps the user's safety limit
// absolute. When the plan under-counts the measured backend (CRAM not yet
// filled, checkpoint reserve below reality), measured+headroom is the honest
// floor, not the stale plan estimate.
func measuredFootprintPlannedFloor(strategy *placement.Strategy, measuredMB, ceilingMB int) int {
	if strategy == nil {
		return 0
	}
	if strategy.PlannedHostFootprintMB <= 0 {
		// No plan-derived host cost: bare CRAM is not a plan floor. Preserve the
		// pre-launch whole-host ceiling so Fix-B only ever raises/keeps it for
		// this plan -- never clamps the scope below the plan ceiling.
		return ceilingMB
	}
	plannedFloor := strategy.PlannedHostFootprintMB + strategy.CRAM
	if measuredMB > 0 && plannedFloor < measuredMB {
		// The plan under-counts the measured backend. measured+headroom is the
		// honest floor, not the stale plan estimate.
		return measuredMB
	}
	return plannedFloor
}

// measuredFootprintCgroupMaxMB is the pure sizing rule for the measured-footprint
// cgroup (Fix B): the larger of measured non-reclaimable footprint + headroom
// and the plan's future capacity requirement, clamped to the whole-host ceiling.
// The plan floor matters because the first canary has not filled CRAM yet.
func measuredFootprintCgroupMaxMB(measuredMB, headroomMB, plannedFloorMB, ceilingMB int) int {
	if measuredMB <= 0 || headroomMB <= 0 {
		return 0
	}
	newMax := measuredMB + headroomMB
	if plannedFloorMB > newMax {
		newMax = plannedFloorMB
	}
	if newMax > ceilingMB {
		return ceilingMB
	}
	return newMax
}

// hostReclaimCeilingMB is the hard ceiling an mmap-backed plan may reach before
// the OOM killer is the right answer.
//
// --ram-limit-percent expresses a whole-host *utilisation* target. For
// reclaimable page cache that is a pressure threshold, not a death sentence: it
// belongs on MemoryHigh, where it makes the kernel drop clean pages. The hard
// ceiling is then the RAM that was actually free, which is the physical bound
// the plan could reach in any case. Containment is preserved -- a runaway
// anonymous allocation still meets a real limit, and MemorySwapMax=0 still
// holds, so nothing spills to disk.
//
// An explicit --ram-budget is a ceiling the user named, so it is left alone;
// only the derived percent target is reinterpreted as reclaim pressure.
func hostReclaimCeilingMB(req *launchRequest, caps *detect.Capabilities) int {
	if req == nil || caps == nil || runtime.GOOS != "linux" {
		return 0
	}
	if req.RamBudgetMB > 0 || req.RAMLimitPercent <= 0 {
		return 0
	}
	ceiling := caps.RAM.FreeMB
	if req.RAMHeadroomMB > 0 {
		ceiling -= req.RAMHeadroomMB
	}
	if ceiling <= 0 {
		return 0
	}
	return ceiling
}

func startLaunchProcess(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, serverArgs []string, timeout time.Duration) (*server.Process, error) {
	startOpts := backendStartOptions(req, caps, be, hostExpertPinningEnv(be, serverArgs), serverArgs)
	if req.ClaudeCode {
		// In Claude Code mode ggrun hands the terminal to the `claude` client, so
		// the backend's ongoing per-request logs must go to a file instead of
		// bleeding into Claude Code's UI.
		scope := claudeLaunchLogScope(req, model, be, serverArgs)
		logPath := claudeServerLogPath(cfg, req.Port, scope)
		// Append, never truncate. The scope is a hash of the launch shape, not of
		// one invocation, so every launch and every retry that lands on the same
		// shape shares this path — and os.Create erased the previous attempt at
		// the moment the next one started. That is the second of two truncation
		// sites (pkg/recovery/recovery.go runOnce is the other); fixing only one
		// still lost the history, as the 2026-08-02 23:11 relaunch showed.
		if lf, ferr := os.OpenFile(logPath, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644); ferr == nil {
			_, _ = fmt.Fprintf(lf, "[ggrun] launch-scope: %s\n", scope)
			// The scope is a hash, so a mismatch between two runs is invisible
			// from the filenames alone -- diagnosing one cost an afternoon.
			// Write the parts it is made of.
			_, _ = fmt.Fprintf(lf, "[ggrun] launch-identity: workload=%s backend=%s shape=%s\n",
				requestWorkloadProfile(req, model), evidenceBackendCacheTag(be), recoveryLaunchIdentity(serverArgs))
			_, _ = fmt.Fprintf(lf, "[ggrun] launch: %s\n", formatCommand(serverArgs))
			fmt.Printf("[claude-code] backend logs -> %s\n", logPath)
			return server.StartWithTimeoutToOptions(serverArgs, req.Port, timeout, lf, lf, startOpts)
		}
	}
	return server.StartWithTimeoutToOptions(serverArgs, req.Port, timeout, os.Stdout, os.Stderr, startOpts)
}

// formatMessageDelimiters renders the role markers for the launch line so a
// template whose markers were misread is visible immediately rather than as an
// unexplained absence of prefix reuse.
func formatMessageDelimiters(delims []claudeauto.MessageDelimiter) string {
	parts := make([]string, 0, len(delims))
	for _, d := range delims {
		parts = append(parts, d.Role+"="+d.Delimiter)
	}
	return strings.Join(parts, " ")
}

// promptsForUpdates keeps the check off commands that must stay silent or fast:
// machine-readable output, the updater itself, and trivial queries.
func promptsForUpdates(args []string) bool {
	if len(args) == 0 {
		return false
	}
	switch args[0] {
	case "help", "--help", "-h", "version", "--version", "-v",
		"update", "--update", "self-update", "dry-run", "detect":
		return false
	}
	for _, a := range args {
		if a == "--emit-server-argv-json" || a == "--json" || a == "--dry-run" {
			return false
		}
	}
	return true
}

// printVRAMLedger shows how each GPU's expert budget was spent. Stranded space
// below one expert layer is structural -- a layer is indivisible -- but more
// than that means something reserved it, and every such question so far has
// been answered by reconstructing this arithmetic from the -ot string and
// nvidia-smi after the fact.
func printVRAMLedger(strategy *placement.Strategy) {
	if strategy == nil || len(strategy.VRAMLedger) == 0 {
		return
	}
	for _, e := range strategy.VRAMLedger {
		role := "split-owner"
		if e.ExpertOnly {
			role = "expert-only"
		}
		fmt.Printf("[launch] CUDA%d %-11s free %6d - fixed %6d = room %6d MiB -> %2d expert layers, %5d MiB stranded\n",
			e.GPU, role, e.FreeMB, e.FixedMB, e.RoomMB, e.ExpertLayers, e.StrandedMB)
	}
}

func printOptimizationSummary(label string, strategy *placement.Strategy, detailed bool) {
	for _, line := range optimizationSummaryLines(label, strategy, detailed) {
		fmt.Println(line)
	}
}

func optimizationSummaryLines(label string, strategy *placement.Strategy, detailed bool) []string {
	if strategy == nil || strategy.Residency == "" || strategy.ResourceLedger == nil {
		return nil
	}
	if label == "" {
		label = "optimize"
	}
	ledger := strategy.ResourceLedger
	evidence := ledger.Evidence
	if evidence == "" {
		evidence = "estimate"
	}
	slots := max(1, strategy.Parallel)
	kv := strategy.KVPlacement
	if strategy.KVType != "" {
		if kv == "" {
			kv = strategy.KVType
		} else {
			kv += "/" + strategy.KVType
		}
	}
	if kv == "" {
		kv = "unspecified"
	}
	lines := []string{fmt.Sprintf("[%s] %s, ctx %d total / %d per agent, kv %s, batch %d/%d, parallel %d, bottleneck %s (%s)",
		label, strategy.Residency, strategy.ContextSize, strategy.ContextSize/slots, kv,
		strategy.BatchSize, strategy.UBatchSize, slots, strategy.OptimizationBottleneck, evidence)}
	if boundary := strategy.OptimizationBoundary; boundary != nil {
		lines = append(lines, fmt.Sprintf("[%s] calculated %d candidates (%d feasible, %d exact): batch %d..%d, ubatch %d..%d, parallel %d..%d, %d topology shape(s)",
			label, boundary.CandidateCount, boundary.FeasibleCount, boundary.ExactCount,
			boundary.MinBatch, boundary.MaxBatch, boundary.MinUBatch, boundary.MaxUBatch,
			boundary.MinParallel, boundary.MaxParallel, len(boundary.Topologies)))
	}
	if !detailed {
		return lines
	}
	for _, device := range ledger.Devices {
		lines = append(lines, fmt.Sprintf("[%s] CUDA%d %-11s free %6d - model %6d - KV %5d - graph/runtime %5d = slack %6d MiB [%s]",
			label, device.GPU, device.Role, device.FreeMB, device.ModelMB, device.ContextMB,
			device.GraphMB+device.RuntimeMB, device.SlackMB, device.Evidence))
	}
	lines = append(lines, fmt.Sprintf("[%s] host free %d - required %d = slack %d MiB (reclaimable weights %d MiB) [%s]",
		label, ledger.Host.FreeMB, ledger.Host.RequiredMB, ledger.Host.SlackMB,
		ledger.Host.Reclaimable, ledger.Host.Evidence))
	return lines
}

// serverProcessPID is the backend's PID, or 0 when it is not running.
func serverProcessPID(p *server.Process) int {
	if p == nil || p.Cmd == nil || p.Cmd.Process == nil {
		return 0
	}
	return p.Cmd.Process.Pid
}

const (
	failedLaunchRAMReleaseToleranceMB = 1024
	failedLaunchGPUReleaseToleranceMB = 64
)

// stopFailedLaunchBeforeAdvisor is the boundary between a failed main-model
// start and the optional support model. StartWithTimeoutToOptions performs its
// own best-effort teardown, but its caller must not assume that succeeded: a
// stubborn loader could otherwise overlap a 100+ GiB allocation with the
// helper. Re-stop the process, require it to be gone, and wait for RAM/VRAM to
// settle back to the hardware snapshot taken before companion/main startup.
func stopFailedLaunchBeforeAdvisor(process *server.Process, baseline *detect.Capabilities, timeout time.Duration) error {
	var stopErr error
	if process != nil {
		stopErr = process.Stop()
	}
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	deadline := time.Now().Add(timeout)
	for {
		processGone := process == nil || !process.IsRunning()
		if processGone && launchResourcesAtBaseline(baseline) {
			if stopErr != nil {
				return fmt.Errorf("failed main-model teardown returned an error despite released resources: %w", stopErr)
			}
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("failed main-model process/resources did not return to baseline within %s; refusing to start support expert", timeout)
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func launchResourcesAtBaseline(baseline *detect.Capabilities) bool {
	if baseline == nil {
		return true
	}
	currentGPU := make(map[int]int, len(baseline.GPUs))
	for _, gpu := range baseline.GPUs {
		currentGPU[gpu.Index] = placement.QueryVRAMUsed(gpu.Index)
	}
	return releaseReadingsAtBaseline(baseline, currentAvailableRAMMB(), currentGPU)
}

// captureLaunchResourceBaseline snapshots the host after companions are live
// but before the main backend starts. Calibration/promotion transitions compare
// against this point, not the earlier hardware discovery snapshot; otherwise a
// legitimate worker looked like leaked VRAM, while a just-stopped main process
// could still be occupying memory when the next preflight sampled the cards.
func captureLaunchResourceBaseline(caps *detect.Capabilities) *detect.Capabilities {
	if caps == nil {
		return nil
	}
	baseline := *caps
	baseline.GPUs = append([]detect.GPU(nil), caps.GPUs...)
	baseline.RAM.FreeMB = currentAvailableRAMMB()
	for i := range baseline.GPUs {
		baseline.GPUs[i].VRAMUsedMB = placement.QueryVRAMUsed(baseline.GPUs[i].Index)
	}
	return &baseline
}

func releaseReadingsAtBaseline(baseline *detect.Capabilities, availableRAMMB int, gpuUsedMB map[int]int) bool {
	if baseline == nil {
		return true
	}
	if baseline.RAM.FreeMB > 0 && availableRAMMB > 0 &&
		availableRAMMB < baseline.RAM.FreeMB-failedLaunchRAMReleaseToleranceMB {
		return false
	}
	for _, gpu := range baseline.GPUs {
		if used := gpuUsedMB[gpu.Index]; used > gpu.VRAMUsedMB+failedLaunchGPUReleaseToleranceMB {
			return false
		}
	}
	return true
}

func currentAvailableRAMMB() int {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		kb, err := strconv.Atoi(fields[1])
		if err != nil {
			return 0
		}
		return kb / 1024
	}
	return 0
}

func claudeServerLogPath(cfg *config.Config, port int, scope string) string {
	logDir := ""
	if cfg != nil {
		logDir = cfg.LogDir
	}
	if logDir == "" {
		logDir = os.TempDir()
	}
	return filepath.Join(logDir, fmt.Sprintf("ggrun-claude-server-v2-%d-%s.log", port, scope))
}

func claudeOOMMarkerPath(cfg *config.Config, req *launchRequest, model *placement.ModelProfile, be *backendInfo, serverArgs []string) string {
	if req == nil {
		return ""
	}
	scope := claudeLaunchLogScope(req, model, be, serverArgs)
	return claudeServerLogPath(cfg, req.Port, scope) + ".oom-recorded"
}

func recordMeasuredLaunchProbes(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, serverLog string, baselineVRAMByGPU map[int]int, serverPID int) map[int]int {
	if cfg == nil || model == nil || strategy == nil || be == nil || serverLog == "" {
		return nil
	}
	cacheBackendTag := scopedProbeBackendTagForStrategy(req, model, be, strategy)
	var gpus []detect.GPU
	if caps != nil {
		gpus = caps.GPUs
	}
	if hasExternalSpecCompanion(strategy) {
		return nil
	}
	if model.IsMoE && len(gpus) > 0 {
		// Build the per-GPU companion VRAM map so the system probe can net it
		// out of the breakdown table's unaccounted column. Without this the
		// reviewer/worker seated on a card shows up as permanent CUDA overhead
		// and latches there (the 2916 MiB bug).
		companionVRAMByGPU := map[int]int{}
		for _, cp := range strategy.CompanionPlacements {
			if cp.GPU < 0 {
				continue
			}
			// The NanoBeige worker stores its measured VRAM under its KV-qualified
			// MeasurementKey (claude_auto.go companionMeasurementKey/recordReviewerVRAM),
			// not its public Name; the strategy only carries Name. Resolve the key via
			// the frozen profile so the netting lookup finds the file the writer wrote.
			key := cp.Name
			if req != nil && req.ReviewerProfile != nil && cp.Name == req.ReviewerProfile.Name {
				key = req.ReviewerProfile.companionMeasurementKey()
			}
			if mb := placement.MeasuredCompanionVRAMMB(cfg.CacheDir, key); mb > 0 {
				companionVRAMByGPU[cp.GPU] += mb
			}
		}
		placement.RunPostLaunchProbe(cfg.CacheDir, gpus, serverLog, serverPID, companionVRAMByGPU)
		placement.RunPostLaunchModelProbeVRAMDelta(cfg.CacheDir, model, strategy, cacheBackendTag, gpus, baselineVRAMByGPU)
	}
	computeByGPU := placement.ParseComputeBuffersByGPU(serverLog)
	probeWritten := placement.RunPostLaunchModelProbe(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, cacheBackendTag, gpus, strategy.Parallel, serverLog)
	placement.RecordPostLaunchContextAllocation(cfg.CacheDir, model, strategy, cacheBackendTag, gpus, serverLog, serverPID)
	placement.RunPostLaunchKVProbe(cfg.CacheDir, model, strategy.ContextSize, strategy.KVType, serverLog, strategy.Parallel)
	if !probeWritten {
		return nil
	}
	return computeByGPU
}

// recordLiveCheckpointEvidence consumes the first real checkpoint allocation
// produced by the functional canary. Startup probes stop at health and cannot
// see this host-RAM state, while waiting until the next launch is too late: the
// caller is about to tighten this process's cgroup memory.max. Persist the exact
// observation for future planning and raise this live strategy's host floor
// before that resize happens.
func recordLiveCheckpointEvidence(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, serverLog string) bool {
	if cfg == nil || model == nil || strategy == nil || be == nil || caps == nil || serverLog == "" {
		return false
	}
	obs := placement.ObservePromptCache(serverLog)
	if obs.LargestCheckpointMB <= 0 {
		return false
	}
	tag := scopedProbeBackendTagForStrategy(req, model, be, strategy)
	if err := placement.RecordPromptCacheObservation(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, tag, caps.GPUs, strategy.Parallel, obs); err != nil {
		fmt.Fprintf(os.Stderr, "[launch] checkpoint measurement cache write failed (launch remains conservatively bounded): %v\n", err)
	}
	beforeMB, afterMB, changed := placement.ApplyMeasuredCheckpointObservation(model, strategy, obs)
	if changed {
		fmt.Fprintf(os.Stderr, "[launch] measured %.3f MiB per context checkpoint; raised the active host reserve from %d to %d MiB before cgroup re-size\n",
			obs.LargestCheckpointMB, beforeMB, afterMB)
	}
	return changed
}

func measuredPromotionOptions(req *launchRequest, model *placement.ModelProfile, be *backendInfo, cacheDir string) placement.Options {
	opts := placementOptionsFromRequest(req, model, be, cacheDir)
	opts.SkipPlacementCache = true
	return opts
}

func maybePromoteMeasuredPlacement(req *launchRequest, cfg *config.Config, be *backendInfo, caps *detect.Capabilities, model *placement.ModelProfile, current *placement.Strategy, currentArgs []string, memoryRecovery *launchMemoryRecovery) (*placement.Strategy, []string, bool) {
	if req == nil || cfg == nil || be == nil || caps == nil || model == nil || current == nil || !model.IsMoE || len(caps.GPUs) == 0 {
		return nil, nil, false
	}
	if req.Calibrate == calibrateOff {
		return nil, nil, false
	}
	// A validated .place entry is already the lifecycle winner for this exact
	// launch shape. Auto mode may consume new measurements for the next planner
	// pass, but must not stop that server to challenge it with an unproven denser
	// layout. --calibrate on remains the explicit opt-in to do so.
	if current.PlacementCacheHit && req.Calibrate != calibrateOn {
		return nil, nil, false
	}
	// A measured KV probe may have been written after the first load. Force the
	// recompute to reload it instead of reusing the pre-launch model struct state.
	// Also bypass the placement cache: reloading the placement that just launched
	// made this calibration pass incapable of filling newly proven free VRAM.
	// This was especially visible when the Claude reviewer changed the baseline:
	// a safe but sparse five-block cache kept winning even when six blocks fit.
	model.MeasuredKVBytesPerTok = nil
	model.MeasuredKVGeometry = nil
	opts := measuredPromotionOptions(req, model, be, cfg.CacheDir)
	next, err := placement.Compute(caps, model, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[launch] calibration: measured placement recompute failed: %v\n", err)
		return nil, nil, false
	}
	claudeCodeSlotAdjust(next, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	if !shouldPromoteMoEPlacement(current, next) {
		return nil, nil, false
	}
	nextArgs := buildLaunchServerArgs(req, cfg, be, caps, model, next)
	if formatCommand(nextArgs) == formatCommand(currentArgs) {
		return nil, nil, false
	}
	if memoryRecovery.isRejected(nextArgs) {
		fmt.Fprintln(os.Stderr, "[launch] allocation measurement proposed an argv already rejected by this launch; retaining the proven-safe placement")
		return nil, nil, false
	}
	return next, nextArgs, true
}

// retainRecoveredBaselineForAllocationPromotion keeps the allocator's denser
// recompute monotonic. Once this lifecycle needed memory recovery, it must not
// immediately promote another allocation-only guess. The workload optimizer is
// separate: it may measure the recovered server in place and challenge one
// non-rejected finalist when the resolved launch has performance headroom.
func retainRecoveredBaselineForAllocationPromotion(req *launchRequest, memoryRecovery *launchMemoryRecovery) bool {
	if memoryRecovery == nil || !memoryRecovery.hasRejections() {
		return false
	}
	return req == nil || req.Calibrate != calibrateOn
}

func startLaunchWithCUDAOOMRecovery(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, serverArgs []string, timeout time.Duration) (*server.Process, *placement.Strategy, []string, error) {
	return startLaunchWithCUDAOOMRecoveryState(req, cfg, model, strategy, be, caps, serverArgs, timeout, newLaunchMemoryRecovery())
}

// startLaunchWithCUDAOOMRecoveryState is the sole production start boundary.
// The caller owns memoryRecovery for the complete launch lifecycle so initial
// load, measured promotion, calibration, restoration, and runtime recovery can
// never forget an argv that an earlier phase disproved.
func startLaunchWithCUDAOOMRecoveryState(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, serverArgs []string, timeout time.Duration, memoryRecovery *launchMemoryRecovery) (*server.Process, *placement.Strategy, []string, error) {
	return startLaunchWithCUDAOOMRecoveryStateMode(req, cfg, model, strategy, be, caps, serverArgs, timeout, memoryRecovery, false, false)
}

// startLaunchExactAdmission starts one optimizer challenger as the exact argv
// that candidate computed. A memory miss records the rejection and returns; it
// does not walk the fit recovery ladder (more CPU experts, derated ubatch, or
// a different checkpoint step). The caller restores the measured baseline.
func startLaunchExactAdmission(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, serverArgs []string, timeout time.Duration, memoryRecovery *launchMemoryRecovery) (*server.Process, *placement.Strategy, []string, error) {
	return startLaunchWithCUDAOOMRecoveryStateMode(req, cfg, model, strategy, be, caps, serverArgs, timeout, memoryRecovery, false, true)
}

// exactAdmissionClass names one rewrite the fit recovery ladder is allowed to
// perform, and that an optimizer challenger must refuse instead.
type exactAdmissionClass string

const (
	exactAdmissionSpec       exactAdmissionClass = "speculation"
	exactAdmissionCompat     exactAdmissionClass = "compatibility"
	exactAdmissionCompanion  exactAdmissionClass = "companion"
	exactAdmissionMemory     exactAdmissionClass = "memory"
	exactAdmissionMMap       exactAdmissionClass = "mmap"
	exactAdmissionHotExperts exactAdmissionClass = "hot-experts"
	exactAdmissionCUDAOOM    exactAdmissionClass = "cuda-oom"
)

// exactAdmissionFailure distinguishes a deterministic refusal to run the
// candidate's exact argv from an ordinary start error such as a health timeout,
// interrupted load, or transient backend failure. Only this typed class may be
// persisted as negative automatic-calibration evidence.
type exactAdmissionFailure struct {
	class   exactAdmissionClass
	message string
	cause   error
}

func (e *exactAdmissionFailure) Error() string {
	if e == nil {
		return ""
	}
	if e.cause != nil {
		return e.message + ": " + e.cause.Error()
	}
	return e.message
}

func (e *exactAdmissionFailure) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

func isStableExactAdmissionFailure(err error) bool {
	var failure *exactAdmissionFailure
	return errors.As(err, &failure)
}

func exactAdmissionError(class exactAdmissionClass, detail string, cause error) error {
	message := ""
	switch class {
	case exactAdmissionSpec:
		message = "exact candidate requires a speculation re-plan; refusing to change its argv"
	case exactAdmissionCompat:
		message = fmt.Sprintf("exact candidate requires backend compatibility adjustment (%s); refusing to change its argv", detail)
	case exactAdmissionCompanion:
		message = "exact candidate's speculative companion was rejected; refusing a target-only argv rewrite"
	case exactAdmissionMemory:
		message = fmt.Sprintf("exact candidate failed memory admission%s; refusing recovery ladder", detail)
	case exactAdmissionMMap:
		message = "exact candidate contradicted mmap pageability; refusing a resident argv rewrite"
	case exactAdmissionHotExperts:
		message = fmt.Sprintf("exact candidate did not activate its hot-expert cache%s; refusing to benchmark a cache-free fallback", detail)
	case exactAdmissionCUDAOOM:
		message = fmt.Sprintf("exact candidate CUDA OOM%s; refusing recovery ladder", detail)
	default:
		message = fmt.Sprintf("exact candidate required an argv rewrite (%s); refusing recovery ladder", class)
	}
	return &exactAdmissionFailure{class: class, message: message, cause: cause}
}

func automaticHotExpertFallbackAllowed(req *launchRequest, strategy *placement.Strategy) bool {
	return req != nil && strategy != nil && strategy.HotExpertCacheSlots > 0 &&
		strings.EqualFold(strings.TrimSpace(req.HotExperts), "auto")
}

func hotExpertFailureDecisionSource(req *launchRequest, cfg *config.Config, model *placement.ModelProfile,
	be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy,
	pending *placement.CalibrationDecision,
) *placement.CalibrationDecision {
	var applied *placement.CalibrationDecision
	if req != nil {
		applied = req.AppliedCalibration
	}
	for _, decision := range []*placement.CalibrationDecision{pending, applied} {
		if decision != nil && decision.ScopeKey != "" {
			copy := *decision
			return &copy
		}
	}
	if cfg == nil {
		return nil
	}
	cacheFree := placement.WithoutHotExpertCache(strategy)
	if cacheFree == nil {
		return nil
	}
	key := calibrationScopeKey(req, model, be, caps, cacheFree)
	decision, err := placement.LoadCalibrationDecision(cfg.CacheDir, key)
	if err != nil {
		return nil
	}
	return decision
}

func invalidateAutomaticHotExpertEvidence(req *launchRequest, cfg *config.Config, model *placement.ModelProfile,
	be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy,
) error {
	if req == nil {
		return nil
	}
	cacheDir := ""
	if cfg != nil {
		cacheDir = cfg.CacheDir
	}
	var failures []error
	if key := verifiedConfigScopeKey(req, model, be, caps); key != "" {
		if err := placement.DeleteVerifiedConfig(cacheDir, key); err != nil {
			failures = append(failures, fmt.Errorf("delete cache-on verified config: %w", err))
		}
	}
	calibrationKeys := map[string]bool{}
	if cacheFree := placement.WithoutHotExpertCache(strategy); cacheFree != nil {
		if key := calibrationScopeKey(req, model, be, caps, cacheFree); key != "" {
			calibrationKeys[key] = true
		}
	}
	if req.AppliedCalibration != nil && req.AppliedCalibration.ScopeKey != "" {
		calibrationKeys[req.AppliedCalibration.ScopeKey] = true
	}
	for key := range calibrationKeys {
		if err := placement.DeleteCalibrationDecision(cacheDir, key); err != nil {
			failures = append(failures, fmt.Errorf("delete cache-on calibration decision: %w", err))
		}
	}
	req.AppliedCalibration = nil
	req.HotExpertsRuntimeDisabled = true
	return errors.Join(failures...)
}

func hotExpertUnavailableDecision(source *placement.CalibrationDecision, strategy *placement.Strategy, reason string) *placement.CalibrationDecision {
	if source == nil || source.ScopeKey == "" {
		return nil
	}
	decision := *source
	decision.Winner = "default"
	decision.ValidationLevel = placement.CalibrationValidationAdmission
	decision.WinnerTPS = decision.DefaultTPS
	decision.WinnerPromptTPS = decision.DefaultPromptTPS
	decision.WinnerMixedTPS = decision.DefaultMixedTPS
	decision.WinnerTurnTimeS = decision.DefaultTurnTimeS
	decision.WinnerTurnMaxS = decision.DefaultTurnMaxS
	decision.WinnerAgentSamples = decision.DefaultAgentSamples
	decision.WinnerWorkloadLanes = decision.DefaultWorkloadLanes
	decision.WinnerCachedTokens = decision.DefaultCachedTokens
	decision.WinnerNewPromptTokens = decision.DefaultNewPromptTokens
	decision.WinnerScore = decision.DefaultScore
	decision.Improvement = 0
	decision.FinalistOutcome = "unavailable"
	decision.FinalistFailureClass = "hot-expert-lifecycle"
	decision.FinalistFailureReason = reason
	decision.MeasuredAt = ""
	if decision.Finalist == "" && strategy != nil && strategy.HotExpertCacheSlots > 0 {
		decision.Finalist = fmt.Sprintf("hot-experts-%d", strategy.HotExpertCacheSlots)
	}
	return &decision
}

// validateExactAdmissionArgv is the backstop for every challenger admission
// path. Individual rewrite sites fail with a specific reason; this invariant
// catches both future sites and an accidental in-place slice mutation before a
// changed process can start under the original candidate's evidence.
func validateExactAdmissionArgv(exact bool, expected string, actual []string) error {
	if !exact {
		return nil
	}
	got := formatCommand(actual)
	if got != expected {
		return fmt.Errorf("exact candidate argv changed during admission; expected %s, got %s", expected, got)
	}
	return nil
}

// restoreLaunchWithCUDAOOMRecoveryState re-enters the start boundary to bring
// back a placement the OUTER launcher has unconditionally committed to (the
// failed-promotion fallback in cmdLaunch and the calibration default/winner
// restarts). The isRejected gate exists to keep the memory-recovery loop from
// resurrecting an argv it disproved; a deliberate restore carries an explicit
// Strategy argument whose loading the caller falls back to no matter what, so
// re-gating it would only convert a working fallback into "die with no server".
// Recovery rejections remain recorded and enforced everywhere the argv is chosen
// by recovery/recompute.
func restoreLaunchWithCUDAOOMRecoveryState(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, serverArgs []string, timeout time.Duration, memoryRecovery *launchMemoryRecovery) (*server.Process, *placement.Strategy, []string, error) {
	return startLaunchWithCUDAOOMRecoveryStateMode(req, cfg, model, strategy, be, caps, serverArgs, timeout, memoryRecovery, true, false)
}

type tunedBatchConstraint struct {
	enabled          bool
	batch            int
	ubatch           int
	performanceTuned bool
}

func tunedBatchConstraintFor(strategy *placement.Strategy) tunedBatchConstraint {
	if strategy == nil || !strategy.BatchTuned || strategy.BatchSize <= 0 || strategy.UBatchSize <= 0 {
		return tunedBatchConstraint{}
	}
	return tunedBatchConstraint{
		enabled:          true,
		batch:            strategy.BatchSize,
		ubatch:           strategy.UBatchSize,
		performanceTuned: strategy.PerformanceTuned,
	}
}

func (constraint tunedBatchConstraint) apply(opts *placement.Options) {
	if !constraint.enabled || opts == nil {
		return
	}
	opts.BatchSize = constraint.batch
	opts.UBatchSize = constraint.ubatch
	opts.BatchSizeExplicit = true
	opts.UBatchSizeExplicit = true
	opts.RuntimePolicy = nil
}

func (constraint tunedBatchConstraint) retain(strategy *placement.Strategy) *placement.Strategy {
	if strategy != nil && constraint.enabled && strategy.BatchSize == constraint.batch && strategy.UBatchSize == constraint.ubatch {
		strategy.BatchTuned = true
		strategy.PerformanceTuned = constraint.performanceTuned
	}
	return strategy
}

// backendMeasuredRecomputeWorthVerifying keeps exact allocator evidence tied
// to the argv that produced it. A newly denser MoE placement is worth another
// contained admission pass; lateral split churn at the same expert residency
// is not. In the latter case the already-proven placement is both safer and no
// less substantiated as a performance choice. Planned/no-allocation evidence
// still has to converge through the ordinary recompute loop.
func backendMeasuredRecomputeWorthVerifying(level memoryEvidenceLevel, current, next *placement.Strategy) bool {
	if level != memoryEvidenceAllocated || current == nil || next == nil {
		return true
	}
	if current.Type == placement.MoEOffload && next.Type == placement.MoEOffload {
		return shouldPromoteMoEPlacement(current, next)
	}
	return true
}

func startLaunchWithCUDAOOMRecoveryStateMode(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, serverArgs []string, timeout time.Duration, memoryRecovery *launchMemoryRecovery, restoreExempt bool, exactAdmission bool) (launchProcess *server.Process, launchStrategy *placement.Strategy, launchArgs []string, launchErr error) {
	const maxRetries = 2
	const maxPreflightReplans = 5
	retries := 0
	preflightReplans := 0
	oomPenalty := map[int]int{}
	if memoryRecovery == nil {
		memoryRecovery = newLaunchMemoryRecovery()
	}
	specDisabled := false
	measuredProductionArgs := ""
	exactCandidateArgs := ""
	if exactAdmission {
		exactCandidateArgs = formatCommand(serverArgs)
	}
	// A calibration winner/candidate owns an exact batch pair even when the user
	// did not spell that pair on the command line. Backend allocation evidence
	// can force a fresh Compute pass before the real server starts; preserve the
	// measured candidate as a placement input through that pass instead of
	// letting the ordinary Claude cold-start policy collapse it back to 128/128.
	tunedBatch := tunedBatchConstraintFor(strategy)
	retainTunedBatch := func(next *placement.Strategy) *placement.Strategy {
		return tunedBatch.retain(next)
	}
	runtimeCaps, visibleToPhysical := runtimeGPUCapabilitiesForLaunch(caps, req, strategy)
	placementOpts := func() placement.Options {
		opts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
		tunedBatch.apply(&opts)
		if specDisabled {
			opts.SpecMode = "off"
		}
		return opts
	}
	// A verified-config reuse hit that fails at any point in the launch boundary
	// (preflight rejection, spec mismatch, compatibility adjustment, start error)
	// is evidence the saved config is no longer safe. Delete the record so the
	// next launch re-derives from scratch — "never retry a broken config blindly"
	// holds for the full-config layer just as it does for the MoE .place cache
	// (support.go:559-576). The deferred check covers every error return and only
	// fires when the launch actually failed (the named launchErr is non-nil).
	reusedFromVerified := strategy != nil && strategy.VerifiedConfigReused
	defer func() {
		if reusedFromVerified && launchErr != nil && req != nil && cfg != nil && model != nil && be != nil && caps != nil {
			if key := verifiedConfigScopeKey(req, model, be, caps); key != "" {
				if delErr := placement.DeleteVerifiedConfig(cfg.CacheDir, key); delErr != nil {
					fmt.Fprintf(os.Stderr, "[launch] warning: could not delete stale verified config: %v\n", delErr)
				} else {
					fmt.Fprintln(os.Stderr, "[verified] verified config reuse hit failed to launch; record deleted, re-deriving fresh")
				}
			}
		}
	}()
	for {
		if err := validateExactAdmissionArgv(exactAdmission, exactCandidateArgs, serverArgs); err != nil {
			return nil, strategy, serverArgs, err
		}
		if !restoreExempt && memoryRecovery.isRejected(serverArgs) {
			return nil, strategy, serverArgs, fmt.Errorf("refusing to retry a memory configuration rejected earlier in this launch lifecycle")
		}
		if !specDisabled && strings.EqualFold(strings.TrimSpace(req.SpecMode), "auto") && strategy != nil && strategy.Draft != nil && strategy.Draft.Type != placement.DraftNone {
			verified := strategy.Draft.VerifiedLaunchIdentity
			if verified == "" || verified != specLaunchIdentity(serverArgs) {
				if exactAdmission {
					return nil, strategy, serverArgs, exactAdmissionError(exactAdmissionSpec, "", nil)
				}
				fmt.Fprintln(os.Stderr, "[spec] final launch flags differ from the verified profile; disabling speculation")
				specDisabled = true
				next, rerr := placement.Compute(caps, model, placementOpts())
				if rerr != nil || next == nil {
					if rerr != nil {
						return nil, strategy, serverArgs, fmt.Errorf("speculative profile mismatch and target-only re-plan failed: %w", rerr)
					}
					return nil, strategy, serverArgs, fmt.Errorf("speculative profile mismatch and target-only re-plan returned no strategy")
				}
				next = retainTunedBatch(next)
				strategy = next
				serverArgs = buildLaunchServerArgs(req, cfg, be, caps, model, next)
				continue
			}
		}
		// Ask the backend's no-alloc accounting whether this placement can even
		// load (~1s) before committing to a real load (15+ min for a big MoE).
		// A measured deficit re-plans exactly like a startup CUDA OOM would —
		// without paying for the load to learn it. Re-planned args loop back
		// here, so every retry is re-gated too.
		if strategy != nil {
			preflight := preflightPlacement(req, be, &configForPreflight{CacheDir: cfg.CacheDir}, runtimeCaps, model, strategy, serverArgs)
			if adjustment := preflight.BackendAdjustment; adjustment != nil {
				if exactAdmission {
					return nil, strategy, serverArgs, exactAdmissionError(exactAdmissionCompat, adjustment.Reason, nil)
				}
				if preflightReplans >= maxPreflightReplans {
					return nil, strategy, serverArgs, fmt.Errorf("backend compatibility adjustment did not converge after %d retries", maxPreflightReplans)
				}
				switch {
				case adjustment.RemoveFlag != "":
					if userExplicitBackendFlag(req, adjustment.RemoveFlag) {
						// Advisory adjustments never reach here: preflight
						// resolves the user-explicit case itself, since the
						// backend started and there is nothing to fail closed on.
						return nil, strategy, serverArgs, fmt.Errorf(
							"backend rejected explicitly supplied %s: %s; refusing to change a user-supplied backend flag",
							adjustment.RemoveFlag, adjustment.Reason,
						)
					}
					if !hasArg(serverArgs, adjustment.RemoveFlag) {
						return nil, strategy, serverArgs, fmt.Errorf(
							"backend requested removal of %s, but that flag is not present in the launch",
							adjustment.RemoveFlag,
						)
					}
					if persistErr := persistBackendCapability(cfg.CacheDir, model, be, adjustment.RemoveFlag, adjustment.Reason, false); persistErr != nil {
						fmt.Fprintf(os.Stderr, "[launch] warning: could not persist measured backend capability: %v\n", persistErr)
					}
					if !disableBackendFlag(req, adjustment.RemoveFlag, adjustment.Reason) {
						return nil, strategy, serverArgs, fmt.Errorf(
							"backend compatibility adjustment for %s repeated without changing the launch",
							adjustment.RemoveFlag,
						)
					}
					if adjustment.RemoveFlag == "--swa-full" {
						req.ExtraArgs = setPassthroughBoolFlag(req.ExtraArgs, "--swa-full", false)
					}
					if adjustment.RequireReplan {
						opts := placementOpts()
						opts.SkipPlacementCache = true
						next, replanErr := placement.Compute(caps, model, opts)
						if replanErr != nil || next == nil {
							if replanErr != nil {
								return nil, strategy, serverArgs, fmt.Errorf("backend feature re-plan failed: %w", replanErr)
							}
							return nil, strategy, serverArgs, fmt.Errorf("backend feature re-plan returned no strategy")
						}
						next = retainTunedBatch(next)
						claudeCodeSlotAdjust(next, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
						nextArgs := buildLaunchServerArgs(req, cfg, be, caps, model, next)
						if formatCommand(nextArgs) == formatCommand(serverArgs) {
							return nil, strategy, serverArgs, fmt.Errorf("backend feature re-plan for %s produced no argument change", adjustment.RemoveFlag)
						}
						if validateErr := validateBackendLaunchArgs(be, nextArgs); validateErr != nil {
							return nil, strategy, serverArgs, validateErr
						}
						preflightReplans++
						strategy, serverArgs = next, nextArgs
						fmt.Fprintf(os.Stderr,
							"[launch] backend/model compatibility: %s; disabling %s, fully recomputing placement, and retrying\n",
							adjustment.Reason, adjustment.RemoveFlag,
						)
						continue
					}
					nextArgs := applyRequestDisabledBackendFlags(serverArgs, req)
					if formatCommand(nextArgs) == formatCommand(serverArgs) {
						return nil, strategy, serverArgs, fmt.Errorf(
							"backend compatibility adjustment for %s produced no argument change",
							adjustment.RemoveFlag,
						)
					}
					preflightReplans++
					serverArgs = nextArgs
					fmt.Fprintf(os.Stderr,
						"[launch] backend/model compatibility: %s; disabling %s and retrying the measured placement\n",
						adjustment.Reason, adjustment.RemoveFlag,
					)
					continue

				case adjustment.KVQualityV != "":
					targetV, vErr := placement.NormalizeKVType(adjustment.KVQualityV)
					if vErr != nil {
						return nil, strategy, serverArgs, fmt.Errorf("backend requested invalid KV V-cache compatibility type %q: %w", adjustment.KVQualityV, vErr)
					}
					// An explicit user-supplied --cache-type-v is not ggrun's to
					// override: the backend rejected it, so fail closed rather than
					// silently rewrite an explicit user flag (same policy as the
					// RemoveFlag case above).
					if req.KVTypeV != "" {
						return nil, strategy, serverArgs, fmt.Errorf(
							"backend rejected explicitly supplied --cache-type-v %s: %s; refusing to change a user-supplied backend flag",
							req.KVTypeV, adjustment.Reason,
						)
					}
					previousV := "auto"
					if strategy != nil && strategy.KVTypeV != "" {
						previousV = strategy.KVTypeV
					} else if strategy != nil {
						previousV = strategy.KVType
					}
					if currentV, _ := placement.NormalizeKVType(req.KVQualityV); currentV == targetV {
						return nil, strategy, serverArgs, fmt.Errorf("backend compatibility adjustment repeated for V cache type %s without changing the launch", targetV)
					}
					req.KVQualityV = targetV
					next, replanErr := placement.Compute(caps, model, placementOpts())
					if replanErr != nil || next == nil {
						if replanErr != nil {
							return nil, strategy, serverArgs, fmt.Errorf("backend-compatible V-cache re-plan failed: %w", replanErr)
						}
						return nil, strategy, serverArgs, fmt.Errorf("backend-compatible V-cache re-plan returned no strategy")
					}
					next = retainTunedBatch(next)
					claudeCodeSlotAdjust(next, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
					nextArgs := buildLaunchServerArgs(req, cfg, be, caps, model, next)
					if formatCommand(nextArgs) == formatCommand(serverArgs) {
						return nil, strategy, serverArgs, fmt.Errorf("backend-compatible V-cache re-plan produced no argument change")
					}
					if validateErr := validateBackendLaunchArgs(be, nextArgs); validateErr != nil {
						return nil, strategy, serverArgs, validateErr
					}
					preflightReplans++
					strategy, serverArgs = next, nextArgs
					fmt.Fprintf(os.Stderr,
						"[launch] backend/model compatibility: %s; promoting V cache %s -> %s, recomputing placement, and retrying\n",
						adjustment.Reason, previousV, targetV,
					)
					continue
				case adjustment.KVQuality != "":
					targetKV, kvErr := placement.NormalizeKVType(adjustment.KVQuality)
					if kvErr != nil {
						return nil, strategy, serverArgs, fmt.Errorf("backend requested invalid KV compatibility type %q: %w", adjustment.KVQuality, kvErr)
					}
					if currentKV, _ := placement.NormalizeKVType(req.KVQuality); currentKV == targetKV {
						return nil, strategy, serverArgs, fmt.Errorf("backend compatibility adjustment repeated for KV type %s without changing the launch", targetKV)
					}
					previousKV := strategy.KVType
					req.KVQuality = targetKV
					req.KVTypeK, req.KVTypeV = "", ""
					next, replanErr := placement.Compute(caps, model, placementOpts())
					if replanErr != nil || next == nil {
						if replanErr != nil {
							return nil, strategy, serverArgs, fmt.Errorf("backend-compatible KV re-plan failed: %w", replanErr)
						}
						return nil, strategy, serverArgs, fmt.Errorf("backend-compatible KV re-plan returned no strategy")
					}
					next = retainTunedBatch(next)
					claudeCodeSlotAdjust(next, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
					nextArgs := buildLaunchServerArgs(req, cfg, be, caps, model, next)
					if formatCommand(nextArgs) == formatCommand(serverArgs) {
						return nil, strategy, serverArgs, fmt.Errorf("backend-compatible KV re-plan produced no argument change")
					}
					if validateErr := validateBackendLaunchArgs(be, nextArgs); validateErr != nil {
						return nil, strategy, serverArgs, validateErr
					}
					preflightReplans++
					strategy, serverArgs = next, nextArgs
					fmt.Fprintf(os.Stderr,
						"[launch] backend/model compatibility: %s; changing KV %s -> %s, recomputing placement, and retrying\n",
						adjustment.Reason, previousKV, targetKV,
					)
					continue
				default:
					return nil, strategy, serverArgs, fmt.Errorf("backend compatibility adjustment had no supported action: %s", adjustment.Reason)
				}
			}
			if preflight.Err != nil {
				if consent, ok := preflight.Err.(*liveMemoryProbeConsentError); ok {
					if err := confirmLiveMemoryProbe(req, consent.Reason, os.Stdin, os.Stderr, stdinIsTerminal()); err != nil {
						return nil, strategy, serverArgs, err
					}
					rememberLiveMemoryProbeConsent(cfg, os.Stderr)
					continue
				}
				// A backend/model error that no ggrun rule classified (no OOM, no
				// backendAdjustmentFromLog match, no known flag) is the launch-failure
				// case the advisor exists for. Consult it for a DIAGNOSIS ONLY: the
				// consultation never mutates the request or argv, and the launch still
				// fails closed exactly as it would have without the advisor.
				var unclassified *backendUnclassifiedProbeError
				if errors.As(preflight.Err, &unclassified) {
					adviseUnclassifiedLaunchFailure(req, cfg, model, be, caps, unclassified.LogExcerpt)
				}
				return nil, strategy, serverArgs, fmt.Errorf("memory preflight failed closed: %w", preflight.Err)
			}
			if preflight.ProbeUnavailable != "" {
				fmt.Fprintf(os.Stderr, "[launch] %s; continuing on the planner's estimate\n", preflight.ProbeUnavailable)
			}
			if preflight.CompanionRejected {
				if exactAdmission {
					return nil, strategy, serverArgs, exactAdmissionError(exactAdmissionCompanion, "", nil)
				}
				specDisabled = true
				opts := placementOpts()
				opts.SkipPlacementCache = false
				next, rerr := placement.Compute(caps, model, opts)
				if rerr != nil || next == nil {
					if rerr != nil {
						return nil, strategy, serverArgs, fmt.Errorf("selected backend rejected speculative companion and target-only re-plan failed: %w", rerr)
					}
					return nil, strategy, serverArgs, fmt.Errorf("selected backend rejected speculative companion and target-only re-plan returned no strategy")
				}
				next = retainTunedBatch(next)
				strategy = next
				serverArgs = buildLaunchServerArgs(req, cfg, be, caps, model, next)
				fmt.Fprintln(os.Stderr, "[launch] continuing with stable target-only serving")
				continue
			}
			if preflight.DoesNotFit {
				memoryRecovery.reject(serverArgs)
				if exactAdmission {
					return nil, strategy, serverArgs, exactAdmissionError(exactAdmissionMemory, fmt.Sprintf(" on CUDA%d (%d MiB deficit)", preflight.Device, preflight.DeficitMB), nil)
				}
				if preflightReplans >= maxPreflightReplans {
					return nil, strategy, serverArgs, fmt.Errorf("memory preflight did not converge after %d re-plans; refusing a real model load", maxPreflightReplans)
				}
				preflightReplans++
				next, nextArgs, method, rerr := recoverPreflightOOM(
					req, cfg, model, be, caps, runtimeCaps, visibleToPhysical,
					strategy, serverArgs, oomPenalty, preflight,
				)
				if rerr != nil {
					return nil, strategy, serverArgs, fmt.Errorf("memory preflight recovery failed closed: %w", rerr)
				}
				strategy, serverArgs = next, nextArgs
				fmt.Fprintf(os.Stderr,
					"[launch] preflight %s after CUDA%d allocation %d MiB (deficit %d MiB, ctx=%d, n-cpu-moe=%d, ubatch=%d)\n",
					method, preflight.Device, preflight.AllocMB, preflight.DeficitMB, strategy.ContextSize, strategy.NCPUMoE, strategy.UBatchSize,
				)
				continue
			}
			if preflight.Evidence.Level != memoryEvidenceNone && exactAdmission {
				// The preflight measured this exact argv. Challenger admission must
				// consume that proof directly; feeding it back through Compute can
				// produce a different split and turn one bounded experiment into an
				// unrequested recovery search.
				if preflight.Evidence.Level == memoryEvidenceAllocated {
					measuredProductionArgs = formatCommand(serverArgs)
				}
			} else if preflight.Evidence.Level != memoryEvidenceNone {
				opts := placementOpts()
				opts.SkipPlacementCache = true
				next, rerr := placement.Compute(caps, model, opts)
				if rerr != nil || next == nil {
					if rerr != nil {
						return nil, strategy, serverArgs, fmt.Errorf("backend-measured placement recompute failed: %w", rerr)
					}
					return nil, strategy, serverArgs, fmt.Errorf("backend-measured placement recompute returned no strategy")
				}
				next = recomputeAndApplyCalibration(req, cfg, model, be, caps, next)
				next = retainTunedBatch(next)
				claudeCodeSlotAdjust(next, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
				nextArgs := buildLaunchServerArgs(req, cfg, be, caps, model, next)
				if changed, rejected := memoryRecovery.recomputeDecision(serverArgs, nextArgs); changed {
					if rejected {
						fmt.Fprintln(os.Stderr, "[launch] backend-measured recompute reproduced an argv rejected by this launch's memory checks; retaining the verified-safe placement")
						if preflight.Evidence.Level == memoryEvidenceAllocated {
							measuredProductionArgs = formatCommand(serverArgs)
						}
					} else if !backendMeasuredRecomputeWorthVerifying(preflight.Evidence.Level, strategy, next) {
						// The exact probe covered serverArgs, not nextArgs. Do not trade
						// a proven placement for an unmeasured lateral tensor split.
						fmt.Fprintln(os.Stderr, "[launch] exact allocation already proves this expert residency; retaining it instead of testing an unmeasured lateral split")
						measuredProductionArgs = formatCommand(serverArgs)
					} else {
						if preflightReplans >= maxPreflightReplans {
							if preflight.Evidence.Level == memoryEvidenceAllocated {
								fmt.Fprintln(os.Stderr, "[launch] measured re-plan budget reached; retaining the exact allocation-proven placement")
								measuredProductionArgs = formatCommand(serverArgs)
							} else {
								return nil, strategy, serverArgs, fmt.Errorf("backend memory plan did not reach a fixed point after %d re-plans; refusing a real model load", maxPreflightReplans)
							}
						} else {
							strategy = next
							serverArgs = nextArgs
							preflightReplans++
							fmt.Fprintf(os.Stderr, "[launch] backend-measured memory re-plan %d/%d changed the exact argv; verifying the new placement before production\n", preflightReplans, maxPreflightReplans)
							continue
						}
					}
				} else {
					fmt.Fprintf(os.Stderr, "[launch] memory plan stable at %s evidence\n", preflight.Evidence.Level)
					if preflight.Evidence.Level == memoryEvidenceAllocated {
						measuredProductionArgs = formatCommand(serverArgs)
					}
				}
			}
		}
		if err := validateHostMemoryContainment(req, caps, strategy); err != nil {
			return nil, strategy, serverArgs, err
		}
		// A preflight or OOM recovery can move additional expert layers to CPU
		// after the initial launch confirmation. Re-check here so no re-plan can
		// silently introduce disk-backed mmap before a real backend start.
		if err := confirmRequiredMMap(req, strategy, os.Stdin, os.Stderr, stdinIsTerminal()); err != nil {
			return nil, strategy, serverArgs, err
		}
		if err := validateExactAdmissionArgv(exactAdmission, exactCandidateArgs, serverArgs); err != nil {
			return nil, strategy, serverArgs, err
		}
		p, err := startLaunchProcess(req, cfg, model, be, caps, serverArgs, timeout)
		if err == nil {
			if mmapErr := validateObservedMMapPageability(cfg, model, be, strategy, p); mmapErr != nil {
				memoryRecovery.reject(serverArgs)
				_ = p.Stop()
				if exactAdmission {
					return nil, strategy, serverArgs, exactAdmissionError(exactAdmissionMMap, "", mmapErr)
				}
				if preflightReplans >= maxPreflightReplans {
					return nil, strategy, serverArgs, fmt.Errorf("mmap pageability correction did not converge: %w", mmapErr)
				}
				preflightReplans++
				opts := placementOpts()
				opts.SkipPlacementCache = true
				next, replanErr := placement.Compute(caps, model, opts)
				if replanErr != nil || next == nil {
					if replanErr != nil {
						return nil, strategy, serverArgs, fmt.Errorf("%w; resident re-plan failed: %v", mmapErr, replanErr)
					}
					return nil, strategy, serverArgs, fmt.Errorf("%w; resident re-plan returned no strategy", mmapErr)
				}
				strategy = recomputeAndApplyCalibration(req, cfg, model, be, caps, next)
				strategy = retainTunedBatch(strategy)
				claudeCodeSlotAdjust(strategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
				serverArgs = buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
				fmt.Fprintln(os.Stderr, "[mmap] exact backend contradicted file-backed loading; recomputing under resident anonymous-memory accounting")
				continue
			}
			if strategy != nil && strategy.HotExpertCacheSlots > 0 {
				logData := ""
				if p != nil && p.LogBuf != nil {
					logData = p.LogBuf.String()
				}
				if hotErr := placement.ValidateHotExpertCacheObservation(strategy, logData); hotErr != nil {
					memoryRecovery.reject(serverArgs)
					_ = p.Stop()
					if exactAdmission {
						return nil, strategy, serverArgs, exactAdmissionError(
							exactAdmissionHotExperts, "", hotErr,
						)
					}
					// Only automatic policy may restore cache-free. Numeric policy is an
					// exact user/config request; any unexpected stale/off state also fails
					// closed instead of silently changing the requested launch.
					if !automaticHotExpertFallbackAllowed(req, strategy) {
						return nil, strategy, serverArgs, fmt.Errorf("hot-expert activation failed: %w", hotErr)
					}
					negativeSource := hotExpertFailureDecisionSource(req, cfg, model, be, caps, strategy, nil)
					// An automatic cached winner may fall back in this lifecycle, but
					// erase the evidence which claimed the cache-on process was valid.
					if invalidateErr := invalidateAutomaticHotExpertEvidence(req, cfg, model, be, caps, strategy); invalidateErr != nil {
						fmt.Fprintf(os.Stderr, "[hot-experts] warning: failed to invalidate all cache-on evidence: %v\n", invalidateErr)
					}
					req.HotExpertPendingNegative = hotExpertUnavailableDecision(
						negativeSource, strategy, "cache-on activation failed before the exact cache-free baseline was restored: "+hotErr.Error(),
					)
					// A hot-cache candidate changes no tensor/KV/batch coordinate. Strip
					// only that feature from the exact candidate rather than recomputing
					// under a potentially different free-VRAM snapshot.
					next := placement.WithoutHotExpertCache(strategy)
					if next == nil {
						return nil, strategy, serverArgs, fmt.Errorf("%w; cache-free fallback returned no strategy", hotErr)
					}
					next.PerformanceTuned = false
					next.VerifiedConfigReused = false
					strategy = next
					claudeCodeSlotAdjust(strategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
					serverArgs = buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
					fmt.Fprintf(os.Stderr, "[hot-experts] %v; invalidated cache-on evidence and restoring the stable cache-free placement\n", hotErr)
					continue
				}
				fmt.Fprintf(os.Stderr,
					"[hot-experts] activated %d slots across %d host expert layers (%s)\n",
					strategy.HotExpertCacheSlots, strategy.HotExpertCacheLayers, strategy.HotExpertCacheEvidence,
				)
			}
			if measuredProductionArgs != "" && measuredProductionArgs == formatCommand(serverArgs) {
				fmt.Fprintln(os.Stderr, "[launch] backend-measured configuration loaded and passed health check")
			}
			// Do not persist a reusable placement here. HTTP health proves only
			// that the loader reached the network loop; the profile controller
			// promotes and writes it after functional/cache/performance canaries.
			return p, strategy, serverArgs, nil
		}

		logData := ""
		var measuredComputeByGPU map[int]int
		if p != nil && p.LogBuf != nil {
			logData = p.LogBuf.String()
			measuredComputeByGPU = recordMeasuredLaunchProbes(req, cfg, model, strategy, be, runtimeCaps, logData, nil, serverProcessPID(p))
		}
		// Diagnose before checking the retry budget: a clean, parseable OOM on
		// the very last allowed attempt still deserves its real cause recorded
		// and reported, instead of surfacing only the process's raw exit error
		// (e.g. a bare "signal: segmentation fault" with no VRAM context).
		device, allocMB, isComputeBuffer, ok := startupLogCUDAOOMDetailed(logData)
		// A startup OOM is not runtime growth. recordMeasuredLaunchProbes above
		// already preserves graph-reserve sizes as compute-buffer measurements;
		// recording the same cudaMalloc again as post-health growth double-counted
		// it on the next placement. Only post-health crash paths record growth.
		if retries >= maxRetries {
			if ok {
				return p, strategy, serverArgs, fmt.Errorf("CUDA OOM on device %d allocating %d MiB (retry budget exhausted after %d attempts): %w", device, allocMB, retries, err)
			}
			return p, strategy, serverArgs, err
		}
		if !ok {
			return p, strategy, serverArgs, err
		}
		memoryRecovery.reject(serverArgs)
		if exactAdmission {
			return p, strategy, serverArgs, exactAdmissionError(exactAdmissionCUDAOOM, fmt.Sprintf(" on device %d allocating %d MiB", device, allocMB), err)
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
		var s *placement.Strategy
		var rerr error
		computeMeasuredOnFailedGPU := measuredComputeByGPU[device] > 0
		physicalDevice := physicalGPUIndex(device, visibleToPhysical)
		if isComputeBuffer && computeMeasuredOnFailedGPU {
			// The failed graph allocation is now the exact compute-buffer reserve
			// used by Compute. Penalizing the card by that allocation as well would
			// charge it twice. Recompute fresh from the measurement alone.
			opts := placementOpts()
			opts.SkipPlacementCache = true
			opts.CacheFile = ""
			s, rerr = placement.Compute(caps, model, opts)
			if rerr == nil && s != nil {
				s = recomputeAndApplyCalibration(req, cfg, model, be, caps, s)
				s = retainTunedBatch(s)
			}
		} else {
			oomPenalty[physicalDevice] += oomOvershoot(caps, physicalDevice, allocMB)
			s, rerr = placement.ReplanAfterOOM(caps, model, placementOpts(), oomPenalty)
			s = retainTunedBatch(s)
		}
		if rerr != nil || s == nil || s.OTString == "" {
			s = nil
		}
		// Apply the same monotonic policy as contained preflight recovery. A
		// measured packer result may move weights at the current ubatch, but a
		// smaller ubatch is not adopted until the failed GPU has first lost one
		// routed expert layer. For a real allocator OOM the exact overshoot is not
		// known, so one layer is the smallest useful black-box experiment.
		var recoveryCandidateArgs []string
		if s != nil {
			recoveryCandidateArgs = buildLaunchServerArgs(req, cfg, be, caps, model, s)
		}
		nextStrategy, nextArgs, method, changed := applyMemoryRecoverySelection(
			req, strategy, serverArgs, s, model, runtimeCaps,
			preflightOutcome{Device: device, AllocMB: allocMB, AllocMBMeasured: allocMB > 0, DeficitMB: 1, IsComputeBuffer: isComputeBuffer},
			recoveryCandidateArgs,
		)
		if !changed {
			return p, strategy, serverArgs, err
		}
		switch method {
		case "replanned":
			if isComputeBuffer && computeMeasuredOnFailedGPU {
				fmt.Fprintf(os.Stderr, "[launch] CUDA OOM on device %d (%d MiB); measured compute buffer and re-planned (n-cpu-moe=%d) without a duplicate penalty\n", device, allocMB, nextStrategy.NCPUMoE)
			} else {
				fmt.Fprintf(os.Stderr, "[launch] CUDA OOM on device %d (%d MiB, over ~%d MiB); re-planned (n-cpu-moe=%d) and retrying\n", device, allocMB, oomPenalty[physicalDevice], nextStrategy.NCPUMoE)
			}
		case "swa-full-withdrawn":
			fmt.Fprintf(os.Stderr, "[launch] CUDA OOM on device %d allocating %d MiB; withdrawing --swa-full (a full-context KV cache this model does not reuse) before touching placement\n", device, allocMB)
		case "expert-derate":
			fmt.Fprintf(os.Stderr, "[launch] CUDA OOM on device %d allocating %d MiB; moving one expert layer off the failed GPU before lowering ubatch\n", device, allocMB)
		case "ubatch-derate":
			fmt.Fprintf(os.Stderr, "[launch] CUDA OOM on device %d allocating %d MiB; no movable expert remains, lowering ubatch to %d\n", device, allocMB, nextStrategy.UBatchSize)
		default:
			return p, strategy, serverArgs, fmt.Errorf("unsupported CUDA OOM recovery method %q", method)
		}
		strategy = nextStrategy
		serverArgs = nextArgs
		retries++
		printVRAMLedger(strategy)
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
	if over < 512 {
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

const unknownRuntimeCUDAOOMReserveMinMB = 2048

// runtimeLogCUDAOOM also recognizes CUDA VMM failures that only report
// "current device" after cuMemCreate aborts. llama.cpp omits reserve_size from
// that diagnostic, so after a real post-health crash we conservatively reserve
// 10% of that device (at least 2 GiB). A repeat adds another such block. This is
// learned only for the exact runtime probe key; normal first launches retain
// measured, margin-free packing.
// modelLoadedLogMarkers are how a backend announces that weights, KV and the
// load-time compute buffers are all in place. Everything logged after one of
// these is serving work; everything before it is still loading.
var modelLoadedLogMarkers = []string{
	"llama_server: model loaded",
	"model loaded",
	"main: model loaded",
	// ggrun's own verdict. A passed health check is the strongest available
	// statement that the backend is serving rather than still allocating.
	"health check ok",
}

// runtimeGrowthWindowStart returns the line index after which an allocation
// failure counts as runtime graph growth, and whether such a point exists.
//
// This gate is the difference between learning and being poisoned. The recorder
// files whatever size failed under PROBED_RUNTIME_GRAPH_GROWTH_MB_<device>, so
// without it a launch that dies while allocating its KV buffer teaches the
// planner that the KV buffer is "growth" -- a budget line it already accounts
// for, now reserved twice. Verified on this project: CUDA0 carried 5504, which
// is exactly that plan's "CUDA0 KV buffer size = 5504.00 MiB".
//
// A log with no load-complete marker never yields growth. That is deliberate:
// an OOM during load is a placement error the planner must fix by moving a
// layer, not a runtime term to reserve, and guessing wrong here corrupts the
// cache for every later launch on the same key.
func runtimeGrowthWindowStart(lines []string) (int, bool) {
	for i := len(lines) - 1; i >= 0; i-- {
		lower := strings.ToLower(lines[i])
		for _, marker := range modelLoadedLogMarkers {
			if strings.Contains(lower, marker) {
				return i, true
			}
		}
	}
	return 0, false
}

func runtimeLogCUDAOOM(logData string, caps *detect.Capabilities, model *placement.ModelProfile, prior map[int]int) (device int, reserveMB int, estimated bool, ok bool) {
	lines := strings.Split(logData, "\n")
	loadedAt, loaded := runtimeGrowthWindowStart(lines)
	if !loaded {
		// Still loading when it died: a placement problem, not a runtime term.
		return 0, 0, false, false
	}
	for i := len(lines) - 1; i > loadedAt; i-- {
		if device, allocMB, ok := recovery.ParseCUDAOOM(lines[i]); ok {
			return device, allocMB, false, true
		}
		device, ok = recovery.ParseCUDADevice(lines[i])
		if !ok {
			continue
		}
		isOOM := false
		for j := i - 1; j >= 0 && j >= i-3; j-- {
			if strings.Contains(strings.ToLower(lines[j]), "cuda error: out of memory") {
				isOOM = true
				break
			}
		}
		if !isOOM {
			continue
		}
		// Reserve exactly one routed expert layer: that is the unit placement
		// moves between GPU and CPU, its size is known exactly from the GGUF
		// ledger, and it is the smallest step that changes the outcome. A tenth
		// of the card, which this used to reserve, is a quantity derived from
		// nothing -- on a 24 GiB device it withheld 2457 MiB, close to two
		// layers, and a second abort compounded it permanently.
		reserveMB = placement.LargestRoutedExpertLayerMB(model)
		if reserveMB <= 0 {
			// No ledger: fall back to the old fraction rather than reserve
			// nothing, since an OOM is proof that something must give.
			reserveMB = unknownRuntimeCUDAOOMReserveMinMB
			if caps != nil {
				for _, gpu := range caps.GPUs {
					if gpu.Index == device {
						if scaled := (gpu.VRAMTotalMB + 9) / 10; scaled > reserveMB {
							reserveMB = scaled
						}
						break
					}
				}
			}
		}
		if prior[device] >= reserveMB {
			reserveMB += prior[device]
		}
		return device, reserveMB, true, true
	}
	return 0, 0, false, false
}

func oomLogFingerprint(logData string) string {
	sum := sha256.Sum256([]byte(logData))
	return fmt.Sprintf("%x", sum[:])
}

// recordRuntimeOOMLog records either the exact failed allocation or the
// bounded VMM fallback above. markerPath prevents a Claude crash recorded on
// exit from being counted again when its previous log is recovered next run.
func recordRuntimeOOMLog(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, logData, markerPath string) (device, reserveMB int, estimated, changed, ok bool, err error) {
	if cfg == nil || model == nil || strategy == nil || be == nil || caps == nil {
		return 0, 0, false, false, false, nil
	}
	fingerprint := oomLogFingerprint(logData)
	if markerPath != "" {
		if data, readErr := os.ReadFile(markerPath); readErr == nil && strings.TrimSpace(string(data)) == fingerprint {
			return 0, 0, false, false, false, nil
		}
	}
	cacheBackendTag := scopedProbeBackendTagForStrategy(req, model, be, strategy)
	prior := placement.RuntimeGraphGrowthByGPU(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, cacheBackendTag, caps.GPUs, strategy.Parallel)
	device, reserveMB, estimated, ok = runtimeLogCUDAOOM(logData, caps, model, prior)
	if !ok {
		return 0, 0, false, false, false, nil
	}
	if err = placement.RecordRuntimeGraphGrowthFromOOM(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, cacheBackendTag, caps.GPUs, strategy.Parallel, device, reserveMB, estimated); err != nil {
		return device, reserveMB, estimated, false, true, err
	}
	changed = reserveMB > prior[device]
	if markerPath != "" {
		if err = os.WriteFile(markerPath, []byte(fingerprint+"\n"), 0600); err != nil {
			return device, reserveMB, estimated, changed, true, err
		}
	}
	return device, reserveMB, estimated, changed, true, nil
}

func classifyRuntimeCgroupOOM(oomKills, peakBytes uint64) (peakMB int, ok bool) {
	if oomKills == 0 {
		return 0, false
	}
	const mib = uint64(1024 * 1024)
	if peakBytes > 0 {
		peakMB = int((peakBytes + mib - 1) / mib)
	}
	return peakMB, true
}

func runtimeCgroupOOM(p *server.Process) (peakMB int, ok bool) {
	if p == nil {
		return 0, false
	}
	oomKills, err := p.MemoryOOMKillCount()
	if err != nil || oomKills == 0 {
		return 0, false
	}
	peakBytes, _ := p.MemoryPeakBytes()
	return classifyRuntimeCgroupOOM(oomKills, peakBytes)
}

// recordRuntimeHostCgroupOOM is the host-memory counterpart to
// recordRuntimeOOMLog. A cgroup kill has no CUDA allocation line, so treating it
// as an unrecognized crash leaves an unsafe active/verified profile behind.
// Revoke that profile and persist any prompt/checkpoint evidence from the log;
// the next launch then derives a fresh host plan instead of replaying the one
// the kernel disproved. markerPath makes the Claude health watcher and client
// exit path idempotent.
func recordRuntimeHostCgroupOOM(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, serverArgs []string, p *server.Process, logData, markerPath string) (peakMB int, hostOOM, recorded bool, err error) {
	peakMB, hostOOM = runtimeCgroupOOM(p)
	if !hostOOM {
		return 0, false, false, nil
	}
	fingerprint := oomLogFingerprint(logData)
	if markerPath != "" {
		if data, readErr := os.ReadFile(markerPath); readErr == nil && strings.TrimSpace(string(data)) == fingerprint {
			return peakMB, true, false, nil
		}
	}
	reason := fmt.Sprintf("host cgroup OOM after health verification (peak %d MiB)", peakMB)
	if err = invalidateRuntimeOOMLaunch(req, cfg, model, be, caps, strategy, serverArgs, reason); err != nil {
		return peakMB, true, false, err
	}
	// Avoid mutating the serving strategy from the Claude health goroutine. The
	// process is already dead; only the exact probe evidence is needed now.
	evidenceStrategy := *strategy
	recordPreviousClaudePromptCache(req, cfg, model, &evidenceStrategy, be, caps, logData)
	if markerPath != "" {
		if err = os.WriteFile(markerPath, []byte(fingerprint+"\n"), 0600); err != nil {
			return peakMB, true, false, err
		}
	}
	return peakMB, true, true, nil
}

func previousClaudeLogMatches(logData string, model *placement.ModelProfile, strategy *placement.Strategy, scope string) bool {
	if model == nil || strategy == nil || strategy.Parallel < 1 || strategy.ContextSize < 1 {
		return false
	}
	if scope == "" || !strings.Contains(logData, "[ggrun] launch-scope: "+scope) {
		return false
	}
	if !strings.Contains(logData, filepath.Base(model.Path)) {
		return false
	}
	// A crash before the health check never reaches slot init, so neither the
	// health marker nor the slots line exists -- and refusing those logs meant
	// the only failures ggrun could not learn from were the ones that stopped it
	// starting. On this project a backend swap invalidated the probe history,
	// placement packed three expert layers too many, and every retry re-read a
	// log it had decided was unusable and repeated the same choice.
	//
	// The scope already pins the exact canonical argv, including context and
	// slot count, so it is sufficient evidence that the log describes this
	// launch. The caller still records nothing unless a real OOM is parsed out
	// of it.
	if !strings.Contains(logData, "health check OK") {
		return true
	}
	wantSlots := fmt.Sprintf("n_slots = %d, n_ctx_slot = %d", strategy.Parallel, strategy.ContextSize/strategy.Parallel)
	return strings.Contains(logData, wantSlots)
}

// recordPreviousClaudePromptCache learns what a saved conversation actually
// cost from the last run's log.
//
// It has to read the previous run rather than this one: eviction and skip lines
// are emitted while serving, and the launch-time buffer ggrun records other
// probes from is captured at the health check, before any request exists. That
// makes this the same shape as the OOM recovery beside it, and it only became
// reliable once the log scope stopped depending on values placement derives --
// before that the previous run's log was filed under a name this one could not
// reconstruct.
// It reports whether the stored measurement grew, which is the caller's cue to
// re-plan: CRAM was already computed from the old value.
func recordPreviousClaudePromptCache(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, logData string) bool {
	if cfg == nil || model == nil || strategy == nil || caps == nil || logData == "" {
		return false
	}
	obs := placement.ObservePromptCache(logData)
	if obs.LargestEntryMB <= 0 && obs.BytesPerToken <= 0 && obs.LargestCheckpointMB <= 0 {
		return false
	}
	tag := scopedProbeBackendTagForStrategy(req, model, be, strategy)
	if err := placement.RecordPromptCacheObservation(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, tag, caps.GPUs, strategy.Parallel, obs); err != nil {
		return false
	}
	promptCacheGrew := obs.LargestEntryMB > strategy.MeasuredPromptCacheEntryMB ||
		obs.BytesPerToken > strategy.MeasuredPromptCacheBPT
	checkpointBefore, checkpointAfter, checkpointGrew := placement.ApplyMeasuredCheckpointObservation(model, strategy, obs)
	switch {
	case obs.Skipped > 0:
		fmt.Printf("[launch] prompt cache: %d prompt(s) exceeded the whole budget last run, largest %.0f MiB; sizing from the measurement\n",
			obs.Skipped, obs.LargestEntryMB)
	case obs.Evicted > 0:
		fmt.Printf("[launch] prompt cache: %d eviction(s) last run, largest entry %.0f MiB; sizing from the measurement\n",
			obs.Evicted, obs.LargestEntryMB)
	}
	if checkpointGrew {
		fmt.Printf("[launch] context checkpoints: measured %.3f MiB each; host reserve grew from %d to %d MiB\n",
			obs.LargestCheckpointMB, checkpointBefore, checkpointAfter)
	}
	return promptCacheGrew || checkpointGrew
}

func recoverPreviousClaudeRuntimeOOM(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, serverArgs []string) (*placement.Strategy, error) {
	if req == nil || !req.ClaudeCode {
		return strategy, nil
	}
	scope := claudeLaunchLogScope(req, model, be, serverArgs)
	logPath := claudeServerLogPath(cfg, req.Port, scope)
	logData, err := os.ReadFile(logPath)
	if err != nil || !previousClaudeLogMatches(string(logData), model, strategy, scope) {
		return strategy, nil
	}
	// A served log carries what the prompt cache actually cost as well. Record
	// it before anything re-plans, so one re-plan picks up both.
	promptCacheGrew := recordPreviousClaudePromptCache(req, cfg, model, strategy, be, caps, string(logData))
	markerPath := logPath + ".oom-recorded"
	device, reserveMB, estimated, changed, ok, err := recordRuntimeOOMLog(req, cfg, model, strategy, be, caps, string(logData), markerPath)
	if err != nil {
		return nil, fmt.Errorf("recover previous Claude runtime OOM: %w", err)
	}
	if !ok || !changed {
		if !promptCacheGrew {
			return strategy, nil
		}
		// CRAM was sized before the measurement existed. Re-plan so the budget
		// reaches the launch rather than waiting for the run after next.
		opts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
		opts.SkipPlacementCache = true
		opts.VerifiedConfigScopeKey = ""
		next, err := placement.Compute(caps, model, opts)
		if err != nil {
			return nil, err
		}
		next = applyCalibrationDecision(req, cfg, model, be, caps, next)
		claudeCodeSlotAdjust(next, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
		fmt.Printf("[launch] prompt cache: re-planned -cram %d -> %d MiB from the measured entry size\n", strategy.CRAM, next.CRAM)
		return next, nil
	}
	if estimated {
		fmt.Printf("[launch] recovered previous CUDA VMM OOM on device %d; llama.cpp omitted its allocation size, reserving %d MiB runtime headroom and re-planning\n", device, reserveMB)
	} else {
		fmt.Printf("[launch] recovered previous CUDA OOM on device %d; reserving the measured %d MiB allocation and re-planning\n", device, reserveMB)
	}
	opts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	opts.SkipPlacementCache = true
	opts.VerifiedConfigScopeKey = ""
	next, err := placement.Compute(caps, model, opts)
	if err != nil {
		return nil, err
	}
	next = applyCalibrationDecision(req, cfg, model, be, caps, next)
	claudeCodeSlotAdjust(next, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	return next, nil
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

// resolveLaunchBackend selects the backend, applies any configured custom
// architecture routing, and preflights the arch. This step is identical across
// every launch path (CLI, TUI, dry-run). Returns nil if no backend is available.
func resolveLaunchBackend(req *launchRequest, model *placement.ModelProfile, caps *detect.Capabilities) *backendInfo {
	be := selectBackendForModel(caps, req, model)
	if be == nil {
		return nil
	}
	be = preferHotExpertFeatureBackend(req, model, be)
	applyBackendFeatureCompatibility(req, model, be)
	preflightBackendArch(model, be, caps, req.AppHome)
	return be
}

// preferHotExpertFeatureBackend selects an already-built feature overlay for
// the exact backend ggrun chose. It does not route by architecture and it does
// not build during a non-interactive launch: provisioning is explicit in the
// TUI/backend command, while every later launch can reuse the validated binary.
func preferHotExpertFeatureBackend(req *launchRequest, model *placement.ModelProfile, selected *backendInfo) *backendInfo {
	if req == nil || model == nil || selected == nil || !model.IsMoE || req.ServerBinExplicit ||
		req.HotExpertsRuntimeDisabled || strings.EqualFold(strings.TrimSpace(req.HotExperts), "off") {
		return selected
	}
	if placement.BackendSupportsHotExpertCache(selected.Help) {
		return selected
	}
	composite := backends.FeatureBackend(selected.Tag, "hot-experts")
	if composite == nil {
		return selected
	}
	candidate := detectRegisteredBackend(composite)
	if candidate == nil || !placement.BackendSupportsHotExpertCache(candidate.Help) {
		fmt.Fprintf(os.Stderr, "Warning: installed hot-experts overlay %q no longer exposes its required cache flags; using base backend %q\n", composite.Tag, selected.Tag)
		return selected
	}
	fmt.Printf("[launch] hot experts enabled: selecting validated overlay %q for base backend %q\n", composite.Tag, selected.Tag)
	return candidate
}

func applyBackendFeatureCompatibility(req *launchRequest, model *placement.ModelProfile, be *backendInfo) {
	if req == nil || be == nil {
		return
	}
	arch := ""
	if model != nil {
		arch = strings.TrimSpace(model.ModelArch)
	}
	isDeepSeek4IK := strings.EqualFold(arch, "deepseek4") && be.IsIK
	// The architecture's KV rule is re-applied here because backend selection can
	// land somewhere the pre-selection pass did not assume, and because a cached
	// or resumed request can arrive with a KV type the rule forbids. Both backend
	// families are held to the same standard: ik_llama accepting a type is not
	// evidence the model computes correctly with it.
	if rule, ok := backends.KVRuleForArch(arch); ok {
		if kvType, err := placement.NormalizeKVType(req.KVQuality); err == nil && !rule.Permits(kvType) {
			family := "mainline"
			if be.IsIK {
				family = "ik_llama"
			}
			fmt.Printf("[launch] %s on %s cannot be trusted with a %s K-cache (%s); promoting this launch to %s and recomputing placement.\n",
				arch, family, kvType, rule.Reason, rule.Target())
			req.KVQuality = rule.Target()
			req.KVTypeK, req.KVTypeV = "", ""
		}
	}
	// ik_llama accepts -khad during argument parsing, but DeepSeek4 rejects it
	// only after loading the weights and creating the context. Avoid paying for
	// that known-bad load. An explicit passthrough -khad remains authoritative
	// and will fail closed rather than being silently changed.
	if isDeepSeek4IK && !hasArg(req.ExtraArgs, "-khad") {
		const reason = "DeepSeek4 has no K-cache Hadamard implementation in ik_llama"
		if disableBackendFlag(req, "-khad", reason) {
			fmt.Printf("[launch] %s; disabling -khad for this model/backend profile.\n", reason)
		}
	}
	if !hasArg(req.ExtraArgs, "--swa-full") {
		return
	}
	// A model with no sliding-window layer (SlidingWindow <= 0) cannot use a full
	// SWA cache at all, regardless of whether the backend lists --swa-full in its
	// --help. Mainline llama-server DOES advertise --swa-full but still silently
	// disables it at load for n_swa==0 models, so this deterministic GGUF gate must
	// run before the help-surface gate below (which would otherwise early-return).
	// It mirrors the -khad DeepSeek4 disable: an explicit user passthrough stays
	// authoritative and fails closed rather than being silently changed.
	if model != nil && model.SlidingWindow <= 0 && !userExplicitBackendFlag(req, "--swa-full") {
		req.ExtraArgs = setPassthroughBoolFlag(req.ExtraArgs, "--swa-full", false)
		archLabel := "this model"
		if arch != "" {
			archLabel = arch
		}
		fmt.Printf("[launch] Full SWA cache is unavailable for %s (no sliding-window layer); disabling it for this launch.\n", archLabel)
		return
	}
	// An empty help surface is unknown, not unsupported. With a real help probe,
	// however, passing an absent option is guaranteed to abort argument parsing.
	if strings.TrimSpace(be.Help) == "" || strings.Contains(be.Help, "--swa-full") {
		return
	}
	if userExplicitBackendFlag(req, "--swa-full") {
		fmt.Printf("[launch] backend %s does not advertise --swa-full; preserving the explicit user request so validation fails closed.\n", be.Path)
		return
	}
	req.ExtraArgs = setPassthroughBoolFlag(req.ExtraArgs, "--swa-full", false)
	archLabel := "this model"
	if arch != "" {
		archLabel = arch
	}
	fmt.Printf("[launch] Full SWA cache is unavailable for %s on backend %s; disabling it for this launch.\n", archLabel, be.Path)
}

func launchHardwareIdentity(caps *detect.Capabilities) string {
	if caps == nil {
		return "unknown"
	}
	parts := make([]string, 0, len(caps.GPUs)+2)
	for _, gpu := range caps.GPUs {
		parts = append(parts, fmt.Sprintf("gpu%d:%s:%d:%s:%s:gen%dx%d:bw%d",
			gpu.Index, gpu.Name, gpu.VRAMTotalMB, gpu.Driver, gpu.PCIBusID,
			gpu.PCIGen, gpu.PCILanes, gpu.BandwidthMBps))
	}
	sort.Strings(parts)
	parts = append(parts, fmt.Sprintf("ram:%d", caps.RAM.TotalMB), fmt.Sprintf("cpu:%s:%d", caps.CPU.Model, caps.CPU.Cores))
	return controller.ScopeKey(parts...)
}

// verifyAndActivateLaunch is the promotion boundary between "the HTTP listener
// came up" and "this profile may be reused automatically". A cache regression
// leaves the server available in a clearly reported degraded state, but it does
// not overwrite the last-known-good placement.
func verifyAndActivateLaunch(req *launchRequest, cfg *config.Config, model *placement.ModelProfile,
	be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy, serverArgs []string, claudeRouterURL string,
) error {
	if req == nil || cfg == nil || model == nil || be == nil || strategy == nil {
		return errors.New("incomplete launch profile")
	}
	hardware := launchHardwareIdentity(caps)
	argsHash := controller.HashArgs(serverArgs)
	scope := launchProfileScope(req, model, be, caps)
	store := controller.Store{CacheDir: cfg.CacheDir}
	if store.IsActive(scope, argsHash) {
		if !req.CalibrationScreened && !req.CalibrationPending && strategy.Type == placement.MoEOffload && strategy.PlacementCachePath != "" {
			_ = placement.SavePlacementCache(strategy.PlacementCachePath, placement.StrategyToCacheEntry(strategy))
		}
		// The already-active fast path is a re-validation of the same profile: the
		// config was already promoted, so refresh the verified config record too.
		saveVerifiedConfigForLaunch(cfg, req, model, be, caps, strategy)
		fmt.Fprintln(os.Stderr, "[verify] exact launch profile is already active; reusing its canary result")
		return nil
	}

	profile, err := store.Begin(controller.Profile{
		Scope:            scope,
		ModelIdentity:    placement.SpecTargetIdentity(model),
		BackendIdentity:  be.Identity,
		HardwareIdentity: hardware,
		ArgsHash:         argsHash,
		Properties: map[string]string{
			"context":      strconv.Itoa(strategy.ContextSize),
			"ubatch":       strconv.Itoa(strategy.UBatchSize),
			"parallel":     strconv.Itoa(strategy.Parallel),
			"kv_type":      strategy.KVType,
			"kv_placement": strategy.KVPlacement,
			"swa_full":     strconv.FormatBool(strategy.SWAFull),
			"checkpoints":  strconv.Itoa(strategy.MaxCheckpoints),
		},
	})
	if err != nil {
		return err
	}
	tag := scopedProbeBackendTagForStrategy(req, model, be, strategy)
	allocationEvidence := "live-server-load"
	var profileGPUs []detect.GPU
	if caps != nil {
		profileGPUs = caps.GPUs
	}
	if allocation, ok := placement.LoadMeasuredAllocation(cfg.CacheDir, model, strategy.ContextSize,
		strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, tag, profileGPUs, strategy.Parallel); ok {
		allocationEvidence = allocation.Evidence
	}
	if _, err = store.Transition(scope, profile.ID, controller.StateAllocationVerified,
		"backend allocation accepted", allocationEvidence); err != nil {
		return err
	}
	if _, err = store.Transition(scope, profile.ID, controller.StateLoadHealthy,
		"server passed health and model-list checks", "health"); err != nil {
		return err
	}

	runner := &benchmark.Runner{
		BaseURL: fmt.Sprintf("http://127.0.0.1:%d", req.Port),
		Model:   filepath.Base(model.Path),
		Timeout: 20 * time.Minute,
	}
	canary, canaryErr := runner.RunCacheCanary()
	if canaryErr != nil || canary == nil || !canary.Functional {
		reason := "functional canary failed"
		if canaryErr != nil {
			reason += ": " + canaryErr.Error()
		} else if canary != nil && canary.Reason != "" {
			reason += ": " + canary.Reason
		}
		_, _ = store.Transition(scope, profile.ID, controller.StateRejected, reason, "cache-canary")
		return errors.New(reason)
	}
	if _, err = store.Transition(scope, profile.ID, controller.StateFunctionalVerified,
		"deterministic completion endpoint responded", "cache-canary"); err != nil {
		return err
	}
	metrics := []controller.Metric{
		{Name: "cold_prompt_tokens", Value: float64(canary.ColdPromptTokens), Unit: "tokens", Source: "cache-canary"},
		{Name: "append_cached_tokens", Value: float64(canary.AppendCachedTokens), Unit: "tokens", Source: "cache-canary"},
		{Name: "branch_cached_tokens", Value: float64(canary.BranchCachedTokens), Unit: "tokens", Source: "cache-canary"},
		{Name: "cold_prompt_tps", Value: canary.ColdPromptTPS, Unit: "tokens/s", Source: "cache-canary"},
	}
	if !canary.Passed {
		reason := canary.Reason
		if reason == "" {
			reason = "prefix-cache canary did not meet reuse thresholds"
		}
		_, _ = store.Transition(scope, profile.ID, controller.StateDegraded, reason, "cache-canary", metrics...)
		fmt.Fprintf(os.Stderr, "[verify] degraded profile: %s; placement will not be promoted and the support expert may analyze it during a maintenance window\n", reason)
		if strategy != nil && strategy.HotExpertCacheSlots > 0 {
			return fmt.Errorf("hot-expert candidate failed cache verification: %s", reason)
		}
		return nil
	}
	if _, err = store.Transition(scope, profile.ID, controller.StateCacheVerified,
		"strict extension, older branch, and replay restored prefix state", "cache-canary", metrics...); err != nil {
		return err
	}
	if req.ClaudeCode {
		if strings.TrimSpace(claudeRouterURL) == "" {
			reason := "Claude workload profile has no running Anthropic router to verify"
			_, _ = store.Transition(scope, profile.ID, controller.StateRejected, reason, "claude-router-canary")
			return errors.New(reason)
		}
		claudeRunner := &benchmark.Runner{
			BaseURL: claudeRouterURL,
			Model:   "local",
			Timeout: 20 * time.Minute,
		}
		if routerErr := claudeRunner.RunClaudeRouterCanary(); routerErr != nil {
			reason := "Claude Anthropic-router canary failed: " + routerErr.Error()
			_, _ = store.Transition(scope, profile.ID, controller.StateRejected, reason, "claude-router-canary")
			return errors.New(reason)
		}
		reviewerEnabled := req.ReviewerProfile != nil &&
			!req.ClaudeReviewerDisabled && claudeCompanionNeeded(nil)
		if reviewerEnabled {
			reviewerRunner := &benchmark.Runner{
				BaseURL: claudeRouterURL,
				Model:   "local",
				Timeout: 20 * time.Minute,
			}
			if reviewerErr := reviewerRunner.RunClaudeReviewerCanary(); reviewerErr != nil {
				reason := "Claude reviewer-route canary failed: " + reviewerErr.Error()
				_, _ = store.Transition(scope, profile.ID, controller.StateRejected, reason, "claude-reviewer-canary")
				return errors.New(reason)
			}
		}
		if reviewerEnabled && req.ReviewerProfile.ServesWorkers {
			workerRunner := &benchmark.Runner{
				BaseURL: claudeRouterURL,
				Model:   claudeauto.UtilityAlias,
				Timeout: 20 * time.Minute,
			}
			if workerErr := workerRunner.RunClaudeRouterCanary(); workerErr != nil {
				reason := "Claude worker-route canary failed: " + workerErr.Error()
				_, _ = store.Transition(scope, profile.ID, controller.StateRejected, reason, "claude-worker-canary")
				return errors.New(reason)
			}
		}
	}
	if _, err = store.Transition(scope, profile.ID, controller.StatePerformanceVerified,
		"cache canary recorded prefill performance and workload gateway passed", "cache-canary+workload-canary"); err != nil {
		return err
	}
	if _, err = store.Transition(scope, profile.ID, controller.StateActive,
		"all required launch checks passed", "profile-controller"); err != nil {
		return err
	}
	if !req.CalibrationScreened && !req.CalibrationPending && strategy.Type == placement.MoEOffload && strategy.PlacementCachePath != "" {
		if err := placement.SavePlacementCache(strategy.PlacementCachePath, placement.StrategyToCacheEntry(strategy)); err != nil {
			return fmt.Errorf("persist verified placement: %w", err)
		}
	}
	// The functional canary passed and the profile reached StateActive: this is
	// the promotion boundary. Save the full verified config so the next launch
	// of this exact scope starts directly from it. Failure degrades to a log —
	// the launch is already active and must not be failed by a cache write.
	saveVerifiedConfigForLaunch(cfg, req, model, be, caps, strategy)
	fmt.Fprintf(os.Stderr, "[verify] active profile: append cache=%d, branch cache=%d tokens\n",
		canary.AppendCachedTokens, canary.BranchCachedTokens)
	return nil
}

// saveVerifiedConfigForLaunch persists the full serving config at the promotion
// boundary (StateActive / working inference). The record is scoped by the same
// CalibrationScopeKey the reuse path hashes against, so save and load can never
// disagree about what launch they describe. A save failure degrades to a stderr
// log — the launch is already active and must never be failed by a cache write.
func saveVerifiedConfigForLaunch(cfg *config.Config, req *launchRequest, model *placement.ModelProfile,
	be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy,
) {
	if cfg == nil || req == nil || model == nil || be == nil || strategy == nil {
		return
	}
	// --no-cached-config is the escape hatch: do not write a verified config for
	// a launch that explicitly asked to derive fresh.
	if req.NoCachedConfig || req.CalibrationScreened || req.CalibrationPending {
		return
	}
	reviewer := ""
	if req.ReviewerProfile != nil {
		reviewer = req.ReviewerProfile.Name
	}
	// The same strategy-free scope key the reuse path hashes against must be
	// used here, so a record saved after StateActive is found by the next
	// launch's pre-Compute lookup (which cannot know the final strategy yet).
	scopeKey := verifiedConfigScopeKey(req, model, be, caps)
	vc := placement.VerifiedConfigToRecord(
		scopeKey,
		filepath.Base(model.Path),
		strategy,
		be.Identity,
		be.Path,
		req.ChatTemplateOverride,
		reviewer,
	)
	if _, err := placement.SaveVerifiedConfig(cfg.CacheDir, vc); err != nil {
		fmt.Fprintf(os.Stderr, "[verify] verified config cache write failed (degrading, launch unaffected): %v\n", err)
		return
	}
	fmt.Fprintf(os.Stderr, "[verify] verified config saved for this scope (direct start on next launch)\n")
}

// verifiedConfigScopeKey computes the strategy-free scope key for the
// verified-config cache. It deliberately omits the base-placement hash: the
// whole point of the record is that the *saved config is the placement*, so the
// reuse lookup must be computable before placement.Compute runs (which is
// exactly what the verified config short-circuits). The request, model, and
// hardware fields are the scope: a change to ctx/parallel/batch/ubatch/KV/
// mmap/memory-policy/swa-full/chat-template/backend/GPU set all produce a
// different key and a clean miss.
// planLogicVersion identifies the placement logic that produced a verified
// config. Bump it whenever a planner change should retire stored plans: VRAM
// bandwidth-aware dense splits, VRAM-budgeted context fit, and granular context
// maximisation all changed what a good plan looks like, and records written
// before them would otherwise replay the old answer indefinitely.
const planLogicVersion = "6"

func verifiedConfigScopeKey(req *launchRequest, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities) string {
	if req == nil || model == nil || be == nil {
		return ""
	}
	if req.NoCachedConfig {
		return ""
	}
	opts := placementOptionsFromRequestCaps(req, model, be, "", caps)
	// A verified config REPLAYS a whole plan and bypasses placement.Compute, so
	// nothing in the hardware/model/backend scope changes when ggrun's own
	// planning logic does -- a record written by an older planner would keep
	// replaying its plan forever. Bind the record to the planner that produced
	// it: bumping planLogicVersion retires every stored config exactly once.
	// Only verified configs carry this; measured calibration data describes the
	// hardware, not our arithmetic, and stays valid across planner changes.
	return placement.NewCalibrationScopeKey(model, caps, opts, nil).String() + "-plan" + planLogicVersion
}

func requestedLaunchPolicyIdentity(req *launchRequest, model *placement.ModelProfile) string {
	if req == nil {
		return "default"
	}
	values := []string{
		"ctx=" + req.CtxFlag,
		"kv-placement=" + req.KVPlacement,
		"kv-quality=" + req.KVQuality,
		"kv-k=" + req.KVTypeK,
		"kv-v=" + req.KVTypeV,
		"cpu=" + strconv.FormatBool(req.CPUMode),
		"gpus=" + req.GPUsFlag,
		"vision=" + strconv.FormatBool(req.VisionAuto),
		"mmproj=" + req.MMProjPath,
		"tune=" + req.TuneCache,
		"spec=" + req.SpecMode,
		"hot-experts=" + req.HotExperts,
		"hot-experts-explicit=" + strconv.FormatBool(req.HotExpertsSet),
		"force-spec-moe=" + strconv.FormatBool(req.ForceSpecMoE),
		"ram-budget=" + strconv.Itoa(req.RamBudgetMB),
		"ram-limit=" + strconv.Itoa(req.RAMLimitPercent),
		"vram-headroom=" + strconv.Itoa(req.VRAMHeadroomMB),
		"ram-headroom=" + strconv.Itoa(req.RAMHeadroomMB),
		"no-mmap=" + strconv.FormatBool(req.NoMMap),
		"force-mmap=" + strconv.FormatBool(req.ForceMMap),
		"parallel=" + strconv.Itoa(req.Parallel),
		"parallel-explicit=" + strconv.FormatBool(req.ParallelSet),
		"threads=" + strconv.Itoa(req.Threads),
		"cache-ram=" + strconv.Itoa(req.CacheRAMMB),
		"claude-max-active=" + strconv.Itoa(req.ClaudeMaxActive),
		"batch=" + strconv.Itoa(req.BatchSize),
		"batch-explicit=" + strconv.FormatBool(req.BatchSizeSet),
		"ubatch=" + strconv.Itoa(req.UBatchSize),
		"ubatch-explicit=" + strconv.FormatBool(req.UBatchSizeSet),
		"benchmark=" + strconv.FormatBool(req.Benchmark),
		"worker-benchmark=" + strconv.FormatBool(req.WorkerBenchmark),
		"workload=" + requestWorkloadProfile(req, model),
	}
	return controller.ScopeKey(values...)
}

func launchModelFamilyIdentity(model *placement.ModelProfile) string {
	if model == nil {
		return ""
	}
	name := strings.TrimSpace(model.Name)
	if name == "" {
		name = strings.TrimSpace(model.Basename)
	}
	if name == "" {
		name = filepath.Base(model.Path)
	}
	return controller.ScopeKey(
		strings.ToLower(name), strings.ToLower(model.ModelArch),
		strconv.Itoa(model.NumLayers), strconv.FormatInt(model.NumParams, 10),
		strconv.Itoa(model.ContextSize), strings.ToLower(model.TokenizerHash),
	)
}

func launchBackendFamilyIdentity(be *backendInfo) string {
	if be == nil {
		return ""
	}
	tag := strings.TrimSpace(be.Tag)
	if tag == "" {
		tag = strings.TrimSpace(be.Dialect)
	}
	if tag == "" {
		tag = filepath.Base(be.Path)
	}
	return strings.ToLower(tag)
}

func launchProfileScope(req *launchRequest, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities) string {
	if model == nil || be == nil {
		return ""
	}
	policy := "default"
	if req != nil {
		policy = req.ProfilePolicyIdentity
		if policy == "" {
			policy = requestedLaunchPolicyIdentity(req, model)
		}
	}
	// Lifecycle canaries are evidence for the planner semantics that interpreted
	// the argv, not just for its current spelling. Reusing an older active record
	// after checkpoint accounting changed would skip the very canary that creates
	// the first exact checkpoint measurement. Keep this version aligned with the
	// full verified-config scope so each planner revision revalidates once.
	return controller.ScopeKey(launchModelFamilyIdentity(model), launchBackendFamilyIdentity(be),
		launchHardwareIdentity(caps), policy, claudeCompanionLifecycleIdentity(req),
		"plan-logic="+planLogicVersion)
}

func claudeCompanionLifecycleIdentity(req *launchRequest) string {
	if req == nil || !req.ClaudeCode {
		return "no-claude-companion"
	}
	if req.ClaudeReviewerDisabled || !claudeCompanionNeeded(nil) {
		return "main-model-fallback"
	}
	if req.ReviewerProfile == nil {
		return "companion-unresolved"
	}
	p := req.ReviewerProfile
	return controller.ScopeKey("separate-companion", p.Name, p.ModelPath, p.BackendPath, p.KVType,
		strconv.Itoa(p.reviewerContextTokens()), p.companionMeasurementKey())
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

	cfg := loadConfigOrExit()
	req.ModelPath = resolveModelPath(req.ModelPath, cfg.ModelDir)

	model, err := parseModel(req.ModelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing model: %v\n", err)
		os.Exit(1)
	}
	warnModelCompatibility(model)
	// Freeze the user's requested serving policy before backend compatibility,
	// placement recovery, or the support controller adjusts generated knobs.
	// Those adjusted argv are candidates within this family, not new families.
	req.ProfilePolicyIdentity = requestedLaunchPolicyIdentity(req, model)

	// A restart races the previous server's teardown: it can hold tens of GB
	// of VRAM for seconds after ggrun exits, and a plan computed in that window
	// bakes the shortage in. Wait for the port it must release anyway.
	launchPort := req.Port
	if launchPort <= 0 {
		launchPort = cfg.Port
	}
	if !waitForPredecessorPort(launchPort, 20*time.Second, os.Stderr) {
		fmt.Fprintf(os.Stderr, "[launch] port %d is still occupied after 20s; continuing, but placement may see its VRAM as used and the bind may fail\n", launchPort)
	}

	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}
	// --gpus selects real hardware, so a name that is not there is a mistake, not
	// a hint. Silently dropping it narrowed the selection (--gpus 0,3 quietly ran
	// on one card) and dropping all of it fell back to EVERY GPU -- the opposite
	// of what someone reserving a card for other work asked for.
	if err := validateRequestedGPUs(caps, req); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	be := resolveLaunchBackend(req, model, caps)
	if recipe := reviewedRecipeRequiredForMain(model.ModelArch, be); recipe != nil {
		if !confirmReviewedBackendInstall(recipe, model.ModelArch, cfg.AssumeYes, os.Stdin, os.Stderr, stdinIsTerminal()) {
			fmt.Fprintf(os.Stderr, "Error: no proven main-model backend for architecture %q; install the reviewed backend with: ggrun backend install %s\n", model.ModelArch, recipe.Name)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "[launch] installing reviewed backend %q before model placement\n", recipe.Name)
		cmdBackendInstall([]string{recipe.Name})
		be = resolveLaunchBackend(req, model, caps)
	}
	if be == nil {
		// No reviewed recipe for this architecture. Search open llama.cpp PRs
		// and Hugging Face GGUF cards for a head fork that adds the loader; if
		// the user declines or nothing is found, the mainline-update offer
		// remains. A scripted non-terminal call never blocks.
		if offerDiscoveredArchFork(req, model, cfg.AssumeYes) {
			be = resolveLaunchBackend(req, model, caps)
		}
		if be == nil && offerMainlineBackendUpdate(req, model, cfg.AssumeYes) {
			be = resolveLaunchBackend(req, model, caps)
		}
		if be == nil {
			fmt.Fprintf(os.Stderr, "Error: %s\n", backendUnavailableMessage(req))
			os.Exit(1)
		}
	}
	applyCachedBackendCapabilities(req, cfg.CacheDir, model, be)
	if env := applyGPUVisibility(req, backendDialect(be)); env != "" {
		fmt.Printf("[launch] GPU restriction: %s\n", env)
	}
	if err := guardPortFree(req.Port, "launch"); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Claude Code's Auto permission checks must not run on the giant coding
	// model: one tool call can otherwise trigger ten extra ~25k-token turns.
	// The dedicated reviewer is a placement companion: its VRAM is reserved in
	// the same ledger before the main model's split is computed, and the planner
	// returns the GPU the reviewer should occupy. The reservation lives on the
	// request so every placement.Compute path — first plan and every re-plan —
	// keeps the reviewer's VRAM accounted.
	req.ReviewerReservation = claudeReviewerReservation(req, caps, cfg.CacheDir)
	// A dense model that cannot fit on GPU even at reduced context should not be
	// silently spilled into system RAM. Ask the user; assumeYes / non-terminal
	// keeps the historical silent behavior. The hook rides the request so every
	// Compute path — first plan and every re-plan — asks the same question.
	req.DenseOffloadPrompt = confirmDenseCPUOffload(cfg.AssumeYes)

	computeStrategy := func(candidateReq *launchRequest) (*placement.Strategy, error) {
		candidate, computeErr := placement.Compute(caps, model, placementOptionsFromRequestCaps(candidateReq, model, be, cfg.CacheDir, caps))
		if computeErr != nil {
			return nil, computeErr
		}
		// A prior calibration for this exact model/hardware/workload scope already
		// measured the fastest placement; apply it instead of the estimate.
		return applyCalibrationDecision(candidateReq, cfg, model, be, caps, candidate), nil
	}

	strategy, err := computeStrategy(req)
	if err != nil {
		strategy, err = tryResidentWithoutClaudeReviewer(req, err, os.Stdin, os.Stderr, stdinIsTerminal(), computeStrategy)
		if err != nil {
			// Layer-1 deterministic replan before any Layer-2 escalation. The
			// first Compute may have consumed a poisoned .place cache; recompute
			// with the cache bypassed (and any derated ubatch preserved), then
			// let the advisor decide only if that deterministic replan also
			// fails. Compute already walks UBatchFitLadder internally, so the
			// replan's only knob beyond SkipPlacementCache is clearing the cache
			// file and preserving a prior derate.
			firstCode := classifyAdvisorFailure(err)
			if replanned, replanErr := deterministicReplanOnPlacementFailure(req, model, be, cfg, caps, nil); replanErr == nil {
				strategy, err = replanned, nil
			} else {
				// Layer-2/Layer-3 boundary: escalate only a genuinely novel failure
				// (or a classified class that recurred after the deterministic
				// budget), and only after the user consents to the consultation.
				// Every declined/absent path returns the ORIGINAL replan error.
				strategy, err = escalatePlacementFailure(req, cfg, model, be, caps, firstCode, replanErr, computeStrategy)
			}
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error computing placement: %s\n", placementErrorMessage(err))
				os.Exit(1)
			}
		}
	}
	// Proactive reviewer-drop gate: a dense model that fits fully on GPU with
	// comfortable spare VRAM does not need a separate Auto reviewer — review
	// traffic routes through the main model. This runs only on the FIRST
	// successful plan; happened-before any reactive retry path, and never on a
	// plan that only fits because the reviewer was already dropped.
	strategy = proactivelyDropReviewerForVRAMModel(req, caps, model, strategy, computeStrategy, os.Stdin, os.Stderr, stdinIsTerminal(), cfg.CacheDir)
	if err := confirmRequiredMMap(req, strategy, os.Stdin, os.Stderr, stdinIsTerminal()); err != nil {
		if !errors.Is(err, errMMapDeclined) {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintln(os.Stderr, "[placement] mmap declined; recomputing a fully resident placement")
		strategy, err = computeStrategy(req)
		if err != nil {
			strategy, err = tryResidentWithoutClaudeReviewer(req, err, os.Stdin, os.Stderr, stdinIsTerminal(), computeStrategy)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error computing resident placement: %s\n", placementErrorMessage(err))
			os.Exit(1)
		}
	}
	if err := validateHostMemoryContainment(req, caps, strategy); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Finalize and parser-check the exact backend command before starting any
	// companion or model process. A ggrun-generated dialect mismatch must be a
	// cheap, explicit pre-launch error, never a reviewer load followed by a giant
	// backend help dump from the contained memory probe.
	claudeCodeSlotAdjust(strategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	var preRecoveryStrategy *placement.Strategy
	var serverArgs []string
	if req.ClaudeCode {
		// Recovery must use the exact final launch identity. Build once before
		// looking for a previous scoped log, then rebuild after a recovered OOM
		// may have changed the placement.
		preRecoveryStrategy = strategy
		serverArgs = buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
		strategy, err = recoverPreviousClaudeRuntimeOOM(req, cfg, model, strategy, be, caps, serverArgs)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	}
	if serverArgs == nil || strategy != preRecoveryStrategy {
		serverArgs = buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
	}
	strategy, serverArgs, err = validateAndRepairBackendArgs(req, cfg, model, be, caps, strategy, serverArgs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if warning := longModelLoadWarning(model, autoStartupTimeout(model)); warning != "" {
		fmt.Fprintln(os.Stderr, warning)
	}

	// Start the reviewer on the GPU the final plan chose (CPU when it placed -1).
	claudeAuto, err := startClaudeAutoReviewer(req, cfg, caps, strategy.CompanionPlacements)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	// The gateway also owns model-aware agent admission. Keep it available when
	// the user selected a non-Auto Claude permission mode and no reviewer is
	// needed, so host-offloaded models still avoid destructive decode interleave.
	if req.ClaudeCode && claudeAuto == nil {
		claudeAuto = &claudeAutoRuntime{reviewerGPU: -1}
	}

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

	fmt.Printf("[launch] %s\n", formatCommand(serverArgs))
	if memMax := backendMemoryMaxMB(req, caps); memMax > 0 {
		fmt.Printf("[launch] backend memory scope: MemoryMax=%d MiB\n", memMax)
	}
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
	runtimeCaps, visibleToPhysical := runtimeGPUCapabilities(caps, req)
	baselineVRAM := map[int]int{}
	if runtimeCaps != nil && len(runtimeCaps.GPUs) > 0 {
		for _, g := range runtimeCaps.GPUs {
			baselineVRAM[g.Index] = placement.QueryVRAMUsed(physicalGPUIndex(g.Index, visibleToPhysical))
		}
	}

	// Companions are now resident. This is the resource floor every later
	// promotion/calibration transition must return to before another main-model
	// process may start.
	resourceBaseline := captureLaunchResourceBaseline(caps)
	launchRecovery := newLaunchMemoryRecovery()
	p, strategy, serverArgs, err := startLaunchWithCUDAOOMRecoveryState(req, cfg, model, strategy, be, caps, serverArgs, timeout, launchRecovery)
	if err != nil {
		claudeAuto.stop()
		if releaseErr := stopFailedLaunchBeforeAdvisor(p, caps, 30*time.Second); releaseErr != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", releaseErr)
			os.Exit(1)
		}
		p, strategy, serverArgs, claudeAuto, err = retryStartWithAdvisor(req, cfg, model, be, caps, strategy, err, timeout, launchRecovery)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error starting server: %v\n", err)
			os.Exit(1)
		}
	}

	fmt.Printf("[launch] Server running on port %d (PID %d)\n", req.Port, p.Cmd.Process.Pid)
	// Record launch usage once per successful launch (never per request) so the
	// TUI can sort its model list by real history. Best-effort write.
	modelusage.RecordLaunch(cfg.CacheDir, req.ModelPath)
	if p.LogBuf != nil {
		recordMeasuredLaunchProbes(req, cfg, model, strategy, be, runtimeCaps, p.LogBuf.String(), baselineVRAM, serverProcessPID(p))
	}
	// Consume allocation measurements before performance calibration. A plan
	// with fewer CPU experts is only a new baseline candidate until benchmarked;
	// promoting it after calibration would replace the measured winner with an
	// unbenchmarked strategy and could undo a faster KV alternate.
	if p.LogBuf != nil {
		if retainRecoveredBaselineForAllocationPromotion(req, launchRecovery) {
			fmt.Printf("[launch] retaining the first proven-safe placement after %d rejected memory configuration(s); measurements will seed the next launch\n", launchRecovery.rejectionCount())
		} else if nextStrategy, nextArgs, ok := maybePromoteMeasuredPlacement(req, cfg, be, caps, model, strategy, serverArgs, launchRecovery); ok {
			fmt.Printf("[launch] allocation measurement fits more GPU experts (%d CPU MoE -> %d); establishing the calibrated baseline\n", strategy.NCPUMoE, nextStrategy.NCPUMoE)
			oldStrategy, oldArgs := strategy, append([]string(nil), serverArgs...)
			// restorePrevious brings back the pre-promotion placement after a
			// failed promotion or resource-release failure. The old args are the
			// currently-serving, health-verified argv — never in the rejected set —
			// so the restore-exempt start boundary is used (a deliberate fallback
			// must not be re-gated into a dead box).
			restorePrevious := func() bool {
				p, strategy, serverArgs, err = restoreLaunchWithCUDAOOMRecoveryState(req, cfg, model, oldStrategy, be, caps, oldArgs, timeout, launchRecovery)
				if err != nil {
					claudeAuto.stop()
					fmt.Fprintf(os.Stderr, "Error restoring previous loaded placement: %v\n", err)
					fmt.Fprintln(os.Stderr, "Error: no server is running after failed measured baseline promotion")
					os.Exit(1)
				}
				return true
			}
			if !stopCalibrationProcessAndWait(p, "measured baseline promotion", resourceBaseline, 30*time.Second) {
				// The old process was force-stopped by Stop(); degrade to restoring
				// it rather than exiting with no server. Mirrors the calibration
				// winner-restore path.
				fmt.Fprintln(os.Stderr, "Error: current server/resources did not release before measured baseline promotion; attempting restore of the previous placement")
				restorePrevious()
			} else {
				fmt.Printf("[launch] %s\n", formatCommand(nextArgs))
				promotedP, promotedStrategy, promotedArgs, promoteErr := startLaunchWithCUDAOOMRecoveryState(req, cfg, model, nextStrategy, be, caps, nextArgs, timeout, launchRecovery)
				if promoteErr != nil {
					if !stopCalibrationProcessAndWait(promotedP, "failed measured baseline", resourceBaseline, 30*time.Second) {
						fmt.Fprintln(os.Stderr, "Error: failed measured baseline did not release resources; refusing an overlapping restore — attempting restore of the previous placement")
						// promotedP was force-killed; the old placement's resources
						// are the pre-promotion baseline, so restoring it is safe.
					}
					fmt.Fprintf(os.Stderr, "[launch] measured baseline failed (%v); restoring the previously loaded placement\n", promoteErr)
					restorePrevious()
				} else {
					p, strategy, serverArgs = promotedP, promotedStrategy, promotedArgs
				}
			}
			fmt.Printf("[launch] Server running on port %d (PID %d)\n", req.Port, p.Cmd.Process.Pid)
			if p.LogBuf != nil {
				recordMeasuredLaunchProbes(req, cfg, model, strategy, be, runtimeCaps, p.LogBuf.String(), baselineVRAM, serverProcessPID(p))
			}
		}
	}
	// First-launch optimization now measures alternatives against the final
	// allocation-informed baseline. Its returned decision remains provisional
	// until the winner passes lifecycle and cache verification below.
	var pendingCalibration *placement.CalibrationDecision
	plannedCandidates := calibrationPlan(req, cfg, model, be, caps, strategy, func(candidate *placement.Strategy) bool {
		candidateArgs := buildLaunchServerArgs(req, cfg, be, caps, model, candidate)
		return launchRecovery.isRejected(candidateArgs)
	})
	if len(plannedCandidates) >= 2 {
		p, strategy, serverArgs, pendingCalibration = runCalibration(req, cfg, model, be, caps, strategy, serverArgs, timeout, p, launchRecovery, resourceBaseline, plannedCandidates)
		if p == nil {
			claudeAuto.stop()
			fmt.Fprintf(os.Stderr, "Error: calibration left no running server\n")
			os.Exit(1)
		}
		fmt.Printf("[launch] Server running on port %d (PID %d)\n", req.Port, p.Cmd.Process.Pid)
	} else {
		strategy = explainServingOptimization(req, cfg, model, be, caps, strategy, false)
		printOptimizationSummary("optimize", strategy, false)
	}
	if req.HotExpertPendingNegative != nil {
		// Activation fallback can happen inside a calibration restart or on a
		// direct verified-config start. In both cases its negative result outranks
		// any stale provisional winner and is persisted only after the baseline
		// passes the common lifecycle below.
		pendingCalibration = req.HotExpertPendingNegative
		req.HotExpertPendingNegative = nil
	}
	req.CalibrationPending = pendingCalibration != nil
	// A Claude profile is not verified by llama's OpenAI endpoint alone. Bring
	// up the actual Anthropic gateway first, including admission and chat-role
	// delimiter transforms, so lifecycle activation can canary /v1/messages.
	claudeRouterURL := ""
	if req.ClaudeCode && claudeAuto != nil {
		if err := claudeAuto.startRouter(cfg, req.Host, req.Port, hasArg(serverArgs, "--mmproj"), claudeMainMaxActive(req, strategy, pendingCalibration), serverArgs); err != nil {
			_ = p.Stop()
			claudeAuto.stop()
			fmt.Fprintf(os.Stderr, "Error starting Claude Auto router: %v\n", err)
			os.Exit(1)
		}
		if p.LogBuf != nil {
			if delims := claudeauto.ParseChatMessageDelimiters(p.LogBuf.String()); len(delims) > 0 {
				claudeAuto.setMessageDelimiters(delims)
				fmt.Printf("[claude-code] chat delimiters read from the backend: %s\n", formatMessageDelimiters(delims))
			}
		}
		if claudeAuto.router != nil {
			claudeRouterURL = claudeAuto.router.URL()
		}
	}
	verificationErr := verifyAndActivateLaunch(req, cfg, model, be, runtimeCaps, strategy, serverArgs, claudeRouterURL)
	if verificationErr != nil && automaticHotExpertFallbackAllowed(req, strategy) {
		hotVerificationErr := verificationErr
		failedHotStrategy := strategy
		negativeSource := hotExpertFailureDecisionSource(req, cfg, model, be, caps, strategy, pendingCalibration)
		fmt.Fprintf(os.Stderr, "[hot-experts] cache-on lifecycle verification failed (%v); verifying the exact cache-free baseline before attributing the failure\n", verificationErr)
		if !stopCalibrationProcessAndWait(p, "failed hot-expert lifecycle verification", resourceBaseline, 30*time.Second) {
			verificationErr = fmt.Errorf("%w; cache-on process did not release cleanly", verificationErr)
		} else {
			if invalidateErr := invalidateAutomaticHotExpertEvidence(req, cfg, model, be, caps, strategy); invalidateErr != nil {
				fmt.Fprintf(os.Stderr, "[hot-experts] warning: failed to invalidate all cache-on evidence: %v\n", invalidateErr)
			}
			pendingCalibration = hotExpertUnavailableDecision(negativeSource, failedHotStrategy,
				"cache-on lifecycle failed while the exact cache-free baseline was awaiting verification: "+hotVerificationErr.Error(),
			)
			req.CalibrationPending = pendingCalibration != nil
			fallback := placement.WithoutHotExpertCache(strategy)
			if fallback == nil {
				verificationErr = fmt.Errorf("%w; cache-free fallback returned no strategy", verificationErr)
			} else {
				fallback.PerformanceTuned = false
				fallback.VerifiedConfigReused = false
				claudeCodeSlotAdjust(fallback, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
				fallbackArgs := buildLaunchServerArgs(req, cfg, be, caps, model, fallback)
				fallbackP, fallbackStrategy, actualFallbackArgs, fallbackErr := restoreLaunchWithCUDAOOMRecoveryState(
					req, cfg, model, fallback, be, caps, fallbackArgs, timeout, launchRecovery,
				)
				if fallbackErr != nil || fallbackP == nil {
					if fallbackErr == nil {
						fallbackErr = errors.New("cache-free fallback returned no process")
					}
					verificationErr = fmt.Errorf("%w; cache-free fallback failed to start: %v", verificationErr, fallbackErr)
				} else if !exactCalibrationCandidate(fallbackArgs, actualFallbackArgs) {
					_ = fallbackP.Stop()
					verificationErr = fmt.Errorf("%w; cache-free fallback required a different recovery argv", verificationErr)
				} else if fallbackVerifyErr := verifyAndActivateLaunch(
					req, cfg, model, be, runtimeCaps, fallbackStrategy, actualFallbackArgs, claudeRouterURL,
				); fallbackVerifyErr != nil {
					_ = fallbackP.Stop()
					verificationErr = fmt.Errorf("%w; cache-free baseline also failed verification: %v", verificationErr, fallbackVerifyErr)
				} else {
					p, strategy, serverArgs = fallbackP, fallbackStrategy, actualFallbackArgs
					verificationErr = nil
					if pendingCalibration != nil {
						pendingCalibration.FinalistFailureReason = "cache-on lifecycle failed while the exact cache-free baseline passed: " + hotVerificationErr.Error()
					}
					req.CalibrationPending = pendingCalibration != nil
				}
			}
		}
	}
	if verificationErr != nil {
		_ = p.Stop()
		claudeAuto.stop()
		fmt.Fprintf(os.Stderr, "Error verifying server profile: %v\n", verificationErr)
		os.Exit(1)
	}
	checkpointPlanChanged := false
	if p.LogBuf != nil {
		checkpointPlanChanged = recordLiveCheckpointEvidence(req, cfg, model, strategy, be, runtimeCaps, p.LogBuf.String())
	}
	if checkpointPlanChanged {
		// verifyAndActivateLaunch persists ordinary profiles at StateActive. The
		// canary measurement above is newer than that write, so refresh the record
		// before this process tightens its cgroup. Pending calibration profiles are
		// intentionally withheld here and saved by the promotion block below.
		saveVerifiedConfigForLaunch(cfg, req, model, be, runtimeCaps, strategy)
	}
	// The functional canary is the first real request: it grows the graph and
	// allocates the first context checkpoints, so this is the earliest honest
	// measurement of the backend's resident footprint. Re-size the scope to
	// measured+headroom now that a wrong pre-launch plan would have already
	// failed.
	resizeScopeToMeasuredFootprint(req, runtimeCaps, strategy, p)
	if pendingCalibration != nil {
		scope := launchProfileScope(req, model, be, runtimeCaps)
		active := controller.Store{CacheDir: cfg.CacheDir}.IsActive(scope, controller.HashArgs(serverArgs))
		mode := effectiveCalibrationMode(req)
		if !active {
			fmt.Fprintf(os.Stderr, "[calibrate] screened winner did not reach an active profile; decision not cached\n")
		} else {
			performanceEvidence := mode != calibrateAuto || automaticCalibrationEvidenceValid(pendingCalibration)
			admissionEvidence := mode == calibrateAuto && automaticCalibrationAdmissionEvidenceValid(pendingCalibration)
			if !performanceEvidence && !admissionEvidence {
				fmt.Fprintf(os.Stderr, "[optimize] winner reached active state but agent-workflow evidence was incomplete; decision not promoted\n")
				req.CalibrationPending = false
			} else {
				if mode == calibrateAuto && performanceEvidence {
					pendingCalibration.ValidationLevel = placement.CalibrationValidationWorkflow
				} else if admissionEvidence {
					pendingCalibration.ValidationLevel = placement.CalibrationValidationAdmission
					pendingCalibration.RecordAdmissionRejection(
						pendingCalibration.Finalist,
						pendingCalibration.FinalistFailureClass,
						pendingCalibration.FinalistFailureReason,
					)
					if previous, loadErr := placement.LoadCalibrationDecision(cfg.CacheDir, pendingCalibration.ScopeKey); loadErr == nil {
						pendingCalibration.MergeAdmissionRejections(previous)
					}
				}
				path, saveErr := placement.SaveCalibrationDecision(cfg.CacheDir, *pendingCalibration)
				if saveErr != nil {
					fmt.Fprintf(os.Stderr, "[optimize] active winner cache failed: %v\n", saveErr)
				} else if mode == calibrateAuto {
					// Only now does the winner become reusable core state. The earlier
					// StateActive transition deliberately skipped verified/place writes
					// while this decision was pending, so a decision-cache failure can
					// never strand a falsely tuned verified config.
					strategy.PerformanceTuned = performanceEvidence
					req.CalibrationPending = false
					saveVerifiedConfigForLaunch(cfg, req, model, be, runtimeCaps, strategy)
					if strategy.Type == placement.MoEOffload && strategy.PlacementCachePath != "" {
						if placeErr := placement.SavePlacementCache(strategy.PlacementCachePath, placement.StrategyToCacheEntry(strategy)); placeErr != nil {
							fmt.Fprintf(os.Stderr, "[optimize] verified placement cache write failed (launch unaffected): %v\n", placeErr)
						}
					}
					if performanceEvidence {
						fmt.Printf("[optimize] workflow winner %s passed clean relaunch, agent, cache, and lifecycle gates; cached at %s\n",
							pendingCalibration.Winner, path)
					} else {
						fmt.Printf("[optimize] exact finalist %s was unavailable; cached scoped negative evidence so identical launches keep the verified baseline without another reload (%s)\n",
							pendingCalibration.Finalist, path)
					}
				} else {
					fmt.Printf("[calibrate] screened winner %s passed launch canaries and was cached for explicit reuse at %s; it is not automatic-eligible without the full workflow gate\n",
						pendingCalibration.Winner, path)
				}
			}
		}
		req.CalibrationPending = false
	}
	// A benchmarked Claude profile measures the exact Claude placement policy
	// (reviewer reservation, slots, batch, sampling) without opening the
	// interactive Claude client. This keeps calibration runs unattended and
	// guarantees the measured server is stopped when the one-shot probe ends.
	if req.Benchmark || req.WorkerBenchmark {
		var benchmarkErr error
		if req.WorkerBenchmark {
			usedVRAMMB := measuredLaunchVRAMMB(runtimeCaps, visibleToPhysical, baselineVRAM)
			benchmarkErr = runOneShotWorkerBenchmark(req.Port, filepath.Base(req.ModelPath), usedVRAMMB)
		} else {
			benchmarkErr = runOneShotBenchmark(req.Port, filepath.Base(req.ModelPath))
		}
		stopErr := p.Stop()
		if stopErr != nil {
			fmt.Fprintf(os.Stderr, "[launch] stop after benchmark: %v\n", stopErr)
		}
		claudeAuto.stop()
		if benchmarkErr != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", benchmarkErr)
			os.Exit(1)
		}
		return
	}
	claudeClientPort := req.Port
	if claudeAuto != nil {
		claudeClientPort = claudeAuto.clientPort(req.Port)
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
					logData := ""
					if p.LogBuf != nil {
						logData = p.LogBuf.String()
					}
					markerPath := claudeOOMMarkerPath(cfg, req, model, be, serverArgs)
					peakMB, hostOOM, recorded, hostErr := recordRuntimeHostCgroupOOM(req, cfg, model, strategy, be, runtimeCaps, serverArgs, p, logData, markerPath)
					if hostErr != nil {
						fmt.Fprintf(os.Stderr, "[launch] could not record host cgroup OOM from health monitor: %v\n", hostErr)
					} else if hostOOM {
						if recorded {
							fmt.Fprintf(os.Stderr, "[launch] health monitor recorded host cgroup OOM at %d MiB; revoked the profile and cached host-state evidence for the next launch\n", peakMB)
						}
					} else if p.LogBuf != nil {
						device, reserveMB, estimated, _, ok, recordErr := recordRuntimeOOMLog(req, cfg, model, strategy, be, caps, logData, markerPath)
						if recordErr != nil {
							fmt.Fprintf(os.Stderr, "[launch] could not record backend OOM from health monitor: %v\n", recordErr)
						} else if ok && estimated {
							fmt.Fprintf(os.Stderr, "[launch] health monitor recorded CUDA VMM OOM on device %d and a %d MiB reserve for the next launch\n", device, reserveMB)
						} else if ok {
							fmt.Fprintf(os.Stderr, "[launch] health monitor recorded CUDA OOM on device %d and a %d MiB reserve for the next launch\n", device, reserveMB)
						}
					}
					healthCancel()
					return
				}
			}
		}()

		clientHost := req.Host
		if claudeAuto != nil {
			clientHost = "127.0.0.1"
		}
		clientArgs, statusLineEnabled := claudeCodeProgressClientArgs(nil, claudeClientPort)
		progressEnabled := !progressDisabled()
		progressStop := func() {}
		if progressEnabled {
			progressStop = startClaudeProgressMonitor(clientHost, claudeClientPort, p.LogBuf, !statusLineEnabled)
		}
		defer progressStop()
		if !progressEnabled {
			fmt.Println("[claude-code] Live request progress disabled by GGRUN_CLAUDE_PROGRESS.")
		} else if statusLineEnabled {
			fmt.Println("[claude-code] Live request progress enabled in Claude's status line.")
		} else {
			fmt.Println("[claude-code] Live request progress enabled in the terminal title (existing Claude status line preserved).")
		}
		sessionSpec, sessionErr := claudeLaunchSession(cfg, req, serverArgs)
		if sessionErr != nil {
			progressStop()
			healthCancel()
			_ = p.Stop()
			claudeAuto.stop()
			fmt.Fprintf(os.Stderr, "Error: %v\n", sessionErr)
			os.Exit(1)
		}
		if code := runClaudeCodeClient(clientHost, claudeClientPort, serverArgs, clientArgs, sessionSpec, strategy.ContextSize); code >= 0 {
			progressStop()
			healthCancel()
			// Record on exit as well as on launch: the workflow run ID is
			// assigned inside Claude Code, so only now is the resume handle
			// complete.
			refreshClaudeSessionRecord(cfg.CacheDir, sessionSpec, req.ModelPath, req.Backend,
				req.Port, claudeStripResumeArgs(req.OriginalArgs), serverArgs)
			// The terminal was handed to `claude`, so a mid-session backend
			// crash isn't something this process can retry live — but it can
			// still be recorded before Stop(), so the NEXT `--claude-code`
			// launch of this exact model/context reserves the measured
			// deficit instead of repeating the same crash blind.
			if !p.IsRunning() && p.LogBuf != nil {
				markerPath := claudeOOMMarkerPath(cfg, req, model, be, serverArgs)
				logData := p.LogBuf.String()
				peakMB, hostOOM, recorded, hostErr := recordRuntimeHostCgroupOOM(req, cfg, model, strategy, be, runtimeCaps, serverArgs, p, logData, markerPath)
				if hostErr != nil {
					fmt.Fprintf(os.Stderr, "[launch] could not record host cgroup OOM: %v\n", hostErr)
				} else if hostOOM {
					if recorded {
						fmt.Fprintf(os.Stderr, "[launch] backend crashed during this session (host cgroup OOM at %d MiB) — profile revoked and host-state evidence recorded for the next launch.\n", peakMB)
					}
				} else if device, reserveMB, estimated, _, ok, recordErr := recordRuntimeOOMLog(req, cfg, model, strategy, be, caps, logData, markerPath); recordErr != nil {
					fmt.Fprintf(os.Stderr, "[launch] could not record backend OOM: %v\n", recordErr)
				} else if ok && estimated {
					fmt.Fprintf(os.Stderr, "[launch] backend crashed during this session (CUDA VMM OOM on device %d; allocation size omitted) — recorded %d MiB runtime reserve for the next launch.\n", device, reserveMB)
				} else if ok {
					fmt.Fprintf(os.Stderr, "[launch] backend crashed during this session (CUDA OOM on device %d, %d MiB) — recorded, next launch of this model/context will reserve for it.\n", device, reserveMB)
				}
			}
			if err := p.Stop(); err != nil {
				fmt.Fprintf(os.Stderr, "[launch] stop after claude: %v\n", err)
			}
			claudeAuto.stop()
			os.Exit(code)
		}
		// `claude` isn't installed — fall back to the copy-paste recipe.
		printClaudeCodeRecipe(clientHost, claudeClientPort, serverArgs)
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
		if peakMB, hostOOM, _, hostErr := recordRuntimeHostCgroupOOM(req, cfg, model, strategy, be, runtimeCaps, serverArgs, p, logData, ""); hostOOM {
			claudeAuto.stop()
			if hostErr != nil {
				fmt.Fprintf(os.Stderr, "[launch] server was killed by its host-memory cgroup at %d MiB, but profile invalidation failed: %v\n", peakMB, hostErr)
			} else {
				fmt.Fprintf(os.Stderr, "[launch] server was killed by its host-memory cgroup at %d MiB. The failed profile was revoked and prompt/checkpoint evidence was cached; the next launch will derive a fresh host plan.\n", peakMB)
			}
			os.Exit(1)
		}
		cacheBackendTag := scopedProbeBackendTagForStrategy(req, model, be, strategy)
		prior := placement.RuntimeGraphGrowthByGPU(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, cacheBackendTag, caps.GPUs, strategy.Parallel)
		device, allocMB, estimated, ok := runtimeLogCUDAOOM(logData, caps, model, prior)
		if !ok {
			claudeAuto.stop()
			fmt.Fprintln(os.Stderr, "[launch] server exited unexpectedly (not a recognized CUDA OOM) — see the log for details.")
			os.Exit(1)
		}

		reason := fmt.Sprintf("CUDA OOM on device %d after health verification", device)
		if err := invalidateRuntimeOOMLaunch(req, cfg, model, be, runtimeCaps, strategy, serverArgs, reason); err != nil {
			claudeAuto.stop()
			fmt.Fprintf(os.Stderr, "[launch] cannot invalidate runtime-failed profile: %v\n", err)
			os.Exit(1)
		}
		if err := placement.RecordRuntimeGraphGrowthFromOOM(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, cacheBackendTag, caps.GPUs, strategy.Parallel, device, allocMB, estimated); err != nil {
			claudeAuto.stop()
			fmt.Fprintf(os.Stderr, "[launch] cannot persist runtime OOM evidence: %v\n", err)
			os.Exit(1)
		}
		if runtimeOOMRetries >= maxRuntimeOOMRetries {
			claudeAuto.stop()
			fmt.Fprintf(os.Stderr, "[launch] server crashed (CUDA OOM on device %d, %d MiB) after %d recovery attempt(s) — giving up. The failed profile and placement cache were revoked; the recorded deficit will drive the next launch.\n", device, allocMB, runtimeOOMRetries)
			os.Exit(1)
		}

		runtimeOOMRetries++
		if estimated {
			fmt.Fprintf(os.Stderr, "[launch] server crashed after health check: CUDA VMM OOM on device %d omitted its allocation size — reserving %d MiB, re-planning and relaunching (attempt %d/%d)...\n",
				device, allocMB, runtimeOOMRetries, maxRuntimeOOMRetries)
		} else {
			fmt.Fprintf(os.Stderr, "[launch] server crashed after health check: CUDA OOM on device %d needing %d MiB more mid-request — recorded, re-planning and relaunching (attempt %d/%d)...\n",
				device, allocMB, runtimeOOMRetries, maxRuntimeOOMRetries)
		}

		nextStrategy, nextArgs, err := replanAfterRuntimeOOM(req, cfg, model, be, caps, serverArgs, launchRecovery)
		if err != nil {
			claudeAuto.stop()
			fmt.Fprintf(os.Stderr, "[launch] re-plan after runtime OOM failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("[launch] %s\n", formatCommand(nextArgs))
		newP, newStrategy, newArgs, err := startLaunchWithCUDAOOMRecoveryState(req, cfg, model, nextStrategy, be, caps, nextArgs, timeout, launchRecovery)
		if err != nil {
			claudeAuto.stop()
			fmt.Fprintf(os.Stderr, "[launch] relaunch after runtime OOM failed: %v\n", err)
			os.Exit(1)
		}
		if newP.LogBuf != nil {
			recordMeasuredLaunchProbes(req, cfg, model, newStrategy, be, runtimeCaps, newP.LogBuf.String(), baselineVRAM, serverProcessPID(newP))
		}
		if err := verifyAndActivateLaunch(req, cfg, model, be, runtimeCaps, newStrategy, newArgs, claudeRouterURL); err != nil {
			_ = newP.Stop()
			claudeAuto.stop()
			fmt.Fprintf(os.Stderr, "[launch] recovered placement failed lifecycle verification: %v\n", err)
			os.Exit(1)
		}
		if newP.LogBuf != nil {
			if recordLiveCheckpointEvidence(req, cfg, model, newStrategy, be, runtimeCaps, newP.LogBuf.String()) {
				saveVerifiedConfigForLaunch(cfg, req, model, be, runtimeCaps, newStrategy)
			}
		}
		resizeScopeToMeasuredFootprint(req, runtimeCaps, newStrategy, newP)
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
		p.Kill()
	case <-time.After(30 * time.Second):
		fmt.Fprintln(os.Stderr, "[launch] Timeout — forcing shutdown...")
		p.Kill()
	}
	claudeAuto.stop()
}

func invalidateRuntimeOOMLaunch(req *launchRequest, cfg *config.Config, model *placement.ModelProfile,
	be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy, serverArgs []string, reason string,
) error {
	if req == nil || cfg == nil || model == nil || be == nil || strategy == nil {
		return errors.New("incomplete runtime OOM profile")
	}
	scope := launchProfileScope(req, model, be, caps)
	store := controller.Store{CacheDir: cfg.CacheDir}
	if _, err := store.RejectActiveIfMatch(scope, controller.HashArgs(serverArgs), reason, "runtime-oom"); err != nil {
		return fmt.Errorf("revoke active profile: %w", err)
	}
	if path := strings.TrimSpace(strategy.PlacementCachePath); path != "" {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("remove stale placement cache: %w", err)
		}
	}
	// A runtime OOM is evidence the declared winner is unsafe at runtime. The
	// decision may have been saved under the default strategy's scope key
	// (runCalibration hashes the pre-calibration strategy) while the serving
	// strategy is the winner (kv-alternate/split-inverted), whose scope key
	// differs. Sweep the default plus every calibration candidate so whichever
	// key the decision was saved under is removed — otherwise the OOM'd
	// placement is re-declared the winner on the next launch.
	keys := []string{calibrationScopeKey(req, model, be, caps, strategy)}
	// The serving winner is not guaranteed to contain enough of the cold
	// baseline to reverse-generate its scope (a KV alternate, for example, does
	// not carry the default KV placement). Recompute the deterministic baseline
	// with full-config and placement replay disabled, then delete the exact key
	// under which runCalibration originally saved the decision.
	defaultOpts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	defaultOpts.VerifiedConfigScopeKey = ""
	defaultOpts.SkipPlacementCache = true
	defaultOpts.CacheFile = ""
	if defaultStrategy, computeErr := placement.Compute(caps, model, defaultOpts); computeErr == nil && defaultStrategy != nil {
		keys = append(keys, calibrationScopeKey(req, model, be, caps, defaultStrategy))
	}
	for _, cand := range calibrationCandidates(req, cfg, model, be, caps, strategy) {
		if cand.Strategy == nil || cand.Strategy == strategy {
			continue
		}
		keys = append(keys, calibrationScopeKey(req, model, be, caps, cand.Strategy))
	}
	for _, key := range keys {
		if err := placement.DeleteCalibrationDecision(cfg.CacheDir, key); err != nil {
			return fmt.Errorf("remove stale calibration decision: %w", err)
		}
	}
	// A ubatch winner does not naturally generate the lower cold default as a
	// reverse candidate, so its serving scope cannot reveal the key under which
	// the decision was saved. Production decisions carry ModelBasename; remove
	// every decision for this model after a runtime OOM rather than risk replay.
	if err := placement.DeleteCalibrationDecisionsForModel(cfg.CacheDir, model); err != nil {
		return fmt.Errorf("remove model calibration decisions: %w", err)
	}
	// A runtime OOM is evidence the saved full config is unsafe at runtime. The
	// verified-config record is scoped by the same strategy-free key the reuse
	// path hashes against, so delete it so the OOM'd placement is never
	// replayed verbatim on the next launch.
	if key := verifiedConfigScopeKey(req, model, be, caps); key != "" {
		if err := placement.DeleteVerifiedConfig(cfg.CacheDir, key); err != nil {
			return fmt.Errorf("remove stale verified config: %w", err)
		}
	}
	return nil
}

// replanAfterRuntimeOOM derives a fresh placement after a post-health CUDA
// OOM, honoring the growth deficit the caller just recorded. It rejects the
// argv that just crashed on the shared lifecycle recovery state, then refuses
// to hand back an argv identical to it: a fresh derivation can legitimately
// reproduce the same placement when the recorded deficit fits inside the
// plan's slack, and re-running it would just re-crash identically. Every
// other recovery path (preflight, startup OOM, measured promotion,
// calibration candidates) already refuses a rejected argv; this is the one
// path that previously relied on the retry counter alone.
func replanAfterRuntimeOOM(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, serverArgs []string, launchRecovery *launchMemoryRecovery) (*placement.Strategy, []string, error) {
	if launchRecovery != nil {
		launchRecovery.reject(serverArgs)
	}
	replanOpts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	// Without this, Compute() prefers the .place cache written when the
	// prior instance loaded cleanly and passed health — which is exactly
	// the placement that just OOM'd mid-request. Skipping it forces a
	// fresh derivation that actually consults the growth deficit the
	// caller just recorded via RecordRuntimeGraphGrowthFromOOM.
	replanOpts.SkipPlacementCache = true
	nextStrategy, err := placement.Compute(caps, model, replanOpts)
	if err != nil {
		return nil, nil, err
	}
	claudeCodeSlotAdjust(nextStrategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	nextArgs := buildLaunchServerArgs(req, cfg, be, caps, model, nextStrategy)
	if formatCommand(nextArgs) == formatCommand(serverArgs) {
		return nil, nil, fmt.Errorf("runtime OOM re-plan reproduced the exact failed argv; refusing an identical relaunch")
	}
	return nextStrategy, nextArgs, nil
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
	if req.ClaudeCode {
		// Generic and community throughput tunes do not encode the agent
		// scheduler, context, cache, or semantic-validation workload. They may
		// override -b/-ub after Claude's fairness adjustment, so fail closed until
		// the validation registry can select profile-scoped evidence.
		fmt.Println("[tune] Skipping automatic generic/community tune for Claude Code; use an explicit --tune-cache after workload validation.")
		return serverArgs
	}
	path := bestTuneCachePath(cacheDir, filepath.Base(req.ModelPath), backendTag, vision, "", tuneHardwareHash(caps))
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

func bestTuneCachePath(cacheDir, modelName, backendTag string, vision bool, workload, hardwareHash string) string {
	if cacheDir == "" || modelName == "" {
		return ""
	}
	rows := tune.ListTunedConfigs(cacheDir, modelName, tuneCacheBackendTag(backendTag), vision)
	for _, row := range rows {
		if row.Workload != workload {
			continue
		}
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
	cfg := loadConfigOrExit()
	req.ModelPath = resolveModelPath(req.ModelPath, cfg.ModelDir)
	model, err := parseModel(req.ModelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing model: %v\n", err)
		os.Exit(1)
	}
	be := resolveLaunchBackend(req, model, caps)
	binPath := "llama-server"
	if be != nil {
		binPath = be.Path
	} else if req.BackendExplicit {
		fmt.Fprintf(os.Stderr, "Error: %s\n", backendUnavailableMessage(req))
		os.Exit(1)
	} else {
		be = &backendInfo{Path: binPath, Tag: "llama"}
	}
	applyCachedBackendCapabilities(req, cfg.CacheDir, model, be)
	placementOpts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	strategy, err := placement.Compute(caps, model, placementOpts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %s\n", placementErrorMessage(err))
		os.Exit(1)
	}
	if err := confirmRequiredMMap(req, strategy, os.Stdin, os.Stderr, stdinIsTerminal()); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if err := validateHostMemoryContainment(req, caps, strategy); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
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
	var pendingReview *tui.LaunchRequest
	for {
		var (
			req *tui.LaunchRequest
			err error
		)
		if pendingReview != nil {
			req, err = tui.RunAfterBackendInstall(pendingReview)
			pendingReview = nil
		} else {
			req, err = tui.Run()
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		if req == nil {
			return
		}
		if req.Update {
			cmdUpdate(nil)
			return
		}
		if len(req.BackendArgs) > 0 {
			cmdBackend(req.BackendArgs)
			if req.ModelPath != "" {
				// A model-aware install carries the current launch settings. Once
				// the recipe registers its architecture route, return to a review
				// screen with that route selected instead of making the user find
				// and configure the model again.
				copyReq := *req
				copyReq.BackendArgs = nil
				pendingReview = &copyReq
				continue
			}
			return
		}

		cfg := loadConfigOrExit()

		if req.DownloadRepo != "" {
			caps, err := detect.Detect()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
				os.Exit(1)
			}
			modelDir := cfg.ModelDir
			if req.DownloadDir != "" {
				modelDir = expandPath(req.DownloadDir)
			}
			if err := os.MkdirAll(modelDir, 0o755); err != nil {
				fmt.Fprintf(os.Stderr, "Error: cannot use %s: %v\n", modelDir, err)
				os.Exit(1)
			}
			if modelDir != cfg.ModelDir {
				fmt.Fprintf(os.Stderr, "[download] destination: %s\n", modelDir)
			}
			d := download.New(modelDir, cfg.CacheDir, cfg.AppHome)
			if err := d.RunQuant(req.DownloadRepo, req.DownloadQuant, caps); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			return
		}

		launchArgs := tuiLaunchArgs(req, cfg)
		if err := tui.SaveLatestLaunch(cfg.CacheDir, req); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not save latest TUI launch configuration: %v\n", err)
		}
		// Launch usage is recorded inside cmdLaunch (the shared real-server
		// path for both CLI and TUI launches), so no record is needed here; the
		// AI-tune path below intentionally skips a server launch.
		if req.AITune {
			cmdTune(launchArgs)
			return
		}
		cmdLaunch(launchArgs)
		return
	}
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

	cfg := loadConfigOrExit()
	req.ModelPath = resolveModelPath(req.ModelPath, cfg.ModelDir)

	model, err := parseModel(req.ModelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing model: %v\n", err)
		os.Exit(1)
	}
	warnModelCompatibility(model)

	be := resolveLaunchBackend(req, model, caps)
	if be == nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", backendUnavailableMessage(req))
		os.Exit(1)
	}
	applyCachedBackendCapabilities(req, cfg.CacheDir, model, be)
	// Match cmdLaunch exactly: Claude Auto's helper consumes VRAM before the
	// main model is packed. Without this reservation a dry-run can claim one
	// additional expert layer fits and disagree with the real launch.
	req.ReviewerReservation = claudeReviewerReservation(req, caps, cfg.CacheDir)

	placementOpts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	strategy, err := placement.Compute(caps, model, placementOpts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %s\n", placementErrorMessage(err))
		os.Exit(1)
	}
	claudeCodeSlotAdjust(strategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	strategy = applyCalibrationDecision(req, cfg, model, be, caps, strategy)
	strategy = explainServingOptimization(req, cfg, model, be, caps, strategy, true)

	if os.Getenv("GGRUN_TRACE_PLACEMENT") != "" {
		printVRAMLedger(strategy)
		fmt.Printf("[trace] NCPUMoE=%d OT=%s\n", strategy.NCPUMoE, strategy.OTString)
	}

	serverArgs := buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
	envPrefix := applyGPUVisibility(req, backendDialect(be))
	if req.EmitServerArgvJSON {
		plan := struct {
			Schema               string                          `json:"schema"`
			ModelPath            string                          `json:"model_path"`
			BackendTag           string                          `json:"backend_tag"`
			BackendID            string                          `json:"backend_identity"`
			ClaudeProfile        string                          `json:"claude_profile,omitempty"`
			LaunchScope          string                          `json:"launch_scope,omitempty"`
			MemoryMaxMB          int                             `json:"memory_max_mb,omitempty"`
			Environment          map[string]string               `json:"environment"`
			ServerArgv           []string                        `json:"server_argv"`
			Residency            placement.ResidencyClass        `json:"residency,omitempty"`
			Bottleneck           string                          `json:"optimization_bottleneck,omitempty"`
			ResourceLedger       *placement.ResourceLedger       `json:"resource_ledger,omitempty"`
			OptimizationBoundary *placement.OptimizationBoundary `json:"optimization_boundary,omitempty"`
			CachedOptimization   *placement.CalibrationDecision  `json:"cached_optimization,omitempty"`
		}{
			Schema:        "ggrun-server-plan-v1",
			ModelPath:     req.ModelPath,
			BackendTag:    be.Tag,
			BackendID:     be.Identity,
			ClaudeProfile: effectiveClaudeProfile(req),
			// The name a crash log would be filed under. Exposed because it is a
			// hash: when recovery silently fails to find a previous OOM, the only
			// way to see why is to compare this between two runs.
			LaunchScope:          claudeLaunchLogScope(req, model, be, serverArgs),
			MemoryMaxMB:          backendMemoryMaxMB(req, caps),
			Environment:          launchPlanEnvironment(serverArgs, envPrefix, be.Path),
			ServerArgv:           serverArgs,
			Residency:            strategy.Residency,
			Bottleneck:           strategy.OptimizationBottleneck,
			ResourceLedger:       strategy.ResourceLedger,
			OptimizationBoundary: strategy.OptimizationBoundary,
			CachedOptimization:   req.AppliedCalibration,
		}
		if err := json.NewEncoder(os.Stdout).Encode(plan); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing launch plan: %v\n", err)
			os.Exit(1)
		}
		return
	}
	printOptimizationSummary("dry-run", strategy, true)
	if envPrefix != "" {
		fmt.Print(envPrefix + " ")
	}
	fmt.Println(formatCommand(serverArgs))
	if s := placement.DraftSummary(strategy.Draft); s != "" {
		fmt.Printf("[spec] %s\n", s)
	}
	printDryRunClaudeNotice(req.ClaudeCode, req.ClaudeReviewerDisabled)
}

// printDryRunClaudeNotice closes a dry-run with the same promise the real
// launch keeps: which Claude Code companion (if any) will start. With the
// reviewer off no reviewer process starts, so promising one would contradict
// the shipped v3.2.8 behavior; say what does happen instead.
func printDryRunClaudeNotice(claudeCode, reviewerDisabled bool) {
	if !claudeCode {
		return
	}
	if reviewerDisabled {
		fmt.Println("[claude-code] A real launch opens Claude Code with reviews handled by the main model (no separate reviewer, per --claude-reviewer off).")
		return
	}
	fmt.Println("[claude-code] A real launch also starts the local Auto reviewer/router and then opens Claude Code.")
}

// launchPlanEnvironment exports the process settings that ggrun's real server
// launcher applies implicitly. A machine-readable argv without these settings
// is not an equivalent launch: in particular, CUDA1/CUDA2 tensor overrides can
// address different physical cards unless CUDA_DEVICE_ORDER=PCI_BUS_ID is
// preserved. Keep the allowlist narrow so a plan never serializes unrelated or
// secret parent-process environment variables.
func launchPlanEnvironment(serverArgs []string, envPrefix string, backendPath ...string) map[string]string {
	allowed := map[string]bool{
		"CUDA_DEVICE_ORDER":        true,
		"CUDA_SCALE_LAUNCH_QUEUES": true,
		"LD_LIBRARY_PATH":          true,
	}
	result := map[string]string{}
	for _, item := range server.ChildEnv(os.Environ(), serverArgs) {
		key, value, ok := strings.Cut(item, "=")
		if ok && allowed[key] {
			result[key] = value
		}
	}
	if key, value, ok := strings.Cut(envPrefix, "="); ok && key != "" {
		result[key] = value
	}
	if len(backendPath) > 0 {
		if stable, ok := libhub.StableLibraryPath(backendPath[0]); ok {
			if inherited := result["LD_LIBRARY_PATH"]; inherited != "" {
				stable += ":" + inherited
			}
			result["LD_LIBRARY_PATH"] = stable
		}
	}
	return result
}

// printClaudeCodeRecipe prints the exact env to point Claude Code at this
// local ggrun endpoint. In Auto mode the port belongs to ggrun's loopback
// router; normal turns stream to the main llama-server and hidden safety turns
// go to the dedicated reviewer.
func printClaudeCodeRecipe(host string, port int, serverArgs []string) {
	clientHost := host
	if clientHost == "" || clientHost == "0.0.0.0" || clientHost == "::" {
		clientHost = "127.0.0.1"
	}
	window := claudeCodeAutocompactWindow(serverArgs, 0)
	pct := claudeCodeAutocompactPct(serverArgs, 0)
	slot := ""
	if ctx := argIntValue(serverArgs, "--ctx-size", "-c", "--ctx"); ctx > 0 {
		par := argIntValue(serverArgs, "--parallel", "-np")
		if par < 1 {
			par = 1
		}
		slot = fmt.Sprintf(" (~%dk per slot at --parallel %d)", ctx/par/1000, par)
	}
	// Foreground tiers use the main alias; Claude's cheap tier uses the router's
	// local-fast label so it reaches the verified worker without leaving ggrun.
	fmt.Println()
	fmt.Println("[claude-code] In another terminal:")
	// Match claudeCodeEnv: drop any real key so the dummy token + local base URL win,
	// otherwise Claude Code prefers the real key and routes to api.anthropic.com.
	fmt.Println("  unset ANTHROPIC_API_KEY")
	fmt.Printf("  export ANTHROPIC_BASE_URL=http://%s:%d ANTHROPIC_AUTH_TOKEN=ggrun\n", clientHost, port)
	fmt.Printf("  export ANTHROPIC_MODEL=local ANTHROPIC_SMALL_FAST_MODEL=%s\n", claudeauto.UtilityAlias)
	fmt.Printf("  export ANTHROPIC_DEFAULT_HAIKU_MODEL=%s ANTHROPIC_DEFAULT_SONNET_MODEL=local ANTHROPIC_DEFAULT_OPUS_MODEL=local\n", claudeauto.UtilityAlias)
	fmt.Printf("  export CLAUDE_CODE_EFFORT_LEVEL=%s  # xhigh is the agentic default; set max for one demanding session\n", envOr("CLAUDE_CODE_EFFORT_LEVEL", "xhigh"))
	fmt.Printf("  export API_TIMEOUT_MS=%d  # maximum safe timer: no practical local-inference deadline\n", claudeNoTimeoutMS)
	fmt.Printf("  export CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS=%d  # background agents may be quiet during local prefill\n", claudeNoTimeoutMS)
	fmt.Println("  export CLAUDE_ENABLE_BYTE_WATCHDOG=0 CLAUDE_ENABLE_STREAM_WATCHDOG=0 API_FORCE_IDLE_TIMEOUT=0")
	fmt.Printf("  export CLAUDE_CODE_AUTO_COMPACT_WINDOW=%d CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=%d  # compact early to fit the real slot%s\n", window, pct, slot)
	claudeArgs, _ := claudeCodeProgressClientArgs(nil, port)
	claudeArgs = claudeCodeWorkflowPromptArgs(claudeArgs)
	claudeArgs = append(claudeCodePermissionArgs(claudeArgs), claudeArgs...)
	claudeArgs = append(claudeArgs, "--disallowedTools", "WebSearch")
	if _, err := exec.LookPath("uvx"); err == nil {
		claudeArgs = append(claudeArgs,
			"--allowedTools", "mcp__ddg-search__search,mcp__ddg-search__fetch_content",
			"--mcp-config", `{"mcpServers":{"ddg-search":{"command":"uvx","args":["duckduckgo-mcp-server"]}}}`,
		)
		fmt.Printf("  %s\n", formatCommand(append([]string{"claude"}, claudeArgs...)))
	} else {
		fmt.Printf("  %s   # add a search MCP (e.g. uvx duckduckgo-mcp-server) for web research\n", formatCommand(append([]string{"claude"}, claudeArgs...)))
	}
}

// claudeCodeEnv returns the child environment that points Claude Code at the
// locally-served models. Foreground tiers map to "local" and the cheap tier to
// local-fast; ANTHROPIC_API_KEY is dropped so the dummy auth token + base URL
// take effect.
func claudeCodeEnv(host string, port int, serverArgs []string, actualCtx int) []string {
	clientHost := host
	if clientHost == "" || clientHost == "0.0.0.0" || clientHost == "::" {
		clientHost = "127.0.0.1"
	}
	var env []string
	for _, kv := range os.Environ() {
		key, _, _ := strings.Cut(kv, "=")
		switch key {
		case "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN",
			"ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL",
			"ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
			"CLAUDE_CODE_EFFORT_LEVEL",
			"API_TIMEOUT_MS", "API_FORCE_IDLE_TIMEOUT", "CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS",
			"CLAUDE_ENABLE_BYTE_WATCHDOG", "CLAUDE_ENABLE_STREAM_WATCHDOG",
			"CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
			"CLAUDE_CODE_ATTRIBUTION_HEADER":
			continue
		}
		env = append(env, kv)
	}
	return append(env,
		fmt.Sprintf("ANTHROPIC_BASE_URL=http://%s:%d", clientHost, port),
		"ANTHROPIC_AUTH_TOKEN=ggrun",
		"ANTHROPIC_MODEL=local",
		// The cheap tiers address the utility alias. ggrun's router resolves it
		// to the companion backend when one is running and falls back to the
		// main model otherwise, so nothing can reach the vendor API either way.
		"ANTHROPIC_SMALL_FAST_MODEL="+claudeauto.UtilityAlias,
		"ANTHROPIC_DEFAULT_HAIKU_MODEL="+claudeauto.UtilityAlias,
		"ANTHROPIC_DEFAULT_SONNET_MODEL=local",
		"ANTHROPIC_DEFAULT_OPUS_MODEL=local",
		// Claude Code prepends an attribution block carrying client version and
		// a fingerprint. Those bytes sit at the very front of the prompt, so
		// they change the first tokens on every request and no later turn can
		// match a prefix -- llama.cpp computes a common prefix of zero, and
		// every context checkpoint it saved is invalidated against pos_next = 0.
		//
		// Measured here: seven checkpoints created across one turn, five
		// invalidated at pos_next = 0, and 0% reuse of 60,127 prompt tokens
		// while the checkpoint machinery was working correctly the whole time.
		// The header is worth nothing against a local backend that has no
		// vendor telemetry to attribute to.
		"CLAUDE_CODE_ATTRIBUTION_HEADER=0",
		// xhigh is Claude Code's recommended balance for coding and agentic work.
		// The official environment variable also accepts auto/max and lets an
		// explicit user choice override this local-workflow default.
		"CLAUDE_CODE_EFFORT_LEVEL="+envOr("CLAUDE_CODE_EFFORT_LEVEL", "xhigh"),
		// JavaScript's maximum safe timer value is effectively no deadline for a
		// local session. It covers foreground requests and queued Workflow fan-out.
		fmt.Sprintf("API_TIMEOUT_MS=%d", claudeNoTimeoutMS),
		// Background agents and streaming each have independent watchdogs. A giant
		// local MoE can spend minutes in prompt processing without producing an event.
		fmt.Sprintf("CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS=%d", claudeNoTimeoutMS),
		"CLAUDE_ENABLE_BYTE_WATCHDOG=0",
		"CLAUDE_ENABLE_STREAM_WATCHDOG=0",
		// Compatibility with Claude Code versions that predate the named watchdogs.
		"API_FORCE_IDLE_TIMEOUT=0",
		// Claude Code's assumed window has changed across releases and model aliases.
		// Give it the backend's REAL per-slot capacity directly, then compact at 75%
		// so an in-flight reply and tool output cannot overflow the slot. The actual
		// context is the resolved strategy context (which the model's native
		// context_length may have capped below the requested --ctx-size) — sizing the
		// autocompact window off the raw --ctx-size lets a request overflow the real
		// slot before compaction fires (the Muse 250k-requested/131k-actual crash).
		// User values win.
		"CLAUDE_CODE_AUTO_COMPACT_WINDOW="+envOr("CLAUDE_CODE_AUTO_COMPACT_WINDOW", strconv.Itoa(claudeCodeAutocompactWindow(serverArgs, actualCtx))),
		"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE="+envOr("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", strconv.Itoa(claudeCodeAutocompactPct(serverArgs, actualCtx))),
	)
}

// envOr returns the current environment value for key, or def if unset/empty.
func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// claudeCodePermissionArgs defaults local Claude Code to Auto. ggrun supplies
// Auto's otherwise-missing classifier with a dedicated local reviewer and routes
// only the hidden safety-monitor calls to it; Workflow, MCP, WebFetch, and Bash can
// therefore run autonomously without bypassing Claude Code's safety rules.
//
// GGRUN_CLAUDE_PERMISSION_MODE can select another current Claude CLI mode. Set it
// to "inherit" to preserve the user's settings.json default. An explicit
// --permission-mode in extraArgs always wins.
func claudeCodePermissionArgs(extraArgs []string) []string {
	for _, arg := range extraArgs {
		if arg == "--permission-mode" || strings.HasPrefix(arg, "--permission-mode=") {
			return nil
		}
	}
	mode := strings.TrimSpace(os.Getenv("GGRUN_CLAUDE_PERMISSION_MODE"))
	if mode == "" {
		mode = "auto"
	}
	if strings.EqualFold(mode, "inherit") {
		return nil
	}
	// Keep invalid environment values from making Claude fail at startup. These
	// choices match the current CLI; "default" is the docs/settings spelling of
	// the CLI's "manual" mode.
	switch strings.ToLower(mode) {
	case "acceptedits":
		mode = "acceptEdits"
	case "auto":
		mode = "auto"
	case "bypasspermissions":
		mode = "bypassPermissions"
	case "manual", "default":
		mode = "manual"
	case "dontask":
		mode = "dontAsk"
	case "plan":
		mode = "plan"
	default:
		mode = "auto"
	}
	return []string{"--permission-mode", mode}
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

// claudeCodeAutocompactWindow returns the backend's actual per-sequence capacity.
// llama.cpp/ik_llama divide --ctx-size across --parallel slots. Passing this value
// explicitly avoids depending on Claude Code's changing assumed window for custom
// model aliases. The 256-token alignment mirrors the backend's slot allocation.
func claudeCodeAutocompactWindow(serverArgs []string, actualCtx int) int {
	ctx := argIntValue(serverArgs, "--ctx-size", "-c", "--ctx")
	if ctx <= 0 {
		return 200000
	}
	// The real per-slot capacity is what can actually overflow. When the resolved
	// strategy context (post model-native-cap) is available and smaller, it is the
	// true ceiling — a requested ctx that got capped must not size the window.
	if actualCtx > 0 && actualCtx < ctx {
		ctx = actualCtx
	}
	parallel := argIntValue(serverArgs, "--parallel", "-np")
	if parallel < 1 {
		parallel = 1
	}
	slot := (ctx / parallel) & ^255
	if slot < 2048 {
		return 2048
	}
	return slot
}

func claudeCodeAutocompactPct(serverArgs []string, actualCtx int) int {
	if argIntValue(serverArgs, "--ctx-size", "-c", "--ctx") <= 0 && actualCtx <= 0 {
		return 25 // unknown ctx: preserve the historical ~50k fallback
	}
	return 75
}

// claudeCodeSearchMCPArgs returns --mcp-config args that wire a no-key DuckDuckGo
// search MCP into Claude Code, replacing the Anthropic-only WebSearch tool that
// can't run against a local endpoint. Returns nil if the user already passed their
// own --mcp-config or no MCP runner (uvx) is installed. The exposed tool surfaces to
// agents and workflows as mcp__ddg-search__search and
// mcp__ddg-search__fetch_content.
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
	args := []string{"--mcp-config", cfg}
	if !hasArg(extraArgs, "--allowedTools") && !hasArg(extraArgs, "--allowed-tools") {
		args = append(args, "--allowedTools", "mcp__ddg-search__search,mcp__ddg-search__fetch_content")
	}
	return args
}

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

// claudeCodeCacheArgs enables chunk-level KV reuse for repeated system, tool,
// and workflow blocks that move after new conversation content is inserted.
// Ordinary prompt caching only reuses a common prefix; cache-reuse can shift a
// later matching chunk into its new position. The value 256 is the conservative
// llama.cpp coding preset. Users can disable it explicitly, and older backends
// remain compatible because support is checked before adding the flag.
func claudeCodeCacheArgs(args []string, claudeCode bool, backendHelp string, shiftableContext bool) []string {
	if !claudeCode || !shiftableContext || !strings.Contains(backendHelp, "--cache-reuse") {
		return args
	}
	hasFlag := func(flag string) bool {
		for _, arg := range args {
			if arg == flag || strings.HasPrefix(arg, flag+"=") {
				return true
			}
		}
		return false
	}
	if hasFlag("--cache-reuse") || hasFlag("--no-cache-prompt") {
		return args
	}
	return append(args, "--cache-reuse", "256")
}

func claudeCodeShiftableContext(model *placement.ModelProfile, strategy *placement.Strategy) bool {
	if strategy != nil && (strategy.HasSSM || strategy.MMProjPath != "") {
		return false
	}
	if model == nil {
		return true
	}
	// Laguna uses multi-position RoPE. llama.cpp's KV shift requires one
	// position per embedding and disables cache_reuse after model load.
	return !strings.EqualFold(strings.TrimSpace(model.ModelArch), "laguna")
}

// runClaudeCodeClient launches Claude Code in the foreground wired to the local
// server, inheriting the terminal. It returns claude's exit code, or -1 if the
// `claude` CLI isn't installed (so the caller can fall back to the recipe).
func runClaudeCodeClient(host string, port int, serverArgs, extraArgs []string, spec *claudeSessionSpec, actualCtx int) int {
	claudePath, err := exec.LookPath("claude")
	if err != nil {
		return -1
	}
	clientHost := host
	if clientHost == "" || clientHost == "0.0.0.0" || clientHost == "::" {
		clientHost = "127.0.0.1"
	}
	fmt.Printf("[claude-code] Claude Code → http://%s:%d\n", clientHost, port)
	args := claudeCodeWorkflowPromptArgs(extraArgs)
	if permissionArgs := claudeCodePermissionArgs(extraArgs); permissionArgs != nil {
		args = append(permissionArgs, args...)
		if len(permissionArgs) == 2 && permissionArgs[1] == "auto" {
			fmt.Println("[claude-code] Permission mode: Auto (dedicated local safety reviewer; fail-closed).")
		} else if len(permissionArgs) == 2 && permissionArgs[1] == "acceptEdits" {
			fmt.Println("[claude-code] Permission mode: acceptEdits (explicit override; shell actions still ask).")
		}
	}
	// Built-in WebSearch is an Anthropic server-side tool; on a local endpoint it
	// can't run, and the model loops on it while the auto-permission classifier
	// fails. Disable it and wire a no-key DuckDuckGo MCP in its place so agents and
	// workflows can still do web research. Skip either if the user passed their own.
	if !hasArg(extraArgs, "--disallowedTools") {
		args = append([]string{"--disallowedTools", "WebSearch"}, args...)
	}
	if mcp := claudeCodeSearchMCPArgs(extraArgs); mcp != nil {
		args = append(mcp, args...)
		fmt.Println("[claude-code] Online research enabled through DuckDuckGo MCP (search + fetch_content).")
	} else {
		fmt.Println("[claude-code] WebSearch disabled (Anthropic-only); install uvx or add a search MCP for web research.")
	}
	sessionArgs, err := claudeSessionArgs(spec, extraArgs, args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[claude-code] %v\n", err)
		return 2
	}
	args = sessionArgs
	if summary := describeClaudeResume(spec); summary != "" {
		fmt.Println(summary)
	}
	// The resume instruction goes last so it becomes Claude's opening turn.
	if prompt := claudeResumePrompt(spec); prompt != "" && spec.Resume {
		args = append(args, prompt)
	}
	releaseInterrupt := holdInterruptForClaude()
	defer releaseInterrupt()
	cmd := exec.Command(claudePath, args...)
	cmd.Env = claudeCodeEnv(host, port, serverArgs, actualCtx)
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
	cfg := loadConfigOrExit()
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

// downloadDirFromArgs pulls an explicit destination out of the download argv.
//
// MODEL_DIR is a single path, so every download landed on whichever filesystem
// it happens to sit on. On this machine that is the 456 GB root volume with
// 67 GB free, while the 1.9 TB volume holding the large quants has 935 GB --
// and the only way to put a model there was to download it elsewhere and
// symlink it in by hand. A destination belongs to one download, not to the
// whole configuration, so it is a flag rather than a config change.
// expandPath resolves a user-supplied directory to an absolute path, expanding
// environment variables and a leading ~. The tilde matters most for the TUI,
// where a destination is typed by hand and "~/2tb-disk/models" would otherwise
// create a directory literally named "~".
func expandPath(dir string) string {
	dir = os.ExpandEnv(strings.TrimSpace(dir))
	if dir == "~" || strings.HasPrefix(dir, "~/") {
		if home, err := os.UserHomeDir(); err == nil && home != "" {
			dir = filepath.Join(home, strings.TrimPrefix(strings.TrimPrefix(dir, "~"), "/"))
		}
	}
	if abs, err := filepath.Abs(dir); err == nil {
		return abs
	}
	return dir
}

// downloadOptionsFromArgs pulls an explicit destination and quant out of the
// download argv. Both accept "--flag value" and "--flag=value".
//
// --quant matters for scripted and unattended use: without it the downloader
// drops into its interactive numbered picker, so a download cannot be started
// from anything that has no terminal attached.
func downloadOptionsFromArgs(args []string, fallbackDir string) (repo, dir, quant string, err error) {
	dir = fallbackDir
	rest := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		key, val, hasVal := strings.Cut(args[i], "=")
		var target *string
		switch key {
		case "--dir", "-dir", "--model-dir":
			target = &dir
		case "--quant", "-quant":
			target = &quant
		default:
			rest = append(rest, args[i])
			continue
		}
		if !hasVal {
			if i+1 >= len(args) {
				return "", "", "", fmt.Errorf("%s needs a value", key)
			}
			i++
			val = args[i]
		}
		if strings.TrimSpace(val) == "" {
			return "", "", "", fmt.Errorf("%s needs a value", key)
		}
		*target = strings.TrimSpace(val)
	}
	if len(rest) < 1 {
		return "", "", "", fmt.Errorf("no repository given")
	}
	return rest[0], expandPath(dir), quant, nil
}

func cmdDownload(args []string) {
	cfg := loadConfigOrExit()
	repo, modelDir, quant, err := downloadOptionsFromArgs(args, cfg.ModelDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Usage: ggrun download <repo/name> [--dir <path>] [--quant <name>]\n  %v\n", err)
		os.Exit(2)
	}

	caps, derr := detect.Detect()
	if derr != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", derr)
		os.Exit(1)
	}

	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "Error: cannot use %s: %v\n", modelDir, err)
		os.Exit(1)
	}
	if modelDir != cfg.ModelDir {
		fmt.Fprintf(os.Stderr, "[download] destination: %s\n", modelDir)
	}

	d := download.New(modelDir, cfg.CacheDir, cfg.AppHome)
	if err := d.RunQuant(repo, quant, caps); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func tuneRoundsFromArgs(args []string, fallback int) (int, error) {
	if fallback <= 0 {
		fallback = 8
	}
	for i := 0; i < len(args); i++ {
		if key, val, ok := strings.Cut(args[i], "="); ok && (key == "--rounds" || key == "-rounds") {
			n, err := parsePositiveFlag(key, val)
			if err != nil {
				return 0, err
			}
			return n, nil
		}
		if args[i] == "--rounds" || args[i] == "-rounds" {
			if i+1 >= len(args) {
				return 0, fmt.Errorf("%s requires a value", args[i])
			}
			n, err := parsePositiveFlag(args[i], args[i+1])
			if err != nil {
				return 0, err
			}
			return n, nil
		}
	}
	return fallback, nil
}

func cmdRecommend(args []string) {
	limit := 5
	firstOnly := false
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-n", "--limit":
			if i+1 < len(args) {
				if n, err := strconv.Atoi(args[i+1]); err == nil && n > 0 {
					limit = n
				}
				i++
			}
		case "--first":
			firstOnly = true
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

	cfg := loadConfigOrExit()
	if cfg.RAMLimitPercent > 0 {
		fmt.Printf("RAM limit: %d%% whole-host utilisation\n", cfg.RAMLimitPercent)
		caps = detect.ApplyRAMLimitPercent(caps, cfg.RAMLimitPercent)
	}
	if headroomMB := parseBudgetMB(cfg.VRAMHeadroom); headroomMB > 0 {
		fmt.Printf("VRAM headroom: %d MB reserved (set via Settings or --vram-headroom)\n", headroomMB)
		caps = detect.ApplyVRAMHeadroom(caps, headroomMB)
	}
	if headroomMB := parseBudgetMB(cfg.RAMHeadroom); headroomMB > 0 {
		fmt.Printf("RAM headroom: %d MB reserved (set via Settings or --ram-headroom)\n", headroomMB)
		caps = detect.ApplyRAMHeadroom(caps, headroomMB)
	}

	cats := recommend.TopCategories(caps, limit)
	if len(cats.Balanced) == 0 {
		fmt.Println("No models in the catalog fit this machine.")
		return
	}
	if firstOnly {
		fmt.Println(cats.Balanced[0].Repo)
		return
	}
	printRecGroup := func(title string, rows []recommend.Recommendation) {
		if len(rows) == 0 {
			return
		}
		fmt.Printf("\n%s\n", title)
		fmt.Printf("  %-36s %-10s %-8s %6s %5s %8s\n", "Model", "Fit", "Quant", "Size", "Qual", "Est.speed")
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
	fmt.Println("\nSpeed is an estimate for ranking; run --benchmark on the downloaded model for a measured result.")
	fmt.Println("Fit uses installed capacity; every launch rechecks currently free RAM and VRAM.")
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

	cfg := loadConfigOrExit()
	rounds, err := tuneRoundsFromArgs(args, cfg.TuneRounds)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(2)
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
		fmt.Fprintf(os.Stderr, "Error: %s\n", backendUnavailableMessage(req))
		os.Exit(1)
	}
	applyCachedBackendCapabilities(req, cfg.CacheDir, model, be)
	if env := applyGPUVisibility(req, backendDialect(be)); env != "" {
		fmt.Printf("[tune] GPU restriction: %s\n", env)
	}

	tuneOpts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	tuneOpts.ReasoningOff = true // tuning measures throughput, so think-free like benchmarks
	strategy, err := placement.Compute(caps, model, tuneOpts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error computing placement: %s\n", placementErrorMessage(err))
		os.Exit(1)
	}
	strategy.BackendTag = backendDialect(be)
	// Tune the same slot/batch policy that a real Claude launch uses. Without
	// this, an agent-parallel tune can benchmark an uncapped hybrid prefill and
	// later override the fairness policy it was meant to improve.
	claudeCodeSlotAdjust(strategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	agentParallel := 0
	agentWorkload := ""
	minTuneImprovement := 0.0
	if req.ClaudeCode && model.HasSSM > 0 && strategy.Parallel > 1 {
		agentParallel = strategy.Parallel
		minTuneImprovement = calibrationMinImprovementPct
		kvTypeV := strategy.KVTypeV
		if kvTypeV == "" {
			kvTypeV = strategy.KVType
		}
		agentWorkload = fmt.Sprintf("%s-p%d-c%d-k%s-v%s-%s-%s",
			tune.AgentParallelWorkloadPrefix, strategy.Parallel, strategy.ContextSize,
			strategy.KVType, kvTypeV, strategy.KVPlacement, strategy.Type)
		fmt.Printf("[tune] parallel agent workload: %d slots; ranking safe flags by repeated cache-backed turn time (batch/ubatch remain placement-owned; use --calibrate on to screen pairs)\n", strategy.Parallel)
	}

	// A completed tune for this model/hardware/backend is reused unless the
	// user explicitly asks for a fresh run with --retune.
	if !hasArg(args, "--retune") {
		cachePath := tune.TuneCachePathForWorkload(cfg.CacheDir, req.ModelPath, gpuNamesFromCaps(caps), strategy.MMProjPath != "", be.Tag, agentWorkload)
		if cachePath != "" && tune.TuneFileComplete(cachePath) {
			fmt.Printf("[tune] Completed tune cache found: %s\n", cachePath)
			if agentWorkload != "" {
				fmt.Println("[tune] Select it with --tune-cache for this exact workload, or use --retune to measure again.")
			} else {
				fmt.Println("[tune] It is applied automatically on launch. Re-run with --retune to tune again.")
			}
			return
		}
	}
	if err := confirmRequiredMMap(req, strategy, os.Stdin, os.Stderr, stdinIsTerminal()); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if err := validateHostMemoryContainment(req, caps, strategy); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	serverArgs := append([]string{be.Path}, strategy.Args(req.ModelPath, req.Port)...)
	serverArgs = append(serverArgs, req.ExtraArgs...)
	serverArgs = applyRequestDisabledBackendFlags(serverArgs, req)
	if memMax := backendMemoryMaxMB(req, caps); memMax > 0 {
		fmt.Printf("[tune] backend memory scope: MemoryMax=%d MiB\n", memMax)
	}
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
		BaseURL:           fmt.Sprintf("http://localhost:%d", req.Port),
		Model:             filepath.Base(req.ModelPath),
		Rounds:            rounds,
		Cache:             cache,
		Caps:              caps,
		Backend:           be.Tag,
		Vision:            strategy.MMProjPath != "",
		MinImprovementPct: minTuneImprovement,
		BenchmarkTimeout:  benchTimeout,
		BackendHelp:       be.Help,
		AgentParallel:     agentParallel,
		Workload:          agentWorkload,
		OnProgress: func(msg string) {
			fmt.Println("[tune]", msg)
		},
		StartServer: func(flags []string) (func(), error) {
			p, err := server.StartWithTimeoutToOptions(flags, req.Port, timeout, os.Stdout, os.Stderr, backendStartOptions(req, caps, be, nil, flags))
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
	tunePath := tune.TuneCachePathForWorkload(cfg.CacheDir, req.ModelPath, gpuNamesFromCaps(caps), strategy.MMProjPath != "", be.Tag, agentWorkload)
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
	port, model, err := parseBenchmarkArgs(args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(2)
	}
	if err := requireBenchmarkServer(port, time.Second); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if err := runOneShotBenchmark(port, model); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// parseBenchmarkArgs accepts a local GGUF as the command's sole positional
// argument even when flags follow it. The standard flag package stops at the
// first positional, which previously discarded both that model and every
// later flag and could benchmark an unrelated server on the default port.
func parseBenchmarkArgs(args []string) (int, string, error) {
	fs := flag.NewFlagSet("benchmark", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	port := fs.Int("port", 8081, "Server port")
	model := fs.String("model", "default", "Model name")

	flagArgs := make([]string, 0, len(args))
	positionals := make([]string, 0, 1)
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--port" || arg == "-port" || arg == "--model" || arg == "-model" {
			flagArgs = append(flagArgs, arg)
			if i+1 >= len(args) {
				return 0, "", fmt.Errorf("%s requires a value", arg)
			}
			i++
			flagArgs = append(flagArgs, args[i])
			continue
		}
		if strings.HasPrefix(arg, "-") {
			flagArgs = append(flagArgs, arg)
			continue
		}
		positionals = append(positionals, arg)
	}
	if err := fs.Parse(flagArgs); err != nil {
		return 0, "", err
	}
	positionals = append(positionals, fs.Args()...)
	if len(positionals) > 1 {
		return 0, "", fmt.Errorf("benchmark accepts at most one model file; usage: ggrun benchmark [model.gguf] [--port N]")
	}
	if _, err := config.ParsePort(strconv.Itoa(*port)); err != nil {
		return 0, "", fmt.Errorf("--port %v", err)
	}
	if len(positionals) == 1 {
		cfg, err := config.Load()
		if err != nil {
			return 0, "", fmt.Errorf("load config: %w", err)
		}
		resolved := resolveModelPath(positionals[0], cfg.ModelDir)
		if _, err := parseModel(resolved); err != nil {
			return 0, "", fmt.Errorf("model %q: %w", positionals[0], err)
		}
		*model = filepath.Base(resolved)
	}
	return *port, *model, nil
}

func requireBenchmarkServer(port int, timeout time.Duration) error {
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), timeout)
	if err != nil {
		return fmt.Errorf("no llama-server on port %d; start one or pass --port", port)
	}
	_ = conn.Close()
	return nil
}

func runOneShotBenchmark(port int, model string) error {
	runner := &benchmark.Runner{
		BaseURL: fmt.Sprintf("http://localhost:%d", port),
		Model:   model,
	}
	res, err := runner.Run()
	if err != nil {
		return err
	}
	data, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println(string(data))
	return nil
}

func runOneShotWorkerBenchmark(port int, model string, peakVRAMMB int) error {
	runner := &benchmark.Runner{
		BaseURL: fmt.Sprintf("http://localhost:%d", port),
		Model:   model,
	}
	throughput, err := runner.Run()
	if err != nil {
		return fmt.Errorf("throughput benchmark: %w", err)
	}
	throughput.PeakVRAMMB = peakVRAMMB
	worker, err := runner.RunWorkerSuite()
	if err != nil {
		return fmt.Errorf("worker benchmark: %w", err)
	}
	report := struct {
		Throughput *benchmark.Result            `json:"throughput"`
		Worker     *benchmark.WorkerSuiteResult `json:"worker"`
	}{Throughput: throughput, Worker: worker}
	data, _ := json.MarshalIndent(report, "", "  ")
	fmt.Println(string(data))
	return nil
}

func measuredLaunchVRAMMB(caps *detect.Capabilities, visibleToPhysical map[int]int, baseline map[int]int) int {
	if caps == nil {
		return 0
	}
	total := 0
	for _, gpu := range caps.GPUs {
		physical := physicalGPUIndex(gpu.Index, visibleToPhysical)
		used := placement.QueryVRAMUsed(physical) - baseline[gpu.Index]
		if used > 0 {
			total += used
		}
	}
	return total
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
	cfg, err := config.Load()
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	// Find the backend FIRST so its tag feeds placement — otherwise the
	// split-mode/flag selection can't tell ik_llama from mainline and emits
	// flags the backend rejects (e.g. `--split-mode row`, unsupported by ik).
	backendReq := &launchRequest{
		ServerBin:       cfg.LlamaServer,
		AppHome:         cfg.AppHome,
		Backend:         cfg.Backend,
		BackendExplicit: configuredBackendExplicit(cfg.Backend),
		// Populate the runtime knobs that enter the calibration scope key the
		// same way the interactive launch request does, so a decision cached by
		// an interactive launch is found by the daemon (and vice versa). Leaving
		// them zero produced a scope key that could never match the interactive
		// save, making the daemon's calibration consume a silent no-op.
		RAMLimitPercent: cfg.RAMLimitPercent,
		RamBudgetMB:     parseBudgetMB(cfg.RamBudget),
		RAMHeadroomMB:   parseBudgetMB(cfg.RAMHeadroom),
		VRAMHeadroomMB:  parseBudgetMB(cfg.VRAMHeadroom),
		KVPlacement:     cfg.KVPlacement,
		KVQuality:       cfg.KVQuality,
		HotExperts:      cfg.HotExperts,
		Host:            cfg.Host,
	}
	if cfg.SWAFull {
		backendReq.ExtraArgs = append(backendReq.ExtraArgs, "--swa-full")
	}
	be := selectBackendForModel(caps, backendReq, model)
	if be == nil {
		return nil, errors.New(backendUnavailableMessage(backendReq))
	}
	applyBackendFeatureCompatibility(backendReq, model, be)
	applyCachedBackendCapabilities(backendReq, cfg.CacheDir, model, be)
	opts := placement.Options{
		ContextSize:             resolveCtxFlag(cfg.CtxValue(), model.CTXTrain),
		KVPlacement:             cfg.KVPlacement,
		KVQuality:               cfg.KVQuality,
		SWAFull:                 hasArg(backendReq.ExtraArgs, "--swa-full"),
		RamBudgetMB:             parseBudgetMB(cfg.RamBudget),
		RAMLimitPercent:         cfg.RAMLimitPercent,
		VRAMHeadroomMB:          parseBudgetMB(cfg.VRAMHeadroom),
		RAMHeadroomMB:           parseBudgetMB(cfg.RAMHeadroom),
		CacheDir:                cfg.CacheDir,
		Host:                    cfg.Host,
		BackendTag:              backendDialect(be),
		BackendCacheTag:         evidenceBackendCacheTag(be),
		BackendIdentity:         be.Identity,
		CPUExpertMMapCapability: backendCPUExpertMMapCapability(be),
		CPUExpertMMapEvidence:   be.CPUExpertMMapEvidence,
		BackendHelp:             be.Help,
		VisionAuto:              cfg.Vision,
		SpecMode:                cfg.Spec,
		HotExperts:              cfg.HotExperts,
	}
	if slots, parseErr := strconv.Atoi(strings.TrimSpace(cfg.HotExperts)); parseErr == nil && slots > 0 {
		opts.HotExpertCacheSlots = slots
	}
	strategy, err := placement.Compute(caps, model, opts)
	if err != nil {
		return nil, fmt.Errorf("compute placement: %w", err)
	}
	// Consume a cached calibration winner just like the interactive launch path,
	// so a daemon/serve model load applies the measured fastest placement instead
	// of the raw estimate. (The daemon is deliberately not given the full
	// OOM-recovery/relaunch lifecycle; this closes the calibration gap without
	// changing the daemon's process model.)
	strategy = applyCalibrationDecision(backendReq, cfg, model, be, caps, strategy)
	strategy.BackendTag = backendDialect(be)
	serverArgs := append([]string{be.Path}, strategy.Args(modelPath, port)...)
	if hasArg(backendReq.ExtraArgs, "--swa-full") {
		serverArgs = append(serverArgs, "--swa-full")
	}
	return serverArgs, nil
}

func daemonMemoryScope(req *launchRequest, caps *detect.Capabilities, be *backendInfo, serverArgs []string, hardOverrideMB int) (highMB, maxMB int, err error) {
	opts := backendStartOptions(req, caps, be, nil, serverArgs)
	highMB, maxMB = opts.MemoryHighMB, opts.MemoryMaxMB
	if hardOverrideMB > 0 {
		maxMB = hardOverrideMB
		highMB = hardOverrideMB
		if argsMMapBackedExperts(be, serverArgs) && opts.MemoryHighMB > 0 && opts.MemoryHighMB < maxMB {
			highMB = opts.MemoryHighMB
		}
	}
	if maxMB <= 0 {
		return 0, 0, fmt.Errorf("configured RAM policy leaves no positive daemon memory ceiling")
	}
	if highMB <= 0 || highMB > maxMB {
		highMB = maxMB
	}
	return highMB, maxMB, nil
}

func cmdDaemon(args []string) {
	fs := flag.NewFlagSet("daemon", flag.ExitOnError)
	modelPath := fs.String("model", "", "Model path")
	port := fs.Int("port", 8081, "Server port")
	controlPort := fs.Int("control-port", 9090, "Control API port")
	startupTimeoutSecs := fs.Int("startup-timeout-secs", 300, "Max seconds to wait for llama-server to become healthy after start/reload")
	memoryMaxMB := fs.Int("memory-max-mb", 0, "Required hard host-memory ceiling for the managed backend")
	ramLimitPercent := fs.Int("ram-limit-percent", 0, "Override the configured whole-host RAM utilisation limit")
	fs.Parse(args)
	if _, err := config.ParsePort(strconv.Itoa(*port)); err != nil {
		fmt.Fprintf(os.Stderr, "Error: --port %v\n", err)
		os.Exit(2)
	}
	if _, err := config.ParsePort(strconv.Itoa(*controlPort)); err != nil {
		fmt.Fprintf(os.Stderr, "Error: --control-port %v\n", err)
		os.Exit(2)
	}
	if *startupTimeoutSecs < 1 {
		fmt.Fprintln(os.Stderr, "Error: --startup-timeout-secs must be a positive integer")
		os.Exit(2)
	}
	if *memoryMaxMB < 0 {
		fmt.Fprintln(os.Stderr, "Error: --memory-max-mb must be non-negative")
		os.Exit(2)
	}
	if *ramLimitPercent != 0 {
		if _, err := config.ParseRAMLimitPercent(strconv.Itoa(*ramLimitPercent)); err != nil {
			fmt.Fprintf(os.Stderr, "Error: --ram-limit-percent %v\n", err)
			os.Exit(2)
		}
	}

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: ggrun daemon --model <model.gguf>")
		os.Exit(2)
	}
	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}
	cfgDaemon := loadConfigOrExit()
	percent := cfgDaemon.RAMLimitPercent
	if *ramLimitPercent != 0 {
		percent = *ramLimitPercent
	}
	memoryReq := &launchRequest{
		RamBudgetMB:     parseBudgetMB(cfgDaemon.RamBudget),
		RAMLimitPercent: percent,
		RAMHeadroomMB:   parseBudgetMB(cfgDaemon.RAMHeadroom),
	}
	hardOverrideMB := *memoryMaxMB

	serverArgs, err := computeServerArgs(*modelPath, *port)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	computeMemory := func(args []string) (int, int, error) {
		if len(args) == 0 {
			return 0, 0, fmt.Errorf("empty daemon backend argv")
		}
		return daemonMemoryScope(memoryReq, caps, detectUsableBackend(args[0]), args, hardOverrideMB)
	}
	highMB, effectiveMaxMB, err := computeMemory(serverArgs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	d := daemon.New(daemon.Config{
		ModelPath:          *modelPath,
		ServerArgs:         serverArgs,
		Port:               *port,
		ControlPort:        *controlPort,
		MemoryMaxMB:        effectiveMaxMB,
		MemoryHighMB:       highMB,
		StartupTimeoutSecs: *startupTimeoutSecs,
		// Let /reload recompute placement when handed a bare model path,
		// so model swaps get the same auto-placement as the initial launch.
		ComputeArgs:   computeServerArgs,
		ComputeMemory: computeMemory,
	})
	errCh := make(chan error, 1)
	go func() { errCh <- d.Start() }()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, shutdownSignals()...)
	defer signal.Stop(sigCh)

	select {
	case err := <-errCh:
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	case sig := <-sigCh:
		fmt.Printf("\n[daemon] received %s; stopping managed server\n", sig)
		if err := d.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "Error during daemon shutdown: %v\n", err)
			os.Exit(1)
		}
	}
}

func cmdConfig(args []string) {
	sub := "show"
	if len(args) > 0 {
		sub = args[0]
	}
	switch sub {
	case "help", "--help", "-h":
		fmt.Fprintln(os.Stderr, "Usage: ggrun config [show|edit|path|reset]")
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

func cmdUpdate(args []string) {
	for _, arg := range args {
		switch arg {
		case "-h", "--help", "help":
			printUpdateScope()
			return
		default:
			fmt.Fprintf(os.Stderr, "unknown update option %q\n", arg)
			fmt.Fprintln(os.Stderr, "Usage: ggrun update [--help]")
			os.Exit(2)
		}
	}
	// Self-update ggrun
	if err := update.SelfUpdate(); err != nil {
		fmt.Fprintf(os.Stderr, "Self-update: %v\n", err)
	}
	if runtime.GOOS == "windows" {
		fmt.Println("Backend updates are handled by the native Windows release bundle.")
	} else {
		// Update every active generic backend plus every registered fork.
		if err := updateAllBackends(); err != nil {
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

func printUpdateScope() {
	appHome := backends.AppHome()
	fmt.Println("Usage: ggrun update")
	fmt.Printf("\nCanonical app home: %s\n", appHome)
	fmt.Println("\nActive generic backend builds:")
	targets := update.BackendBuildTargetsAt(appHome)
	if len(targets) == 0 {
		fmt.Println("  (none found)")
	}
	for _, target := range targets {
		fmt.Printf("  %-28s %s\n", target.Label, target.BuildDir)
	}
	fmt.Println("\nRegistered fork backends:")
	forks := backends.Load()
	sortRegisteredBackendsForUpdate(forks)
	if len(forks) == 0 {
		fmt.Println("  (none registered)")
	}
	for _, fork := range forks {
		fmt.Printf("  %-28s %s\n", fork.Tag, fork.Path)
	}
	fmt.Println("\nEach build is rebuilt independently. A failed build keeps the previous working binary, and remaining backends continue.")
}

func updateAllBackends() error {
	var updateErrs []error
	if err := update.UpdateBackendsAtAppHome(backends.AppHome()); err != nil {
		updateErrs = append(updateErrs, err)
	}

	forks := backends.Load()
	sortRegisteredBackendsForUpdate(forks)
	if len(forks) == 0 {
		fmt.Println("\nNo registered fork backends found.")
		return errors.Join(updateErrs...)
	}
	fmt.Println("\nRegistered fork update summary:")
	updateErrs = append(updateErrs, updateRegisteredBackendList(forks, updateRegisteredBackend)...)
	return errors.Join(updateErrs...)
}

// sortRegisteredBackendsForUpdate keeps ordinary/base forks ahead of composed
// overlays. desiredBackendRecipe intentionally follows the base's installed
// pin, so update-all must advance the base before rebuilding its derivative.
func sortRegisteredBackendsForUpdate(forks []backends.Backend) {
	sort.SliceStable(forks, func(i, j int) bool {
		iComposed := len(forks[i].Features) > 0
		jComposed := len(forks[j].Features) > 0
		if iComposed != jComposed {
			return !iComposed
		}
		return strings.ToLower(forks[i].Tag) < strings.ToLower(forks[j].Tag)
	})
}

func updateRegisteredBackendList(forks []backends.Backend, updater func(string) error) []error {
	var updateErrs []error
	for _, fork := range forks {
		if err := updater(fork.Tag); err != nil {
			fmt.Printf("  %-28s failed (kept previous build): %v\n", fork.Tag, err)
			updateErrs = append(updateErrs, fmt.Errorf("%s: %w", fork.Tag, err))
			continue
		}
		fmt.Printf("  %-28s checked\n", fork.Tag)
	}
	return updateErrs
}

// isIKOnlyArch reports whether a model architecture can only be loaded by
// ik_llama.cpp; mainline llama.cpp rejects these with "unknown model architecture".
func isIKOnlyArch(arch string) bool {
	return backends.RequiredBackendForArch(arch) == "ik_llama"
}

// availableIKBinary returns the path of a detected ik_llama.cpp server binary, if any.
func availableIKBinary(caps *detect.Capabilities, configuredAppHome ...string) string {
	seen := map[string]bool{}
	cands := append([]string(nil), backendSearchPaths(configuredAppHome...)...)
	if caps != nil {
		for _, b := range caps.Backends {
			cands = append(cands, b.Path)
		}
	}
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
func preflightBackendArch(model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, configuredAppHome ...string) {
	if model == nil || be == nil {
		return
	}
	if !be.IsIK && isIKOnlyArch(model.ModelArch) {
		preflightIKOnlyArch(model, be, caps, configuredAppHome...)
		return
	}
	suggestForkForArch(model.ModelArch, be)
}

// suggestForkForArch warns when the resolved backend does not know the model's
// architecture and a reviewed recipe does. Without it the launch dies inside the
// loader -- a Laguna model on mainline reports only a tensor count -- and
// nothing connects that to the fork that would serve it.
//
// It warns rather than exits. The probe reads a backend's own library graph,
// which it can only do for ELF objects; a Windows build that keeps its
// architectures in a DLL would probe as unsupported while working perfectly, and
// refusing that launch would be a worse failure than the one being replaced. A
// genuinely unsupported architecture still fails at load, now with the fix named
// beforehand.
func suggestForkForArch(arch string, be *backendInfo) {
	if arch == "" || be == nil || be.Path == "" {
		return
	}
	supported, probed := backends.BackendSupportsArch(be.Path, arch)
	if !probed || supported {
		return
	}
	recipes := backends.RecipesForArch(arch)
	if len(recipes) == 0 {
		return
	}
	fmt.Fprintf(os.Stderr,
		"[launch] warning: backend %s does not appear to support model architecture %q.\n", be.Path, arch)
	for _, r := range recipes {
		fmt.Fprintf(os.Stderr, "[launch]   reviewed fork available: ggrun backend install %s   (%s)\n", r.Name, r.Description)
	}
	fmt.Fprintln(os.Stderr, "[launch] continuing anyway; if the model fails to load, install the fork above.")
}

func preflightIKOnlyArch(model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, configuredAppHome ...string) {
	fmt.Fprintf(os.Stderr,
		"Error: model architecture %q needs the ik_llama.cpp backend, but the selected backend is mainline llama.cpp.\n"+
			"  backend binary: %s\n", model.ModelArch, be.Path)
	if ik := availableIKBinary(caps, configuredAppHome...); ik != "" {
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
		fmt.Fprintln(os.Stderr, "[warning] DeepSeek V4 Flash is tagged as deepseek2. Re-convert it with current mainline llama.cpp so general.architecture=deepseek4; this GGUF may be rejected.")
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
		Path:                      path,
		Name:                      info.Name,
		Basename:                  info.Basename,
		QuantizedBy:               info.QuantizedBy,
		SizeBytes:                 totalBytes,
		TotalSizeMB:               totalSizeMB,
		NumLayers:                 info.BlockCount,
		NumParams:                 info.EstimateParams(),
		IsMoE:                     info.IsMoE,
		NumExperts:                numExperts,
		ContextSize:               info.ContextLength,
		HiddenSize:                info.EmbeddingLength,
		HeadCount:                 headCount,
		HeadCountKV:               info.HeadCountKV,
		KeyLength:                 info.KeyLength,
		ValueLength:               info.ValueLength,
		VocabSize:                 info.VocabSize,
		TokenizerModel:            info.TokenizerModel,
		TokenizerPre:              info.TokenizerPre,
		TokenizerHash:             info.TokenizerHash,
		QuantType:                 "", // not parsed from gguf.py output
		ExpertBytes:               info.ExpertBytes,
		NonExpertBytes:            info.NonExpertBytes,
		TokenEmbdBytes:            info.TokenEmbdBytes,
		OutputBytes:               info.OutputBytes,
		ShexpBytes:                info.ShexpBytes,
		ExpertAuxBytes:            info.ExpertAuxBytes,
		ExpertLayerBytes:          append([]int64(nil), info.ExpertLayerBytes...),
		RoutedExpertLayerBytes:    append([]int64(nil), info.RoutedExpertLayerBytes...),
		ShexpLayerBytes:           append([]int64(nil), info.ShexpLayerBytes...),
		ExpertAuxLayerBytes:       append([]int64(nil), info.ExpertAuxLayerBytes...),
		NonExpertLayerBytes:       append([]int64(nil), info.NonExpertLayerBytes...),
		Fused:                     info.Fused,
		EmbeddingLength:           info.EmbeddingLength,
		FeedForwardLength:         info.FeedForwardLength,
		ExpertUsedCount:           info.ExpertUsed,
		ExpertFF:                  info.ExpFF,
		ExpertSharedFF:            info.ExpSharedFF,
		ExpertSharedCount:         info.ExpertSharedCount,
		ExpertSharedCountInferred: info.ExpertSharedCountInferred != 0,
		LeadingDense:              info.LeadingDense,
		LeadingDenseInferred:      info.LeadingDenseInferred != 0,
		RopeDim:                   info.NRot,
		HasSSM:                    info.SSM,
		FullAttnInterval:          info.FullAttnInterval,
		SlidingWindow:             info.SlidingWindow,
		HasShexp:                  info.HasShexp,
		KVLoraRank:                info.KVLoraRank,
		QLoraRank:                 info.QLoraRank,
		KeyLengthMLA:              info.KeyLengthMLA,
		ValueLengthMLA:            info.ValueLengthMLA,
		CTXTrain:                  info.ContextLength,
		ModelArch:                 info.Architecture,
		NextNPredictLayers:        info.NextNPredictLayers,
	}
}

// parseModel calls parse_gguf.py to extract real model metadata.
// For multi-part models, it sums all shard files for total size.
func parseModel(path string) (*placement.ModelProfile, error) {
	if _, _, err := modelstore.ResolveGGUFShardFiles(path); err != nil {
		return nil, fmt.Errorf("model file %q: %w", path, err)
	}
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
			profile.ExpertAuxBytes = int64(float64(profile.ExpertAuxBytes) * scale)
			scaleLayerBytes(profile.ExpertLayerBytes, scale)
			scaleLayerBytes(profile.RoutedExpertLayerBytes, scale)
			scaleLayerBytes(profile.ShexpLayerBytes, scale)
			scaleLayerBytes(profile.ExpertAuxLayerBytes, scale)
			scaleLayerBytes(profile.NonExpertLayerBytes, scale)
		}
	}
	// SizeBytes is authoritative after multi-shard discovery/rescaling. Keep the
	// MiB summary in sync: auto KV placement and strategy selection consume
	// TotalSizeMB, and a stale parser-table value can make an oversized MoE look
	// as though it fits wholly in VRAM.
	if profile.SizeBytes > 0 {
		profile.TotalSizeMB = int((profile.SizeBytes + 1048576 - 1) / 1048576)
	}

	return profile, nil
}

func scaleLayerBytes(values []int64, scale float64) {
	for i := range values {
		values[i] = int64(float64(values[i]) * scale)
	}
}

// totalModelSize returns the total bytes of a model, including all shards.
func totalModelSize(path string) int64 {
	files, _, err := modelstore.ResolveGGUFShardFiles(path)
	if err != nil {
		return 0
	}
	var total int64
	for _, file := range files {
		// os.Stat follows file symlinks. Summing lstat sizes once shrank a
		// 146GB sharded model to a few hundred bytes and invalidated placement.
		if info, statErr := os.Stat(file); statErr == nil {
			total += info.Size()
		}
	}
	return total
}

type backendInfo struct {
	Path                    string
	IsIK                    bool
	SupportsReasoning       bool
	Tag                     string
	Dialect                 string // placement/flag family: llama, ik_llama, vulkan, metal
	Help                    string
	Identity                string // version/build hash; invalidates speculative performance profiles
	CPUExpertMMapCapability placement.CPUExpertMMapCapability
	CPUExpertMMapEvidence   string
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

func findBackend(caps *detect.Capabilities, configuredAppHome ...string) *backendInfo {
	// Try detected backends first
	if caps != nil {
		for _, b := range caps.Backends {
			if b.Name == "llama-server" || b.Name == "ik_llama" || b.Name == "ik_llama-server" {
				if info := detectUsableBackend(b.Path); info != nil {
					return info
				}
			}
		}
	}
	for _, p := range backendSearchPaths(configuredAppHome...) {
		if p != "" {
			if _, err := os.Stat(p); err == nil {
				if info := detectUsableBackend(p); info != nil {
					return info
				}
			}
		}
	}
	return nil
}

func backendSearchPaths(configuredAppHome ...string) []string {
	home := os.Getenv("HOME")
	if home == "" {
		home, _ = os.UserHomeDir()
	}
	appHome := ""
	explicitAppHome := false
	for _, candidate := range configuredAppHome {
		if candidate = strings.TrimSpace(candidate); candidate != "" {
			appHome = candidate
			explicitAppHome = true
			break
		}
	}
	if appHome == "" {
		// backends.AppHome already handles LLM_APP_HOME, validated executable
		// ancestry, the recorded app-home pointer, and bounded discovery. Using
		// that shared resolver prevents ~/.local/bin/ggrun from incorrectly
		// treating ~/.local as the state tree while the configured install lives
		// elsewhere.
		appHome = backends.AppHome()
	}
	paths := []string{
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
	}
	// A configured APP_HOME is an explicit installation boundary. Global
	// detection still runs after these paths in selectBackend/findBackend, but
	// ad-hoc source trees under $HOME must not jump ahead of that detection or
	// override the selected production tree.
	if explicitAppHome {
		return paths
	}
	return append(paths,
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
	)
}

// backendLoaderFailed reports a binary that cannot even load (missing NCCL,
// wrong ELF, etc.). Those must not win auto-selection over a working Vulkan/CPU
// backend sitting next to them.
func backendLoaderFailed(help string) bool {
	return strings.Contains(help, "cannot open shared object file") ||
		strings.Contains(help, "error while loading shared libraries") ||
		strings.Contains(help, "Exec format error") ||
		strings.Contains(help, "cannot execute binary file")
}

func detectUsableBackend(path string) *backendInfo {
	info := detectBackend(path)
	if info == nil || backendLoaderFailed(info.Help) {
		return nil
	}
	return info
}

// detectBackend runs --help to determine if this is ik_llama.cpp fork.
// llama-server --help returns exit code 1, so we check the output regardless of error.
func detectBackend(path string) *backendInfo {
	info := &backendInfo{Path: path, Tag: "llama", Dialect: "llama"}
	cmd := exec.Command(path, "--help")
	if hubDir, ok, _ := libhub.Setup(path); ok {
		defer libhub.Cleanup(hubDir)
		cmd.Env = libhub.ApplyHubToChildEnv(os.Environ(), hubDir)
	}
	out, _ := cmd.CombinedOutput()
	help := string(out)
	info.Help = help
	info.Identity = backendBuildIdentity(path)
	lowerBase := strings.ToLower(filepath.Base(path))
	lowerDir := strings.ToLower(filepath.Dir(path))
	if strings.Contains(help, "ikawrakow") || strings.Contains(help, "split-mode-graph") {
		info.IsIK = true
		info.Tag = "ik_llama"
		info.Dialect = "ik_llama"
	} else if strings.Contains(lowerBase, "vulkan") || strings.Contains(lowerDir, "build-vulkan") {
		info.Tag = "vulkan"
		info.Dialect = "vulkan"
	} else if runtime.GOOS == "darwin" {
		// macOS llama.cpp builds default to Metal; placement must not emit
		// CUDA/Vulkan device-routing flags for them.
		info.Tag = "metal"
		info.Dialect = "metal"
	}
	if strings.Contains(help, "--reasoning") {
		info.SupportsReasoning = true
	}
	info.CPUExpertMMapCapability, info.CPUExpertMMapEvidence = probedCPUExpertMMapCapability(info)
	return info
}

// probedCPUExpertMMapCapability classifies loader memory behavior from the
// backend binary probe, not from the user-facing selection tag. ik-derived
// loaders are known to copy CPU experts into anonymous CUDA-host buffers.
// A successfully probed non-ik llama-server uses the reviewed mapped-CPU loader
// family. Empty/broken help is deliberately unknown and therefore resident-only
// until live evidence or an updated reviewed classifier says otherwise.
func probedCPUExpertMMapCapability(info *backendInfo) (placement.CPUExpertMMapCapability, string) {
	if info == nil {
		return placement.CPUExpertMMapUnknown, "backend probe unavailable"
	}
	identity := strings.TrimSpace(info.Identity)
	if identity == "" {
		identity = filepath.Base(info.Path)
	}
	if info.IsIK || placement.BackendUsesAnonymousCPUExperts(backendDialect(info)) {
		return placement.CPUExpertMMapAnonymous, "help-probed anonymous CPU-expert loader: " + identity
	}
	if strings.TrimSpace(info.Help) == "" || backendLoaderFailed(info.Help) {
		return placement.CPUExpertMMapUnknown, "loader memory behavior not proven: " + identity
	}
	return placement.CPUExpertMMapFileBacked, "help-probed mapped CPU-expert loader: " + identity
}

func backendBuildIdentity(path string) string {
	cmd := exec.Command(path, "--version")
	if hubDir, ok, _ := libhub.Setup(path); ok {
		defer libhub.Cleanup(hubDir)
		cmd.Env = libhub.ApplyHubToChildEnv(os.Environ(), hubDir)
	}
	out, _ := cmd.CombinedOutput()
	material := strings.TrimSpace(string(out))
	if fi, err := os.Stat(path); err == nil {
		material += fmt.Sprintf("\n%s\n%d\n%d", filepath.Base(path), fi.Size(), fi.ModTime().UnixNano())
	}
	if material == "" {
		material = path
	}
	sum := sha256.Sum256([]byte(material))
	return fmt.Sprintf("%s-%x", filepath.Base(path), sum[:12])
}
