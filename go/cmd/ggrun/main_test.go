package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
	"github.com/raketenkater/ggrun/pkg/tui"
)

func writeFakeBackend(t *testing.T, name, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+body), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestParseLaunchArgsMMapPolicy(t *testing.T) {
	req, err := parseLaunchArgs([]string{"model.gguf", "--mmap"})
	if err != nil {
		t.Fatal(err)
	}
	if !req.ForceMMap || req.NoMMap {
		t.Fatalf("--mmap policy not preserved: %#v", req)
	}
	req, err = parseLaunchArgs([]string{"model.gguf", "--mmap", "--no-mmap"})
	if err != nil {
		t.Fatal(err)
	}
	if !req.NoMMap || req.ForceMMap {
		t.Fatalf("last mmap policy must win: %#v", req)
	}
}

func TestConfirmRequiredMMap(t *testing.T) {
	strategy := &placement.Strategy{MMap: true, MMapRequired: true}
	req := &launchRequest{}
	var output bytes.Buffer
	if err := confirmRequiredMMap(req, strategy, strings.NewReader("yes\n"), &output, true); err != nil {
		t.Fatal(err)
	}
	if !req.ForceMMap || !strings.Contains(output.String(), "Use mmap?") {
		t.Fatalf("confirmation did not approve mmap: req=%#v output=%q", req, output.String())
	}
	if err := confirmRequiredMMap(&launchRequest{}, strategy, strings.NewReader(""), &output, false); err == nil || !strings.Contains(err.Error(), "--mmap") {
		t.Fatalf("non-interactive launch must require explicit --mmap, got %v", err)
	}
}

func TestConfirmLiveMemoryProbeRequiresExplicitNonInteractiveConsent(t *testing.T) {
	var output bytes.Buffer
	req := &launchRequest{}
	if err := confirmLiveMemoryProbe(req, "full load required", strings.NewReader("yes\n"), &output, true); err != nil {
		t.Fatal(err)
	}
	if !req.AllowLiveMemoryProbe || !strings.Contains(output.String(), "contained live memory probe") {
		t.Fatalf("interactive consent was not retained: req=%#v output=%q", req, output.String())
	}
	if err := confirmLiveMemoryProbe(&launchRequest{}, "full load required", strings.NewReader(""), &output, false); err == nil || !strings.Contains(err.Error(), "--allow-live-memory-probe") {
		t.Fatalf("non-interactive probe must require an explicit flag, got %v", err)
	}
}

func TestIsLlamaServerExecutableIgnoresArguments(t *testing.T) {
	if !isLlamaServerExecutable("/opt/bin/llama-server-cuda") || !isLlamaServerExecutable("/opt/bin/ik_llama-server") {
		t.Fatal("known server executable was not recognized")
	}
	if isLlamaServerExecutable("/usr/bin/rtk") || isLlamaServerExecutable("/tmp/ggrun") {
		t.Fatal("wrapper containing a --server-bin argument must not be recognized")
	}
}

func TestBackendMemoryMaxUsesDetectedBudgetAndHeadroom(t *testing.T) {
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 120000}}
	if runtime.GOOS != "linux" {
		if got := backendMemoryMaxMB(&launchRequest{}, caps); got != 0 {
			t.Fatalf("non-Linux launch cap = %d, want disabled", got)
		}
		return
	}
	if got := backendMemoryMaxMB(&launchRequest{}, caps); got != 120000 {
		t.Fatalf("detected free RAM cap = %d, want 120000", got)
	}
	if got := backendMemoryMaxMB(&launchRequest{RamBudgetMB: 96000}, caps); got != 96000 {
		t.Fatalf("RAM budget cap = %d, want 96000", got)
	}
	percentCaps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 128727, FreeMB: 125239}}
	if got := backendMemoryMaxMB(&launchRequest{RAMLimitPercent: 90}, percentCaps); got != 112366 {
		t.Fatalf("90%% whole-host cap = %d, want 112366", got)
	}
	if got := backendMemoryMaxMB(&launchRequest{RAMHeadroomMB: 8192}, caps); got != 111808 {
		t.Fatalf("headroom-adjusted cap = %d, want 111808", got)
	}
	if got := backendMemoryMaxMB(&launchRequest{RamBudgetMB: 64000, RAMHeadroomMB: 4096}, caps); got != 59904 {
		t.Fatalf("budget/headroom cap = %d, want 59904", got)
	}
	if got := backendMemoryMaxMB(&launchRequest{RAMHeadroomMB: 200000}, caps); got != 0 {
		t.Fatalf("non-positive cap = %d, want disabled", got)
	}
}

func TestBackendStartOptionsArePlacementStrategyIndependent(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("backend memory scopes are Linux-only")
	}
	req := &launchRequest{RAMHeadroomMB: 8192}
	caps := &detect.Capabilities{RAM: detect.RAMInfo{FreeMB: 120000}}
	single := &placement.Strategy{Type: placement.SingleGPU}
	multi := &placement.Strategy{Type: placement.MultiGPUDense}

	singleOpts := backendStartOptions(req, caps, nil)
	multiOpts := backendStartOptions(req, caps, nil)

	if single.Type != placement.SingleGPU || multi.Type != placement.MultiGPUDense {
		t.Fatalf("test setup broken: single=%s multi=%s", single.Type, multi.Type)
	}
	if singleOpts.MemoryMaxMB != 111808 || multiOpts.MemoryMaxMB != 111808 {
		t.Fatalf("dense strategy memory scopes differ or use wrong cap: single=%d multi=%d", singleOpts.MemoryMaxMB, multiOpts.MemoryMaxMB)
	}
}

func TestHostMemoryPlacementRequiresExplicitContainmentBudget(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("host-memory containment is Linux-only")
	}
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 120000}}
	hostStrategies := []*placement.Strategy{
		{Type: placement.CPUOnly},
		{Type: placement.DenseCPUOffload},
		{Type: placement.MoEOffload},
	}
	for _, strategy := range hostStrategies {
		if err := validateHostMemoryContainment(&launchRequest{}, caps, strategy); err == nil {
			t.Fatalf("%s placement accepted without an explicit RAM safety limit", strategy.Type)
		}
		if err := validateHostMemoryContainment(&launchRequest{RamBudgetMB: 96000}, caps, strategy); err != nil {
			t.Fatalf("%s placement rejected explicit RAM budget: %v", strategy.Type, err)
		}
		if err := validateHostMemoryContainment(&launchRequest{RAMHeadroomMB: 8192}, caps, strategy); err != nil {
			t.Fatalf("%s placement rejected explicit RAM headroom: %v", strategy.Type, err)
		}
		if err := validateHostMemoryContainment(&launchRequest{RAMLimitPercent: 90}, caps, strategy); err != nil {
			t.Fatalf("%s placement rejected RAM limit percent: %v", strategy.Type, err)
		}
	}
	for _, strategy := range []*placement.Strategy{{Type: placement.SingleGPU}, {Type: placement.MultiGPUDense}} {
		if err := validateHostMemoryContainment(&launchRequest{}, caps, strategy); err != nil {
			t.Fatalf("fully GPU-resident %s placement unexpectedly rejected: %v", strategy.Type, err)
		}
	}
}

func TestClaudeCodeParallelIsFeaturePolicyForDeepseek4(t *testing.T) {
	req := &launchRequest{ClaudeCode: true}
	model := &placement.ModelProfile{ModelArch: "deepseek4", CTXTrain: 1048576}
	be := &backendInfo{Tag: "llama"}
	opts := placementOptionsFromRequest(req, model, be, t.TempDir())
	if !opts.RequireMeasuredBuffers {
		t.Fatal("production placement must require measured buffer evidence")
	}
	if opts.Parallel != 4 {
		t.Fatalf("claude-code should request four slots over the shared mainline placement, got %d", opts.Parallel)
	}
	if opts.ContextSize != 1048576 {
		t.Fatalf("claude-code auto context should use the 1M native window, got %d", opts.ContextSize)
	}
	explicit := &launchRequest{ClaudeCode: true, CtxFlag: "262144"}
	if got := placementOptionsFromRequest(explicit, model, be, t.TempDir()).ContextSize; got != 262144 {
		t.Fatalf("explicit Claude Code context must win, got %d", got)
	}
	be = &backendInfo{Tag: "ik_llama"}
	opts = placementOptionsFromRequest(req, &placement.ModelProfile{ModelArch: "qwen3moe"}, be, t.TempDir())
	if opts.Parallel != 4 {
		t.Fatalf("claude-code on other models initially requests 4 slots, got %d", opts.Parallel)
	}
	if opts.ContextSize != 131072 {
		t.Fatalf("unknown model context should use the portable 2x64k baseline, got %d", opts.ContextSize)
	}
}

func TestClaudeCodeProfilesSelectExpectedAutomaticParallelism(t *testing.T) {
	model := &placement.ModelProfile{ModelArch: "deepseek4", CTXTrain: 1048576}
	be := &backendInfo{Tag: "llama"}
	for _, tc := range []struct {
		name string
		req  *launchRequest
		want int
	}{
		{"default_preserves_parallel_workflow", &launchRequest{ClaudeCode: true, Parallel: 1}, 4},
		{"default_preserves_higher_configured_parallel", &launchRequest{ClaudeCode: true, Parallel: 8}, 8},
		{"parallel_profile_preserves_parallel_workflow", &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileParallel}, 4},
		{"parallel_profile_overrides_stale_configured_parallel", &launchRequest{ClaudeCode: true, Parallel: 8, ClaudeProfile: claudeProfileParallel}, 4},
		{"interactive_keeps_single_foreground_slot", &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileInteractive}, 1},
		{"interactive_overrides_stale_configured_parallel", &launchRequest{ClaudeCode: true, Parallel: 8, ClaudeProfile: claudeProfileInteractive}, 1},
		{"interactive_keeps_explicit_parallel", &launchRequest{ClaudeCode: true, Parallel: 2, ParallelSet: true, ClaudeProfile: claudeProfileInteractive}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := placementOptionsFromRequest(tc.req, model, be, t.TempDir()).Parallel; got != tc.want {
				t.Fatalf("parallel=%d, want %d", got, tc.want)
			}
		})
	}
}

func TestClaudeWorkloadProfileScopesCacheEvidence(t *testing.T) {
	model := &placement.ModelProfile{ModelArch: "deepseek4", CTXTrain: 1048576}
	be := &backendInfo{Tag: "llama"}
	parallelDefault := &launchRequest{ClaudeCode: true, Parallel: 1}
	parallelExplicit := &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileParallel}
	interactive := &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileInteractive}

	parallelScope := requestWorkloadProfile(parallelDefault, model)
	if parallelScope == "" {
		t.Fatal("Claude Code default must have a non-empty workload cache scope")
	}
	if got := requestWorkloadProfile(parallelExplicit, model); got != parallelScope {
		t.Fatalf("default and explicit agent-parallel should share behavior scope: got %q, want %q", got, parallelScope)
	}
	interactiveScope := requestWorkloadProfile(interactive, model)
	if interactiveScope == parallelScope {
		t.Fatalf("interactive and parallel profiles shared workload scope %q", interactiveScope)
	}
	if got := scopedProbeBackendTag(interactive, model, be); got == be.Tag {
		t.Fatalf("interactive profile reused the unscoped backend tag %q", got)
	}
	if got := placementOptionsFromRequest(interactive, model, be, t.TempDir()).WorkloadProfile; got != interactiveScope {
		t.Fatalf("placement workload scope=%q, want %q", got, interactiveScope)
	}
}

func TestPlacementEvidenceUsesExactBackendBuildIdentity(t *testing.T) {
	model := &placement.ModelProfile{ModelArch: "deepseek4", CTXTrain: 1048576}
	req := &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileInteractive}
	buildA := &backendInfo{Tag: "llama", Identity: "llama-server-cuda-build-a"}
	buildB := &backendInfo{Tag: "llama", Identity: "llama-server-cuda-build-b"}

	tagA := evidenceBackendCacheTag(buildA)
	tagB := evidenceBackendCacheTag(buildB)
	if tagA == tagB || tagA == buildA.Tag || tagB == buildB.Tag {
		t.Fatalf("backend evidence tags must isolate builds: A=%q B=%q", tagA, tagB)
	}
	if got := scopedProbeBackendTag(req, model, buildA); got == scopedProbeBackendTag(req, model, buildB) {
		t.Fatalf("probe scope reused evidence across backend builds: %q", got)
	}
	if got := placementOptionsFromRequest(req, model, buildA, t.TempDir()).BackendCacheTag; got != tagA {
		t.Fatalf("placement cache tag=%q, want build-scoped %q", got, tagA)
	}
}

func TestClaudeServerLogScopeTracksProfileBuildAndFinalArgs(t *testing.T) {
	cfg := config.Defaults()
	cfg.LogDir = t.TempDir()
	model := &placement.ModelProfile{Path: "model.gguf", ModelArch: "deepseek4", CTXTrain: 65536}
	interactive := &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileInteractive, Port: 8081}
	parallel := &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileParallel, Port: 8081}
	buildA := &backendInfo{Tag: "llama", Identity: "build-a"}
	buildB := &backendInfo{Tag: "llama", Identity: "build-b"}
	argsA := []string{"/tmp/llama-server", "-m", "model.gguf", "-b", "512", "--port", "8081"}
	argsB := []string{"/tmp/llama-server", "-m", "model.gguf", "-b", "128", "--port", "8081"}

	scopeA := claudeLaunchLogScope(interactive, model, buildA, argsA)
	pathA := claudeServerLogPath(cfg, interactive.Port, scopeA)
	if pathA == claudeServerLogPath(cfg, parallel.Port, claudeLaunchLogScope(parallel, model, buildA, argsA)) {
		t.Fatal("interactive and parallel profiles shared a recoverable Claude log")
	}
	if pathA == claudeServerLogPath(cfg, interactive.Port, claudeLaunchLogScope(interactive, model, buildB, argsA)) {
		t.Fatal("different backend builds shared a recoverable Claude log")
	}
	if pathA == claudeServerLogPath(cfg, interactive.Port, claudeLaunchLogScope(interactive, model, buildA, argsB)) {
		t.Fatal("different final launch args shared a recoverable Claude log")
	}

	strategy := &placement.Strategy{Parallel: 1, ContextSize: 65536}
	log := "[ggrun] launch-scope: " + scopeA + "\n" +
		"health check OK model.gguf\n" +
		"n_slots = 1, n_ctx_slot = 65536\n"
	if !previousClaudeLogMatches(log, model, strategy, scopeA) {
		t.Fatal("scoped current log should be recoverable")
	}
	if previousClaudeLogMatches(log, model, strategy, "other-scope") {
		t.Fatal("log from another final launch scope was accepted for recovery")
	}
}

func TestHybridMoECalibrationDoesNotPromoteSplitOwnerAfterHealthCheck(t *testing.T) {
	req := &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileInteractive}
	model := &placement.ModelProfile{ModelArch: "deepseek4", IsMoE: true}
	current := &placement.Strategy{Type: placement.MoEOffload}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}}}
	next, args, ok := maybePromoteMeasuredPlacement(req, &config.Config{}, &backendInfo{Tag: "llama", Identity: "build"}, caps, model, current, []string{"llama-server"})
	if ok || next != nil || args != nil {
		t.Fatalf("hybrid MoE must not auto-promote from load-time evidence: next=%v args=%v ok=%v", next, args, ok)
	}
}

func TestClaudeCodeInteractiveProfileKeepsSSMPrefillBatch(t *testing.T) {
	req := &launchRequest{
		ClaudeCode:    true,
		Parallel:      1,
		ClaudeProfile: claudeProfileInteractive,
	}
	opts := placementOptionsFromRequest(req, &placement.ModelProfile{ModelArch: "deepseek4", CTXTrain: 1048576}, &backendInfo{Tag: "llama"}, t.TempDir())
	s := &placement.Strategy{ContextSize: opts.ContextSize, Parallel: opts.Parallel, BatchSize: 2048, HasSSM: true}
	claudeCodeSlotAdjust(s, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet)
	if s.Parallel != 1 || s.BatchSize != 2048 {
		t.Fatalf("interactive Claude profile changed foreground prefill setup: parallel=%d batch=%d", s.Parallel, s.BatchSize)
	}
}

func TestParseClaudeProfile(t *testing.T) {
	isolateConfig(t)
	for _, tc := range []struct {
		args []string
		want string
	}{
		{[]string{"model.gguf", "--claude-code", "--claude-profile", "agent-interactive"}, claudeProfileInteractive},
		{[]string{"model.gguf", "--claude-code", "--claude-profile=AGENT-PARALLEL"}, claudeProfileParallel},
	} {
		req, err := parseLaunchArgs(tc.args)
		if err != nil {
			t.Fatalf("parse %v: %v", tc.args, err)
		}
		if req.ClaudeProfile != tc.want {
			t.Fatalf("profile=%q, want %q for %v", req.ClaudeProfile, tc.want, tc.args)
		}
	}
	if _, err := parseLaunchArgs([]string{"model.gguf", "--claude-profile", "fastest"}); err == nil {
		t.Fatal("invalid Claude profile was accepted")
	}
	if _, err := parseLaunchArgs([]string{"model.gguf", "--claude-profile", claudeProfileInteractive}); err == nil {
		t.Fatal("Claude profile without --claude-code was accepted")
	}
}

func TestParseEmitServerArgvJSON(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"model.gguf", "--emit-server-argv-json"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.EmitServerArgvJSON {
		t.Fatal("--emit-server-argv-json was not retained for dry-run planning")
	}
}

func TestLaunchPlanEnvironmentMatchesServerChildCUDAContract(t *testing.T) {
	t.Setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
	oldQueue, hadQueue := os.LookupEnv("CUDA_SCALE_LAUNCH_QUEUES")
	if err := os.Unsetenv("CUDA_SCALE_LAUNCH_QUEUES"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if hadQueue {
			_ = os.Setenv("CUDA_SCALE_LAUNCH_QUEUES", oldQueue)
		} else {
			_ = os.Unsetenv("CUDA_SCALE_LAUNCH_QUEUES")
		}
	})
	env := launchPlanEnvironment(
		[]string{"llama-server", "--tensor-split", "1,0,0"},
		"CUDA_VISIBLE_DEVICES=2,0",
	)
	if got := env["CUDA_DEVICE_ORDER"]; got != "PCI_BUS_ID" {
		t.Fatalf("CUDA_DEVICE_ORDER=%q, want PCI_BUS_ID", got)
	}
	if got := env["CUDA_SCALE_LAUNCH_QUEUES"]; got != "4x" {
		t.Fatalf("CUDA_SCALE_LAUNCH_QUEUES=%q, want 4x", got)
	}
	if got := env["CUDA_VISIBLE_DEVICES"]; got != "2,0" {
		t.Fatalf("CUDA_VISIBLE_DEVICES=%q, want 2,0", got)
	}
}

func TestLaunchPlanEnvironmentIncludesStableBackendLibraries(t *testing.T) {
	t.Setenv("LD_LIBRARY_PATH", "")
	t.Setenv("LLM_SERVER_LIB_HUB", "")
	binDir := filepath.Join(t.TempDir(), "build-cuda", "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	backend := filepath.Join(binDir, "llama-server")
	if err := os.WriteFile(backend, []byte("backend"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(binDir, "libllama.so"), []byte("library"), 0o644); err != nil {
		t.Fatal(err)
	}

	env := launchPlanEnvironment([]string{backend}, "", backend)
	if got := env["LD_LIBRARY_PATH"]; got != binDir {
		t.Fatalf("LD_LIBRARY_PATH=%q, want stable backend directory %q", got, binDir)
	}
}

func TestTUILaunchArgsPassSelectedBackend(t *testing.T) {
	cfg := config.Defaults()
	cfg.Backend = "ik_llama"
	args := tuiLaunchArgs(&tui.LaunchRequest{
		ModelPath:   "model.gguf",
		Port:        8081,
		KVPlacement: "auto",
		KVQuality:   "mid",
		Backend:     "ik_llama",
		ClaudeCode:  true,
	}, cfg)
	if !hasAdjacentArg(args, "--backend", "ik_llama") {
		t.Fatalf("selected backend should be explicit so route-arch cannot override it, got %v", args)
	}
	if !hasAdjacentArg(args, "--ctx-size", "fit") {
		t.Fatalf("TUI fit context should map to CLI fit args, got %v", args)
	}
	if !hasArg(args, "--claude-code") {
		t.Fatalf("Claude Code toggle not preserved: %v", args)
	}
}

func TestParseLaunchArgsPlansDirectKVTypeOnce(t *testing.T) {
	t.Setenv("LLM_CONFIG", filepath.Join(t.TempDir(), "config"))
	req, err := parseLaunchArgs([]string{
		"model.gguf", "--cache-type-k", "q5_1", "--cache-type-v=q5_1",
	})
	if err != nil {
		t.Fatalf("parse direct KV cache type: %v", err)
	}
	if req.KVQuality != "q5_1" || len(req.ExtraArgs) != 0 {
		t.Fatalf("direct KV flags must become the planned type, got quality=%q extra=%v", req.KVQuality, req.ExtraArgs)
	}

	strategy, err := placement.Compute(&detect.Capabilities{
		CPU: detect.CPUInfo{Cores: 4}, RAM: detect.RAMInfo{TotalMB: 16384, FreeMB: 16384},
	}, &placement.ModelProfile{
		SizeBytes: 1, NumLayers: 32, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}, placement.Options{CPUMode: true, ContextSize: 32768, KVQuality: req.KVQuality})
	if err != nil {
		t.Fatalf("plan direct KV cache type: %v", err)
	}
	if strategy.KVType != "q5_1" {
		t.Fatalf("strategy KV type = %q, want q5_1", strategy.KVType)
	}
	args := strategy.Args("model.gguf", 8081)
	if !hasAdjacentArg(args, "--cache-type-k", "q5_1") || !hasAdjacentArg(args, "--cache-type-v", "q5_1") {
		t.Fatalf("strategy did not emit q5_1 K/V flags: %v", args)
	}
	if got := strings.Count(strings.Join(args, " "), "--cache-type-k"); got != 1 {
		t.Fatalf("cache type K flag emitted %d times, want once: %v", got, args)
	}
}

func TestParseLaunchArgsRejectsMixedKVTypes(t *testing.T) {
	t.Setenv("LLM_CONFIG", filepath.Join(t.TempDir(), "config"))
	_, err := parseLaunchArgs([]string{
		"model.gguf", "--cache-type-k", "q8_0", "--cache-type-v", "q5_1",
	})
	if err == nil || !strings.Contains(err.Error(), "mixed") {
		t.Fatalf("mixed cache types must fail before an unsafe placement, got %v", err)
	}
}

func TestTUILaunchArgsPreserveMaxContext(t *testing.T) {
	cfg := config.Defaults()
	args := tuiLaunchArgs(&tui.LaunchRequest{ModelPath: "model.gguf", CtxFlag: "max"}, cfg)
	if !hasAdjacentArg(args, "--ctx-size", "max") {
		t.Fatalf("TUI max context should pass CLI max, got %v", args)
	}
}

func TestTUILaunchArgsPassNonDefaultBackend(t *testing.T) {
	cfg := config.Defaults()
	cfg.Backend = "ik_llama"
	args := tuiLaunchArgs(&tui.LaunchRequest{ModelPath: "model.gguf", Backend: "custom"}, cfg)
	if !hasAdjacentArg(args, "--backend", "custom") {
		t.Fatalf("non-default backend selection should be explicit, got %v", args)
	}
}

func hasAdjacentArg(args []string, key, val string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == key && args[i+1] == val {
			return true
		}
	}
	return false
}

func TestParseModelMissingFileReportsModelPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.gguf")
	_, err := parseModel(path)
	if err == nil {
		t.Fatal("expected missing model to fail")
	}
	if !strings.Contains(err.Error(), "model file") || !strings.Contains(err.Error(), "missing.gguf") {
		t.Fatalf("expected model-file path error, got %v", err)
	}
	if strings.Contains(err.Error(), "parse_gguf.py failed") {
		t.Fatalf("missing model should not be reported as parser failure: %v", err)
	}
}

func TestShouldPromoteMoEPlacement(t *testing.T) {
	cur := &placement.Strategy{Type: placement.MoEOffload, NCPUMoE: 37}
	next := &placement.Strategy{Type: placement.MoEOffload, NCPUMoE: 35}
	if !shouldPromoteMoEPlacement(cur, next) {
		t.Fatalf("expected fewer CPU MoE layers to promote")
	}
	if shouldPromoteMoEPlacement(cur, &placement.Strategy{Type: placement.MoEOffload, NCPUMoE: 37}) {
		t.Fatalf("equal CPU MoE layers must not promote")
	}
	if shouldPromoteMoEPlacement(&placement.Strategy{Type: placement.SingleGPU}, next) {
		t.Fatalf("non-MoE-offload current placement must not promote")
	}
}

func TestMeasuredPromotionBypassesPlacementCache(t *testing.T) {
	opts := measuredPromotionOptions(
		&launchRequest{CtxFlag: "32768"},
		&placement.ModelProfile{ModelArch: "qwen3moe", CTXTrain: 32768},
		&backendInfo{Tag: "llama"},
		t.TempDir(),
	)
	if !opts.SkipPlacementCache {
		t.Fatal("measured promotion must recompute instead of reloading the sparse placement it is meant to improve")
	}
}

func TestStartupLogCUDAOOM(t *testing.T) {
	log := "loading\n" +
		"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2206.07 MiB on device 0: cudaMalloc failed: out of memory\n" +
		"segmentation fault"
	device, allocMB, ok := startupLogCUDAOOM(log)
	if !ok || device != 0 || allocMB != 2207 {
		t.Fatalf("cuda oom parse = device %d alloc %d ok %v", device, allocMB, ok)
	}
}

func TestRuntimeLogCUDAOOMRecognizesVMMFormat(t *testing.T) {
	log := strings.Join([]string{
		"[launch] health check OK after 5m1s",
		"CUDA error: out of memory",
		"  current device: 0, in function alloc at ggml-cuda.cu:529",
		"  cuMemCreate(&handle, reserve_size, &prop, 0)",
	}, "\n")
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24564}}}
	device, reserveMB, estimated, ok := runtimeLogCUDAOOM(log, caps, nil)
	if !ok || !estimated || device != 0 || reserveMB != 2457 {
		t.Fatalf("VMM OOM = device %d reserve %d estimated=%v ok=%v", device, reserveMB, estimated, ok)
	}
	_, repeatedReserve, _, ok := runtimeLogCUDAOOM(log, caps, map[int]int{0: reserveMB})
	if !ok || repeatedReserve != 4914 {
		t.Fatalf("repeated VMM OOM reserve = %d ok=%v, want 4914", repeatedReserve, ok)
	}
}

func TestRuntimeLogCUDAOOMPrefersExactAllocation(t *testing.T) {
	log := "allocating 1679.00 MiB on device 2: cudaMalloc failed: out of memory"
	device, reserveMB, estimated, ok := runtimeLogCUDAOOM(log, nil, nil)
	if !ok || estimated || device != 2 || reserveMB != 1679 {
		t.Fatalf("exact OOM = device %d reserve %d estimated=%v ok=%v", device, reserveMB, estimated, ok)
	}
}

func TestPreviousClaudeLogMatchesRuntimeShape(t *testing.T) {
	model := &placement.ModelProfile{Path: "/models/DeepSeek-V4-00001-of-00004.gguf"}
	strategy := &placement.Strategy{ContextSize: 1048576, Parallel: 4}
	const scope = "exact-final-launch-scope"
	log := "[ggrun] launch-scope: " + scope + "\n" +
		"loading model '/models/DeepSeek-V4-00001-of-00004.gguf'\n" +
		"initializing, n_slots = 4, n_ctx_slot = 262144, kv_unified = 'false'\n" +
		"[launch] health check OK after 5m1s\n"
	if !previousClaudeLogMatches(log, model, strategy, scope) {
		t.Fatal("matching previous Claude runtime log was rejected")
	}
	strategy.Parallel = 8
	if previousClaudeLogMatches(log, model, strategy, scope) {
		t.Fatal("log from a different parallel/context shape must not be recovered")
	}
}

func TestStartupComputeMeasurementMustMatchFailedGPU(t *testing.T) {
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	model := &placement.ModelProfile{Path: "/models/model.gguf", NumLayers: 43, NumExperts: 256}
	strategy := &placement.Strategy{
		ContextSize: 1048576,
		UBatchSize:  64,
		KVQuality:   "high",
		KVPlacement: "gpu",
		KVType:      "f16",
		Parallel:    1,
	}
	be := &backendInfo{Tag: "llama"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0}, {Index: 1}}}
	log := "CUDA1 compute buffer size = 100.00 MiB\n" +
		"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 8000.00 MiB on device 0: cudaMalloc failed: out of memory\n" +
		"ggml_gallocr_reserve_n: graph_reserve failed\n"

	measured := recordMeasuredLaunchProbes(nil, cfg, model, strategy, be, caps, log, nil)
	device, _, isCompute, ok := startupLogCUDAOOMDetailed(log)
	if !ok || !isCompute || device != 0 {
		t.Fatalf("failed allocation parse = device %d compute=%v ok=%v", device, isCompute, ok)
	}
	if measured[device] != 0 {
		t.Fatalf("another GPU's probe must not suppress the failed GPU penalty: %v", measured)
	}
}

func TestRouteArchBackendKeepsRegisteredTag(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake-backend probe uses a shell script")
	}
	t.Setenv("LLM_APP_HOME", t.TempDir())
	backendPath := writeFakeBackend(t, "custom-server", "echo llama server help\n")
	if err := backends.Save([]backends.Backend{{Tag: "custom", Path: backendPath, RouteArch: "custom_moe"}}); err != nil {
		t.Fatalf("save backends: %v", err)
	}
	be := routeArchBackend(&backendInfo{Path: "/main/llama-server", Tag: "llama"}, &placement.ModelProfile{ModelArch: "custom_moe"}, &launchRequest{})
	if be == nil || be.Path != backendPath || be.Tag != "custom" {
		t.Fatalf("expected routed custom backend tag, got %#v", be)
	}
}

func TestRouteArchBackendPreservesIKDialectBehindRecipeTag(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake-backend probe uses a shell script")
	}
	t.Setenv("LLM_APP_HOME", t.TempDir())
	backendPath := writeFakeBackend(t, "hy3-server", "echo 'ikawrakow split-mode-graph'\n")
	if err := backends.Save([]backends.Backend{{Tag: "hy3", Path: backendPath, RouteArch: "hy_v3"}}); err != nil {
		t.Fatalf("save backends: %v", err)
	}
	be := routeArchBackend(&backendInfo{Path: "/main/llama-server", Tag: "llama"}, &placement.ModelProfile{ModelArch: "hy_v3"}, &launchRequest{})
	if be == nil || be.Tag != "hy3" || backendDialect(be) != "ik_llama" || !be.IsIK {
		t.Fatalf("expected HY3 identity with IK dialect, got %#v", be)
	}
	opts := placementOptionsFromRequest(&launchRequest{}, &placement.ModelProfile{}, be, t.TempDir())
	if opts.BackendTag != "ik_llama" {
		t.Fatalf("placement got recipe tag instead of IK dialect: %#v", opts)
	}
	if want := evidenceBackendCacheTag(be); opts.BackendCacheTag != want {
		t.Fatalf("placement probes are not isolated to the exact HY3 fork build: got=%q want=%q opts=%#v", opts.BackendCacheTag, want, opts)
	}
}

func TestHY3CompatibilityArgsUseOnlyDerivedMetadata(t *testing.T) {
	model := &placement.ModelProfile{
		ModelArch:                 "hy_v3",
		ExpertSharedCount:         1,
		ExpertSharedCountInferred: true,
		LeadingDense:              1,
		LeadingDenseInferred:      true,
	}
	got := hy3CompatibilityArgs(nil, model, &backendInfo{Tag: "hy3"})
	want := []string{
		"--override-kv", "hy_v3.expert_shared_count=int:1",
		"--override-kv", "hy_v3.leading_dense_block_count=int:1",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("HY3 derived args = %#v, want %#v", got, want)
	}

	got = hy3CompatibilityArgs([]string{"--override-kv", "hy_v3.expert_shared_count=int:2"}, model, &backendInfo{Tag: "hy3"})
	if !reflect.DeepEqual(got, []string{"--override-kv", "hy_v3.leading_dense_block_count=int:1"}) {
		t.Fatalf("explicit expert override must win, got %#v", got)
	}
	if got := hy3CompatibilityArgs(nil, model, &backendInfo{Tag: "llama"}); got != nil {
		t.Fatalf("non-HY3 backend must not receive compatibility args: %#v", got)
	}
}

func TestHY3TemplateArgsUseBundledTemplateWithoutOverridingUser(t *testing.T) {
	root := t.TempDir()
	template := filepath.Join(root, "models", "templates", "Hy3.jinja")
	if err := os.MkdirAll(filepath.Dir(template), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(template, []byte("template"), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(root, "build-cuda", "bin", "llama-server")
	if err := os.MkdirAll(filepath.Dir(bin), 0o755); err != nil {
		t.Fatal(err)
	}
	got := hy3TemplateArgs(nil, &backendInfo{Tag: "hy3", Path: bin})
	want := []string{"--chat-template-file", template}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("HY3 template args = %#v, want %#v", got, want)
	}
	if got := hy3TemplateArgs([]string{"--chat-template", "chatml"}, &backendInfo{Tag: "hy3", Path: bin}); got != nil {
		t.Fatalf("explicit user chat template must win: %#v", got)
	}
}

func TestBackendBuildJobsCapsHeavyCompilers(t *testing.T) {
	if got := backendBuildJobs("cuda", 256); got != 8 {
		t.Fatalf("CUDA build jobs = %d, want 8", got)
	}
	if got := backendBuildJobs("cpu", 256); got != 16 {
		t.Fatalf("CPU build jobs = %d, want 16", got)
	}
	if got := backendBuildJobs("cuda", 4); got != 4 {
		t.Fatalf("small host CUDA build jobs = %d, want 4", got)
	}
	if got := backendBuildJobs("cuda", 0); got != 1 {
		t.Fatalf("invalid CPU count build jobs = %d, want 1", got)
	}
}

func TestRouteArchBackendKeepsExplicitBackend(t *testing.T) {
	be := routeArchBackend(&backendInfo{Path: "/main/llama-server", Tag: "llama"}, &placement.ModelProfile{ModelArch: "deepseek4"}, &launchRequest{Backend: "llama", BackendExplicit: true})
	if be == nil || be.Path != "/main/llama-server" || be.Tag != "llama" {
		t.Fatalf("explicit backend must not be route-arch overridden, got %#v", be)
	}
}

func TestConfiguredBackendExplicit(t *testing.T) {
	if !configuredBackendExplicit("llama") || !configuredBackendExplicit("custom") {
		t.Fatal("named configured backends must be explicit")
	}
	if configuredBackendExplicit("") || configuredBackendExplicit("auto") {
		t.Fatal("empty/auto backend must stay implicit")
	}
}

func TestBackendGPUCapableProbe(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake-backend probe uses a shell script")
	}
	cpuBin := writeFakeBackend(t, "cpu-server", "echo 'Available devices:'\n")
	gpuBin := writeFakeBackend(t, "gpu-server",
		"echo 'Available devices:'\necho '  CUDA0: NVIDIA GeForce RTX 4070 (11873 MiB, 11710 MiB free)'\n")

	if capable, probed := backendGPUCapable(cpuBin); !probed || capable {
		t.Fatalf("cpu-only build: want probed=true capable=false, got probed=%v capable=%v", probed, capable)
	}
	if capable, probed := backendGPUCapable(gpuBin); !probed || !capable {
		t.Fatalf("gpu build: want probed=true capable=true, got probed=%v capable=%v", probed, capable)
	}
	if _, probed := backendGPUCapable(filepath.Join(t.TempDir(), "nope")); probed {
		t.Fatal("missing binary must report probed=false so caps stays unchanged")
	}
}

func TestGateBackendGPUStripsGPUsForCPUBuild(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake-backend probe uses a shell script")
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Name: "RTX 4070", VRAMTotalMB: 12288}}}

	cpuBe := &backendInfo{Path: writeFakeBackend(t, "cpu-server", "echo 'Available devices:'\n")}
	if got := gateBackendGPU(cpuBe, caps); len(got.GPUs) != 0 {
		t.Fatalf("CPU-only backend: GPUs should be stripped, got %d", len(got.GPUs))
	}
	if len(caps.GPUs) != 1 {
		t.Fatal("gateBackendGPU must not mutate the caller's caps")
	}

	gpuBe := &backendInfo{Path: writeFakeBackend(t, "gpu-server",
		"echo 'Available devices:'\necho '  CUDA0: NVIDIA GeForce RTX 4070'\n")}
	if got := gateBackendGPU(gpuBe, caps); len(got.GPUs) != 1 {
		t.Fatalf("GPU-capable backend: GPUs must be kept, got %d", len(got.GPUs))
	}
}

func isolateConfig(t *testing.T) {
	t.Helper()
	t.Setenv("LLM_CONFIG", filepath.Join(t.TempDir(), "missing-config"))
	for _, k := range []string{
		"LLM_PORT", "LLM_CTX_SIZE", "LLM_KV_PLACEMENT", "LLM_KV_QUALITY",
		"LLM_BACKEND", "LLAMA_SERVER", "LLM_HOST", "LLM_SPEC", "LLM_VISION",
	} {
		t.Setenv(k, "")
	}
}

func TestParseLaunchArgsLegacyModelFirst(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{
		"/models/test.gguf", "--dry-run", "--ctx-size", "fit",
		"--kv-placement", "gpu", "--kv-quality", "high", "--spec", "ngram",
		"--mmproj", "/models/mmproj.gguf", "--ram-budget", "48GB",
		"--", "--no-mmap",
	})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if req.ModelPath != "/models/test.gguf" {
		t.Fatalf("model mismatch: %s", req.ModelPath)
	}
	if req.CtxFlag != "fit" || req.KVPlacement != "gpu" || req.KVQuality != "high" {
		t.Fatalf("placement flags mismatch: %#v", req)
	}
	if req.Host != "127.0.0.1" {
		t.Fatalf("expected safe loopback host, got %q", req.Host)
	}
	if req.SpecMode != "ngram" || req.MMProjPath != "/models/mmproj.gguf" || req.RamBudgetMB != 48*1024 {
		t.Fatalf("advanced flags mismatch: %#v", req)
	}
	if !req.NoMMap {
		t.Fatalf("--no-mmap must feed placement, got %#v", req)
	}
	if len(req.ExtraArgs) != 0 {
		t.Fatalf("extra args mismatch: %v", req.ExtraArgs)
	}
}

func TestParseLaunchArgsDefaultKVQualityIsAuto(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"/models/test.gguf"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if req.KVQuality != "auto" {
		t.Fatalf("default KV quality must remain model-aware auto, got %q", req.KVQuality)
	}
}

func TestParseLaunchArgsNoMMapFeedsPlacement(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"model.gguf", "--no-mmap", "-kv", "gpu"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.NoMMap {
		t.Fatalf("expected --no-mmap to set launch request")
	}
	if len(req.ExtraArgs) != 0 {
		t.Fatalf("--no-mmap must not remain a raw backend arg: %v", req.ExtraArgs)
	}
	opts := placementOptionsFromRequest(req, &placement.ModelProfile{CTXTrain: 32768}, &backendInfo{Tag: "llama"}, t.TempDir())
	if !opts.NoMMap {
		t.Fatalf("expected placement options to receive NoMMap")
	}
}

func TestParseLaunchArgsNoMMapAfterDelimiterStillFeedsPlacement(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"model.gguf", "--", "--no-mmap", "--draft-max", "8"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.NoMMap {
		t.Fatalf("expected passthrough --no-mmap to be promoted into placement")
	}
	want := []string{"--draft-max", "8"}
	if len(req.ExtraArgs) != len(want) || req.ExtraArgs[0] != want[0] || req.ExtraArgs[1] != want[1] {
		t.Fatalf("extra args mismatch: got %v want %v", req.ExtraArgs, want)
	}
}

func TestParseLaunchArgsEqualsForms(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{
		"--port=9090", "--ctx-size=max", "--backend=ik_llama",
		"--gpus=1,3", "--host=127.0.0.1", "--spec=draft", "--parallel=4",
		"--ram-limit-percent=88", "--allow-live-memory-probe=true", "model.gguf",
	})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if req.Port != 9090 || req.CtxFlag != "max" || req.Backend != "ik_llama" {
		t.Fatalf("equals flags mismatch: %#v", req)
	}
	if req.GPUsFlag != "1,3" || req.Host != "127.0.0.1" || req.SpecMode != "draft" || req.Parallel != 4 {
		t.Fatalf("equals placement mismatch: %#v", req)
	}
	if req.RAMLimitPercent != 88 {
		t.Fatalf("RAM limit percent = %d, want 88", req.RAMLimitPercent)
	}
	if !req.AllowLiveMemoryProbe {
		t.Fatal("equals-form live memory probe consent was not retained")
	}
}

func TestExplicitBatchFlagsFeedPlacementInsteadOfExtraArgs(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{
		"model.gguf", "--batch-size=512", "-ub", "256",
	})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.BatchSizeSet || req.BatchSize != 512 || !req.UBatchSizeSet || req.UBatchSize != 256 {
		t.Fatalf("explicit batch flags were not retained: %#v", req)
	}
	if len(req.ExtraArgs) != 0 {
		t.Fatalf("explicit placement flags must not remain late extra args: %v", req.ExtraArgs)
	}
	if _, err := parseLaunchArgs([]string{"model.gguf", "--batch-size", "128", "--ubatch-size", "256"}); err == nil {
		t.Fatal("batch smaller than microbatch was accepted")
	}
}

func TestBenchmarkCompatArgs(t *testing.T) {
	args := benchmarkCompatArgs([]string{"/models/test.gguf", "--benchmark", "--port", "9090"})
	if len(args) != 4 || args[0] != "--model" || args[1] != "test.gguf" || args[2] != "--port" || args[3] != "9090" {
		t.Fatalf("unexpected benchmark args: %v", args)
	}

	args = benchmarkCompatArgs([]string{"--model=/models/test.gguf", "--benchmark", "--port=9091"})
	if len(args) != 4 || args[0] != "--model" || args[1] != "test.gguf" || args[2] != "--port" || args[3] != "9091" {
		t.Fatalf("unexpected equals benchmark args: %v", args)
	}

	args = benchmarkCompatArgs([]string{"--model", "/models/test.gguf", "--benchmark"})
	if len(args) != 2 || args[0] != "--model" || args[1] != "test.gguf" {
		t.Fatalf("unexpected explicit model benchmark args: %v", args)
	}
}

func TestAutoStartupTimeoutDoublesHugeMoE(t *testing.T) {
	model := &placement.ModelProfile{
		SizeBytes: 146 * 1024 * 1024 * 1024,
		IsMoE:     true,
	}
	if got := autoStartupTimeout(model); got != 30*time.Minute {
		t.Fatalf("huge MoE timeout mismatch: got %v", got)
	}
}

func TestAutoStartupTimeoutDoublesBaseTimeout(t *testing.T) {
	model := &placement.ModelProfile{SizeBytes: 1024 * 1024}
	if got := autoStartupTimeout(model); got != 8*time.Minute {
		t.Fatalf("base timeout mismatch: got %v", got)
	}
}

func TestKnownCommandAcceptsUpdateAlias(t *testing.T) {
	if !knownCommand("update") {
		t.Fatal("expected update command to be known")
	}
	if !knownCommand("--update") {
		t.Fatal("expected legacy --update alias to be known")
	}
}

func TestResolveCtxFlag(t *testing.T) {
	if got := resolveCtxFlag("fit", 131072); got != 0 {
		t.Fatalf("fit should resolve to auto 0, got %d", got)
	}
	if got := resolveCtxFlag("max", 131072); got != 131072 {
		t.Fatalf("max should resolve to native ctx, got %d", got)
	}
	if got := resolveCtxFlag("8192", 131072); got != 8192 {
		t.Fatalf("manual ctx mismatch: %d", got)
	}
}

func TestParseLaunchArgsFlagFirstLaunch(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"--cpu", "--ctx-size", "2048", "--parallel", "2", "model.gguf"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.CPUMode || req.CtxFlag != "2048" || req.Parallel != 2 || req.ModelPath != "model.gguf" {
		t.Fatalf("flag-first parse mismatch: %#v", req)
	}
}

func TestParseLaunchArgsBenchmark(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"model.gguf", "--benchmark", "--port", "9090"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.Benchmark || req.ModelPath != "model.gguf" || req.Port != 9090 {
		t.Fatalf("benchmark parse mismatch: %#v", req)
	}
}

func TestSelectBackendBackendFlagOverridesConfiguredServerBin(t *testing.T) {
	dir := t.TempDir()
	ikPath := filepath.Join(dir, "ik-llama-server")
	vulkanPath := filepath.Join(dir, "vulkan-llama-server")
	if err := os.WriteFile(ikPath, []byte("#!/bin/sh\necho ikawrakow split-mode-graph\n"), 0755); err != nil {
		t.Fatalf("write ik backend: %v", err)
	}
	if err := os.WriteFile(vulkanPath, []byte("#!/bin/sh\necho vulkan backend\n"), 0755); err != nil {
		t.Fatalf("write vulkan backend: %v", err)
	}

	caps := &detect.Capabilities{Backends: []detect.Backend{
		{Name: "llama-server", Path: vulkanPath},
	}}
	req := &launchRequest{
		ServerBin:       ikPath,
		Backend:         "vulkan",
		BackendExplicit: true,
	}
	be := selectBackend(caps, req)
	if be == nil || be.Path != vulkanPath || be.Tag != "vulkan" {
		t.Fatalf("expected explicit backend to override configured server bin, got %#v", be)
	}
}

func TestSelectBackendExplicitServerBinWins(t *testing.T) {
	dir := t.TempDir()
	ikPath := filepath.Join(dir, "ik-llama-server")
	vulkanPath := filepath.Join(dir, "vulkan-llama-server")
	if err := os.WriteFile(ikPath, []byte("#!/bin/sh\necho ikawrakow split-mode-graph\n"), 0755); err != nil {
		t.Fatalf("write ik backend: %v", err)
	}
	if err := os.WriteFile(vulkanPath, []byte("#!/bin/sh\necho vulkan backend\n"), 0755); err != nil {
		t.Fatalf("write vulkan backend: %v", err)
	}

	caps := &detect.Capabilities{Backends: []detect.Backend{
		{Name: "llama-server", Path: vulkanPath},
	}}
	req := &launchRequest{
		ServerBin:         ikPath,
		ServerBinExplicit: true,
		Backend:           "vulkan",
		BackendExplicit:   true,
	}
	be := selectBackend(caps, req)
	if be == nil || be.Path != ikPath || be.Tag != "ik_llama" {
		t.Fatalf("expected explicit server bin to win, got %#v", be)
	}
}

func TestDetectBackendCUDAHelpMentionVulkanStaysLlama(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake-backend probe uses a shell script")
	}
	bin := writeFakeBackend(t, "llama-server-cuda", "echo 'Vulkan appears in generic help text'\n")
	info := detectBackend(bin)
	if info.Tag != "llama" {
		t.Fatalf("CUDA/mainline path should stay llama even when help mentions Vulkan, got %#v", info)
	}
}

func TestBackendMatchesVulkanAliases(t *testing.T) {
	info := &backendInfo{Path: "/home/me/llama.cpp/build-vulkan/bin/llama-server", Tag: "vulkan"}
	if !backendMatches(info, "llama-server", "vulkan") {
		t.Fatalf("expected vulkan backend match")
	}
	if !backendMatches(info, "llama-server", "llama-vk") {
		t.Fatalf("expected llama-vk backend alias match")
	}
}

func TestResolveModelPathUsesConfiguredModelDir(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(model, []byte("gguf"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	got := resolveModelPath("model.gguf", dir)
	if got != model {
		t.Fatalf("expected configured model dir path, got %s", got)
	}
}

func TestApplyTuneCacheAutoSelectsBest(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	cachePath := filepath.Join(cacheDir, "tune_model.gguf_4_hw12345678_vulkan.json")
	doc := `{
		"model": "model.gguf",
		"baseline_gen_tps": 100.0,
		"baseline_wins": false,
		"best_config": {
			"name": "threads12",
			"flags": {"--threads": "12"},
			"gen_tps": 120.0,
			"pp_tps": 300.0
		},
		"rounds": 1,
		"tuned_at": "2026-05-28T00:00:00Z"
	}`
	if err := os.WriteFile(cachePath, []byte(doc), 0644); err != nil {
		t.Fatalf("write tune cache: %v", err)
	}

	args := applyTuneCache(&launchRequest{ModelPath: modelPath}, []string{"llama-server", "--threads", "8"}, cacheDir, "vulkan", false, nil)
	if !hasArgValue(args, "--threads", "12") {
		t.Fatalf("expected cached --threads override, got %v", args)
	}
}

func TestApplyTuneCacheSkipsMemoryExpandingOverrideWhenVRAMHeadroomIsLow(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	cachePath := filepath.Join(cacheDir, "tune_model.gguf_4_hw12345678_vulkan.json")
	doc := `{
		"model": "model.gguf",
		"baseline_gen_tps": 100.0,
		"baseline_wins": false,
		"best_config": {
			"name": "larger-ubatch",
			"flags": {"-ub": "2048"},
			"gen_tps": 120.0,
			"pp_tps": 300.0
		},
		"rounds": 1,
		"tuned_at": "2026-06-02T00:00:00Z"
	}`
	if err := os.WriteFile(cachePath, []byte(doc), 0644); err != nil {
		t.Fatalf("write tune cache: %v", err)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, VRAMUsedMB: 21000}}}
	base := []string{"llama-server", "--device", "Vulkan0", "-ub", "1024"}

	args := applyTuneCache(&launchRequest{ModelPath: modelPath, TuneCache: cachePath}, base, cacheDir, "vulkan", false, caps)
	if !hasArgValue(args, "-ub", "1024") {
		t.Fatalf("expected low-headroom guard to keep base -ub, got %v", args)
	}
}

func TestApplyTuneCacheAllowsNonVRAMOverrideWhenVRAMHeadroomIsLow(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	cachePath := filepath.Join(cacheDir, "tune_model.gguf_4_hw12345678_vulkan.json")
	doc := `{
		"model": "model.gguf",
		"baseline_gen_tps": 100.0,
		"baseline_wins": false,
		"best_config": {
			"name": "threads12",
			"flags": {"--threads": "12"},
			"gen_tps": 120.0,
			"pp_tps": 300.0
		},
		"rounds": 1,
		"tuned_at": "2026-06-02T00:00:00Z"
	}`
	if err := os.WriteFile(cachePath, []byte(doc), 0644); err != nil {
		t.Fatalf("write tune cache: %v", err)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, VRAMUsedMB: 24000}}}
	base := []string{"llama-server", "--device", "Vulkan0", "--threads", "8"}

	args := applyTuneCache(&launchRequest{ModelPath: modelPath, TuneCache: cachePath}, base, cacheDir, "vulkan", false, caps)
	if !hasArgValue(args, "--threads", "12") {
		t.Fatalf("expected non-VRAM override to apply, got %v", args)
	}
}

func TestApplyTuneCacheDoesNotCrossBackend(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	cachePath := filepath.Join(cacheDir, "tune_model.gguf_4_hw12345678_vulkan.json")
	doc := `{
		"model": "model.gguf",
		"baseline_gen_tps": 100.0,
		"baseline_wins": false,
		"best_config": {
			"name": "threads12",
			"flags": {"--threads": "12"},
			"gen_tps": 120.0,
			"pp_tps": 300.0
		},
		"rounds": 1,
		"tuned_at": "2026-05-28T00:00:00Z"
	}`
	if err := os.WriteFile(cachePath, []byte(doc), 0644); err != nil {
		t.Fatalf("write tune cache: %v", err)
	}

	args := applyTuneCache(&launchRequest{ModelPath: modelPath}, []string{"llama-server", "--threads", "8"}, cacheDir, "llama", false, nil)
	if !hasArgValue(args, "--threads", "8") {
		t.Fatalf("expected backend-scoped cache to be ignored, got %v", args)
	}
}

func TestBestTuneCachePathFiltersHardwareHash(t *testing.T) {
	cacheDir := t.TempDir()
	cachePath := filepath.Join(cacheDir, "tune_model.gguf_4_hwdeadbeef_vulkan.json")
	doc := `{
		"model": "model.gguf",
		"baseline_gen_tps": 100.0,
		"baseline_wins": false,
		"best_config": {
			"name": "threads12",
			"flags": {"--threads": "12"},
			"gen_tps": 120.0,
			"pp_tps": 300.0
		},
		"rounds": 1,
		"tuned_at": "2026-05-28T00:00:00Z"
	}`
	if err := os.WriteFile(cachePath, []byte(doc), 0644); err != nil {
		t.Fatalf("write tune cache: %v", err)
	}
	if got := bestTuneCachePath(cacheDir, "model.gguf", "vulkan", false, "badc0ffe"); got != "" {
		t.Fatalf("expected wrong-hardware cache to be ignored, got %s", got)
	}
	if got := bestTuneCachePath(cacheDir, "model.gguf", "vulkan", false, "deadbeef"); got != cachePath {
		t.Fatalf("expected matching hardware cache, got %s", got)
	}
}

func TestApplyTuneCacheSkipsAutomaticGenericTuneForClaudeCode(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	cachePath := filepath.Join(cacheDir, "tune_model.gguf_4_hwdeadbeef_vulkan.json")
	doc := `{
		"model": "model.gguf",
		"baseline_gen_tps": 100.0,
		"baseline_wins": false,
		"best_config": {"name": "threads12", "flags": {"--threads": "12"}, "gen_tps": 120.0, "pp_tps": 300.0},
		"rounds": 1,
		"tuned_at": "2026-05-28T00:00:00Z"
	}`
	if err := os.WriteFile(cachePath, []byte(doc), 0644); err != nil {
		t.Fatalf("write generic tune cache: %v", err)
	}
	base := []string{"llama-server", "--threads", "8"}
	got := applyTuneCache(&launchRequest{ModelPath: modelPath, ClaudeCode: true}, base, cacheDir, "vulkan", false, nil)
	if !hasArgValue(got, "--threads", "8") {
		t.Fatalf("automatic generic tune changed Claude Code args: %v", got)
	}
	got = applyTuneCache(&launchRequest{ModelPath: modelPath, ClaudeCode: true, TuneCache: cachePath}, base, cacheDir, "vulkan", false, nil)
	if !hasArgValue(got, "--threads", "12") {
		t.Fatalf("explicit Claude Code tune was not honored: %v", got)
	}
}

func hasArgValue(args []string, flag, value string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag && args[i+1] == value {
			return true
		}
	}
	return false
}

func TestBackendSearchPathsIncludeAppHomeBackend(t *testing.T) {
	appHome := filepath.Join(t.TempDir(), "ggrun")
	t.Setenv("LLM_APP_HOME", appHome)
	paths := backendSearchPaths()
	want := filepath.Join(appHome, ".bin", "llama-server")
	for _, path := range paths {
		if path == want {
			return
		}
	}
	t.Fatalf("missing app-home backend path %s in %#v", want, paths)
}

func TestFirstPositionalSkipsParallelValue(t *testing.T) {
	// --parallel takes a value; "2" must not be mistaken for the model arg.
	got := firstPositional([]string{"--parallel", "2", "unsloth/Qwen-GGUF", "--download"})
	if got != "unsloth/Qwen-GGUF" {
		t.Fatalf("expected repo positional, got %q", got)
	}
	got = firstPositional([]string{"-c", "32768", "model.gguf"})
	if got != "model.gguf" {
		t.Fatalf("expected model.gguf, got %q", got)
	}
	got = firstPositional([]string{"--ram-headroom", "2G", "org/model-GGUF", "--download"})
	if got != "org/model-GGUF" {
		t.Fatalf("--ram-headroom value was treated as positional: got %q", got)
	}
	got = firstPositional([]string{"--ram-limit-percent", "90", "org/model-GGUF", "--download"})
	if got != "org/model-GGUF" {
		t.Fatalf("--ram-limit-percent value was treated as positional: got %q", got)
	}
	got = firstPositional([]string{"--claude-profile", "agent-interactive", "org/model-GGUF", "--download"})
	if got != "org/model-GGUF" {
		t.Fatalf("--claude-profile value was treated as positional: got %q", got)
	}
}

func TestParseLaunchArgsRejectsInvalidSafetyFlags(t *testing.T) {
	cases := []struct {
		name string
		args []string
	}{
		{"port text", []string{"model.gguf", "--port", "abc"}},
		{"port zero", []string{"model.gguf", "--port=0"}},
		{"parallel text", []string{"model.gguf", "--parallel", "many"}},
		{"parallel zero", []string{"model.gguf", "--parallel=0"}},
		{"vram headroom text", []string{"model.gguf", "--vram-headroom", "two-gig"}},
		{"ram headroom negative", []string{"model.gguf", "--ram-headroom=-2G"}},
		{"ram percent zero", []string{"model.gguf", "--ram-limit-percent=0"}},
		{"ram percent high", []string{"model.gguf", "--ram-limit-percent", "101"}},
		{"gpu token", []string{"model.gguf", "--gpus", "0,fast"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			isolateConfig(t)
			if _, err := parseLaunchArgs(tc.args); err == nil {
				t.Fatalf("parseLaunchArgs(%v) accepted invalid input", tc.args)
			}
		})
	}
}

func TestParseLongContextValidationArgsStripsRecorderFlags(t *testing.T) {
	tokens, gpuUsed, launchArgs, err := parseLongContextValidationArgs([]string{
		"model.gguf", "--prompt-tokens", "60000", "--gpu-used", "0:16224,1:9059", "--parallel", "4",
	})
	if err != nil {
		t.Fatalf("parse validation args: %v", err)
	}
	if tokens != 60000 {
		t.Fatalf("tokens=%d, want 60000", tokens)
	}
	if gpuUsed[0] != 16224 || gpuUsed[1] != 9059 {
		t.Fatalf("gpu-used=%v", gpuUsed)
	}
	if strings.Join(launchArgs, " ") != "model.gguf --parallel 4" {
		t.Fatalf("launch args leaked recorder flags: %v", launchArgs)
	}
}

func TestPlacementOptionsNeverMapsInvalidGPUToZero(t *testing.T) {
	opts := placementOptionsFromRequest(
		&launchRequest{GPUsFlag: "not-a-gpu"},
		&placement.ModelProfile{}, &backendInfo{Tag: "llama"}, t.TempDir(),
	)
	if len(opts.GPUs) != 0 {
		t.Fatalf("invalid GPU input became placement GPUs %v", opts.GPUs)
	}
}

func TestApplyGPUVisibilitySetsEnv(t *testing.T) {
	t.Setenv("CUDA_VISIBLE_DEVICES", "")
	req := &launchRequest{GPUsFlag: "2,0"}
	env := applyGPUVisibility(req, "ik_llama")
	if env != "CUDA_VISIBLE_DEVICES=0,2" {
		t.Fatalf("unexpected env assignment: %q", env)
	}
	if os.Getenv("CUDA_VISIBLE_DEVICES") != "0,2" {
		t.Fatalf("CUDA_VISIBLE_DEVICES not set: %q", os.Getenv("CUDA_VISIBLE_DEVICES"))
	}

	t.Setenv("GGML_VK_VISIBLE_DEVICES", "")
	env = applyGPUVisibility(&launchRequest{GPUsFlag: "1"}, "vulkan")
	if env != "GGML_VK_VISIBLE_DEVICES=1" {
		t.Fatalf("unexpected vulkan env assignment: %q", env)
	}
}

func TestApplyGPUVisibilityNoFlagNoEnv(t *testing.T) {
	if env := applyGPUVisibility(&launchRequest{}, "ik_llama"); env != "" {
		t.Fatalf("expected no env assignment without --gpus, got %q", env)
	}
	if env := applyGPUVisibility(&launchRequest{GPUsFlag: "abc"}, "ik_llama"); env != "" {
		t.Fatalf("expected no env assignment for invalid --gpus, got %q", env)
	}
}

func TestRuntimeGPUCapabilitiesMatchesVisibilityRenumbering(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, Name: "large", VRAMTotalMB: 24576},
		{Index: 1, Name: "slow", VRAMTotalMB: 12288},
		{Index: 2, Name: "fast", VRAMTotalMB: 12282},
	}}
	runtime, mapping := runtimeGPUCapabilities(caps, &launchRequest{GPUsFlag: "2,1"})
	if runtime == nil || len(runtime.GPUs) != 2 {
		t.Fatalf("runtime GPU filter mismatch: %#v", runtime)
	}
	if runtime.GPUs[0].Name != "slow" || runtime.GPUs[0].Index != 0 || runtime.GPUs[1].Name != "fast" || runtime.GPUs[1].Index != 1 {
		t.Fatalf("visible GPU order/renumber mismatch: %#v", runtime.GPUs)
	}
	if mapping[0] != 1 || mapping[1] != 2 || physicalGPUIndex(1, mapping) != 2 {
		t.Fatalf("visible-to-physical mapping mismatch: %#v", mapping)
	}
}

func TestClaudeCodeAutocompactPct(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want int
	}{
		{"parallel4_65k_slot", []string{"--ctx-size", "262144", "--parallel", "4"}, 24},
		{"parallel8_32k_slot", []string{"--ctx-size", "262144", "--parallel", "8"}, 12},
		{"parallel1_full_ctx_caps_at_90", []string{"--ctx-size", "262144", "--parallel", "1"}, 90},
		{"no_parallel_defaults_to_1", []string{"--ctx-size", "65536"}, 24},
		{"tiny_slot_floors_at_5", []string{"--ctx-size", "8192", "--parallel", "8"}, 5},
		{"missing_ctx_keeps_legacy_default", []string{"--parallel", "4"}, 25},
		{"short_ctx_alias", []string{"-c", "131072", "-np", "4"}, 12},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := claudeCodeAutocompactPct(tc.args); got != tc.want {
				t.Fatalf("claudeCodeAutocompactPct(%v) = %d, want %d", tc.args, got, tc.want)
			}
		})
	}
}

func TestArgIntValue(t *testing.T) {
	args := []string{"-m", "model.gguf", "--ctx-size", "4096", "--parallel", "4", "--flag"}
	if got := argIntValue(args, "--ctx-size", "-c"); got != 4096 {
		t.Fatalf("--ctx-size = %d, want 4096", got)
	}
	if got := argIntValue(args, "--parallel", "-np"); got != 4 {
		t.Fatalf("--parallel = %d, want 4", got)
	}
	if got := argIntValue(args, "--missing"); got != -1 {
		t.Fatalf("--missing = %d, want -1", got)
	}
	// trailing flag with no value must not panic or misparse
	if got := argIntValue(args, "--flag"); got != -1 {
		t.Fatalf("--flag (no value) = %d, want -1", got)
	}
	// last-wins: a user value appended after the strategy's must override it, to
	// mirror llama.cpp/ik_llama (which honor the final repeated flag).
	dup := []string{"--ctx-size", "262144", "--parallel", "4", "--ctx-size", "16384"}
	if got := argIntValue(dup, "--ctx-size", "-c"); got != 16384 {
		t.Fatalf("last-wins --ctx-size = %d, want 16384", got)
	}
	// an unparseable later value is skipped, falling back to the last parseable one
	if got := argIntValue([]string{"--ctx-size", "8192", "--ctx-size", "max"}, "--ctx-size"); got != 8192 {
		t.Fatalf("last parseable --ctx-size = %d, want 8192", got)
	}
}

func TestClaudeCodeAutocompactPctLastWinsOnUserOverride(t *testing.T) {
	// strategy emits 262144/4 (pct 24); a user appends --ctx-size 16384 (backend
	// last-wins → 16384/4 = 4096 slot → pct floors at 5). Must reflect the real slot.
	args := []string{"--ctx-size", "262144", "--parallel", "4", "--ctx-size", "16384"}
	if got := claudeCodeAutocompactPct(args); got != 5 {
		t.Fatalf("autocompact pct with user override = %d, want 5", got)
	}
}

func TestClaudeCodeSearchMCPArgsRespectsUserConfig(t *testing.T) {
	if got := claudeCodeSearchMCPArgs([]string{"--mcp-config", "mine.json"}); got != nil {
		t.Fatalf("expected nil when user passed --mcp-config, got %v", got)
	}
}

func TestClaudeCodeSearchMCPArgsEnablesResearchTools(t *testing.T) {
	binDir := t.TempDir()
	uvx := filepath.Join(binDir, "uvx")
	if err := os.WriteFile(uvx, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", binDir)

	got := claudeCodeSearchMCPArgs(nil)
	joined := strings.Join(got, " ")
	for _, want := range []string{"--mcp-config", "duckduckgo-mcp-server", "--allowedTools", "mcp__ddg-search__search", "mcp__ddg-search__fetch_content"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("missing %q in research MCP args: %v", want, got)
		}
	}

	got = claudeCodeSearchMCPArgs([]string{"--allowed-tools", "mine"})
	if hasArg(got, "--allowedTools") || hasArg(got, "--allowed-tools") {
		t.Fatalf("user allowed-tools must not be overridden, got %v", got)
	}
}

func TestClaudeCodePermissionArgsDefaultsToLocalAuto(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "")
	got := claudeCodePermissionArgs(nil)
	if len(got) != 2 || got[0] != "--permission-mode" || got[1] != "auto" {
		t.Fatalf("local Claude launch must use the routed Auto reviewer, got %v", got)
	}
}

func TestClaudeCodePermissionArgsRespectsOverrides(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "auto")
	if got := claudeCodePermissionArgs(nil); len(got) != 2 || got[1] != "auto" {
		t.Fatalf("environment override not respected: %v", got)
	}
	if got := claudeCodePermissionArgs([]string{"--permission-mode", "plan"}); got != nil {
		t.Fatalf("explicit CLI mode must win, got %v", got)
	}
	if got := claudeCodePermissionArgs([]string{"--permission-mode=manual"}); got != nil {
		t.Fatalf("explicit equals-form CLI mode must win, got %v", got)
	}
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "inherit")
	if got := claudeCodePermissionArgs(nil); got != nil {
		t.Fatalf("inherit must preserve settings.json mode, got %v", got)
	}
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "not-a-mode")
	if got := claudeCodePermissionArgs(nil); len(got) != 2 || got[1] != "auto" {
		t.Fatalf("invalid override must fail safe to routed Auto, got %v", got)
	}
}

func TestClaudeCodeAliasArgs(t *testing.T) {
	base := []string{"-m", "model.gguf", "--port", "8081"}
	// claude-code mode appends --alias local so /v1/models advertises "local"
	got := claudeCodeAliasArgs(base, true)
	if argIndexOf(got, "--alias") < 0 || got[argIndexOf(got, "--alias")+1] != "local" {
		t.Fatalf("expected --alias local appended, got %v", got)
	}
	// non-claude-code mode is a no-op
	if got := claudeCodeAliasArgs(base, false); len(got) != len(base) {
		t.Fatalf("expected no change outside claude-code mode, got %v", got)
	}
	// a user-set alias is respected (not doubled)
	user := []string{"-m", "model.gguf", "--alias", "mymodel"}
	if got := claudeCodeAliasArgs(user, true); len(got) != len(user) {
		t.Fatalf("expected user --alias preserved without doubling, got %v", got)
	}
	if got := claudeCodeAliasArgs([]string{"-a", "x"}, true); len(got) != 2 {
		t.Fatalf("expected short -a alias respected, got %v", got)
	}
}

func TestClaudeCodeCacheArgs(t *testing.T) {
	base := []string{"llama-server", "-m", "model.gguf"}
	got := claudeCodeCacheArgs(base, true, "--cache-prompt --cache-reuse N", true)
	if !hasArgValue(got, "--cache-reuse", "256") {
		t.Fatalf("expected Claude cache reuse default, got %v", got)
	}
	if got := claudeCodeCacheArgs(base, false, "--cache-reuse N", true); len(got) != len(base) {
		t.Fatalf("expected no cache change outside Claude mode, got %v", got)
	}
	if got := claudeCodeCacheArgs(base, true, "--cache-prompt", true); len(got) != len(base) {
		t.Fatalf("expected unsupported backend to remain unchanged, got %v", got)
	}
	if got := claudeCodeCacheArgs(base, true, "--cache-reuse N", false); len(got) != len(base) {
		t.Fatalf("expected recurrent context to skip unsupported cache shifting, got %v", got)
	}
	for _, user := range [][]string{
		{"llama-server", "--cache-reuse", "0"},
		{"llama-server", "--cache-reuse=0"},
		{"llama-server", "--no-cache-prompt"},
	} {
		got := claudeCodeCacheArgs(user, true, "--cache-reuse N", true)
		if len(got) != len(user) {
			t.Fatalf("expected user cache override preserved, input %v got %v", user, got)
		}
	}
}

func argIndexOf(args []string, want string) int {
	for i, a := range args {
		if a == want {
			return i
		}
	}
	return -1
}

func TestClaudeCodeEnvDisablesIdleTimeoutForLocalBackend(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "real-key")
	t.Setenv("API_TIMEOUT_MS", "")
	t.Setenv("API_FORCE_IDLE_TIMEOUT", "")
	t.Setenv("CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS", "")
	t.Setenv("CLAUDE_ENABLE_BYTE_WATCHDOG", "")
	t.Setenv("CLAUDE_ENABLE_STREAM_WATCHDOG", "")
	t.Setenv("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "")
	t.Setenv("CLAUDE_CODE_EFFORT_LEVEL", "")
	env := claudeCodeEnv("0.0.0.0", 8081, []string{"llama-server", "--ctx-size", "1048576", "--parallel", "4"})

	if envHasPrefix(env, "ANTHROPIC_API_KEY=") {
		t.Fatalf("claude-code env must drop real ANTHROPIC_API_KEY: %v", env)
	}
	for _, want := range []string{
		"ANTHROPIC_BASE_URL=http://127.0.0.1:8081",
		"API_TIMEOUT_MS=2147483647",
		"API_FORCE_IDLE_TIMEOUT=0",
		"CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS=2147483647",
		"CLAUDE_ENABLE_BYTE_WATCHDOG=0",
		"CLAUDE_ENABLE_STREAM_WATCHDOG=0",
		"CLAUDE_CODE_EFFORT_LEVEL=xhigh",
	} {
		if !envContains(env, want) {
			t.Fatalf("missing %s in claude-code env: %v", want, env)
		}
	}

	t.Setenv("CLAUDE_CODE_EFFORT_LEVEL", "max")
	overridden := claudeCodeEnv("127.0.0.1", 8081, nil)
	if !envContains(overridden, "CLAUDE_CODE_EFFORT_LEVEL=max") {
		t.Fatalf("explicit Claude effort override was not preserved: %v", overridden)
	}
}

func envContains(env []string, want string) bool {
	for _, kv := range env {
		if kv == want {
			return true
		}
	}
	return false
}

func envHasPrefix(env []string, prefix string) bool {
	for _, kv := range env {
		if strings.HasPrefix(kv, prefix) {
			return true
		}
	}
	return false
}

func TestClaudeCodeSlotAdjust(t *testing.T) {
	cases := []struct {
		name         string
		ctx, par     int
		claudeCode   bool
		explicit     bool
		wantParallel int
	}{
		{"large_ctx_keeps_4", 262144, 4, true, false, 4},
		{"fit_32k_drops_to_1", 32768, 4, true, false, 1}, // the MiniMax-M3 regression: 8k slots
		{"128k_drops_to_2", 131072, 4, true, false, 2},   // 65k slots
		{"tiny_ctx_floors_at_1", 8192, 4, true, false, 1},
		{"not_claude_mode_untouched", 32768, 4, false, false, 4},
		{"parallel_1_untouched", 32768, 1, true, false, 1},
		{"explicit_parallel_kept", 65536, 2, true, true, 2}, // user tuning a big MoE: 2x32k slots
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := &placement.Strategy{ContextSize: tc.ctx, Parallel: tc.par}
			claudeCodeSlotAdjust(s, tc.claudeCode, tc.explicit, false)
			if s.Parallel != tc.wantParallel {
				t.Fatalf("ctx=%d par=%d cc=%v: got parallel %d, want %d", tc.ctx, tc.par, tc.claudeCode, s.Parallel, tc.wantParallel)
			}
		})
	}
}

func TestClaudeCodeHybridUsesFairPromptBatch(t *testing.T) {
	s := &placement.Strategy{ContextSize: 1048576, Parallel: 2, BatchSize: 2048, HasSSM: true}
	claudeCodeSlotAdjust(s, true, false, false)
	if s.BatchSize != claudeHybridBatch {
		t.Fatalf("hybrid Claude batch=%d, want %d", s.BatchSize, claudeHybridBatch)
	}

	nonClaude := &placement.Strategy{ContextSize: 1048576, Parallel: 2, BatchSize: 2048, HasSSM: true}
	claudeCodeSlotAdjust(nonClaude, false, false, false)
	if nonClaude.BatchSize != 2048 {
		t.Fatalf("non-Claude batch was changed: %d", nonClaude.BatchSize)
	}
}

func TestClaudeCodeHybridExplicitBatchOverridesFairnessCap(t *testing.T) {
	s := &placement.Strategy{ContextSize: 65536, Parallel: 4, BatchSize: 512, HasSSM: true}
	claudeCodeSlotAdjust(s, true, true, true)
	if s.BatchSize != 512 {
		t.Fatalf("explicit hybrid Claude batch=%d, want 512", s.BatchSize)
	}
}

func TestClaudeCodeHybridSingleSlotKeepsPlacementBatch(t *testing.T) {
	for _, tc := range []struct {
		name          string
		ctx, parallel int
		explicit      bool
	}{
		{"single_slot", 1048576, 1, false},
		// The automatic 4-slot default becomes one slot at this context. Verify
		// batch fairness is evaluated after that normalization.
		{"auto_reduced_to_single_slot", 32768, 4, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := &placement.Strategy{ContextSize: tc.ctx, Parallel: tc.parallel, BatchSize: 2048, HasSSM: true}
			claudeCodeSlotAdjust(s, true, tc.explicit, false)
			if s.Parallel != 1 {
				t.Fatalf("parallel=%d, want final single slot", s.Parallel)
			}
			if s.BatchSize != 2048 {
				t.Fatalf("single-slot hybrid batch=%d, want placement-selected 2048", s.BatchSize)
			}
		})
	}
}

func TestClaudeCodeSamplingArgs(t *testing.T) {
	base := []string{"-m", "model.gguf"}
	got := claudeCodeSamplingArgs(base, true, nil)
	for _, want := range []string{"--presence-penalty", "--repeat-penalty", "--repeat-last-n", "--top-k", "--top-p", "--min-p"} {
		if !hasArg(got, want) {
			t.Fatalf("expected %s in claude-code sampling defaults, got %v", want, got)
		}
	}
	// non-claude-code: untouched
	if got := claudeCodeSamplingArgs(base, false, nil); len(got) != len(base) {
		t.Fatalf("expected no sampling flags outside claude-code mode, got %v", got)
	}
	// user-set flag wins: not doubled, others still added
	user := []string{"-m", "model.gguf", "--presence-penalty", "1.5"}
	got = claudeCodeSamplingArgs(user, true, nil)
	n := 0
	for _, a := range got {
		if a == "--presence-penalty" {
			n++
		}
	}
	if n != 1 || !hasArg(got, "--top-k") {
		t.Fatalf("expected user presence-penalty kept once + other defaults added, got %v", got)
	}
}

func TestClaudeCodeSamplingArgsDeepSeek4(t *testing.T) {
	base := []string{"-m", "model.gguf"}
	model := &placement.ModelProfile{ModelArch: "deepseek4"}
	got := claudeCodeSamplingArgs(base, true, model)
	for _, want := range []string{"--temp", "--top-k", "--top-p", "--min-p", "--reasoning-budget"} {
		if !hasArg(got, want) {
			t.Fatalf("expected %s in deepseek4 claude-code defaults, got %v", want, got)
		}
	}
	if got[argIndexOf(got, "--top-k")+1] != "40" || got[argIndexOf(got, "--min-p")+1] != "0.05" || got[argIndexOf(got, "--reasoning-budget")+1] != "0" {
		t.Fatalf("unexpected deepseek4 defaults: %v", got)
	}

	user := []string{"-m", "model.gguf", "--reasoning-budget", "-1", "--top-k", "10"}
	got = claudeCodeSamplingArgs(user, true, model)
	if got[argIndexOf(got, "--reasoning-budget")+1] != "-1" || got[argIndexOf(got, "--top-k")+1] != "10" {
		t.Fatalf("user deepseek4 sampling overrides should win, got %v", got)
	}
}

// A models dir full of symlinks (e.g. shards linked from another disk) must be
// sized via the link targets. Summing entry.Info() (lstat) once shrank a 146GB
// sharded model to 365 bytes; the parseModel drift-rescale then crushed
// ExpertBytes with it and placement pinned all expert layers onto one GPU.
func TestTotalModelSizeFollowsSymlinkedShards(t *testing.T) {
	realDir := t.TempDir()
	linkDir := t.TempDir()
	var want int64
	for i := 1; i <= 3; i++ {
		name := fmt.Sprintf("big-%05d-of-00003.gguf", i)
		data := bytes.Repeat([]byte{0xAB}, 1000*i)
		if err := os.WriteFile(filepath.Join(realDir, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(filepath.Join(realDir, name), filepath.Join(linkDir, name)); err != nil {
			t.Fatal(err)
		}
		want += int64(len(data))
	}

	if got := totalModelSize(filepath.Join(linkDir, "big-00001-of-00003.gguf")); got != want {
		t.Fatalf("symlinked shards: totalModelSize = %d, want %d", got, want)
	}
	if got := totalModelSize(filepath.Join(realDir, "big-00001-of-00003.gguf")); got != want {
		t.Fatalf("real shards: totalModelSize = %d, want %d", got, want)
	}
}

func TestShouldPromoteMoEPlacementIncludesSubpinSqueeze(t *testing.T) {
	current := &placement.Strategy{
		Type:     placement.MoEOffload,
		NCPUMoE:  32,
		OTString: `blk\.(0|1)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,exps=CPU`,
	}
	fewerCPULayers := &placement.Strategy{
		Type:     placement.MoEOffload,
		NCPUMoE:  31,
		OTString: current.OTString,
	}
	if !shouldPromoteMoEPlacement(current, fewerCPULayers) {
		t.Fatal("expected fewer CPU MoE layers to promote")
	}

	subpinSqueeze := &placement.Strategy{
		Type:     placement.MoEOffload,
		NCPUMoE:  32,
		OTString: current.OTString + `,blk\.(2)\.ffn_(gate_up|up_gate|gate|up)_exps.*=CUDA0`,
	}
	if !shouldPromoteMoEPlacement(current, subpinSqueeze) {
		t.Fatal("expected same-NCPUMoE subpin squeeze to promote")
	}

	same := *current
	if shouldPromoteMoEPlacement(current, &same) {
		t.Fatal("unchanged placement must not promote")
	}
}

// TestWaitForShutdownOrCrashDetectsProcessDeath guards the exact bug this
// function fixes: cmdLaunch's "Press Ctrl+C to stop" wait used to block only
// on the shutdown signal, so a backend that crashed on its own (a real CUDA
// OOM well after health check, reproduced 2026-07-08/09 on a long request)
// left the wrapper silently hung forever with no idea its child had died.
func TestWaitForShutdownOrCrashDetectsProcessDeath(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX short-lived process")
	}
	cmd := exec.Command("sh", "-c", "exit 0")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start fake process: %v", err)
	}
	// Production always reaps via a background cmd.Wait() (server.go's
	// StartWithTimeoutTo) — that's what populates Cmd.ProcessState, which
	// IsRunning() checks. Without it the child would sit as a zombie and
	// never look "not running", which would silently mask this test.
	go func() { _ = cmd.Wait() }()
	p := &server.Process{Cmd: cmd}
	sigCh := make(chan os.Signal, 1)

	done := make(chan bool, 1)
	go func() { done <- waitForShutdownOrCrash(p, sigCh) }()

	select {
	case crashed := <-done:
		if !crashed {
			t.Fatal("expected crashed=true when the process exits on its own")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("waitForShutdownOrCrash did not detect process death in time")
	}
}

func TestWaitForShutdownOrCrashRespondsToSignal(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX long-lived process")
	}
	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start fake process: %v", err)
	}
	defer func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	}()
	p := &server.Process{Cmd: cmd}
	sigCh := make(chan os.Signal, 1)

	done := make(chan bool, 1)
	go func() { done <- waitForShutdownOrCrash(p, sigCh) }()

	sigCh <- os.Interrupt
	select {
	case crashed := <-done:
		if crashed {
			t.Fatal("expected crashed=false when a shutdown signal arrives while the process is still running")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("waitForShutdownOrCrash did not respond to the signal in time")
	}
}
