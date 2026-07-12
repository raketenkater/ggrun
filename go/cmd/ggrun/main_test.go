package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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

func TestClaudeCodeParallelIsFeaturePolicyForDeepseek4(t *testing.T) {
	req := &launchRequest{ClaudeCode: true}
	model := &placement.ModelProfile{ModelArch: "deepseek4", CTXTrain: 1048576}
	be := &backendInfo{Tag: "llama"}
	opts := placementOptionsFromRequest(req, model, be, t.TempDir())
	if opts.Parallel != 4 {
		t.Fatalf("claude-code should layer four slots over the shared mainline placement, got %d", opts.Parallel)
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
		t.Fatalf("claude-code on other models keeps 4 slots, got %d", opts.Parallel)
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

func TestStartupLogCUDAOOM(t *testing.T) {
	log := "loading\n" +
		"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2206.07 MiB on device 0: cudaMalloc failed: out of memory\n" +
		"segmentation fault"
	device, allocMB, ok := startupLogCUDAOOM(log)
	if !ok || device != 0 || allocMB != 2207 {
		t.Fatalf("cuda oom parse = device %d alloc %d ok %v", device, allocMB, ok)
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

	measured := recordMeasuredLaunchProbes(cfg, model, strategy, be, caps, log, nil)
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
		"--gpus=1,3", "--host=127.0.0.1", "--spec=draft", "--parallel=4", "model.gguf",
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
	t.Setenv("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "")
	env := claudeCodeEnv("0.0.0.0", 8081, []string{"llama-server", "--ctx-size", "1048576", "--parallel", "4"})

	if envHasPrefix(env, "ANTHROPIC_API_KEY=") {
		t.Fatalf("claude-code env must drop real ANTHROPIC_API_KEY: %v", env)
	}
	for _, want := range []string{
		"ANTHROPIC_BASE_URL=http://127.0.0.1:8081",
		"API_TIMEOUT_MS=14400000",
		"API_FORCE_IDLE_TIMEOUT=0",
	} {
		if !envContains(env, want) {
			t.Fatalf("missing %s in claude-code env: %v", want, env)
		}
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
			claudeCodeSlotAdjust(s, tc.claudeCode, tc.explicit)
			if s.Parallel != tc.wantParallel {
				t.Fatalf("ctx=%d par=%d cc=%v: got parallel %d, want %d", tc.ctx, tc.par, tc.claudeCode, s.Parallel, tc.wantParallel)
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
