package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/raketenkater/llm-server/pkg/detect"
)

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
	if req.SpecMode != "ngram" || req.MMProjPath != "/models/mmproj.gguf" || req.RamBudgetMB != 48*1024 {
		t.Fatalf("advanced flags mismatch: %#v", req)
	}
	if len(req.ExtraArgs) != 1 || req.ExtraArgs[0] != "--no-mmap" {
		t.Fatalf("extra args mismatch: %v", req.ExtraArgs)
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
	appHome := filepath.Join(t.TempDir(), "llm-server")
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
