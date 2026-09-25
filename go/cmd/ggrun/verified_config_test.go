package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestSaveVerifiedConfigWritesScopedRecord(t *testing.T) {
	cacheDir := t.TempDir()
	cfg := &config.Config{CacheDir: cacheDir}
	model := &placement.ModelProfile{Path: "/models/target.gguf", SizeBytes: 1 << 30, TotalSizeMB: 1024, ModelArch: "llama", NumLayers: 32}
	backend := &backendInfo{Tag: "llama", Identity: "backend-build", Path: "/usr/bin/llama-server"}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "3090", VRAMTotalMB: 24576, VRAMUsedMB: 0, BandwidthMBps: 30000}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	strategy := &placement.Strategy{
		Type: placement.SingleGPU, ContextSize: 8192, MainGPU: 0,
		KVPlacement: "gpu", KVType: "q8_0", BatchSize: 2048, UBatchSize: 512,
		Parallel: 1, MMap: true, FlashAttention: true, BackendTag: "llama",
		Threads: 8, ThreadsBatch: 8,
	}
	req := &launchRequest{CtxFlag: "8192", KVQuality: "mid", KVPlacement: "gpu", Parallel: 1}

	saveVerifiedConfigForLaunch(cfg, req, model, backend, caps, strategy)

	key := verifiedConfigScopeKey(req, model, backend, caps)
	if key == "" {
		t.Fatal("expected a non-empty verified config scope key")
	}
	path := placement.VerifiedConfigPath(cacheDir, key)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("verified config not saved at %s: %v", path, err)
	}
	loaded, err := placement.LoadVerifiedConfig(cacheDir, key)
	if err != nil {
		t.Fatalf("load saved verified config: %v", err)
	}
	if loaded.StrategyType != placement.SingleGPU || loaded.MainGPU != 0 || loaded.ContextSize != 8192 {
		t.Fatalf("saved config lost strategy: %+v", loaded)
	}
	if loaded.BackendIdentity != "backend-build" || loaded.BackendPath != "/usr/bin/llama-server" {
		t.Fatalf("saved config lost backend provenance: %+v", loaded)
	}
}

func TestSaveVerifiedConfigSkipsNoCachedConfig(t *testing.T) {
	cacheDir := t.TempDir()
	cfg := &config.Config{CacheDir: cacheDir}
	model := &placement.ModelProfile{Path: "/models/target.gguf", SizeBytes: 1 << 30, TotalSizeMB: 1024, ModelArch: "llama", NumLayers: 32}
	backend := &backendInfo{Tag: "llama", Identity: "backend-build", Path: "/usr/bin/llama-server"}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "3090", VRAMTotalMB: 24576, VRAMUsedMB: 0, BandwidthMBps: 30000}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	strategy := &placement.Strategy{Type: placement.SingleGPU, ContextSize: 8192, MainGPU: 0, BatchSize: 2048, UBatchSize: 512, Parallel: 1, MMap: true, FlashAttention: true}
	req := &launchRequest{CtxFlag: "8192", KVQuality: "mid", KVPlacement: "gpu", Parallel: 1, NoCachedConfig: true}

	saveVerifiedConfigForLaunch(cfg, req, model, backend, caps, strategy)

	key := verifiedConfigScopeKey(req, model, backend, caps)
	if key != "" {
		t.Fatalf("--no-cached-config must not produce a verified-config scope key, got %q", key)
	}
	if entries, _ := filepath.Glob(filepath.Join(cacheDir, "verified-configs", "verified-*.json")); len(entries) != 0 {
		t.Fatalf("--no-cached-config must not write a verified config, found %v", entries)
	}
}

func TestSaveVerifiedConfigSkipsScreenedCalibrationWinner(t *testing.T) {
	cacheDir := t.TempDir()
	cfg := &config.Config{CacheDir: cacheDir}
	model := &placement.ModelProfile{Path: "/models/target.gguf", SizeBytes: 1 << 30, TotalSizeMB: 1024, ModelArch: "llama", NumLayers: 32}
	backend := &backendInfo{Tag: "llama", Identity: "backend-build", Path: "/usr/bin/llama-server"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "3090", VRAMTotalMB: 24576}}, RAM: detect.RAMInfo{TotalMB: 65536}, CPU: detect.CPUInfo{Cores: 8}}
	strategy := &placement.Strategy{Type: placement.SingleGPU, ContextSize: 8192, MainGPU: 0, BatchSize: 256, UBatchSize: 256, Parallel: 2}
	req := &launchRequest{CtxFlag: "8192", KVQuality: "mid", KVPlacement: "gpu", Parallel: 2, CalibrationScreened: true}

	saveVerifiedConfigForLaunch(cfg, req, model, backend, caps, strategy)

	if entries, _ := filepath.Glob(filepath.Join(cacheDir, "verified-configs", "verified-*.json")); len(entries) != 0 {
		t.Fatalf("screened calibration winner leaked into automatic verified configs: %v", entries)
	}
}

func TestSaveVerifiedConfigSkipsPendingOptimizerDecision(t *testing.T) {
	cacheDir := t.TempDir()
	cfg := &config.Config{CacheDir: cacheDir}
	model := &placement.ModelProfile{Path: "/models/target.gguf", SizeBytes: 1 << 30, TotalSizeMB: 1024, ModelArch: "llama", NumLayers: 32}
	backend := &backendInfo{Tag: "llama", Identity: "backend-build", Path: "/usr/bin/llama-server"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "3090", VRAMTotalMB: 24576}}, RAM: detect.RAMInfo{TotalMB: 65536}, CPU: detect.CPUInfo{Cores: 8}}
	strategy := &placement.Strategy{Type: placement.SingleGPU, ContextSize: 8192, MainGPU: 0, BatchSize: 256, UBatchSize: 256, Parallel: 2, PerformanceTuned: true}
	req := &launchRequest{CtxFlag: "8192", KVQuality: "mid", KVPlacement: "gpu", Parallel: 2, CalibrationPending: true}

	saveVerifiedConfigForLaunch(cfg, req, model, backend, caps, strategy)
	if entries, _ := filepath.Glob(filepath.Join(cacheDir, "verified-configs", "verified-*.json")); len(entries) != 0 {
		t.Fatalf("pending optimizer winner leaked into verified configs: %v", entries)
	}
	req.CalibrationPending = false
	saveVerifiedConfigForLaunch(cfg, req, model, backend, caps, strategy)
	if entries, _ := filepath.Glob(filepath.Join(cacheDir, "verified-configs", "verified-*.json")); len(entries) != 1 {
		t.Fatalf("promoted optimizer winner was not persisted: %v", entries)
	}
}

func TestInvalidateRuntimeOOMDeletesVerifiedConfig(t *testing.T) {
	cacheDir := t.TempDir()
	cfg := &config.Config{CacheDir: cacheDir}
	model := &placement.ModelProfile{Path: "/models/target.gguf", SizeBytes: 1 << 30, TotalSizeMB: 1024, ModelArch: "llama", NumLayers: 32}
	backend := &backendInfo{Tag: "llama", Identity: "backend-build", Path: "/usr/bin/llama-server"}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "3090", VRAMTotalMB: 24576, VRAMUsedMB: 0, BandwidthMBps: 30000}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	strategy := &placement.Strategy{Type: placement.SingleGPU, ContextSize: 8192, MainGPU: 0, BatchSize: 2048, UBatchSize: 512, Parallel: 1, MMap: true, FlashAttention: true}
	req := &launchRequest{CtxFlag: "8192", KVQuality: "mid", KVPlacement: "gpu", Parallel: 1}

	saveVerifiedConfigForLaunch(cfg, req, model, backend, caps, strategy)
	key := verifiedConfigScopeKey(req, model, backend, caps)
	if key == "" {
		t.Fatal("expected a scope key")
	}
	if _, err := os.Stat(placement.VerifiedConfigPath(cacheDir, key)); err != nil {
		t.Fatalf("verified config should exist before invalidation: %v", err)
	}

	args := []string{"llama-server", "-m", model.Path, "--ctx-size", "8192"}
	if err := invalidateRuntimeOOMLaunch(req, cfg, model, backend, caps, strategy, args, "CUDA OOM after health"); err != nil {
		t.Fatalf("invalidate: %v", err)
	}
	if _, err := os.Stat(placement.VerifiedConfigPath(cacheDir, key)); !os.IsNotExist(err) {
		t.Fatal("runtime OOM must delete the verified config record")
	}
}

func TestVerifiedConfigScopeKeyChangesWithRequest(t *testing.T) {
	model := &placement.ModelProfile{Path: "/models/target.gguf", SizeBytes: 1 << 30, TotalSizeMB: 1024, ModelArch: "llama", NumLayers: 32}
	backend := &backendInfo{Tag: "llama", Identity: "backend-build", Path: "/usr/bin/llama-server"}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "3090", VRAMTotalMB: 24576, VRAMUsedMB: 0, BandwidthMBps: 30000}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	base := &launchRequest{CtxFlag: "8192", KVQuality: "mid", KVPlacement: "gpu", Parallel: 1}
	key := verifiedConfigScopeKey(base, model, backend, caps)

	// Same shape → same key (a reuse hit).
	if again := verifiedConfigScopeKey(base, model, backend, caps); again != key {
		t.Fatalf("same scope must produce the same key: %s != %s", again, key)
	}
	// Different context → different key (clean miss).
	diffCtx := &launchRequest{CtxFlag: "16384", KVQuality: "mid", KVPlacement: "gpu", Parallel: 1}
	if got := verifiedConfigScopeKey(diffCtx, model, backend, caps); got == key {
		t.Fatal("a different context must produce a different verified-config scope key")
	}
	// Different chat template → different key (clean miss).
	diffTemplate := &launchRequest{CtxFlag: "8192", KVQuality: "mid", KVPlacement: "gpu", Parallel: 1, ChatTemplateOverride: "qwen3.8"}
	if got := verifiedConfigScopeKey(diffTemplate, model, backend, caps); got == key {
		t.Fatal("a forced chat template must produce a different verified-config scope key")
	}
	// Different backend → different key (clean miss).
	diffBackend := &backendInfo{Tag: "ik_llama", Identity: "ik-build", Path: "/usr/bin/ik_llama"}
	if got := verifiedConfigScopeKey(base, model, diffBackend, caps); got == key {
		t.Fatal("a different backend must produce a different verified-config scope key")
	}
	diffKV := &launchRequest{CtxFlag: "8192", KVQuality: "q8_0", KVPlacement: "gpu", Parallel: 1}
	if got := verifiedConfigScopeKey(diffKV, model, backend, caps); got == key {
		t.Fatal("a different KV quality must produce a different verified-config scope key")
	}
	diffSWA := &launchRequest{CtxFlag: "8192", KVQuality: "mid", KVPlacement: "gpu", Parallel: 1, ExtraArgs: []string{"--swa-full"}}
	if got := verifiedConfigScopeKey(diffSWA, model, backend, caps); got == key {
		t.Fatal("full SWA must produce a different verified-config scope key")
	}
	moved := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "3090", VRAMTotalMB: 24576, PCIBusID: "0000:65:00.0", PCIGen: 3, PCILanes: 16}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	movedReq := &launchRequest{CtxFlag: "8192", KVQuality: "mid", KVPlacement: "gpu", Parallel: 1}
	if got := verifiedConfigScopeKey(movedReq, model, backend, moved); got == key {
		t.Fatal("a different physical device must produce a different verified-config scope key")
	}
}
