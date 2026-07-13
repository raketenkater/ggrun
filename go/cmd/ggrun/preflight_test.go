package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestFindFitParamsDoesNotCrossCustomForksViaPATH(t *testing.T) {
	dir := t.TempDir()
	pathBin := filepath.Join(dir, "path-bin")
	if err := os.MkdirAll(pathBin, 0o755); err != nil {
		t.Fatal(err)
	}
	fit := filepath.Join(pathBin, "llama-fit-params")
	if err := os.WriteFile(fit, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", pathBin)
	customServer := filepath.Join(dir, "custom-fork", "bin", "llama-server")
	if got := findFitParamsBin(customServer); got != "" {
		t.Fatalf("custom fork must not use unrelated PATH fit-params: %s", got)
	}
}

func TestPreflightArgsKeepsOnlyMemoryShapingFlags(t *testing.T) {
	serverArgs := []string{
		"-m", "model.gguf",
		"--host", "0.0.0.0", "--port", "8081",
		"--ctx-size", "1048576",
		"--flash-attn", "on",
		"-b", "2048", "-ub", "512",
		"--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
		"--jinja",
		"--threads", "8", "--threads-batch", "8",
		"--parallel", "4",
		"-ngl", "999",
		"--fit", "off",
		"--tensor-split", "0.86,0.03,0.11",
		"--split-mode", "layer",
		"-ot", `blk\.(0|1)\.ffn_.*=CUDA0,exps=CPU`,
		"--n-cpu-moe", "36",
		"--no-mmap",
		"--alias", "local",
		"--presence-penalty", "1.0",
	}
	want := []string{
		"--fit-print", "on",
		"-m", "model.gguf",
		"--ctx-size", "1048576",
		"--flash-attn", "on",
		"-b", "2048", "-ub", "512",
		"--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
		"--parallel", "4",
		"-ngl", "999",
		"--tensor-split", "0.86,0.03,0.11",
		"--split-mode", "layer",
		"-ot", `blk\.(0|1)\.ffn_.*=CUDA0,exps=CPU`,
		"--n-cpu-moe", "36",
	}
	if got := preflightArgs(serverArgs); !reflect.DeepEqual(got, want) {
		t.Fatalf("preflightArgs:\n got  %q\n want %q", got, want)
	}
}

func TestDraftPreflightServerArgs(t *testing.T) {
	strategy := &placement.Strategy{
		BackendTag: "llama", ContextSize: 1048576, BatchSize: 2048, UBatchSize: 256,
		Parallel: 4, FlashAttention: true,
		Draft: &placement.DraftConfig{
			Type: placement.DraftDFlash, Path: "dspark.gguf", DraftGPU: 2,
			CTXSizeDraft: 1048576, KVTypeDraft: "q8_0", GPULayersDraft: "all",
		},
	}
	want := []string{
		"llama-fit-draft", "-m", "dspark.gguf", "-c", "1048576",
		"-b", "2048", "-ub", "256", "-ctk", "q8_0", "-ctv", "q8_0",
		"-np", "4", "-ngl", "all", "--device", "CUDA2", "--flash-attn", "on",
	}
	if got := draftPreflightServerArgs(strategy); !reflect.DeepEqual(got, want) {
		t.Fatalf("draft preflight args:\n got  %q\n want %q", got, want)
	}
}

func TestMergePreflightDevicesAddsCompanionMemory(t *testing.T) {
	target := []preflightDevice{
		{Name: "CUDA0", ModelMB: 15000, ContextMB: 3000, ComputeMB: 2000},
		{Name: "CUDA1", ModelMB: 9000, ContextMB: 200, ComputeMB: 600},
		{Name: "Host", ModelMB: 110000, ComputeMB: 20},
	}
	draft := []preflightDevice{
		{Name: "CUDA1", ModelMB: 11000, ContextMB: 400, ComputeMB: 900},
		{Name: "Host", ModelMB: 50, ContextMB: 10, ComputeMB: 5},
	}
	got := mergePreflightDevices(target, draft)
	want := []preflightDevice{
		{Name: "CUDA0", ModelMB: 15000, ContextMB: 3000, ComputeMB: 2000},
		{Name: "CUDA1", ModelMB: 20000, ContextMB: 600, ComputeMB: 1500},
		{Name: "Host", ModelMB: 110050, ContextMB: 10, ComputeMB: 25},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("merged preflight rows:\n got  %#v\n want %#v", got, want)
	}
}

func TestEmbeddedMTPPreflightReservationIsConservativePerGPU(t *testing.T) {
	model := &placement.ModelProfile{
		NumLayers: 33, NextNPredictLayers: 1, HasSSM: 1, FullAttnInterval: 4,
		HeadCountKV: 4, KeyLength: 256, ValueLength: 256,
	}
	strategy := &placement.Strategy{
		ContextSize: 262144, KVPlacement: "gpu",
		Draft: &placement.DraftConfig{Type: placement.DraftMTP, SpecType: "draft-mtp"},
	}
	target := []preflightDevice{
		{Name: "CUDA0", ComputeMB: 1600},
		{Name: "CUDA1", ComputeMB: 600},
		{Name: "Host", ComputeMB: 10},
	}
	got, err := embeddedMTPPreflightReservation(model, strategy, target)
	if err != nil {
		t.Fatal(err)
	}
	want := []preflightDevice{
		{Name: "CUDA0", ContextMB: 1024, ComputeMB: 1600},
		{Name: "CUDA1", ContextMB: 1024, ComputeMB: 1024},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("embedded MTP reservation:\n got  %#v\n want %#v", got, want)
	}
}

func TestEmbeddedMTPPreflightRejectsUnprovenCPUKV(t *testing.T) {
	model := &placement.ModelProfile{NextNPredictLayers: 1, HeadCountKV: 4, KeyLength: 128, ValueLength: 128}
	strategy := &placement.Strategy{
		ContextSize: 32768, KVPlacement: "cpu",
		Draft: &placement.DraftConfig{Type: placement.DraftMTP, SpecType: "draft-mtp"},
	}
	if _, err := embeddedMTPPreflightReservation(model, strategy, []preflightDevice{{Name: "CUDA0", ComputeMB: 1000}}); err == nil {
		t.Fatal("embedded MTP with unmeasured CPU KV must fail closed")
	}
}

func TestPreflightWorstDeficit(t *testing.T) {
	// Real shape from the 2026-07-07 DeepSeek-V4 launch: 3090Ti + 3060 + 4070,
	// fit-print rows in MiB (model, context, compute).
	devs := []preflightDevice{
		{Name: "CUDA0", ModelMB: 15648, ContextMB: 3238, ComputeMB: 2184},
		{Name: "CUDA1", ModelMB: 9070, ContextMB: 179, ComputeMB: 599},
		{Name: "CUDA2", ModelMB: 10248, ContextMB: 351, ComputeMB: 599},
		{Name: "Host", ModelMB: 114162, ContextMB: 0, ComputeMB: 17}, // ignored
	}
	gpus := []detect.GPU{
		{Index: 0, VRAMTotalMB: 24564},
		{Index: 1, VRAMTotalMB: 12288},
		{Index: 2, VRAMTotalMB: 12282},
	}

	// With the measured 678 MB overhead everything fits (CUDA2 has ~406 MiB slack).
	dev, deficit, _ := preflightWorstDeficit(devs, gpus, map[int]int{0: 678, 1: 678, 2: 678}, nil)
	if dev != -1 || deficit != 0 {
		t.Fatalf("expected fit, got device %d deficit %d", dev, deficit)
	}

	// Occupy 1 GB on the 4070: CUDA2 must be reported with the exact overshoot.
	gpus[2].VRAMUsedMB = 1024
	dev, deficit, summary := preflightWorstDeficit(devs, gpus, map[int]int{0: 678, 1: 678, 2: 678}, nil)
	if dev != 2 {
		t.Fatalf("expected CUDA2 deficit, got device %d (summary %s)", dev, summary)
	}
	want := (10248 + 351 + 599 + 678) - (12282 - 1024)
	if deficit != want {
		t.Fatalf("deficit = %d, want %d", deficit, want)
	}
}

func TestPreflightContextTotalIncludesHostAndGPU(t *testing.T) {
	devs := []preflightDevice{
		{Name: "CUDA0", ContextMB: 6252},
		{Name: "CUDA1", ContextMB: 0},
		{Name: "CUDA2", ContextMB: 649},
		{Name: "Host", ContextMB: 31},
		{Name: "ignored-negative", ContextMB: -10},
	}
	if got := preflightContextTotalMB(devs); got != 6932 {
		t.Fatalf("total context = %d MiB, want 6932", got)
	}
}

func TestPreflightWorstDeficitIncludesMeasuredRuntimeGrowth(t *testing.T) {
	devs := []preflightDevice{
		{Name: "CUDA2", ModelMB: 10248, ContextMB: 351, ComputeMB: 599},
	}
	gpus := []detect.GPU{{Index: 2, VRAMTotalMB: 12282}}

	dev, deficit, summary := preflightWorstDeficit(devs, gpus, map[int]int{2: 678}, map[int]int{2: 1000})
	if dev != 2 {
		t.Fatalf("expected CUDA2 deficit, got device %d (summary %s)", dev, summary)
	}
	want := (10248 + 351 + 599 + 678 + 1000) - 12282
	if deficit != want {
		t.Fatalf("deficit = %d, want %d", deficit, want)
	}
	if !strings.Contains(summary, "fit=11198 overhead=678 runtime=1000") {
		t.Fatalf("summary missing exact terms: %s", summary)
	}
}

func TestPreflightWorstDeficitIgnoresUnknownDevices(t *testing.T) {
	devs := []preflightDevice{
		{Name: "CUDA5", ModelMB: 99999},
		{Name: "Vulkan0", ModelMB: 99999},
	}
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 8192}}
	if dev, deficit, _ := preflightWorstDeficit(devs, gpus, map[int]int{0: 600}, nil); dev != -1 || deficit != 0 {
		t.Fatalf("unknown devices must not produce deficits, got dev %d deficit %d", dev, deficit)
	}
}
