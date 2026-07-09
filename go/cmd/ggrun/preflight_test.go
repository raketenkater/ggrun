package main

import (
	"reflect"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

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
