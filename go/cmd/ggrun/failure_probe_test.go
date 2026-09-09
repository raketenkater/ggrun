package main

import (
	"os"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestRecordFailedLaunchProbesKeepsComputeButNoHealthyEvidence(t *testing.T) {
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	model := &placement.ModelProfile{Path: "model.gguf", Basename: "model", IsMoE: true}
	strategy := &placement.Strategy{ContextSize: 32768, UBatchSize: 256, KVQuality: "high", KVPlacement: "gpu", KVType: "f16", Parallel: 1}
	be := &backendInfo{Tag: "llama", Identity: "failure-probe-test"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0}, {Index: 1}}}
	tag := scopedProbeBackendTagForStrategy(nil, model, be, strategy)
	if err := placement.RecordRuntimeGraphGrowthFromOOM(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, tag, caps.GPUs, strategy.Parallel, 1, 777, false); err != nil {
		t.Fatalf("seed OOM evidence: %v", err)
	}
	log := "llama_context: model loaded\n" +
		"sched_reserve: CUDA0 compute buffer size = 1234.00 MiB\n" +
		"sched_reserve: CUDA1 compute buffer size = 2345.00 MiB\n" +
		"CUDA error: out of memory\ncurrent device: 1, in function ggml_cuda_graph_evaluate_and_capture\n"

	got := recordFailedLaunchProbes(nil, cfg, model, strategy, be, caps, log)
	if got[0] != 1234 || got[1] != 2345 {
		t.Fatalf("failed launch lost parsed compute buffers: %#v", got)
	}
	if growth := placement.RuntimeGraphGrowthByGPU(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, tag, caps.GPUs, strategy.Parallel); growth[1] != 777 {
		t.Fatalf("failed launch changed existing OOM reserve: %#v", growth)
	}
	entries, err := os.ReadDir(cfg.CacheDir)
	if err != nil || len(entries) == 0 {
		t.Fatalf("probe cache was not written: %v", err)
	}
	var probe string
	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), ".probe") {
			data, readErr := os.ReadFile(cfg.CacheDir + "/" + entry.Name())
			if readErr != nil {
				t.Fatal(readErr)
			}
			probe = string(data)
		}
	}
	if !strings.Contains(probe, "PROBED_COMPUTE_BUF_MB_CUDA0=1234") || !strings.Contains(probe, "PROBED_COMPUTE_BUF_MB_CUDA1=2345") {
		t.Fatalf("failed launch did not persist its narrow compute evidence: %s", probe)
	}
	if !strings.Contains(probe, "PROBED_RUNTIME_GRAPH_GROWTH_FROM_OOM_CUDA1=1") || !strings.Contains(probe, "PROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA1=777") {
		t.Fatalf("existing OOM source/reserve was not preserved: %s", probe)
	}
	if _, ok := placement.LoadMeasuredAllocation(cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality, strategy.KVPlacement, tag, caps.GPUs, strategy.Parallel); ok {
		t.Fatal("failed launch created a complete healthy allocation record")
	}
}
