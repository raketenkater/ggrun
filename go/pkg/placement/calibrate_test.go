package placement

import (
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestCalibrationCandidatesSingleGPUIsNoOp(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}}}
	model := &ModelProfile{Path: "m.gguf", IsMoE: true}
	base := &Strategy{Type: MoEOffload, KVPlacement: "cpu"}
	got := CalibrationCandidates(caps, model, base, Options{})
	if len(got) != 1 || got[0].Name != "default" {
		t.Fatalf("single GPU must not produce alternatives, got %+v", got)
	}
}

func TestCalibrationCandidatesCPUOnlyIsNoOp(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 12288},
	}}
	base := &Strategy{Type: CPUOnly}
	got := CalibrationCandidates(caps, &ModelProfile{Path: "m.gguf"}, base, Options{})
	if len(got) != 1 {
		t.Fatalf("CPU-only must not produce alternatives, got %+v", got)
	}
}

func TestCalibrationCandidatesMoEAddsKVAlternate(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 32000},
		{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 8000},
	}}
	model := &ModelProfile{
		Path: "m.gguf", IsMoE: true, TotalSizeMB: 64 * 1024, NumLayers: 60, NumExperts: 128,
		ExpertBytes: 56 * 1024 * 1024 * 1024, NonExpertBytes: 8 * 1024 * 1024 * 1024,
	}
	caps.RAM = detect.RAMInfo{TotalMB: 131072, FreeMB: 131072}
	caps.CPU = detect.CPUInfo{Cores: 16}
	base := &Strategy{Type: MoEOffload, KVPlacement: "cpu", NCPUMoE: 40, OTString: "blk.ffn=CPU"}
	got := CalibrationCandidates(caps, model, base, Options{ContextSize: 8192, KVPlacement: "cpu"})
	if len(got) != 2 {
		t.Fatalf("MoE multi-GPU should offer a KV alternate, got %d candidates", len(got))
	}
	if got[0].Name != "default" || got[1].Name != "kv-alternate" {
		t.Fatalf("unexpected candidate names: %+v", got)
	}
	alt := got[1].Strategy
	if alt.KVPlacement != "gpu" {
		t.Fatalf("KV alternate should flip cpu->gpu, got %q", alt.KVPlacement)
	}
	// The alternate must not alias the base's expert split.
	if alt == base {
		t.Fatal("candidate aliases the base strategy")
	}
	if base.KVPlacement != "cpu" {
		t.Fatalf("base mutated to %q", base.KVPlacement)
	}
}

func TestCalibrationCandidatesMoEKVGPUFlipsToCPU(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 12288},
	}}
	caps.RAM = detect.RAMInfo{TotalMB: 131072, FreeMB: 131072}
	caps.CPU = detect.CPUInfo{Cores: 16}
	base := &Strategy{Type: MoEOffload, KVPlacement: "gpu"}
	model := &ModelProfile{
		Path: "m.gguf", IsMoE: true, TotalSizeMB: 64 * 1024, NumLayers: 60, NumExperts: 128,
		ExpertBytes: 56 * 1024 * 1024 * 1024, NonExpertBytes: 8 * 1024 * 1024 * 1024,
	}
	got := CalibrationCandidates(caps, model, base, Options{ContextSize: 8192, KVPlacement: "gpu"})
	if len(got) != 2 || got[1].Strategy.KVPlacement != "cpu" {
		t.Fatalf("KV=gpu should alternate to cpu, got %+v", got)
	}
}

func TestCalibrationCandidatesDenseSplitInversion(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 32000},
		{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 8000},
	}}
	base := &Strategy{Type: MultiGPUDense, TensorSplit: []float64{0.75, 0.25}, MainGPU: 0}
	got := CalibrationCandidates(caps, &ModelProfile{Path: "m.gguf"}, base, Options{})
	if len(got) != 2 || got[1].Name != "split-inverted" {
		t.Fatalf("dense multi-GPU should offer a split inversion, got %+v", got)
	}
	inv := got[1].Strategy
	if inv.TensorSplit[0] != 0.25 || inv.TensorSplit[1] != 0.75 {
		t.Fatalf("split not inverted: %v", inv.TensorSplit)
	}
	if inv.MainGPU != 1 {
		t.Fatalf("main GPU should follow the larger share, got %d", inv.MainGPU)
	}
	// Base untouched.
	if base.TensorSplit[0] != 0.75 || base.MainGPU != 0 {
		t.Fatal("base strategy mutated by inversion")
	}
}

func TestCalibrationCandidatesDenseSymmetricSplitSkipped(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 24576},
	}}
	base := &Strategy{Type: MultiGPUDense, TensorSplit: []float64{0.5, 0.5}, MainGPU: 0}
	got := CalibrationCandidates(caps, &ModelProfile{Path: "m.gguf"}, base, Options{})
	if len(got) != 1 {
		t.Fatalf("a symmetric split inverts to itself and must be skipped, got %+v", got)
	}
}

func TestCalibrationDecisionRoundTrip(t *testing.T) {
	dir := t.TempDir()
	key := CalibrationScopeKey{
		ModelIdentity: "m", BackendIdentity: "b", HardwareID: "h",
		WorkloadProfile: "claude-agent-parallel-v1", ContextSize: 131072,
		Parallel: 4, UBatchSize: 256, KVQuality: "mid",
	}.String()
	d := CalibrationDecision{
		ScopeKey: key, Winner: "kv-alternate",
		DefaultTPS: 20.5, WinnerTPS: 24.1, Improvement: 17.5,
	}
	if _, err := SaveCalibrationDecision(dir, d); err != nil {
		t.Fatalf("save: %v", err)
	}
	loaded, err := LoadCalibrationDecision(dir, key)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.Winner != "kv-alternate" || loaded.WinnerTPS != 24.1 {
		t.Fatalf("round trip lost decision: %+v", loaded)
	}
	// A different scope key must not read this decision.
	if _, err := LoadCalibrationDecision(dir, "other-scope"); err == nil {
		t.Fatal("stale/foreign scope must not load a decision")
	}
	// The file lives under the calibration namespace.
	if filepath.Dir(CalibrationPath(dir, key)) != filepath.Join(dir, "calibration") {
		t.Fatalf("unexpected calibration path %q", CalibrationPath(dir, key))
	}
}
