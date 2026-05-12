package placement

import (
	"testing"

	"github.com/raketenkater/llm-server/pkg/detect"
)

func TestComputeDenseFits(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "model.gguf",
		SizeBytes:   15 * 1024 * 1024 * 1024, // 15GB
		NumLayers:   64,
		NumParams:   32_000_000_000,
		IsMoE:       false,
		ContextSize: 32768,
		HiddenSize:  4096,
	}
	opts := Options{KVPlacement: "auto", KVQuality: "mid"}
	strat, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.GPULayers == 0 {
		t.Fatalf("expected some layers on GPU, got %d", strat.GPULayers)
	}
	if strat.ContextSize == 0 {
		t.Fatalf("context size should not be zero")
	}
}

func TestComputeDenseTooLarge(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 8192}},
		RAM:  detect.RAMInfo{TotalMB: 32768},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path:        "model.gguf",
		SizeBytes:   40 * 1024 * 1024 * 1024, // 40GB
		NumLayers:   80,
		NumParams:   70_000_000_000,
		IsMoE:       false,
		ContextSize: 32768,
		HiddenSize:  8192,
	}
	opts := Options{KVPlacement: "auto", KVQuality: "mid"}
	strat, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	// 40GB model on 24GB GPU -> dense_cpu_offload
	if strat.Type != DenseCPUOffload {
		t.Fatalf("expected dense_cpu_offload strategy, got %s", strat.Type)
	}
	if strat.GPULayers != 999 {
		t.Fatalf("expected GPULayers=999, got %d", strat.GPULayers)
	}
}

func TestComputeMoE(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "moe.gguf",
		SizeBytes:   100 * 1024 * 1024 * 1024, // 100GB
		NumLayers:   64,
		NumParams:   200_000_000_000,
		IsMoE:       true,
		NumExperts:  64,
		ContextSize: 32768,
		HiddenSize:  6144,
	}
	opts := Options{KVPlacement: "auto", KVQuality: "mid"}
	strat, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.NCPUMoE == 0 {
		t.Fatalf("expected CPU experts for large MoE")
	}
	if strat.GPULayers == 0 && strat.NCPUMoE == 0 {
		t.Fatalf("expected some GPU layers or CPU experts for MoE")
	}
}

func TestComputeCPUOnly(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{},
		RAM:  detect.RAMInfo{TotalMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "model.gguf",
		SizeBytes:   10 * 1024 * 1024 * 1024,
		NumLayers:   32,
		NumParams:   8_000_000_000,
		IsMoE:       false,
		ContextSize: 32768,
		HiddenSize:  4096,
	}
	opts := Options{CPUMode: true}
	strat, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.GPULayers != 0 {
		t.Fatalf("expected CPU-only mode")
	}
}

func TestArgs(t *testing.T) {
	s := &Strategy{
		ContextSize:    4096,
		GPULayers:      32,
		KVQuality:      "mid",
		FlashAttention: true,
		Threads:        16,
		BatchSize:      2048,
		UBatchSize:     512,
	}
	args := s.Args("/models/test.gguf", 8081)
	if len(args) == 0 {
		t.Fatalf("args should not be empty")
	}
	joined := ""
	for _, a := range args {
		joined += a + " "
	}
	if !contains(args, "-m") {
		t.Fatalf("args missing -m")
	}
	if !contains(args, "/models/test.gguf") {
		t.Fatalf("args missing model path")
	}
	if !contains(args, "--port") {
		t.Fatalf("args missing --port")
	}
	if !contains(args, "--flash-attn") {
		t.Fatalf("args missing --flash-attn")
	}
}

func contains(slice []string, val string) bool {
	for _, v := range slice {
		if v == val {
			return true
		}
	}
	return false
}

func TestNormalizeSplit(t *testing.T) {
	split := normalizeSplit([]float64{12288, 12288})
	if len(split) != 2 {
		t.Fatalf("expected 2 values")
	}
	if split[0] != 0.5 || split[1] != 0.5 {
		t.Fatalf("expected equal split, got %v", split)
	}
}

func TestBuildOTString(t *testing.T) {
	gpus := []detect.GPU{
		{Index: 0},
		{Index: 1},
	}
	gpuOrder := []int{0, 1}

	// Single layer on GPU0
	ot := buildOTString([]int{1, 0}, gpus, gpuOrder)
	if ot != `blk\.(0)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,exps=CPU` {
		t.Fatalf("single-layer OT mismatch: %s", ot)
	}

	// Multiple layers on GPU0
	ot = buildOTString([]int{5, 0}, gpus, gpuOrder)
	expected := `blk\.(0|1|2|3|4)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,exps=CPU`
	if ot != expected {
		t.Fatalf("multi-layer OT mismatch:\n  got:      %s\n  expected: %s", ot, expected)
	}

	// Layers on both GPUs
	ot = buildOTString([]int{2, 3}, gpus, gpuOrder)
	expected = `blk\.(0|1)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,blk\.(2|3|4)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA1,exps=CPU`
	if ot != expected {
		t.Fatalf("two-gpu OT mismatch:\n  got:      %s\n  expected: %s", ot, expected)
	}
}

func TestDefaultContextSize(t *testing.T) {
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 65536},
	}
	model := &ModelProfile{
		NumLayers:   32,
		HiddenSize:  4096,
		ContextSize: 0,
	}
	ctx := defaultContextSize(model, caps)
	if ctx < 4096 {
		t.Fatalf("context too small: %d", ctx)
	}
}

func TestComputeDenseMultiGPU(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 12288},
			{Index: 1, VRAMTotalMB: 12288},
		},
		RAM: detect.RAMInfo{TotalMB: 65536},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "model.gguf",
		SizeBytes:   15 * 1024 * 1024 * 1024,
		NumLayers:   64,
		NumParams:   32_000_000_000,
		IsMoE:       false,
		ContextSize: 32768,
		HiddenSize:  4096,
	}
	strat, err := Compute(caps, model, Options{})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if len(strat.TensorSplit) != 2 {
		t.Fatalf("expected tensor split for multi-GPU, got %v", strat.TensorSplit)
	}
	if strat.SplitMode != "row" {
		t.Fatalf("expected split-mode row for mainline multi-GPU, got %s", strat.SplitMode)
	}
}

func TestComputeMoEMultiGPU(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576},
			{Index: 1, VRAMTotalMB: 24576},
		},
		RAM: detect.RAMInfo{TotalMB: 65536},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "moe.gguf",
		SizeBytes:   100 * 1024 * 1024 * 1024,
		NumLayers:   64,
		NumParams:   200_000_000_000,
		IsMoE:       true,
		NumExperts:  64,
		ContextSize: 32768,
		HiddenSize:  6144,
	}
	strat, err := Compute(caps, model, Options{})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	// MoE uses NCPUMoE for CPU expert offload
	if strat.NCPUMoE == 0 {
		t.Fatalf("expected MoE CPU expert offload")
	}
}

func TestArgsFull(t *testing.T) {
	s := &Strategy{
		ContextSize:    4096,
		GPULayers:      32,
		MainGPU:        0,
		TensorSplit:    []float64{0.5, 0.5},
		SplitMode:      "layer",
		KVPlacement:    "cpu",
		KVQuality:      "high",
		NCPUMoE:        64,
		FlashAttention: true,
		MMap:           false,
		MLock:          true,
		Threads:        16,
		BatchSize:      2048,
		UBatchSize:     512,
	}
	args := s.Args("/models/test.gguf", 8081)
	checks := map[string]bool{
		"-m": false, "--port": false, "--ctx-size": false, "-ngl": false,
		"--flash-attn": false,
	}
	for _, a := range args {
		if _, ok := checks[a]; ok {
			checks[a] = true
		}
	}
	for k, v := range checks {
		if !v {
			t.Fatalf("args missing %s", k)
		}
	}
}
