package placement

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
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
	// 40GB model on 8GB GPU with 128GB RAM -> dense_cpu_offload
	// Total system memory must exceed model overhead (40GB * 130% = 52GB)
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 8192}},
		RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
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
	if strat.Type != DenseCPUOffload {
		t.Fatalf("expected dense_cpu_offload strategy, got %s", strat.Type)
	}
	if strat.GPULayers != 999 {
		t.Fatalf("expected GPULayers=999, got %d", strat.GPULayers)
	}
}

func TestComputeMoE(t *testing.T) {
	// 40GB MoE on 24GB GPU with 128GB RAM
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "moe.gguf",
		SizeBytes:   40 * 1024 * 1024 * 1024, // 40GB
		NumLayers:   64,
		NumParams:   70_000_000_000,
		IsMoE:       true,
		NumExperts:  64,
		ContextSize: 32768,
		HiddenSize:  4096,
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
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
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
	host := ""
	for i, arg := range args {
		if arg == "--host" && i+1 < len(args) {
			host = args[i+1]
		}
	}
	if host != "127.0.0.1" {
		t.Fatalf("expected loopback host, got %q in args %v", host, args)
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

var otDevicePattern = regexp.MustCompile(`blk\\\.\(([^)]*)\).*=(?:CUDA|Vulkan)(\d+)`)

func parseOTLayersByDevice(t *testing.T, ot string) map[int][]int {
	t.Helper()
	out := map[int][]int{}
	for _, part := range strings.Split(ot, ",") {
		m := otDevicePattern.FindStringSubmatch(part)
		if m == nil {
			continue
		}
		device, err := strconv.Atoi(m[2])
		if err != nil {
			t.Fatalf("parse device from %q: %v", part, err)
		}
		for _, raw := range strings.Split(m[1], "|") {
			layer, err := strconv.Atoi(raw)
			if err != nil {
				t.Fatalf("parse layer from %q: %v", part, err)
			}
			out[device] = append(out[device], layer)
		}
	}
	return out
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
	ot := buildOTString([]int{1, 0}, gpus, gpuOrder, "")
	if ot != `blk\.(0)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,exps=CPU` {
		t.Fatalf("single-layer OT mismatch: %s", ot)
	}

	// Multiple layers on GPU0
	ot = buildOTString([]int{5, 0}, gpus, gpuOrder, "")
	expected := `blk\.(0|1|2|3|4)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,exps=CPU`
	if ot != expected {
		t.Fatalf("multi-layer OT mismatch:\n  got:      %s\n  expected: %s", ot, expected)
	}

	// Layers on both GPUs
	ot = buildOTString([]int{2, 3}, gpus, gpuOrder, "")
	expected = `blk\.(0|1)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,blk\.(2|3|4)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA1,exps=CPU`
	if ot != expected {
		t.Fatalf("two-gpu OT mismatch:\n  got:      %s\n  expected: %s", ot, expected)
	}

	// Vulkan uses Vulkan device names in override tensors.
	ot = buildOTString([]int{1, 0}, gpus, gpuOrder, "vulkan")
	if ot != `blk\.(0)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=Vulkan0,exps=CPU` {
		t.Fatalf("vulkan OT mismatch: %s", ot)
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
	// 60GB MoE on two 24GB GPUs with 128GB RAM
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576},
			{Index: 1, VRAMTotalMB: 24576},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "moe.gguf",
		SizeBytes:   60 * 1024 * 1024 * 1024,
		NumLayers:   64,
		NumParams:   120_000_000_000,
		IsMoE:       true,
		NumExperts:  64,
		ContextSize: 32768,
		HiddenSize:  4096,
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

func TestComputeMoEHeterogeneousMultiGPUExactLedger(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24576, VRAMUsedMB: 822, BandwidthMBps: 31504},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, VRAMUsedMB: 574, BandwidthMBps: 12000},
			{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12288, VRAMUsedMB: 660, BandwidthMBps: 25203},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 78000},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path:            "MiniMax-M3.gguf",
		TotalSizeMB:     149 * 1024,
		SizeBytes:       149 * 1024 * 1024 * 1024,
		NumLayers:       60,
		IsMoE:           true,
		NumExperts:      128,
		LeadingDense:    3,
		ExpertBytes:     int64(57 * 2500 * 1024 * 1024),
		NonExpertBytes:  int64(6500 * 1024 * 1024),
		ContextSize:     32768,
		EmbeddingLength: 6144,
		HeadCountKV:     4,
		KeyLength:       128,
		ValueLength:     128,
		ExpertUsedCount: 4,
		ExpertFF:        3072,
	}

	strat, err := Compute(caps, model, Options{ContextSize: 32768, KVQuality: "low", CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.Type != MoEOffload {
		t.Fatalf("expected MoE offload, got %s", strat.Type)
	}
	if len(strat.TensorSplit) != len(caps.GPUs) {
		t.Fatalf("expected tensor split for every visible GPU, got %v", strat.TensorSplit)
	}
	if strat.SplitMode != "layer" {
		t.Fatalf("expected MoE split-mode layer, got %q", strat.SplitMode)
	}
	if !strings.Contains(strat.OTString, "exps=CPU") {
		t.Fatalf("expected CPU expert catch-all in -ot, got %s", strat.OTString)
	}

	assignments := parseOTLayersByDevice(t, strat.OTString)
	for device, layers := range assignments {
		for _, layer := range layers {
			if layer < model.LeadingDense {
				t.Fatalf("device %d pinned leading dense layer %d in %s", device, layer, strat.OTString)
			}
		}
	}
	expertPerLayerMB := ceilDivInt(bytesToMiBCeil(model.ExpertBytes), model.NumLayers-model.LeadingDense)
	nonExpertTotalMB := bytesToMiBCeil(model.NonExpertBytes)
	kvTotalMB := computeKVTotalMB(model, strat.ContextSize, strat.KVType)
	fixedPerGPU := 600 + computeFloorMB
	for gi, gpu := range caps.GPUs {
		usedMB := fixedPerGPU + splitShareMB(nonExpertTotalMB, strat.TensorSplit, gi) + splitShareMB(kvTotalMB, strat.TensorSplit, gi) + len(assignments[gpu.Index])*expertPerLayerMB
		if usedMB > gpu.VRAMFreeMB() {
			t.Fatalf("gpu %d over budget: used=%dMB free=%dMB split=%v ot=%s", gpu.Index, usedMB, gpu.VRAMFreeMB(), strat.TensorSplit, strat.OTString)
		}
	}
}

func TestComputeMoEMultiGPUFullyFitsExpertsOnGPU(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "GPU A", VRAMTotalMB: 24576, BandwidthMBps: 20000},
			{Index: 1, Name: "GPU B", VRAMTotalMB: 24576, BandwidthMBps: 20000},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:            "moe.gguf",
		TotalSizeMB:     32 * 1024,
		SizeBytes:       32 * 1024 * 1024 * 1024,
		NumLayers:       32,
		IsMoE:           true,
		NumExperts:      64,
		ExpertBytes:     16 * 1024 * 1024 * 1024,
		NonExpertBytes:  16 * 1024 * 1024 * 1024,
		ContextSize:     32768,
		EmbeddingLength: 4096,
		HeadCountKV:     8,
		KeyLength:       128,
		ValueLength:     128,
	}

	strat, err := Compute(caps, model, Options{ContextSize: 32768, KVQuality: "low", CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	assignments := parseOTLayersByDevice(t, strat.OTString)
	totalPinned := 0
	for _, layers := range assignments {
		totalPinned += len(layers)
	}
	if totalPinned != model.NumLayers {
		t.Fatalf("expected all %d expert layers on GPU, got %d via %s", model.NumLayers, totalPinned, strat.OTString)
	}
	if strat.NCPUMoE != 0 {
		t.Fatalf("expected no CPU MoE layers when experts fit, got %d", strat.NCPUMoE)
	}
}

func TestComputeMoESingleGPUDoesNotEmitTensorSplit(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "RTX", VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:           "moe.gguf",
		TotalSizeMB:    48 * 1024,
		SizeBytes:      48 * 1024 * 1024 * 1024,
		NumLayers:      48,
		IsMoE:          true,
		NumExperts:     64,
		ExpertBytes:    40 * 1024 * 1024 * 1024,
		NonExpertBytes: 8 * 1024 * 1024 * 1024,
		ContextSize:    32768,
		HeadCountKV:    8,
		KeyLength:      128,
		ValueLength:    128,
	}
	strat, err := Compute(caps, model, Options{ContextSize: 32768, KVQuality: "low", CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if len(strat.TensorSplit) != 0 {
		t.Fatalf("single-GPU MoE should not emit tensor split, got %v", strat.TensorSplit)
	}
	if !strings.Contains(strat.OTString, "exps=CPU") {
		t.Fatalf("single-GPU MoE still needs CPU catch-all, got %s", strat.OTString)
	}
}

func TestPlacementCacheRequiresTensorSplitForAssignments(t *testing.T) {
	path := filepath.Join(t.TempDir(), "old.conf")
	if err := os.WriteFile(path, []byte("CACHED_GPU_ASSIGNMENTS=\"0:0:4\"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}}}
	if _, err := LoadPlacementCache(path, caps, 0); err == nil {
		t.Fatal("expected old MoE assignment cache without tensor split to be rejected")
	}
}

func TestPlacementCacheRoundTripsTensorSplit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.conf")
	entry := &CacheEntry{
		GPUAssignments: []GPUAssignment{{CUDAIndex: 0, Start: 3, Count: 4}},
		TensorSplit:    []float64{0.5, 0.25, 0.25},
		SplitMode:      "layer",
		NCPUMoE:        53,
		BatchSize:      2048,
		UBatchSize:     512,
	}
	if err := SavePlacementCache(path, entry); err != nil {
		t.Fatalf("save cache: %v", err)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 12288}, {Index: 2, VRAMTotalMB: 12288}}}
	loaded, err := LoadPlacementCache(path, caps, 0)
	if err != nil {
		t.Fatalf("load cache: %v", err)
	}
	if loaded.SplitMode != "layer" || len(loaded.TensorSplit) != 3 || loaded.TensorSplit[0] != 0.5 {
		t.Fatalf("tensor split did not round trip: %+v", loaded)
	}
}

func TestDerateCUDAOOMArgsMovesExpertLayersToCPU(t *testing.T) {
	model := &ModelProfile{
		NumLayers:    60,
		LeadingDense: 3,
		ExpertBytes:  int64(57 * 2500 * 1024 * 1024),
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576},
		{Index: 1, VRAMTotalMB: 12288, VRAMUsedMB: 574},
	}}
	args := []string{
		"--tensor-split", "0.67,0.33",
		"--split-mode", "layer",
		"-b", "2048",
		"-ub", "512",
		"--parallel", "1",
		"-ot", `blk\.(3|4|5|6)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,blk\.(7|8|9|10)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA1,exps=CPU`,
		"--n-cpu-moe", "49",
	}
	newArgs, entry, ok := DerateCUDAOOMArgs(args, model, caps, 1, 11876)
	if !ok {
		t.Fatal("expected CUDA OOM args to derate")
	}
	newOT := newArgs[argIndex(newArgs, "-ot")+1]
	assignments := parseOTLayersByDevice(t, newOT)
	if got := len(assignments[1]); got != 3 {
		t.Fatalf("expected device 1 to drop one layer, got %d via %s", got, newOT)
	}
	if currentNCPUMoE(newArgs) != 50 {
		t.Fatalf("expected --n-cpu-moe to increment to 50, got args %v", newArgs)
	}
	if entry == nil || len(entry.GPUAssignments) != 2 || entry.NCPUMoE != 50 || len(entry.TensorSplit) != 2 {
		t.Fatalf("unexpected cache entry: %+v", entry)
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

func TestComputeDraftNgramOptIn(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false}

	draft := ComputeDraft(model, caps, Options{SpecMode: "ngram"})
	if draft.Type != DraftNgram {
		t.Fatalf("expected ngram draft, got %s", draft.Type)
	}
	args := DraftFlags(draft)
	if !contains(args, "--spec-ngram-map-k-size-n") {
		t.Fatalf("ngram flags missing expected values: %v", args)
	}
}

func TestComputeDraftSpecAutoTuneRequiresSupport(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false}

	plain := ComputeDraft(model, caps, Options{SpecMode: "ngram", BackendTag: "vulkan", BackendHelp: "--spec-type [ngram-map-k]"})
	if contains(DraftFlags(plain), "--spec-autotune") {
		t.Fatalf("did not expect spec-autotune without backend support: %v", DraftFlags(plain))
	}
	supported := ComputeDraft(model, caps, Options{SpecMode: "ngram", BackendTag: "vulkan", BackendHelp: "--spec-type [ngram-map-k] --spec-autotune"})
	if !contains(DraftFlags(supported), "--spec-autotune") {
		t.Fatalf("expected spec-autotune when backend advertises it: %v", DraftFlags(supported))
	}
}

func TestComputeDraftMTPIKOnly(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "moe.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: true, NextNPredictLayers: 1}

	blocked := ComputeDraft(model, caps, Options{SpecMode: "mtp", BackendTag: "llama"})
	if blocked.Type != DraftNone {
		t.Fatalf("expected mainline MTP to be skipped, got %s", blocked.Type)
	}
	mtp := ComputeDraft(model, caps, Options{SpecMode: "mtp", BackendTag: "ik_llama"})
	if mtp.Type != DraftMTP {
		t.Fatalf("expected ik MTP, got %s", mtp.Type)
	}
	args := DraftFlags(mtp)
	if !contains(args, "--multi-token-prediction") || !contains(args, "--spec-type") {
		t.Fatalf("MTP flags missing expected values: %v", args)
	}
}

func TestDraftFlagsIKDraftUsesCanonicalSpecType(t *testing.T) {
	cfg := &DraftConfig{Type: DraftModel, BackendTag: "ik_llama", Path: "draft.gguf", DraftGPU: 0, CTXSizeDraft: 8192, KVTypeDraft: "q8_0", ThreadsDraft: 2, DraftMax: 16, PSplit: 0.10, SpecAutoTune: true}
	args := DraftFlags(cfg)
	if !contains(args, "--spec-type") || !contains(args, "draft:n_max=16") {
		t.Fatalf("expected canonical IK draft spec-type, got %v", args)
	}
	if contains(args, "--draft-max") || contains(args, "--spec-draft-n-max") {
		t.Fatalf("IK draft flags should not use legacy draft max flags: %v", args)
	}
	if !contains(args, "--p-split") || !contains(args, "--spec-autotune") {
		t.Fatalf("IK draft flags missing p-split/autotune: %v", args)
	}
}

func TestDraftFlagsIKMTPUsesCanonicalNMax(t *testing.T) {
	cfg := &DraftConfig{Type: DraftMTP, BackendTag: "ik_llama", SpecType: "mtp", DraftMax: 4, MTPFlag: true}
	args := DraftFlags(cfg)
	if !contains(args, "--spec-type") || !contains(args, "mtp:n_max=4") || !contains(args, "--multi-token-prediction") {
		t.Fatalf("expected canonical IK MTP flags, got %v", args)
	}
	if contains(args, "--draft-max") || contains(args, "--spec-draft-n-max") {
		t.Fatalf("IK MTP flags should not use legacy draft max flags: %v", args)
	}
}

func TestComputeDraftMTPSkipsWithoutNextNLayers(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false}
	draft := ComputeDraft(model, caps, Options{SpecMode: "mtp", BackendTag: "ik_llama"})
	if draft.Type != DraftNone {
		t.Fatalf("expected MTP to skip without NextN layers, got %s", draft.Type)
	}
}

func TestComputeDraftMoERequiresExplicitOverride(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "moe.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: true}

	blocked := ComputeDraft(model, caps, Options{SpecMode: "ngram"})
	if blocked.Type != DraftNone {
		t.Fatalf("expected MoE speculative decoding to be gated, got %s", blocked.Type)
	}
	forced := ComputeDraft(model, caps, Options{SpecMode: "ngram", ForceSpecMoE: true})
	if forced.Type != DraftNgram {
		t.Fatalf("expected force override to enable ngram, got %s", forced.Type)
	}
}

func TestFindOrDownloadDraftIgnoresInvalidLocalWhenDownloadsSkipped(t *testing.T) {
	t.Setenv("LLM_SERVER_SKIP_DRAFT_DOWNLOAD", "1")
	dir := t.TempDir()
	bad := filepath.Join(dir, "draft-model.gguf")
	if err := os.WriteFile(bad, []byte("not gguf"), 0644); err != nil {
		t.Fatalf("write bad draft: %v", err)
	}
	model := &ModelProfile{Path: filepath.Join(dir, "target.gguf"), TotalSizeMB: 1024, VocabSize: 1}
	if got := findOrDownloadDraftCandidate(model, dir, "ik_llama"); got != "" {
		t.Fatalf("expected invalid local draft to be ignored, got %s", got)
	}
}

func TestDraftCandidateFiltersNonTextArtifacts(t *testing.T) {
	for _, name := range []string{"mmproj-F16.gguf", "vision-projector.gguf", "clip-model.gguf", "Qwen_Qwen3.6-35B-A3B-imatrix.gguf"} {
		if !isNonTextDraftGGUFName(name) {
			t.Fatalf("expected %s to be rejected as non-text draft", name)
		}
		if draftFilenameLooksRelevantForKind(name, "draft") {
			t.Fatalf("did not expect projector %s to be relevant", name)
		}
	}
	if isNonTextDraftGGUFName("Qwen3.5-0.8B-Q4_K_M.gguf") {
		t.Fatal("text draft model was incorrectly rejected")
	}
}

func TestDraftValidationRepoWideMismatch(t *testing.T) {
	if !draftValidationRepoWideMismatch(fmt.Errorf("vocab mismatch: draft=1 target=2")) {
		t.Fatal("expected vocab mismatch to stop repo")
	}
	if !draftValidationRepoWideMismatch(fmt.Errorf("architecture mismatch draft=llama target=qwen")) {
		t.Fatal("expected architecture mismatch to stop repo")
	}
	if draftValidationRepoWideMismatch(fmt.Errorf("incomplete file")) {
		t.Fatal("did not expect incomplete file to stop repo")
	}
}

func TestDraftCandidateRankPrefersQ4Draft(t *testing.T) {
	q4 := draftCandidateRank("Qwen3.5-0.8B-Q4_K_M.gguf", "draft")
	bf16 := draftCandidateRank("Qwen3.5-0.8B-BF16.gguf", "draft")
	iq2 := draftCandidateRank("Qwen3.5-0.8B-IQ2_M.gguf", "draft")
	if !(q4 < bf16 && q4 < iq2) {
		t.Fatalf("expected Q4 draft rank to win, got q4=%d bf16=%d iq2=%d", q4, bf16, iq2)
	}
}

func TestHFSpecSearchQueriesForQwenDraft(t *testing.T) {
	model := &ModelProfile{Path: "Qwen3.6-27B-Q5_K_M.gguf", Basename: "Qwen3.6-27B", ModelArch: "qwen35"}
	queries := hfSpecSearchQueries(model, "draft")
	joined := strings.ToLower(strings.Join(queries, "\n"))
	for _, want := range []string{"qwen3.6 27b draft gguf", "qwen3.6 0.8b gguf", "qwen3.5 0.8b gguf"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("expected query %q in %#v", want, queries)
		}
	}
}

func TestHFRepoLooksRelevantForSmallDraft(t *testing.T) {
	model := &ModelProfile{Path: "Qwen3.6-27B-Q5_K_M.gguf", Basename: "Qwen3.6-27B", ModelArch: "qwen35"}
	if !hfRepoLooksRelevant("bartowski/Qwen3.6-0.8B-GGUF", model, "draft") {
		t.Fatal("expected small same-family repo to be considered as draft candidate")
	}
	if !hfRepoLooksRelevant("unsloth/Qwen3.5-0.8B-GGUF", model, "draft") {
		t.Fatal("expected qwen3.5 architecture-compatible repo to be considered as draft candidate")
	}
	if hfRepoLooksRelevant("unsloth/Qwen3.6-27B-GGUF", model, "draft") {
		t.Fatal("did not expect full-size target repo to be considered as draft candidate")
	}
	if hfRepoLooksRelevant("bartowski/Qwen_Qwen3.6-35B-A3B-GGUF", model, "draft") {
		t.Fatal("did not expect 35B/A3B full MoE repo to be considered as draft candidate")
	}
	if repoLooksLikeDraftRepo("bartowski/qwen_qwen3.6-35b-a3b-gguf") {
		t.Fatal("35B/A3B should not match the small 3B draft heuristic")
	}
	if !hfRepoLooksRelevant("Ex0bit/Qwen3.6-27B-PRISM-EAGLE3", model, "eagle3") {
		t.Fatal("expected target EAGLE repo to be considered relevant")
	}
}

func TestHFCandidateSizeOK(t *testing.T) {
	model := &ModelProfile{TotalSizeMB: 1000}
	if !hfCandidateSizeOK(&http.Response{ContentLength: 250 * 1024 * 1024}, model) {
		t.Fatal("expected small candidate to pass")
	}
	if hfCandidateSizeOK(&http.Response{ContentLength: 500 * 1024 * 1024}, model) {
		t.Fatal("expected oversized candidate to be rejected")
	}
}

func TestHFResolveURLKeepsPathSeparators(t *testing.T) {
	got := hfResolveURL("org/repo", "folder/a b.gguf")
	want := "https://huggingface.co/org/repo/resolve/main/folder/a%20b.gguf"
	if got != want {
		t.Fatalf("resolve URL mismatch: %s", got)
	}
}

func TestSameDraftArchitecture(t *testing.T) {
	if !sameDraftArchitecture("qwen2", "qwen2") {
		t.Fatal("expected matching architecture to pass")
	}
	if sameDraftArchitecture("qwen2", "llama") {
		t.Fatal("expected mismatched architecture to fail")
	}
	if !sameDraftArchitecture("", "llama") {
		t.Fatal("missing target metadata should not reject a draft")
	}
}

func TestComputeDraftAutoDoesNotFallbackToNgram(t *testing.T) {
	t.Setenv("LLM_SERVER_SKIP_DRAFT_DOWNLOAD", "1")
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: t.TempDir() + "/model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false}
	help := "--spec-type [none|draft-simple|draft-mtp|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod] --spec-ngram-mod-n-match --spec-ngram-mod-n-min --spec-ngram-mod-n-max"

	draft := ComputeDraft(model, caps, Options{SpecMode: "auto", BackendTag: "vulkan", BackendHelp: help})
	if draft.Type != DraftNone {
		t.Fatalf("expected auto to stay off without MTP/EAGLE/draft, got type=%s spec=%s", draft.Type, draft.SpecType)
	}
}

func TestComputeDraftAutoPrefersMTPWhenAvailable(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false, NextNPredictLayers: 1}

	draft := ComputeDraft(model, caps, Options{SpecMode: "auto", BackendTag: "ik_llama"})
	if draft.Type != DraftMTP || draft.SpecType != "mtp" {
		t.Fatalf("expected auto MTP, got type=%s spec=%s", draft.Type, draft.SpecType)
	}
}

func TestDraftFlagsEagle3(t *testing.T) {
	cfg := &DraftConfig{Type: DraftEagle3, BackendTag: "vulkan", Path: "eagle.gguf", DraftGPU: 0, CTXSizeDraft: 8192, KVTypeDraft: "q8_0", ThreadsDraft: 2, DraftMax: 8}
	args := DraftFlags(cfg)
	if !contains(args, "--spec-type") || !contains(args, "eagle3") || !contains(args, "--model-draft") {
		t.Fatalf("EAGLE-3 flags missing expected values: %v", args)
	}
}

func TestComputeDraftDraftDoesNotFallbackToNgram(t *testing.T) {
	t.Setenv("LLM_SERVER_SKIP_DRAFT_DOWNLOAD", "1")
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: t.TempDir() + "/model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false}
	help := "--spec-type [none|ngram-map-k|ngram-mod] --spec-ngram-mod-n-match"

	draft := ComputeDraft(model, caps, Options{SpecMode: "draft", BackendTag: "llama", BackendHelp: help})
	if draft.Type != DraftNone {
		t.Fatalf("expected explicit draft mode to skip without compatible draft model, got %s", draft.Type)
	}
}

func TestComputeDraftNgramK4VFlags(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false}
	help := "--spec-type [none|ngram-map-k|ngram-map-k4v] --spec-ngram-map-k4v-size-n --spec-ngram-map-k4v-size-m --spec-ngram-map-k4v-min-hits"

	draft := ComputeDraft(model, caps, Options{SpecMode: "ngram-k4v", BackendTag: "vulkan", BackendHelp: help})
	if draft.Type != DraftNgram || draft.SpecType != "ngram-map-k4v" {
		t.Fatalf("expected ngram-map-k4v, got type=%s spec=%s", draft.Type, draft.SpecType)
	}
	args := DraftFlags(draft)
	if !contains(args, "--spec-ngram-map-k4v-size-n") || !contains(args, "--spec-draft-n-max") {
		t.Fatalf("ngram-map-k4v flags missing expected values: %v", args)
	}
}

func TestComputeDraftMainlineMTPWhenAdvertised(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false, NextNPredictLayers: 1}
	help := "--spec-type [none|draft-simple|draft-mtp|ngram-map-k]"

	draft := ComputeDraft(model, caps, Options{SpecMode: "mtp", BackendTag: "llama", BackendHelp: help})
	if draft.Type != DraftMTP || draft.SpecType != "draft-mtp" {
		t.Fatalf("expected mainline draft-mtp, got type=%s spec=%s", draft.Type, draft.SpecType)
	}
	args := DraftFlags(draft)
	if contains(args, "--multi-token-prediction") {
		t.Fatalf("mainline draft-mtp should not get ik MTP flag: %v", args)
	}
}

func TestDraftDeviceUsesVulkanDialect(t *testing.T) {
	cfg := &DraftConfig{Type: DraftModel, BackendTag: "vulkan", Path: "draft.gguf", DraftGPU: 1, DraftMax: 3}
	args := DraftFlags(cfg)
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--device-draft" && args[i+1] == "Vulkan1" {
			return
		}
	}
	t.Fatalf("expected Vulkan draft device flag, got %v", args)
}

func TestArgsCPUOnlyIncludesZeroGPULayers(t *testing.T) {
	s := &Strategy{
		Type:           CPUOnly,
		ContextSize:    4096,
		KVQuality:      "mid",
		KVType:         "q4_0",
		FlashAttention: true,
		Threads:        8,
		ThreadsBatch:   8,
		BatchSize:      512,
		UBatchSize:     256,
	}
	args := s.Args("/models/test.gguf", 8081)
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "-ngl" && args[i+1] == "0" {
			return
		}
	}
	t.Fatalf("expected CPU-only args to include -ngl 0, got %v", args)
}

func TestArgsSingleGPUPinsDevice(t *testing.T) {
	s := &Strategy{
		Type:           SingleGPU,
		ContextSize:    4096,
		MainGPU:        0,
		KVType:         "q4_0",
		FlashAttention: true,
		Threads:        8,
		ThreadsBatch:   8,
		BatchSize:      8192,
		UBatchSize:     1024,
	}
	args := s.Args("/models/test.gguf", 8081)
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--device" && args[i+1] == "CUDA0" {
			return
		}
	}
	t.Fatalf("expected single-GPU args to include --device CUDA0, got %v", args)
}

func TestArgsSingleGPUPinsVulkanDevice(t *testing.T) {
	s := &Strategy{
		Type:           SingleGPU,
		ContextSize:    4096,
		MainGPU:        0,
		BackendTag:     "vulkan",
		KVType:         "q4_0",
		FlashAttention: true,
		Threads:        8,
		ThreadsBatch:   8,
		BatchSize:      8192,
		UBatchSize:     1024,
	}
	args := s.Args("/models/test.gguf", 8081)
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--device" && args[i+1] == "Vulkan0" {
			return
		}
	}
	t.Fatalf("expected single-GPU args to include --device Vulkan0, got %v", args)
}

func TestRestrictGPUsFiltersAndRenumbers(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 4070", VRAMTotalMB: 12288},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288},
			{Index: 2, Name: "RTX 3090", VRAMTotalMB: 24576},
		},
		RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU: detect.CPUInfo{Cores: 16},
	}
	out, err := restrictGPUs(caps, []int{1, 2})
	if err != nil {
		t.Fatalf("restrictGPUs failed: %v", err)
	}
	if len(out.GPUs) != 2 {
		t.Fatalf("expected 2 GPUs after restriction, got %d", len(out.GPUs))
	}
	if out.GPUs[0].Name != "RTX 3060" || out.GPUs[1].Name != "RTX 3090" {
		t.Fatalf("wrong GPUs selected: %v", out.GPUs)
	}
	// Renumbered from 0 to match CUDA_VISIBLE_DEVICES enumeration.
	if out.GPUs[0].Index != 0 || out.GPUs[1].Index != 1 {
		t.Fatalf("expected renumbered indices 0,1; got %d,%d", out.GPUs[0].Index, out.GPUs[1].Index)
	}
	// Original caps untouched.
	if len(caps.GPUs) != 3 || caps.GPUs[1].Index != 1 {
		t.Fatalf("restrictGPUs mutated input caps")
	}
}

func TestRestrictGPUsNoMatchErrors(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 12288}},
	}
	if _, err := restrictGPUs(caps, []int{5}); err == nil {
		t.Fatal("expected error for non-existent GPU index")
	}
}

func TestRestrictGPUsEmptyPassthrough(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0}, {Index: 1}},
	}
	out, err := restrictGPUs(caps, nil)
	if err != nil || out != caps {
		t.Fatalf("expected passthrough for empty restriction, got %v %v", out, err)
	}
}

func TestComputeHonorsGPURestriction(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576},
			{Index: 1, VRAMTotalMB: 12288},
		},
		RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "model.gguf",
		SizeBytes:   4 * 1024 * 1024 * 1024,
		TotalSizeMB: 4 * 1024,
		NumLayers:   32,
		ContextSize: 32768,
		HiddenSize:  4096,
		HeadCountKV: 8,
		KeyLength:   128,
		ValueLength: 128,
	}
	strat, err := Compute(caps, model, Options{GPUs: []int{1}, KVPlacement: "auto", KVQuality: "low"})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	// With only GPU 1 visible there is exactly one device, so no tensor split
	// across two devices may be emitted.
	if len(strat.TensorSplit) > 1 {
		t.Fatalf("expected single-GPU placement under restriction, got split %v", strat.TensorSplit)
	}
}

func TestApplyRAMBudgetOverridesDetectedRAM(t *testing.T) {
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 8192, FreeMB: 1024},
	}
	out := applyRAMBudget(caps, 65536)
	if out == caps {
		t.Fatalf("expected budgeted capabilities copy")
	}
	if out.RAM.TotalMB != 65536 || out.RAM.FreeMB != 65536 {
		t.Fatalf("expected explicit RAM budget to be used, got %+v", out.RAM)
	}
	if caps.RAM.TotalMB != 8192 || caps.RAM.FreeMB != 1024 {
		t.Fatalf("applyRAMBudget mutated input caps: %+v", caps.RAM)
	}
}

func TestCPUOnlyAutoContextUsesRAMBudget(t *testing.T) {
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 8192, FreeMB: 1495},
		CPU: detect.CPUInfo{Cores: 2},
	}
	model := &ModelProfile{
		Path:        "moe.gguf",
		TotalSizeMB: 1024,
		NumLayers:   40,
		IsMoE:       true,
		HeadCountKV: 2,
		KeyLength:   256,
		ValueLength: 256,
		CTXTrain:    262144,
	}
	strat, err := Compute(caps, model, Options{CPUMode: true, RamBudgetMB: 512000, KVQuality: "low"})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.Type != CPUOnly {
		t.Fatalf("expected CPU-only strategy, got %s", strat.Type)
	}
	if strat.ContextSize != 262144 {
		t.Fatalf("expected RAM-budgeted CPU auto-context 262144, got %d", strat.ContextSize)
	}
}

func TestArgsMetalSkipsDeviceRouting(t *testing.T) {
	s := &Strategy{
		Type:        SingleGPU,
		ContextSize: 32768,
		KVType:      "q8_0",
		BatchSize:   4096,
		UBatchSize:  512,
		Threads:     8,
		BackendTag:  "metal",
		MainGPU:     0,
		GPULayers:   999,
	}
	args := s.Args("model.gguf", 8081)
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "-ngl 999") {
		t.Fatalf("metal must offload with -ngl 999, got: %s", joined)
	}
	for _, banned := range []string{"--device", "-mg", "--run-time-repack"} {
		for _, a := range args {
			if a == banned {
				t.Fatalf("metal args must not contain %s: %s", banned, joined)
			}
		}
	}
}

func TestComputeAppleSiliconSingleGPU(t *testing.T) {
	// A 32GB M-series Mac: one synthesized GPU with 24GB working set.
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "Apple M2 Pro", VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 32768, FreeMB: 26214},
		CPU:  detect.CPUInfo{Cores: 10},
	}
	model := &ModelProfile{
		Path:        "model.gguf",
		SizeBytes:   4 * 1024 * 1024 * 1024,
		TotalSizeMB: 4 * 1024,
		NumLayers:   32,
		ContextSize: 32768,
		HiddenSize:  4096,
		HeadCountKV: 8,
		KeyLength:   128,
		ValueLength: 128,
	}
	strat, err := Compute(caps, model, Options{KVPlacement: "auto", KVQuality: "low", BackendTag: "metal"})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.Type == CPUOnly {
		t.Fatal("Apple Silicon must not fall back to CPU-only placement")
	}
	args := strat.Args("model.gguf", 8081)
	for i, a := range args {
		if a == "-ngl" && args[i+1] == "0" {
			t.Fatal("Apple Silicon launch must not disable GPU offload")
		}
	}
}

func TestKVOnCPUFreesVRAMForExperts(t *testing.T) {
	// A big MoE with KV on CPU must place MORE expert layers on the GPU than the
	// same model with KV on GPU — because the CPU-KV frees the VRAM that would
	// otherwise be (wrongly) reserved for a cache that isn't there.
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	mk := func() *ModelProfile {
		return &ModelProfile{
			Path: "moe.gguf", SizeBytes: 40 * 1024 * 1024 * 1024,
			NumLayers: 64, NumParams: 70_000_000_000, IsMoE: true, NumExperts: 64,
			ContextSize: 32768, HiddenSize: 4096,
			HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
			ExpertBytes: 36 * 1024 * 1024 * 1024, NonExpertBytes: 4 * 1024 * 1024 * 1024,
			CTXTrain: 32768,
		}
	}
	gpuKV, err := Compute(caps, mk(), Options{ContextSize: 32768, KVPlacement: "gpu", KVQuality: "mid"})
	if err != nil {
		t.Fatalf("gpu-kv compute: %v", err)
	}
	cpuKV, err := Compute(caps, mk(), Options{ContextSize: 32768, KVPlacement: "cpu", KVQuality: "mid"})
	if err != nil {
		t.Fatalf("cpu-kv compute: %v", err)
	}
	if cpuKV.NCPUMoE >= gpuKV.NCPUMoE {
		t.Fatalf("KV-on-CPU should offload FEWER experts to CPU (more on GPU): cpu-kv NCPUMoE=%d, gpu-kv NCPUMoE=%d", cpuKV.NCPUMoE, gpuKV.NCPUMoE)
	}
}
