package placement

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// ---------------------------------------------------------------------------
// Args() — fork-specific flag emission
// ---------------------------------------------------------------------------

func baseStrategy() *Strategy {
	return &Strategy{
		Type:           SingleGPU,
		ContextSize:    32768,
		GPULayers:      999,
		KVPlacement:    "gpu",
		KVType:         "q8_0",
		FlashAttention: true,
		MMap:           true,
		Threads:        8,
		ThreadsBatch:   8,
		BatchSize:      4096,
		UBatchSize:     512,
		BackendTag:     "llama",
		Host:           "127.0.0.1",
	}
}

func argsContain(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

func argsContainPair(args []string, flag, value string) bool {
	for i := 0; i < len(args)-1; i++ {
		if args[i] == flag && args[i+1] == value {
			return true
		}
	}
	return false
}

func TestArgs_FlashAttnConditional(t *testing.T) {
	s := baseStrategy()

	// FlashAttention = true → "--flash-attn", "on" present
	s.FlashAttention = true
	args := s.Args("/models/test.gguf", 8080)
	if !argsContainPair(args, "--flash-attn", "on") {
		t.Errorf("expected --flash-attn on when FlashAttention=true, got: %v", args)
	}

	// FlashAttention = false → absent
	s.FlashAttention = false
	args = s.Args("/models/test.gguf", 8080)
	if argsContain(args, "--flash-attn") {
		t.Errorf("expected no --flash-attn when FlashAttention=false, got: %v", args)
	}
}

func TestArgs_KVOffloadFromStrategy(t *testing.T) {
	s := baseStrategy()

	// KVPlacement = "gpu" → --kv-offload
	s.KVPlacement = "gpu"
	args := s.Args("/models/test.gguf", 8080)
	if !argsContain(args, "--kv-offload") {
		t.Errorf("expected --kv-offload when KVPlacement=gpu, got: %v", args)
	}
	if argsContain(args, "--no-kv-offload") {
		t.Errorf("did not expect --no-kv-offload when KVPlacement=gpu")
	}

	// KVPlacement = "cpu" → --no-kv-offload
	s.KVPlacement = "cpu"
	args = s.Args("/models/test.gguf", 8080)
	if !argsContain(args, "--no-kv-offload") {
		t.Errorf("expected --no-kv-offload when KVPlacement=cpu, got: %v", args)
	}
	if argsContain(args, "--kv-offload") {
		t.Errorf("did not expect --kv-offload when KVPlacement=cpu")
	}
}

func TestArgs_MaxCheckpointsRespected(t *testing.T) {
	s := baseStrategy()

	// MaxCheckpoints = 3 → emitted
	s.MaxCheckpoints = 3
	args := s.Args("/models/test.gguf", 8080)
	if !argsContainPair(args, "--ctx-checkpoints", "3") {
		t.Errorf("expected --ctx-checkpoints 3, got: %v", args)
	}

	// MaxCheckpoints = 0 → emitted as "0" (explicit disable)
	s.MaxCheckpoints = 0
	args = s.Args("/models/test.gguf", 8080)
	if !argsContainPair(args, "--ctx-checkpoints", "0") {
		t.Errorf("expected --ctx-checkpoints 0 (explicit disable), got: %v", args)
	}

	// MaxCheckpoints = -1 → NOT emitted (means "not computed")
	s.MaxCheckpoints = -1
	args = s.Args("/models/test.gguf", 8080)
	if argsContain(args, "--ctx-checkpoints") {
		t.Errorf("expected no --ctx-checkpoints when MaxCheckpoints=-1, got: %v", args)
	}
}

func TestArgs_CRAMAlwaysEmitted(t *testing.T) {
	s := baseStrategy()

	// CRAM = 0 → still emitted as "-cram 0"
	s.CRAM = 0
	args := s.Args("/models/test.gguf", 8080)
	if !argsContainPair(args, "-cram", "0") {
		t.Errorf("expected -cram 0 to always be emitted, got: %v", args)
	}

	// CRAM = 2048 → emitted
	s.CRAM = 2048
	args = s.Args("/models/test.gguf", 8080)
	if !argsContainPair(args, "-cram", "2048") {
		t.Errorf("expected -cram 2048, got: %v", args)
	}
}

func TestArgs_MoEOffloadEmitsOTAndNCPUMoE(t *testing.T) {
	s := baseStrategy()
	s.Type = MoEOffload
	s.OTString = `blk\.(0|1|2)\.ffn_.*=CUDA0,exps=CPU`
	s.NCPUMoE = 5
	s.UseNativeOffload = false

	args := s.Args("/models/test.gguf", 8080)

	// -ot must be present for strict MoE offload
	if !argsContainPair(args, "-ot", s.OTString) {
		t.Errorf("expected -ot with OTString for MoE offload, got: %v", args)
	}
	// --n-cpu-moe must be present
	if !argsContainPair(args, "--n-cpu-moe", "5") {
		t.Errorf("expected --n-cpu-moe 5, got: %v", args)
	}
}

func TestArgs_NCPUMoEOnlyWhenPositive(t *testing.T) {
	s := baseStrategy()
	s.Type = MoEOffload

	// NCPUMoE = 0 → not emitted
	s.NCPUMoE = 0
	args := s.Args("/models/test.gguf", 8080)
	if argsContain(args, "--n-cpu-moe") {
		t.Errorf("expected no --n-cpu-moe when NCPUMoE=0, got: %v", args)
	}

	// NCPUMoE = 12 → emitted
	s.NCPUMoE = 12
	args = s.Args("/models/test.gguf", 8080)
	if !argsContainPair(args, "--n-cpu-moe", "12") {
		t.Errorf("expected --n-cpu-moe 12, got: %v", args)
	}
}

func TestArgs_NoMmap(t *testing.T) {
	s := baseStrategy()

	s.MMap = true
	args := s.Args("/models/test.gguf", 8080)
	if argsContain(args, "--no-mmap") {
		t.Errorf("expected no --no-mmap when MMap=true")
	}

	s.MMap = false
	args = s.Args("/models/test.gguf", 8080)
	if !argsContain(args, "--no-mmap") {
		t.Errorf("expected --no-mmap when MMap=false, got: %v", args)
	}
}

func TestArgs_ParallelDefault(t *testing.T) {
	s := baseStrategy()

	// Parallel = 0 → emits "--parallel 1"
	s.Parallel = 0
	args := s.Args("/models/test.gguf", 8080)
	if !argsContainPair(args, "--parallel", "1") {
		t.Errorf("expected --parallel 1 when Parallel=0, got: %v", args)
	}

	// Parallel = 4 → emits "--parallel 4"
	s.Parallel = 4
	args = s.Args("/models/test.gguf", 8080)
	if !argsContainPair(args, "--parallel", "4") {
		t.Errorf("expected --parallel 4, got: %v", args)
	}
}

func TestArgs_CPUOnly(t *testing.T) {
	s := baseStrategy()
	s.Type = CPUOnly
	s.GPULayers = 0

	args := s.Args("/models/test.gguf", 8080)
	if !argsContainPair(args, "-ngl", "0") {
		t.Errorf("expected -ngl 0 for CPUOnly, got: %v", args)
	}
}

func TestArgs_ReasoningOff(t *testing.T) {
	s := baseStrategy()

	s.ReasoningOff = false
	args := s.Args("/models/test.gguf", 8080)
	if argsContain(args, "--reasoning") {
		t.Errorf("expected no --reasoning when ReasoningOff=false")
	}

	s.ReasoningOff = true
	args = s.Args("/models/test.gguf", 8080)
	if !argsContainPair(args, "--reasoning", "off") {
		t.Errorf("expected --reasoning off when ReasoningOff=true, got: %v", args)
	}
}

func TestArgs_HasSSM(t *testing.T) {
	s := baseStrategy()

	s.HasSSM = false
	args := s.Args("/models/test.gguf", 8080)
	if argsContain(args, "--no-context-shift") {
		t.Errorf("expected no --no-context-shift when HasSSM=false")
	}

	s.HasSSM = true
	args = s.Args("/models/test.gguf", 8080)
	if !argsContain(args, "--no-context-shift") {
		t.Errorf("expected --no-context-shift when HasSSM=true, got: %v", args)
	}
}

func TestArgs_TimeoutAlwaysEmitted(t *testing.T) {
	s := baseStrategy()
	args := s.Args("/models/test.gguf", 8080)
	if !argsContainPair(args, "--timeout", "2147483647") {
		t.Errorf("expected --timeout 2147483647, got: %v", args)
	}
}

func TestArgs_OTStringEmitted(t *testing.T) {
	s := baseStrategy()
	s.Type = MoEOffload
	s.UseNativeOffload = false
	s.OTString = `blk\.(0|1|2)\.ffn_.*=CUDA0,exps=CPU`

	args := s.Args("/models/test.gguf", 8080)
	if !argsContainPair(args, "-ot", s.OTString) {
		t.Errorf("expected -ot with OTString, got: %v", args)
	}
}

// ---------------------------------------------------------------------------
// buildMoEOffload — fork-specific native offload logic
// ---------------------------------------------------------------------------

func moeModel() *ModelProfile {
	return &ModelProfile{
		Path:              "/models/test-moe.gguf",
		Name:              "TestMoE",
		SizeBytes:         50 * 1024 * 1024 * 1024, // 50 GB
		TotalSizeMB:       50 * 1024,
		NumLayers:         60,
		IsMoE:             true,
		NumExperts:        64,
		ExpertBytes:       45 * 1024 * 1024 * 1024, // 45 GB experts
		NonExpertBytes:    5 * 1024 * 1024 * 1024,  // 5 GB non-expert
		ContextSize:       32768,
		EmbeddingLength:   4096,
		FeedForwardLength: 16384,
		HeadCountKV:       8,
		KeyLength:         128,
		ValueLength:       128,
	}
}

func smallGPUCaps() *detect.Capabilities {
	return &detect.Capabilities{
		CPU: detect.CPUInfo{Cores: 8, Threads: 16, Model: "Test CPU"},
		RAM: detect.RAMInfo{TotalMB: 128 * 1024, FreeMB: 100 * 1024},
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 4070", VRAMTotalMB: 12288, VRAMUsedMB: 512, BandwidthMBps: 504000},
		},
	}
}

func largeGPUCaps() *detect.Capabilities {
	return &detect.Capabilities{
		CPU: detect.CPUInfo{Cores: 16, Threads: 32, Model: "Test CPU"},
		RAM: detect.RAMInfo{TotalMB: 256 * 1024, FreeMB: 200 * 1024},
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 4090", VRAMTotalMB: 24576, VRAMUsedMB: 512, BandwidthMBps: 1008000},
			{Index: 1, Name: "RTX 4090", VRAMTotalMB: 24576, VRAMUsedMB: 512, BandwidthMBps: 1008000},
		},
	}
}

func boundaryGPUCaps() *detect.Capabilities {
	return &detect.Capabilities{
		CPU: detect.CPUInfo{Cores: 8, Threads: 16, Model: "Test CPU"},
		RAM: detect.RAMInfo{TotalMB: 128 * 1024, FreeMB: 100 * 1024},
		GPUs: []detect.GPU{
			{Index: 0, Name: "GPU16GB", VRAMTotalMB: 16384, VRAMUsedMB: 0, BandwidthMBps: 500000},
		},
	}
}

func TestBuildMoEOffload_SmallGPU_UsesStrictOffload(t *testing.T) {
	caps := smallGPUCaps() // 12 GB — strict offload, not native
	model := moeModel()
	s := &Strategy{
		ContextSize:  32768,
		KVPlacement:  "gpu",
		KVType:       "q8_0",
		MMap:         true,
		Threads:      8,
		ThreadsBatch: 8,
		BatchSize:    4096,
		UBatchSize:   512,
		BackendTag:   "llama",
		IsMoE:        true,
		GPULayers:    999,
	}
	opts := Options{BackendTag: "llama"}

	result, err := buildMoEOffload(s, caps, model, model.TotalSizeMB, 1024, opts)
	if err != nil {
		t.Fatalf("buildMoEOffload failed: %v", err)
	}

	// Must NOT use native offload (removed — always strict)
	if result.UseNativeOffload {
		t.Errorf("expected UseNativeOffload=false (native offload removed)")
	}

	// OTString should be empty for single GPU (-ot gated on numGPUs > 1)
	if result.OTString != "" {
		t.Errorf("expected empty OTString for single-GPU MoE (no -ot needed), got %q", result.OTString)
	}

	// NCPUMoE should be > 0 (most experts on CPU for a 12 GB GPU)
	if result.NCPUMoE <= 0 {
		t.Errorf("expected NCPUMoE > 0 for MoE on 12 GB GPU, got %d", result.NCPUMoE)
	}
}

func TestBuildMoEOffload_LargeGPU_SetsOTAndNCPUMoE(t *testing.T) {
	caps := largeGPUCaps() // 24 GB > 16 GB threshold
	model := moeModel()
	s := &Strategy{
		ContextSize:  32768,
		KVPlacement:  "gpu",
		KVType:       "q8_0",
		MMap:         true,
		Threads:      16,
		ThreadsBatch: 16,
		BatchSize:    4096,
		UBatchSize:   512,
		BackendTag:   "llama",
		IsMoE:        true,
		GPULayers:    999,
	}
	opts := Options{BackendTag: "llama"}

	result, err := buildMoEOffload(s, caps, model, model.TotalSizeMB, 1024, opts)
	if err != nil {
		t.Fatalf("buildMoEOffload failed: %v", err)
	}

	// Must NOT force native offload (GPUs are >= 16 GB)
	if result.UseNativeOffload {
		t.Errorf("expected UseNativeOffload=false for MoE + GPU >= 16GB")
	}

	// OTString should be non-empty (strict mode builds explicit placement)
	if result.OTString == "" {
		t.Errorf("expected non-empty OTString for strict MoE offload")
	}

	// OTString should end with "exps=CPU" catch-all
	if !strings.HasSuffix(result.OTString, "exps=CPU") {
		t.Errorf("expected OTString to end with 'exps=CPU', got %q", result.OTString)
	}
}

func TestBuildMoEOffload_BoundaryAt16GB(t *testing.T) {
	// Exactly 16384 MB → strict offload (native offload removed entirely)
	caps := boundaryGPUCaps()
	model := moeModel()
	s := &Strategy{
		ContextSize:  32768,
		KVPlacement:  "gpu",
		KVType:       "q8_0",
		MMap:         true,
		Threads:      8,
		ThreadsBatch: 8,
		BatchSize:    4096,
		UBatchSize:   512,
		BackendTag:   "llama",
		IsMoE:        true,
		GPULayers:    999,
	}
	opts := Options{BackendTag: "llama"}

	result, err := buildMoEOffload(s, caps, model, model.TotalSizeMB, 1024, opts)
	if err != nil {
		t.Fatalf("buildMoEOffload failed: %v", err)
	}

	// Native offload is removed — always strict
	if result.UseNativeOffload {
		t.Errorf("expected UseNativeOffload=false (native offload removed)")
	}
	// Should have explicit NCPUMoE
	if result.NCPUMoE <= 0 {
		t.Errorf("expected NCPUMoE > 0 for strict MoE offload, got %d", result.NCPUMoE)
	}
}

// ---------------------------------------------------------------------------
// defaultFlashAttention
// ---------------------------------------------------------------------------

func TestDefaultFlashAttention(t *testing.T) {
	model := moeModel()
	opts := Options{}

	// KV on GPU → flash attention enabled
	if !defaultFlashAttention(model, opts, "gpu") {
		t.Errorf("expected FlashAttention=true when kvPlacement=gpu")
	}

	// KV on CPU → flash attention disabled (FA kernel needs GPU-resident KV)
	if defaultFlashAttention(model, opts, "cpu") {
		t.Errorf("expected FlashAttention=false when kvPlacement=cpu")
	}
}