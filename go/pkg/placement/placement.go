package placement

import (
	"crypto/md5"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/llm-server/pkg/detect"
)

// Bash llm-server constants (lines 260-279)
const (
	vramOverheadPercent = 130  // model size * this / 100 = estimated VRAM needed
	computePerGPUMB     = 512  // legacy; non-MoE single-GPU sizing only
	computeFloorMB      = 1024 // cited llama.cpp compute floor; CUDA overhead measured separately
	minCramMB           = 512
	systemHeadroomMB    = 5120
	singleGPUHeadroomMB = 4096 // extra VRAM headroom for single-GPU mode
)

// StrategyType selects how the model is placed.
type StrategyType string

const (
	CPUOnly         StrategyType = "cpu_only"
	SingleGPU       StrategyType = "single_gpu"
	MultiGPUDense   StrategyType = "multi_gpu_dense"
	DenseCPUOffload StrategyType = "dense_cpu_offload"
	MoEOffload      StrategyType = "moe_offload"
)

// Strategy represents the computed placement for a model on this hardware.
type Strategy struct {
	Type           StrategyType `json:"type"`
	ContextSize    int          `json:"context_size"`
	GPULayers      int          `json:"gpu_layers"`      // always 999; llama-server decides
	TensorSplit    []float64    `json:"tensor_split,omitempty"`
	SplitMode      string       `json:"split_mode,omitempty"` // graph, layer, row
	MainGPU        int          `json:"main_gpu,omitempty"`
	KVPlacement    string       `json:"kv_placement"`         // gpu, cpu, auto
	KVQuality      string       `json:"kv_quality"`           // high, mid, low
	KVType         string       `json:"kv_type"`              // f16, q8_0, q4_0
	NCPUMoE        int          `json:"n_cpu_moe,omitempty"`    // for MoE offload
	OTString       string       `json:"ot_string,omitempty"`    // -ot override-tensor flags
	MMap           bool         `json:"mmap"`
	MLock          bool         `json:"mlock"`
	FlashAttention bool         `json:"flash_attention"`
	Threads        int          `json:"threads"`
	BatchSize      int          `json:"batch_size"`
	UBatchSize     int          `json:"ubatch_size"`
	BackendTag     string       `json:"backend_tag,omitempty"` // "llama" or "ik_llama"
	IsMoE          bool         `json:"is_moe"`
	ReasoningOff   bool         `json:"reasoning_off"`         // default off for OpenAI compat
	ThreadsBatch   int          `json:"threads_batch"`         // batch threads (logical cores)
	Parallel       int          `json:"parallel,omitempty"`
	CRAM           int          `json:"cram,omitempty"`        // prompt cache MB
	MaxCheckpoints int          `json:"max_checkpoints,omitempty"`
	UseCUDAGraphs  bool         `json:"use_cuda_graphs,omitempty"`
	Host           string       `json:"host,omitempty"`        // listen address
	HasSSM         bool         `json:"has_ssm,omitempty"`     // SSM/Mamba hybrid flag
	Draft          *DraftConfig `json:"draft,omitempty"`       // speculative decoding config
}

// ModelProfile describes the GGUF model.
type ModelProfile struct {
	Path               string `json:"path"`
	SizeBytes          int64  `json:"size_bytes"`
	TotalSizeMB        int    `json:"total_size_mb"`     // includes multi-part shards
	NumLayers          int    `json:"num_layers"`
	NumParams          int64  `json:"num_params"`
	IsMoE              bool   `json:"is_moe"`
	NumExperts         int    `json:"num_experts,omitempty"`
	ContextSize        int    `json:"context_size"`
	HiddenSize         int    `json:"hidden_size"`
	HeadCount          int    `json:"head_count"`
	HeadCountKV        int    `json:"head_count_kv"`
	KeyLength          int    `json:"key_length"`
	ValueLength        int    `json:"value_length"`
	VocabSize          int    `json:"vocab_size"`
	QuantType          string `json:"quant_type"`
	ExpertBytes        int64  `json:"expert_bytes"`
	NonExpertBytes     int64  `json:"non_expert_bytes"`
	Fused              int    `json:"fused"`
	EmbeddingLength    int    `json:"embedding_length"`
	FeedForwardLength  int    `json:"feed_forward_length"`
	KVLoraRank         int    `json:"kv_lora_rank"`
	QLoraRank          int    `json:"q_lora_rank"`
	RopeDim            int    `json:"rope_dim"`
	KeyLengthMLA       int    `json:"key_length_mla"`
	ValueLengthMLA     int    `json:"value_length_mla"`
	HasSSM             int    `json:"has_ssm"`
	SlidingWindow      int    `json:"sliding_window"`
	FullAttnInterval   int    `json:"full_attn_interval"`
	HasShexp           int    `json:"has_shexp"`
	CTXTrain           int    `json:"ctx_train"`
	ModelArch          string `json:"model_arch"`
	ExpertUsedCount    int    `json:"expert_used_count,omitempty"`
	ExpertFF           int    `json:"expert_ff,omitempty"`
	ExpertSharedFF     int    `json:"expert_shared_ff,omitempty"`
}

// Options allows user overrides.
type Options struct {
	ContextSize int
	KVPlacement string // auto, gpu, cpu
	KVQuality   string // high, mid, low
	GPUs        []int  // restrict to specific GPUs
	CPUMode     bool
	RamBudgetMB int
	BackendTag  string // "llama" or "ik_llama"
	NoMMap      bool
	Parallel    int
	CacheFile   string // path to placement cache for MoE recovery
	CacheDir    string // path to llm-server cache dir (for probes)
	Host        string // listen address (default 0.0.0.0)
}

// Compute builds a Strategy from hardware capabilities and model profile.
func Compute(caps *detect.Capabilities, model *ModelProfile, opts Options) (*Strategy, error) {
	s := &Strategy{
		ContextSize:    opts.ContextSize,
		KVPlacement:    opts.KVPlacement,
		KVQuality:      opts.KVQuality,
		MMap:           !opts.NoMMap,
		MLock:          false,
		Threads:        caps.CPU.Cores,   // physical cores (bash uses physical)
		ThreadsBatch:   caps.CPU.Cores,   // physical cores (bash uses physical for both)
		BackendTag:     opts.BackendTag,
		IsMoE:          model.IsMoE,
		GPULayers:      999,
		FlashAttention: true,
		ReasoningOff:   true, // match bash: --reasoning off
		HasSSM:         model.HasSSM == 1,
		Host:           opts.Host,
	}

	if s.ContextSize <= 0 {
		s.ContextSize = defaultContextSize(model, caps)
	}
	if opts.KVPlacement == "" {
		s.KVPlacement = "auto"
	}
	if opts.KVQuality == "" {
		s.KVQuality = "low" // match bash default: q4_0, minimum VRAM
	}
	if opts.Parallel > 0 {
		s.Parallel = opts.Parallel
	}

	// Total size MB
	totalSizeMB := model.TotalSizeMB
	if totalSizeMB <= 0 {
		totalSizeMB = int(model.SizeBytes / 1024 / 1024)
	}

	// KV cache type selection — try compact types first for large models
	s.KVType = kvTypeFromQuality(s.KVQuality)

	// Auto-fit context: prefer single GPU (faster), only go multi-GPU if model doesn't fit
	if opts.ContextSize <= 0 {
		// Try single-GPU first: compute context using only the best GPU's VRAM
		singleCtx, singleKVType := computeAutoContextSizeSingleGPU(caps, model, totalSizeMB, s.KVType, opts)
		singleKVTotal := computeKVTotalMB(model, singleCtx, singleKVType)

		// Check if model actually fits on single GPU at this context
		// Use VRAMFreeMB (not total) — desktop/compositor uses some VRAM
		bestFreeVRAM := 0
		for _, g := range caps.GPUs {
			if g.VRAMFreeMB() > bestFreeVRAM {
				bestFreeVRAM = g.VRAMFreeMB()
			}
		}

		// Load overhead for single-GPU check (measured, not guessed)
		sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
		cudaOverheadMB := 600
		if sysProbe != nil {
			cudaOverheadMB = sysProbe.CUDAOverheadMB
		}
		computeBufMB := computeFloorMB
		pc := loadProbeCache(opts.CacheDir, model, singleCtx, 512, singleKVType)
		if pc != nil {
			computeBufMB = pc.ComputeBufMB
		}

		singleGPUNeeded := totalSizeMB + cudaOverheadMB + computeBufMB + singleKVTotal
		singleGPUUsable := bestFreeVRAM - 1024

		if singleGPUNeeded <= singleGPUUsable && singleCtx >= 32768 {
			// Model fits on single GPU — use it (faster, simpler)
			s.ContextSize = singleCtx
			s.KVType = singleKVType
		} else {
			// Doesn't fit on single GPU — try multi-GPU
			multiCtx, multiKVType := computeAutoContextSize(caps, model, totalSizeMB, s.KVType, opts)
			s.ContextSize = multiCtx
			s.KVType = multiKVType
		}
	}

	// Compute KV cache size
	kvTotalMB := computeKVTotalMB(model, s.ContextSize, s.KVType)

	// Batch sizes based on fit
	bestGPUVRAM := 0
	for _, g := range caps.GPUs {
		if g.VRAMTotalMB > bestGPUVRAM {
			bestGPUVRAM = g.VRAMTotalMB
		}
	}

	fitsOnGPU := totalSizeMB*110/100 <= bestGPUVRAM
	if fitsOnGPU && bestGPUVRAM > totalSizeMB+singleGPUHeadroomMB {
		s.BatchSize = 8192
		s.UBatchSize = 1024
	} else if fitsOnGPU {
		s.BatchSize = 4096
		s.UBatchSize = 512
	} else {
		s.BatchSize = 2048
		s.UBatchSize = 512
	}

	// Try cached placement first (MoE only)
	if opts.CacheFile != "" && model.IsMoE {
		cache, err := LoadPlacementCache(opts.CacheFile, caps, kvTotalMB)
		if err == nil && cache != nil {
			s.Type = MoEOffload
			s.BatchSize = cache.BatchSize
			s.UBatchSize = cache.UBatchSize
			s.Parallel = cache.Parallel
			s.NCPUMoE = cache.NCPUMoE
			s.MMap = !opts.NoMMap
			if cache.MMap {
				s.MMap = true
			}
			if cache.KVUnified {
				s.KVPlacement = "gpu"
			}
			if len(cache.GPUAssignments) > 0 {
				otString := buildOTStringFromAssignments(cache.GPUAssignments, caps.GPUs, model.NumLayers)
				if otString != "" {
					s.OTString = otString
				}
			}
			return s, nil
		}
	}

	// Strategy selection (matches bash choose_strategy() exactly)
	strategy := chooseStrategy(caps, model, totalSizeMB, kvTotalMB, opts)
	s.Type = strategy

	var err error
	switch strategy {
	case CPUOnly:
		s, err = buildCPUOnly(s, caps, model, opts)
	case SingleGPU:
		s, err = buildSingleGPU(s, caps, model, totalSizeMB, kvTotalMB, opts)
	case MultiGPUDense:
		s, err = buildMultiGPUDense(s, caps, model, totalSizeMB, kvTotalMB, opts)
	case DenseCPUOffload:
		s, err = buildDenseCPUOffload(s, caps, model, totalSizeMB, kvTotalMB, opts)
	case MoEOffload:
		s, err = buildMoEOffload(s, caps, model, totalSizeMB, kvTotalMB, opts)
	}
	if err != nil {
		return nil, err
	}

	// OOM guard: refuse if model+KV+compute don't fit (non-MoE only)
	if strategy != MoEOffload {
		if err := checkMemoryOrDie(caps, model, s, totalSizeMB, kvTotalMB, opts); err != nil {
			return nil, err
		}
	}

	// Compute CRAM (prompt cache) — matches bash CRAM logic
	cram, maxCheckpoints := computeCRAM(caps, model, s, totalSizeMB, kvTotalMB)
	s.CRAM = cram
	s.MaxCheckpoints = maxCheckpoints

	// Default host
	if s.Host == "" {
		s.Host = "0.0.0.0"
	}

	return s, nil
}

// chooseStrategy mirrors bash choose_strategy() exactly (lines 2703-2739).
func chooseStrategy(caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) StrategyType {
	numGPUs := len(caps.GPUs)

	if opts.CPUMode || numGPUs == 0 {
		return CPUOnly
	}

	// Load system probe for CUDA overhead (measured, not guessed)
	sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
	cudaOverheadMB := 0
	if sysProbe != nil {
		cudaOverheadMB = sysProbe.CUDAOverheadMB
	} else if numGPUs > 0 {
		cudaOverheadMB = 600 // conservative estimate per GPU
	}

	// Load model probe for compute buffer
	computeBufMB := computeFloorMB // 1024 default
	pc := loadProbeCache(opts.CacheDir, model, opts.ContextSize, 512, opts.KVQuality)
	if pc != nil {
		computeBufMB = pc.ComputeBufMB
	}

	// Single GPU: model + overhead fits in best GPU
	// Use FREE VRAM (desktop/compositor uses some VRAM)
	bestFreeVRAM := 0
	for _, g := range caps.GPUs {
		if g.VRAMFreeMB() > bestFreeVRAM {
			bestFreeVRAM = g.VRAMFreeMB()
		}
	}
	singleGPUUsable := bestFreeVRAM - 1024
	if singleGPUUsable < 0 {
		singleGPUUsable = 0
	}
	// Use measured overhead: model weights + CUDA overhead + compute buffer + KV
	singleGPUNeeded := totalSizeMB + cudaOverheadMB + computeBufMB + kvTotalMB
	if singleGPUNeeded <= singleGPUUsable {
		return SingleGPU
	}

	// Multi-GPU dense: model fits across ALL GPUs (sum of FREE VRAM)
	if !model.IsMoE {
		totalFreeVRAM := 0
		for _, g := range caps.GPUs {
			totalFreeVRAM += g.VRAMFreeMB()
		}
		// Use measured overhead per GPU: model + (cudaOverhead + computeBuf) * numGPUs + KV
		vramNeeded := totalSizeMB + (cudaOverheadMB+computeBufMB)*numGPUs + kvTotalMB
		if vramNeeded <= totalFreeVRAM {
			return MultiGPUDense
		}
	}

	// MoE expert offload
	if model.IsMoE {
		return MoEOffload
	}

	// Dense model with CPU spill
	return DenseCPUOffload
}

func buildCPUOnly(s *Strategy, caps *detect.Capabilities, model *ModelProfile, opts Options) (*Strategy, error) {
	s.GPULayers = 0
	s.MMap = true
	s.BatchSize = 512
	s.UBatchSize = 256
	return s, nil
}

func buildSingleGPU(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	gpuOrder := orderGPUsByBandwidth(caps.GPUs)
	mainIdx := 0
	if len(gpuOrder) > 0 {
		mainIdx = gpuOrder[0]
	}
	s.MainGPU = caps.GPUs[mainIdx].Index
	return s, nil
}

func buildMultiGPUDense(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	numGPUs := len(caps.GPUs)

	// Load system probe for CUDA overhead (same as MoE path)
	sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
	cudaOverheadMB := 0
	if sysProbe != nil {
		cudaOverheadMB = sysProbe.CUDAOverheadMB
	} else if numGPUs > 0 {
		cudaOverheadMB = 600 // conservative estimate per GPU
	}

	// Load model probe for compute buffer (same as MoE path)
	probeHit := false
	computeBufMB := computeFloorMB // 1024 default
	pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality)
	if pc != nil {
		probeHit = true
		computeBufMB = pc.ComputeBufMB
	}

	// Per-layer costs
	weightPerLayerMB := totalSizeMB / model.NumLayers
	if weightPerLayerMB <= 0 {
		weightPerLayerMB = 1
	}
	kvPerLayerMB := kvTotalMB / model.NumLayers
	if kvPerLayerMB < 1 && kvTotalMB > 0 {
		kvPerLayerMB = 1
	}

	// KV-first GPU reserve: weighted by free VRAM * PCIe bandwidth
	gpuKVReserveMB := kvReserveByBandwidth(kvTotalMB, caps.GPUs, seqRange(numGPUs), kvPerLayerMB)

	// Calculate effective free VRAM per GPU for model weights
	// effectiveFree = freeVRAM - cudaOverhead - computeBuf - kvReserve
	// Tensor-split proportional to effectiveFree * bandwidth (matches bash VRAM_FREE * BANDWIDTH)
	split := make([]float64, numGPUs)
	gpuOrder := orderGPUsByBandwidth(caps.GPUs)
	totalWeighted := 0.0
	for idx := 0; idx < numGPUs; idx++ {
		gi := gpuOrder[idx]
		g := caps.GPUs[gi]
		overhead := cudaOverheadMB + computeBufMB + gpuKVReserveMB[gi]
		effective := g.VRAMFreeMB() - overhead
		if effective < 0 {
			effective = 0
		}
		bw := float64(g.BandwidthMBps)
		if bw <= 0 {
			bw = 1.0 // fallback for unknown bandwidth
		}
		totalWeighted += float64(effective) * bw
	}
	for idx := 0; idx < numGPUs; idx++ {
		gi := gpuOrder[idx]
		g := caps.GPUs[gi]
		overhead := cudaOverheadMB + computeBufMB + gpuKVReserveMB[gi]
		effective := g.VRAMFreeMB() - overhead
		if effective < 0 {
			effective = 0
		}
		bw := float64(g.BandwidthMBps)
		if bw <= 0 {
			bw = 1.0
		}
		if totalWeighted > 0 {
			split[gi] = float64(effective) * bw / totalWeighted
		}
	}
	s.TensorSplit = normalizeSplit(split)

	// Find smallest GPU subset that fits the model
	// Use effective capacity (free - overhead) not just total VRAM
	gpuOrderBW := orderGPUsByBandwidth(caps.GPUs)
	bestGPUCount := numGPUs
	for n := 2; n <= numGPUs; n++ {
		subsetCapacity := 0
		for j := 0; j < n; j++ {
			gi := gpuOrderBW[j]
			g := caps.GPUs[gi]
			overhead := cudaOverheadMB + computeBufMB + gpuKVReserveMB[gi]
			effective := g.VRAMFreeMB() - overhead
			if effective < 0 {
				effective = 0
			}
			subsetCapacity += effective
		}
		modelWeightMB := totalSizeMB + kvTotalMB/2 // model weights + partial KV overhead
		if modelWeightMB <= subsetCapacity {
			bestGPUCount = n
			break
		}
	}

	// Zero out GPUs not in the selected subset
	if bestGPUCount < numGPUs {
		for idx := bestGPUCount; idx < numGPUs; idx++ {
			gi := gpuOrderBW[idx]
			split[gi] = 0
		}
		s.TensorSplit = normalizeSplit(split)
	}

	// Prefer layer split for ik_llama (avoids NCCL P2P which fails
	// on mixed GPU architectures like Ampere+Ada), row for mainline
	if opts.BackendTag == "ik_llama" {
		s.SplitMode = "layer"
	} else {
		s.SplitMode = "row"
	}

	_ = probeHit // used for logging/debugging

	return s, nil
}

func buildDenseCPUOffload(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	numGPUs := len(caps.GPUs)
	if numGPUs > 1 {
		// Load system probe for CUDA overhead
		sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
		cudaOverheadMB := 0
		if sysProbe != nil {
			cudaOverheadMB = sysProbe.CUDAOverheadMB
		} else {
			cudaOverheadMB = 600
		}

		// Load model probe for compute buffer
		computeBufMB := computeFloorMB
		pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality)
		if pc != nil {
			computeBufMB = pc.ComputeBufMB
		}

		// KV per layer for reserve
		kvPerLayerMB := kvTotalMB / model.NumLayers
		if kvPerLayerMB < 1 && kvTotalMB > 0 {
			kvPerLayerMB = 1
		}

		// KV-first GPU reserve: weighted by VRAM * PCIe bandwidth
		gpuKVReserveMB := kvReserveByBandwidth(kvTotalMB, caps.GPUs, nil, kvPerLayerMB)

		// Tensor split proportional to effective free VRAM * bandwidth
		split := make([]float64, numGPUs)
		gpuOrder := orderGPUsByBandwidth(caps.GPUs)
		totalWeighted := 0.0
		for idx := 0; idx < numGPUs; idx++ {
			gi := gpuOrder[idx]
			g := caps.GPUs[gi]
			overhead := cudaOverheadMB + computeBufMB + gpuKVReserveMB[gi]
			effective := g.VRAMFreeMB() - overhead
			if effective < 0 {
				effective = 0
			}
			bw := float64(g.BandwidthMBps)
			if bw <= 0 {
				bw = 1.0
			}
			totalWeighted += float64(effective) * bw
		}
		for idx := 0; idx < numGPUs; idx++ {
			gi := gpuOrder[idx]
			g := caps.GPUs[gi]
			overhead := cudaOverheadMB + computeBufMB + gpuKVReserveMB[gi]
			effective := g.VRAMFreeMB() - overhead
			if effective < 0 {
				effective = 0
			}
			bw := float64(g.BandwidthMBps)
			if bw <= 0 {
				bw = 1.0
			}
			if totalWeighted > 0 {
				split[gi] = float64(effective) * bw / totalWeighted
			}
		}
		s.TensorSplit = normalizeSplit(split)

		if opts.BackendTag == "ik_llama" {
			s.SplitMode = "layer"
		} else {
			s.SplitMode = "row"
		}
	}
	s.MMap = false
	return s, nil
}

// buildMoEOffload computes per-GPU layer caps and builds -ot flags.
// Mirrors bash MoE Phase 1 logic (lines 5453-5776).
func buildMoEOffload(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	numGPUs := len(caps.GPUs)
	gpuOrder := orderGPUsByFreeVRAM(caps.GPUs)

	// Per-layer costs (lines 5453-5454)
	expertTotalMB := int(model.ExpertBytes / 1024 / 1024)
	if expertTotalMB <= 0 {
		expertTotalMB = totalSizeMB * 90 / 100
	}
	nonExpertTotalMB := int(model.NonExpertBytes / 1024 / 1024)
	if nonExpertTotalMB <= 0 {
		nonExpertTotalMB = totalSizeMB - expertTotalMB
	}
	if nonExpertTotalMB < 0 {
		nonExpertTotalMB = 0
	}

	expertPerLayerMB := expertTotalMB / model.NumLayers
	if expertPerLayerMB <= 0 {
		expertPerLayerMB = 1
	}
	nonExpertPerLayerMB := nonExpertTotalMB / model.NumLayers
	if nonExpertPerLayerMB <= 0 {
		nonExpertPerLayerMB = 1
	}
	costPerLayerMB := expertPerLayerMB + nonExpertPerLayerMB

	// Main GPU globals (lines 5465-5466)
	mainGPUGlobalsMB := nonExpertTotalMB - nonExpertPerLayerMB*model.NumLayers/2
	if mainGPUGlobalsMB < 0 {
		mainGPUGlobalsMB = 0
	}

	// Load system probe (lines 5476-5485)
	sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
	sysCUDAOverheadMB := 0
	if sysProbe != nil {
		sysCUDAOverheadMB = sysProbe.CUDAOverheadMB
	} else if len(caps.GPUs) > 0 {
		// Fallback: estimate CUDA context overhead when no measured cache exists.
		// Typical per-GPU overhead is 500-800 MB. Use 600 MB as conservative estimate.
		// This prevents OOM on first launch before the post-launch probe creates a cache.
		sysCUDAOverheadMB = 600
	}

	// Load probe cache (lines 5487-5508)
	probeHit := false
	placementLabel := "cold-start"
	var probedComputeBufMB int

	pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality)
	if pc != nil {
		probeHit = true
		placementLabel = "probe-hit"
		probedComputeBufMB = pc.ComputeBufMB
	}

	// DET_FIXED_PER_GPU (lines 5494-5508)
	var detFixedPerGPU int
	if probeHit {
		detFixedPerGPU = sysCUDAOverheadMB + probedComputeBufMB
	} else {
		detFixedPerGPU = sysCUDAOverheadMB + computeFloorMB
	}

	// KV per layer (for reserve rounding)
	kvPerLayerMB := kvTotalMB / model.NumLayers
	if kvPerLayerMB < 1 && kvTotalMB > 0 {
		kvPerLayerMB = 1
	}

	// KV-first GPU reserve: weighted by VRAM * PCIe bandwidth (bash lines 5058-5083)
	gpuKVReserveMB := kvReserveByBandwidth(kvTotalMB, caps.GPUs, gpuOrder, kvPerLayerMB)

	// Hard ceilings: _recompute_gpu_layer_caps (lines 5521-5538)
	maxGPULayersPer := make([]int, numGPUs)
	maxGPULayers := 0
	for i := 0; i < numGPUs; i++ {
		gi := gpuOrder[i]
		overhead := detFixedPerGPU + gpuKVReserveMB[gi]
		if i == 0 {
			overhead += mainGPUGlobalsMB
		}
		usable := caps.GPUs[gi].VRAMFreeMB() - overhead
		if usable < 0 {
			usable = 0
		}
		cap := 0
		if costPerLayerMB > 0 {
			cap = usable / costPerLayerMB
		}
		if cap > model.NumLayers {
			cap = model.NumLayers
		}
		maxGPULayersPer[gi] = cap
		maxGPULayers += cap
	}
	if maxGPULayers > model.NumLayers {
		maxGPULayers = model.NumLayers
	}

	// Hard ceilings: _recompute_cpu_layer_caps (lines 5562-5588)
	kvPlacementEffective := s.KVPlacement
	if kvPlacementEffective == "auto" {
		kvPlacementEffective = "gpu"
	}
	cpuKVRAMMB := 0
	if kvPlacementEffective == "cpu" {
		cpuKVRAMMB = kvTotalMB
	}

	// Strict ceiling (--no-mmap path)
	ramOverheadPreMB := 1024 + 2048 + totalSizeMB/500 // cuda_host + graph_scratch + mmap_pt (simplified)
	cpuBudgetStrict := caps.RAM.FreeMB - ramOverheadPreMB - cpuKVRAMMB
	if cpuBudgetStrict < 0 {
		cpuBudgetStrict = 0
	}
	maxCPULayersStrict := 0
	if expertPerLayerMB > 0 {
		maxCPULayersStrict = cpuBudgetStrict / expertPerLayerMB
	}
	if maxCPULayersStrict > model.NumLayers {
		maxCPULayersStrict = model.NumLayers
	}

	// Mmap-aware ceiling
	preWorkingSetFloor := ramOverheadPreMB + cpuKVRAMMB + 8*expertPerLayerMB
	maxCPULayersMMap := 0
	if caps.RAM.FreeMB >= preWorkingSetFloor {
		maxCPULayersMMap = model.NumLayers
	}

	maxCPULayers := maxCPULayersStrict
	ceilCPULabel := "strict --no-mmap"
	if maxGPULayers+maxCPULayersStrict < model.NumLayers &&
		!opts.NoMMap &&
		maxCPULayersMMap > maxCPULayersStrict {
		maxCPULayers = maxCPULayersMMap
		ceilCPULabel = "mmap (page-cache)"
	}

	// Does-not-fit guard (lines 5655-5675)
	if maxGPULayers+maxCPULayers < model.NumLayers {
		gap := model.NumLayers - maxGPULayers - maxCPULayers
		gapVRAMMB := gap * costPerLayerMB
		gapRAMMB := gap * expertPerLayerMB
		return nil, fmt.Errorf(
			"Model does not fit on this system.\n"+
				"  Required:    %d layers\n"+
				"  GPU cap:     %d layers across %d GPU(s)\n"+
				"  CPU cap:     %d layers (%s)\n"+
				"  Gap:         %d layers — need ~%dMB more free VRAM or ~%dMB more RAM\n"+
				"\n  Options:\n"+
				"    1. Free VRAM (close other GPU workloads, --gpus to add a card)\n"+
				"    2. Drop --no-mmap so kernel can page experts on demand\n"+
				"    3. Use a smaller quantization or smaller model",
			model.NumLayers, maxGPULayers, numGPUs, maxCPULayers, ceilCPULabel,
			gap, gapVRAMMB, gapRAMMB)
	}

	// Initial layer assignment (lines 5677-5694)
	layersPerGPU := make([]int, numGPUs)
	nextLayer := 0
	totalGPULayers := 0

	for i := 0; i < numGPUs; i++ {
		gi := gpuOrder[i]
		localOverhead := detFixedPerGPU + gpuKVReserveMB[gi]
		if i == 0 {
			localOverhead += mainGPUGlobalsMB
		}
		usableMB := caps.GPUs[gi].VRAMFreeMB() - localOverhead
		if usableMB < 0 {
			usableMB = 0
		}
		layers := 0
		if costPerLayerMB > 0 {
			layers = usableMB / costPerLayerMB
		}
		remain := model.NumLayers - nextLayer
		if layers > remain {
			layers = remain
		}
		layersPerGPU[gi] = layers
		nextLayer += layers
		totalGPULayers += layers
	}

	layersCPU := model.NumLayers - totalGPULayers

	// RAM safety check and Phase 1 redistribution (lines 5708-5776)
	cpuExpertMB := layersCPU * expertPerLayerMB

	// Exact RAM overhead (lines 5709-5727)
	cudaHostMB := 1024
	graphScratchMB := 2048
	mmapPTMB := totalSizeMB / 500

	// CPU activation memory (lines 5547-5559)
	// Uses exact GGUF metadata: EXPERT_USED_COUNT * EXPERT_FF + EXPERT_SHARED_FF
	var actFFN int
	if model.NumExperts > 0 && model.ExpertUsedCount > 0 && model.ExpertFF > 0 {
		// MoE: use exact expert_ff from GGUF metadata
		actFFN = model.ExpertUsedCount * model.ExpertFF
		if model.ExpertSharedFF > 0 {
			actFFN += model.ExpertSharedFF
		}
	} else {
		actFFN = model.FeedForwardLength
	}
	if model.KVLoraRank > 0 {
		actFFN += model.KVLoraRank + model.QLoraRank
	}
	cpuActMB := s.UBatchSize * (model.EmbeddingLength + actFFN) * 4 * 2 / 1048576
	if cpuActMB < 64 {
		cpuActMB = 64
	}
	ramOverheadMB := cudaHostMB + graphScratchMB + mmapPTMB + cpuActMB

	cpuKVMB := 0
	if kvPlacementEffective == "cpu" {
		cpuKVMB = kvTotalMB
	}
	ramNeeded := cpuExpertMB + cpuKVMB + ramOverheadMB
	ramAvailMB := caps.RAM.FreeMB

	// Phase 1: If CPU layers would OOM, push layers to GPUs
	// REDIST_BUMPS is empty in bash (Phase 1 disabled: cold-start is already deterministic)
	// But we keep the logic structure for completeness
	if ramNeeded > ramAvailMB && layersCPU > 0 && numGPUs > 0 {
		redistBumps := []int{70, 80, 90, 100}
		for _, bump := range redistBumps {
			nextLayer = 0
			totalGPULayers = 0
			for i := 0; i < numGPUs; i++ {
				gi := gpuOrder[i]
				pct := bump - i*5
				if pct < 30 {
					pct = 30
				}
				localOverhead := computeFloorMB
				if i == 0 {
					localOverhead += mainGPUGlobalsMB
				}
				availableMB := caps.GPUs[gi].VRAMFreeMB() - localOverhead
				if availableMB < 0 {
					availableMB = 0
				}
				budgetMB := availableMB * pct / 100
				layers := 0
				if costPerLayerMB > 0 {
					layers = budgetMB / costPerLayerMB
					// Hard-max floor: on final bump pass, allow up to raw hard-fit count
					if layers == 0 && bump == redistBumps[len(redistBumps)-1] {
						layers = availableMB / costPerLayerMB
					}
				}
				remain := model.NumLayers - nextLayer
				if layers > remain {
					layers = remain
				}
				layersPerGPU[gi] = layers
				nextLayer += layers
				totalGPULayers += layers
			}
			layersCPU = model.NumLayers - totalGPULayers
			cpuExpertMB = layersCPU * expertPerLayerMB
			ramNeeded = cpuExpertMB + cpuKVMB + ramOverheadMB
			if ramNeeded <= ramAvailMB {
				break
			}
		}
	}

	// Mmap decision for MoE (lines 5562-5588)
	// If strict ceiling doesn't fit and user didn't force --no-mmap,
	// use mmap (page-cache) if working set fits
	if totalSizeMB > ramAvailMB {
		s.MMap = true
	} else if ramNeeded > ramAvailMB {
		workingSetFloor := ramOverheadMB + cpuKVMB + 8*expertPerLayerMB
		if ramAvailMB >= workingSetFloor {
			s.MMap = true
		}
	} else {
		s.MMap = false
	}

	// Build -ot string (lines 5876-5891)
	if totalGPULayers > 0 {
		otString := buildOTString(layersPerGPU, caps.GPUs, gpuOrder)
		if otString != "" {
			s.OTString = otString
		}
	}

	// NCPUMoE = number of experts to keep on CPU
	if layersCPU > 0 {
		s.NCPUMoE = model.NumExperts
	}

	_ = placementLabel // used for logging in bash
	_ = maxGPULayersPer
	_ = ceilCPULabel

	return s, nil
}

// buildOTString builds the -ot override-tensor string for MoE.
// Matches bash build_ot_string() exactly: explicit layer list with escaped dots.
func buildOTString(layersPerGPU []int, gpus []detect.GPU, gpuOrder []int) string {
	var parts []string
	expertPattern := `ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp)`

	nextLayer := 0
	for _, gi := range gpuOrder {
		count := layersPerGPU[gi]
		if count > 0 {
			start := nextLayer
			last := start + count - 1
			cudaIdx := gpus[gi].Index
			// Build explicit layer list like bash: 0|1|2|...|31
			var layerParts []string
			for l := start; l <= last; l++ {
				layerParts = append(layerParts, fmt.Sprintf("%d", l))
			}
			layerRange := stringsJoin(layerParts, "|")
			parts = append(parts, fmt.Sprintf(`blk\.(%s)\.%s.*=CUDA%d`, layerRange, expertPattern, cudaIdx))
			nextLayer += count
		}
	}
	parts = append(parts, "exps=CPU")

	return stringsJoin(parts, ",")
}

func buildOTStringFromAssignments(assignments []GPUAssignment, gpus []detect.GPU, numLayers int) string {
	var parts []string
	expertPattern := `ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp)`

	nextLayer := 0
	for _, assign := range assignments {
		if assign.Count <= 0 {
			continue
		}
		start := assign.Start
		last := start + assign.Count - 1
		var layerParts []string
		for l := start; l <= last; l++ {
			layerParts = append(layerParts, fmt.Sprintf("%d", l))
		}
		layerRange := stringsJoin(layerParts, "|")
		parts = append(parts, fmt.Sprintf(`blk\.(%s)\.%s.*=CUDA%d`, layerRange, expertPattern, assign.CUDAIndex))
		nextLayer += assign.Count
	}
	parts = append(parts, "exps=CPU")
	return stringsJoin(parts, ",")
}

func stringsJoin(parts []string, sep string) string {
	if len(parts) == 0 {
		return ""
	}
	result := parts[0]
	for i := 1; i < len(parts); i++ {
		result += sep + parts[i]
	}
	return result
}

// computeKVTotalMB calculates exact KV cache size.
func computeKVTotalMB(model *ModelProfile, ctxSize int, kvType string) int {
	var kvElemsTotal int

	hasMLA := model.KVLoraRank > 0
	hasSSM := model.HasSSM == 1
	hasISWA := model.SlidingWindow > 0

	if hasMLA {
		// MLA: compressed c^{KV} + RoPE'd key once per layer
		kvElemsTotal = model.NumLayers * ctxSize * (model.KVLoraRank + model.RopeDim)
	} else if hasSSM {
		var attnLayers int
		if model.FullAttnInterval > 0 {
			attnLayers = model.NumLayers / model.FullAttnInterval
			if attnLayers < 1 {
				attnLayers = 1
			}
		} else if model.HeadCountKV == 0 {
			attnLayers = 0
		} else {
			attnLayers = (model.NumLayers + 1) / 2
		}
		kvBytesPerLayerPerToken := model.HeadCountKV * (model.KeyLength + model.ValueLength)
		kvElemsTotal = attnLayers * ctxSize * kvBytesPerLayerPerToken
	} else if hasISWA {
		swaPeriod := 6
		switch model.ModelArch {
		case "gemma2", "cohere2", "exaone4", "llama4":
			swaPeriod = 4
		case "gemma3":
			swaPeriod = 6
		case "plamo3":
			swaPeriod = 8
		}
		fullLayers := (model.NumLayers + swaPeriod - 1) / swaPeriod
		swaLayers := model.NumLayers - fullLayers
		swaCtx := ctxSize
		if swaCtx > model.SlidingWindow {
			swaCtx = model.SlidingWindow
		}
		kvBytesPerLayerPerToken := model.HeadCountKV * (model.KeyLength + model.ValueLength)
		kvElemsTotal = fullLayers*ctxSize*kvBytesPerLayerPerToken + swaLayers*swaCtx*kvBytesPerLayerPerToken
	} else {
		// Standard GQA/MQA
		kvBytesPerLayerPerToken := model.HeadCountKV * (model.KeyLength + model.ValueLength)
		kvElemsTotal = model.NumLayers * ctxSize * kvBytesPerLayerPerToken
	}

	var bytesPerElem float64
	switch kvType {
	case "q4_0":
		bytesPerElem = 0.5625
	case "q8_0":
		bytesPerElem = 1.0625
	case "f16":
		bytesPerElem = 2.0
	default:
		bytesPerElem = 1.0625 // q8_0 default
	}

	return int(float64(kvElemsTotal) * bytesPerElem / 1024 / 1024)
}

func kvTypeFromQuality(quality string) string {
	switch quality {
	case "high":
		return "f16"
	case "mid":
		return "q8_0"
	case "low":
		return "q4_0"
	default:
		return "q8_0"
	}
}

func orderGPUsByBandwidth(gpus []detect.GPU) []int {
	indices := make([]int, len(gpus))
	for i := range gpus {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		gi := gpus[indices[i]]
		gj := gpus[indices[j]]
		// Primary: bandwidth DESC
		if gi.BandwidthMBps != gj.BandwidthMBps {
			return gi.BandwidthMBps > gj.BandwidthMBps
		}
		// Secondary: VRAM total DESC
		if gi.VRAMTotalMB != gj.VRAMTotalMB {
			return gi.VRAMTotalMB > gj.VRAMTotalMB
		}
		// Tertiary: PCI index ASC (lower index = closer to CPU)
		return gi.Index < gj.Index
	})
	return indices
}

// kvReserveByBandwidth distributes total KV cache across GPUs weighted by
// free VRAM * PCIe bandwidth. Faster GPUs get more KV cache (matches bash).
func kvReserveByBandwidth(kvTotalMB int, gpus []detect.GPU, order []int, kvPerLayerMB int) []int {
	reserve := make([]int, len(gpus))
	var totalWeighted int64
	for _, g := range gpus {
		bw := g.BandwidthMBps
		if bw <= 0 { bw = 1 }
		totalWeighted += int64(g.VRAMFreeMB()) * int64(bw)
	}
	if kvTotalMB <= 0 || totalWeighted <= 0 {
		return reserve
	}
	useOrder := order
	if len(useOrder) == 0 {
		useOrder = seqRange(len(gpus))
	}
	for _, gi := range useOrder {
		bw := gpus[gi].BandwidthMBps
		if bw <= 0 { bw = 1 }
		w := int64(gpus[gi].VRAMFreeMB()) * int64(bw)
		share := int((int64(kvTotalMB)*w + totalWeighted - 1) / totalWeighted)
		if kvPerLayerMB > 0 {
			share = ((share + kvPerLayerMB - 1) / kvPerLayerMB) * kvPerLayerMB
		}
		reserve[gi] = share
	}
	return reserve
}

func seqRange(n int) []int {
	r := make([]int, n)
	for i := range r { r[i] = i }
	return r
}

func orderGPUsByFreeVRAM(gpus []detect.GPU) []int {
	indices := make([]int, len(gpus))
	for i := range gpus {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		vi := gpus[indices[i]].VRAMFreeMB()
		vj := gpus[indices[j]].VRAMFreeMB()
		if vi == vj {
			return gpus[indices[i]].Index < gpus[indices[j]].Index
		}
		return vi > vj
	})
	return indices
}

func normalizeSplit(split []float64) []float64 {
	var total float64
	for _, v := range split {
		total += v
	}
	if total == 0 {
		return split
	}
	for i := range split {
		split[i] = math.Round(split[i]/total*100) / 100
	}
	return split
}

// checkMemoryOrDie mirrors bash check_memory_or_die() (lines 2620-2654).
// OOM guard: refuse to launch if model+KV+compute don't fit.
func checkMemoryOrDie(caps *detect.Capabilities, model *ModelProfile, s *Strategy, totalSizeMB, kvTotalMB int, opts Options) error {
	numGPUs := len(caps.GPUs)

	// Load system probe for CUDA overhead (measured, not guessed)
	sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
	cudaOverheadMB := 0
	if sysProbe != nil {
		cudaOverheadMB = sysProbe.CUDAOverheadMB
	} else if numGPUs > 0 {
		cudaOverheadMB = 600 // conservative estimate per GPU
	}

	// Load model probe for compute buffer
	computeBufMB := computeFloorMB // 1024 default
	pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality)
	if pc != nil {
		computeBufMB = pc.ComputeBufMB
	}

	// Model weights + per-GPU overhead (CUDA context + compute buffer)
	// For single GPU, only count 1 GPU's overhead
	overheadGPUs := numGPUs
	if s.Type == SingleGPU {
		overheadGPUs = 1
	}
	modelOverheadMB := totalSizeMB + (cudaOverheadMB+computeBufMB)*overheadGPUs
	neededMB := modelOverheadMB + kvTotalMB

	var poolMB int
	var poolLabel string

	switch s.Type {
	case SingleGPU:
		// Best GPU's free VRAM
		bestFree := 0
		for _, g := range caps.GPUs {
			if g.VRAMFreeMB() > bestFree {
				bestFree = g.VRAMFreeMB()
			}
		}
		poolMB = bestFree
		poolLabel = "best GPU"
	case MultiGPUDense:
		// Total free VRAM across all GPUs
		for _, g := range caps.GPUs {
			poolMB += g.VRAMFreeMB()
		}
		poolLabel = "all GPUs"
	case CPUOnly:
		poolMB = caps.RAM.FreeMB
		poolLabel = "RAM"
	case DenseCPUOffload:
		// Total system memory (GPU + RAM) since model is split
		for _, g := range caps.GPUs {
			poolMB += g.VRAMFreeMB()
		}
		poolMB += caps.RAM.FreeMB
		poolLabel = "system memory"
	}

	if neededMB > poolMB {
		// Back-solve max safe context
		maxKVMB := poolMB - modelOverheadMB
		if maxKVMB < 0 {
			maxKVMB = 0
		}
		maxCtx := 0
		if kvTotalMB > 0 {
			maxCtx = maxKVMB * s.ContextSize / kvTotalMB
		}

		msg := fmt.Sprintf(
			"ERROR: Model does not fit in %s.\n"+
				"  Model weights:          %dMB\n"+
				"  CUDA overhead (%d GPU): %dMB\n"+
				"  Compute buffers (%d):   %dMB\n"+
				"  KV cache (ctx=%d):      %dMB\n"+
				"  -----------------------------\n"+
				"  Total needed:          %dMB\n"+
				"  Available (%s):        %dMB\n"+
				"  Shortfall:             %dMB\n",
			poolLabel, totalSizeMB, numGPUs, cudaOverheadMB*numGPUs,
			numGPUs, computeBufMB*numGPUs, s.ContextSize, kvTotalMB,
			neededMB, poolLabel, poolMB, neededMB-poolMB)

		if maxCtx > 0 {
			msg += fmt.Sprintf("\n  Max safe context at this memory: --ctx-size %d", maxCtx)
		} else {
			msg += "\n  Model weights alone exceed available memory."
		}
		msg += "\n  Or use a smaller quantization / model."
		return fmt.Errorf("%s", msg)
	}
	return nil
}

// computeCRAM calculates prompt cache size from remaining memory after load.
// Mirrors bash CRAM logic (lines 2009-2211, 2319-2342).
func computeCRAM(caps *detect.Capabilities, model *ModelProfile, s *Strategy, totalSizeMB, kvTotalMB int) (int, int) {
	numGPUs := len(caps.GPUs)

	// Fits on GPU? (model fits entirely in VRAM)
	fitsOnGPU := false
	switch s.Type {
	case SingleGPU, MultiGPUDense:
		fitsOnGPU = true
	}

	// RAM_AFTER_LOAD (lines 2009-2017)
	var ramAfterLoad int
	if fitsOnGPU {
		ramAfterLoad = caps.RAM.FreeMB
	} else {
		ramOnCPU := totalSizeMB - caps.TotalVRAM()
		if ramOnCPU < 0 {
			ramOnCPU = 0
		}
		ramAfterLoad = caps.RAM.FreeMB - ramOnCPU
		if ramAfterLoad < 0 {
			ramAfterLoad = 0
		}
	}

	// Single-GPU / CPU-only CRAM (lines 2202-2204)
	cram := ramAfterLoad / 10
	if cram > 16384 {
		cram = 16384
	}
	if cram < minCramMB {
		cram = 0
	}

	maxCheckpoints := 0

	// Multi-GPU CRAM (lines 2319-2342)
	if numGPUs > 1 && s.Type != CPUOnly {
		// TOTAL_VRAM_MB = sum of FREE VRAM
		totalFreeVRAM := 0
		for _, g := range caps.GPUs {
			totalFreeVRAM += g.VRAMFreeMB()
		}
		modelOnGPUMB := totalSizeMB * vramOverheadPercent / 100
		if modelOnGPUMB > totalFreeVRAM {
			modelOnGPUMB = totalFreeVRAM
		}
		vramHeadroom := totalFreeVRAM - modelOnGPUMB - kvTotalMB - computePerGPUMB*numGPUs
		if vramHeadroom < 0 {
			vramHeadroom = 0
		}
		cacheRAMMB := vramHeadroom / 2
		if cacheRAMMB > 4096 {
			cacheRAMMB = 4096
		}
		if cacheRAMMB < 256 {
			cacheRAMMB = 0
			maxCheckpoints = 0
		} else {
			maxCheckpoints = cacheRAMMB / 200
			if maxCheckpoints < 2 {
				maxCheckpoints = 2
			}
			if maxCheckpoints > 16 {
				maxCheckpoints = 16
			}
		}
		cram = cacheRAMMB
	}

	return cram, maxCheckpoints
}

func defaultContextSize(model *ModelProfile, caps *detect.Capabilities) int {
	if model.ContextSize > 0 && model.ContextSize < 32768 {
		return model.ContextSize
	}
	return 32768
}

// computeAutoContextSizeSingleGPU computes the largest context that fits on
// a SINGLE GPU (the best one). Used to prefer single-GPU mode (faster).
// Matches bash max_context_fit() (lines 2186-2265).
func computeAutoContextSizeSingleGPU(caps *detect.Capabilities, model *ModelProfile, totalSizeMB int, preferredKVType string, opts Options) (int, string) {
	// Find best single GPU by total VRAM
	bestVRAM := 0
	for _, g := range caps.GPUs {
		if g.VRAMTotalMB > bestVRAM {
			bestVRAM = g.VRAMTotalMB
		}
	}

	// Total hardware = best GPU VRAM + free RAM (matches bash)
	totalHWMB := bestVRAM + caps.RAM.FreeMB

	// Fixed overhead: model weights + 8GB headroom (bash: TOTAL_SIZE_MB + 8192)
	fixedOverheadMB := totalSizeMB + 8192

	// If model doesn't fit at all, return minimum
	if totalHWMB <= fixedOverheadMB {
		return 32768, preferredKVType
	}

	// KV budget = total hardware - model - headroom
	kvBudgetMB := totalHWMB - fixedOverheadMB
	if kvBudgetMB <= 0 {
		return 32768, preferredKVType
	}

	// Try KV types in order
	kvTypes := []string{preferredKVType, "q8_0", "q4_0"}
	seen := make(map[string]bool)
	var orderedTypes []string
	for _, t := range kvTypes {
		if !seen[t] {
			seen[t] = true
			orderedTypes = append(orderedTypes, t)
		}
	}

	for _, kvType := range orderedTypes {
		refCtx := 32768
		refKVTotalMB := computeKVTotalMB(model, refCtx, kvType)
		if refKVTotalMB <= 0 {
			continue
		}
		kvBytesPerToken := float64(refKVTotalMB) * 1048576.0 / float64(refCtx)
		maxCtxRaw := int(float64(kvBudgetMB) * 1048576.0 / kvBytesPerToken)

		hwCapCtx := maxCtxRaw
		if model.CTXTrain > 0 && model.CTXTrain < hwCapCtx {
			hwCapCtx = model.CTXTrain
		}

		powerOfTwoValues := []int{32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304}
		suggestedCtx := 32768
		for _, c := range powerOfTwoValues {
			if c <= hwCapCtx {
				suggestedCtx = c
			}
		}

		if suggestedCtx >= 32768 {
			return suggestedCtx, kvType
		}
	}

	return 32768, "q4_0"
}

// computeAutoContextSize computes the largest context that fits in available
// hardware memory, mirroring bash max_context_fit() (lines 2186-2265).
// Uses TOTAL_VRAM + RAM_AVAIL (matches bash exactly).
func computeAutoContextSize(caps *detect.Capabilities, model *ModelProfile, totalSizeMB int, preferredKVType string, opts Options) (int, string) {
	// Total hardware = all GPU VRAM + free RAM
	totalVRAM := 0
	for _, g := range caps.GPUs {
		totalVRAM += g.VRAMTotalMB
	}
	totalHWMB := totalVRAM + caps.RAM.FreeMB

	// Fixed overhead: model weights + 8GB headroom (bash: TOTAL_SIZE_MB + 8192)
	fixedOverheadMB := totalSizeMB + 8192

	// If model doesn't fit at all, return minimum
	if totalHWMB <= fixedOverheadMB {
		return 32768, preferredKVType
	}

	// KV budget = total hardware - model - headroom
	kvBudgetMB := totalHWMB - fixedOverheadMB
	if kvBudgetMB <= 0 {
		return 32768, preferredKVType
	}

	// Try KV types in order: preferred, then q8_0, then q4_0
	kvTypes := []string{preferredKVType, "q8_0", "q4_0"}
	seen := make(map[string]bool)
	var orderedTypes []string
	for _, t := range kvTypes {
		if !seen[t] {
			seen[t] = true
			orderedTypes = append(orderedTypes, t)
		}
	}

	for _, kvType := range orderedTypes {
		refCtx := 32768
		refKVTotalMB := computeKVTotalMB(model, refCtx, kvType)
		if refKVTotalMB <= 0 {
			continue
		}
		kvBytesPerToken := float64(refKVTotalMB) * 1048576.0 / float64(refCtx)
		maxCtxRaw := int(float64(kvBudgetMB) * 1048576.0 / kvBytesPerToken)

		hwCapCtx := maxCtxRaw
		if model.CTXTrain > 0 && model.CTXTrain < hwCapCtx {
			hwCapCtx = model.CTXTrain
		}

		powerOfTwoValues := []int{32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304}
		suggestedCtx := 32768
		for _, c := range powerOfTwoValues {
			if c <= hwCapCtx {
				suggestedCtx = c
			}
		}

		if suggestedCtx >= 32768 {
			return suggestedCtx, kvType
		}
	}

	// Nothing fits well — return minimum with most compact type
	return 32768, "q4_0"
}

// Args converts a Strategy into llama-server command-line arguments.
func (s *Strategy) Args(modelPath string, port int) []string {
	host := s.Host
	if host == "" {
		host = "0.0.0.0"
	}
	args := []string{
		"-m", modelPath,
		"--host", host,
		"--port", fmt.Sprintf("%d", port),
		"--ctx-size", fmt.Sprintf("%d", s.ContextSize),
		"--flash-attn", "on",
		"-b", fmt.Sprintf("%d", s.BatchSize),
		"-ub", fmt.Sprintf("%d", s.UBatchSize),
		"--cache-type-k", s.KVType,
		"--cache-type-v", s.KVType,
		"--jinja",
		"--threads", fmt.Sprintf("%d", s.Threads),
		"--threads-batch", fmt.Sprintf("%d", s.ThreadsBatch),
	}

	if s.KVPlacement == "cpu" {
		args = append(args, "--no-kv-offload")
	} else if s.KVPlacement == "gpu" {
		args = append(args, "--kv-offload")
	}

	if s.ReasoningOff {
		args = append(args, "--reasoning", "off")
	}

	// SSM/Mamba models need --no-context-shift (bash line 2029-2033)
	if s.HasSSM {
		args = append(args, "--no-context-shift")
	}

	if s.Parallel > 0 {
		args = append(args, "--parallel", fmt.Sprintf("%d", s.Parallel))
	}

	// GPU offloading: ALWAYS -ngl 999
	if len(s.TensorSplit) > 0 || s.Type != CPUOnly {
		args = append(args, "-ngl", "999")
		if s.MainGPU >= 0 && len(s.TensorSplit) == 0 {
			args = append(args, "-mg", fmt.Sprintf("%d", s.MainGPU))
		}
	}

	if len(s.TensorSplit) > 0 {
		var splitStr string
		for i, v := range s.TensorSplit {
			if i > 0 {
				splitStr += ","
			}
			splitStr += fmt.Sprintf("%.2f", v)
		}
		args = append(args, "--tensor-split", splitStr)
	}

	if s.SplitMode != "" {
		args = append(args, "--split-mode", s.SplitMode)
	}

	if s.OTString != "" {
		args = append(args, "-ot", s.OTString)
	}

	if s.NCPUMoE > 0 {
		args = append(args, "--n-cpu-moe", fmt.Sprintf("%d", s.NCPUMoE))
	}

	if !s.MMap {
		args = append(args, "--no-mmap")
	}

	if s.MLock {
		args = append(args, "--mlock")
	}

	if s.CRAM > 0 {
		args = append(args, "-cram", fmt.Sprintf("%d", s.CRAM))
		if s.MaxCheckpoints > 0 {
			args = append(args, "--ctx-checkpoints", fmt.Sprintf("%d", s.MaxCheckpoints))
		}
	}

	// ik_llama.cpp fork specific flags
	if s.BackendTag == "ik_llama" {
		args = append(args, "--run-time-repack")
		args = append(args, "-khad")
		args = append(args, "--defrag-thold", "0.1")

		if s.IsMoE {
			args = append(args, "-muge")
			args = append(args, "-ger")
		}

		if len(s.TensorSplit) > 0 || s.Type != CPUOnly {
			args = append(args, "-mqkv")
		}
	}

	// Speculative decoding flags (draft model or ngram)
	if s.Draft != nil && s.Draft.Type != DraftNone {
		args = append(args, DraftFlags(s.Draft)...)
	}

	return args
}

// loadSystemProbe tries to load measured CUDA overhead from cache.
// Uses GPU signature hash matching bash system_probe cache (lines 5216-5228).
func loadSystemProbe(cacheDir string, gpus []detect.GPU) *systemProbe {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "llm-server")
	}
	// Compute GPU signature hash: sort(names+drivers), MD5, take first 12 chars
	gpuSig := gpuSignatureHash(gpus)
	path := filepath.Join(cacheDir, fmt.Sprintf("system_%s.cache", gpuSig))
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	sp := &systemProbe{}
	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(strings.Trim(parts[1], `"`))
		if key == "SYS_CUDA_OVERHEAD_MB" {
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				sp.CUDAOverheadMB = v
			}
		}
	}
	if sp.CUDAOverheadMB == 0 {
		return nil
	}
	return sp
}

// gpuSignatureHash computes MD5 hash of sorted GPU name+driver pairs.
// Matches bash: nvidia-smi --query-gpu=name,driver_version | sort | md5sum | cut -c1-12
func gpuSignatureHash(gpus []detect.GPU) string {
	var parts []string
	for _, g := range gpus {
		parts = append(parts, fmt.Sprintf("%s, %s", g.Name, g.Driver))
	}
	sort.Strings(parts)
	input := strings.Join(parts, "\n") + "\n"
	h := md5.New()
	h.Write([]byte(input))
	return fmt.Sprintf("%x", h.Sum(nil))[:12]
}

// RunPostLaunchProbe measures actual CUDA overhead after a successful server launch.
// It reads current VRAM usage from nvidia-smi, parses buffer sizes from the
// server's captured stderr log, and caches the result for future launches.
// Matches bash post-launch probe (lines 6174-6228).
func RunPostLaunchProbe(cacheDir string, gpus []detect.GPU, serverLog string) {
	if len(gpus) == 0 || serverLog == "" {
		return
	}
	// Only write if no cached value exists
	sp := loadSystemProbe(cacheDir, gpus)
	if sp != nil && sp.CUDAOverheadMB > 0 {
		return
	}

	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "llm-server")
	}

	// Use primary GPU (first in list, sorted by PCI bus ID)
	primary := gpus[0]

	// Get current VRAM used on primary GPU via nvidia-smi
	primaryUsedMB := queryVRAMUsed(primary.Index)
	if primaryUsedMB <= 0 {
		return
	}

	// Parse buffer sizes from server log
	modelBufMB, kvBufMB, computeBufMB := parseBuffersFromLog(serverLog, primary.Index)
	accounted := modelBufMB + kvBufMB + computeBufMB

	cudaOverhead := primaryUsedMB - accounted
	if cudaOverhead <= 0 || cudaOverhead >= primaryUsedMB {
		return
	}

	// Write system probe cache
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return
	}
	gpuSig := gpuSignatureHash(gpus)
	path := filepath.Join(cacheDir, fmt.Sprintf("system_%s.cache", gpuSig))
	driver := gpus[0].Driver
	if driver == "" {
		driver = "unknown"
	}
	content := fmt.Sprintf(
		"# System probe (post-launch measurement) for %s\n"+
			"# Driver: %s\n"+
			"# Generated: %s\n"+
			"# Measurements: VRAM_used=%dMB model_buf=%dMB kv_buf=%dMB compute_buf=%dMB\n"+
			"SYS_CUDA_OVERHEAD_MB=%d\n",
		primary.Name, driver, time.Now().Format(time.RFC3339),
		primaryUsedMB, modelBufMB, kvBufMB, computeBufMB, cudaOverhead,
	)
	if err := os.WriteFile(path, []byte(content), 0644); err == nil {
		fmt.Fprintf(os.Stderr, "  System probe written: cuda_overhead=%dMB (measured from launch)\n", cudaOverhead)
	}
}

// queryVRAMUsed returns current nvidia-smi memory.used for a given GPU index.
func queryVRAMUsed(gpuIndex int) int {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=memory.used", "--format=csv,noheader,nounits",
		"-i", fmt.Sprintf("%d", gpuIndex),
	).Output()
	if err != nil {
		return 0
	}
	s := strings.TrimSpace(string(out))
	v, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return v
}

// parseBuffersFromLog parses llama-server log for CUDA buffer sizes on a specific GPU.
// Returns modelBufMB, kvBufMB, computeBufMB.
// Handles both mainline ("CUDAN model buffer size = X MiB") and
// ik_llama ("CUDAN buffer size = X MiB") formats.
func parseBuffersFromLog(log string, gpuIndex int) (modelBufMB, kvBufMB, computeBufMB int) {
	cudaTag := fmt.Sprintf("CUDA%d", gpuIndex)

	var maxModelBuf, maxComputeBuf float64
	var totalKVBuf float64
	var kvCount int

	lines := strings.Split(log, "\n")
	for _, line := range lines {
		if !strings.Contains(line, cudaTag) {
			continue
		}

		// Model buffer: "CUDA0 model buffer size = X MiB" or "CUDA0 buffer size = X MiB"
		if strings.Contains(line, "buffer size =") && !strings.Contains(line, "KV") && !strings.Contains(line, "compute") {
			if v := parseMiB(line); v > maxModelBuf {
				maxModelBuf = v
			}
		}

		// KV buffer: "CUDA0 KV buffer size = X MiB"
		if strings.Contains(line, "KV buffer size =") {
			if v := parseMiB(line); v > 0 {
				totalKVBuf += v
				kvCount++
			}
		}

		// Compute buffer: "CUDA0 compute buffer size = X MiB"
		if strings.Contains(line, "compute buffer size =") {
			if v := parseMiB(line); v > maxComputeBuf {
				maxComputeBuf = v
			}
		}
	}

	modelBufMB = int(maxModelBuf + 0.5)
	computeBufMB = int(maxComputeBuf + 0.5)
	if totalKVBuf > 0 && kvCount > 0 {
		kvBufMB = int(totalKVBuf/float64(kvCount) + 0.5)
	}
	return
}

// parseMiB extracts a floating-point MiB value from a log line containing "X MiB".
func parseMiB(line string) float64 {
	idx := strings.LastIndex(line, "=")
	if idx < 0 {
		return 0
	}
	rest := strings.TrimSpace(line[idx+1:])
	rest = strings.TrimSuffix(rest, " MiB")
	rest = strings.TrimSpace(rest)
	v, err := strconv.ParseFloat(rest, 64)
	if err != nil {
		return 0
	}
	return v
}

// loadProbeCache tries to load per-model probe data.
// Uses MD5 hash key matching bash probe_cache_file() (lines 5193-5207).
func loadProbeCache(cacheDir string, model *ModelProfile, ctxSize int, ubatch int, kvQuality string) *probeCache {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "llm-server", "probes")
	}
	// Compute MD5 hash key matching bash: model_name:layers:experts:embd:ff:ctx:ubatch:kv_quality
	modelName := filepath.Base(model.Path)
	key := fmt.Sprintf("%s:%d:%d:%d:%d:%d:%d:%s",
		modelName, model.NumLayers, model.NumExperts,
		model.EmbeddingLength, model.FeedForwardLength,
		ctxSize, ubatch, kvQuality)
	hash := md5Hash12(key)
	path := filepath.Join(cacheDir, hash+".probe")

	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	content := string(data)
	pc := &probeCache{}
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		k := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(strings.Trim(parts[1], `"`))
		switch k {
		case "PROBED_COMPUTE_BUF_MB":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				pc.ComputeBufMB = v
			}
		case "PROBED_KV_PER_LAYER_MB":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				pc.KVPerLayerMB = v
			}
		}
	}
	if pc.ComputeBufMB > 0 || pc.KVPerLayerMB > 0 {
		return pc
	}
	return nil
}

// md5Hash12 computes first 12 chars of MD5 hash of input string.
func md5Hash12(input string) string {
	h := md5.New()
	h.Write([]byte(input))
	return fmt.Sprintf("%x", h.Sum(nil))[:12]
}

// internal types for probe loading
type systemProbe struct {
	CUDAOverheadMB int
}

type probeCache struct {
	ComputeBufMB int
	KVPerLayerMB int
}

// WriteProbeCache writes measured compute buffer and KV per layer to cache.
// Mirrors bash write_probe_cache() (lines 5398-5437).
func WriteProbeCache(cacheDir, modelName string, computeBufMB, kvPerLayerMB int) error {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "llm-server", "probes")
	}
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return err
	}
	// Sanitize model name for filename
	safeName := strings.ReplaceAll(modelName, "/", "_")
	path := filepath.Join(cacheDir, safeName+".probe")
	content := fmt.Sprintf(
		"# Probe cache for %s\n"+
		"# Generated: %s\n"+
		"PROBED_COMPUTE_BUF_MB=%d\n"+
		"PROBED_KV_PER_LAYER_MB=%d\n",
		modelName, time.Now().Format(time.RFC3339), computeBufMB, kvPerLayerMB,
	)
	return os.WriteFile(path, []byte(content), 0644)
}

// ParseLogForProbe extracts compute_buf and kv_per_layer from server log output.
// Looks for lines like: "CUDA0 compute buffer size = 1410.12 MiB"
func ParseLogForProbe(logData string) (computeBufMB, kvPerLayerMB int) {
	lines := strings.Split(logData, "\n")

	// Find largest compute buffer across CUDA devices
	var maxComputeBuf float64
	for _, line := range lines {
		// Pattern: "CUDA0 compute buffer size = 1410.12 MiB"
		if idx := strings.Index(line, "compute buffer size ="); idx >= 0 {
			rest := line[idx+len("compute buffer size ="):]
			rest = strings.TrimSpace(rest)
			rest = strings.TrimSuffix(rest, " MiB")
			if v, err := strconv.ParseFloat(rest, 64); err == nil && v > maxComputeBuf {
				maxComputeBuf = v
			}
		}
	}

	// Sum KV buffer sizes across CUDA devices
	var totalKVBuf float64
	var kvCount int
	for _, line := range lines {
		if idx := strings.Index(line, "KV buffer size ="); idx >= 0 {
			rest := line[idx+len("KV buffer size ="):]
			rest = strings.TrimSpace(rest)
			rest = strings.TrimSuffix(rest, " MiB")
			if v, err := strconv.ParseFloat(rest, 64); err == nil {
				totalKVBuf += v
				kvCount++
			}
		}
	}

	if maxComputeBuf > 0 {
		computeBufMB = int(maxComputeBuf + 0.5)
	}
	if totalKVBuf > 0 && kvCount > 0 {
		// Bash divides by GPU layer count; we approximate with kvCount (devices with KV)
		kvPerLayerMB = int(totalKVBuf/float64(kvCount) + 0.5)
	}

	return
}
