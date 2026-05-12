package placement

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/raketenkater/llm-server/pkg/detect"
)

// Bash llm-server constants (lines 260-279)
const (
	vramOverheadPercent = 130  // model size * this / 100 = estimated VRAM needed
	computePerGPUMB     = 512  // legacy; non-MoE single-GPU sizing only
	computeFloorMB      = 1024 // cited from llama.cpp common/common.cpp
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
	Parallel       int          `json:"parallel,omitempty"`
	CRAM           int          `json:"cram,omitempty"`        // prompt cache MB
	MaxCheckpoints int          `json:"max_checkpoints,omitempty"`
	UseCUDAGraphs  bool         `json:"use_cuda_graphs,omitempty"`
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
}

// Compute builds a Strategy from hardware capabilities and model profile.
func Compute(caps *detect.Capabilities, model *ModelProfile, opts Options) (*Strategy, error) {
	s := &Strategy{
		ContextSize:    opts.ContextSize,
		KVPlacement:    opts.KVPlacement,
		KVQuality:      opts.KVQuality,
		MMap:           !opts.NoMMap,
		MLock:          false,
		Threads:        caps.CPU.Cores,
		BackendTag:     opts.BackendTag,
		IsMoE:          model.IsMoE,
		GPULayers:      999,
		FlashAttention: true,
		ReasoningOff:   false,
	}

	if s.ContextSize <= 0 {
		s.ContextSize = defaultContextSize(model, caps)
	}
	if opts.KVPlacement == "" {
		s.KVPlacement = "auto"
	}
	if opts.KVQuality == "" {
		s.KVQuality = "mid"
	}
	if opts.Parallel > 0 {
		s.Parallel = opts.Parallel
	}

	// KV cache type selection
	s.KVType = kvTypeFromQuality(s.KVQuality)

	// Total size MB
	totalSizeMB := model.TotalSizeMB
	if totalSizeMB <= 0 {
		totalSizeMB = int(model.SizeBytes / 1024 / 1024)
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
		if err := checkMemoryOrDie(caps, model, s, totalSizeMB, kvTotalMB); err != nil {
			return nil, err
		}
	}

	// Compute CRAM (prompt cache)
	s.CRAM, s.MaxCheckpoints = computeCRAM(caps, model, s, totalSizeMB, kvTotalMB)

	return s, nil
}

// chooseStrategy mirrors bash choose_strategy() exactly (lines 2703-2739).
func chooseStrategy(caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) StrategyType {
	numGPUs := len(caps.GPUs)

	if opts.CPUMode || numGPUs == 0 {
		return CPUOnly
	}

	// Single GPU: model + overhead fits in best GPU
	// Use TOTAL VRAM (minus 1GB safety), not FREE — pre-launch free VRAM dips
	// from desktop/compositor would spuriously reject single-GPU.
	bestVRAM := 0
	for _, g := range caps.GPUs {
		if g.VRAMTotalMB > bestVRAM {
			bestVRAM = g.VRAMTotalMB
		}
	}
	singleGPUUsable := bestVRAM - 1024
	if singleGPUUsable < 0 {
		singleGPUUsable = 0
	}
	// Tighter estimate (110%) for single GPU
	singleGPUNeeded := totalSizeMB*110/100 + kvTotalMB + computePerGPUMB
	if singleGPUNeeded <= singleGPUUsable {
		return SingleGPU
	}

	// Multi-GPU dense: model fits across ALL GPUs (sum of FREE VRAM)
	if !model.IsMoE {
		totalFreeVRAM := 0
		for _, g := range caps.GPUs {
			totalFreeVRAM += g.VRAMFreeMB()
		}
		vramNeeded := totalSizeMB*vramOverheadPercent/100 + kvTotalMB + computePerGPUMB
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
	gpuOrder := orderGPUsByVRAM(caps.GPUs)
	mainIdx := 0
	if len(gpuOrder) > 0 {
		mainIdx = gpuOrder[0]
	}
	s.MainGPU = caps.GPUs[mainIdx].Index
	return s, nil
}

func buildMultiGPUDense(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	numGPUs := len(caps.GPUs)

	// Tensor split across ALL GPUs proportional to VRAM
	split := make([]float64, numGPUs)
	totalVRAM := 0
	for _, g := range caps.GPUs {
		totalVRAM += g.VRAMTotalMB
	}
	for i, g := range caps.GPUs {
		if totalVRAM > 0 {
			split[i] = float64(g.VRAMTotalMB) / float64(totalVRAM)
		}
	}
	s.TensorSplit = normalizeSplit(split)

	// Prefer graph split for ik_llama, layer for mainline
	if opts.BackendTag == "ik_llama" {
		s.SplitMode = "graph"
	} else {
		s.SplitMode = "row"
	}

	return s, nil
}

func buildDenseCPUOffload(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	numGPUs := len(caps.GPUs)
	if numGPUs > 1 {
		if opts.BackendTag == "ik_llama" {
			s.SplitMode = "graph"
		} else {
			s.SplitMode = "row"
		}
		// Tensor split across all GPUs
		split := make([]float64, numGPUs)
		totalVRAM := 0
		for _, g := range caps.GPUs {
			totalVRAM += g.VRAMTotalMB
		}
		for i, g := range caps.GPUs {
			if totalVRAM > 0 {
				split[i] = float64(g.VRAMTotalMB) / float64(totalVRAM)
			}
		}
		s.TensorSplit = normalizeSplit(split)
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
	sysProbe := loadSystemProbe(opts.CacheDir)
	sysCUDAOverheadMB := 0
	if sysProbe != nil {
		sysCUDAOverheadMB = sysProbe.CUDAOverheadMB
	}

	// Load probe cache (lines 5487-5508)
	probeHit := false
	placementLabel := "cold-start"
	var probedComputeBufMB int

	pc := loadProbeCache(opts.CacheDir, filepath.Base(model.Path))
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

	// Compute per-GPU KV reserves proportional to free VRAM (lines 5058-5083)
	gpuKVReserveMB := make([]int, numGPUs)
	totalFreeVRAM := 0
	for _, g := range caps.GPUs {
		totalFreeVRAM += g.VRAMFreeMB()
	}

	if s.KVPlacement != "cpu" && kvTotalMB > 0 && totalFreeVRAM > 0 && numGPUs > 0 {
		for i := 0; i < numGPUs; i++ {
			gi := gpuOrder[i]
			// share = (KV_TOTAL_MB * GPU_VRAM_FREE[gi] + total_free - 1) / total_free
			share := (kvTotalMB*caps.GPUs[gi].VRAMFreeMB() + totalFreeVRAM - 1) / totalFreeVRAM
			// Round up to KV_PER_LAYER_MB multiples
			if kvPerLayerMB > 0 {
				share = ((share + kvPerLayerMB - 1) / kvPerLayerMB) * kvPerLayerMB
			}
			gpuKVReserveMB[gi] = share
		}
	}

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

	// CPU activation memory
	var actFFN int
	if model.NumExperts > 0 && model.FeedForwardLength > 0 {
		// MoE: use feed_forward_length as proxy for expert_ff
		actFFN = model.FeedForwardLength
		if model.KVLoraRank > 0 {
			actFFN += model.KVLoraRank + model.QLoraRank
		}
	} else {
		actFFN = model.FeedForwardLength
		if model.KVLoraRank > 0 {
			actFFN += model.KVLoraRank + model.QLoraRank
		}
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

func orderGPUsByVRAM(gpus []detect.GPU) []int {
	indices := make([]int, len(gpus))
	for i := range gpus {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		vi := gpus[indices[i]].VRAMTotalMB
		vj := gpus[indices[j]].VRAMTotalMB
		if vi == vj {
			return gpus[indices[i]].Index < gpus[indices[j]].Index
		}
		return vi > vj
	})
	return indices
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
func checkMemoryOrDie(caps *detect.Capabilities, model *ModelProfile, s *Strategy, totalSizeMB, kvTotalMB int) error {
	modelOverheadMB := totalSizeMB * vramOverheadPercent / 100
	neededMB := modelOverheadMB + kvTotalMB + computePerGPUMB

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
		maxKVMB := poolMB - modelOverheadMB - computePerGPUMB
		if maxKVMB < 0 {
			maxKVMB = 0
		}
		maxCtx := 0
		if kvTotalMB > 0 {
			maxCtx = maxKVMB * s.ContextSize / kvTotalMB
		}

		msg := fmt.Sprintf(
			"ERROR: Model does not fit in %s.\n"+
				"  Model (with overhead): %dMB\n"+
				"  KV cache (ctx=%d):     %dMB\n"+
				"  Compute buffers:       %dMB\n"+
				"  -----------------------------\n"+
				"  Total needed:          %dMB\n"+
				"  Available (%s):        %dMB\n"+
				"  Shortfall:             %dMB\n",
			poolLabel, modelOverheadMB, s.ContextSize, kvTotalMB, computePerGPUMB,
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

// Args converts a Strategy into llama-server command-line arguments.
func (s *Strategy) Args(modelPath string, port int) []string {
	args := []string{
		"-m", modelPath,
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", port),
		"--ctx-size", fmt.Sprintf("%d", s.ContextSize),
		"--flash-attn", "on",
		"-b", fmt.Sprintf("%d", s.BatchSize),
		"-ub", fmt.Sprintf("%d", s.UBatchSize),
		"--cache-type-k", s.KVType,
		"--cache-type-v", s.KVType,
		"--jinja",
		"--threads", fmt.Sprintf("%d", s.Threads),
		"--threads-batch", fmt.Sprintf("%d", s.Threads),
	}

	if s.KVPlacement == "cpu" {
		args = append(args, "--no-kv-offload")
	} else if s.KVPlacement == "gpu" {
		args = append(args, "--kv-offload")
	}

	if s.ReasoningOff {
		args = append(args, "--reasoning", "off")
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

	return args
}

// loadSystemProbe tries to load measured CUDA overhead from cache.
func loadSystemProbe(cacheDir string) *systemProbe {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "llm-server")
	}
	path := filepath.Join(cacheDir, "system.probe")
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

// loadProbeCache tries to load per-model probe data.
func loadProbeCache(cacheDir, modelName string) *probeCache {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "llm-server", "probes")
	}
	entries, err := os.ReadDir(cacheDir)
	if err != nil {
		return nil
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".probe") {
			continue
		}
		path := filepath.Join(cacheDir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		content := string(data)
		if !strings.Contains(content, modelName) {
			continue
		}
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
			key := strings.TrimSpace(parts[0])
			val := strings.TrimSpace(strings.Trim(parts[1], `"`))
			switch key {
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
	}
	return nil
}

// internal types for probe loading
type systemProbe struct {
	CUDAOverheadMB int
}

type probeCache struct {
	ComputeBufMB int
	KVPerLayerMB int
}
