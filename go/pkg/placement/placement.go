package placement

import (
	"fmt"
	"math"
	"sort"

	"github.com/raketenkater/llm-server/pkg/detect"
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
	Path            string  `json:"path"`
	SizeBytes       int64   `json:"size_bytes"`
	TotalSizeMB     int     `json:"total_size_mb"`     // includes multi-part shards
	NumLayers       int     `json:"num_layers"`
	NumParams       int64   `json:"num_params"`
	IsMoE           bool    `json:"is_moe"`
	NumExperts      int     `json:"num_experts,omitempty"`
	ContextSize     int     `json:"context_size"`
	HiddenSize      int     `json:"hidden_size"`
	HeadCount       int     `json:"head_count"`
	HeadCountKV     int     `json:"head_count_kv"`
	KeyLength       int     `json:"key_length"`
	ValueLength     int     `json:"value_length"`
	VocabSize       int     `json:"vocab_size"`
	QuantType       string  `json:"quant_type"`
	ExpertBytes     int64   `json:"expert_bytes"`
	NonExpertBytes  int64   `json:"non_expert_bytes"`
	Fused           int     `json:"fused"`
	EmbeddingLength int     `json:"embedding_length"`
	FeedForwardLength int   `json:"feed_forward_length"`
	KVLoraRank      int     `json:"kv_lora_rank"`
	QLoraRank       int     `json:"q_lora_rank"`
	RopeDim         int     `json:"rope_dim"`
	KeyLengthMLA    int     `json:"key_length_mla"`
	ValueLengthMLA  int     `json:"value_length_mla"`
	HasSSM          int     `json:"has_ssm"`
	SlidingWindow   int     `json:"sliding_window"`
	FullAttnInterval int    `json:"full_attn_interval"`
	HasShexp        int     `json:"has_shexp"`
	CTXTrain        int     `json:"ctx_train"`
	ModelArch       string  `json:"model_arch"`
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
		ReasoningOff:   true,
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

	// Memory analysis
	totalVRAM := caps.TotalVRAM()
	totalSizeMB := model.TotalSizeMB
	if totalSizeMB <= 0 {
		totalSizeMB = int(model.SizeBytes / 1024 / 1024)
	}

	// Compute KV cache size for memory analysis
	kvTotalMB := computeKVTotalMB(model, s.ContextSize, s.KVType)

	// Batch sizes based on fit
	bestGPUVRAM := 0
	for _, g := range caps.GPUs {
		if g.VRAMTotalMB > bestGPUVRAM {
			bestGPUVRAM = g.VRAMTotalMB
		}
	}

	fitsOnGPU := totalSizeMB*110/100 <= totalVRAM
	singleGPUHeadroomMB := 4096
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
			// Build strategy from cache
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
			// Build OT string from cached assignments
			if len(cache.GPUAssignments) > 0 {
				otString := buildOTStringFromAssignments(cache.GPUAssignments, caps.GPUs, model.NumLayers)
				if otString != "" {
					s.OTString = otString
				}
			}
			return s, nil
		}
	}

	// Strategy selection
	strategy := chooseStrategy(caps, model, totalSizeMB, kvTotalMB, opts)
	s.Type = strategy

	switch strategy {
	case CPUOnly:
		return buildCPUOnly(s, caps, model, opts)
	case SingleGPU:
		return buildSingleGPU(s, caps, model, opts)
	case MultiGPUDense:
		return buildMultiGPUDense(s, caps, model, totalSizeMB, kvTotalMB, opts)
	case DenseCPUOffload:
		return buildDenseCPUOffload(s, caps, model, opts)
	case MoEOffload:
		return buildMoEOffload(s, caps, model, totalSizeMB, kvTotalMB, opts)
	}

	return s, nil
}

// chooseStrategy mirrors bash choose_strategy()
func chooseStrategy(caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) StrategyType {
	numGPUs := len(caps.GPUs)

	if opts.CPUMode || numGPUs == 0 {
		return CPUOnly
	}

	// Single GPU check: model + overhead fits in best GPU
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
	computePerGPU := 1024 // MB compute floor
	singleGPUNeeded := totalSizeMB*110/100 + kvTotalMB + computePerGPU
	if singleGPUNeeded <= singleGPUUsable {
		return SingleGPU
	}

	// Multi-GPU dense check
	if !model.IsMoE {
		// Check subset of fastest GPUs (ordered by VRAM)
		gpuOrder := orderGPUsByVRAM(caps.GPUs)
		for n := 2; n <= numGPUs; n++ {
			subsetVRAM := 0
			for j := 0; j < n && j < len(gpuOrder); j++ {
				gi := gpuOrder[j]
				usable := caps.GPUs[gi].VRAMTotalMB - 1024
				if usable < 0 {
					usable = 0
				}
				subsetVRAM += usable
			}
			vramNeeded := totalSizeMB*130/100 + kvTotalMB + computePerGPU*n
			if vramNeeded <= subsetVRAM {
				return MultiGPUDense
			}
		}
	}

	// MoE offload
	if model.IsMoE {
		return MoEOffload
	}

	// Dense CPU offload
	return DenseCPUOffload
}

func buildCPUOnly(s *Strategy, caps *detect.Capabilities, model *ModelProfile, opts Options) (*Strategy, error) {
	s.GPULayers = 0
	s.MMap = true
	s.BatchSize = 512
	s.UBatchSize = 256
	return s, nil
}

func buildSingleGPU(s *Strategy, caps *detect.Capabilities, model *ModelProfile, opts Options) (*Strategy, error) {
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
	gpuOrder := orderGPUsByVRAM(caps.GPUs)

	// Find smallest subset of fastest GPUs that can fit the model
	useGPUCount := numGPUs
	computePerGPU := 1024
	for n := 2; n <= numGPUs; n++ {
		subsetVRAM := 0
		for j := 0; j < n && j < len(gpuOrder); j++ {
			gi := gpuOrder[j]
			usable := caps.GPUs[gi].VRAMTotalMB - 1024
			if usable < 0 {
				usable = 0
			}
			subsetVRAM += usable
		}
		vramNeeded := totalSizeMB*130/100 + kvTotalMB + computePerGPU*n
		if vramNeeded <= subsetVRAM {
			useGPUCount = n
			break
		}
	}

	// Build tensor-split proportional to VRAM
	selected := make(map[int]bool)
	for j := 0; j < useGPUCount && j < len(gpuOrder); j++ {
		selected[gpuOrder[j]] = true
	}

	totalVRAM := 0
	for i, g := range caps.GPUs {
		if selected[i] {
			totalVRAM += g.VRAMTotalMB
		}
	}

	split := make([]float64, numGPUs)
	for i, g := range caps.GPUs {
		if selected[i] && totalVRAM > 0 {
			split[i] = float64(g.VRAMTotalMB) / float64(totalVRAM)
		} else {
			split[i] = 0.0
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

func buildDenseCPUOffload(s *Strategy, caps *detect.Capabilities, model *ModelProfile, opts Options) (*Strategy, error) {
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
func buildMoEOffload(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	numGPUs := len(caps.GPUs)
	gpuOrder := orderGPUsByFreeVRAM(caps.GPUs)

	// Per-layer costs
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

	// Main GPU globals (embeddings + output head)
	mainGPUGlobalsMB := nonExpertTotalMB - nonExpertPerLayerMB*model.NumLayers/2
	if mainGPUGlobalsMB < 0 {
		mainGPUGlobalsMB = 0
	}

	// CUDA overhead (measured or default)
	cudaOverheadMB := 0
	computeFloorMB := 1024
	fixedPerGPU := cudaOverheadMB + computeFloorMB

	// KV reserves
	kvPerLayerMB := kvTotalMB / model.NumLayers
	if kvPerLayerMB < 1 && kvTotalMB > 0 {
		kvPerLayerMB = 1
	}

	// Compute per-GPU layer caps
	gpuKVReserveMB := make([]int, numGPUs)
	totalFreeVRAM := 0
	for _, g := range caps.GPUs {
		totalFreeVRAM += g.VRAMFreeMB()
	}

	if s.KVPlacement != "cpu" && kvTotalMB > 0 && totalFreeVRAM > 0 && numGPUs > 0 {
		for i := 0; i < numGPUs; i++ {
			gi := gpuOrder[i]
			share := (kvTotalMB * caps.GPUs[gi].VRAMFreeMB()) / totalFreeVRAM
			if kvPerLayerMB > 0 {
				share = ((share + kvPerLayerMB - 1) / kvPerLayerMB) * kvPerLayerMB
			}
			gpuKVReserveMB[gi] = share
		}
	}

	// Layer distribution
	layersPerGPU := make([]int, numGPUs)
	nextLayer := 0
	totalGPULayers := 0

	for i := 0; i < numGPUs; i++ {
		gi := gpuOrder[i]
		localOverhead := fixedPerGPU + gpuKVReserveMB[gi]
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

	// RAM safety check
	cpuExpertMB := layersCPU * expertPerLayerMB
	ramOverheadMB := 1024 + 2048 + totalSizeMB/500 + 512 // cuda_host + graph_scratch + mmap_pt + cpu_act
	cpuKVMB := 0
	if s.KVPlacement == "cpu" {
		cpuKVMB = kvTotalMB
	}
	ramNeeded := cpuExpertMB + cpuKVMB + ramOverheadMB
	ramAvailMB := caps.RAM.FreeMB

	if ramNeeded > ramAvailMB && layersCPU > 0 && numGPUs > 0 {
		// Push layers to GPUs
		for bump := 70; bump <= 100; bump += 10 {
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
					if layers == 0 && bump == 100 {
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

	// mmap decision for MoE
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

	// Build -ot string
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

	return s, nil
}

// buildOTString builds the -ot override-tensor string for MoE.
// Matches bash build_ot_string() exactly: explicit layer list with escaped dots.
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
