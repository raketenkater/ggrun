package placement

import (
	"crypto/md5"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// VRAM and compute sizing constants
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
	GPULayers      int          `json:"gpu_layers"` // always 999; llama-server decides
	TensorSplit    []float64    `json:"tensor_split,omitempty"`
	SplitMode      string       `json:"split_mode,omitempty"` // graph, layer, row
	MainGPU        int          `json:"main_gpu,omitempty"`
	KVPlacement    string       `json:"kv_placement"`        // gpu, cpu, auto
	KVQuality      string       `json:"kv_quality"`          // high, mid, low
	KVType         string       `json:"kv_type"`             // f16, q8_0, q4_0
	NCPUMoE        int          `json:"n_cpu_moe,omitempty"` // for MoE offload
	OTString       string       `json:"ot_string,omitempty"` // -ot override-tensor flags
	MMap           bool         `json:"mmap"`
	MLock          bool         `json:"mlock"`
	FlashAttention bool         `json:"flash_attention"`
	Threads        int          `json:"threads"`
	BatchSize      int          `json:"batch_size"`
	UBatchSize     int          `json:"ubatch_size"`
	BackendTag     string       `json:"backend_tag,omitempty"` // "llama" or "ik_llama"
	IsMoE          bool         `json:"is_moe"`
	ReasoningOff   bool         `json:"reasoning_off"` // default off for OpenAI compat
	ThreadsBatch   int          `json:"threads_batch"` // batch threads (logical cores)
	Parallel       int          `json:"parallel,omitempty"`
	CRAM           int          `json:"cram,omitempty"` // prompt cache MB
	MaxCheckpoints int          `json:"max_checkpoints,omitempty"`
	UseCUDAGraphs  bool         `json:"use_cuda_graphs,omitempty"`
	Host           string       `json:"host,omitempty"`        // listen address
	HasSSM         bool         `json:"has_ssm,omitempty"`     // SSM/Mamba hybrid flag
	Draft          *DraftConfig `json:"draft,omitempty"`       // speculative decoding config
	MMProjPath     string       `json:"mmproj_path,omitempty"` // vision projector GGUF
	MMProjSizeMB   int          `json:"-"`                     // mmproj VRAM on primary GPU
}

// ModelProfile describes the GGUF model.
type ModelProfile struct {
	Path               string `json:"path"`
	Name               string `json:"name,omitempty"`         // GGUF metadata: model name
	Basename           string `json:"basename,omitempty"`     // GGUF metadata: model basename
	QuantizedBy        string `json:"quantized_by,omitempty"` // GGUF metadata: quantizer (e.g. "unsloth")
	SizeBytes          int64  `json:"size_bytes"`
	TotalSizeMB        int    `json:"total_size_mb"` // includes multi-part shards
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
	LeadingDense       int    `json:"leading_dense,omitempty"`
	NextNPredictLayers int    `json:"nextn_predict_layers,omitempty"`
}

// Options allows user overrides.
type Options struct {
	ContextSize    int
	KVPlacement    string // auto, gpu, cpu
	KVQuality      string // high, mid, low
	GPUs           []int  // restrict to specific GPUs
	CPUMode        bool
	RamBudgetMB    int
	VRAMHeadroomMB int    // hold back this much total VRAM as a safety margin
	RAMHeadroomMB  int    // hold back this much system RAM as a safety margin
	BackendTag     string // "llama" or "ik_llama"
	NoMMap         bool
	Parallel       int
	CacheFile      string // path to placement cache for MoE recovery
	CacheDir       string // path to ggrun cache dir (for probes)
	Host           string // listen address (default 0.0.0.0)
	VisionAuto     bool   // auto-detect mmproj for vision
	MMProjPath     string // explicit vision projector GGUF
	SpecMode       string // off, auto, draft, eagle3, ngram, ngram-mod, ngram-k4v, mtp
	BackendHelp    string // llama-server --help output for dialect-specific flags
	ForceSpecMoE   bool   // allow speculative decoding on MoE despite default gate
}

func applyRAMBudget(caps *detect.Capabilities, budgetMB int) *detect.Capabilities {
	if budgetMB <= 0 || caps == nil {
		return caps
	}
	capped := *caps
	capped.RAM.FreeMB = budgetMB
	capped.RAM.TotalMB = budgetMB
	return &capped
}

// Compute builds a Strategy from hardware capabilities and model profile.
func Compute(caps *detect.Capabilities, model *ModelProfile, opts Options) (*Strategy, error) {
	var err error
	caps, err = restrictGPUs(caps, opts.GPUs)
	if err != nil {
		return nil, err
	}

	caps = applyRAMBudget(caps, opts.RamBudgetMB)
	caps = detect.ApplyVRAMHeadroom(caps, opts.VRAMHeadroomMB)
	caps = detect.ApplyRAMHeadroom(caps, opts.RAMHeadroomMB)

	s := &Strategy{
		ContextSize:    opts.ContextSize,
		KVPlacement:    opts.KVPlacement,
		KVQuality:      opts.KVQuality,
		MMap:           !opts.NoMMap,
		MLock:          false,
		Threads:        caps.CPU.Cores, // physical cores
		ThreadsBatch:   caps.CPU.Cores, // physical cores
		BackendTag:     opts.BackendTag,
		IsMoE:          model.IsMoE,
		GPULayers:      999,
		FlashAttention: true,
		ReasoningOff:   true, // default reasoning off for OpenAI-compatible output
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
		s.KVQuality = "low" // q4_0, minimum VRAM
	}
	if opts.Parallel > 0 {
		s.Parallel = opts.Parallel
	}

	// Vision: use an explicit projector, or auto-detect one when --vision is set.
	if opts.MMProjPath != "" {
		if err := validateMMProj(opts.MMProjPath, model.Name, model.Basename); err != nil {
			return nil, err
		}
		s.MMProjPath = opts.MMProjPath
		if fi, err := os.Stat(opts.MMProjPath); err == nil {
			s.MMProjSizeMB = int(fi.Size() / 1024 / 1024)
		}
	} else if opts.VisionAuto && model.Path != "" {
		if path, err := findOrDownloadMMProj(model.Path, opts.CacheDir, model.Name, model.Basename, model.QuantizedBy); err == nil {
			s.MMProjPath = path
			if fi, err := os.Stat(path); err == nil {
				s.MMProjSizeMB = int(fi.Size() / 1024 / 1024)
			}
		} else {
			fmt.Fprintf(os.Stderr, "[vision] %v\n", err)
		}
	}

	// Total size MB (model + mmproj if vision)
	totalSizeMB := model.TotalSizeMB + s.MMProjSizeMB
	if totalSizeMB <= 0 {
		totalSizeMB = int(model.SizeBytes / 1024 / 1024)
	}

	// KV cache type selection — try compact types first for large models
	s.KVType = kvTypeFromQuality(s.KVQuality)

	// Auto-fit context: compute both single-GPU and multi-GPU, pick the larger.
	if opts.ContextSize <= 0 {
		if opts.CPUMode || len(caps.GPUs) == 0 {
			cpuCaps := *caps
			cpuCaps.GPUs = nil
			s.ContextSize, s.KVType = computeAutoContextSize(&cpuCaps, model, totalSizeMB, s.KVType, opts)
		} else {
			sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
			cudaOH := 600
			if sysProbe != nil {
				cudaOH = sysProbe.CUDAOverheadMB
			}
			cbuf := computeFloorMB

			bestFree := 0
			for _, g := range caps.GPUs {
				if g.VRAMFreeMB() > bestFree {
					bestFree = g.VRAMFreeMB()
				}
			}

			// Single-GPU estimate
			singleCtx, singleKV := computeAutoContextSizeSingleGPU(caps, model, totalSizeMB, s.KVType, opts)
			singleKVM := computeKVTotalMB(model, singleCtx, singleKV)
			singleFits := (totalSizeMB+cudaOH+cbuf+singleKVM) <= (bestFree-1024) && singleCtx >= 32768

			// Multi-GPU estimate
			multiCtx, multiKV := computeAutoContextSize(caps, model, totalSizeMB, s.KVType, opts)
			multiKVM := computeKVTotalMB(model, multiCtx, multiKV)
			multiFree := 0
			for _, g := range caps.GPUs {
				multiFree += g.VRAMFreeMB()
			}
			multiFits := (totalSizeMB+(cudaOH+cbuf)*len(caps.GPUs)+multiKVM) <= multiFree && multiCtx >= 32768

			if multiFits && multiCtx > singleCtx {
				s.ContextSize, s.KVType = multiCtx, multiKV
			} else if singleFits {
				s.ContextSize, s.KVType = singleCtx, singleKV
			} else if multiFits {
				s.ContextSize, s.KVType = multiCtx, multiKV
			} else {
				s.ContextSize, s.KVType = 32768, "q4_0"
			}
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
			if len(cache.TensorSplit) > 0 {
				s.TensorSplit = normalizeSplit(cache.TensorSplit)
				s.SplitMode = cache.SplitMode
				if s.SplitMode == "" {
					s.SplitMode = "layer"
				}
			}
			if len(cache.GPUAssignments) > 0 {
				otString := buildOTStringFromAssignments(cache.GPUAssignments, caps.GPUs, model.NumLayers, opts.BackendTag)
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

	// Vision override: mmproj needs extra VRAM — force multi-GPU
	if s.MMProjPath != "" && strategy == SingleGPU && len(caps.GPUs) > 1 {
		if model.IsMoE {
			strategy = MoEOffload
			s.Type = MoEOffload
		} else {
			strategy = MultiGPUDense
			s.Type = MultiGPUDense
		}
	}

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

	if opts.SpecMode != "" && opts.SpecMode != "off" {
		s.Draft = ComputeDraft(model, caps, opts)
	}

	// Compute CRAM (prompt cache)
	cram, maxCheckpoints := computeCRAM(caps, model, s, totalSizeMB, kvTotalMB)
	s.CRAM = cram
	s.MaxCheckpoints = maxCheckpoints

	// Default host
	if s.Host == "" {
		s.Host = "0.0.0.0"
	}

	return s, nil
}

// restrictGPUs filters caps.GPUs to the user-selected device indices (--gpus).
// Devices are renumbered from 0 because the launcher restricts visibility via
// CUDA_VISIBLE_DEVICES / GGML_VK_VISIBLE_DEVICES, so the backend enumerates
// only the selected devices starting at index 0.
func restrictGPUs(caps *detect.Capabilities, want []int) (*detect.Capabilities, error) {
	if caps == nil || len(want) == 0 || len(caps.GPUs) == 0 {
		return caps, nil
	}
	wanted := make(map[int]bool, len(want))
	for _, idx := range want {
		wanted[idx] = true
	}
	filtered := *caps
	filtered.GPUs = nil
	for _, g := range caps.GPUs {
		if wanted[g.Index] {
			gg := g
			gg.Index = len(filtered.GPUs)
			filtered.GPUs = append(filtered.GPUs, gg)
		}
	}
	if len(filtered.GPUs) == 0 {
		return nil, fmt.Errorf("--gpus %v matches no detected GPU (have %d GPUs)", want, len(caps.GPUs))
	}
	return &filtered, nil
}

// chooseStrategy selects the placement strategy from hardware and model size.
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

	// KV-first GPU reserve: VRAM-proportional (KV reads are VRAM-local)
	gpuKVReserveMB := kvReserveByBandwidth(kvTotalMB, caps.GPUs, seqRange(numGPUs), kvPerLayerMB)

	// Tensor-split: proportional to free VRAM only.
	// llama-server distributes BOTH model weights AND KV cache by this ratio.
	// Using effective free (subtracting KV reserve) causes OOM because
	// llama-server puts KV back proportionally to the split anyway.
	gpuOrder := orderGPUsByBandwidth(caps.GPUs)
	split := make([]float64, numGPUs)
	totalFree := 0.0
	for _, g := range caps.GPUs {
		totalFree += float64(g.VRAMFreeMB())
	}
	if totalFree > 0 {
		for _, gi := range gpuOrder {
			free := float64(caps.GPUs[gi].VRAMFreeMB())
			if gi == gpuOrder[0] && s.MMProjSizeMB > 0 {
				free -= float64(s.MMProjSizeMB)
				if free < 0 {
					free = 0
				}
			}
			split[gi] = free / totalFree
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

// buildMoEOffload computes a fully specified multi-GPU MoE plan: tensor split
// for non-expert/KV tensors plus override-tensor pins for expert tensors.
func buildMoEOffload(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	numGPUs := len(caps.GPUs)
	if numGPUs == 0 {
		return buildCPUOnly(s, caps, model, opts)
	}
	if model.NumLayers <= 0 {
		return nil, fmt.Errorf("MoE placement requires model layer count")
	}

	gpuOrder := orderGPUsByBandwidth(caps.GPUs)
	s.MainGPU = caps.GPUs[gpuOrder[0]].Index
	if numGPUs > 1 {
		s.SplitMode = "layer"
	}

	moeStartLayer := model.LeadingDense
	if moeStartLayer < 0 || moeStartLayer >= model.NumLayers {
		moeStartLayer = 0
	}
	moeLayerCount := model.NumLayers - moeStartLayer
	if moeLayerCount <= 0 {
		moeLayerCount = model.NumLayers
		moeStartLayer = 0
	}

	// Per-layer costs
	expertTotalMB := bytesToMiBCeil(model.ExpertBytes)
	if expertTotalMB <= 0 {
		expertTotalMB = totalSizeMB * 90 / 100
	}
	nonExpertTotalMB := bytesToMiBCeil(model.NonExpertBytes)
	if nonExpertTotalMB <= 0 {
		nonExpertTotalMB = totalSizeMB - expertTotalMB
	}
	if nonExpertTotalMB < 0 {
		nonExpertTotalMB = 0
	}
	expertPerLayerMB := ceilDivInt(expertTotalMB, moeLayerCount)
	if expertPerLayerMB <= 0 {
		expertPerLayerMB = 1
	}
	nonExpertPerLayerMB := ceilDivInt(nonExpertTotalMB, model.NumLayers)
	if nonExpertPerLayerMB <= 0 {
		nonExpertPerLayerMB = 1
	}

	// Load system probe
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

	// Load probe cache
	computeBufMB := computeFloorMB
	pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality)
	if pc != nil && pc.ComputeBufMB > 0 {
		computeBufMB = pc.ComputeBufMB
	}
	fixedPerGPU := sysCUDAOverheadMB + computeBufMB

	// Use only GPUs that can carry CUDA/compute overhead plus their emitted
	// tensor-split share of all non-expert weights and KV. The split is
	// bandwidth-weighted: under --split-mode layer, the GPU that owns a
	// layer's non-expert weights also computes that layer — including
	// streaming its CPU-resident experts over PCIe. Weighting the split by
	// measured PCIe bandwidth (not just free VRAM) concentrates layer
	// ownership on the fastest-link GPU, so CPU-expert streaming avoids
	// bottlenecking on a slow PCIe link (e.g. a card stuck at x1).
	// When bandwidth is unknown or uniform across GPUs, this degenerates
	// to free-VRAM-proportional (the previous behaviour).
	used := make([]bool, numGPUs)
	for i, g := range caps.GPUs {
		used[i] = g.VRAMFreeMB() > fixedPerGPU
	}
	var split []float64
	for {
		rawSplit := make([]float64, numGPUs)
		totalWeighted := 0.0
		for i, g := range caps.GPUs {
			if used[i] {
				totalWeighted += float64(g.VRAMFreeMB()) * gpuSplitWeight(g)
			}
		}
		if totalWeighted <= 0 {
			return nil, fmt.Errorf("Model does not fit on this system: no GPU has free VRAM after CUDA/compute overhead")
		}
		for i, g := range caps.GPUs {
			if used[i] {
				rawSplit[i] = float64(g.VRAMFreeMB()) * gpuSplitWeight(g) / totalWeighted
			}
		}
		split = normalizeSplit(rawSplit)

		removed := false
		for i, g := range caps.GPUs {
			if !used[i] {
				continue
			}
			nonExpertShareMB := splitShareMB(nonExpertTotalMB, split, i)
			kvShareMB := splitShareMB(kvTotalMB, split, i)
			if fixedPerGPU+nonExpertShareMB+kvShareMB > g.VRAMFreeMB() {
				used[i] = false
				removed = true
			}
		}
		if !removed {
			break
		}
	}
	if numGPUs > 1 {
		s.TensorSplit = split
	}

	// Per-GPU expert capacity under the exact emitted split.
	maxGPULayersPer := make([]int, numGPUs)
	maxGPULayers := 0
	for _, gi := range gpuOrder {
		if split[gi] <= 0 {
			continue
		}
		g := caps.GPUs[gi]
		nonExpertShareMB := splitShareMB(nonExpertTotalMB, split, gi)
		kvShareMB := splitShareMB(kvTotalMB, split, gi)
		roomMB := g.VRAMFreeMB() - fixedPerGPU - nonExpertShareMB - kvShareMB
		if roomMB < 0 {
			roomMB = 0
		}
		capLayers := roomMB / expertPerLayerMB
		if capLayers > moeLayerCount {
			capLayers = moeLayerCount
		}
		maxGPULayersPer[gi] = capLayers
		maxGPULayers += capLayers
	}
	if maxGPULayers > moeLayerCount {
		maxGPULayers = moeLayerCount
	}

	// Hard ceilings: _recompute_cpu_layer_caps
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
	if maxCPULayersStrict > moeLayerCount {
		maxCPULayersStrict = moeLayerCount
	}

	// Mmap-aware ceiling
	preWorkingSetFloor := ramOverheadPreMB + cpuKVRAMMB + 8*expertPerLayerMB
	maxCPULayersMMap := 0
	if caps.RAM.FreeMB >= preWorkingSetFloor {
		maxCPULayersMMap = moeLayerCount
	}

	maxCPULayers := maxCPULayersStrict
	ceilCPULabel := "strict --no-mmap"
	if maxGPULayers+maxCPULayersStrict < moeLayerCount &&
		!opts.NoMMap &&
		maxCPULayersMMap > maxCPULayersStrict {
		maxCPULayers = maxCPULayersMMap
		ceilCPULabel = "mmap (page-cache)"
	}

	// Does-not-fit guard
	if maxGPULayers+maxCPULayers < moeLayerCount {
		gap := moeLayerCount - maxGPULayers - maxCPULayers
		gapVRAMMB := gap * expertPerLayerMB
		gapRAMMB := gap * expertPerLayerMB
		return nil, fmt.Errorf(
			"Model does not fit on this system.\n"+
				"  Required:    %d MoE layers\n"+
				"  GPU cap:     %d layers across %d GPU(s)\n"+
				"  CPU cap:     %d layers (%s)\n"+
				"  Gap:         %d layers — need ~%dMB more free VRAM or ~%dMB more RAM\n"+
				"\n  Options:\n"+
				"    1. Free VRAM (close other GPU workloads, --gpus to add a card)\n"+
				"    2. Drop --no-mmap so kernel can page experts on demand\n"+
				"    3. Use a smaller quantization or smaller model",
			moeLayerCount, maxGPULayers, numGPUs, maxCPULayers, ceilCPULabel,
			gap, gapVRAMMB, gapRAMMB)
	}

	// Initial layer assignment
	layersPerGPU := make([]int, numGPUs)
	totalGPULayers := 0
	remainingMoELayers := moeLayerCount

	for _, gi := range gpuOrder {
		layers := maxGPULayersPer[gi]
		if layers > remainingMoELayers {
			layers = remainingMoELayers
		}
		layersPerGPU[gi] = layers
		totalGPULayers += layers
		remainingMoELayers -= layers
		if remainingMoELayers == 0 {
			break
		}
	}

	layersCPU := moeLayerCount - totalGPULayers

	// RAM safety check
	cpuExpertMB := layersCPU * expertPerLayerMB

	// Exact RAM overhead
	cudaHostMB := 1024
	graphScratchMB := 2048
	mmapPTMB := totalSizeMB / 500

	// CPU activation memory
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

	// Mmap decision for MoE
	// If strict ceiling doesn't fit and user didn't force --no-mmap,
	// use mmap (page-cache) if working set fits
	if opts.NoMMap {
		s.MMap = false
	} else if totalSizeMB > ramAvailMB {
		s.MMap = true
	} else if ramNeeded > ramAvailMB {
		workingSetFloor := ramOverheadMB + cpuKVMB + 8*expertPerLayerMB
		if ramAvailMB >= workingSetFloor {
			s.MMap = true
		}
	} else {
		s.MMap = false
	}

	// Build -ot string. Always include the exps=CPU catch-all so expert
	// tensors never follow the backend's default layer split by accident.
	otString := buildOTStringFromStart(layersPerGPU, caps.GPUs, gpuOrder, moeStartLayer, opts.BackendTag)
	if otString != "" {
		s.OTString = otString
	}

	// NCPUMoE is a CPU expert-layer count, not an expert count.
	if layersCPU > 0 {
		s.NCPUMoE = layersCPU
	}

	_ = nonExpertPerLayerMB

	return s, nil
}

func bytesToMiBCeil(n int64) int {
	if n <= 0 {
		return 0
	}
	return int((n + 1048576 - 1) / 1048576)
}

func ceilDivInt(n, d int) int {
	if n <= 0 || d <= 0 {
		return 0
	}
	return (n + d - 1) / d
}

func splitShareMB(totalMB int, split []float64, idx int) int {
	if totalMB <= 0 || idx < 0 || idx >= len(split) || split[idx] <= 0 {
		return 0
	}
	totalSplit := 0.0
	for _, v := range split {
		if v > 0 {
			totalSplit += v
		}
	}
	if totalSplit <= 0 {
		return 0
	}
	return int(math.Ceil(float64(totalMB) * split[idx] / totalSplit))
}

// DerateCUDAOOMArgs moves enough expert layers from the failed device back to
// CPU to cover a cudaMalloc load failure, then returns the rewritten argv.
func DerateCUDAOOMArgs(args []string, model *ModelProfile, caps *detect.Capabilities, device, allocMB int) ([]string, *CacheEntry, bool) {
	if model == nil || model.NumLayers <= 0 || allocMB <= 0 {
		return nil, nil, false
	}
	_, moeLayers := moeLayerRange(model)
	expertPerLayerMB := ceilDivInt(bytesToMiBCeil(model.ExpertBytes), moeLayers)
	if expertPerLayerMB <= 0 {
		return nil, nil, false
	}
	overshootMB := allocMB
	if caps != nil {
		for _, g := range caps.GPUs {
			if g.Index == device && allocMB > g.VRAMFreeMB() {
				overshootMB = allocMB - g.VRAMFreeMB()
				break
			}
		}
	}
	dropLayers := ceilDivInt(overshootMB, expertPerLayerMB)
	if dropLayers <= 0 {
		dropLayers = 1
	}

	otIdx := argIndex(args, "-ot", "--override-tensor")
	if otIdx < 0 || otIdx+1 >= len(args) {
		return nil, nil, false
	}
	assignments := parseOTAssignments(args[otIdx+1])
	if len(assignments) == 0 {
		return nil, nil, false
	}
	remainingDrop := dropLayers
	actualDrop := 0
	for i := range assignments {
		if assignments[i].CUDAIndex != device || assignments[i].Count <= 0 {
			continue
		}
		drop := remainingDrop
		if drop > assignments[i].Count {
			drop = assignments[i].Count
		}
		assignments[i].Count -= drop
		actualDrop += drop
		remainingDrop -= drop
		if remainingDrop == 0 {
			break
		}
	}
	if actualDrop == 0 {
		return nil, nil, false
	}

	newArgs := append([]string(nil), args...)
	newArgs[otIdx+1] = buildOTStringFromAssignments(assignments, nil, model.NumLayers, "")
	setOrAppendArg(&newArgs, "--n-cpu-moe", strconv.Itoa(currentNCPUMoE(args)+actualDrop))

	entry := cacheEntryFromArgs(newArgs, assignments)
	return newArgs, entry, true
}

func moeLayerRange(model *ModelProfile) (int, int) {
	if model == nil || model.NumLayers <= 0 {
		return 0, 0
	}
	start := model.LeadingDense
	if start < 0 || start >= model.NumLayers {
		start = 0
	}
	count := model.NumLayers - start
	if count <= 0 {
		return 0, model.NumLayers
	}
	return start, count
}

var otAssignmentPattern = regexp.MustCompile(`blk\\\.\(([^)]*)\).*=(?:CUDA|Vulkan)(\d+)`)

func parseOTAssignments(ot string) []GPUAssignment {
	var out []GPUAssignment
	for _, part := range strings.Split(ot, ",") {
		m := otAssignmentPattern.FindStringSubmatch(part)
		if m == nil {
			continue
		}
		device, err := strconv.Atoi(m[2])
		if err != nil {
			continue
		}
		layers := strings.Split(m[1], "|")
		if len(layers) == 0 {
			continue
		}
		start, err := strconv.Atoi(layers[0])
		if err != nil {
			continue
		}
		out = append(out, GPUAssignment{CUDAIndex: device, Start: start, Count: len(layers)})
	}
	return out
}

func argIndex(args []string, names ...string) int {
	for i, arg := range args {
		for _, name := range names {
			if arg == name {
				return i
			}
		}
	}
	return -1
}

func setOrAppendArg(args *[]string, name, value string) {
	if idx := argIndex(*args, name); idx >= 0 {
		if idx+1 < len(*args) {
			(*args)[idx+1] = value
			return
		}
	}
	*args = append(*args, name, value)
}

func currentNCPUMoE(args []string) int {
	idx := argIndex(args, "--n-cpu-moe")
	if idx < 0 || idx+1 >= len(args) {
		return 0
	}
	v, _ := strconv.Atoi(args[idx+1])
	return v
}

func cacheEntryFromArgs(args []string, assignments []GPUAssignment) *CacheEntry {
	entry := &CacheEntry{GPUAssignments: positiveAssignments(assignments)}
	if idx := argIndex(args, "--tensor-split"); idx >= 0 && idx+1 < len(args) {
		entry.TensorSplit = parseTensorSplit(args[idx+1])
	}
	if idx := argIndex(args, "--split-mode"); idx >= 0 && idx+1 < len(args) {
		entry.SplitMode = args[idx+1]
	}
	if idx := argIndex(args, "--n-cpu-moe"); idx >= 0 && idx+1 < len(args) {
		entry.NCPUMoE, _ = strconv.Atoi(args[idx+1])
	}
	if idx := argIndex(args, "-b", "--batch-size"); idx >= 0 && idx+1 < len(args) {
		entry.BatchSize, _ = strconv.Atoi(args[idx+1])
	}
	if idx := argIndex(args, "-ub", "--ubatch-size"); idx >= 0 && idx+1 < len(args) {
		entry.UBatchSize, _ = strconv.Atoi(args[idx+1])
	}
	if idx := argIndex(args, "--parallel", "-np"); idx >= 0 && idx+1 < len(args) {
		entry.Parallel, _ = strconv.Atoi(args[idx+1])
	}
	entry.MMap = argIndex(args, "--no-mmap") < 0
	return entry
}

func positiveAssignments(assignments []GPUAssignment) []GPUAssignment {
	out := make([]GPUAssignment, 0, len(assignments))
	for _, a := range assignments {
		if a.Count > 0 {
			out = append(out, a)
		}
	}
	return out
}

// buildOTString builds the -ot override-tensor string for MoE.
// Builds the -ot override-tensor string: explicit layer list with escaped dots.
func buildOTString(layersPerGPU []int, gpus []detect.GPU, gpuOrder []int, backendTag string) string {
	return buildOTStringFromStart(layersPerGPU, gpus, gpuOrder, 0, backendTag)
}

func buildOTStringFromStart(layersPerGPU []int, gpus []detect.GPU, gpuOrder []int, startLayer int, backendTag string) string {
	var parts []string
	expertPattern := `ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp)`

	nextLayer := startLayer
	for _, gi := range gpuOrder {
		count := layersPerGPU[gi]
		if count > 0 {
			start := nextLayer
			last := start + count - 1
			cudaIdx := gpus[gi].Index
			// Build explicit layer list, e.g. 0|1|2|...|31
			var layerParts []string
			for l := start; l <= last; l++ {
				layerParts = append(layerParts, fmt.Sprintf("%d", l))
			}
			layerRange := stringsJoin(layerParts, "|")
			parts = append(parts, fmt.Sprintf(`blk\.(%s)\.%s.*=%s`, layerRange, expertPattern, deviceName(backendTag, cudaIdx)))
			nextLayer += count
		}
	}
	parts = append(parts, "exps=CPU")

	return stringsJoin(parts, ",")
}

func buildOTStringFromAssignments(assignments []GPUAssignment, gpus []detect.GPU, numLayers int, backendTag string) string {
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
		parts = append(parts, fmt.Sprintf(`blk\.(%s)\.%s.*=%s`, layerRange, expertPattern, deviceName(backendTag, assign.CUDAIndex)))
		nextLayer += assign.Count
	}
	parts = append(parts, "exps=CPU")
	return stringsJoin(parts, ",")
}

func deviceName(backendTag string, index int) string {
	if strings.EqualFold(backendTag, "vulkan") {
		return fmt.Sprintf("Vulkan%d", index)
	}
	return fmt.Sprintf("CUDA%d", index)
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

// kvReserveByBandwidth distributes KV cache proportionally to free VRAM.
// KV reads are VRAM-local — PCIe bandwidth does not affect KV access speed.
func kvReserveByBandwidth(kvTotalMB int, gpus []detect.GPU, order []int, kvPerLayerMB int) []int {
	reserve := make([]int, len(gpus))
	totalFree := 0
	for _, g := range gpus {
		totalFree += g.VRAMFreeMB()
	}
	if kvTotalMB <= 0 || totalFree <= 0 {
		return reserve
	}
	useOrder := order
	if len(useOrder) == 0 {
		useOrder = seqRange(len(gpus))
	}
	for _, gi := range useOrder {
		share := (kvTotalMB*gpus[gi].VRAMFreeMB() + totalFree - 1) / totalFree
		if kvPerLayerMB > 0 {
			share = ((share + kvPerLayerMB - 1) / kvPerLayerMB) * kvPerLayerMB
		}
		reserve[gi] = share
	}
	return reserve
}

func seqRange(n int) []int {
	r := make([]int, n)
	for i := range r {
		r[i] = i
	}
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

// gpuSplitWeight returns the weight applied to a GPU's free VRAM when
// computing the tensor-split for MoE offload. Under --split-mode layer the
// GPU that owns a layer's non-expert weights computes that layer, so
// CPU-resident experts stream over that GPU's PCIe link. Weighting by PCIe
// bandwidth concentrates ownership on fast-link GPUs. Returns 1.0 when
// bandwidth is unknown so the split degenerates to free-VRAM-proportional.
func gpuSplitWeight(g detect.GPU) float64 {
	if g.BandwidthMBps <= 0 {
		return 1.0
	}
	return float64(g.BandwidthMBps)
}

// checkMemoryOrDie refuses to launch when model + KV + compute buffers exceed the pool.
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
func computeCRAM(caps *detect.Capabilities, model *ModelProfile, s *Strategy, totalSizeMB, kvTotalMB int) (int, int) {
	numGPUs := len(caps.GPUs)

	// Fits on GPU? (model fits entirely in VRAM)
	fitsOnGPU := false
	switch s.Type {
	case SingleGPU, MultiGPUDense:
		fitsOnGPU = true
	}

	// RAM remaining after weights load
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

	// Single-GPU / CPU-only CRAM
	cram := ramAfterLoad / 10
	if cram > 16384 {
		cram = 16384
	}
	if cram < minCramMB {
		cram = 0
	}

	maxCheckpoints := 0

	// Multi-GPU CRAM
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
func computeAutoContextSizeSingleGPU(caps *detect.Capabilities, model *ModelProfile, totalSizeMB int, preferredKVType string, opts Options) (int, string) {
	// Find best single GPU by total VRAM
	bestVRAM := 0
	for _, g := range caps.GPUs {
		if g.VRAMTotalMB > bestVRAM {
			bestVRAM = g.VRAMTotalMB
		}
	}

	// Total hardware for single GPU: best GPU VRAM + up to 4GB RAM (not entire system)
	// Single GPU context shouldn't use entire system RAM — the model must fit on ONE GPU.
	totalHWMB := bestVRAM + 4096

	// Fixed overhead: model weights + 8GB headroom
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
// hardware memory, .
// Uses TOTAL_VRAM + RAM_AVAIL.
func computeAutoContextSize(caps *detect.Capabilities, model *ModelProfile, totalSizeMB int, preferredKVType string, opts Options) (int, string) {
	// Total hardware = all GPU VRAM + free RAM
	totalVRAM := 0
	for _, g := range caps.GPUs {
		totalVRAM += g.VRAMTotalMB
	}
	totalHWMB := totalVRAM + caps.RAM.FreeMB

	// Fixed overhead: model weights + 8GB headroom
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

	// SSM/Mamba models need --no-context-shift
	if s.HasSSM {
		args = append(args, "--no-context-shift")
	}

	if s.Parallel > 0 {
		args = append(args, "--parallel", fmt.Sprintf("%d", s.Parallel))
	} else {
		args = append(args, "--parallel", "1")
	}

	// Vision support: auto-detected mmproj
	if s.MMProjPath != "" {
		args = append(args, "--mmproj", s.MMProjPath)
	}

	// GPU offloading. CPU-only still
	// prints -ngl 0 so compatibility tests and user scripts can see the mode.
	if s.Type == CPUOnly {
		args = append(args, "-ngl", "0")
	} else if len(s.TensorSplit) > 0 || s.Type != CPUOnly {
		args = append(args, "-ngl", "999")
		// Metal has exactly one logical device — device-routing flags are
		// CUDA/Vulkan concepts and llama-server rejects unknown device names.
		if s.MainGPU >= 0 && len(s.TensorSplit) == 0 && !strings.EqualFold(s.BackendTag, "metal") {
			args = append(args, "-mg", fmt.Sprintf("%d", s.MainGPU))
			if s.Type == SingleGPU {
				args = append(args, "--device", deviceName(s.BackendTag, s.MainGPU))
			}
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

	// Speculative decoding flags (MTP, EAGLE-3, draft model, or explicit ngram)
	if s.Draft != nil && s.Draft.Type != DraftNone {
		args = append(args, DraftFlags(s.Draft)...)
	}

	return args
}

// loadSystemProbe tries to load measured CUDA overhead from cache.
// Keys the probe cache by a GPU-signature hash.
func loadSystemProbe(cacheDir string, gpus []detect.GPU) *systemProbe {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
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
// GPU signature: nvidia-smi --query-gpu=name,driver_version | sort | md5sum | cut -c1-12
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
// Parses the server log after launch to record measured overhead.
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
		cacheDir = filepath.Join(home, ".cache", "ggrun")
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
// Keys the probe cache file by an MD5 of the model + runtime signature.
func loadProbeCache(cacheDir string, model *ModelProfile, ctxSize int, ubatch int, kvQuality string) *probeCache {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun", "probes")
	}
	// MD5 hash key over model_name:layers:experts:embd:ff:ctx:ubatch:kv_quality
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
// Writes measured compute-buffer and KV sizes to the probe cache.
func WriteProbeCache(cacheDir, modelName string, computeBufMB, kvPerLayerMB int) error {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun", "probes")
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
		// Approximate per-device KV share by kvCount (devices holding KV)
		kvPerLayerMB = int(totalKVBuf/float64(kvCount) + 0.5)
	}

	return
}
