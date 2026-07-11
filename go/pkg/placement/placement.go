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
	// Cards below this fraction of the fastest PCIe link are too slow to own
	// regular layer slots in MoE layer-split mode, but can still be useful as
	// expert-only VRAM when one or more whole expert layers fit.
	expertOnlyMaxBandwidthRatio = 0.10
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
	Type        StrategyType `json:"type"`
	ContextSize int          `json:"context_size"`
	GPULayers   int          `json:"gpu_layers"` // always 999; llama-server decides
	TensorSplit []float64    `json:"tensor_split,omitempty"`
	SplitMode   string       `json:"split_mode,omitempty"` // graph, layer, row
	MainGPU     int          `json:"main_gpu,omitempty"`
	KVPlacement string       `json:"kv_placement"`        // gpu, cpu, auto
	KVQuality   string       `json:"kv_quality"`          // high, mid, low
	KVType      string       `json:"kv_type"`             // f16, q8_0, q4_0
	NCPUMoE     int          `json:"n_cpu_moe,omitempty"` // for MoE offload
	OTString    string       `json:"ot_string,omitempty"` // -ot override-tensor flags
	// PlacementCachePath is the keyed file where this exact placement (model +
	// ctx + ubatch + kv placement + backend + GPU set) is persisted, so a load
	// that lands right — or is corrected by OOM-recovery — is reused next launch
	// instead of re-predicted. Runtime-only; not part of the serialized strategy.
	PlacementCachePath string `json:"-"`
	// BackendSupportsFit is true when the backend's --help lists -fit/--fit. ggrun
	// then emits `--fit off` for GPU launches to disable the backend's own memory
	// auto-fitting, which is redundant with ggrun's explicit placement.
	BackendSupportsFit bool         `json:"-"`
	MMap               bool         `json:"mmap"`
	MLock              bool         `json:"mlock"`
	FlashAttention     bool         `json:"flash_attention"`
	Threads            int          `json:"threads"`
	BatchSize          int          `json:"batch_size"`
	UBatchSize         int          `json:"ubatch_size"`
	BackendTag         string       `json:"backend_tag,omitempty"` // "llama" or "ik_llama"
	IsMoE              bool         `json:"is_moe"`
	ReasoningOff       bool         `json:"reasoning_off"` // default off for OpenAI compat
	ThreadsBatch       int          `json:"threads_batch"` // batch threads (logical cores)
	Parallel           int          `json:"parallel,omitempty"`
	CRAM               int          `json:"cram,omitempty"` // prompt cache MB
	MaxCheckpoints     int          `json:"max_checkpoints,omitempty"`
	UseCUDAGraphs      bool         `json:"use_cuda_graphs,omitempty"`
	Host               string       `json:"host,omitempty"`        // listen address
	HasSSM             bool         `json:"has_ssm,omitempty"`     // SSM/Mamba hybrid flag
	Draft              *DraftConfig `json:"draft,omitempty"`       // speculative decoding config
	MMProjPath         string       `json:"mmproj_path,omitempty"` // vision projector GGUF
	MMProjSizeMB       int          `json:"-"`                     // mmproj VRAM on primary GPU
}

// ModelProfile describes the GGUF model.
type ModelProfile struct {
	Path              string `json:"path"`
	Name              string `json:"name,omitempty"`         // GGUF metadata: model name
	Basename          string `json:"basename,omitempty"`     // GGUF metadata: model basename
	QuantizedBy       string `json:"quantized_by,omitempty"` // GGUF metadata: quantizer (e.g. "unsloth")
	SizeBytes         int64  `json:"size_bytes"`
	TotalSizeMB       int    `json:"total_size_mb"` // includes multi-part shards
	NumLayers         int    `json:"num_layers"`
	NumParams         int64  `json:"num_params"`
	IsMoE             bool   `json:"is_moe"`
	NumExperts        int    `json:"num_experts,omitempty"`
	ContextSize       int    `json:"context_size"`
	HiddenSize        int    `json:"hidden_size"`
	HeadCount         int    `json:"head_count"`
	HeadCountKV       int    `json:"head_count_kv"`
	KeyLength         int    `json:"key_length"`
	ValueLength       int    `json:"value_length"`
	VocabSize         int    `json:"vocab_size"`
	QuantType         string `json:"quant_type"`
	ExpertBytes       int64  `json:"expert_bytes"`
	NonExpertBytes    int64  `json:"non_expert_bytes"`
	TokenEmbdBytes    int64  `json:"token_embd_bytes,omitempty"` // subset of NonExpertBytes; always host-resident
	OutputBytes       int64  `json:"output_bytes,omitempty"`     // subset of NonExpertBytes; whole on the last split device
	ShexpBytes        int64  `json:"shexp_bytes,omitempty"`      // subset of ExpertBytes; stays on the layer's device
	Fused             int    `json:"fused"`
	EmbeddingLength   int    `json:"embedding_length"`
	FeedForwardLength int    `json:"feed_forward_length"`
	KVLoraRank        int    `json:"kv_lora_rank"`
	QLoraRank         int    `json:"q_lora_rank"`
	RopeDim           int    `json:"rope_dim"`
	KeyLengthMLA      int    `json:"key_length_mla"`
	ValueLengthMLA    int    `json:"value_length_mla"`
	HasSSM            int    `json:"has_ssm"`
	SlidingWindow     int    `json:"sliding_window"`
	FullAttnInterval  int    `json:"full_attn_interval"`
	HasShexp          int    `json:"has_shexp"`
	CTXTrain          int    `json:"ctx_train"`
	ModelArch         string `json:"model_arch"`
	ExpertUsedCount   int    `json:"expert_used_count,omitempty"`
	// MeasuredKVBytesPerTok maps a KV cache type (e.g. "q8_0") to the KV cache
	// bytes-per-token that llama.cpp ACTUALLY allocated on a previous launch of
	// this model, read back from the backend log. It is the ground truth for
	// compressed-attention models (MLA/CSA-HCA/SWA) where the GGUF formula is
	// unreliable; computeKVTotalMB prefers it over the formula when present.
	MeasuredKVBytesPerTok map[string]float64 `json:"-"`
	ExpertFF              int                `json:"expert_ff,omitempty"`
	ExpertSharedFF        int                `json:"expert_shared_ff,omitempty"`
	LeadingDense          int                `json:"leading_dense,omitempty"`
	NextNPredictLayers    int                `json:"nextn_predict_layers,omitempty"`
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
	Host           string // listen address (default 127.0.0.1)
	VisionAuto     bool   // auto-detect mmproj for vision
	MMProjPath     string // explicit vision projector GGUF
	SpecMode       string // off, auto, draft, eagle3, ngram, ngram-mod, ngram-k4v, mtp
	BackendHelp    string // llama-server --help output for dialect-specific flags
	ForceSpecMoE   bool   // allow speculative decoding on MoE despite default gate
	ReasoningOff   bool   // emit `--reasoning off` (benchmark/tune only; normal serving keeps the model's thinking)
	// SkipPlacementCache disables loading the keyed .place cache for this Compute.
	// Set during a corrective OOM re-plan so it derives fresh from the penalized
	// VRAM instead of reloading the placement that just OOM'd.
	SkipPlacementCache bool
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

	// Load any KV cache size llama.cpp measured for this model on a prior launch,
	// so context sizing uses measured truth (exact for compressed attention).
	if model.MeasuredKVBytesPerTok == nil {
		if rates := loadMeasuredKVRates(opts.CacheDir, model); rates != nil {
			model.MeasuredKVBytesPerTok = rates
		}
	}

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
		FlashAttention: true, // finalized against the resolved KV placement before each return
		// Thinking stays ON for normal serving (backend default `--reasoning auto`);
		// only benchmark/tune opt in to `--reasoning off` for clean, fast measurement.
		ReasoningOff: opts.ReasoningOff,
		HasSSM:       model.HasSSM == 1,
		Host:         opts.Host,
		// ggrun sets explicit placement (-ngl/-ot/--tensor-split), so the backend's
		// own auto memory-fitting (-fit) is redundant with this explicit plan.
		// Disable it when the backend supports the flag.
		BackendSupportsFit: backendHelpSupports(opts.BackendHelp, "-fit"),
	}

	if s.ContextSize <= 0 {
		s.ContextSize = defaultContextSize(model, caps)
	}
	if opts.KVPlacement == "" {
		s.KVPlacement = "auto"
	}
	if opts.KVQuality == "" {
		// q8_0 KV cache: near-lossless, preserves model quality. The fitting
		// logic falls back to q4_0 only when VRAM genuinely can't hold q8_0.
		s.KVQuality = "mid"
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

	// Resolve KV placement "auto" → concrete value up-front so every caller and
	// every explicit-context retry sees the same placement policy.
	if s.KVPlacement == "auto" || s.KVPlacement == "" {
		if opts.CPUMode || len(caps.GPUs) == 0 {
			s.KVPlacement = "cpu"
		} else {
			sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
			perGPUOH := perGPUVRAMOverheadMB(sysProbe, 0)
			s.KVPlacement = resolveAutoKVPlacement(caps, model, totalSizeMB, perGPUOH*len(caps.GPUs))
		}
	}

	// Auto-fit context: compute both single-GPU and multi-GPU, pick the larger.
	if opts.ContextSize <= 0 {
		if opts.CPUMode || len(caps.GPUs) == 0 {
			cpuCaps := *caps
			cpuCaps.GPUs = nil
			s.ContextSize, s.KVType = computeAutoContextSize(&cpuCaps, model, totalSizeMB, s.KVType, opts)
		} else {
			sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
			// Per-GPU non-weight VRAM overhead, derived per-component (measured
			// CUDA overhead + compute buffer) — no flat guess, no extra margin.
			perGPUOH := perGPUVRAMOverheadMB(sysProbe, 0)

			bestFree := 0
			for _, g := range caps.GPUs {
				if g.VRAMFreeMB() > bestFree {
					bestFree = g.VRAMFreeMB()
				}
			}

			// Single-GPU estimate
			singleCtx, singleKV := computeAutoContextSizeSingleGPU(caps, model, totalSizeMB, s.KVType, opts)
			singleKVM := computeKVTotalMB(model, singleCtx, singleKV)
			singleFits := (totalSizeMB+perGPUOH+singleKVM) <= bestFree && singleCtx >= 32768

			// Multi-GPU estimate
			multiCtx, multiKV := computeAutoContextSize(caps, model, totalSizeMB, s.KVType, opts)
			multiKVM := computeKVTotalMB(model, multiCtx, multiKV)
			multiFree := 0
			for _, g := range caps.GPUs {
				multiFree += g.VRAMFreeMB()
			}
			multiFits := (totalSizeMB+perGPUOH*len(caps.GPUs)+multiKVM) <= multiFree && multiCtx >= 32768

			if multiFits && multiCtx > singleCtx {
				s.ContextSize, s.KVType = multiCtx, multiKV
			} else if singleFits {
				s.ContextSize, s.KVType = singleCtx, singleKV
			} else if multiFits {
				s.ContextSize, s.KVType = multiCtx, multiKV
			} else {
				// The model doesn't fit wholly in VRAM (a big MoE offloading experts
				// to CPU). Don't collapse to the 32768/q4_0 floor — size the context
				// by where its KV will actually live. --kv-placement drives it:
				// gpu → VRAM-bounded (safe, experts offload); cpu → RAM-bounded
				// (large window); auto → gpu if it fits, else cpu for a big MoE.
				placement := s.KVPlacement
				if placement == "auto" || placement == "" {
					placement = resolveAutoKVPlacement(caps, model, totalSizeMB, perGPUOH*len(caps.GPUs))
				}
				s.KVPlacement = placement
				s.ContextSize, s.KVType = computeAutoContextSizeKVPlacement(caps, model, totalSizeMB, s.KVType, placement, opts)
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

	// Batch tier by exact fit on the best single GPU: model + measured CUDA
	// overhead + KV + that tier's actual compute buffer must fit in VRAM. No
	// percentage guess, no fixed headroom — each tier's compute buffer is the
	// real cost of running that (u)batch (firstLaunchComputeBufMB).
	batchBaseMB := totalSizeMB + measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs)) + kvTotalMB
	switch {
	case batchBaseMB+firstLaunchComputeBufMB(model, 1024) <= bestGPUVRAM:
		s.BatchSize, s.UBatchSize = 8192, 1024
	case batchBaseMB+firstLaunchComputeBufMB(model, 512) <= bestGPUVRAM:
		s.BatchSize, s.UBatchSize = 4096, 512
	default:
		s.BatchSize, s.UBatchSize = 2048, 512
	}

	// Persist/reuse this exact placement under a key that includes kv placement,
	// ctx, ubatch, backend, and the GPU set — computed from the now-resolved
	// strategy so the launcher (save) and this load agree byte-for-byte.
	s.PlacementCachePath = PlacementCachePathFor(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, opts.BackendTag, caps.GPUs, s.Parallel, splitCompactKey(s.TensorSplit))

	// Try cached placement first (MoE only). Prefer the keyed placement cache
	// (remembers a load that landed right or was OOM-corrected); fall back to a
	// tune-cache file if one was selected.
	placeFile := opts.CacheFile
	if s.PlacementCachePath != "" {
		if _, statErr := os.Stat(s.PlacementCachePath); statErr == nil {
			placeFile = s.PlacementCachePath
		}
	}
	if opts.SkipPlacementCache {
		placeFile = ""
	}
	if placeFile != "" && model.IsMoE {
		cache, err := LoadPlacementCache(placeFile, caps, kvTotalMB)
		if err == nil && cache != nil {
			s.Type = MoEOffload
			s.BatchSize = cache.BatchSize
			s.UBatchSize = cache.UBatchSize
			s.Parallel = cache.Parallel
			// The caller's requested slot count wins over what happened to be
			// cached: parallel shapes slots and compute buffers, not the weight
			// layout the cache exists to remember. Without this, a placement
			// cached by a --parallel 1 CLI run silently stripped Claude Code
			// mode down to a single slot.
			if opts.Parallel > 0 && opts.Parallel != cache.Parallel {
				s.Parallel = opts.Parallel
			}
			s.NCPUMoE = cache.NCPUMoE
			// Restore the cached mmap decision, don't reset it: the entry was
			// saved from a load that landed right, and CACHED_MMAP=0 means it
			// loaded resident (--no-mmap). Resetting to mmap-allowed here made
			// a cache hit silently drop --no-mmap and page 100GB+ of experts
			// from disk. An explicit user --no-mmap still always wins.
			s.MMap = cache.MMap && !opts.NoMMap
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
			// Prefer the exact cached -ot (preserves sub-layer gate+up pins);
			// fall back to rebuilding from GPU assignments for legacy caches.
			if cache.OTString != "" {
				s.OTString = cache.OTString
			} else if len(cache.GPUAssignments) > 0 {
				otString := buildOTStringFromAssignments(cache.GPUAssignments, caps.GPUs, model.NumLayers, opts.BackendTag)
				if otString != "" {
					s.OTString = otString
				}
			}
			s.FlashAttention = defaultFlashAttention(model, opts, s.KVPlacement)
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
		preUBatch := *s // buildMoEOffload returns (nil, err) on hard failure, losing s.UBatchSize
		s, err = buildMoEOffload(s, caps, model, totalSizeMB, kvTotalMB, opts)
		s, err = maximizeMoEGPUFitByUBatch(&preUBatch, s, err, caps, model, totalSizeMB, kvTotalMB, opts)
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
		s.Host = "127.0.0.1"
	}

	s.FlashAttention = defaultFlashAttention(model, opts, s.KVPlacement)

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

	// Load measured CUDA overhead. Missing probe data is unknown and contributes 0;
	// preflight/startup OOM recording supplies measured data for later launches.
	cudaOverheadMB := measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs))

	// Load model probe for compute buffer
	computeBufMB := computeFloorMB // 1024 default
	pc := loadProbeCache(opts.CacheDir, model, opts.ContextSize, 512, opts.KVQuality, opts.KVPlacement, opts.BackendTag, caps.GPUs, opts.Parallel)
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
	// Use measured overhead: model weights + CUDA overhead + compute buffer + KV.
	// Do not subtract a static reserve from free VRAM here; free VRAM is already
	// the measured allocator-visible capacity after resident desktop/process usage.
	singleGPUNeeded := totalSizeMB + cudaOverheadMB + computeBufMB + kvTotalMB
	if singleGPUNeeded <= bestFreeVRAM {
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

	// Load measured CUDA overhead. Missing probe data is unknown and contributes 0.
	cudaOverheadMB := measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs))

	// Load model probe for compute buffer (same as MoE path)
	probeHit := false
	computeBufMB := computeFloorMB // 1024 default
	pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, opts.BackendTag, caps.GPUs, s.Parallel)
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
	// on mixed GPU architectures like Ampere+Ada). For mainline dense
	// multi-GPU use "tensor": "--split-mode row" triggers a GQA assert
	// in ggml-cuda.cu and segfaults mid-request on GQA models (audit #7).
	if opts.BackendTag == "ik_llama" {
		s.SplitMode = "layer"
	} else {
		s.SplitMode = "tensor"
	}

	_ = probeHit // used for logging/debugging

	return s, nil
}

func buildDenseCPUOffload(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	numGPUs := len(caps.GPUs)
	if numGPUs > 1 {
		// Load measured CUDA overhead. Missing probe data is unknown and contributes 0.
		cudaOverheadMB := measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs))

		// Load model probe for compute buffer
		computeBufMB := computeFloorMB
		pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, opts.BackendTag, caps.GPUs, s.Parallel)
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
			s.SplitMode = "tensor"
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
	// Input embeddings never leave host memory (llama.cpp keeps the input layer
	// on CPU), so they must not be charged against per-GPU VRAM budgets; they
	// are counted on the RAM side below instead.
	tokenEmbdMB := bytesToMiBCeil(model.TokenEmbdBytes)
	if tokenEmbdMB > 0 && tokenEmbdMB < nonExpertTotalMB {
		nonExpertTotalMB -= tokenEmbdMB
	} else {
		tokenEmbdMB = 0
	}
	// The output head is not pro-rata: llama.cpp assigns it to the device that
	// owns the last of its n_layer+1 split slots. Charging it proportionally
	// once OOM'd the smallest GPU (output.weight ~1GB landed whole on the 4070
	// while placement had budgeted an 0.11 share of it).
	outputMB := bytesToMiBCeil(model.OutputBytes)
	if outputMB > 0 && outputMB < nonExpertTotalMB {
		nonExpertTotalMB -= outputMB
	} else {
		outputMB = 0
	}
	// Shared experts ride with their layer's owning device — the exps=CPU
	// catch-all does not match "_shexp", so even CPU-offloaded layers keep
	// their shared expert in VRAM. GPU whole-layer pins already include shexp
	// in expertPerLayerMB; the CPU side must exclude it.
	shexpTotalMB := bytesToMiBCeil(model.ShexpBytes)
	if shexpTotalMB < 0 || shexpTotalMB >= expertTotalMB {
		shexpTotalMB = 0
	}
	expertPerLayerMB := ceilDivInt(expertTotalMB, moeLayerCount)
	if expertPerLayerMB <= 0 {
		expertPerLayerMB = 1
	}
	// RAM (and gate+up chunk) cost of a CPU-offloaded expert layer: routed
	// experts only, shared expert stays on the owning GPU.
	expertCPUPerLayerMB := ceilDivInt(expertTotalMB-shexpTotalMB, moeLayerCount)
	if expertCPUPerLayerMB <= 0 {
		expertCPUPerLayerMB = expertPerLayerMB
	}
	nonExpertForGPU := nonExpertTotalMB - bytesToMiBCeil(model.TokenEmbdBytes)
	if nonExpertForGPU < 0 {
		nonExpertForGPU = 0
	}
	nonExpertPerLayerMB := ceilDivInt(nonExpertForGPU, model.NumLayers)
	if nonExpertPerLayerMB <= 0 {
		nonExpertPerLayerMB = 1
	}

	// Load measured CUDA overhead per GPU. Missing probe entries are unknown
	// components, not free VRAM; they contribute 0 to whole-layer fitting but
	// block optional remainder squeeze below until a successful launch measures
	// them.
	sysCUDAOverheadByGPU := SystemCUDAOverheadByGPU(opts.CacheDir, caps.GPUs)

	// Load per-model/runtime probe cache. Until a model has completed one launch
	// with these settings, use a first-launch fallback that keeps the main GPU
	// conservative without charging the full prompt graph to every secondary GPU.
	pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, opts.BackendTag, caps.GPUs, s.Parallel)
	fixedPerGPU := make([]int, numGPUs)
	expertOnlyFixedPerGPU := make([]int, numGPUs)
	for i, g := range caps.GPUs {
		computeBufMB := firstLaunchComputeBufMBForGPU(model, s.UBatchSize, i, gpuOrder)
		runtimeGrowthMB := 0
		if pc != nil {
			// Use the aggregate (primary) compute buffer for split-owner cost
			// accounting. The per-GPU values in the probe cache are measured
			// for the specific placement that ran (e.g. GPU1 expert-only =
			// 299 MB, GPU0 split-owner = 33 GB at ub=256). When the placement
			// changes between measurement and this computation — which is
			// exactly what a re-plan does — the per-GPU values are wrong: a
			// GPU that was expert-only (299 MB) might become a split-owner
			// (needing the full ~33 GB), or vice versa. The aggregate
			// (primary's compute buffer) is the maximum any split-owner would
			// need, so using it for all GPUs is conservative but correct.
			// Expert-only GPUs use expertOnlyComputeReserveMB which caps at
			// the compute floor, so the aggregate doesn't affect them.
			if pc.ComputeBufMB > 0 {
				computeBufMB = pc.ComputeBufMB
			}
			// Real long requests can need more than the load-time graph
			// reserve accounts for — llama-server's context-checkpoint state
			// save (tools/server/server-context.cpp create_checkpoint) lives
			// entirely outside llama_context::sched_reserve(), so no no-alloc
			// measurement (ours or llama.cpp's own --fit) ever sees it.
			// Reproduced 2026-07-08: DeepSeek-V4 crashed needing +1679MB on
			// CUDA2 at 16384 real tokens despite a startup reserve sized for
			// the full max context. Once a canary or a real crash measures
			// this growth (RecordRuntimeGraphGrowth/-FromOOM), reserve it here
			// so the next placement for this exact key packs around it
			// instead of rediscovering the deficit by crashing again.
			runtimeGrowthMB = pc.RuntimeGraphGrowthByGPU[g.Index]
		}
		fixedPerGPU[i] = sysCUDAOverheadByGPU[g.Index] + computeBufMB + runtimeGrowthMB
		expertOnlyFixedPerGPU[i] = sysCUDAOverheadByGPU[g.Index] + expertOnlyComputeReserveMB(computeBufMB) + runtimeGrowthMB
	}
	expertOnlyGPU := expertOnlySlowGPUs(caps.GPUs, fixedPerGPU, expertOnlyFixedPerGPU, expertPerLayerMB, nonExpertPerLayerMB)

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
	// Only reserve GPU VRAM for the KV cache when it actually lives on the GPU.
	// With KV on CPU (big-MoE auto-placement, or --kv-placement cpu) the experts
	// must be free to fill that VRAM instead — otherwise every GPU sits half-empty
	// holding room for a KV cache that's in system RAM.
	gpuKVTotalMB := kvTotalMB
	if s.KVPlacement == "cpu" {
		gpuKVTotalMB = 0
	}

	// nonExpertChargeMB is the exact VRAM a GPU pays for non-expert weights
	// under llama.cpp's real slot assignment: its owned repeating layers times
	// the per-layer non-expert bytes, the shared experts of its owned layers,
	// and the whole output head iff it owns the last slot. This replaces the
	// old pro-rata split share, which spread the ~1GB output head across all
	// GPUs and OOM'd whichever small card actually received it.
	nonExpertChargeMB := func(owned []int, outputDev, gi int) int {
		perLayer := float64(nonExpertTotalMB) / float64(model.NumLayers)
		perShexp := 0.0
		if moeLayerCount > 0 {
			perShexp = float64(shexpTotalMB) / float64(moeLayerCount)
		}
		charge := float64(owned[gi]) * (perLayer + perShexp)
		if gi == outputDev {
			charge += float64(outputMB)
		}
		return int(charge + 0.999)
	}

	used := make([]bool, numGPUs)
	for i, g := range caps.GPUs {
		free := g.VRAMFreeMB()
		fixed := fixedPerGPU[i]
		used[i] = !expertOnlyGPU[i] && free > fixed
	}
	var split []float64
	var ownedLayers []int
	outputDev := -1
	for {
		rawSplit := make([]float64, numGPUs)
		totalWeighted := 0.0
		for i, g := range caps.GPUs {
			if used[i] {
				// Weight by VRAM available AFTER the fixed per-GPU CUDA/compute
				// reserve, not raw free VRAM, so high-overhead GPUs are not
				// over-allocated (audit #4).
				avail := float64(g.VRAMFreeMB() - fixedPerGPU[i])
				totalWeighted += avail * gpuSplitWeight(g)
			}
		}
		if totalWeighted <= 0 {
			return nil, fmt.Errorf("Model does not fit on this system: no GPU has free VRAM after CUDA/compute overhead")
		}
		for i, g := range caps.GPUs {
			if used[i] {
				avail := float64(g.VRAMFreeMB() - fixedPerGPU[i])
				rawSplit[i] = avail * gpuSplitWeight(g) / totalWeighted
			}
		}
		split = normalizeSplit(rawSplit)
		ownedLayers, outputDev = layerOwnership(split, model.NumLayers)

		removed := false
		for i, g := range caps.GPUs {
			if !used[i] {
				continue
			}
			kvShareMB := ownedShareMB(gpuKVTotalMB, ownedLayers, model.NumLayers, i)
			if fixedPerGPU[i]+nonExpertChargeMB(ownedLayers, outputDev, i)+kvShareMB > g.VRAMFreeMB() {
				used[i] = false
				removed = true
			}
		}
		if !removed {
			break
		}
	}
	// Post-split retrofit: a GPU eliminated from the split (KV or compute buffer
	// exceeded its share) but not already classified expert-only can still hold
	// whole expert layers. Without this, such a GPU sits idle while its VRAM
	// could offload expert layers from CPU RAM. Reclassify it as expert-only
	// when it can fit at least one complete expert layer under the expert-only
	// reserve, so the expert-capacity loop below picks it up. Uses VRAMTotalMB
	// (not the penalized VRAMFreeMB) because the OOM penalty from a prior hybrid
	// placement attempt does not apply to an expert-only placement — the deficit
	// was caused by dense weights + KV, which the expert-only GPU does not carry.
	for i, g := range caps.GPUs {
		if expertOnlyGPU[i] || used[i] || split[i] > 0 {
			continue
		}
		if i >= len(expertOnlyFixedPerGPU) {
			continue
		}
		if g.VRAMTotalMB-expertOnlyFixedPerGPU[i] >= expertPerLayerMB {
			expertOnlyGPU[i] = true
		}
	}
	if numGPUs > 1 {
		s.TensorSplit = split
	}

	// Per-GPU expert capacity under the exact emitted split. roomMBPer keeps the
	// exact VRAM budget for experts on each GPU so sub-layer packing can later
	// fill the remainder that whole-layer flooring leaves stranded.
	maxGPULayersPer := make([]int, numGPUs)
	roomMBPer := make([]int, numGPUs)
	maxGPULayers := 0
	for _, gi := range gpuOrder {
		if split[gi] <= 0 && !expertOnlyGPU[gi] {
			continue
		}
		g := caps.GPUs[gi]
		kvShareMB := ownedShareMB(gpuKVTotalMB, ownedLayers, model.NumLayers, gi)
		fixedMB := fixedPerGPU[gi]
		if expertOnlyGPU[gi] {
			fixedMB = expertOnlyFixedPerGPU[gi]
		}
		// Expert-only GPUs carry the non-expert tensors (norms, attention) for
		// every layer pinned to them via -ot, even though ownedLayers==0. Keep
		// the compute-buffer floor: expert-only GPUs run the per-layer forward
		// pass for their pinned layers, so the graph reserve must cover norms,
		// routing, and replicated layer tensors. The probe data shows the
		// actual expert-only compute buffer is small (~299 MB for 3 layers on
		// DeepSeek-V4), well under the 1024 MB floor.
		if expertOnlyGPU[gi] && fixedMB < computeFloorMB {
			fixedMB = computeFloorMB
		}
		nonExpertCharge := nonExpertChargeMB(ownedLayers, outputDev, gi)
		if expertOnlyGPU[gi] {
			nonExpertCharge = 0 // zero here; per-layer cost accounts for it below
		}
		var roomMB int
		if expertOnlyGPU[gi] {
			// Expert-only GPUs don't own dense layers or KV, so their room is
			// the full VRAM minus the expert-only fixed costs. Use VRAMTotalMB
			// to avoid the OOM penalty from a prior hybrid placement attempt —
			// the penalty was for a different placement type (split-owner with
			// dense + KV), not for expert-only. The preflight gate validates
			// the actual fit before the load.
			roomMB = g.VRAMTotalMB - fixedMB
		} else {
			roomMB = g.VRAMFreeMB() - fixedMB - nonExpertCharge - kvShareMB
		}
		if roomMB < 0 {
			roomMB = 0
		}
		roomMBPer[gi] = roomMB
		perLayerCost := expertPerLayerMB
		if expertOnlyGPU[gi] {
			nonExpertPerLayer := float64(nonExpertTotalMB) / float64(model.NumLayers)
			perLayerCost = expertPerLayerMB + int(nonExpertPerLayer+0.5)
		}
		capLayers := roomMB / perLayerCost
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

	// Strict ceiling (--no-mmap path). Uses the real measured-overhead
	// formula (not a hand-copied subset) so this ceiling and the later real
	// RAM check (checkMemoryOrDie -> ramRuntimeOverheadMB) can't disagree —
	// a stale copy here was missing the cpuActMB term, letting this ceiling
	// accept a CPU-layer count the real check would then reject.
	ramOverheadPreMB := ramRuntimeOverheadMB(model, s.UBatchSize, totalSizeMB)
	cpuBudgetStrict := caps.RAM.FreeMB - ramOverheadPreMB - cpuKVRAMMB
	if cpuBudgetStrict < 0 {
		cpuBudgetStrict = 0
	}
	maxCPULayersStrict := 0
	if expertCPUPerLayerMB > 0 {
		maxCPULayersStrict = cpuBudgetStrict / expertCPUPerLayerMB
	}
	if maxCPULayersStrict > moeLayerCount {
		maxCPULayersStrict = moeLayerCount
	}

	// Mmap-aware ceiling.
	//
	// Expert weight bytes live in a read-only, file-backed mmap: the kernel can
	// always evict those clean pages under memory pressure and re-fault them
	// from the model file, so they never have to be simultaneously resident.
	// What CANNOT be reclaimed is the runtime's own anonymous memory — CUDA
	// host staging, graph scratch, the mmap page-table, CPU activation buffers
	// (ramOverheadPreMB), and the KV cache (cpuKVRAMMB, continuously
	// read/written, not file-backed) — if that doesn't fit in free RAM, no
	// amount of expert-page eviction helps, and mmap would thrash from the
	// first token. That is the real (measured, not guessed) floor: it replaces
	// both the old flat "8 layers" floor (too lenient — let a model whose
	// working set vastly exceeds even non-reclaimable RAM through) and the
	// "full footprint must fit" floor (too strict — defeated the entire point
	// of mmap, which is to hold LESS than the full footprint resident).
	preWorkingSetFloor := ramOverheadPreMB + cpuKVRAMMB
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

	// Sub-layer expert packing (GPU squeeze): whole-layer packing floors each
	// GPU's capacity, stranding up to ~expertPerLayerMB of VRAM per card. Fill
	// that remainder with the gate+up projections (2/3 of a layer) of the next
	// CPU-bound layers — down stays on CPU — so stranded VRAM becomes expert
	// residency: more experts on GPU (faster) and that weight leaves system RAM
	// (breathing room). Sizing is exact: gate+up is 2/3 of the file-anchored
	// per-layer ROUTED expert bytes — the shared expert is not part of a chunk
	// (it already lives on the layer's owning GPU).
	gateUpChunkMB := 2 * expertCPUPerLayerMB / 3
	remainderMB := make([]int, numGPUs)
	for gi := range caps.GPUs {
		r := roomMBPer[gi] - layersPerGPU[gi]*expertPerLayerMB
		// Expert-only slow-link GPUs are used for whole expert layers only. A
		// partial gate+up pin on a PCIe x1 device creates cross-device expert
		// traffic without making a full layer self-contained.
		if expertOnlyGPU[gi] {
			r = 0
		}
		if r < 0 {
			r = 0
		}
		remainderMB[gi] = r
	}
	var subPins []subExpertPin
	movedOffCPUMB := 0
	if hasMeasuredCUDAOverheadForActiveGPUs(sysCUDAOverheadByGPU, caps.GPUs, split) {
		subPins, movedOffCPUMB = packGateUpChunks(remainderMB, gpuOrder, gateUpChunkMB, moeStartLayer+totalGPULayers, layersCPU)
	}

	// RAM safety check. Sub-layer pins move gate+up off the CPU, so the resident
	// CPU expert footprint shrinks by exactly the packed bytes. CPU layers cost
	// only their routed experts — shared experts stay in VRAM.
	cpuExpertMB := layersCPU*expertCPUPerLayerMB - movedOffCPUMB
	if cpuExpertMB < 0 {
		cpuExpertMB = 0
	}

	// Non-weight RAM overhead, derived per-component (see ramRuntimeOverheadMB):
	// CUDA host staging + graph scratch + mmap page table + CPU activation.
	ramOverheadMB := ramRuntimeOverheadMB(model, s.UBatchSize, totalSizeMB)

	cpuKVMB := 0
	if kvPlacementEffective == "cpu" {
		cpuKVMB = kvTotalMB
	}
	ramNeeded := cpuExpertMB + cpuKVMB + ramOverheadMB + tokenEmbdMB
	ramAvailMB := caps.RAM.FreeMB

	// Mmap decision for MoE — VRAM-aware, no guessed margin.
	//
	// mmap is a question about RAM, not total model size. Expert layers placed
	// on the GPUs live in VRAM and cost zero system RAM, so what matters is the
	// CPU-resident footprint only: CPU-side experts + CPU-side KV + activation/
	// compute overhead (that is exactly ramNeeded). Compare it against the real
	// available RAM (MemAvailable, already net of any user --ram-headroom) with
	// no fudge factor — a guessed % or fixed-MB cushion breaks at scale (trivial
	// on a 1TB box, wasteful as a percentage on a huge model). If the resident
	// footprint fits, load resident (fast, no SSD paging); otherwise mmap while
	// the working set fits. A deliberate reserve is the user's --ram-headroom.
	//
	// The old test keyed off totalSizeMB, which ignored VRAM entirely: a 146GB
	// model with ~40GB of experts on the GPUs (leaving ~100GB on CPU, well under
	// 122GB RAM) was forced onto mmap and paged from SSD for no reason.
	if opts.NoMMap {
		s.MMap = false
	} else if ramNeeded > ramAvailMB {
		// The CPU-resident expert bytes counted in ramNeeded are clean,
		// file-backed mmap pages — evictable and re-fault-able from the model
		// file, so they don't have to all be resident at once. What must
		// actually fit is the runtime's non-reclaimable anonymous memory: KV
		// cache (continuously read/written) plus compute/activation overhead.
		// (Same reasoning as preWorkingSetFloor above; keep both in sync.)
		workingSetFloor := ramOverheadMB + cpuKVMB
		if ramAvailMB >= workingSetFloor {
			s.MMap = true
		}
	} else {
		s.MMap = false
	}

	// Build -ot string. Always include the exps=CPU catch-all so expert
	// tensors never follow the backend's default layer split by accident.
	otString := buildOTStringWithSubPins(layersPerGPU, subPins, caps.GPUs, gpuOrder, moeStartLayer, opts.BackendTag)
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

// UBatchFitLadder are ubatch sizes tried, largest first, when the default
// ubatch's compute buffer leaves literally zero VRAM for any MoE expert
// layer. Flash-attention's compute buffer scales with ubatch at large
// context — measured on DeepSeek-V4 at ctx 1048576, f16 KV, 3-way tensor
// split: ~17-20GB/GPU at ubatch 512 vs ~2.5-2.8GB/GPU at ubatch 64. A
// smaller ubatch trades prefill batch size for GPU-resident experts, which
// is the better trade once every decode step would otherwise stream all
// experts from CPU RAM. Exported so the launch-time preflight (cmd/ggrun)
// can pre-measure every rung via the fit-params oracle before this ladder
// runs, instead of the retry silently falling back to the first-launch
// heuristic for ubatch values that were never actually measured.
var UBatchFitLadder = []int{256, 128, 64}

// numGPUsExcluded counts GPUs that got no tensor-split share and no explicit
// expert pins in a multi-GPU MoE placement. A zero split is acceptable for a
// deliberately expert-only slow PCIe GPU; it is only broken when the GPU is
// completely unused.
func numGPUsExcluded(s *Strategy, gpus []detect.GPU) int {
	numGPUs := len(gpus)
	if s == nil || numGPUs <= 1 || len(s.TensorSplit) != numGPUs {
		return 0
	}
	excluded := 0
	for i, v := range s.TensorSplit {
		if v <= 0 && !otStringUsesDevice(s.OTString, gpus[i].Index) {
			excluded++
		}
	}
	return excluded
}

// maximizeMoEGPUFitByUBatch retries buildMoEOffload at a smaller ubatch when
// the default ubatch produces a placement that wastes GPU capacity: (a)
// starves every expert layer off every GPU (NCPUMoE == total MoE layers),
// (b) fails to fit at all — flash-attention's compute buffer can be large
// enough at big contexts that compute+KV alone exceed a GPU's VRAM before any
// expert weight is even considered, surfacing as a hard "does not fit" error
// rather than a soft zero-experts result — or (c) excludes a whole GPU from
// the tensor split even though other GPUs are in use. base carries the
// pre-call strategy fields (ContextSize, KVPlacement, UBatchSize, ...) since
// a hard failure returns (nil, err) and loses them. Each retry reuses
// buildMoEOffload's own measured-or-heuristic compute accounting for that
// ubatch — no new margin is invented, only a different, real ubatch is
// tried — and stops at the largest ladder rung that measurably improves the
// placement, preserving as much prefill batching as the VRAM allows.
func maximizeMoEGPUFitByUBatch(base, s *Strategy, err error, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	if base == nil || model == nil {
		return s, err
	}
	_, moeCount := moeLayerRange(model)
	if moeCount <= 0 {
		return s, err
	}
	numGPUs := 0
	if caps != nil {
		numGPUs = len(caps.GPUs)
	}
	var gpus []detect.GPU
	if caps != nil {
		gpus = caps.GPUs
	}
	baseExcluded := numGPUsExcluded(s, gpus)
	if err == nil && s != nil && s.NCPUMoE == 0 && baseExcluded == 0 {
		return s, err // already maximal — every expert layer is on a GPU and every GPU is in use
	}
	best, bestErr, bestExcluded := s, err, baseExcluded
	bestNCPUMoE := moeCount + 1
	if err == nil && s != nil {
		bestNCPUMoE = s.NCPUMoE
	}
	for _, ub := range UBatchFitLadder {
		if ub >= base.UBatchSize {
			continue
		}
		cand := *base
		cand.UBatchSize = ub
		if cand.BatchSize < ub {
			cand.BatchSize = ub
		}
		next, cerr := buildMoEOffload(&cand, caps, model, totalSizeMB, kvTotalMB, opts)
		if cerr != nil {
			continue
		}
		nextExcluded := numGPUsExcluded(next, gpus)
		if next.NCPUMoE >= bestNCPUMoE && nextExcluded >= bestExcluded {
			continue // not an improvement over the best rung found so far
		}
		reason := "left no VRAM for MoE experts"
		if baseExcluded > 0 {
			reason = fmt.Sprintf("stranded %d of %d GPUs with zero tensor-split share", baseExcluded, numGPUs)
		}
		fmt.Fprintf(os.Stderr,
			"[placement] ubatch %d %s at this context/KV type — using ubatch %d instead (%d expert layer(s) on GPU, %d/%d GPUs used)\n",
			base.UBatchSize, reason, ub, moeCount-next.NCPUMoE, numGPUs-nextExcluded, numGPUs)
		best, bestErr, bestNCPUMoE, bestExcluded = next, nil, next.NCPUMoE, nextExcluded
	}
	return best, bestErr
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

// layerOwnership mirrors llama.cpp's tensor-split slot assignment exactly
// (llama-model.cpp): n_layer+1 slots — the repeating layers plus the output
// head — are distributed by upper_bound over the cumulative normalized split.
// Returns the owned repeating-layer count per device and the index of the
// device that owns the output slot (-1 if no device has a share). The input
// layer (token embeddings) always stays on the CPU and owns no slot.
func layerOwnership(split []float64, numLayers int) (owned []int, outputDev int) {
	owned = make([]int, len(split))
	layerDevices, outputDev := layerDeviceAssignments(split, numLayers)
	for _, dev := range layerDevices {
		if dev >= 0 && dev < len(owned) {
			owned[dev]++
		}
	}
	return owned, outputDev
}

// layerDeviceAssignments returns the exact device index for every repeating
// layer plus the output slot. Keeping the per-layer ownership is needed when a
// cost applies only to the MoE suffix (for example shared experts).
func layerDeviceAssignments(split []float64, numLayers int) (layerDevices []int, outputDev int) {
	layerDevices = make([]int, numLayers)
	for i := range layerDevices {
		layerDevices[i] = -1
	}
	outputDev = -1
	sum := 0.0
	for _, v := range split {
		if v > 0 {
			sum += v
		}
	}
	if sum <= 0 || numLayers <= 0 {
		return
	}
	cum := make([]float64, len(split))
	c := 0.0
	for i, v := range split {
		if v > 0 {
			c += v
		}
		cum[i] = c / sum
	}
	slots := numLayers + 1
	for slot := 0; slot < slots; slot++ {
		f := float64(slot) / float64(slots)
		dev := -1
		for i := range cum {
			if cum[i] > f {
				dev = i
				break
			}
		}
		if dev < 0 {
			dev = len(split) - 1
		}
		if slot == numLayers {
			outputDev = dev
		} else {
			layerDevices[slot] = dev
		}
	}
	return layerDevices, outputDev
}

// ownedShareMB charges a device its owned-layer fraction of a per-layer total
// (e.g. the KV cache, which llama.cpp allocates on each layer's device).
func ownedShareMB(totalMB int, owned []int, numLayers, idx int) int {
	if totalMB <= 0 || numLayers <= 0 || idx < 0 || idx >= len(owned) || owned[idx] <= 0 {
		return 0
	}
	return int(math.Ceil(float64(totalMB) * float64(owned[idx]) / float64(numLayers)))
}

// hasMeasuredCUDAOverheadForActiveGPUs gates optional sub-layer squeeze.
// VERIFICATION: cold-cache placement must not treat an unmeasured CUDA context
// as free remainder; no percentage/static reserve is hidden here.
func hasMeasuredCUDAOverheadForActiveGPUs(overheadByGPU map[int]int, gpus []detect.GPU, split []float64) bool {
	if len(gpus) == 0 || len(split) == 0 {
		return false
	}
	for i, g := range gpus {
		if i >= len(split) || split[i] <= 0 {
			continue
		}
		if overheadByGPU[g.Index] <= 0 {
			return false
		}
	}
	return true
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

// ReplanAfterOOM recomputes the full placement after a cudaMalloc OOM, with the
// failed device(s) penalized by how much they overshot. Because it re-runs the
// real packer, the correction is fill-preserving: it refits the failed card
// tightly (partial gate+up chunks, not whole layers) AND reclaims stranded VRAM
// on the other cards via the sub-pin squeeze — so experts move off system RAM
// instead of a blind whole-layer drop that over-corrects and erases the squeeze.
// penaltyMB is keyed by GPU Index and accumulates across retries. Returns the new
// strategy, or an error if there's nothing to replan or it no longer fits.
func ReplanAfterOOM(caps *detect.Capabilities, model *ModelProfile, opts Options, penaltyMB map[int]int) (*Strategy, error) {
	if caps == nil || model == nil || len(penaltyMB) == 0 {
		return nil, fmt.Errorf("replan: nothing to do")
	}
	c2 := *caps
	c2.GPUs = append([]detect.GPU(nil), caps.GPUs...)
	any := false
	for i := range c2.GPUs {
		if p := penaltyMB[c2.GPUs[i].Index]; p > 0 {
			c2.GPUs[i].VRAMUsedMB += p // shrink usable VRAM by the overshoot
			any = true
		}
	}
	if !any {
		return nil, fmt.Errorf("replan: no matching device")
	}
	o := opts
	o.SkipPlacementCache = true // derive fresh, don't reload the placement that OOM'd
	o.CacheFile = ""
	return Compute(&c2, model, o)
}

// currentUBatch reads the launch args' current -ub/--ubatch-size value, or 0
// if unset/unparseable.
func currentUBatch(args []string) int {
	idx := argIndex(args, "-ub", "--ubatch-size")
	if idx < 0 || idx+1 >= len(args) {
		return 0
	}
	v, _ := strconv.Atoi(args[idx+1])
	return v
}

// nextUBatchDown returns the next smaller rung on the same fit ladder
// maximizeMoEGPUFitByUBatch uses at placement time, or ok=false if current is
// already at or below the ladder's floor.
func nextUBatchDown(current int) (int, bool) {
	for _, rung := range UBatchFitLadder {
		if rung < current {
			return rung, true
		}
	}
	return 0, false
}

// DerateCUDAOOMArgs recovers from a cudaMalloc load failure. isComputeBuffer
// distinguishes the two failure classes the caller can observe in the
// backend log: a graph_reserve/gallocr (compute-buffer) OOM scales with
// ubatch, not expert-layer placement, so shrinking ubatch one rung down the
// same ladder used at placement time is tried first; a model-weight
// allocation failure (isComputeBuffer=false) goes straight to moving expert
// layers from the failed device back to CPU, since ubatch has no bearing on
// weight tensor size.
func DerateCUDAOOMArgs(args []string, model *ModelProfile, caps *detect.Capabilities, device, allocMB int, isComputeBuffer bool) ([]string, *CacheEntry, bool) {
	if model == nil || model.NumLayers <= 0 || allocMB <= 0 {
		return nil, nil, false
	}
	if isComputeBuffer {
		if next, ok := nextUBatchDown(currentUBatch(args)); ok {
			newArgs := append([]string(nil), args...)
			setOrAppendArg(&newArgs, "-ub", strconv.Itoa(next))
			// Keep the in-memory Strategy's UBatchSize in sync with serverArgs —
			// applyDeratedPlacementEntry applies this to strategy, which is what
			// the success path persists to the .place cache. Without it, a cache
			// hit later would resurrect the OOM'd, too-large ubatch.
			return newArgs, &CacheEntry{UBatchSize: next}, true
		}
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
	// Persist the resolved KV placement so a cache hit re-applies it (audit #3):
	// without this, no .place cache carries CACHED_KVUNIFIED and the load-side
	// check at placement.go:397 never fires.
	if argIndex(args, "--kv-offload") >= 0 {
		entry.KVUnified = true
	} else if argIndex(args, "--no-kv-offload") >= 0 {
		entry.KVUnified = false
	}
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
	expertPattern := `ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp)`

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

// subExpertPin pins one MoE layer's gate+up expert projections (2/3 of the
// layer's expert weight) to a specific GPU. The layer's down projection is left
// unpinned so the exps=CPU catch-all keeps it in system RAM.
type subExpertPin struct {
	Layer int // absolute layer index
	GI    int // position in caps.GPUs
}

// packGateUpChunks fills the VRAM that whole-layer packing floors away. Whole
// layers cost ~expertPerLayerMB each, so each GPU strands up to that much VRAM
// (its remainder). A layer's gate+up projections are 2/3 of its expert weight;
// pinning them onto a GPU with leftover room — down stays on CPU — turns stranded
// VRAM into expert residency: more experts on the GPU, that much less in system
// RAM. Greedy by remainder: each CPU-bound layer goes to the GPU (bandwidth
// order for ties) with the most room that can still hold a gate+up chunk.
// remainderMB is indexed by caps.GPUs position; gpuOrder gives the fill order.
// Returns the pins and the total expert MB moved off the CPU.
func packGateUpChunks(remainderMB []int, gpuOrder []int, gateUpChunkMB, cpuStartLayer, cpuLayerCount int) ([]subExpertPin, int) {
	if gateUpChunkMB <= 0 || cpuLayerCount <= 0 {
		return nil, 0
	}
	rem := make([]int, len(remainderMB))
	copy(rem, remainderMB)
	var pins []subExpertPin
	movedMB := 0
	for i := 0; i < cpuLayerCount; i++ {
		best := -1
		for _, gi := range gpuOrder {
			if gi < 0 || gi >= len(rem) {
				continue
			}
			if rem[gi] >= gateUpChunkMB && (best < 0 || rem[gi] > rem[best]) {
				best = gi
			}
		}
		if best < 0 {
			break // no GPU can hold another gate+up chunk
		}
		pins = append(pins, subExpertPin{Layer: cpuStartLayer + i, GI: best})
		rem[best] -= gateUpChunkMB
		movedMB += gateUpChunkMB
	}
	return pins, movedMB
}

// buildOTStringWithSubPins is buildOTStringFromStart plus optional sub-layer
// gate+up pins. Whole-layer pins come first, then the gate+up pins, then the
// exps=CPU catch-all — first-match-wins (see llama.cpp arg.cpp) keeps each
// partial layer's gate+up on its GPU while down falls through to CPU. With no
// sub-pins the output is identical to buildOTStringFromStart.
func buildOTStringWithSubPins(layersPerGPU []int, subPins []subExpertPin, gpus []detect.GPU, gpuOrder []int, startLayer int, backendTag string) string {
	var parts []string
	expertPattern := `ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp)`
	gateUpPattern := `ffn_(gate_up|up_gate|gate|up)_(ch|)exps`

	nextLayer := startLayer
	for _, gi := range gpuOrder {
		count := layersPerGPU[gi]
		if count > 0 {
			start := nextLayer
			last := start + count - 1
			var layerParts []string
			for l := start; l <= last; l++ {
				layerParts = append(layerParts, fmt.Sprintf("%d", l))
			}
			parts = append(parts, fmt.Sprintf(`blk\.(%s)\.%s.*=%s`, stringsJoin(layerParts, "|"), expertPattern, deviceName(backendTag, gpus[gi].Index)))
			nextLayer += count
		}
	}

	// Group sub-pins by GPU, preserving first-seen order, one compact rule per
	// GPU, emitted before the exps=CPU catch-all.
	if len(subPins) > 0 {
		byGPU := map[int][]int{}
		var order []int
		for _, p := range subPins {
			if _, ok := byGPU[p.GI]; !ok {
				order = append(order, p.GI)
			}
			byGPU[p.GI] = append(byGPU[p.GI], p.Layer)
		}
		for _, gi := range order {
			var layerParts []string
			for _, l := range byGPU[gi] {
				layerParts = append(layerParts, fmt.Sprintf("%d", l))
			}
			parts = append(parts, fmt.Sprintf(`blk\.(%s)\.%s.*=%s`, stringsJoin(layerParts, "|"), gateUpPattern, deviceName(backendTag, gpus[gi].Index)))
		}
	}

	parts = append(parts, "exps=CPU")
	return stringsJoin(parts, ",")
}

func buildOTStringFromAssignments(assignments []GPUAssignment, gpus []detect.GPU, numLayers int, backendTag string) string {
	var parts []string
	expertPattern := `ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp)`

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

func otStringUsesDevice(ot string, index int) bool {
	return strings.Contains(ot, fmt.Sprintf("=CUDA%d", index)) ||
		strings.Contains(ot, fmt.Sprintf("=Vulkan%d", index))
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
	// Prefer the KV size llama.cpp actually allocated on a previous launch (read
	// back from its log) — it is exact for every attention scheme, including the
	// compressed ones (MLA / CSA-HCA / sliding-window) the formula below can't
	// model. Falls through to the per-arch estimate when we have no measurement.
	if r, ok := model.MeasuredKVBytesPerTok[strings.ToLower(kvType)]; ok && r > 0 {
		return int(r*float64(ctxSize)/1048576.0 + 0.5)
	}

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

// expertOnlySlowGPUs classifies GPUs as expert-only devices: they must not own
// regular layer slots, but their VRAM can still hold whole expert layers. A GPU
// is an expert-only candidate when EITHER (a) its PCIe link is very slow
// relative to the fastest GPU (bandwidth ratio <= expertOnlyMaxBandwidthRatio),
// so owning dense layers would bottleneck the layer pipeline, OR (b) it cannot
// fit the split-owner compute reserve plus one dense layer's non-expert weight
// (capacity trigger), so a dense split slot would be wasted on it. The
// classification uses the normal split-owner reserve to make sure at least one
// true layer owner remains, and the expert-only reserve to decide whether a
// candidate card can carry at least one complete expert layer.
func expertOnlySlowGPUs(gpus []detect.GPU, splitFixedPerGPU, expertOnlyFixedPerGPU []int, expertPerLayerMB, nonExpertPerLayerMB int) []bool {
	expertOnly := make([]bool, len(gpus))
	if len(gpus) <= 1 || expertPerLayerMB <= 0 {
		return expertOnly
	}
	maxBandwidth := 0
	for _, g := range gpus {
		if g.BandwidthMBps > maxBandwidth {
			maxBandwidth = g.BandwidthMBps
		}
	}
	if maxBandwidth <= 0 {
		return expertOnly
	}

	splitCandidates := 0
	for i, g := range gpus {
		if i < len(splitFixedPerGPU) && g.VRAMFreeMB() > splitFixedPerGPU[i] {
			splitCandidates++
		}
	}
	candidates := make([]int, 0, len(gpus))
	for i, g := range gpus {
		if i >= len(splitFixedPerGPU) || i >= len(expertOnlyFixedPerGPU) || g.BandwidthMBps <= 0 {
			continue
		}
		slowLink := float64(g.BandwidthMBps)/float64(maxBandwidth) <= expertOnlyMaxBandwidthRatio
		cantFitDense := nonExpertPerLayerMB > 0 && g.VRAMFreeMB()-splitFixedPerGPU[i] < nonExpertPerLayerMB
		if !slowLink && !cantFitDense {
			continue
		}
		if g.VRAMFreeMB()-expertOnlyFixedPerGPU[i] < expertPerLayerMB {
			continue
		}
		candidates = append(candidates, i)
	}
	sort.Slice(candidates, func(i, j int) bool {
		gi := gpus[candidates[i]]
		gj := gpus[candidates[j]]
		if gi.BandwidthMBps != gj.BandwidthMBps {
			return gi.BandwidthMBps < gj.BandwidthMBps
		}
		if gi.VRAMFreeMB() != gj.VRAMFreeMB() {
			return gi.VRAMFreeMB() > gj.VRAMFreeMB()
		}
		return gi.Index < gj.Index
	})
	for _, i := range candidates {
		if splitCandidates <= 1 {
			break
		}
		expertOnly[i] = true
		splitCandidates--
	}
	return expertOnly
}

// expertOnlyComputeReserveMB is the compute margin for a GPU that will only run
// explicitly pinned experts. It keeps measured/recorded CUDA and runtime growth
// outside this helper and caps the prompt-graph reserve at llama.cpp's compute
// floor, because an expert-only GPU does not own regular prompt-processing
// layer slots or KV. The probe cache's per-GPU compute buffer is measured for
// the placement that actually ran (split-owner with dense layers), which is
// dramatically larger than what an expert-only GPU needs (e.g. DeepSeek-V4:
// ~9.8 GB split-owner vs ~299 MB expert-only on the same card). Using the
// split-owner value for an expert-only GPU blocks it from receiving any expert
// layers at all. Cap at the compute floor: it is a conservative reserve for
// the small expert-projection graph, and the preflight gate catches any real
// overflow before the load.
func expertOnlyComputeReserveMB(splitOwnerComputeMB int) int {
	return computeFloorMB
}

// modelAwareHeadroom estimates the non-weight VRAM/RAM the runtime needs beyond
// the model weights (prompt-graph compute buffer + a small runtime-growth
// margin). Replaces the flat 8 GiB guess previously hard-coded in the auto
// context-size paths so large models reserve enough and small models don't
// waste context capacity (audit #8).
func modelAwareHeadroom(model *ModelProfile) int {
	return firstLaunchComputeBufMB(model, 512) + 1024
}

// firstLaunchComputeBufMB is a conservative compute-buffer reservation used until
// the post-launch probe measures the real value for this model + settings. The
// prompt-processing graph scales with ubatch AND model shape: activation working
// set per token is roughly proportional to hidden_size * num_layers. The old flat
// ~4 MiB/ubatch estimate (ub 512 -> 2048 MiB) under-reserved large MoE graphs by
// ~4.4x — once expert-packing filled the GPU, llama.cpp's compute-buffer alloc
// OOM'd ("failed to create context", V4 needs ~9020 MiB at ub 512). We now size
// from the model: bytes per (token·hidden·layer) ≈ 42, calibrated so V4 reserves
// ~9412 MiB (covers the measured ~9020). Over-estimating is safe — the probe cache
// overrides this with the measured value after the first launch; under-estimating
// is fatal (OOM crash). A nil model falls back to the old per-ubatch heuristic.
func firstLaunchComputeBufMB(model *ModelProfile, uBatch int) int {
	est := uBatch * 4
	if model != nil && model.HiddenSize > 0 && model.NumLayers > 0 {
		per := float64(model.HiddenSize) * float64(model.NumLayers) * 42.0 / 1e6
		est = int(float64(uBatch) * per)
	}
	if est < computeFloorMB {
		est = computeFloorMB
	}
	if est > 16384 {
		est = 16384
	}
	return est
}

func firstLaunchComputeBufMBForGPU(model *ModelProfile, uBatch, gpuPos int, order []int) int {
	primary := len(order) == 0 || order[0] == gpuPos
	if primary {
		return firstLaunchComputeBufMB(model, uBatch)
	}
	est := firstLaunchComputeBufMB(model, uBatch) / 2
	if est < computeFloorMB {
		est = computeFloorMB
	}
	return est
}

// perGPUVRAMOverheadMB is the non-weight VRAM a single GPU needs at runtime:
// measured CUDA context/allocator overhead plus the compute buffer. Missing
// CUDA probe data is unknown and contributes 0; no static fallback margin is
// hidden here.
func perGPUVRAMOverheadMB(sysProbe *systemProbe, uBatch int) int {
	return measuredCUDAOverheadMB(sysProbe) + firstLaunchComputeBufMB(nil, uBatch)
}

// measuredCUDAOverheadMB is the measured CUDA context/allocator overhead per GPU.
// Missing probe data contributes 0 so callers cannot accidentally depend on a
// fabricated reserve.
func measuredCUDAOverheadMB(sysProbe *systemProbe) int {
	if sysProbe != nil && sysProbe.CUDAOverheadMB > 0 {
		return sysProbe.CUDAOverheadMB
	}
	return 0
}

// ramRuntimeOverheadMB is the non-weight system RAM the runtime needs: CUDA host
// pinned staging, compute-graph scratch, the mmap page table, and CPU activation
// buffers. Each term is derived from model dims / file size rather than a flat
// guessed reserve. (mmapPT is exact: a 4KB-page table is fileSize/4096*8B ≈
// fileSize/512. The host/graph terms are the last constants still pending a
// measurement probe — kept here as the single place to swap in measured values.)
func ramRuntimeOverheadMB(model *ModelProfile, uBatch, totalSizeMB int) int {
	const cudaHostMB = 1024
	const graphScratchMB = 2048
	mmapPTMB := totalSizeMB / 500

	actFFN := model.FeedForwardLength
	if model.NumExperts > 0 && model.ExpertUsedCount > 0 && model.ExpertFF > 0 {
		actFFN = model.ExpertUsedCount * model.ExpertFF
		if model.ExpertSharedFF > 0 {
			actFFN += model.ExpertSharedFF
		}
	}
	if model.KVLoraRank > 0 {
		actFFN += model.KVLoraRank + model.QLoraRank
	}
	cpuActMB := uBatch * (model.EmbeddingLength + actFFN) * 4 * 2 / 1048576
	if cpuActMB < 64 {
		cpuActMB = 64
	}
	return cudaHostMB + graphScratchMB + mmapPTMB + cpuActMB
}

// checkMemoryOrDie refuses to launch when model + KV + compute buffers exceed the pool.
// OOM guard: refuse to launch if model+KV+compute don't fit.
func checkMemoryOrDie(caps *detect.Capabilities, model *ModelProfile, s *Strategy, totalSizeMB, kvTotalMB int, opts Options) error {
	numGPUs := len(caps.GPUs)

	// Load measured CUDA overhead. Missing probe data is unknown and contributes 0;
	// preflight/startup OOM recording supplies measured data for later launches.
	cudaOverheadMB := measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs))

	// Load model probe for compute buffer
	computeBufMB := computeFloorMB // 1024 default
	pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, opts.BackendTag, caps.GPUs, s.Parallel)
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

	// -1 means "not computed for this strategy type" (single-GPU/CPU-only):
	// leave the backend's own default alone rather than silently emitting a
	// disable decision nothing actually derived. The multi-GPU branch below
	// always overwrites this with a real, headroom-based value (0 included).
	maxCheckpoints := -1

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

// defaultFlashAttention decides whether ggrun forces `--flash-attn on`. The
// decision must follow the resolved KV placement because the CUDA FA kernel
// requires GPU-resident KV.
func defaultFlashAttention(model *ModelProfile, opts Options, kvPlacement string) bool {
	// llama.cpp auto-disables flash attention whenever the KV cache isn't
	// GPU-resident — the FA CUDA kernel
	// needs its KV tensor on the same device doing the attention compute.
	// Claiming FlashAttention here when kvPlacement=="cpu" would emit a
	// self-contradicting `--flash-attn on --no-kv-offload` command; for
	// deepseek4 specifically that also silently re-opens the unbounded
	// compute-buffer growth this flag exists to prevent (see [[Task #10]]).
	return kvPlacement != "cpu"
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

	// Fixed overhead: model weights + model-aware headroom (audit #8)
	fixedOverheadMB := totalSizeMB + modelAwareHeadroom(model)

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

// resolveAutoKVPlacement decides gpu vs cpu for the KV cache when --kv-placement
// is "auto". A model that fits in VRAM keeps its KV on GPU (fast, VRAM to spare).
// A big MoE whose experts must offload to CPU puts KV on CPU instead: that frees
// VRAM for more expert layers (the decode-bandwidth bottleneck) and unlocks a much
// larger context. A dense model bigger than VRAM keeps KV on GPU (its only spot).
func resolveAutoKVPlacement(caps *detect.Capabilities, model *ModelProfile, totalSizeMB, vramOverheadMB int) string {
	freeVRAM := 0
	for _, g := range caps.GPUs {
		freeVRAM += g.VRAMFreeMB()
	}
	if totalSizeMB+vramOverheadMB <= freeVRAM {
		return "gpu"
	}
	if model.IsMoE {
		if strings.EqualFold(model.ModelArch, "deepseek4") {
			// KV on CPU makes llama.cpp auto-disable flash attention, and
			// deepseek4's non-FA graph materializes score tensors that grow
			// with real token position (~98 KiB/token measured 2026-07-09) —
			// no load-time reserve can cover that. Mainline therefore keeps
			// DeepSeek4 KV on GPU and trades expert VRAM for a bounded FA graph.
			return "gpu"
		}
		return "cpu"
	}
	return "gpu"
}

// computeAutoContextSizeKVPlacement computes the largest context whose KV cache
// fits in the memory implied by placement: VRAM for "gpu", system RAM for "cpu".
// For a MoE, "gpu" keeps only the non-expert weights on GPU and reserves the rest
// of VRAM for KV (experts offload to CPU), while "cpu" leaves VRAM for experts and
// puts the (large) KV in RAM. This is what makes --kv-placement drive the context
// ceiling instead of a fixed VRAM+RAM budget that can overflow a GPU-pinned KV.
func computeAutoContextSizeKVPlacement(caps *detect.Capabilities, model *ModelProfile, totalSizeMB int, preferredKVType, placement string, opts Options) (int, string) {
	freeVRAM := 0
	for _, g := range caps.GPUs {
		freeVRAM += g.VRAMFreeMB()
	}
	// Overhead is derived per-component, not a flat guess: VRAM side = measured
	// CUDA overhead + compute buffer per GPU; RAM side = host/graph/mmap/activation.
	sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
	vramOverheadMB := perGPUVRAMOverheadMB(sysProbe, 0) * len(caps.GPUs)

	var kvBudgetMB int
	if placement == "cpu" {
		// KV lives in RAM, sharing it with the weights that don't fit in VRAM.
		weightsInRAM := totalSizeMB - freeVRAM
		if weightsInRAM < 0 {
			weightsInRAM = 0
		}
		kvBudgetMB = caps.RAM.FreeMB - weightsInRAM - ramRuntimeOverheadMB(model, 0, totalSizeMB)
	} else {
		// KV lives in VRAM alongside the GPU-resident weights. Dense: the whole
		// model. MoE: only the non-expert weights (experts offload to CPU), so the
		// rest of VRAM is free for KV.
		gpuResident := totalSizeMB
		if model.IsMoE {
			if ne := bytesToMiBCeil(model.NonExpertBytes); ne > 0 {
				gpuResident = ne
				// Input embeddings stay in host memory, never in VRAM.
				if te := bytesToMiBCeil(model.TokenEmbdBytes); te > 0 && te < ne {
					gpuResident = ne - te
				}
			}
		}
		kvBudgetMB = freeVRAM - gpuResident - vramOverheadMB
	}
	if kvBudgetMB <= 0 {
		return 32768, preferredKVType
	}

	seen := map[string]bool{}
	for _, kvType := range []string{preferredKVType, "q8_0", "q4_0"} {
		if seen[kvType] {
			continue
		}
		seen[kvType] = true
		refCtx := 32768
		refKVMB := computeKVTotalMB(model, refCtx, kvType)
		if refKVMB <= 0 {
			continue
		}
		kvBytesPerToken := float64(refKVMB) * 1048576.0 / float64(refCtx)
		maxCtx := int(float64(kvBudgetMB) * 1048576.0 / kvBytesPerToken)
		if model.CTXTrain > 0 && model.CTXTrain < maxCtx {
			maxCtx = model.CTXTrain
		}
		best := 0
		for _, c := range []int{32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304} {
			if c <= maxCtx {
				best = c
			}
		}
		if best >= 32768 {
			return best, kvType
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

	// Fixed overhead: model weights + model-aware headroom (audit #8)
	fixedOverheadMB := totalSizeMB + modelAwareHeadroom(model)

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
		host = "127.0.0.1"
	}
	args := []string{
		"-m", modelPath,
		"--host", host,
		"--port", fmt.Sprintf("%d", port),
		"--ctx-size", fmt.Sprintf("%d", s.ContextSize),
	}
	if s.FlashAttention {
		args = append(args, "--flash-attn", "on")
	}
	args = append(args,
		"-b", fmt.Sprintf("%d", s.BatchSize),
		"-ub", fmt.Sprintf("%d", s.UBatchSize),
		"--cache-type-k", s.KVType,
		"--cache-type-v", s.KVType,
		"--jinja",
		"--threads", fmt.Sprintf("%d", s.Threads),
		"--threads-batch", fmt.Sprintf("%d", s.ThreadsBatch),
	)

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
		// Disable the backend's own memory auto-fit: ggrun already sets explicit
		// placement, so a second fit pass is redundant. Only emit the option when
		// the selected backend supports it.
		if s.BackendSupportsFit {
			args = append(args, "--fit", "off")
		}
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

	// CRAM is always a real, derived decision (never "not applicable") — 0
	// must reach the backend as an explicit "-cram 0" (disable), not silence
	// that lets llama-server fall back to its own 8192 MiB default. Same for
	// MaxCheckpoints when computeCRAM actually evaluated it (>= 0): nesting
	// this inside "CRAM > 0" used to mean a correctly-computed "0, disable
	// checkpoints — VRAM is too tight" was silently dropped, leaving
	// llama-server's default of 32 checkpoints active. That default's context
	// checkpoint save (tools/server/server-context.cpp create_checkpoint)
	// needs backend memory sched_reserve() never accounts for, and is exactly
	// what crashed DeepSeek-V4 mid-request on 2026-07-08 despite a placement
	// that had loaded clean and passed health check.
	args = append(args, "-cram", fmt.Sprintf("%d", s.CRAM))
	if s.MaxCheckpoints >= 0 {
		args = append(args, "--ctx-checkpoints", fmt.Sprintf("%d", s.MaxCheckpoints))
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

	// Server --timeout: some custom backends default to 600s, which kills
	// long Claude Code requests mid-stream. Match the client timeout so
	// queued subagents and large prompt-processing runs don't time out
	// server-side (Codex audit #5).
	args = append(args, "--timeout", "1800")

	return args
}

// loadSystemProbe tries to load measured CUDA overhead from cache.
// Keys the probe cache by a GPU-signature hash.
// SystemCUDAOverheadMB returns the legacy measured CUDA context overhead. New
// placement/preflight code must use SystemCUDAOverheadByGPU so headroom remains
// per-device and measured-only; this helper is kept for old call sites/tests.
func SystemCUDAOverheadMB(cacheDir string, gpus []detect.GPU) int {
	if p := loadSystemProbe(cacheDir, gpus); p != nil && p.CUDAOverheadMB > 0 {
		return p.CUDAOverheadMB
	}
	return 0
}

// SystemCUDAOverheadByGPU returns measured CUDA context/allocator overhead keyed
// by CUDA device index. Missing entries are intentionally absent: callers must
// not fill them with static margins.
func SystemCUDAOverheadByGPU(cacheDir string, gpus []detect.GPU) map[int]int {
	sp := loadSystemProbe(cacheDir, gpus)
	if sp == nil {
		return nil
	}
	out := map[int]int{}
	for k, v := range sp.CUDAOverheadByGPU {
		if v > 0 {
			out[k] = v
		}
	}
	if len(out) == 0 && sp.CUDAOverheadMB > 0 {
		for _, g := range gpus {
			out[g.Index] = sp.CUDAOverheadMB
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func loadSystemProbe(cacheDir string, gpus []detect.GPU) *systemProbe {
	explicitCacheDir := cacheDir != ""
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	// Compute GPU signature hash: sort(names+drivers), MD5, take first 12 chars
	gpuSig := gpuSignatureHash(gpus)
	path := filepath.Join(cacheDir, fmt.Sprintf("system_%s.cache", gpuSig))
	data, err := os.ReadFile(path)
	if err != nil && explicitCacheDir {
		// App-local installs used to read ~/.cache/ggrun before LLM_APP_HOME
		// became authoritative. Preserve those measured CUDA values and migrate
		// them lazily instead of treating the missing new-path file as zero.
		home, _ := os.UserHomeDir()
		legacyPath := filepath.Join(home, ".cache", "ggrun", fmt.Sprintf("system_%s.cache", gpuSig))
		if legacyPath != path {
			if legacyData, legacyErr := os.ReadFile(legacyPath); legacyErr == nil {
				data, err = legacyData, nil
				if mkErr := os.MkdirAll(filepath.Dir(path), 0755); mkErr == nil {
					_ = os.WriteFile(path, legacyData, 0644)
				}
			}
		}
	}
	if err != nil {
		return nil
	}
	sp := &systemProbe{CUDAOverheadByGPU: map[int]int{}}
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
		switch {
		case key == "SYS_CUDA_OVERHEAD_MB":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				sp.CUDAOverheadMB = v
			}
		case strings.HasPrefix(key, "SYS_CUDA_OVERHEAD_MB_CUDA"):
			idxRaw := strings.TrimPrefix(key, "SYS_CUDA_OVERHEAD_MB_CUDA")
			idx, idxErr := strconv.Atoi(idxRaw)
			v, valErr := strconv.Atoi(val)
			if idxErr == nil && valErr == nil && idx >= 0 && v >= 0 {
				sp.CUDAOverheadByGPU[idx] = v
			}
		}
	}
	if sp.CUDAOverheadMB == 0 && len(sp.CUDAOverheadByGPU) == 0 {
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
// kvCachePath is the per-model cache of measured KV bytes-per-token. Keyed by
// model basename + byte size so requantizations/different models never collide.
func kvCachePath(cacheDir string, model *ModelProfile) string {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	base := model.Basename
	if base == "" {
		base = filepath.Base(model.Path)
	}
	return filepath.Join(cacheDir, fmt.Sprintf("kv_%s_%d.cache", base, model.SizeBytes))
}

// loadMeasuredKVRates reads the per-model KV cache into a kvType→bytes/token map.
func loadMeasuredKVRates(cacheDir string, model *ModelProfile) map[string]float64 {
	path := kvCachePath(cacheDir, model)
	data, err := os.ReadFile(path)
	if err != nil && cacheDir != "" {
		// See loadSystemProbe: older installs wrote exact KV measurements to the
		// user cache even when the current app uses an app-local cache directory.
		// A compressed-attention model can be overestimated by many GiB without
		// this value, so migrate the measurement rather than reverting to formula.
		legacyPath := kvCachePath("", model)
		if legacyPath != path {
			if legacyData, legacyErr := os.ReadFile(legacyPath); legacyErr == nil {
				data, err = legacyData, nil
				if mkErr := os.MkdirAll(filepath.Dir(path), 0755); mkErr == nil {
					_ = os.WriteFile(path, legacyData, 0644)
				}
			}
		}
	}
	if err != nil {
		return nil
	}
	out := map[string]float64{}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// format: KV_BYTES_PER_TOK_<kvtype>=<float>
		const pfx = "KV_BYTES_PER_TOK_"
		if !strings.HasPrefix(line, pfx) {
			continue
		}
		kv := strings.SplitN(strings.TrimPrefix(line, pfx), "=", 2)
		if len(kv) != 2 {
			continue
		}
		if v, err := strconv.ParseFloat(strings.TrimSpace(kv[1]), 64); err == nil && v > 0 {
			out[strings.ToLower(strings.TrimSpace(kv[0]))] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// parseKVBufferTotalMB extracts the model's TOTAL KV cache allocation (MiB) at the
// launched context from a backend log. llama.cpp's wording varies across versions
// and backends, so match all known forms: an aggregate "KV self size = X MiB" /
// "KV cache size = X MiB" line (already the total — take it directly), otherwise
// SUM the per-device "... KV buffer size = X MiB" lines across CUDA devices + CPU.
// Returns 0 when the log carries no KV line (caller falls back to the formula or
// the VRAM-delta probe).
func parseKVBufferTotalMB(log string) float64 {
	var aggregate, bufSum float64
	for _, line := range strings.Split(log, "\n") {
		low := strings.ToLower(line)
		if !strings.Contains(low, "=") {
			continue
		}
		switch {
		case strings.Contains(low, "kv self size"), strings.Contains(low, "kv cache size"):
			if v := parseMiB(line); v > aggregate {
				aggregate = v // aggregate line: the total, printed once
			}
		case strings.Contains(low, "kv buffer size"):
			bufSum += parseMiB(line) // per-device: sum across GPUs + CPU
		}
	}
	if aggregate > 0 {
		return aggregate
	}
	return bufSum
}

// kvBytesPerTokenFromVRAMDelta derives KV bytes-per-token from two launches that
// differ ONLY in context size. Weights, compute buffers, and CUDA overhead are
// identical across the two, so the VRAM difference is pure KV cache — exact for
// every architecture and independent of whether the backend logs its KV size at
// all. Returns 0 if the samples are unusable.
func kvBytesPerTokenFromVRAMDelta(ctxA, vramA_MB, ctxB, vramB_MB int) float64 {
	dCtx := ctxB - ctxA
	dVRAM := vramB_MB - vramA_MB
	if dCtx < 0 {
		dCtx, dVRAM = -dCtx, -dVRAM
	}
	if dCtx == 0 || dVRAM <= 0 {
		return 0
	}
	return float64(dVRAM) * 1048576.0 / float64(dCtx)
}

// setCtxSizeArg returns a copy of args with --ctx-size set to ctx (adding it if
// absent). Used by the VRAM-delta probe to launch the same placement twice at
// different contexts.
func setCtxSizeArg(args []string, ctx int) []string {
	out := make([]string, len(args))
	copy(out, args)
	for i := 0; i < len(out)-1; i++ {
		if out[i] == "--ctx-size" || out[i] == "-c" {
			out[i+1] = strconv.Itoa(ctx)
			return out
		}
	}
	return append(out, "--ctx-size", strconv.Itoa(ctx))
}

// measureLoadedVRAM launches backendPath with args, waits for VRAM to plateau
// (the model + KV finished allocating), returns total VRAM used across gpus (MiB),
// then kills the process. Log-independent — it reads nvidia-smi, not stderr.
func measureLoadedVRAM(backendPath string, args []string, gpus []detect.GPU, timeout time.Duration) int {
	cmd := exec.Command(backendPath, args...)
	cmd.Stdout = nil
	cmd.Stderr = nil
	if err := cmd.Start(); err != nil {
		return 0
	}
	defer func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	}()

	deadline := time.Now().Add(timeout)
	prev, stable := -1, 0
	for time.Now().Before(deadline) {
		time.Sleep(1500 * time.Millisecond)
		total := 0
		for _, g := range gpus {
			total += QueryVRAMUsed(g.Index)
		}
		// Plateau = two consecutive readings within 64 MiB, above a floor.
		if total > 512 && prev > 512 && absInt(total-prev) <= 64 {
			stable++
			if stable >= 2 {
				return total
			}
		} else {
			stable = 0
		}
		prev = total
	}
	if prev > 512 {
		return prev
	}
	return 0
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// ProbeKVViaVRAMDelta measures a model's KV bytes-per-token by launching the same
// placement (baseArgs, minus the binary) twice — at a small and a larger context —
// and attributing the VRAM difference entirely to KV. It is the log-independent
// fallback for backends that don't print their KV buffer size. Requires roughly
// idle GPUs for a clean delta. Writes the per-model KV cache and returns true on
// success. Best-effort: returns false if either launch fails to allocate.
//
// NOTE: the two-launch flow needs live validation on real hardware before it is
// auto-invoked; the arithmetic (kvBytesPerTokenFromVRAMDelta) and cache round-trip
// are unit-tested.
func ProbeKVViaVRAMDelta(backendPath string, baseArgs []string, gpus []detect.GPU, cacheDir string, model *ModelProfile, kvType string) bool {
	if backendPath == "" || model == nil || len(gpus) == 0 {
		return false
	}
	if kvType == "" {
		kvType = "q8_0"
	}
	const ctxA, ctxB = 8192, 65536
	loadTimeout := 15 * time.Minute

	vramA := measureLoadedVRAM(backendPath, setCtxSizeArg(baseArgs, ctxA), gpus, loadTimeout)
	if vramA <= 0 {
		return false
	}
	vramB := measureLoadedVRAM(backendPath, setCtxSizeArg(baseArgs, ctxB), gpus, loadTimeout)
	if vramB <= 0 {
		return false
	}
	rate := kvBytesPerTokenFromVRAMDelta(ctxA, vramA, ctxB, vramB)
	if rate <= 0 {
		return false
	}
	writeMeasuredKVRate(cacheDir, model, strings.ToLower(kvType), rate,
		fmt.Sprintf("VRAM-delta probe: ctx %d=%dMB, ctx %d=%dMB", ctxA, vramA, ctxB, vramB))
	return true
}

// RunPostLaunchKVProbe reads the KV cache size llama.cpp actually allocated at
// ctxSize from the backend log and caches it as bytes-per-token for this model +
// kvType, so future launches size the context from measured truth instead of the
// per-arch GGUF formula. A successful launch deliberately refreshes a preflight
// estimate because the final buffer log is the most precise measurement.
func RunPostLaunchKVProbe(cacheDir string, model *ModelProfile, ctxSize int, kvType, serverLog string) {
	if model == nil || ctxSize <= 0 || serverLog == "" {
		return
	}
	if kvType == "" {
		kvType = "q8_0"
	}
	kvType = strings.ToLower(kvType)
	totalKVMB := parseKVBufferTotalMB(serverLog)
	if totalKVMB <= 0 {
		return
	}
	bytesPerTok := totalKVMB * 1048576.0 / float64(ctxSize)
	if bytesPerTok <= 0 {
		return
	}
	writeMeasuredKVRate(cacheDir, model, kvType, bytesPerTok,
		fmt.Sprintf("launch log: ctx=%d total_kv=%.0fMB", ctxSize, totalKVMB))
}

// RecordMeasuredContextMB records the backend-authoritative total context
// allocation reported by llama-fit-params. For placement purposes this is the
// exact quantity computeKVTotalMB must reserve: summing every device row keeps
// GPU-, CPU-, and mixed-KV placements on the same accounting path. The model is
// updated in memory as well as on disk so an immediate preflight re-plan sees
// the measurement without waiting for another process or launch.
func RecordMeasuredContextMB(cacheDir string, model *ModelProfile, ctxSize int, kvType string, totalContextMB int) {
	if model == nil || ctxSize <= 0 || totalContextMB <= 0 {
		return
	}
	if kvType == "" {
		kvType = "q8_0"
	}
	kvType = strings.ToLower(kvType)
	bytesPerTok := float64(totalContextMB) * 1048576.0 / float64(ctxSize)
	if bytesPerTok <= 0 {
		return
	}
	writeMeasuredKVRate(cacheDir, model, kvType, bytesPerTok,
		fmt.Sprintf("fit-params preflight: ctx=%d total_context=%dMB", ctxSize, totalContextMB))
}

// RecordMeasuredComputeBuffers records per-GPU compute-buffer MiB measured by
// the no-alloc fit-params preflight (cmd/ggrun/preflight.go) for the exact
// ctx/ubatch/KV/backend/GPU-set key, merging with (not clobbering) any
// runtime-growth or KV-per-layer data already cached for that same key. This
// lets the FIRST launch attempt for a shape use real numbers instead of the
// first-launch heuristic (ubatch*4, clamped 1024-4096) — which measurably
// under-estimates flash-attention's compute buffer at large context (e.g.
// ~17-20GB actual vs a 4096MB clamp for DeepSeek-V4 at ctx 1M, f16 KV).
func RecordMeasuredComputeBuffers(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, computeByGPU map[int]int) error {
	if model == nil || ctxSize <= 0 || ubatch <= 0 || len(computeByGPU) == 0 {
		return nil
	}
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	growth := map[int]int{}
	kvPerLayerMB := 0
	if pc != nil {
		for k, v := range pc.RuntimeGraphGrowthByGPU {
			growth[k] = v
		}
		kvPerLayerMB = pc.KVPerLayerMB
	}
	return writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel, computeByGPU, growth, kvPerLayerMB)
}

// RunPostLaunchModelProbe records measured compute-buffer data for the exact
// model/runtime placement that just loaded. Future placement can use this instead
// of the first-launch compute estimate.
// RuntimeGraphGrowthByGPU returns measured post-health graph growth keyed by CUDA
// device. These values are populated only from observed runtime allocation growth
// or exact cudaMalloc failures for the same runtime signature; missing means
// unknown, not zero-margin proof.
func RuntimeGraphGrowthByGPU(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) map[int]int {
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	if pc == nil || len(pc.RuntimeGraphGrowthByGPU) == 0 {
		return nil
	}
	out := map[int]int{}
	for k, v := range pc.RuntimeGraphGrowthByGPU {
		if v > 0 {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// HasRuntimeGraphGrowthProbe reports whether all active GPUs have measured
// runtime-growth data for this exact runtime signature. This is a verification
// marker for future agents: no static fallback is hidden behind this predicate.
func HasRuntimeGraphGrowthProbe(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) bool {
	growth := RuntimeGraphGrowthByGPU(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	if len(gpus) == 0 || len(growth) == 0 {
		return false
	}
	for _, g := range gpus {
		if _, ok := growth[g.Index]; !ok {
			return false
		}
	}
	return true
}

// RecordRuntimeGraphGrowth stores exact per-device runtime graph growth for the
// current runtime signature. The values must come from measurement: VRAM delta
// after a canary or an exact cudaMalloc allocation request parsed from backend
// logs. It intentionally never adds static margins.
func RecordRuntimeGraphGrowth(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, growthByGPU map[int]int) error {
	if model == nil || ctxSize <= 0 || ubatch <= 0 || len(growthByGPU) == 0 {
		return nil
	}
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	computeByGPU := map[int]int{}
	kvPerLayerMB := 0
	mergedGrowth := map[int]int{}
	if pc != nil {
		for k, v := range pc.ComputeBufByGPU {
			computeByGPU[k] = v
		}
		for k, v := range pc.RuntimeGraphGrowthByGPU {
			mergedGrowth[k] = v
		}
		kvPerLayerMB = pc.KVPerLayerMB
	}
	for idx, v := range growthByGPU {
		if v > mergedGrowth[idx] {
			mergedGrowth[idx] = v
		}
	}
	return writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel, computeByGPU, mergedGrowth, kvPerLayerMB)
}

// RecordRuntimeGraphGrowthFromOOM records a runtime graph allocation observed in
// a cudaMalloc OOM line. allocMB is already the backend-requested size, so using
// it as growth is measured accounting, not a reserve margin.
func RecordRuntimeGraphGrowthFromOOM(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel, device, allocMB int) error {
	if device < 0 || allocMB <= 0 {
		return nil
	}
	return RecordRuntimeGraphGrowth(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel, map[int]int{device: allocMB})
}

// RunPostLaunchModelProbeVRAMDelta writes per-GPU compute-buffer probe cache from
// nvidia-smi VRAM delta (current - baseline) instead of log parsing. Log-independent;
// works even when the server binary suppresses LLAMA_LOG_INFO. It estimates model
// weight per GPU from the placement OT assignments, subtracts that from the VRAM
// delta, and stores the remainder as the measured compute-buffer value for that GPU.
func computeBuffersFromVRAMDelta(
	model *ModelProfile, strategy *Strategy, gpus []detect.GPU,
	baselineVRAMByGPU, usedVRAMByGPU, overheadByGPU map[int]int,
) map[int]int {
	if model == nil || strategy == nil || model.NumLayers <= 0 || len(gpus) == 0 {
		return nil
	}
	// Partial gate+up pins do not expose their byte size in the OT string. A
	// guessed model subtraction would turn this fallback into poisoned compute
	// data, so rely on fit/log probes for those placements.
	if strings.Contains(strategy.OTString, `ffn_(gate_up|up_gate|gate|up)_(ch|)exps`) {
		return nil
	}

	assignments := parseOTAssignments(strategy.OTString)
	expertLayersByGPU := map[int]int{}
	for _, a := range assignments {
		expertLayersByGPU[a.CUDAIndex] += a.Count
	}
	moeLayers := model.NumLayers - model.LeadingDense
	if moeLayers <= 0 {
		moeLayers = model.NumLayers
	}
	expertPerLayerMB := ceilDivInt(bytesToMiBCeil(model.ExpertBytes), moeLayers)

	nonExpertTotalMB := bytesToMiBCeil(model.NonExpertBytes)
	if tokenMB := bytesToMiBCeil(model.TokenEmbdBytes); tokenMB > 0 && tokenMB < nonExpertTotalMB {
		nonExpertTotalMB -= tokenMB
	}
	outputMB := bytesToMiBCeil(model.OutputBytes)
	if outputMB > 0 && outputMB < nonExpertTotalMB {
		nonExpertTotalMB -= outputMB
	} else {
		outputMB = 0
	}
	shexpTotalMB := bytesToMiBCeil(model.ShexpBytes)
	if shexpTotalMB < 0 || shexpTotalMB >= bytesToMiBCeil(model.ExpertBytes) {
		shexpTotalMB = 0
	}
	owned, outputDev := layerOwnership(strategy.TensorSplit, model.NumLayers)
	layerDevices, _ := layerDeviceAssignments(strategy.TensorSplit, model.NumLayers)
	moeOwned := make([]int, len(strategy.TensorSplit))
	moeStart := model.LeadingDense
	if moeStart < 0 || moeStart >= model.NumLayers {
		moeStart = 0
	}
	for layer := moeStart; layer < len(layerDevices); layer++ {
		if dev := layerDevices[layer]; dev >= 0 && dev < len(moeOwned) {
			moeOwned[dev]++
		}
	}

	kvTotalMB := 0
	if strings.EqualFold(strategy.KVPlacement, "gpu") && strategy.ContextSize > 0 {
		kvTotalMB = computeKVTotalMB(model, strategy.ContextSize, strategy.KVType)
	}

	computeByGPU := map[int]int{}
	for gi, g := range gpus {
		overheadMB, measured := overheadByGPU[g.Index]
		if !measured {
			continue
		}
		usedMB := usedVRAMByGPU[g.Index]
		baselineMB := baselineVRAMByGPU[g.Index]
		if usedMB <= baselineMB {
			continue
		}

		modelMB := expertLayersByGPU[g.Index] * expertPerLayerMB
		if gi < len(owned) && owned[gi] > 0 {
			modelMB += ownedShareMB(nonExpertTotalMB, owned, model.NumLayers, gi)
		}
		if gi < len(moeOwned) && moeOwned[gi] > 0 {
			modelMB += int(math.Ceil(float64(shexpTotalMB) * float64(moeOwned[gi]) / float64(moeLayers)))
		}
		if gi == outputDev {
			modelMB += outputMB
		}
		kvShareMB := ownedShareMB(kvTotalMB, owned, model.NumLayers, gi)
		bufMB := usedMB - baselineMB - overheadMB - modelMB - kvShareMB
		if bufMB > 0 {
			computeByGPU[g.Index] = bufMB
		}
	}
	if len(computeByGPU) == 0 {
		return nil
	}
	return computeByGPU
}

func RunPostLaunchModelProbeVRAMDelta(
	cacheDir string, model *ModelProfile, strategy *Strategy,
	backendTag string, gpus []detect.GPU, baselineVRAMByGPU map[int]int,
) bool {
	if model == nil || strategy == nil || len(gpus) == 0 || len(baselineVRAMByGPU) == 0 {
		return false
	}
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	// This is strictly a fallback for backends that expose neither fit-params nor
	// compute-buffer log rows. Never replace an authoritative probe already
	// recorded by preflight for this exact runtime signature.
	existing := loadProbeCache(cacheDir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, backendTag, gpus, strategy.Parallel)
	if existing != nil && len(existing.ComputeBufByGPU) > 0 {
		return false
	}

	usedVRAMByGPU := map[int]int{}
	for _, g := range gpus {
		usedMB := QueryVRAMUsed(g.Index)
		if usedMB > 0 {
			usedVRAMByGPU[g.Index] = usedMB
		}
	}
	computeByGPU := computeBuffersFromVRAMDelta(model, strategy, gpus, baselineVRAMByGPU, usedVRAMByGPU, SystemCUDAOverheadByGPU(cacheDir, gpus))
	if len(computeByGPU) == 0 {
		return false
	}

	// Preserve any runtime-growth history from a previous OOM so the probe
	// cache does not silently erase it (audit cross-check #3).
	var mergedGrowth map[int]int
	if existing != nil {
		mergedGrowth = existing.RuntimeGraphGrowthByGPU
	}

	if err := writeProbeCacheForModel(cacheDir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, backendTag, gpus, strategy.Parallel, computeByGPU, mergedGrowth, 0); err == nil {
		indices := make([]int, 0, len(computeByGPU))
		for idx := range computeByGPU {
			indices = append(indices, idx)
		}
		sort.Ints(indices)
		parts := make([]string, 0, len(indices))
		for _, idx := range indices {
			parts = append(parts, fmt.Sprintf("CUDA%d=%dMB", idx, computeByGPU[idx]))
		}
		fmt.Fprintf(os.Stderr, "  VRAM probe: compute_buf %s\n", strings.Join(parts, ", "))
		return true
	}
	return false
}

func RunPostLaunchModelProbe(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, serverLog string) bool {
	if model == nil || ctxSize <= 0 || ubatch <= 0 || serverLog == "" {
		return false
	}
	computeByGPU := ParseComputeBuffersByGPU(serverLog)
	_, kvPerLayerMB := ParseLogForProbe(serverLog)
	if len(computeByGPU) == 0 && kvPerLayerMB <= 0 {
		return false
	}
	if err := writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel, computeByGPU, nil, kvPerLayerMB); err == nil {
		if len(computeByGPU) == 0 {
			return true
		}
		parts := make([]string, 0, len(computeByGPU))
		indices := make([]int, 0, len(computeByGPU))
		for idx := range computeByGPU {
			indices = append(indices, idx)
		}
		sort.Ints(indices)
		for _, idx := range indices {
			parts = append(parts, fmt.Sprintf("CUDA%d=%dMB", idx, computeByGPU[idx]))
		}
		fmt.Fprintf(os.Stderr, "  Model probe written: compute_buf %s\n", strings.Join(parts, ", "))
		return true
	}
	return false
}

// writeMeasuredKVRate records a measured KV bytes-per-token for model+kvType,
// merging with any rates already cached for other kvTypes.
func writeMeasuredKVRate(cacheDir string, model *ModelProfile, kvType string, bytesPerTok float64, note string) {
	kvType = strings.ToLower(kvType)
	rates := loadMeasuredKVRates(cacheDir, model)
	if rates == nil {
		rates = map[string]float64{}
	}
	for k, v := range model.MeasuredKVBytesPerTok {
		if v > 0 && rates[k] <= 0 {
			rates[k] = v
		}
	}
	rates[kvType] = bytesPerTok
	model.MeasuredKVBytesPerTok = rates

	path := kvCachePath(cacheDir, model)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return
	}
	var b strings.Builder
	fmt.Fprintf(&b, "# Measured KV cache for %s (%s)\n", model.Basename, note)
	for k, v := range rates {
		fmt.Fprintf(&b, "KV_BYTES_PER_TOK_%s=%.4f\n", k, v)
	}
	if err := os.WriteFile(path, []byte(b.String()), 0644); err == nil {
		fmt.Fprintf(os.Stderr, "  KV probe: %s = %.0f bytes/token (%s)\n", kvType, bytesPerTok, note)
	}
}

func RunPostLaunchProbe(cacheDir string, gpus []detect.GPU, serverLog string) {
	if len(gpus) == 0 || serverLog == "" {
		return
	}
	// Only write if per-device measured values already exist. Legacy global-only
	// files are upgraded the next time a launch provides per-device accounting.
	sp := loadSystemProbe(cacheDir, gpus)
	if sp != nil && len(sp.CUDAOverheadByGPU) > 0 {
		return
	}

	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}

	overheadByGPU := map[int]int{}
	for _, gpu := range gpus {
		usedMB := QueryVRAMUsed(gpu.Index)
		if usedMB <= 0 {
			continue
		}
		modelBufMB, kvBufMB, computeBufMB := parseBuffersFromLog(serverLog, gpu.Index)
		accounted := modelBufMB + kvBufMB + computeBufMB
		if accounted <= 0 {
			continue
		}
		cudaOverhead := usedMB - accounted
		if cudaOverhead <= 0 || cudaOverhead >= usedMB {
			continue
		}
		overheadByGPU[gpu.Index] = cudaOverhead
	}
	if len(overheadByGPU) == 0 {
		return
	}

	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return
	}
	gpuSig := gpuSignatureHash(gpus)
	path := filepath.Join(cacheDir, fmt.Sprintf("system_%s.cache", gpuSig))
	var b strings.Builder
	fmt.Fprintf(&b, "# System probe (post-launch per-device measurement)\n")
	fmt.Fprintf(&b, "# Generated: %s\n", time.Now().Format(time.RFC3339))
	indices := make([]int, 0, len(overheadByGPU))
	for idx := range overheadByGPU {
		indices = append(indices, idx)
	}
	sort.Ints(indices)
	legacyMax := 0
	parts := make([]string, 0, len(indices))
	for _, idx := range indices {
		v := overheadByGPU[idx]
		if v > legacyMax {
			legacyMax = v
		}
		fmt.Fprintf(&b, "SYS_CUDA_OVERHEAD_MB_CUDA%d=%d\n", idx, v)
		parts = append(parts, fmt.Sprintf("CUDA%d=%dMB", idx, v))
	}
	// Compatibility for older readers. This is still measured data, not a margin.
	fmt.Fprintf(&b, "SYS_CUDA_OVERHEAD_MB=%d\n", legacyMax)
	if err := os.WriteFile(path, []byte(b.String()), 0644); err == nil {
		fmt.Fprintf(os.Stderr, "  System probe written: cuda_overhead %s\n", strings.Join(parts, ", "))
	}
}

// QueryVRAMUsed returns current nvidia-smi memory.used for a given GPU index.
func QueryVRAMUsed(gpuIndex int) int {
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
	// Take the number between "=" and the FIRST "MiB" after it, so lines with
	// trailing detail (e.g. "KV self size = X MiB, K (f16): Y MiB, ...") parse the
	// aggregate X and not the per-component values.
	rest := line[idx+1:]
	mib := strings.Index(rest, "MiB")
	if mib < 0 {
		return 0
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(rest[:mib]), 64)
	if err != nil {
		return 0
	}
	return v
}

// loadProbeCache tries to load per-model/runtime probe data.
// Keys the probe cache file by an MD5 of the model + placement runtime signature.
func loadProbeCache(cacheDir string, model *ModelProfile, ctxSize int, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) *probeCache {
	path := probeCachePath(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, probeParallelKey(parallel))
	if path == "" {
		return nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	content := string(data)
	pc := &probeCache{ComputeBufByGPU: map[int]int{}, RuntimeGraphGrowthByGPU: map[int]int{}}
	schemaVersion := 1
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
		switch {
		case k == "PROBE_CACHE_SCHEMA":
			if v, err := strconv.Atoi(val); err == nil && v > 0 {
				schemaVersion = v
			}
		case k == "PROBED_COMPUTE_BUF_MB":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				pc.ComputeBufMB = v
			}
		case strings.HasPrefix(k, "PROBED_COMPUTE_BUF_MB_CUDA"):
			idxRaw := strings.TrimPrefix(k, "PROBED_COMPUTE_BUF_MB_CUDA")
			idx, idxErr := strconv.Atoi(idxRaw)
			v, valErr := strconv.Atoi(val)
			if idxErr == nil && valErr == nil && idx >= 0 && v >= 0 {
				pc.ComputeBufByGPU[idx] = v
			}
		case strings.HasPrefix(k, "PROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA"):
			idxRaw := strings.TrimPrefix(k, "PROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA")
			idx, idxErr := strconv.Atoi(idxRaw)
			v, valErr := strconv.Atoi(val)
			if idxErr == nil && valErr == nil && idx >= 0 && v >= 0 {
				pc.RuntimeGraphGrowthByGPU[idx] = v
			}
		case k == "PROBED_KV_PER_LAYER_MB":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				pc.KVPerLayerMB = v
			}
		}
	}
	if schemaVersion < 2 {
		// Before schema 2, a startup graph-reserve OOM was written once as the
		// measured compute buffer and again as "runtime growth" from the same
		// cudaMalloc line (rounding made the second value exactly compute or
		// compute+1 MiB). Summing both excluded otherwise viable GPUs. Drop only
		// that unambiguous legacy duplicate; genuine post-health growth is kept.
		for idx, growth := range pc.RuntimeGraphGrowthByGPU {
			compute := pc.ComputeBufByGPU[idx]
			if compute > 0 && growth >= compute && growth <= compute+1 {
				delete(pc.RuntimeGraphGrowthByGPU, idx)
			}
		}
	}
	if pc.ComputeBufMB > 0 || len(pc.ComputeBufByGPU) > 0 || len(pc.RuntimeGraphGrowthByGPU) > 0 || pc.KVPerLayerMB > 0 {
		return pc
	}
	return nil
}

func probeCachePath(cacheDir string, model *ModelProfile, ctxSize int, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) string {
	if model == nil {
		return ""
	}
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun", "probes")
	}
	if kvPlacement == "" {
		kvPlacement = "auto"
	}
	if kvQuality == "" {
		kvQuality = "mid"
	}
	if backendTag == "" {
		backendTag = "llama"
	}
	// MD5 hash key over model/runtime/placement. Compute buffers differ with KV
	// placement, backend, GPU set, and parallel slots. Tensor split is derived
	// after probes are loaded, so it cannot honestly participate in this key.
	modelName := filepath.Base(model.Path)
	// Keep the historical trailing separator so serial (parallel key 0) cache
	// paths remain compatible with measurements from before slot isolation.
	key := fmt.Sprintf("%s:%d:%d:%d:%d:%d:%d:%s:%s:%s:%s:%d:",
		modelName, model.NumLayers, model.NumExperts,
		model.EmbeddingLength, model.FeedForwardLength,
		ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpuSignatureHash(gpus), parallel)
	hash := md5Hash12(key)
	return filepath.Join(cacheDir, hash+".probe")
}

// PlacementCachePathFor returns the keyed file that persists the validated MoE
// placement for this exact model + runtime + hardware. Keyed identically to the
// probe cache (kv placement, ctx, ubatch, backend, GPU set all part of the key)
// so kv=gpu vs kv=cpu — or two context sizes — never share a cache entry. Both
// the fit (load) and OOM-recovery / success (save) use this same path, so a
// placement that loads cleanly is remembered instead of re-predicted.
func PlacementCachePathFor(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, tensorSplit string) string {
	if model == nil {
		return ""
	}
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	if kvPlacement == "" {
		kvPlacement = "auto"
	}
	if kvQuality == "" {
		kvQuality = "mid"
	}
	if backendTag == "" {
		backendTag = "llama"
	}
	key := fmt.Sprintf("place:%s:%d:%d:%d:%d:%d:%d:%s:%s:%s:%s:%d:%s",
		filepath.Base(model.Path), model.NumLayers, model.NumExperts,
		model.EmbeddingLength, model.FeedForwardLength,
		ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpuSignatureHash(gpus), parallel, tensorSplit)
	return filepath.Join(cacheDir, md5Hash12(key)+".place")
}

// md5Hash12 computes first 12 chars of MD5 hash of input string.
func md5Hash12(input string) string {
	h := md5.New()
	h.Write([]byte(input))
	return fmt.Sprintf("%x", h.Sum(nil))[:12]
}

// internal types for probe loading
type systemProbe struct {
	CUDAOverheadMB    int
	CUDAOverheadByGPU map[int]int
}

type probeCache struct {
	ComputeBufMB            int
	ComputeBufByGPU         map[int]int
	RuntimeGraphGrowthByGPU map[int]int
	KVPerLayerMB            int
}

const probeCacheSchema = 2

// probeParallelKey preserves the legacy serial key (0) for normal --parallel 1
// launches while isolating multi-slot graph measurements such as Claude Code's
// --parallel 4. Graph allocation changes with slot count; sharing those probes
// was another form of compute-buffer double accounting across launch modes.
func probeParallelKey(parallel int) int {
	if parallel <= 1 {
		return 0
	}
	return parallel
}

// splitCompactKey returns a compact string for a tensor-split vector, suitable
// for cache-key inclusion so placements with different split ratios don't share
// cached compute-buffer measurements (audit cross-check #6).
func splitCompactKey(split []float64) string {
	if len(split) == 0 {
		return "0"
	}
	parts := make([]string, len(split))
	for i, v := range split {
		parts[i] = fmt.Sprintf("%.2f", v)
	}
	return strings.Join(parts, ",")
}

// WriteProbeCache writes a legacy probe cache. Prefer WriteProbeCacheForModel,
// which writes to the same runtime-signature key that loadProbeCache reads.
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

// WriteProbeCacheForModel writes measured compute-buffer and KV sizes to the
// per-model/runtime cache consumed by placement. computeByGPU is keyed by CUDA
// device index as emitted in the backend log.
func WriteProbeCacheForModel(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, computeByGPU map[int]int, kvPerLayerMB int) error {
	return writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, 0, computeByGPU, nil, kvPerLayerMB)
}

func writeProbeCacheForModel(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, computeByGPU map[int]int, runtimeGrowthByGPU map[int]int, kvPerLayerMB int) error {
	parallelKey := probeParallelKey(parallel)
	path := probeCachePath(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallelKey)
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	maxCompute := 0
	for _, v := range computeByGPU {
		if v > maxCompute {
			maxCompute = v
		}
	}
	var b strings.Builder
	fmt.Fprintf(&b, "# Probe cache for %s\n", filepath.Base(model.Path))
	fmt.Fprintf(&b, "# Generated: %s\n", time.Now().Format(time.RFC3339))
	fmt.Fprintf(&b, "# ctx=%d ubatch=%d kv_quality=%s kv_placement=%s backend=%s gpu_sig=%s parallel=%d\n", ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpuSignatureHash(gpus), parallelKey)
	fmt.Fprintf(&b, "PROBE_CACHE_SCHEMA=%d\n", probeCacheSchema)
	if maxCompute > 0 {
		fmt.Fprintf(&b, "PROBED_COMPUTE_BUF_MB=%d\n", maxCompute)
	}
	indices := make([]int, 0, len(computeByGPU))
	for idx := range computeByGPU {
		indices = append(indices, idx)
	}
	sort.Ints(indices)
	for _, idx := range indices {
		if computeByGPU[idx] > 0 {
			fmt.Fprintf(&b, "PROBED_COMPUTE_BUF_MB_CUDA%d=%d\n", idx, computeByGPU[idx])
		}
	}
	growthIndices := make([]int, 0, len(runtimeGrowthByGPU))
	for idx := range runtimeGrowthByGPU {
		growthIndices = append(growthIndices, idx)
	}
	sort.Ints(growthIndices)
	for _, idx := range growthIndices {
		if runtimeGrowthByGPU[idx] > 0 {
			fmt.Fprintf(&b, "PROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA%d=%d\n", idx, runtimeGrowthByGPU[idx])
		}
	}
	if kvPerLayerMB > 0 {
		fmt.Fprintf(&b, "PROBED_KV_PER_LAYER_MB=%d\n", kvPerLayerMB)
	}
	return os.WriteFile(path, []byte(b.String()), 0644)
}

var memoryBreakdownComputePattern = regexp.MustCompile(`CUDA([0-9]+).*?\(\s*-?[0-9]+\s*=\s*-?[0-9]+\s*\+\s*-?[0-9]+\s*\+\s*([0-9]+)\s*\)`)

// ParseLogForProbe extracts compute_buf and kv_per_layer from server log output.
// Looks for lines like: "CUDA0 compute buffer size = 1410.12 MiB" and fit
// breakdown rows like: "CUDA0 ... ( self = model + context + compute )".
func ParseLogForProbe(logData string) (computeBufMB, kvPerLayerMB int) {
	computeByGPU := ParseComputeBuffersByGPU(logData)
	for _, v := range computeByGPU {
		if v > computeBufMB {
			computeBufMB = v
		}
	}

	// Sum KV buffer sizes across CUDA devices
	var totalKVBuf float64
	var kvCount int
	lines := strings.Split(logData, "\n")
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

	if totalKVBuf > 0 && kvCount > 0 {
		// Approximate per-device KV share by kvCount (devices holding KV)
		kvPerLayerMB = int(totalKVBuf/float64(kvCount) + 0.5)
	}

	return
}

// ParseComputeBuffersByGPU extracts measured/planned compute-buffer MiB by CUDA
// index. It accepts both final llama.cpp buffer lines and the memory breakdown
// table printed during fit.
func ParseComputeBuffersByGPU(logData string) map[int]int {
	out := map[int]int{}
	for _, line := range strings.Split(logData, "\n") {
		if idx := strings.Index(line, "compute buffer size ="); idx >= 0 {
			cudaIdx := cudaIndexFromLine(line)
			if cudaIdx < 0 {
				continue
			}
			rest := line[idx+len("compute buffer size ="):]
			rest = strings.TrimSpace(rest)
			rest = strings.TrimSuffix(rest, " MiB")
			if v, err := strconv.ParseFloat(rest, 64); err == nil && v > 0 {
				mb := int(v + 0.5)
				if mb > out[cudaIdx] {
					out[cudaIdx] = mb
				}
			}
			continue
		}
		if m := memoryBreakdownComputePattern.FindStringSubmatch(line); m != nil {
			cudaIdx, idxErr := strconv.Atoi(m[1])
			mb, mbErr := strconv.Atoi(m[2])
			if idxErr == nil && mbErr == nil && mb > out[cudaIdx] {
				out[cudaIdx] = mb
			}
		}
	}
	return out
}

func cudaIndexFromLine(line string) int {
	idx := strings.Index(line, "CUDA")
	if idx < 0 {
		return -1
	}
	start := idx + len("CUDA")
	end := start
	for end < len(line) && line[end] >= '0' && line[end] <= '9' {
		end++
	}
	if end == start {
		return -1
	}
	n, err := strconv.Atoi(line[start:end])
	if err != nil {
		return -1
	}
	return n
}
