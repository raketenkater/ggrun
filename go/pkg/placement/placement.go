package placement

import (
	"fmt"
	"math"

	"github.com/raketenkater/llm-server/pkg/detect"
)

// Strategy represents the computed placement for a model on this hardware.
type Strategy struct {
	ContextSize    int      `json:"context_size"`
	GPULayers      int      `json:"gpu_layers"`
	TensorSplit    []float64 `json:"tensor_split,omitempty"`
	SplitMode      string   `json:"split_mode,omitempty"`
	MainGPU        int      `json:"main_gpu,omitempty"`
	KVPlacement    string   `json:"kv_placement"`
	KVQuality      string   `json:"kv_quality"`
	NCPUMoE        int      `json:"n_cpu_moe,omitempty"`
	MMap           bool     `json:"mmap"`
	MLock          bool     `json:"mlock"`
	FlashAttention bool     `json:"flash_attention"`
	Threads        int      `json:"threads"`
	BatchSize      int      `json:"batch_size"`
	UBatchSize     int      `json:"ubatch_size"`
}

// ModelProfile describes the GGUF model.
type ModelProfile struct {
	Path        string `json:"path"`
	SizeBytes   int64  `json:"size_bytes"`
	NumLayers   int    `json:"num_layers"`
	NumParams   int64  `json:"num_params"`
	IsMoE       bool   `json:"is_moe"`
	NumExperts  int    `json:"num_experts,omitempty"`
	ContextSize int    `json:"context_size"`
	HiddenSize  int    `json:"hidden_size"`
	HeadCount   int    `json:"head_count"`
	VocabSize   int    `json:"vocab_size"`
	QuantType   string `json:"quant_type"`
}

// Compute builds a Strategy from hardware capabilities and model profile.
func Compute(caps *detect.Capabilities, model *ModelProfile, opts Options) (*Strategy, error) {
	s := &Strategy{
		ContextSize: opts.ContextSize,
		KVPlacement: opts.KVPlacement,
		KVQuality:   opts.KVQuality,
		MMap:        true,
		MLock:       false,
		Threads:     caps.CPU.Cores,
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

	totalVRAM := caps.TotalVRAM()
	modelSizeMB := int(model.SizeBytes / 1024 / 1024)

	// Dense vs MoE logic
	if model.IsMoE {
		return computeMoE(s, caps, model, totalVRAM, modelSizeMB, opts)
	}
	return computeDense(s, caps, model, totalVRAM, modelSizeMB, opts)
}

// Options allows user overrides.
type Options struct {
	ContextSize int
	KVPlacement string
	KVQuality   string
	GPUs        []int // restrict to specific GPUs
	CPUMode     bool
	RamBudgetMB int
}

func computeDense(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalVRAM, modelSizeMB int, opts Options) (*Strategy, error) {
	numGPUs := len(caps.GPUs)
	if opts.CPUMode || numGPUs == 0 {
		s.GPULayers = 0
		s.MMap = true
		return s, nil
	}

	if numGPUs == 1 {
		// Single GPU: try to fit everything
		gpuVRAM := caps.GPUs[0].VRAMTotalMB
		overhead := estimateOverhead(model, s.ContextSize)
		available := gpuVRAM - overhead

		if modelSizeMB <= available {
			s.GPULayers = model.NumLayers
			s.MainGPU = 0
			if opts.KVPlacement == "auto" {
				s.KVPlacement = "gpu"
			}
		} else {
			// Partial offload
			layerSize := modelSizeMB / model.NumLayers
			maxLayers := available / layerSize
			if maxLayers < 0 {
				maxLayers = 0
			}
			if maxLayers > model.NumLayers {
				maxLayers = model.NumLayers
			}
			s.GPULayers = maxLayers
			s.MainGPU = 0
			if opts.KVPlacement == "auto" {
				s.KVPlacement = "gpu"
			}
		}
	} else {
		// Multi-GPU: tensor split
		s.GPULayers = model.NumLayers
		s.SplitMode = "layer"
		var split []float64
		for _, g := range caps.GPUs {
			split = append(split, float64(g.VRAMTotalMB))
		}
		s.TensorSplit = normalizeSplit(split)
		if opts.KVPlacement == "auto" {
			s.KVPlacement = "gpu"
		}
	}

	s.FlashAttention = true
	s.BatchSize = 2048
	s.UBatchSize = 512
	return s, nil
}

func computeMoE(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalVRAM, modelSizeMB int, opts Options) (*Strategy, error) {
	numGPUs := len(caps.GPUs)
	if opts.CPUMode || numGPUs == 0 {
		s.GPULayers = 0
		s.NCPUMoE = model.NumExperts
		s.MMap = true
		return s, nil
	}

	gpuVRAM := caps.GPUs[0].VRAMTotalMB
	if numGPUs > 1 {
		// Use total for estimation on primary
		gpuVRAM = totalVRAM
	}

	overhead := estimateOverhead(model, s.ContextSize)
	available := gpuVRAM - overhead

	if modelSizeMB <= available && numGPUs == 1 {
		// Fits on single GPU
		s.GPULayers = model.NumLayers
		s.MainGPU = 0
		s.NCPUMoE = 0
		if opts.KVPlacement == "auto" {
			s.KVPlacement = "gpu"
		}
	} else {
		// MoE offload: GPU gets attention + some FFN, CPU gets experts
		attnSize := modelSizeMB * 20 / 100 // rough: 20% is attention
		if attnSize <= available {
			s.GPULayers = model.NumLayers
		} else {
			layerSize := modelSizeMB / model.NumLayers
			maxLayers := available / layerSize
			if maxLayers < 0 {
				maxLayers = 0
			}
			if maxLayers > model.NumLayers {
				maxLayers = model.NumLayers
			}
			s.GPULayers = maxLayers
		}
		// Route remaining experts to CPU
		s.NCPUMoE = model.NumExperts
		if opts.KVPlacement == "auto" {
			s.KVPlacement = "cpu"
		}
	}

	if numGPUs > 1 {
		s.SplitMode = "layer"
		var split []float64
		for _, g := range caps.GPUs {
			split = append(split, float64(g.VRAMTotalMB))
		}
		s.TensorSplit = normalizeSplit(split)
	}

	s.FlashAttention = true
	s.BatchSize = 2048
	s.UBatchSize = 512
	return s, nil
}

func estimateOverhead(model *ModelProfile, ctxSize int) int {
	// Rough estimate: KV cache size
	// bytes_per_token ≈ 2 * num_layers * hidden_size * head_count / head_count * 2 bytes (fp16)
	// Simplified: use a heuristic based on model size and context
	baseOverhead := 512 // MB for CUDA context, etc.
	kvFactor := 1
	if model.IsMoE {
		kvFactor = 2 // MoE tends to have larger KV overhead
	}
	kvMB := (ctxSize * model.NumLayers * model.HiddenSize * 2 * kvFactor) / (1024 * 1024)
	return baseOverhead + kvMB
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
	// Fit to RAM/VRAM
	ramMB := caps.RAM.TotalMB
	vramMB := caps.TotalVRAM()
	maxMem := ramMB
	if vramMB > maxMem {
		maxMem = vramMB
	}
	// Heuristic: ~1MB per 1K context tokens per layer for KV cache
	kvPer1K := (model.NumLayers * model.HiddenSize * 2) / (1024 * 1024)
	if kvPer1K <= 0 {
		kvPer1K = 1
	}
	maxCtx := (maxMem / 4) / kvPer1K * 1024 // use 1/4 of max memory
	if maxCtx > 32768 {
		return 32768
	}
	if maxCtx < 4096 {
		return 4096
	}
	return maxCtx
}

// Args converts a Strategy into llama-server command-line arguments.
func (s *Strategy) Args(modelPath string, port int) []string {
	args := []string{
		"-m", modelPath,
		"--port", fmt.Sprintf("%d", port),
		"-c", fmt.Sprintf("%d", s.ContextSize),
		"-ngl", fmt.Sprintf("%d", s.GPULayers),
		"--threads", fmt.Sprintf("%d", s.Threads),
	}

	if s.MainGPU >= 0 && len(s.TensorSplit) == 0 {
		args = append(args, "-mg", fmt.Sprintf("%d", s.MainGPU))
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
		args = append(args, "-sm", s.SplitMode)
	}

	if s.KVPlacement == "cpu" {
		args = append(args, "--kv-placement", "cpu")
	}

	if s.KVQuality == "high" {
		args = append(args, "--cache-type-k", "f16", "--cache-type-v", "f16")
	} else if s.KVQuality == "mid" {
		args = append(args, "--cache-type-k", "q8_0", "--cache-type-v", "q8_0")
	} else if s.KVQuality == "low" {
		args = append(args, "--cache-type-k", "q4_0", "--cache-type-v", "q4_0")
	}

	if s.NCPUMoE > 0 {
		args = append(args, "--n-cpu-moe", fmt.Sprintf("%d", s.NCPUMoE))
	}

	if s.FlashAttention {
		args = append(args, "-fa")
	}

	if s.MMap {
		args = append(args, "--mmap")
	} else {
		args = append(args, "--no-mmap")
	}

	if s.MLock {
		args = append(args, "--mlock")
	}

	if s.BatchSize > 0 {
		args = append(args, "-b", fmt.Sprintf("%d", s.BatchSize))
	}
	if s.UBatchSize > 0 {
		args = append(args, "-ub", fmt.Sprintf("%d", s.UBatchSize))
	}

	return args
}
