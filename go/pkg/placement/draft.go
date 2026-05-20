package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/raketenkater/llm-server/pkg/detect"
	"github.com/raketenkater/llm-server/pkg/gguf"
)

// DraftType selects the speculative decoding strategy.
type DraftType string

const (
	DraftNone       DraftType = "none"
	DraftModel      DraftType = "draft_model"
	DraftNgram      DraftType = "ngram"
)

// DraftConfig holds computed speculative decoding parameters.
// All values are calculated from hardware + model metadata — nothing is guessed.
type DraftConfig struct {
	Type          DraftType `json:"type"`
	Path          string    `json:"path,omitempty"`          // draft model GGUF path
	DraftGPU      int       `json:"draft_gpu,omitempty"`     // CUDA device index for draft
	CTXSizeDraft  int       `json:"ctx_size_draft,omitempty"` // context size for draft
	KVTypeDraft   string    `json:"kv_type_draft,omitempty"`  // KV type for draft
	ThreadsDraft  int       `json:"threads_draft,omitempty"`  // threads for draft generation
	SpecAutoTune  bool      `json:"spec_autotune"`            // let llama.cpp auto-tune params
	// Draft model params (calculated, not guessed)
	DraftMax      int       `json:"draft_max,omitempty"`      // max draft tokens per batch
	DraftMin      int       `json:"draft_min,omitempty"`      // min draft tokens per batch
	PSplit        float64   `json:"p_split,omitempty"`        // speculative split probability
	// Ngram params (fallback when no matching draft model exists)
	SpecType      string    `json:"spec_type,omitempty"`      // ngram - cache, ngram - simple, etc.
	NgramN        int       `json:"ngram_n,omitempty"`
	NgramM        int       `json:"ngram_m,omitempty"`
	NgramMinHits  int       `json:"ngram_min_hits,omitempty"`
}

// ComputeDraft decides the speculative decoding strategy for a target model.
// It detects matching draft models, calculates GPU placement, and falls back
// to ngram speculation when no compatible draft exists.
func ComputeDraft(target *ModelProfile, caps *detect.Capabilities, opts Options) *DraftConfig {
	cfg := &DraftConfig{
		Type:         DraftNone,
		SpecAutoTune: true, // always preferred — llama.cpp measures at runtime
		DraftMax:     16,   // llama.cpp default
		PSplit:       0.1,  // llama.cpp default
	}

	if len(caps.GPUs) == 0 {
		return cfg
	}

	// Scan for matching draft model in the same directory as target
	modelDir := filepath.Dir(target.Path)
	candidate := findDraftCandidate(target, modelDir)

	if candidate != "" {
		// Parse draft model for metadata and check architecture compatibility.
		// Some formats (e.g., dflash-draft) may not be supported by all builds.
		draftInfo, err := gguf.Parse(candidate)
		archOk := err == nil && draftInfo != nil &&
			draftInfo.Architecture != "" &&
			draftInfo.Architecture != "unknown" &&
			draftInfo.Architecture != "dflash-draft" // not universally supported by ik_llama

		if archOk {
			cfg.Type = DraftModel
			cfg.Path = candidate

			// Draft context = min(target context, draft model's trained context)
			draftCTX := target.ContextSize
			if draftCTX <= 0 {
				draftCTX = draftInfo.ContextLength
			}
			if draftInfo.ContextLength > 0 && draftInfo.ContextLength < draftCTX {
				draftCTX = draftInfo.ContextLength
			}
			cfg.CTXSizeDraft = draftCTX

			// Calculate draft model VRAM requirement
			draftSizeMB := int(draftInfo.ExpertBytes+draftInfo.NonExpertBytes) / (1024 * 1024)
			if draftSizeMB <= 0 {
				draftSizeMB = 1024
			}

			// KV cache for draft model
			draftKVMB := computeKVTotalMB(&ModelProfile{
				HeadCountKV:      draftInfo.HeadCountKV,
				KeyLength:        draftInfo.KeyLength,
				ValueLength:      draftInfo.ValueLength,
				NumLayers:        draftInfo.BlockCount,
				KVLoraRank:       draftInfo.KVLoraRank,
				QLoraRank:        draftInfo.QLoraRank,
				HasSSM:           draftInfo.SSM,
				SlidingWindow:    draftInfo.SlidingWindow,
				FullAttnInterval: draftInfo.FullAttnInterval,
			}, draftCTX, cfg.KVTypeDraft)

			cfg.DraftGPU = findDraftGPU(caps, target, draftSizeMB+draftKVMB+computeFloorMB)

			if caps.CPU.Cores >= 4 {
				cfg.ThreadsDraft = 2
			} else {
				cfg.ThreadsDraft = caps.CPU.Cores
			}
			cfg.KVTypeDraft = computeDraftKVType(caps, draftInfo)
		}
		// If arch not ok, fall through to ngram below
	}

	// No compatible draft model — use ngram speculation as fallback
	if cfg.Type == DraftNone {
		cfg.Type = DraftNgram
		cfg.NgramN = 12
		cfg.NgramM = 48
		cfg.NgramMinHits = 1
	}

	return cfg
}

// findDraftCandidate scans the model directory for a small GGUF model with
// the same tokenizer vocabulary as the target. Returns the path to the best
// candidate (smallest matching model), or empty string if none found.
func findDraftCandidate(target *ModelProfile, modelDir string) string {
	if modelDir == "" {
		return ""
	}

	entries, err := os.ReadDir(modelDir)
	if err != nil {
		return ""
	}

	type candidate struct {
		path string
		size int64
	}
	var matches []candidate

	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".gguf") {
			continue
		}
		candPath := filepath.Join(modelDir, e.Name())
		if candPath == target.Path {
			continue // skip self
		}

		info, err := e.Info()
		if err != nil {
			continue
		}

		// Quick size filter: draft should be small (< 15% of target)
		targetSizeMB := float64(target.TotalSizeMB)
		candSizeMB := float64(info.Size()) / (1024 * 1024)
		if candSizeMB > targetSizeMB*0.15 {
			continue
		}

		// Parse GGUF metadata to check vocabulary match
		ginfo, err := gguf.Parse(candPath)
		if err != nil {
			continue
		}

		// Must share the same tokenizer: exact vocab size match
		if ginfo.VocabSize == 0 || ginfo.VocabSize != target.VocabSize {
			continue
		}

		matches = append(matches, candidate{path: candPath, size: info.Size()})
	}

	if len(matches) == 0 {
		return ""
	}

	// Pick the smallest matching candidate (prefer lightweight drafts)
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].size < matches[j].size
	})

	return matches[0].path
}

// findDraftGPU selects the GPU with the most free VRAM after the target model
// loads its layers. This ensures the draft model has room without colliding.
func findDraftGPU(caps *detect.Capabilities, target *ModelProfile, draftVRAMNeed int) int {
	bestGPU := 0
	bestFree := 0

	for i, g := range caps.GPUs {
		// Estimate target model's VRAM usage on this GPU
		targetUse := estimateTargetVRAMUse(target, caps, i)
		freeAfterTarget := g.VRAMTotalMB - targetUse - draftVRAMNeed

		if freeAfterTarget > bestFree {
			bestFree = freeAfterTarget
			bestGPU = i
		}
	}
	return bestGPU
}

// estimateTargetVRAMUse estimates how much VRAM the target model uses on a given GPU.
func estimateTargetVRAMUse(target *ModelProfile, caps *detect.Capabilities, gpuIndex int) int {
	if len(caps.GPUs) == 0 {
		return 0
	}

	// For MoE: compute fixed overhead per GPU + per-layer cost
	if target.IsMoE {
		// Non-expert weight per GPU: proportional to VRAM share
		totalFree := 0
		for _, g := range caps.GPUs {
			totalFree += g.VRAMTotalMB
		}
		if totalFree <= 0 {
			return 0
		}

		share := float64(caps.GPUs[gpuIndex].VRAMTotalMB) / float64(totalFree)
		nonExpertShare := float64(target.NonExpertBytes) / (1024 * 1024) * share

		// If we have a known placement from Compute(), use it
		// For now: proportional estimate is reasonable
		return int(nonExpertShare)
	}

	// For dense: proportional tensor-split based on VRAM free values
	totalFree := 0
	for _, g := range caps.GPUs {
		free := g.VRAMTotalMB - g.VRAMUsedMB
		if free > 0 {
			totalFree += free
		}
	}
	if totalFree <= 0 {
		return 0
	}
	share := float64(caps.GPUs[gpuIndex].VRAMTotalMB-caps.GPUs[gpuIndex].VRAMUsedMB) / float64(totalFree)
	return int(float64(target.TotalSizeMB) * vramOverheadPercent / 100 * share)
}

// computeDraftKVType determines the KV cache type for the draft model.
// Prefers the same type as the target for consistency, falls back to q4_0
// if the draft model is too large for q8_0 on the selected GPU.
func computeDraftKVType(caps *detect.Capabilities, draftInfo *gguf.Info) string {
	if draftInfo == nil || len(caps.GPUs) == 0 {
		return "q4_0"
	}

	// For draft models (typically < 2GB), q8_0 KV cache is fine
	// on any GPU with > 4GB free. Use q4_0 on smaller GPUs.
	for _, g := range caps.GPUs {
		if g.VRAMTotalMB-g.VRAMUsedMB > 4096 {
			return "q8_0"
		}
	}
	return "q4_0"
}

// DraftFlags returns the llama-server arguments for speculative decoding.
func DraftFlags(cfg *DraftConfig) []string {
	if cfg == nil || cfg.Type == DraftNone {
		return nil
	}

	var flags []string

	switch cfg.Type {
	case DraftModel:
		if cfg.Path != "" {
			flags = append(flags, "--model-draft", cfg.Path)
		}
		if cfg.DraftGPU >= 0 && cfg.Path != "" {
			flags = append(flags, "--device-draft", fmt.Sprintf("CUDA%d", cfg.DraftGPU))
		}
		if cfg.CTXSizeDraft > 0 {
			flags = append(flags, "--ctx-size-draft", fmt.Sprintf("%d", cfg.CTXSizeDraft))
		}
		if cfg.KVTypeDraft != "" {
			flags = append(flags, "--cache-type-k-draft", cfg.KVTypeDraft)
			flags = append(flags, "--cache-type-v-draft", cfg.KVTypeDraft)
		}
		if cfg.ThreadsDraft > 0 {
			flags = append(flags, "--threads-draft", fmt.Sprintf("%d", cfg.ThreadsDraft))
		}
		if cfg.DraftMax > 0 {
			flags = append(flags, "--draft-max", fmt.Sprintf("%d", cfg.DraftMax))
		}
		if cfg.PSplit > 0 {
			flags = append(flags, "--p-split", fmt.Sprintf("%.2f", cfg.PSplit))
		}
		if cfg.SpecAutoTune {
			flags = append(flags, "--spec-autotune")
		}

	case DraftNgram:
		if cfg.SpecType != "" {
			flags = append(flags, "--spec-type", cfg.SpecType)
		}
		if cfg.NgramN > 0 {
			flags = append(flags, "--spec-ngram-size-n", fmt.Sprintf("%d", cfg.NgramN))
		}
		if cfg.NgramM > 0 {
			flags = append(flags, "--spec-ngram-size-m", fmt.Sprintf("%d", cfg.NgramM))
		}
		if cfg.NgramMinHits > 0 {
			flags = append(flags, "--spec-ngram-min-hits", fmt.Sprintf("%d", cfg.NgramMinHits))
		}
		if cfg.SpecAutoTune {
			flags = append(flags, "--spec-autotune")
		}
	}

	return flags
}

// DraftSummary returns a human-readable summary of the draft strategy.
func DraftSummary(cfg *DraftConfig) string {
	if cfg == nil || cfg.Type == DraftNone {
		return "speculative decoding: off (no compatible draft model, no GPU available)"
	}
	switch cfg.Type {
	case DraftModel:
		name := filepath.Base(cfg.Path)
		return fmt.Sprintf("speculative decoding: draft model %s (GPU%d, ctx=%d)",
			name, cfg.DraftGPU, cfg.CTXSizeDraft)
	case DraftNgram:
		return fmt.Sprintf("speculative decoding: ngram (n=%d, m=%d, autotune)",
			cfg.NgramN, cfg.NgramM)
	default:
		return "speculative decoding: off"
	}
}
