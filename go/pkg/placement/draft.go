package placement

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/raketenkater/llm-server/pkg/detect"
	"github.com/raketenkater/llm-server/pkg/gguf"
)

// DraftType selects the speculative decoding strategy.
type DraftType string

const (
	DraftNone  DraftType = "none"
	DraftModel DraftType = "draft_model"
	DraftNgram DraftType = "ngram"
	DraftMTP   DraftType = "mtp"
)

// DraftConfig holds computed speculative decoding parameters.
// All values are calculated from hardware + model metadata — nothing is guessed.
type DraftConfig struct {
	Type         DraftType `json:"type"`
	BackendTag   string    `json:"backend_tag,omitempty"`    // backend dialect for spec flags
	Path         string    `json:"path,omitempty"`           // draft model GGUF path
	DraftGPU     int       `json:"draft_gpu,omitempty"`      // CUDA device index for draft
	CTXSizeDraft int       `json:"ctx_size_draft,omitempty"` // context size for draft
	KVTypeDraft  string    `json:"kv_type_draft,omitempty"`  // KV type for draft
	ThreadsDraft int       `json:"threads_draft,omitempty"`  // threads for draft generation
	SpecAutoTune bool      `json:"spec_autotune"`            // let llama.cpp auto-tune params
	// Draft model params (calculated, not guessed)
	DraftMax int     `json:"draft_max,omitempty"` // max draft tokens per batch
	DraftMin int     `json:"draft_min,omitempty"` // min draft tokens per batch
	PSplit   float64 `json:"p_split,omitempty"`   // speculative split probability
	// Ngram params (fallback when no matching draft model exists)
	SpecType     string `json:"spec_type,omitempty"` // ngram-map-k, ngram-mod, mtp, etc.
	MTPFlag      bool   `json:"mtp_flag,omitempty"`  // ik_llama legacy MTP enable flag
	NgramN       int    `json:"ngram_n,omitempty"`
	NgramM       int    `json:"ngram_m,omitempty"`
	NgramMinHits int    `json:"ngram_min_hits,omitempty"`
}

// ComputeDraft decides the speculative decoding strategy for a target model.
// It only enables draft-model speculation when a compatible local draft exists;
// ngram speculation is explicit because it needs workload-specific proof.
func ComputeDraft(target *ModelProfile, caps *detect.Capabilities, opts Options) *DraftConfig {
	cfg := &DraftConfig{
		Type:         DraftNone,
		BackendTag:   opts.BackendTag,
		SpecAutoTune: specAutoTuneSupported(opts.BackendTag, opts.BackendHelp),
		DraftMax:     16,
		PSplit:       0.1,
	}

	mode := normalizeSpecMode(opts.SpecMode)
	if mode == "off" || caps == nil || len(caps.GPUs) == 0 || target == nil {
		return cfg
	}

	if mode == "mtp" {
		if !modelSupportsMTP(target) {
			fmt.Fprintf(os.Stderr, "[spec] MTP requires a model with NextN/MTP prediction layers; skipping\n")
			return cfg
		}
		if backendSupportsMTP(opts.BackendTag) {
			cfg.Type = DraftMTP
			cfg.SpecType = "mtp"
			cfg.MTPFlag = true
			return cfg
		}
		if backendHelpSupports(opts.BackendHelp, "draft-mtp") {
			cfg.Type = DraftMTP
			cfg.SpecType = "draft-mtp"
			return cfg
		}
		fmt.Fprintf(os.Stderr, "[spec] MTP requires ik_llama or a llama.cpp backend with draft-mtp support; skipping\n")
		return cfg
	}

	if target.IsMoE && !opts.ForceSpecMoE {
		return cfg
	}

	if isNgramMode(mode) {
		configureNgramDraft(cfg, target, opts, mode)
		return cfg
	}
	if mode != "auto" && mode != "draft" {
		fmt.Fprintf(os.Stderr, "[spec] unknown speculative decoding mode %q; skipping\n", opts.SpecMode)
		return cfg
	}

	// Scan for matching draft model in the same directory as target. Auto mode
	// enables a validated draft model when possible, then falls back to explicit
	// no-draft ngram speculation if the backend supports it.
	modelDir := filepath.Dir(target.Path)
	candidate := findOrDownloadDraftCandidate(target, modelDir, opts.BackendTag)

	if candidate != "" {
		draftInfo, err := validateDraftCandidate(candidate, target, opts.BackendTag)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[spec] rejecting draft %s: %v\n", filepath.Base(candidate), err)
		} else {
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
			return cfg
		}
	}

	if mode == "auto" {
		configureNgramDraft(cfg, target, opts, "auto")
		if cfg.Type == DraftNgram {
			fmt.Fprintf(os.Stderr, "[spec] no compatible draft model found; using %s self-speculation\n", cfg.SpecType)
		}
	}
	return cfg
}

func normalizeSpecMode(mode string) string {
	mode = strings.ToLower(strings.TrimSpace(mode))
	switch mode {
	case "", "off", "none", "false", "0":
		return "off"
	case "auto":
		return "auto"
	case "draft", "draft-model", "draft_model", "model":
		return "draft"
	case "ngram", "ngram-map", "ngram-map-k", "ngram-k":
		return "ngram"
	case "ngram-mod", "ngram_mod", "mod", "self", "self-spec", "self-speculative":
		return "ngram-mod"
	case "ngram-k4v", "ngram-map-k4v", "k4v":
		return "ngram-k4v"
	case "mtp", "draft-mtp":
		return "mtp"
	default:
		return mode
	}
}

func modelSupportsMTP(target *ModelProfile) bool {
	if target == nil {
		return false
	}
	return target.NextNPredictLayers > 0
}

func isNgramMode(mode string) bool {
	return mode == "ngram" || mode == "ngram-mod" || mode == "ngram-k4v"
}

func configureNgramDraft(cfg *DraftConfig, target *ModelProfile, opts Options, mode string) {
	specType := chooseNgramSpecType(opts, mode)
	if specType == "" {
		return
	}
	cfg.Type = DraftNgram
	cfg.SpecType = specType
	cfg.NgramMinHits = 1

	switch specType {
	case "ngram-mod":
		cfg.NgramN = 24
		cfg.DraftMin = 48
		cfg.DraftMax = 64
		if target != nil && !target.IsMoE && mode != "auto" {
			cfg.DraftMin = 8
			cfg.DraftMax = 48
		}
	case "ngram-map-k4v":
		cfg.NgramN = 8
		cfg.NgramM = 8
		cfg.NgramMinHits = 2
		cfg.DraftMax = 64
	default:
		cfg.NgramN = 12
		cfg.NgramM = 48
		cfg.NgramMinHits = 1
	}
}

func chooseNgramSpecType(opts Options, mode string) string {
	if backendSupportsMTP(opts.BackendTag) {
		if mode == "ngram-mod" || mode == "ngram-k4v" {
			fmt.Fprintf(os.Stderr, "[spec] %s is not supported by ik_llama; using ngram map-k\n", mode)
		}
		return "ngram - map - k"
	}

	supports := func(specType string) bool {
		if opts.BackendHelp == "" {
			return specType == "ngram-map-k"
		}
		return backendHelpSupports(opts.BackendHelp, specType)
	}
	fallbackMapK := func() string {
		if supports("ngram-map-k") {
			return "ngram-map-k"
		}
		if supports("ngram-mod") {
			return "ngram-mod"
		}
		return ""
	}

	switch mode {
	case "auto":
		if supports("ngram-mod") {
			return "ngram-mod"
		}
		if supports("ngram-map-k4v") {
			return "ngram-map-k4v"
		}
		return fallbackMapK()
	case "ngram-mod":
		if supports("ngram-mod") {
			return "ngram-mod"
		}
		fmt.Fprintf(os.Stderr, "[spec] backend does not expose ngram-mod; using ngram-map-k\n")
		return fallbackMapK()
	case "ngram-k4v":
		if supports("ngram-map-k4v") {
			return "ngram-map-k4v"
		}
		fmt.Fprintf(os.Stderr, "[spec] backend does not expose ngram-map-k4v; using ngram-map-k\n")
		return fallbackMapK()
	default:
		return fallbackMapK()
	}
}

func specAutoTuneSupported(backendTag, backendHelp string) bool {
	if backendSupportsMTP(backendTag) {
		return true
	}
	return backendHelpSupports(backendHelp, "spec-autotune")
}

func backendHelpSupports(help, token string) bool {
	if help == "" || token == "" {
		return false
	}
	return strings.Contains(strings.ToLower(help), strings.ToLower(token))
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

		// Quick size filter: draft should be small (< 15% of target) when
		// target size is known.
		if target.TotalSizeMB > 0 {
			targetSizeMB := float64(target.TotalSizeMB)
			candSizeMB := float64(info.Size()) / (1024 * 1024)
			if candSizeMB > targetSizeMB*0.15 {
				continue
			}
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

func backendSupportsMTP(backendTag string) bool {
	backendTag = strings.ToLower(strings.TrimSpace(backendTag))
	return backendTag == "ik" || backendTag == "ik_llama" || strings.Contains(backendTag, "ik_llama")
}

func ngramSpecType(backendTag string) string {
	if backendSupportsMTP(backendTag) {
		return "ngram - map - k"
	}
	return "ngram-map-k"
}

func findOrDownloadDraftCandidate(target *ModelProfile, modelDir, backendTag string) string {
	if local := findDraftCandidate(target, modelDir); local != "" {
		return local
	}
	if os.Getenv("LLM_SERVER_SKIP_DRAFT_DOWNLOAD") == "1" {
		return ""
	}
	path, err := downloadDraftCandidate(target, modelDir, backendTag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[spec] %v\n", err)
		return ""
	}
	return path
}

func validateDraftCandidate(path string, target *ModelProfile, backendTag string) (*gguf.Info, error) {
	info, err := gguf.Parse(path)
	if err != nil {
		return nil, fmt.Errorf("parse failed: %w", err)
	}
	if info == nil {
		return nil, fmt.Errorf("empty metadata")
	}
	if info.Architecture == "" || info.Architecture == "unknown" {
		return nil, fmt.Errorf("unknown architecture")
	}
	if info.Architecture == "dflash-draft" && backendSupportsMTP(backendTag) {
		return nil, fmt.Errorf("dflash-draft is not supported by ik_llama")
	}
	if target != nil && target.VocabSize > 0 && info.VocabSize > 0 && info.VocabSize != target.VocabSize {
		return nil, fmt.Errorf("vocab mismatch: draft=%d target=%d", info.VocabSize, target.VocabSize)
	}
	expectedBytes := info.NonExpertBytes + info.ExpertBytes
	if expectedBytes > 0 {
		fi, err := os.Stat(path)
		if err != nil {
			return nil, fmt.Errorf("stat failed: %w", err)
		}
		if fi.Size() < int64(expectedBytes) {
			return nil, fmt.Errorf("incomplete file: %d bytes, expected at least %d", fi.Size(), expectedBytes)
		}
		if target != nil && target.TotalSizeMB > 0 {
			candMB := float64(fi.Size()) / (1024 * 1024)
			if candMB > float64(target.TotalSizeMB)*0.30 {
				return nil, fmt.Errorf("draft too large: %.0fMB target=%dMB", candMB, target.TotalSizeMB)
			}
		}
	}
	return info, nil
}

func downloadDraftCandidate(target *ModelProfile, modelDir, backendTag string) (string, error) {
	if target == nil || modelDir == "" {
		return "", fmt.Errorf("no model directory for draft lookup")
	}
	basename := draftLookupBase(target)
	if basename == "" {
		return "", fmt.Errorf("no model basename for draft lookup")
	}

	var repoCandidates []string
	addRepo := func(repo string) {
		repo = strings.Trim(repo, "/")
		if repo == "" {
			return
		}
		for _, existing := range repoCandidates {
			if existing == repo {
				return
			}
		}
		repoCandidates = append(repoCandidates, repo)
	}
	if target.QuantizedBy != "" {
		addRepo(target.QuantizedBy + "/" + basename + "-GGUF")
		if target.Name != "" && target.Name != basename {
			addRepo(target.QuantizedBy + "/" + target.Name + "-GGUF")
		}
	}
	for _, q := range []string{"unsloth", "bartowski", "lmstudio-community"} {
		if q == target.QuantizedBy {
			continue
		}
		addRepo(q + "/" + basename + "-GGUF")
	}

	safeBasename := sanitizeFilename(basename)
	if safeBasename == "" {
		safeBasename = "model"
	}
	client := &http.Client{Timeout: 30 * time.Second}
	for _, repo := range repoCandidates {
		paths := listRepoDraftCandidates(client, repo)
		seen := map[string]bool{}
		for _, remotePath := range paths {
			if remotePath == "" || seen[remotePath] {
				continue
			}
			seen[remotePath] = true
			remoteBase := filepath.Base(remotePath)
			safeRemote := sanitizeFilename(remoteBase)
			if safeRemote == "" {
				safeRemote = "draft.gguf"
			}
			dest := filepath.Join(modelDir, "draft-"+safeBasename+"-"+safeRemote)
			if _, err := os.Stat(dest); err == nil {
				if _, err := validateDraftCandidate(dest, target, backendTag); err == nil {
					fmt.Fprintf(os.Stderr, "[spec] Found compatible draft model: %s\n", dest)
					return dest, nil
				}
			}

			dlURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, url.PathEscape(remotePath))
			headResp, err := client.Head(dlURL)
			if err != nil || headResp.StatusCode != http.StatusOK {
				if headResp != nil && headResp.Body != nil {
					headResp.Body.Close()
				}
				continue
			}
			headResp.Body.Close()

			fmt.Fprintf(os.Stderr, "[spec] Downloading draft model from %s: %s\n", repo, remotePath)
			tmpDest := dest + ".tmp"
			if err := downloadFile(client, dlURL, tmpDest); err != nil {
				os.Remove(tmpDest)
				continue
			}
			if !isGGUF(tmpDest) {
				fmt.Fprintf(os.Stderr, "[spec] Downloaded draft is not a valid GGUF, removing\n")
				os.Remove(tmpDest)
				continue
			}
			if _, err := validateDraftCandidate(tmpDest, target, backendTag); err != nil {
				fmt.Fprintf(os.Stderr, "[spec] Downloaded draft does not match: %v, removing\n", err)
				os.Remove(tmpDest)
				continue
			}
			if err := os.Rename(tmpDest, dest); err != nil {
				os.Remove(tmpDest)
				return "", fmt.Errorf("rename draft: %w", err)
			}
			fmt.Fprintf(os.Stderr, "[spec] Downloaded draft model: %s\n", dest)
			return dest, nil
		}
	}
	return "", fmt.Errorf("no compatible draft model found on HuggingFace")
}

func draftLookupBase(target *ModelProfile) string {
	for _, value := range []string{target.Basename, target.Name} {
		value = strings.TrimSpace(value)
		if value != "" {
			return trimQuantSuffix(value)
		}
	}
	base := strings.TrimSuffix(filepath.Base(target.Path), ".gguf")
	return trimQuantSuffix(base)
}

func trimQuantSuffix(name string) string {
	for _, suffix := range []string{"Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q4_K_L", "Q5_K_S", "Q5_K_M", "Q5_K_L", "Q6_K", "Q8_0", "F16", "F32", "BF16", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL"} {
		if strings.HasSuffix(name, "-"+suffix) {
			return strings.TrimSuffix(name, "-"+suffix)
		}
	}
	return name
}

func listRepoDraftCandidates(client *http.Client, repo string) []string {
	apiURL := fmt.Sprintf("https://huggingface.co/api/models/%s/tree/main?recursive=1", repo)
	resp, err := client.Get(apiURL)
	if err != nil || resp.StatusCode != http.StatusOK {
		if resp != nil && resp.Body != nil {
			resp.Body.Close()
		}
		return nil
	}
	defer resp.Body.Close()

	var items []struct {
		Path string `json:"path"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&items); err != nil {
		return nil
	}
	var paths []string
	for _, item := range items {
		name := strings.ToLower(filepath.Base(item.Path))
		if strings.HasSuffix(name, ".gguf") && draftFilenameLooksRelevant(name) {
			paths = append(paths, item.Path)
		}
	}
	sort.Slice(paths, func(i, j int) bool {
		return draftCandidateRank(paths[i]) < draftCandidateRank(paths[j])
	})
	return paths
}

func draftFilenameLooksRelevant(name string) bool {
	return strings.Contains(name, "draft") ||
		strings.Contains(name, "eagle") ||
		strings.Contains(name, "dflash") ||
		strings.Contains(name, "mtp")
}

func draftCandidateRank(path string) int {
	name := strings.ToLower(filepath.Base(path))
	score := 10
	switch {
	case strings.Contains(name, "draft"):
		score = 0
	case strings.Contains(name, "eagle"):
		score = 1
	case strings.Contains(name, "dflash"):
		score = 2
	case strings.Contains(name, "mtp"):
		score = 3
	}
	if strings.Contains(name, "q8_0") || strings.Contains(name, "f16") || strings.Contains(name, "bf16") {
		score += 1
	}
	return score
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
	ikDialect := backendSupportsMTP(cfg.BackendTag)
	draftMaxFlag := "--spec-draft-n-max"
	pSplitFlag := "--spec-draft-p-split"
	if ikDialect {
		draftMaxFlag = "--draft-max"
		pSplitFlag = "--p-split"
	}

	switch cfg.Type {
	case DraftModel:
		if cfg.Path != "" {
			flags = append(flags, "--model-draft", cfg.Path)
		}
		if cfg.DraftGPU >= 0 && cfg.Path != "" {
			flags = append(flags, "--device-draft", draftDeviceName(cfg.BackendTag, cfg.DraftGPU))
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
			flags = append(flags, draftMaxFlag, fmt.Sprintf("%d", cfg.DraftMax))
		}
		if cfg.DraftMin > 0 && !ikDialect {
			flags = append(flags, "--spec-draft-n-min", fmt.Sprintf("%d", cfg.DraftMin))
		}
		if cfg.PSplit > 0 {
			flags = append(flags, pSplitFlag, fmt.Sprintf("%.2f", cfg.PSplit))
		}
		if cfg.SpecAutoTune {
			flags = append(flags, "--spec-autotune")
		}

	case DraftNgram:
		specType := cfg.SpecType
		if specType == "" {
			specType = ngramSpecType(cfg.BackendTag)
		}
		flags = append(flags, "--spec-type", specType)
		if ikDialect {
			if cfg.NgramN > 0 {
				flags = append(flags, "--spec-ngram-size-n", fmt.Sprintf("%d", cfg.NgramN))
			}
			if cfg.NgramM > 0 {
				flags = append(flags, "--spec-ngram-size-m", fmt.Sprintf("%d", cfg.NgramM))
			}
			if cfg.NgramMinHits > 0 {
				flags = append(flags, "--spec-ngram-min-hits", fmt.Sprintf("%d", cfg.NgramMinHits))
			}
		} else {
			switch specType {
			case "ngram-mod":
				if cfg.NgramN > 0 {
					flags = append(flags, "--spec-ngram-mod-n-match", fmt.Sprintf("%d", cfg.NgramN))
				}
				if cfg.DraftMin > 0 {
					flags = append(flags, "--spec-ngram-mod-n-min", fmt.Sprintf("%d", cfg.DraftMin))
				}
				if cfg.DraftMax > 0 {
					flags = append(flags, "--spec-ngram-mod-n-max", fmt.Sprintf("%d", cfg.DraftMax))
				}
			case "ngram-map-k4v":
				if cfg.NgramN > 0 {
					flags = append(flags, "--spec-ngram-map-k4v-size-n", fmt.Sprintf("%d", cfg.NgramN))
				}
				if cfg.NgramM > 0 {
					flags = append(flags, "--spec-ngram-map-k4v-size-m", fmt.Sprintf("%d", cfg.NgramM))
				}
				if cfg.NgramMinHits > 0 {
					flags = append(flags, "--spec-ngram-map-k4v-min-hits", fmt.Sprintf("%d", cfg.NgramMinHits))
				}
				if cfg.DraftMax > 0 {
					flags = append(flags, draftMaxFlag, fmt.Sprintf("%d", cfg.DraftMax))
				}
			default:
				if cfg.NgramN > 0 {
					flags = append(flags, "--spec-ngram-map-k-size-n", fmt.Sprintf("%d", cfg.NgramN))
				}
				if cfg.NgramM > 0 {
					flags = append(flags, "--spec-ngram-map-k-size-m", fmt.Sprintf("%d", cfg.NgramM))
				}
				if cfg.NgramMinHits > 0 {
					flags = append(flags, "--spec-ngram-map-k-min-hits", fmt.Sprintf("%d", cfg.NgramMinHits))
				}
			}
		}
		if cfg.SpecAutoTune {
			flags = append(flags, "--spec-autotune")
		}

	case DraftMTP:
		specType := cfg.SpecType
		if specType == "" {
			if ikDialect {
				specType = "mtp"
			} else {
				specType = "draft-mtp"
			}
		}
		flags = append(flags, "--spec-type", specType)
		if ikDialect || cfg.MTPFlag {
			flags = append(flags, "--multi-token-prediction")
		}
		if cfg.DraftMax > 0 {
			flags = append(flags, draftMaxFlag, fmt.Sprintf("%d", cfg.DraftMax))
		}
		if cfg.SpecAutoTune {
			flags = append(flags, "--spec-autotune")
		}
	}

	return flags
}

func draftDeviceName(backendTag string, gpu int) string {
	if strings.Contains(strings.ToLower(backendTag), "vulkan") {
		return fmt.Sprintf("Vulkan%d", gpu)
	}
	return fmt.Sprintf("CUDA%d", gpu)
}

// DraftSummary returns a human-readable summary of the draft strategy.
func DraftSummary(cfg *DraftConfig) string {
	if cfg == nil || cfg.Type == DraftNone {
		return ""
	}
	switch cfg.Type {
	case DraftModel:
		name := filepath.Base(cfg.Path)
		return fmt.Sprintf("speculative decoding: draft model %s (GPU%d, ctx=%d)",
			name, cfg.DraftGPU, cfg.CTXSizeDraft)
	case DraftNgram:
		autotune := ""
		if cfg.SpecAutoTune {
			autotune = ", autotune"
		}
		if cfg.SpecType == "ngram-mod" {
			return fmt.Sprintf("speculative decoding: ngram-mod (match=%d, min=%d, max=%d%s)",
				cfg.NgramN, cfg.DraftMin, cfg.DraftMax, autotune)
		}
		return fmt.Sprintf("speculative decoding: %s (n=%d, m=%d%s)",
			cfg.SpecType, cfg.NgramN, cfg.NgramM, autotune)
	case DraftMTP:
		return fmt.Sprintf("speculative decoding: MTP (%s)", cfg.SpecType)
	default:
		return "speculative decoding: off"
	}
}
