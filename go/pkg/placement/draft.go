package placement

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/gguf"
)

// DraftType selects the speculative decoding strategy.
type DraftType string

const (
	DraftNone   DraftType = "none"
	DraftModel  DraftType = "draft_model"
	DraftEagle3 DraftType = "eagle3"
	DraftNgram  DraftType = "ngram"
	DraftMTP    DraftType = "mtp"
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
	// Ngram params (explicit/profile-gated only; not an auto fallback)
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
		configureMTPDraft(cfg, target, opts, true)
		return cfg
	}

	if target.IsMoE && !opts.ForceSpecMoE {
		return cfg
	}

	if isNgramMode(mode) {
		configureNgramDraft(cfg, target, opts, mode)
		return cfg
	}

	modelDir := filepath.Dir(target.Path)

	if mode == "auto" {
		if configureMTPDraft(cfg, target, opts, false) {
			return cfg
		}
		if configureEagle3Draft(cfg, target, caps, opts, modelDir, false) {
			return cfg
		}
		if configureValidatedDraftModel(cfg, target, caps, opts, findOrDownloadDraftCandidate(target, modelDir, opts.BackendTag), DraftModel, "") {
			return cfg
		}
		fmt.Fprintf(os.Stderr, "[spec] auto found no compatible MTP/EAGLE/draft path; leaving speculative decoding off\n")
		return cfg
	}

	if mode == "eagle3" {
		configureEagle3Draft(cfg, target, caps, opts, modelDir, true)
		return cfg
	}

	if mode == "draft" {
		configureValidatedDraftModel(cfg, target, caps, opts, findOrDownloadDraftCandidate(target, modelDir, opts.BackendTag), DraftModel, "")
		return cfg
	}

	fmt.Fprintf(os.Stderr, "[spec] unknown speculative decoding mode %q; skipping\n", opts.SpecMode)
	return cfg
}

func configureMTPDraft(cfg *DraftConfig, target *ModelProfile, opts Options, verbose bool) bool {
	if !modelSupportsMTP(target) {
		if verbose {
			fmt.Fprintf(os.Stderr, "[spec] MTP requires a model with NextN/MTP prediction layers; skipping\n")
		}
		return false
	}
	if backendSupportsMTP(opts.BackendTag) {
		cfg.Type = DraftMTP
		cfg.SpecType = "mtp"
		cfg.MTPFlag = true
		return true
	}
	if backendHelpSupports(opts.BackendHelp, "draft-mtp") {
		cfg.Type = DraftMTP
		cfg.SpecType = "draft-mtp"
		return true
	}
	if verbose {
		fmt.Fprintf(os.Stderr, "[spec] MTP requires ik_llama or a llama.cpp backend with draft-mtp support; skipping\n")
	}
	return false
}

func configureEagle3Draft(cfg *DraftConfig, target *ModelProfile, caps *detect.Capabilities, opts Options, modelDir string, verbose bool) bool {
	if !backendSupportsEagle3(opts) {
		if verbose {
			fmt.Fprintf(os.Stderr, "[spec] EAGLE-3 requires a backend that advertises eagle3 support; skipping\n")
		}
		return false
	}
	candidate := findOrDownloadEagleCandidate(target, modelDir, opts.BackendTag)
	if candidate == "" {
		if verbose {
			fmt.Fprintf(os.Stderr, "[spec] no compatible EAGLE-3 draft model found; skipping\n")
		}
		return false
	}
	return configureValidatedDraftModel(cfg, target, caps, opts, candidate, DraftEagle3, "eagle3")
}

func configureValidatedDraftModel(cfg *DraftConfig, target *ModelProfile, caps *detect.Capabilities, opts Options, candidate string, draftType DraftType, specType string) bool {
	if candidate == "" {
		return false
	}
	draftInfo, err := validateDraftCandidate(candidate, target, opts.BackendTag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[spec] rejecting draft %s: %v\n", filepath.Base(candidate), err)
		return false
	}
	if draftType == DraftModel && !sameDraftArchitecture(target.ModelArch, draftInfo.Architecture) {
		fmt.Fprintf(os.Stderr, "[spec] rejecting draft %s: architecture mismatch draft=%s target=%s\n", filepath.Base(candidate), draftInfo.Architecture, target.ModelArch)
		return false
	}
	cfg.Type = draftType
	cfg.SpecType = specType
	cfg.Path = candidate

	draftCTX := target.ContextSize
	if draftCTX <= 0 {
		draftCTX = draftInfo.ContextLength
	}
	if draftInfo.ContextLength > 0 && draftInfo.ContextLength < draftCTX {
		draftCTX = draftInfo.ContextLength
	}
	cfg.CTXSizeDraft = draftCTX

	draftSizeMB := int(draftInfo.ExpertBytes+draftInfo.NonExpertBytes) / (1024 * 1024)
	if draftSizeMB <= 0 {
		draftSizeMB = 1024
	}
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
	return true
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
	case "eagle", "eagle3", "eagle-3":
		return "eagle3"
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

func backendSupportsEagle3(opts Options) bool {
	return !backendSupportsMTP(opts.BackendTag) && backendHelpSupports(opts.BackendHelp, "eagle3")
}

func sameDraftArchitecture(targetArch, draftArch string) bool {
	targetArch = strings.ToLower(strings.TrimSpace(targetArch))
	draftArch = strings.ToLower(strings.TrimSpace(draftArch))
	if targetArch == "" || draftArch == "" || targetArch == "unknown" || draftArch == "unknown" {
		return true
	}
	return targetArch == draftArch
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

		// os.Stat follows symlinks; e.Info() would report the link's own size.
		info, err := os.Stat(candPath)
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

func findEagleCandidate(target *ModelProfile, modelDir string) string {
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
		if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") || !strings.Contains(strings.ToLower(e.Name()), "eagle") {
			continue
		}
		candPath := filepath.Join(modelDir, e.Name())
		if target != nil && candPath == target.Path {
			continue
		}
		// os.Stat follows symlinks; e.Info() would report the link's own size.
		info, err := os.Stat(candPath)
		if err != nil {
			continue
		}
		ginfo, err := gguf.Parse(candPath)
		if err != nil {
			continue
		}
		if target != nil && target.VocabSize > 0 && ginfo.VocabSize > 0 && ginfo.VocabSize != target.VocabSize {
			continue
		}
		matches = append(matches, candidate{path: candPath, size: info.Size()})
	}
	if len(matches) == 0 {
		return ""
	}
	sort.Slice(matches, func(i, j int) bool { return matches[i].size < matches[j].size })
	return matches[0].path
}

func backendSupportsMTP(backendTag string) bool {
	backendTag = strings.ToLower(strings.TrimSpace(backendTag))
	return backendTag == "ik" || backendTag == "ik_llama" || strings.Contains(backendTag, "ik_llama")
}

func ngramSpecType(backendTag string) string {
	return "ngram-map-k"
}

func findOrDownloadDraftCandidate(target *ModelProfile, modelDir, backendTag string) string {
	if local := findDraftCandidate(target, modelDir); local != "" {
		if _, err := validateDraftCandidate(local, target, backendTag); err == nil {
			return local
		} else {
			fmt.Fprintf(os.Stderr, "[spec] ignoring local draft %s: %v\n", filepath.Base(local), err)
		}
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

func findOrDownloadEagleCandidate(target *ModelProfile, modelDir, backendTag string) string {
	if local := findEagleCandidate(target, modelDir); local != "" {
		if _, err := validateDraftCandidate(local, target, backendTag); err == nil {
			return local
		} else {
			fmt.Fprintf(os.Stderr, "[spec] ignoring local EAGLE draft %s: %v\n", filepath.Base(local), err)
		}
	}
	if os.Getenv("LLM_SERVER_SKIP_DRAFT_DOWNLOAD") == "1" {
		return ""
	}
	path, err := downloadEagleCandidate(target, modelDir, backendTag)
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
	return downloadSpecCandidate(target, modelDir, backendTag, "draft")
}

func downloadEagleCandidate(target *ModelProfile, modelDir, backendTag string) (string, error) {
	return downloadSpecCandidate(target, modelDir, backendTag, "eagle3")
}

func downloadSpecCandidate(target *ModelProfile, modelDir, backendTag, kind string) (string, error) {
	if target == nil || modelDir == "" {
		return "", fmt.Errorf("no model directory for draft lookup")
	}
	basename := draftLookupBase(target)
	if basename == "" {
		return "", fmt.Errorf("no model basename for draft lookup")
	}

	client := &http.Client{Timeout: 30 * time.Second}
	downloadClient := &http.Client{Timeout: 20 * time.Minute}

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
	for _, repo := range searchHFDraftRepos(client, target, kind) {
		addRepo(repo)
	}

	safeBasename := sanitizeFilename(basename)
	if safeBasename == "" {
		safeBasename = "model"
	}
	for _, repo := range repoCandidates {
		paths := listRepoDraftCandidates(client, repo, kind)
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

			dlURL := hfResolveURL(repo, remotePath)
			headResp, err := client.Head(dlURL)
			if err != nil || headResp.StatusCode != http.StatusOK || !hfCandidateSizeOK(headResp, target) {
				if headResp != nil && headResp.Body != nil {
					headResp.Body.Close()
				}
				continue
			}
			headResp.Body.Close()

			fmt.Fprintf(os.Stderr, "[spec] Downloading draft model from %s: %s\n", repo, remotePath)
			tmpDest := dest + ".tmp"
			if err := downloadFile(downloadClient, dlURL, tmpDest); err != nil {
				fmt.Fprintf(os.Stderr, "[spec] download failed for %s: %v\n", remotePath, err)
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
				if draftValidationRepoWideMismatch(err) {
					break
				}
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

func searchHFDraftRepos(client *http.Client, target *ModelProfile, kind string) []string {
	repos := []string{}
	seen := map[string]bool{}
	addRepo := func(repo string) {
		repo = strings.Trim(repo, "/")
		if repo == "" || seen[repo] {
			return
		}
		seen[repo] = true
		repos = append(repos, repo)
	}
	for _, query := range hfSpecSearchQueries(target, kind) {
		apiURL := "https://huggingface.co/api/models?limit=20&search=" + url.QueryEscape(query)
		resp, err := client.Get(apiURL)
		if err != nil || resp.StatusCode != http.StatusOK {
			if resp != nil && resp.Body != nil {
				resp.Body.Close()
			}
			continue
		}
		var rows []struct {
			ID      string `json:"id"`
			ModelID string `json:"modelId"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&rows); err != nil {
			resp.Body.Close()
			continue
		}
		resp.Body.Close()
		for _, row := range rows {
			repo := row.ID
			if repo == "" {
				repo = row.ModelID
			}
			if hfRepoLooksRelevant(repo, target, kind) {
				addRepo(repo)
			}
		}
	}
	return repos
}

func hfSpecSearchQueries(target *ModelProfile, kind string) []string {
	base := draftLookupBase(target)
	pretty := strings.ReplaceAll(base, "-", " ")
	family := draftFamilyName(base)
	queries := []string{}
	add := func(q string) {
		q = strings.Join(strings.Fields(q), " ")
		if q == "" {
			return
		}
		for _, existing := range queries {
			if strings.EqualFold(existing, q) {
				return
			}
		}
		queries = append(queries, q)
	}
	if kind == "eagle3" {
		add(pretty + " EAGLE3 GGUF")
		add(pretty + " EAGLE-3 GGUF")
		add(base + " EAGLE3")
		return queries
	}
	add(pretty + " draft GGUF")
	add(pretty + " drafter GGUF")
	add(pretty + " speculative GGUF")
	add(base + " draft")
	if family != "" {
		add(family + " draft GGUF")
		add(family + " 0.5B GGUF")
		add(family + " 0.6B GGUF")
		add(family + " 0.8B GGUF")
		add(family + " 1.5B GGUF")
		add(family + " 3B GGUF")
	}
	if strings.Contains(strings.ToLower(target.ModelArch), "qwen35") || strings.Contains(strings.ToLower(base), "qwen3.6") {
		add("Qwen3.5 0.8B GGUF")
		add("Qwen3.5 draft GGUF")
	}
	return queries
}

func draftFamilyName(base string) string {
	base = strings.TrimSpace(base)
	if base == "" {
		return ""
	}
	parts := strings.Split(base, "-")
	if len(parts) == 0 {
		return base
	}
	if strings.Contains(strings.ToLower(parts[len(parts)-1]), "b") && len(parts) > 1 {
		parts = parts[:len(parts)-1]
	}
	return strings.Join(parts, " ")
}

func hfRepoLooksRelevant(repo string, target *ModelProfile, kind string) bool {
	repoLower := strings.ToLower(repo)
	baseLower := strings.ToLower(draftLookupBase(target))
	familyLower := compactHFToken(draftFamilyName(draftLookupBase(target)))
	compactRepo := compactHFToken(repoLower)
	archLower := compactHFToken(target.ModelArch)
	if kind == "eagle3" {
		return strings.Contains(repoLower, "eagle") && (baseLower == "" || strings.Contains(compactRepo, compactHFToken(baseLower)) || strings.Contains(compactRepo, familyLower))
	}
	if strings.Contains(repoLower, "draft") || strings.Contains(repoLower, "drafter") || strings.Contains(repoLower, "dflash") || strings.Contains(repoLower, "speculative") {
		return true
	}
	return repoLooksLikeDraftRepo(repoLower) && (familyLower == "" || strings.Contains(compactRepo, familyLower) || (archLower != "" && strings.Contains(compactRepo, archLower)))
}

var smallDraftSizeRE = regexp.MustCompile(`(?i)(^|[-_/ .])(0\.5|0\.6|0\.8|1|1\.5|2|3)b($|[-_/ .])`)

func repoLooksLikeDraftRepo(repoLower string) bool {
	for _, token := range []string{"draft", "drafter", "dflash", "speculative", "eagle"} {
		if strings.Contains(repoLower, token) {
			return true
		}
	}
	return smallDraftSizeRE.MatchString(repoLower)
}

func compactHFToken(value string) string {
	value = strings.ToLower(value)
	for _, old := range []string{"-", "_", ".", "/", " "} {
		value = strings.ReplaceAll(value, old, "")
	}
	return value
}

func hfCandidateSizeOK(resp *http.Response, target *ModelProfile) bool {
	if resp == nil || target == nil || target.TotalSizeMB <= 0 || resp.ContentLength <= 0 {
		return true
	}
	maxBytes := int64(target.TotalSizeMB) * 1024 * 1024 * 30 / 100
	return resp.ContentLength <= maxBytes
}

func hfResolveURL(repo, remotePath string) string {
	parts := strings.Split(remotePath, "/")
	for i, part := range parts {
		parts[i] = url.PathEscape(part)
	}
	return fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, strings.Join(parts, "/"))
}

func draftValidationRepoWideMismatch(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "vocab mismatch") || strings.Contains(msg, "architecture mismatch") || strings.Contains(msg, "not supported")
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

func listRepoDraftCandidates(client *http.Client, repo, kind string) []string {
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
		if strings.HasSuffix(name, ".gguf") && !isNonTextDraftGGUFName(name) && (draftFilenameLooksRelevantForKind(name, kind) || (kind == "draft" && repoLooksLikeDraftRepo(strings.ToLower(repo)))) {
			paths = append(paths, item.Path)
		}
	}
	sort.Slice(paths, func(i, j int) bool {
		return draftCandidateRank(paths[i], kind) < draftCandidateRank(paths[j], kind)
	})
	return paths
}

func draftFilenameLooksRelevant(name string) bool {
	return draftFilenameLooksRelevantForKind(name, "draft")
}

func draftFilenameLooksRelevantForKind(name, kind string) bool {
	name = strings.ToLower(name)
	if isNonTextDraftGGUFName(name) {
		return false
	}
	if kind == "eagle3" {
		return strings.Contains(name, "eagle")
	}
	return strings.Contains(name, "draft") ||
		strings.Contains(name, "dflash")
}

func isNonTextDraftGGUFName(name string) bool {
	name = strings.ToLower(filepath.Base(name))
	for _, token := range []string{"mmproj", "projector", "vision", "clip", "siglip", "vit", "encoder", "imatrix", "calibration", "dataset", "tokenizer"} {
		if strings.Contains(name, token) {
			return true
		}
	}
	return false
}

func draftCandidateRank(path, kind string) int {
	name := strings.ToLower(filepath.Base(path))
	score := 10
	switch {
	case kind == "eagle3" && strings.Contains(name, "eagle3"):
		score = 0
	case kind == "eagle3" && strings.Contains(name, "eagle"):
		score = 1
	case strings.Contains(name, "draft"):
		score = 0
	case strings.Contains(name, "dflash"):
		score = 2
	case strings.Contains(name, "mtp"):
		score = 3
	}
	return score*10 + draftQuantRank(name)
}

func draftQuantRank(name string) int {
	name = strings.ToLower(name)
	switch {
	case strings.Contains(name, "q4_k_m") || strings.Contains(name, "q4_0") || strings.Contains(name, "q4_k_s"):
		return 0
	case strings.Contains(name, "iq4") || strings.Contains(name, "ud-q4"):
		return 1
	case strings.Contains(name, "q5_k_m") || strings.Contains(name, "q5_k_s"):
		return 2
	case strings.Contains(name, "q3") || strings.Contains(name, "iq3"):
		return 3
	case strings.Contains(name, "q6") || strings.Contains(name, "q8") || strings.Contains(name, "f16") || strings.Contains(name, "bf16"):
		return 4
	case strings.Contains(name, "q2") || strings.Contains(name, "iq2"):
		return 5
	default:
		return 6
	}
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

		// Approximate this GPU's non-expert weight by its share of total VRAM;
		// exact per-tensor placement isn't needed for this estimate.
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
		pSplitFlag = "--p-split"
	}

	switch cfg.Type {
	case DraftModel, DraftEagle3:
		specType := cfg.SpecType
		if cfg.Type == DraftEagle3 && specType == "" {
			specType = "eagle3"
		}
		if cfg.Type == DraftModel && ikDialect {
			if specType == "" {
				specType = "draft"
			}
			flags = append(flags, "--spec-type", specTypeWithNMax(specType, cfg.DraftMax))
		} else if specType != "" {
			flags = append(flags, "--spec-type", specType)
		}
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
		if cfg.DraftMax > 0 && !(ikDialect && cfg.Type == DraftModel) {
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
		if ikDialect {
			flags = append(flags, "--spec-type", specTypeWithNMax(specType, cfg.DraftMax))
		} else {
			flags = append(flags, "--spec-type", specType)
		}
		if ikDialect || cfg.MTPFlag {
			flags = append(flags, "--multi-token-prediction")
		}
		if cfg.DraftMax > 0 && !ikDialect {
			flags = append(flags, draftMaxFlag, fmt.Sprintf("%d", cfg.DraftMax))
		}
		if cfg.SpecAutoTune {
			flags = append(flags, "--spec-autotune")
		}
	}

	return flags
}

func specTypeWithNMax(specType string, nMax int) string {
	if nMax <= 0 || strings.Contains(specType, ":") {
		return specType
	}
	return fmt.Sprintf("%s:n_max=%d", specType, nMax)
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
	case DraftEagle3:
		name := filepath.Base(cfg.Path)
		return fmt.Sprintf("speculative decoding: EAGLE-3 %s (GPU%d, ctx=%d)",
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
