package recommend

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

//go:embed catalog.json
var catalogJSON []byte

// QuantOption describes one downloadable GGUF quant for a candidate repo.
// SizeGB is the summed size of all split GGUF files for that quant.
type QuantOption struct {
	Name           string  `json:"name"`
	SizeGB         float64 `json:"size_gb"`
	SizeBytes      int64   `json:"size_bytes,omitempty"`
	QualityPenalty float64 `json:"quality_penalty,omitempty"`
	Dynamic        bool    `json:"dynamic,omitempty"` // Unsloth UD- quant: loss * 0.7
}

// Candidate is a GGUF repository that ggrun can offer as a first-run
// download. Quality is the intelligence-first ranking signal refreshed from
// the checked-in catalog; speed is display metadata only.
type Candidate struct {
	Name           string        `json:"name"`
	Repo           string        `json:"repo"`
	Family         string        `json:"family"`
	SizeGB         float64       `json:"size_gb"`
	Quality        int           `json:"quality"`
	Speed          int           `json:"speed"`
	MoE            bool          `json:"moe"`
	TotalParamsB   float64       `json:"total_params_b,omitempty"`
	ActiveParamsB  float64       `json:"active_params_b,omitempty"`
	Quants         []QuantOption `json:"quants,omitempty"`
	Notes          string        `json:"notes"`
	AAQuery        string        `json:"aa_query,omitempty"`
	AASlug         string        `json:"aa_slug,omitempty"`
	AAIntelligence float64       `json:"aa_intelligence_index,omitempty"`
	AAOutputTPS    float64       `json:"aa_output_tps,omitempty"`
	AAUpdatedAt    string        `json:"aa_updated_at,omitempty"`

	// GGUF geometry read from the binary header by the catalog builder
	// (tools/models/update_recommendations.py fetch_gguf_arch — one HF range
	// request per repo). These are the exact fields the placement engine
	// (go/pkg/placement/placement.go) uses to compute launch overhead, so the
	// recommender can size a quant with the same formula instead of estimating.
	// All omitempty: legacy catalog entries without geometry fall back to the
	// size-only overhead estimate.
	Arch         string `json:"arch,omitempty"`
	Layers       int    `json:"layers,omitempty"`
	Experts      int    `json:"experts,omitempty"`
	ExpertUsed   int    `json:"exp_used,omitempty"`
	ExpertFF     int    `json:"exp_ff,omitempty"`
	ExpertShFF   int    `json:"exp_shared_ff,omitempty"`
	Embedding    int    `json:"embd,omitempty"`
	FeedForward  int    `json:"ff,omitempty"`
	HeadCountKV  int    `json:"hkv,omitempty"`
	KeyLength    int    `json:"kl,omitempty"`
	ValueLength  int    `json:"vl,omitempty"`
	KVLoraRank   int    `json:"kv_lora,omitempty"`
	QLoraRank    int    `json:"q_lora,omitempty"`
	LeadingDense int    `json:"leading_dense,omitempty"`
	TrainCtx     int    `json:"ctx_train,omitempty"`
}

// Recommendation is a candidate ranked for the current machine.
type Recommendation struct {
	Candidate
	Fit                  string
	BackendHint          string
	Reason               string
	Score                int
	QuantName            string
	QuantSizeGB          float64
	MemoryNeedGB         float64
	AdjustedIntelligence float64 // base AA index * quality retained after quantization
	QualityRetained      float64 // fraction of base intelligence kept by this quant (0,1]
	PredictedTPS         float64 // estimated decode tok/s on this machine (0 = unknown)
	SpeedTier            int     // 2 interactive, 1 usable, 0 slow, -1 unknown
}

type catalogDoc struct {
	Version     int         `json:"version"`
	GeneratedAt string      `json:"generated_at"`
	Source      string      `json:"source"`
	Attribution string      `json:"attribution"`
	Candidates  []Candidate `json:"candidates"`
}

const Attribution = "Artificial Analysis intelligence data is used when available; cached locally and filtered by ggrun hardware fit"

// DisplayFit shortens internal fit-mode labels for display in the CLI and TUI.
func DisplayFit(fit string) string {
	switch fit {
	case "single GPU":
		return "GPU"
	case "multi-GPU":
		return "multi-GPU"
	case "MoE RAM+VRAM", "GPU plus RAM":
		return "GPU+RAM"
	case "CPU RAM":
		return "CPU"
	default:
		return fit
	}
}

func Shortlist() []Candidate {
	var doc catalogDoc
	if err := json.Unmarshal(catalogBytes(), &doc); err == nil && len(doc.Candidates) > 0 {
		return doc.Candidates
	}
	return fallbackShortlist()
}

func CatalogAttribution() string {
	var doc catalogDoc
	if err := json.Unmarshal(catalogBytes(), &doc); err == nil && doc.Attribution != "" {
		return doc.Attribution
	}
	return Attribution
}

func allRecommendations(caps *detect.Capabilities) []Recommendation {
	return allRecommendationsWithQuantFilter(caps, nil)
}

// allRecommendationsWithQuantFilter ranks each runnable model at its
// highest-quality fitting quant (intelligence-first). Used by Smartest and, with
// a quant filter, by Fastest.
func allRecommendationsWithQuantFilter(caps *detect.Capabilities, allowQuant func(QuantOption) bool) []Recommendation {
	return collectRecommendations(caps, allowQuant, better)
}

// allRecommendationsBalanced selects each model's representative quant by the
// blended score (effective intelligence * speed) instead of by raw
// intelligence, so a model surfaces at its best *practical* quant in Best
// overall — e.g. a 27B dense at Q5 (~40 tok/s) rather than BF16 (~3 tok/s on RAM
// spill). Smartest keeps the intelligence-first selector so quality-first picks
// still surface there.
func allRecommendationsBalanced(caps *detect.Capabilities) []Recommendation {
	return collectRecommendations(caps, nil, betterByScore)
}

// collectRecommendations evaluates every runnable catalog model on this machine,
// choosing one representative quant per model with isBetter. Models whose
// architecture no bundled backend can load are skipped (see archRunnable).
func collectRecommendations(caps *detect.Capabilities, allowQuant func(QuantOption) bool, isBetter func(a, b Recommendation) bool) []Recommendation {
	var rows []Recommendation
	for _, c := range Shortlist() {
		if !archRunnable(c) {
			continue
		}
		if rec, ok := evaluateWithSelector(caps, c, allowQuant, isBetter); ok {
			rows = append(rows, rec)
		}
	}
	return rows
}

func Top(caps *detect.Capabilities, limit int) []Recommendation {
	if limit <= 0 {
		limit = 5
	}
	rows := allRecommendations(caps)
	sortRecommendations(rows)
	if len(rows) > limit {
		rows = rows[:limit]
	}
	return rows
}

// Categories groups recommendations by intent so the intelligence/speed/fit
// tradeoff is explicit instead of collapsed into one ranked list.
type Categories struct {
	Balanced []Recommendation // best blend of intelligence, speed and fit
	Smartest []Recommendation // highest effective intelligence that fits (may be slow)
	Fastest  []Recommendation // fastest while still genuinely capable
}

// TopCategories returns up to n picks per category. Each category is ranked and
// deduped independently (one quant per model within a category); a model may
// appear in more than one category when it genuinely is both, say, the best
// overall blend and the smartest that fits.
func TopCategories(caps *detect.Capabilities, n int) Categories {
	if n <= 0 {
		n = 4
	}
	rows := allRecommendations(caps)
	if len(rows) == 0 {
		return Categories{}
	}

	maxEff := 0.0
	for _, r := range rows {
		if r.AdjustedIntelligence > maxEff {
			maxEff = r.AdjustedIntelligence
		}
	}

	// Each category dedups within itself (one quant per model). Categories do
	// NOT steal models from each other — cross-category dedup left "Smartest"
	// showing a leftover low-intelligence pick once "Best overall" had taken the
	// genuinely smarter models above it.
	take := func(pool []Recommendation) []Recommendation {
		seen := map[string]bool{}
		out := make([]Recommendation, 0, n)
		for _, r := range pool {
			if seen[r.Repo] {
				continue
			}
			seen[r.Repo] = true
			out = append(out, r)
			if len(out) == n {
				break
			}
		}
		return out
	}

	// Balanced: each model at its best *practical* quant (blended score), then
	// ranked by that score. This uses a separate evaluation pass (betterByScore)
	// so a fast Q5 represents a model here even though Smartest below still sees
	// the intelligence-first reps in `rows` (e.g. the same model at BF16).
	balancedPool := allRecommendationsBalanced(caps)
	sortRecommendations(balancedPool)
	balanced := take(balancedPool)

	// Smartest: effective intelligence, lightly usability-weighted so a
	// marginally-smarter quant that crawls (e.g. a 27B at BF16 @ 3 tok/s) does
	// not beat a nearly-as-smart, far faster quant of the same model (the same
	// 27B at Q5 @ 40 tok/s). Genuinely-smartest-but-slow picks (a big MoE at
	// ~6 tok/s) still surface — the weighting only bites below usableTPS.
	smartPool := append([]Recommendation(nil), rows...)
	smartKey := func(r Recommendation) float64 {
		return r.AdjustedIntelligence * usabilityFactor(r.PredictedTPS)
	}
	sort.SliceStable(smartPool, func(i, j int) bool {
		ki, kj := smartKey(smartPool[i]), smartKey(smartPool[j])
		if ki == kj {
			return smartPool[i].QuantSizeGB < smartPool[j].QuantSizeGB
		}
		return ki > kj
	})
	smartest := take(smartPool)

	// Fastest: highest predicted tok/s among models that are still capable
	// (>= 40% of the best effective intelligence on this machine). The floor
	// is lower than Smartest's implicit bar because this category is about
	// speed — a 158 t/s model at 53% of max intelligence is genuinely useful.
	// Use a Q4-class ceiling so Fastest means fast/small by default; Q5/Q6/Q8
	// and BF16/F16/F32 still belong in Best overall and Smartest when memory
	// allows.
	// Fastest does NOT dedup against Balanced/Smartest: the fastest models
	// often also score well in Best overall (speedFactor caps at interactive),
	// and deduping them out would leave Fastest showing slow leftovers.
	floor := 0.40 * maxEff
	fastRows := allRecommendationsWithQuantFilter(caps, fastestQuantAllowed)
	fastPool := make([]Recommendation, 0, len(fastRows))
	for _, r := range fastRows {
		if r.AdjustedIntelligence >= floor && r.PredictedTPS > 0 {
			fastPool = append(fastPool, r)
		}
	}
	sort.SliceStable(fastPool, func(i, j int) bool {
		return fastPool[i].PredictedTPS > fastPool[j].PredictedTPS
	})
	fastest := take(fastPool)

	return Categories{Balanced: balanced, Smartest: smartest, Fastest: fastest}
}

func sortRecommendations(rows []Recommendation) {
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Score == rows[j].Score {
			if rows[i].AdjustedIntelligence == rows[j].AdjustedIntelligence {
				return rows[i].QuantSizeGB < rows[j].QuantSizeGB
			}
			return rows[i].AdjustedIntelligence > rows[j].AdjustedIntelligence
		}
		return rows[i].Score > rows[j].Score
	})
}

type hardwareBudget struct {
	largestVRAM int
	totalVRAM   int
	usableRAM   int // freeRAM minus a safety reserve (for the estimate path)
	freeRAM     int // raw available RAM (for the exact-overhead path)
	gpuCount    int
}

func hardware(caps *detect.Capabilities) hardwareBudget {
	budget := hardwareBudget{}
	totalRAM := 8192
	if caps != nil {
		budget.gpuCount = len(caps.GPUs)
		for _, g := range caps.GPUs {
			budget.totalVRAM += g.VRAMTotalMB
			if g.VRAMTotalMB > budget.largestVRAM {
				budget.largestVRAM = g.VRAMTotalMB
			}
		}
		if caps.RAM.TotalMB > 0 {
			totalRAM = caps.RAM.TotalMB
		}
	}
	// The recommender is a planning tool: base RAM on total hardware capacity,
	// not currently-available RAM (MemAvailable / AvailPhys), which reflects
	// running processes. A user with M3 already loaded still wants to see what
	// their machine *can* run when idle. The launcher separately uses FreeMB
	// for actual allocation. TotalMB is correctly detected on Linux, macOS,
	// and Windows.
	reserve := 4096
	if totalRAM >= 65536 {
		reserve = 8192
	}
	budget.usableRAM = totalRAM - reserve // legacy estimate path: OS + safety reserve
	budget.freeRAM = totalRAM - 2048      // exact-overhead path: minimal OS reserve
	if budget.usableRAM < 0 {
		budget.usableRAM = 0
	}
	if budget.freeRAM < 0 {
		budget.freeRAM = 0
	}
	return budget
}

func evaluate(caps *detect.Capabilities, c Candidate) (Recommendation, bool) {
	return evaluateWithQuantFilter(caps, c, nil)
}

func evaluateWithQuantFilter(caps *detect.Capabilities, c Candidate, allowQuant func(QuantOption) bool) (Recommendation, bool) {
	return evaluateWithSelector(caps, c, allowQuant, better)
}

// evaluateWithSelector picks one representative quant for a candidate. isBetter
// decides which of two fitting quants wins: better (intelligence-first, the
// default and Smartest) or betterByScore (blended, Best overall).
func evaluateWithSelector(caps *detect.Capabilities, c Candidate, allowQuant func(QuantOption) bool, isBetter func(a, b Recommendation) bool) (Recommendation, bool) {
	budget := hardware(caps)
	backend := backendHint(caps)
	base := modelIntelligence(c)
	if base <= 0 {
		return Recommendation{}, false
	}

	quants := quantOptions(c)
	var best Recommendation
	ok := false
	for _, q := range quants {
		if q.SizeGB <= 0 {
			continue
		}
		if allowQuant != nil && !allowQuant(q) {
			continue
		}
		// Reject physically impossible catalog sizes (e.g. a phantom F16 at
		// 0.9GB for a 30B model from incomplete Hugging Face shard metadata).
		if !plausibleQuantSize(c, q) {
			continue
		}
		fit, reason, _, needGB, fits := fitQuant(budget, caps, c, q)
		if !fits {
			continue
		}
		retained := quantQualityRetention(q.Name, c.TotalParamsB, q.Dynamic)
		effIntel := base * retained
		if effIntel <= 0 {
			continue
		}
		tps := predictDecodeTPS(caps, c, q)
		score := effIntel * speedFactor(tps)
		rec := Recommendation{
			Candidate:            c,
			Fit:                  fit,
			BackendHint:          backend,
			Reason:               speedReason(reason, tps),
			QuantName:            q.Name,
			QuantSizeGB:          q.SizeGB,
			MemoryNeedGB:         needGB,
			AdjustedIntelligence: effIntel,
			QualityRetained:      retained,
			PredictedTPS:         tps,
			SpeedTier:            speedTier(tps),
			Score:                int(score * 1000),
		}
		if !ok || isBetter(rec, best) {
			best = rec
			ok = true
		}
	}
	return best, ok
}

func fastestQuantAllowed(q QuantOption) bool {
	name := strings.ToUpper(q.Name)
	switch {
	case strings.Contains(name, "F32"), strings.Contains(name, "BF16"), strings.Contains(name, "F16"), strings.Contains(name, "F8"):
		return false
	case strings.Contains(name, "Q8"), strings.Contains(name, "Q6"), strings.Contains(name, "IQ6"), strings.Contains(name, "Q5"), strings.Contains(name, "IQ5"):
		return false
	case strings.Contains(name, "MXFP4"), strings.Contains(name, "MXP4"), strings.Contains(name, "IQ4"), strings.Contains(name, "Q4"), strings.Contains(name, "I4"):
		return true
	case strings.Contains(name, "IQ3"), strings.Contains(name, "Q3"), strings.Contains(name, "IQ2"), strings.Contains(name, "Q2"), strings.Contains(name, "IQ1"), strings.Contains(name, "Q1"):
		return true
	default:
		return false
	}
}

func speedReason(fitReason string, tps float64) string {
	if tps <= 0 {
		return fitReason
	}
	return fmt.Sprintf("%s; ~%.0f tok/s predicted", fitReason, tps)
}

// better decides which of two quants of the SAME candidate to recommend.
// Within one model, prefer the highest effective intelligence that fits —
// a user who picked a 428B MoE is choosing capability, and a 90%-retained
// quant should not lose to an 80% one because it decodes a few tok/s slower
// on a model that is already in the "usable, not interactive" speed tier.
// Speed only breaks a true quality tie; download size breaks a speed tie.
func better(a, b Recommendation) bool {
	if a.AdjustedIntelligence != b.AdjustedIntelligence {
		return a.AdjustedIntelligence > b.AdjustedIntelligence
	}
	if a.PredictedTPS != b.PredictedTPS {
		return a.PredictedTPS > b.PredictedTPS
	}
	// True tie on capability and speed: prefer the smaller download.
	return a.QuantSizeGB < b.QuantSizeGB
}

// betterByScore decides which of two quants of the SAME candidate represents it
// in Best overall. The key is the blended score (effective intelligence * speed)
// weighted by fit efficiency. This matters because speedFactor saturates once a
// quant clears the interactive knee, so every fast-enough quant ties on raw
// score and the pick collapses to max retention — i.e. a single-GPU Q5 (~26 t/s,
// 99% quality, leaves two GPUs free) would lose to a multi-GPU Q8 (~17 t/s,
// 99.8% quality, saturates the box) over a 0.8% quality gap. fitFactor restores
// the "and fit" half of the category: among near-equal-quality quants the one
// that leaves the machine usable wins, so Best overall lands on the practical
// daily-driver quant rather than drifting into Smartest territory. It also lets
// a fast Q5 beat a BF16 that only fits by spilling to RAM. A scalar key keeps
// the pairwise selection in evaluateWithSelector a proper (transitive) ordering.
func betterByScore(a, b Recommendation) bool {
	ka := float64(a.Score) * fitFactor(a.Fit)
	kb := float64(b.Score) * fitFactor(b.Fit)
	if ka != kb {
		return ka > kb
	}
	if a.PredictedTPS != b.PredictedTPS {
		return a.PredictedTPS > b.PredictedTPS
	}
	if a.QuantSizeGB != b.QuantSizeGB {
		return a.QuantSizeGB < b.QuantSizeGB
	}
	return a.AdjustedIntelligence > b.AdjustedIntelligence
}

// fitFactor gently discounts placements that consume more of the machine, so the
// Best-overall selector prefers a quant that leaves hardware free when quality
// and speed are otherwise comparable. Multipliers are mild (>=0.80) so they only
// break near-ties — a model that can only run via RAM spill still scores on its
// merits. Applied only in betterByScore (within one model); cross-model ranking
// and the quality-first selector (better) are unaffected.
func fitFactor(fit string) float64 {
	switch fit {
	case "single GPU":
		return 1.00
	case "multi-GPU":
		return 0.92
	case "MoE RAM+VRAM":
		return 0.90
	case "GPU plus RAM":
		return 0.85
	case "CPU RAM":
		return 0.80
	default:
		return 1.00
	}
}

// unrunnableArch lists model architectures present in the catalog that no
// bundled backend (mainline llama.cpp or ik_llama.cpp) can load yet. The
// recommender must not surface them: the launcher would only fail with
// "unknown architecture". They stay in catalog.json so a power user pointing
// LLAMA_SERVER at a custom build can still find them — e.g. deepseek4 (DeepSeek
// V4 Pro/Flash) loads only on antirez/llama.cpp PR #22378, which ggrun already
// warns about in warnModelCompatibility.
//
// This is a launch-time stopgap. The durable fix stamps a per-candidate
// `runnable` flag at catalog-build time from the bundled backend's arch table
// (tools/models/update_recommendations.py) so it can never drift from the
// shipped binaries; see the data-driven-arch-filter task.
var unrunnableArch = map[string]bool{
	"deepseek4":           true, // DeepSeek V4 Pro/Flash — needs antirez/llama.cpp PR #22378
	"bailingmoe2.5":       true, // InclusionAI Ling 2.6 Flash
	"longcat-flash-ngram": true, // LongCat Flash Lite
	"mllama":              true, // Llama 3.2 Vision — not registered in the bundled build
}

// archRunnable reports whether the bundled backends can load this candidate's
// architecture. Entries without arch metadata (legacy catalog rows) are kept:
// the launcher's own preflight still guards them at load time.
func archRunnable(c Candidate) bool {
	if c.Arch == "" {
		return true
	}
	return !unrunnableArch[strings.ToLower(c.Arch)]
}

func quantOptions(c Candidate) []QuantOption {
	if len(c.Quants) > 0 {
		out := append([]QuantOption(nil), c.Quants...)
		sort.SliceStable(out, func(i, j int) bool { return out[i].SizeGB < out[j].SizeGB })
		return out
	}
	name := "catalog"
	if c.SizeGB > 0 {
		name = "auto"
	}
	return []QuantOption{{Name: name, SizeGB: c.SizeGB}}
}

func fitQuant(b hardwareBudget, caps *detect.Capabilities, c Candidate, q QuantOption) (fit, reason string, fitPenalty, needGB float64, ok bool) {
	modelMB := int(q.SizeGB * 1024)

	// When the catalog carries real GGUF geometry, compute overhead per fit
	// mode with the placement engine's exact terms — the overhead differs by
	// mode (single-GPU has no pinned-host/mmap overhead; RAM-split does).
	// hasGeometry gates this; legacy entries use the single estimate below.
	type modeOverhead struct {
		overheadMB int
		fit        string
		reason     string
		penalty    float64
		budgetMB   int
	}
	candidates := []modeOverhead{}
	if hasGeometry(c) {
		ctx := recommendAutoContext(caps, c, modelMB)
		kvMB := recommendKVMB(c, ctx, "q4_0")
		// GPU-resident modes (single/multi-GPU): model + KV + compute buffer.
		// No cudaHost/mmapPT — the model lives in VRAM, not pinned host memory.
		gpuOH := 2048 + kvMB // graph scratch + KV
		// RAM-split modes (MoE RAM+VRAM, GPU+RAM, CPU): add pinned-host + mmap
		// page-table + CPU activation, matching placement.buildMoEOffload /
		// buildDenseCPUOffload.
		ramOH := gpuOH + 1024 + modelMB/500 + recommendCPUActMB(c)

		candidates = []modeOverhead{
			{gpuOH, "single GPU", "single GPU", 0, b.largestVRAM},
			{gpuOH, "multi-GPU", "multi-GPU", 0.25, b.totalVRAM},
			{ramOH, "MoE RAM+VRAM", "MoE RAM+VRAM", 1.0, b.totalVRAM + b.freeRAM},
			{ramOH, "GPU plus RAM", "GPU plus RAM", 2.0, b.totalVRAM + b.freeRAM},
			{ramOH, "CPU RAM", "CPU RAM", 4.0, b.freeRAM},
		}
	} else {
		// Legacy: single estimate, same for all modes.
		oh := estimateOverheadMB(modelMB, caps, c)
		candidates = []modeOverhead{
			{oh, "single GPU", "single GPU", 0, b.largestVRAM},
			{oh, "multi-GPU", "multi-GPU", 0.25, b.totalVRAM},
			{oh, "MoE RAM+VRAM", "MoE RAM+VRAM", 1.0, b.totalVRAM + b.usableRAM},
			{oh, "GPU plus RAM", "GPU plus RAM", 2.0, b.totalVRAM + b.usableRAM},
			{oh, "CPU RAM", "CPU RAM", 4.0, b.usableRAM},
		}
	}

	// Evaluate fit modes in priority order, matching the original switch.
	modes := candidates
	if !c.MoE {
		// Dense models skip the MoE RAM+VRAM mode.
		modes = make([]modeOverhead, 0, len(candidates))
		for _, m := range candidates {
			if m.fit != "MoE RAM+VRAM" {
				modes = append(modes, m)
			}
		}
	}
	for i, m := range modes {
		needMB := modelMB + m.overheadMB
		needGB = float64(needMB) / 1024
		fits := false
		switch m.fit {
		case "single GPU":
			fits = b.gpuCount > 0 && needMB <= m.budgetMB
		case "multi-GPU":
			fits = b.gpuCount > 1 && needMB <= m.budgetMB
		case "MoE RAM+VRAM":
			fits = b.gpuCount > 0 && needMB <= m.budgetMB && c.MoE
		case "GPU plus RAM":
			fits = b.gpuCount > 0 && needMB <= m.budgetMB
		case "CPU RAM":
			fits = b.gpuCount == 0 && needMB <= m.budgetMB
		}
		if fits {
			reason = fmt.Sprintf("%s fits in %s (%.1fGB model + ~%.1fGB overhead)", q.Name, m.fit, q.SizeGB, float64(m.overheadMB)/1024)
			return m.fit, reason, m.penalty, needGB, true
		}
		_ = i
	}
	// No mode fit; report the last computed needGB for the default case.
	return "", "", 0, needGB, false
}

func estimateOverheadMB(modelMB int, caps *detect.Capabilities, c Candidate) int {
	if modelMB <= 0 {
		return 2048
	}
	// Legacy fallback for catalog entries without GGUF geometry. The primary
	// path (fitQuant with hasGeometry) computes per-mode exact overhead; this
	// is only reached when the catalog predates geometry support.
	var overhead int
	if c.MoE && c.ActiveParamsB > 0 {
		computeMB := maxInt(2048, int(c.ActiveParamsB*512))
		kvMB := int(float64(modelMB) * 0.06)
		overhead = computeMB + kvMB
	} else if c.MoE {
		overhead = 4096 + int(float64(modelMB)*0.06)
	} else {
		overhead = 2048 + int(float64(modelMB)*0.10)
	}
	if overhead < 2048 {
		return 2048
	}
	if overhead > 12288 {
		return 12288
	}
	return overhead
}

// hasGeometry reports whether the catalog entry carries the GGUF header fields
// needed to compute overhead exactly. Requires the layer count plus enough KV
// geometry to size the cache; everything else has a sane default.
func hasGeometry(c Candidate) bool {
	return c.Layers > 0 && c.HeadCountKV > 0 && c.KeyLength > 0 && c.ValueLength > 0
}

// recommendCtxMin is the minimum context the launcher targets (matches
// placement.defaultContextSize's 32k floor for capable machines).
const recommendCtxMin = 32768

// recommendCPUActMB computes the CPU activation memory (ubatch * (embd + ffn)
// * 4 bytes * 2) using real GGUF geometry, matching placement.buildMoEOffload's
// actFFN. For MoE, ffn = exp_used*exp_ff + exp_shared_ff; for dense, falls back
// to feed_forward_length. MLA models add kv_lora + q_lora ranks.
func recommendCPUActMB(c Candidate) int {
	const ubatchDefault = 512
	var ffn int
	if c.ExpertUsed > 0 && c.ExpertFF > 0 {
		ffn = c.ExpertUsed * c.ExpertFF
		if c.ExpertShFF > 0 {
			ffn += c.ExpertShFF
		}
	} else if c.FeedForward > 0 {
		ffn = c.FeedForward
	}
	if c.KVLoraRank > 0 {
		ffn += c.KVLoraRank + c.QLoraRank
	}
	embd := c.Embedding
	if embd <= 0 {
		embd = 4096
	}
	cpuActMB := ubatchDefault * (embd + ffn) * 4 * 2 / 1048576
	if cpuActMB < 64 {
		cpuActMB = 64
	}
	return cpuActMB
}

// recommendAutoContext mirrors placement.computeAutoContextSize /
// computeAutoContextSizeSingleGPU at the recommender's level of detail: pick
// the largest power-of-two context (>= 32k, <= train ctx) whose KV fits in the
// memory left after model weights + headroom on this user's hardware.
//
// Crucially it matches the launcher's MoE behaviour: when the model spans
// VRAM+RAM (model > total VRAM, the MoE RAM+VRAM split case), the weights
// consume the memory and the launcher stays at 32k — so the recommender does
// too. Context only scales up when the model fits in VRAM with room to spare,
// exactly as the launcher's single-GPU / multi-GPU-dense paths do.
func recommendAutoContext(caps *detect.Capabilities, c Candidate, modelMB int) int {
	if caps == nil || len(caps.GPUs) == 0 {
		return recommendCtxMin
	}
	totalVRAM := 0
	bestVRAM := 0
	for _, g := range caps.GPUs {
		totalVRAM += g.VRAMTotalMB
		if g.VRAMTotalMB > bestVRAM {
			bestVRAM = g.VRAMTotalMB
		}
	}
	// MoE RAM+VRAM split: model is larger than total VRAM, so weights fill the
	// VRAM and spill into RAM. The launcher stays at 32k here (weights dominate,
	// no room for bigger context). Match that.
	if c.MoE && modelMB > totalVRAM {
		return recommendCtxMin
	}
	// Model fits in VRAM (dense, or small MoE): context can scale with the
	// headroom. Use best-GPU VRAM + a small RAM allowance, mirroring the
	// launcher's single-GPU path (the fastest mode it prefers).
	ramAllowance := 4096
	if caps.RAM.FreeMB > 0 && caps.RAM.FreeMB < ramAllowance {
		ramAllowance = caps.RAM.FreeMB
	}
	totalHWMB := bestVRAM + ramAllowance
	fixedOverheadMB := modelMB + 8192
	if totalHWMB <= fixedOverheadMB {
		return recommendCtxMin
	}
	kvBudgetMB := totalHWMB - fixedOverheadMB
	if kvBudgetMB <= 0 {
		return recommendCtxMin
	}
	refKV := recommendKVMB(c, recommendCtxMin, "q4_0")
	if refKV <= 0 {
		return recommendCtxMin
	}
	kvBytesPerToken := float64(refKV) * 1048576.0 / float64(recommendCtxMin)
	maxCtxRaw := int(float64(kvBudgetMB) * 1048576.0 / kvBytesPerToken)
	hwCapCtx := maxCtxRaw
	if c.TrainCtx > 0 && c.TrainCtx < hwCapCtx {
		hwCapCtx = c.TrainCtx
	}
	for _, p := range []int{1048576, 524288, 262144, 131072, 65536, 32768} {
		if p <= hwCapCtx {
			return p
		}
	}
	return recommendCtxMin
}

// recommendKVMB computes the KV cache size in MB for a candidate at a given
// context and KV type, using the same per-element bytes and GQA/MLA structure
// as placement.computeKVTotalMB. Exported geometry fields drive it; missing
// geometry returns 0 (caller then falls back to the size-based estimate).
func recommendKVMB(c Candidate, ctxSize int, kvType string) int {
	if c.Layers <= 0 || c.HeadCountKV <= 0 {
		return 0
	}
	var kvElemsTotal int
	if c.KVLoraRank > 0 {
		// MLA: compressed c^{KV} + RoPE'd key once per layer. RopeDim is not in
		// the recommender's geometry set; use KeyLength as a conservative
		// stand-in (RoPE dim <= key length for current MLA models).
		ropeDim := c.KeyLength
		if ropeDim <= 0 {
			ropeDim = 64
		}
		kvElemsTotal = c.Layers * ctxSize * (c.KVLoraRank + ropeDim)
	} else {
		// Standard GQA/MQA
		kvBytesPerLayerPerToken := c.HeadCountKV * (c.KeyLength + c.ValueLength)
		kvElemsTotal = c.Layers * ctxSize * kvBytesPerLayerPerToken
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
		bytesPerElem = 1.0625
	}
	return int(float64(kvElemsTotal) * bytesPerElem / 1024 / 1024)
}

func quantPenalty(q QuantOption) float64 {
	if q.QualityPenalty > 0 {
		return q.QualityPenalty
	}
	name := strings.ToUpper(q.Name)
	switch {
	case strings.Contains(name, "BF16") || strings.Contains(name, "F16") || strings.Contains(name, "F32"):
		return 0
	case strings.Contains(name, "Q8"):
		return 0.4
	case strings.Contains(name, "Q6") || strings.Contains(name, "IQ6"):
		return 0.8
	case strings.Contains(name, "Q5") || strings.Contains(name, "IQ5"):
		return 1.5
	case strings.Contains(name, "MXFP4") || strings.Contains(name, "MXP4"):
		return 2.6
	case strings.Contains(name, "Q4") || strings.Contains(name, "IQ4") || strings.Contains(name, "I4"):
		return 3.0
	case strings.Contains(name, "Q3") || strings.Contains(name, "IQ3"):
		return 5.0
	case strings.Contains(name, "Q2") || strings.Contains(name, "IQ2"):
		return 8.0
	case strings.Contains(name, "Q1") || strings.Contains(name, "IQ1"):
		return 12.0
	default:
		return 3.5
	}
}

func modelIntelligence(c Candidate) float64 {
	if c.AAIntelligence > 0 {
		return c.AAIntelligence
	}
	if c.Quality > 0 {
		return float64(c.Quality) / 1.65
	}
	return 0
}

func intelligenceScore(c Candidate) int {
	if c.Quality > 0 {
		return c.Quality
	}
	if c.AAIntelligence > 0 {
		score := int(c.AAIntelligence*1.65 + 0.5)
		if score < 1 {
			return 1
		}
		if score > 100 {
			return 100
		}
		return score
	}
	return 0
}

func backendHint(caps *detect.Capabilities) string {
	if caps == nil || len(caps.GPUs) == 0 {
		return "CPU"
	}
	if hasBackend(caps, "ik") || hasBackend(caps, "cuda") {
		return "CUDA / ik_llama"
	}
	if hasBackend(caps, "vulkan") {
		return "Vulkan fast path"
	}
	if strings.EqualFold(caps.OS, "linux") {
		return "Vulkan fallback on Linux"
	}
	return "GPU backend"
}

func hasBackend(caps *detect.Capabilities, needle string) bool {
	needle = strings.ToLower(needle)
	for _, b := range caps.Backends {
		if strings.Contains(strings.ToLower(b.Name), needle) || strings.Contains(strings.ToLower(b.Path), needle) {
			return true
		}
	}
	return false
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func fallbackShortlist() []Candidate {
	return []Candidate{
		{Name: "Qwen3.6 27B Instruct", Repo: "unsloth/Qwen3.6-27B-GGUF", Family: "Qwen", SizeGB: 18.5, Quality: 84, Speed: 48, Notes: "current dense Qwen quality target for capable local machines"},
		{Name: "Llama 3.2 3B Instruct", Repo: "bartowski/Llama-3.2-3B-Instruct-GGUF", Family: "Llama", SizeGB: 2.2, Quality: 58, Speed: 98, Notes: "very small fallback for laptops and CPU installs"},
		{Name: "Llama 3.1 8B Instruct", Repo: "MaziyarPanahi/Meta-Llama-3.1-8B-Instruct-GGUF", Family: "Llama", SizeGB: 5.0, Quality: 70, Speed: 82, Notes: "balanced quality and speed on 8-12GB GPUs"},
	}
}
