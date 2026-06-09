package recommend

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/raketenkater/llm-server/pkg/detect"
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
}

// Candidate is a GGUF repository that llm-server can offer as a first-run
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
	AdjustedIntelligence float64
}

type catalogDoc struct {
	Version     int         `json:"version"`
	GeneratedAt string      `json:"generated_at"`
	Source      string      `json:"source"`
	Attribution string      `json:"attribution"`
	Candidates  []Candidate `json:"candidates"`
}

const Attribution = "Artificial Analysis intelligence data is used when available; cached locally and filtered by llm-server hardware fit"

func Shortlist() []Candidate {
	var doc catalogDoc
	if err := json.Unmarshal(catalogJSON, &doc); err == nil && len(doc.Candidates) > 0 {
		return doc.Candidates
	}
	return fallbackShortlist()
}

func CatalogAttribution() string {
	var doc catalogDoc
	if err := json.Unmarshal(catalogJSON, &doc); err == nil && doc.Attribution != "" {
		return doc.Attribution
	}
	return Attribution
}

func Top(caps *detect.Capabilities, limit int) []Recommendation {
	if limit <= 0 {
		limit = 5
	}
	var rows []Recommendation
	for _, c := range Shortlist() {
		if rec, ok := evaluate(caps, c); ok {
			rows = append(rows, rec)
		}
	}
	sortRecommendations(rows)
	if len(rows) > limit {
		rows = rows[:limit]
	}
	return rows
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
	usableRAM   int
	gpuCount    int
}

func hardware(caps *detect.Capabilities) hardwareBudget {
	budget := hardwareBudget{}
	totalRAM := 8192
	freeRAM := totalRAM
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
		if caps.RAM.FreeMB > 0 && caps.RAM.FreeMB <= totalRAM {
			freeRAM = caps.RAM.FreeMB
		} else {
			freeRAM = totalRAM
		}
	}
	reserve := 4096
	if totalRAM >= 65536 {
		reserve = 8192
	}
	budget.usableRAM = freeRAM - reserve
	if budget.usableRAM < 0 {
		budget.usableRAM = 0
	}
	return budget
}

func evaluate(caps *detect.Capabilities, c Candidate) (Recommendation, bool) {
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
		fit, reason, fitPenalty, needGB, fits := fitQuant(budget, c, q)
		if !fits {
			continue
		}
		adjusted := base - quantPenalty(q) - fitPenalty
		if adjusted <= 0 {
			continue
		}
		rec := Recommendation{
			Candidate:            c,
			Fit:                  fit,
			BackendHint:          backend,
			Reason:               reason,
			QuantName:            q.Name,
			QuantSizeGB:          q.SizeGB,
			MemoryNeedGB:         needGB,
			AdjustedIntelligence: adjusted,
			Score:                int(adjusted * 1000),
		}
		if !ok || better(rec, best) {
			best = rec
			ok = true
		}
	}
	return best, ok
}

func better(a, b Recommendation) bool {
	if a.Score != b.Score {
		return a.Score > b.Score
	}
	if a.AdjustedIntelligence != b.AdjustedIntelligence {
		return a.AdjustedIntelligence > b.AdjustedIntelligence
	}
	return a.QuantSizeGB > b.QuantSizeGB
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

func fitQuant(b hardwareBudget, c Candidate, q QuantOption) (fit, reason string, fitPenalty, needGB float64, ok bool) {
	modelMB := int(q.SizeGB * 1024)
	overheadMB := estimateOverheadMB(modelMB, c)
	needMB := modelMB + overheadMB
	needGB = float64(needMB) / 1024

	switch {
	case b.gpuCount > 0 && needMB <= b.largestVRAM:
		return "single GPU", fmt.Sprintf("%s fits in the largest GPU (%.1fGB model + ~%.1fGB overhead)", q.Name, q.SizeGB, float64(overheadMB)/1024), 0, needGB, true
	case b.gpuCount > 1 && needMB <= b.totalVRAM:
		return "multi-GPU", fmt.Sprintf("%s fits across %d GPUs (%.1fGB model + ~%.1fGB overhead)", q.Name, b.gpuCount, q.SizeGB, float64(overheadMB)/1024), 0.25, needGB, true
	case b.gpuCount > 0 && needMB <= b.totalVRAM+b.usableRAM && c.MoE:
		return "MoE RAM+VRAM", fmt.Sprintf("%s fits with GPU attention and RAM expert/offload path (%.1fGB model + ~%.1fGB overhead)", q.Name, q.SizeGB, float64(overheadMB)/1024), 1.0, needGB, true
	case b.gpuCount > 0 && needMB <= b.totalVRAM+b.usableRAM:
		return "GPU plus RAM", fmt.Sprintf("%s fits with partial GPU offload (%.1fGB model + ~%.1fGB overhead)", q.Name, q.SizeGB, float64(overheadMB)/1024), 2.0, needGB, true
	case b.gpuCount == 0 && needMB <= b.usableRAM:
		return "CPU RAM", fmt.Sprintf("%s fits in system RAM (%.1fGB model + ~%.1fGB overhead)", q.Name, q.SizeGB, float64(overheadMB)/1024), 4.0, needGB, true
	default:
		return "", "", 0, needGB, false
	}
}

func estimateOverheadMB(modelMB int, c Candidate) int {
	if modelMB <= 0 {
		return 2048
	}
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
