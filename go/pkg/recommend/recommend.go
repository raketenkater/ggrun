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

// Candidate is a GGUF repository that llm-server can offer as a first-run
// download. Scores are product signals refreshed from the checked-in catalog.
type Candidate struct {
	Name           string  `json:"name"`
	Repo           string  `json:"repo"`
	Family         string  `json:"family"`
	SizeGB         float64 `json:"size_gb"`
	Quality        int     `json:"quality"`
	Speed          int     `json:"speed"`
	MoE            bool    `json:"moe"`
	Notes          string  `json:"notes"`
	AAQuery        string  `json:"aa_query,omitempty"`
	AASlug         string  `json:"aa_slug,omitempty"`
	AAIntelligence float64 `json:"aa_intelligence_index,omitempty"`
	AAOutputTPS    float64 `json:"aa_output_tps,omitempty"`
	AAUpdatedAt    string  `json:"aa_updated_at,omitempty"`
}

// Recommendation is a candidate ranked for the current machine.
type Recommendation struct {
	Candidate
	Fit         string
	BackendHint string
	Reason      string
	Score       int
}

type catalogDoc struct {
	Version     int         `json:"version"`
	GeneratedAt string      `json:"generated_at"`
	Source      string      `json:"source"`
	Attribution string      `json:"attribution"`
	Candidates  []Candidate `json:"candidates"`
}

const Attribution = "Artificial Analysis leaderboard data is used when available; cached locally and filtered by llm-server hardware fit"

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
		limit = 3
	}
	var rows []Recommendation
	for _, c := range Shortlist() {
		if rec, ok := evaluate(caps, c); ok {
			rows = append(rows, rec)
		}
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Score == rows[j].Score {
			return rows[i].SizeGB < rows[j].SizeGB
		}
		return rows[i].Score > rows[j].Score
	})
	if len(rows) > limit {
		rows = rows[:limit]
	}
	return rows
}

func evaluate(caps *detect.Capabilities, c Candidate) (Recommendation, bool) {
	largestVRAM := 0
	totalVRAM := 0
	totalRAM := 0
	freeRAM := 0
	gpuCount := 0
	if caps != nil {
		gpuCount = len(caps.GPUs)
		for _, g := range caps.GPUs {
			totalVRAM += g.VRAMTotalMB
			if g.VRAMTotalMB > largestVRAM {
				largestVRAM = g.VRAMTotalMB
			}
		}
		totalRAM = caps.RAM.TotalMB
		freeRAM = caps.RAM.FreeMB
	}
	if totalRAM <= 0 {
		totalRAM = 8192
	}
	if freeRAM <= 0 || freeRAM > totalRAM {
		freeRAM = totalRAM
	}

	modelMB := int(c.SizeGB * 1024)
	overhead := 2048
	if c.MoE {
		overhead = 4096
	}
	needMB := modelMB + overhead
	backend := backendHint(caps)
	rec := Recommendation{Candidate: c, BackendHint: backend}

	fitTier := 0
	switch {
	case gpuCount > 0 && needMB <= largestVRAM:
		rec.Fit = "single GPU"
		rec.Reason = fmt.Sprintf("fits the largest GPU with about %.1fGB model size", c.SizeGB)
		fitTier = 5
	case gpuCount > 1 && needMB <= totalVRAM:
		rec.Fit = "multi-GPU"
		rec.Reason = fmt.Sprintf("fits across %d GPUs with tensor split", gpuCount)
		fitTier = 4
	case c.MoE && gpuCount > 0 && needMB <= totalVRAM+freeRAM:
		rec.Fit = "MoE CPU expert offload"
		rec.Reason = "fits with GPU attention and CPU expert fallback"
		fitTier = 3
	case gpuCount > 0 && needMB <= totalVRAM+freeRAM && c.SizeGB <= 48:
		rec.Fit = "GPU plus CPU offload"
		rec.Reason = "fits with partial GPU offload; slower than full VRAM fit"
		fitTier = 2
	case gpuCount == 0 && needMB <= freeRAM/2:
		rec.Fit = "CPU"
		rec.Reason = "fits in system RAM for CPU serving"
		fitTier = 1
	default:
		return Recommendation{}, false
	}

	score := c.Quality*10 + c.Speed + fitTier*80
	if fitTier <= 2 && c.SizeGB > 20 {
		score -= 120
	}
	if c.MoE && fitTier < 3 {
		score -= 80
	}
	rec.Score = score
	return rec, true
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

func fallbackShortlist() []Candidate {
	return []Candidate{
		{Name: "Qwen3.5 4B Instruct", Repo: "unsloth/Qwen3.5-4B-GGUF", Family: "Qwen", SizeGB: 2.8, Quality: 66, Speed: 95, Notes: "small, fast, strong default for first local runs"},
		{Name: "Llama 3.2 3B Instruct", Repo: "bartowski/Llama-3.2-3B-Instruct-GGUF", Family: "Llama", SizeGB: 2.2, Quality: 58, Speed: 98, Notes: "very small fallback for laptops and CPU installs"},
		{Name: "Llama 3.1 8B Instruct", Repo: "MaziyarPanahi/Meta-Llama-3.1-8B-Instruct-GGUF", Family: "Llama", SizeGB: 5.0, Quality: 70, Speed: 82, Notes: "balanced quality and speed on 8-12GB GPUs"},
	}
}
