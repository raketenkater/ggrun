package recommend

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestShortlistLoadsEmbeddedCatalog(t *testing.T) {
	rows := Shortlist()
	if len(rows) < 3 {
		t.Fatalf("expected embedded catalog rows, got %d", len(rows))
	}
	for _, row := range rows {
		if row.Name == "" || row.Repo == "" || row.SizeGB <= 0 {
			t.Fatalf("invalid catalog row: %#v", row)
		}
	}
}

func TestTopFiltersByHardwareFit(t *testing.T) {
	caps := &detect.Capabilities{
		OS:       "linux",
		RAM:      detect.RAMInfo{TotalMB: 32768, FreeMB: 28000},
		CPU:      detect.CPUInfo{Cores: 8},
		GPUs:     []detect.GPU{{Name: "Test GPU", VRAMTotalMB: 12288}},
		Backends: []detect.Backend{{Name: "llama-server", Path: "/bin/llama-server-vulkan"}},
	}
	rows := Top(caps, 3)
	if len(rows) != 3 {
		t.Fatalf("expected three recommendations, got %d", len(rows))
	}
	for _, row := range rows {
		if row.Repo == "" || row.Fit == "" || row.BackendHint == "" || row.QuantName == "" {
			t.Fatalf("incomplete recommendation: %#v", row)
		}
	}
}

func TestCPURecommendationsStaySmall(t *testing.T) {
	caps := &detect.Capabilities{
		OS:  "linux",
		RAM: detect.RAMInfo{TotalMB: 16384, FreeMB: 14000},
		CPU: detect.CPUInfo{Cores: 8},
	}
	rows := Top(caps, 3)
	if len(rows) == 0 {
		t.Fatal("expected at least one CPU recommendation")
	}
	for _, row := range rows {
		if row.QuantSizeGB > 15 {
			t.Fatalf("CPU recommendation quant is too large: %#v", row)
		}
	}
}

func TestRecommendationScoreIgnoresSpeed(t *testing.T) {
	caps := &detect.Capabilities{
		OS:       "linux",
		RAM:      detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		GPUs:     []detect.GPU{{Name: "Test GPU", VRAMTotalMB: 24576}},
		Backends: []detect.Backend{{Name: "llama-server", Path: "/bin/llama-server-vulkan"}},
	}
	fastButWeak, ok := evaluate(caps, Candidate{Name: "fast", Repo: "repo/fast", SizeGB: 4, Quality: 20, Speed: 100})
	if !ok {
		t.Fatal("expected fast candidate to fit")
	}
	slowButSmart, ok := evaluate(caps, Candidate{Name: "smart", Repo: "repo/smart", SizeGB: 4, Quality: 80, Speed: 1})
	if !ok {
		t.Fatal("expected smart candidate to fit")
	}
	if slowButSmart.Score <= fastButWeak.Score {
		t.Fatalf("expected intelligence score to win regardless of speed: smart=%d fast=%d", slowButSmart.Score, fastButWeak.Score)
	}
}

func TestEvaluateChoosesBestFittingQuant(t *testing.T) {
	// 16GB total RAM + 24GB VRAM: Q8_0 (30GB) fits via GPU+RAM split.
	// The recommender uses total hardware capacity (not current FreeMB),
	// so Q8_0 — the highest-quality quant that fits — wins over Q4_K_M.
	caps := &detect.Capabilities{
		OS:       "linux",
		RAM:      detect.RAMInfo{TotalMB: 16384, FreeMB: 8192},
		GPUs:     []detect.GPU{{Name: "Test GPU", VRAMTotalMB: 24576}},
		Backends: []detect.Backend{{Name: "llama-server", Path: "/bin/llama-server-vulkan"}},
	}
	candidate := Candidate{
		Name:           "quant target",
		Repo:           "repo/quant-target",
		AAIntelligence: 50,
		Quants: []QuantOption{
			{Name: "Q2_K", SizeGB: 10},
			{Name: "Q4_K_M", SizeGB: 18},
			{Name: "Q8_0", SizeGB: 30},
		},
	}
	rec, ok := evaluate(caps, candidate)
	if !ok {
		t.Fatal("expected candidate to fit")
	}
	if rec.QuantName != "Q8_0" || rec.Fit != "GPU plus RAM" {
		t.Fatalf("expected Q8_0 via GPU+RAM (highest quality that fits), got %#v", rec)
	}
}

func TestCatalogPrefersValidCacheOverEmbedded(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("LLM_CACHE_DIR", dir)

	// No cache yet → falls back to embedded (which has many models).
	if got := Shortlist(); len(got) < 3 {
		t.Fatalf("embedded fallback should yield models, got %d", len(got))
	}

	// A valid cached catalog takes precedence.
	cached := `{"version":99,"generated_at":"2099-01-01T00:00:00Z","candidates":[` +
		strings_repeatModel(minModels) + `]}`
	if err := os.WriteFile(filepath.Join(dir, "catalog.json"), []byte(cached), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := Shortlist(); len(got) != minModels || got[0].Repo != "test/model-0" {
		t.Fatalf("expected cached catalog to win, got %d rows first=%v", len(got), got[0].Repo)
	}

	// A truncated/garbage cache is rejected; embedded is used instead.
	if err := os.WriteFile(filepath.Join(dir, "catalog.json"), []byte(`{"candidates":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := Shortlist(); len(got) < 3 || got[0].Repo == "test/model-0" {
		t.Fatalf("invalid cache should be ignored in favour of embedded, got %d", len(got))
	}
}

func strings_repeatModel(n int) string {
	parts := make([]string, n)
	for i := range parts {
		parts[i] = `{"name":"M","repo":"test/model-` + itoa(i) + `","size_gb":4,"aa_intelligence_index":10,"quants":[{"name":"Q4_K_M","size_gb":4}]}`
	}
	return join(parts, ",")
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	s := ""
	for i > 0 {
		s = string(rune('0'+i%10)) + s
		i /= 10
	}
	return s
}

func join(parts []string, sep string) string {
	out := ""
	for i, p := range parts {
		if i > 0 {
			out += sep
		}
		out += p
	}
	return out
}

func TestTopCategoriesDedupeAndPopulate(t *testing.T) {
	caps := &detect.Capabilities{
		OS:  "linux",
		RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 120000},
		GPUs: []detect.GPU{
			{Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24576},
			{Name: "NVIDIA GeForce RTX 3060", VRAMTotalMB: 12288},
		},
		Backends: []detect.Backend{{Name: "cuda", Path: "/bin/llama-server-cuda"}},
	}
	cats := TopCategories(caps, 4)
	if len(cats.Balanced) == 0 {
		t.Fatal("expected balanced recommendations")
	}
	// Each category dedups within itself (one quant per model); categories MAY
	// overlap — a model can legitimately be both the best overall blend and the
	// smartest that fits. (Cross-category dedup used to surface a leftover
	// low-intelligence pick in Smartest, so it was removed.)
	for _, group := range [][]Recommendation{cats.Balanced, cats.Smartest, cats.Fastest} {
		if len(group) > 4 {
			t.Fatalf("category exceeded requested size: %d", len(group))
		}
		seen := map[string]bool{}
		for _, r := range group {
			if seen[r.Repo] {
				t.Fatalf("model %s appears twice in the same category", r.Repo)
			}
			seen[r.Repo] = true
		}
	}
}

func TestPlausibilityGuardDropsPhantomQuant(t *testing.T) {
	caps := &detect.Capabilities{
		OS:       "linux",
		RAM:      detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		GPUs:     []detect.GPU{{Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24576}},
		Backends: []detect.Backend{{Name: "cuda", Path: "/bin/llama-server-cuda"}},
	}
	// Mirrors the real catalog bug: a 30B model with a phantom F16=0.9GB row from
	// incomplete Hugging Face shard metadata. It must never be recommended.
	c := Candidate{
		Name: "phantom", Repo: "unsloth/Phantom-30B-GGUF", AAIntelligence: 40, TotalParamsB: 30,
		Quants: []QuantOption{
			{Name: "F16", SizeGB: 0.9},
			{Name: "Q4_K_M", SizeGB: 17},
		},
	}
	rec, ok := evaluate(caps, c)
	if !ok {
		t.Fatal("expected candidate to fit on a real quant")
	}
	if rec.QuantName == "F16" {
		t.Fatalf("phantom F16=0.9GB quant should have been rejected, got %#v", rec)
	}
	if rec.QuantName != "Q4_K_M" {
		t.Fatalf("expected the plausible Q4_K_M quant, got %q", rec.QuantName)
	}
}

func TestLargeModelsTolerateQuantizationBetter(t *testing.T) {
	small := quantQualityRetention("IQ3_XXS", 4, false)
	large := quantQualityRetention("IQ3_XXS", 120, false)
	if large <= small {
		t.Fatalf("expected a 120B to retain more quality at IQ3 than a 4B: large=%.3f small=%.3f", large, small)
	}
	// Unsloth dynamic quants should beat a generic quant of the same model.
	generic := quantQualityRetention("IQ4_XS", 30, false)
	dynamic := quantQualityRetention("IQ4_XS", 30, true)
	if dynamic <= generic {
		t.Fatalf("expected dynamic quant to retain more: dynamic=%.3f generic=%.3f", dynamic, generic)
	}
}

func TestFastCapableModelOutranksSlowGiant(t *testing.T) {
	// Modest single-GPU rig: a giant only fits by spilling to RAM (slow), a mid
	// model runs fully in VRAM (fast). The fast capable model should rank first.
	caps := &detect.Capabilities{
		OS:       "linux",
		RAM:      detect.RAMInfo{TotalMB: 196608, FreeMB: 184000},
		GPUs:     []detect.GPU{{Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12288}},
		Backends: []detect.Backend{{Name: "cuda", Path: "/bin/llama-server-cuda"}},
	}
	fastMid := Candidate{Name: "fast-mid", Repo: "x/fast", AAIntelligence: 30, TotalParamsB: 9,
		Quants: []QuantOption{{Name: "Q5_K_M", SizeGB: 6.5}}}
	slowGiant := Candidate{Name: "slow-giant", Repo: "x/giant", AAIntelligence: 34, MoE: true,
		TotalParamsB: 200, ActiveParamsB: 30, Quants: []QuantOption{{Name: "Q4_K_M", SizeGB: 95}}}
	fr, ok1 := evaluate(caps, fastMid)
	gr, ok2 := evaluate(caps, slowGiant)
	if !ok1 || !ok2 {
		t.Fatalf("expected both to fit: fast=%v giant=%v", ok1, ok2)
	}
	if fr.PredictedTPS <= gr.PredictedTPS {
		t.Fatalf("expected fast-mid to be predicted faster: fast=%.1f giant=%.1f", fr.PredictedTPS, gr.PredictedTPS)
	}
	if fr.Score <= gr.Score {
		t.Fatalf("expected fast capable model to outrank slow giant: fast=%d giant=%d", fr.Score, gr.Score)
	}
}

func TestEvaluateMoEFitsAcrossRAMAndVRAM(t *testing.T) {
	caps := &detect.Capabilities{
		OS:       "linux",
		RAM:      detect.RAMInfo{TotalMB: 196608, FreeMB: 180224},
		GPUs:     []detect.GPU{{Name: "Test GPU", VRAMTotalMB: 24576}},
		Backends: []detect.Backend{{Name: "llama-server", Path: "/bin/llama-server-vulkan"}},
	}
	candidate := Candidate{
		Name:           "moe target",
		Repo:           "repo/moe-target",
		AAIntelligence: 50,
		MoE:            true,
		TotalParamsB:   230,
		ActiveParamsB:  10,
		Quants:         []QuantOption{{Name: "Q4_K_M", SizeGB: 127}},
	}
	rec, ok := evaluate(caps, candidate)
	if !ok {
		t.Fatal("expected MoE candidate to fit across RAM and VRAM")
	}
	if rec.Fit != "MoE RAM+VRAM" {
		t.Fatalf("expected MoE RAM+VRAM fit, got %#v", rec)
	}
}
