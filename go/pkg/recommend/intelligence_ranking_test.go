package recommend

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func intelligenceTestHardware() *detect.Capabilities {
	return &detect.Capabilities{
		OS: "linux", RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		GPUs:     []detect.GPU{{Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24576}},
		Backends: []detect.Backend{{Name: "cuda", Path: "/fixture/llama-server"}},
	}
}

func intelligenceTestCatalog(t *testing.T, candidates ...Candidate) {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("LLM_CACHE_DIR", dir)
	for len(candidates) < minModels {
		candidates = append(candidates, Candidate{Name: "filler", Repo: fmt.Sprintf("fixture/filler-%d", len(candidates)),
			AAIntelligence: 1, TotalParamsB: 4, SizeGB: 2.6, Quants: []QuantOption{{Name: "Q4_K_M", SizeGB: 2.6}}})
	}
	b, err := json.Marshal(catalogDoc{Version: 99, GeneratedAt: "2099-01-01T00:00:00Z", Candidates: candidates})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "catalog.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestModelIntelligenceOutranksQuantRetentionGuess(t *testing.T) {
	strong := Candidate{Name: "strong", Repo: "fixture/strong", AAIntelligence: 55, TotalParamsB: 30,
		SizeGB: 14, Quants: []QuantOption{{Name: "IQ3_XXS", SizeGB: 14}}}
	weak := Candidate{Name: "weak", Repo: "fixture/weak", AAIntelligence: 50, TotalParamsB: 8,
		SizeGB: 8.5, Quants: []QuantOption{{Name: "Q8_0", SizeGB: 8.5}}}
	intelligenceTestCatalog(t, strong, weak)
	caps := intelligenceTestHardware()
	a, _ := evaluate(caps, strong)
	b, _ := evaluate(caps, weak)
	if a.PredictedTPS < usableTPS || b.PredictedTPS < usableTPS || a.AdjustedIntelligence >= b.AdjustedIntelligence {
		t.Fatalf("fixture no longer reproduces the misleading quant ordering: strong=%+v weak=%+v", a, b)
	}
	cats := TopCategories(caps, 2)
	for name, rows := range map[string][]Recommendation{"overall": cats.Balanced, "smartest": cats.Smartest, "top": Top(caps, 2)} {
		if len(rows) != 2 || rows[0].Repo != strong.Repo || rows[1].Repo != weak.Repo {
			t.Fatalf("%s ranked an unmeasured quant-retention guess above model intelligence: %+v", name, rows)
		}
	}
}

func TestUsableSpeedDoesNotOutweighModelIntelligence(t *testing.T) {
	// The stronger dense model remains usable on this CPU budget, while the
	// weaker small model is much faster. Both clear the same usability floor.
	caps := &detect.Capabilities{OS: "linux", RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU: detect.CPUInfo{Cores: 16}, HostMemoryBandwidthMBps: 60000}
	strong := Candidate{Name: "strong", Repo: "fixture/strong", AAIntelligence: 55, TotalParamsB: 8,
		Quants: []QuantOption{{Name: "Q4_K_M", SizeGB: 4.8}}}
	weak := Candidate{Name: "weak", Repo: "fixture/weak", AAIntelligence: 50, TotalParamsB: 4,
		Quants: []QuantOption{{Name: "Q4_K_M", SizeGB: 2.6}}}
	a, okA := evaluate(caps, strong)
	b, okB := evaluate(caps, weak)
	if !okA || !okB || a.PredictedTPS < usableTPS || a.PredictedTPS >= interactiveTPS || b.PredictedTPS <= a.PredictedTPS {
		t.Fatalf("fixture does not span usable/fast speeds: strong=%+v weak=%+v", a, b)
	}
	if a.Score <= b.Score {
		t.Fatalf("extra speed overrode usable model intelligence: strong=%d weak=%d", a.Score, b.Score)
	}
}

func TestCategoriesChoosePracticalQuantBeforeRankingModel(t *testing.T) {
	c := Candidate{Name: "strong dense", Repo: "fixture/dense", AAIntelligence: 55, TotalParamsB: 27, SizeGB: 19,
		Quants: []QuantOption{{Name: "Q4_K_M", SizeGB: 16}, {Name: "Q5_K_M", SizeGB: 19}, {Name: "BF16", SizeGB: 55}}}
	intelligenceTestCatalog(t, c)
	cats := TopCategories(intelligenceTestHardware(), 1)
	for name, rows := range map[string][]Recommendation{"overall": cats.Balanced, "smartest": cats.Smartest} {
		if len(rows) != 1 || rows[0].Repo != c.Repo || rows[0].QuantName == "BF16" || rows[0].PredictedTPS < usableTPS {
			t.Fatalf("%s ranked the model using an unnecessarily slow quant: %+v", name, rows)
		}
	}
}

func TestLowBitPreferenceDoesNotOverrideCrossModelIntelligence(t *testing.T) {
	for _, quant := range []QuantOption{{Name: "IQ1_S", SizeGB: 2}, {Name: "UD-Q2_K_XL", SizeGB: 2.9}} {
		t.Run(quant.Name, func(t *testing.T) {
			strong := Candidate{Name: "low bit", Repo: "fixture/low-bit", AAIntelligence: 90, TotalParamsB: 8,
				SizeGB: quant.SizeGB, Quants: []QuantOption{quant}}
			regular := Candidate{Name: "regular", Repo: "fixture/regular", AAIntelligence: 50, TotalParamsB: 8, SizeGB: 4.8,
				Quants: []QuantOption{{Name: "Q4_K_M", SizeGB: 4.8}}}
			intelligenceTestCatalog(t, strong, regular)
			caps := intelligenceTestHardware()
			cats := TopCategories(caps, 2)
			for name, rows := range map[string][]Recommendation{"overall": cats.Balanced, "smartest": cats.Smartest, "top": Top(caps, 2)} {
				if len(rows) != 2 || rows[0].Repo != strong.Repo || rows[1].Repo != regular.Repo {
					t.Fatalf("%s used quant class to override model capability: %+v", name, rows)
				}
			}
		})
	}
}

func TestPracticalQuantPrefersUsableRegularVariant(t *testing.T) {
	// Both variants are usable. A low-bit speed advantage must not default to
	// aggressive compression when this same model has a usable ordinary quant.
	c := Candidate{Name: "dense", Repo: "fixture/dense", AAIntelligence: 55, TotalParamsB: 27,
		Quants: []QuantOption{{Name: "IQ2_XXS", SizeGB: 8}, {Name: "Q4_K_M", SizeGB: 16}}}
	r, ok := evaluateWithSelector(intelligenceTestHardware(), c, nil, betterByScore)
	if !ok || r.QuantName != "Q4_K_M" || r.PredictedTPS < usableTPS {
		t.Fatalf("did not choose the usable ordinary quant: %+v", r)
	}
}

func TestPracticalQuantAllowsLowBitWhenRegularVariantIsSlow(t *testing.T) {
	caps := intelligenceTestHardware()
	caps.GPUs[0].VRAMTotalMB = 12288
	c := Candidate{Name: "dense", Repo: "fixture/dense", AAIntelligence: 55, TotalParamsB: 27,
		Quants: []QuantOption{{Name: "Q2_K", SizeGB: 9.5}, {Name: "Q8_0", SizeGB: 29}}}
	quality, ok := evaluate(caps, c)
	if !ok || quality.QuantName != "Q8_0" || quality.PredictedTPS >= usableTPS {
		t.Fatalf("fixture no longer has a slow ordinary quant: %+v", quality)
	}
	r, ok := evaluateWithSelector(caps, c, nil, betterByScore)
	if !ok || r.QuantName != "Q2_K" || r.PredictedTPS < usableTPS {
		t.Fatalf("a merely fitting slow ordinary quant blocked a usable fallback: %+v", r)
	}
}

func TestDisplayedIntelligenceIsIndependentOfQuantGuess(t *testing.T) {
	r := Recommendation{Candidate: Candidate{AAIntelligence: 55.2}, QualityRetained: .55, AdjustedIntelligence: 30.36}
	if got := DisplayIntelligence(r); got != "55.2" {
		t.Fatalf("display presented estimated quant accuracy as benchmark: %q", got)
	}
	if got := DisplayIntelligence(Recommendation{Candidate: Candidate{Quality: 66}}); got != "~40.0" {
		t.Fatalf("fallback intelligence estimate was not labelled: %q", got)
	}
	if got := DisplayIntelligence(Recommendation{}); got != "—" {
		t.Fatalf("missing intelligence invented a score: %q", got)
	}
}
