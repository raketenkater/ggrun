package recommend

import (
	"math"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// The 2026-09-10 catalog refresh shipped every row without parameter counts.
// Nothing noticed for two weeks: speed prediction quietly returned 0, "Fastest"
// disappeared and 193 GB models led "Best overall" on a 48 GB VRAM machine.
// These run against the real shipped catalog so a refresh cannot do it again.
func TestShippedCatalogPredictsSpeed(t *testing.T) {
	caps := referenceRig()
	rows := Shortlist()
	predicted := 0
	for _, c := range rows {
		for _, q := range c.Quants {
			if predictDecodeTPS(caps, c, q) > 0 {
				predicted++
				break
			}
		}
	}
	if predicted < len(rows)*9/10 {
		t.Fatalf("speed predicted for %d of %d catalog models; a refresh dropped the inputs", predicted, len(rows))
	}
	cats := TopCategories(caps, 5)
	if len(cats.Fastest) == 0 {
		t.Fatal("Fastest is empty: speed no longer reaches the ranking")
	}
	for _, r := range cats.Balanced {
		if r.PredictedTPS <= 0 {
			t.Fatalf("Best overall ranks %s with no speed estimate", r.Name)
		}
	}
}

func TestInferParamsFromGeometryAndName(t *testing.T) {
	// MiniMax-M3's shipped geometry: 60 layers (3 dense), 128 experts, 4 used.
	c := inferParams(Candidate{
		Name: "MiniMax-M3", Layers: 60, LeadingDense: 3, Experts: 128, ExpertUsed: 4,
		ExpertFF: 3072, Embedding: 6144, FeedForward: 12288, HeadCountKV: 4, KeyLength: 128,
		Quants: []QuantOption{{Name: "UD-IQ4_XS", SizeGB: 193.3}, {Name: "Q8_0", SizeGB: 437}},
	})
	if !c.MoE || c.ActiveParamsB < 15 || c.ActiveParamsB > 25 || c.TotalParamsB < 350 {
		t.Fatalf("MiniMax-M3 inferred moe=%v total=%.1f active=%.1f", c.MoE, c.TotalParamsB, c.ActiveParamsB)
	}
	// A vendor's own counts in the name beat any formula.
	n := inferParams(Candidate{Name: "NVIDIA Nemotron 3 Ultra 550B A55B (Reasoning)", Layers: 88, Experts: 512, ExpertUsed: 22, ExpertFF: 2688, Embedding: 8192})
	if n.TotalParamsB != 550 || n.ActiveParamsB != 55 || !n.MoE {
		t.Fatalf("name counts ignored: total=%.1f active=%.1f", n.TotalParamsB, n.ActiveParamsB)
	}
	// Explicit catalog values are kept.
	e := inferParams(Candidate{Name: "X 30B A3B", TotalParamsB: 31, ActiveParamsB: 3.3, MoE: true})
	if e.TotalParamsB != 31 || e.ActiveParamsB != 3.3 {
		t.Fatalf("explicit values overwritten: %+v", e)
	}
	// Geometry that implies reading most weights per token is not trusted.
	h := inferParams(Candidate{Name: "hybrid", Layers: 10, Experts: 4, ExpertUsed: 3, ExpertFF: 4096, Embedding: 4096,
		Quants: []QuantOption{{Name: "Q8_0", SizeGB: 2}}})
	if h.ActiveParamsB != 0 {
		t.Fatalf("implausible active count accepted: %.1f of %.1f", h.ActiveParamsB, h.TotalParamsB)
	}
	if got := paramsFromQuantSizes([]QuantOption{{Name: "Q4_K_M", SizeGB: 5.6}, {Name: "Q8_0", SizeGB: 10.6}, {Name: "BF16", SizeGB: 20}}); math.Abs(got-10) > 0.5 {
		t.Fatalf("median params from sizes = %.2f, want ~10", got)
	}
}

func referenceRig() *detect.Capabilities {
	return &detect.Capabilities{
		OS:  "linux",
		RAM: detect.RAMInfo{TotalMB: 217088, FreeMB: 200000},
		CPU: detect.CPUInfo{Cores: 14},
		GPUs: []detect.GPU{
			{Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12282},
			{Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24564},
			{Name: "NVIDIA GeForce RTX 3060", VRAMTotalMB: 12288},
		},
		Backends: []detect.Backend{{Name: "cuda", Path: "/bin/llama-server"}},
	}
}
