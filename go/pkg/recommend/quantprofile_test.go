package recommend

import (
	"math"
	"testing"

	"github.com/raketenkater/ggrun/pkg/gguf"
)

func TestQuantProfileFromCatalogSize(t *testing.T) {
	// A "BF16" upload of a model whose experts are stored at 4 bits: the
	// size over the parameter count shows ~4.5 bits per weight, not 16.
	c := Candidate{Name: "big", Repo: "fixture/big", Revision: "abc", TotalParamsB: 2779.662}
	q := QuantOption{Name: "BF16", SizeBytes: 1561525956960, SizeGB: 1454.28}
	p := quantProfileFor(c, q)
	if p.Source != QuantSourceSize || math.Abs(p.BPW-4.494) > 0.01 {
		t.Fatalf("profile %+v", p)
	}
	if p.Imatrix != "" || p.FileKey != "fixture/big@abc/BF16:1561525956960" {
		t.Fatalf("profile %+v", p)
	}
	// Without size_bytes, size_gb is GiB.
	q.SizeBytes = 0
	if p := quantProfileFor(c, q); math.Abs(p.BPW-4.494) > 0.01 {
		t.Fatalf("GiB fallback %+v", p)
	}
}

func TestQuantProfileNominalWhenParamsCameFromSizes(t *testing.T) {
	c := inferParams(Candidate{Name: "unnamed", Repo: "fixture/x", Quants: []QuantOption{
		{Name: "Q4_K_M", SizeGB: 5.6}, {Name: "Q8_0", SizeGB: 10.6}, {Name: "BF16", SizeGB: 20}}})
	if !c.paramsFromSizes || c.TotalParamsB <= 0 {
		t.Fatalf("params %+v", c)
	}
	if p := quantProfileFor(c, c.Quants[0]); p.Source != QuantSourceNominal || p.BPW <= 0 {
		t.Fatalf("profile %+v", p)
	}
	named := inferParams(Candidate{Name: "X 30B A3B", Quants: c.Quants})
	if named.paramsFromSizes {
		t.Fatal("a count from the model name is not size-derived")
	}
}

func TestQuantProfileRejectsImplausibleWidthAndMissingParams(t *testing.T) {
	c := Candidate{Repo: "fixture/x", TotalParamsB: 30}
	if p := quantProfileFor(c, QuantOption{Name: "Q4_0", SizeBytes: 1 << 30}); p.BPW != 0 || p.Source != "" {
		t.Fatalf("projector-sized file got a width: %+v", p)
	}
	if p := quantProfileFor(Candidate{Repo: "fixture/x"}, QuantOption{Name: "Q4_0", SizeBytes: 1 << 34}); p.BPW != 0 {
		t.Fatalf("no params, no width: %+v", p)
	}
}

func TestQuantProfilePrefersTheLocalHeader(t *testing.T) {
	t.Cleanup(func() { SetLocalQuantProfile(nil) })
	SetLocalQuantProfile(func(c Candidate, q QuantOption) (gguf.QuantProfile, bool) {
		if q.Name != "Q2_K" {
			return gguf.QuantProfile{}, false
		}
		return gguf.QuantProfile{Params: 1000, Bytes: 350, ExpertParams: 800, ExpertBytes: 250, Imatrix: gguf.ImatrixKeys}, true
	})
	c := Candidate{Repo: "fixture/x", TotalParamsB: 30}
	p := quantProfileFor(c, QuantOption{Name: "Q2_K", SizeBytes: 12e9})
	if p.Source != QuantSourceHeader || math.Abs(p.BPW-2.8) > 1e-9 || math.Abs(p.ExpertBPW-2.5) > 1e-9 ||
		math.Abs(p.NonExpertBPW-4) > 1e-9 || p.Imatrix != gguf.ImatrixKeys {
		t.Fatalf("profile %+v", p)
	}
	if p := quantProfileFor(c, QuantOption{Name: "Q4_K_M", SizeBytes: 18e9}); p.Source != QuantSourceSize {
		t.Fatalf("a file not on disk falls back to its size: %+v", p)
	}
}

func TestImatrixFromName(t *testing.T) {
	for name, want := range map[string]string{
		"UD-Q4_K_XL": ImatrixDynamic, "UD-IQ2_XXS": ImatrixDynamic,
		"IQ2_XXS": ImatrixName, "IQ1_S": ImatrixName, "IQ2_M": ImatrixName, "Qwen_IQ3_XXS": ImatrixName,
		"Q2_K": "", "IQ4_XS": "", "IQ3_S": "", "Q4_K_M": "", "BF16": "",
	} {
		if got := imatrixFromName(QuantOption{Name: name}); got != want {
			t.Errorf("%s: %q, want %q", name, got, want)
		}
		if got := (QuantProfile{Imatrix: imatrixFromName(QuantOption{Name: name})}).Imatrixed(); got != (want != "") {
			t.Errorf("%s: Imatrixed %v", name, got)
		}
	}
	if got := imatrixFromName(QuantOption{Name: "Q3_K_XL", Dynamic: true}); got != ImatrixDynamic {
		t.Errorf("dynamic flag: %q", got)
	}
}

func TestEvaluateAttachesTheChosenFilesProfile(t *testing.T) {
	c := Candidate{Name: "fixture 30B", Repo: "fixture/p", AAIntelligence: 40, TotalParamsB: 30,
		Quants: []QuantOption{{Name: "IQ4_XS", SizeGB: 15.2, SizeBytes: 16320875724}}}
	intelligenceTestCatalog(t, c)
	rec, ok := evaluate(intelligenceTestHardware(), c)
	if !ok {
		t.Fatal("fixture does not fit")
	}
	if rec.QuantProfile.Source != QuantSourceSize || math.Abs(rec.QuantProfile.BPW-4.352) > 0.01 {
		t.Fatalf("profile %+v", rec.QuantProfile)
	}
}

// A MoE known only by its size gets an expert width from its geometry.
func TestQuantProfileEstimatesExpertWidthFromGeometry(t *testing.T) {
	c := Candidate{Repo: "fixture/moe", MoE: true, TotalParamsB: 100, Layers: 10, Experts: 64, ExpertUsed: 4,
		ExpertFF: 1024, Embedding: 4096, HeadCountKV: 8, KeyLength: 128}
	_, share, ok := moeRoutedShares(c)
	if !ok {
		t.Fatal("fixture geometry incomplete")
	}
	// A file whose experts are 2.5 bpw and backbone 8.5.
	overall := share*2.5 + (1-share)*assumedBackboneBPW
	size := int64(overall * c.TotalParamsB * 1e9 / 8)
	p := quantProfileFor(c, QuantOption{Name: "IQ2_XS", SizeBytes: size})
	if !p.ExpertEstimated || math.Abs(p.ExpertBPW-2.5) > 0.01 || p.ExpertBPW >= p.BPW {
		t.Fatalf("profile %+v (share %.3f)", p, share)
	}
	c.MoE = false
	if p := quantProfileFor(c, QuantOption{Name: "IQ2_XS", SizeBytes: size}); p.ExpertBPW != 0 || p.ExpertEstimated {
		t.Fatalf("dense model got an expert width: %+v", p)
	}
}
