package recommend

import (
	"math"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// bytesAt is the file size of a model of paramsB at bpw bits per weight.
func bytesAt(paramsB, bpw float64) int64 { return int64(paramsB * 1e9 * bpw / 8) }

func gib(bytes int64) float64 { return float64(bytes) / (1 << 30) }

func bigLowBitMoE(intelligence float64) Candidate {
	size := bytesAt(120, 2.6)
	return Candidate{Name: "Big 120B A12B", Repo: "fixture/big-moe", AAIntelligence: intelligence, TotalParamsB: 120, ActiveParamsB: 12, MoE: true,
		Quants: []QuantOption{{Name: "UD-IQ2_XXS", SizeBytes: size, SizeGB: gib(size), Dynamic: true}}}
}

func smallDenseQ8(intelligence float64) Candidate {
	size := bytesAt(8, 8.5)
	return Candidate{Name: "Small 8B", Repo: "fixture/small-dense", AAIntelligence: intelligence, TotalParamsB: 8,
		Quants: []QuantOption{{Name: "Q8_0", SizeBytes: size, SizeGB: gib(size)}}}
}

// D9: no bit floor. A big MoE that fits only at ~2.6 bits stays eligible, and
// a small 8-bit model whose estimated quality is under 40% of it no longer
// leads on speed alone.
func TestBigImatrixLowBitMoEOutranksSmallDenseBelowTheFloor(t *testing.T) {
	big, small := bigLowBitMoE(60), smallDenseQ8(18)
	intelligenceTestCatalog(t, big, small)
	caps := intelligenceTestHardware()
	b, okB := evaluate(caps, big)
	s, okS := evaluate(caps, small)
	if !okB || !okS || b.EstimatedQuality <= s.EstimatedQuality || b.ReferenceTurnSeconds <= s.ReferenceTurnSeconds {
		t.Fatalf("fixture: big %+v small %+v", b, s)
	}
	if b.Retention < 0.80 || b.Retention > 0.90 || s.Retention < 0.99 {
		t.Fatalf("retention big %.3f small %.3f", b.Retention, s.Retention)
	}
	cats := TopCategories(caps, 3)
	for name, rows := range map[string][]Recommendation{"overall": cats.Balanced, "top": Top(caps, 3)} {
		if len(rows) < 2 || rows[0].Repo != big.Repo || !rows[0].QualityEligible {
			t.Fatalf("%s: %v", name, repos(rows))
		}
		for _, r := range rows[1:] {
			if r.QualityEligible {
				t.Fatalf("%s: %s is above the floor", name, r.Repo)
			}
		}
	}
	if cats.FirstPick == nil || cats.FirstPick.Repo != big.Repo {
		t.Fatalf("first pick %+v (%s)", cats.FirstPick, cats.FirstPickNote)
	}
}

// Among rows above the floor the reference turn decides, so the faster small
// model leads Best overall while Smartest keeps the capability order.
func TestReferenceTurnRanksEligibleRows(t *testing.T) {
	big, small := bigLowBitMoE(60), smallDenseQ8(30)
	intelligenceTestCatalog(t, big, small)
	caps := intelligenceTestHardware()
	cats := TopCategories(caps, 2)
	if len(cats.Balanced) != 2 || cats.Balanced[0].Repo != small.Repo || !cats.Balanced[1].QualityEligible {
		t.Fatalf("overall %v", repos(cats.Balanced))
	}
	if len(cats.Smartest) == 0 || cats.Smartest[0].Repo != big.Repo {
		t.Fatalf("smartest %v", repos(cats.Smartest))
	}
	if cats.FirstPick == nil || cats.FirstPick.Repo != small.Repo {
		t.Fatalf("first pick %+v", cats.FirstPick)
	}
}

func TestPlainQ2KLosesMoreThanImatrixIQ2OfTheSameSize(t *testing.T) {
	for _, c := range []Candidate{
		{TotalParamsB: 8},
		{TotalParamsB: 120, ActiveParamsB: 12, MoE: true},
		{TotalParamsB: 700, ActiveParamsB: 40, MoE: true},
	} {
		plain, _ := estimatedRetention(c, QuantProfile{BPW: 2.7})
		imat, _ := estimatedRetention(c, QuantProfile{BPW: 2.7, Imatrix: ImatrixName})
		if plain >= imat {
			t.Fatalf("%+v: plain %.3f >= imatrix %.3f", c, plain, imat)
		}
	}
	// Bigger models lose less, and size never makes a low-bit file lossless.
	small, _ := estimatedRetention(Candidate{TotalParamsB: 8}, QuantProfile{BPW: 2.7, Imatrix: ImatrixName})
	huge, _ := estimatedRetention(Candidate{TotalParamsB: 2000}, QuantProfile{BPW: 2.7, Imatrix: ImatrixName})
	oneBit, _ := estimatedRetention(Candidate{TotalParamsB: 2000}, QuantProfile{BPW: 1.7, Imatrix: ImatrixName})
	if small >= huge || huge > 0.90 || oneBit > 0.70 {
		t.Fatalf("small %.3f huge %.3f one-bit %.3f", small, huge, oneBit)
	}
	// Retention falls monotonically with width.
	prev := 1.0
	for bpw := 16.0; bpw >= 1.0; bpw -= 0.1 {
		r, ok := estimatedRetention(Candidate{TotalParamsB: 30}, QuantProfile{BPW: bpw})
		if !ok || r > prev+1e-12 || r < minRetention {
			t.Fatalf("bpw %.1f: %.4f after %.4f", bpw, r, prev)
		}
		prev = r
	}
	if _, ok := estimatedRetention(Candidate{TotalParamsB: 30}, QuantProfile{}); ok {
		t.Fatal("no width, no estimate")
	}
}

func TestNoQualifierPicksNothingForAutomaticDownload(t *testing.T) {
	size := bytesAt(30, 2.6)
	plain := Candidate{Name: "Plain 30B", Repo: "fixture/plain-q2k", AAIntelligence: 50, TotalParamsB: 30,
		Quants: []QuantOption{{Name: "Q2_K", SizeBytes: size, SizeGB: gib(size)}}}
	intelligenceTestCatalog(t, plain)
	cats := TopCategories(intelligenceTestHardware(), 3)
	if len(cats.Balanced) == 0 || cats.Balanced[0].Repo != plain.Repo {
		t.Fatalf("the choices must still be shown: %v", repos(cats.Balanced))
	}
	if cats.FirstPick != nil || cats.FirstPickNote != NoFirstPickNote {
		t.Fatalf("first pick %+v note %q", cats.FirstPick, cats.FirstPickNote)
	}

	eligible := Recommendation{Candidate: Candidate{Repo: "a"}, QualityEligible: true}
	for name, c := range map[string]struct {
		rows []Recommendation
		want string
	}{
		"below floor":  {[]Recommendation{{Candidate: Candidate{Repo: "x"}}, eligible}, "a"},
		"no route":     {[]Recommendation{{Candidate: Candidate{Repo: "x"}, QualityEligible: true, BackendStatus: BackendUnsupported}, eligible}, "a"},
		"plain 2-bit":  {[]Recommendation{{Candidate: Candidate{Repo: "x"}, QualityEligible: true, QuantProfile: QuantProfile{BPW: 2.9}}, eligible}, "a"},
		"imatrix 2bit": {[]Recommendation{{Candidate: Candidate{Repo: "x"}, QualityEligible: true, QuantProfile: QuantProfile{BPW: 2.2, Imatrix: ImatrixName}}, eligible}, "x"},
		"3 bits":       {[]Recommendation{{Candidate: Candidate{Repo: "x"}, QualityEligible: true, QuantProfile: QuantProfile{BPW: 3.0}}, eligible}, "x"},
		"none":         {[]Recommendation{{Candidate: Candidate{Repo: "x"}}}, ""},
	} {
		got, note := chooseFirstPick(c.rows)
		switch {
		case c.want == "" && (got != nil || note == ""):
			t.Errorf("%s: %+v %q", name, got, note)
		case c.want != "" && (got == nil || got.Repo != c.want || note != ""):
			t.Errorf("%s: %+v %q, want %s", name, got, note, c.want)
		}
	}
}

func TestQualityFloorIsFortyPercentOfTheBestVisibleRow(t *testing.T) {
	rows := []Recommendation{
		{Candidate: Candidate{Repo: "best", AAIntelligence: 50}, EstimatedQuality: 50},
		{Candidate: Candidate{Repo: "edge", AAIntelligence: 30}, EstimatedQuality: 20},
		{Candidate: Candidate{Repo: "under", AAIntelligence: 60}, EstimatedQuality: 19.9},
		{Candidate: Candidate{Repo: "no estimate", AAIntelligence: 21}},
	}
	markQualityEligible(rows)
	for i, want := range []bool{true, true, false, true} {
		if rows[i].QualityEligible != want {
			t.Errorf("%s: eligible %v", rows[i].Repo, rows[i].QualityEligible)
		}
	}
}

func TestQualityAndTurnLabels(t *testing.T) {
	r := Recommendation{Retention: 0.9123, QuantProfile: QuantProfile{BPW: 2.71, Imatrix: "keys", Source: QuantSourceHeader}}
	for _, c := range []struct {
		r    Recommendation
		want string
	}{
		{r, "quality ≈ 0.91× · 2.7 bpw · imatrix"},
		{Recommendation{Retention: 0.95, RetentionMeasured: true, QuantProfile: r.QuantProfile}, "measured 0.95× · 2.7 bpw · imatrix"},
		{Recommendation{Retention: 0.97, QuantProfile: QuantProfile{BPW: 4.84, Imatrix: ImatrixDynamic, Source: QuantSourceSize}}, "quality ≈ 0.97× · 4.8 bpw · dynamic"},
		{Recommendation{Retention: 0.96, QuantProfile: QuantProfile{BPW: 4.84, Source: QuantSourceNominal}}, "quality ≈ 0.96× · ~4.8 bpw"},
		{Recommendation{}, "quality: no estimate"},
	} {
		if got := QualityLabel(c.r); got != c.want {
			t.Errorf("QualityLabel = %q, want %q", got, c.want)
		}
	}
	for secs, want := range map[float64]string{0: "turn: no estimate", 42.4: "turn ~42 s", 900: "turn ~15 min"} {
		if got := ReferenceTurnLabel(Recommendation{ReferenceTurnSeconds: secs}); got != want {
			t.Errorf("ReferenceTurnLabel(%v) = %q, want %q", secs, got, want)
		}
	}
}

func TestMeasuredRetentionReplacesTheEstimate(t *testing.T) {
	t.Cleanup(func() { SetMeasuredRetention(nil) })
	SetMeasuredRetention(func(key string) (float64, bool) { return 0.95, key == "fixture@r/Q2_K:1" })
	r := Recommendation{Candidate: Candidate{AAIntelligence: 40, TotalParamsB: 8}, QuantProfile: QuantProfile{BPW: 2.7, FileKey: "fixture@r/Q2_K:1"}}
	applyQualityEstimate(&r)
	if !r.RetentionMeasured || r.Retention != 0.95 || math.Abs(r.EstimatedQuality-38) > 1e-9 || r.RetentionModel != "" {
		t.Fatalf("%+v", r)
	}
	r.QuantProfile.FileKey = "other"
	applyQualityEstimate(&r)
	if r.RetentionMeasured || r.RetentionModel != RetentionModel || r.Retention >= 0.95 {
		t.Fatalf("%+v", r)
	}
}

func TestTurnTiesKeepThePracticalOrder(t *testing.T) {
	row := func(repo string, score int, secs float64) Recommendation {
		return Recommendation{Candidate: Candidate{Repo: repo, AAIntelligence: 40}, EstimatedQuality: 40, Score: score, ReferenceTurnSeconds: secs, SpeedTier: 2}
	}
	for _, order := range [][]Recommendation{
		{row("low", 100, 30.01), row("high", 200, 30.04), row("fast", 50, 20)},
		{row("fast", 50, 20), row("high", 200, 30.04), row("low", 100, 30.01)},
		{row("high", 200, 30.04), row("low", 100, 30.01), row("fast", 50, 20)},
	} {
		sortRecommendations(order)
		if got := repos(order); got[0] != "fast" || got[1] != "high" || got[2] != "low" {
			t.Fatalf("order %v", got)
		}
	}
}

// Rows without quant data or speed keep today's order (catalog intelligence)
// and say so.
func TestRowsWithoutQuantDataKeepTodaysOrder(t *testing.T) {
	caps := &detect.Capabilities{OS: "linux", RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		GPUs: []detect.GPU{{Name: "Test GPU", VRAMTotalMB: 24576}}}
	var rows []Recommendation
	for _, c := range []Candidate{
		{Name: "mid", Repo: "repo/mid", SizeGB: 4, Quality: 50},
		{Name: "top", Repo: "repo/top", SizeGB: 4, Quality: 80},
		{Name: "low", Repo: "repo/low", SizeGB: 4, Quality: 20},
	} {
		r, ok := evaluate(caps, c)
		if !ok {
			t.Fatalf("%s does not fit", c.Name)
		}
		if r.Retention != 0 || r.ReferenceTurnSeconds != 0 || QualityLabel(r) != "quality: no estimate" || ReferenceTurnLabel(r) != "turn: no estimate" {
			t.Fatalf("%s invented an estimate: %+v", c.Name, r)
		}
		rows = append(rows, r)
	}
	sortRecommendations(rows)
	if got := repos(rows); got[0] != "repo/top" || got[1] != "repo/mid" || got[2] != "repo/low" {
		t.Fatalf("order %v", got)
	}
	if pick, _ := chooseFirstPick(rows); pick == nil || pick.Repo != "repo/top" {
		t.Fatalf("first pick %+v", pick)
	}
}

// Prefill rates sit near the October 13 README observations they were fitted
// to (one host): dense 27B on GPU 1,008-1,391 tok/s, CPU-only dense 27B 16,
// CPU-only 35B-A3B 127.
func TestPrefillEstimateMatchesTheFittedObservations(t *testing.T) {
	gpu := hardware(&detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 131072}, GPUs: []detect.GPU{{Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24576}}})
	cpu := hardware(&detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 16384}})
	dense := Candidate{TotalParamsB: 27}
	moe := Candidate{TotalParamsB: 35, ActiveParamsB: 3, MoE: true}
	for name, c := range map[string]struct {
		got, lo, hi float64
	}{
		"gpu dense": {predictPrefillTPSForFit(gpu, dense, QuantOption{SizeGB: 18}, "single GPU", 4096), 700, 1400},
		"cpu dense": {predictPrefillTPSForFit(cpu, dense, QuantOption{SizeGB: 8}, "CPU RAM", 0), 10, 20},
		"cpu moe":   {predictPrefillTPSForFit(cpu, moe, QuantOption{SizeGB: 10}, "CPU RAM", 0), 90, 160},
	} {
		if c.got < c.lo || c.got > c.hi || math.IsNaN(c.got) {
			t.Errorf("%s: %.1f tok/s, want %.0f-%.0f", name, c.got, c.lo, c.hi)
		}
	}
	if got := predictPrefillTPSForFit(gpu, Candidate{}, QuantOption{SizeGB: 4}, "single GPU", 0); got != 0 {
		t.Fatalf("no parameters, no prefill estimate: %.1f", got)
	}
	if got := referenceTurnSeconds(1000, 0); got != 0 {
		t.Fatalf("unpriced turn got %.1f", got)
	}
	if got := referenceTurnSeconds(1000, 50); math.Abs(got-25) > 1e-9 {
		t.Fatalf("turn %.2f, want 25", got)
	}
}
