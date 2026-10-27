package recommend

import (
	"math"
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

func TestFastestQuantCeilingFallsBackToQ4(t *testing.T) {
	caps := &detect.Capabilities{
		OS:       "linux",
		RAM:      detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		GPUs:     []detect.GPU{{Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24576}},
		Backends: []detect.Backend{{Name: "cuda", Path: "/bin/llama-server-cuda"}},
	}
	candidate := Candidate{
		Name:           "fast small",
		Repo:           "repo/fast-small",
		AAIntelligence: 50,
		TotalParamsB:   4,
		Quants: []QuantOption{
			{Name: "Q3_K_M", SizeGB: 2.1},
			{Name: "Q4_K_M", SizeGB: 2.6},
			{Name: "Q5_K_M", SizeGB: 2.9},
			{Name: "Q8_0", SizeGB: 4.2},
			{Name: "BF16", SizeGB: 7.9},
		},
	}
	best, ok := evaluate(caps, candidate)
	if !ok {
		t.Fatal("expected candidate to fit")
	}
	if best.QuantName != "BF16" {
		t.Fatalf("expected normal recommendation to keep highest quality, got %q", best.QuantName)
	}
	fast, ok := evaluateWithQuantFilter(caps, candidate, fastestQuantAllowed)
	if !ok {
		t.Fatal("expected fastest-path candidate to fit under the Q4 ceiling")
	}
	if fast.QuantName != "Q4_K_M" {
		t.Fatalf("expected fastest path to cap at Q4_K_M, got %q", fast.QuantName)
	}
	for _, q := range []QuantOption{{Name: "Q5_K_M"}, {Name: "Q6_K"}, {Name: "Q8_0"}, {Name: "BF16"}, {Name: "F16"}, {Name: "F32"}} {
		if fastestQuantAllowed(q) {
			t.Fatalf("%s should not be allowed in Fastest", q.Name)
		}
	}
	for _, q := range []QuantOption{{Name: "Q4_K_M"}, {Name: "IQ4_XS"}, {Name: "MXFP4"}, {Name: "Q3_K_M"}} {
		if !fastestQuantAllowed(q) {
			t.Fatalf("%s should be allowed in Fastest", q.Name)
		}
	}
}

func TestBalancedPrefersBlendedQuantOverBF16(t *testing.T) {
	// 24 GB GPU + 64 GB RAM: a ~27B dense fits at Q4/Q5 entirely in VRAM (fast),
	// but its BF16 (~55 GB) only fits by spilling to RAM (slow). The local
	// quality-only selector keeps BF16; the public practical selector should
	// pick the fast quant instead. This is the Qwen3.6-27B bug.
	caps := &detect.Capabilities{
		OS:       "linux",
		RAM:      detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		GPUs:     []detect.GPU{{Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24576}},
		Backends: []detect.Backend{{Name: "cuda", Path: "/bin/llama-server-cuda"}},
	}
	c := Candidate{
		Name:           "dense 27b",
		Repo:           "repo/dense-27b",
		AAIntelligence: 70,
		TotalParamsB:   27,
		Quants: []QuantOption{
			{Name: "Q4_K_M", SizeGB: 16},
			{Name: "Q5_K_M", SizeGB: 19},
			{Name: "BF16", SizeGB: 55},
		},
	}
	smart, ok := evaluate(caps, c)
	if !ok {
		t.Fatal("expected candidate to fit")
	}
	if smart.QuantName != "BF16" {
		t.Fatalf("quality-only selector should keep highest-quality BF16, got %q", smart.QuantName)
	}
	balanced, ok := evaluateWithSelector(caps, c, nil, betterByScore)
	if !ok {
		t.Fatal("expected blended candidate to fit")
	}
	if balanced.QuantName == "BF16" {
		t.Fatalf("Best overall should pick a fast practical quant, not BF16 (BF16 @ ~%.0f tok/s)", smart.PredictedTPS)
	}
	if balanced.PredictedTPS <= smart.PredictedTPS {
		t.Fatalf("blended pick should be faster than the BF16 pick: balanced %.1f vs smart %.1f tok/s", balanced.PredictedTPS, smart.PredictedTPS)
	}
}

func TestArchRunnableFiltersUnsupported(t *testing.T) {
	// Architectures no bundled backend can load must be filtered out of
	// recommendations (the launcher would only fail with "unknown architecture").
	for _, arch := range []string{"bailingmoe2.5", "longcat-flash-ngram", "mllama"} {
		if archRunnable(Candidate{Arch: arch}) {
			t.Fatalf("arch %q should be filtered (no bundled backend loads it)", arch)
		}
	}
	// Supported arches, and entries without arch metadata, must be kept.
	// deepseek4 is runnable since mainline PR #24162 (2026-06-29).
	for _, arch := range []string{"qwen35", "qwen35moe", "minimax-m3", "gpt-oss", "llama", "deepseek2", "deepseek4", "DeepSeek4", ""} {
		if !archRunnable(Candidate{Arch: arch}) {
			t.Fatalf("arch %q is runnable and must not be filtered", arch)
		}
	}
}

// A model no upstream backend loads still runs: its first launch finds and
// builds a fork backend. The stamp only labels it.
func TestForkOnlyArchitecturesStayRecommendable(t *testing.T) {
	no, yes := false, true
	fork := Candidate{Arch: "axk2", Runnable: &no}
	if !archRunnable(fork) || !fork.NeedsForkBackend() {
		t.Fatal("a fork-only model was hidden or not labelled")
	}
	if (Candidate{Arch: "k2-horizon", Runnable: &yes}).NeedsForkBackend() || (Candidate{Arch: "qwen35"}).NeedsForkBackend() {
		t.Fatal("an upstream or unstamped model was labelled as needing a fork")
	}
	// The explicit blocklist still applies when a stamp is absent.
	if archRunnable(Candidate{Arch: "longcat-flash-ngram"}) {
		t.Fatal("blocklisted arch was recommended without a stamp")
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

func TestMoESpeedEstimateIncludesDispatchCost(t *testing.T) {
	// Measured DeepSeek V4 Flash IQ4 on this heterogeneous 48 GiB setup decodes
	// at about 6 tok/s. The estimator should remain useful for ranking instead
	// of reporting the active-parameter bandwidth ceiling (about 9 tok/s).
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 120000},
		GPUs: []detect.GPU{
			{Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24576},
			{Name: "NVIDIA GeForce RTX 3060", VRAMTotalMB: 12288},
			{Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12288},
		},
	}
	c := Candidate{MoE: true, TotalParamsB: 284, ActiveParamsB: 13}
	got := predictDecodeTPS(caps, c, QuantOption{Name: "UD-IQ4_XS", SizeGB: 128.43})
	if got < 5.0 || got > 7.5 {
		t.Fatalf("DeepSeek V4-class MoE estimate %.1f tok/s is outside the measured useful range", got)
	}
}

// Measured on one rig with host bandwidth probed at 26.27 GB/s. The estimate
// used to charge the always-read backbone at RAM speed and discount measured
// bandwidth by 0.5, predicting GLM-5.3-Flash under 4 tok/s.
func TestMoESpeedEstimateMatchesMeasuredDecode(t *testing.T) {
	t.Setenv("LLM_CACHE_DIR", t.TempDir())
	gpuRig := referenceRig()
	gpuRig.HostMemoryBandwidthMBps = 26273
	cpuOnly := &detect.Capabilities{OS: "linux", RAM: detect.RAMInfo{TotalMB: 16384, FreeMB: 16384},
		CPU: detect.CPUInfo{Cores: 14}, HostMemoryBandwidthMBps: 26273}
	for _, tc := range []struct {
		name     string
		caps     *detect.Capabilities
		quant    QuantOption
		measured float64
	}{
		// eval-time decode while serving agent work, 2026-09-14
		{"Z AI GLM-5.3-Flash", gpuRig, QuantOption{Name: "UD-Q3_K_XL", SizeGB: 137.4, Dynamic: true}, 6.35},
		// 128-token synthetic decode, 2026-09-16
		{"Alibaba Qwen3.8-Flash-Next", gpuRig, QuantOption{Name: "UD-Q3_K_XL", SizeGB: 83.8, Dynamic: true}, 21.82},
		// CPU only, 16 GiB, relaunch ledger decode, October 8
		{"Alibaba Qwen3.6 35B A3B (Reasoning)", cpuOnly, QuantOption{Name: "UD-IQ2_XXS", SizeGB: 10.0, Dynamic: true}, 15.3},
	} {
		var c Candidate
		for _, row := range Shortlist() {
			if row.Name == tc.name {
				c = row
			}
		}
		// Only the expert layout drives the split; MLA rows may lack hkv.
		if _, _, ok := moeRoutedShares(c); !c.MoE || !ok {
			t.Fatalf("%s: catalog row lost its MoE geometry: %+v", tc.name, c)
		}
		c.Quants = []QuantOption{tc.quant}
		rec, ok := evaluate(tc.caps, c)
		if !ok {
			t.Fatalf("%s %s no longer fits", tc.name, tc.quant.Name)
		}
		t.Logf("%s %s (%s): predicted %.2f tok/s, measured %.2f", tc.name, tc.quant.Name, rec.Fit, rec.PredictedTPS, tc.measured)
		if math.Abs(rec.PredictedTPS-tc.measured) > 0.30*tc.measured {
			t.Errorf("%s %s (%s): predicted %.2f tok/s, measured %.2f", tc.name, tc.quant.Name, rec.Fit, rec.PredictedTPS, tc.measured)
		}
	}
}

func TestMoESpeedEstimateKeepsBackboneOnGPU(t *testing.T) {
	caps := referenceRig()
	caps.HostMemoryBandwidthMBps = 26273
	geo := Candidate{MoE: true, TotalParamsB: 320.759, Layers: 46, LeadingDense: 3, Experts: 288,
		ExpertUsed: 8, ExpertFF: 2048, Embedding: 4096, FeedForward: 12288, HeadCountKV: 1, KeyLength: 512}
	geo.ActiveParamsB = activeParamsFromGeometry(geo)
	quant := QuantOption{Name: "UD-Q3_K_XL", SizeGB: 137.4}
	// Same active count with no layout: the VRAM share is spread evenly and
	// the backbone streams from RAM.
	flat := Candidate{MoE: true, TotalParamsB: geo.TotalParamsB, ActiveParamsB: geo.ActiveParamsB}
	split, even := predictDecodeTPS(caps, geo, quant), predictDecodeTPS(caps, flat, quant)
	if split <= even {
		t.Fatalf("backbone-on-GPU estimate %.2f is not above the even spread %.2f", split, even)
	}
	// A resident model has no CPU layers to pay for.
	small := QuantOption{Name: "UD-Q3_K_XL", SizeGB: 20}
	if a, b := predictDecodeTPS(caps, geo, small), predictDecodeTPS(caps, flat, small); math.Abs(a-b) > 1e-9 {
		t.Fatalf("resident MoE estimate depends on layout: %.2f vs %.2f", a, b)
	}
}

// The per-layer dispatch cost was fitted on one 26 GB/s rig. It must not cap
// a fast CPU-only host near 1/(layers*0.7 ms): every term scales with the
// measured bandwidth, so the estimate does too.
func TestMoESpeedEstimateScalesWithCPUOnlyBandwidth(t *testing.T) {
	c := Candidate{MoE: true, TotalParamsB: 34.66, Layers: 40, Experts: 256, ExpertUsed: 8,
		ExpertFF: 512, ExpertShFF: 512, Embedding: 2048, HeadCountKV: 2, KeyLength: 256}
	c.ActiveParamsB = activeParamsFromGeometry(c)
	quant := QuantOption{Name: "UD-IQ2_XXS", SizeGB: 10.0}
	host := func(mbps int) *detect.Capabilities {
		return &detect.Capabilities{OS: "linux", RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
			CPU: detect.CPUInfo{Cores: 64}, HostMemoryBandwidthMBps: mbps}
	}
	slow, fast := predictDecodeTPS(host(26273), c, quant), predictDecodeTPS(host(200000), c, quant)
	if want := slow * 200000 / 26273; math.Abs(fast-want) > 0.01*want {
		t.Fatalf("200 GB/s CPU-only estimate %.1f tok/s, want %.1f (26 GB/s gives %.1f)", fast, want, slow)
	}
	if fast < 60 {
		t.Fatalf("3B-active MoE on a 200 GB/s CPU-only host capped at %.1f tok/s", fast)
	}
}

// Without a measured host bandwidth the layout may only move always-read
// bytes to the GPU: no catalog MoE may come out slower than the even spread,
// and a CPU-only host keeps the even spread exactly.
func TestMoESpeedEstimateUnmeasuredHostsNotSlowerThanEvenSpread(t *testing.T) {
	t.Setenv("LLM_CACHE_DIR", t.TempDir())
	ram := func(gb int) detect.RAMInfo { return detect.RAMInfo{TotalMB: gb << 10, FreeMB: gb << 10} }
	cuda := []detect.Backend{{Name: "cuda", Path: "/bin/llama-server"}}
	hosts := map[string]*detect.Capabilities{
		"cpu-only 64 GB": {OS: "linux", RAM: ram(64), CPU: detect.CPUInfo{Cores: 16}},
		"3090 + 64 GB": {OS: "linux", RAM: ram(64), CPU: detect.CPUInfo{Cores: 16}, Backends: cuda,
			GPUs: []detect.GPU{{Name: "NVIDIA GeForce RTX 3090", VRAMTotalMB: 24576}}},
		"4060 laptop + 32 GB": {OS: "windows", RAM: ram(32), CPU: detect.CPUInfo{Cores: 8}, Backends: cuda,
			GPUs: []detect.GPU{{Name: "NVIDIA GeForce RTX 4060 Laptop GPU", VRAMTotalMB: 8188}}},
	}
	checked := 0
	for _, c := range Shortlist() {
		if !c.MoE || c.ActiveParamsB <= 0 {
			continue
		}
		flat := Candidate{MoE: true, TotalParamsB: c.TotalParamsB, ActiveParamsB: c.ActiveParamsB}
		for name, caps := range hosts {
			for _, q := range c.Quants {
				got, even := predictDecodeTPS(caps, c, q), predictDecodeTPS(caps, flat, q)
				if got < even*(1-1e-9) || (len(caps.GPUs) == 0 && math.Abs(got-even) > 1e-9*even) {
					t.Errorf("%s %s on %s: %.2f tok/s, even spread %.2f", c.Name, q.Name, name, got, even)
				}
				checked++
			}
		}
	}
	if checked == 0 {
		t.Fatal("no catalog MoE rows checked")
	}
}

func TestMoESpeedEstimateUsesMeasuredHostMemoryBandwidth(t *testing.T) {
	candidate := Candidate{MoE: true, TotalParamsB: 200, ActiveParamsB: 12}
	quant := QuantOption{Name: "Q4_K_M", SizeGB: 110}
	slow := &detect.Capabilities{
		HostMemoryBandwidthMBps: 30000,
		GPUs:                    []detect.GPU{{Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 8192}},
	}
	fast := &detect.Capabilities{
		HostMemoryBandwidthMBps: 120000,
		GPUs:                    []detect.GPU{{Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 8192}},
	}
	slowTPS := predictDecodeTPS(slow, candidate, quant)
	fastTPS := predictDecodeTPS(fast, candidate, quant)
	if fastTPS <= slowTPS*2 {
		t.Fatalf("measured host bandwidth did not materially affect offload estimate: slow %.2f, fast %.2f", slowTPS, fastTPS)
	}
}

func TestPredictDecodeTPSDenseSingleGPUUsesFastestFittingCard(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24564},
		{Name: "NVIDIA GeForce RTX 3060", VRAMTotalMB: 12288},
		{Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12282},
	}}
	candidate := Candidate{TotalParamsB: 27}
	quant := QuantOption{Name: "Q5_K_M", SizeGB: 18.2}
	got := predictDecodeTPS(caps, candidate, quant)
	if got < 36 || got > 43 {
		t.Fatalf("dense single-GPU prediction = %.2f t/s, want near measured 40.15", got)
	}
}

func TestPredictDecodeTPSSmallDenseIncludesKernelBoundScaling(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24564}}}
	candidate := Candidate{TotalParamsB: 4}
	quant := QuantOption{Name: "Q4_K_M", SizeGB: 2.6}
	got := predictDecodeTPS(caps, candidate, quant)
	if got < 165 || got > 205 {
		t.Fatalf("small dense prediction = %.2f t/s, want near measured 180", got)
	}
}

func TestUnrunnableArchesLoaded(t *testing.T) {
	// The embedded blocklist must parse and contain exactly the known-bad
	// arches (lowercased). This guards the data-driven load in init() against
	// silent regressions in unrunnable_arches.json.
	loaded, err := loadUnrunnableArches(unrunnableArchesJSON)
	if err != nil {
		t.Fatalf("embedded unrunnable_arches.json failed to parse: %v", err)
	}
	want := map[string]bool{
		"bailingmoe2.5":       true,
		"longcat-flash-ngram": true,
		"mllama":              true,
	}
	if len(loaded) != len(want) {
		t.Fatalf("expected %d unrunnable arches, got %d (%v)", len(want), len(loaded), loaded)
	}
	for arch := range want {
		if !loaded[arch] {
			t.Fatalf("expected unrunnable arch %q in loaded blocklist, got %v", arch, loaded)
		}
	}
	// init() must have populated the package var from the same data.
	for arch := range want {
		if !unrunnableArch[arch] {
			t.Fatalf("unrunnableArch package var missing %q (init did not load it)", arch)
		}
	}
}

func TestUnrunnableArchesFallbackOnBadJSON(t *testing.T) {
	// A malformed data file must not break the blocklist: loadUnrunnableArches
	// returns an error and the caller falls back to the hardcoded set.
	if _, err := loadUnrunnableArches([]byte("{not json")); err == nil {
		t.Fatal("expected parse error for malformed JSON, got nil")
	}
	// The fallback set must still cover the known-bad arches.
	for _, arch := range []string{"bailingmoe2.5", "longcat-flash-ngram", "mllama"} {
		if !fallbackUnrunnableArch[arch] {
			t.Fatalf("fallback blocklist missing %q", arch)
		}
	}
}
