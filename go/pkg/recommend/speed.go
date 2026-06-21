package recommend

import (
	"math"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// This file models the two real axes a recommendation should rank on:
//
//  1. effective intelligence = base AA index * quality retained after quantization,
//     where retention depends on bits-per-weight AND model size (large models keep
//     far more quality at aggressive quants than small ones), with a bonus for
//     Unsloth dynamic ("UD") quants that are built to minimise loss.
//
//  2. predicted decode tok/s on the *current* machine, from a memory-bandwidth
//     model: per-token bytes = active params * bytes/weight, read at a blend of
//     VRAM and system-RAM bandwidth weighted by how much of the model fits in VRAM.
//
// Both are approximations tuned to keep ordering correct and to gate genuinely
// unusable picks; they are not a substitute for `--ai-tune`, which measures.

// Speed tiers used to rank usable models ahead of technically-fits-but-crawls ones.
const (
	interactiveTPS = 15.0 // smooth interactive use
	usableTPS      = 6.0  // tolerable, but not snappy

	ramBandwidthGBps  = 60.0 // dual-channel DDR4/DDR5 ballpark; the offload bottleneck
	decodeEfficiency  = 0.70 // fraction of peak bandwidth llama.cpp realises at decode
	maxPredictedTPS   = 400.0
	defaultVRAMBWGBps = 350.0
)

// bytesPerWeight is the effective on-disk/in-memory bytes per parameter for a
// quant, i.e. GB per billion params. Used for both size plausibility and the
// per-token read estimate.
func bytesPerWeight(quant string) float64 {
	q := strings.ToUpper(quant)
	switch {
	case strings.Contains(q, "F32"):
		return 4.0
	case strings.Contains(q, "BF16") || strings.Contains(q, "F16"):
		return 2.0
	case strings.Contains(q, "F8"):
		return 1.0
	case strings.Contains(q, "Q8"):
		return 1.06
	case strings.Contains(q, "Q6") || strings.Contains(q, "IQ6"):
		return 0.82
	case strings.Contains(q, "Q5") || strings.Contains(q, "IQ5"):
		return 0.69
	case strings.Contains(q, "MXFP4") || strings.Contains(q, "MXP4"):
		return 0.53
	case strings.Contains(q, "IQ4"):
		return 0.51
	case strings.Contains(q, "Q4") || strings.Contains(q, "I4"):
		return 0.56
	case strings.Contains(q, "IQ3"):
		return 0.41
	case strings.Contains(q, "Q3"):
		return 0.44
	case strings.Contains(q, "IQ2"):
		return 0.28
	case strings.Contains(q, "Q2"):
		return 0.33
	case strings.Contains(q, "IQ1") || strings.Contains(q, "Q1"):
		return 0.20
	default:
		return 0.55
	}
}

// quantQualityRetention returns the fraction (0,1] of base intelligence retained
// after quantization. Loss shrinks with model size and for Unsloth dynamic quants.
func quantQualityRetention(quant string, totalParamsB float64, dynamic bool) float64 {
	q := strings.ToUpper(quant)
	var loss float64
	switch {
	case strings.Contains(q, "F32"), strings.Contains(q, "BF16"), strings.Contains(q, "F16"):
		loss = 0.0
	case strings.Contains(q, "Q8"):
		loss = 0.004
	case strings.Contains(q, "Q6"), strings.Contains(q, "IQ6"):
		loss = 0.010
	case strings.Contains(q, "Q5"), strings.Contains(q, "IQ5"):
		loss = 0.020
	case strings.Contains(q, "IQ4"):
		loss = 0.040
	case strings.Contains(q, "MXFP4"), strings.Contains(q, "MXP4"):
		loss = 0.050
	case strings.Contains(q, "Q4"), strings.Contains(q, "I4"):
		loss = 0.045
	case strings.Contains(q, "IQ3"):
		loss = 0.075
	case strings.Contains(q, "Q3"):
		loss = 0.090
	case strings.Contains(q, "IQ2"):
		loss = 0.150
	case strings.Contains(q, "Q2"):
		loss = 0.200
	case strings.Contains(q, "IQ1"), strings.Contains(q, "Q1"):
		loss = 0.380
	default:
		loss = 0.060
	}
	if loss == 0 {
		return 1.0
	}
	// Size attenuation: bigger models tolerate quantization far better.
	// (13/params)^0.4, clamped — ~1.6x loss for a 4B, ~0.45x for a 122B.
	att := 1.0
	if totalParamsB > 0 {
		att = math.Pow(13.0/totalParamsB, 0.4)
		att = clampF(att, 0.45, 1.8)
	}
	if dynamic {
		loss *= 0.70 // Unsloth dynamic quants minimise loss at the same size
	}
	r := 1.0 - loss*att
	// Hard ceilings: very low-bit quants are heavily degraded no matter how big
	// the model is, so size attenuation must not let them masquerade as
	// near-lossless (a 1-bit 400B is not "88% as smart").
	if c := retentionCeiling(q); r > c {
		r = c
	}
	return clampF(r, 0.20, 1.0)
}

// retentionCeiling caps how much quality a quant can possibly retain, regardless
// of model size.
func retentionCeiling(q string) float64 {
	switch {
	case strings.Contains(q, "IQ1"), strings.Contains(q, "Q1"):
		return 0.55
	case strings.Contains(q, "IQ2"):
		return 0.80
	case strings.Contains(q, "Q2"):
		return 0.82
	case strings.Contains(q, "IQ3"):
		return 0.90
	case strings.Contains(q, "Q3"):
		return 0.92
	default:
		return 1.0
	}
}

// vramBandwidthGBps estimates a GPU's memory bandwidth from its name.
func vramBandwidthGBps(name string) float64 {
	n := strings.ToLower(name)
	table := []struct {
		sub string
		bw  float64
	}{
		{"3090 ti", 1008}, {"3090", 936},
		{"4090", 1008}, {"4080 super", 736}, {"4080", 717},
		{"4070 ti super", 672}, {"4070 ti", 504}, {"4070 super", 504}, {"4070", 504},
		{"4060 ti", 288}, {"4060", 272},
		{"3080 ti", 912}, {"3080", 760}, {"3070", 448},
		{"3060 ti", 448}, {"3060", 360}, {"3050", 224},
		{"a100", 1555}, {"h200", 4800}, {"h100", 3350}, {"a6000", 768}, {"a40", 696},
		{"l40", 864}, {"a5000", 768}, {"a4000", 448}, {"l4", 300},
		{"2080 ti", 616}, {"2080", 448}, {"2070", 448}, {"2060", 336},
		{"v100", 900}, {"p100", 732}, {"t4", 320},
		{"mi300", 5300}, {"mi250", 3200}, {"mi210", 1600},
		{"7900 xtx", 960}, {"7900 xt", 800}, {"7800 xt", 624}, {"6900 xt", 512},
		{"m3 ultra", 800}, {"m2 ultra", 800}, {"m1 ultra", 800},
		{"m4 max", 546}, {"m3 max", 400}, {"m2 max", 400}, {"m1 max", 400},
		{"m4 pro", 273}, {"m3 pro", 150}, {"m2 pro", 200}, {"m1 pro", 200},
		{"apple m", 120},
	}
	for _, e := range table {
		if strings.Contains(n, e.sub) {
			return e.bw
		}
	}
	return defaultVRAMBWGBps
}

// weightedVRAMBandwidth returns VRAM bandwidth averaged across GPUs, weighted by
// each GPU's share of total VRAM (where the model's weights end up living).
func weightedVRAMBandwidth(caps *detect.Capabilities) float64 {
	if caps == nil || len(caps.GPUs) == 0 {
		return 0
	}
	var num, den float64
	for _, g := range caps.GPUs {
		w := float64(g.VRAMTotalMB)
		if w <= 0 {
			w = 1
		}
		num += vramBandwidthGBps(g.Name) * w
		den += w
	}
	if den == 0 {
		return defaultVRAMBWGBps
	}
	return num / den
}

// predictDecodeTPS estimates decode tok/s for a candidate+quant on this machine.
// Returns 0 when params are unknown (callers then skip speed gating).
func predictDecodeTPS(caps *detect.Capabilities, c Candidate, quant QuantOption) float64 {
	activeB := c.TotalParamsB
	if c.MoE && c.ActiveParamsB > 0 {
		activeB = c.ActiveParamsB
	}
	if activeB <= 0 || quant.SizeGB <= 0 {
		return 0
	}
	perTokenGB := activeB * bytesPerWeight(quant.Name) // GB read per token
	if perTokenGB <= 0 {
		return 0
	}

	vramBW := weightedVRAMBandwidth(caps)
	budget := hardware(caps)
	usableVRAMGB := float64(budget.totalVRAM) / 1024.0 * 0.88
	// fraction of the model's weights resident in VRAM
	f := 0.0
	if vramBW > 0 {
		f = clampF(usableVRAMGB/quant.SizeGB, 0, 1)
	}
	// blended seconds-per-GB: VRAM-resident bytes read fast, RAM-resident slow.
	secPerGB := f/vramBWorRAM(vramBW) + (1-f)/ramBandwidthGBps
	if secPerGB <= 0 {
		return 0
	}
	tps := decodeEfficiency / (perTokenGB * secPerGB)
	return clampF(tps, 0, maxPredictedTPS)
}

func vramBWorRAM(vramBW float64) float64 {
	if vramBW <= 0 {
		return ramBandwidthGBps
	}
	return vramBW
}

// speedFactor smoothly down-weights slow picks so a smart-but-crawling giant
// ranks below a fast capable model, without hard tier cliffs. Returns 1.0 when
// speed is unknown (no params) so intelligence alone decides.
func speedFactor(tps float64) float64 {
	if tps <= 0 {
		return 1.0
	}
	return clampF(0.30+0.70*(tps/interactiveTPS), 0.30, 1.0)
}

// usabilityFactor lightly down-weights picks too slow to actually use, so the
// "Smartest" category does not surface a marginally-smarter quant that crawls
// (e.g. a 27B at BF16 @ 3 tok/s) when a nearly-as-smart, far faster quant of the
// same model exists (the same 27B at Q5 @ 40 tok/s). 1.0 at/above usableTPS,
// ramping to 0.5 at 0 tok/s. Unknown speed (no params) returns 1.0 so it does
// not penalize models we can't predict.
func usabilityFactor(tps float64) float64 {
	if tps <= 0 || tps >= usableTPS {
		return 1.0
	}
	return 0.5 + 0.5*(tps/usableTPS)
}

// speedTier buckets a predicted tok/s. -1 means "unknown" (no params) so callers
// can fall back to pure intelligence ordering.
func speedTier(tps float64) int {
	switch {
	case tps <= 0:
		return -1
	case tps >= interactiveTPS:
		return 2
	case tps >= usableTPS:
		return 1
	default:
		return 0
	}
}

// plausibleQuantSize rejects catalog rows whose size is physically impossible for
// the quant + param count (e.g. a phantom "F16 = 0.9 GB" for a 30B model, which
// comes from incomplete Hugging Face shard-size metadata).
func plausibleQuantSize(c Candidate, q QuantOption) bool {
	if c.TotalParamsB <= 0 || q.SizeGB <= 0 {
		return true // not enough info to judge; don't drop
	}
	expected := c.TotalParamsB * bytesPerWeight(q.Name)
	// allow generous slack for embeddings/metadata and quant-mix variance
	return q.SizeGB >= expected*0.6 && q.SizeGB <= expected*1.8
}

func clampF(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
