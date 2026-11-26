package recommend

import (
	"fmt"
	"math"
	"strings"
)

// Estimated quality of one (model, GGUF file): catalog intelligence of the base
// model times a retention factor from the file's effective bits per weight,
// the model's size, and whether an importance matrix shaped the quant. It is an
// estimate, shown as one ("quality ≈ 0.91×"), and conservative by design.
//
// What the sources establish (read 2026-11-26):
//   - llama.cpp tools/perplexity/README.md (LLaMA 3 8B, mean KL divergence vs
//     FP16): Q8_0 0.0014, Q6_K 0.0055, Q5_K_M 0.011, Q4_K_M 0.031, Q3_K_M 0.102,
//     Q2_K 0.445 without an imatrix; with one (WT 10m) Q4_K_M 0.028, Q3_K_M
//     0.084, Q2_K 0.332, IQ2_M 0.326, IQ2_XXS 0.812, IQ1_S 2.25. An imatrix cut
//     KLD to 0.74-0.90 of the plain quant at the same type.
//     https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md
//     (the Debian copy named in the plan, sources.debian.org/src/llama.cpp/
//     7965+dfsg-1/tools/perplexity/README.md, returned 404 that day).
//   - Artefact2's KLD table (Mistral-7B, imatrix, median KLD by bpw): IQ2_XXS
//     2.20 bpw 0.175, Q2_K 3.00 0.059, IQ3_XXS 3.21 0.033, IQ4_XS 4.32 0.0088,
//     Q5_K_M 5.67 0.0043, Q6_K 6.57 0.0032.
//     https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9
//   - Unsloth's GGUF benchmarks (a 35B-A3B MoE, mean KLD): UD-IQ2_XXS 0.185,
//     UD-Q2_K_XL 0.097, UD-IQ3_XXS 0.050, UD-Q4_K_XL 0.014, Q8_K_XL 0.0026; a
//     non-dynamic imatrix upload of similar size measured 1.3-2x their KLD, and
//     the page says imatrix "generally helps on lower bits".
//     https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks
//   - Rule of thumb: "2-bit quants approach perplexity similar to a model of
//     half the parameter count", stated as "probably not a hard-and-fast rule".
//     https://rentry.co/llama-cpp-quants-or-fine-ill-do-it-myself-then-pt-2
//
// What is inference here, not measured: no source maps KL divergence to a share
// of benchmark intelligence, and none gives a size-scaling law. The loss curve
// below is chosen so its ordering follows the KLD tables (steep below ~3.5 bpw,
// near zero from 6.5) and its 2-3 bpw losses (15-25% for an 8B) are larger
// than the half-parameter rule implies for most families. The size term
// (smaller loss for larger models) follows the 35B-A3B MoE's lower KLD than
// the 7-8B dense models at the same quant. Both are kept deliberately mild and
// the floors in minLoss stop size from making a low-bit file look lossless.

// RetentionModel names this curve. It is stamped on every estimate so a
// recorded recommendation says which curve produced its quality figure.
const RetentionModel = "bpw-size-imatrix-v1"

// referenceLoss is the share of quality lost by a ~8B dense model quantized
// to a width (effective bits per weight) without an importance matrix.
var referenceLoss = []struct{ bpw, loss float64 }{
	{1.5, 0.60}, {2.0, 0.45}, {2.5, 0.32}, {3.0, 0.20}, {3.5, 0.12},
	{4.5, 0.05}, {5.5, 0.02}, {6.5, 0.01}, {8.5, 0.005}, {16, 0},
}

const (
	// imatrixLossFactor: the weaker end of llama.cpp's imatrix effect on KLD
	// (0.74-0.90). Dynamic quants are imatrix quants with per-tensor choices;
	// they get the same factor until a local measurement says more.
	imatrixLossFactor = 0.80
	// referenceParamsB is the model size referenceLoss describes.
	referenceParamsB = 8.0
	// sizeExponent and its clamp keep the size effect mild: 0.81x the loss at
	// 16B, 0.5x from ~80B effective, 1.3x at 3.3B and below.
	sizeExponent = 0.30
	minSizeScale = 0.50
	maxSizeScale = 1.30
	minRetention = 0.20
)

func lossAt(bpw float64) float64 {
	pts := referenceLoss
	if bpw <= pts[0].bpw {
		return math.Min(0.80, pts[0].loss+(pts[0].bpw-bpw)*0.3)
	}
	for i := 1; i < len(pts); i++ {
		if bpw <= pts[i].bpw {
			a, b := pts[i-1], pts[i]
			return a.loss + (b.loss-a.loss)*(bpw-a.bpw)/(b.bpw-a.bpw)
		}
	}
	return 0
}

// minLoss is the loss no model size removes: a 1-bit file of a huge model is
// still heavily degraded.
func minLoss(bpw float64) float64 {
	switch {
	case bpw < 2.0:
		return 0.40
	case bpw < 2.5:
		return 0.25
	case bpw < 3.0:
		return 0.15
	case bpw < 3.5:
		return 0.08
	case bpw < 4.5:
		return 0.03
	default:
		return 0
	}
}

// sizeScale scales the loss by model size. A MoE counts as the geometric mean
// of its total and active parameters: its experts are many but each token sees
// few of them (an assumption, not a measurement).
func sizeScale(c Candidate) float64 {
	size := c.TotalParamsB
	if c.MoE && c.ActiveParamsB > 0 && c.ActiveParamsB < c.TotalParamsB {
		size = math.Sqrt(c.TotalParamsB * c.ActiveParamsB)
	}
	if size <= 0 {
		return 1
	}
	return clampF(math.Pow(referenceParamsB/size, sizeExponent), minSizeScale, maxSizeScale)
}

// estimatedRetention is the retention factor for one file; ok is false when
// the file's width is unknown.
func estimatedRetention(c Candidate, p QuantProfile) (float64, bool) {
	if p.BPW <= 0 {
		return 0, false
	}
	loss := math.Max(lossAt(p.BPW)*sizeScale(c), minLoss(p.BPW))
	if p.Imatrixed() {
		loss *= imatrixLossFactor
	}
	return clampF(1-loss, minRetention, 1), true
}

// measuredRetention answers a locally measured retention (for example from a
// KL-divergence run) for a file identity (QuantProfile.FileKey). Nothing
// measures yet; the launcher installs it once a measurement exists.
var measuredRetention func(fileKey string) (float64, bool)

// SetMeasuredRetention installs that lookup. nil disables it.
func SetMeasuredRetention(fn func(fileKey string) (float64, bool)) {
	measuredRetention = fn
}

// applyQualityEstimate fills a row's retention and estimated quality.
func applyQualityEstimate(r *Recommendation) {
	r.Retention, r.RetentionMeasured, r.EstimatedQuality, r.RetentionModel = 0, false, 0, ""
	if measuredRetention != nil && r.QuantProfile.FileKey != "" {
		if v, ok := measuredRetention(r.QuantProfile.FileKey); ok && v > 0 && v <= 1 {
			r.Retention, r.RetentionMeasured = v, true
		}
	}
	if !r.RetentionMeasured {
		v, ok := estimatedRetention(r.Candidate, r.QuantProfile)
		if !ok {
			return
		}
		r.Retention, r.RetentionModel = v, RetentionModel
	}
	r.EstimatedQuality = modelIntelligence(r.Candidate) * r.Retention
}

// QualityLabel describes a row's quality estimate, e.g.
// "quality ≈ 0.91× · 2.7 bpw · imatrix".
func QualityLabel(r Recommendation) string {
	if r.Retention <= 0 {
		return "quality: no estimate"
	}
	head := fmt.Sprintf("quality ≈ %.2f×", r.Retention)
	if r.RetentionMeasured {
		head = fmt.Sprintf("measured %.2f×", r.Retention)
	}
	parts := []string{head}
	if bpw := r.QuantProfile.BPW; bpw > 0 {
		format := "%.1f bpw"
		if r.QuantProfile.Source == QuantSourceNominal {
			format = "~%.1f bpw"
		}
		parts = append(parts, fmt.Sprintf(format, bpw))
	}
	switch r.QuantProfile.Imatrix {
	case "":
	case ImatrixDynamic:
		parts = append(parts, "dynamic")
	default:
		parts = append(parts, "imatrix")
	}
	return strings.Join(parts, " · ")
}

// The reference agent turn: new prompt tokens to prefill, then output tokens.
const (
	referencePromptTokens = 15000
	referenceOutputTokens = 500
)

// Prefill is compute-bound, so the estimate is a rate in billions of active
// parameters times tokens per second, by where the active weights run. Fitted
// to the October 13 README rows (one host, first long prompt of the session):
// dense 27B on GPUs 1,008-1,391 tok/s (27,000 is the low end), 35B-A3B MoE on
// GPUs 3,185-3,645 tok/s (10,000 = 3.3B x ~3,000), CPU only 27B dense 16 tok/s
// and 35B-A3B 127 tok/s (both ~400, 380 used). A 26B-A4B MoE split GPU+RAM
// measured 265 tok/s where this model gives ~200. GPU model and CPU core count
// are not modelled (review R-04): an estimate for ranking only.
const (
	gpuDensePrefillRate    = 27000.0
	gpuMoEPrefillRate      = 10000.0
	cpuPrefillRate         = 380.0
	maxPredictedPrefillTPS = 20000.0
)

// predictPrefillTPSForFit estimates prompt processing tok/s for a quant at a
// fit, splitting the active weights between GPU and CPU the way the decode
// estimate does. 0 means unknown (no parameter count).
func predictPrefillTPSForFit(b hardwareBudget, c Candidate, q QuantOption, fit string, gpuOverheadMB int) float64 {
	activeB := c.TotalParamsB
	if c.MoE && c.ActiveParamsB > 0 {
		activeB = c.ActiveParamsB
	}
	if activeB <= 0 || q.SizeGB <= 0 {
		return 0
	}
	usableVRAMGB := float64(b.totalVRAM) / 1024.0 * 0.88
	if fit == "CPU RAM" || b.gpuCount == 0 {
		usableVRAMGB = 0
	}
	var gpuB float64
	rate := gpuDensePrefillRate
	if c.MoE {
		rate = gpuMoEPrefillRate
		backbone, backboneFileGB := 1.0, q.SizeGB
		if routed, fileShare, ok := moeRoutedShares(c); ok {
			backbone, backboneFileGB = 1-routed, q.SizeGB*(1-fileShare)
		}
		fBackbone := clampF(usableVRAMGB/backboneFileGB, 0, 1)
		fExpert := 1.0
		if expertFileGB := q.SizeGB - backboneFileGB; expertFileGB > 0 {
			fExpert = clampF((usableVRAMGB-backboneFileGB)/expertFileGB, 0, 1)
		}
		gpuB = activeB * (backbone*fBackbone + (1-backbone)*fExpert)
	} else {
		f := 1.0
		if fit != "single GPU" && fit != "multi-GPU" {
			f = clampF((usableVRAMGB-float64(gpuOverheadMB)/1024)/q.SizeGB, 0, 1)
		}
		gpuB = activeB * f
	}
	cpuB := math.Max(0, activeB-gpuB)
	sec := gpuB/rate + cpuB/cpuPrefillRate
	if sec <= 0 {
		return 0
	}
	return clampF(1/sec, 0, maxPredictedPrefillTPS)
}

// referenceTurnSeconds prices the reference turn; 0 (unpriced) when either
// rate is unknown.
func referenceTurnSeconds(prefillTPS, decodeTPS float64) float64 {
	if prefillTPS <= 0 || decodeTPS <= 0 {
		return 0
	}
	return referencePromptTokens/prefillTPS + referenceOutputTokens/decodeTPS
}

// ReferenceTurnLabel shows the priced reference turn, e.g. "turn ~42 s".
func ReferenceTurnLabel(r Recommendation) string {
	s := r.ReferenceTurnSeconds
	switch {
	case s <= 0:
		return "turn: no estimate"
	case s < 600:
		return fmt.Sprintf("turn ~%.0f s", s)
	default:
		return fmt.Sprintf("turn ~%.0f min", s/60)
	}
}

// ReferenceTurnNote explains the reference turn and the quality estimate.
const ReferenceTurnNote = "A turn is 15,000 new prompt tokens plus 500 output tokens at estimated speeds. Quality ≈ is an estimate from bits per weight, model size and imatrix use, relative to the full-precision model."
