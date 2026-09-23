package recommend

import (
	"regexp"
	"sort"
	"strconv"
)

// Since the 2026-09-10 catalog refresh the Artificial Analysis rows arrive
// without parameter counts, so every candidate carried total_params_b = 0,
// active_params_b = 0 and moe = false (MiniMax-M3 with 128 experts included).
// predictDecodeTPS returns 0 without parameters, which silently removed speed
// from ranking: "Best overall" collapsed into "Smartest", "Fastest" came out
// empty, and 175-200 GB models that crawl on offload led the list.
//
// The catalog still carries each model's quant sizes and GGUF geometry, which
// determine both numbers independently of any third-party row.

// inferParams fills missing parameter counts and the MoE flag from quant
// sizes and GGUF geometry. Explicit catalog values are kept.
func inferParams(c Candidate) Candidate {
	if c.Experts > 1 && c.ExpertUsed > 0 && c.ExpertUsed < c.Experts {
		c.MoE = true
	}
	if total, active, ok := paramsFromName(c.Name); ok {
		if c.TotalParamsB <= 0 {
			c.TotalParamsB = total
		}
		if c.ActiveParamsB <= 0 && active < total {
			c.ActiveParamsB = active
			c.MoE = true
		}
	}
	if c.TotalParamsB <= 0 {
		c.TotalParamsB = paramsFromQuantSizes(c.Quants)
	}
	if c.MoE && c.ActiveParamsB <= 0 && c.TotalParamsB > 0 {
		// Hybrid layouts (Mamba/attention mixes) break the per-layer formula;
		// a routed model reading most of its weights per token is not a real
		// reading of the geometry, so leave speed unpredicted instead.
		if active := activeParamsFromGeometry(c); active < c.TotalParamsB*0.5 {
			c.ActiveParamsB = active
		}
	}
	return c
}

// "397B-A17B", "550B A55B": vendors put exact counts in the name.
var nameParams = regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)B[- ]A(\d+(?:\.\d+)?)B\b`)

func paramsFromName(name string) (total, active float64, ok bool) {
	m := nameParams.FindStringSubmatch(name)
	if m == nil {
		return 0, 0, false
	}
	total, _ = strconv.ParseFloat(m[1], 64)
	active, _ = strconv.ParseFloat(m[2], 64)
	return total, active, total > 0 && active > 0
}

// paramsFromQuantSizes estimates total parameters (billions) as the median of
// size / bytes-per-weight across the published quants. The median keeps one
// mislabelled or partially listed file from moving the estimate.
func paramsFromQuantSizes(quants []QuantOption) float64 {
	var est []float64
	for _, q := range quants {
		if bpw := bytesPerWeight(q.Name); q.SizeGB > 0 && bpw > 0 {
			est = append(est, q.SizeGB/bpw)
		}
	}
	if len(est) == 0 {
		return 0
	}
	sort.Float64s(est)
	return est[len(est)/2]
}

// activeParamsFromGeometry counts what one token reads: exp_used of the
// experts' (gate, up, down) matrices of embd x exp_ff in every MoE layer, plus
// the always-read tensors - attention, shared experts and dense leading
// layers. Embeddings are left out: a token reads one row, not the matrix.
// It does not depend on the size-derived total, which can land just under the
// expert tensors alone (MiniMax-M3). Returns 0 when the geometry is incomplete,
// which leaves speed unpredicted rather than guessed.
func activeParamsFromGeometry(c Candidate) float64 {
	moeLayers := c.Layers - c.LeadingDense
	if moeLayers <= 0 || c.Embedding <= 0 || c.ExpertFF <= 0 || c.Experts <= 0 || c.ExpertUsed <= 0 {
		return 0
	}
	embd := float64(c.Embedding)
	routed := float64(moeLayers) * float64(c.ExpertUsed) * 3 * embd * float64(c.ExpertFF)
	shared := float64(moeLayers) * 3 * embd * float64(c.ExpertShFF)
	dense := float64(c.LeadingDense) * 3 * embd * float64(c.FeedForward)
	// Q and O projections are embd x embd; K and V are embd x (kv heads x head
	// width). Without head geometry assume full-width K/V (an upper bound).
	kvWidth := embd
	if c.HeadCountKV > 0 && c.KeyLength > 0 {
		kvWidth = float64(c.HeadCountKV * c.KeyLength)
	}
	attention := float64(c.Layers) * (2*embd*embd + 2*embd*kvWidth)
	return (routed + shared + dense + attention) / 1e9
}
