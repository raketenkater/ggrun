package recommend

import (
	"encoding/json"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Measured October 8: Qwen3.8-27B UD-IQ3_XXS on one RTX 4070 with 32 GiB RAM
// kept 52% of its weights on the GPU and decoded 4.2-4.8 tok/s; the
// recommender had predicted 31.5, as if fully resident, then 8.2.
func TestSpilledDenseFitPredictsTheSpill(t *testing.T) {
	var doc catalogDoc
	if err := json.Unmarshal(catalogJSON, &doc); err != nil {
		t.Fatal(err)
	}
	var candidate Candidate
	for _, row := range doc.Candidates {
		if row.Name == "Alibaba Qwen3.8 27B (Xhigh)" {
			candidate = inferParams(row)
		}
	}
	if candidate.Name == "" {
		t.Skip("catalog no longer lists Qwen3.8 27B")
	}
	caps := &detect.Capabilities{
		OS:       "linux",
		RAM:      detect.RAMInfo{TotalMB: 32768, FreeMB: 32768},
		CPU:      detect.CPUInfo{Cores: 14},
		GPUs:     []detect.GPU{{Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12282}},
		Backends: []detect.Backend{{Name: "cuda", Path: "/bin/llama-server-cuda"}},
	}
	onlyIQ3 := func(q QuantOption) bool { return q.Name == "UD-IQ3_XXS" }
	spilled, ok := evaluateWithSelector(caps, candidate, onlyIQ3, betterByScore)
	if !ok || spilled.Fit != "GPU plus RAM" {
		t.Fatalf("UD-IQ3_XXS on the 4070: ok=%v fit %s, want GPU plus RAM", ok, spilled.Fit)
	}
	if spilled.PredictedTPS < 3.5 || spilled.PredictedTPS > 5.5 || spilled.SpeedTier != 0 {
		t.Fatalf("UD-IQ3_XXS spilled predicted %.1f tok/s (tier %d); measured 4.2-4.8", spilled.PredictedTPS, spilled.SpeedTier)
	}
	// The practical pick is then a quant that stays on the card. Measured
	// October 9 (UD-IQ2_XXS, ik_llama 1fddd12 + #2356): ~28 tok/s and every task
	// correct in 58-382 s, against 601-1,610 s for the spilled UD-IQ3_XXS.
	rec, ok := evaluateWithSelector(caps, candidate, nil, betterByScore)
	if !ok || rec.Fit != "single GPU" {
		t.Fatalf("practical pick: ok=%v %s %s %.1f tok/s, want a quant resident on the card", ok, rec.QuantName, rec.Fit, rec.PredictedTPS)
	}
	// Resident on one 24 GiB card it keeps the measured-close prediction.
	caps.GPUs = []detect.GPU{{Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24564}}
	caps.RAM = detect.RAMInfo{TotalMB: 65536, FreeMB: 65536}
	rec, ok = evaluateWithSelector(caps, candidate, nil, betterByScore)
	if !ok || rec.Fit != "single GPU" || rec.PredictedTPS < 25 {
		t.Fatalf("resident prediction changed: ok=%v %s %s %.1f", ok, rec.QuantName, rec.Fit, rec.PredictedTPS)
	}
}
