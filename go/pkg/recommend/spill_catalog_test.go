package recommend

import (
	"encoding/json"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Measured October 8: Qwen3.8-27B UD-IQ3_XXS on one RTX 4070 with 32 GiB RAM
// kept 52% of its weights on the GPU and decoded 4.2-4.8 tok/s; the
// recommender had predicted 31.5, as if fully resident.
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
	rec, ok := evaluateWithSelector(caps, candidate, nil, betterByScore)
	if !ok || rec.Fit != "GPU plus RAM" {
		t.Fatalf("unexpected pick: ok=%v %s %s", ok, rec.QuantName, rec.Fit)
	}
	if rec.PredictedTPS > 12 {
		t.Fatalf("%s predicted %.1f tok/s for a dense model that spills to RAM; measured 4.2-4.8", rec.QuantName, rec.PredictedTPS)
	}
	// Resident on one 24 GiB card it keeps the measured-close prediction.
	caps.GPUs = []detect.GPU{{Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24564}}
	caps.RAM = detect.RAMInfo{TotalMB: 65536, FreeMB: 65536}
	rec, ok = evaluateWithSelector(caps, candidate, nil, betterByScore)
	if !ok || rec.Fit != "single GPU" || rec.PredictedTPS < 25 {
		t.Fatalf("resident prediction changed: ok=%v %s %s %.1f", ok, rec.QuantName, rec.Fit, rec.PredictedTPS)
	}
}
