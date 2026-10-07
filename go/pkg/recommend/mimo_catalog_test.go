package recommend

import (
	"encoding/json"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestMiMoCatalogRequiresMainModelCapacity(t *testing.T) {
	var doc catalogDoc
	if err := json.Unmarshal(catalogJSON, &doc); err != nil {
		t.Fatal(err)
	}
	models := map[string]Candidate{}
	for _, row := range doc.Candidates {
		models[row.Repo] = inferParams(row)
	}
	for _, tc := range []struct {
		name    string
		repo    string
		ramMB   int
		vramMB  int
		wantFit bool
	}{
		{"pro_consumer", "pcuenq/MiMo-V2.6-Pro-RL-GGUF", 32 * 1024, 12 * 1024, false},
		{"flash_consumer", "ggml-org/MiMo-V2.6-Flash-RL-GGUF", 32 * 1024, 12 * 1024, false},
		{"pro_workstation", "pcuenq/MiMo-V2.6-Pro-RL-GGUF", 256 * 1024, 48 * 1024, false},
		{"flash_workstation", "ggml-org/MiMo-V2.6-Flash-RL-GGUF", 256 * 1024, 48 * 1024, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			candidate, found := models[tc.repo]
			if !found {
				t.Fatalf("missing catalog entry for %s", tc.repo)
			}
			caps := &detect.Capabilities{
				OS:       "linux",
				RAM:      detect.RAMInfo{TotalMB: tc.ramMB, FreeMB: tc.ramMB},
				CPU:      detect.CPUInfo{Cores: 8},
				GPUs:     []detect.GPU{{Name: "Test GPU", VRAMTotalMB: tc.vramMB}},
				Backends: []detect.Backend{{Name: "cuda", Path: "/bin/llama-server-cuda"}},
			}
			rec, fit := evaluate(caps, candidate)
			if fit != tc.wantFit {
				t.Fatalf("fit=%v want %v, recommendation=%+v", fit, tc.wantFit, rec)
			}
			if fit && (rec.QuantSizeGB < 100 || rec.QuantName != "MXFP4" && rec.QuantName != "Q2_K") {
				t.Fatalf("companion artifact offered as a main model: %+v", rec)
			}
		})
	}
}
