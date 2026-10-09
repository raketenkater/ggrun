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
	// Rows are found by model name: the daily regeneration may resolve a
	// different upstream repository for the same model, which must not fail
	// the catalog job's validation step.
	models := map[string]Candidate{}
	for _, row := range doc.Candidates {
		models[row.Name] = inferParams(row)
	}
	for _, tc := range []struct {
		name    string
		model   string
		ramMB   int
		vramMB  int
		wantFit bool
	}{
		{"pro_consumer", "Xiaomi MiMo-V2.6-Pro", 32 * 1024, 12 * 1024, false},
		{"flash_consumer", "Xiaomi MiMo-V2.6-Flash", 32 * 1024, 12 * 1024, false},
		{"pro_workstation", "Xiaomi MiMo-V2.6-Pro", 256 * 1024, 48 * 1024, false},
		{"flash_workstation", "Xiaomi MiMo-V2.6-Flash", 256 * 1024, 48 * 1024, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			candidate, found := models[tc.model]
			if !found {
				t.Fatalf("missing catalog entry for %s", tc.model)
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

// A companion artifact (MTP/DFlash/draft head, projector, adapter) or a
// partial shard set listed as a main-model quant is far smaller than the
// model's real quants: complete quants span ~1.5 to 16 bits per weight
// (about 1:11), a draft head is ~1:100. Holds for any correct regeneration.
func TestCatalogQuantsAreCompleteMainModels(t *testing.T) {
	var doc catalogDoc
	if err := json.Unmarshal(catalogJSON, &doc); err != nil {
		t.Fatal(err)
	}
	for _, row := range doc.Candidates {
		var largest int64
		for _, q := range row.Quants {
			if q.SizeBytes > largest {
				largest = q.SizeBytes
			}
		}
		for _, q := range row.Quants {
			if q.SizeBytes > 0 && q.SizeBytes*25 < largest {
				t.Errorf("%s (%s): quant %s is %d bytes against %d for the largest; a companion or partial artifact is listed as a main-model quant",
					row.Name, row.Repo, q.Name, q.SizeBytes, largest)
			}
		}
	}
}
