package recommend

import (
	"testing"

	"github.com/raketenkater/llm-server/pkg/detect"
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
		if row.Repo == "" || row.Fit == "" || row.BackendHint == "" {
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
		if row.SizeGB > 10 {
			t.Fatalf("CPU recommendation is too large: %#v", row)
		}
	}
}
