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
