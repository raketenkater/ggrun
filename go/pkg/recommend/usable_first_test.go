package recommend

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Measured October 8 on CPU with 16 GiB: Qwen3.8-27B UD-IQ2_XXS decoded ~3 tok/s
// and prefilled ~13 tok/s, so a Claude Code task timed out after an hour, while
// Qwen3.6-35B-A3B UD-IQ2_XXS finished its tasks in minutes. Best overall must
// not pick the slow dense model over a usable one; Smartest still may.
func TestBestOverallRanksUsableModelsFirst(t *testing.T) {
	t.Setenv("LLM_CACHE_DIR", t.TempDir())
	caps := &detect.Capabilities{OS: "linux", RAM: detect.RAMInfo{TotalMB: 16384, FreeMB: 16384},
		CPU: detect.CPUInfo{Cores: 14}, HostMemoryBandwidthMBps: 26273}
	const slow = "Alibaba Qwen3.8 27B (Xhigh)"
	var slowRec Recommendation
	for _, r := range allRecommendationsBalanced(caps) {
		if r.Name == slow {
			slowRec = r
		}
	}
	if slowRec.Name == "" || slowRec.PredictedTPS <= 0 || slowRec.PredictedTPS >= usableTPS {
		t.Skipf("catalog no longer has a slow %s on this budget: %+v", slow, slowRec)
	}
	cats := TopCategories(caps, 3)
	for name, rows := range map[string][]Recommendation{"overall": cats.Balanced, "top": Top(caps, 3)} {
		if len(rows) == 0 || rows[0].PredictedTPS < usableTPS {
			t.Fatalf("%s picked a model below the usable floor: %+v", name, rows)
		}
		for i := 1; i < len(rows); i++ {
			if rows[i-1].SpeedTier == 0 && rows[i].SpeedTier != 0 {
				t.Fatalf("%s ranked slow %s above usable %s", name, rows[i-1].Name, rows[i].Name)
			}
		}
	}
	if len(cats.Smartest) == 0 || cats.Smartest[0].Name != slow {
		t.Fatalf("smartest must keep the higher-intelligence slow model: %+v", cats.Smartest)
	}
}
