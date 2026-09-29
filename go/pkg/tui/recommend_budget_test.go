package tui

import (
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/recommend"
	"reflect"
	"testing"
)

func TestRecommendationBudgetMatchesCLIPlanning(t *testing.T) {
	t.Setenv("LLM_CACHE_DIR", t.TempDir())
	caps := &detect.Capabilities{CPU: detect.CPUInfo{Cores: 8}, RAM: detect.RAMInfo{TotalMB: 128 * 1024, FreeMB: 64 * 1024}, GPUs: []detect.GPU{{Index: 0, Name: "RTX 4070", VRAMTotalMB: 12 * 1024}}}
	m := Model{caps: caps, ramBudgetMB: 32 * 1024, ramLimitPercent: 95, ramHeadroomMB: 1024}
	m.refreshRecommendations()
	want := recommend.TopCategories(recommend.PlanningCapabilities(caps, 32*1024, 95, 0, 1024), 4)
	if !reflect.DeepEqual(m.recommendationGroups, want) {
		t.Fatal("TUI ignored configured RAM ceiling")
	}
	before := m.recommendationGroups
	caps.RAM.FreeMB = 1024
	m.refreshRecommendations()
	if !reflect.DeepEqual(before, m.recommendationGroups) {
		t.Fatal("running another model changed capacity-based recommendations")
	}
	unrestricted := recommend.TopCategories(recommend.PlanningCapabilities(caps, 0, 95, 0, 1024), 4)
	if reflect.DeepEqual(before, unrestricted) {
		t.Fatal("fixture does not distinguish restricted recommendations")
	}
}
