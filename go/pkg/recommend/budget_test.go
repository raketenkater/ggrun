package recommend

import (
	"github.com/raketenkater/ggrun/pkg/detect"
	"reflect"
	"testing"
)

func TestPlanningCapabilitiesUsesCapacityAndHonorsCeilings(t *testing.T) {
	original := detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 64 * 1024, FreeMB: 8 * 1024}, GPUs: []detect.GPU{{Index: 2, VRAMTotalMB: 12 * 1024}}}
	for _, tc := range []struct {
		name                            string
		budget, percent, headroom, want int
	}{
		{"explicit overrides percentage then headroom", 32 * 1024, 25, 4 * 1024, 28 * 1024},
		{"percentage of capacity", 0, 75, 4 * 1024, 44 * 1024},
		{"ceiling cannot invent RAM", 128 * 1024, 95, 0, 64 * 1024},
		{"exhausted", 0, 50, 64 * 1024, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := PlanningCapabilities(&original, tc.budget, tc.percent, 1024, tc.headroom)
			if got.RAM.TotalMB != tc.want || got.RAM.FreeMB != tc.want || got.GPUs[0].VRAMTotalMB != 11*1024 {
				t.Fatalf("unexpected budget: %+v", got)
			}
			busy := original
			busy.RAM.FreeMB = 1024
			if other := PlanningCapabilities(&busy, tc.budget, tc.percent, 1024, tc.headroom); !reflect.DeepEqual(other, got) {
				t.Fatal("live RAM use changed planning budget")
			}
			if tc.want == 0 && hardware(got).usableRAM != 0 {
				t.Fatal("exhausted RAM restored by fallback")
			}
			got.GPUs[0].VRAMTotalMB = 0
			if original.GPUs[0].VRAMTotalMB != 12*1024 || original.RAM.TotalMB != 64*1024 {
				t.Fatal("mutated source inventory")
			}
		})
	}
}
