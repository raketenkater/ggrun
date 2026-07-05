package placement

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestResolveAutoKVPlacement(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{{VRAMTotalMB: 24576}, {VRAMTotalMB: 12288}, {VRAMTotalMB: 12288}}} // 48G
	cases := []struct {
		name        string
		totalSizeMB int
		isMoE       bool
		want        string
	}{
		{"dense_fits_vram_gpu", 20000, false, "gpu"},
		{"big_moe_offloads_cpu", 116000, true, "cpu"},
		{"dense_too_big_still_gpu", 116000, false, "gpu"},
		{"small_moe_fits_gpu", 8000, true, "gpu"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := &ModelProfile{IsMoE: tc.isMoE}
			// derived per-component overhead: 3 GPUs × (600 CUDA + 1024 compute)
			const vramOverheadMB = 3 * (600 + 1024)
			if got := resolveAutoKVPlacement(caps, m, tc.totalSizeMB, vramOverheadMB); got != tc.want {
				t.Fatalf("resolveAutoKVPlacement(%dMB, moe=%v) = %q, want %q", tc.totalSizeMB, tc.isMoE, got, tc.want)
			}
		})
	}
}
