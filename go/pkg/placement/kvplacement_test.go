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
		arch        string
		want        string
	}{
		{"dense_fits_vram_gpu", 20000, false, "llama", "gpu"},
		{"big_moe_offloads_cpu", 116000, true, "qwen3moe", "cpu"},
		{"dense_too_big_still_gpu", 116000, false, "llama", "gpu"},
		{"small_moe_fits_gpu", 8000, true, "qwen3moe", "gpu"},
		// deepseek4 without flash attention grows compute scratch with real
		// token position (~98 KiB/token measured) — KV must stay on GPU so FA
		// stays enabled, even for a big offloading MoE.
		{"deepseek4_big_moe_keeps_kv_gpu", 140000, true, "deepseek4", "gpu"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := &ModelProfile{IsMoE: tc.isMoE, ModelArch: tc.arch}
			// derived per-component overhead: no CUDA probe data here, so only compute buffer is charged
			const vramOverheadMB = 3 * computeFloorMB
			if got := resolveAutoKVPlacement(caps, m, tc.totalSizeMB, vramOverheadMB); got != tc.want {
				t.Fatalf("resolveAutoKVPlacement(%dMB, moe=%v, arch=%s) = %q, want %q", tc.totalSizeMB, tc.isMoE, tc.arch, got, tc.want)
			}
		})
	}
}
