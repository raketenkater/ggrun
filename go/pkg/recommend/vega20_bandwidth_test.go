package recommend

import "testing"

// Vega 20 cards (4,096-bit HBM2 at 2.0 Gbps) are priced at 1,024 GB/s, by the
// names their drivers report; other cards keep their entries.
func TestVega20MemoryBandwidth(t *testing.T) {
	for _, name := range []string{
		"AMD Instinct MI60 / MI50",
		"AMD Instinct MI60 / MI50 (RADV VEGA20)",
		"AMD Radeon Pro VII / Radeon Instinct MI50 (Vega 20)",
		"AMD Radeon VII",
		"AMD Radeon Pro VII",
	} {
		if got := vramBandwidthGBps(name); got != 1024 {
			t.Errorf("%s: %.0f GB/s, want 1024", name, got)
		}
	}
	for name, want := range map[string]float64{"AMD Instinct MI300X": 5300, "AMD Instinct MI210": 1600, "NVIDIA GeForce RTX 3090 Ti": 1008, "Unknown GPU": defaultVRAMBWGBps} {
		if got := vramBandwidthGBps(name); got != want {
			t.Errorf("%s: %.0f GB/s, want %.0f", name, got, want)
		}
	}
}
