package main

import (
	"runtime"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestHostGrowthReserveMatchesContainmentPlatform(t *testing.T) {
	cases := []struct {
		name      string
		req       *launchRequest
		linuxWant int
	}{
		{"absent", nil, 0},
		{"configured", &launchRequest{CgroupHeadroomMB: 4096}, 4096},
		{"already reserved by recovery", &launchRequest{CgroupHeadroomMB: 4096, PlacementHostReserveMB: 4096}, 0},
		{"disabled", &launchRequest{}, 0},
		{"negative", &launchRequest{CgroupHeadroomMB: -1}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want := 0
			if runtime.GOOS == "linux" {
				want = tc.linuxWant
			}
			if got := hostGrowthReserveMB(tc.req); got != want {
				t.Fatalf("%s reserve = %d, want %d", runtime.GOOS, got, want)
			}
		})
	}
}

// The macOS packaging runner has 5375 MiB available. A 1 MiB synthetic model
// was refused because the Linux-only 4096 MiB gate reserve was charged on top
// of real CPU runtime buffers. The native Mac/Windows contract jobs run this.
func TestNonLinuxSmallCPUHostKeepsRuntimeAndExplicitLimits(t *testing.T) {
	if runtime.GOOS == "linux" {
		t.Skip("Linux intentionally keeps its containment reserve")
	}
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 8192, FreeMB: 5375}, CPU: detect.CPUInfo{Cores: 4}}
	model := &placement.ModelProfile{ModelArch: "llama", SizeBytes: 1 << 20, NumLayers: 2, HeadCountKV: 1, KeyLength: 16, ValueLength: 16, EmbeddingLength: 64, FeedForwardLength: 128, CTXTrain: 2048, ContextSize: 2048}
	req := &launchRequest{CPUMode: true, CtxFlag: "fit", CgroupHeadroomMB: 4096}
	opts := placementOptionsFromRequestCaps(req, model, &backendInfo{Tag: "llama"}, t.TempDir(), caps)
	// Exercise the cold estimate as well as the normal measured-buffer path.
	opts.RequireMeasuredBuffers = false
	strategy, err := placement.Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("tiny model refused on a small %s CPU host: %v", runtime.GOOS, err)
	}
	if strategy.PlannedHostFootprintMB <= 1 || strategy.PlannedHostFootprintMB > caps.RAM.FreeMB {
		t.Fatalf("runtime accounting lost or over budget: %d MiB", strategy.PlannedHostFootprintMB)
	}
	opts.RamBudgetMB = 1
	if _, err = placement.Compute(caps, model, opts); err == nil {
		t.Fatal("explicit insufficient RAM budget ignored")
	}
	opts.RamBudgetMB = 0
	opts.RAMHeadroomMB = caps.RAM.FreeMB
	if _, err = placement.Compute(caps, model, opts); err == nil {
		t.Fatal("explicit RAM headroom ignored")
	}
}
