package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestRecommendCPUOnlyDropsEveryGPU(t *testing.T) {
	cfg := config.Defaults()
	opts, err := parseRecommendArgs([]string{"--cpu", "--ram-budget", "12G"}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "RTX 4070", VRAMTotalMB: 12282}},
		RAM:  detect.RAMInfo{TotalMB: 212000, FreeMB: 200000},
	}
	planned, err := recommendationCapabilities(caps, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(planned.GPUs) != 0 {
		t.Fatalf("CPU-only recommendation kept GPUs: %+v", planned.GPUs)
	}
	if _, err := parseRecommendArgs([]string{"--cpu", "--gpus", "0"}, cfg); err == nil {
		t.Fatal("--cpu with --gpus accepted")
	}
}
