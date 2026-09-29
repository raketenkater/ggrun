package main

import (
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"testing"
)

func TestRecommendRestrictionsAndValidation(t *testing.T) {
	cfg := &config.Config{RamBudget: "16G", RAMLimitPercent: 95}
	opts, err := parseRecommendArgs([]string{"--gpus", "2", "--ram-budget=32G", "--vram-headroom", "1G", "-n3", "--json"}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if opts.ramBudgetMB != 32768 || opts.limit != 3 || !opts.json {
		t.Fatalf("options: %+v", opts)
	}
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 128 * 1024, FreeMB: 4096}, GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24 * 1024}, {Index: 2, VRAMTotalMB: 12 * 1024}}}
	got, err := recommendationCapabilities(caps, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.GPUs) != 1 || got.GPUs[0].Index != 2 || got.GPUs[0].VRAMTotalMB != 11*1024 || got.RAM.TotalMB != 32*1024 {
		t.Fatalf("restrictions ignored: %+v", got)
	}
	if len(caps.GPUs) != 2 || caps.GPUs[1].VRAMTotalMB != 12*1024 {
		t.Fatal("changed detected GPUs")
	}
	opts.gpus = "1"
	if _, err := recommendationCapabilities(caps, opts); err == nil {
		t.Fatal("accepted nonexistent physical GPU")
	}
	inherited, err := parseRecommendArgs(nil, cfg)
	if err != nil || inherited.ramBudgetMB != 16*1024 {
		t.Fatalf("config budget lost: %+v %v", inherited, err)
	}
	for _, args := range [][]string{{"--unknown"}, {"--ram-budget"}, {"--ram-budget=-1"}, {"--gpus=-1"}, {"--gpus=0,0"}, {"--ram-limit-percent=101"}, {"-n0"}, {"nonsense"}, {"--first", "--json"}} {
		if _, err := parseRecommendArgs(args, cfg); err == nil {
			t.Errorf("accepted invalid arguments %v", args)
		}
	}
}
