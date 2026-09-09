package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestOracleHotExpertChargeIsLocalAndPerGPU(t *testing.T) {
	rows := []preflightDevice{
		{Name: "CUDA0", ModelMB: 100, ContextMB: 20, ComputeMB: 30},
		{Name: "CUDA1", ModelMB: 110, ContextMB: 21, ComputeMB: 31},
		{Name: "Host", ModelMB: 50},
	}
	raw := append([]preflightDevice(nil), rows...)
	strategy := &placement.Strategy{HotExpertCacheSlots: 32, HotExpertCacheVRAMByGPU: map[int]int{0: 700, 1: 900}}
	got, err := addOracleHotExpertCacheCharges(rows, []detect.GPU{{Index: 0}, {Index: 1}}, strategy,
		[]string{"--moe-expert-cache", "32", "--moe-expert-cache-inserts", "2"})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(rows, raw) {
		t.Fatalf("raw oracle rows mutated: %#v", rows)
	}
	if got[0].UnaccountedMB != 700 || got[1].UnaccountedMB != 900 || got[2].UnaccountedMB != 0 {
		t.Fatalf("cache charge rows = %#v", got)
	}
}

func TestOracleHotExpertChargeFailsClosed(t *testing.T) {
	base := []preflightDevice{{Name: "CUDA0", ModelMB: 100}}
	selected := []detect.GPU{{Index: 0}, {Index: 1}}
	cases := []struct {
		name   string
		args   []string
		charge map[int]int
		want   string
	}{
		{"missing argv", nil, map[int]int{0: 1}, "argv"},
		{"slot mismatch", []string{"--moe-expert-cache", "16"}, map[int]int{0: 1}, "slots"},
		{"missing charge", []string{"--moe-expert-cache", "32"}, nil, "per-GPU"},
		{"missing row", []string{"--moe-expert-cache", "32"}, map[int]int{1: 1}, "matching oracle row"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := addOracleHotExpertCacheCharges(base, selected,
				&placement.Strategy{HotExpertCacheSlots: 32, HotExpertCacheVRAMByGPU: tc.charge}, tc.args)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestPreflightPlacementChargesCacheOnlyAfterRawOracleEvidence(t *testing.T) {
	dir := t.TempDir()
	server := filepath.Join(dir, "llama-server")
	fit := filepath.Join(dir, "llama-fit-params")
	if err := os.WriteFile(server, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fit, []byte("#!/bin/sh\necho 'CUDA0 100 20 30'\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "test", VRAMTotalMB: 10000}}, RAM: detect.RAMInfo{FreeMB: 100000}}
	model := &placement.ModelProfile{Path: filepath.Join(dir, "model.gguf"), Name: "model", TotalSizeMB: 100}
	args := []string{"llama-server", "-m", model.Path, "--ctx-size", "1", "-ub", "1", "--moe-expert-cache", "32"}
	strategy := &placement.Strategy{ContextSize: 1, UBatchSize: 1, BatchSize: 1, Parallel: 1, KVQuality: "q8_0", KVPlacement: "gpu", HotExpertCacheSlots: 32, HotExpertCacheVRAMByGPU: map[int]int{0: 9900}}
	outcome := preflightPlacement(&launchRequest{}, &backendInfo{Path: server, Tag: "llama", Dialect: "llama"}, &configForPreflight{CacheDir: dir}, caps, model, strategy, args)
	if outcome.Err != nil {
		t.Fatal(outcome.Err)
	}
	if !outcome.DoesNotFit || outcome.DeficitMB <= 0 {
		t.Fatalf("cache-on oracle did not charge deficit: %+v", outcome)
	}
	if len(outcome.Evidence.Devices) != 1 || outcome.Evidence.Devices[0].UnaccountedMB != 0 {
		t.Fatalf("raw oracle evidence was mutated: %+v", outcome.Evidence.Devices)
	}
	off := *strategy
	off.HotExpertCacheSlots = 0
	off.HotExpertCacheVRAMByGPU = nil
	offOutcome := preflightPlacement(&launchRequest{}, &backendInfo{Path: server, Tag: "llama", Dialect: "llama"}, &configForPreflight{CacheDir: filepath.Join(dir, "off")}, caps, model, &off, args[:len(args)-2])
	if offOutcome.Err != nil || offOutcome.DoesNotFit {
		t.Fatalf("cache-off oracle was charged: %v %+v", offOutcome.Err, offOutcome)
	}
}

func TestOracleCacheArgvAndStrategyMustAgree(t *testing.T) {
	rows := []preflightDevice{{Name: "CUDA0", ModelMB: 100}}
	gpus := []detect.GPU{{Index: 0}}
	for _, tc := range []struct {
		name    string
		args    []string
		slots   int
		wantErr bool
	}{
		{"off", nil, 0, false},
		{"explicit zero", []string{"--moe-expert-cache", "0"}, 0, false},
		{"equals", []string{"--moe-expert-cache=16"}, 16, false},
		{"unexpected on", []string{"--moe-expert-cache", "16"}, 0, true},
		{"invalid", []string{"--moe-expert-cache", "abc"}, 0, true},
		{"negative", []string{"--moe-expert-cache=-1"}, 0, true},
		{"missing value", []string{"--moe-expert-cache"}, 0, true},
		{"conflicting", []string{"--moe-expert-cache", "16", "--moe-expert-cache=8"}, 16, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := &placement.Strategy{HotExpertCacheSlots: tc.slots, HotExpertCacheVRAMByGPU: map[int]int{0: 10}}
			got, err := addOracleHotExpertCacheCharges(rows, gpus, s, tc.args)
			if (err != nil) != tc.wantErr {
				t.Fatalf("rows=%v err=%v wantErr=%v", got, err, tc.wantErr)
			}
			if err == nil && tc.slots == 0 && !reflect.DeepEqual(rows, got) {
				t.Fatal("disabled cache changed rows")
			}
		})
	}
	if _, err := addOracleHotExpertCacheCharges(rows, gpus, nil, []string{"--moe-expert-cache=16"}); err == nil {
		t.Fatal("unplanned cache argv accepted without strategy")
	}
}
