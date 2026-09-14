package main

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func fitTestModel(ctxTrain, sizeMB int) *placement.ModelProfile {
	return &placement.ModelProfile{
		Path: "m.gguf", TotalSizeMB: sizeMB, SizeBytes: int64(sizeMB) * 1024 * 1024,
		NumLayers: 32, ContextSize: ctxTrain, CTXTrain: ctxTrain,
		HiddenSize: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
}

// placementOptionsFromRequestCaps reads backend identity fields, so tests supply
// a minimal real backend rather than nil.
func fitTestBackend() *backendInfo {
	return &backendInfo{Path: "/usr/bin/llama-server", Dialect: "llama", Tag: "llama", Identity: "test"}
}

func fitTestCaps(vramMB int) *detect.Capabilities {
	return &detect.Capabilities{
		GPUs: []detect.GPU{{
			Index: 0, Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: vramMB, VRAMUsedMB: 300,
			BandwidthMBps: 15760, MemBandwidthMBps: 504048,
		}},
		RAM: detect.RAMInfo{TotalMB: 262144, FreeMB: 200000},
		CPU: detect.CPUInfo{Cores: 24},
	}
}

// Claude Code's default context used to be the model's native window outright,
// so "fit" meant "max" and a KV cache far too large for VRAM was planned and
// then offloaded to the host. The default must be bounded by what fits.
func TestClaudeCodeDefaultContextIsBoundedByVRAM(t *testing.T) {
	model := fitTestModel(262144, 7000)
	caps := fitTestCaps(12282) // no room for a 262144 KV alongside 7 GB of weights
	req := &launchRequest{ClaudeCode: true, CtxFlag: "fit"}

	opts := placementOptionsFromRequestCaps(req, model, fitTestBackend(), t.TempDir(), caps)
	if opts.ContextSize != 0 {
		t.Fatalf("fit was pre-resolved to %d before placement; want the shared automatic resolver", opts.ContextSize)
	}
	strategy, err := placement.Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("compute fit: %v", err)
	}
	if strategy.ContextSize >= model.CTXTrain {
		t.Errorf("Claude Code fit chose %d (native %d); it ignored the VRAM budget",
			strategy.ContextSize, model.CTXTrain)
	}
	if strategy.ContextSize <= 0 {
		t.Fatalf("no context chosen: %d", strategy.ContextSize)
	}
	if !strategy.ContextAuto || strategy.ContextFitTier != "gpu-resident" {
		t.Fatalf("fit did not return a GPU-resident automatic boundary: %+v", strategy)
	}
}

// Ample VRAM must retain the native window per agent, not divide one native
// window across all concurrent slots.
func TestClaudeCodeDefaultContextKeepsNativeWhenVRAMAllows(t *testing.T) {
	model := fitTestModel(65536, 4000)
	caps := fitTestCaps(24564)
	req := &launchRequest{ClaudeCode: true, CtxFlag: "fit"}

	opts := placementOptionsFromRequestCaps(req, model, fitTestBackend(), t.TempDir(), caps)
	if opts.ContextSize != 0 || opts.AutoContextMax != 1048576 {
		t.Fatalf("fit options = ctx %d cap %d, want unresolved ctx and total workload cap %d",
			opts.ContextSize, opts.AutoContextMax, 1048576)
	}
	strategy, err := placement.Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("compute fit: %v", err)
	}
	if strategy.Parallel != 4 || strategy.ContextSize != 4*model.CTXTrain {
		t.Errorf("context=%d slots=%d, want four native %d-token windows; VRAM was ample", strategy.ContextSize, strategy.Parallel, model.CTXTrain)
	}
}

// An explicit context is the user's call and must never be lowered, however
// badly it fits -- the launcher warns instead.
func TestExplicitContextIsNeverLowered(t *testing.T) {
	model := fitTestModel(262144, 7000)
	caps := fitTestCaps(12282)
	req := &launchRequest{ClaudeCode: true, CtxFlag: "262144"}

	opts := placementOptionsFromRequestCaps(req, model, fitTestBackend(), t.TempDir(), caps)
	if opts.ContextSize != 262144 {
		t.Errorf("explicit --ctx 262144 became %d; a user override must stand", opts.ContextSize)
	}
}

// "max" is an explicit choice too, and must survive intact.
func TestExplicitMaxContextIsNeverLowered(t *testing.T) {
	model := fitTestModel(262144, 7000)
	caps := fitTestCaps(12282)
	req := &launchRequest{ClaudeCode: true, CtxFlag: "max"}

	opts := placementOptionsFromRequestCaps(req, model, fitTestBackend(), t.TempDir(), caps)
	if opts.ContextSize != model.CTXTrain {
		t.Errorf("--ctx max became %d, want the native %d", opts.ContextSize, model.CTXTrain)
	}
}

// A verified config replays a whole plan and never re-runs placement, so a
// record written by an older planner must not survive a planner change. The
// scope key is bound to planLogicVersion for exactly that reason.
func TestVerifiedConfigScopeKeyIsBoundToThePlanner(t *testing.T) {
	model := fitTestModel(262144, 7000)
	caps := fitTestCaps(24564)
	req := &launchRequest{ClaudeCode: true, CtxFlag: "fit"}

	key := verifiedConfigScopeKey(req, model, fitTestBackend(), caps)
	if key == "" {
		t.Fatal("no scope key produced")
	}
	if !strings.HasSuffix(key, "-plan"+planLogicVersion) {
		t.Errorf("scope key %q is not bound to the planner version; stored plans would outlive a planner change", key)
	}
}

// --no-cached-config must still disable the record entirely.
func TestVerifiedConfigScopeKeyEmptyWhenCacheDisabled(t *testing.T) {
	model := fitTestModel(262144, 7000)
	req := &launchRequest{ClaudeCode: true, CtxFlag: "fit", NoCachedConfig: true}
	if key := verifiedConfigScopeKey(req, model, fitTestBackend(), fitTestCaps(24564)); key != "" {
		t.Errorf("scope key = %q, want empty with --no-cached-config", key)
	}
}
