package main

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

const canaryOOMLog = `srv  llama_server: model loaded
CUDA error: out of memory
  current device: 0, in function alloc at ggml-cuda.cu:464
/src/ggml-cuda.cu:139: CUDA error`

// A CUDA OOM in the first requests after load ended the launch at verification
// and taught nothing, so every relaunch re-derived the plan that died. It must
// record a reserve under the failed plan's scope, and a repeat must stack.
func TestVerificationOOMRecordsAStackingReserve(t *testing.T) {
	cfg := &config.Config{CacheDir: t.TempDir()}
	model := &placement.ModelProfile{Path: "/models/m.gguf", Name: "m", ModelArch: "test", SizeBytes: 1234,
		RoutedExpertLayerBytes: []int64{242 << 20, 240 << 20}}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 12282}},
		RAM: detect.RAMInfo{TotalMB: 32768}, CPU: detect.CPUInfo{Model: "cpu", Cores: 8}}
	be := &backendInfo{Tag: "ik_llama", Identity: "ik-build"}
	req := &launchRequest{CtxFlag: "262144", KVQuality: "q8_0", KVPlacement: "gpu", Parallel: 1}
	req.ProfilePolicyIdentity = requestedLaunchPolicyIdentity(req, model)
	strategy := &placement.Strategy{Type: placement.MoEOffload, ContextSize: 262144, UBatchSize: 512,
		KVQuality: "q8_0", KVPlacement: "gpu", Parallel: 1}
	args := []string{"llama-server", "-m", model.Path, "-ub", "512"}
	tag := scopedProbeBackendTagForStrategy(req, model, be, strategy)
	growth := func() int {
		return placement.RuntimeGraphGrowthByGPU(cfg.CacheDir, model, 262144, 512, "q8_0", "gpu", tag, caps.GPUs, 1)[0]
	}

	msg, ok := learnFromVerificationOOM(req, cfg, model, be, caps, caps, strategy, args, canaryOOMLog)
	if !ok || !strings.Contains(msg, "Launch again") {
		t.Fatalf("verification OOM not learned: ok=%v msg=%q", ok, msg)
	}
	first := growth()
	if first < 242 {
		t.Fatalf("recorded reserve %d MiB, want at least one 242 MiB layer", first)
	}
	if _, ok := learnFromVerificationOOM(req, cfg, model, be, caps, caps, strategy, args, canaryOOMLog); !ok {
		t.Fatal("repeat OOM not learned")
	}
	if second := growth(); second <= first {
		t.Fatalf("repeat reserve %d MiB did not grow past %d MiB", second, first)
	}
}

func TestVerificationFailureWithoutOOMLearnsNothing(t *testing.T) {
	cfg := &config.Config{CacheDir: t.TempDir()}
	model := &placement.ModelProfile{Path: "/models/m.gguf", ModelArch: "test"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 12282}}}
	strategy := &placement.Strategy{ContextSize: 4096, UBatchSize: 512, Parallel: 1}
	for _, log := range []string{
		"srv  llama_server: model loaded\nfunctional canary did not get a bounded answer\n",
		"CUDA error: out of memory\n  current device: 0, in function alloc at x.cu:1\n", // still loading: placement, not runtime
	} {
		if msg, ok := learnFromVerificationOOM(&launchRequest{}, cfg, model, &backendInfo{}, caps, caps, strategy, nil, log); ok {
			t.Fatalf("non-runtime failure was learned as an OOM: %q", msg)
		}
	}
}
