package placement

import (
	"github.com/raketenkater/ggrun/pkg/detect"
	"testing"
)

// The GLM5Next backend combines attention caches with KDA recurrent state even
// when the GGUF has no generic SSM bit. A branch requires that state restored.
func TestGLM5NextRecurrentCheckpointPolicy(t *testing.T) {
	for _, arch := range []string{"glm5next", "GLM5NEXT"} {
		model := &ModelProfile{ModelArch: arch, Path: "glm.gguf", SizeBytes: 1024 * 1024, ContextSize: 8192, NumLayers: 8, HiddenSize: 64}
		if !requiresScopedContextEvidence(model) {
			t.Errorf("%s compound state must not use global KV geometry", arch)
		}
		caps := &detect.Capabilities{}
		caps.CPU.Cores = 4
		caps.RAM.TotalMB = 65536
		caps.RAM.FreeMB = 60000
		strategy, err := Compute(caps, model, Options{ContextSize: 4096, Parallel: 1, BackendHelp: "--checkpoint-min-step --no-context-shift"})
		if err != nil {
			t.Fatal(err)
		}
		if !strategy.HasSSM || strategy.CheckpointMinStep != checkpointMinStepFloor {
			t.Fatalf("%s missing recurrent policy: recurrent=%v step=%d", arch, strategy.HasSSM, strategy.CheckpointMinStep)
		}
		args := strategy.Args("glm.gguf", 8081)
		if !hasAdjacentArgPlacement(args, "--checkpoint-min-step", "512") {
			t.Fatalf("branch checkpoint policy missing from argv: %v", args)
		}
	}
	if requiresScopedContextEvidence(&ModelProfile{ModelArch: "glm4"}) {
		t.Fatal("unrelated GLM architectures must not inherit recurrent policy")
	}
	// An older verified record must not bypass the newly required state policy.
	model := &ModelProfile{ModelArch: "glm5next", Path: "glm.gguf", SizeBytes: 1024 * 1024, ContextSize: 8192, NumLayers: 8, HiddenSize: 64}
	caps := &detect.Capabilities{CPU: detect.CPUInfo{Cores: 4}, RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 60000}}
	opts := Options{ContextSize: 4096, Parallel: 1, CacheDir: t.TempDir(), VerifiedConfigScopeKey: "glm-replay", BackendHelp: "--checkpoint-min-step --no-context-shift"}
	fresh, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatal(err)
	}
	vc := VerifiedConfigToRecord(opts.VerifiedConfigScopeKey, "glm.gguf", fresh, "test", "test", "", "")
	vc.HasSSM = false
	vc.CheckpointMinStep = 0
	if _, err := SaveVerifiedConfig(opts.CacheDir, vc); err != nil {
		t.Fatal(err)
	}
	restored, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatal(err)
	}
	if restored.VerifiedConfigReused || !restored.HasSSM || restored.CheckpointMinStep != 512 {
		t.Fatalf("stale GLM verified policy replayed: %+v", restored)
	}

	// A matching corrected record still takes the direct replay path.
	vc = VerifiedConfigToRecord(opts.VerifiedConfigScopeKey, "glm.gguf", restored, "test", "test", "", "")
	if _, err := SaveVerifiedConfig(opts.CacheDir, vc); err != nil {
		t.Fatal(err)
	}
	reused, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !reused.VerifiedConfigReused {
		t.Fatal("matching recurrent checkpoint policy was unnecessarily discarded")
	}

}
