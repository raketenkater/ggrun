package main

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func preflightRecoveryFixture() (*placement.ModelProfile, *detect.Capabilities, []string, *placement.Strategy) {
	model := &placement.ModelProfile{
		NumLayers:    60,
		LeadingDense: 3,
		ExpertBytes:  int64(57 * 2500 * 1024 * 1024),
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576},
		{Index: 1, VRAMTotalMB: 12288},
	}}
	ot := `blk\.(3|4|5|6)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,blk\.(7|8|9|10)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA1,exps=CPU`
	args := []string{
		"llama-server", "-b", "2048", "-ub", "512",
		"--tensor-split", "0.67,0.33", "--split-mode", "layer",
		"-ot", ot, "--n-cpu-moe", "49",
	}
	strategy := &placement.Strategy{
		BatchSize: 2048, UBatchSize: 512, TensorSplit: []float64{0.67, 0.33},
		SplitMode: "layer", OTString: ot, NCPUMoE: 49,
	}
	return model, caps, args, strategy
}

func TestAllocationOOMOutcomePreservesExactRecoveryEvidence(t *testing.T) {
	got := allocationOOMOutcome(preflightOutcome{Device: -1}, &ikAllocationOOMError{
		Device: 0, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if !got.DoesNotFit || got.Device != 0 || got.AllocMB != 2132 || got.DeficitMB != 74 || !got.IsComputeBuffer {
		t.Fatalf("allocation outcome lost exact evidence: %+v", got)
	}
}

func TestUnchangedComputeReplanLowersUBatch(t *testing.T) {
	model, caps, args, unchanged := preflightRecoveryFixture()
	next, entry, method, ok := selectChangedPreflightRecovery(args, unchanged, model, caps, preflightOutcome{
		Device: 0, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if !ok || method != "ubatch-derate" {
		t.Fatalf("compute recovery = method %q ok=%v", method, ok)
	}
	if entry == nil || entry.UBatchSize != 256 || !strings.Contains(effectiveMemoryArgsFingerprint(next), "ubatch=256") {
		t.Fatalf("compute recovery did not lower ubatch: entry=%+v args=%v", entry, next)
	}
}

func TestUnchangedWeightReplanMovesExpertLayer(t *testing.T) {
	model, caps, args, unchanged := preflightRecoveryFixture()
	next, entry, method, ok := selectChangedPreflightRecovery(args, unchanged, model, caps, preflightOutcome{
		Device: 1, AllocMB: 2500, DeficitMB: 74,
	})
	if !ok || method != "expert-derate" {
		t.Fatalf("weight recovery = method %q ok=%v", method, ok)
	}
	if entry == nil || entry.NCPUMoE != 50 || !strings.Contains(effectiveMemoryArgsFingerprint(next), "n-cpu-moe=50") {
		t.Fatalf("weight recovery did not move one expert layer: entry=%+v args=%v", entry, next)
	}
}

func TestChangedPackerResultWinsBeforeFallback(t *testing.T) {
	model, caps, args, candidate := preflightRecoveryFixture()
	candidate.UBatchSize = 256
	next, entry, method, ok := selectChangedPreflightRecovery(args, candidate, model, caps, preflightOutcome{
		Device: 0, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if !ok || method != "replanned" || entry != nil {
		t.Fatalf("changed packer result = method %q entry=%+v ok=%v", method, entry, ok)
	}
	if !strings.Contains(effectiveMemoryArgsFingerprint(next), "ubatch=256") {
		t.Fatalf("changed packer result was not selected: %v", next)
	}
}

func TestEffectiveDuplicateOverrideForbidsIdenticalRetry(t *testing.T) {
	model, caps, args, candidate := preflightRecoveryFixture()
	args = append(args, "-ub", "512") // later user value remains authoritative
	candidate.UBatchSize = 256
	_, _, _, ok := selectChangedPreflightRecovery(args, candidate, model, caps, preflightOutcome{
		Device: 0, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if ok {
		t.Fatal("recovery must reject a syntactically changed but effectively identical retry")
	}
}

func TestNoGenericWeightLeverFailsClosed(t *testing.T) {
	model := &placement.ModelProfile{NumLayers: 32}
	args := []string{"llama-server", "-ub", "512"}
	_, _, _, ok := selectChangedPreflightRecovery(args, nil, model, &detect.Capabilities{}, preflightOutcome{
		Device: 0, AllocMB: 1024, DeficitMB: 1,
	})
	if ok {
		t.Fatal("weight OOM without a placement lever must fail closed")
	}
}
