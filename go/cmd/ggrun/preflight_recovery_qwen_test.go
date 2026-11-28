package main

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// Drives the recovery selector with the exact Qwen3.8-Flash-Next argv and
// oracle-planned outcome that fails to launch, and pins what it actually does.
// Six fixes were attempted and reverted from inference about which branch runs;
// this test exists so the next change starts from observation instead.
//
// What it establishes today:
//   - the expert lever is selected and does real work: CUDA2 4 -> 3 layers,
//     n-cpu-moe 22 -> 23, so recovery is not stalled on a no-op;
//   - the ubatch lever is available and would give 256 -> 64, so it is not
//     disqualified by the synthesised AllocMB as previously recorded;
//   - the pins are whole-layer (the pattern includes `down`), so the expert
//     pricing is legitimate rather than a partial-pin mis-estimate.
//
// The open question is therefore none of those: it is why the re-measured
// deficit falls only ~7 MiB per round when a whole expert layer leaves CUDA2.
// Each round re-plans and re-measures, so the next step is to compare the
// per-device ledger before and after one accepted derate.
func TestQwenFlashNextRecoverySelectorShape(t *testing.T) {
	ot := `blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=CUDA1,blk\.(13|14|15|16|17|18)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=CUDA0,blk\.(19|20|21|22)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=CUDA2,exps=CPU`
	args := []string{
		"llama-server", "--ctx-size", "262144", "-b", "2048", "-ub", "256",
		"--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
		"--kv-offload", "--parallel", "1", "-ngl", "999",
		"--tensor-split", "0.27,0.59,0.14", "--split-mode", "layer",
		"-ot", ot, "--n-cpu-moe", "22",
	}
	model := &placement.ModelProfile{
		NumLayers: 48, LeadingDense: 0, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
		ExpertBytes: int64(52) * 1024 * 1024 * 1024,
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 12282}, {Index: 1, VRAMTotalMB: 24564}, {Index: 2, VRAMTotalMB: 12288},
	}}
	outcome := preflightOutcome{
		Device: 2, AllocMB: 101, AllocMBMeasured: false, DeficitMB: 101,
		IsComputeBuffer: false, DoesNotFit: true,
		Evidence: memoryPlanEvidence{Level: memoryEvidenceOraclePlanned},
	}

	if got := placement.CurrentUBatch(args); got != 256 {
		t.Fatalf("fixture ubatch %d, want 256", got)
	}
	if got := placement.CurrentGPUExpertLayers(args, 2); got != 4 {
		t.Fatalf("fixture CUDA2 expert layers %d, want 4", got)
	}

	nextArgs, entry, method, ok := selectChangedPreflightRecovery(args, nil, model, caps, "", outcome)
	if !ok || method != "expert-derate" || entry == nil {
		t.Fatalf("selector -> ok=%v method=%q entry=%v", ok, method, entry)
	}
	if got := placement.CurrentGPUExpertLayers(nextArgs, 2); got != 3 {
		t.Fatalf("CUDA2 expert layers %d after recovery, want 3", got)
	}
	if got := entry.NCPUMoE; got != 23 {
		t.Fatalf("n-cpu-moe %d after recovery, want 23", got)
	}
	if got := placement.CurrentUBatch(nextArgs); got != 256 {
		t.Fatalf("expert recovery changed ubatch to %d; it should not", got)
	}

	// The ubatch lever is available for this shape, contrary to an earlier
	// recorded diagnosis. It is simply not selected while the expert lever
	// reports success first.
	ubatchArgs, ubatchEntry, ubatchOK := placement.DerateCUDAOOMArgsForDeficit(args, model, caps, "", 2, 101, 101, true)
	if !ubatchOK || ubatchEntry == nil || ubatchEntry.UBatchSize >= 256 {
		t.Fatalf("ubatch lever unavailable: ok=%v entry=%v", ubatchOK, ubatchEntry)
	}
	if got := placement.CurrentUBatch(ubatchArgs); got >= 256 {
		t.Fatalf("ubatch lever produced %d, want a reduction", got)
	}
}

// The selector hands the launch's dialect to the expert derate: the same
// Qwen argv served by a Vulkan build keeps Vulkan device names after recovery.
func TestRecoverySelectorDerateKeepsVulkanDeviceNames(t *testing.T) {
	ot := `blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=Vulkan1,blk\.(13|14|15|16|17|18)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=Vulkan0,blk\.(19|20|21|22)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=Vulkan2,exps=CPU`
	want := strings.Replace(ot, `blk\.(19|20|21|22)\.`, `blk\.(19|20|21)\.`, 1)
	args := []string{
		"llama-server", "--ctx-size", "262144", "-b", "2048", "-ub", "256",
		"--parallel", "1", "-ngl", "999", "--split-mode", "layer",
		"-ot", ot, "--n-cpu-moe", "22",
	}
	model := &placement.ModelProfile{NumLayers: 48, ExpertBytes: int64(52) * 1024 * 1024 * 1024}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24564}, {Index: 1, VRAMTotalMB: 32752}, {Index: 2, VRAMTotalMB: 12288}}}
	outcome := preflightOutcome{Device: 2, AllocMB: 101, DeficitMB: 101, DoesNotFit: true}

	nextArgs, entry, method, ok := selectChangedPreflightRecovery(args, nil, model, caps, "vulkan", outcome)
	if !ok || method != "expert-derate" || entry == nil {
		t.Fatalf("selector -> ok=%v method=%q entry=%v", ok, method, entry)
	}
	idx := -1
	for i, arg := range nextArgs {
		if arg == "-ot" {
			idx = i
		}
	}
	if idx < 0 || idx+1 >= len(nextArgs) || nextArgs[idx+1] != want {
		t.Fatalf("vulkan recovery -ot = %v\nwant %s", nextArgs, want)
	}
	if entry.OTString != want {
		t.Fatalf("derated cache entry -ot %q, want %q", entry.OTString, want)
	}
}
