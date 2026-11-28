package placement

import (
	"strings"
	"testing"
)

// The ROCm dialect (a HIP build, set only for an AMD --gpus selection) names
// devices ROCm<i> everywhere placement writes one, and the recovery derate
// parses and rebuilds a ROCm -ot in that dialect. CUDA-family tags keep
// today's names.
func TestROCmDialectDeviceNames(t *testing.T) {
	if got := deviceName("rocm", 1); got != "ROCm1" {
		t.Fatalf("deviceName = %q", got)
	}
	if got := draftDeviceName("rocm", 0); got != "ROCm0" {
		t.Fatalf("draftDeviceName = %q", got)
	}
	for _, tag := range []string{"", "llama", "ik_llama"} {
		if deviceName(tag, 1) != "CUDA1" || draftDeviceName(tag, 1) != "CUDA1" {
			t.Fatalf("tag %q changed the CUDA names", tag)
		}
	}
	if !otStringUsesDevice(`blk\.(3|4)\.ffn_up_exps.*=ROCm1,exps=CPU`, 1) {
		t.Fatal("a ROCm -ot does not use its device")
	}

	model := &ModelProfile{NumLayers: 48, ExpertBytes: int64(52) * 1024 * 1024 * 1024}
	before := []GPUAssignment{{CUDAIndex: 1, Start: 0, Count: 13}, {CUDAIndex: 0, Start: 13, Count: 6}, {CUDAIndex: 2, Start: 19, Count: 4}}
	after := []GPUAssignment{{CUDAIndex: 1, Start: 0, Count: 13}, {CUDAIndex: 0, Start: 13, Count: 6}, {CUDAIndex: 2, Start: 19, Count: 3}}
	args := []string{"llama-server", "-ub", "256", "-ot", buildOTStringFromAssignments(before, nil, model.NumLayers, "rocm"), "--n-cpu-moe", "22"}
	if !strings.Contains(args[4], "=ROCm2,") {
		t.Fatalf("ROCm -ot %s", args[4])
	}
	newArgs, entry, ok := DerateCUDAOOMArgsForDeficit(args, model, nil, "rocm", 2, 101, 101, false)
	if !ok || entry == nil {
		t.Fatalf("ROCm expert derate did not apply: %v", newArgs)
	}
	ot := newArgs[argIndex(newArgs, "-ot")+1]
	if want := buildOTStringFromAssignments(after, nil, model.NumLayers, "rocm"); ot != want || entry.OTString != ot {
		t.Fatalf("ROCm derate -ot\n%s\nwant\n%s", ot, want)
	}
	if strings.Contains(ot, "CUDA") || strings.Contains(ot, "Vulkan") || currentNCPUMoE(newArgs) != 23 {
		t.Fatalf("ROCm derate wrote another dialect or the wrong n-cpu-moe: %s %v", ot, newArgs)
	}
}
