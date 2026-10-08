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

// The cuBLAS form of the same ceiling failure must drive the same recovery.
func TestCublasUnsupportedValueAtTheCeilingIsASizelessOOM(t *testing.T) {
	const log = `srv  llama_server: model loaded
CUDA error: an unsupported value or parameter was passed to the function
  current device: 0, in function ggml_cuda_op_mul_mat_cublas at ggml-cuda.cu:1878
  cublasSgemm_v2(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N, row_diff, src1_ncols, ne10, &alpha, src0_ddf_i, ne00, src1_ddf1_i, ne10, &beta, dst_dd_i, ldc)
/src/ggml-cuda.cu:139: CUDA error`
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 11873}}}
	model := &placement.ModelProfile{Path: "m.gguf", RoutedExpertLayerBytes: []int64{242 << 20}}
	device, reserve, estimated, ok := runtimeLogCUDAOOM(log, caps, model, nil)
	if !ok || device != 0 || !estimated || reserve != 242 {
		t.Fatalf("cuBLAS ceiling failure not read as a size-less OOM: device=%d reserve=%d estimated=%v ok=%v", device, reserve, estimated, ok)
	}
	// The same error text from another function is not this failure.
	other := strings.Replace(log, "ggml_cuda_op_mul_mat_cublas", "ggml_cuda_flash_attn_ext", 1)
	if _, _, _, ok := runtimeLogCUDAOOM(other, caps, model, nil); ok {
		t.Fatal("an unsupported-value error outside the cuBLAS GEMM was taken for an OOM")
	}
}

// A --gpus launch runs on a restricted, renumbered GPU set and its measured
// probes are filed under it. Learned growth filed under the full detected set
// never matched, so four relaunches recorded 242..968 MiB and planned
// identically.
func TestRuntimeGrowthIsFiledUnderTheRuntimeGPUSet(t *testing.T) {
	full := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0}, {Index: 1}, {Index: 2}}}
	restricted := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0}}}
	if got := runtimeGrowthCaps(full, restricted); got != restricted {
		t.Fatal("growth must use the GPU set the backend ran on")
	}
	if got := runtimeGrowthCaps(full, nil); got != full {
		t.Fatal("without a runtime set, fall back to the detected set")
	}
	if got := runtimeGrowthCaps(full, &detect.Capabilities{}); got != full {
		t.Fatal("an empty runtime set (CPU launch) must fall back to the detected set")
	}
}
