package main

import (
	"os"
	"path/filepath"
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

// The B2 canary failure, verbatim from the acceptance log
// (growth-learning-b2.first.ggrun.log): Qwen3.6-35B-A3B on one RTX 4070 with
// 11,710 MiB free at load and 8,393 + 2,783 + 245 MiB placed, so 290 MiB left
// when the first request's cuBLAS GEMM failed.
const b2CublasCeilingLog = `ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes, VRAM: 11873 MiB
CUDA0: using device CUDA0 - 11710 MiB free
llm_load_tensors:        CPU buffer size =  1854.81 MiB
llm_load_tensors:      CUDA0 buffer size =  8392.99 MiB
llama_kv_cache_init:      CUDA0 KV buffer size =  2782.82 MiB
llama_init_from_model: KV self size  = 2720.00 MiB, K (q8_0): 1360.00 MiB, V (q8_0): 1360.00 MiB
llama_init_from_model:        CPU  output buffer size =     0.95 MiB
llama_init_from_model:      CUDA0 compute buffer size =   244.50 MiB
llama_init_from_model:  CUDA_Host compute buffer size =   130.01 MiB
INFO [                    main] model loaded | tid="136938303602688" timestamp=1791457873
VERB [            update_slots] decoding batch | tid="136938303602688" timestamp=1791457887 n_tokens=1
CUDA error: an unsupported value or parameter was passed to the function
  current device: 0, in function ggml_cuda_op_mul_mat_cublas at /__w/ggrun/ggrun/.ik_llama.cpp/ggml/src/ggml-cuda.cu:1878
  cublasSgemm_v2(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N, row_diff, src1_ncols, ne10, &alpha, src0_ddf_i, ne00, src1_ddf1_i, ne10, &beta, dst_dd_i, ldc)
/__w/ggrun/ggrun/.ik_llama.cpp/ggml/src/ggml-cuda.cu:139: CUDA error`

// The B2 serving failure, verbatim (B2-q36-35b-warm log): KV on the host, so
// 1,798 MiB left at load, and the serving recorder had measured 1,672 MiB of
// growth when the GEMM failed.
const b2CublasServingLog = `CUDA0: using device CUDA0 - 11710 MiB free
llm_load_tensors:      CUDA0 buffer size =  9490.99 MiB
llama_kv_cache_init:        CPU KV buffer size =  2782.81 MiB
llama_init_from_model:      CUDA0 compute buffer size =   421.10 MiB
llama_init_from_model:  CUDA_Host compute buffer size =   130.10 MiB
INFO [                    main] model loaded | tid="123706209792000" timestamp=1791463062
VERB [            update_slots] decoding batch | tid="123706209792000" timestamp=1791463375 n_tokens=1
CUDA error: an unsupported value or parameter was passed to the function
  current device: 0, in function ggml_cuda_op_mul_mat_cublas at /__w/ggrun/ggrun/.ik_llama.cpp/ggml/src/ggml-cuda.cu:1878
  cublasSgemm_v2(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N, row_diff, src1_ncols, ne10, &alpha, src0_ddf_i, ne00, src1_ddf1_i, ne10, &beta, dst_dd_i, ldc)
/__w/ggrun/ggrun/.ik_llama.cpp/ggml/src/ggml-cuda.cu:139: CUDA error`

const cublasUnsupportedValue = "CUDA error: an unsupported value or parameter was passed to the function"

// cuBLAS reports "an unsupported value or parameter" (INVALID_VALUE) for shape
// and build incompatibilities too, so it is an OOM only when the device was at
// its ceiling, and the diagnostics keep the backend's own line.
func TestCublasUnsupportedValueIsAnOOMOnlyAtTheCeiling(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 11873}}}
	model := &placement.ModelProfile{Path: "m.gguf", RoutedExpertLayerBytes: []int64{242 << 20}}

	device, reserve, estimated, suspect, ok := runtimeLogCUDAOOM(b2CublasCeilingLog, caps, model, nil, nil)
	if !ok || device != 0 || !estimated || reserve != 242 {
		t.Fatalf("B2 ceiling failure not read as a size-less OOM: device=%d reserve=%d estimated=%v ok=%v", device, reserve, estimated, ok)
	}
	if !strings.Contains(suspect, cublasUnsupportedValue) || !strings.Contains(suspect, "290 MiB left") {
		t.Fatalf("suspected OOM lost the backend's error or its evidence: %q", suspect)
	}

	// Serving: only the measured growth puts the device at its ceiling.
	if _, _, _, _, ok := runtimeLogCUDAOOM(b2CublasServingLog, caps, model, nil, nil); ok {
		t.Fatal("1,798 MiB left at load and no measured growth was taken for an OOM")
	}
	if _, _, _, suspect, ok := runtimeLogCUDAOOM(b2CublasServingLog, caps, model, nil, map[int]int{0: 1672}); !ok || !strings.Contains(suspect, "126 MiB left") {
		t.Fatalf("B2 serving failure after 1,672 MiB of measured growth not read as an OOM: ok=%v suspect=%q", ok, suspect)
	}

	for name, log := range map[string]string{
		// The same line with 12 GiB still free on the device.
		"far from the ceiling": strings.Replace(b2CublasCeilingLog, "11710 MiB free", "23719 MiB free", 1),
		// No free-memory report: nothing corroborates an OOM.
		"no memory evidence": "srv  llama_server: model loaded\n" + cublasUnsupportedValue +
			"\n  current device: 0, in function ggml_cuda_op_mul_mat_cublas at ggml-cuda.cu:1878\n",
		// Text in a logged prompt is not the backend's accounting.
		"evidence in a prompt": "srv  llama_server: model loaded\n" +
			`VERB [ batch_pending_prompt] prompt tokenized | prompt_tokens="CUDA0: using device CUDA0 - 11710 MiB free` + "\n" +
			`llm_load_tensors:      CUDA0 buffer size = 11500.00 MiB"` + "\n" +
			cublasUnsupportedValue + "\n  current device: 0, in function ggml_cuda_op_mul_mat_cublas at ggml-cuda.cu:1878\n",
		// The same error text from another function is not this failure.
		"another function": strings.Replace(b2CublasCeilingLog, "ggml_cuda_op_mul_mat_cublas", "ggml_cuda_flash_attn_ext", 1),
	} {
		if _, reserve, _, suspect, ok := runtimeLogCUDAOOM(log, caps, model, nil, nil); ok {
			t.Errorf("%s: cuBLAS error taken for an OOM (reserve %d MiB, %q)", name, reserve, suspect)
		}
	}
}

// Real llama.cpp load lines (fork-auto serve.log): 23,841 MiB free, 2,178 +
// 19,916 + 883 MiB placed on CUDA1.
func TestCUDADeviceHeadroomReadsMainlineLoadLines(t *testing.T) {
	lines := strings.Split(`0.00.603.112 I llama_prepare_model_devices: using device CUDA1 (NVIDIA GeForce RTX 3090 Ti) (0000:65:00.0) - 23841 MiB free
0.00.827.970 I load_tensors:        CUDA1 model buffer size =  2177.95 MiB
0.01.190.870 I llama_kv_cache:      CUDA1 KV buffer size = 19915.50 MiB
0.01.233.018 I sched_reserve:      CUDA1 compute buffer size =   883.20 MiB`, "\n")
	if headroom, ok := cudaDeviceHeadroomMB(lines, 1, 0); !ok || headroom != 864 {
		t.Fatalf("CUDA1 headroom = %d ok=%v, want 864", headroom, ok)
	}
	if headroom, ok := cudaDeviceHeadroomMB(lines, 1, 400); !ok || headroom != 464 {
		t.Fatalf("CUDA1 headroom after 400 MiB growth = %d ok=%v, want 464", headroom, ok)
	}
	if _, ok := cudaDeviceHeadroomMB(lines, 0, 0); ok {
		t.Fatal("a device the backend never reported has no headroom evidence")
	}
}

// Without memory evidence the cuBLAS error is an ordinary backend failure:
// verification learning and the Claude Code recorder revoke nothing and
// reserve nothing, and the serving error quotes the backend's line. The
// placement cache file stands in for the revocation, which deletes it.
func TestCublasErrorFarFromTheCeilingLearnsNothing(t *testing.T) {
	cfg := &config.Config{CacheDir: t.TempDir()}
	model := &placement.ModelProfile{Path: "/models/m.gguf", Name: "m", ModelArch: "test", SizeBytes: 1234,
		RoutedExpertLayerBytes: []int64{242 << 20}}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24112}},
		RAM: detect.RAMInfo{TotalMB: 32768}, CPU: detect.CPUInfo{Model: "cpu", Cores: 8}}
	be := &backendInfo{Tag: "ik_llama", Identity: "ik-build"}
	req := &launchRequest{CtxFlag: "262144", KVQuality: "q8_0", KVPlacement: "gpu", Parallel: 1}
	req.ProfilePolicyIdentity = requestedLaunchPolicyIdentity(req, model)
	strategy := &placement.Strategy{Type: placement.MoEOffload, ContextSize: 262144, UBatchSize: 256,
		KVQuality: "q8_0", KVPlacement: "gpu", Parallel: 1}
	args := []string{"llama-server", "-m", model.Path, "-ub", "256"}
	tag := scopedProbeBackendTagForStrategy(req, model, be, strategy)
	growth := func() int {
		return placement.RuntimeGraphGrowthByGPU(cfg.CacheDir, model, 262144, 256, "q8_0", "gpu", tag, caps.GPUs, 1)[0]
	}
	strategy.PlacementCachePath = filepath.Join(cfg.CacheDir, "placement.json")
	if err := os.WriteFile(strategy.PlacementCachePath, []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	far := strings.Replace(b2CublasCeilingLog, "11710 MiB free", "23719 MiB free", 1)

	if msg, ok := learnFromVerificationOOM(req, cfg, model, be, caps, caps, strategy, args, far); ok || msg != "" {
		t.Fatalf("verification learned an OOM from a cuBLAS error with 12 GiB free: ok=%v msg=%q", ok, msg)
	}
	marker := filepath.Join(cfg.CacheDir, "claude.oom-recorded")
	if _, _, _, _, _, ok, err := recordRuntimeOOMLog(req, cfg, model, strategy, be, caps, far, marker); ok || err != nil {
		t.Fatalf("Claude Code recorder took the cuBLAS error for an OOM: ok=%v err=%v", ok, err)
	}
	if got := growth(); got != 0 {
		t.Fatalf("a non-memory cuBLAS failure reserved %d MiB", got)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("an unrecorded failure wrote the OOM marker: %v", err)
	}
	if _, err := os.Stat(strategy.PlacementCachePath); err != nil {
		t.Fatalf("a non-memory cuBLAS failure revoked the plan: %v", err)
	}
	if msg := unrecognizedCrashMessage(far); !strings.Contains(msg, cublasUnsupportedValue) {
		t.Fatalf("serving error lost the backend's line: %q", msg)
	}

	// At the ceiling the same consumer learns, and still names the backend error.
	msg, ok := learnFromVerificationOOM(req, cfg, model, be, caps, caps, strategy, args, b2CublasCeilingLog)
	if !ok || !strings.Contains(msg, "suspected CUDA OOM on device 0") || !strings.Contains(msg, cublasUnsupportedValue) {
		t.Fatalf("ceiling failure not learned with its original error: ok=%v msg=%q", ok, msg)
	}
	if got := growth(); got != 242 {
		t.Fatalf("ceiling failure reserved %d MiB, want one 242 MiB layer", got)
	}
	if _, err := os.Stat(strategy.PlacementCachePath); !os.IsNotExist(err) {
		t.Fatalf("ceiling failure left the failed plan's placement cache: %v", err)
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
