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

	msg, ok := learnFromVerificationOOM(req, cfg, model, be, caps, caps, strategy, args, canaryOOMLog, nil)
	if !ok || !strings.Contains(msg, "Launch again") {
		t.Fatalf("verification OOM not learned: ok=%v msg=%q", ok, msg)
	}
	first := growth()
	if first < 242 {
		t.Fatalf("recorded reserve %d MiB, want at least one 242 MiB layer", first)
	}
	if _, ok := learnFromVerificationOOM(req, cfg, model, be, caps, caps, strategy, args, canaryOOMLog, nil); !ok {
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
		if msg, ok := learnFromVerificationOOM(&launchRequest{}, cfg, model, &backendInfo{}, caps, caps, strategy, nil, log, nil); ok {
			t.Fatalf("non-runtime failure was learned as an OOM: %q", msg)
		}
	}
}

// The B2 canary failure, verbatim from the acceptance log
// (growth-learning-b2.first.ggrun.log): Qwen3.6-35B-A3B on one RTX 4070. The
// harness's nvidia-smi had the device at 11,871 of 11,873 MiB when the first
// request's cuBLAS GEMM failed.
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
// the device had 1,798 MiB left by the load's own accounting and only reached
// its ceiling by growing while serving.
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

// A request whose prompt quotes a backend log, as one escaped VERB line: the
// owner's normal workflow when working on ggrun through Claude Code.
const pastedBackendLogRequest = `VERB [      log_server_request] request | tid="123706209792000" timestamp=1791463300 request="{\"messages\":[{\"role\":\"user\",\"content\":\"CUDA0: using device CUDA0 - 11710 MiB free\\nllm_load_tensors:      CUDA0 buffer size =  8392.99 MiB\\nINFO [ main] model loaded\"}]}"`

// insertBeforeFailure puts lines between the load-complete marker and the
// failing request.
func insertBeforeFailure(log, lines string) string {
	return strings.Replace(log, "VERB [            update_slots] decoding batch", lines+"\nVERB [            update_slots] decoding batch", 1)
}

// cuBLAS reports "an unsupported value or parameter" (INVALID_VALUE) for shape
// and build incompatibilities too, so it is an OOM only when the device's free
// VRAM, sampled after the failure, shows it at its ceiling. Log text never
// stands in for that sample, and the diagnostics keep the backend's own line.
func TestCublasUnsupportedValueIsAnOOMOnlyAtTheCeiling(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 11873}}}
	model := &placement.ModelProfile{Path: "m.gguf", RoutedExpertLayerBytes: []int64{242 << 20}}
	atCeiling := map[int]int{0: 2} // 11,871 of 11,873 MiB, as recorded

	device, reserve, estimated, suspect, ok := runtimeLogCUDAOOM(b2CublasCeilingLog, caps, model, nil, atCeiling)
	if !ok || device != 0 || !estimated || reserve != 242 {
		t.Fatalf("B2 ceiling failure not read as a size-less OOM: device=%d reserve=%d estimated=%v ok=%v", device, reserve, estimated, ok)
	}
	if !strings.Contains(suspect, cublasUnsupportedValue) || !strings.Contains(suspect, "2 MiB free") {
		t.Fatalf("suspected OOM lost the backend's error or its evidence: %q", suspect)
	}
	for name, log := range map[string]string{
		"serving, KV on the host": b2CublasServingLog,
		// A prompt quoting free memory and a load marker cannot hide it.
		"prompt quotes a load": insertBeforeFailure(b2CublasCeilingLog, pastedBackendLogRequest),
	} {
		if _, _, _, suspect, ok := runtimeLogCUDAOOM(log, caps, model, nil, atCeiling); !ok || !strings.Contains(suspect, cublasUnsupportedValue) {
			t.Errorf("%s: ceiling failure not read as an OOM: ok=%v suspect=%q", name, ok, suspect)
		}
	}

	for name, c := range map[string]struct {
		log  string
		free map[int]int
	}{
		"no sample":                      {b2CublasCeilingLog, nil},
		"packed but not grown":           {b2CublasCeilingLog, map[int]int{0: 188}}, // the B2 plan right after load
		"far from the ceiling":           {b2CublasCeilingLog, map[int]int{0: 12000}},
		"another device at its ceiling":  {b2CublasCeilingLog, map[int]int{1: 2}},
		"another function":               {strings.Replace(b2CublasCeilingLog, "ggml_cuda_op_mul_mat_cublas", "ggml_cuda_flash_attn_ext", 1), atCeiling},
		"prompt pastes a full device":    {insertBeforeFailure(b2CublasServingLog, pastedBackendLogRequest), map[int]int{0: 1798}},
		"prompt pastes, nothing sampled": {insertBeforeFailure(b2CublasServingLog, pastedBackendLogRequest), nil},
	} {
		if _, reserve, _, suspect, ok := runtimeLogCUDAOOM(c.log, caps, model, nil, c.free); ok {
			t.Errorf("%s: cuBLAS error taken for an OOM (reserve %d MiB, %q)", name, reserve, suspect)
		}
	}
}

// The sample is matched to runtime devices by PCI bus ID, so a --gpus launch
// that renumbered the 3090 Ti to CUDA0 reads the 3090 Ti's free memory.
func TestFreeVRAMByRuntimeDeviceMatchesBusIDs(t *testing.T) {
	smi := "00000000:17:00.0, 9000\n00000000:65:00.0, 2\n00000000:B3:00.0, 11000\nnot a row\n"
	gpus := []detect.GPU{{Index: 0, PCIBusID: "00000000:65:00.0"}, {Index: 1, PCIBusID: "00000000:b3:00.0"}}
	if got := freeVRAMByRuntimeDevice(smi, gpus); len(got) != 2 || got[0] != 2 || got[1] != 11000 {
		t.Fatalf("free VRAM by runtime device = %v, want map[0:2 1:11000]", got)
	}
	if got := freeVRAMByRuntimeDevice(smi, []detect.GPU{{Index: 0}}); got != nil {
		t.Fatalf("a GPU without a bus ID was sampled: %v", got)
	}
	if got := freeVRAMByRuntimeDevice("", gpus); got != nil {
		t.Fatalf("no nvidia-smi output still produced a sample: %v", got)
	}
}

// The quoted line is the failure that ended serving, not a fatal-looking line
// from the load phase.
func TestUnrecognizedCrashQuotesTheServingFailure(t *testing.T) {
	log := "/src/llama.cpp/src/llama-model.cpp:812: unknown key, using default\n" + b2CublasCeilingLog
	if got := runtimeFatalLine(log); got != cublasUnsupportedValue {
		t.Fatalf("runtime fatal line = %q, want the cuBLAS error", got)
	}
	if msg := unrecognizedCrashMessage(log); !strings.Contains(msg, cublasUnsupportedValue) || strings.Contains(msg, "unknown key") {
		t.Fatalf("crash message = %q", msg)
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
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 11873}},
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
	marker := filepath.Join(cfg.CacheDir, "claude.oom-recorded")

	for name, free := range map[string]map[int]int{"room on the device": {0: 9000}, "no sample": nil} {
		if msg, ok := learnFromVerificationOOM(req, cfg, model, be, caps, caps, strategy, args, b2CublasCeilingLog, free); ok || msg != "" {
			t.Fatalf("%s: verification learned an OOM from a cuBLAS error: ok=%v msg=%q", name, ok, msg)
		}
		if _, _, _, _, _, ok, err := recordRuntimeOOMLog(req, cfg, model, strategy, be, caps, b2CublasCeilingLog, marker, free); ok || err != nil {
			t.Fatalf("%s: Claude Code recorder took the cuBLAS error for an OOM: ok=%v err=%v", name, ok, err)
		}
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
	if msg := unrecognizedCrashMessage(b2CublasCeilingLog); !strings.Contains(msg, cublasUnsupportedValue) {
		t.Fatalf("serving error lost the backend's line: %q", msg)
	}

	// At the ceiling the same consumer learns, and still names the backend error.
	msg, ok := learnFromVerificationOOM(req, cfg, model, be, caps, caps, strategy, args, b2CublasCeilingLog, map[int]int{0: 2})
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
