package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestParseKVBufferTotalMB(t *testing.T) {
	// Multi-GPU KV split + a CPU buffer: total must SUM, not average.
	log := strings.Join([]string{
		"llama: CUDA0 model buffer size = 12000.00 MiB",
		"llama: CUDA0 KV buffer size =  2000.00 MiB",
		"llama: CUDA1 KV buffer size =  1000.00 MiB",
		"llama: CPU KV buffer size =   500.00 MiB",
		"llama: CUDA0 compute buffer size =  800.00 MiB",
	}, "\n")
	if got := parseKVBufferTotalMB(log); got != 3500 {
		t.Fatalf("total KV = %.0f, want 3500", got)
	}
}

func TestMeasuredKVRateBeatsFormula(t *testing.T) {
	// A model whose formula would say one thing, but a measured rate overrides it.
	model := &ModelProfile{
		NumLayers: 43, HeadCountKV: 1, KeyLength: 512, ValueLength: 512,
		MeasuredKVBytesPerTok: map[string]float64{"q8_0": 8192}, // 8 KiB/token (measured)
	}
	// 8192 bytes/token * 131072 tokens / 1MiB = 1024 MiB exactly
	if got := computeKVTotalMB(model, 131072, "q8_0"); got != 1024 {
		t.Fatalf("measured KV = %d MiB, want 1024", got)
	}
	// A kvType with no measurement falls back to the formula (non-zero, different).
	if got := computeKVTotalMB(model, 131072, "f16"); got == 1024 || got <= 0 {
		t.Fatalf("f16 should use formula, got %d", got)
	}
}

func TestKVProbeRoundTrip(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Basename: "TestModel", SizeBytes: 12345, Path: "/x/TestModel.gguf"}
	log := "llama: CUDA0 KV buffer size = 4096.00 MiB\nllama: CUDA1 KV buffer size = 4096.00 MiB\n"
	// ctx 262144, total KV 8192 MiB → 8192*1MiB/262144 = 32768 bytes/token
	RunPostLaunchKVProbe(dir, model, 262144, "q8_0", log)
	rates := loadMeasuredKVRates(dir, model)
	if rates == nil || rates["q8_0"] < 32700 || rates["q8_0"] > 32800 {
		t.Fatalf("round-trip rate = %v, want ~32768", rates)
	}
}

func TestRecordMeasuredContextMBUpdatesImmediatePlacementState(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	dir := t.TempDir()
	model := &ModelProfile{
		Basename:  "DeepSeek-V4-Flash",
		SizeBytes: 137903959808,
		MeasuredKVBytesPerTok: map[string]float64{
			"q8_0": 4096,
			"f16":  1,
		},
	}

	RecordMeasuredContextMB(dir, model, 524288, "f16", 3456)
	if got := model.MeasuredKVBytesPerTok["f16"]; got != 6912 {
		t.Fatalf("in-memory f16 rate = %.2f, want 6912", got)
	}
	if got := model.MeasuredKVBytesPerTok["q8_0"]; got != 4096 {
		t.Fatalf("existing in-memory rate was lost: %.2f", got)
	}
	if got := computeKVTotalMB(model, 524288, "f16"); got != 3456 {
		t.Fatalf("immediate placement context = %d MiB, want 3456", got)
	}
	// The final successful-launch log is still the most precise measurement and
	// refines the no-allocation preflight value.
	RunPostLaunchKVProbe(dir, model, 524288, "f16", "llama: CUDA0 KV buffer size = 3450.00 MiB")
	if got := model.MeasuredKVBytesPerTok["f16"]; got != 6900 {
		t.Fatalf("successful launch did not refine rate: %.2f", got)
	}

	model.MeasuredKVBytesPerTok = nil
	rates := loadMeasuredKVRates(dir, model)
	if rates["f16"] != 6900 || rates["q8_0"] != 4096 {
		t.Fatalf("persisted rates = %#v, want f16=6900 and q8_0=4096", rates)
	}
}

func TestLegacyMeasuredCachesMigrateToAppCache(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	appCache := filepath.Join(t.TempDir(), "app-cache")
	model := &ModelProfile{
		Basename:  "Deepseek-V4-Flash",
		Path:      "/models/DeepSeek-V4-Flash.gguf",
		SizeBytes: 137903959808,
	}
	legacyKV := kvCachePath("", model)
	if err := os.MkdirAll(filepath.Dir(legacyKV), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(legacyKV, []byte("KV_BYTES_PER_TOK_f16=6912.2500\n"), 0644); err != nil {
		t.Fatal(err)
	}

	rates := loadMeasuredKVRates(appCache, model)
	if rates == nil || rates["f16"] != 6912.25 {
		t.Fatalf("legacy KV rate was not loaded: %#v", rates)
	}
	if _, err := os.Stat(kvCachePath(appCache, model)); err != nil {
		t.Fatalf("legacy KV cache was not migrated: %v", err)
	}

	gpus := []detect.GPU{{Index: 0, Name: "RTX 3090 Ti", Driver: "580"}}
	systemName := fmt.Sprintf("system_%s.cache", gpuSignatureHash(gpus))
	legacySystem := filepath.Join(home, ".cache", "ggrun", systemName)
	if err := os.WriteFile(legacySystem, []byte("SYS_CUDA_OVERHEAD_MB_CUDA0=488\nSYS_CUDA_OVERHEAD_MB=488\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if got := SystemCUDAOverheadByGPU(appCache, gpus)[0]; got != 488 {
		t.Fatalf("legacy CUDA overhead = %d, want 488", got)
	}
	if _, err := os.Stat(filepath.Join(appCache, systemName)); err != nil {
		t.Fatalf("legacy system cache was not migrated: %v", err)
	}
}

func TestLoadProbeCacheDropsLegacyStartupOOMDoubleCount(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "/models/deepseek4.gguf", NumLayers: 43, NumExperts: 256, EmbeddingLength: 4096}
	gpus := []detect.GPU{{Index: 2, Name: "RTX 4070", Driver: "580"}}
	path := probeCachePath(dir, model, 1048576, 64, "high", "gpu", "llama", gpus, 0)
	legacy := "PROBED_COMPUTE_BUF_MB=8616\n" +
		"PROBED_COMPUTE_BUF_MB_CUDA2=8616\n" +
		"PROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA2=8617\n"
	if err := os.WriteFile(path, []byte(legacy), 0644); err != nil {
		t.Fatal(err)
	}
	got := loadProbeCache(dir, model, 1048576, 64, "high", "gpu", "llama", gpus, 1)
	if got == nil || got.ComputeBufByGPU[2] != 8616 {
		t.Fatalf("legacy compute measurement lost: %#v", got)
	}
	if _, poisoned := got.RuntimeGraphGrowthByGPU[2]; poisoned {
		t.Fatalf("startup compute OOM was still double-counted as runtime growth: %#v", got)
	}

	// Schema-2 growth is known to come from a post-health crash and must remain,
	// even in the unlikely event its measured size equals the compute buffer.
	if err := writeProbeCacheForModel(dir, model, 1048576, 64, "high", "gpu", "llama", gpus, 1,
		map[int]int{2: 8616}, map[int]int{2: 8616}, 0); err != nil {
		t.Fatal(err)
	}
	got = loadProbeCache(dir, model, 1048576, 64, "high", "gpu", "llama", gpus, 1)
	if got == nil || got.RuntimeGraphGrowthByGPU[2] != 8616 {
		t.Fatalf("schema-2 runtime growth was not preserved: %#v", got)
	}
}

func TestParseComputeBuffersByGPU(t *testing.T) {
	log := strings.Join([]string{
		"llama: CUDA0 compute buffer size = 800.40 MiB",
		"common_memory_breakdown_print: |   - CUDA0 (RTX 3090 Ti) | 24111 = 23830 + ( 18668 =  16442 +      26 +    2199) +      -18387 |",
		"common_memory_breakdown_print: |   - CUDA1 (RTX 3060)    | 11909 = 11790 + (  5244 =   5032 +      13 +     197) +       -5125 |",
		"common_memory_breakdown_print: |   - CUDA2 (RTX 4070)    | 11873 = 11704 + (  6193 =   5875 +      12 +     306) +       -6024 |",
	}, "\n")
	got := ParseComputeBuffersByGPU(log)
	if got[0] != 2199 || got[1] != 197 || got[2] != 306 {
		t.Fatalf("compute buffers = %#v, want CUDA0=2199 CUDA1=197 CUDA2=306", got)
	}
	if max, _ := ParseLogForProbe(log); max != 2199 {
		t.Fatalf("max compute buffer = %d, want 2199", max)
	}
}

func TestComputeBuffersFromVRAMDeltaKeepsExpertOnlyGPUExact(t *testing.T) {
	const mib = int64(1048576)
	model := &ModelProfile{
		NumLayers:      43,
		ExpertBytes:    4300 * mib,
		NonExpertBytes: 430 * mib,
		OutputBytes:    43 * mib,
		MeasuredKVBytesPerTok: map[string]float64{
			"f16": 430,
		},
	}
	strategy := &Strategy{
		ContextSize: 1048576,
		KVPlacement: "gpu",
		KVType:      "f16",
		TensorSplit: []float64{0.89, 0, 0.11},
		OTString:    `blk\.(0|1|2)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp).*=CUDA1,exps=CPU`,
	}
	gpus := []detect.GPU{{Index: 0}, {Index: 1}, {Index: 2}}
	baseline := map[int]int{0: 10, 1: 20, 2: 30}
	overhead := map[int]int{0: 100, 1: 200, 2: 300}
	owned, outputDev := layerOwnership(strategy.TensorSplit, model.NumLayers)
	if owned[1] != 0 || outputDev != 2 {
		t.Fatalf("fixture ownership = %v output=%d", owned, outputDev)
	}

	// CUDA1 has no regular layers or KV share. Its delta is exactly CUDA
	// overhead + three full expert layers + a 74 MiB compute buffer.
	used := map[int]int{
		1: baseline[1] + overhead[1] + 3*100 + 74,
	}
	// CUDA2 owns the output slot; charge the output head once, plus its exact
	// owned-layer shares, leaving a 50 MiB compute buffer.
	nonExpertBodyMB := 430 - 43
	model2 := ownedShareMB(nonExpertBodyMB, owned, model.NumLayers, 2) + 43
	kv2 := ownedShareMB(430, owned, model.NumLayers, 2)
	used[2] = baseline[2] + overhead[2] + model2 + kv2 + 50

	got := computeBuffersFromVRAMDelta(model, strategy, gpus, baseline, used, overhead)
	if got[1] != 74 {
		t.Fatalf("expert-only CUDA1 compute = %d MiB, want 74 (all=%v)", got[1], got)
	}
	if got[2] != 50 {
		t.Fatalf("output-owner CUDA2 compute = %d MiB, want 50 (all=%v)", got[2], got)
	}

	strategy.OTString = `blk\.(3)\.ffn_(gate_up|up_gate|gate|up)_(ch|)exps.*=CUDA2,exps=CPU`
	if poisoned := computeBuffersFromVRAMDelta(model, strategy, gpus, baseline, used, overhead); poisoned != nil {
		t.Fatalf("partial expert pins must skip the inexact fallback: %v", poisoned)
	}
}

func TestProbeCacheRoundTripRuntimeKey(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{
		Path:              "/models/model.gguf",
		NumLayers:         43,
		NumExperts:        256,
		EmbeddingLength:   4096,
		FeedForwardLength: 0,
	}
	gpus := []detect.GPU{
		{Index: 0, Name: "RTX 3090 Ti", Driver: "580"},
		{Index: 1, Name: "RTX 3060", Driver: "580"},
		{Index: 2, Name: "RTX 4070", Driver: "580"},
	}
	compute := map[int]int{0: 2199, 1: 197, 2: 306}
	if err := WriteProbeCacheForModel(dir, model, 1048576, 512, "mid", "gpu", "llama", gpus, compute, 1024); err != nil {
		t.Fatalf("write probe: %v", err)
	}
	got := loadProbeCache(dir, model, 1048576, 512, "mid", "gpu", "llama", gpus, 1)
	if got == nil || got.ComputeBufByGPU[0] != 2199 || got.ComputeBufByGPU[1] != 197 || got.ComputeBufByGPU[2] != 306 || got.KVPerLayerMB != 1024 {
		t.Fatalf("loaded probe = %#v", got)
	}
	if wrongPlacement := loadProbeCache(dir, model, 1048576, 512, "mid", "cpu", "llama", gpus, 1); wrongPlacement != nil {
		t.Fatalf("probe must not cross KV placement: %#v", wrongPlacement)
	}
	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 1048576, 512, "mid", "gpu", "llama", gpus, 1, 2, 1000); err != nil {
		t.Fatalf("record runtime growth: %v", err)
	}
	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 1048576, 512, "mid", "gpu", "llama", gpus, 1, 2, 900); err != nil {
		t.Fatalf("record lower runtime growth: %v", err)
	}
	got = loadProbeCache(dir, model, 1048576, 512, "mid", "gpu", "llama", gpus, 1)
	if got == nil || got.ComputeBufByGPU[0] != 2199 || got.ComputeBufByGPU[2] != 306 || got.RuntimeGraphGrowthByGPU[2] != 1000 || got.KVPerLayerMB != 1024 {
		t.Fatalf("loaded runtime growth probe = %#v", got)
	}
	if growth := RuntimeGraphGrowthByGPU(dir, model, 1048576, 512, "mid", "gpu", "llama", gpus, 1); growth[2] != 1000 {
		t.Fatalf("runtime growth = %#v, want CUDA2=1000", growth)
	}
}

func TestProbeCacheSeparatesParallelSlotCounts(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "/models/model.gguf", NumLayers: 43, NumExperts: 256, EmbeddingLength: 4096}
	gpus := []detect.GPU{{Index: 0, Name: "RTX 3090 Ti", Driver: "580"}}

	if err := writeProbeCacheForModel(dir, model, 1048576, 64, "high", "gpu", "llama", gpus, 1,
		map[int]int{0: 8000}, map[int]int{0: 500}, 0); err != nil {
		t.Fatal(err)
	}
	if err := writeProbeCacheForModel(dir, model, 1048576, 64, "high", "gpu", "llama", gpus, 4,
		map[int]int{0: 12000}, map[int]int{0: 900}, 0); err != nil {
		t.Fatal(err)
	}

	serial := loadProbeCache(dir, model, 1048576, 64, "high", "gpu", "llama", gpus, 1)
	parallel := loadProbeCache(dir, model, 1048576, 64, "high", "gpu", "llama", gpus, 4)
	if serial == nil || serial.ComputeBufByGPU[0] != 8000 || serial.RuntimeGraphGrowthByGPU[0] != 500 {
		t.Fatalf("serial probe crossed signatures: %#v", serial)
	}
	if parallel == nil || parallel.ComputeBufByGPU[0] != 12000 || parallel.RuntimeGraphGrowthByGPU[0] != 900 {
		t.Fatalf("parallel probe crossed signatures: %#v", parallel)
	}
}

func TestParseKVBufferWordings(t *testing.T) {
	// aggregate "KV self size" wins over per-device buffers
	agg := "llama_context: KV self size  = 5120.00 MiB, K (f16): 2560.00 MiB, V (f16): 2560.00 MiB"
	if got := parseKVBufferTotalMB(agg); got < 5119 || got > 5121 {
		t.Fatalf("KV self size = %.0f, want ~5120", got)
	}
	// "KV cache size" wording
	if got := parseKVBufferTotalMB("llm: KV cache size = 3000.00 MiB"); got < 2999 || got > 3001 {
		t.Fatalf("KV cache size = %.0f, want ~3000", got)
	}
	// falls back to summing per-device buffer lines when no aggregate present
	perdev := "CUDA0 KV buffer size = 1000.00 MiB\nCUDA1 KV buffer size = 1000.00 MiB"
	if got := parseKVBufferTotalMB(perdev); got < 1999 || got > 2001 {
		t.Fatalf("summed buffers = %.0f, want ~2000", got)
	}
	if got := parseKVBufferTotalMB("no kv here"); got != 0 {
		t.Fatalf("no KV line should be 0, got %.0f", got)
	}
}

func TestKVBytesPerTokenFromVRAMDelta(t *testing.T) {
	// ctx 8192 -> 8000MB, ctx 65536 -> 12000MB. delta 4000MB over 57344 tokens.
	got := kvBytesPerTokenFromVRAMDelta(8192, 8000, 65536, 12000)
	want := 4000.0 * 1048576.0 / 57344.0
	if got < want-1 || got > want+1 {
		t.Fatalf("delta rate = %.1f, want ~%.1f", got, want)
	}
	// order-independent
	if r := kvBytesPerTokenFromVRAMDelta(65536, 12000, 8192, 8000); r < want-1 || r > want+1 {
		t.Fatalf("reversed = %.1f, want ~%.1f", r, want)
	}
	// non-increasing VRAM (noise) → 0
	if r := kvBytesPerTokenFromVRAMDelta(8192, 8000, 65536, 7900); r != 0 {
		t.Fatalf("noisy delta should be 0, got %.1f", r)
	}
}

func TestSetCtxSizeArg(t *testing.T) {
	got := setCtxSizeArg([]string{"-m", "x", "--ctx-size", "32768", "--jinja"}, 8192)
	if got[3] != "8192" {
		t.Fatalf("ctx not replaced: %v", got)
	}
	got = setCtxSizeArg([]string{"-m", "x"}, 8192)
	if got[len(got)-2] != "--ctx-size" || got[len(got)-1] != "8192" {
		t.Fatalf("ctx not appended: %v", got)
	}
}
