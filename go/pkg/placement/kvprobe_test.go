package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
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

func TestConcurrentRuntimeGrowthWritersMergeWithoutLosingMeasuredEvidence(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "/models/concurrent-moe.gguf", Basename: "concurrent-moe.gguf", TotalSizeMB: 70000}
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 24564}, {Index: 1, VRAMTotalMB: 24564}}
	var wg sync.WaitGroup
	errs := make(chan error, 3)
	writes := []func() error{
		func() error {
			return RecordRuntimeGraphGrowthFromOOM(dir, model, 131072, 512, "q4_0", "gpu", "build-a", gpus, 1, 0, 3000, true)
		},
		func() error {
			return RecordRuntimeGraphGrowthFromOOM(dir, model, 131072, 512, "q4_0", "gpu", "build-a", gpus, 1, 0, 900, false)
		},
		func() error {
			return RecordRuntimeGraphGrowthFromOOM(dir, model, 131072, 512, "q4_0", "gpu", "build-a", gpus, 1, 1, 1200, false)
		},
	}
	for _, write := range writes {
		wg.Add(1)
		go func(write func() error) {
			defer wg.Done()
			errs <- write()
		}(write)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	pc := loadProbeCache(dir, model, 131072, 512, "q4_0", "gpu", "build-a", gpus, 1)
	if pc == nil || pc.RuntimeGraphGrowthByGPU[0] != 900 || pc.RuntimeGraphGrowthEstimatedByGPU[0] || pc.RuntimeGraphGrowthByGPU[1] != 1200 {
		t.Fatalf("concurrent merge lost or downgraded evidence: %#v", pc)
	}
}

func TestMeasuredKVRateBeatsFormula(t *testing.T) {
	// A model whose formula would say one thing, but a measured rate overrides it.
	model := &ModelProfile{
		NumLayers: 43, HeadCountKV: 1, KeyLength: 512, ValueLength: 512,
		MeasuredKVBytesPerTok: map[string]float64{"q8_0": 8192}, // 8 KiB/token (measured)
	}
	// 8192 bytes/token * 131072 tokens / 1MiB = 1024 MiB exactly
	if got := computeKVTotalMB(model, 131072, "q8_0", false); got != 1024 {
		t.Fatalf("measured KV = %d MiB, want 1024", got)
	}
	// A kvType with no measurement falls back to the formula (non-zero, different).
	if got := computeKVTotalMB(model, 131072, "f16", false); got == 1024 || got <= 0 {
		t.Fatalf("f16 should use formula, got %d", got)
	}
}

func TestKVProbeRoundTrip(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Basename: "TestModel", SizeBytes: 12345, Path: "/x/TestModel.gguf"}
	log := "llama: CUDA0 KV buffer size = 4096.00 MiB\nllama: CUDA1 KV buffer size = 4096.00 MiB\n"
	// ctx 262144, total KV 8192 MiB → 8192*1MiB/262144 = 32768 bytes/token
	RunPostLaunchKVProbe(dir, model, 262144, "q8_0", log, 1)
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

	RecordMeasuredContextMB(dir, model, 524288, "f16", 3456, false)
	if got := model.MeasuredKVBytesPerTok["f16"]; got != 6912 {
		t.Fatalf("in-memory f16 rate = %.2f, want 6912", got)
	}
	if got := model.MeasuredKVBytesPerTok["q8_0"]; got != 4096 {
		t.Fatalf("existing in-memory rate was lost: %.2f", got)
	}
	if got := computeKVTotalMB(model, 524288, "f16", false); got != 3456 {
		t.Fatalf("immediate placement context = %d MiB, want 3456", got)
	}
	// The final successful-launch log is still the most precise measurement and
	// refines the no-allocation preflight value.
	RunPostLaunchKVProbe(dir, model, 524288, "f16", "llama: CUDA0 KV buffer size = 3450.00 MiB", 1)
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
	systemName := fmt.Sprintf("system_%s.cache", gpuIdentityHash(gpus))
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
	// This file records nothing about how busy the machine was when its numbers
	// were taken, so none of them can be trusted -- not just the growth entry
	// that duplicated the compute buffer. Discarding it subsumes the older
	// double-count repair, which is why that repair no longer exists.
	if got := loadProbeCache(dir, model, 1048576, 64, "high", "gpu", "llama", gpus, 1); got != nil {
		t.Fatalf("probe with no recorded measurement conditions was used: %#v", got)
	}

	// Written through the current writer, so the conditions are recorded and the
	// growth entry passed the post-serving gate: it must survive, even in the
	// unlikely event its measured size equals the compute buffer.
	if err := writeProbeCacheForModel(dir, model, 1048576, 64, "high", "gpu", "llama", gpus, 1,
		map[int]int{2: 8616}, map[int]int{2: 8616}, nil, 0); err != nil {
		t.Fatal(err)
	}
	got := loadProbeCache(dir, model, 1048576, 64, "high", "gpu", "llama", gpus, 1)
	if got == nil || got.RuntimeGraphGrowthByGPU[2] != 8616 {
		t.Fatalf("gated runtime growth was not preserved: %#v", got)
	}
}

func TestProbeCacheKeepsMaximumAcrossPlacementVariants(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "/models/moe.gguf", NumLayers: 43, NumExperts: 256, EmbeddingLength: 4096}
	gpus := []detect.GPU{{Index: 0}, {Index: 1}}
	if err := RecordMeasuredComputeBuffers(dir, model, 1048576, 256, "high", "gpu", "llama", gpus, 4,
		map[int]int{0: 2423, 1: 74}); err != nil {
		t.Fatal(err)
	}
	if err := RecordMeasuredComputeBuffers(dir, model, 1048576, 256, "high", "gpu", "llama", gpus, 4,
		map[int]int{0: 8927, 1: 299}); err != nil {
		t.Fatal(err)
	}
	// A later smaller placement must not erase the larger graph reserve.
	if err := RecordMeasuredComputeBuffers(dir, model, 1048576, 256, "high", "gpu", "llama", gpus, 4,
		map[int]int{0: 1000, 1: 50}); err != nil {
		t.Fatal(err)
	}
	got := loadProbeCache(dir, model, 1048576, 256, "high", "gpu", "llama", gpus, 4)
	if got == nil || got.ComputeBufByGPU[0] != 8927 || got.ComputeBufByGPU[1] != 299 || got.ComputeBufMB != 8927 {
		t.Fatalf("maximum placement-dependent graph reserve was not preserved: %#v", got)
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
	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 1048576, 512, "mid", "gpu", "llama", gpus, 1, 2, 1000, false); err != nil {
		t.Fatalf("record runtime growth: %v", err)
	}
	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 1048576, 512, "mid", "gpu", "llama", gpus, 1, 2, 900, false); err != nil {
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
		map[int]int{0: 8000}, map[int]int{0: 500}, nil, 0); err != nil {
		t.Fatal(err)
	}
	if err := writeProbeCacheForModel(dir, model, 1048576, 64, "high", "gpu", "llama", gpus, 4,
		map[int]int{0: 12000}, map[int]int{0: 900}, nil, 0); err != nil {
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

// A CUDA VMM failure carries no allocation size, so ggrun falls back to a flat
// fraction of the card. Two aborts on a 24 GiB card compounded that guess to
// 4914 MiB -- 3.6 expert layers withheld permanently -- and one of the aborts
// was a malformed launch that proved nothing about capacity.
func TestEstimatedRuntimeGrowthDoesNotAccumulate(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "/models/big-moe.gguf", Basename: "big-moe.gguf", TotalSizeMB: 70000}
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 24564}}

	for i := 0; i < 3; i++ {
		if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 1048576, 512, "q4_0", "cpu", "llama", gpus, 4, 0, 2457, true); err != nil {
			t.Fatalf("record estimate %d: %v", i, err)
		}
	}
	got := RuntimeGraphGrowthByGPU(dir, model, 1048576, 512, "q4_0", "cpu", "llama", gpus, 4)
	if got[0] != 2457 {
		t.Errorf("three identical estimates gave %d MiB, want 2457 (no accumulation)", got[0])
	}
}

// A measurement is better evidence than a fraction of the card, even when it is
// smaller. Backing off is safe: a CUDA out-of-memory aborts the backend, which
// ggrun derates and restarts, rather than taking the host down.
func TestMeasuredRuntimeGrowthReplacesAnEstimate(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "/models/big-moe.gguf", Basename: "big-moe.gguf", TotalSizeMB: 70000}
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 24564}}

	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 1048576, 512, "q4_0", "cpu", "llama", gpus, 4, 0, 2457, true); err != nil {
		t.Fatal(err)
	}
	// A real allocation size, smaller than the guess.
	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 1048576, 512, "q4_0", "cpu", "llama", gpus, 4, 0, 900, false); err != nil {
		t.Fatal(err)
	}
	if got := RuntimeGraphGrowthByGPU(dir, model, 1048576, 512, "q4_0", "cpu", "llama", gpus, 4); got[0] != 900 {
		t.Errorf("measurement did not replace the estimate: got %d MiB, want 900", got[0])
	}
	// And a later guess must not raise a value that was actually observed.
	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 1048576, 512, "q4_0", "cpu", "llama", gpus, 4, 0, 2457, true); err != nil {
		t.Fatal(err)
	}
	if got := RuntimeGraphGrowthByGPU(dir, model, 1048576, 512, "q4_0", "cpu", "llama", gpus, 4); got[0] != 900 {
		t.Errorf("an estimate overwrote a measurement: got %d MiB, want 900", got[0])
	}
}

// Measured values still accumulate by maximum: a larger observed allocation is
// strictly better evidence than a smaller one.
func TestMeasuredRuntimeGrowthKeepsTheLargestObservation(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "/models/big-moe.gguf", Basename: "big-moe.gguf", TotalSizeMB: 70000}
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 24564}}
	for _, mb := range []int{1000, 1800, 900} {
		if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 1048576, 512, "q4_0", "cpu", "llama", gpus, 4, 0, mb, false); err != nil {
			t.Fatal(err)
		}
	}
	if got := RuntimeGraphGrowthByGPU(dir, model, 1048576, 512, "q4_0", "cpu", "llama", gpus, 4); got[0] != 1800 {
		t.Errorf("measured growth = %d MiB, want the largest observation 1800", got[0])
	}
}

// Nothing else shrinks learned growth, so a scope taxed by a crash that no
// longer reproduces needs a way back.
func TestClearRuntimeGraphGrowthKeepsComputeBuffers(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "/models/big-moe.gguf", Basename: "big-moe.gguf", TotalSizeMB: 70000}
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 24564}}

	if err := writeProbeCacheForModel(dir, model, 1048576, 512, "q4_0", "cpu", "llama", gpus, 4,
		map[int]int{0: 6117}, map[int]int{0: 4914}, map[int]bool{0: true}, 128); err != nil {
		t.Fatal(err)
	}
	if err := ClearRuntimeGraphGrowth(dir, model, 1048576, 512, "q4_0", "cpu", "llama", gpus, 4); err != nil {
		t.Fatalf("clear: %v", err)
	}
	pc := loadProbeCache(dir, model, 1048576, 512, "q4_0", "cpu", "llama", gpus, 4)
	if pc == nil {
		t.Fatal("probe cache removed entirely; only learned growth should be cleared")
	}
	if len(pc.RuntimeGraphGrowthByGPU) != 0 {
		t.Errorf("growth not cleared: %#v", pc.RuntimeGraphGrowthByGPU)
	}
	// The expensive measurements must survive.
	if pc.ComputeBufByGPU[0] != 6117 || pc.KVPerLayerMB != 128 {
		t.Errorf("clear discarded measured buffers: %#v", pc)
	}
}

// A --swa-full preflight measures every windowed layer at full depth, so the
// total it reports describes that launch and no other. Recorded as a plain rate
// it makes the next launch reserve the swa-full figure: measured on Laguna,
// 6912 MB at ctx 131072 stored 55296 B/token against a true 13864, and the
// resulting over-reservation forced auto-fit to push KV onto the host.
func TestRecordMeasuredContextMBRefusesSWAFullTotals(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	dir := t.TempDir()
	model := &ModelProfile{
		Basename:              "Laguna-S-2.1",
		SizeBytes:             73395172000,
		SlidingWindow:         512,
		MeasuredKVBytesPerTok: map[string]float64{"q4_0": 13864.5},
	}
	RecordMeasuredContextMB(dir, model, 131072, "q4_0", 6912, true)
	if got := model.MeasuredKVBytesPerTok["q4_0"]; got != 13864.5 {
		t.Fatalf("a swa-full total overwrote the plain rate: %.2f, want 13864.5", got)
	}

	// A model with no windowed layers cannot be distorted by the flag, so its
	// measurement is still worth keeping.
	dense := &ModelProfile{Basename: "Dense", SizeBytes: 1, MeasuredKVBytesPerTok: map[string]float64{}}
	RecordMeasuredContextMB(dir, dense, 131072, "q4_0", 6912, true)
	if got := dense.MeasuredKVBytesPerTok["q4_0"]; got != 55296 {
		t.Fatalf("dense model rate = %.2f, want 55296 recorded normally", got)
	}
}

func TestPostLaunchKVProbeRefusesParallelTotals(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "m.gguf", Basename: "m.gguf", SizeBytes: 4242}

	// A plain single-slot launch establishes the truth first.
	serialLog := "llama: CUDA0 KV buffer size = 1000.00 MiB\nllama: CUDA1 KV buffer size = 1000.00 MiB\n"
	RunPostLaunchKVProbe(dir, model, 262144, "q8_0", serialLog, 1)
	rates := loadMeasuredKVRates(dir, model)
	if rates == nil || rates["q8_0"] < 7900 || rates["q8_0"] > 8100 {
		t.Fatalf("serial rate = %v, want ~8000 bytes/token (2 MiB/KV per ctx token)", rates)
	}

	// Then a --parallel 4 launch prints 4x the KV total (4 sequences).
	parallelLog := "llama: CUDA0 KV buffer size = 4000.00 MiB\nllama: CUDA1 KV buffer size = 4000.00 MiB\n"
	RunPostLaunchKVProbe(dir, model, 262144, "q8_0", parallelLog, 4)

	// The parallel launch must NOT overwrite the model-wide rate.
	rates = loadMeasuredKVRates(dir, model)
	if rates == nil || rates["q8_0"] < 7900 || rates["q8_0"] > 8100 {
		t.Fatalf("parallel=4 launch poisoned the rate: %v, want the serial ~8000 preserved", rates)
	}
}

func TestParseKVBufferMultipleAggregateLines(t *testing.T) {
	log := "llama_init_from_model: KV self size  = 13824.00 MiB, K (q4_0): 6912.00 MiB, V (q4_0): 6912.00 MiB\n" +
		"llama_init_from_model: KV cache size =  3000.00 MiB\n"
	if got := parseKVBufferTotalMB(log); got != 16824 {
		t.Fatalf("multi-region aggregate: got %.0f, want 16824 (13824+3000)", got)
	}
}

// Old oracle records may have measured GPU KV while serving requested CPU KV.
func TestProbeCacheDoesNotReusePrePolicyOracleKey(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "/models/policy.gguf", NumLayers: 12, EmbeddingLength: 512}
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 8192}}
	oldKey := fmt.Sprintf("probe:v7:%s:%d:%d:%d:%d:%d:%d:%s:%s:%s:%s:%d:",
		SpecTargetIdentity(model), model.NumLayers, model.NumExperts, model.EmbeddingLength, model.FeedForwardLength,
		4096, 128, "q8_0", "cpu", "llama", gpuSignatureHash(gpus), 0)
	oldPath := filepath.Join(dir, md5Hash12(oldKey)+".probe")
	if err := os.WriteFile(oldPath, []byte("PROBE_CACHE_SCHEMA=7\nPROBED_COMPUTE_BUF_MB_CUDA0=123\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := loadProbeCache(dir, model, 4096, 128, "q8_0", "cpu", "llama", gpus, 1); got != nil {
		t.Fatalf("reused oracle evidence with stripped memory policy: %+v", got)
	}
	if err := RecordMeasuredComputeBuffers(dir, model, 4096, 128, "q8_0", "cpu", "llama", gpus, 1, map[int]int{0: 456}); err != nil {
		t.Fatal(err)
	}
	if got := loadProbeCache(dir, model, 4096, 128, "q8_0", "cpu", "llama", gpus, 1); got == nil || got.ComputeBufByGPU[0] != 456 {
		t.Fatalf("fresh shape evidence not reusable: %+v", got)
	}
}

// Growth is what a healthy launch allocated beyond everything the backend's own
// log accounts for. Every input must be measured; a device missing any of them
// yields no entry, because an unexplained remainder is unknown, not growth.
func TestRuntimeGraphGrowthFromVRAMDeltaRecordsOnlyClosedAccounting(t *testing.T) {
	gpus := []detect.GPU{{Index: 0}, {Index: 1}, {Index: 2}, {Index: 3}}
	log := strings.Join([]string{
		"llama: CUDA0 model buffer size =  6941.00 MiB",
		"llama: CUDA0 KV buffer size =   274.00 MiB",
		"llama: CUDA0 compute buffer size =   210.00 MiB",
		"llama: CUDA1 model buffer size =  1000.00 MiB",
		"llama: CUDA1 compute buffer size =   100.00 MiB",
		"llama: CUDA3 model buffer size =  1000.00 MiB",
		"llama: CUDA3 compute buffer size =   100.00 MiB",
	}, "\n")
	baseline := map[int]int{0: 1, 1: 1, 2: 1, 3: 1}
	used := map[int]int{0: 7600, 1: 5000, 2: 5000, 3: 1100}
	// CUDA1 has no measured system overhead.
	overhead := map[int]int{0: 100, 2: 100, 3: 100}

	got := runtimeGraphGrowthFromVRAMDelta(gpus, baseline, used, overhead, log)

	// CUDA0 closes: 7600 - 1 - 100 - (6941 + 274 + 210).
	if len(got) != 1 || got[0] != 74 {
		t.Fatalf("growth = %v, want only CUDA0=74", got)
	}
	if _, ok := got[1]; ok {
		t.Fatal("recorded growth for a device with no measured overhead")
	}
	if _, ok := got[2]; ok {
		t.Fatal("recorded growth for a device the log accounts nothing for")
	}
	if _, ok := got[3]; ok {
		t.Fatal("recorded growth where the accounting did not close")
	}
}

// The reserve may now come down, but never past an allocation that was actually
// observed to fail. Before this, growth was recorded only on OOM paths, so a
// cold key borrowed a foreign model's failure and never released it.
func TestSuccessMeasurementFillsAnEmptyKeyAndNeverLowersAnObservedOOM(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "/models/reserve-moe.gguf", Basename: "reserve-moe.gguf", TotalSizeMB: 140000}
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 12282}, {Index: 1, VRAMTotalMB: 24564}}

	// CUDA1 really did fail at 4000 MiB for this exact signature.
	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 500736, 128, "mid", "gpu", "llama", gpus, 1, 1, 4000, false); err != nil {
		t.Fatal(err)
	}
	// A launch then reaches healthy serving and measures far less on both.
	if err := RecordRuntimeGraphGrowth(dir, model, 500736, 128, "mid", "gpu", "llama", gpus, 1,
		map[int]int{0: 210, 1: 309}); err != nil {
		t.Fatal(err)
	}

	got := RuntimeGraphGrowthByGPU(dir, model, 500736, 128, "mid", "gpu", "llama", gpus, 1)
	if got[0] != 210 {
		t.Fatalf("CUDA0 growth = %d, want the success measurement 210 on a key that knew nothing", got[0])
	}
	if got[1] != 4000 {
		t.Fatalf("CUDA1 growth = %d, want the observed OOM 4000 preserved", got[1])
	}
}

// The legacy system-probe migration must find a file written under a SUPERSEDED
// GPU-set signature, because that is the only case it exists for.
//
// It previously rebuilt the legacy filename with the CURRENT hash, so it could
// only match a file already at the current name — i.e. never. Two signature
// changes landed in this branch (the raw measured bandwidth was removed, then
// g.Index was dropped), so every pre-existing system probe on an existing install
// is exactly the case the old code could not reach: dead code that silently
// stopped preserving measured CUDA values.
func TestLegacySystemProbeIsFoundUnderASupersededSignature(t *testing.T) {
	gpus := []detect.GPU{{Index: 0, Name: "RTX 3090 Ti", Driver: "580",
		PCIBusID: "00000000:65:00.0", PCIGen: 3, PCILanes: 16, VRAMTotalMB: 24564,
		BandwidthMBps: 12321}}
	superseded := supersededGPUSetHashes(gpus)
	if len(superseded) == 0 {
		t.Fatal("no superseded signatures are reconstructible, so the migration " +
			"cannot reach any pre-existing file")
	}

	for _, hash := range superseded {
		home := t.TempDir()
		t.Setenv("HOME", home)
		legacyDir := filepath.Join(home, ".cache", "ggrun")
		if err := os.MkdirAll(legacyDir, 0o755); err != nil {
			t.Fatal(err)
		}
		name := "system_" + hash + ".cache"
		if err := os.WriteFile(filepath.Join(legacyDir, name),
			[]byte("SYS_CUDA_OVERHEAD_MB_CUDA0=488\nSYS_CUDA_OVERHEAD_MB=488\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if got := SystemCUDAOverheadByGPU(t.TempDir(), gpus)[0]; got != 488 {
			t.Errorf("a probe under superseded signature %s was not migrated: "+
				"got %d, want 488", hash, got)
		}
	}
}

// And it must REFUSE a probe measured on a different GPU set. Adopting one would
// import another machine's overhead into this plan, which is worse than having no
// measurement at all — a wrong number silently shapes placement.
//
// Found the hard way: an unguarded search picked up a 2026-07-08 probe from this
// box's real ~/.cache/ggrun and changed a context-fit test's answer.
func TestLegacySystemProbeRefusesADifferentGPUSet(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	legacyDir := filepath.Join(home, ".cache", "ggrun")
	if err := os.MkdirAll(legacyDir, 0o755); err != nil {
		t.Fatal(err)
	}
	// A well-formed probe body under a name that matches no GPU set this build
	// could have produced from the set under test.
	if err := os.WriteFile(filepath.Join(legacyDir, "system_ffffffffffff.cache"),
		[]byte("# System probe\nSYS_CUDA_OVERHEAD_MB_CUDA0=488\nSYS_CUDA_OVERHEAD_MB=488\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gpus := []detect.GPU{{Index: 0, Name: "RTX 4070", Driver: "580",
		PCIBusID: "00000000:17:00.0", PCIGen: 3, PCILanes: 16, VRAMTotalMB: 12282,
		BandwidthMBps: 12192}}
	if got := SystemCUDAOverheadByGPU(t.TempDir(), gpus)[0]; got != 0 {
		t.Errorf("a probe from a DIFFERENT GPU set was adopted: got %d, want 0 — "+
			"another machine's overhead must not shape this plan", got)
	}
}
