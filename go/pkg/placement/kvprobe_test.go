package placement

import (
	"strings"
	"testing"
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
