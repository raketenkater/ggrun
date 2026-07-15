package placement

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/gguf"
)

// writeSpecGGUF writes the metadata surface the speculative resolver uses.
// It intentionally has no tensors: these tests exercise identity/compatibility,
// while gguf's own package tests cover tensor-span accounting.
func writeSpecGGUF(t *testing.T, path, arch, tokenizerModel, tokenizerPre string, embd, ctx, vocab, nextN int) {
	t.Helper()
	type kv struct {
		key    string
		typeID uint32
		str    string
		u32    uint32
		array  int
	}
	kvs := []kv{
		{key: "general.architecture", typeID: 8, str: arch},
		{key: arch + ".embedding_length", typeID: 4, u32: uint32(embd)},
		{key: arch + ".context_length", typeID: 4, u32: uint32(ctx)},
		{key: arch + ".nextn_predict_layers", typeID: 4, u32: uint32(nextN)},
		{key: "tokenizer.ggml.model", typeID: 8, str: tokenizerModel},
		{key: "tokenizer.ggml.pre", typeID: 8, str: tokenizerPre},
		{key: "tokenizer.ggml.tokens", typeID: 9, array: vocab},
	}
	buf := new(bytes.Buffer)
	buf.WriteString("GGUF")
	_ = binary.Write(buf, binary.LittleEndian, uint32(3))
	_ = binary.Write(buf, binary.LittleEndian, uint64(0))
	_ = binary.Write(buf, binary.LittleEndian, uint64(len(kvs)))
	writeString := func(s string) {
		_ = binary.Write(buf, binary.LittleEndian, uint64(len(s)))
		buf.WriteString(s)
	}
	for _, item := range kvs {
		writeString(item.key)
		_ = binary.Write(buf, binary.LittleEndian, item.typeID)
		switch item.typeID {
		case 4:
			_ = binary.Write(buf, binary.LittleEndian, item.u32)
		case 8:
			writeString(item.str)
		case 9:
			_ = binary.Write(buf, binary.LittleEndian, uint32(0)) // array<uint8>
			_ = binary.Write(buf, binary.LittleEndian, uint64(item.array))
			buf.Write(make([]byte, item.array))
		}
	}
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write spec GGUF: %v", err)
	}
}

func saveEligibleSpecProfile(t *testing.T, target *ModelProfile, caps *detect.Capabilities, opts Options, kind, companion string, draftMax int) {
	t.Helper()
	scope := NewSpecProfileScope(target, caps, opts, kind, companion)
	maxPrompt := 4096
	if scope.ContextSize >= 60000 {
		maxPrompt = 60000
	}
	_, err := SaveSpecPerformanceProfile(opts.CacheDir, SpecPerformanceProfile{
		Scope: scope, LaunchIdentity: "test-launch", DraftMax: draftMax, BaselineTPS: 100, SpeculativeTPS: 110, ImprovementPct: 10, WallImprovementPct: 8,
		PromptCases: 9, RepeatedRounds: 3, MaxPromptTokens: maxPrompt,
		CorrectnessPassed: true, StabilityPassed: true, ParallelLoadPassed: true, Complete: true,
	})
	if err != nil {
		t.Fatalf("save speculative profile: %v", err)
	}
}

func TestComputeDenseFits(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "model.gguf",
		SizeBytes:   15 * 1024 * 1024 * 1024, // 15GB
		NumLayers:   64,
		NumParams:   32_000_000_000,
		IsMoE:       false,
		ContextSize: 32768,
		HiddenSize:  4096,
	}
	opts := Options{KVPlacement: "auto", KVQuality: "mid"}
	strat, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.GPULayers == 0 {
		t.Fatalf("expected some layers on GPU, got %d", strat.GPULayers)
	}
	if strat.ContextSize == 0 {
		t.Fatalf("context size should not be zero")
	}
}

func TestComputeDenseTooLarge(t *testing.T) {
	// 40GB model on 8GB GPU with 128GB RAM -> dense_cpu_offload
	// Total system memory must exceed model overhead (40GB * 130% = 52GB)
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 8192}},
		RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path:        "model.gguf",
		SizeBytes:   40 * 1024 * 1024 * 1024, // 40GB
		NumLayers:   80,
		NumParams:   70_000_000_000,
		IsMoE:       false,
		ContextSize: 32768,
		HiddenSize:  8192,
	}
	opts := Options{KVPlacement: "auto", KVQuality: "mid"}
	strat, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.Type != DenseCPUOffload {
		t.Fatalf("expected dense_cpu_offload strategy, got %s", strat.Type)
	}
	if strat.GPULayers != 999 {
		t.Fatalf("expected GPULayers=999, got %d", strat.GPULayers)
	}
}

func TestComputeSingleGPUChoosesFastestDeviceThatActuallyFits(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 8192, BandwidthMBps: 32000},
			{Index: 1, VRAMTotalMB: 24576, BandwidthMBps: 8000},
		},
		RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "model.gguf", SizeBytes: 14 * 1024 * 1024 * 1024,
		NumLayers: 40, ContextSize: 4096, HiddenSize: 2048,
	}
	strat, err := Compute(caps, model, Options{ContextSize: 4096, KVPlacement: "gpu", KVQuality: "low"})
	if err != nil {
		t.Fatal(err)
	}
	if strat.Type != SingleGPU || strat.MainGPU != 1 {
		t.Fatalf("expected the fitting CUDA1, got type=%s main=%d", strat.Type, strat.MainGPU)
	}
}

func TestComputeBatchTierUsesFreeNotTotalVRAM(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, VRAMUsedMB: 8192}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "dense.gguf", SizeBytes: 10 * 1024 * 1024 * 1024,
		NumLayers: 40, ContextSize: 4096, HiddenSize: 4096,
	}
	strategy, err := Compute(caps, model, Options{ContextSize: 4096, KVPlacement: "gpu", KVQuality: "low"})
	if err != nil {
		t.Fatal(err)
	}
	if strategy.Type != SingleGPU {
		t.Fatalf("expected single GPU, got %s", strategy.Type)
	}
	if strategy.UBatchSize != 512 || strategy.BatchSize != 4096 {
		t.Fatalf("batch tier ignored occupied VRAM: batch=%d ubatch=%d", strategy.BatchSize, strategy.UBatchSize)
	}
}

func TestDenseCPUOffloadLetsBackendFitExactLayers(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 8192, BandwidthMBps: 8000}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "model.gguf", SizeBytes: 20 * 1024 * 1024 * 1024,
		NumLayers: 40, ContextSize: 4096, HiddenSize: 2048,
	}
	strat, err := Compute(caps, model, Options{
		ContextSize: 4096, KVPlacement: "cpu", KVQuality: "low",
		BackendHelp: "-fit, --fit [on|off]",
	})
	if err != nil {
		t.Fatal(err)
	}
	if strat.Type != DenseCPUOffload {
		t.Fatalf("expected dense CPU offload, got %s", strat.Type)
	}
	args := strat.Args(model.Path, 8081)
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--fit on") {
		t.Fatalf("dense CPU offload must enable backend fit: %s", joined)
	}
	for _, forbidden := range []string{"-ngl", "--tensor-split", "--split-mode"} {
		if contains(args, forbidden) {
			t.Fatalf("dense CPU offload must leave %s unset for backend fit: %s", forbidden, joined)
		}
	}
}

func TestComputeMoE(t *testing.T) {
	// 40GB MoE on 24GB GPU with 128GB RAM
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "moe.gguf",
		SizeBytes:   40 * 1024 * 1024 * 1024, // 40GB
		NumLayers:   64,
		NumParams:   70_000_000_000,
		IsMoE:       true,
		NumExperts:  64,
		ContextSize: 32768,
		HiddenSize:  4096,
	}
	opts := Options{KVPlacement: "auto", KVQuality: "mid"}
	strat, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.NCPUMoE == 0 {
		t.Fatalf("expected CPU experts for large MoE")
	}
	if strat.GPULayers == 0 && strat.NCPUMoE == 0 {
		t.Fatalf("expected some GPU layers or CPU experts for MoE")
	}
}

func TestComputeCPUOnly(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "model.gguf",
		SizeBytes:   10 * 1024 * 1024 * 1024,
		NumLayers:   32,
		NumParams:   8_000_000_000,
		IsMoE:       false,
		ContextSize: 32768,
		HiddenSize:  4096,
	}
	opts := Options{CPUMode: true}
	strat, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.GPULayers != 0 {
		t.Fatalf("expected CPU-only mode")
	}
}

func TestComputeCPUOnlyPreservesNoMMap(t *testing.T) {
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "model.gguf", SizeBytes: 8 * 1024 * 1024 * 1024,
		NumLayers: 32, ContextSize: 4096, HiddenSize: 4096,
	}
	strategy, err := Compute(caps, model, Options{CPUMode: true, NoMMap: true, ContextSize: 4096})
	if err != nil {
		t.Fatal(err)
	}
	if strategy.MMap {
		t.Fatal("CPU-only placement discarded explicit no-mmap")
	}
	if args := strategy.Args(model.Path, 8081); !contains(args, "--no-mmap") {
		t.Fatalf("CPU-only no-mmap was not emitted: %v", args)
	}
}

func TestComputeCPUOnlyReservesHostRuntimeMemory(t *testing.T) {
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 12288, FreeMB: 12288},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "model.gguf", SizeBytes: 10 * 1024 * 1024 * 1024,
		NumLayers: 40, ContextSize: 4096, HiddenSize: 4096,
	}
	_, err := Compute(caps, model, Options{CPUMode: true, ContextSize: 4096, KVPlacement: "cpu", KVQuality: "low"})
	if err == nil || !strings.Contains(err.Error(), "Host runtime buffers") {
		t.Fatalf("expected host runtime memory refusal, got %v", err)
	}
}

func TestComputeCPUOnlyDoesNotChargeDetectedGPUOverhead(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576},
			{Index: 1, VRAMTotalMB: 12288},
			{Index: 2, VRAMTotalMB: 12288},
		},
		RAM: detect.RAMInfo{TotalMB: 4096, FreeMB: 4096},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "tiny.gguf", SizeBytes: 1 * 1024 * 1024,
		NumLayers: 2, ContextSize: 512, HiddenSize: 128,
	}
	strategy, err := Compute(caps, model, Options{CPUMode: true, ContextSize: 512, KVPlacement: "cpu", KVQuality: "low"})
	if err != nil {
		t.Fatalf("CPU-only placement charged unused GPU overhead: %v", err)
	}
	if strategy.Type != CPUOnly {
		t.Fatalf("expected CPU-only strategy, got %s", strategy.Type)
	}
}

func TestArgs(t *testing.T) {
	s := &Strategy{
		ContextSize:    4096,
		GPULayers:      32,
		KVQuality:      "mid",
		FlashAttention: true,
		Threads:        16,
		BatchSize:      2048,
		UBatchSize:     512,
	}
	args := s.Args("/models/test.gguf", 8081)
	if len(args) == 0 {
		t.Fatalf("args should not be empty")
	}
	joined := ""
	for _, a := range args {
		joined += a + " "
	}
	if !contains(args, "-m") {
		t.Fatalf("args missing -m")
	}
	if !contains(args, "/models/test.gguf") {
		t.Fatalf("args missing model path")
	}
	if !contains(args, "--port") {
		t.Fatalf("args missing --port")
	}
	host := ""
	timeout := ""
	for i, arg := range args {
		if arg == "--host" && i+1 < len(args) {
			host = args[i+1]
		}
		if arg == "--timeout" && i+1 < len(args) {
			timeout = args[i+1]
		}
	}
	if host != "127.0.0.1" {
		t.Fatalf("expected loopback host, got %q in args %v", host, args)
	}
	if timeout != "2147483647" {
		t.Fatalf("expected no practical server request timeout, got %q in args %v", timeout, args)
	}
	if !contains(args, "--flash-attn") {
		t.Fatalf("args missing --flash-attn")
	}
}

// Regression: DeepSeek-V4-Flash MXFP4 on the real 3090Ti+3060+4070 box. The
// parser once under-sized MXFP4 tensors (unknown ggml type 39 → 0.5 B/elem
// guess instead of 17 B / 32 elems), so expertPerLayerMB came out 3098 instead
// of the real 3290 and placement pinned 5 expert layers on GPU0 — a guaranteed
// CUDA OOM discovered only after a 15-minute model load. With exact bytes and
// the cold-cache MoE graph reserve, the first plan must remain within every
// device ledger while still using an expert-storage GPU.
func TestComputeDeepSeekV4FlashFirstLaunchExactBudget(t *testing.T) {
	// Real PCIe links on this box: 3090Ti gen3 x16, 3060 gen3 x1 (!), 4070
	// gen3 x4. The x1 card is slow enough that it must be expert-only, not
	// a tiny regular layer owner in the old observed 0.86/0.03/0.11 split.
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, BandwidthMBps: 15754},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, BandwidthMBps: 985},
			{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12282, BandwidthMBps: 3938},
		},
		RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 120000},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path:            "DeepSeek-V4-Flash-MXFP4-00001-of-00005.gguf",
		Basename:        "DeepSeek-V4-Flash-MXFP4-00001-of-00005.gguf",
		SizeBytes:       156378344860,
		TotalSizeMB:     149134,
		NumLayers:       43,
		IsMoE:           true,
		NumExperts:      256,
		ExpertUsedCount: 6,
		ExpertFF:        2048,
		ExpertBytes:     148319502336, // real on-disk spans (parse_gguf.py)
		NonExpertBytes:  8053508160,
		TokenEmbdBytes:  1059061760,
		OutputBytes:     1059061760, // lands whole on the last split device (observed: CUDA2)
		ShexpBytes:      1149763584, // ~25.5MB/layer, stays on the layer's device
		ContextSize:     1048576,
		CTXTrain:        1048576,
		EmbeddingLength: 4096,
		HiddenSize:      4096,
		HeadCountKV:     1,
		KVLoraRank:      512,
		QLoraRank:       1024,
		SlidingWindow:   128,
		ModelArch:       "deepseek4",
		MeasuredKVBytesPerTok: map[string]float64{
			"q8_0": 6912.25, // measured: launch log, ctx=1048576 total_kv=6912MB
		},
	}

	strat, err := Compute(caps, model, Options{
		ContextSize: 1048576,
		KVPlacement: "cpu",
		KVQuality:   "mid",
		BackendTag:  "llama",
		Parallel:    1,
		CacheDir:    t.TempDir(),
	})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.Type != MoEOffload {
		t.Fatalf("expected MoE offload, got %s", strat.Type)
	}
	if strat.TensorSplit[1] != 0 {
		t.Fatalf("expected x1 GPU to be expert-only with zero tensor split, got %v", strat.TensorSplit)
	}
	if !otStringUsesDevice(strat.OTString, 1) {
		t.Fatalf("expected x1 GPU to still receive full expert pins, got OT %s", strat.OTString)
	}

	expertPerLayerMB := ceilDivInt(bytesToMiBCeil(model.ExpertBytes), model.NumLayers)
	if expertPerLayerMB != 3290 {
		t.Fatalf("expected real 3290MB per expert layer, got %d", expertPerLayerMB)
	}

	// Whole-layer pins carry the shared expert ("_shexp"); gate+up sub-pins
	// don't and cost only 2/3 of a layer, so count them separately.
	wholeLayersByDevice := map[int]int{}
	for _, part := range strings.Split(strat.OTString, ",") {
		m := otDevicePattern.FindStringSubmatch(part)
		if m == nil || !strings.Contains(part, "_shexp") {
			continue
		}
		device, err := strconv.Atoi(m[2])
		if err != nil {
			t.Fatalf("parse device from %q: %v", part, err)
		}
		wholeLayersByDevice[device] += len(strings.Split(m[1], "|"))
	}
	if wholeLayersByDevice[0] > 5 {
		t.Fatalf("GPU0 cannot hold %d whole expert layers (only 5 fit without measured CUDA overhead): %s",
			wholeLayersByDevice[0], strat.OTString)
	}
	totalWholeLayers := 0
	for _, n := range wholeLayersByDevice {
		totalWholeLayers += n
	}

	// Exact per-GPU ledger with real byte sizes and llama.cpp's real slot
	// assignment: nothing may exceed free VRAM. Input embeddings stay on the
	// CPU; the output head lands whole on the last split device; shared
	// experts ride with their layer's owner.
	nonExpertPoolMB := bytesToMiBCeil(model.NonExpertBytes) -
		bytesToMiBCeil(model.TokenEmbdBytes) - bytesToMiBCeil(model.OutputBytes)
	owned, outputDev := layerOwnership(strat.TensorSplit, model.NumLayers)
	if outputDev != 0 {
		t.Fatalf("expected output head on the sole dense-layer owner (CUDA0), got %d (split %v)", outputDev, strat.TensorSplit)
	}
	perLayerNonExp := float64(nonExpertPoolMB) / float64(model.NumLayers)
	perLayerShexp := float64(bytesToMiBCeil(model.ShexpBytes)) / float64(model.NumLayers)
	expertMBByDevice := otExpertMBByDevice(t, strat.OTString, expertPerLayerMB)
	for gi, gpu := range caps.GPUs {
		fixed := firstLaunchComputeBufMBForGPU(model, strat.UBatchSize, gi, orderGPUsByBandwidth(caps.GPUs))
		if strat.TensorSplit[gi] == 0 && otStringUsesDevice(strat.OTString, gpu.Index) {
			fixed = computeFloorMB // expert-only graph reserve
		}
		usedMB := fixed + int(float64(owned[gi])*(perLayerNonExp+perLayerShexp)) + expertMBByDevice[gpu.Index]
		if gi == outputDev {
			usedMB += bytesToMiBCeil(model.OutputBytes)
		}
		if usedMB > gpu.VRAMFreeMB() {
			t.Fatalf("gpu %d over budget: used=%dMB free=%dMB owned=%v split=%v ot=%s",
				gpu.Index, usedMB, gpu.VRAMFreeMB(), owned, strat.TensorSplit, strat.OTString)
		}
	}

	// The cold plan is intentionally conservative until fit-params measures this
	// exact graph. It must still offload at least one complete expert layer.
	if totalWholeLayers < 1 {
		t.Fatalf("expected at least one whole expert layer on GPU, got %d: %s",
			totalWholeLayers, strat.OTString)
	}
	for _, part := range strings.Split(strat.OTString, ",") {
		if otDevicePattern.MatchString(part) && !strings.Contains(part, "_shexp") {
			t.Fatalf("cold-cache placement must not use sub-layer squeeze without measured CUDA overhead: %s", strat.OTString)
		}
	}
	if strat.KVPlacement != "cpu" {
		t.Fatalf("expected KV on CPU, got %q", strat.KVPlacement)
	}
}

// Regression for the real no-flag DeepSeek-V4 launch on the 3090Ti+3060+4070
// host. The exact 1M-context KV measurement and CUDA overhead originally lived
// in ~/.cache/ggrun, while the app later moved to an app-local cache. Falling
// back to formula charged 16.4 GiB instead of the measured 6.9 GiB; legacy
// startup-OOM probes then charged compute a second time as runtime growth. The
// solver rejected a model that fits, before preflight could correct anything.
func TestComputeDeepSeekV4FullContextMigratesExactMeasurements(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	cacheDir := filepath.Join(t.TempDir(), "app-cache")
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		t.Fatal(err)
	}

	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "NVIDIA GeForce RTX 3090 Ti", Driver: "580.159.03", VRAMTotalMB: 24564, VRAMUsedMB: 453, BandwidthMBps: 15760},
			{Index: 1, Name: "NVIDIA GeForce RTX 3060", Driver: "580.159.03", VRAMTotalMB: 12288, VRAMUsedMB: 379, BandwidthMBps: 985},
			{Index: 2, Name: "NVIDIA GeForce RTX 4070", Driver: "580.159.03", VRAMTotalMB: 12282, VRAMUsedMB: 409, BandwidthMBps: 3940},
		},
		RAM: detect.RAMInfo{TotalMB: 128730, FreeMB: 123424},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path:            "/models/UD-IQ4_XS/DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf",
		Basename:        "Deepseek-V4-Flash",
		SizeBytes:       137903959808,
		TotalSizeMB:     131515,
		NumLayers:       43,
		IsMoE:           true,
		NumExperts:      256,
		ExpertUsedCount: 6,
		ExpertFF:        2048,
		ExpertBytes:     131240296448,
		NonExpertBytes:  6658320448,
		TokenEmbdBytes:  562626560,
		OutputBytes:     434380800,
		ShexpBytes:      1149763584,
		ContextSize:     1048576,
		CTXTrain:        1048576,
		HiddenSize:      4096,
		EmbeddingLength: 4096,
		HeadCountKV:     1,
		KeyLength:       512,
		ValueLength:     512,
		RopeDim:         64,
		SlidingWindow:   128,
		ModelArch:       "deepseek4",
	}

	legacyDir := filepath.Join(home, ".cache", "ggrun")
	if err := os.MkdirAll(legacyDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(kvCachePath("", model), []byte("KV_BYTES_PER_TOK_f16=6912.2500\n"), 0644); err != nil {
		t.Fatal(err)
	}
	systemPath := filepath.Join(legacyDir, fmt.Sprintf("system_%s.cache", gpuSignatureHash(caps.GPUs)))
	systemData := "SYS_CUDA_OVERHEAD_MB_CUDA0=488\n" +
		"SYS_CUDA_OVERHEAD_MB_CUDA1=311\n" +
		"SYS_CUDA_OVERHEAD_MB_CUDA2=367\n" +
		"SYS_CUDA_OVERHEAD_MB=488\n"
	if err := os.WriteFile(systemPath, []byte(systemData), 0644); err != nil {
		t.Fatal(err)
	}

	// These are the real mainline fit-params measurements. The growth rows are
	// legacy poison: the same startup allocation rounded up by one MiB.
	legacyProbes := map[int]string{
		256: "PROBED_COMPUTE_BUF_MB=33893\nPROBED_COMPUTE_BUF_MB_CUDA0=33893\nPROBED_COMPUTE_BUF_MB_CUDA1=299\nPROBED_COMPUTE_BUF_MB_CUDA2=33697\nPROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA0=33894\n",
		128: "PROBED_COMPUTE_BUF_MB=17074\nPROBED_COMPUTE_BUF_MB_CUDA0=17074\nPROBED_COMPUTE_BUF_MB_CUDA1=149\nPROBED_COMPUTE_BUF_MB_CUDA2=16976\nPROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA0=17075\n",
		64:  "PROBED_COMPUTE_BUF_MB=8664\nPROBED_COMPUTE_BUF_MB_CUDA0=8664\nPROBED_COMPUTE_BUF_MB_CUDA1=74\nPROBED_COMPUTE_BUF_MB_CUDA2=8616\nPROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA2=8617\n",
	}
	for ubatch, data := range legacyProbes {
		path := probeCachePath(cacheDir, model, 1048576, ubatch, "high", "gpu", "llama", caps.GPUs, 0)
		if err := os.WriteFile(path, []byte(data), 0644); err != nil {
			t.Fatal(err)
		}
	}

	strat, err := Compute(caps, model, Options{
		KVPlacement: "auto",
		KVQuality:   "high",
		BackendTag:  "llama",
		Parallel:    1,
		CacheDir:    cacheDir,
	})
	if err != nil {
		t.Fatalf("full-context placement should fit: %v", err)
	}
	if strat.ContextSize != 1048576 {
		t.Fatalf("context was lowered to %d; want the full 1048576", strat.ContextSize)
	}
	if strat.KVPlacement != "gpu" || strat.KVType != "f16" {
		t.Fatalf("mainline DeepSeek4 must keep f16 KV on GPU, got placement=%s type=%s", strat.KVPlacement, strat.KVType)
	}
	if got := model.MeasuredKVBytesPerTok["f16"]; got != 6912.25 {
		t.Fatalf("exact KV measurement was not migrated, got %.2f", got)
	}
	if len(strat.TensorSplit) != 3 || strat.TensorSplit[1] != 0 {
		t.Fatalf("slow CUDA1 must remain expert-only, split=%v", strat.TensorSplit)
	}
	if !otStringUsesDevice(strat.OTString, 1) {
		t.Fatalf("expert-only CUDA1 should still be filled with expert weights: %s", strat.OTString)
	}
	for _, part := range strings.Split(strat.OTString, ",") {
		if otDevicePattern.MatchString(part) && !strings.Contains(part, "_shexp") {
			t.Fatalf("automatic planner must emit complete expert layers only: %s", strat.OTString)
		}
	}
}

func TestComputeDeepSeekV4Parallel4UsesMeasuredStableWholeLayerPlan(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, BandwidthMBps: 15754},
			{Index: 1, Name: "RTX 3060 x1", VRAMTotalMB: 12288, BandwidthMBps: 985},
			{Index: 2, Name: "RTX 4070 x4", VRAMTotalMB: 12282, BandwidthMBps: 3938},
		},
		RAM: detect.RAMInfo{TotalMB: 128730, FreeMB: 123424},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path:      "/models/DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf",
		SizeBytes: 137903959808, TotalSizeMB: 131515,
		NumLayers: 43, IsMoE: true, NumExperts: 256, ExpertUsedCount: 6, ExpertFF: 2048,
		ExpertBytes: 131240296448, NonExpertBytes: 6658320448,
		TokenEmbdBytes: 562626560, OutputBytes: 434380800, ShexpBytes: 1149763584,
		ContextSize: 1048576, CTXTrain: 1048576, HiddenSize: 4096, EmbeddingLength: 4096,
		HeadCountKV: 1, KeyLength: 512, ValueLength: 512, ModelArch: "deepseek4",
		MeasuredKVBytesPerTok: map[string]float64{"f16": 7012},
	}
	cacheDir := t.TempDir()
	systemData := "SYS_CUDA_OVERHEAD_MB_CUDA0=488\n" +
		"SYS_CUDA_OVERHEAD_MB_CUDA1=311\n" +
		"SYS_CUDA_OVERHEAD_MB_CUDA2=367\n" +
		"SYS_CUDA_OVERHEAD_MB=488\n"
	if err := os.WriteFile(filepath.Join(cacheDir, fmt.Sprintf("system_%s.cache", gpuSignatureHash(caps.GPUs))), []byte(systemData), 0644); err != nil {
		t.Fatal(err)
	}

	strat, err := Compute(caps, model, Options{
		ContextSize: 1048576, KVPlacement: "gpu", KVQuality: "high",
		BackendTag: "llama", Parallel: 4, CacheDir: cacheDir,
	})
	if err != nil {
		t.Fatal(err)
	}
	if strat.UBatchSize != 256 {
		t.Fatalf("ubatch=%d, want largest stable rung 256", strat.UBatchSize)
	}
	if len(strat.TensorSplit) != 3 || strat.TensorSplit[0] != 1 || strat.TensorSplit[1] != 0 || strat.TensorSplit[2] != 0 {
		t.Fatalf("dense split=%v, want 1,0,0", strat.TensorSplit)
	}
	layers := parseOTLayersByDevice(t, strat.OTString)
	if len(layers[0]) != 0 || len(layers[1]) != 3 || len(layers[2]) != 3 {
		t.Fatalf("expert layers by device=%v, want CUDA0=0 CUDA1=3 CUDA2=3 (OT=%s)", layers, strat.OTString)
	}
	if strat.NCPUMoE != 37 {
		t.Fatalf("n-cpu-moe=%d, want 37", strat.NCPUMoE)
	}
	for _, part := range strings.Split(strat.OTString, ",") {
		if otDevicePattern.MatchString(part) && !strings.Contains(part, "_shexp") {
			t.Fatalf("stable plan must not contain partial expert pins: %s", strat.OTString)
		}
	}
}

// TestMaximizeMoEGPUFitByUBatchRescuesZeroExpertPlacement reproduces the
// 2026-07-08 "no expert layers landed on GPU" report: DeepSeek-V4 at ctx
// 1048576 with f16 KV (mainline requires f16 KV for correct dsv4 output —
// q8_0 KV computes garbage) needs a flash-attention compute buffer that
// scales with ubatch and dwarfs the model at ubatch 512. Compute-buffer
// values below are from a live `llama-fit-params --fit-print on` run against
// this exact model/ctx/KV/tensor-split (2026-07-08), not a guess.
func TestMaximizeMoEGPUFitByUBatchRescuesZeroExpertPlacement(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, BandwidthMBps: 15754},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, BandwidthMBps: 985},
			{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12282, BandwidthMBps: 3938},
		},
		RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 120000},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path:            "DeepSeek-V4-Flash-MXFP4-00001-of-00005.gguf",
		Basename:        "DeepSeek-V4-Flash-MXFP4-00001-of-00005.gguf",
		SizeBytes:       156378344860,
		TotalSizeMB:     149134,
		NumLayers:       43,
		IsMoE:           true,
		NumExperts:      256,
		ExpertUsedCount: 6,
		ExpertFF:        2048,
		ExpertBytes:     148319502336,
		NonExpertBytes:  8053508160,
		TokenEmbdBytes:  1059061760,
		OutputBytes:     1059061760,
		ShexpBytes:      1149763584,
		ContextSize:     1048576,
		CTXTrain:        1048576,
		EmbeddingLength: 4096,
		HiddenSize:      4096,
		HeadCountKV:     1,
		KVLoraRank:      512,
		QLoraRank:       1024,
		SlidingWindow:   128,
		ModelArch:       "deepseek4",
		MeasuredKVBytesPerTok: map[string]float64{
			"q8_0": 6912.25, // measured: launch log, ctx=1048576 total_kv=6912MB
			// f16 KV derived from the measured q8_0 rate by the same byte-width
			// ratio the code already uses (f16 bytesPerElem 2.0 vs q8_0 1.0625)
			// — not an independent guess, deepseek4's MLA-compressed KV just
			// scales with element width like any other cache type.
			"f16": 6912.25 * 2.0 / 1.0625,
		},
	}

	cacheDir := t.TempDir()
	gpus := caps.GPUs
	// Measured (fit-params, 2026-07-08): compute buffer per GPU at ctx
	// 1048576, f16 KV, parallel 4, real 0.86/0.03/0.11 tensor split.
	measured := map[int]map[int]int{
		512: {0: 17970, 1: 20573, 2: 20612}, // eats the whole card before any expert fits
		256: {0: 9113, 1: 10413, 2: 10432},
		128: {0: 4684, 1: 5332, 2: 5342},
		64:  {0: 2470, 1: 2793, 2: 2797},
	}
	for ub, byGPU := range measured {
		if err := writeProbeCacheForModel(cacheDir, model, 1048576, ub, "high", "gpu", "llama", gpus, 4, byGPU, nil, 0); err != nil {
			t.Fatalf("seed probe cache ubatch=%d: %v", ub, err)
		}
	}

	strat, err := Compute(caps, model, Options{
		ContextSize: 1048576,
		KVPlacement: "gpu",
		KVQuality:   "high",
		BackendTag:  "llama",
		Parallel:    4,
		CacheDir:    cacheDir,
	})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.Type != MoEOffload {
		t.Fatalf("expected MoE offload, got %s", strat.Type)
	}
	if strat.UBatchSize != 64 {
		t.Fatalf("expected the fixture's largest usable ubatch 64, got %d", strat.UBatchSize)
	}
	_, moeCount := moeLayerRange(model)
	if strat.NCPUMoE >= moeCount {
		t.Fatalf("expected at least one expert layer on GPU after the ubatch retry, got NCPUMoE=%d of %d total (ubatch=%d)",
			strat.NCPUMoE, moeCount, strat.UBatchSize)
	}
	if !strings.Contains(strat.OTString, "exps") {
		t.Fatalf("expected at least one GPU expert pin in -ot, got %q", strat.OTString)
	}
}

// TestComputeDeepSeekV4KeepsOneRecurrentCheckpoint is the end-to-end
// version of the bounded hybrid policy: runs the real
// Compute() -> computeCRAM -> Args() pipeline against the exact hardware and
// model shape used for 128GB DeepSeek-V4. VRAM is too tight for host prompt
// CRAM, but the machine has ample host headroom for one recurrent-state
// checkpoint. The default of 32 remains forbidden.
func TestComputeDeepSeekV4KeepsOneRecurrentCheckpoint(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, BandwidthMBps: 15754},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, BandwidthMBps: 985},
			{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12282, BandwidthMBps: 3938},
		},
		RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 120000},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path:            "DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf",
		Basename:        "DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf",
		SizeBytes:       137898617344,
		TotalSizeMB:     131511,
		NumLayers:       44,
		IsMoE:           true,
		NumExperts:      256,
		ExpertUsedCount: 6,
		ExpertFF:        2048,
		ExpertBytes:     131240296448,
		NonExpertBytes:  6658320448,
		ModelArch:       "deepseek4",
		ContextSize:     1048576,
		CTXTrain:        1048576,
		MeasuredKVBytesPerTok: map[string]float64{
			"f16": 6912.25,
		},
	}
	strat, err := Compute(caps, model, Options{
		ContextSize: 1048576,
		KVPlacement: "cpu",
		KVQuality:   "high",
		BackendTag:  "llama",
		Parallel:    1,
		CacheDir:    t.TempDir(),
	})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.CRAM != 0 {
		t.Fatalf("expected computeCRAM to disable the prompt cache under tight VRAM, got CRAM=%d", strat.CRAM)
	}
	if strat.MaxCheckpoints != 1 {
		t.Fatalf("expected one bounded recurrent checkpoint, got MaxCheckpoints=%d", strat.MaxCheckpoints)
	}
	if strat.TensorSplit[1] != 0 {
		t.Fatalf("expected x1 GPU to be expert-only with zero tensor split, got %v", strat.TensorSplit)
	}
	if !otStringUsesDevice(strat.OTString, 1) {
		t.Fatalf("expected x1 GPU to still receive full expert pins, got OT %s", strat.OTString)
	}
	args := strat.Args("/models/test.gguf", 8081)
	if !hasAdjacentArgPlacement(args, "-cram", "0") {
		t.Fatalf("expected explicit '-cram 0' in emitted args, got %v", args)
	}
	if !hasAdjacentArgPlacement(args, "--ctx-checkpoints", "1") {
		t.Fatalf("expected explicit '--ctx-checkpoints 1' in emitted args, got %v", args)
	}
	if !contains(args, "--no-context-shift") {
		t.Fatalf("expected DeepSeek4 recurrent context shifting to be disabled, got %v", args)
	}
}

// TestMaximizeMoEGPUFitByUBatchRescuesExcludedGPU reproduces the 2026-07-08
// incident: a claude-code launch at ctx 262144 landed on tensor-split
// 0.00,0.00,1.00 — two GPUs (36GB combined) got zero share while the model's
// CPU-offloaded remainder filled system RAM to the last byte. The ladder
// previously only rescued a placement with literally zero experts anywhere;
// a placement that "succeeds" but strands whole GPUs is just as broken and
// slipped through untouched.
func TestMaximizeMoEGPUFitByUBatchRescuesExcludedGPU(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, BandwidthMBps: 15754},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, BandwidthMBps: 985},
			{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12282, BandwidthMBps: 3938},
		},
		RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 120000},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path:            "DeepSeek-V4-Flash-MXFP4-00001-of-00005.gguf",
		Basename:        "DeepSeek-V4-Flash-MXFP4-00001-of-00005.gguf",
		SizeBytes:       156378344860,
		TotalSizeMB:     149134,
		NumLayers:       43,
		IsMoE:           true,
		NumExperts:      256,
		ExpertUsedCount: 6,
		ExpertFF:        2048,
		ExpertBytes:     148319502336,
		NonExpertBytes:  8053508160,
		TokenEmbdBytes:  1059061760,
		OutputBytes:     1059061760,
		ShexpBytes:      1149763584,
		ContextSize:     262144,
		CTXTrain:        1048576,
		EmbeddingLength: 4096,
		HiddenSize:      4096,
		HeadCountKV:     1,
		KVLoraRank:      512,
		QLoraRank:       1024,
		SlidingWindow:   128,
		ModelArch:       "deepseek4",
		MeasuredKVBytesPerTok: map[string]float64{
			"f16": 6912.25 * 2.0 / 1.0625,
		},
	}

	cacheDir := t.TempDir()
	gpus := caps.GPUs
	// A GPU with a much larger measured compute buffer at ubatch 512 than at
	// smaller ubatch — enough that at 512 it can't cover its own overhead+KV
	// share and gets removed from the split entirely, exactly like CUDA0/1
	// did in the real incident.
	measured := map[int]map[int]int{
		512: {0: 21000, 1: 11000, 2: 2000},
		256: {0: 10000, 1: 5000, 2: 1200},
		128: {0: 5000, 1: 2500, 2: 700},
		64:  {0: 2500, 1: 1300, 2: 500},
	}
	for ub, byGPU := range measured {
		if err := writeProbeCacheForModel(cacheDir, model, 262144, ub, "high", "gpu", "llama", gpus, 4, byGPU, nil, 0); err != nil {
			t.Fatalf("seed probe cache ubatch=%d: %v", ub, err)
		}
	}

	strat, err := Compute(caps, model, Options{
		ContextSize: 262144,
		KVPlacement: "gpu",
		KVQuality:   "high",
		BackendTag:  "llama",
		Parallel:    4,
		CacheDir:    cacheDir,
	})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.Type != MoEOffload {
		t.Fatalf("expected MoE offload, got %s", strat.Type)
	}
	if excluded := numGPUsExcluded(strat, caps.GPUs); excluded > 0 {
		t.Fatalf("expected the ladder to rescue every GPU into the split, got %d excluded (ubatch=%d, split=%v)",
			excluded, strat.UBatchSize, strat.TensorSplit)
	}
	if strat.UBatchSize >= 512 {
		t.Fatalf("expected the ladder to drop below the default ubatch that stranded a GPU, got %d", strat.UBatchSize)
	}
}

func TestComputeMoEUsesSlowPCIeGPUAsExpertOnly(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, BandwidthMBps: 15754},
			{Index: 1, Name: "RTX 3060 x1", VRAMTotalMB: 12288, BandwidthMBps: 985},
			{Index: 2, Name: "RTX 4070 x4", VRAMTotalMB: 12282, BandwidthMBps: 3938},
		},
		RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 120000},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path:           "slow-pcie-moe.gguf",
		TotalSizeMB:    50 * 1024,
		SizeBytes:      50 * 1024 * 1024 * 1024,
		NumLayers:      12,
		IsMoE:          true,
		NumExperts:     128,
		ExpertBytes:    int64(12 * 3500 * 1024 * 1024),
		NonExpertBytes: int64(8 * 1024 * 1024 * 1024),
		ContextSize:    32768,
		HeadCountKV:    0,
	}

	strat, err := Compute(caps, model, Options{ContextSize: 32768, KVPlacement: "cpu", KVQuality: "low", CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if len(strat.TensorSplit) != len(caps.GPUs) {
		t.Fatalf("expected tensor split for all devices, got %v", strat.TensorSplit)
	}
	if strat.TensorSplit[1] != 0 {
		t.Fatalf("expected x1 GPU to be excluded from tensor-split ownership, got split %v", strat.TensorSplit)
	}
	if !otStringUsesDevice(strat.OTString, 1) {
		t.Fatalf("expected x1 GPU to still receive whole expert pins, got OT %s", strat.OTString)
	}
	if strat.TensorSplit[2] != 0 || !otStringUsesDevice(strat.OTString, 2) {
		t.Fatalf("expected x4 GPU to be whole-expert storage too, split=%v OT=%s", strat.TensorSplit, strat.OTString)
	}
	if excluded := numGPUsExcluded(strat, caps.GPUs); excluded != 0 {
		t.Fatalf("expert-only GPU must count as used, got excluded=%d split=%v ot=%s", excluded, strat.TensorSplit, strat.OTString)
	}
	for _, part := range strings.Split(strat.OTString, ",") {
		if strings.Contains(part, "=CUDA1") && !strings.Contains(part, "_shexp") {
			t.Fatalf("slow expert-only GPU must get full expert layers, not partial sub-pins: %s", strat.OTString)
		}
	}
}

func TestExpertOnlySlowGPUUsesExpertReserveNotSplitOwnerReserve(t *testing.T) {
	gpus := []detect.GPU{
		{Index: 0, Name: "fast", VRAMTotalMB: 24576, BandwidthMBps: 16000},
		{Index: 1, Name: "slow-x1", VRAMTotalMB: 12288, BandwidthMBps: 900},
		{Index: 2, Name: "medium-x8", VRAMTotalMB: 12288, BandwidthMBps: 8000},
	}
	// The slow GPU would fail a normal split-owner reserve, but it can safely
	// hold whole expert layers after the grounded expert-only reserve is used.
	// nonExpertPerLayerMB is small so only the bandwidth trigger fires here.
	splitFixed := []int{2048, 10000, 2048}
	expertOnlyFixed := []int{2048, 1024, 2048}
	expertOnly := expertOnlySlowGPUs(gpus, splitFixed, expertOnlyFixed, 3000, 1)
	if !expertOnly[1] {
		t.Fatalf("expected slow x1 GPU to classify expert-only, got %v", expertOnly)
	}
	if expertOnly[0] || expertOnly[2] {
		t.Fatalf("only the slow x1 GPU should classify expert-only, got %v", expertOnly)
	}
}

func TestExpertOnlyCapacityRespectsCurrentFreeVRAM(t *testing.T) {
	gpus := []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576, VRAMUsedMB: 0, BandwidthMBps: 16000},
		{Index: 1, VRAMTotalMB: 12288, VRAMUsedMB: 10500, BandwidthMBps: 1000},
	}
	expertOnly := expertOnlySlowGPUs(gpus, []int{2000, 2000}, []int{1000, 1000}, 3000, 200)
	if expertOnly[1] {
		t.Fatalf("slow GPU with only %d MiB free must not be classified as able to store a 3000 MiB expert layer", gpus[1].VRAMFreeMB())
	}
}

// TestExpertOnlyCapacityTrigger verifies the OR capacity path: a GPU whose
// PCIe link is fast enough to own dense layers (bandwidth ratio above 0.33)
// but whose VRAM cannot fit the split-owner compute reserve plus one dense
// layer's non-expert weight is still classified expert-only.
func TestExpertOnlyCapacityTrigger(t *testing.T) {
	gpus := []detect.GPU{
		{Index: 0, Name: "fast-big", VRAMTotalMB: 24576, BandwidthMBps: 16000},
		{Index: 1, Name: "fast-small", VRAMTotalMB: 4096, BandwidthMBps: 16000},
	}
	// GPU1 has the same fast link as GPU0 (ratio 1.0, above 0.33), so the
	// bandwidth trigger does NOT fire. But its VRAM after the split-owner
	// reserve is too small for one dense layer (nonExpertPerLayerMB=2000),
	// so the capacity trigger must classify it expert-only. Its expert-only
	// reserve leaves enough room for one expert layer (3000 MB).
	splitFixed := []int{2048, 3000}
	expertOnlyFixed := []int{2048, 512}
	expertOnly := expertOnlySlowGPUs(gpus, splitFixed, expertOnlyFixed, 3000, 2000)
	if !expertOnly[1] {
		t.Fatalf("expected small-VRAM GPU to classify expert-only via capacity trigger, got %v", expertOnly)
	}
	if expertOnly[0] {
		t.Fatalf("fast-big GPU should not classify expert-only, got %v", expertOnly)
	}
}

// TestExpertOnlyRetrofitAfterSplitElimination verifies the post-split retrofit:
// a GPU that is NOT classified expert-only by bandwidth or capacity (its link
// is fast and it fits compute+dense pre-split) but gets eliminated from the
// tensor split by KV/compute pressure is retrofitted as expert-only so its
// VRAM is not left idle.
func TestExpertOnlyRetrofitAfterSplitElimination(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, BandwidthMBps: 15754},
			{Index: 1, Name: "RTX 3060 x1", VRAMTotalMB: 12288, BandwidthMBps: 985},
			{Index: 2, Name: "RTX 4070 x8", VRAMTotalMB: 12282, BandwidthMBps: 8000},
		},
		RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 120000},
		CPU: detect.CPUInfo{Cores: 8},
	}
	// Model sized so GPU2 is NOT bandwidth-expert-only (ratio ~0.51 > 0.33)
	// and fits compute+dense pre-split, but a large KV cache at 1M ctx
	// eliminates it from the tensor split. The retrofit must then give it
	// expert layers instead of leaving it idle.
	model := &ModelProfile{
		Path:           "retrofit-moe.gguf",
		TotalSizeMB:    50 * 1024,
		SizeBytes:      50 * 1024 * 1024 * 1024,
		NumLayers:      12,
		IsMoE:          true,
		NumExperts:     128,
		ExpertBytes:    int64(12 * 3500 * 1024 * 1024),
		NonExpertBytes: int64(8 * 1024 * 1024 * 1024),
		ContextSize:    32768,
		HeadCountKV:    0,
	}
	strat, err := Compute(caps, model, Options{ContextSize: 32768, KVPlacement: "cpu", KVQuality: "low", CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	// GPU2 must not be stranded: either it has a tensor-split share, or it is
	// expert-only and appears in the -ot string with =CUDA2.
	if strat.TensorSplit[2] <= 0 && !otStringUsesDevice(strat.OTString, 2) {
		t.Fatalf("GPU2 stranded: split=%v and not in OT %s", strat.TensorSplit, strat.OTString)
	}
	if excluded := numGPUsExcluded(strat, caps.GPUs); excluded != 0 {
		t.Fatalf("no GPU should be excluded, got excluded=%d split=%v ot=%s", excluded, strat.TensorSplit, strat.OTString)
	}
}

// TestRecordMeasuredComputeBuffersMergesNotClobbers guards the preflight's
// write path (cmd/ggrun/preflight.go): recording fresh compute-buffer
// measurements must not erase runtime-growth or KV-per-layer data already
// recorded for the same key, since writeProbeCacheForModel fully rewrites
// the cache file rather than patching it.
func TestRecordMeasuredComputeBuffersMergesNotClobbers(t *testing.T) {
	cacheDir := t.TempDir()
	model := &ModelProfile{Path: "model.gguf", NumLayers: 43, NumExperts: 256}
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 24564}}

	if err := RecordRuntimeGraphGrowth(cacheDir, model, 1048576, 512, "high", "gpu", "llama", gpus, 1, map[int]int{0: 1000}); err != nil {
		t.Fatalf("seed runtime growth: %v", err)
	}
	if err := RecordMeasuredComputeBuffers(cacheDir, model, 1048576, 512, "high", "gpu", "llama", gpus, 1, map[int]int{0: 17970}); err != nil {
		t.Fatalf("record compute buffers: %v", err)
	}

	pc := loadProbeCache(cacheDir, model, 1048576, 512, "high", "gpu", "llama", gpus, 1)
	if pc == nil {
		t.Fatal("expected a probe cache entry")
	}
	if pc.ComputeBufByGPU[0] != 17970 {
		t.Fatalf("expected recorded compute buffer 17970, got %d", pc.ComputeBufByGPU[0])
	}
	if pc.RuntimeGraphGrowthByGPU[0] != 1000 {
		t.Fatalf("expected prior runtime growth 1000 preserved, got %d", pc.RuntimeGraphGrowthByGPU[0])
	}
}

func TestArgsOmitFlashAttentionWhenDisabled(t *testing.T) {
	s := &Strategy{
		ContextSize:    4096,
		GPULayers:      32,
		KVQuality:      "mid",
		FlashAttention: false,
		Threads:        16,
		BatchSize:      2048,
		UBatchSize:     512,
	}
	args := s.Args("/models/test.gguf", 8081)
	if contains(args, "--flash-attn") {
		t.Fatalf("args should leave flash attention at backend default when disabled: %v", args)
	}
}

func TestFlashAttentionFollowsKVPlacement(t *testing.T) {
	m := &ModelProfile{ModelArch: "deepseek4"}
	if defaultFlashAttention(m, Options{BackendTag: "llama"}, "cpu") {
		t.Fatal("KV on CPU auto-disables flash attention; claiming it is on would emit a self-contradicting --flash-attn on --no-kv-offload command")
	}
	if !defaultFlashAttention(m, Options{BackendTag: "llama"}, "gpu") {
		t.Fatal("mainline deepseek4 with KV on GPU must default flash attention on (bounds the compute buffer; see Task #10)")
	}
}

func contains(slice []string, val string) bool {
	for _, v := range slice {
		if v == val {
			return true
		}
	}
	return false
}

var otDevicePattern = regexp.MustCompile(`blk\\\.\(([^)]*)\).*=(?:CUDA|Vulkan)(\d+)`)

func parseOTLayersByDevice(t *testing.T, ot string) map[int][]int {
	t.Helper()
	out := map[int][]int{}
	for _, part := range strings.Split(ot, ",") {
		m := otDevicePattern.FindStringSubmatch(part)
		if m == nil {
			continue
		}
		device, err := strconv.Atoi(m[2])
		if err != nil {
			t.Fatalf("parse device from %q: %v", part, err)
		}
		for _, raw := range strings.Split(m[1], "|") {
			layer, err := strconv.Atoi(raw)
			if err != nil {
				t.Fatalf("parse layer from %q: %v", part, err)
			}
			out[device] = append(out[device], layer)
		}
	}
	return out
}

// otExpertMBByDevice charges each -ot pin its real VRAM: a whole-layer pin (its
// pattern includes the shared expert, "_shexp") costs the full expertPerLayerMB,
// while a sub-layer gate+up pin (down stays on CPU) costs 2/3 of it — matching
// buildOTStringWithSubPins / packGateUpChunks.
func otExpertMBByDevice(t *testing.T, ot string, expertPerLayerMB int) map[int]int {
	t.Helper()
	out := map[int]int{}
	for _, part := range strings.Split(ot, ",") {
		m := otDevicePattern.FindStringSubmatch(part)
		if m == nil {
			continue
		}
		device, err := strconv.Atoi(m[2])
		if err != nil {
			t.Fatalf("parse device from %q: %v", part, err)
		}
		nLayers := len(strings.Split(m[1], "|"))
		per := expertPerLayerMB
		if !strings.Contains(part, "_shexp") { // gate+up-only sub-pin
			per = 2 * expertPerLayerMB / 3
		}
		out[device] += nLayers * per
	}
	return out
}

func TestNormalizeSplit(t *testing.T) {
	split := normalizeSplit([]float64{12288, 12288})
	if len(split) != 2 {
		t.Fatalf("expected 2 values")
	}
	if split[0] != 0.5 || split[1] != 0.5 {
		t.Fatalf("expected equal split, got %v", split)
	}
}

func TestBuildOTString(t *testing.T) {
	gpus := []detect.GPU{
		{Index: 0},
		{Index: 1},
	}
	gpuOrder := []int{0, 1}

	// Patterns include chunked expert weights plus the routed/hash-gate tensors
	// needed to dispatch those experts on the assigned device.

	// Single layer on GPU0
	ot := buildOTString([]int{1, 0}, gpus, gpuOrder, "")
	if ot != `blk\.(0)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=CUDA0,exps=CPU` {
		t.Fatalf("single-layer OT mismatch: %s", ot)
	}

	// Multiple layers on GPU0
	ot = buildOTString([]int{5, 0}, gpus, gpuOrder, "")
	expected := `blk\.(0|1|2|3|4)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=CUDA0,exps=CPU`
	if ot != expected {
		t.Fatalf("multi-layer OT mismatch:\n  got:      %s\n  expected: %s", ot, expected)
	}

	// Layers on both GPUs
	ot = buildOTString([]int{2, 3}, gpus, gpuOrder, "")
	expected = `blk\.(0|1)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=CUDA0,blk\.(2|3|4)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=CUDA1,exps=CPU`
	if ot != expected {
		t.Fatalf("two-gpu OT mismatch:\n  got:      %s\n  expected: %s", ot, expected)
	}

	// Vulkan uses Vulkan device names in override tensors.
	ot = buildOTString([]int{1, 0}, gpus, gpuOrder, "vulkan")
	if ot != `blk\.(0)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=Vulkan0,exps=CPU` {
		t.Fatalf("vulkan OT mismatch: %s", ot)
	}

}

func TestDefaultContextSize(t *testing.T) {
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 65536},
	}
	model := &ModelProfile{
		NumLayers:   32,
		HiddenSize:  4096,
		ContextSize: 0,
	}
	ctx := defaultContextSize(model, caps)
	if ctx < 4096 {
		t.Fatalf("context too small: %d", ctx)
	}
}

func TestComputeDenseMultiGPU(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 12288},
			{Index: 1, VRAMTotalMB: 12288},
		},
		RAM: detect.RAMInfo{TotalMB: 65536},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "model.gguf",
		SizeBytes:   15 * 1024 * 1024 * 1024,
		NumLayers:   64,
		NumParams:   32_000_000_000,
		IsMoE:       false,
		ContextSize: 32768,
		HiddenSize:  4096,
	}
	strat, err := Compute(caps, model, Options{})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if len(strat.TensorSplit) != 2 {
		t.Fatalf("expected tensor split for multi-GPU, got %v", strat.TensorSplit)
	}
	if strat.SplitMode != "layer" {
		t.Fatalf("expected portable split-mode layer for heterogeneous dense multi-GPU, got %s", strat.SplitMode)
	}
}

func TestComputeMoEMultiGPU(t *testing.T) {
	// 60GB MoE on two 24GB GPUs with 128GB RAM
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576},
			{Index: 1, VRAMTotalMB: 24576},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "moe.gguf",
		SizeBytes:   60 * 1024 * 1024 * 1024,
		NumLayers:   64,
		NumParams:   120_000_000_000,
		IsMoE:       true,
		NumExperts:  64,
		ContextSize: 32768,
		HiddenSize:  4096,
	}
	strat, err := Compute(caps, model, Options{})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	// MoE uses NCPUMoE for CPU expert offload
	if strat.NCPUMoE == 0 {
		t.Fatalf("expected MoE CPU expert offload")
	}
}

func TestComputeMoEHeterogeneousMultiGPUExactLedger(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24576, VRAMUsedMB: 822, BandwidthMBps: 31504},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, VRAMUsedMB: 574, BandwidthMBps: 12000},
			{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12288, VRAMUsedMB: 660, BandwidthMBps: 25203},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 78000},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path:            "MiniMax-M3.gguf",
		TotalSizeMB:     149 * 1024,
		SizeBytes:       149 * 1024 * 1024 * 1024,
		NumLayers:       60,
		IsMoE:           true,
		NumExperts:      128,
		LeadingDense:    3,
		ExpertBytes:     int64(57 * 2500 * 1024 * 1024),
		NonExpertBytes:  int64(6500 * 1024 * 1024),
		ContextSize:     32768,
		EmbeddingLength: 6144,
		HeadCountKV:     4,
		KeyLength:       128,
		ValueLength:     128,
		ExpertUsedCount: 4,
		ExpertFF:        3072,
	}

	strat, err := Compute(caps, model, Options{ContextSize: 32768, KVQuality: "low", CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.Type != MoEOffload {
		t.Fatalf("expected MoE offload, got %s", strat.Type)
	}
	if len(strat.TensorSplit) != len(caps.GPUs) {
		t.Fatalf("expected tensor split for every visible GPU, got %v", strat.TensorSplit)
	}
	if strat.SplitMode != "layer" {
		t.Fatalf("expected MoE split-mode layer, got %q", strat.SplitMode)
	}
	if !strings.Contains(strat.OTString, "exps=CPU") {
		t.Fatalf("expected CPU expert catch-all in -ot, got %s", strat.OTString)
	}

	assignments := parseOTLayersByDevice(t, strat.OTString)
	for device, layers := range assignments {
		for _, layer := range layers {
			if layer < model.LeadingDense {
				t.Fatalf("device %d pinned leading dense layer %d in %s", device, layer, strat.OTString)
			}
		}
	}
	expertPerLayerMB := ceilDivInt(bytesToMiBCeil(model.ExpertBytes), model.NumLayers-model.LeadingDense)
	nonExpertTotalMB := bytesToMiBCeil(model.NonExpertBytes)
	kvTotalMB := computeKVTotalMB(model, strat.ContextSize, strat.KVType)
	fixedPerGPU := computeFloorMB
	expertMBByDevice := otExpertMBByDevice(t, strat.OTString, expertPerLayerMB)
	for gi, gpu := range caps.GPUs {
		usedMB := fixedPerGPU + splitShareMB(nonExpertTotalMB, strat.TensorSplit, gi) + splitShareMB(kvTotalMB, strat.TensorSplit, gi) + expertMBByDevice[gpu.Index]
		if usedMB > gpu.VRAMFreeMB() {
			t.Fatalf("gpu %d over budget: used=%dMB free=%dMB split=%v ot=%s", gpu.Index, usedMB, gpu.VRAMFreeMB(), strat.TensorSplit, strat.OTString)
		}
	}
}

func TestComputeMoEMultiGPUFullyFitsExpertsOnGPU(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "GPU A", VRAMTotalMB: 24576, BandwidthMBps: 20000},
			{Index: 1, Name: "GPU B", VRAMTotalMB: 24576, BandwidthMBps: 20000},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:            "moe.gguf",
		TotalSizeMB:     32 * 1024,
		SizeBytes:       32 * 1024 * 1024 * 1024,
		NumLayers:       32,
		IsMoE:           true,
		NumExperts:      64,
		ExpertBytes:     16 * 1024 * 1024 * 1024,
		NonExpertBytes:  16 * 1024 * 1024 * 1024,
		ContextSize:     32768,
		EmbeddingLength: 4096,
		HeadCountKV:     8,
		KeyLength:       128,
		ValueLength:     128,
	}

	strat, err := Compute(caps, model, Options{ContextSize: 32768, KVQuality: "low", CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	assignments := parseOTLayersByDevice(t, strat.OTString)
	totalPinned := 0
	for _, layers := range assignments {
		totalPinned += len(layers)
	}
	if totalPinned != model.NumLayers {
		t.Fatalf("expected all %d expert layers on GPU, got %d via %s", model.NumLayers, totalPinned, strat.OTString)
	}
	if strat.NCPUMoE != 0 {
		t.Fatalf("expected no CPU MoE layers when experts fit, got %d", strat.NCPUMoE)
	}
}

func TestFirstLaunchComputeBufForGPUKeepsEverySplitOwnerConservative(t *testing.T) {
	order := []int{2, 0, 1}
	primary := firstLaunchComputeBufMBForGPU(nil, 512, 2, order)
	secondary := firstLaunchComputeBufMBForGPU(nil, 512, 0, order)
	if primary != firstLaunchComputeBufMB(nil, 512) {
		t.Fatalf("primary fallback = %d, want %d", primary, firstLaunchComputeBufMB(nil, 512))
	}
	if secondary != primary {
		t.Fatalf("secondary split-owner fallback = %d, want full graph reserve %d", secondary, primary)
	}
}

func TestFirstLaunchComputeBufMoEScalesWithFanoutAndParallel(t *testing.T) {
	model := &ModelProfile{
		IsMoE: true, ModelArch: "deepseek4", HiddenSize: 4096, NumLayers: 44, ExpertUsedCount: 6,
	}
	serial := firstLaunchComputeBufMBParallel(model, 256, 1)
	parallel4 := firstLaunchComputeBufMBParallel(model, 256, 4)
	if serial < 33000 || serial > 37000 {
		t.Fatalf("serial MoE graph reserve = %d MiB, want measured-scale ~34 GiB", serial)
	}
	if parallel4 < 8500 || parallel4 > 9500 {
		t.Fatalf("parallel-4 MoE graph reserve = %d MiB, want measured-scale ~8.9 GiB", parallel4)
	}
	if serial < parallel4*3 || serial > parallel4*5 {
		t.Fatalf("parallel scaling inconsistent: serial=%d parallel4=%d", serial, parallel4)
	}
}

func TestComputeMoESingleGPUDoesNotEmitTensorSplit(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "RTX", VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:           "moe.gguf",
		TotalSizeMB:    48 * 1024,
		SizeBytes:      48 * 1024 * 1024 * 1024,
		NumLayers:      48,
		IsMoE:          true,
		NumExperts:     64,
		ExpertBytes:    40 * 1024 * 1024 * 1024,
		NonExpertBytes: 8 * 1024 * 1024 * 1024,
		ContextSize:    32768,
		HeadCountKV:    8,
		KeyLength:      128,
		ValueLength:    128,
	}
	strat, err := Compute(caps, model, Options{ContextSize: 32768, KVQuality: "low", CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if len(strat.TensorSplit) != 0 {
		t.Fatalf("single-GPU MoE should not emit tensor split, got %v", strat.TensorSplit)
	}
	if !strings.Contains(strat.OTString, "exps=CPU") {
		t.Fatalf("single-GPU MoE still needs CPU catch-all, got %s", strat.OTString)
	}
}

func TestPlacementCacheRequiresTensorSplitForAssignments(t *testing.T) {
	path := filepath.Join(t.TempDir(), "old.conf")
	if err := os.WriteFile(path, []byte("CACHED_GPU_ASSIGNMENTS=\"0:0:4\"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}}}
	if _, err := LoadPlacementCache(path, caps, 0); err == nil {
		t.Fatal("expected old MoE assignment cache without tensor split to be rejected")
	}
}

func TestPlacementCacheRoundTripsTensorSplit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.conf")
	entry := &CacheEntry{
		GPUAssignments: []GPUAssignment{{CUDAIndex: 0, Start: 3, Count: 4}},
		TensorSplit:    []float64{0.5, 0.25, 0.25},
		SplitMode:      "layer",
		NCPUMoE:        53,
		BatchSize:      2048,
		UBatchSize:     512,
	}
	if err := SavePlacementCache(path, entry); err != nil {
		t.Fatalf("save cache: %v", err)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 12288}, {Index: 2, VRAMTotalMB: 12288}}}
	loaded, err := LoadPlacementCache(path, caps, 0)
	if err != nil {
		t.Fatalf("load cache: %v", err)
	}
	if loaded.SplitMode != "layer" || len(loaded.TensorSplit) != 3 || loaded.TensorSplit[0] != 0.5 {
		t.Fatalf("tensor split did not round trip: %+v", loaded)
	}
}

func TestCachedHybridPlacementRecomputesRuntimeCheckpointPolicy(t *testing.T) {
	path := filepath.Join(t.TempDir(), "hybrid-cache.conf")
	if err := SavePlacementCache(path, &CacheEntry{
		TensorSplit: []float64{0.67, 0.33}, SplitMode: "layer", NCPUMoE: 30,
		BatchSize: 2048, UBatchSize: 128, Parallel: 2, MMap: false,
	}); err != nil {
		t.Fatal(err)
	}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576},
			{Index: 1, VRAMTotalMB: 12288},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "hybrid-moe.gguf", TotalSizeMB: 65536, NumLayers: 40,
		IsMoE: true, NumExperts: 64, HasSSM: 1, ContextSize: 131072,
		HiddenSize: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	strategy, err := Compute(caps, model, Options{
		ContextSize: 131072, Parallel: 2, CacheFile: path,
		KVPlacement: "gpu", KVQuality: "mid",
	})
	if err != nil {
		t.Fatal(err)
	}
	if strategy.MaxCheckpoints != 1 {
		t.Fatalf("cached hybrid placement checkpoints=%d, want one", strategy.MaxCheckpoints)
	}
	if args := strategy.Args(model.Path, 8081); !hasAdjacentArgPlacement(args, "--ctx-checkpoints", "1") {
		t.Fatalf("cached hybrid placement did not emit checkpoint policy: %v", args)
	}
}

func TestDerateCUDAOOMArgsMovesExpertLayersToCPU(t *testing.T) {
	model := &ModelProfile{
		NumLayers:    60,
		LeadingDense: 3,
		ExpertBytes:  int64(57 * 2500 * 1024 * 1024),
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576},
		{Index: 1, VRAMTotalMB: 12288, VRAMUsedMB: 574},
	}}
	args := []string{
		"--tensor-split", "0.67,0.33",
		"--split-mode", "layer",
		"-b", "2048",
		"-ub", "512",
		"--parallel", "1",
		"-ot", `blk\.(3|4|5|6)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,blk\.(7|8|9|10)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA1,exps=CPU`,
		"--n-cpu-moe", "49",
	}
	newArgs, entry, ok := DerateCUDAOOMArgs(args, model, caps, 1, 11876, false)
	if !ok {
		t.Fatal("expected CUDA OOM args to derate")
	}
	newOT := newArgs[argIndex(newArgs, "-ot")+1]
	assignments := parseOTLayersByDevice(t, newOT)
	if got := len(assignments[1]); got != 3 {
		t.Fatalf("expected device 1 to drop one layer, got %d via %s", got, newOT)
	}
	if currentNCPUMoE(newArgs) != 50 {
		t.Fatalf("expected --n-cpu-moe to increment to 50, got args %v", newArgs)
	}
	if entry == nil || len(entry.GPUAssignments) != 2 || entry.NCPUMoE != 50 || len(entry.TensorSplit) != 2 {
		t.Fatalf("unexpected cache entry: %+v", entry)
	}
}

func TestDerateCUDAOOMArgsShrinksUBatchForComputeBufferOOM(t *testing.T) {
	model := &ModelProfile{
		NumLayers:    60,
		LeadingDense: 3,
		ExpertBytes:  int64(57 * 2500 * 1024 * 1024),
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576},
		{Index: 1, VRAMTotalMB: 12288, VRAMUsedMB: 574},
	}}
	args := []string{
		"--tensor-split", "0.67,0.33",
		"--split-mode", "layer",
		"-b", "2048",
		"-ub", "512",
		"--parallel", "1",
		"-ot", `blk\.(3|4|5|6)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,blk\.(7|8|9|10)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA1,exps=CPU`,
		"--n-cpu-moe", "49",
	}
	newArgs, entry, ok := DerateCUDAOOMArgs(args, model, caps, 1, 599, true)
	if !ok {
		t.Fatal("expected a compute-buffer OOM to derate by shrinking ubatch")
	}
	if got := currentUBatch(newArgs); got != 256 {
		t.Fatalf("expected -ub to step down to 256, got %d via %v", got, newArgs)
	}
	// Expert layout must be untouched — a compute-buffer OOM has nothing to
	// do with which expert layers are GPU-resident.
	if currentNCPUMoE(newArgs) != 49 {
		t.Fatalf("expected --n-cpu-moe to stay at 49, got args %v", newArgs)
	}
	if entry == nil || entry.UBatchSize != 256 {
		t.Fatalf("expected cache entry to carry the new ubatch, got %+v", entry)
	}

	// Once ubatch is already at the ladder floor, fall back to the layer-drop lever.
	args[argIndex(args, "-ub")+1] = "64"
	newArgs, _, ok = DerateCUDAOOMArgs(args, model, caps, 1, 599, true)
	if !ok {
		t.Fatal("expected fallback to layer-drop once ubatch is at its floor")
	}
	if currentUBatch(newArgs) != 64 {
		t.Fatalf("expected -ub to stay at the floor, got %v", newArgs)
	}
	if currentNCPUMoE(newArgs) != 50 {
		t.Fatalf("expected the layer-drop fallback to fire, got %v", newArgs)
	}
}

func TestArgsFull(t *testing.T) {
	s := &Strategy{
		ContextSize:    4096,
		GPULayers:      32,
		MainGPU:        0,
		TensorSplit:    []float64{0.5, 0.5},
		SplitMode:      "layer",
		KVPlacement:    "cpu",
		KVQuality:      "high",
		NCPUMoE:        64,
		FlashAttention: true,
		MMap:           false,
		MLock:          true,
		Threads:        16,
		BatchSize:      2048,
		UBatchSize:     512,
	}
	args := s.Args("/models/test.gguf", 8081)
	checks := map[string]bool{
		"-m": false, "--port": false, "--ctx-size": false, "-ngl": false,
		"--flash-attn": false,
	}
	for _, a := range args {
		if _, ok := checks[a]; ok {
			checks[a] = true
		}
	}
	for k, v := range checks {
		if !v {
			t.Fatalf("args missing %s", k)
		}
	}
}

func TestComputeDraftNgramOptIn(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false}

	draft := ComputeDraft(model, caps, Options{SpecMode: "ngram"})
	if draft.Type != DraftNgram {
		t.Fatalf("expected ngram draft, got %s", draft.Type)
	}
	args := DraftFlags(draft)
	if !contains(args, "--spec-ngram-map-k-size-n") {
		t.Fatalf("ngram flags missing expected values: %v", args)
	}
}

func TestComputeDraftSpecAutoTuneRequiresSupport(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false}

	plain := ComputeDraft(model, caps, Options{SpecMode: "ngram", BackendTag: "vulkan", BackendHelp: "--spec-type [ngram-map-k]"})
	if contains(DraftFlags(plain), "--spec-autotune") {
		t.Fatalf("did not expect spec-autotune without backend support: %v", DraftFlags(plain))
	}
	supported := ComputeDraft(model, caps, Options{SpecMode: "ngram", BackendTag: "vulkan", BackendHelp: "--spec-type [ngram-map-k] --spec-autotune"})
	if !contains(DraftFlags(supported), "--spec-autotune") {
		t.Fatalf("expected spec-autotune when backend advertises it: %v", DraftFlags(supported))
	}
}

func TestComputeDraftMTPIKOnly(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "moe.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: true, NextNPredictLayers: 1}

	blocked := ComputeDraft(model, caps, Options{SpecMode: "mtp", BackendTag: "llama"})
	if blocked.Type != DraftNone {
		t.Fatalf("expected mainline MTP to be skipped, got %s", blocked.Type)
	}
	mtp := ComputeDraft(model, caps, Options{SpecMode: "mtp", BackendTag: "ik_llama"})
	if mtp.Type != DraftMTP {
		t.Fatalf("expected ik MTP, got %s", mtp.Type)
	}
	args := DraftFlags(mtp)
	if !contains(args, "--multi-token-prediction") || !contains(args, "--spec-type") {
		t.Fatalf("MTP flags missing expected values: %v", args)
	}
}

func TestDraftFlagsIKDraftUsesCanonicalSpecType(t *testing.T) {
	cfg := &DraftConfig{Type: DraftModel, BackendTag: "ik_llama", Path: "draft.gguf", DraftGPU: 0, CTXSizeDraft: 8192, KVTypeDraft: "q8_0", ThreadsDraft: 2, DraftMax: 16, PSplit: 0.10, SpecAutoTune: true}
	args := DraftFlags(cfg)
	if !contains(args, "--spec-type") || !contains(args, "draft:n_max=16") {
		t.Fatalf("expected canonical IK draft spec-type, got %v", args)
	}
	if contains(args, "--draft-max") || contains(args, "--spec-draft-n-max") {
		t.Fatalf("IK draft flags should not use legacy draft max flags: %v", args)
	}
	if !contains(args, "--p-split") || !contains(args, "--spec-autotune") {
		t.Fatalf("IK draft flags missing p-split/autotune: %v", args)
	}
}

func TestDraftFlagsIKMTPUsesCanonicalNMax(t *testing.T) {
	cfg := &DraftConfig{Type: DraftMTP, BackendTag: "ik_llama", SpecType: "mtp", DraftMax: 4, MTPFlag: true}
	args := DraftFlags(cfg)
	if !contains(args, "--spec-type") || !contains(args, "mtp:n_max=4") || !contains(args, "--multi-token-prediction") {
		t.Fatalf("expected canonical IK MTP flags, got %v", args)
	}
	if contains(args, "--draft-max") || contains(args, "--spec-draft-n-max") {
		t.Fatalf("IK MTP flags should not use legacy draft max flags: %v", args)
	}
}

func TestComputeDraftMTPSkipsWithoutNextNLayers(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false}
	draft := ComputeDraft(model, caps, Options{SpecMode: "mtp", BackendTag: "ik_llama"})
	if draft.Type != DraftNone {
		t.Fatalf("expected MTP to skip without NextN layers, got %s", draft.Type)
	}
}

func TestComputeDraftMoERequiresExplicitOverride(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "moe.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: true}

	blocked := ComputeDraft(model, caps, Options{SpecMode: "ngram"})
	if blocked.Type != DraftNone {
		t.Fatalf("expected MoE speculative decoding to be gated, got %s", blocked.Type)
	}
	forced := ComputeDraft(model, caps, Options{SpecMode: "ngram", ForceSpecMoE: true})
	if forced.Type != DraftNgram {
		t.Fatalf("expected force override to enable ngram, got %s", forced.Type)
	}
}

func TestFindOrDownloadDraftIgnoresInvalidLocalWhenDownloadsSkipped(t *testing.T) {
	t.Setenv("LLM_SERVER_SKIP_DRAFT_DOWNLOAD", "1")
	dir := t.TempDir()
	bad := filepath.Join(dir, "draft-model.gguf")
	if err := os.WriteFile(bad, []byte("not gguf"), 0644); err != nil {
		t.Fatalf("write bad draft: %v", err)
	}
	model := &ModelProfile{Path: filepath.Join(dir, "target.gguf"), TotalSizeMB: 1024, VocabSize: 1}
	if got := findOrDownloadDraftCandidate(model, dir, "ik_llama"); got != "" {
		t.Fatalf("expected invalid local draft to be ignored, got %s", got)
	}
}

func TestDraftCandidateFiltersNonTextArtifacts(t *testing.T) {
	for _, name := range []string{"mmproj-F16.gguf", "vision-projector.gguf", "clip-model.gguf", "Qwen_Qwen3.6-35B-A3B-imatrix.gguf"} {
		if !isNonTextDraftGGUFName(name) {
			t.Fatalf("expected %s to be rejected as non-text draft", name)
		}
		if draftFilenameLooksRelevantForKind(name, "draft") {
			t.Fatalf("did not expect projector %s to be relevant", name)
		}
	}
	if isNonTextDraftGGUFName("Qwen3.5-0.8B-Q4_K_M.gguf") {
		t.Fatal("text draft model was incorrectly rejected")
	}
}

func TestDraftValidationRepoWideMismatch(t *testing.T) {
	if !draftValidationRepoWideMismatch(fmt.Errorf("vocab mismatch: draft=1 target=2")) {
		t.Fatal("expected vocab mismatch to stop repo")
	}
	if !draftValidationRepoWideMismatch(fmt.Errorf("architecture mismatch draft=llama target=qwen")) {
		t.Fatal("expected architecture mismatch to stop repo")
	}
	if draftValidationRepoWideMismatch(fmt.Errorf("incomplete file")) {
		t.Fatal("did not expect incomplete file to stop repo")
	}
}

func TestDraftCandidateRankPrefersQ4Draft(t *testing.T) {
	q4 := draftCandidateRank("Qwen3.5-0.8B-Q4_K_M.gguf", "draft")
	bf16 := draftCandidateRank("Qwen3.5-0.8B-BF16.gguf", "draft")
	iq2 := draftCandidateRank("Qwen3.5-0.8B-IQ2_M.gguf", "draft")
	if !(q4 < bf16 && q4 < iq2) {
		t.Fatalf("expected Q4 draft rank to win, got q4=%d bf16=%d iq2=%d", q4, bf16, iq2)
	}
}

func TestHFSpecSearchQueriesForQwenDraft(t *testing.T) {
	model := &ModelProfile{Path: "Qwen3.6-27B-Q5_K_M.gguf", Basename: "Qwen3.6-27B", ModelArch: "qwen35"}
	queries := hfSpecSearchQueries(model, "draft")
	joined := strings.ToLower(strings.Join(queries, "\n"))
	for _, want := range []string{"qwen3.6 27b draft gguf", "qwen3.6 0.8b gguf", "qwen3.5 0.8b gguf"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("expected query %q in %#v", want, queries)
		}
	}
}

func TestHFRepoLooksRelevantForSmallDraft(t *testing.T) {
	model := &ModelProfile{Path: "Qwen3.6-27B-Q5_K_M.gguf", Basename: "Qwen3.6-27B", ModelArch: "qwen35"}
	if !hfRepoLooksRelevant("bartowski/Qwen3.6-0.8B-GGUF", model, "draft") {
		t.Fatal("expected small same-family repo to be considered as draft candidate")
	}
	if !hfRepoLooksRelevant("unsloth/Qwen3.5-0.8B-GGUF", model, "draft") {
		t.Fatal("expected qwen3.5 architecture-compatible repo to be considered as draft candidate")
	}
	if hfRepoLooksRelevant("unsloth/Qwen3.6-27B-GGUF", model, "draft") {
		t.Fatal("did not expect full-size target repo to be considered as draft candidate")
	}
	if hfRepoLooksRelevant("bartowski/Qwen_Qwen3.6-35B-A3B-GGUF", model, "draft") {
		t.Fatal("did not expect 35B/A3B full MoE repo to be considered as draft candidate")
	}
	if repoLooksLikeDraftRepo("bartowski/qwen_qwen3.6-35b-a3b-gguf") {
		t.Fatal("35B/A3B should not match the small 3B draft heuristic")
	}
	if !hfRepoLooksRelevant("Ex0bit/Qwen3.6-27B-PRISM-EAGLE3", model, "eagle3") {
		t.Fatal("expected target EAGLE repo to be considered relevant")
	}
}

func TestHFCandidateSizeOK(t *testing.T) {
	model := &ModelProfile{TotalSizeMB: 1000}
	if !hfCandidateSizeOK(&http.Response{ContentLength: 250 * 1024 * 1024}, model) {
		t.Fatal("expected small candidate to pass")
	}
	if hfCandidateSizeOK(&http.Response{ContentLength: 500 * 1024 * 1024}, model) {
		t.Fatal("expected oversized candidate to be rejected")
	}
}

func TestDownloadFileResumesPartialContent(t *testing.T) {
	payload := []byte("GGUF" + strings.Repeat("spec-data-", 100))
	partial := 137
	rangeSeen := ""
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rangeSeen = r.Header.Get("Range")
		if rangeSeen != fmt.Sprintf("bytes=%d-", partial) {
			t.Errorf("unexpected Range header %q", rangeSeen)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", partial, len(payload)-1, len(payload)))
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write(payload[partial:])
	}))
	defer srv.Close()

	dest := filepath.Join(t.TempDir(), "draft.gguf.tmp")
	if err := os.WriteFile(dest, payload[:partial], 0644); err != nil {
		t.Fatal(err)
	}
	if err := downloadFile(srv.Client(), srv.URL, dest); err != nil {
		t.Fatalf("resume download: %v", err)
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("resumed file mismatch: got %d bytes, want %d", len(got), len(payload))
	}
}

func TestDownloadFileRestartsWhenServerIgnoresRange(t *testing.T) {
	payload := []byte("GGUF-complete")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
	}))
	defer srv.Close()
	dest := filepath.Join(t.TempDir(), "draft.gguf.tmp")
	if err := os.WriteFile(dest, []byte("GGUF-partial"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := downloadFile(srv.Client(), srv.URL, dest); err != nil {
		t.Fatalf("restart download: %v", err)
	}
	got, _ := os.ReadFile(dest)
	if !bytes.Equal(got, payload) {
		t.Fatalf("server ignored Range but destination was not safely replaced: %q", got)
	}
}

func TestVerifyFileSHA256(t *testing.T) {
	path := filepath.Join(t.TempDir(), "artifact.gguf")
	payload := []byte("GGUF-pinned-artifact")
	if err := os.WriteFile(path, payload, 0644); err != nil {
		t.Fatal(err)
	}
	const sum = "0f5a7057cb6f53b9d50ff176006243f55c321b35b7e92944a2f9b4f05f81f898"
	if err := verifyFileSHA256(path, int64(len(payload)), sum); err != nil {
		t.Fatalf("valid pinned artifact rejected: %v", err)
	}
	if err := verifyFileSHA256(path, int64(len(payload)+1), sum); err == nil || !strings.Contains(err.Error(), "size=") {
		t.Fatalf("expected size mismatch, got %v", err)
	}
	if err := verifyFileSHA256(path, int64(len(payload)), strings.Repeat("0", 64)); err == nil || !strings.Contains(err.Error(), "sha256=") {
		t.Fatalf("expected SHA mismatch, got %v", err)
	}
}

func TestHFResolveURLKeepsPathSeparators(t *testing.T) {
	got := hfResolveURL("org/repo", "folder/a b.gguf")
	want := "https://huggingface.co/org/repo/resolve/main/folder/a%20b.gguf"
	if got != want {
		t.Fatalf("resolve URL mismatch: %s", got)
	}
}

func TestKnownDFlashDownloadIsRevisionPinned(t *testing.T) {
	repo := "Lucebox/DeepSeek-V4-Flash-DSpark-Drafter-GGUF"
	revision := knownSpecializedRepoRevision(repo)
	if revision == "" || revision == "main" {
		t.Fatalf("known DFlash repository must be immutable, got revision %q", revision)
	}
	got := hfResolveURLAt(repo, "folder/a b.gguf", revision)
	want := "https://huggingface.co/" + repo + "/resolve/" + revision + "/folder/a%20b.gguf"
	if got != want {
		t.Fatalf("pinned resolve URL mismatch: %s", got)
	}
}

func TestSameDraftArchitecture(t *testing.T) {
	if !sameDraftArchitecture("qwen2", "qwen2") {
		t.Fatal("expected matching architecture to pass")
	}
	if sameDraftArchitecture("qwen2", "llama") {
		t.Fatal("expected mismatched architecture to fail")
	}
	if !sameDraftArchitecture("", "llama") {
		t.Fatal("missing target metadata should not reject a draft")
	}
}

func TestComputeDraftAutoDoesNotFallbackToNgram(t *testing.T) {
	t.Setenv("LLM_SERVER_SKIP_DRAFT_DOWNLOAD", "1")
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: t.TempDir() + "/model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false}
	help := "--spec-type [none|draft-simple|draft-mtp|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod] --spec-ngram-mod-n-match --spec-ngram-mod-n-min --spec-ngram-mod-n-max"

	draft := ComputeDraft(model, caps, Options{SpecMode: "auto", BackendTag: "vulkan", BackendHelp: help})
	if draft.Type != DraftNone {
		t.Fatalf("expected auto to stay off without MTP/EAGLE/draft, got type=%s spec=%s", draft.Type, draft.SpecType)
	}
}

func TestComputeDraftAutoPrefersMTPWhenAvailable(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: true, NextNPredictLayers: 1}

	opts := Options{SpecMode: "auto", BackendTag: "ik_llama", BackendIdentity: "ik-build-a", CacheDir: t.TempDir(), ContextSize: 32768}
	if draft := ComputeDraft(model, caps, opts); draft.Type != DraftNone {
		t.Fatalf("Auto must stay off without a performance profile, got type=%s", draft.Type)
	}
	saveEligibleSpecProfile(t, model, caps, opts, "mtp", "", 3)
	draft := ComputeDraft(model, caps, opts)
	if draft.Type != DraftMTP || draft.SpecType != "mtp" {
		t.Fatalf("expected auto MTP, got type=%s spec=%s", draft.Type, draft.SpecType)
	}
	if draft.DraftMax != 3 {
		t.Fatalf("profile ceiling not applied: %d", draft.DraftMax)
	}
}

func TestDraftFlagsEagle3(t *testing.T) {
	cfg := &DraftConfig{Type: DraftEagle3, BackendTag: "vulkan", Path: "eagle.gguf", DraftGPU: 0, CTXSizeDraft: 8192, KVTypeDraft: "q8_0", ThreadsDraft: 2, DraftMax: 8}
	args := DraftFlags(cfg)
	if !contains(args, "--spec-type") || !contains(args, "eagle3") || !contains(args, "--model-draft") {
		t.Fatalf("EAGLE-3 flags missing expected values: %v", args)
	}
}

func TestComputeDraftDraftDoesNotFallbackToNgram(t *testing.T) {
	t.Setenv("LLM_SERVER_SKIP_DRAFT_DOWNLOAD", "1")
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: t.TempDir() + "/model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false}
	help := "--spec-type [none|ngram-map-k|ngram-mod] --spec-ngram-mod-n-match"

	draft := ComputeDraft(model, caps, Options{SpecMode: "draft", BackendTag: "llama", BackendHelp: help})
	if draft.Type != DraftNone {
		t.Fatalf("expected explicit draft mode to skip without compatible draft model, got %s", draft.Type)
	}
}

func TestComputeDraftNgramK4VFlags(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false}
	help := "--spec-type [none|ngram-map-k|ngram-map-k4v] --spec-ngram-map-k4v-size-n --spec-ngram-map-k4v-size-m --spec-ngram-map-k4v-min-hits"

	draft := ComputeDraft(model, caps, Options{SpecMode: "ngram-k4v", BackendTag: "vulkan", BackendHelp: help})
	if draft.Type != DraftNgram || draft.SpecType != "ngram-map-k4v" {
		t.Fatalf("expected ngram-map-k4v, got type=%s spec=%s", draft.Type, draft.SpecType)
	}
	args := DraftFlags(draft)
	if !contains(args, "--spec-ngram-map-k4v-size-n") || !contains(args, "--spec-draft-n-max") {
		t.Fatalf("ngram-map-k4v flags missing expected values: %v", args)
	}
}

func TestComputeDraftMainlineMTPWhenAdvertised(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, IsMoE: false, NextNPredictLayers: 1}
	help := "--spec-type [none|draft-simple|draft-mtp|ngram-map-k]"

	draft := ComputeDraft(model, caps, Options{SpecMode: "mtp", BackendTag: "llama", BackendHelp: help})
	if draft.Type != DraftMTP || draft.SpecType != "draft-mtp" {
		t.Fatalf("expected mainline draft-mtp, got type=%s spec=%s", draft.Type, draft.SpecType)
	}
	args := DraftFlags(draft)
	if contains(args, "--multi-token-prediction") {
		t.Fatalf("mainline draft-mtp should not get ik MTP flag: %v", args)
	}
	foundDraftMax := false
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--spec-draft-n-max" {
			foundDraftMax = true
			if args[i+1] != "2" {
				t.Fatalf("MTP must use the conservative two-token default, got %v", args)
			}
		}
	}
	if !foundDraftMax {
		t.Fatalf("MTP must emit the conservative two-token draft ceiling, got %v", args)
	}
}

func TestComputeDraftParallelMTPRespectsBackendCapability(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 1024, NumLayers: 32, ContextSize: 32768, NextNPredictLayers: 1}

	ik := ComputeDraft(model, caps, Options{SpecMode: "mtp", BackendTag: "ik_llama", Parallel: 4})
	if ik.Type != DraftNone {
		t.Fatalf("ik_llama server does not support speculative parallel slots, got %s", ik.Type)
	}

	mainline := ComputeDraft(model, caps, Options{
		SpecMode: "mtp", BackendTag: "llama", BackendHelp: "--spec-type draft-mtp", Parallel: 4,
	})
	if mainline.Type != DraftMTP {
		t.Fatalf("mainline parallel MTP should remain available, got %s", mainline.Type)
	}
}

func TestComputeDraftFindsLocalMTPOnlyCompanion(t *testing.T) {
	t.Setenv("LLM_SERVER_SKIP_DRAFT_DOWNLOAD", "1")
	dir := t.TempDir()
	companion := filepath.Join(dir, "Qwen3.5-9B-MTP-ONLY-Q4_K_M.gguf")
	writeSpecGGUF(t, companion, "qwen35", "gpt2", "qwen35", 4096, 262144, 64, 1)
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	target := &ModelProfile{
		Path: filepath.Join(dir, "Qwen3.5-9B-Q4_K_M.gguf"), TotalSizeMB: 6000,
		ModelArch: "qwen35", EmbeddingLength: 4096, ContextSize: 262144,
		VocabSize: 64, TokenizerModel: "gpt2", TokenizerPre: "qwen35",
	}
	draft := ComputeDraft(target, caps, Options{SpecMode: "mtp", BackendTag: "llama", BackendHelp: "--spec-type draft-mtp --spec-draft-model"})
	if draft.Type != DraftMTP || draft.Path != companion || draft.SpecType != "draft-mtp" {
		t.Fatalf("expected local MTP-only companion, got %#v", draft)
	}
	args := DraftFlags(draft)
	for _, want := range []string{"--model-draft", companion, "--spec-draft-ngl", "all"} {
		if !contains(args, want) {
			t.Fatalf("MTP companion flags missing %q: %v", want, args)
		}
	}
	if contains(args, "--ctx-size-draft") {
		t.Fatalf("current mainline removed --ctx-size-draft; inherited context must not emit it: %v", args)
	}
}

func TestMTPCompanionRejectsTokenizerMismatch(t *testing.T) {
	t.Setenv("LLM_SERVER_SKIP_DRAFT_DOWNLOAD", "1")
	dir := t.TempDir()
	companion := filepath.Join(dir, "model-MTP-ONLY.gguf")
	writeSpecGGUF(t, companion, "qwen35", "gpt2", "wrong-pre", 4096, 262144, 64, 1)
	target := &ModelProfile{
		Path: filepath.Join(dir, "model.gguf"), TotalSizeMB: 6000,
		ModelArch: "qwen35", EmbeddingLength: 4096, VocabSize: 64,
		TokenizerModel: "gpt2", TokenizerPre: "qwen35",
	}
	if got := findSpecializedCandidate(target, dir, Options{BackendTag: "llama"}, "mtp"); got != "" {
		t.Fatalf("expected tokenizer-mismatched MTP head to be rejected, got %s", got)
	}
}

func TestComputeDraftAutoUsesDFlashForDeepSeekV4MoE(t *testing.T) {
	t.Setenv("LLM_SERVER_SKIP_DRAFT_DOWNLOAD", "1")
	dir := t.TempDir()
	drafter := filepath.Join(dir, "DeepSeek-V4-Flash-DSpark-draft-Q4.gguf")
	writeSpecGGUF(t, drafter, "deepseek4-dflash-draft", "gpt2", "joyai-llm", 4096, 1048576, 64, 0)
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, Name: "RTX"}},
		RAM:  detect.RAMInfo{TotalMB: 196608, FreeMB: 196608},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	target := &ModelProfile{
		Path: filepath.Join(dir, "DeepSeek-V4-Flash-Q4.gguf"), TotalSizeMB: 137000,
		Name: "DeepSeek V4 Flash", ModelArch: "deepseek4", IsMoE: true,
		EmbeddingLength: 4096, ContextSize: 1048576, VocabSize: 64,
		TokenizerModel: "gpt2", TokenizerPre: "joyai-llm",
	}
	help := "--spec-type none,draft-mtp,draft-dflash --spec-draft-model"
	opts := Options{
		SpecMode: "auto", BackendTag: "llama", BackendHelp: help,
		SpecCandidateValidator: func(string) error { return nil }, CacheDir: dir,
	}
	saveEligibleSpecProfile(t, target, caps, opts, "dflash", drafter, 2)
	draft := ComputeDraft(target, caps, opts)
	if draft.Type != DraftDFlash || draft.Path != drafter || draft.SpecType != "draft-dflash" {
		t.Fatalf("expected DFlash before the generic MoE gate, got %#v", draft)
	}
	args := DraftFlags(draft)
	if !contains(args, "draft-dflash") || !contains(args, "--model-draft") {
		t.Fatalf("DFlash flags missing: %v", args)
	}

	mtp := ComputeDraft(target, caps, Options{SpecMode: "mtp", BackendTag: "llama", BackendHelp: help})
	if mtp.Type != DraftNone {
		t.Fatalf("DeepSeek V4 DFlash must not be mislabeled as MTP: %#v", mtp)
	}
}

func TestSpecializedHFDiscovery(t *testing.T) {
	qwen := &ModelProfile{Path: "Qwen3.5-9B-Q4_K_M.gguf", Basename: "Qwen3.5-9B", ModelArch: "qwen35"}
	mtpQueries := strings.ToLower(strings.Join(hfSpecSearchQueries(qwen, "mtp"), "\n"))
	if !strings.Contains(mtpQueries, "qwen3.5 9b mtp only gguf") {
		t.Fatalf("MTP-only query missing: %s", mtpQueries)
	}
	if !hfRepoLooksRelevant("a4lg/Qwen3.5-9B-MTP-ONLY-GGUF", qwen, "mtp") {
		t.Fatal("expected same-target MTP-only repo to be relevant")
	}
	deepseek := &ModelProfile{Path: "DeepSeek-V4-Flash-Q4.gguf", Name: "DeepSeek V4 Flash", Basename: "DeepSeek-V4-Flash", ModelArch: "deepseek4"}
	known := knownSpecializedRepos(deepseek, "dflash")
	if len(known) != 0 {
		t.Fatalf("unverified DeepSeek V4 DFlash repo must not be auto-selected: %v", known)
	}
	if reason := unsupportedSpecializedRepo(deepSeekV4DFlashRepo, "dflash"); reason == "" {
		t.Fatal("verified-incompatible DeepSeek V4 drafter must be blocked")
	}
}

func TestBackendValidatorRejectsMetadataCompatibleDFlash(t *testing.T) {
	t.Setenv("LLM_SERVER_SKIP_DRAFT_DOWNLOAD", "1")
	dir := t.TempDir()
	drafter := filepath.Join(dir, "generic-dflash-draft.gguf")
	writeSpecGGUF(t, drafter, "deepseek4-dflash-draft", "gpt2", "joyai-llm", 4096, 1048576, 64, 0)
	target := &ModelProfile{
		Path: filepath.Join(dir, "DeepSeek-V4-Flash.gguf"), TotalSizeMB: 137000,
		ModelArch: "deepseek4", IsMoE: true, EmbeddingLength: 4096,
		VocabSize: 64, TokenizerModel: "gpt2", TokenizerPre: "joyai-llm",
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}}, CPU: detect.CPUInfo{Cores: 16}}
	draft := ComputeDraft(target, caps, Options{
		SpecMode: "auto", BackendTag: "llama", BackendHelp: "--spec-type draft-dflash",
		SpecCandidateValidator: func(string) error { return fmt.Errorf("invalid ggml type 101") },
	})
	if draft.Type != DraftNone {
		t.Fatalf("backend-rejected DFlash must fall back to off, got %#v", draft)
	}
}

func TestSpecializedArchitectureFilterRunsBeforeDownload(t *testing.T) {
	deepseek := &ModelProfile{ModelArch: "deepseek4"}
	if specializedArchitectureCompatible(deepseek, "mtp", "qwen35") {
		t.Fatal("a Qwen MTP model named after DeepSeek must be rejected before download")
	}
	if !specializedArchitectureCompatible(deepseek, "dflash", "deepseek4-dflash-draft") {
		t.Fatal("expected same-family DeepSeek DFlash architecture to pass")
	}
	if specializedArchitectureCompatible(deepseek, "dflash", "qwen35-dflash-draft") {
		t.Fatal("cross-family DFlash architecture must be rejected")
	}
	qwen := &ModelProfile{ModelArch: "qwen35"}
	if !specializedArchitectureCompatible(qwen, "mtp", "qwen35") {
		t.Fatal("same-family Qwen MTP-only architecture should pass")
	}
}

func TestSpecializedIdentityUsesTokenizerHashAndFailsClosed(t *testing.T) {
	target := &ModelProfile{
		ModelArch: "qwen35", EmbeddingLength: 4096, VocabSize: 64,
		TokenizerModel: "gpt2", TokenizerPre: "qwen35", TokenizerHash: strings.Repeat("a", 64),
	}
	matching := &gguf.Info{
		Architecture: "qwen35", EmbeddingLength: 4096, VocabSize: 64,
		TokenizerModel: "gpt2", TokenizerPre: "qwen35", TokenizerHash: strings.Repeat("a", 64),
	}
	if err := validateSpecializedCompatibilityIdentity(target, matching, "mtp", "llama"); err != nil {
		t.Fatalf("exact identity rejected: %v", err)
	}
	mismatched := *matching
	mismatched.TokenizerHash = strings.Repeat("b", 64)
	if err := validateSpecializedCompatibilityIdentity(target, &mismatched, "mtp", "llama"); err == nil || !strings.Contains(err.Error(), "tokenizer hash mismatch") {
		t.Fatalf("tokenizer hash mismatch was not rejected: %v", err)
	}
	missing := *matching
	missing.TokenizerHash = ""
	if err := validateSpecializedCompatibilityIdentity(target, &missing, "mtp", "llama"); err == nil || !strings.Contains(err.Error(), "missing on one side") {
		t.Fatalf("missing tokenizer hash was not rejected: %v", err)
	}
}

func TestReviewedDeepSeekV4MTPManifestStaysBlockedForLlama(t *testing.T) {
	target := &ModelProfile{ModelArch: "deepseek4"}
	if specializedArchitectureCompatibleForBackend(target, "mtp", "deepseek4_mtp_support", "llama") {
		t.Fatal("DS4-specific MTP architecture must not be authorized for llama-server")
	}
	if specializedArchitectureCompatibleForBackend(target, "mtp", "deepseek4_mtp_support", "ds4") {
		t.Fatal("a known artifact must remain blocked until its manifest is AutoApproved")
	}
	if reason := unsupportedSpecializedRepo(deepSeekV4MTPRepo, "mtp"); !strings.Contains(reason, "DS4-specific") {
		t.Fatalf("reviewed MTP artifact missing deterministic rejection reason: %q", reason)
	}
	if revision := knownSpecializedRepoRevision(deepSeekV4MTPRepo); revision != deepSeekV4MTPRevision {
		t.Fatalf("MTP manifest revision=%q, want %q", revision, deepSeekV4MTPRevision)
	}
	manifest, ok := specializedArtifactFor(deepSeekV4MTPRepo, deepSeekV4MTPFile)
	if !ok || manifest.Size != deepSeekV4MTPSize || manifest.SHA256 != deepSeekV4MTPSHA256 {
		t.Fatalf("MTP provenance manifest incomplete: ok=%v manifest=%+v", ok, manifest)
	}
}

func TestAutoSpecializedDiscoveryRequiresReviewedManifest(t *testing.T) {
	if openSpecializedDiscoveryAllowed(Options{SpecMode: "auto"}, "mtp") {
		t.Fatal("Auto must not promote mutable MTP search results")
	}
	if openSpecializedDiscoveryAllowed(Options{SpecMode: "auto"}, "dflash") {
		t.Fatal("Auto must not promote mutable DFlash search results")
	}
	if !openSpecializedDiscoveryAllowed(Options{SpecMode: "mtp"}, "mtp") {
		t.Fatal("explicit MTP testing should retain discovery behind validation gates")
	}
	if !openSpecializedDiscoveryAllowed(Options{SpecMode: "auto"}, "draft") {
		t.Fatal("generic draft discovery policy should remain unchanged")
	}
}

func TestEmbeddedMTPContextReservationUsesOnlyNextNLayers(t *testing.T) {
	model := &ModelProfile{
		NumLayers: 33, NextNPredictLayers: 1, HasSSM: 1, FullAttnInterval: 4,
		HeadCountKV: 4, KeyLength: 256, ValueLength: 256,
	}
	if got := EmbeddedMTPContextMB(model, 262144, "f16"); got != 1024 {
		t.Fatalf("embedded MTP context=%d MiB, want 1024", got)
	}
	if got := EmbeddedMTPContextMB(model, 262144, "q8_0"); got != 544 {
		t.Fatalf("quantized embedded MTP context=%d MiB, want 544", got)
	}
}

func TestAutoCompanionRequiresSelectedBackendLoader(t *testing.T) {
	opts := Options{SpecMode: "auto"}
	if err := validateSpecCandidateBackend("mtp.gguf", opts); err == nil || !strings.Contains(err.Error(), "no-allocation") {
		t.Fatalf("Auto accepted an unverified companion: %v", err)
	}
	opts.SpecMode = "mtp"
	if err := validateSpecCandidateBackend("mtp.gguf", opts); err != nil {
		t.Fatalf("explicit testing should remain possible: %v", err)
	}
}

func TestDraftDeviceUsesVulkanDialect(t *testing.T) {
	cfg := &DraftConfig{Type: DraftModel, BackendTag: "vulkan", Path: "draft.gguf", DraftGPU: 1, DraftMax: 3}
	args := DraftFlags(cfg)
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--device-draft" && args[i+1] == "Vulkan1" {
			return
		}
	}
	t.Fatalf("expected Vulkan draft device flag, got %v", args)
}

// TestArgsEmitsExplicitZeroCacheAndCheckpoints guards the 2026-07-08 crash:
// computeCRAM can correctly decide "0, disable" for both CRAM and
// MaxCheckpoints when VRAM is too tight for a big multi-GPU MoE, but the old
// "if s.CRAM > 0" gate (with MaxCheckpoints nested inside it) silently
// dropped both flags whenever the answer was 0 — leaving llama-server's own
// defaults (cache-ram 8192 MiB, ctx-checkpoints 32) active. That default's
// checkpoint state-save lives entirely outside the backend's own memory
// accounting and crashed DeepSeek-V4 mid-request despite a placement that
// had already loaded clean and passed health check.
func TestArgsEmitsExplicitZeroCacheAndCheckpoints(t *testing.T) {
	s := &Strategy{
		Type:           MoEOffload,
		ContextSize:    1048576,
		KVQuality:      "high",
		KVType:         "f16",
		FlashAttention: true,
		Threads:        8,
		ThreadsBatch:   8,
		BatchSize:      2048,
		UBatchSize:     64,
		CRAM:           0,
		MaxCheckpoints: 0,
	}
	args := s.Args("/models/test.gguf", 8081)
	if !hasAdjacentArgPlacement(args, "-cram", "0") {
		t.Fatalf("expected explicit '-cram 0' when computeCRAM decided cache is unsafe, got %v", args)
	}
	if !hasAdjacentArgPlacement(args, "--ctx-checkpoints", "0") {
		t.Fatalf("expected explicit '--ctx-checkpoints 0' when computeCRAM decided checkpoints are unsafe, got %v", args)
	}
}

func TestHybridPromptCacheKeepsOneBoundedCheckpoint(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 8192, FreeMB: 8192},
	}
	s := &Strategy{Type: SingleGPU, HasSSM: true, Parallel: 2}
	cram, checkpoints := computeCRAM(caps, &ModelProfile{HasSSM: 1}, s, 4096, 256)
	if checkpoints != 1 {
		t.Fatalf("hybrid model checkpoints=%d, want one rolling checkpoint; cram=%d", checkpoints, cram)
	}
	s.CRAM = cram
	s.MaxCheckpoints = checkpoints
	s.ContextSize = 131072
	s.KVType = "q8_0"
	s.Threads = 8
	s.ThreadsBatch = 8
	s.BatchSize = 512
	s.UBatchSize = 128
	if args := s.Args("model.gguf", 8081); !hasAdjacentArgPlacement(args, "--ctx-checkpoints", "1") {
		t.Fatalf("hybrid strategy did not emit its bounded checkpoint: %v", args)
	}
}

func TestHybridPromptCacheDisablesCheckpointWithoutHostHeadroom(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 768, FreeMB: 768},
	}
	s := &Strategy{Type: SingleGPU, HasSSM: true, Parallel: 2}
	_, checkpoints := computeCRAM(caps, &ModelProfile{HasSSM: 1}, s, 4096, 256)
	if checkpoints != 0 {
		t.Fatalf("memory-constrained hybrid model checkpoints=%d, want disabled", checkpoints)
	}
}

// TestArgsOmitsCheckpointsWhenNotComputedForStrategyType guards the other
// direction: computeCRAM never runs its real headroom math for single-GPU/
// CPU-only strategies (MaxCheckpoints stays at the -1 "not computed"
// sentinel), so Args() must not fabricate a disable decision nothing
// actually derived — the backend's own default should apply. CRAM itself IS
// always a real decision for every strategy type, so it must always emit,
// including an explicit 0.
func TestArgsOmitsCheckpointsWhenNotComputedForStrategyType(t *testing.T) {
	s := &Strategy{
		Type:           SingleGPU,
		ContextSize:    32768,
		KVQuality:      "mid",
		KVType:         "q8_0",
		FlashAttention: true,
		Threads:        8,
		ThreadsBatch:   8,
		BatchSize:      2048,
		UBatchSize:     512,
		CRAM:           0,
		MaxCheckpoints: -1,
	}
	args := s.Args("/models/test.gguf", 8081)
	if !hasAdjacentArgPlacement(args, "-cram", "0") {
		t.Fatalf("expected CRAM to always emit explicitly (it's always a real decision), got %v", args)
	}
	for _, a := range args {
		if a == "--ctx-checkpoints" {
			t.Fatalf("expected no --ctx-checkpoints when computeCRAM never evaluated it for this strategy type, got %v", args)
		}
	}
}

func hasAdjacentArgPlacement(args []string, key, val string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == key && args[i+1] == val {
			return true
		}
	}
	return false
}

func TestArgsCPUOnlyIncludesZeroGPULayers(t *testing.T) {
	s := &Strategy{
		Type:           CPUOnly,
		ContextSize:    4096,
		KVQuality:      "mid",
		KVType:         "q4_0",
		FlashAttention: true,
		Threads:        8,
		ThreadsBatch:   8,
		BatchSize:      512,
		UBatchSize:     256,
	}
	args := s.Args("/models/test.gguf", 8081)
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "-ngl" && args[i+1] == "0" {
			return
		}
	}
	t.Fatalf("expected CPU-only args to include -ngl 0, got %v", args)
}

func TestArgsSingleGPUPinsDevice(t *testing.T) {
	s := &Strategy{
		Type:           SingleGPU,
		ContextSize:    4096,
		MainGPU:        0,
		KVType:         "q4_0",
		FlashAttention: true,
		Threads:        8,
		ThreadsBatch:   8,
		BatchSize:      8192,
		UBatchSize:     1024,
	}
	args := s.Args("/models/test.gguf", 8081)
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--device" && args[i+1] == "CUDA0" {
			return
		}
	}
	t.Fatalf("expected single-GPU args to include --device CUDA0, got %v", args)
}

func TestArgsSingleGPUPinsVulkanDevice(t *testing.T) {
	s := &Strategy{
		Type:           SingleGPU,
		ContextSize:    4096,
		MainGPU:        0,
		BackendTag:     "vulkan",
		KVType:         "q4_0",
		FlashAttention: true,
		Threads:        8,
		ThreadsBatch:   8,
		BatchSize:      8192,
		UBatchSize:     1024,
	}
	args := s.Args("/models/test.gguf", 8081)
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--device" && args[i+1] == "Vulkan0" {
			return
		}
	}
	t.Fatalf("expected single-GPU args to include --device Vulkan0, got %v", args)
}

func TestRestrictGPUsFiltersAndRenumbers(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 4070", VRAMTotalMB: 12288},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288},
			{Index: 2, Name: "RTX 3090", VRAMTotalMB: 24576},
		},
		RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU: detect.CPUInfo{Cores: 16},
	}
	out, err := restrictGPUs(caps, []int{1, 2})
	if err != nil {
		t.Fatalf("restrictGPUs failed: %v", err)
	}
	if len(out.GPUs) != 2 {
		t.Fatalf("expected 2 GPUs after restriction, got %d", len(out.GPUs))
	}
	if out.GPUs[0].Name != "RTX 3060" || out.GPUs[1].Name != "RTX 3090" {
		t.Fatalf("wrong GPUs selected: %v", out.GPUs)
	}
	// Renumbered from 0 to match CUDA_VISIBLE_DEVICES enumeration.
	if out.GPUs[0].Index != 0 || out.GPUs[1].Index != 1 {
		t.Fatalf("expected renumbered indices 0,1; got %d,%d", out.GPUs[0].Index, out.GPUs[1].Index)
	}
	// Original caps untouched.
	if len(caps.GPUs) != 3 || caps.GPUs[1].Index != 1 {
		t.Fatalf("restrictGPUs mutated input caps")
	}
}

func TestRestrictGPUsNoMatchErrors(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 12288}},
	}
	if _, err := restrictGPUs(caps, []int{5}); err == nil {
		t.Fatal("expected error for non-existent GPU index")
	}
}

func TestRestrictGPUsEmptyPassthrough(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0}, {Index: 1}},
	}
	out, err := restrictGPUs(caps, nil)
	if err != nil || out != caps {
		t.Fatalf("expected passthrough for empty restriction, got %v %v", out, err)
	}
}

func TestComputeHonorsGPURestriction(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576},
			{Index: 1, VRAMTotalMB: 12288},
		},
		RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "model.gguf",
		SizeBytes:   4 * 1024 * 1024 * 1024,
		TotalSizeMB: 4 * 1024,
		NumLayers:   32,
		ContextSize: 32768,
		HiddenSize:  4096,
		HeadCountKV: 8,
		KeyLength:   128,
		ValueLength: 128,
	}
	strat, err := Compute(caps, model, Options{GPUs: []int{1}, KVPlacement: "auto", KVQuality: "low"})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	// With only GPU 1 visible there is exactly one device, so no tensor split
	// across two devices may be emitted.
	if len(strat.TensorSplit) > 1 {
		t.Fatalf("expected single-GPU placement under restriction, got split %v", strat.TensorSplit)
	}
}

func TestApplyRAMBudgetOverridesDetectedRAM(t *testing.T) {
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 8192, FreeMB: 1024},
	}
	out := applyRAMBudget(caps, 65536)
	if out == caps {
		t.Fatalf("expected budgeted capabilities copy")
	}
	if out.RAM.TotalMB != 65536 || out.RAM.FreeMB != 65536 {
		t.Fatalf("expected explicit RAM budget to be used, got %+v", out.RAM)
	}
	if caps.RAM.TotalMB != 8192 || caps.RAM.FreeMB != 1024 {
		t.Fatalf("applyRAMBudget mutated input caps: %+v", caps.RAM)
	}
}

func TestCPUOnlyAutoContextUsesRAMBudget(t *testing.T) {
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 8192, FreeMB: 1495},
		CPU: detect.CPUInfo{Cores: 2},
	}
	model := &ModelProfile{
		Path:        "moe.gguf",
		TotalSizeMB: 1024,
		NumLayers:   40,
		IsMoE:       true,
		HeadCountKV: 2,
		KeyLength:   256,
		ValueLength: 256,
		CTXTrain:    262144,
	}
	strat, err := Compute(caps, model, Options{CPUMode: true, RamBudgetMB: 512000, KVQuality: "low"})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.Type != CPUOnly {
		t.Fatalf("expected CPU-only strategy, got %s", strat.Type)
	}
	if strat.ContextSize != 262144 {
		t.Fatalf("expected RAM-budgeted CPU auto-context 262144, got %d", strat.ContextSize)
	}
}

func TestArgsMetalSkipsDeviceRouting(t *testing.T) {
	s := &Strategy{
		Type:        SingleGPU,
		ContextSize: 32768,
		KVType:      "q8_0",
		BatchSize:   4096,
		UBatchSize:  512,
		Threads:     8,
		BackendTag:  "metal",
		MainGPU:     0,
		GPULayers:   999,
	}
	args := s.Args("model.gguf", 8081)
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "-ngl 999") {
		t.Fatalf("metal must offload with -ngl 999, got: %s", joined)
	}
	for _, banned := range []string{"--device", "-mg", "--run-time-repack"} {
		for _, a := range args {
			if a == banned {
				t.Fatalf("metal args must not contain %s: %s", banned, joined)
			}
		}
	}
}

func TestComputeAppleSiliconSingleGPU(t *testing.T) {
	// A 32GB M-series Mac: one synthesized GPU with 24GB working set.
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "Apple M2 Pro", VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 32768, FreeMB: 26214},
		CPU:  detect.CPUInfo{Cores: 10},
	}
	model := &ModelProfile{
		Path:        "model.gguf",
		SizeBytes:   4 * 1024 * 1024 * 1024,
		TotalSizeMB: 4 * 1024,
		NumLayers:   32,
		ContextSize: 32768,
		HiddenSize:  4096,
		HeadCountKV: 8,
		KeyLength:   128,
		ValueLength: 128,
	}
	strat, err := Compute(caps, model, Options{KVPlacement: "auto", KVQuality: "low", BackendTag: "metal"})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.Type == CPUOnly {
		t.Fatal("Apple Silicon must not fall back to CPU-only placement")
	}
	args := strat.Args("model.gguf", 8081)
	for i, a := range args {
		if a == "-ngl" && args[i+1] == "0" {
			t.Fatal("Apple Silicon launch must not disable GPU offload")
		}
	}
}

func TestMmapDecisionIsVRAMAware(t *testing.T) {
	// mmap is a question about RAM, not total model size. The same big MoE that
	// exceeds total VRAM should load RESIDENT (no mmap) when the GPUs absorb
	// enough experts that the CPU remainder fits in RAM — and only fall back to
	// mmap when little VRAM leaves a CPU remainder that overflows RAM.
	//
	// The old decision keyed off totalSizeMB > ramAvail, so BOTH cases below —
	// identical model, identical RAM — would have been forced onto mmap. The
	// VRAM-aware decision must flip: big-VRAM => resident, small-VRAM => mmap.
	mk := func() *ModelProfile {
		return &ModelProfile{
			Path: "moe.gguf", SizeBytes: 100 * 1024 * 1024 * 1024,
			NumLayers: 64, NumParams: 100_000_000_000, IsMoE: true, NumExperts: 64,
			ContextSize: 32768, HiddenSize: 4096,
			HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
			ExpertBytes: 92 * 1024 * 1024 * 1024, NonExpertBytes: 8 * 1024 * 1024 * 1024,
			CTXTrain: 32768,
		}
	}
	// 80GB RAM; 100GB model exceeds total VRAM in both cases below.
	bigVRAM := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 24576}, {Index: 2, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 81920, FreeMB: 81920},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	smallVRAM := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 81920, FreeMB: 81920},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	opts := Options{ContextSize: 32768, KVPlacement: "cpu", KVQuality: "mid"}

	big, err := Compute(bigVRAM, mk(), opts)
	if err != nil {
		t.Fatalf("big-vram compute: %v", err)
	}
	small, err := Compute(smallVRAM, mk(), opts)
	if err != nil {
		t.Fatalf("small-vram compute: %v", err)
	}
	if big.MMap {
		t.Errorf("big-VRAM MoE: CPU remainder fits in RAM, expected resident (MMap=false), got MMap=true")
	}
	if !small.MMap {
		t.Errorf("small-VRAM MoE: CPU remainder overflows RAM, expected mmap (MMap=true), got MMap=false")
	}
}

func TestKVOnCPUFreesVRAMForExperts(t *testing.T) {
	// A big MoE with KV on CPU must place MORE expert layers on the GPU than the
	// same model with KV on GPU — because the CPU-KV frees the VRAM that would
	// otherwise be (wrongly) reserved for a cache that isn't there.
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	mk := func() *ModelProfile {
		return &ModelProfile{
			Path: "moe.gguf", SizeBytes: 40 * 1024 * 1024 * 1024,
			NumLayers: 64, NumParams: 70_000_000_000, IsMoE: true, NumExperts: 64,
			ContextSize: 32768, HiddenSize: 4096,
			HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
			ExpertBytes: 36 * 1024 * 1024 * 1024, NonExpertBytes: 4 * 1024 * 1024 * 1024,
			CTXTrain: 32768,
		}
	}
	gpuKV, err := Compute(caps, mk(), Options{ContextSize: 32768, KVPlacement: "gpu", KVQuality: "mid"})
	if err != nil {
		t.Fatalf("gpu-kv compute: %v", err)
	}
	cpuKV, err := Compute(caps, mk(), Options{ContextSize: 32768, KVPlacement: "cpu", KVQuality: "mid"})
	if err != nil {
		t.Fatalf("cpu-kv compute: %v", err)
	}
	if cpuKV.NCPUMoE >= gpuKV.NCPUMoE {
		t.Fatalf("KV-on-CPU should offload FEWER experts to CPU (more on GPU): cpu-kv NCPUMoE=%d, gpu-kv NCPUMoE=%d", cpuKV.NCPUMoE, gpuKV.NCPUMoE)
	}
}

func TestExactKVTypesAreSizedAndPreserved(t *testing.T) {
	model := &ModelProfile{
		NumLayers: 32, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	q5 := computeKVTotalMB(model, 1048576, "q5_1")
	q8 := computeKVTotalMB(model, 1048576, "q8_0")
	if q5 != 49152 {
		t.Fatalf("q5_1 KV size = %d MiB, want 49152 MiB", q5)
	}
	if q8 != 69632 {
		t.Fatalf("q8_0 KV size = %d MiB, want 69632 MiB", q8)
	}
	if q5 >= q8 {
		t.Fatalf("q5_1 must use less memory than q8_0: q5=%d q8=%d", q5, q8)
	}

	if got := kvTypesForAutoContext("q5_1", "q5_1"); len(got) != 1 || got[0] != "q5_1" {
		t.Fatalf("explicit cache type must not silently fall back: %v", got)
	}
	if got := fallbackKVType("q5_1", "q5_1"); got != "q5_1" {
		t.Fatalf("exact cache type fallback = %q, want q5_1", got)
	}
}

func TestNormalizeKVType(t *testing.T) {
	for input, want := range map[string]string{
		"high": "f16", "mid": "q8_0", "low": "q4_0", "Q5_1": "q5_1", "fp32": "f32",
	} {
		got, err := NormalizeKVType(input)
		if err != nil || got != want {
			t.Fatalf("NormalizeKVType(%q) = %q, %v; want %q, nil", input, got, err, want)
		}
	}
	if _, err := NormalizeKVType("q6_k"); err == nil {
		t.Fatal("unsupported cache type must be rejected before placement")
	}
}
