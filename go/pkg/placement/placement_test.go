package placement

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
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

func TestComputeHonorsExplicitBatchBeforeLaunch(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path: "model.gguf", SizeBytes: 15 * 1024 * 1024 * 1024,
		NumLayers: 64, NumParams: 32_000_000_000, ContextSize: 32768, HiddenSize: 4096,
	}
	strategy, err := Compute(caps, model, Options{ContextSize: 32768, BatchSize: 512, UBatchSize: 256})
	if err != nil {
		t.Fatalf("compute: %v", err)
	}
	if strategy.BatchSize != 512 || strategy.UBatchSize != 256 {
		t.Fatalf("explicit batch was not planned: batch=%d ubatch=%d", strategy.BatchSize, strategy.UBatchSize)
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

func TestFindDraftGPUFailsClosedAndReturnsPhysicalIndex(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 3, VRAMTotalMB: 12288, BandwidthMBps: 8000},
		{Index: 7, VRAMTotalMB: 24576, BandwidthMBps: 32000},
	}}
	target := &ModelProfile{IsMoE: true}
	if got := findDraftGPU(caps, target, 8000); got != 7 {
		t.Fatalf("draft GPU = %d, want physical CUDA7", got)
	}
	if got := findDraftGPU(caps, target, 25000); got != -1 {
		t.Fatalf("no-fit draft returned CUDA%d instead of failing closed", got)
	}
}

func TestApplyCompanionReservationsSeatsAndReserves(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 32000},
			{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 8000},
			{Index: 2, VRAMTotalMB: 12288, BandwidthMBps: 1000},
		},
	}
	reserved, placements, err := applyCompanionReservations(caps, []CompanionReservation{
		{Name: "claude-auto-reviewer", VRAMMB: 2600, AllowCPU: true},
	})
	if err != nil {
		t.Fatalf("applyCompanionReservations: %v", err)
	}
	if len(placements) != 1 || placements[0].GPU != 2 {
		t.Fatalf("expected the slowest-link GPU 2, got %+v", placements)
	}
	// The reservation shows as used VRAM on the chosen GPU in the returned copy.
	for _, g := range reserved.GPUs {
		if g.Index == 2 && g.VRAMUsedMB != 2600 {
			t.Fatalf("GPU2 used = %d, want 2600", g.VRAMUsedMB)
		}
	}
	// The caller's original caps are untouched.
	for _, g := range caps.GPUs {
		if g.VRAMUsedMB != 0 {
			t.Fatalf("input caps mutated: GPU%d used = %d", g.Index, g.VRAMUsedMB)
		}
	}
}

func TestApplyCompanionReservationsHonorsPreference(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 32000},
			{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 8000},
		},
	}
	_, placements, err := applyCompanionReservations(caps, []CompanionReservation{
		{Name: "reviewer", VRAMMB: 2600, GPUPreference: []int{1, 0}, AllowCPU: true},
	})
	if err != nil {
		t.Fatalf("applyCompanionReservations: %v", err)
	}
	if len(placements) != 1 || placements[0].GPU != 1 {
		t.Fatalf("explicit preference [1,0] should seat GPU 1, got %+v", placements)
	}
}

func TestApplyCompanionReservationsCPUFallbackAndError(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 2048, VRAMUsedMB: 1024}},
	}
	// Fits nowhere on GPU: AllowCPU seats it at -1.
	_, placements, err := applyCompanionReservations(caps, []CompanionReservation{
		{Name: "reviewer", VRAMMB: 2600, AllowCPU: true},
	})
	if err != nil {
		t.Fatalf("CPU-allowed reservation should not error: %v", err)
	}
	if len(placements) != 1 || placements[0].GPU != -1 {
		t.Fatalf("expected CPU seat (-1), got %+v", placements)
	}
	// Disallowing CPU turns a no-fit into an explicit error, not a silent fallback.
	if _, _, err := applyCompanionReservations(caps, []CompanionReservation{
		{Name: "reviewer", VRAMMB: 2600, AllowCPU: false},
	}); err == nil {
		t.Fatal("expected an error when no GPU fits and CPU is disallowed")
	}
}

func TestComputeReservesCompanionBeforeSplit(t *testing.T) {
	// Reserving the reviewer's footprint in the ledger must produce a resolved
	// seat on the strategy, and — on a multi-GPU host — must seat the reviewer
	// on the slowest-link GPU, keeping the fast GPU free for the model.
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 32000},
			{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 1000},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path: "model.gguf", SizeBytes: 20 * 1024 * 1024 * 1024,
		NumLayers: 48, ContextSize: 32768, HiddenSize: 4096,
	}
	companion := []CompanionReservation{{Name: "claude-auto-reviewer", VRAMMB: 2600, AllowCPU: true}}
	strat, err := Compute(caps, model, Options{ContextSize: 32768, KVPlacement: "gpu", KVQuality: "low", Companions: companion})
	if err != nil {
		t.Fatalf("compute with companion: %v", err)
	}
	if len(strat.CompanionPlacements) != 1 {
		t.Fatalf("expected one companion placement, got %+v", strat.CompanionPlacements)
	}
	if strat.CompanionPlacements[0].Name != "claude-auto-reviewer" {
		t.Fatalf("companion name lost: %+v", strat.CompanionPlacements[0])
	}
	// Slowest-link GPU 1 must be the seat — the 24GB fast GPU stays free.
	if strat.CompanionPlacements[0].GPU != 1 {
		t.Fatalf("reviewer should sit on slow GPU 1, got %d", strat.CompanionPlacements[0].GPU)
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

func TestStrategyArgsAvoidsFitValueForBooleanFork(t *testing.T) {
	strategy := &Strategy{
		Type:                 SingleGPU,
		ContextSize:          4096,
		KVPlacement:          "gpu",
		KVType:               "f16",
		BatchSize:            512,
		UBatchSize:           256,
		Threads:              8,
		ThreadsBatch:         8,
		Parallel:             1,
		TensorSplit:          []float64{1},
		BackendSupportsFit:   true,
		BackendFitTakesValue: false,
	}
	args := strategy.Args("model.gguf", 8081)
	if contains(args, "--fit") {
		t.Fatalf("boolean-only --fit backend must not receive an explicit disable value: %v", args)
	}
}

func TestStrategyArgsRespectKVOffloadDialectAndCacheDefault(t *testing.T) {
	base := Strategy{
		Type:         SingleGPU,
		ContextSize:  4096,
		KVPlacement:  "gpu",
		KVType:       "f16",
		BatchSize:    512,
		UBatchSize:   256,
		Threads:      8,
		ThreadsBatch: 8,
		Parallel:     1,
		TensorSplit:  []float64{1},
	}

	ikArgs := base.Args("model.gguf", 8081)
	if contains(ikArgs, "--kv-offload") {
		t.Fatalf("backend without positive KV flag must rely on GPU-KV default: %v", ikArgs)
	}
	if entry := cacheEntryFromArgs(ikArgs, nil); !entry.KVUnified {
		t.Fatalf("omitted positive KV flag must still cache GPU KV: %#v", entry)
	}

	mainline := base
	mainline.BackendSupportsKVOffload = true
	mainlineArgs := mainline.Args("model.gguf", 8081)
	if !contains(mainlineArgs, "--kv-offload") {
		t.Fatalf("backend advertising --kv-offload must receive it: %v", mainlineArgs)
	}

	cpu := base
	cpu.KVPlacement = "cpu"
	cpuArgs := cpu.Args("model.gguf", 8081)
	if !contains(cpuArgs, "--no-kv-offload") {
		t.Fatalf("CPU KV must retain explicit negative flag: %v", cpuArgs)
	}
	if entry := cacheEntryFromArgs(cpuArgs, nil); entry.KVUnified {
		t.Fatalf("CPU KV cache state = GPU: %#v", entry)
	}
}

func TestComputeDetectsPositiveKVOffloadFromBackendHelp(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{Path: "model.gguf", SizeBytes: 1024 * 1024 * 1024, NumLayers: 16, ContextSize: 4096, HiddenSize: 1024}

	ik, err := Compute(caps, model, Options{ContextSize: 4096, KVPlacement: "gpu", BackendHelp: "--no-kv-offload"})
	if err != nil {
		t.Fatal(err)
	}
	if ik.BackendSupportsKVOffload {
		t.Fatalf("negative-only backend help must not advertise positive KV offload")
	}
	if contains(ik.Args(model.Path, 8081), "--kv-offload") {
		t.Fatalf("negative-only backend emitted unsupported positive KV flag")
	}

	mainline, err := Compute(caps, model, Options{ContextSize: 4096, KVPlacement: "gpu", BackendHelp: "--no-kv-offload --kv-offload"})
	if err != nil {
		t.Fatal(err)
	}
	if !mainline.BackendSupportsKVOffload {
		t.Fatal("positive KV option in backend help was not detected")
	}
	if !contains(mainline.Args(model.Path, 8081), "--kv-offload") {
		t.Fatalf("backend advertising positive KV flag did not receive it")
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

// TestArgsMixedKVTypeVEmitsDistinctVCache verifies the V-leg override: a
// strategy that forced V to f16 (Inkling) still compresses K but must not emit
// a compressed --cache-type-v, because the backend rejects it outright.
func TestArgsMixedKVTypeVEmitsDistinctVCache(t *testing.T) {
	s := &Strategy{
		ContextSize:    4096,
		GPULayers:      32,
		KVQuality:      "mid",
		KVType:         "q8_0",
		KVTypeV:        "f16",
		FlashAttention: true,
		Threads:        16,
		BatchSize:      2048,
		UBatchSize:     512,
	}
	args := s.Args("/models/inkling.gguf", 8081)
	argV := ""
	for i, a := range args {
		if a == "--cache-type-v" && i+1 < len(args) {
			argV = args[i+1]
		}
		if a == "--cache-type-k" && i+1 < len(args) {
			if args[i+1] != "q8_0" {
				t.Fatalf("K cache must stay compressed q8_0, got %q in %v", args[i+1], args)
			}
		}
	}
	if argV != "f16" {
		t.Fatalf("V cache must be promoted to f16 while K is compressed, got %q in %v", argV, args)
	}
	// A strategy without the override keeps the unified pair.
	plain := (&Strategy{ContextSize: 4096, KVType: "q4_0", FlashAttention: true, Threads: 8, BatchSize: 1024, UBatchSize: 512}).Args("/models/x.gguf", 8081)
	for i, a := range plain {
		if a == "--cache-type-v" && i+1 < len(plain) && plain[i+1] != "q4_0" {
			t.Fatalf("unified strategy must emit V == K, got V %q in %v", plain[i+1], plain)
		}
	}
}

// TestResolveKVQualityInklingKeepsCompressedKWithFV pins the Inkling rule: a
// compressed K request is legal (that is where the memory savings are), and the
// unified quality still resolves to a compressed K type. The V-leg f16 pinning
// happens in Compute, not here.
func TestResolveKVQualityInklingKeepsCompressedKWithFV(t *testing.T) {
	model := &ModelProfile{ModelArch: "inkling"}
	for _, quality := range []string{"auto", "mid", "low", "q8_0", "q4_0"} {
		got, err := resolveKVQuality(model, quality, "llama")
		if err != nil {
			t.Errorf("Inkling compressed K quality %q must not be rejected (K may compress): %v", quality, err)
			continue
		}
		if got == "" {
			t.Errorf("Inkling quality %q resolved to empty", quality)
		}
	}
	// Explicit invalid type still fails with the normal validation error.
	if _, err := resolveKVQuality(model, "nonsense", "llama"); err == nil {
		t.Error("invalid KV type accepted for Inkling")
	}
}

// TestComputeInklingPinsVCacheToF16 verifies Compute emits a f16 V override for
// the inkling architecture regardless of the requested K quality, so the launch
// never sends a compressed --cache-type-v to the rejecting backend.
func TestComputeInklingPinsVCacheToF16(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, VRAMUsedMB: 0, BandwidthMBps: 15754}},
		RAM:  detect.RAMInfo{TotalMB: 128512, FreeMB: 120000},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "inkling-small.gguf", Basename: "inkling-small.gguf",
		ModelArch: "inkling", SizeBytes: 8 * 1024 * 1024 * 1024, TotalSizeMB: 8192,
		NumLayers: 32, IsMoE: false, ContextSize: 32768,
		EmbeddingLength: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	for _, quality := range []string{"auto", "low", "q4_0", "q8_0"} {
		opts := Options{ContextSize: 32768, KVQuality: quality, KVPlacement: "gpu", CacheDir: t.TempDir()}
		strategy, err := Compute(caps, model, opts)
		if err != nil {
			t.Fatalf("Compute inkling quality %q: %v", quality, err)
		}
		if strategy.KVTypeV != "f16" {
			t.Errorf("Inkling quality %q must pin V cache to f16, got KVTypeV=%q KVType=%q", quality, strategy.KVTypeV, strategy.KVType)
		}
		args := strategy.Args(model.Path, 8081)
		argV := ""
		for i, a := range args {
			if a == "--cache-type-v" && i+1 < len(args) {
				argV = args[i+1]
			}
		}
		if argV != "f16" {
			t.Errorf("Inkling quality %q emitted --cache-type-v %q, want f16 (args %v)", quality, argV, args)
		}
	}
	// A non-inkling model is unaffected.
	other := &ModelProfile{
		Path: "qwen.gguf", Basename: "qwen.gguf",
		ModelArch: "qwen3", SizeBytes: 8 * 1024 * 1024 * 1024, TotalSizeMB: 8192,
		NumLayers: 32, IsMoE: false, ContextSize: 32768,
		EmbeddingLength: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	otherStrategy, err := Compute(caps, other, Options{ContextSize: 32768, KVQuality: "low", KVPlacement: "gpu", CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("Compute qwen3: %v", err)
	}
	if otherStrategy.KVTypeV != "" {
		t.Errorf("non-Inkling model must not set KVTypeV, got %q", otherStrategy.KVTypeV)
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
			"q8_0": 6912.25, // measured: launch log, ctx=1048576 total_kv=6912MB
		},
	}

	// The old fixture exercised an invalid exact q8_0/CPU-KV mainline path. Keep
	// the regression meaningful by asserting that a correctness-first planner
	// rejects it before it can reach the unsafe placement ledger below. The legacy
	// "mid" preset is intentionally upgraded to F16 on V4; exact q8_0 remains an
	// explicit compressed-KV request and must fail.
	if _, err := Compute(caps, model, Options{
		ContextSize: 262144,
		KVPlacement: "cpu",
		KVQuality:   "q8_0",
		BackendTag:  "llama",
		Parallel:    1,
		CacheDir:    t.TempDir(),
	}); err == nil || !strings.Contains(err.Error(), "requires f16 or bf16 KV") {
		t.Fatalf("compressed mainline V4 KV must fail before planning, got %v", err)
	}

	strat, err := Compute(caps, model, Options{
		ContextSize: 262144,
		KVPlacement: "gpu",
		KVQuality:   "high",
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
		fixed := firstLaunchComputeBufMBForGPUParallelAtContext(
			model, strat.UBatchSize, strat.Parallel, strat.ContextSize, gi, orderGPUsByBandwidth(caps.GPUs),
		)
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
	if strat.KVPlacement != "gpu" {
		t.Fatalf("expected correctness-first f16 KV on GPU, got %q", strat.KVPlacement)
	}
}

// Regression for the real no-flag DeepSeek-V4 launch on the 3090Ti+3060+4070
// host. The exact 1M-context KV measurement and CUDA overhead originally lived
// in ~/.cache/ggrun, while the app later moved to an app-local cache. Falling
// back to formula charged 16.4 GiB instead of the measured 6.9 GiB; legacy
// startup-OOM probes then charged compute a second time as runtime growth. The
// solver rejected a model that fits, before preflight could correct anything.
func TestComputeDeepSeekV4DoesNotUseLegacyGlobalKVToForceFullContext(t *testing.T) {
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
	systemPath := filepath.Join(legacyDir, fmt.Sprintf("system_%s.cache", gpuIdentityHash(caps.GPUs)))
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
	if strat.ContextSize >= 1048576 {
		t.Fatalf("legacy model-wide KV rate forced unsafe full context: %d", strat.ContextSize)
	}
	if strat.KVPlacement != "gpu" || strat.KVType != "f16" {
		t.Fatalf("mainline DeepSeek4 must keep f16 KV on GPU, got placement=%s type=%s", strat.KVPlacement, strat.KVType)
	}
	if got := model.MeasuredKVBytesPerTok["f16"]; got != 6912.25 {
		t.Fatalf("exact KV measurement was not migrated, got %.2f", got)
	}
	if strat.ContextAllocationEvidence != "" {
		t.Fatalf("legacy rate was mislabeled as exact allocation evidence: %q", strat.ContextAllocationEvidence)
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
	if err := os.WriteFile(filepath.Join(cacheDir, fmt.Sprintf("system_%s.cache", gpuIdentityHash(caps.GPUs))), []byte(systemData), 0644); err != nil {
		t.Fatal(err)
	}
	for _, ubatch := range []int{512, 256, 128, 64} {
		if err := RecordMeasuredAllocation(cacheDir, model, 1048576, ubatch, "high", "gpu", "llama", caps.GPUs, 4,
			MeasuredAllocation{Evidence: "fit-params", ContextTotalMB: 7012}); err != nil {
			t.Fatalf("seed exact allocation ubatch=%d: %v", ubatch, err)
		}
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

// MoE capacity is geometry-driven for every architecture. DeepSeek4 must use
// the same split-owner packing path as an ordinary MoE instead of losing an
// entire GPU to an architecture-name policy.
func TestComputeDeepSeekV4SplitOwnerUsesGenericMeasuredCapacity(t *testing.T) {
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
		ContextSize: 65536, CTXTrain: 1048576, HiddenSize: 4096, EmbeddingLength: 4096,
		HeadCountKV: 1, KeyLength: 512, ValueLength: 512, ModelArch: "deepseek4",
		MeasuredKVBytesPerTok: map[string]float64{"f16": 6912.25},
	}
	cacheDir := t.TempDir()
	opts := Options{
		ContextSize: 65536, KVPlacement: "gpu", KVQuality: "high",
		BackendTag: "llama", Parallel: 1, CacheDir: cacheDir,
	}

	cold, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("cold placement: %v", err)
	}
	coldLayers := parseOTLayersByDevice(t, cold.OTString)
	if got := len(coldLayers[0]); got == 0 {
		t.Fatalf("cold V4 plan left the CUDA0 split owner empty: %s", cold.OTString)
	}
	if len(coldLayers[1])+len(coldLayers[2]) == 0 {
		t.Fatalf("cold V4 plan failed to use secondary expert storage: %s", cold.OTString)
	}

	// A measurement for another parallel shape is deliberately not evidence for
	// this foreground/P1 launch.
	if err := writeProbeCacheForModel(cacheDir, model, 65536, 512, "high", "gpu", "llama", caps.GPUs, 4,
		map[int]int{0: 2048, 1: 128, 2: 128}, nil, nil, 0); err != nil {
		t.Fatalf("write mismatched probe: %v", err)
	}
	wrongShape, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("placement with mismatched probe: %v", err)
	}
	if got, want := len(parseOTLayersByDevice(t, wrongShape.OTString)[0]), len(coldLayers[0]); got != want {
		t.Fatalf("parallel-4 probe changed parallel-1 CUDA0 capacity: got %d layers, want %d (%s)", got, want, wrongShape.OTString)
	}

	// An exact measurement refines the generic capacity calculation directly;
	// no architecture-specific validation token is required.
	if err := writeProbeCacheForModel(cacheDir, model, 65536, 512, "high", "gpu", "llama", caps.GPUs, 1,
		map[int]int{0: 2048, 1: 128, 2: 128}, nil, nil, 0); err != nil {
		t.Fatalf("write exact probe: %v", err)
	}
	measured, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("placement with exact probe: %v", err)
	}
	if got := len(parseOTLayersByDevice(t, measured.OTString)[0]); got == 0 {
		t.Fatalf("exact measurement left CUDA0 empty: %s", measured.OTString)
	}
}

// TestComputeSplitOwnerChargesPerGPUComputeNotAggregate guards Fix A.1: the
// split-owner compute reserve must come from the per-GPU measurement for THIS
// device, not the primary GPU's aggregate. The crash reproduced the failure:
// DeepSeek-V4 at ctx 1M / ub=512 carried PROBED_COMPUTE_BUF_MB=17970
// (oracle-planned, the primary's value) while the secondary split-owner's
// per-GPU value was an order of magnitude smaller. Charging every split-owner
// the primary's aggregate collapsed the secondary card to 0 expert layers;
// charging each its own per-GPU value packs it. A model with free GPU VRAM must
// therefore get MORE experts on GPU, not be under-filled.
func TestComputeSplitOwnerChargesPerGPUComputeNotAggregate(t *testing.T) {
	// Two fast (non-expert-only) GPUs so both hold real tensor-split shares. The
	// secondary split owner (GPU1) is the card the old aggregate charge starved:
	// it is charged the primary's 17970 instead of its own 599.
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "GPU A", VRAMTotalMB: 24564, BandwidthMBps: 16000, VRAMUsedMB: 500},
			{Index: 1, Name: "GPU B", VRAMTotalMB: 24564, BandwidthMBps: 16000, VRAMUsedMB: 300},
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
		ContextSize: 65536, CTXTrain: 1048576, HiddenSize: 4096, EmbeddingLength: 4096,
		HeadCountKV: 1, KeyLength: 512, ValueLength: 512, ModelArch: "deepseek4",
		MeasuredKVBytesPerTok: map[string]float64{"f16": 6912.25},
	}
	cacheDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(cacheDir, fmt.Sprintf("system_%s.cache", gpuIdentityHash(caps.GPUs))),
		[]byte("SYS_CUDA_OVERHEAD_MB_CUDA0=488\nSYS_CUDA_OVERHEAD_MB_CUDA1=311\nSYS_CUDA_OVERHEAD_MB=488\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	opts := Options{
		ContextSize: 65536, KVPlacement: "gpu", KVQuality: "high",
		BackendTag: "llama", Parallel: 1, CacheDir: cacheDir,
	}
	// Seed the probe cache with a LARGE aggregate compute buffer (17970, the
	// primary's real ctx-1M value) and a small per-GPU value for the secondary
	// split owner. Under the old aggregate charge GPU1 pays 17970 for compute;
	// with the fix it pays 599.
	if err := writeProbeCacheForModel(cacheDir, model, 65536, 512, "high", "gpu", "llama", caps.GPUs, 1,
		map[int]int{0: 17970, 1: 599}, nil, nil, 0); err != nil {
		t.Fatal(err)
	}

	// Call buildMoEOffload directly (bypassing the ubatch ladder) so the test
	// isolates the split-owner compute charge at ub=512 rather than letting the
	// ladder descend to a smaller ubatch whose compute buffer would mask the bug.
	kvTotalMB := computeKVTotalMB(model, 65536, "f16", false)
	base := &Strategy{
		Type: MoEOffload, ContextSize: 65536, KVPlacement: "gpu", KVQuality: "high",
		UBatchSize: 512, BatchSize: 512, Parallel: 1, BackendTag: "llama",
	}
	strat, err := buildMoEOffload(base, caps, model, model.TotalSizeMB, kvTotalMB, opts)
	if err != nil {
		t.Fatalf("buildMoEOffload should fit: %v", err)
	}
	layers := parseOTLayersByDevice(t, strat.OTString)
	// GPU1 must hold a real tensor-split share (both links are fast, so neither
	// is demoted to expert-only) AND pack expert layers. Under the old
	// aggregate-to-all charge it paid 17970 MiB of compute and got 0 layers; with
	// its per-GPU 599 it packs. This is the regression's core assertion: a
	// split-owner with free VRAM is filled, not under-filled.
	if len(layers[1]) == 0 {
		t.Fatalf("GPU1 under-filled: got 0 expert layers (OT=%s); per-GPU compute charge must leave room", strat.OTString)
	}
	if strat.TensorSplit[1] <= 0 {
		t.Fatalf("GPU1 should hold a tensor-split share, got split=%v (test setup)", strat.TensorSplit)
	}
	// The primary split-owner must also hold whole layers.
	if len(layers[0]) == 0 {
		t.Fatalf("GPU0 under-filled: got 0 expert layers (OT=%s)", strat.OTString)
	}
	// With the compute charged per-GPU the plan packs more GPU layers than a plan
	// that charged every split-owner the full 17970 aggregate would. Verify the
	// plan is not degenerate.
	totalGPU := len(layers[0]) + len(layers[1])
	if totalGPU <= 2 {
		t.Fatalf("GPU under-fill: only %d expert layers on all GPUs (OT=%s)", totalGPU, strat.OTString)
	}
}

// TestWriteProbeEvidencePriorityLiveBeatsOracle guards Fix A.2: a live-allocated
// compute-buffer measurement must supersede a later oracle-planned prediction for
// the same key, in EITHER direction — and a fresh oracle must not clobber a prior
// live observation.
func TestWriteProbeEvidencePriorityLiveBeatsOracle(t *testing.T) {
	cacheDir := t.TempDir()
	model := &ModelProfile{Path: "evidence-model.gguf", NumLayers: 43, NumExperts: 256}
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 24564}, {Index: 1, VRAMTotalMB: 12288}, {Index: 2, VRAMTotalMB: 12282}}

	// 1. Oracle runs first: writes 9022 (prediction).
	if err := writeProbeCacheForModel(cacheDir, model, 65536, 512, "high", "gpu", "llama", gpus, 1,
		map[int]int{0: 9022, 1: 599, 2: 599}, nil, nil, 0, probeMeasurements{ComputeBufEvidence: "oracle-planned"}); err != nil {
		t.Fatal(err)
	}
	// 2. A live launch measures the real buffers: smaller values, observed.
	if err := writeProbeCacheForModel(cacheDir, model, 65536, 512, "high", "gpu", "llama", gpus, 1,
		map[int]int{0: 3649, 1: 121, 2: 599}, nil, nil, 0, probeMeasurements{ComputeBufEvidence: "live-allocated"}); err != nil {
		t.Fatal(err)
	}
	pc := loadProbeCache(cacheDir, model, 65536, 512, "high", "gpu", "llama", gpus, 1)
	if pc == nil {
		t.Fatal("expected a probe cache entry")
	}
	if got := pc.ComputeBufByGPU[0]; got != 3649 {
		t.Fatalf("live-allocated GPU0 must supersede oracle 9022, got %d", got)
	}
	if got := pc.ComputeBufByGPU[1]; got != 121 {
		t.Fatalf("live-allocated GPU1 must supersede oracle 599, got %d", got)
	}
	if !observedAllocationEvidence(pc.ComputeBufEvidence) {
		t.Fatalf("file evidence must remain observed after a live measurement, got %q", pc.ComputeBufEvidence)
	}

	// 3. A later fit-oracle run must NOT clobber the observed measurement.
	if err := writeProbeCacheForModel(cacheDir, model, 65536, 512, "high", "gpu", "llama", gpus, 1,
		map[int]int{0: 9022, 1: 599, 2: 599}, nil, nil, 0, probeMeasurements{ComputeBufEvidence: "oracle-planned"}); err != nil {
		t.Fatal(err)
	}
	pc = loadProbeCache(cacheDir, model, 65536, 512, "high", "gpu", "llama", gpus, 1)
	if pc == nil {
		t.Fatal("expected a probe cache entry")
	}
	if got := pc.ComputeBufByGPU[0]; got != 3649 {
		t.Fatalf("later oracle-planned must not clobber live-allocated GPU0: got %d, want 3649", got)
	}
	if !observedAllocationEvidence(pc.ComputeBufEvidence) {
		t.Fatalf("file evidence must stay observed after an oracle attempt, got %q", pc.ComputeBufEvidence)
	}
}

// TestLadderDescentsWhenBasePlanCPUOverFills guards Fix A.5: the ubatch ladder
// must descend when the base plan leaves a pathological fraction of MoE layers
// on CPU even though it "fits", instead of returning the under-filled plan.
func TestLadderDescentsWhenBasePlanCPUOverFills(t *testing.T) {
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
		ContextSize: 65536, CTXTrain: 1048576, HiddenSize: 4096, EmbeddingLength: 4096,
		HeadCountKV: 1, KeyLength: 512, ValueLength: 512, ModelArch: "deepseek4",
		MeasuredKVBytesPerTok: map[string]float64{"f16": 6912.25},
	}
	cacheDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(cacheDir, fmt.Sprintf("system_%s.cache", gpuIdentityHash(caps.GPUs))),
		[]byte("SYS_CUDA_OVERHEAD_MB_CUDA0=488\nSYS_CUDA_OVERHEAD_MB_CUDA1=311\nSYS_CUDA_OVERHEAD_MB_CUDA2=367\nSYS_CUDA_OVERHEAD_MB=488\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// A probe whose aggregate compute buffer is huge at ub=512 (forcing a CPU
	// over-fill) but smaller at ub=256/128. The ladder must descend to a rung
	// that puts >=75% of MoE layers on GPU.
	for _, ub := range []int{512, 256, 128, 64} {
		compute := 9022
		if ub <= 256 {
			compute = 2500
		}
		if err := writeProbeCacheForModel(cacheDir, model, 65536, ub, "high", "gpu", "llama", caps.GPUs, 1,
			map[int]int{0: compute, 1: 121, 2: 121}, nil, nil, 0); err != nil {
			t.Fatal(err)
		}
	}
	opts := Options{
		ContextSize: 65536, KVPlacement: "gpu", KVQuality: "high",
		BackendTag: "llama", Parallel: 1, CacheDir: cacheDir,
	}
	strat, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("placement should fit: %v", err)
	}
	// The base ub=512 plan's compute buffer (9022) starves GPU capacity and
	// produces a pathological CPU over-fill. The ladder must descend to a smaller
	// ubatch whose compute buffer is smaller. This is the regression's core
	// assertion: the plan must NOT keep the 41-layer-ish ub=512 CPU over-fill.
	if strat.UBatchSize >= 512 {
		t.Fatalf("ladder kept the over-filled ubatch=512 plan (n-cpu-moe=%d)", strat.NCPUMoE)
	}
	// The descend rung's smaller compute buffer must buy at least ONE more GPU
	// layer than the base ub=512 plan's ~2-3. Compare against the well-known
	// healthy V4 footprint: n-cpu-moe must be strictly below the base's count,
	// and comfortably below a plan that never descended.
	if strat.NCPUMoE >= 41 {
		t.Fatalf("ladder did not reduce the CPU over-fill: n-cpu-moe=%d (OT=%s)", strat.NCPUMoE, strat.OTString)
	}
}

func TestComputeHybridMoEPacksEverySplitOwnerByCapacity(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "GPU A", VRAMTotalMB: 24564, BandwidthMBps: 15754},
			{Index: 1, Name: "GPU B", VRAMTotalMB: 24564, BandwidthMBps: 15754},
		},
		RAM: detect.RAMInfo{TotalMB: 262144, FreeMB: 250000},
		CPU: detect.CPUInfo{Cores: 16},
	}
	const gib = int64(1024 * 1024 * 1024)
	model := &ModelProfile{
		Path: "hybrid-two-split-owners.gguf", SizeBytes: 104 * gib, TotalSizeMB: 104 * 1024,
		NumLayers: 32, IsMoE: true, NumExperts: 64, ExpertUsedCount: 4, ExpertFF: 2048,
		ExpertBytes: 96 * gib, NonExpertBytes: 8 * gib,
		TokenEmbdBytes: 512 * 1024 * 1024, OutputBytes: 512 * 1024 * 1024,
		ContextSize: 65536, CTXTrain: 65536, HiddenSize: 4096, EmbeddingLength: 4096,
		HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
		ModelArch: "deepseek4", MeasuredKVBytesPerTok: map[string]float64{"f16": 4096},
	}
	cacheDir := t.TempDir()
	opts := Options{
		ContextSize: 65536, KVPlacement: "gpu", KVQuality: "high",
		BackendTag: "llama", Parallel: 1, CacheDir: cacheDir,
	}

	strategy, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("cold placement: %v", err)
	}
	if len(strategy.TensorSplit) != 2 || strategy.TensorSplit[0] <= 0 || strategy.TensorSplit[1] <= 0 {
		t.Fatalf("expected two regular split owners, got split=%v", strategy.TensorSplit)
	}
	layers := parseOTLayersByDevice(t, strategy.OTString)
	if len(layers[0]) == 0 || len(layers[1]) == 0 {
		t.Fatalf("cold hybrid plan failed to use both split owners: %s", strategy.OTString)
	}

	if err := writeProbeCacheForModel(cacheDir, model, 65536, 512, "high", "gpu", "llama", caps.GPUs, 1,
		map[int]int{0: 1024, 1: 1024}, nil, nil, 0); err != nil {
		t.Fatalf("write startup probe: %v", err)
	}
	strategy, err = Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("placement with startup probe: %v", err)
	}
	layers = parseOTLayersByDevice(t, strategy.OTString)
	if len(layers[0]) == 0 || len(layers[1]) == 0 {
		t.Fatalf("measured hybrid plan failed to use both split owners: %s", strategy.OTString)
	}
}

func TestResolveKVQualityDeepSeekV4MainlineRequiresF16(t *testing.T) {
	model := &ModelProfile{ModelArch: "deepseek4"}

	got, err := resolveKVQuality(model, "", "llama")
	if err != nil || got != "high" {
		t.Fatalf("default V4 KV quality = %q, %v; want high/f16", got, err)
	}

	got, err = resolveKVQuality(model, "auto", "llama")
	if err != nil || got != "high" {
		t.Fatalf("auto V4 KV quality = %q, %v; want high/f16", got, err)
	}

	got, err = resolveKVQuality(model, "high", "llama")
	if err != nil || got != "high" {
		t.Fatalf("explicit F16 V4 KV quality = %q, %v; want high/f16", got, err)
	}

	got, err = resolveKVQuality(model, "mid", "llama")
	if err != nil || got != "high" {
		t.Fatalf("legacy/default mid V4 KV quality = %q, %v; want safe high/f16", got, err)
	}

	if _, err := resolveKVQuality(model, "q8_0", "llama"); err == nil || !strings.Contains(err.Error(), "requires f16 or bf16 KV") {
		t.Fatalf("exact compressed V4 KV must fail with correctness error, got %v", err)
	}

	got, err = resolveKVQuality(&ModelProfile{ModelArch: "qwen3"}, "", "llama")
	if err != nil || got != "mid" {
		t.Fatalf("generic default KV quality = %q, %v; want mid", got, err)
	}
	got, err = resolveKVQuality(&ModelProfile{ModelArch: "qwen3"}, "auto", "llama")
	if err != nil || got != "mid" {
		t.Fatalf("generic auto KV quality = %q, %v; want mid", got, err)
	}
}

// TestResolveKVQualityDeepSeekV4MainlineAllowsBF16 guards FIX 1: mainline
// llama.cpp must accept a bf16 K-cache for deepseek4, mirroring the reviewed
// arch rule (pkg/backends/archconstraints.go permits f16 and bf16). bf16
// carries no requantization loss on the FP8 attention weights, so it is as
// correct as f16. Before this test, the mainline branch rejected bf16 with a
// "requires f16 KV" error, which made ik_llama the only viable auto backend for
// a bf16 launch and routed the large-CPU-expert model to a loader whose
// anonymous CUDA-host experts OOM-killed it.
func TestResolveKVQualityDeepSeekV4MainlineAllowsBF16(t *testing.T) {
	model := &ModelProfile{ModelArch: "deepseek4"}

	got, err := resolveKVQuality(model, "bf16", "llama")
	if err != nil {
		t.Fatalf("explicit bf16 V4 KV rejected by mainline: %v", err)
	}
	if got != "bf16" {
		t.Fatalf("explicit bf16 V4 KV = %q, want bf16 preserved", got)
	}

	// bf16 must stay accepted across the whole mainline family (vulkan, metal).
	for _, backendTag := range []string{"vulkan", "metal"} {
		got, err := resolveKVQuality(model, "bf16", backendTag)
		if err != nil || got != "bf16" {
			t.Errorf("bf16 V4 KV on %s = %q, %v; want bf16 preserved", backendTag, got, err)
		}
	}

	// f16 still maps to the f16 default "high", and compressed KV is still
	// rejected on mainline for correctness.
	got, err = resolveKVQuality(model, "f16", "llama")
	if err != nil || got != "high" {
		t.Fatalf("f16 V4 KV = %q, %v; want high/f16", got, err)
	}
	for _, compressed := range []string{"q8_0", "q4_0", "low"} {
		if _, err := resolveKVQuality(model, compressed, "llama"); err == nil {
			t.Errorf("compressed V4 KV %q accepted by mainline; must fail for correctness", compressed)
		}
	}
	// Legacy presets auto/mid are promoted to f16, not rejected.
	got, err = resolveKVQuality(model, "mid", "llama")
	if err != nil || got != "high" {
		t.Fatalf("legacy mid V4 KV = %q, %v; want safe high/f16", got, err)
	}
}

func TestResolveKVQualityDeepSeekV4IKRejectsUnsupportedKCache(t *testing.T) {
	model := &ModelProfile{ModelArch: "deepseek4"}
	for _, supported := range []string{"auto", "mid", "q8_0", "high", "f16", "bf16"} {
		if _, err := resolveKVQuality(model, supported, "ik_llama"); err != nil {
			t.Errorf("supported IK V4 KV type %q rejected: %v", supported, err)
		}
	}
	for _, unsupported := range []string{"low", "q4_0", "q5_1"} {
		if _, err := resolveKVQuality(model, unsupported, "ik_llama"); err == nil || !strings.Contains(err.Error(), "supports only") {
			t.Errorf("unsupported IK V4 KV type %q accepted: %v", unsupported, err)
		}
	}
}

func TestResolveKVQualityDeepSeekV4VulkanUsesMainlineSafetyRule(t *testing.T) {
	model := &ModelProfile{ModelArch: "deepseek4"}
	if _, err := resolveKVQuality(model, "q8_0", "vulkan"); err == nil || !strings.Contains(err.Error(), "requires f16 or bf16 KV") {
		t.Fatalf("Vulkan mainline accepted compressed DeepSeek4 KV: %v", err)
	}
	if got, err := resolveKVQuality(model, "f16", "vulkan"); err != nil || got != "high" {
		t.Fatalf("Vulkan f16 DeepSeek4 KV = %q, %v", got, err)
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
		if err := writeProbeCacheForModel(cacheDir, model, 1048576, ub, "high", "gpu", "llama", gpus, 4, byGPU, nil, nil, 0); err != nil {
			t.Fatalf("seed probe cache ubatch=%d: %v", ub, err)
		}
		if err := RecordMeasuredAllocation(cacheDir, model, 1048576, ub, "high", "gpu", "llama", gpus, 4,
			MeasuredAllocation{Evidence: "fit-params", ContextTotalMB: 13012}); err != nil {
			t.Fatalf("seed exact context ubatch=%d: %v", ub, err)
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

// TestComputeDeepSeekV4KeepsBoundedRecurrentCheckpoints is the end-to-end
// version of the bounded hybrid policy: runs the real
// Compute() -> computeCRAM -> Args() pipeline against the exact hardware and
// model shape used for 128GB DeepSeek-V4.
//
// This test previously asserted `-cram 0`, on the reasoning that VRAM was too
// tight for the prompt cache. That was wrong: the prompt cache is held in host
// RAM, and this fixture keeps roughly 37 GiB free after the weights load. The
// old expectation disabled caching precisely on the models whose re-prefill
// costs the most. The bounded-checkpoint property the test is named for is
// unchanged -- a bounded 4..16 set, never the backend default of 32.
func TestComputeDeepSeekV4KeepsBoundedRecurrentCheckpoints(t *testing.T) {
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
	// The prompt cache is host RAM, so a VRAM-saturated model still gets a
	// branch-capable bounded set.
	if strat.CRAM < minCramMB {
		t.Fatalf("expected a host-RAM prompt cache, got CRAM=%d", strat.CRAM)
	}
	// The bound is the safety one -- the cache must not crowd out the weights'
	// working set inside the memory scope -- not the old tenth-of-free-RAM
	// policy, which was too small to hold even one saved slot and so evicted a
	// conversation's prefix on every new admission.
	if budget := 37623 * 2 / 3; strat.CRAM > budget {
		t.Fatalf("prompt cache %d MiB exceeds the host RAM budget %d after load", strat.CRAM, budget)
	}
	if strat.MaxCheckpoints < hybridCheckpointMinimum || strat.MaxCheckpoints > hybridCheckpointMaximum {
		t.Fatalf("expected bounded recurrent checkpoints, got MaxCheckpoints=%d", strat.MaxCheckpoints)
	}
	if strat.TensorSplit[1] != 0 {
		t.Fatalf("expected x1 GPU to be expert-only with zero tensor split, got %v", strat.TensorSplit)
	}
	if !otStringUsesDevice(strat.OTString, 1) {
		t.Fatalf("expected x1 GPU to still receive full expert pins, got OT %s", strat.OTString)
	}
	args := strat.Args("/models/test.gguf", 8081)
	if hasAdjacentArgPlacement(args, "-cram", "0") {
		t.Fatalf("prompt cache disabled despite ample host RAM, got %v", args)
	}
	if !hasAdjacentArgPlacement(args, "--ctx-checkpoints", strconv.Itoa(strat.MaxCheckpoints)) {
		t.Fatalf("expected explicit bounded checkpoint count in emitted args, got %v", args)
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
		if err := writeProbeCacheForModel(cacheDir, model, 262144, ub, "high", "gpu", "llama", gpus, 4, byGPU, nil, nil, 0); err != nil {
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
	if excluded := numGPUsExcluded(strat, caps.GPUs, model.NumLayers); excluded > 0 {
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
	if excluded := numGPUsExcluded(strat, caps.GPUs, model.NumLayers); excluded != 0 {
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

func TestRequireMeasuredBuffersRemovesColdGuessButUsesRecordedCompute(t *testing.T) {
	cacheDir := t.TempDir()
	gpus := []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 8192}}
	caps := &detect.Capabilities{GPUs: gpus}
	model := &ModelProfile{Path: "model.gguf", SizeBytes: 7000 * 1024 * 1024, TotalSizeMB: 7000, NumLayers: 32}
	strategy := &Strategy{ContextSize: 32768, UBatchSize: 512, KVQuality: "mid", KVPlacement: "gpu"}
	opts := Options{CacheDir: cacheDir, BackendTag: "llama", RequireMeasuredBuffers: true}

	if got := chooseStrategy(caps, model, strategy, 7000, 500, opts); got != SingleGPU {
		t.Fatalf("cold measured-only strategy = %s, want %s", got, SingleGPU)
	}
	if err := RecordMeasuredComputeBuffers(cacheDir, model, 32768, 512, "mid", "gpu", "llama", gpus, 0, map[int]int{0: 1200}); err != nil {
		t.Fatal(err)
	}
	if got := chooseStrategy(caps, model, strategy, 7000, 500, opts); got != DenseCPUOffload {
		t.Fatalf("recorded 1200 MiB compute was ignored: got %s", got)
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
	if excluded := numGPUsExcluded(strat, caps.GPUs, model.NumLayers); excluded != 0 {
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

func TestDeepSeekV4KeepsGPUKVAndFlashAttention(t *testing.T) {
	m := &ModelProfile{ModelArch: "deepseek4", IsMoE: true}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}}}
	if got := resolveAutoKVPlacement(caps, m, 120000, 4096, 1024); got != "gpu" {
		t.Fatalf("DeepSeek-V4 auto KV placement = %q, want gpu so flash attention remains available", got)
	}
	if defaultFlashAttention(m, Options{BackendTag: "llama"}, "cpu") {
		t.Fatal("KV on CPU auto-disables flash attention; claiming it is on would emit a self-contradicting --flash-attn on --no-kv-offload command")
	}
	if !defaultFlashAttention(m, Options{BackendTag: "llama"}, "gpu") {
		t.Fatal("mainline deepseek4 with KV on GPU must default flash attention on (bounds the compute buffer; see Task #10)")
	}
	if !defaultFlashAttention(m, Options{BackendTag: "ik_llama"}, "gpu") {
		t.Fatal("ik_llama deepseek4 with KV on GPU must emit --flash-attn on")
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

// TestComputeDenseMultiGPUBalancedSplit recreates the Muse-Glimmer crash: a
// 30 GB dense model on 24/12/12 GB GPUs where the fast pair (CUDA0 gen4x16 +
// CUDA2 gen4x16) alone fits the model. The old subset-fit loop dropped the slow
// gen3x1 CUDA1 (split 0.00) and then CUDA2 OOM'd on its compute buffer. The
// balanced fix must give EVERY GPU a nonzero share so all free VRAM contributes.
func TestComputeDenseMultiGPUBalancedSplit(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24576, VRAMUsedMB: 736, BandwidthMBps: 31504},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, VRAMUsedMB: 6329, BandwidthMBps: 985},
			{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12288, VRAMUsedMB: 578, BandwidthMBps: 31504},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 100000},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "Muse-Glimmer.gguf",
		TotalSizeMB: 30804,
		SizeBytes:   30804 * 1024 * 1024,
		NumLayers:   80,
		IsMoE:       false,
		ContextSize: 16384,
		HiddenSize:  5120,
		CTXTrain:    16384,
		HeadCountKV: 8,
		KeyLength:   128,
		ValueLength: 128,
	}
	// ctx=16384 gives KV=1440, so the fast pair (CUDA0+CUDA2) fits the model in
	// the old subset-fit loop and CUDA1 got zeroed. With ctx=32768 (KV=2880) the
	// pair did NOT fit and the old code kept CUDA1 at 0.14 — not the crash.
	strat, err := Compute(caps, model, Options{ContextSize: 16384, KVPlacement: "gpu", KVQuality: "low", CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.Type != MultiGPUDense {
		t.Fatalf("expected multi_gpu_dense, got %s", strat.Type)
	}
	if len(strat.TensorSplit) != 3 {
		t.Fatalf("expected a 3-GPU split, got %v", strat.TensorSplit)
	}
	for gi, share := range strat.TensorSplit {
		if share <= 0 {
			t.Fatalf("GPU%d got a zero share; every GPU must contribute, split=%v", gi, strat.TensorSplit)
		}
	}
	// The split must still FIT each GPU: model+KV share + per-GPU compute reserve.
	totalSizeMB := model.TotalSizeMB
	kvTotal := computeKVTotalMB(model, 16384, strat.KVType, false)
	cudaOH := measuredCUDAOverheadMB(loadSystemProbe(t.TempDir(), caps.GPUs))
	for gi := 0; gi < 3; gi++ {
		share := strat.TensorSplit[gi]
		need := int(float64(totalSizeMB)*share) + int(float64(kvTotal)*share) + cudaOH + computeFloorMB
		free := caps.GPUs[gi].VRAMFreeMB()
		if need > free {
			t.Fatalf("GPU%d needs %d MiB but has %d free (share=%.2f)", gi, need, free, share)
		}
	}
}

// TestComputeDenseComputeReservePreventsOOM verifies the per-GPU compute buffer
// is reserved BEFORE the model split, using the Muse-Glimmer crash hardware and
// a probe cache carrying per-GPU measured compute buffers. CUDA2 free VRAM only
// fits its model+KV share plus ~752 MiB compute; the split must be sized so
// CUDA2's own compute allocation fits. This exercises the per-GPU compute path
// (ComputeBufByGPU), which the old dense code ignored — it charged the uniform
// aggregate (752) to every GPU and gave CUDA2 a 0.33 share needing 11392 MiB of
// its 11710, the razor-thin margin that OOM'd.
func TestComputeDenseComputeReservePreventsOOM(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24576, VRAMUsedMB: 736, BandwidthMBps: 31504},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, VRAMUsedMB: 6329, BandwidthMBps: 985},
			{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12288, VRAMUsedMB: 578, BandwidthMBps: 31504},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 100000},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "Muse-Glimmer.gguf",
		TotalSizeMB: 30804,
		SizeBytes:   30804 * 1024 * 1024,
		NumLayers:   80,
		IsMoE:       false,
		ContextSize: 16384,
		HiddenSize:  5120,
		CTXTrain:    16384,
		HeadCountKV: 8,
		KeyLength:   128,
		ValueLength: 128,
	}
	cacheDir := t.TempDir()
	// Per-GPU measured compute buffers: CUDA0 and CUDA2 need 752 MiB, CUDA1 (the
	// slow gen3x1 card, expert-ish role) needs only 64 MiB. A uniform aggregate
	// charge would over-reserve CUDA1 and under-reserve CUDA2.
	if err := writeProbeCacheForModel(cacheDir, model, 16384, 512, "low", "gpu", "llama", caps.GPUs, 1,
		map[int]int{0: 752, 1: 64, 2: 752}, nil, nil, 0); err != nil {
		t.Fatalf("write probe: %v", err)
	}
	strat, err := Compute(caps, model, Options{ContextSize: 16384, KVPlacement: "gpu", KVQuality: "low", BackendTag: "llama", CacheDir: cacheDir})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.Type != MultiGPUDense {
		t.Fatalf("expected multi_gpu_dense, got %s", strat.Type)
	}
	if len(strat.TensorSplit) != 3 || strat.TensorSplit[1] <= 0 {
		t.Fatalf("expected a 3-GPU split with CUDA1 contributing, got %v", strat.TensorSplit)
	}
	// Every GPU must fit its own model+KV share PLUS its measured compute buffer
	// (the per-GPU value the planner reserved before the split). CUDA2 is the
	// crash card: its compute buffer (752 MiB) must fit alongside its share.
	totalSizeMB := model.TotalSizeMB
	kvTotal := computeKVTotalMB(model, 16384, strat.KVType, false)
	measuredCompute := map[int]int{0: 752, 1: 64, 2: 752}
	cudaOH := measuredCUDAOverheadMB(loadSystemProbe(t.TempDir(), caps.GPUs))
	for gi := 0; gi < 3; gi++ {
		share := strat.TensorSplit[gi]
		need := int(float64(totalSizeMB)*share) + int(float64(kvTotal)*share) + cudaOH + measuredCompute[gi]
		free := caps.GPUs[gi].VRAMFreeMB()
		if need > free {
			t.Fatalf("GPU%d needs %d MiB but has %d free (share=%.2f, compute=%d); compute reserve not accounted", gi, need, free, share, measuredCompute[gi])
		}
	}
}

// TestComputeDenseSmallModelStillUsesSingleGPU verifies the tiny-model case is
// preserved: a small dense model that fits one GPU stays on SingleGPU (no split,
// no multi-GPU). The balanced-split change must not route small models onto more
// GPUs than needed.
func TestComputeDenseSmallModelStillUsesSingleGPU(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 31504},
			{Index: 1, VRAMTotalMB: 24576, BandwidthMBps: 31504},
			{Index: 2, VRAMTotalMB: 24576, BandwidthMBps: 31504},
		},
		RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path:        "small-dense.gguf",
		TotalSizeMB: 8192,
		SizeBytes:   8 * 1024 * 1024 * 1024,
		NumLayers:   32,
		IsMoE:       false,
		ContextSize: 32768,
		HiddenSize:  2048,
		CTXTrain:    32768,
		HeadCountKV: 8,
		KeyLength:   128,
		ValueLength: 128,
	}
	strat, err := Compute(caps, model, Options{ContextSize: 32768, KVPlacement: "gpu", KVQuality: "low", CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.Type != SingleGPU {
		t.Fatalf("expected single_gpu for a small model, got %s", strat.Type)
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

func TestComputeMoEForcedSplitOwnerRecomputesCompletePlacement(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 4070", VRAMTotalMB: 12288, BandwidthMBps: 4000},
			{Index: 1, Name: "RTX 3090 Ti", VRAMTotalMB: 24576, BandwidthMBps: 16000},
			{Index: 2, Name: "RTX 3060", VRAMTotalMB: 12288, BandwidthMBps: 1000},
		},
		RAM: detect.RAMInfo{TotalMB: 217088, FreeMB: 200000},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path: "roomy-moe.gguf", TotalSizeMB: 68 * 1024,
		SizeBytes: 68 * 1024 * 1024 * 1024,
		NumLayers: 32, LeadingDense: 2, IsMoE: true,
		NumExperts: 64, ExpertUsedCount: 4,
		ExpertBytes:    60 * 1024 * 1024 * 1024,
		NonExpertBytes: 8 * 1024 * 1024 * 1024,
		ContextSize:    32768, CTXTrain: 32768, HiddenSize: 4096,
		EmbeddingLength: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	owner := 1
	strategy, err := Compute(caps, model, Options{
		ContextSize: 32768, KVPlacement: "gpu", KVQuality: "low",
		Parallel: 2, NoMMap: true, CacheDir: t.TempDir(), MoESplitOwnerGPU: &owner,
	})
	if err != nil {
		t.Fatalf("forced-owner placement failed: %v", err)
	}
	if strategy.Type != MoEOffload {
		t.Fatalf("type=%s, want MoE offload", strategy.Type)
	}
	if strategy.MainGPU != owner || strategy.PlacementPolicy != "owner-1" {
		t.Fatalf("owner provenance lost: main=%d policy=%q", strategy.MainGPU, strategy.PlacementPolicy)
	}
	if len(strategy.TensorSplit) != 3 || strategy.TensorSplit[0] != 0 ||
		strategy.TensorSplit[1] != 1 || strategy.TensorSplit[2] != 0 {
		t.Fatalf("split=%v, want sole CUDA1 backbone owner", strategy.TensorSplit)
	}
	for _, entry := range strategy.VRAMLedger {
		if entry.GPU == owner && entry.ExpertOnly {
			t.Fatalf("owner GPU was marked expert-only: %+v", entry)
		}
		if entry.GPU != owner && !entry.ExpertOnly {
			t.Fatalf("non-owner GPU retained ordinary layers: %+v", entry)
		}
	}
	missing := 99
	if _, err := Compute(caps, model, Options{
		ContextSize: 32768, KVPlacement: "gpu", KVQuality: "low",
		NoMMap: true, CacheDir: t.TempDir(), MoESplitOwnerGPU: &missing,
	}); err == nil || !strings.Contains(err.Error(), "is not selected") {
		t.Fatalf("missing forced owner did not fail closed: %v", err)
	}
}

func TestDefaultMoEPackDoesNotAdoptCalibrationOwnerPolicy(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 4070", VRAMTotalMB: 12288, BandwidthMBps: 4000},
			{Index: 1, Name: "RTX 3090 Ti", VRAMTotalMB: 24576, BandwidthMBps: 16000},
			{Index: 2, Name: "RTX 3060", VRAMTotalMB: 12288, BandwidthMBps: 1000},
		},
		RAM: detect.RAMInfo{TotalMB: 217088, FreeMB: 200000},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path: "roomy-moe.gguf", TotalSizeMB: 68 * 1024,
		SizeBytes: 68 * 1024 * 1024 * 1024,
		NumLayers: 32, LeadingDense: 2, IsMoE: true,
		NumExperts: 64, ExpertUsedCount: 4,
		ExpertBytes:    60 * 1024 * 1024 * 1024,
		NonExpertBytes: 8 * 1024 * 1024 * 1024,
		ContextSize:    32768, CTXTrain: 32768, HiddenSize: 4096,
		EmbeddingLength: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	strategy, err := Compute(caps, model, Options{
		ContextSize: 32768, KVPlacement: "gpu", KVQuality: "low",
		Parallel: 2, NoMMap: true, CacheDir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("default pack failed: %v", err)
	}
	if strings.HasPrefix(strategy.PlacementPolicy, "owner-") {
		t.Fatalf("fit-first default adopted an unmeasured performance topology: main=%d policy=%q", strategy.MainGPU, strategy.PlacementPolicy)
	}
}

func TestDefaultMoEPackSpreadsBackboneWhenNoGPUHoldsKV(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "GPU A", VRAMTotalMB: 16384, BandwidthMBps: 16000},
			{Index: 1, Name: "GPU B", VRAMTotalMB: 16384, BandwidthMBps: 16000},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 120000},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path: "kv-spread-moe.gguf", TotalSizeMB: 80 * 1024,
		SizeBytes: 80 * 1024 * 1024 * 1024,
		NumLayers: 32, IsMoE: true, NumExperts: 64, ExpertUsedCount: 4,
		ExpertBytes:    60 * 1024 * 1024 * 1024,
		NonExpertBytes: 12 * 1024 * 1024 * 1024,
		ContextSize:    262144, CTXTrain: 262144, HiddenSize: 4096,
		EmbeddingLength: 4096, HeadCountKV: 16, KeyLength: 256, ValueLength: 256,
		MeasuredKVBytesPerTok: map[string]float64{"f16": 32768},
	}
	strategy, err := Compute(caps, model, Options{
		ContextSize: 262144, KVPlacement: "gpu", KVQuality: "high",
		NoMMap: true, CacheDir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("spread pack failed: %v", err)
	}
	if len(strategy.TensorSplit) != 2 || strategy.TensorSplit[0] <= 0 || strategy.TensorSplit[1] <= 0 {
		t.Fatalf("backbone+KV that does not fit on one GPU must keep a multi-owner split, got %v", strategy.TensorSplit)
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
	kvTotalMB := computeKVTotalMB(model, strat.ContextSize, strat.KVType, false)
	fixedPerGPU := computeFloorMB
	expertMBByDevice := otExpertMBByDevice(t, strat.OTString, expertPerLayerMB)
	for gi, gpu := range caps.GPUs {
		usedMB := fixedPerGPU + splitShareMB(nonExpertTotalMB, strat.TensorSplit, gi) + splitShareMB(kvTotalMB, strat.TensorSplit, gi) + expertMBByDevice[gpu.Index]
		if usedMB > gpu.VRAMFreeMB() {
			t.Fatalf("gpu %d over budget: used=%dMB free=%dMB split=%v ot=%s", gpu.Index, usedMB, gpu.VRAMFreeMB(), strat.TensorSplit, strat.OTString)
		}
	}
}

func TestComputeMoEExactLayerLedgerDoesNotDoubleChargeMovedTensors(t *testing.T) {
	const mib64 = int64(1024 * 1024)
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 15360}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "exact-moe.gguf", NumLayers: 2, IsMoE: true, NumExperts: 8,
		ContextSize: 4096, HiddenSize: 512, HeadCountKV: 1, KeyLength: 64, ValueLength: 64,
		TokenEmbdBytes:         4 * 1024 * mib64,
		ExpertBytes:            10 * 1024 * mib64,
		NonExpertBytes:         (4*1024 + 2*1024 + 200) * mib64,
		ShexpBytes:             2 * 1024 * mib64,
		ExpertAuxBytes:         200 * mib64,
		RoutedExpertLayerBytes: []int64{4 * 1024 * mib64, 4 * 1024 * mib64},
		ShexpLayerBytes:        []int64{1024 * mib64, 1024 * mib64},
		ExpertAuxLayerBytes:    []int64{100 * mib64, 100 * mib64},
		NonExpertLayerBytes:    []int64{1024 * mib64, 1024 * mib64},
	}
	model.SizeBytes = model.ExpertBytes + model.NonExpertBytes
	model.TotalSizeMB = bytesToMiBCeil(model.SizeBytes)

	strat, err := Compute(caps, model, Options{
		ContextSize: 4096, UBatchSize: 64, KVPlacement: "cpu", CacheDir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.NCPUMoE != 0 {
		t.Fatalf("exact ledger should fit both routed layers without double-charging shared/router tensors; n-cpu-moe=%d, ot=%s", strat.NCPUMoE, strat.OTString)
	}
	layers := parseOTLayersByDevice(t, strat.OTString)[0]
	if len(layers) != 2 {
		t.Fatalf("expected both expert layers on GPU0, got %v in %s", layers, strat.OTString)
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
	// A small MoE that fits across GPUs now goes to MultiGPUDense (all layers
	// on GPU via tensor-split + split-mode layer), not MoEOffload — so there is
	// no -ot/exps=CPU and no CPU MoE layers.
	if strat.NCPUMoE != 0 {
		t.Fatalf("expected no CPU MoE layers when experts fit, got %d", strat.NCPUMoE)
	}
	if strat.OTString != "" {
		t.Fatalf("a fitting multi-GPU MoE must not emit -ot (would imply CPU experts), got %q", strat.OTString)
	}
	if len(strat.TensorSplit) < 2 {
		t.Fatalf("a fitting multi-GPU MoE must use a tensor split, got %v", strat.TensorSplit)
	}
	if strat.Type != MultiGPUDense {
		t.Fatalf("a fitting multi-GPU MoE must be MultiGPUDense, got %s", strat.Type)
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

func TestFirstLaunchComputeBufDeepSeek4ScalesDownAt65KContext(t *testing.T) {
	model := &ModelProfile{
		IsMoE: true, ModelArch: "deepseek4", HiddenSize: 4096, NumLayers: 44, ExpertUsedCount: 6,
	}
	order := []int{0, 1, 2}
	at1M := firstLaunchComputeBufMBForGPUParallelAtContext(model, 256, 1, 1048576, 0, order)
	at65K := firstLaunchComputeBufMBForGPUParallelAtContext(model, 256, 1, 65536, 0, order)
	if at1M < 33000 || at1M > 37000 {
		t.Fatalf("1M graph reserve = %d MiB, want measured-scale ~34 GiB", at1M)
	}
	if at65K < 2800 || at65K > 3400 {
		t.Fatalf("65k graph reserve = %d MiB, want conservative scale near 3 GiB", at65K)
	}
	if at65K >= at1M/4 {
		t.Fatalf("65k context did not materially reduce graph reserve: 65k=%d 1M=%d", at65K, at1M)
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
	if !strategy.PlacementCacheHit {
		t.Fatal("validated placement cache load was not marked as a cache hit")
	}
	if strategy.MaxCheckpoints < hybridCheckpointMinimum || strategy.MaxCheckpoints > hybridCheckpointMaximum {
		t.Fatalf("cached hybrid placement checkpoints=%d, want %d..%d", strategy.MaxCheckpoints, hybridCheckpointMinimum, hybridCheckpointMaximum)
	}
	if args := strategy.Args(model.Path, 8081); !hasAdjacentArgPlacement(args, "--ctx-checkpoints", strconv.Itoa(strategy.MaxCheckpoints)) {
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
	if entry.OTString != newOT {
		t.Fatalf("derated cache OT = %q, want exact live argv %q", entry.OTString, newOT)
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

func TestDerateCUDAOOMArgsForDeficitJumpsPastInsufficientRung(t *testing.T) {
	model := &ModelProfile{NumLayers: 32}
	args := []string{"llama-server", "-b", "2048", "-ub", "256"}
	newArgs, entry, ok := DerateCUDAOOMArgsForDeficit(args, model, nil, 0, 17495, 12666, true)
	if !ok || entry == nil {
		t.Fatalf("measured compute deficit did not produce a ubatch recovery: entry=%+v ok=%v", entry, ok)
	}
	if got := currentUBatch(newArgs); got != 64 || entry.UBatchSize != 64 {
		t.Fatalf("17,495 MiB allocation / 12,666 MiB deficit selected ubatch %d, want 64: %v", got, newArgs)
	}
}

func TestCurrentUBatchMatchesBackendLastValueWins(t *testing.T) {
	args := []string{"llama-server", "-ub", "512", "--ubatch-size", "64"}
	if got := CurrentUBatch(args); got != 64 {
		t.Fatalf("effective ubatch = %d, want final override 64", got)
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
		NumLayers: 60, NumExperts: 256, SizeBytes: 137000 * 1024 * 1024,
		ExpertBytes: 120000 * 1024 * 1024, NonExpertBytes: 17000 * 1024 * 1024,
		EmbeddingLength: 4096, ContextSize: 1048576, VocabSize: 64,
		TokenizerModel: "gpt2", TokenizerPre: "joyai-llm",
	}
	help := "--spec-type none,draft-mtp,draft-dflash --spec-draft-model"
	opts := Options{
		SpecMode: "auto", BackendTag: "llama", BackendHelp: help,
		SpecCandidateValidator: func(string) error { return nil }, CacheDir: dir,
		ContextSize: 1048576, KVPlacement: "cpu", KVQuality: "high", RequireMeasuredBuffers: true,
	}
	saveEligibleSpecProfile(t, target, caps, opts, "dflash", drafter, 2)
	draft := ComputeDraft(target, caps, opts)
	if draft.Type != DraftDFlash || draft.Path != drafter || draft.SpecType != "draft-dflash" {
		t.Fatalf("expected DFlash before the generic MoE gate, got %#v", draft)
	}
	if draft.KVTypeDraft != "q4_0" {
		t.Fatalf("DFlash draft KV type = %q, want q4_0 default", draft.KVTypeDraft)
	}
	args := DraftFlags(draft)
	if !contains(args, "draft-dflash") || !contains(args, "--model-draft") || !contains(args, "q4_0") {
		t.Fatalf("DFlash flags missing: %v", args)
	}
	strategy, err := Compute(caps, target, opts)
	if err != nil {
		t.Fatalf("compute with DFlash reservation: %v", err)
	}
	if strategy.Draft == nil || strategy.Draft.Type != DraftDFlash {
		t.Fatalf("target placement lost DFlash: %#v", strategy.Draft)
	}
	if len(strategy.CompanionPlacements) != 1 || strategy.CompanionPlacements[0].Name != "spec-dflash" || strategy.CompanionPlacements[0].GPU != strategy.Draft.DraftGPU {
		t.Fatalf("DFlash was not reserved before target placement: draft=%#v companions=%+v", strategy.Draft, strategy.CompanionPlacements)
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

func TestHybridPromptCacheKeepsBranchCapableBoundedCheckpoints(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 8192, FreeMB: 8192},
	}
	s := &Strategy{Type: SingleGPU, HasSSM: true, Parallel: 2}
	cram, checkpoints := computeCRAM(caps, &ModelProfile{HasSSM: 1}, s, 4096, 256)
	if checkpoints < hybridCheckpointMinimum || checkpoints > hybridCheckpointMaximum {
		t.Fatalf("hybrid model checkpoints=%d, want %d..%d; cram=%d", checkpoints, hybridCheckpointMinimum, hybridCheckpointMaximum, cram)
	}
	s.CRAM = cram
	s.MaxCheckpoints = checkpoints
	s.ContextSize = 131072
	s.KVType = "q8_0"
	s.Threads = 8
	s.ThreadsBatch = 8
	s.BatchSize = 512
	s.UBatchSize = 128
	if args := s.Args("model.gguf", 8081); !hasAdjacentArgPlacement(args, "--ctx-checkpoints", strconv.Itoa(checkpoints)) {
		t.Fatalf("hybrid strategy did not emit its bounded checkpoint: %v", args)
	}
}

func TestCheckpointSpacingFlagFollowsBackendHelpDialect(t *testing.T) {
	cases := []struct {
		name string
		help string
		tag  string
		want string
	}{
		{name: "mainline", help: "  --checkpoint-min-step N", tag: "llama", want: "--checkpoint-min-step"},
		{name: "ik", help: "  --ctx-checkpoints-interval N", tag: "ik_llama", want: "--ctx-checkpoints-interval"},
		{name: "known unsupported", help: "llama-server options", tag: "llama", want: "-"},
		{name: "legacy ik fallback", tag: "ik_llama", want: "--ctx-checkpoints-interval"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := backendCheckpointMinStepFlag(tc.help, tc.tag); got != tc.want {
				t.Fatalf("flag = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestArgsUsesIKCheckpointIntervalInsteadOfMainlineFlag(t *testing.T) {
	s := &Strategy{
		Type:                         SingleGPU,
		ContextSize:                  131072,
		KVQuality:                    "mid",
		KVType:                       "q8_0",
		Threads:                      8,
		ThreadsBatch:                 8,
		BatchSize:                    128,
		UBatchSize:                   512,
		CRAM:                         4096,
		MaxCheckpoints:               1,
		CheckpointMinStep:            512,
		BackendTag:                   "ik_llama",
		BackendCheckpointMinStepFlag: "--ctx-checkpoints-interval",
	}
	args := s.Args("model.gguf", 8081)
	if !hasAdjacentArgPlacement(args, "--ctx-checkpoints-interval", "512") {
		t.Fatalf("ik checkpoint interval missing: %v", args)
	}
	for _, arg := range args {
		if arg == "--checkpoint-min-step" {
			t.Fatalf("mainline-only checkpoint flag leaked into ik argv: %v", args)
		}
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

func TestRAMBudgetOverridesPercentLimit(t *testing.T) {
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 128727, FreeMB: 125239},
	}
	if got := applyRAMPolicy(caps, Options{RAMLimitPercent: 90}).RAM.FreeMB; got != 112366 {
		t.Fatalf("percent policy free RAM = %d, want 112366", got)
	}
	if got := applyRAMPolicy(caps, Options{RamBudgetMB: 64000, RAMLimitPercent: 90}).RAM.FreeMB; got != 64000 {
		t.Fatalf("explicit RAM budget = %d, want 64000", got)
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

func TestIKLlamaCPUExpertsCannotUseMMapAsRAMCapacity(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, VRAMUsedMB: 1, VRAMReservedMB: 452, BandwidthMBps: 15760},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, VRAMUsedMB: 1, VRAMReservedMB: 378, BandwidthMBps: 985},
			{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12282, VRAMUsedMB: 1, VRAMReservedMB: 408, BandwidthMBps: 3940},
		},
		RAM: detect.RAMInfo{TotalMB: 112160, FreeMB: 112160},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "/models/MiniMax-M3.gguf", SizeBytes: 159407162624, TotalSizeMB: 152022,
		NumLayers: 60, IsMoE: true, NumExperts: 128, ExpertUsedCount: 4, ExpertFF: 3072,
		LeadingDense: 3, ExpertBytes: 151315808256, NonExpertBytes: 8083051008,
		TokenEmbdBytes: 1008322560, OutputBytes: 1008322560, ShexpBytes: 2661285888,
		ContextSize: 65536, CTXTrain: 1048576, HiddenSize: 6144, EmbeddingLength: 6144,
		HeadCountKV: 4, KeyLength: 128, ValueLength: 128, ModelArch: "minimax-m3",
	}
	opts := Options{ContextSize: 65536, KVPlacement: "cpu", KVQuality: "high", UBatchSize: 512, BackendTag: "ik_llama", CacheDir: t.TempDir()}
	_, err := Compute(caps, model, opts)
	if err == nil || !strings.Contains(err.Error(), "anonymous host memory") {
		t.Fatalf("ik_llama mmap must not bypass resident RAM: %v", err)
	}

	opts.BackendTag = "llama"
	strategy, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("file-backed backend should retain the mmap option: %v", err)
	}
	if !strategy.MMap || !strategy.MMapRequired {
		t.Fatalf("expected explicit mmap confirmation marker, got mmap=%v required=%v", strategy.MMap, strategy.MMapRequired)
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
	q5 := computeKVTotalMB(model, 1048576, "q5_1", false)
	q8 := computeKVTotalMB(model, 1048576, "q8_0", false)
	if q5 != 49152 {
		t.Fatalf("q5_1 KV size = %d MiB, want 49152 MiB", q5)
	}
	if q8 != 69632 {
		t.Fatalf("q8_0 KV size = %d MiB, want 69632 MiB", q8)
	}
	if q5 >= q8 {
		t.Fatalf("q5_1 must use less memory than q8_0: q5=%d q8=%d", q5, q8)
	}

	if got := kvTypesForAutoContext(model, "q5_1", "q5_1"); len(got) != 1 || got[0] != "q5_1" {
		t.Fatalf("explicit cache type must not silently fall back: %v", got)
	}
	if got := fallbackKVType(model, "q5_1", "q5_1"); got != "q5_1" {
		t.Fatalf("exact cache type fallback = %q, want q5_1", got)
	}
}

func TestOddHeadDimensionUsesSafeKVType(t *testing.T) {
	model := &ModelProfile{
		Path: "stories15m.gguf", SizeBytes: 32 * 1024 * 1024,
		NumLayers: 6, ContextSize: 2048, HeadCountKV: 1,
		KeyLength: 48, ValueLength: 48,
	}
	for _, quality := range []string{"", "auto", "mid", "low"} {
		got, err := resolveKVQuality(model, quality, "llama")
		if err != nil {
			t.Fatalf("preset %q rejected instead of promoting safely: %v", quality, err)
		}
		if kvTypeFromQuality(got) != "f16" {
			t.Fatalf("preset %q resolved to %q/%q, want f16", quality, got, kvTypeFromQuality(got))
		}
	}

	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 8192, FreeMB: 8192},
		CPU: detect.CPUInfo{Cores: 4},
	}
	strategy, err := Compute(caps, model, Options{CPUMode: true, ContextSize: 2048, KVQuality: "auto"})
	if err != nil {
		t.Fatalf("compute odd-head-dimension model: %v", err)
	}
	if strategy.KVType != "f16" {
		t.Fatalf("odd-head-dimension plan selected %q KV, want f16", strategy.KVType)
	}
}

func TestOddHeadDimensionRejectsExactQuantizedKVType(t *testing.T) {
	model := &ModelProfile{KeyLength: 48, ValueLength: 48}
	for _, kvType := range []string{"q8_0", "q5_1", "q4_0"} {
		_, err := resolveKVQuality(model, kvType, "llama")
		if err == nil || !strings.Contains(err.Error(), "--kv-quality f16") {
			t.Fatalf("exact %s error = %v, want a calm f16 recovery hint", kvType, err)
		}
	}
}

func TestAutoContextKVLadderFiltersUnsafeBlockTypes(t *testing.T) {
	model := &ModelProfile{KeyLength: 48, ValueLength: 48}
	for _, quality := range []string{"high", "auto", "mid"} {
		got := kvTypesForAutoContext(model, "f16", quality)
		if len(got) != 1 || got[0] != "f16" {
			t.Fatalf("quality %q unsafe ladder = %v, want [f16]", quality, got)
		}
		if fallback := fallbackKVType(model, "f16", quality); fallback != "f16" {
			t.Fatalf("quality %q unsafe fallback = %q, want f16", quality, fallback)
		}
	}
}

func TestUnknownHeadDimensionSkipsKVBlockGuard(t *testing.T) {
	for _, model := range []*ModelProfile{{KeyLength: 0, ValueLength: 48}, {KeyLength: 48, ValueLength: 0}} {
		got, err := resolveKVQuality(model, "q8_0", "llama")
		if err != nil || got != "q8_0" {
			t.Fatalf("unknown head dimension must preserve q8_0: got %q, err %v", got, err)
		}
	}
}

// A GGUF that omits attention.key_length/value_length is not unconstrained:
// llama.cpp derives the head width from n_embd / n_head and enforces the
// block-size rule against it. Treating the absent keys as "no constraint" made
// ggrun emit --cache-type-k q8_0 for stories260K (n_embd 64, n_head 8, so an
// 8-wide head), and llama-server died during startup with "K cache type q8_0
// with block size 32 does not divide n_embd_head_k=8".
func TestDerivedHeadDimensionDrivesKVBlockGuard(t *testing.T) {
	model := &ModelProfile{
		Path: "stories260k.gguf", SizeBytes: 2 * 1024 * 1024,
		NumLayers: 5, ContextSize: 2048, HeadCountKV: 8,
		EmbeddingLength: 64, HeadCount: 8, // n_embd_head_k = 8, not divisible by 32
	}
	for _, quality := range []string{"", "auto", "mid", "low"} {
		got, err := resolveKVQuality(model, quality, "llama")
		if err != nil {
			t.Fatalf("preset %q rejected instead of promoting safely: %v", quality, err)
		}
		if kvTypeFromQuality(got) != "f16" {
			t.Fatalf("preset %q resolved to %q/%q, want f16", quality, got, kvTypeFromQuality(got))
		}
	}

	_, err := resolveKVQuality(model, "q8_0", "llama")
	if err == nil || !strings.Contains(err.Error(), "key_length=8") {
		t.Fatalf("exact q8_0 error = %v, want the derived width reported, not the absent zero", err)
	}

	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 8192, FreeMB: 8192},
		CPU: detect.CPUInfo{Cores: 4},
	}
	strategy, cerr := Compute(caps, model, Options{CPUMode: true, ContextSize: 2048, KVQuality: "auto"})
	if cerr != nil {
		t.Fatalf("compute derived-head-dimension model: %v", cerr)
	}
	if strategy.KVType != "f16" {
		t.Fatalf("derived-head-dimension plan selected %q KV, want f16", strategy.KVType)
	}
}

// The derivation must not make the guard paranoid: a model whose derived head
// width is a clean multiple of 32 keeps its quantized cache.
func TestDerivedHeadDimensionKeepsQuantizedKVWhenDivisible(t *testing.T) {
	model := &ModelProfile{EmbeddingLength: 4096, HeadCount: 32} // n_embd_head_k = 128
	got, err := resolveKVQuality(model, "q8_0", "llama")
	if err != nil || got != "q8_0" {
		t.Fatalf("divisible derived head dimension must keep q8_0: got %q, err %v", got, err)
	}
}

func TestNormalizeKVType(t *testing.T) {
	for input, want := range map[string]string{
		"auto": "q8_0", "high": "f16", "mid": "q8_0", "low": "q4_0", "Q5_1": "q5_1", "fp32": "f32",
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

// TestPromptCacheSizedFromHostRAMNotVRAM pins the distinction that broke
// caching on this project's own hardware: --cache-ram is a host-RAM prompt
// cache (server_prompt_cache copies slot state out via
// llama_state_seq_get_data into host buffers), but it was sized from VRAM
// headroom. A model large enough to saturate VRAM therefore got -cram 0, which
// also disabled --cache-idle-slots, so an agent evicted from a slot lost its
// entire prefix instead of parking it in tens of GiB of idle RAM.
func TestPromptCacheSizedFromHostRAMNotVRAM(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, BandwidthMBps: 15754},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, BandwidthMBps: 985},
			{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12282, BandwidthMBps: 3938},
		},
		// VRAM is fully committed to weights; host RAM is nearly empty.
		RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 120000},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "big-moe.gguf", Basename: "big-moe.gguf",
		SizeBytes: 73000000000, TotalSizeMB: 70000,
		NumLayers: 32, IsMoE: true, NumExperts: 128, ExpertUsedCount: 8,
		ExpertFF: 2048, ExpertBytes: 68000000000, NonExpertBytes: 4000000000,
		ModelArch: "qwen3moe", ContextSize: 262144, CTXTrain: 262144,
	}
	cram, _ := computeCRAM(caps, model, &Strategy{Type: MoEOffload, Parallel: 4}, 70000, 8000)

	if cram < minCramMB {
		t.Fatalf("prompt cache disabled (%d MiB) despite ~99 GiB of host RAM free after load", cram)
	}
	// It must still be a fraction of host RAM, not unbounded.
	ramAfterLoad := 120000 - (70000 - caps.TotalVRAM())
	if cram > ramAfterLoad/10+1 {
		t.Errorf("prompt cache %d MiB exceeds one tenth of the %d MiB host budget", cram, ramAfterLoad)
	}
}

// cram is derived from free RAM, so it moves a few MiB between otherwise
// identical runs. That drift reached the launch argv, and the Claude crash log
// was filed under a hash of the argv, so two aborts differing only by `-cram
// 9752` against `-cram 9742` were stored under different names and neither
// could teach the next launch.
//
// Quantizing absorbs the ordinary drift. It is not a guarantee -- free RAM does
// cross a boundary sometimes, measured going 9728 -> 9216 across three
// consecutive plans -- so this test pins the granularity, not reproducibility.
// Recovery is made independent of the value instead, by excluding it from the
// launch identity.
func TestPromptCacheBudgetIsQuantizedAgainstFreeRAMDrift(t *testing.T) {
	newCaps := func(freeMB int) *detect.Capabilities {
		return &detect.Capabilities{
			GPUs: []detect.GPU{
				{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, BandwidthMBps: 15754},
				{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, BandwidthMBps: 985},
				{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12282, BandwidthMBps: 3938},
			},
			RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: freeMB},
			CPU: detect.CPUInfo{Cores: 8},
		}
	}
	model := &ModelProfile{
		Path: "big-moe.gguf", Basename: "big-moe.gguf",
		SizeBytes: 73000000000, TotalSizeMB: 70000,
		NumLayers: 32, IsMoE: true, NumExperts: 128, ExpertUsedCount: 8,
		ExpertFF: 2048, ExpertBytes: 68000000000, NonExpertBytes: 4000000000,
		ModelArch: "qwen3moe", ContextSize: 262144, CTXTrain: 262144,
	}
	// The real drift that broke recovery: 120000 vs 119900 MiB free moved cram
	// by ten MiB and the launch stopped reproducing.
	base, baseCkpt := computeCRAM(newCaps(120000), model, &Strategy{Type: MoEOffload, Parallel: 4}, 70000, 8000)
	drift, driftCkpt := computeCRAM(newCaps(119900), model, &Strategy{Type: MoEOffload, Parallel: 4}, 70000, 8000)
	if base != drift {
		t.Errorf("cram moved %d -> %d MiB on 100 MiB of free-RAM drift", base, drift)
	}
	if baseCkpt != driftCkpt {
		t.Errorf("ctx-checkpoints moved %d -> %d on the same drift", baseCkpt, driftCkpt)
	}
	if base%cramQuantumMB != 0 {
		t.Errorf("cram %d MiB is not on a %d MiB boundary", base, cramQuantumMB)
	}
	// Quantizing rounds down so the host-RAM ceiling above it still holds.
	ramAfterLoad := 120000 - (70000 - newCaps(120000).TotalVRAM())
	if base > ramAfterLoad*2/3 {
		t.Errorf("cram %d MiB exceeds the %d MiB host budget", base, ramAfterLoad*2/3)
	}
}

// The point of a measured entry size is that the budget follows the slot count.
// Without one it does not: `slots` was referenced only inside a branch gated on
// a measurement nothing ever recorded, so one slot and four slots produced the
// identical 9728 MiB and no A/B over --parallel could mean anything.
func TestPromptCacheBudgetScalesWithSlotsOnceMeasured(t *testing.T) {
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
		Path: "big-moe.gguf", Basename: "big-moe.gguf",
		SizeBytes: 73000000000, TotalSizeMB: 70000,
		NumLayers: 32, IsMoE: true, NumExperts: 128, ExpertUsedCount: 8,
		ExpertFF: 2048, ExpertBytes: 68000000000, NonExpertBytes: 4000000000,
		ModelArch: "qwen3moe", ContextSize: 262144, CTXTrain: 262144,
	}
	// The entry size actually observed on this project.
	const entryMB float64 = 6494.703

	unmeasured := map[int]int{}
	measured := map[int]int{}
	for _, slots := range []int{1, 2, 4} {
		u, _ := computeCRAM(caps, model, &Strategy{Type: MoEOffload, Parallel: slots}, 70000, 8000)
		m, _ := computeCRAM(caps, model,
			&Strategy{Type: MoEOffload, Parallel: slots, MeasuredPromptCacheEntryMB: entryMB}, 70000, 8000)
		unmeasured[slots], measured[slots] = u, m
	}
	if unmeasured[1] != unmeasured[4] {
		t.Errorf("without a measurement the budget should not vary: 1 slot=%d, 4 slots=%d", unmeasured[1], unmeasured[4])
	}
	if !(measured[1] < measured[2] && measured[2] < measured[4]) {
		t.Errorf("measured budget did not grow with slots: %d / %d / %d", measured[1], measured[2], measured[4])
	}
	// Four slots need one resident entry each, plus one so a conversation is
	// stored before the one it replaces is dropped.
	wantRaw := int(math.Ceil(entryMB * 5))
	want := ((wantRaw + cramQuantumMB - 1) / cramQuantumMB) * cramQuantumMB
	if measured[4] != want {
		t.Errorf("4-slot budget %d MiB, want %d (ceiling of 5 x %.0f MiB entry)", measured[4], want, entryMB)
	}
	// It still may not exceed the host budget the weights need.
	ramAfterLoad := 120000 - (70000 - caps.TotalVRAM())
	if measured[4] > ramAfterLoad*2/3 {
		t.Errorf("measured budget %d MiB exceeds the %d MiB host ceiling", measured[4], ramAfterLoad*2/3)
	}
}

// A genuinely RAM-starved host must still disable the cache rather than push
// the machine into swap.
func TestPromptCacheStaysDisabledWhenHostRAMIsExhausted(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, BandwidthMBps: 15754},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, BandwidthMBps: 985},
		},
		RAM: detect.RAMInfo{TotalMB: 64000, FreeMB: 40000},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "huge.gguf", Basename: "huge.gguf",
		SizeBytes: 84000000000, TotalSizeMB: 80000,
		NumLayers: 32, IsMoE: true, NumExperts: 128, ExpertUsedCount: 8,
		ExpertFF: 2048, ExpertBytes: 78000000000, NonExpertBytes: 2000000000,
		ModelArch: "qwen3moe", ContextSize: 65536, CTXTrain: 65536,
	}
	// 80 GiB model, 36 GiB VRAM: ~44 GiB must live in 40 GiB of RAM, leaving
	// nothing for a cache.
	cram, _ := computeCRAM(caps, model, &Strategy{Type: MoEOffload, Parallel: 4}, 80000, 4000)
	if cram != 0 {
		t.Errorf("expected the prompt cache disabled on a RAM-starved host, got %d MiB", cram)
	}
}

func TestThreadsOptionOverridesPhysicalCoreDefault(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 65536},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "model.gguf", SizeBytes: 15 * 1024 * 1024 * 1024,
		NumLayers: 64, NumParams: 32_000_000_000, ContextSize: 32768, HiddenSize: 4096,
	}

	// Physical cores stay the default: CPU-resident experts are bandwidth-bound
	// and SMT siblings share the ports that bound them.
	base, err := Compute(caps, model, Options{})
	if err != nil {
		t.Fatalf("compute: %v", err)
	}
	if base.Threads != 8 || base.ThreadsBatch != 8 {
		t.Errorf("default threads = %d/%d, want 8/8", base.Threads, base.ThreadsBatch)
	}

	// An explicit request wins, so the default can be measured rather than
	// assumed. Batch threads follow, matching how the pair is emitted.
	over, err := Compute(caps, model, Options{Threads: 16})
	if err != nil {
		t.Fatalf("compute: %v", err)
	}
	if over.Threads != 16 || over.ThreadsBatch != 16 {
		t.Errorf("overridden threads = %d/%d, want 16/16", over.Threads, over.ThreadsBatch)
	}

	// Zero means "unset", not "zero threads".
	zero, err := Compute(caps, model, Options{Threads: 0})
	if err != nil {
		t.Fatalf("compute: %v", err)
	}
	if zero.Threads != 8 {
		t.Errorf("threads with 0 = %d, want the 8-core default", zero.Threads)
	}
}

func TestArgsPinsPhysicalCoresForPrefillAndDecode(t *testing.T) {
	s := &Strategy{
		ContextSize: 8192, BatchSize: 512, UBatchSize: 256,
		KVType: "q8_0", KVPlacement: "gpu", Threads: 14, ThreadsBatch: 14, NCPUMoE: 8,
		CPURange: "0-13", CPUStrict: true,
		BackendSupportsCPURange:       true,
		BackendSupportsCPURangeBatch:  true,
		BackendSupportsCPUStrict:      true,
		BackendSupportsCPUStrictBatch: true,
	}
	args := strings.Join(s.Args("m.gguf", 8081), " ")
	for _, want := range []string{
		"--cpu-range 0-13", "--cpu-strict 1",
		"--cpu-range-batch 0-13", "--cpu-strict-batch 1",
	} {
		if !strings.Contains(args, want) {
			t.Fatalf("args missing %q: %s", want, args)
		}
	}
}

func TestArgsDoesNotApplyCPUExpertAffinityToGPUOnlyStrategy(t *testing.T) {
	s := &Strategy{
		ContextSize: 8192, BatchSize: 512, UBatchSize: 256,
		KVType: "q8_0", KVPlacement: "gpu", Threads: 14, ThreadsBatch: 14,
		CPURange: "0-13", CPUStrict: true, BackendSupportsCPURange: true,
	}
	if args := strings.Join(s.Args("m.gguf", 8081), " "); strings.Contains(args, "--cpu-range") {
		t.Fatalf("unmeasured CPU-expert affinity leaked into a GPU-only strategy: %s", args)
	}
}

func TestArgsRequiresExactStrictBatchCapability(t *testing.T) {
	s := &Strategy{
		ContextSize: 8192, BatchSize: 512, UBatchSize: 256, NCPUMoE: 8,
		KVType: "q8_0", KVPlacement: "gpu", Threads: 14, ThreadsBatch: 14,
		CPURange: "0-13", CPUStrict: true,
		BackendSupportsCPURange: true, BackendSupportsCPURangeBatch: true,
		BackendSupportsCPUStrict: true,
	}
	args := strings.Join(s.Args("m.gguf", 8081), " ")
	if !strings.Contains(args, "--cpu-range-batch 0-13") || strings.Contains(args, "--cpu-strict-batch") {
		t.Fatalf("unsupported strict-batch flag emission: %s", args)
	}
}

func TestCPUAffinityCapabilityUsesExactFlagTokens(t *testing.T) {
	s := &Strategy{}
	caps := &detect.Capabilities{CPU: detect.CPUInfo{PhysicalIDs: []int{0, 1, 2, 3}}}
	configureCPUAffinity(s, caps, "--cpu-range-batch lo-hi\n--cpu-strict-batch 0|1")
	if s.BackendSupportsCPURange || s.BackendSupportsCPUStrict || s.CPURange != "" {
		t.Fatalf("batch-only flags became proof of main affinity support: %+v", s)
	}
}

func TestComputeSetsCPURangeFromPhysicalIDs(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 40000},
		CPU:  detect.CPUInfo{Cores: 14, Threads: 28, PhysicalIDs: []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}},
	}
	model := &ModelProfile{
		Path: "model.gguf", SizeBytes: 15 * 1024 * 1024 * 1024,
		NumLayers: 64, NumParams: 32_000_000_000, ContextSize: 32768, HiddenSize: 4096,
	}
	s, err := Compute(caps, model, Options{BackendHelp: "--cpu-range lo-hi\n--cpu-strict\n--cpu-range-batch lo-hi\n--cpu-strict-batch\n"})
	if err != nil {
		t.Fatal(err)
	}
	if s.CPURange != "0-13" || !s.CPUStrict {
		t.Fatalf("cpu affinity = range %q strict %v, want 0-13/true", s.CPURange, s.CPUStrict)
	}
	s.NCPUMoE = 1
	joined := strings.Join(s.Args("m.gguf", 8081), " ")
	if !strings.Contains(joined, "--cpu-range 0-13") || !strings.Contains(joined, "--cpu-range-batch 0-13") {
		t.Fatalf("emitted args missing prefill+decode pin: %s", joined)
	}
}

func TestComputeOmitsCPURangeWithoutBackendSupport(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 40000},
		CPU:  detect.CPUInfo{Cores: 8, PhysicalIDs: []int{0, 1, 2, 3, 4, 5, 6, 7}},
	}
	model := &ModelProfile{
		Path: "model.gguf", SizeBytes: 15 * 1024 * 1024 * 1024,
		NumLayers: 64, NumParams: 32_000_000_000, ContextSize: 32768, HiddenSize: 4096,
	}
	s, err := Compute(caps, model, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if s.CPURange != "" {
		t.Fatalf("cpu-range leaked without backend help: %q", s.CPURange)
	}
}

func TestCacheRAMOverrideBeatsDerivedBudget(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 120000},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "model.gguf", SizeBytes: 15 * 1024 * 1024 * 1024,
		NumLayers: 64, NumParams: 32_000_000_000, ContextSize: 32768, HiddenSize: 4096,
	}

	derived, err := Compute(caps, model, Options{})
	if err != nil {
		t.Fatalf("compute: %v", err)
	}
	// Whatever it derives, it must stay inside the safety budget: the cache
	// shares a memory scope with the weights, and overshooting turns a slow
	// launch into a failed one.
	if derived.CRAM > caps.RAM.FreeMB*2/3 {
		t.Errorf("derived CRAM = %d exceeds the safety budget", derived.CRAM)
	}

	// An explicit budget wins, including above that cap -- one measured entry
	// here was ~6.5 GiB, so holding one per slot needs far more than 16 GiB.
	over, err := Compute(caps, model, Options{CacheRAMMB: 40960})
	if err != nil {
		t.Fatalf("compute: %v", err)
	}
	if over.CRAM != 40960 {
		t.Errorf("overridden CRAM = %d, want 40960", over.CRAM)
	}

	// Zero means "derive it", not "disable the cache".
	zero, err := Compute(caps, model, Options{CacheRAMMB: 0})
	if err != nil {
		t.Fatalf("compute: %v", err)
	}
	if zero.CRAM != derived.CRAM {
		t.Errorf("CRAM with 0 = %d, want the derived %d", zero.CRAM, derived.CRAM)
	}
}

// growthCarryFixture is the real rig: a split-owner 3090 Ti plus two 12 GB
// expert-only cards. Runtime growth was measured on the split owner only, which
// is what makes the per-device rule load-bearing.
func growthCarryFixture() ([]detect.GPU, *ModelProfile) {
	gpus := []detect.GPU{
		{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, BandwidthMBps: 15754},
		{Index: 1, Name: "RTX 3060 x1", VRAMTotalMB: 12288, BandwidthMBps: 985},
		{Index: 2, Name: "RTX 4070 x4", VRAMTotalMB: 12282, BandwidthMBps: 3938},
	}
	model := &ModelProfile{
		Path:      "/models/DeepSeek-V4-Flash-UD-Q3_K_XL-00001-of-00004.gguf",
		NumLayers: 43, IsMoE: true, NumExperts: 256, ModelArch: "deepseek4",
	}
	return gpus, model
}

// Runtime graph growth cannot be measured before a request exists, so a cold
// ctx/ubatch key would budget 0 and OOM on its first real request. The carry
// takes a MEASURED value from a related key of the same model and applies it
// only to the device that actually carried it. Charging the split owner's
// 5504 MiB to the 12 GB expert-only cards would budget them negative.
func TestRelatedModelRuntimeGraphGrowthCarriesMeasuredValuePerDevice(t *testing.T) {
	cacheDir := t.TempDir()
	gpus, model := growthCarryFixture()

	// A related key: same model/GPUs/parallel, different ctx and ubatch.
	if err := writeProbeCacheForModel(cacheDir, model, 131072, 512, "high", "gpu", "llama", gpus, 1,
		map[int]int{0: 1391}, map[int]int{0: 5504}, nil, 0); err != nil {
		t.Fatalf("write related probe: %v", err)
	}

	related := RelatedModelRuntimeGraphGrowth(cacheDir, model, gpus, 1, "llama")
	if related == nil {
		t.Fatal("cold key got no carry from a measured related key")
	}
	if got := related[0]; got != 5504 {
		t.Fatalf("CUDA0 carry = %d, want the measured 5504", got)
	}
	for _, dev := range []int{1, 2} {
		if got := related[dev]; got != 0 {
			t.Fatalf("split-owner growth charged to expert-only CUDA%d: %d MiB", dev, got)
		}
	}
}

// The carry takes the largest measured value across related keys: growth scales
// with model graph shape, so the biggest observation for this model is the one
// the cold key must survive.
func TestRelatedModelRuntimeGraphGrowthTakesPerDeviceMaximum(t *testing.T) {
	cacheDir := t.TempDir()
	gpus, model := growthCarryFixture()

	if err := writeProbeCacheForModel(cacheDir, model, 65536, 512, "high", "gpu", "llama", gpus, 1,
		nil, map[int]int{0: 2457}, nil, 0); err != nil {
		t.Fatalf("write first related probe: %v", err)
	}
	if err := writeProbeCacheForModel(cacheDir, model, 131072, 1024, "high", "gpu", "llama", gpus, 1,
		nil, map[int]int{0: 5504}, nil, 0); err != nil {
		t.Fatalf("write second related probe: %v", err)
	}

	if got := RelatedModelRuntimeGraphGrowth(cacheDir, model, gpus, 1, "llama")[0]; got != 5504 {
		t.Fatalf("carry = %d, want the larger measured 5504", got)
	}
}

// What is NOT evidence for a carry. Each of these was a correction from the
// adversarial verify of the design; relaxing any of them either launders a guess
// into a reserve or breaks the cross-parallel contract.
func TestRelatedModelRuntimeGraphGrowthRejectsNonEvidence(t *testing.T) {
	gpus, model := growthCarryFixture()

	t.Run("estimated growth is not a measurement", func(t *testing.T) {
		cacheDir := t.TempDir()
		if err := writeProbeCacheForModel(cacheDir, model, 131072, 512, "high", "gpu", "llama", gpus, 1,
			nil, map[int]int{0: 5504}, map[int]bool{0: true}, 0); err != nil {
			t.Fatalf("write estimated probe: %v", err)
		}
		if got := RelatedModelRuntimeGraphGrowth(cacheDir, model, gpus, 1, "llama"); got[0] != 0 {
			t.Fatalf("estimated growth carried as measured evidence: %d MiB", got[0])
		}
	})

	// A different slot count is not evidence. The recorder files the size of
	// whatever allocation failed, so a probe can hold a KV-buffer figure
	// mislabelled as growth (verified: CUDA0=5504 was that plan's KV buffer).
	// Widening the match spreads a bad measurement instead of containing it.
	t.Run("a different parallel is not evidence", func(t *testing.T) {
		cacheDir := t.TempDir()
		if err := writeProbeCacheForModel(cacheDir, model, 131072, 512, "high", "gpu", "llama", gpus, 4,
			nil, map[int]int{0: 5504}, nil, 0); err != nil {
			t.Fatalf("write parallel-4 probe: %v", err)
		}
		if got := RelatedModelRuntimeGraphGrowth(cacheDir, model, gpus, 1, "llama"); got[0] != 0 {
			t.Fatalf("parallel-4 measurement carried into a parallel-1 launch: %d MiB", got[0])
		}
	})

	t.Run("a different model is not evidence", func(t *testing.T) {
		cacheDir := t.TempDir()
		other := *model
		other.Path = "/models/Laguna-120B-Q4_K_M.gguf"
		if err := writeProbeCacheForModel(cacheDir, &other, 131072, 512, "high", "gpu", "llama", gpus, 1,
			nil, map[int]int{0: 4914}, nil, 0); err != nil {
			t.Fatalf("write foreign-model probe: %v", err)
		}
		if got := RelatedModelRuntimeGraphGrowth(cacheDir, model, gpus, 1, "llama"); got[0] != 0 {
			t.Fatalf("another model's growth carried: %d MiB", got[0])
		}
	})

	t.Run("a different GPU set is not evidence", func(t *testing.T) {
		cacheDir := t.TempDir()
		otherGPUs := []detect.GPU{{Index: 0, Name: "RTX 4090", VRAMTotalMB: 24564, BandwidthMBps: 20000}}
		if err := writeProbeCacheForModel(cacheDir, model, 131072, 512, "high", "gpu", "llama", otherGPUs, 1,
			nil, map[int]int{0: 5504}, nil, 0); err != nil {
			t.Fatalf("write foreign-hardware probe: %v", err)
		}
		if got := RelatedModelRuntimeGraphGrowth(cacheDir, model, gpus, 1, "llama"); got[0] != 0 {
			t.Fatalf("another rig's growth carried: %d MiB", got[0])
		}
	})

	t.Run("an empty cache yields no carry", func(t *testing.T) {
		if got := RelatedModelRuntimeGraphGrowth(t.TempDir(), model, gpus, 1, "llama"); got != nil {
			t.Fatalf("empty cache produced a carry: %v", got)
		}
	})
}

// The measurement class the planner was missing: whatever a device holds beyond
// the buffers the backend reports. It captures CUDA context, graph capture and
// anything else no no-alloc oracle can see, on any backend, without needing a
// breakdown table or the right PID.
func TestWholeDeviceOverheadMBDerivesTheUnreportedRemainder(t *testing.T) {
	// The real 2026-08-05 reading: CUDA0 held 23922 MiB against 23151 MiB of
	// reported model+KV+compute, on a 24564 MiB card, with no companion.
	if got, ok := wholeDeviceOverheadMB(23922, 23151, 0, 24564); !ok || got != 771 {
		t.Fatalf("CUDA0 overhead = %d (ok=%v), want 771", got, ok)
	}
	// CUDA2 the same launch.
	if got, ok := wholeDeviceOverheadMB(11473, 11036, 0, 12282); !ok || got != 437 {
		t.Fatalf("CUDA2 overhead = %d (ok=%v), want 437", got, ok)
	}

	// A companion on the card is a separate tenant with its own budget line.
	// Charging it here is how a 12 GB card once latched 2916 MiB of "overhead":
	// without the subtraction this same reading looks like 6634 MiB of context.
	withCompanion, ok := wholeDeviceOverheadMB(12300, 5666, 6218, 12288)
	if !ok || withCompanion != 416 {
		t.Fatalf("overhead net of companion = %d (ok=%v), want 416", withCompanion, ok)
	}
	// Without the subtraction the same reading is 6634 MiB, which the outlier
	// ceiling rejects outright -- so the device would be left unmeasured rather
	// than latched. Both behaviours are safe; the subtraction is what turns an
	// unusable reading into a usable one.
	if bare, ok := wholeDeviceOverheadMB(12300, 5666, 0, 12288); ok {
		t.Fatalf("unsubtracted companion reading was accepted as %d MiB", bare)
	}
}

// What it refuses to record. A wrong value here is not one bad launch; it taxes
// every future plan on this hardware signature.
func TestWholeDeviceOverheadMBRejectsNonEvidence(t *testing.T) {
	for _, tc := range []struct {
		name                              string
		live, accounted, companion, total int
	}{
		{"foreign workload above the outlier ceiling", 20000, 4000, 0, 24564},
		{"nothing reported for this device", 12000, 0, 0, 24564},
		{"no live reading", 0, 11036, 0, 12282},
		{"reported exceeds live", 11036, 11473, 0, 12282},
		{"companion accounts for all of it", 11885, 5666, 6219, 12288},
		{"unknown card size", 23922, 23151, 0, 0},
	} {
		if got, ok := wholeDeviceOverheadMB(tc.live, tc.accounted, tc.companion, tc.total); ok {
			t.Fatalf("%s recorded %d MiB", tc.name, got)
		}
	}
}

// The prompt cache used to be sized against `totalSize - totalVRAM`, which
// charges the model for every byte of VRAM installed instead of the bytes the
// weights actually occupy there — VRAM also holds KV, compute buffers and CUDA
// context. Measured on this host serving DeepSeek-V4-Flash UD-Q3_K_XL at
// --n-cpu-moe 33, from the backend's own load_tensors report:
//
//	CUDA0 11998.86 + CUDA1 10734.00 + CUDA2 10736.96 = 33470 MiB of weights
//	CUDA_Host                                        = 88793 MiB of weights
//	total                                            = 122263 MiB
//
// The old subtraction called the host share 122268-49134 = 73134 MiB, understating
// it by ~15.7 GiB, and every downstream figure — the cache budget and the hybrid
// checkpoint headroom derived from it — was computed against RAM that was never
// free. The backend was OOM-killed by its own memory scope while saving prompt
// state.
func TestPromptCacheIsSizedAgainstHostWeightsNotInstalledVRAM(t *testing.T) {
	const (
		freeRAMMB       = 120000
		totalSizeMB     = 122268 // model total, all shards
		hostWeightsMB   = 88793  // CUDA_Host model buffer, measured
		kvTotalMB       = 6400
		measuredEntryMB = 6656 // one saved conversation, measured
		slots           = 4
	)
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, BandwidthMBps: 15754},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, BandwidthMBps: 985},
			{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12282, BandwidthMBps: 3938},
		},
		RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: freeRAMMB},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "v4.gguf", Basename: "v4.gguf",
		SizeBytes: 128225000000, TotalSizeMB: totalSizeMB,
		NumLayers: 44, IsMoE: true, NumExperts: 256, ExpertUsedCount: 8,
		ModelArch: "deepseek4", ContextSize: 131072, CTXTrain: 131072,
	}
	newStrategy := func(footprintMB int) *Strategy {
		return &Strategy{
			Type: MoEOffload, Parallel: slots, HasSSM: true,
			MeasuredPromptCacheEntryMB: measuredEntryMB,
			PlannedHostFootprintMB:     footprintMB,
		}
	}

	cram, _ := computeCRAM(caps, model, newStrategy(hostWeightsMB), totalSizeMB, kvTotalMB)
	if cram < minCramMB {
		t.Fatalf("prompt cache disabled (%d MiB) with %d MiB of host RAM left after weights",
			cram, freeRAMMB-hostWeightsMB)
	}
	// The invariant the OOM-kill violated: the weights and the cache they share
	// a memory scope with must both fit in the RAM that was free.
	if committed := hostWeightsMB + cram; committed > freeRAMMB {
		t.Errorf("plan commits %d MiB of host RAM (%d weights + %d cache) but only %d MiB was free",
			committed, hostWeightsMB, cram, freeRAMMB)
	}

	// Without a plan-derived footprint the old subtraction still runs, and on
	// these numbers it overcommits. Pin that so the fallback is never mistaken
	// for an equivalent answer.
	legacy, _ := computeCRAM(caps, model, newStrategy(0), totalSizeMB, kvTotalMB)
	if legacy <= cram {
		t.Fatalf("expected the VRAM-install subtraction to grant more than the measured footprint does, got legacy=%d measured=%d", legacy, cram)
	}
	if hostWeightsMB+legacy <= freeRAMMB {
		t.Fatalf("fixture no longer reproduces the overcommit: legacy cram %d + %d weights fits in %d MiB",
			legacy, hostWeightsMB, freeRAMMB)
	}
}

// Charging the cache for mmap-backed expert pages would disable it on exactly
// the hosts that page happily today: those bytes are clean and file-backed, so
// the kernel reclaims them under pressure. Only the anonymous working set is
// genuinely spoken for.
func TestPromptCacheChargesOnlyTheAnonymousWorkingSetUnderMmap(t *testing.T) {
	const residentMB, workingSetMB = 88793, 4200

	mainline := Options{CPUExpertMMapCapability: CPUExpertMMapFileBacked}
	if got := hostFootprintForCache(residentMB, workingSetMB, false, mainline); got != residentMB {
		t.Errorf("resident plan should charge the whole footprint, got %d want %d", got, residentMB)
	}
	if got := hostFootprintForCache(residentMB, workingSetMB, true, mainline); got != workingSetMB {
		t.Errorf("mmap plan should charge only the working set, got %d want %d", got, workingSetMB)
	}
	if got := hostFootprintForCache(-1, -1, false, mainline); got != 0 {
		t.Errorf("a negative footprint must clamp to 0 so it reads as 'not derived', got %d", got)
	}
}

func TestCachedMMapWorkingSetIncludesEmbeddingsAndCheckpoints(t *testing.T) {
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 100000},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		IsMoE: true, NumLayers: 2, TotalSizeMB: 4096,
		ExpertBytes: 2 * 1024 * 1024 * 1024, TokenEmbdBytes: 512 * 1024 * 1024,
		SlidingWindow: 512,
	}
	strategy := &Strategy{ContextSize: 8192, Parallel: 2, KVPlacement: "gpu"}
	cache := &CacheEntry{NCPUMoE: 1, UBatchSize: 128, Parallel: 2, KVUnified: true, MMap: true}
	opts := Options{ForceMMap: true, CPUExpertMMapCapability: CPUExpertMMapFileBacked}
	const kvTotalMB = 8192
	fits, mmap, _, footprint := cachedMoEHostMemoryFits(caps, model, strategy, cache, model.TotalSizeMB, kvTotalMB, opts)
	if !fits || !mmap {
		t.Fatalf("fixture did not produce an mmap-backed cached plan: fits=%v mmap=%v", fits, mmap)
	}
	want := plannedRAMRuntimeOverheadMB(caps, model, cache.UBatchSize, model.TotalSizeMB, opts) +
		bytesToMiBCeil(model.TokenEmbdBytes) + checkpointFootprintMB(model, 0, strategy.Parallel, kvTotalMB, strategy.ContextSize)
	if footprint != want {
		t.Fatalf("mmap working set = %d MiB, want exact fixed host state %d MiB", footprint, want)
	}
}

// A fully-resident dense placement must derive a real PlannedHostFootprintMB
// (runtime overhead + token embeddings + CPU-side KV + checkpoint reserve), not
// leave it at 0. With it at 0, Fix-B's measured-footprint resize collapsed the
// cgroup ceiling to bare CRAM -- the prompt-cache budget -- and the backend
// OOM'd against its own scope on a long prompt (Qwen3.8 27B crash: scope capped
// at ~11 GB = CRAM, server killed at 10.9 GB on a 36k-token prompt). The dense
// builders (SingleGPU / MultiGPUDense) set the field; this test pins it.
func TestDenseBuildersDerivePlannedHostFootprint(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, BandwidthMBps: 15754},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, BandwidthMBps: 985},
			{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12282, BandwidthMBps: 3938},
		},
		RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 114844},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "qwen3.gguf", Basename: "qwen3.gguf",
		SizeBytes: 27000000000, TotalSizeMB: 27000,
		NumLayers: 40, FeedForwardLength: 14336, EmbeddingLength: 5120,
		TokenEmbdBytes: 50000000, ContextSize: 32768, CTXTrain: 32768,
	}
	kvTotalMB := computeKVTotalMB(model, 32768, "f16", false)

	// MultiGPUDense with CPU-side KV must charge overhead + embeddings + KV.
	multi := &Strategy{Type: MultiGPUDense, ContextSize: 32768, UBatchSize: 512, BatchSize: 512, Parallel: 1, KVPlacement: "cpu"}
	s, err := buildMultiGPUDense(multi, caps, model, model.TotalSizeMB, kvTotalMB, Options{ContextSize: 32768, KVPlacement: "cpu", Parallel: 1})
	if err != nil {
		t.Fatalf("buildMultiGPUDense should fit: %v", err)
	}
	if s.PlannedHostFootprintMB <= 0 {
		t.Fatalf("MultiGPUDense must derive a real PlannedHostFootprintMB, got 0")
	}
	// The footprint must cover at least the runtime overhead + token embeddings
	// (and, with CPU KV, the KV itself) -- well above zero.
	overhead := ramRuntimeOverheadMB(model, s.UBatchSize, model.TotalSizeMB)
	embd := bytesToMiBCeil(model.TokenEmbdBytes)
	if s.PlannedHostFootprintMB < overhead+embd+kvTotalMB {
		t.Fatalf("MultiGPUDense host footprint %d < overhead %d + embeddings %d + cpuKV %d",
			s.PlannedHostFootprintMB, overhead, embd, kvTotalMB)
	}

	// SingleGPU (KV on GPU) must charge overhead + embeddings. Use a smaller
	// model that fits one GPU's VRAM (weights + overhead + KV on the 3090 Ti).
	singleModel := &ModelProfile{
		Path: "small-dense.gguf", Basename: "small-dense.gguf",
		SizeBytes: 10000000000, TotalSizeMB: 10000,
		NumLayers: 40, FeedForwardLength: 14336, EmbeddingLength: 5120,
		TokenEmbdBytes: 50000000, ContextSize: 32768, CTXTrain: 32768,
	}
	singleKV := computeKVTotalMB(singleModel, 32768, "f16", false)
	single := &Strategy{Type: SingleGPU, ContextSize: 32768, UBatchSize: 512, BatchSize: 512, Parallel: 1, KVPlacement: "gpu"}
	ss, err := buildSingleGPU(single, caps, singleModel, singleModel.TotalSizeMB, singleKV, Options{ContextSize: 32768, KVPlacement: "gpu", Parallel: 1})
	if err != nil {
		t.Fatalf("buildSingleGPU should fit: %v", err)
	}
	if ss.PlannedHostFootprintMB <= 0 {
		t.Fatalf("SingleGPU must derive a real PlannedHostFootprintMB, got 0")
	}
	singleOverhead := ramRuntimeOverheadMB(singleModel, ss.UBatchSize, singleModel.TotalSizeMB)
	singleEmbd := bytesToMiBCeil(singleModel.TokenEmbdBytes)
	if ss.PlannedHostFootprintMB < singleOverhead+singleEmbd {
		t.Fatalf("SingleGPU host footprint %d < overhead %d + embeddings %d", ss.PlannedHostFootprintMB, singleOverhead, singleEmbd)
	}
}

// computeCRAM must charge the dense plan's resident host footprint, not the
// whole free RAM. The old `ramAfterLoad = FreeMB` for SingleGPU/MultiGPUDense
// handed the prompt cache the entire free RAM, so a dense model with a large
// CRAM shared its cgroup budget with the server's own footprint and OOM'd on a
// long prompt (Qwen3.8 27B: -cram 11264 against an ~11 GB ceiling). Once the
// builders derive PlannedHostFootprintMB the cache is sized against RAM that is
// really left.
func TestComputeCRAMChargesDensePlannedHostFootprint(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, BandwidthMBps: 15754},
			{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288, BandwidthMBps: 985},
			{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12282, BandwidthMBps: 3938},
		},
		RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 114844},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "qwen3.gguf", Basename: "qwen3.gguf",
		SizeBytes: 27000000000, TotalSizeMB: 27000,
		NumLayers: 40, FeedForwardLength: 14336, EmbeddingLength: 5120,
		TokenEmbdBytes: 50000000, ContextSize: 32768, CTXTrain: 32768,
	}

	// A dense plan WITH a derived host footprint must charge it.
	footprint := 3000
	dense := &Strategy{Type: MultiGPUDense, Parallel: 1, PlannedHostFootprintMB: footprint}
	cram, _ := computeCRAM(caps, model, dense, 27000, 2000)
	if cram+footprint > 114844 {
		t.Fatalf("dense plan commits %d cache + %d footprint = %d MiB, over the %d MiB free RAM",
			cram, footprint, cram+footprint, 114844)
	}
	// The cache must not be sized as if the whole free RAM were available: a
	// dense plan with the footprint charged yields strictly less than the old
	// free-RAM rule.
	legacyCram, _ := computeCRAM(caps, model, &Strategy{Type: MultiGPUDense, Parallel: 1}, 27000, 2000)
	if legacyCram == 0 {
		t.Fatalf("legacy path disabled the cache entirely, test fixture broken")
	}
	if cram >= legacyCram {
		t.Fatalf("charged footprint should reduce the cache: got %d, legacy (uncharged) %d", cram, legacyCram)
	}
}

// A growth figure written before the post-serving gate cannot be told apart
// from a load-time allocation that was misfiled as growth. The one measured on
// this project was 5504 MiB on CUDA0 -- exactly that plan's
// `CUDA0 KV buffer size = 5504.00 MiB` -- and because re-probing carries growth
// forward, it was reserved on every later plan for the signature: four expert
// layers pushed to the CPU and decode roughly halved.
func TestGrowthRecordedBeforeTheServingGateIsRetired(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564}}
	model := &ModelProfile{Path: "v4.gguf", Basename: "v4.gguf", TotalSizeMB: 122268}
	header := "# Probe cache for v4.gguf\n# ctx=131072 ubatch=512 kv_quality=high kv_placement=gpu backend=llama@x gpu_sig=" +
		gpuSignatureHash(gpus) + " parallel=2\n"

	write := func(schema int, extra string) *probeCache {
		body := header + fmt.Sprintf("PROBE_CACHE_SCHEMA=%d\n", schema) + extra +
			"PROBED_COMPUTE_BUF_MB_CUDA0=1391\n" +
			"PROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA0=5504\n"
		path := probeCachePath(dir, model, 131072, 512, "high", "gpu", "llama@x", gpus, 2)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatalf("write probe: %v", err)
		}
		return loadProbeCache(dir, model, 131072, 512, "high", "gpu", "llama@x", gpus, 2)
	}

	// A file from before the conditions were recorded is discarded whole, which
	// takes the ungated growth with it.
	if stale := write(probeGrowthGateSchema-1, ""); stale != nil {
		t.Errorf("probe with no recorded measurement conditions was used: %+v", stale)
	}

	// With conditions recorded and unchanged, the file is usable and gated growth
	// survives. gpus carry no free-VRAM reading here, so nothing looks busier.
	current := write(probeCacheSchema, "PROBED_FREE_VRAM=\"0:24111\"\n")
	if current == nil {
		t.Fatal("current-schema probe measured on an idle machine must load")
	}
	if current.RuntimeGraphGrowthByGPU[0] != 5504 {
		t.Errorf("growth written through the gate must be kept, got %+v", current.RuntimeGraphGrowthByGPU)
	}
	if current.ComputeBufByGPU[0] != 1391 {
		t.Errorf("measured compute buffer lost, got %d", current.ComputeBufByGPU[0])
	}
}

// The mechanism behind the run that put 3 of 43 expert layers on GPU: a probe
// taken while the previous server was still releasing the cards recorded a
// CUDA2 compute buffer of 8641 MiB against a healthy 149 MiB, and the next
// launch believed it. 12282 total - 8641 compute - 359 overhead = 3282 MiB of
// room against a 3292 MiB expert layer, so the card got nothing -- short by
// 10 MiB. That plan was then cached in turn.
func TestProbeMeasuredWhileTheMachineWasBusyIsNotReused(t *testing.T) {
	gpus := func(free int) []detect.GPU {
		return []detect.GPU{{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12282, VRAMUsedMB: 12282 - free}}
	}
	const busyFree, idleFree = 2000, 11873

	if !probeMeasuredUnderDuress(map[int]int{2: busyFree}, gpus(idleFree)) {
		t.Error("a probe measured with 2000 MiB free must not be reused now that 11873 MiB is free")
	}
	if probeMeasuredUnderDuress(map[int]int{2: idleFree}, gpus(idleFree)) {
		t.Error("a probe measured under today's conditions must stay usable")
	}
	// Ordinary allocator jitter must not invalidate a good probe.
	if probeMeasuredUnderDuress(map[int]int{2: idleFree}, gpus(idleFree+planFreeVRAMSlackMB-1)) {
		t.Error("sub-slack drift invalidated a probe")
	}
	// The machine being busier now than at probe time is not duress: the probe
	// saw more headroom than exists today, and the launch preflight handles a
	// plan that asks for too much.
	if probeMeasuredUnderDuress(map[int]int{2: idleFree}, gpus(busyFree)) {
		t.Error("a probe from an idle machine was rejected merely because the machine is busy now")
	}
	// No recorded conditions, and a device with no entry, are both unjudgeable.
	if !probeMeasuredUnderDuress(nil, gpus(idleFree)) {
		t.Error("a probe with no recorded conditions must not be trusted")
	}
	if !probeMeasuredUnderDuress(map[int]int{0: idleFree}, gpus(idleFree)) {
		t.Error("a probe missing this device's conditions must not be trusted")
	}
}

// The planner takes no static margins, so plannedRAMRuntimeOverheadMB returns 0
// under RequireMeasuredBuffers. That is right about guessing and wrong about
// measuring: ~1.9 GiB of real host overhead sat outside every plan on this rig.
// Numbers below are from a live DeepSeek-V4 launch at --n-cpu-moe 37.
func TestHostOverheadIsMeasuredFromWhatTheMemoryScopeHolds(t *testing.T) {
	const (
		anonMB, shmemMB, slabMB = 1680, 91522, 212
		hostModelMB             = 91448 // CUDA_Host model buffer
		hostComputeMB           = 36    // CUDA_Host compute buffer
		hostOutputMB            = 1     // CUDA_Host output buffer
	)
	live := anonMB + shmemMB + slabMB
	accounted := hostModelMB + hostComputeMB + hostOutputMB

	got, ok := hostOverheadMB(live, accounted)
	if !ok {
		t.Fatalf("live=%d accounted=%d should yield a measurement", live, accounted)
	}
	if want := live - accounted; got != want {
		t.Errorf("host overhead %d MiB, want %d", got, want)
	}
	// llama-server's own VmRSS that moment was 93878 MiB. The cgroup reading
	// must agree with it closely or one of the two is measuring the wrong thing.
	if diff := 93878 - live; diff < 0 || diff > 93878/100 {
		t.Errorf("cgroup non-reclaimable %d MiB disagrees with VmRSS 93878 MiB by %d", live, diff)
	}

	// Page cache must never be charged: it was 15.5 GiB here and is reclaimable.
	if inflated, _ := hostOverheadMB(live+15872, accounted); inflated > 0 && inflated < 15872 {
		t.Errorf("including page cache changed the answer to %d MiB", inflated)
	}
	// Guards.
	if _, ok := hostOverheadMB(accounted-1, accounted); ok {
		t.Error("a reading below the declared buffers must be rejected, not negative")
	}
	if _, ok := hostOverheadMB(live, 0); ok {
		t.Error("no parsed host buffers means no measurement, not a full-size overhead")
	}
	if _, ok := hostOverheadMB(live, live/8); ok {
		t.Error("a delta near the whole reading means the buffers were not parsed; must be rejected")
	}
}

// The log accumulates across launch attempts inside one ggrun run: the first
// attempt reported an 86136 MiB host model buffer before --swa-full was
// withdrawn, the surviving one 91448 MiB. Summing them would invent 84 GiB.
func TestHostBufferParsingTakesMaximaNotSums(t *testing.T) {
	log := strings.Join([]string{
		"load_tensors:    CUDA_Host model buffer size = 86136.56 MiB",
		"load_tensors:        CUDA0 model buffer size = 11998.86 MiB",
		"load_tensors:    CUDA_Host model buffer size = 91448.56 MiB",
		"llama_context:  CUDA_Host  output buffer size =     0.99 MiB",
		"sched_reserve:  CUDA_Host compute buffer size =    36.32 MiB",
		"sched_reserve:      CUDA0 compute buffer size =  1391.00 MiB",
	}, "\n")

	got := parseHostBuffersFromLog(log)
	if want := 91448 + 36 + 1; got < want-2 || got > want+2 {
		t.Errorf("host buffers %d MiB, want ~%d (max model + compute + output)", got, want)
	}
	if got > 100000 {
		t.Errorf("host buffers %d MiB — the two load attempts were summed", got)
	}
}

func TestLoadSystemProbeDropsCurrentSchemaOutliers(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{
		{Index: 0, Name: "RTX 4070", VRAMTotalMB: 12282},
		{Index: 1, Name: "RTX 3090 Ti", VRAMTotalMB: 24564},
		{Index: 2, Name: "RTX 3060", VRAMTotalMB: 12288},
	}
	path := filepath.Join(dir, fmt.Sprintf("system_%s.cache", gpuIdentityHash(gpus)))
	// Schema 2 still latched a 4230 MiB "overhead" on CUDA1 after a DeepSeek
	// load. That is graph/KV, not CUDA context; charging it on the next launch
	// moved four expert layers onto the CPU.
	body := "SYS_PROBE_SCHEMA=2\nSYS_CUDA_OVERHEAD_MB_CUDA0=1614\n" +
		"SYS_CUDA_OVERHEAD_MB_CUDA1=4230\nSYS_CUDA_OVERHEAD_MB_CUDA2=568\nSYS_CUDA_OVERHEAD_MB=4230\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	sp := loadSystemProbe(dir, gpus)
	if sp == nil {
		t.Fatal("system probe missing")
	}
	if _, ok := sp.CUDAOverheadByGPU[1]; ok {
		t.Fatalf("CUDA1 outlier %d was trusted as CUDA overhead", sp.CUDAOverheadByGPU[1])
	}
	if sp.CUDAOverheadByGPU[0] != 1614 || sp.CUDAOverheadByGPU[2] != 568 {
		t.Fatalf("peer overheads were discarded: %+v", sp.CUDAOverheadByGPU)
	}
}

func TestStableAutoContextDoesNotInflateClaudeBatchGraph(t *testing.T) {
	opts := Options{
		RuntimePolicy: func(s *Strategy) {
			s.BatchSize = 128
			s.UBatchSize = 64
		},
	}
	got := stableAutoContextOptions(opts)
	if got.BatchSize != 0 || got.UBatchSize != 0 {
		t.Fatalf("auto-fit search overrode the serving graph with %d/%d", got.BatchSize, got.UBatchSize)
	}
	if got.RuntimePolicy == nil {
		t.Fatal("runtime policy was stripped from auto-fit search")
	}
}

// The host term is measured on its own schedule, but needing it must not cause

// the cards to be measured again. A whole-device delta charges everything it
// cannot attribute to "CUDA overhead", and what it cannot attribute depends on
// the launch: a single-GPU Qwen3.6-27B at -b 8192 -ub 1024 wrote CUDA0=3850 MiB
// over a DeepSeek-V4 measurement of 1327, and applying that back to V4 cost
// CUDA0 its only expert layer (9 GPU layers -> 8, n-cpu-moe 34 -> 35).
func TestHostOverheadProbeDoesNotRemeasureAlreadyMeasuredCards(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{
		{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564},
		{Index: 1, Name: "RTX 3060", VRAMTotalMB: 12288},
		{Index: 2, Name: "RTX 4070", VRAMTotalMB: 12282},
	}
	path := filepath.Join(dir, fmt.Sprintf("system_%s.cache", gpuIdentityHash(gpus)))
	good := "SYS_PROBE_SCHEMA=2\nSYS_CUDA_OVERHEAD_MB_CUDA0=1327\n" +
		"SYS_CUDA_OVERHEAD_MB_CUDA1=301\nSYS_CUDA_OVERHEAD_MB_CUDA2=359\nSYS_CUDA_OVERHEAD_MB=1327\n"
	if err := os.WriteFile(path, []byte(good), 0o644); err != nil {
		t.Fatal(err)
	}

	// A log that would make the whole-device source produce a different, larger
	// answer for CUDA0 if it were allowed to run.
	log := "load_tensors:        CUDA0 model buffer size = 11000.00 MiB\n" +
		"sched_reserve:      CUDA0 compute buffer size =  1000.00 MiB\n"
	RunPostLaunchProbe(dir, gpus, log, 0, nil)

	sp := loadSystemProbe(dir, gpus)
	if sp == nil {
		t.Fatal("system probe disappeared")
	}
	for idx, want := range map[int]int{0: 1327, 1: 301, 2: 359} {
		if got := sp.CUDAOverheadByGPU[idx]; got != want {
			t.Errorf("CUDA%d overhead was re-measured to %d, want the existing %d", idx, got, want)
		}
	}
}

// mmap makes CPU-side experts reclaimable only where the loader actually maps
// them. Under ik_llama they land in anonymous memory regardless of the flag:
// MiniMax-M3's mmap and resident rows had identical cgroups (anon 113363 MB,
// file 449 MB) on 2026-08-07. Charging only the anonymous working set there
// predicted a 458 MB host footprint against a real 113 GB.
func TestHostFootprintChargesFullResidentWhenBackendDoesNotMap(t *testing.T) {
	const resident = 111130 // MiniMax-M3's CPU experts plus overhead
	const workingSet = 458  // what mmap would appear to cost

	anonymous := Options{CPUExpertMMapCapability: CPUExpertMMapAnonymous}
	fileBacked := Options{CPUExpertMMapCapability: CPUExpertMMapFileBacked}
	if got := hostFootprintForCache(resident, workingSet, true, anonymous); got != resident {
		t.Fatalf("ik_llama does not map CPU experts, so the plan must charge the full %d MB, got %d", resident, got)
	}
	// Mainline does map them (DeepSeek-V4: anon 1.9 GB, file 113 GB), so the
	// reclaimable bytes must not be charged or the prompt cache is disabled on
	// exactly the hosts that page happily.
	if got := hostFootprintForCache(resident, workingSet, true, fileBacked); got != workingSet {
		t.Fatalf("mainline maps CPU experts, so only the working set is charged: want %d, got %d", workingSet, got)
	}
	// An explicitly unknown exact backend is never granted reclaimable capacity.
	if got := hostFootprintForCache(resident, workingSet, true, Options{CPUExpertMMapCapability: CPUExpertMMapUnknown}); got != resident {
		t.Fatalf("unknown backend capability must fail closed, got %d", got)
	}

	// Without mmap the backend is irrelevant: the bytes are resident either way.
	for _, tag := range []string{"llama", "ik_llama", ""} {
		if got := hostFootprintForCache(resident, workingSet, false, Options{BackendTag: tag}); got != resident {
			t.Fatalf("resident plan on %q must charge %d, got %d", tag, resident, got)
		}
	}
}

// An ik_llama-derived fork routes to placement under its recipe tag (hy3),
// not "ik_llama" (detectRegisteredBackend sets info.Tag = cb.Tag). The mmap
// band and the working-set-only cache charge must not be granted to a loader
// that still allocates CPU experts in anonymous CUDA-host memory.
func TestHostFootprintChargesFullResidentForIKDerivedForkTags(t *testing.T) {
	const resident = 111130 // MiniMax-M3 CPU experts plus overhead
	const workingSet = 458  // what mmap would appear to cost
	for _, tag := range []string{"hy3", "ik", "ik_llama-fork"} {
		if got := hostFootprintForCache(resident, workingSet, true, Options{BackendTag: tag}); got != resident {
			t.Errorf("%q is an ik-derived loader: mmap must charge the full %d MB, got %d", tag, resident, got)
		}
	}
	// Mainline still maps (DeepSeek-V4 measured anon 1.9GB / file 113GB): the
	// reclaimable bytes must not be charged or the cache is disabled there.
	if got := hostFootprintForCache(resident, workingSet, true, Options{CPUExpertMMapCapability: CPUExpertMMapFileBacked}); got != workingSet {
		t.Errorf("mainline maps CPU experts: want working set %d, got %d", workingSet, got)
	}
}

// And an Options-level assertion that the band predicate agrees.
func TestMMapBandDeniedForIKDerivedFork(t *testing.T) {
	if mmapCanPageCPUExperts(Options{BackendTag: "hy3"}) {
		t.Error("hy3 uses the anonymous CUDA-host loader: mmap must not be considered reclaimable")
	}
	if !mmapCanPageCPUExperts(Options{BackendTag: "llama"}) {
		t.Error("mainline maps CPU experts: mmap reclaimability must be kept")
	}
	if mmapCanPageCPUExperts(Options{BackendTag: "llama", CPUExpertMMapCapability: CPUExpertMMapUnknown}) {
		t.Error("an explicit unknown exact-backend capability must override a friendly tag")
	}
}

// --- Dense auto-context reduction (dense models must fit on GPU) ---

// denseReduceCaps is the 24/12/12 GB test rig (CUDA1 partially occupied, mirroring
// the real box) with a combined ~41.5 GB of free VRAM.
func denseReduceCaps() *detect.Capabilities {
	return &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576, VRAMUsedMB: 736},
			{Index: 1, VRAMTotalMB: 12288, VRAMUsedMB: 6329},
			{Index: 2, VRAMTotalMB: 12288, VRAMUsedMB: 578},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 100000},
		CPU: detect.CPUInfo{Cores: 16},
	}
}

// denseReduceModel builds a dense model whose KV geometry is expensive enough that
// an auto (RAM-pool-sized) context overflows GPU memory while a smaller context
// fits. free = 41509 MiB; model + 3*1024 overhead = 33876 MiB leaves ~7633 MiB of
// KV headroom, so ctx=65536 (KV ~4250 MiB) fits but ctx=131072 (KV ~8500 MiB)
// does not.
func denseReduceModel() *ModelProfile {
	bpc := 8000.0 * 1024 * 1024 / (52.0 * 65536)
	return &ModelProfile{
		Path:        "DenseBig.gguf",
		TotalSizeMB: 30804,
		SizeBytes:   30804 << 20,
		NumLayers:   53,
		HiddenSize:  5120,
		CTXTrain:    1048576,
		MeasuredKVGeometry: map[string]KVGeometry{
			"f16": {FullLayers: 52, SWALayers: 0, BytesPerCellPerLayer: bpc},
		},
	}
}

// TestComputeDenseAutoReduceFitsGPU verifies that a dense model whose auto context
// would spill to host RAM is reduced to a context that fits entirely on GPU.
func TestComputeDenseAutoReduceFitsGPU(t *testing.T) {
	caps := denseReduceCaps()
	model := denseReduceModel()
	strat, err := Compute(caps, model, Options{KVPlacement: "auto", KVQuality: "mid", SWAFull: true, CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if strat.Type != MultiGPUDense {
		t.Fatalf("expected multi_gpu_dense (dense model must fit on GPU after auto ctx reduction), got %s", strat.Type)
	}
	kv := computeKVTotalMB(model, strat.ContextSize, strat.KVType, true)
	// model + 3 * computeFloor overhead + KV must fit in free VRAM.
	if 30804+3*computeFloorMB+kv > caps.GPUs[0].VRAMFreeMB()+caps.GPUs[1].VRAMFreeMB()+caps.GPUs[2].VRAMFreeMB() {
		t.Fatalf("reduced ctx=%d still does not fit on GPU: model+overhead+kv = %d > free", strat.ContextSize, 30804+3*computeFloorMB+kv)
	}
}

// TestComputeDenseAutoReducePicksLargestFit verifies the reduction chooses the
// LARGEST context rung that still fits entirely on GPU (reduces as little as
// needed), not the smallest.
func TestComputeDenseAutoReducePicksLargestFit(t *testing.T) {
	caps := denseReduceCaps()
	model := denseReduceModel()
	strat, err := Compute(caps, model, Options{KVPlacement: "auto", KVQuality: "mid", SWAFull: true, CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	// Auto-sizing takes the largest window the VRAM holds, not the largest power
	// of two below it: the old ladder answered 65536 here where ~116k fits, so
	// nearly half the window was discarded. Assert the property rather than a
	// constant -- the chosen context must fit, and one granule more must not.
	free := caps.GPUs[0].VRAMFreeMB() + caps.GPUs[1].VRAMFreeMB() + caps.GPUs[2].VRAMFreeMB()
	fixed := 30804 + 3*computeFloorMB
	kv := computeKVTotalMB(model, strat.ContextSize, strat.KVType, true)
	if fixed+kv > free {
		t.Fatalf("ctx=%d does not fit: model+overhead+kv = %d > %d free", strat.ContextSize, fixed+kv, free)
	}
	if strat.ContextSize <= 65536 {
		t.Fatalf("ctx=%d is no better than the old power-of-two rung; the window is not being maximised", strat.ContextSize)
	}
	nextKV := computeKVTotalMB(model, strat.ContextSize+contextGranularity, strat.KVType, true)
	if fixed+nextKV <= free {
		t.Fatalf("ctx=%d is not maximal: %d also fits", strat.ContextSize, strat.ContextSize+contextGranularity)
	}
}

// TestComputeDenseAutoReduceDeclinedFailsHard verifies that when a dense model
// cannot fit on GPU even at minimum context and the user declines host offload,
// Compute returns a hard error instead of silently launching into system RAM.
func TestComputeDenseAutoReduceDeclinedFailsHard(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576, VRAMUsedMB: 736},
			{Index: 1, VRAMTotalMB: 12288, VRAMUsedMB: 6329},
			{Index: 2, VRAMTotalMB: 12288, VRAMUsedMB: 578},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 100000},
		CPU: detect.CPUInfo{Cores: 16},
	}
	// 40 GB dense on ~41.5 GB free: even at 32768 ctx the model weights +
	// 3*1024 overhead already approach the pool, and the compute-buffer probe
	// reserve pushes it over at every context.
	model := &ModelProfile{
		Path: "DenseTooBig.gguf", TotalSizeMB: 40000, SizeBytes: 40000 << 20, NumLayers: 53, HiddenSize: 5120, CTXTrain: 1048576,
		MeasuredKVGeometry: map[string]KVGeometry{"f16": {FullLayers: 52, SWALayers: 0, BytesPerCellPerLayer: 64}},
	}
	dir := t.TempDir()
	for _, ctx := range []int{32768, 65536, 131072, 262144, 524288} {
		WriteProbeCacheForModel(dir, model, ctx, 512, "mid", "gpu", "llama", caps.GPUs, map[int]int{0: 4000, 1: 2000, 2: 2000}, 0)
	}
	declined := false
	_, err := Compute(caps, model, Options{
		KVPlacement: "gpu", KVQuality: "mid", CacheDir: dir,
		RequireMeasuredBuffers: true, BatchSize: 512, UBatchSize: 512,
		DenseCPUOffloadPrompt: func(totalSizeMB, freeGPUVRAMMB int) bool {
			declined = true
			return false
		},
	})
	if err == nil {
		t.Fatalf("expected a hard error when the user declines host offload for a model too big for GPU")
	}
	if !declined {
		t.Fatalf("expected the DenseCPUOffloadPrompt hook to be consulted")
	}
}

// TestComputeDenseAutoReduceAcceptedOffload verifies that when a dense model
// cannot fit on GPU at any context but the user accepts host offload (or the
// caller is non-terminal / assume-yes), today's DenseCPUOffload behavior is
// preserved.
func TestComputeDenseAutoReduceAcceptedOffload(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576, VRAMUsedMB: 736},
			{Index: 1, VRAMTotalMB: 12288, VRAMUsedMB: 6329},
			{Index: 2, VRAMTotalMB: 12288, VRAMUsedMB: 578},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 100000},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path: "DenseTooBig.gguf", TotalSizeMB: 40000, SizeBytes: 40000 << 20, NumLayers: 53, HiddenSize: 5120, CTXTrain: 1048576,
		MeasuredKVGeometry: map[string]KVGeometry{"f16": {FullLayers: 52, SWALayers: 0, BytesPerCellPerLayer: 64}},
	}
	dir := t.TempDir()
	for _, ctx := range []int{32768, 65536, 131072, 262144, 524288} {
		WriteProbeCacheForModel(dir, model, ctx, 512, "mid", "gpu", "llama", caps.GPUs, map[int]int{0: 4000, 1: 2000, 2: 2000}, 0)
	}
	accepted := false
	strat, err := Compute(caps, model, Options{
		KVPlacement: "gpu", KVQuality: "mid", CacheDir: dir,
		RequireMeasuredBuffers: true, BatchSize: 512, UBatchSize: 512,
		DenseCPUOffloadPrompt: func(totalSizeMB, freeGPUVRAMMB int) bool {
			accepted = true
			return true
		},
	})
	if err != nil {
		t.Fatalf("accepted host offload should compute: %v", err)
	}
	if !accepted {
		t.Fatalf("expected the prompt hook to fire")
	}
	if strat.Type != DenseCPUOffload {
		t.Fatalf("expected dense_cpu_offload when the user accepts host offload, got %s", strat.Type)
	}
	// No prompt (nil hook) = non-terminal/assume-yes default: silent host offload.
	strat2, err := Compute(caps, model, Options{
		KVPlacement: "gpu", KVQuality: "mid", CacheDir: dir,
		RequireMeasuredBuffers: true, BatchSize: 512, UBatchSize: 512,
	})
	if err != nil {
		t.Fatalf("nil-prompt host offload should compute: %v", err)
	}
	if strat2.Type != DenseCPUOffload {
		t.Fatalf("expected silent dense_cpu_offload without a prompt hook, got %s", strat2.Type)
	}
}

// TestComputeDenseAutoReduceHonorsExplicitCtx verifies that an explicit --ctx-size
// is a user choice and is honored unchanged: the auto-reduction must not fire.
func TestComputeDenseAutoReduceHonorsExplicitCtx(t *testing.T) {
	caps := denseReduceCaps()
	model := denseReduceModel()
	dir := t.TempDir()
	// The reduction would reduce 262144 to 65536; an explicit 262144 must survive.
	for _, ctx := range []int{32768, 65536, 131072, 262144, 524288} {
		WriteProbeCacheForModel(dir, model, ctx, 512, "mid", "gpu", "llama", caps.GPUs, map[int]int{0: 4000, 1: 2000, 2: 2000}, 0)
	}
	strat, err := Compute(caps, model, Options{
		ContextSize: 262144, KVPlacement: "gpu", KVQuality: "mid", CacheDir: dir,
		RequireMeasuredBuffers: true, BatchSize: 512, UBatchSize: 512,
		DenseCPUOffloadPrompt: func(totalSizeMB, freeGPUVRAMMB int) bool {
			t.Fatalf("prompt must not fire for an explicit --ctx-size")
			return false
		},
	})
	if err != nil {
		t.Fatalf("explicit ctx compute failed: %v", err)
	}
	if strat.ContextSize != 262144 {
		t.Fatalf("explicit ctx=262144 must be honored unchanged, got %d", strat.ContextSize)
	}
}

// TestComputeDenseAutoReduceMoEUnchanged verifies the MoE path is untouched: MoE
// models keep their existing MoEOffload strategy and never go through the dense
// reduction (which is gated on !IsMoE).
func TestComputeDenseAutoReduceMoEUnchanged(t *testing.T) {
	caps := denseReduceCaps()
	model := &ModelProfile{
		Path: "MoE.gguf", TotalSizeMB: 30000, SizeBytes: 30000 << 20, NumLayers: 32, HiddenSize: 4096, CTXTrain: 1048576,
		HeadCountKV: 4, KeyLength: 128, ValueLength: 128,
		IsMoE: true, NumExperts: 64, ExpertUsedCount: 6, ExpertFF: 4096,
		NonExpertBytes: 10 << 30,
	}
	strat, err := Compute(caps, model, Options{KVPlacement: "auto", KVQuality: "mid", CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("MoE compute failed: %v", err)
	}
	if strat.Type != MoEOffload {
		t.Fatalf("expected MoE to keep moe_offload, got %s", strat.Type)
	}
}

// TestComputeDenseAutoReduceSingleGPUUnchanged verifies the single-GPU path is
// untouched: a dense model that fits on one GPU stays SingleGPU.
func TestComputeDenseAutoReduceSingleGPUUnchanged(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576, VRAMUsedMB: 736},
		},
		RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path: "DenseSmall.gguf", TotalSizeMB: 20000, SizeBytes: 20000 << 20, NumLayers: 32, HiddenSize: 4096, CTXTrain: 1048576,
		HeadCountKV: 4, KeyLength: 128, ValueLength: 128,
	}
	strat, err := Compute(caps, model, Options{KVPlacement: "gpu", KVQuality: "mid", CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("single-GPU compute failed: %v", err)
	}
	if strat.Type != SingleGPU {
		t.Fatalf("expected single_gpu, got %s", strat.Type)
	}
}

// TestStrategyVRAMHeadroomMB verifies the conservative spare-VRAM estimator the
// proactive reviewer-drop gate relies on: dense plans report positive headroom
// when they fit comfortably, and return 0 for offload/CPU-only plans that are
// not fully GPU-resident (or for a plan that does not fit).
func TestStrategyVRAMHeadroomMB(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24564, VRAMUsedMB: 500, BandwidthMBps: 15754},
	}}
	model := &ModelProfile{Path: "/models/small.gguf", TotalSizeMB: 2741, SizeBytes: 2741 << 20, NumLayers: 36}

	// A small dense model on a 24 GB card: used = 2741 + compute floor (1024) +
	// KV estimate + CUDA overhead (0 when unchached) = ~4 GB, while the card has
	// ~24 GB free — the headroom must be large and comfortably positive.
	single := &Strategy{Type: SingleGPU, MainGPU: 0, ContextSize: 65536, KVType: "q8_0", KVPlacement: "gpu"}
	if h := StrategyVRAMHeadroomMB(caps, model, single, ""); h < 3*1024 {
		t.Fatalf("single-GPU headroom = %d, want >= 3 GiB for a comfortably fitting small model", h)
	}

	// A model that cannot fit at all (weights alone exceed the GPU) reports 0.
	big := &ModelProfile{Path: "/models/big.gguf", TotalSizeMB: 30000, SizeBytes: 30000 << 20, NumLayers: 64}
	if h := StrategyVRAMHeadroomMB(caps, big, &Strategy{Type: SingleGPU, MainGPU: 0, ContextSize: 8192, KVType: "q8_0", KVPlacement: "gpu"}, ""); h != 0 {
		t.Fatalf("over-capacity single-GPU headroom = %d, want 0", h)
	}

	// Offload/CPU-only plans are never priced: the proactive gate must treat them
	// as having no spare VRAM so it never drops a companion from one.
	offload := &Strategy{Type: MoEOffload, IsMoE: true, ContextSize: 65536, KVType: "q8_0"}
	if h := StrategyVRAMHeadroomMB(caps, model, offload, ""); h != 0 {
		t.Fatalf("MoE offload headroom = %d, want 0", h)
	}
	if h := StrategyVRAMHeadroomMB(caps, model, &Strategy{Type: CPUOnly}, ""); h != 0 {
		t.Fatalf("CPUOnly headroom = %d, want 0", h)
	}
	if h := StrategyVRAMHeadroomMB(caps, model, &Strategy{Type: DenseCPUOffload}, ""); h != 0 {
		t.Fatalf("DenseCPUOffload headroom = %d, want 0", h)
	}
}
