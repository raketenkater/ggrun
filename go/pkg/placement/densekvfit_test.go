package placement

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// partlyOccupiedGPUs returns n 24 GB cards with only freeMB left each.
func partlyOccupiedGPUs(n, freeMB int) []detect.GPU {
	gpus := make([]detect.GPU, n)
	for i := range gpus {
		gpus[i] = detect.GPU{Index: i, VRAMTotalMB: 24000, VRAMUsedMB: 24000 - freeMB, BandwidthMBps: 16000}
	}
	return gpus
}

func assertNoNegativeDeviceSlack(t *testing.T, caps *detect.Capabilities, model *ModelProfile, s *Strategy, opts Options) {
	t.Helper()
	ledger := BuildResourceLedger(caps, model, s, opts)
	if len(ledger.Devices) == 0 {
		t.Fatalf("ledger has no devices: %+v", ledger)
	}
	for _, device := range ledger.Devices {
		if device.SlackMB < 0 {
			t.Fatalf("accepted plan has negative slack on CUDA%d: %+v (ctx %d, KV %s)",
				device.GPU, device, s.ContextSize, s.KVPlacement)
		}
	}
}

// A 30 GB dense model on two cards with 6 GB free each fits only by CPU
// offload. The pooled check admits the native 262,144 context (16 GiB of KV
// on the GPUs); each card's ledger does not. The search must stop where every
// card's KV share still fits.
func TestAutomaticContextDenseOffloadKVFitsEachGPU(t *testing.T) {
	caps := strategyFitCaps(partlyOccupiedGPUs(2, 6000)...)
	model := strategyFitDenseModel(30000, 262144)
	opts := strategyFitOptions()
	opts.CacheDir = measuredOverheadCache(t, caps.GPUs, 300)

	got, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("dense offload fit: %v", err)
	}
	if got.Type != DenseCPUOffload || got.KVPlacement != "gpu" {
		t.Fatalf("plan = %s with KV on %s, want dense CPU offload with KV on gpu", got.Type, got.KVPlacement)
	}
	if got.ContextSize >= 262144 {
		t.Fatalf("context %d kept the pooled-check boundary", got.ContextSize)
	}
	assertNoNegativeDeviceSlack(t, caps, model, got, opts)
	if !strings.Contains(got.ContextFitEvidence, "KV does not fit") {
		t.Fatalf("boundary evidence does not name the per-GPU KV limit: %q", got.ContextFitEvidence)
	}
}

// With no room for KV on any card even at the context floor, automatic KV
// placement moves KV to the host instead of accepting an overfull card.
func TestAutomaticContextDenseOffloadMovesKVToHostWhenNoGPUFits(t *testing.T) {
	caps := strategyFitCaps(partlyOccupiedGPUs(2, 3500)...)
	model := strategyFitDenseModel(30000, 262144)
	opts := strategyFitOptions()
	opts.KVPlacement = ""
	opts.CacheDir = measuredOverheadCache(t, caps.GPUs, 300)

	got, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("dense offload fit: %v", err)
	}
	if got.Type != DenseCPUOffload || got.KVPlacement != "cpu" {
		t.Fatalf("plan = %s with KV on %s, want dense CPU offload with KV on cpu", got.Type, got.KVPlacement)
	}
	assertNoNegativeDeviceSlack(t, caps, model, got, opts)
}

// KV pinned to the GPUs that cannot hold it is a refusal, not a plan.
func TestAutomaticContextDenseOffloadRefusesGPUKVThatCannotFit(t *testing.T) {
	caps := strategyFitCaps(partlyOccupiedGPUs(2, 3500)...)
	model := strategyFitDenseModel(30000, 262144)
	opts := strategyFitOptions()
	opts.CacheDir = measuredOverheadCache(t, caps.GPUs, 300)

	got, err := Compute(caps, model, opts)
	if err == nil {
		t.Fatalf("accepted %s ctx %d with KV on %s; want a refusal", got.Type, got.ContextSize, got.KVPlacement)
	}
	if !strings.Contains(err.Error(), "KV does not fit") {
		t.Fatalf("refusal does not name the per-GPU KV limit: %v", err)
	}
}

// An explicit context is still planned (exact admission decides), and its
// ledger reports the overfull cards that dry run refuses.
func TestExplicitContextDenseOffloadLedgerReportsKVOverflow(t *testing.T) {
	caps := strategyFitCaps(partlyOccupiedGPUs(2, 6000)...)
	model := strategyFitDenseModel(30000, 262144)
	opts := strategyFitOptions()
	opts.CacheDir = measuredOverheadCache(t, caps.GPUs, 300)
	opts.ContextSize = 262144

	got, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("explicit dense offload: %v", err)
	}
	ledger := BuildResourceLedger(caps, model, got, opts)
	over := DenseOffloadKVOverflow(got, &ledger)
	if len(over) == 0 {
		t.Fatalf("16 GiB of GPU KV on two 6 GB cards reported no overflow: %+v", ledger.Devices)
	}
	if text := FormatKVOverflow(over); !strings.Contains(text, "CUDA") || !strings.Contains(text, "slack -") {
		t.Fatalf("overflow text = %q", text)
	}
}

func TestDenseOffloadKVOverflowScope(t *testing.T) {
	ledger := &ResourceLedger{Devices: []DeviceResourceLedger{
		{GPU: 0, ContextMB: 4000, SlackMB: -10},
		{GPU: 1, ContextMB: 0, SlackMB: -50},
		{GPU: 2, ContextMB: 4000, SlackMB: 0},
	}}
	over := DenseOffloadKVOverflow(&Strategy{Type: DenseCPUOffload}, ledger)
	if len(over) != 1 || over[0].GPU != 0 {
		t.Fatalf("overflow = %+v, want only CUDA0 (carries KV, negative slack)", over)
	}
	for _, s := range []*Strategy{nil, {Type: MoEOffload}, {Type: MultiGPUDense}, {Type: CPUOnly}} {
		if got := DenseOffloadKVOverflow(s, ledger); got != nil {
			t.Fatalf("non-dense-offload plan %+v reported overflow %+v", s, got)
		}
	}
	if got := DenseOffloadKVOverflow(&Strategy{Type: DenseCPUOffload}, nil); got != nil {
		t.Fatalf("missing ledger reported overflow %+v", got)
	}
}

func TestCPUOnlyPlanLabelsNoGPUBottleneck(t *testing.T) {
	gpus := []detect.GPU{
		{Index: 0, VRAMTotalMB: 24000, BandwidthMBps: 16000},
		{Index: 1, VRAMTotalMB: 12000, BandwidthMBps: 2000},
	}
	caps := &detect.Capabilities{
		GPUs: gpus, RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 120000},
		CPU: detect.CPUInfo{Cores: 16}, HostMemoryBandwidthMBps: 40000,
	}
	model := strategyFitDenseModel(8000, 32768)
	opts := Options{CacheDir: measuredOverheadCache(t, gpus, 300)}
	base := &Strategy{
		Type: CPUOnly, ContextSize: 8192, Parallel: 1, BatchSize: 2048, UBatchSize: 512,
		KVPlacement: "cpu", KVQuality: "mid", KVType: "q8_0",
	}
	frontier := AnalyzeCandidateFrontier(caps, model, opts, []CalibrationCandidate{{Name: "default", Strategy: base}})
	if base.Residency != ResidencyCPUResident {
		t.Fatalf("CPU-only plan residency = %q, want %q", base.Residency, ResidencyCPUResident)
	}
	if base.OptimizationBottleneck != "CPU layer service" || strings.Contains(base.OptimizationBottleneck, "GPU") {
		t.Fatalf("CPU-only bottleneck = %q", base.OptimizationBottleneck)
	}
	if frontier[0].Estimate.ActiveGPUs != 0 {
		t.Fatalf("CPU-only plan counted %d active GPUs", frontier[0].Estimate.ActiveGPUs)
	}

	paged := cloneStrategy(base)
	paged.Residency = ""
	paged.MMapRequired = true
	AnalyzeCandidateFrontier(caps, model, opts, []CalibrationCandidate{{Name: "default", Strategy: paged}})
	if paged.Residency != ResidencyNonResident {
		t.Fatalf("mmap-required CPU-only residency = %q, want %q", paged.Residency, ResidencyNonResident)
	}
}

func TestSingleGPUPlanIgnoresIdleDeviceForBottleneck(t *testing.T) {
	gpus := []detect.GPU{
		{Index: 0, VRAMTotalMB: 24000, BandwidthMBps: 16000},
		{Index: 1, VRAMTotalMB: 12000, BandwidthMBps: 2000},
	}
	caps := &detect.Capabilities{
		GPUs: gpus, RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 60000}, CPU: detect.CPUInfo{Cores: 16},
	}
	model := strategyFitDenseModel(8000, 32768)
	opts := Options{CacheDir: measuredOverheadCache(t, gpus, 300)}
	s := &Strategy{
		Type: SingleGPU, MainGPU: 0, ContextSize: 8192, Parallel: 1, BatchSize: 2048, UBatchSize: 512,
		KVPlacement: "cpu", KVQuality: "mid", KVType: "q8_0",
	}
	est := AnalyzeStrategy(caps, model, s, opts)
	if est.ActiveGPUs != 1 || est.Bottleneck != "GPU 0 layer service" {
		t.Fatalf("single-GPU estimate: %d active GPUs, bottleneck %q; want 1 and GPU 0", est.ActiveGPUs, est.Bottleneck)
	}
}

func batchRuleOptions(t *testing.T, caps *detect.Capabilities) Options {
	t.Helper()
	return Options{
		KVPlacement: "gpu", KVQuality: "mid", SkipCachedConfig: true,
		CacheDir: measuredOverheadCache(t, caps.GPUs, 300),
	}
}

// The automatic context is searched on a pinned batch pair, but the chosen
// point gets the same batch rule as the same context typed explicitly.
func TestAutomaticContextUsesTheExplicitContextBatchRule(t *testing.T) {
	caps := strategyFitCaps(detect.GPU{Index: 0, VRAMTotalMB: 24000})
	model := strategyFitDenseModel(2000, 32768)
	opts := batchRuleOptions(t, caps)

	auto, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("automatic context: %v", err)
	}
	explicitOpts := opts
	explicitOpts.ContextSize = auto.ContextSize
	explicit, err := Compute(caps, model, explicitOpts)
	if err != nil {
		t.Fatalf("explicit context: %v", err)
	}
	if explicit.UBatchSize != 1024 {
		t.Fatalf("roomy explicit plan took batch %d/%d; the test needs the large tier", explicit.BatchSize, explicit.UBatchSize)
	}
	if auto.BatchSize != explicit.BatchSize || auto.UBatchSize != explicit.UBatchSize {
		t.Fatalf("ctx %d: automatic batch %d/%d, explicit %d/%d",
			auto.ContextSize, auto.BatchSize, auto.UBatchSize, explicit.BatchSize, explicit.UBatchSize)
	}
}

func TestAutomaticBatchNeverExceedsContext(t *testing.T) {
	caps := strategyFitCaps(detect.GPU{Index: 0, VRAMTotalMB: 24000})
	model := strategyFitDenseModel(2000, 32768)
	opts := batchRuleOptions(t, caps)
	opts.ContextSize = 4096

	got, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("small context: %v", err)
	}
	if got.BatchSize > 4096 || got.UBatchSize > got.BatchSize {
		t.Fatalf("ctx 4096 planned batch %d/%d", got.BatchSize, got.UBatchSize)
	}

	// An explicit batch is the user's constraint and is not capped.
	opts.BatchSize, opts.BatchSizeExplicit = 8192, true
	got, err = Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("explicit batch: %v", err)
	}
	if got.BatchSize != 8192 {
		t.Fatalf("explicit batch 8192 became %d", got.BatchSize)
	}
}
