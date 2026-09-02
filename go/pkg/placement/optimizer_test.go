package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func optimizerTestGPUs() []detect.GPU {
	return []detect.GPU{
		{Index: 0, Name: "fast", VRAMTotalMB: 32768, MemBandwidthMBps: 1_000_000, BandwidthMBps: 24000},
		{Index: 1, Name: "slow", VRAMTotalMB: 32768, MemBandwidthMBps: 100_000, BandwidthMBps: 8000},
	}
}

func TestMeasuredAllocationMustMatchTensorDistribution(t *testing.T) {
	gpus := optimizerTestGPUs()
	model := &ModelProfile{TotalSizeMB: 10000, SizeBytes: 10000 * 1024 * 1024}
	strategy := &Strategy{Type: MultiGPUDense, TensorSplit: []float64{0.8, 0.2}, KVPlacement: "cpu"}
	matching := MeasuredAllocation{
		ContextTotalMB: 100, ContextHostMB: 100,
		ModelByGPU: map[int]int{0: 8000, 1: 2000},
	}
	if !measuredAllocationMatchesStrategy(matching, model, strategy, gpus) {
		t.Fatal("matching allocation distribution was rejected")
	}
	reversed := matching
	reversed.ModelByGPU = map[int]int{0: 2000, 1: 8000}
	if measuredAllocationMatchesStrategy(reversed, model, strategy, gpus) {
		t.Fatal("allocation from the reversed tensor split became exact evidence")
	}
}

func TestMeasuredAllocationIdentityAcceptsUnlabelledGuardPeakOnlyForExactPlacement(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	model := &ModelProfile{Path: modelPath, TotalSizeMB: 10000, SizeBytes: 10000 * 1024 * 1024}
	gpus := optimizerTestGPUs()
	caps := &detect.Capabilities{
		GPUs: gpus,
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
	}
	strategy := &Strategy{
		Type: MultiGPUDense, TensorSplit: []float64{0.8, 0.2}, SplitMode: "layer", MainGPU: 0,
		ContextSize: 32768, Parallel: 1, BatchSize: 2048, UBatchSize: 512,
		KVPlacement: "gpu", KVQuality: "high", KVType: "q8_0", FlashAttention: true,
		PlanFreeVRAM: map[int]int{0: 32768, 1: 32768},
	}
	allocation := MeasuredAllocation{
		Evidence:          "allocation-verified",
		PlacementIdentity: AllocationPlacementIdentity(strategy),
		ContextTotalMB:    768,
		ContextByGPU:      map[int]int{0: 512, 1: 256},
		// The guard observed exact peaks, but this backend did not label model
		// buffers. The aggregate therefore lives in unaccounted rows.
		UnaccountedByGPU:  map[int]int{0: 9000, 1: 3000},
		UnaccountedHostMB: 2048,
	}
	if err := RecordMeasuredAllocation(dir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, "test", gpus, strategy.Parallel, allocation); err != nil {
		t.Fatal(err)
	}
	loaded, ok := LoadMeasuredAllocation(dir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, "test", gpus, strategy.Parallel)
	if !ok || loaded.PlacementIdentity != allocation.PlacementIdentity {
		t.Fatalf("allocation identity did not round-trip: ok=%t loaded=%+v", ok, loaded)
	}
	ledger := BuildResourceLedger(caps, model, strategy, Options{CacheDir: dir, BackendCacheTag: "test"})
	if !ledger.Exact || !ledger.Fits || ledger.Devices[0].RequiredMB != 9512 || ledger.Devices[1].RequiredMB != 3256 {
		t.Fatalf("matching guarded allocation was not exact: %+v", ledger)
	}

	different := cloneStrategy(strategy)
	different.TensorSplit = []float64{0.2, 0.8}
	ledger = BuildResourceLedger(caps, model, different, Options{CacheDir: dir, BackendCacheTag: "test"})
	if ledger.Exact {
		t.Fatalf("different placement reused exact guarded allocation: %+v", ledger)
	}
}

func TestPostLaunchContextObservationPreservesMatchingGuardedBreakdown(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	model := &ModelProfile{Path: modelPath, TotalSizeMB: 10000, SizeBytes: 10000 * 1024 * 1024}
	gpus := optimizerTestGPUs()
	strategy := &Strategy{
		Type: MultiGPUDense, TensorSplit: []float64{0.5, 0.5}, ContextSize: 32768,
		Parallel: 1, BatchSize: 2048, UBatchSize: 512, KVPlacement: "gpu", KVQuality: "high", KVType: "q8_0",
	}
	identity := AllocationPlacementIdentity(strategy)
	if err := RecordMeasuredAllocation(dir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, "test", gpus, strategy.Parallel, MeasuredAllocation{
			Evidence: "allocation-verified", PlacementIdentity: identity, ContextTotalMB: 100,
			ContextByGPU: map[int]int{0: 50, 1: 50}, ModelByGPU: map[int]int{0: 5000, 1: 5000},
			UnaccountedByGPU: map[int]int{0: 200, 1: 200}, UnaccountedHostMB: 300,
		}); err != nil {
		t.Fatal(err)
	}
	logData := "llama_kv_cache_init: CUDA0 KV buffer size = 60.00 MiB\nllama_kv_cache_init: CUDA1 KV buffer size = 60.00 MiB\n"
	if !RecordPostLaunchContextAllocation(dir, model, strategy, "test", gpus, logData) {
		t.Fatal("post-launch context observation was not recorded")
	}
	loaded, ok := LoadMeasuredAllocation(dir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, "test", gpus, strategy.Parallel)
	if !ok || loaded.ContextTotalMB != 120 || loaded.ModelByGPU[0] != 5000 || loaded.UnaccountedByGPU[1] != 200 || loaded.PlacementIdentity != identity {
		t.Fatalf("sparse post-launch observation erased guarded breakdown: ok=%t loaded=%+v", ok, loaded)
	}
}

func TestRealQwenRecurrentLogPromotesCompleteLiveAllocation(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "qwen3.8-flash.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	model := &ModelProfile{
		Path: modelPath, TotalSizeMB: 85800, SizeBytes: 85800 * 1024 * 1024,
		NumLayers: 48, NumExperts: 128, IsMoE: true,
	}
	gpus := []detect.GPU{
		{Index: 0, Name: "RTX 4070", VRAMTotalMB: 12282},
		{Index: 1, Name: "RTX 3090 Ti", VRAMTotalMB: 24564},
		{Index: 2, Name: "RTX 3060", VRAMTotalMB: 12288},
	}
	systemPath := filepath.Join(dir, fmt.Sprintf("system_%s.cache", gpuSignatureHash(gpus)))
	system := fmt.Sprintf("SYS_PROBE_SCHEMA=%d\n", systemProbeSchema) +
		"SYS_CUDA_OVERHEAD_MB_CUDA0=500\n" +
		"SYS_CUDA_OVERHEAD_MB_CUDA1=600\n" +
		"SYS_CUDA_OVERHEAD_MB_CUDA2=400\n" +
		"SYS_CUDA_OVERHEAD_MB=600\nSYS_HOST_OVERHEAD_MB=750\n"
	if err := os.WriteFile(systemPath, []byte(system), 0o600); err != nil {
		t.Fatal(err)
	}
	strategy := &Strategy{
		Type: MoEOffload, TensorSplit: []float64{0.29, 0.61, 0.10},
		ContextSize: 262144, Parallel: 1, BatchSize: 2048, UBatchSize: 256,
		KVPlacement: "gpu", KVQuality: "high", KVType: "q8_0", NCPUMoE: 31,
	}
	logData := strings.Join([]string{
		"load_tensors:          CPU model buffer size = 27465.95 MiB",
		"load_tensors:        CUDA0 model buffer size =  5403.70 MiB",
		"load_tensors:        CUDA1 model buffer size = 13118.31 MiB",
		"load_tensors:        CUDA2 model buffer size =  5037.61 MiB",
		"load_tensors:    CUDA_Host model buffer size = 34781.64 MiB",
		"llama_context:  CUDA_Host  output buffer size =     0.95 MiB",
		"llama_kv_cache:      CUDA0 KV buffer size =  1536.00 MiB",
		"llama_kv_cache:      CUDA1 KV buffer size =  4096.00 MiB",
		"llama_kv_cache:      CUDA2 KV buffer size =   512.00 MiB",
		"llama_kv_cache:      CUDA0 KV buffer size =   576.00 MiB",
		"llama_kv_cache:      CUDA1 KV buffer size =  1536.00 MiB",
		"llama_kv_cache:      CUDA2 KV buffer size =   192.00 MiB",
		"sched_reserve:      CUDA0 compute buffer size =  1159.25 MiB",
		"sched_reserve:      CUDA1 compute buffer size =  1029.85 MiB",
		"sched_reserve:      CUDA2 compute buffer size =   822.00 MiB",
		"sched_reserve:  CUDA_Host compute buffer size =   937.05 MiB",
	}, "\n")

	if !RecordPostLaunchContextAllocation(dir, model, strategy, "qwen", gpus, logData) {
		t.Fatal("real log was not recorded")
	}
	allocation, ok := LoadMeasuredAllocation(dir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, "qwen", gpus, strategy.Parallel)
	if !ok || allocation.Evidence != "live-allocated" {
		t.Fatalf("live allocation did not become authoritative: ok=%t allocation=%+v", ok, allocation)
	}
	if allocation.ContextTotalMB != 8448 ||
		allocation.ContextByGPU[0] != 2112 || allocation.ContextByGPU[1] != 5632 || allocation.ContextByGPU[2] != 704 {
		t.Fatalf("recurrent KV regions were not summed per device: %+v", allocation)
	}
	if allocation.ModelByGPU[0] != 5404 || allocation.ModelByGPU[1] != 13118 || allocation.ModelByGPU[2] != 5038 {
		t.Fatalf("model distribution was not parsed: %+v", allocation.ModelByGPU)
	}
	if allocation.ModelHostMB != 62248 {
		t.Fatalf("CPU and CUDA_Host model buffers = %d, want 62248", allocation.ModelHostMB)
	}
	ledger := BuildResourceLedger(&detect.Capabilities{
		GPUs: gpus, RAM: detect.RAMInfo{TotalMB: 262144, FreeMB: 196608},
	}, model, strategy, Options{CacheDir: dir, BackendCacheTag: "qwen"})
	if !ledger.Exact || !ledger.Fits || ledger.Evidence != "live-allocated" {
		t.Fatalf("real live allocation did not unlock an exact fitting ledger: %+v", ledger)
	}

	// A later fit run for this key is only a prediction and must not demote the
	// healthy live ledger that topology and hot-expert search depend on.
	if err := RecordMeasuredAllocation(dir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, "qwen", gpus, strategy.Parallel, MeasuredAllocation{
			Evidence: "oracle-planned", PlacementIdentity: AllocationPlacementIdentity(strategy),
			ContextTotalMB: 8448, ContextByGPU: map[int]int{0: 2112, 1: 5632, 2: 704},
		}); err != nil {
		t.Fatal(err)
	}
	afterOracle, ok := LoadMeasuredAllocation(dir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, "qwen", gpus, strategy.Parallel)
	if !ok || afterOracle.Evidence != "live-allocated" || afterOracle.ModelByGPU[1] != 13118 {
		t.Fatalf("oracle demoted live allocation: ok=%t allocation=%+v", ok, afterOracle)
	}
}

func TestBatchAndParallelFrontiersRespectIntentWithoutFixedPairs(t *testing.T) {
	base := &Strategy{ContextSize: 196608, Parallel: 2, BatchSize: 2048, UBatchSize: 512}
	pairs := calibrationBatchNeighbors(base, Options{BatchSizeExplicit: true})
	seenLargeUBatch := false
	for _, pair := range pairs {
		if pair.batch != base.BatchSize || pair.ubatch > pair.batch {
			t.Fatalf("explicit batch or invariant changed: %+v", pair)
		}
		seenLargeUBatch = seenLargeUBatch || pair.ubatch == 1024
	}
	if !seenLargeUBatch {
		t.Fatal("bounded frontier did not challenge a larger ubatch")
	}
	if got := calibrationBatchNeighbors(base, Options{BatchSizeExplicit: true, UBatchSizeExplicit: true}); len(got) != 0 {
		t.Fatalf("both explicit batch coordinates were challenged: %+v", got)
	}

	parallel := calibrationParallelNeighbors(base, Options{ParallelSlotTarget: 32768})
	seen := map[int]bool{}
	for _, value := range parallel {
		seen[value] = true
	}
	for _, want := range []int{1, 3, 4, 5, 6} {
		if !seen[want] {
			t.Fatalf("legal parallel=%d missing from frontier %v", want, parallel)
		}
	}
}

func TestEqualCostFrontierPreservesNearestGeneratorOrder(t *testing.T) {
	frontier := AnalyzeCandidateFrontier(nil, nil, Options{}, []CalibrationCandidate{
		{Name: "default"},
		{Name: "batch-near", Estimate: CandidateEstimate{AgentCost: 1, Feasible: true}},
		{Name: "batch-lexically-earlier-but-far", Estimate: CandidateEstimate{AgentCost: 1, Feasible: true}},
	})
	if len(frontier) != 3 || frontier[1].Name != "batch-near" {
		t.Fatalf("equal-cost frontier lost generator proximity order: %+v", frontier)
	}
}

func TestMaterialHeadroomChangeIsDirectional(t *testing.T) {
	base := &Strategy{BatchSize: 2048, UBatchSize: 512, Parallel: 2, KVPlacement: "gpu"}
	lowerSlots := cloneStrategy(base)
	lowerSlots.Parallel = 1
	if materialHeadroomChange(base, lowerSlots) {
		t.Fatal("lower-resource slot count was classified as residual headroom")
	}
	largerGraph := cloneStrategy(base)
	largerGraph.UBatchSize = 1024
	if !materialHeadroomChange(base, largerGraph) {
		t.Fatal("larger physical microbatch did not consume headroom")
	}
}

func TestAnalyzeFrontierUsesExactHeadroomAndPersistsBoundary(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	model := &ModelProfile{
		Path: modelPath, TotalSizeMB: 10000, SizeBytes: 10000 * 1024 * 1024,
		NumLayers: 40, EmbeddingLength: 4096,
	}
	caps := &detect.Capabilities{
		GPUs:                    optimizerTestGPUs(),
		RAM:                     detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:                     detect.CPUInfo{Cores: 16, Threads: 32},
		HostMemoryBandwidthMBps: 50000,
	}
	opts := Options{CacheDir: dir, BackendCacheTag: "test", WorkloadProfile: "agent", WorkloadConcurrency: 1}
	base := &Strategy{
		Type: MultiGPUDense, TensorSplit: []float64{0.5, 0.5}, MainGPU: 0,
		ContextSize: 32768, Parallel: 1, BatchSize: 2048, UBatchSize: 512,
		KVPlacement: "cpu", KVQuality: "high", KVType: "q8_0",
		PlanFreeVRAM: map[int]int{0: 32768, 1: 32768},
	}
	if err := RecordMeasuredAllocation(
		dir, model, base.ContextSize, base.UBatchSize, base.KVQuality,
		base.KVPlacement, backendCacheTag(opts), caps.GPUs, base.Parallel,
		MeasuredAllocation{
			Evidence: "backend-measured", ContextTotalMB: 100, ContextHostMB: 100,
			ModelByGPU: map[int]int{0: 5000, 1: 5000},
		},
	); err != nil {
		t.Fatal(err)
	}
	fastHeavy := cloneStrategy(base)
	fastHeavy.TensorSplit = []float64{0.9, 0.1}
	frontier := AnalyzeCandidateFrontier(caps, model, opts, []CalibrationCandidate{
		{Name: "default", Strategy: base},
		{Name: "fast-heavy", Strategy: fastHeavy},
	})
	if base.ResourceLedger == nil || !base.ResourceLedger.Exact {
		t.Fatalf("baseline did not consume its exact allocation: %+v", base.ResourceLedger)
	}
	if frontier[1].Strategy.ResourceLedger == nil || frontier[1].Strategy.ResourceLedger.Exact {
		t.Fatal("different split reused the baseline's exact allocation")
	}
	if base.Residency != ResidencyRoomy {
		t.Fatalf("exact headroom with a faster feasible neighbor classified as %q", base.Residency)
	}
	if base.OptimizationBoundary == nil || base.OptimizationBoundary.CandidateCount != 2 ||
		base.OptimizationBoundary.FeasibleCount != 2 || base.OptimizationBoundary.ExactCount != 1 {
		t.Fatalf("incomplete explored boundary: %+v", base.OptimizationBoundary)
	}
}

func TestAnalyzeFrontierClassifiesRoomyFromExactBatchHeadroom(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	model := &ModelProfile{
		Path: modelPath, TotalSizeMB: 8000, SizeBytes: 8000 * 1024 * 1024,
		NumLayers: 32, EmbeddingLength: 4096,
	}
	caps := &detect.Capabilities{
		GPUs: optimizerTestGPUs()[:1],
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 16, Threads: 32},
	}
	opts := Options{CacheDir: dir, BackendCacheTag: "test"}
	base := &Strategy{
		Type: SingleGPU, MainGPU: 0, ContextSize: 32768, Parallel: 1,
		BatchSize: 2048, UBatchSize: 512, KVPlacement: "cpu", KVQuality: "high", KVType: "q8_0",
		PlanFreeVRAM: map[int]int{0: 32768},
	}
	if err := RecordMeasuredAllocation(
		dir, model, base.ContextSize, base.UBatchSize, base.KVQuality,
		base.KVPlacement, backendCacheTag(opts), caps.GPUs, base.Parallel,
		MeasuredAllocation{
			Evidence: "backend-measured", ContextTotalMB: 100, ContextHostMB: 100,
			ModelByGPU: map[int]int{0: 8000},
		},
	); err != nil {
		t.Fatal(err)
	}
	largerBatch := cloneStrategy(base)
	largerBatch.BatchSize = 4096
	frontier := AnalyzeCandidateFrontier(caps, model, opts, []CalibrationCandidate{
		{Name: "default", Strategy: base},
		{Name: "batch-4096-ubatch-512", Strategy: largerBatch},
	})
	if !frontier[1].Estimate.Feasible {
		t.Fatalf("larger batch did not fit the exact slack: %+v", frontier[1].Estimate)
	}
	if base.Residency != ResidencyRoomy {
		t.Fatalf("exact slack that admits a larger batch classified as %q", base.Residency)
	}
	if !frontier[1].Estimate.ActionableGain {
		t.Fatal("larger batch was not marked as actionable headroom")
	}
}

func TestPackedMoEHasPerformanceHeadroomWhenHostCanAbsorbAnExpertLayer(t *testing.T) {
	model := &ModelProfile{
		TotalSizeMB: 120000, SizeBytes: 120000 * 1024 * 1024,
		NumLayers: 40, IsMoE: true, ExpertBytes: 100000 * 1024 * 1024,
	}
	base := &Strategy{
		Type: MoEOffload, ContextSize: 32768, Parallel: 2, BatchSize: 128, UBatchSize: 64,
		KVPlacement: "gpu", KVQuality: "high", KVType: "f16", NCPUMoE: 34,
		TensorSplit: []float64{0.3, 0.7}, PlacementPolicy: "link",
		ResourceLedger: &ResourceLedger{Exact: true, Fits: true, Host: HostResourceLedger{SlackMB: 30000}},
	}
	alternate := cloneStrategy(base)
	alternate.TensorSplit = []float64{0.2, 0.8}
	alternate.PlacementPolicy = "memory"
	frontier := []CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true}},
		{Name: "moe-topology-memory", Strategy: alternate, Estimate: CandidateEstimate{Feasible: true}},
	}
	if !moeTopologyExperimentHeadroom(model, base, frontier) {
		t.Fatal("system-roomy MoE was confused with an intentionally packed GPU")
	}
	// The threshold is one layer derived from this model, not a fixed cushion.
	base.ResourceLedger.Host.SlackMB = 2499
	if moeTopologyExperimentHeadroom(model, base, frontier) {
		t.Fatal("genuinely tight host was admitted without one expert layer of fallback room")
	}
}

func TestEstimateStrategyCostPrefersFewerCPUExpertLayers(t *testing.T) {
	model := &ModelProfile{
		TotalSizeMB: 120000, SizeBytes: 120000 * 1024 * 1024,
		NumLayers: 40, IsMoE: true, ExpertBytes: 100000 * 1024 * 1024,
		NumExperts: 256, ExpertUsedCount: 8,
	}
	caps := &detect.Capabilities{
		GPUs: optimizerTestGPUs(), HostMemoryBandwidthMBps: 50000,
	}
	ledger := ResourceLedger{Fits: true, Devices: []DeviceResourceLedger{
		{GPU: 0, Active: true, ModelMB: 1000, RequiredMB: 3000, BandwidthMBps: 1_000_000},
		{GPU: 1, Active: true, ModelMB: 1000, RequiredMB: 3000, BandwidthMBps: 100_000},
	}}
	base := &Strategy{
		Type: MoEOffload, Parallel: 2, UBatchSize: 64, NCPUMoE: 36,
		TensorSplit: []float64{0.5, 0.5},
	}
	denser := cloneStrategy(base)
	denser.NCPUMoE = 32
	opts := Options{WorkloadConcurrency: 2, WorkloadProfile: "claude-agent-parallel-v4:test"}
	baseEstimate := EstimateStrategyCost(caps, model, base, opts, ledger)
	denserEstimate := EstimateStrategyCost(caps, model, denser, opts, ledger)
	if baseEstimate.Bottleneck != "CPU expert bandwidth" || denserEstimate.AgentCost >= baseEstimate.AgentCost {
		t.Fatalf("denser GPU expert pack was not preferred: base=%+v denser=%+v", baseEstimate, denserEstimate)
	}
}

func TestTightLiveCandidatesKeepProvenShape(t *testing.T) {
	base := &Strategy{
		Type: MultiGPUDense, MainGPU: 0, TensorSplit: []float64{0.5, 0.5},
		BatchSize: 2048, UBatchSize: 512, Residency: ResidencyTight,
		ResourceLedger: &ResourceLedger{Exact: true, Fits: true},
	}
	same := cloneStrategy(base)
	same.BatchSize = 4096
	same.Residency = ""
	reshuffle := cloneStrategy(base)
	reshuffle.TensorSplit = []float64{0.9, 0.1}
	reshuffle.Residency = ""
	got := TightLiveCandidates([]CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true}},
		{Name: "topology", Strategy: reshuffle, Estimate: CandidateEstimate{Feasible: true, AgentCost: 0.5}},
		{Name: "batch-4096", Strategy: same, Estimate: CandidateEstimate{Feasible: true, AgentCost: 1}},
	})
	if len(got) != 2 || got[1].Name != "batch-4096" {
		t.Fatalf("tight live set should keep the proven split: %+v", got)
	}
}

func TestTightLiveCandidatesKeepDenserGPUExpertPack(t *testing.T) {
	base := &Strategy{
		Type: MoEOffload, KVPlacement: "gpu", NCPUMoE: 38,
		TensorSplit:    []float64{0.26, 0.63, 0.11},
		OTString:       "blk.(0|1|2)=CUDA1",
		Residency:      ResidencyTight,
		ResourceLedger: &ResourceLedger{Exact: true, Fits: true},
		VRAMLedger: []GPULedgerEntry{
			{GPU: 1, ExpertLayers: 3}, {GPU: 0, ExpertLayers: 1}, {GPU: 2, ExpertLayers: 1},
		},
	}
	denser := cloneStrategy(base)
	denser.NCPUMoE = 34
	denser.TensorSplit = []float64{0.27, 0.65, 0.09}
	denser.OTString = "blk.(0|1|2|3|4)=CUDA1,blk.(5|6)=CUDA0,blk.(7|8)=CUDA2"
	denser.VRAMLedger = []GPULedgerEntry{
		{GPU: 1, ExpertLayers: 5}, {GPU: 0, ExpertLayers: 2}, {GPU: 2, ExpertLayers: 2},
	}
	mmap := cloneStrategy(denser)
	mmap.MMapRequired = true
	got := TightLiveCandidates([]CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true}},
		{Name: "moe-topology-memory", Strategy: denser, Estimate: CandidateEstimate{Feasible: true, AgentCost: 0.8}},
		{Name: "mmap", Strategy: mmap, Estimate: CandidateEstimate{Feasible: true, AgentCost: 0.7}},
	})
	if len(got) != 2 || got[1].Name != "moe-topology-memory" {
		t.Fatalf("tight live set dropped the denser GPU expert pack: %+v", got)
	}
}

func TestDeviceBalanceBottleneckFlagsIdleComputeGPU(t *testing.T) {
	s := &Strategy{Type: MultiGPUDense, TensorSplit: []float64{0.27, 0.65, 0.09}}
	got := DeviceBalanceBottleneck(s, []GPUUtilSample{
		{GPU: 0, SMPercent: 4, MemPercent: 80},
		{GPU: 1, SMPercent: 2, MemPercent: 70},
		{GPU: 2, SMPercent: 96, MemPercent: 90},
	})
	if got != "GPU 2 saturated (96% SM) while GPU 1 is idle (2% SM)" {
		t.Fatalf("imbalanced topology bottleneck=%q", got)
	}
	if got := DeviceBalanceBottleneck(s, nil); got != "" {
		t.Fatalf("missing samples must fail closed, got %q", got)
	}
}

func TestAnalyzeCandidateFrontierRanksMoEOwnerWhenCriticalPathIsCheaper(t *testing.T) {
	base := &Strategy{
		Type: MoEOffload, ContextSize: 32768, Parallel: 2,
		BatchSize: 128, UBatchSize: 64, KVPlacement: "gpu", KVQuality: "high", KVType: "bf16",
		NCPUMoE: 36, TensorSplit: []float64{0.27, 0.65, 0.08}, MainGPU: 0,
		ResourceLedger: &ResourceLedger{Fits: true, Devices: []DeviceResourceLedger{
			{GPU: 0, ModelMB: 3000, RequiredMB: 4000, Active: true, BandwidthMBps: 4000},
			{GPU: 1, ModelMB: 9000, RequiredMB: 10000, Active: true, BandwidthMBps: 16000},
			{GPU: 2, ModelMB: 1000, RequiredMB: 2000, Active: true, BandwidthMBps: 1000},
		}},
	}
	giantUBatch := cloneStrategy(base)
	giantUBatch.BatchSize = 8192
	giantUBatch.UBatchSize = 8192
	owner := cloneStrategy(base)
	owner.TensorSplit = []float64{0, 1, 0}
	owner.MainGPU = 1
	owner.PlacementPolicy = "owner-1"
	frontier := AnalyzeCandidateFrontier(&detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 12288, BandwidthMBps: 4000},
			{Index: 1, VRAMTotalMB: 24576, BandwidthMBps: 16000},
			{Index: 2, VRAMTotalMB: 12288, BandwidthMBps: 1000},
		},
		RAM:                     detect.RAMInfo{TotalMB: 217088, FreeMB: 200000},
		HostMemoryBandwidthMBps: 50000,
	}, &ModelProfile{
		TotalSizeMB: 68000, SizeBytes: 68000 * 1024 * 1024,
		NumLayers: 32, LeadingDense: 2, IsMoE: true,
		ExpertBytes: 60 * 1024 * 1024 * 1024, NonExpertBytes: 8 * 1024 * 1024 * 1024,
		NumExperts: 64, ExpertUsedCount: 4,
	}, Options{}, []CalibrationCandidate{
		{Name: "default", Strategy: base},
		{Name: "batch-8192-ubatch-8192", Strategy: giantUBatch},
		{Name: "moe-owner-1", Strategy: owner},
	})
	if len(frontier) < 3 || frontier[1].Name != "moe-owner-1" {
		t.Fatalf("ranked %+v, want cheaper moe-owner-1 first", namesOf(frontier))
	}
	if !(frontier[1].Estimate.AgentCost < frontier[2].Estimate.AgentCost) {
		t.Fatalf("owner was ordered by name rather than cost: owner=%g alternate=%g",
			frontier[1].Estimate.AgentCost, frontier[2].Estimate.AgentCost)
	}
}

func namesOf(candidates []CalibrationCandidate) []string {
	out := make([]string, len(candidates))
	for i, candidate := range candidates {
		out[i] = candidate.Name
	}
	return out
}

func TestSelectDeviceBalanceFinalistRelievesMeasuredBusyGPU(t *testing.T) {
	strategy := func(model0, model1, batch int) *Strategy {
		return &Strategy{
			Type: MultiGPUDense, ContextSize: 32768, Parallel: 2,
			BatchSize: batch, UBatchSize: 128, KVPlacement: "gpu", KVQuality: "mid", KVType: "q8_0",
			ResourceLedger: &ResourceLedger{Fits: true, Devices: []DeviceResourceLedger{
				{GPU: 0, ModelMB: model0, Active: true}, {GPU: 1, ModelMB: model1, Active: true},
			}},
		}
	}
	base := strategy(7000, 3000, 256)
	wrongCoordinate := strategy(3000, 7000, 512)
	balanced := strategy(4000, 6000, 256)
	frontier := []CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true, AgentCost: 1}},
		{Name: "batch-512", Strategy: wrongCoordinate, Estimate: CandidateEstimate{Feasible: true, AgentCost: 0.5}},
		{Name: "topology-capacity", Strategy: balanced, Estimate: CandidateEstimate{Feasible: true, AgentCost: 0.9}},
	}
	signal := AnalyzeDeviceBalance(base, []GPUUtilSample{{GPU: 0, SMPercent: 96}, {GPU: 1, SMPercent: 3}})
	got, ok := SelectDeviceBalanceFinalist(frontier, signal)
	if !ok || got.Name != "topology-capacity" {
		t.Fatalf("telemetry finalist=%+v ok=%t", got, ok)
	}
}

func TestSelectDeviceBalanceFinalistChoosesCheapestRelievingMoETopology(t *testing.T) {
	strategy := func(split []float64, modelMB []int) *Strategy {
		return &Strategy{
			Type: MoEOffload, ContextSize: 1048576, Parallel: 2,
			BatchSize: 128, UBatchSize: 64, KVPlacement: "gpu", KVQuality: "high", KVType: "bf16",
			NCPUMoE: 36, TensorSplit: split,
			ResourceLedger: &ResourceLedger{Fits: true, Devices: []DeviceResourceLedger{
				{GPU: 0, ModelMB: modelMB[0], Active: true},
				{GPU: 1, ModelMB: modelMB[1], Active: true},
				{GPU: 2, ModelMB: modelMB[2], Active: true},
			}},
		}
	}
	base := strategy([]float64{0.23, 0.67, 0.10}, []int{3000, 9000, 1000})
	reweighted := strategy([]float64{0.15, 0.75, 0.10}, []int{2000, 10000, 1000})
	// CUDA0 stores more routed experts in this candidate, but owns no ordinary
	// layers. Sparse expert bytes must not hide removal of its serial backbone.
	owner1 := strategy([]float64{0, 1, 0}, []int{5000, 7000, 1000})
	owner1.PlacementPolicy = "owner-1"
	frontier := []CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true, AgentCost: 1}},
		{Name: "moe-topology-memory", Strategy: reweighted, Estimate: CandidateEstimate{Feasible: true, AgentCost: 0.8}},
		{Name: "moe-owner-1", Strategy: owner1, Estimate: CandidateEstimate{Feasible: true, AgentCost: 0.9}},
	}
	signal := AnalyzeDeviceBalance(base, []GPUUtilSample{
		{GPU: 0, SMPercent: 98}, {GPU: 1, SMPercent: 1}, {GPU: 2, SMPercent: 2},
	})
	got, ok := SelectDeviceBalanceFinalist(frontier, signal)
	if !ok || got.Name != "moe-topology-memory" {
		t.Fatalf("role-aware finalist=%+v ok=%t", got, ok)
	}
}

func TestAnalyzeDeviceBalanceIgnoresIdleMoEExpertStorage(t *testing.T) {
	s := &Strategy{
		Type: MoEOffload, TensorSplit: []float64{0, 1}, MainGPU: 1,
		ResourceLedger: &ResourceLedger{Fits: true, Devices: []DeviceResourceLedger{
			{GPU: 0, ModelMB: 8000, Active: true},
			{GPU: 1, ModelMB: 7000, Active: true},
		}},
		VRAMLedger: []GPULedgerEntry{
			{GPU: 0, ExpertOnly: true, ExpertLayers: 4},
			{GPU: 1, ExpertLayers: 2},
		},
	}
	signal := AnalyzeDeviceBalance(s, []GPUUtilSample{
		{GPU: 0, SMPercent: 1}, {GPU: 1, SMPercent: 96},
	})
	if signal.Observed || signal.Imbalanced {
		t.Fatalf("idle expert storage was mistaken for a second serial compute stage: %+v", signal)
	}
}

func TestEstimateMoECostPricesGPUExpertResidency(t *testing.T) {
	const mib = int64(1024 * 1024)
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 16000, MemBandwidthMBps: 1_000_000},
			{Index: 1, VRAMTotalMB: 24576, BandwidthMBps: 16000, MemBandwidthMBps: 100_000},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 120000}, HostMemoryBandwidthMBps: 50000,
	}
	model := &ModelProfile{
		NumLayers: 10, IsMoE: true, NumExperts: 10, ExpertUsedCount: 1,
		ExpertBytes: 10 * 1024 * mib, NonExpertBytes: 1024 * mib,
		EmbeddingLength: 4096,
	}
	ledger := ResourceLedger{Fits: true, Devices: []DeviceResourceLedger{
		{GPU: 0, ModelMB: 11024, RequiredMB: 12000, Active: true, BandwidthMBps: 1_000_000},
		{GPU: 1, ModelMB: 10240, RequiredMB: 11000, Active: true, BandwidthMBps: 100_000},
	}}
	strategy := func(expertGPU int) *Strategy {
		s := &Strategy{
			Type: MoEOffload, ContextSize: 32768, Parallel: 1,
			BatchSize: 128, UBatchSize: 64, KVPlacement: "gpu", KVType: "f16",
			TensorSplit: []float64{1, 0}, MainGPU: 0,
			VRAMLedger: []GPULedgerEntry{{GPU: expertGPU, ExpertLayers: 10, ExpertOnly: expertGPU != 0}},
		}
		copyLedger := ledger
		s.ResourceLedger = &copyLedger
		return s
	}
	fast := EstimateStrategyCost(caps, model, strategy(0), Options{WorkloadConcurrency: 1}, ledger)
	slow := EstimateStrategyCost(caps, model, strategy(1), Options{WorkloadConcurrency: 1}, ledger)
	if fast.GPUExpertCost <= 0 || slow.GPUExpertCost <= fast.GPUExpertCost*5 {
		t.Fatalf("GPU expert service was not priced by its execution device: fast=%+v slow=%+v", fast, slow)
	}
	if slow.AgentCost <= fast.AgentCost {
		t.Fatalf("slow expert GPU ranked ahead of fast residency: fast=%g slow=%g", fast.AgentCost, slow.AgentCost)
	}
}

func TestSelectDeviceBalanceFinalistPreservesTightMoEResidency(t *testing.T) {
	strategy := func(model0, model1, cpuMoE int) *Strategy {
		return &Strategy{
			Type: MoEOffload, ContextSize: 32768, Parallel: 1,
			BatchSize: 256, UBatchSize: 128, KVPlacement: "gpu", KVQuality: "mid", KVType: "q8_0",
			NCPUMoE: cpuMoE,
			ResourceLedger: &ResourceLedger{Fits: true, Devices: []DeviceResourceLedger{
				{GPU: 0, ModelMB: model0, Active: true}, {GPU: 1, ModelMB: model1, Active: true},
			}},
		}
	}
	base := strategy(7000, 3000, 30)
	moreCPU := strategy(3000, 7000, 31)
	frontier := []CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true}},
		{Name: "more-cpu-offload", Strategy: moreCPU, Estimate: CandidateEstimate{Feasible: true}},
	}
	signal := AnalyzeDeviceBalance(base, []GPUUtilSample{{GPU: 0, SMPercent: 90}, {GPU: 1, SMPercent: 4}})
	if got, ok := SelectDeviceBalanceFinalist(frontier, signal); ok {
		t.Fatalf("less-resident topology became a finalist: %+v", got)
	}
}

func TestSelectDeviceBalanceFinalistMeasuresReliefEvenWhenStaticPriorIsUncertain(t *testing.T) {
	strategy := func(split []float64, cpuMoE int) *Strategy {
		return &Strategy{
			Type: MoEOffload, Residency: ResidencyTight, ContextSize: 32768, Parallel: 1,
			BatchSize: 256, UBatchSize: 128, KVPlacement: "gpu", KVQuality: "mid", KVType: "q8_0",
			NCPUMoE: cpuMoE, TensorSplit: split,
			ResourceLedger: &ResourceLedger{Exact: true, Fits: true, Devices: []DeviceResourceLedger{
				{GPU: 0, ModelMB: 7000, Active: true}, {GPU: 1, ModelMB: 3000, Active: true},
			}},
		}
	}
	base := strategy([]float64{0.7, 0.3}, 30)
	challenger := strategy([]float64{0.3, 0.7}, 30)
	frontier := []CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true, AgentCost: 1}},
		// Static prediction is slightly worse; measured imbalance is precisely why
		// one live A/B is still necessary.
		{Name: "moe-owner-1", Strategy: challenger, Estimate: CandidateEstimate{Feasible: true, AgentCost: 1.05}},
	}
	signal := AnalyzeDeviceBalance(base, []GPUUtilSample{{GPU: 0, SMPercent: 95}, {GPU: 1, SMPercent: 3}})
	got, ok := SelectDeviceBalanceFinalist(frontier, signal)
	if !ok || got.Name != "moe-owner-1" {
		t.Fatalf("measured imbalance did not authorize bounded challenger: got=%+v ok=%t", got, ok)
	}
}

func TestDenseSubsetSplitRejectsIdleMembersOfMask(t *testing.T) {
	gpus := optimizerTestGPUs()
	free := []int{32768, 32768}
	fixed := []int{1024, 1024}
	if split, ok := denseSubsetSplit(gpus, free, fixed, 8000, 0b11, "critical"); ok {
		t.Fatalf("faster card can hold the model, but a zero-weight second GPU was accepted: %v", split)
	}
	split, ok := denseSubsetSplit(gpus, free, fixed, 40000, 0b11, "critical")
	if !ok || len(split) < 2 || split[0] <= 0 || split[1] <= 0 {
		t.Fatalf("needed both GPUs, got ok=%v split=%v", ok, split)
	}
	zeroCapacity := []int{32768, 1024}
	if split, ok = denseSubsetSplit(gpus, zeroCapacity, fixed, 8000, 0b11, "capacity"); ok {
		t.Fatalf("mask member with no remaining capacity was accepted: %v", split)
	}
}

func TestTopologyRankingPrefersFastSingleGPUWhenItFits(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: optimizerTestGPUs(),
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 16, Threads: 32},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 8000, SizeBytes: 8000 * 1024 * 1024, NumLayers: 32, EmbeddingLength: 4096}
	base := &Strategy{
		Type: MultiGPUDense, TensorSplit: []float64{0.5, 0.5}, MainGPU: 0,
		ContextSize: 32768, Parallel: 1, BatchSize: 2048, UBatchSize: 512,
		KVPlacement: "cpu", KVQuality: "high", KVType: "q8_0",
	}
	alternates := denseTopologyCandidates(caps, model, base, Options{})
	frontier := []CalibrationCandidate{{Name: "default", Strategy: base}}
	frontier = append(frontier, alternates...)
	frontier = AnalyzeCandidateFrontier(caps, model, Options{}, frontier)
	if len(frontier) < 3 {
		t.Fatalf("single-device topology candidates missing: %+v", frontier)
	}
	if frontier[1].Name != "single-gpu-0" {
		t.Fatalf("calculated finalist=%q, want fastest fitting single GPU", frontier[1].Name)
	}
}
