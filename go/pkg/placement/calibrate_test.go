package placement

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestCalibrationCandidatesSingleGPUOffloadedMoEAddsUBatchLadder(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 32000}},
		RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 131072}, CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path: "m.gguf", IsMoE: true, TotalSizeMB: 64 * 1024, NumLayers: 60, NumExperts: 128,
		ExpertBytes: 56 * 1024 * 1024 * 1024, NonExpertBytes: 8 * 1024 * 1024 * 1024,
		ContextSize: 8192, HiddenSize: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	base := &Strategy{Type: MoEOffload, KVPlacement: "cpu", KVQuality: "mid", KVType: "q8_0", ContextSize: 8192, Parallel: 1, BatchSize: 2048, UBatchSize: 512}
	opts := Options{ContextSize: 8192, KVPlacement: "cpu", KVQuality: "mid", Parallel: 1, CacheDir: t.TempDir()}
	if _, err := recomputeBatchCandidate(caps, model, base, opts, 2048, 1024); err != nil {
		t.Fatalf("recompute 2048/1024: %v", err)
	}
	got := CalibrationCandidates(caps, model, base, opts)
	if len(got) < 2 || got[0].Name != "default" {
		t.Fatalf("single-GPU offloaded MoE has no bounded neighbors: %+v", got)
	}
	foundLargerUBatch := false
	for _, candidate := range got[1:] {
		if candidate.Strategy == nil || candidate.Strategy.Type != MoEOffload ||
			candidate.Strategy.UBatchSize > candidate.Strategy.BatchSize {
			t.Fatalf("candidate was not a legal full MoE placement: %+v", candidate)
		}
		if candidate.Strategy.BatchSize == 2048 && candidate.Strategy.UBatchSize == 1024 {
			foundLargerUBatch = true
		}
	}
	if !foundLargerUBatch {
		t.Fatalf("single-GPU offloaded MoE omitted the proved larger ubatch: %+v", got)
	}
}

func TestCalibrationCandidatesCPUOnlyIsNoOp(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 12288},
	}}
	base := &Strategy{Type: CPUOnly}
	got := CalibrationCandidates(caps, &ModelProfile{Path: "m.gguf"}, base, Options{})
	if len(got) != 1 {
		t.Fatalf("CPU-only must not produce alternatives, got %+v", got)
	}
}

func TestCalibrationCandidatesMoEAddsKVAlternate(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 32000},
		{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 8000},
	}}
	model := &ModelProfile{
		Path: "m.gguf", IsMoE: true, TotalSizeMB: 64 * 1024, NumLayers: 60, NumExperts: 128,
		ExpertBytes: 56 * 1024 * 1024 * 1024, NonExpertBytes: 8 * 1024 * 1024 * 1024,
		ContextSize: 8192, HiddenSize: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	caps.RAM = detect.RAMInfo{TotalMB: 131072, FreeMB: 131072}
	caps.CPU = detect.CPUInfo{Cores: 16}
	base := &Strategy{
		Type: MoEOffload, KVPlacement: "cpu", KVQuality: "mid", KVType: "q8_0",
		ContextSize: 8192, Parallel: 1, BatchSize: 2048, UBatchSize: 512,
		NCPUMoE: 40, OTString: "blk.ffn=CPU",
	}
	opts := Options{ContextSize: 8192, KVPlacement: "cpu", KVQuality: "mid", Parallel: 1, CacheDir: t.TempDir()}
	if _, err := recomputeBatchCandidate(caps, model, base, opts, 2048, 1024); err != nil {
		t.Fatalf("recompute 2048/1024: %v", err)
	}
	got := CalibrationCandidates(caps, model, base, opts)
	if len(got) < 3 || got[0].Name != "default" {
		t.Fatalf("MoE multi-GPU should offer batch and KV neighbors, got %d candidates: %+v", len(got), got)
	}
	var alt *Strategy
	for _, candidate := range got {
		if candidate.Name == "kv-alternate" {
			alt = candidate.Strategy
			break
		}
	}
	if alt == nil {
		t.Fatalf("MoE candidate set omitted KV alternate: %+v", got)
	}
	if alt.KVPlacement != "gpu" {
		t.Fatalf("KV alternate should flip cpu->gpu, got %q", alt.KVPlacement)
	}
	// The alternate must not alias the base's expert split.
	if alt == base {
		t.Fatal("candidate aliases the base strategy")
	}
	if base.KVPlacement != "cpu" {
		t.Fatalf("base mutated to %q", base.KVPlacement)
	}
}

func TestCalibrationCandidatesMoEKVGPUFlipsToCPU(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 12288},
	}}
	caps.RAM = detect.RAMInfo{TotalMB: 131072, FreeMB: 131072}
	caps.CPU = detect.CPUInfo{Cores: 16}
	base := &Strategy{Type: MoEOffload, KVPlacement: "gpu"}
	model := &ModelProfile{
		Path: "m.gguf", IsMoE: true, TotalSizeMB: 64 * 1024, NumLayers: 60, NumExperts: 128,
		ExpertBytes: 56 * 1024 * 1024 * 1024, NonExpertBytes: 8 * 1024 * 1024 * 1024,
	}
	got := CalibrationCandidates(caps, model, base, Options{ContextSize: 8192, KVPlacement: "gpu"})
	found := false
	for _, candidate := range got {
		if candidate.Name == "kv-alternate" && candidate.Strategy.KVPlacement == "cpu" {
			found = true
		}
	}
	if !found {
		t.Fatalf("KV=gpu should alternate to cpu, got %+v", got)
	}
}

func TestCalibrationCandidatesMoEUBatchRespectsExplicitRequest(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 12288}},
		RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path: "m.gguf", IsMoE: true, TotalSizeMB: 64 * 1024, NumLayers: 60, NumExperts: 128,
		ExpertBytes: 56 * 1024 * 1024 * 1024, NonExpertBytes: 8 * 1024 * 1024 * 1024,
		ContextSize: 8192, HiddenSize: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	base := &Strategy{
		Type: MoEOffload, KVPlacement: "cpu", KVQuality: "mid", KVType: "q8_0",
		ContextSize: 8192, Parallel: 1, BatchSize: 2048, UBatchSize: 512,
	}
	got := CalibrationCandidates(caps, model, base, Options{
		ContextSize: 8192, KVPlacement: "cpu", KVQuality: "mid", Parallel: 1,
		UBatchSize: 512, UBatchSizeExplicit: true,
	})
	for _, candidate := range got {
		if candidate.Name == "ubatch-1024" || candidate.Name == "ubatch-2048" {
			t.Fatalf("explicit ubatch was challenged by %q", candidate.Name)
		}
	}
}

func TestCalibrationScopeSeparatesExplicitAndAutomaticUBatch(t *testing.T) {
	model := &ModelProfile{Path: "m.gguf"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "gpu", VRAMTotalMB: 24576}}}
	base := &Strategy{ContextSize: 8192, Parallel: 1, BatchSize: 2048, UBatchSize: 512}
	auto := NewCalibrationScopeKey(model, caps, Options{}, base).String()
	explicit := NewCalibrationScopeKey(model, caps, Options{UBatchSizeExplicit: true}, base).String()
	if auto == explicit {
		t.Fatal("automatic and user-explicit ubatch shared a calibration scope")
	}
}

func TestLatestCalibrationDecisionForModelAndStatusLine(t *testing.T) {
	dir := t.TempDir()
	older := CalibrationDecision{
		ScopeKey: "a", Winner: "default", ModelBasename: "model.gguf",
		MeasuredAt: "2026-08-01T00:00:00Z", BaselineResidency: ResidencyTight,
		FinalistOutcome: "baseline-won",
	}
	newer := CalibrationDecision{
		ScopeKey: "b", Winner: "batch-256-ubatch-256", ModelBasename: "model.gguf",
		MeasuredAt: "2026-08-26T00:00:00Z", BaselineResidency: ResidencyRoomy,
		FinalistOutcome: "promoted", BaselineBottleneck: "GPU 2 compute idle",
	}
	if _, err := SaveCalibrationDecision(dir, older); err != nil {
		t.Fatal(err)
	}
	if _, err := SaveCalibrationDecision(dir, newer); err != nil {
		t.Fatal(err)
	}
	got, err := LatestCalibrationDecisionForModel(dir, "model.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if got.Winner != "batch-256-ubatch-256" || got.MeasuredAt != newer.MeasuredAt {
		t.Fatalf("did not return the newest decision: %+v", got)
	}
	line := FormatCalibrationStatus(got)
	if !strings.Contains(line, "roomy-resident") || !strings.Contains(line, "promoted") ||
		!strings.Contains(line, "GPU 2 compute idle") {
		t.Fatalf("status line omitted inspect fields: %q", line)
	}
	if line := FormatCalibrationStatus(nil); !strings.Contains(line, "cold estimate") {
		t.Fatalf("nil decision status=%q", line)
	}
	unavailable := &CalibrationDecision{
		Winner: "default", Finalist: "ubatch-2048", FinalistOutcome: "unavailable",
		BaselineResidency: ResidencyRoomy,
	}
	line = FormatCalibrationStatus(unavailable)
	if !strings.Contains(line, "default retained") || !strings.Contains(line, "finalist ubatch-2048 unavailable") ||
		strings.Contains(line, "unavailable default") {
		t.Fatalf("unavailable-finalist status misidentified the serving baseline: %q", line)
	}
}

func TestCalibrationModelBasenameUnifiesShardedStatusIdentity(t *testing.T) {
	const shard = "DeepSeek-V4-Flash-UD-Q3_K_XL-00001-of-00004.gguf"
	const canonical = "DeepSeek-V4-Flash-UD-Q3_K_XL.gguf"
	if got := CalibrationModelBasename("/models/" + shard); got != canonical {
		t.Fatalf("canonical shard identity=%q, want %q", got, canonical)
	}
	dir := t.TempDir()
	if _, err := SaveCalibrationDecision(dir, CalibrationDecision{
		ScopeKey: "sharded", Winner: "default", ModelBasename: shard,
	}); err != nil {
		t.Fatal(err)
	}
	decision, err := LatestCalibrationDecisionForModel(dir, canonical)
	if err != nil || decision.Winner != "default" {
		t.Fatalf("normalized TUI name did not find shard decision: decision=%+v err=%v", decision, err)
	}
}

func TestCalibrationStatusIgnoresStaleSchemaDecisions(t *testing.T) {
	dir := t.TempDir()
	stale := CalibrationDecision{
		SchemaVersion: CalibrationSchemaVersion - 1,
		ScopeKey:      "stale", Winner: "bad-winner", ModelBasename: "model.gguf",
		MeasuredAt: "2099-01-01T00:00:00Z",
	}
	if _, err := SaveCalibrationDecision(dir, stale); err != nil {
		t.Fatal(err)
	}
	current := CalibrationDecision{
		ScopeKey: "current", Winner: "default", ModelBasename: "model.gguf",
		MeasuredAt: "2026-08-26T00:00:00Z",
	}
	if _, err := SaveCalibrationDecision(dir, current); err != nil {
		t.Fatal(err)
	}
	got, err := LatestCalibrationDecisionForModel(dir, "model.gguf")
	if err != nil || got.Winner != "default" {
		t.Fatalf("latest current decision=%+v err=%v", got, err)
	}
	listed := ListCalibrationDecisions(dir, 0)
	if len(listed) != 1 || listed[0].SchemaVersion != CalibrationSchemaVersion {
		t.Fatalf("status listed stale decisions: %+v", listed)
	}
}

func TestAdmissionDecisionSuppressesOnlyItsExactAutomaticRetry(t *testing.T) {
	decision := &CalibrationDecision{
		Winner: "default", ValidationLevel: CalibrationValidationAdmission,
		Finalist: "ubatch-2048", FinalistOutcome: "unavailable",
		FinalistFailureClass: "cuda-oom", FinalistFailureReason: "exact candidate CUDA OOM",
	}
	if decision.AutomaticEligible() {
		t.Fatal("admission-only evidence became automatic performance evidence")
	}
	if !decision.SuppressesAutomaticAdmissionRetry("ubatch-2048") {
		t.Fatal("exact unavailable finalist was not suppressed")
	}
	if decision.SuppressesAutomaticAdmissionRetry("parallel-2") {
		t.Fatal("one failed finalist suppressed a different candidate")
	}
	decision.FinalistFailureReason = ""
	if decision.SuppressesAutomaticAdmissionRetry("ubatch-2048") {
		t.Fatal("unexplained admission failure suppressed a retry")
	}
	decision.ValidationLevel = CalibrationValidationWorkflow
	if decision.SuppressesAutomaticAdmissionRetry("ubatch-2048") {
		t.Fatal("workflow result was treated as admission-only evidence")
	}
}

func TestAdmissionDecisionAccumulatesScopedRejectedCoordinates(t *testing.T) {
	previous := &CalibrationDecision{
		Winner: "default", ValidationLevel: CalibrationValidationAdmission,
		Finalist: "ubatch-2048", FinalistOutcome: "unavailable",
		FinalistFailureClass: "memory", FinalistFailureReason: "CUDA0 deficit",
	}
	current := &CalibrationDecision{
		Winner: "default", ValidationLevel: CalibrationValidationAdmission,
		Finalist: "hot-experts-23", FinalistOutcome: "unavailable",
		FinalistFailureClass: "telemetry", FinalistFailureReason: "no cache hits",
	}
	current.RecordAdmissionRejection(current.Finalist, current.FinalistFailureClass, current.FinalistFailureReason)
	current.MergeAdmissionRejections(previous)
	for _, finalist := range []string{"ubatch-2048", "hot-experts-23"} {
		if !current.SuppressesAutomaticAdmissionRetry(finalist) {
			t.Fatalf("merged decision forgot rejected coordinate %q: %+v", finalist, current.RejectedFinalists)
		}
	}
	if current.SuppressesAutomaticAdmissionRetry("moe-owner-1") {
		t.Fatal("scoped rejection suppressed an untested coordinate")
	}
}

func TestCalibrationScopeIncludesFullCorePolicy(t *testing.T) {
	model := &ModelProfile{Path: "m.gguf", SizeBytes: 1234}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "gpu", VRAMTotalMB: 24576}}}
	base := &Strategy{ContextSize: 32768, Parallel: 2, BatchSize: 2048, UBatchSize: 512, Type: SingleGPU}
	baseline := Options{
		SamplingProfile: "default", BackendHelp: "--fit", BackendIdentity: "build",
		CPUExpertMMapCapability: CPUExpertMMapFileBacked, CPUExpertMMapEvidence: "probe-a",
	}
	baseKey := NewCalibrationScopeKey(model, caps, baseline, base).String()
	variants := []Options{baseline, baseline, baseline, baseline, baseline}
	variants[0].ParallelExplicit = true
	variants[1].SamplingProfile = "greedy"
	variants[2].Companions = []CompanionReservation{{Name: "reviewer", VRAMMB: 4096, GPUPreference: []int{0}}}
	variants[3].SpecMode = "mtp"
	variants[4].CPUExpertMMapCapability = CPUExpertMMapUnknown
	for i, opts := range variants {
		if got := NewCalibrationScopeKey(model, caps, opts, base).String(); got == baseKey {
			t.Fatalf("policy variant %d shared the baseline calibration scope", i)
		}
	}
}

func TestCalibrationScopeSeparatesHostExecutionPolicy(t *testing.T) {
	model := &ModelProfile{Path: "m.gguf", SizeBytes: 1234}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "gpu", VRAMTotalMB: 24576}}}
	base := &Strategy{
		ContextSize: 32768, Parallel: 2, BatchSize: 2048, UBatchSize: 512,
		Threads: 14, ThreadsBatch: 14, CPURange: "0-13", CPUStrict: true,
	}
	baseline := NewCalibrationScopeKey(model, caps, Options{}, base).String()
	variants := []*Strategy{}
	for i := 0; i < 4; i++ {
		copyBase := *base
		variants = append(variants, &copyBase)
	}
	variants[0].Threads = 12
	variants[1].ThreadsBatch = 12
	variants[2].CPURange = ""
	variants[3].CPUStrict = false
	for i, variant := range variants {
		if got := NewCalibrationScopeKey(model, caps, Options{}, variant).String(); got == baseline {
			t.Fatalf("host execution variant %d shared calibration evidence", i)
		}
	}
}

func TestCalibrationCandidatesGeneralDenseSearchesBatchAndSlots(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "gpu", VRAMTotalMB: 24576, BandwidthMBps: 32000}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536}, CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "dense.gguf", TotalSizeMB: 8192, SizeBytes: 8192 * 1024 * 1024,
		NumLayers: 32, HiddenSize: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	opts := Options{ContextSize: 131072, KVPlacement: "gpu", KVQuality: "mid", Parallel: 1, CacheDir: t.TempDir()}
	base, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("compute baseline: %v", err)
	}
	got := CalibrationCandidates(caps, model, base, opts)
	foundBatch, foundParallel := false, false
	for _, candidate := range got[1:] {
		if candidate.Strategy == nil || !sameCalibrationResidency(base, candidate.Strategy) {
			t.Fatalf("optimizer emitted an unproved residency transition: %+v", candidate)
		}
		if candidate.Strategy.BatchSize != base.BatchSize || candidate.Strategy.UBatchSize != base.UBatchSize {
			foundBatch = true
		}
		if candidate.Strategy.Parallel != base.Parallel {
			foundParallel = true
		}
	}
	if !foundBatch || !foundParallel {
		t.Fatalf("general dense optimizer omitted a core dimension: batch=%v parallel=%v candidates=%+v", foundBatch, foundParallel, got)
	}
	explicitParallel := opts
	explicitParallel.ParallelExplicit = true
	for _, candidate := range CalibrationCandidates(caps, model, base, explicitParallel) {
		if candidate.Strategy != nil && candidate.Strategy.Parallel != base.Parallel {
			t.Fatalf("explicit parallel was challenged by %+v", candidate)
		}
	}
}

func TestCalibrationResidencyRejectsGPUToDenseCPUFallback(t *testing.T) {
	base := &Strategy{
		Type: MoEOffload, KVPlacement: "cpu", ContextSize: 32768,
	}
	fallback := &Strategy{
		Type: DenseCPUOffload, KVPlacement: "cpu", ContextSize: 32768,
	}
	if sameCalibrationResidency(base, fallback) {
		t.Fatal("GPU-backed CPU-KV MoE candidate silently became dense CPU offload")
	}
}

func TestCalibrationCandidatesParallelAgentTunesValidBatchPairs(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 12288},
	}}
	model := &ModelProfile{Path: "qwen.gguf", HasSSM: 1}
	base := &Strategy{
		Type: MultiGPUDense, HasSSM: true, Parallel: 2,
		BatchSize: 128, UBatchSize: 128,
		TensorSplit: []float64{0.25, 0.75}, MainGPU: 1,
	}
	got := CalibrationCandidates(caps, model, base, Options{WorkloadProfile: "claude-agent-parallel-v4:test"})
	want := []batchPair{{256, 128}, {256, 256}, {512, 256}, {512, 512}}
	if len(got) < 1+len(want) {
		t.Fatalf("missing agent batch candidates: %+v", got)
	}
	for i, pair := range want {
		candidate := got[i+1]
		if candidate.Strategy.BatchSize != pair.batch || candidate.Strategy.UBatchSize != pair.ubatch {
			t.Fatalf("candidate %d = %d/%d, want %d/%d", i, candidate.Strategy.BatchSize, candidate.Strategy.UBatchSize, pair.batch, pair.ubatch)
		}
		if candidate.Strategy.UBatchSize > candidate.Strategy.BatchSize {
			t.Fatalf("candidate %q violates batch invariant", candidate.Name)
		}
		if candidate.Strategy.ContextAllocationMB != 0 || candidate.Strategy.ContextAllocationEvidence != "" {
			t.Fatalf("candidate %q retained foreign ubatch evidence: %+v", candidate.Name, candidate.Strategy)
		}
	}
}

func TestCalibrationCandidatesParallelAgentPreservesExplicitBatchIntent(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}}}
	model := &ModelProfile{Path: "qwen.gguf", HasSSM: 1}
	base := &Strategy{Type: SingleGPU, HasSSM: true, Parallel: 2, BatchSize: 256, UBatchSize: 128}
	for _, opts := range []Options{
		{WorkloadProfile: "claude-agent-parallel-v4:test", BatchSizeExplicit: true},
		{WorkloadProfile: "claude-agent-parallel-v4:test", UBatchSizeExplicit: true},
	} {
		got := CalibrationCandidates(caps, model, base, opts)
		if len(got) != 1 {
			t.Fatalf("explicit batch intent was challenged: %+v", got)
		}
	}
}

func TestNormalizeBatchSizesHonorsIntentAndUpdatesCheckpoint(t *testing.T) {
	model := &ModelProfile{SlidingWindow: 512}
	auto := &Strategy{BatchSize: 128, UBatchSize: 512, CheckpointMinStep: 1024}
	if !NormalizeBatchSizes(auto, model, false, false) || auto.BatchSize != 128 || auto.UBatchSize != 128 {
		t.Fatalf("automatic inverted pair not lowered safely: %+v", auto)
	}
	if auto.CheckpointMinStep != checkpointMinStep(model, 128) {
		t.Fatalf("checkpoint spacing did not follow normalized ubatch: %+v", auto)
	}
	explicitU := &Strategy{BatchSize: 128, UBatchSize: 512}
	if !NormalizeBatchSizes(explicitU, model, false, true) || explicitU.BatchSize != 512 || explicitU.UBatchSize != 512 {
		t.Fatalf("explicit ubatch did not raise automatic batch: %+v", explicitU)
	}
}

func TestCalibrationCandidatesDenseSplitInversion(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 32000},
		{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 8000},
	}}
	base := &Strategy{Type: MultiGPUDense, TensorSplit: []float64{0.75, 0.25}, MainGPU: 0}
	got := CalibrationCandidates(caps, &ModelProfile{Path: "m.gguf"}, base, Options{})
	if len(got) != 2 || got[1].Name != "split-inverted" {
		t.Fatalf("dense multi-GPU should offer a split inversion, got %+v", got)
	}
	inv := got[1].Strategy
	if inv.TensorSplit[0] != 0.25 || inv.TensorSplit[1] != 0.75 {
		t.Fatalf("split not inverted: %v", inv.TensorSplit)
	}
	if inv.MainGPU != 1 {
		t.Fatalf("main GPU should follow the larger share, got %d", inv.MainGPU)
	}
	// Base untouched.
	if base.TensorSplit[0] != 0.75 || base.MainGPU != 0 {
		t.Fatal("base strategy mutated by inversion")
	}
}

func TestCalibrationCandidatesDenseSymmetricSplitSkipped(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 24576},
	}}
	base := &Strategy{Type: MultiGPUDense, TensorSplit: []float64{0.5, 0.5}, MainGPU: 0}
	got := CalibrationCandidates(caps, &ModelProfile{Path: "m.gguf"}, base, Options{})
	if len(got) != 1 {
		t.Fatalf("a symmetric split inverts to itself and must be skipped, got %+v", got)
	}
}

func TestMoETopologyCandidatesIncludeCompleteSplitOwners(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 16384, BandwidthMBps: 16000},
			{Index: 1, VRAMTotalMB: 24576, BandwidthMBps: 16000},
			{Index: 2, VRAMTotalMB: 16384, BandwidthMBps: 16000},
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
	opts := Options{
		ContextSize: 32768, KVPlacement: "gpu", KVQuality: "low",
		Parallel: 2, NoMMap: true, CacheDir: t.TempDir(),
	}
	base, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, candidate := range moeTopologyCandidates(caps, model, base, opts) {
		if candidate.Name != "moe-owner-1" {
			continue
		}
		found = true
		if got := candidate.Strategy.TensorSplit; len(got) != 3 || got[0] != 0 || got[1] != 1 || got[2] != 0 {
			t.Fatalf("owner candidate was not fully recomputed: %v", got)
		}
	}
	if !found {
		t.Fatal("candidate frontier omitted the CUDA1 backbone-owner hypothesis")
	}
}

func TestCalibrationDecisionRoundTrip(t *testing.T) {
	dir := t.TempDir()
	key := CalibrationScopeKey{
		ModelIdentity: "m", BackendIdentity: "b", HardwareID: "h",
		WorkloadProfile: "claude-agent-parallel-v1", ContextSize: 131072,
		Parallel: 4, UBatchSize: 256, KVQuality: "mid",
	}.String()
	d := CalibrationDecision{
		ScopeKey: key, Winner: "kv-alternate",
		DefaultTPS: 20.5, WinnerTPS: 24.1, Improvement: 17.5,
		BaselineResidency: ResidencyRoomy, Finalist: "kv-alternate", FinalistOutcome: "promoted",
		ExploredBoundary: &OptimizationBoundary{CandidateCount: 12, FeasibleCount: 8, MaxUBatch: 2048},
	}
	if _, err := SaveCalibrationDecision(dir, d); err != nil {
		t.Fatalf("save: %v", err)
	}
	loaded, err := LoadCalibrationDecision(dir, key)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.Winner != "kv-alternate" || loaded.WinnerTPS != 24.1 ||
		loaded.BaselineResidency != ResidencyRoomy || loaded.FinalistOutcome != "promoted" ||
		loaded.ExploredBoundary == nil || loaded.ExploredBoundary.CandidateCount != 12 {
		t.Fatalf("round trip lost decision: %+v", loaded)
	}
	// A different scope key must not read this decision.
	if _, err := LoadCalibrationDecision(dir, "other-scope"); err == nil {
		t.Fatal("stale/foreign scope must not load a decision")
	}
	// The file lives under the calibration namespace.
	if filepath.Dir(CalibrationPath(dir, key)) != filepath.Join(dir, "calibration") {
		t.Fatalf("unexpected calibration path %q", CalibrationPath(dir, key))
	}
}

func TestCalibrationScopeIncludesRequestedWorkloadConcurrency(t *testing.T) {
	base := CalibrationScopeKey{ModelIdentity: "m", BackendIdentity: "b", HardwareID: "h", WorkloadProfile: "parallel"}
	base.WorkloadConcurrency = 2
	two := base.String()
	base.WorkloadConcurrency = 4
	if four := base.String(); four == two {
		t.Fatal("two-agent and four-agent optimization shared a decision scope")
	}
}

func TestDeleteCalibrationDecisionsForModel(t *testing.T) {
	dir := t.TempDir()
	for _, decision := range []CalibrationDecision{
		{ScopeKey: "scope-a", ModelBasename: "a.gguf", Winner: "ubatch-1024"},
		{ScopeKey: "scope-b", ModelBasename: "b.gguf", Winner: "default"},
	} {
		if _, err := SaveCalibrationDecision(dir, decision); err != nil {
			t.Fatalf("save %s: %v", decision.ScopeKey, err)
		}
	}
	if err := DeleteCalibrationDecisionsForModel(dir, &ModelProfile{Path: "/models/a.gguf"}); err != nil {
		t.Fatalf("delete model decisions: %v", err)
	}
	if _, err := LoadCalibrationDecision(dir, "scope-a"); err == nil {
		t.Fatal("target model calibration decision survived deletion")
	}
	if _, err := LoadCalibrationDecision(dir, "scope-b"); err != nil {
		t.Fatalf("other model calibration decision was removed: %v", err)
	}
}
