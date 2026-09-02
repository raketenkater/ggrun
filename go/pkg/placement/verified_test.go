package placement

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestVerifiedConfigRoundTrip(t *testing.T) {
	dir := t.TempDir()
	key := CalibrationScopeKey{
		ModelIdentity: "m", BackendIdentity: "b", HardwareID: "h",
		WorkloadProfile: "claude-agent-parallel-v1", ContextSize: 131072,
		Parallel: 4, UBatchSize: 256, KVQuality: "mid", SWAFull: true,
	}.String()

	s := &Strategy{
		Type:                     MoEOffload,
		HasSSM:                   true,
		IsMoE:                    true,
		ContextSize:              131072,
		KVPlacement:              "cpu",
		KVType:                   "q8_0",
		NCPUMoE:                  40,
		OTString:                 "blk.0-9.ffn_up,ffn_gate,ffn_down=GPU0,exps=CPU",
		MainGPU:                  0,
		TensorSplit:              []float64{0.6, 0.4},
		SplitMode:                "layer",
		BatchSize:                2048,
		UBatchSize:               512,
		Parallel:                 4,
		Threads:                  8,
		ThreadsBatch:             8,
		MMap:                     true,
		MMapRequired:             true,
		CPUExpertMMapCapability:  CPUExpertMMapFileBacked,
		CPUExpertMMapEvidence:    "live exact-build proof",
		ReclaimableHostWeightsMB: 81920,
		BatchTuned:               true,
		PerformanceTuned:         true,
		FlashAttention:           true,
		SWAFull:                  true,
		CRAM:                     4096,
		MeasuredCheckpointMB:     871.690,
		MaxCheckpoints:           8,
		BackendTag:               "llama",
		ModelBasename:            "V4.gguf",
	}
	vc := VerifiedConfigToRecord(key, "V4.gguf", s, "be-ident", "/path/to/llama", "qwen", "qwen3.5")
	if _, err := SaveVerifiedConfig(dir, vc); err != nil {
		t.Fatalf("save: %v", err)
	}
	loaded, err := LoadVerifiedConfig(dir, key)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.StrategyType != MoEOffload || loaded.NCPUMoE != 40 || loaded.ContextSize != 131072 {
		t.Fatalf("round trip lost strategy: %+v", loaded)
	}
	if loaded.OTString != s.OTString || loaded.ChatTemplate != "qwen" || loaded.Reviewer != "qwen3.5" {
		t.Fatalf("round trip lost fields: %+v", loaded)
	}
	if loaded.BackendIdentity != "be-ident" || loaded.BackendPath != "/path/to/llama" {
		t.Fatalf("provenance lost: %+v", loaded)
	}
	if !loaded.HasSSM || !loaded.IsMoE {
		t.Fatalf("model semantics lost: %+v", loaded)
	}
	if loaded.MeasuredCheckpointMB != s.MeasuredCheckpointMB {
		t.Fatalf("checkpoint measurement lost: got %.3f want %.3f", loaded.MeasuredCheckpointMB, s.MeasuredCheckpointMB)
	}
	if !loaded.MMapRequired || loaded.CPUExpertMMapCapability != CPUExpertMMapFileBacked ||
		loaded.CPUExpertMMapEvidence != "live exact-build proof" || loaded.ReclaimableHostWeightsMB != 81920 ||
		!loaded.BatchTuned || !loaded.PerformanceTuned || loaded.PerformanceEvidenceSchema != CalibrationSchemaVersion {
		t.Fatalf("optimizer/mmap evidence lost: %+v", loaded)
	}
	// The file lives under the verified-configs namespace with a hashed name.
	p := VerifiedConfigPath(dir, key)
	if filepath.Dir(p) != filepath.Join(dir, "verified-configs") {
		t.Fatalf("unexpected verified path %q", p)
	}
	if !strings.HasPrefix(filepath.Base(p), "verified-") || !strings.HasSuffix(filepath.Base(p), ".json") {
		t.Fatalf("unexpected verified filename %q", filepath.Base(p))
	}
}

func TestVerifiedConfigRequiresCurrentPerformanceEvidence(t *testing.T) {
	vc := &VerifiedConfig{
		StrategyType: SingleGPU, ContextSize: 8192, BatchSize: 2048, UBatchSize: 512,
		Parallel: 1, PerformanceTuned: true,
	}
	legacy := VerifiedToStrategy(vc, Options{}, nil)
	if legacy.PerformanceTuned {
		t.Fatal("a verified placement without current calibration evidence suppressed the optimizer")
	}
	vc.PerformanceEvidenceSchema = CalibrationSchemaVersion
	current := VerifiedToStrategy(vc, Options{}, nil)
	if !current.PerformanceTuned {
		t.Fatal("current performance evidence was not restored")
	}
}

func TestVerifiedConfigRoundTripsHotExpertShapeAndRuntimeEvidence(t *testing.T) {
	strategy := &Strategy{
		Type: MoEOffload, ContextSize: 32768, BatchSize: 2048, UBatchSize: 256,
		Parallel: 1, NCPUMoE: 20, OTString: "exps=CPU", BackendTag: "llama",
		HotExpertCacheSlots: 8, HotExpertCacheInserts: 2, HotExpertCacheLayers: 20,
		HotExpertCacheVRAMByGPU:   map[int]int{0: 1024, 2: 768},
		HotExpertCacheLayersByGPU: map[int]int{0: 12, 2: 8},
		HotExpertCacheEvidence:    "exact allocation; measured telemetry",
		HotExpertCacheSteps:       1024, HotExpertCacheHits: 75,
		HotExpertCacheMisses: 25, HotExpertCacheHitRate: 75,
	}
	record := VerifiedConfigToRecord("scope", "moe.gguf", strategy, "backend", "/server", "", "")
	restored := VerifiedToStrategy(&record, Options{
		BackendHelp: "--moe-expert-cache N\n--moe-expert-cache-inserts N",
	}, nil)
	if restored.HotExpertCacheSlots != 8 || restored.HotExpertCacheInserts != 2 ||
		restored.HotExpertCacheLayers != 20 || restored.HotExpertCacheVRAMByGPU[2] != 768 ||
		restored.HotExpertCacheLayersByGPU[0] != 12 || restored.HotExpertCacheSteps != 1024 ||
		restored.HotExpertCacheHits != 75 || restored.HotExpertCacheMisses != 25 ||
		restored.HotExpertCacheHitRate != 75 || !restored.BackendSupportsHotExpertCache {
		t.Fatalf("hot-expert verified evidence was not restored: %+v", restored)
	}
	restored.HotExpertCacheVRAMByGPU[0] = 1
	if record.HotExpertCacheVRAMByGPU[0] != 1024 {
		t.Fatal("verified hot-expert VRAM map aliases the runtime strategy")
	}

	unsupported := VerifiedToStrategy(&record, Options{BackendHelp: "--moe-expert-cache N"}, nil)
	if strings.Contains(strings.Join(unsupported.Args("m.gguf", 8081), " "), "--moe-expert-cache") {
		t.Fatal("verified cache shape emitted flags on a backend missing the exact capability pair")
	}
}

func TestVerifiedConfigRecomputesCurrentCPUAffinityCapabilities(t *testing.T) {
	vc := &VerifiedConfig{
		StrategyType: MoEOffload, ContextSize: 8192, BatchSize: 512, UBatchSize: 256,
		Parallel: 1, NCPUMoE: 8,
	}
	caps := &detect.Capabilities{CPU: detect.CPUInfo{Cores: 4, PhysicalIDs: []int{0, 1, 2, 3}}}
	s := VerifiedToStrategy(vc, Options{BackendHelp: "--cpu-range lo-hi\n--cpu-strict 0|1\n"}, caps)
	args := strings.Join(s.Args("m.gguf", 8081), " ")
	if !strings.Contains(args, "--cpu-range 0-3") || !strings.Contains(args, "--cpu-strict 1") {
		t.Fatalf("verified-config reuse lost current affinity capability: %s", args)
	}
}

func TestVerifiedConfigRejectsScopeMismatch(t *testing.T) {
	dir := t.TempDir()
	key := CalibrationScopeKey{
		ModelIdentity: "m", BackendIdentity: "b", HardwareID: "h",
		ContextSize: 131072, Parallel: 4, KVQuality: "mid",
	}.String()
	s := &Strategy{Type: SingleGPU, ContextSize: 131072, MainGPU: 0, BatchSize: 2048, UBatchSize: 512, Parallel: 4, MMap: true, FlashAttention: true, BackendTag: "llama"}
	vc := VerifiedConfigToRecord(key, "m.gguf", s, "b", "/p", "", "")
	if _, err := SaveVerifiedConfig(dir, vc); err != nil {
		t.Fatalf("save: %v", err)
	}
	// A different scope key must not load this record.
	if _, err := LoadVerifiedConfig(dir, "other-scope"); err == nil {
		t.Fatal("stale/foreign scope must not load a verified config")
	}
	// A record whose stored ScopeKey disagrees with the lookup must be rejected.
	// Write the file directly under one key while the record claims another.
	tampered := vc
	tampered.ScopeKey = "claimed-by-record"
	data, _ := json.MarshalIndent(tampered, "", "  ")
	if err := os.MkdirAll(filepath.Dir(VerifiedConfigPath(dir, key)), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(VerifiedConfigPath(dir, key), append(data, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadVerifiedConfig(dir, key); err == nil {
		t.Fatal("a tampered scope key must be rejected on load")
	}
}

func TestVerifiedConfigRejectsStaleSchema(t *testing.T) {
	dir := t.TempDir()
	key := "scopehash"
	s := &Strategy{Type: SingleGPU, ContextSize: 8192, MainGPU: 0, BatchSize: 2048, UBatchSize: 512, MMap: true, FlashAttention: true}
	vc := VerifiedConfigToRecord(key, "m.gguf", s, "b", "/p", "", "")
	vc.SchemaVersion = VerifiedConfigSchemaVersion + 1
	if _, err := SaveVerifiedConfig(dir, vc); err != nil {
		t.Fatalf("save: %v", err)
	}
	if _, err := LoadVerifiedConfig(dir, key); err == nil {
		t.Fatal("a stale-schema record must be rejected on load")
	}
}

func TestVerifiedConfigScopeKeyWidening(t *testing.T) {
	model := &ModelProfile{Path: "m.gguf", SizeBytes: 1 << 30, TotalSizeMB: 1024, ModelArch: "llama", NumLayers: 32, EmbeddingLength: 4096}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "3090", VRAMTotalMB: 24576, BandwidthMBps: 30000}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	base := &Strategy{Type: SingleGPU, KVPlacement: "gpu", MMap: true, ContextSize: 8192, BatchSize: 2048, UBatchSize: 512}

	opts := Options{ContextSize: 8192, KVQuality: "mid", KVPlacement: "gpu", Parallel: 1, BackendIdentity: "b", SWAFull: false}
	baseKey := NewCalibrationScopeKey(model, caps, opts, base).String()

	optsSWA := opts
	optsSWA.SWAFull = true
	swaKey := NewCalibrationScopeKey(model, caps, optsSWA, base).String()

	optsTemplate := opts
	optsTemplate.ChatTemplate = "qwen3.8"
	templateKey := NewCalibrationScopeKey(model, caps, optsTemplate, base).String()

	if baseKey == swaKey {
		t.Error("swa-full must widen the scope key")
	}
	if baseKey == templateKey {
		t.Error("a forced chat template must widen the scope key")
	}
	if swaKey == templateKey {
		t.Error("swa-full and template must produce distinct keys")
	}
	// Same opts produce a deterministic key.
	again := NewCalibrationScopeKey(model, caps, opts, base).String()
	if again != baseKey {
		t.Error("scope key must be deterministic")
	}
}

func TestVerifiedConfigStaleFreeVRAMGuard(t *testing.T) {
	dir := t.TempDir()
	key := "scopehash"
	s := &Strategy{Type: SingleGPU, ContextSize: 8192, MainGPU: 0, BatchSize: 2048, UBatchSize: 512, MMap: true, FlashAttention: true,
		PlanFreeVRAM: map[int]int{0: 4096}}
	vc := VerifiedConfigToRecord(key, "m.gguf", s, "b", "/p", "", "")
	if _, err := SaveVerifiedConfig(dir, vc); err != nil {
		t.Fatalf("save: %v", err)
	}
	// A GPU now showing much more free VRAM than at save time makes the record
	// stale-pessimistic.
	roomier := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 4096 + planFreeVRAMSlackMB + 500, VRAMUsedMB: 0}}}
	if _, stale := verifiedConfigFreeVRAMStale(&vc, roomier); !stale {
		t.Fatal("a plan computed under tighter VRAM must be flagged stale-pessimistic")
	}
	// A GPU with comparable or tighter VRAM is not stale.
	similar := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 4096 + planFreeVRAMSlackMB/2, VRAMUsedMB: 0}}}
	if _, stale := verifiedConfigFreeVRAMStale(&vc, similar); stale {
		t.Fatal("a plan computed under similar VRAM must not be flagged stale")
	}
}

func TestClearModelCachesRemovesVerifiedConfig(t *testing.T) {
	dir := t.TempDir()
	key := CalibrationScopeKey{
		ModelIdentity: "m", BackendIdentity: "b", HardwareID: "h",
		ContextSize: 8192, Parallel: 1, KVQuality: "mid",
	}.String()
	s := &Strategy{Type: SingleGPU, ContextSize: 8192, MainGPU: 0, BatchSize: 2048, UBatchSize: 512, MMap: true, FlashAttention: true}
	vc := VerifiedConfigToRecord(key, "target.gguf", s, "b", "/p", "", "")
	if _, err := SaveVerifiedConfig(dir, vc); err != nil {
		t.Fatalf("save: %v", err)
	}
	// Save a second record for a different model that must survive.
	otherKey := "otherscope"
	other := VerifiedConfigToRecord(otherKey, "other.gguf", s, "b", "/p", "", "")
	if _, err := SaveVerifiedConfig(dir, other); err != nil {
		t.Fatalf("save other: %v", err)
	}
	model := &ModelProfile{Path: filepath.Join(dir, "target.gguf"), Basename: "target.gguf"}
	files, err := ClearModelCaches(dir, model)
	if err != nil {
		t.Fatalf("clear: %v", err)
	}
	if files < 1 {
		t.Fatalf("expected at least 1 file removed, got %d", files)
	}
	if _, err := os.Stat(VerifiedConfigPath(dir, key)); !os.IsNotExist(err) {
		t.Fatal("target model's verified config should be removed")
	}
	if _, err := os.Stat(VerifiedConfigPath(dir, otherKey)); err != nil {
		t.Fatal("another model's verified config must survive a clear")
	}
}

func TestComputeReusesVerifiedConfigForDense(t *testing.T) {
	dir := t.TempDir()
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "3090", VRAMTotalMB: 24576, VRAMUsedMB: 512, BandwidthMBps: 30000}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: filepath.Join(dir, "target.gguf"), SizeBytes: 4 << 30, TotalSizeMB: 4096,
		ModelArch: "llama", NumLayers: 32, EmbeddingLength: 4096, IsMoE: false,
		ContextSize: 32768,
	}
	opts := Options{
		ContextSize: 8192, KVQuality: "mid", KVPlacement: "gpu", Parallel: 1,
		CacheDir: dir, BackendIdentity: "b", BackendTag: "llama", SWAFull: false,
	}
	key := NewCalibrationScopeKey(model, caps, opts, nil).String()

	// First launch: compute the estimate, then promote it to a verified config
	// (the launcher does this after StateActive).
	first, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("first compute: %v", err)
	}
	if first.PlacementCacheHit {
		t.Fatal("first launch must not be a reuse hit")
	}
	vc := VerifiedConfigToRecord(key, model.Basename, first, "b", "/p", "", "")
	if _, err := SaveVerifiedConfig(dir, vc); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Second launch with the exact same scope: must start directly from the
	// saved config — no re-estimate, PlacementCacheHit + VerifiedConfigReused.
	reuseOpts := opts
	reuseOpts.VerifiedConfigScopeKey = key
	second, err := Compute(caps, model, reuseOpts)
	if err != nil {
		t.Fatalf("reuse compute: %v", err)
	}
	if !second.PlacementCacheHit || !second.VerifiedConfigReused {
		t.Fatalf("matching-scope launch must reuse the verified config (hit=%v reused=%v)", second.PlacementCacheHit, second.VerifiedConfigReused)
	}
	if second.Type != first.Type || second.ContextSize != first.ContextSize || second.MainGPU != first.MainGPU {
		t.Fatalf("reused strategy diverged from saved: %+v vs %+v", second, first)
	}

	// A different scope key (different context) must NOT reuse.
	otherOpts := opts
	otherOpts.ContextSize = 16384
	otherOpts.VerifiedConfigScopeKey = NewCalibrationScopeKey(model, caps, otherOpts, nil).String()
	other, err := Compute(caps, model, otherOpts)
	if err != nil {
		t.Fatalf("other-scope compute: %v", err)
	}
	if other.VerifiedConfigReused {
		t.Fatal("a non-matching scope must not reuse the verified config")
	}
}

func TestComputeReusesVerifiedConfigForMoE(t *testing.T) {
	dir := t.TempDir()
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "3090", VRAMTotalMB: 24576, VRAMUsedMB: 512, BandwidthMBps: 31504},
			{Index: 1, Name: "3060", VRAMTotalMB: 12288, VRAMUsedMB: 512, BandwidthMBps: 12000},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 120000},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path: filepath.Join(dir, "moe.gguf"), SizeBytes: 146 << 30, TotalSizeMB: 146 * 1024,
		NumLayers: 43, IsMoE: true, NumExperts: 256,
		ExpertBytes: int64(43 * 3289 * 1024 * 1024), NonExpertBytes: int64(7680 * 1024 * 1024),
		ContextSize: 32768, EmbeddingLength: 4096, HeadCountKV: 1, KeyLength: 512, ValueLength: 512,
		ExpertUsedCount: 6, ExpertFF: 2048, ModelArch: "llama",
	}
	opts := Options{
		ContextSize: 8192, KVQuality: "low", KVPlacement: "cpu", Parallel: 1,
		CacheDir: dir, BackendIdentity: "b", BackendTag: "llama", SWAFull: false,
	}
	first, err := Compute(caps, model, opts)
	if err != nil || first.Type != MoEOffload {
		t.Fatalf("first compute: type=%v err=%v", first.Type, err)
	}
	if first.OTString == "" {
		t.Fatalf("MoE first compute produced no -ot")
	}
	key := NewCalibrationScopeKey(model, caps, opts, nil).String()
	vc := VerifiedConfigToRecord(key, model.Basename, first, "b", "/p", "", "")
	if _, err := SaveVerifiedConfig(dir, vc); err != nil {
		t.Fatalf("save: %v", err)
	}

	reuseOpts := opts
	reuseOpts.VerifiedConfigScopeKey = key
	second, err := Compute(caps, model, reuseOpts)
	if err != nil {
		t.Fatalf("reuse compute: %v", err)
	}
	if !second.VerifiedConfigReused {
		t.Fatal("matching-scope MoE launch must reuse the verified config")
	}
	if second.OTString != first.OTString || second.NCPUMoE != first.NCPUMoE {
		t.Fatalf("reused MoE placement diverged: ot=%q vs %q, ncpumoe=%d vs %d",
			second.OTString, first.OTString, second.NCPUMoE, first.NCPUMoE)
	}
}

func TestComputeVerifiedConfigRespectsNoCachedConfig(t *testing.T) {
	dir := t.TempDir()
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "3090", VRAMTotalMB: 24576, VRAMUsedMB: 512, BandwidthMBps: 30000}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: filepath.Join(dir, "target.gguf"), SizeBytes: 4 << 30, TotalSizeMB: 4096,
		ModelArch: "llama", NumLayers: 32, EmbeddingLength: 4096, IsMoE: false,
	}
	opts := Options{
		ContextSize: 8192, KVQuality: "mid", KVPlacement: "gpu", Parallel: 1,
		CacheDir: dir, BackendIdentity: "b", BackendTag: "llama",
	}
	key := NewCalibrationScopeKey(model, caps, opts, nil).String()
	first, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("first compute: %v", err)
	}
	vc := VerifiedConfigToRecord(key, model.Basename, first, "b", "/p", "", "")
	if _, err := SaveVerifiedConfig(dir, vc); err != nil {
		t.Fatalf("save: %v", err)
	}

	// --no-cached-config (SkipCachedConfig) must bypass verified-config reuse.
	skipOpts := opts
	skipOpts.SkipCachedConfig = true
	skipOpts.VerifiedConfigScopeKey = key
	skipped, err := Compute(caps, model, skipOpts)
	if err != nil {
		t.Fatalf("skip compute: %v", err)
	}
	if skipped.VerifiedConfigReused {
		t.Fatal("--no-cached-config must bypass verified-config reuse")
	}
}

func TestVerifiedConfigToStrategyRestoresRequestOwnedKnobs(t *testing.T) {
	vc := &VerifiedConfig{
		StrategyType:   MoEOffload,
		HasSSM:         true,
		IsMoE:          true,
		ContextSize:    8192,
		KVPlacement:    "gpu",
		KVType:         "q8_0",
		NCPUMoE:        30,
		OTString:       "exps=CPU",
		BatchSize:      1024,
		UBatchSize:     256,
		Parallel:       2,
		Threads:        4,
		ThreadsBatch:   4,
		MMap:           true,
		FlashAttention: true,
		BackendTag:     "llama",
		MainGPU:        0,
		TensorSplit:    []float64{0.6, 0.4},
	}
	// The caller's requested knobs win (placement.go:1026-1034 generalized).
	opts := Options{Parallel: 4, BatchSize: 4096, UBatchSize: 512, Threads: 8, NoMMap: true}
	caps := &detect.Capabilities{CPU: detect.CPUInfo{Cores: 16}}
	s := VerifiedToStrategy(vc, opts, caps)
	if s.Parallel != 4 || s.BatchSize != 4096 || s.UBatchSize != 512 {
		t.Fatalf("request-owned knobs must win: parallel=%d batch=%d ubatch=%d", s.Parallel, s.BatchSize, s.UBatchSize)
	}
	if s.Threads != 8 || s.ThreadsBatch != 8 {
		t.Fatalf("requested threads must win: threads=%d batch=%d", s.Threads, s.ThreadsBatch)
	}
	if s.MMap {
		t.Fatal("an explicit user --no-mmap must win over the saved mmap value")
	}
	if s.NCPUMoE != 30 || s.OTString != "exps=CPU" || s.Type != MoEOffload {
		t.Fatalf("placement identity must restore verbatim: %+v", s)
	}
	if !s.HasSSM || !s.IsMoE {
		t.Fatalf("model semantics must restore verbatim: %+v", s)
	}
	// The saved values are used when the caller did not request overrides.
	opts2 := Options{BackendHelp: "--fit [on|off]\n--kv-offload\n"}
	s2 := VerifiedToStrategy(vc, opts2, caps)
	if s2.BatchSize != 1024 || s2.UBatchSize != 256 || s2.Parallel != 2 {
		t.Fatalf("saved knobs must apply when caller has no override: %+v", s2)
	}
	if len(s2.TensorSplit) != 2 || s2.TensorSplit[0] != 0.6 {
		t.Fatalf("tensor split not restored: %v", s2.TensorSplit)
	}
	args := strings.Join(s2.Args("model.gguf", 8081), " ")
	if !strings.Contains(args, "--kv-offload") || !strings.Contains(args, "--fit off") {
		t.Fatalf("verified replay dropped current-backend placement semantics: %q", args)
	}
}
