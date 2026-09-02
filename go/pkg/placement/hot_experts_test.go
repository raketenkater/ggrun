package placement

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

const hotExpertTestMiB = int64(1024 * 1024)

func hotExpertFixture() (*detect.Capabilities, *ModelProfile, *Strategy, Options) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 3, Name: "router-a", VRAMTotalMB: 24576},
			{Index: 7, Name: "router-b", VRAMTotalMB: 24576},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU: detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path: "moe.gguf", Basename: "moe.gguf", NumLayers: 4, IsMoE: true,
		NumExperts: 4, ExpertUsedCount: 2,
		RoutedExpertLayerBytes: []int64{
			4 * hotExpertTestMiB,
			8 * hotExpertTestMiB,
			8 * hotExpertTestMiB,
			12 * hotExpertTestMiB,
		},
	}
	gpus := caps.GPUs
	strategy := &Strategy{
		Type: MoEOffload, NCPUMoE: 3, Parallel: 1, MainGPU: 3,
		TensorSplit: []float64{1, 1},
		OTString: buildOTStringWithSubPins(
			[]int{1, 0}, []subExpertPin{{Layer: 1, GI: 1}},
			gpus, []int{0, 1}, 0, "llama",
		),
	}
	opts := Options{BackendHelp: "--moe-expert-cache N\n--moe-expert-cache-inserts N"}
	return caps, model, strategy, opts
}

func TestBackendSupportsHotExpertCacheRequiresBothExactFlags(t *testing.T) {
	if !BackendSupportsHotExpertCache("--moe-expert-cache N\n--moe-expert-cache-inserts N") {
		t.Fatal("exact capability pair was not detected")
	}
	for _, help := range []string{
		"--moe-expert-cache N",
		"--moe-expert-cache-inserts N",
		"--moe-expert-cache-extra N\n--moe-expert-cache-inserts-extra N",
	} {
		if BackendSupportsHotExpertCache(help) {
			t.Fatalf("near/missing capability was accepted: %q", help)
		}
	}
}

func TestHotExpertPinnedLayersClassifiesGeneratedWholeAndPartialRules(t *testing.T) {
	_, _, strategy, _ := hotExpertFixture()
	whole, partial, err := hotExpertPinnedLayers(strategy.OTString)
	if err != nil {
		t.Fatal(err)
	}
	if !whole[0] || len(whole) != 1 || !partial[1] || len(partial) != 1 {
		t.Fatalf("whole=%v partial=%v ot=%s", whole, partial, strategy.OTString)
	}
	unsafe := `blk\.(0)\.ffn_((gate|up)_(ch|)exps|gate_tid2eid|exp_probs_b).*=CUDA3,exps=CPU`
	if _, _, err := hotExpertPinnedLayers(unsafe); err == nil {
		t.Fatal("router auxiliaries without the exact complete expert rule were trusted")
	}
}

func TestHotExpertCacheShapeExcludesPinnedLayersAndUsesPhysicalRouterOwners(t *testing.T) {
	caps, model, strategy, opts := hotExpertFixture()
	shape, err := hotExpertCacheShapeFor(caps, model, strategy, opts)
	if err != nil {
		t.Fatal(err)
	}
	if shape.layers != 2 || shape.layersByGPU[3] != 1 || shape.layersByGPU[7] != 1 {
		t.Fatalf("unexpected host-layer routing: %+v", shape)
	}
	if shape.perSlotBytesByGPU[3] != 2*hotExpertTestMiB || shape.perSlotBytesByGPU[7] != 3*hotExpertTestMiB {
		t.Fatalf("unexpected expert-slice strides: %+v", shape.perSlotBytesByGPU)
	}
	wantFixed3 := 2*hotExpertTestMiB + 256 + 4*hotExpertCacheAlignmentBytes
	wantFixed7 := 3*hotExpertTestMiB + 256 + 4*hotExpertCacheAlignmentBytes
	if shape.fixedBytesByGPU[3] != wantFixed3 || shape.fixedBytesByGPU[7] != wantFixed7 {
		t.Fatalf("unexpected dummy/table/alignment charge: %+v", shape.fixedBytesByGPU)
	}
}

func TestHotExpertCacheEligibilityFailsClosed(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*ModelProfile, *Strategy, *Options)
	}{
		{"missing-backend-capability", func(_ *ModelProfile, _ *Strategy, o *Options) { o.BackendHelp = "--moe-expert-cache N" }},
		{"dense", func(m *ModelProfile, _ *Strategy, _ *Options) { m.IsMoE = false }},
		{"not-offloaded", func(_ *ModelProfile, s *Strategy, _ *Options) { s.NCPUMoE = 0 }},
		{"mmap-last-resort", func(_ *ModelProfile, s *Strategy, _ *Options) { s.MMapRequired = true }},
		{"parallel-slots", func(_ *ModelProfile, s *Strategy, _ *Options) { s.Parallel = 2 }},
		{"row-split", func(_ *ModelProfile, s *Strategy, _ *Options) { s.SplitMode = "row" }},
		{"speculative", func(_ *ModelProfile, s *Strategy, _ *Options) { s.Draft = &DraftConfig{Type: DraftNgram} }},
		{"fused-gate-up", func(m *ModelProfile, _ *Strategy, _ *Options) { m.Fused = 1 }},
		{"missing-layer-bytes", func(m *ModelProfile, _ *Strategy, _ *Options) { m.RoutedExpertLayerBytes = nil }},
		{"missing-cpu-catch-all", func(_ *ModelProfile, s *Strategy, _ *Options) { s.OTString = "" }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			caps, model, strategy, opts := hotExpertFixture()
			tc.mutate(model, strategy, &opts)
			if shape, err := hotExpertCacheShapeFor(caps, model, strategy, opts); err == nil || shape != nil {
				t.Fatalf("ineligible shape was accepted: shape=%+v err=%v", shape, err)
			}
		})
	}
}

func TestHotExpertCacheLayoutAndTightestDeviceSlotBound(t *testing.T) {
	shape := &hotExpertCacheShape{
		perSlotBytesByGPU: map[int]int64{3: 10 * hotExpertTestMiB, 7: 20 * hotExpertTestMiB},
		fixedBytesByGPU:   map[int]int64{3: 5 * hotExpertTestMiB, 7: 10 * hotExpertTestMiB},
		layersByGPU:       map[int]int{3: 1, 7: 1}, layers: 2,
	}
	layout, err := hotExpertCacheLayout(shape, 4)
	if err != nil {
		t.Fatal(err)
	}
	if layout[3] != 45 || layout[7] != 90 {
		t.Fatalf("layout=%v", layout)
	}
	ledger := ResourceLedger{Exact: true, Fits: true, Devices: []DeviceResourceLedger{
		{GPU: 3, SlackMB: 105}, {GPU: 7, SlackMB: 90},
	}}
	model := &ModelProfile{NumExperts: 8, ExpertUsedCount: 2}
	if got := hotExpertCacheMaxSlots(model, shape, ledger); got != 4 {
		t.Fatalf("max slots=%d, want tightest-device bound 4", got)
	}
	model.ExpertUsedCount = 5
	if got := hotExpertCacheMaxSlots(model, shape, ledger); got != 0 {
		t.Fatalf("sub-active-expert cache should be rejected, got %d slots", got)
	}
	ledger.Exact = false
	model.ExpertUsedCount = 2
	if got := hotExpertCacheMaxSlots(model, shape, ledger); got != 0 {
		t.Fatalf("estimated ledger admitted %d slots", got)
	}
	overflowShape := &hotExpertCacheShape{
		perSlotBytesByGPU: map[int]int64{3: 2}, fixedBytesByGPU: map[int]int64{3: 0},
	}
	if _, err := hotExpertCacheLayout(overflowShape, int(^uint(0)>>1)); err == nil {
		t.Fatal("overflowing explicit cache geometry was accepted")
	}
}

func TestExplicitHotExpertSlotsCannotExceedModelExpertCount(t *testing.T) {
	caps, model, strategy, opts := hotExpertFixture()
	opts.HotExperts = "5"
	opts.HotExpertCacheSlots = 5
	if got, err := finalizeHotExpertCache(caps, model, opts, strategy); err == nil || got != nil {
		t.Fatalf("over-wide explicit cache was accepted: strategy=%+v err=%v", got, err)
	}
}

func TestRequiredHotExpertsDoesNotDegradeToCacheFree(t *testing.T) {
	caps, model, strategy, opts := hotExpertFixture()
	opts.HotExperts = "auto"
	if got, err := finalizeHotExpertCache(caps, model, opts, cloneStrategy(strategy)); err != nil || got == nil {
		t.Fatalf("auto should retain its cache-free fallback without exact evidence: strategy=%+v err=%v", got, err)
	}
	opts.HotExperts = "on"
	if got, err := finalizeHotExpertCache(caps, model, opts, cloneStrategy(strategy)); err == nil || got != nil {
		t.Fatalf("required hot experts silently degraded cache-free: strategy=%+v err=%v", got, err)
	}
}

func TestAutomaticHotExpertCandidateRequiresExactCacheFreeAllocation(t *testing.T) {
	caps, model, base, opts := hotExpertFixture()
	dir := t.TempDir()
	model.Path = filepath.Join(dir, "moe.gguf")
	model.Basename = "moe.gguf"
	model.SizeBytes = 32 * hotExpertTestMiB
	model.TotalSizeMB = 32
	model.ExpertBytes = model.SizeBytes
	if err := os.WriteFile(model.Path, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	base.ContextSize = 4096
	base.BatchSize = 512
	base.UBatchSize = 128
	base.KVPlacement = "cpu"
	base.KVQuality = "high"
	base.KVType = "q8_0"
	base.PlanFreeVRAM = map[int]int{3: 24576, 7: 24576}
	opts.CacheDir = dir
	opts.BackendCacheTag = "test-hot"
	opts.HotExperts = "auto"

	withoutMeasurement := CalibrationCandidates(caps, model, base, opts)
	for _, candidate := range withoutMeasurement[1:] {
		if strings.HasPrefix(candidate.Name, "hot-experts-") {
			t.Fatalf("estimated baseline generated automatic cache candidate: %+v", candidate)
		}
	}
	if len(base.OptimizationExclusions) == 0 || !strings.Contains(base.OptimizationExclusions[0], "exact cache-free") {
		t.Fatalf("missing hot-expert exclusion reason: %+v", base.OptimizationExclusions)
	}

	if err := RecordMeasuredAllocation(
		dir, model, base.ContextSize, base.UBatchSize, base.KVQuality,
		base.KVPlacement, backendCacheTag(opts), caps.GPUs, base.Parallel,
		MeasuredAllocation{
			Evidence: "allocation-verified", PlacementIdentity: AllocationPlacementIdentity(base),
			ContextTotalMB: 100, ContextHostMB: 100,
			ModelByGPU:  map[int]int{3: 1000, 7: 1000},
			ModelHostMB: 100, UnaccountedByGPU: map[int]int{3: 100, 7: 100},
		},
	); err != nil {
		t.Fatal(err)
	}
	withMeasurement := CalibrationCandidates(caps, model, base, opts)
	if len(withMeasurement) < 2 || withMeasurement[1].Name != "hot-experts-4" ||
		withMeasurement[1].Strategy == nil || withMeasurement[1].Strategy.HotExpertCacheSlots != 4 ||
		withMeasurement[1].Strategy.ResourceLedger == nil || !withMeasurement[1].Strategy.ResourceLedger.Exact ||
		!withMeasurement[1].Strategy.ResourceLedger.Fits {
		t.Fatalf("exact residual headroom did not produce the bounded first challenger: %+v", withMeasurement)
	}
}

func TestHotExpertPriorityDemotesGPULayersToReserveMinSlots(t *testing.T) {
	caps, model, base, opts := hotExpertFixture()
	dir := t.TempDir()
	model.Path = filepath.Join(dir, "moe.gguf")
	model.Basename = "moe.gguf"
	model.SizeBytes = 32 * hotExpertTestMiB
	model.TotalSizeMB = 32
	model.ExpertBytes = model.SizeBytes
	model.ExpertUsedCount = 2
	if err := os.WriteFile(model.Path, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	base.ContextSize = 4096
	base.BatchSize = 512
	base.UBatchSize = 128
	base.KVPlacement = "cpu"
	base.KVQuality = "high"
	base.KVType = "q8_0"
	base.NCPUMoE = 2
	base.PlanFreeVRAM = map[int]int{3: 24576, 7: 24576}
	opts.CacheDir = dir
	opts.BackendCacheTag = "test-hot-priority"
	opts.HotExperts = "auto"
	if err := RecordMeasuredAllocation(
		dir, model, base.ContextSize, base.UBatchSize, base.KVQuality,
		base.KVPlacement, backendCacheTag(opts), caps.GPUs, base.Parallel,
		MeasuredAllocation{
			Evidence: "allocation-verified", PlacementIdentity: AllocationPlacementIdentity(base),
			ContextTotalMB: 100, ContextHostMB: 100,
			ModelByGPU:       map[int]int{3: 24500, 7: 1000},
			UnaccountedByGPU: map[int]int{3: 50, 7: 50},
			ModelHostMB:      100,
		},
	); err != nil {
		t.Fatal(err)
	}
	got := CalibrationCandidates(caps, model, base, opts)
	if len(got) < 2 || !strings.HasPrefix(got[1].Name, "hot-experts-") {
		t.Fatalf("auto hot experts did not produce a cache-on challenger: %+v", got)
	}
	hot := got[1].Strategy
	if hot == nil || hot.HotExpertCacheSlots < 2 {
		t.Fatalf("priority challenger lacked a useful cache: %+v", hot)
	}
	demoted := hot.NCPUMoE > base.NCPUMoE || hot.OTString != base.OTString
	if hot.ResourceLedger == nil {
		t.Fatal("priority challenger has no resource ledger")
	}
	if demoted && (hot.ResourceLedger.Exact ||
		!strings.Contains(hot.ResourceLedger.Evidence, "derived-expert-demotion")) {
		t.Fatalf("changed expert residency was mislabeled as exact allocation evidence: %+v", hot.ResourceLedger)
	}
	for _, device := range hot.ResourceLedger.Devices {
		if device.RequiredMB+device.SlackMB != device.FreeMB {
			t.Fatalf("GPU%d priority ledger is internally inconsistent: %+v", device.GPU, device)
		}
	}
	if demoted && (hot.ResourceLedger.Host.RequiredMB <= 0 ||
		hot.ResourceLedger.Host.RequiredMB+hot.ResourceLedger.Host.SlackMB != hot.ResourceLedger.Host.FreeMB) {
		t.Fatalf("demoted expert bytes were not charged consistently to host: %+v", hot.ResourceLedger.Host)
	}
	if hot.NCPUMoE <= base.NCPUMoE && !strings.Contains(hot.HotExpertCacheEvidence, "priority") &&
		hot.OTString == base.OTString {
		// Leftover may already fit after demotion is unnecessary; still require
		// a cache-on first challenger while auto is on.
		if hot.HotExpertCacheSlots < 2 {
			t.Fatalf("auto path did not turn hot experts on: %+v", hot)
		}
	}
	if got[0].Name != "default" || got[0].Strategy.HotExpertCacheSlots != 0 {
		t.Fatalf("packed cache-free baseline must remain candidate 0: %+v", got[0])
	}
	analyzed := AnalyzeCandidateFrontier(caps, model, opts, got)
	var analyzedHot *CalibrationCandidate
	for i := range analyzed {
		if strings.HasPrefix(analyzed[i].Name, "hot-experts-") {
			analyzedHot = &analyzed[i]
			break
		}
	}
	if analyzedHot == nil || analyzedHot.Strategy == nil || !analyzedHot.Estimate.Feasible {
		t.Fatalf("priority hot-expert challenger disappeared from the analyzed frontier: %+v", analyzed)
	}
	if demoted && (analyzedHot.Strategy.ResourceLedger == nil || analyzedHot.Strategy.ResourceLedger.Exact) {
		t.Fatalf("frontier analysis reused cache-free allocation authority for changed residency: %+v", analyzedHot.Strategy.ResourceLedger)
	}
}

func TestHotExpertDemotePinnedLayerRemovesHighestLayer(t *testing.T) {
	_, _, strategy, _ := hotExpertFixture()
	strategy.VRAMLedger = []GPULedgerEntry{{GPU: 3, ExpertLayers: 2}, {GPU: 7, ExpertLayers: 1}}
	before := strategy.NCPUMoE
	if !hotExpertDemotePinnedLayer(strategy, 0) {
		t.Fatal("expected to demote wholly pinned layer 0")
	}
	if strategy.NCPUMoE != before+1 {
		t.Fatalf("NCPUMoE=%d, want %d", strategy.NCPUMoE, before+1)
	}
	whole, _, err := hotExpertPinnedLayers(strategy.OTString)
	if err != nil {
		t.Fatal(err)
	}
	if whole[0] {
		t.Fatalf("layer 0 still pinned: %s", strategy.OTString)
	}
	if strategy.VRAMLedger[0].ExpertLayers != 1 || strategy.VRAMLedger[1].ExpertLayers != 1 {
		t.Fatalf("VRAM ledger was not updated for the demoted GPU: %+v", strategy.VRAMLedger)
	}
}

func TestFinalizeAutoHotExpertsAppliesDemotedCacheOn(t *testing.T) {
	caps, model, base, opts := hotExpertFixture()
	dir := t.TempDir()
	model.Path = filepath.Join(dir, "moe.gguf")
	model.Basename = "moe.gguf"
	model.SizeBytes = 32 * hotExpertTestMiB
	model.TotalSizeMB = 32
	model.ExpertBytes = model.SizeBytes
	model.ExpertUsedCount = 2
	if err := os.WriteFile(model.Path, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	base.ContextSize = 4096
	base.BatchSize = 512
	base.UBatchSize = 128
	base.KVPlacement = "cpu"
	base.KVQuality = "high"
	base.KVType = "q8_0"
	base.NCPUMoE = 2
	base.VRAMLedger = []GPULedgerEntry{{GPU: 3, ExpertLayers: 1}, {GPU: 7, ExpertLayers: 0}}
	base.PlanFreeVRAM = map[int]int{3: 24576, 7: 24576}
	opts.CacheDir = dir
	opts.BackendCacheTag = "test-hot-finalize-auto"
	opts.HotExperts = "auto"
	if err := RecordMeasuredAllocation(
		dir, model, base.ContextSize, base.UBatchSize, base.KVQuality,
		base.KVPlacement, backendCacheTag(opts), caps.GPUs, base.Parallel,
		MeasuredAllocation{
			Evidence: "allocation-verified", PlacementIdentity: AllocationPlacementIdentity(base),
			ContextTotalMB: 100, ContextHostMB: 100,
			ModelByGPU:       map[int]int{3: 24500, 7: 1000},
			UnaccountedByGPU: map[int]int{3: 50, 7: 50},
			ModelHostMB:      100,
		},
	); err != nil {
		t.Fatal(err)
	}
	got, err := finalizeHotExpertCache(caps, model, opts, base)
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || got.HotExpertCacheSlots < 2 {
		t.Fatalf("auto finalize did not turn the cache on: %+v err=%v", got, err)
	}
	args := strings.Join(got.Args("moe.gguf", 8080), " ")
	if !strings.Contains(args, "--moe-expert-cache") {
		t.Fatalf("auto cache-on strategy omitted the backend flag: %s", args)
	}
}

func TestHotExpertDerivedDemotionLedgerMovesBytesToHostAndLosesExactness(t *testing.T) {
	base := ResourceLedger{
		Exact: true, Fits: true, Evidence: "live-allocated",
		Devices: []DeviceResourceLedger{{GPU: 3, FreeMB: 1000, ModelMB: 700, RequiredMB: 900, SlackMB: 100}},
		Host:    HostResourceLedger{FreeMB: 5000, ModelMB: 1000, RequiredMB: 1500, SlackMB: 3500},
	}
	got := hotExpertLedgerWithSlack(base, map[int]int{3: 164})
	if got.Exact || !got.Fits || !strings.Contains(got.Evidence, "derived-expert-demotion") {
		t.Fatalf("derived ledger authority=%+v", got)
	}
	dev := got.Devices[0]
	if dev.ModelMB != 636 || dev.RequiredMB != 836 || dev.SlackMB != 164 || dev.RequiredMB+dev.SlackMB != dev.FreeMB {
		t.Fatalf("device bytes were not moved exactly once: %+v", dev)
	}
	if got.Host.ModelMB != 1064 || got.Host.RequiredMB != 1564 || got.Host.SlackMB != 3436 ||
		got.Host.RequiredMB+got.Host.SlackMB != got.Host.FreeMB {
		t.Fatalf("host did not receive demoted bytes exactly once: %+v", got.Host)
	}
}

func TestFinalizeHotExpertCachePreservesVerifiedRuntimeEvidence(t *testing.T) {
	caps, model, strategy, opts := hotExpertFixture()
	dir := t.TempDir()
	model.Path = filepath.Join(dir, "moe.gguf")
	model.Basename = "moe.gguf"
	model.SizeBytes = 32 * hotExpertTestMiB
	model.TotalSizeMB = 32
	model.ExpertBytes = model.SizeBytes
	if err := os.WriteFile(model.Path, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	strategy.ContextSize = 4096
	strategy.BatchSize = 512
	strategy.UBatchSize = 128
	strategy.KVPlacement = "cpu"
	strategy.KVQuality = "high"
	strategy.KVType = "q8_0"
	strategy.PlanFreeVRAM = map[int]int{3: 24576, 7: 24576}
	opts.CacheDir = dir
	opts.BackendCacheTag = "test-hot"
	opts.HotExperts = "auto"
	if err := RecordMeasuredAllocation(
		dir, model, strategy.ContextSize, strategy.UBatchSize, strategy.KVQuality,
		strategy.KVPlacement, backendCacheTag(opts), caps.GPUs, strategy.Parallel,
		MeasuredAllocation{
			Evidence: "allocation-verified", PlacementIdentity: AllocationPlacementIdentity(strategy),
			ContextTotalMB: 100, ContextHostMB: 100,
			ModelByGPU: map[int]int{3: 1000, 7: 1000}, ModelHostMB: 100,
			UnaccountedByGPU: map[int]int{3: 100, 7: 100},
		},
	); err != nil {
		t.Fatal(err)
	}
	shape, err := hotExpertCacheShapeFor(caps, model, strategy, opts)
	if err != nil {
		t.Fatal(err)
	}
	if err := assignHotExpertCache(strategy, shape, 4); err != nil {
		t.Fatal(err)
	}
	strategy.HotExpertCacheEvidence += "; measured aggregate runtime hit/miss telemetry"
	strategy.HotExpertCacheSteps = 1024
	strategy.HotExpertCacheHits = 75
	strategy.HotExpertCacheMisses = 25
	strategy.HotExpertCacheHitRate = 75

	got, err := finalizeHotExpertCache(caps, model, opts, strategy)
	if err != nil {
		t.Fatal(err)
	}
	if got.HotExpertCacheSteps != 1024 || got.HotExpertCacheHits != 75 ||
		got.HotExpertCacheMisses != 25 || got.HotExpertCacheHitRate != 75 ||
		!strings.Contains(got.HotExpertCacheEvidence, "measured aggregate") {
		t.Fatalf("revalidation erased verified runtime evidence: %+v", got)
	}
}

func TestHotExpertCacheLedgerChargesExactlyOnce(t *testing.T) {
	strategy := &Strategy{HotExpertCacheSlots: 4, HotExpertCacheVRAMByGPU: map[int]int{3: 45, 7: 90}}
	base := ResourceLedger{Exact: true, Fits: true, Evidence: "allocation", Devices: []DeviceResourceLedger{
		{GPU: 3, RequiredMB: 1000, SlackMB: 200},
		{GPU: 7, RequiredMB: 2000, SlackMB: 100},
	}}
	ledger := base
	ledger.Devices = append([]DeviceResourceLedger(nil), base.Devices...)
	applyHotExpertCacheLedger(&ledger, strategy, false)
	if ledger.Devices[0].RequiredMB != 1045 || ledger.Devices[0].SlackMB != 155 || ledger.Devices[0].HotExpertCacheMB != 45 ||
		ledger.Devices[1].RequiredMB != 2090 || ledger.Devices[1].SlackMB != 10 || ledger.Devices[1].HotExpertCacheMB != 90 {
		t.Fatalf("cache charge=%+v", ledger.Devices)
	}
	alreadyMeasured := base
	alreadyMeasured.Devices = append([]DeviceResourceLedger(nil), base.Devices...)
	applyHotExpertCacheLedger(&alreadyMeasured, strategy, true)
	if alreadyMeasured.Devices[0].RequiredMB != 1000 || alreadyMeasured.Devices[0].HotExpertCacheMB != 0 {
		t.Fatalf("measured cache allocation was charged twice: %+v", alreadyMeasured.Devices)
	}
}

func TestHotExpertStartupObservationValidation(t *testing.T) {
	strategy := &Strategy{
		HotExpertCacheSlots: 4, HotExpertCacheInserts: 2, HotExpertCacheLayers: 2,
		HotExpertCacheVRAMByGPU: map[int]int{3: 45, 7: 90},
	}
	valid := "prefix MoE expert cache enabled: 2 layers x 4 slots, 2 inserts/step, 134.5 MiB device memory suffix"
	if err := ValidateHotExpertCacheObservation(strategy, valid); err != nil {
		t.Fatal(err)
	}
	for _, logData := range []string{
		"MoE expert cache enabled: 2 layers x 3 slots, 2 inserts/step, 100.0 MiB device memory",
		"MoE expert cache enabled: 2 layers x 4 slots, 2 inserts/step, 200.0 MiB device memory",
		"failed to allocate moe cache buffer",
		"unrelated startup output",
	} {
		if err := ValidateHotExpertCacheObservation(strategy, logData); err == nil {
			t.Fatalf("invalid activation was accepted: %q", logData)
		}
	}
}

func TestHotExpertRuntimeTelemetryUsesLastConsistentWindow(t *testing.T) {
	strategy := &Strategy{HotExpertCacheSlots: 4}
	logData := "moe-cache: steps=512 hits=10 misses=90 hit-rate=10.0%\n" +
		"debug moe-cache: steps=1024 hits=75 misses=25 hit-rate=75.0%"
	telemetry, err := ValidateHotExpertCacheTelemetry(strategy, logData)
	if err != nil {
		t.Fatal(err)
	}
	if telemetry.Steps != 1024 || telemetry.Hits != 75 || telemetry.Misses != 25 || telemetry.HitRate != 75 {
		t.Fatalf("telemetry=%+v", telemetry)
	}
	for _, invalid := range []string{
		"unrelated log",
		"moe-cache: steps=511 hits=1 misses=1 hit-rate=50.0%",
		"moe-cache: steps=512 hits=0 misses=100 hit-rate=0.0%",
		"moe-cache: steps=512 hits=75 misses=25 hit-rate=70.0%",
	} {
		if got, err := ValidateHotExpertCacheTelemetry(strategy, invalid); err == nil || (got.Observed && math.IsNaN(got.HitRate)) {
			t.Fatalf("invalid telemetry was accepted: %q => %+v, %v", invalid, got, err)
		}
	}
}

func TestHotExpertArgsAndCacheScopeAreCapabilityGated(t *testing.T) {
	strategy := &Strategy{
		HotExpertCacheSlots: 4, HotExpertCacheInserts: 2,
		BackendSupportsHotExpertCache: true,
	}
	args := strings.Join(strategy.Args("model.gguf", 8080), " ")
	if !strings.Contains(args, "--moe-expert-cache 4") || !strings.Contains(args, "--moe-expert-cache-inserts 2") {
		t.Fatalf("hot-expert flags missing: %s", args)
	}
	strategy.BackendSupportsHotExpertCache = false
	if args := strings.Join(strategy.Args("model.gguf", 8080), " "); strings.Contains(args, "--moe-expert-cache") {
		t.Fatalf("unsupported backend received hot-expert flags: %s", args)
	}

	tag := ScopedBackendRuntimeFeatureTag("llama|workload=agent|hot-experts=4,inserts=2", false, 8, 3)
	if strings.Count(tag, "hot-experts=") != 1 || !strings.Contains(tag, "hot-experts=8,inserts=3") {
		t.Fatalf("stale runtime feature scope survived: %s", tag)
	}
	if off := ScopedBackendRuntimeFeatureTag(tag, false, 0, 0); strings.Contains(off, "hot-experts=") {
		t.Fatalf("cache-free scope retained hot-expert allocation tag: %s", off)
	}

	caps, model, base, opts := hotExpertFixture()
	opts.BackendIdentity = "backend-build"
	opts.HotExperts = "off"
	offKey := NewCalibrationScopeKey(model, caps, opts, base).String()
	opts.HotExperts = "auto"
	autoKey := NewCalibrationScopeKey(model, caps, opts, base).String()
	opts.HotExperts = "8"
	opts.HotExpertCacheSlots = 8
	numericKey := NewCalibrationScopeKey(model, caps, opts, base).String()
	if offKey == autoKey || offKey == numericKey || autoKey == numericKey {
		t.Fatalf("hot-expert policies shared calibration scope: off=%s auto=%s numeric=%s", offKey, autoKey, numericKey)
	}
}

func TestWithoutHotExpertCacheDeepCopiesAndClearsEvidence(t *testing.T) {
	original := &Strategy{
		HotExpertCacheSlots: 4, HotExpertCacheInserts: 2, HotExpertCacheLayers: 2,
		HotExpertCacheVRAMByGPU: map[int]int{3: 45}, HotExpertCacheLayersByGPU: map[int]int{3: 1},
		HotExpertCacheEvidence: "planned; measured", HotExpertCacheSteps: 512,
		HotExpertCacheHits: 10, HotExpertCacheMisses: 90, HotExpertCacheHitRate: 10,
	}
	copy := WithoutHotExpertCache(original)
	if copy == original || copy.HotExpertCacheSlots != 0 || copy.HotExpertCacheVRAMByGPU != nil ||
		copy.HotExpertCacheSteps != 0 || copy.HotExpertCacheEvidence != "" {
		t.Fatalf("cache-free copy=%+v", copy)
	}
	if original.HotExpertCacheSlots != 4 || original.HotExpertCacheVRAMByGPU[3] != 45 || original.HotExpertCacheSteps != 512 {
		t.Fatalf("original was mutated: %+v", original)
	}
}
