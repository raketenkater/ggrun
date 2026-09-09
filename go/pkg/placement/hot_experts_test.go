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
		Type: MoEOffload, NCPUMoE: 3, Parallel: 1, MainGPU: 3, GPULayers: 999,
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

func TestRequiredHotExpertsBootstrapsAnUnmeasuredBaselineButFailsClosedOtherwise(t *testing.T) {
	caps, model, strategy, opts := hotExpertFixture()
	opts.HotExperts = "auto"
	if got, err := finalizeHotExpertCache(caps, model, opts, cloneStrategy(strategy)); err != nil || got == nil {
		t.Fatalf("auto should retain its cache-free fallback without exact evidence: strategy=%+v err=%v", got, err)
	}

	// `on` with a baseline that has simply never been allocation-measured is the
	// chicken-and-egg case: the cache cannot be sized without evidence, and only
	// a completed launch records it. Serve the baseline once and say so, rather
	// than refusing this launch shape forever.
	opts.HotExperts = "on"
	got, err := finalizeHotExpertCache(caps, model, opts, cloneStrategy(strategy))
	if err != nil || got == nil {
		t.Fatalf("required hot experts refused to bootstrap an unmeasured baseline: strategy=%+v err=%v", got, err)
	}
	if got.HotExpertCacheSlots != 0 {
		t.Fatalf("bootstrap launch must serve cache-free: %+v", got)
	}
	announced := false
	for _, x := range got.OptimizationExclusions {
		// Match the durable half of the promise, not the exact phrasing: the
		// message now also names the pinned topology it will replay.
		if strings.Contains(x, "next launch") {
			announced = true
		}
	}
	if !announced {
		t.Fatalf("bootstrap was not announced to the user: %+v", got.OptimizationExclusions)
	}

	// A genuine incompatibility (here: the backend does not advertise the exact
	// flag pair) must still fail closed under `on` — the bootstrap is scoped to
	// the missing-measurement case only.
	incompatible := opts
	incompatible.BackendHelp = "--some-other-flag N"
	if got, err := finalizeHotExpertCache(caps, model, incompatible, cloneStrategy(strategy)); err == nil || got != nil {
		t.Fatalf("required hot experts silently degraded on a real incompatibility: strategy=%+v err=%v", got, err)
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
	if len(base.OptimizationExclusions) == 0 || !strings.Contains(base.OptimizationExclusions[0], "allocation-measured") {
		t.Fatalf("missing hot-expert exclusion reason: %+v", base.OptimizationExclusions)
	}

	if err := RecordMeasuredAllocation(
		dir, model, base.ContextSize, base.UBatchSize, base.KVQuality,
		base.KVPlacement, backendCacheTag(opts), caps.GPUs, base.Parallel,
		MeasuredAllocation{
			Evidence: "allocation-verified", PlacementIdentity: AllocationPlacementIdentity(base, model),
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
		withMeasurement[1].Strategy.ResourceLedger == nil || withMeasurement[1].Strategy.ResourceLedger.Exact ||
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
			Evidence: "allocation-verified", PlacementIdentity: AllocationPlacementIdentity(base, model),
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

func TestFinalizeAutoHotExpertsKeepsPackedDefaultWithCacheOnChallenger(t *testing.T) {
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
			Evidence: "allocation-verified", PlacementIdentity: AllocationPlacementIdentity(base, model),
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
	// `auto` serves the packed cache-free layout as the fail-closed default; the
	// cache-on placement is the challenger the live A/B must win (invariant 7).
	if got == nil || got.HotExpertCacheSlots != 0 {
		t.Fatalf("auto finalize did not keep the packed cache-free default: %+v err=%v", got, err)
	}
	args := strings.Join(got.Args("moe.gguf", 8080), " ")
	if strings.Contains(args, "--moe-expert-cache") {
		t.Fatalf("packed default serve still emitted the cache flag: %s", args)
	}
	challenger := got.HotExpertCacheChallenger
	if challenger == nil || challenger.HotExpertCacheSlots < 2 {
		t.Fatalf("auto finalize did not attach a cache-on challenger: %+v", challenger)
	}
	if challenger.HotExpertCacheFreeBaseline == nil {
		t.Fatal("cache-on challenger did not capture its packed cache-free baseline")
	}
	challengerArgs := strings.Join(challenger.Args("moe.gguf", 8080), " ")
	if !strings.Contains(challengerArgs, "--moe-expert-cache") {
		t.Fatalf("cache-on challenger omitted the backend flag: %s", challengerArgs)
	}
	// The challenger's captured baseline reproduces the packed topology exactly.
	restored := RestorePackedCacheFreeBaseline(challenger)
	if restored == nil || restored.HotExpertCacheSlots != 0 ||
		restored.NCPUMoE != base.NCPUMoE || restored.OTString != base.OTString {
		t.Fatalf("restore did not reproduce the packed pre-demotion topology: %+v (base NCPUMoE=%d OT=%s)",
			restored, base.NCPUMoE, base.OTString)
	}
}

func TestRestorePackedCacheFreeBaselineReproducesPackedTopology(t *testing.T) {
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
	opts.BackendCacheTag = "test-restore-baseline"
	opts.HotExperts = "auto"
	// GPU 3 nearly full forces the priority challenger to demote a GPU expert layer.
	if err := RecordMeasuredAllocation(
		dir, model, base.ContextSize, base.UBatchSize, base.KVQuality,
		base.KVPlacement, backendCacheTag(opts), caps.GPUs, base.Parallel,
		MeasuredAllocation{
			Evidence: "allocation-verified", PlacementIdentity: AllocationPlacementIdentity(base, model),
			ContextTotalMB: 100, ContextHostMB: 100,
			ModelByGPU:       map[int]int{3: 24500, 7: 1000},
			UnaccountedByGPU: map[int]int{3: 50, 7: 50},
			ModelHostMB:      100,
		},
	); err != nil {
		t.Fatal(err)
	}

	wantOT := base.OTString
	wantNCPU := base.NCPUMoE
	challenger, err := hotExpertCacheCandidate(caps, model, base, opts)
	if err != nil || challenger == nil {
		t.Fatalf("no cache-on challenger: %v", err)
	}
	if challenger.HotExpertCacheFreeBaseline == nil {
		t.Fatal("challenger did not capture a packed cache-free baseline")
	}
	demoted := challenger.NCPUMoE != wantNCPU || challenger.OTString != wantOT
	if !demoted {
		t.Skip("challenger did not demote on this fixture; nothing to restore")
	}

	restored := RestorePackedCacheFreeBaseline(challenger)
	if restored == nil {
		t.Fatal("restore returned nil despite a captured baseline")
	}
	if restored.HotExpertCacheSlots != 0 || restored.HotExpertCacheVRAMByGPU != nil {
		t.Fatalf("restored baseline still carries cache fields: %+v", restored)
	}
	if restored.NCPUMoE != wantNCPU || restored.OTString != wantOT {
		t.Fatalf("restore did not reproduce the packed topology: NCPUMoE %d/%d OT %q/%q",
			restored.NCPUMoE, wantNCPU, restored.OTString, wantOT)
	}
	// Per-GPU expert-layer count must not regress below the packed baseline.
	for _, e := range restored.VRAMLedger {
		for _, b := range base.VRAMLedger {
			if e.GPU == b.GPU && e.ExpertLayers < b.ExpertLayers {
				t.Fatalf("GPU%d lost expert layers on restore: %d < %d", e.GPU, e.ExpertLayers, b.ExpertLayers)
			}
		}
	}
	// A challenger with no snapshot (legacy path) returns nil so the caller
	// recomputes rather than serving a feature-stripped demoted strategy.
	challenger.HotExpertCacheFreeBaseline = nil
	if RestorePackedCacheFreeBaseline(challenger) != nil {
		t.Fatal("restore invented a baseline for a snapshot-less challenger")
	}
}

func TestCalibrationCandidatesKeepPackedDefaultWithHandedOffChallenger(t *testing.T) {
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
	opts.BackendCacheTag = "test-handoff-challenger"
	opts.HotExperts = "auto"
	if err := RecordMeasuredAllocation(
		dir, model, base.ContextSize, base.UBatchSize, base.KVQuality,
		base.KVPlacement, backendCacheTag(opts), caps.GPUs, base.Parallel,
		MeasuredAllocation{
			Evidence: "allocation-verified", PlacementIdentity: AllocationPlacementIdentity(base, model),
			ContextTotalMB: 100, ContextHostMB: 100,
			ModelByGPU: map[int]int{3: 1000, 7: 1000}, ModelHostMB: 100,
			UnaccountedByGPU: map[int]int{3: 100, 7: 100},
		},
	); err != nil {
		t.Fatal(err)
	}

	served, err := finalizeHotExpertCache(caps, model, opts, base)
	if err != nil || served == nil {
		t.Fatalf("finalize: %v", err)
	}
	if served.HotExpertCacheSlots != 0 || served.HotExpertCacheChallenger == nil {
		t.Fatalf("auto finalize did not hand off a challenger over a packed default: %+v", served)
	}

	got := CalibrationCandidates(caps, model, served, opts)
	if len(got) < 2 || got[0].Name != "default" || got[0].Strategy.HotExpertCacheSlots != 0 {
		t.Fatalf("packed cache-free layout is not calibration candidate 0: %+v", got)
	}
	sawHot := false
	for _, c := range got[1:] {
		if strings.HasPrefix(c.Name, "hot-experts-") {
			sawHot = true
			if c.Strategy.HotExpertCacheSlots < 2 {
				t.Fatalf("cache-on challenger lacked a useful cache: %+v", c.Strategy)
			}
		}
	}
	if !sawHot {
		t.Fatalf("auto path did not offer the cache-on challenger for the A/B: %+v", got)
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
			Evidence: "allocation-verified", PlacementIdentity: AllocationPlacementIdentity(strategy, model),
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
	if ledger.Exact {
		t.Fatal("cache-free allocation cannot prove cache-on graph allocation")
	}
	if ledger.Devices[0].RequiredMB != 1045 || ledger.Devices[0].SlackMB != 155 || ledger.Devices[0].HotExpertCacheMB != 45 ||
		ledger.Devices[1].RequiredMB != 2090 || ledger.Devices[1].SlackMB != 10 || ledger.Devices[1].HotExpertCacheMB != 90 {
		t.Fatalf("cache charge=%+v", ledger.Devices)
	}
	alreadyMeasured := base
	alreadyMeasured.Devices = append([]DeviceResourceLedger(nil), base.Devices...)
	applyHotExpertCacheLedger(&alreadyMeasured, strategy, true)
	if !alreadyMeasured.Exact {
		t.Fatal("actual cache-on allocation lost its authority")
	}
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

// A backend that caches FEWER layers than planned is running a smaller, cheaper
// cache -- not a silent degradation to cache-off. Requiring exact equality
// discarded a working 6520 MiB cache on glm5next (2026-09-02): the backend
// acknowledged 41 layers where ggrun counted 42, because ggrun's cacheable-layer
// walk includes an MTP/NextN block the backend does not cache, and the feature
// never ran once all day as a result.
func TestHotExpertObservationAcceptsFewerCachedLayersThanPlanned(t *testing.T) {
	plan := &Strategy{
		HotExpertCacheSlots: 14, HotExpertCacheInserts: 2, HotExpertCacheLayers: 42,
		HotExpertCacheVRAMByGPU: map[int]int{0: 3400, 1: 3400},
	}
	ack := "operator(): MoE expert cache enabled: 41 layers x 14 slots, 2 inserts/step, 6520.3 MiB device memory"
	if err := ValidateHotExpertCacheObservation(plan, ack); err != nil {
		t.Fatalf("a 41-of-42-layer cache must be accepted: %v", err)
	}

	// More layers than planned is memory this plan never budgeted: still refused.
	over := "operator(): MoE expert cache enabled: 43 layers x 14 slots, 2 inserts/step, 6520.3 MiB device memory"
	if err := ValidateHotExpertCacheObservation(plan, over); err == nil {
		t.Fatal("a cache covering MORE layers than planned must be refused")
	}
	// A different shape means a different configuration under test: still refused.
	shape := "operator(): MoE expert cache enabled: 41 layers x 8 slots, 2 inserts/step, 6520.3 MiB device memory"
	if err := ValidateHotExpertCacheObservation(plan, shape); err == nil {
		t.Fatal("a cache with a different slot count must be refused")
	}
	// Zero covered layers is not a cache at all.
	none := "operator(): MoE expert cache enabled: 0 layers x 14 slots, 2 inserts/step, 0.0 MiB device memory"
	if err := ValidateHotExpertCacheObservation(plan, none); err == nil {
		t.Fatal("a cache covering no layers must be refused")
	}
	// A backend that degraded to cache-off is still caught by Enabled.
	if err := ValidateHotExpertCacheObservation(plan, "failed to allocate moe cache buffer"); err == nil {
		t.Fatal("a degraded cache-off backend must still be refused")
	}
}

// ledgerWithSlack builds an exact, fitting ledger whose devices carry the given
// slack. Slot arithmetic reads slack, so this is the knob that decides how many
// slots a placement can seat before any expert layer is demoted.
func ledgerWithSlack(caps *detect.Capabilities, slackMB int) ResourceLedger {
	ledger := ResourceLedger{Exact: true, Fits: true, Evidence: "live-allocated"}
	for _, g := range caps.GPUs {
		ledger.Devices = append(ledger.Devices, DeviceResourceLedger{
			GPU: g.Index, Active: true, FreeMB: slackMB, SlackMB: slackMB,
		})
	}
	return ledger
}

// TestExplicitSlotRequestFreesLayersToReachIt is the change's reason for
// existing. The automatic path stops as soon as it clears
// hotExpertMinUsefulSlots, which on GLM 5.3 Flash produced a 14-slot cache
// measuring ~26% hit rate and costing 12% decode: large enough to pay uploads,
// too small to earn them back. The sizes the patch author measured gains at
// (K=48-64) are unreachable that way on a rig whose VRAM is already committed,
// which makes the useful range untestable rather than merely unchosen.
//
// Invariant 3: a named slot count is a constraint, so ggrun must free what it
// needs rather than refuse.
func TestExplicitSlotRequestFreesLayersToReachIt(t *testing.T) {
	caps, model, strategy, opts := hotExpertFixture()

	// Slack enough for a small cache immediately, but not the requested one:
	// reaching it requires giving up resident expert layers.
	base := cloneStrategy(strategy)
	ledger := ledgerWithSlack(caps, 24)

	shape, adjusted, dropped, err := hotExpertFreeLayersForSlots(
		caps, model, base, opts, ledger, 3)
	if err != nil {
		t.Fatalf("an explicit request the rig can satisfy must not be refused: %v", err)
	}
	if shape == nil {
		t.Fatal("a satisfied request must return the cache shape it fits")
	}
	if !adjusted.Fits {
		t.Error("the returned ledger must fit")
	}
	if dropped < 0 {
		t.Errorf("demoted count must not be negative; got %d", dropped)
	}
	// Whatever it demoted, the result must actually seat the request. The
	// returned ledger carries Exact=false on purpose -- the placement is not
	// exact until the backend admits it -- so slot arithmetic is checked the way
	// the function itself does it, against the exact source evidence.
	slotLedger := adjusted
	slotLedger.Exact = ledger.Exact
	if slots := hotExpertCacheMaxSlots(model, shape, slotLedger); slots < 3 {
		t.Errorf("after demoting %d layer(s) the placement seats %d slots, want >= 3", dropped, slots)
	}
	if adjusted.Exact {
		t.Error("a demoted placement must not claim exact evidence before admission")
	}
}

// TestExplicitSlotRequestReportsTheCeiling: when even a fully demoted placement
// cannot seat the request, the error must name what this rig can actually
// reach. Bisecting a slot count by hand across six-minute model loads is not a
// reasonable way to find that number.
func TestExplicitSlotRequestReportsTheCeiling(t *testing.T) {
	caps, model, strategy, opts := hotExpertFixture()
	base := cloneStrategy(strategy)

	// Almost no slack: no amount of demotion seats a large cache.
	_, _, _, err := hotExpertFreeLayersForSlots(
		caps, model, base, opts, ledgerWithSlack(caps, 1), 4096)
	if err == nil {
		t.Fatal("an unreachable request must fail rather than silently under-deliver")
	}
	if !strings.Contains(err.Error(), "tops out near") {
		t.Errorf("error must report the reachable ceiling; got %q", err)
	}
	if !strings.Contains(err.Error(), "4096") {
		t.Errorf("error must name the request that could not be met; got %q", err)
	}
}

// TestExplicitSlotRequestDemotesOnlyWhenNeeded: a request the packed layout
// already satisfies must cost no resident expert layer at all.
func TestExplicitSlotRequestDemotesOnlyWhenNeeded(t *testing.T) {
	caps, model, strategy, opts := hotExpertFixture()
	base := cloneStrategy(strategy)
	before := base.NCPUMoE

	_, _, dropped, err := hotExpertFreeLayersForSlots(
		caps, model, base, opts, ledgerWithSlack(caps, 4096), 1)
	if err != nil {
		t.Fatalf("a trivially satisfiable request must succeed: %v", err)
	}
	if dropped != 0 {
		t.Errorf("demoted %d layer(s) for a request that already fit", dropped)
	}
	if base.NCPUMoE != before {
		t.Errorf("expert placement changed for a request that already fit: %d -> %d", before, base.NCPUMoE)
	}
}

// TestPriorityCandidateDemotesTowardTargetNotMinimum is the crossover defect.
//
// The loop stopped at hotExpertMinUsefulSlots (ExpertUsedCount), which is the
// point below which a cache cannot function -- not the point at which it is
// worth having. Measured 2026-09-08: that produced 14 slots on a rig that seats
// 34, and K=14 costs 10.3% decode while K=32 earns 6.0%. The first layout
// clearing ExpertUsedCount is reliably on the losing side of the crossover.
func TestPriorityCandidateDemotesTowardTargetNotMinimum(t *testing.T) {
	caps, model, strategy, opts := hotExpertFixture()
	// Slack that already clears minUseful without demoting anything, so a loop
	// that stops at the minimum returns immediately with a small cache.
	ledger := ledgerWithSlack(caps, 40)

	got, err := hotExpertPriorityCandidate(caps, model, strategy, opts, ledger)
	if err != nil {
		t.Fatalf("candidate: %v", err)
	}
	minUseful := hotExpertMinUsefulSlots(model)
	if got.HotExpertCacheSlots <= minUseful {
		t.Errorf("sized to %d slots, the bare minimum of %d; the loop must pursue the target",
			got.HotExpertCacheSlots, minUseful)
	}
}

// TestPriorityCandidateSettlesForBestReachable: the target routinely exceeds
// what a rig can seat -- 48 against a measured ceiling of 34 on the GLM rig --
// so running out of layers to demote must yield the largest cache that fits,
// not a failure and not the first one over the minimum.
func TestPriorityCandidateSettlesForBestReachable(t *testing.T) {
	caps, model, strategy, opts := hotExpertFixture()
	// The shared fixture has 4 experts and tiny layers, so its target is the
	// 2-slot minimum and every slot is nearly free. Give it a realistic shape:
	// 288 experts targeting 48 slots, with expert layers large enough that slot
	// count is actually bounded by VRAM.
	model.NumExperts = 288
	model.ExpertUsedCount = 8
	for i := range model.RoutedExpertLayerBytes {
		model.RoutedExpertLayerBytes[i] = 288 * 10 * hotExpertTestMiB // ~10 MiB per expert
	}
	target := hotExpertTargetSlots(model)

	// Enough to be useful, nowhere near the target even after every demotion.
	got, err := hotExpertPriorityCandidate(caps, model, strategy, opts, ledgerWithSlack(caps, 300))
	if err != nil {
		t.Fatalf("an unreachable target must not fail the candidate: %v", err)
	}
	if got.HotExpertCacheSlots >= target {
		t.Fatalf("fixture should not reach the target; got %d >= %d", got.HotExpertCacheSlots, target)
	}
	if got.HotExpertCacheSlots < hotExpertMinUsefulSlots(model) {
		t.Errorf("settled below the minimum useful size: %d", got.HotExpertCacheSlots)
	}
	if got.ResourceLedger == nil || !got.ResourceLedger.Fits {
		t.Error("the returned best-reachable layout must fit")
	}
}

// TestPriorityCandidateRefusesWhenNothingUsefulFits keeps the floor: a rig with
// no room must still fail closed rather than emit a token cache.
func TestPriorityCandidateRefusesWhenNothingUsefulFits(t *testing.T) {
	caps, model, strategy, opts := hotExpertFixture()
	_, err := hotExpertPriorityCandidate(caps, model, strategy, opts, ledgerWithSlack(caps, 1))
	if err == nil {
		t.Fatal("a rig that cannot seat the minimum must refuse")
	}
	if !strings.Contains(err.Error(), "best reachable") {
		t.Errorf("refusal should report what was reachable; got %q", err)
	}
}

// TestHotExpertTargetScalesWithModel: the target must follow the model's expert
// count, not a constant. A 32-expert model does not need 48 slots to cover its
// hot set, and encoding one rig's number would be invariant 8 all over again.
func TestHotExpertTargetScalesWithModel(t *testing.T) {
	big := &ModelProfile{NumExperts: 288, ExpertUsedCount: 8}
	small := &ModelProfile{NumExperts: 32, ExpertUsedCount: 4}
	if t1, t2 := hotExpertTargetSlots(big), hotExpertTargetSlots(small); t1 <= t2 {
		t.Errorf("a 288-expert model should target more slots than a 32-expert one: %d vs %d", t1, t2)
	}
	if got := hotExpertTargetSlots(small); got > small.NumExperts {
		t.Errorf("target %d exceeds the model's entire expert count %d", got, small.NumExperts)
	}
	// Never below the point a cache stops functioning.
	tiny := &ModelProfile{NumExperts: 4, ExpertUsedCount: 2}
	if got := hotExpertTargetSlots(tiny); got < hotExpertMinUsefulSlots(tiny) {
		t.Errorf("target %d is below the minimum useful %d", got, hotExpertMinUsefulSlots(tiny))
	}
}
