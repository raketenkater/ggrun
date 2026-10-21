package placement

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// dsv41Profile mirrors parse_gguf.py schema 3 on the pinned
// prometheusAIR/DeepSeek-V4.1-Flash-1017 IQ2_XS headers (revision 06b82ad):
// 171.7 GiB of execution weights plus 194.6 GiB of Engram tables in layers 1
// and 14.
func dsv41Profile() *ModelProfile {
	const layers = 40
	routed := make([]int64, layers)
	shared := make([]int64, layers)
	nonExpert := make([]int64, layers)
	lookup := make([]int64, layers)
	for i := range routed {
		routed[i] = (175620096000 - 1504051200) / layers
		shared[i] = 1504051200 / layers
		nonExpert[i] = (8764295168 - 2*1323827200) / layers
	}
	lookup[1], lookup[14] = 208902215200/2, 208902215200/2
	return &ModelProfile{
		Path: "/models/DeepSeek-V4.1-Flash-1017-IQ2_XS-00001-of-00010.gguf", SizeBytes: 393292445504, TotalSizeMB: 375072,
		NumLayers: layers, IsMoE: true, NumExperts: 384, ExpertUsedCount: 6, ExpertFF: 2304, ExpertSharedCount: 1,
		HiddenSize: 5120, EmbeddingLength: 5120, HeadCountKV: 1, KeyLength: 512, ValueLength: 512,
		ContextSize: 65536, CTXTrain: 1048576, ModelArch: "deepseek41", HasShexp: 1,
		ExpertBytes: 175620096000, NonExpertBytes: 8764295168,
		SparseLookupBytes: 208902215200, SparseLookupKind: "engram", SparseLookupLayerBytes: lookup,
		TokenEmbdBytes: 1323827200, OutputBytes: 1323827200, ShexpBytes: 1504051200,
		RoutedExpertLayerBytes: routed, ShexpLayerBytes: shared, NonExpertLayerBytes: nonExpert,
	}
}

func dsv41Host(freeRAMMB int) *detect.Capabilities {
	total := 217096
	if freeRAMMB > total {
		total = freeRAMMB + 10000
	}
	return &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 4070", VRAMTotalMB: 12282, VRAMUsedMB: 1, VRAMReservedMB: 408},
			{Index: 1, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, VRAMUsedMB: 272, VRAMReservedMB: 452},
			{Index: 2, Name: "RTX 3060", VRAMTotalMB: 12288, VRAMUsedMB: 1, VRAMReservedMB: 378},
		},
		RAM: detect.RAMInfo{TotalMB: total, FreeMB: freeRAMMB},
		CPU: detect.CPUInfo{Cores: 14},
	}
}

// dsv41Opts describes the command layer's inputs: capability file-backed only
// under a reviewed loader contract, and the lazy flag the backend advertises.
func dsv41Opts(t *testing.T, capability SparseLookupCapability) Options {
	return Options{
		ContextSize: 32768, KVPlacement: "gpu", KVQuality: "high", UBatchSize: 512,
		BackendTag: "llama", BackendIdentity: "smalinin/llama.cpp 425badfe", CacheDir: t.TempDir(),
		CPUExpertMMapCapability: CPUExpertMMapFileBacked, SparseLookupCapability: capability,
		SparseLookupEvidence: "test", SparseLookupFlag: SparseLookupLazyFlag,
	}
}

func hasArgPair(args []string, flag, value string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag && args[i+1] == value {
			return true
		}
	}
	return false
}

const dsv41TierMB = 199225

// Without a reviewed loader contract the tables are ordinary layer tensors:
// charged with their layer, never treated as reclaimable, and "--lazy-mode
// off" is stated when the backend advertises the flag. On this host the
// layers holding ~97 GiB tables cannot go to a GPU, so the refusal says why.
func TestSparseLookupWithoutContractLoadsWithLayers(t *testing.T) {
	for _, capability := range []SparseLookupCapability{"", SparseLookupUnknown, SparseLookupAnonymous} {
		_, err := Compute(dsv41Host(204700), dsv41Profile(), dsv41Opts(t, capability))
		if err == nil || !strings.Contains(err.Error(), "load with their layers") || strings.Contains(err.Error(), "must stay file-backed") {
			t.Fatalf("capability %q: want an actionable layer-tier refusal, got %v", capability, err)
		}
	}
}

// PR #93 review finding 4: a capable roomy host keeps an accurately charged
// resident option instead of a blanket lazy-loading requirement.
func TestRoomyHostHasResidentSparseLookupPlan(t *testing.T) {
	model := dsv41Profile()
	// Unknown loader, CPU-only on 900 GB: tables charged with their layers.
	opts := dsv41Opts(t, SparseLookupUnknown)
	opts.CPUMode = true
	s, err := Compute(dsv41Host(900000), model, opts)
	if err != nil {
		t.Fatalf("roomy CPU host with an unreviewed loader must still plan: %v", err)
	}
	if s.SparseLookupPolicy != SparseLookupLayer || s.PlannedHostFootprintMB < model.TotalSizeMB {
		t.Fatalf("layer tier must charge the whole model to host: policy=%s footprint=%d model=%d", s.SparseLookupPolicy, s.PlannedHostFootprintMB, model.TotalSizeMB)
	}
	if !hasArgPair(s.Args(model.Path, 8080), SparseLookupLazyFlag, "off") {
		t.Fatalf("an advertised lazy flag is stated off for layer tables: %v", s.Args(model.Path, 8080))
	}
	// Reviewed loader on the same host: resident lazy tier, charged once.
	s, err = Compute(dsv41Host(900000), model, dsv41Opts(t, SparseLookupFileBacked))
	if err != nil {
		t.Fatalf("roomy host with a reviewed loader: %v", err)
	}
	paged, err := Compute(dsv41Host(900000), model, func() Options {
		o := dsv41Opts(t, SparseLookupFileBacked)
		o.sparseLookupPolicy = SparseLookupPaged
		return o
	}())
	if err != nil {
		t.Fatal(err)
	}
	// The resident plan sizes its runtime against RAM already reduced by the
	// tier, so compare against the paged plan's footprint within that margin.
	if s.SparseLookupPolicy != SparseLookupResident || paged.PlannedHostFootprintMB >= dsv41TierMB ||
		s.PlannedHostFootprintMB < paged.PlannedHostFootprintMB+dsv41TierMB-2048 {
		t.Fatalf("roomy host: policy=%s footprint=%d, want resident with the paged footprint %d + tier %d",
			s.SparseLookupPolicy, s.PlannedHostFootprintMB, paged.PlannedHostFootprintMB, dsv41TierMB)
	}
	if !hasArgPair(s.Args(model.Path, 8080), SparseLookupLazyFlag, "on") || s.MMapRequired {
		t.Fatalf("resident lazy tier: args=%v mmapRequired=%v", s.Args(model.Path, 8080), s.MMapRequired)
	}
}

// A constrained host pages the tier and keeps execution weights resident.
func TestConstrainedHostPagesTierAndKeepsExecutionResident(t *testing.T) {
	model := dsv41Profile()
	s, err := Compute(dsv41Host(204700), model, dsv41Opts(t, SparseLookupFileBacked))
	if err != nil {
		t.Fatalf("execution weights fit RAM+VRAM; the lookup tier must not block the plan: %v", err)
	}
	if s.Type != MoEOffload || s.SparseLookupMB != dsv41TierMB || !StrategyUsesFileBackedSparseLookup(s) {
		t.Fatalf("type=%s sparse=%d policy=%q, want moe_offload with a %d MiB paged tier", s.Type, s.SparseLookupMB, s.SparseLookupPolicy, dsv41TierMB)
	}
	if s.MMap || s.MMapRequired {
		t.Fatalf("expert paging must not be planned when execution weights fit resident: mmap=%v required=%v", s.MMap, s.MMapRequired)
	}
	if s.PlannedHostFootprintMB <= 0 || s.PlannedHostFootprintMB >= dsv41TierMB {
		t.Fatalf("planned host footprint %d MiB must exclude the %d MiB paged tier", s.PlannedHostFootprintMB, dsv41TierMB)
	}
	args := s.Args(model.Path, 8080)
	if !hasArgPair(args, SparseLookupLazyFlag, "on") || !strings.Contains(strings.Join(args, " "), "--no-mmap") {
		t.Fatalf("argv must state lazy lookup reads and resident experts: %v", args)
	}
	// Automatic context: a resident tier that needs expert paging loses to a
	// paged tier with resident execution weights.
	opts := dsv41Opts(t, SparseLookupFileBacked)
	opts.ContextSize = 0
	if auto, err := Compute(dsv41Host(204700), model, opts); err != nil || auto.SparseLookupPolicy != SparseLookupPaged || auto.MMapRequired {
		t.Fatalf("auto context: %v policy=%v", err, auto)
	}
}

// The defect this tier fixes: schema-2 accounting folds both ~97 GiB tables
// into layers 1 and 14 as ordinary backbone, which bills them to a GPU.
func TestSchema2EngramAccountingCannotPlaceTheModel(t *testing.T) {
	model := dsv41Profile()
	model.NonExpertBytes += model.SparseLookupBytes
	model.NonExpertLayerBytes[1] += model.SparseLookupBytes / 2
	model.NonExpertLayerBytes[14] += model.SparseLookupBytes / 2
	model.SparseLookupBytes, model.SparseLookupKind, model.SparseLookupLayerBytes = 0, "", nil
	if s, err := Compute(dsv41Host(204700), model, dsv41Opts(t, SparseLookupFileBacked)); err == nil {
		t.Fatalf("schema-2 accounting unexpectedly produced a plan: type=%s ncpumoe=%d", s.Type, s.NCPUMoE)
	}
}

func TestSparseLookupTierRefusesAutomaticExpertPaging(t *testing.T) {
	model := dsv41Profile()
	_, err := Compute(dsv41Host(100000), model, dsv41Opts(t, SparseLookupFileBacked))
	if err == nil || !strings.Contains(err.Error(), "will not also page experts") || !strings.Contains(err.Error(), "resident tier") {
		t.Fatalf("expert paging beside a paged lookup tier must need explicit consent, got %v", err)
	}
	opts := dsv41Opts(t, SparseLookupFileBacked)
	opts.ForceMMap = true
	s, err := Compute(dsv41Host(100000), model, opts)
	if err != nil {
		t.Fatalf("explicit --mmap keeps expert paging as a last resort: %v", err)
	}
	if !s.MMap || !s.MMapRequired || !hasArgPair(s.Args(model.Path, 8080), SparseLookupLazyFlag, "on") {
		t.Fatalf("explicit last resort: mmap=%v required=%v args=%v", s.MMap, s.MMapRequired, s.Args(model.Path, 8080))
	}
}

// Explicit user lazy choices are constraints, never silently replaced.
func TestExplicitLazyChoiceIsHonored(t *testing.T) {
	model := dsv41Profile()
	opts := dsv41Opts(t, SparseLookupFileBacked)
	opts.SparseLookupLazy = "off"
	opts.CPUMode = true
	s, err := Compute(dsv41Host(900000), model, opts)
	if err != nil || s.SparseLookupPolicy != SparseLookupLayer || !hasArgPair(s.Args(model.Path, 8080), SparseLookupLazyFlag, "off") {
		t.Fatalf("explicit off: %v %+v", err, s)
	}
	opts = dsv41Opts(t, SparseLookupUnknown)
	opts.SparseLookupLazy = "on"
	if _, err := Compute(dsv41Host(900000), model, opts); err == nil || !strings.Contains(err.Error(), "no reviewed lazy-loader contract") {
		t.Fatalf("explicit on without a contract must be refused with the reason, got %v", err)
	}
	opts = dsv41Opts(t, SparseLookupFileBacked)
	opts.SparseLookupLazy = "on"
	if s, err := Compute(dsv41Host(204700), model, opts); err != nil || !StrategyUsesLazySparseLookup(s) {
		t.Fatalf("explicit on with a contract: %v %+v", err, s)
	}
}

// synthDenseProfile is a SYNTHETIC dense second architecture with a small
// sparse table kind. It exercises the generic tier policy only; it does not
// claim any real model with such tables is supported.
func synthDenseProfile(tierMiB int) *ModelProfile {
	const layers = 32
	nonExpert := make([]int64, layers)
	lookup := make([]int64, layers)
	for i := range nonExpert {
		nonExpert[i] = 6000 << 20 / layers
	}
	lookup[0], lookup[16] = int64(tierMiB)<<20/2, int64(tierMiB)<<20/2
	total := 6000 + tierMiB
	return &ModelProfile{
		Path: "/models/synthrow-dense.gguf", TotalSizeMB: total, SizeBytes: int64(total) << 20,
		NumLayers: layers, HiddenSize: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
		ContextSize: 32768, CTXTrain: 32768, ModelArch: "synthrow",
		NonExpertBytes: 6000 << 20, NonExpertLayerBytes: nonExpert,
		SparseLookupBytes: int64(tierMiB) << 20, SparseLookupKind: "synthetic-rowtable", SparseLookupLayerBytes: lookup,
		MeasuredKVBytesPerTok: map[string]float64{"f16": 128 * 1024, "q8_0": 64 * 1024},
	}
}

func synthOpts(t *testing.T, capability SparseLookupCapability) Options {
	return Options{
		ContextSize: 8192, KVPlacement: "auto", KVQuality: "mid", CacheDir: t.TempDir(), BackendTag: "llama",
		SparseLookupCapability: capability, SparseLookupFlag: SparseLookupLazyFlag,
	}
}

func synthCaps(ramFreeMB int, gpus ...detect.GPU) *detect.Capabilities {
	return &detect.Capabilities{GPUs: gpus, RAM: detect.RAMInfo{TotalMB: ramFreeMB + 4096, FreeMB: ramFreeMB}, CPU: detect.CPUInfo{Cores: 8}}
}

// SYNTHETIC matrix: dense model, small table, CPU-only / single-GPU /
// multi-GPU, with and without a reviewed loader.
func TestSyntheticDenseTierPolicyMatrix(t *testing.T) {
	single := []detect.GPU{{Index: 0, VRAMTotalMB: 24576}}
	multi := []detect.GPU{{Index: 0, VRAMTotalMB: 12288}, {Index: 1, VRAMTotalMB: 12288}}
	for _, tc := range []struct {
		name       string
		caps       *detect.Capabilities
		cpu        bool
		capability SparseLookupCapability
		want       SparseLookupPolicy
	}{
		{"cpu-only, unknown loader", synthCaps(64000), true, SparseLookupUnknown, SparseLookupLayer},
		{"cpu-only, reviewed loader", synthCaps(64000), true, SparseLookupFileBacked, SparseLookupResident},
		{"single GPU, unknown loader", synthCaps(64000, single...), false, SparseLookupUnknown, SparseLookupLayer},
		{"single GPU, reviewed loader", synthCaps(64000, single...), false, SparseLookupFileBacked, SparseLookupResident},
		{"multi GPU, unknown loader", synthCaps(64000, multi...), false, SparseLookupUnknown, SparseLookupLayer},
		{"multi GPU, reviewed loader", synthCaps(64000, multi...), false, SparseLookupFileBacked, SparseLookupResident},
	} {
		model := synthDenseProfile(2048)
		opts := synthOpts(t, tc.capability)
		opts.CPUMode = tc.cpu
		s, err := Compute(tc.caps, model, opts)
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		if s.SparseLookupPolicy != tc.want || s.SparseLookupMB != 2048 {
			t.Errorf("%s: policy=%s tier=%d, want %s 2048", tc.name, s.SparseLookupPolicy, s.SparseLookupMB, tc.want)
		}
		want := "off"
		if tc.want != SparseLookupLayer {
			want = "on"
		}
		if !hasArgPair(s.Args(model.Path, 8080), SparseLookupLazyFlag, want) {
			t.Errorf("%s: lazy flag must be stated %s: %v", tc.name, want, s.Args(model.Path, 8080))
		}
		if tc.cpu && s.PlannedHostFootprintMB > 0 && s.PlannedHostFootprintMB < model.TotalSizeMB {
			t.Errorf("%s: CPU-only footprint %d must include the %d MiB model with its tables", tc.name, s.PlannedHostFootprintMB, model.TotalSizeMB)
		}
	}
	// A backend without the flag gets no lazy option at all.
	opts := synthOpts(t, SparseLookupUnknown)
	opts.SparseLookupFlag = ""
	if s, err := Compute(synthCaps(64000, single...), synthDenseProfile(2048), opts); err != nil || strings.Contains(strings.Join(s.Args("m", 1), " "), SparseLookupLazyFlag) {
		t.Fatalf("flagless backend: %v %v", err, s)
	}
}

// The layer tier is real resident memory on its layer's device: a table that
// does not fit the GPU next to the model must change the plan.
func TestLayerTierIsChargedToTheLayerDevice(t *testing.T) {
	caps := synthCaps(64000, detect.GPU{Index: 0, VRAMTotalMB: 12288})
	small, err := Compute(caps, synthDenseProfile(256), synthOpts(t, SparseLookupUnknown))
	if err != nil {
		t.Fatal(err)
	}
	big, err := Compute(caps, synthDenseProfile(16384), synthOpts(t, SparseLookupUnknown))
	if err == nil && big.Type == small.Type && big.GPULayers == small.GPULayers {
		t.Fatalf("a 16 GiB layer table on a 12 GiB GPU did not change the plan: %s/%d", big.Type, big.GPULayers)
	}
}

func TestOrdinaryModelIgnoresSparseLookupInputs(t *testing.T) {
	model := dsv41Profile()
	model.SparseLookupBytes, model.SparseLookupKind, model.SparseLookupLayerBytes, model.ModelArch = 0, "", nil, "deepseek4"
	model.SizeBytes -= 208902215200
	model.TotalSizeMB -= dsv41TierMB
	for _, capability := range []SparseLookupCapability{SparseLookupUnknown, SparseLookupFileBacked} {
		for _, lazy := range []string{"", "on", "off"} {
			opts := dsv41Opts(t, capability)
			opts.SparseLookupLazy = lazy
			s, err := Compute(dsv41Host(204700), model, opts)
			if err != nil {
				t.Fatalf("capability %q lazy %q must not affect a model without a lookup tier: %v", capability, lazy, err)
			}
			if s.SparseLookupMB != 0 || s.SparseLookupPolicy != "" || strings.Contains(strings.Join(s.Args(model.Path, 8080), " "), SparseLookupLazyFlag) {
				t.Fatalf("ordinary model gained a lookup tier: %d MiB %q, args %v", s.SparseLookupMB, s.SparseLookupPolicy, s.Args(model.Path, 8080))
			}
		}
	}
	plain := &Strategy{Type: MoEOffload, ContextSize: 4096, MMap: false}
	before := AllocationPlacementIdentity(plain)
	plain.SparseLookupCapability = SparseLookupFileBacked
	if AllocationPlacementIdentity(plain) != before {
		t.Fatal("allocation identity of a plan without a lookup tier must not change")
	}
	plain.SparseLookupMB, plain.SparseLookupPolicy = 1, SparseLookupPaged
	paged := AllocationPlacementIdentity(plain)
	plain.SparseLookupPolicy = SparseLookupResident
	if paged == before || AllocationPlacementIdentity(plain) == paged {
		t.Fatal("the lookup tier and its policy must scope allocation evidence")
	}
	if tag := backendCacheTag(Options{BackendTag: "llama"}); strings.Contains(tag, "lookup=") {
		t.Fatalf("models without a tier keep their cache namespace: %q", tag)
	}
	if backendCacheTag(Options{BackendTag: "llama", sparseLookupPolicy: SparseLookupPaged}) == backendCacheTag(Options{BackendTag: "llama", sparseLookupPolicy: SparseLookupResident}) {
		t.Fatal("paged and resident evidence must not share a cache namespace")
	}
}

func TestResidentTierJoinsFootprintOnce(t *testing.T) {
	model := dsv41Profile()
	opts := dsv41Opts(t, SparseLookupFileBacked)
	opts.sparseLookupPolicy = SparseLookupResident
	s := &Strategy{PlannedHostFootprintMB: 150000}
	applySparseLookupTier(s, model, opts)
	applySparseLookupTier(s, model, opts)
	if s.PlannedHostFootprintMB != 150000+dsv41TierMB {
		t.Fatalf("repeated application: footprint %d", s.PlannedHostFootprintMB)
	}
	// An underived footprint stays underived (the whole-host ceiling applies).
	z := &Strategy{}
	applySparseLookupTier(z, model, opts)
	if z.PlannedHostFootprintMB != 0 {
		t.Fatalf("underived footprint became %d", z.PlannedHostFootprintMB)
	}
}

func TestVerifiedConfigKeepsSparseLookupTier(t *testing.T) {
	model := dsv41Profile()
	opts := dsv41Opts(t, SparseLookupFileBacked)
	s, err := Compute(dsv41Host(204700), model, opts)
	if err != nil {
		t.Fatal(err)
	}
	vc := VerifiedConfigToRecord("scope", "model", s, "id", "/bin/llama-server", "", "")
	restored := VerifiedToStrategy(&vc, opts, dsv41Host(204700))
	if restored == nil || restored.SparseLookupMB != s.SparseLookupMB || !StrategyUsesFileBackedSparseLookup(restored) {
		t.Fatalf("verified round trip lost the lookup tier: %+v", restored)
	}
	if !hasArgPair(restored.Args(model.Path, 8080), SparseLookupLazyFlag, "on") {
		t.Fatal("relaunch from a verified config must keep lazy lookup reads")
	}
	// A record is evidence only for the policy that produced it.
	opts.sparseLookupPolicy = SparseLookupPaged
	if !sparseLookupRecordMatches(restored, model, opts) {
		t.Fatal("matching record rejected")
	}
	opts.sparseLookupPolicy = SparseLookupResident
	if sparseLookupRecordMatches(restored, model, opts) {
		t.Fatal("a paged record was accepted for the resident policy")
	}
	restored.SparseLookupPolicy = ""
	opts.sparseLookupPolicy = SparseLookupPaged
	if sparseLookupRecordMatches(restored, model, opts) {
		t.Fatal("a record from before tier policies was accepted")
	}
}

// PR #93 re-review finding 1: a measurement the command side records for a
// sparse-policy plan (full model profile, its own scoped tag) must be found
// by the policy's planning pass (reduced model view, backendCacheTag), and
// must not leak into another policy or backend build.
func TestSparsePolicyEvidenceRoundTrip(t *testing.T) {
	full := dsv41Profile()
	for _, policy := range []SparseLookupPolicy{SparseLookupLayer, SparseLookupResident, SparseLookupPaged} {
		view := sparseLookupModelView(full, policy)
		if SpecTargetIdentity(view) != SpecTargetIdentity(full) || kvCachePath("/c", view) != kvCachePath("/c", full) {
			t.Fatalf("%s view changed the artifact identity", policy)
		}
		if view.ArtifactSizeBytes() != full.SizeBytes || view.ArtifactTotalSizeMB() != full.TotalSizeMB {
			t.Fatalf("%s view: artifact %d/%d, want %d/%d", policy, view.ArtifactSizeBytes(), view.ArtifactTotalSizeMB(), full.SizeBytes, full.TotalSizeMB)
		}
	}
	o := dsv41Opts(t, SparseLookupFileBacked)
	caps := dsv41Host(204700)
	first, err := Compute(caps, full, o)
	if err != nil || first.SparseLookupPolicy != SparseLookupPaged {
		t.Fatalf("first plan: %v %+v", err, first)
	}
	// The command side writes with the full profile and its own tag, scoped by
	// the plan's policy (scopedProbeBackendTagForStrategy).
	writerTag := SparseLookupScopedTag(backendCacheTag(Options{BackendTag: o.BackendTag}), first.SparseLookupPolicy)
	if err := RecordMeasuredComputeBuffers(o.CacheDir, full, first.ContextSize, first.UBatchSize, first.KVQuality, first.KVPlacement,
		writerTag, caps.GPUs, first.Parallel, map[int]int{1: 2913}); err != nil {
		t.Fatal(err)
	}
	reader := o
	reader.sparseLookupPolicy = SparseLookupPaged
	if pc := reader.loadProbeCacheForStrategy(sparseLookupModelView(full, SparseLookupPaged), first, caps.GPUs); pc == nil {
		t.Fatalf("replan cannot read the measurement: writer %q reader %q", writerTag, backendCacheTag(reader))
	}
	other := o
	other.sparseLookupPolicy = SparseLookupResident
	if pc := other.loadProbeCacheForStrategy(sparseLookupModelView(full, SparseLookupResident), first, caps.GPUs); pc != nil {
		t.Fatal("a paged measurement leaked into the resident policy")
	}
	build := reader
	build.BackendTag = "llama@another-build"
	if pc := build.loadProbeCacheForStrategy(sparseLookupModelView(full, SparseLookupPaged), first, caps.GPUs); pc != nil {
		t.Fatal("a measurement leaked to another backend build")
	}
	// Replan and relaunch from the verified record keep the same evidence scope.
	again, err := Compute(caps, full, o)
	if err != nil || again.SparseLookupPolicy != SparseLookupPaged {
		t.Fatalf("replan: %v %+v", err, again)
	}
	if err := RecordMeasuredComputeBuffers(o.CacheDir, full, again.ContextSize, again.UBatchSize, again.KVQuality, again.KVPlacement,
		SparseLookupScopedTag(backendCacheTag(Options{BackendTag: o.BackendTag}), again.SparseLookupPolicy), caps.GPUs, again.Parallel, map[int]int{1: 2913}); err != nil {
		t.Fatal(err)
	}
	// A relaunch without an explicit ubatch restores the recorded shape.
	relaunch := reader
	relaunch.UBatchSize = 0
	vc := VerifiedConfigToRecord("scope", "model", again, "id", "/bin/llama-server", "", "")
	if restored := VerifiedToStrategy(&vc, relaunch, caps); restored == nil || restored.SparseLookupPolicy != SparseLookupPaged ||
		relaunch.loadProbeCacheForStrategy(sparseLookupModelView(full, SparseLookupPaged), restored, caps.GPUs) == nil {
		t.Fatalf("relaunch from the verified record cannot read the measurement: %+v", restored)
	}
}

// PR #93 final review P2: the Claude relaunch refresh receives the full
// profile and request options. It must read the prompt-cache measurement the
// command side recorded under the plan's policy-scoped tag, size CRAM on the
// same accounting Compute used, and stay isolated from other policies and
// from ordinary models.
func TestRuntimeRefreshReadsPolicyScopedPromptCache(t *testing.T) {
	for _, tc := range []struct {
		free   int
		policy SparseLookupPolicy
	}{{204700, SparseLookupPaged}, {900000, SparseLookupResident}} {
		full, caps := dsv41Profile(), dsv41Host(tc.free)
		opts := dsv41Opts(t, SparseLookupFileBacked)
		s, err := Compute(caps, full, opts)
		if err != nil || s.SparseLookupPolicy != tc.policy {
			t.Fatalf("%s fixture: %v %+v", tc.policy, err, s)
		}
		// Without new evidence the refresh reproduces Compute's sizing.
		if same := RefreshRuntimeCachePolicy(caps, full, s, opts); same.CRAM != s.CRAM || same.MaxCheckpoints != s.MaxCheckpoints {
			t.Fatalf("%s: refresh without a measurement changed sizing: cram %d -> %d, checkpoints %d -> %d",
				tc.policy, s.CRAM, same.CRAM, s.MaxCheckpoints, same.MaxCheckpoints)
		}
		plain := backendCacheTag(Options{BackendTag: opts.BackendTag})
		other := SparseLookupResident
		if tc.policy == SparseLookupResident {
			other = SparseLookupPaged
		}
		record := func(tag string, entryMB float64) {
			t.Helper()
			if err := RecordPromptCacheObservation(opts.CacheDir, full, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement,
				tag, caps.GPUs, s.Parallel, PromptCacheObservation{LargestEntryMB: entryMB}); err != nil {
				t.Fatal(err)
			}
		}
		// Evidence for another policy or for the unscoped namespace is not this plan's.
		record(plain, 99000)
		record(SparseLookupScopedTag(plain, other), 98000)
		if leaked := RefreshRuntimeCachePolicy(caps, full, s, opts); leaked.MeasuredPromptCacheEntryMB != 0 {
			t.Fatalf("%s refresh read foreign evidence: %.0f MiB", tc.policy, leaked.MeasuredPromptCacheEntryMB)
		}
		// An entry whose slots+1 working set exceeds the derived budget.
		slots := max(1, s.Parallel)
		entryMB := float64(s.CRAM/(slots+1) + 2048)
		record(SparseLookupScopedTag(plain, tc.policy), entryMB)
		next := RefreshRuntimeCachePolicy(caps, full, s, opts)
		if next.MeasuredPromptCacheEntryMB != entryMB {
			t.Fatalf("%s: relaunch refresh lost the measurement: got %.0f, writer tag %q", tc.policy, next.MeasuredPromptCacheEntryMB, SparseLookupScopedTag(plain, tc.policy))
		}
		if next.CRAM <= s.CRAM {
			t.Fatalf("%s: measured entry did not reach CRAM: %d -> %d", tc.policy, s.CRAM, next.CRAM)
		}
		if next.SparseLookupPolicy != s.SparseLookupPolicy || next.ContextSize != s.ContextSize || next.PlannedHostFootprintMB != s.PlannedHostFootprintMB {
			t.Fatalf("%s: refresh changed the plan: %+v", tc.policy, next)
		}
	}

	// An ordinary model keeps the unscoped namespace and ignores scoped entries.
	model := dsv41Profile()
	model.SparseLookupBytes, model.SparseLookupKind, model.SparseLookupLayerBytes, model.ModelArch = 0, "", nil, "deepseek4"
	model.SizeBytes -= 208902215200
	model.TotalSizeMB -= dsv41TierMB
	caps, opts := dsv41Host(204700), dsv41Opts(t, SparseLookupFileBacked)
	s, err := Compute(caps, model, opts)
	if err != nil || s.SparseLookupPolicy != "" {
		t.Fatalf("ordinary fixture: %v %+v", err, s)
	}
	plain := backendCacheTag(Options{BackendTag: opts.BackendTag})
	if err := RecordPromptCacheObservation(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement,
		SparseLookupScopedTag(plain, SparseLookupPaged), caps.GPUs, s.Parallel, PromptCacheObservation{LargestEntryMB: 97000}); err != nil {
		t.Fatal(err)
	}
	if got := RefreshRuntimeCachePolicy(caps, model, s, opts); got.MeasuredPromptCacheEntryMB != 0 {
		t.Fatalf("ordinary model read a policy-scoped entry: %.0f", got.MeasuredPromptCacheEntryMB)
	}
	if err := RecordPromptCacheObservation(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement,
		plain, caps.GPUs, s.Parallel, PromptCacheObservation{LargestEntryMB: 3000}); err != nil {
		t.Fatal(err)
	}
	if got := RefreshRuntimeCachePolicy(caps, model, s, opts); got.MeasuredPromptCacheEntryMB != 3000 {
		t.Fatalf("ordinary model lost its own entry: %.0f", got.MeasuredPromptCacheEntryMB)
	}
}

// The resource ledger is another exported reader of a finished strategy: a
// paged or resident tier is priced on the policy's view (tables are not GPU
// weights) whether the caller passes the full profile or Compute's view.
func TestResourceLedgerPricesLookupTierOnPolicyView(t *testing.T) {
	for _, free := range []int{204700, 900000} {
		full, caps := dsv41Profile(), dsv41Host(free)
		opts := dsv41Opts(t, SparseLookupFileBacked)
		s, err := Compute(caps, full, opts)
		if err != nil {
			t.Fatal(err)
		}
		// The command side records a measured graph under the plan's scoped tag.
		writerTag := SparseLookupScopedTag(backendCacheTag(Options{BackendTag: opts.BackendTag}), s.SparseLookupPolicy)
		if err := RecordMeasuredComputeBuffers(opts.CacheDir, full, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement,
			writerTag, caps.GPUs, s.Parallel, map[int]int{0: 7001, 1: 7002, 2: 7003}); err != nil {
			t.Fatal(err)
		}
		internal := opts
		internal.sparseLookupPolicy = s.SparseLookupPolicy
		external := BuildResourceLedger(caps, full, s, opts)
		if pc := internal.loadProbeCacheForStrategy(full, s, caps.GPUs); pc == nil {
			t.Fatalf("%s: fixture measurement not readable under %q", s.SparseLookupPolicy, writerTag)
		}
		planned := BuildResourceLedger(caps, sparseLookupModelView(full, s.SparseLookupPolicy), s, internal)
		gpuModelMB := 0
		for i := range external.Devices {
			if external.Devices[i].ModelMB != planned.Devices[i].ModelMB || external.Devices[i].GraphMB != planned.Devices[i].GraphMB ||
				external.Devices[i].RequiredMB != planned.Devices[i].RequiredMB {
				t.Fatalf("%s device %d: external %+v, planned %+v", s.SparseLookupPolicy, i, external.Devices[i], planned.Devices[i])
			}
			gpuModelMB += external.Devices[i].ModelMB
		}
		if external.Host != planned.Host || external.Fits != planned.Fits {
			t.Fatalf("%s host: external %+v fits=%v, planned %+v fits=%v", s.SparseLookupPolicy, external.Host, external.Fits, planned.Host, planned.Fits)
		}
		if len(external.MissingComputeGPUs) != 0 {
			t.Fatalf("%s: measured graphs not read: missing %v", s.SparseLookupPolicy, external.MissingComputeGPUs)
		}
		if gpuModelMB >= dsv41TierMB {
			t.Fatalf("%s: %d MiB of GPU model rows include the lookup tables", s.SparseLookupPolicy, gpuModelMB)
		}
	}
}
