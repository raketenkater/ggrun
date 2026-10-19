package placement

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// dsv41Profile mirrors parse_gguf.py schema 3 on the pinned
// prometheusAIR/DeepSeek-V4.1-Flash-1017 IQ2_XS headers (revision 06b82ad):
// 171.7 GiB of execution weights plus 194.6 GiB of Engram tables.
func dsv41Profile() *ModelProfile {
	const layers = 40
	routed := make([]int64, layers)
	shared := make([]int64, layers)
	nonExpert := make([]int64, layers)
	for i := range routed {
		routed[i] = (175620096000 - 1504051200) / layers
		shared[i] = 1504051200 / layers
		nonExpert[i] = (8764295168 - 2*1323827200) / layers
	}
	return &ModelProfile{
		Path: "/models/DeepSeek-V4.1-Flash-1017-IQ2_XS-00001-of-00010.gguf", SizeBytes: 393292445504, TotalSizeMB: 375072,
		NumLayers: layers, IsMoE: true, NumExperts: 384, ExpertUsedCount: 6, ExpertFF: 2304, ExpertSharedCount: 1,
		HiddenSize: 5120, EmbeddingLength: 5120, HeadCountKV: 1, KeyLength: 512, ValueLength: 512,
		ContextSize: 65536, CTXTrain: 1048576, ModelArch: "deepseek41", HasShexp: 1,
		ExpertBytes: 175620096000, NonExpertBytes: 8764295168, SparseLookupBytes: 208902215200, SparseLookupKind: "engram",
		TokenEmbdBytes: 1323827200, OutputBytes: 1323827200, ShexpBytes: 1504051200,
		RoutedExpertLayerBytes: routed, ShexpLayerBytes: shared, NonExpertLayerBytes: nonExpert,
	}
}

func dsv41Host(freeRAMMB int) *detect.Capabilities {
	return &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 4070", VRAMTotalMB: 12282, VRAMUsedMB: 1, VRAMReservedMB: 408},
			{Index: 1, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, VRAMUsedMB: 272, VRAMReservedMB: 452},
			{Index: 2, Name: "RTX 3060", VRAMTotalMB: 12288, VRAMUsedMB: 1, VRAMReservedMB: 378},
		},
		RAM: detect.RAMInfo{TotalMB: 217096, FreeMB: freeRAMMB},
		CPU: detect.CPUInfo{Cores: 14},
	}
}

func dsv41Opts(t *testing.T, capability SparseLookupCapability) Options {
	return Options{
		ContextSize: 32768, KVPlacement: "gpu", KVQuality: "high", UBatchSize: 512,
		BackendTag: "llama", BackendIdentity: "smalinin/llama.cpp 425badfe", CacheDir: t.TempDir(),
		CPUExpertMMapCapability: CPUExpertMMapFileBacked, SparseLookupCapability: capability,
		SparseLookupEvidence: "test",
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

func TestSparseLookupTierFailsClosedWithoutLazyLoader(t *testing.T) {
	for _, capability := range []SparseLookupCapability{"", SparseLookupUnknown, SparseLookupAnonymous} {
		_, err := Compute(dsv41Host(204700), dsv41Profile(), dsv41Opts(t, capability))
		if err == nil || !strings.Contains(err.Error(), SparseLookupLazyFlag) {
			t.Fatalf("capability %q: a lookup tier without a lazy loader must be refused, got %v", capability, err)
		}
	}
}

func TestSparseLookupTierStaysOffGPUAndOutOfResidentLedger(t *testing.T) {
	model := dsv41Profile()
	s, err := Compute(dsv41Host(204700), model, dsv41Opts(t, SparseLookupFileBacked))
	if err != nil {
		t.Fatalf("execution weights fit RAM+VRAM; the lookup tier must not block the plan: %v", err)
	}
	wantMB := bytesToMiBCeil(model.SparseLookupBytes)
	if s.Type != MoEOffload || s.SparseLookupMB != wantMB || !StrategyUsesFileBackedSparseLookup(s) {
		t.Fatalf("type=%s sparse=%d capability=%q, want moe_offload with %d MiB file-backed", s.Type, s.SparseLookupMB, s.SparseLookupCapability, wantMB)
	}
	// Resident execution weights fit, so experts stay anonymous and resident;
	// the lookup tables are the only file-backed tier.
	if s.MMap || s.MMapRequired {
		t.Fatalf("expert paging must not be planned when execution weights fit resident: mmap=%v required=%v", s.MMap, s.MMapRequired)
	}
	if s.PlannedHostFootprintMB <= 0 || s.PlannedHostFootprintMB >= wantMB {
		t.Fatalf("planned host footprint %d MiB must exclude the %d MiB lookup tier", s.PlannedHostFootprintMB, wantMB)
	}
	args := s.Args(model.Path, 8080)
	if !hasArgPair(args, SparseLookupLazyFlag, "on") || !hasArgPair(args, "--n-cpu-moe", "0") && s.NCPUMoE <= 0 {
		t.Fatalf("argv must state lazy lookup reads and CPU experts: %v", args)
	}
	if !strings.Contains(strings.Join(args, " "), "--no-mmap") {
		t.Fatalf("resident experts need --no-mmap (lazy tables stay mapped regardless): %v", args)
	}
}

// The defect this tier fixes: a schema-2 profile folds both ~97 GiB tables
// into layers 1 and 14 as ordinary backbone, which bills them to a GPU.
func TestSchema2EngramAccountingCannotPlaceTheModel(t *testing.T) {
	model := dsv41Profile()
	model.NonExpertBytes += model.SparseLookupBytes
	model.NonExpertLayerBytes[1] += model.SparseLookupBytes / 2
	model.NonExpertLayerBytes[14] += model.SparseLookupBytes / 2
	model.SparseLookupBytes, model.SparseLookupKind = 0, ""
	if s, err := Compute(dsv41Host(204700), model, dsv41Opts(t, SparseLookupFileBacked)); err == nil {
		t.Fatalf("schema-2 accounting unexpectedly produced a plan: type=%s ncpumoe=%d", s.Type, s.NCPUMoE)
	}
}

func TestSparseLookupTierRefusesAutomaticExpertPaging(t *testing.T) {
	model := dsv41Profile()
	_, err := Compute(dsv41Host(100000), model, dsv41Opts(t, SparseLookupFileBacked))
	if err == nil || !strings.Contains(err.Error(), "will not also page experts") {
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

func TestOrdinaryModelIgnoresSparseLookupCapability(t *testing.T) {
	model := dsv41Profile()
	model.SparseLookupBytes, model.SparseLookupKind, model.ModelArch = 0, "", "deepseek4"
	model.SizeBytes -= 208902215200
	for _, capability := range []SparseLookupCapability{SparseLookupUnknown, SparseLookupFileBacked} {
		s, err := Compute(dsv41Host(204700), model, dsv41Opts(t, capability))
		if err != nil {
			t.Fatalf("capability %q must not affect a model without a lookup tier: %v", capability, err)
		}
		if s.SparseLookupMB != 0 || strings.Contains(strings.Join(s.Args(model.Path, 8080), " "), SparseLookupLazyFlag) {
			t.Fatalf("ordinary model gained a lookup tier: %d MiB, args %v", s.SparseLookupMB, s.Args(model.Path, 8080))
		}
	}
	plain := &Strategy{Type: MoEOffload, ContextSize: 4096, MMap: false}
	before := AllocationPlacementIdentity(plain)
	plain.SparseLookupCapability = SparseLookupFileBacked
	if AllocationPlacementIdentity(plain) != before {
		t.Fatal("allocation identity of a plan without a lookup tier must not change")
	}
	plain.SparseLookupMB = 1
	if AllocationPlacementIdentity(plain) == before {
		t.Fatal("a lookup tier must scope allocation evidence")
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
}
