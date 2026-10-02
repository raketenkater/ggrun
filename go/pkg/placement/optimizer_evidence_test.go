package placement

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func missingComputeFixture(t *testing.T) (*detect.Capabilities, *ModelProfile, *Strategy, Options) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(path, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	caps := &detect.Capabilities{GPUs: optimizerTestGPUs(), RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 60000}}
	model := &ModelProfile{Path: path, TotalSizeMB: 8000, SizeBytes: 8000 * 1048576, NumLayers: 32, EmbeddingLength: 4096}
	s := &Strategy{Type: MultiGPUDense, TensorSplit: []float64{.5, .5}, SplitMode: "layer", MainGPU: 0,
		ContextSize: 32768, ContextAllocationMB: 512, Parallel: 1, BatchSize: 2048, UBatchSize: 64,
		KVPlacement: "gpu", KVQuality: "q8_0", KVType: "q8_0", PlanFreeVRAM: map[int]int{0: 32768, 1: 32768}}
	return caps, model, s, Options{CacheDir: dir, BackendCacheTag: "fixture", RequireMeasuredBuffers: true}
}

func TestMissingComputeIsAdmissionNeededNotFit(t *testing.T) {
	caps, model, base, opts := missingComputeFixture(t)
	for _, ub := range []int{64, 128, 256, 512, 2048} {
		s := cloneStrategy(base)
		s.UBatchSize = ub
		ledger := BuildResourceLedger(caps, model, s, opts)
		if ledger.Fits || ledger.Exact || !ledger.NeedsAdmission || len(ledger.MissingComputeGPUs) != 2 {
			t.Fatalf("ubatch %d acquired false fit proof: %+v", ub, ledger)
		}
		est := EstimateStrategyCost(caps, model, s, opts, ledger)
		if est.Feasible || !est.NeedsAdmission || est.Confidence != "unknown-memory" || math.IsInf(est.AgentCost, 0) {
			t.Fatalf("unknown candidate lost its bounded admission path: %+v", est)
		}
		if _, err := json.Marshal(est); err != nil {
			t.Fatal(err)
		}
	}
}

func TestMissingComputeDoesNotHideKnownCapacityDeficit(t *testing.T) {
	caps, model, s, opts := missingComputeFixture(t)
	s.PlanFreeVRAM[0] = 1
	ledger := BuildResourceLedger(caps, model, s, opts)
	if ledger.Fits || ledger.NeedsAdmission || ledger.Devices[0].SlackMB >= 0 {
		t.Fatalf("known deficit became mere uncertainty: %+v", ledger)
	}
}

func TestComputeEvidenceDistinguishesPartialAndZeroRows(t *testing.T) {
	for _, complete := range []bool{false, true} {
		t.Run(map[bool]string{false: "missing-peer", true: "measured-zero-peer"}[complete], func(t *testing.T) {
			caps, model, s, opts := missingComputeFixture(t)
			rows := map[int]int{0: 512}
			if complete {
				rows[1] = 0
			}
			if err := RecordMeasuredComputeBuffers(opts.CacheDir, model, s.ContextSize, s.UBatchSize,
				s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel, rows); err != nil {
				t.Fatal(err)
			}
			ledger := BuildResourceLedger(caps, model, s, opts)
			if ledger.Fits != complete || ledger.NeedsAdmission == complete {
				t.Fatalf("partial/zero rows confused: %+v", ledger)
			}
			if !complete && (len(ledger.MissingComputeGPUs) != 1 || ledger.MissingComputeGPUs[0] != 1) {
				t.Fatalf("missing peer borrowed another GPU's aggregate: %+v", ledger)
			}
			if complete {
				if err := RecordMeasuredComputeBuffers(opts.CacheDir, model, s.ContextSize, s.UBatchSize,
					s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel, map[int]int{0: 768}); err != nil {
					t.Fatal(err)
				}
				if merged := BuildResourceLedger(caps, model, s, opts); !merged.Fits || merged.NeedsAdmission {
					t.Fatalf("partial refresh erased explicit zero peer: %+v", merged)
				}
			}
			other := cloneStrategy(s)
			other.UBatchSize *= 2
			if ledger := BuildResourceLedger(caps, model, other, opts); ledger.Fits || !ledger.NeedsAdmission {
				t.Fatalf("changed ubatch inherited compute proof: %+v", ledger)
			}
		})
	}
}

func TestUnknownFrontierKeepsNearestAdmissionAndBaseline(t *testing.T) {
	caps, model, base, opts := missingComputeFixture(t)
	near, far := cloneStrategy(base), cloneStrategy(base)
	near.UBatchSize, far.UBatchSize = 128, 2048
	frontier := AnalyzeCandidateFrontier(caps, model, opts, []CalibrationCandidate{
		{Name: "default", Strategy: base}, {Name: "ubatch-128", Strategy: near}, {Name: "ubatch-2048", Strategy: far},
	})
	if frontier[0].Strategy != base || frontier[1].Strategy != near {
		t.Fatalf("missing evidence reordered the nearest admission: %+v", frontier)
	}
	if base.Residency == ResidencyRoomy || base.OptimizationBoundary.FeasibleCount != 0 || base.OptimizationBoundary.UnmeasuredCount != 3 {
		t.Fatalf("unknown graph costs invented headroom: %+v", base.OptimizationBoundary)
	}
	// Exact baseline proof permits a same-shape neighbor to reach contained
	// admission, but uncertainty never authorizes a topology change.
	base.Residency, base.ResourceLedger = ResidencyTight, &ResourceLedger{Fits: true, Exact: true}
	far.TensorSplit = []float64{.8, .2}
	got := TightLiveCandidates(frontier)
	if len(got) != 2 || got[0].Strategy != base || got[1].Strategy != near {
		t.Fatalf("tight unknown admission lost its boundary: %+v", got)
	}
}

func TestCPUOnlyLedgerDoesNotRequireGPUCompute(t *testing.T) {
	caps, model, s, opts := missingComputeFixture(t)
	caps.GPUs = nil
	s.Type = CPUOnly
	ledger := BuildResourceLedger(caps, model, s, opts)
	if !ledger.Fits || ledger.NeedsAdmission || len(ledger.MissingComputeGPUs) != 0 {
		t.Fatalf("CPU serving depends on GPU evidence: %+v", ledger)
	}
}

// Staged host-expert transfer makes a huge microbatch look far cheaper per
// token. That prior must still not outrank measured memory evidence or reorder
// unmeasured rungs away from the nearest admission.
func TestStagingPriorCannotPromoteUnmeasuredMicrobatch(t *testing.T) {
	caps, model, base, opts := missingComputeFixture(t)
	model.IsMoE, model.NumExperts, model.ExpertUsedCount = true, 256, 8
	model.ExpertBytes, model.NumLayers = 6000*1048576, 32
	base.Type, base.NCPUMoE = MoEOffload, 32
	opts.BackendTag, opts.BackendHelp = "llama", "--op-offload, --no-op-offload"
	near, mid, far := cloneStrategy(base), cloneStrategy(base), cloneStrategy(base)
	near.UBatchSize, mid.UBatchSize, far.UBatchSize = 128, 256, 2048
	for _, s := range []*Strategy{base, mid} {
		if err := RecordMeasuredComputeBuffers(opts.CacheDir, model, s.ContextSize, s.UBatchSize,
			s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel, map[int]int{0: 900, 1: 900}); err != nil {
			t.Fatal(err)
		}
	}
	frontier := AnalyzeCandidateFrontier(caps, model, opts, []CalibrationCandidate{
		{Name: "default", Strategy: base}, {Name: "ubatch-2048", Strategy: far},
		{Name: "ubatch-128", Strategy: near}, {Name: "ubatch-256", Strategy: mid},
	})
	got := []*Strategy{frontier[0].Strategy, frontier[1].Strategy, frontier[2].Strategy, frontier[3].Strategy}
	if got[0] != base || got[1] != mid || got[2] != far || got[3] != near {
		t.Fatalf("order %v; want baseline, measured 256, then unmeasured in generator order", namesOf(frontier))
	}
	if frontier[2].Estimate.HostExpert == nil || !frontier[2].Estimate.HostExpert.PrefillStaged ||
		frontier[2].Estimate.PrefillCost >= frontier[1].Estimate.PrefillCost {
		t.Fatalf("fixture no longer exercises a cheaper unmeasured prior: %+v", frontier[2].Estimate)
	}
	if frontier[2].Estimate.Feasible || base.OptimizationBoundary.FeasibleCount != 2 {
		t.Fatalf("unmeasured rung counted as fitting: %+v", base.OptimizationBoundary)
	}
}

// Generator order still decides among unknown-memory candidates, but a
// candidate predicted slower than the baseline never goes first.
func TestUnknownFrontierNeverTestsAPredictedLossFirst(t *testing.T) {
	caps, model, base, opts := missingComputeFixture(t)
	slower, faster := cloneStrategy(base), cloneStrategy(base)
	slower.UBatchSize, faster.UBatchSize = 32, 128
	frontier := AnalyzeCandidateFrontier(caps, model, opts, []CalibrationCandidate{
		{Name: "default", Strategy: base}, {Name: "ubatch-32", Strategy: slower}, {Name: "ubatch-128", Strategy: faster},
	})
	if frontier[1].Strategy != faster {
		t.Fatalf("finalist %s is predicted slower than the baseline; order %v", frontier[1].Name, namesOf(frontier))
	}
	if frontier[2].Estimate.AgentCost < frontier[0].Estimate.AgentCost {
		t.Fatal("fixture no longer has a predicted loss")
	}
}
