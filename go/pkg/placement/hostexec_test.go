package placement

import (
	"math"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

const mainlineHelp = "--op-offload, --no-op-offload  whether to offload host tensor operations to device (default: true)"

func TestResolveHostExpertExecutionScopesPolicy(t *testing.T) {
	gpus := []detect.GPU{{Index: 1, BandwidthMBps: 12000}, {Index: 2, BandwidthMBps: 6000}}
	caps := &detect.Capabilities{GPUs: gpus}
	cases := []struct {
		name       string
		caps       *detect.Capabilities
		opts       Options
		mode       HostExpertExecutionMode
		minBatch   int
		stagingGPU int
		confidence string
	}{
		{"flag off", caps, Options{BackendTag: "llama", BackendHelp: mainlineHelp, HostWeightOffload: "off"}, HostExpertExecCPU, 0, -1, "effective-setting"},
		{"cpu only", &detect.Capabilities{}, Options{BackendTag: "llama", BackendHelp: mainlineHelp}, HostExpertExecCPU, 0, -1, "effective-setting"},
		{"ik fork", caps, Options{BackendTag: "ik_llama", BackendHelp: mainlineHelp}, HostExpertExecUnknown, 0, -1, "unknown"},
		{"vulkan", caps, Options{BackendTag: "vulkan", BackendHelp: mainlineHelp}, HostExpertExecUnknown, 0, -1, "unknown"},
		{"no switch", caps, Options{BackendTag: "llama", BackendHelp: "--flash-attn"}, HostExpertExecUnknown, 0, -1, "unknown"},
		// The first enumerated device of a --gpus subset, not physical GPU 0.
		{"default", caps, Options{BackendTag: "llama", BackendHelp: mainlineHelp}, HostExpertExecStaged, 32, 1, "family-default"},
		{"env", caps, Options{BackendTag: "llama", BackendHelp: mainlineHelp, OpOffloadMinBatchEnv: "128", OpOffloadMinBatchEnvSet: true}, HostExpertExecStaged, 128, 1, "family-default"},
		// atoi("junk") is 0 in the backend: every batch, even decode, is staged.
		{"env junk", caps, Options{BackendTag: "llama", BackendHelp: mainlineHelp, OpOffloadMinBatchEnv: "junk", OpOffloadMinBatchEnvSet: true}, HostExpertExecStaged, 0, 1, "family-default"},
	}
	for _, c := range cases {
		got := ResolveHostExpertExecution(c.caps, c.opts)
		if got.Mode != c.mode || got.MinBatch != c.minBatch || got.StagingGPU != c.stagingGPU || got.Confidence != c.confidence {
			t.Errorf("%s: %+v", c.name, got)
		}
	}
	staged := ResolveHostExpertExecution(caps, Options{BackendTag: "llama", BackendHelp: mainlineHelp})
	if staged.StagedAt(31) || !staged.StagedAt(32) || staged.StagedAt(1) {
		t.Fatalf("threshold edges wrong: %+v", staged)
	}
	if staged.Fingerprint() == ResolveHostExpertExecution(caps, Options{BackendTag: "llama", BackendHelp: mainlineHelp, HostWeightOffload: "off"}).Fingerprint() {
		t.Fatal("policies share a fingerprint")
	}
}

// hostExpertFixture keeps every routed expert in host memory, the recorded
// all-experts-on-CPU shape, with 8 of 256 experts routed per token.
func hostExpertFixture() (*detect.Capabilities, *ModelProfile, *Strategy, ResourceLedger) {
	model := &ModelProfile{
		TotalSizeMB: 130000, SizeBytes: 130000 * 1048576, NumLayers: 48, IsMoE: true,
		ExpertBytes: 114192 * 1048576, NumExperts: 256, ExpertUsedCount: 8,
	}
	caps := &detect.Capabilities{HostMemoryBandwidthMBps: 60000, GPUs: []detect.GPU{
		{Index: 0, MemBandwidthMBps: 504000, BandwidthMBps: 12000},
		{Index: 1, MemBandwidthMBps: 1008000, BandwidthMBps: 12000},
	}}
	s := &Strategy{Type: MoEOffload, Parallel: 1, UBatchSize: 64, NCPUMoE: 48, TensorSplit: []float64{.25, .75}}
	ledger := ResourceLedger{Fits: true, Devices: []DeviceResourceLedger{
		{GPU: 0, Active: true, ModelMB: 4000, RequiredMB: 9000, BandwidthMBps: 504000},
		{GPU: 1, Active: true, ModelMB: 10000, RequiredMB: 18000, BandwidthMBps: 1008000},
	}}
	return caps, model, s, ledger
}

// Staged transfer is touched bytes over (microbatch x copy rate), in the same
// seconds-per-token units as decode, and it is counted once.
func TestStagedHostExpertTransferIsPerTokenAndCountedOnce(t *testing.T) {
	caps, model, base, ledger := hostExpertFixture()
	staged := Options{BackendTag: "llama", BackendHelp: mainlineHelp}
	cpu := Options{BackendTag: "llama", BackendHelp: mainlineHelp, HostWeightOffload: "off"}
	// Expected GiB crossing the link per prompt token under uniform routing
	// (docs/mimo-current-run-math-review-20260929.md): 1.514, 0.856, 0.436.
	wantGiB := map[int]float64{64: 1.514, 128: 0.856, 256: 0.436}
	prev := math.Inf(1)
	for _, ub := range []int{64, 128, 256} {
		s := cloneStrategy(base)
		s.UBatchSize = ub
		est := EstimateStrategyCost(caps, model, s, staged, ledger)
		h := est.HostExpert
		if h == nil || !h.PrefillStaged || h.CopyRateMBps != 12000 || h.ComputePriced {
			t.Fatalf("ubatch %d not priced as staged: %+v", ub, h)
		}
		gib := h.TransferSecPerToken * float64(h.CopyRateMBps) / 1024
		if math.Abs(gib-wantGiB[ub]) > 0.001 {
			t.Fatalf("ubatch %d moves %.4f GiB/token, want %.3f", ub, gib, wantGiB[ub])
		}
		if h.TransferSecPerToken >= prev {
			t.Fatalf("larger microbatch did not amortise the transfer")
		}
		prev = h.TransferSecPerToken
		if est.PrefillBottleneck != "GPU 0 host-expert staging link" {
			t.Fatalf("prefill bottleneck %q", est.PrefillBottleneck)
		}
		// Decode is one token: below the threshold it executes on CPU, so
		// staging must not change it.
		if cpuEst := EstimateStrategyCost(caps, model, s, cpu, ledger); cpuEst.DecodeCost != est.DecodeCost {
			t.Fatalf("staging policy changed decode: %v vs %v", est.DecodeCost, cpuEst.DecodeCost)
		}
		// A staged microbatch does not also pay the host-read prior: raising
		// host bandwidth above the link cannot change its prefill cost.
		faster := *caps
		faster.HostMemoryBandwidthMBps = 120000
		if again := EstimateStrategyCost(&faster, model, s, staged, ledger); again.PrefillCost != est.PrefillCost {
			t.Fatalf("host read double-counted with staging: %v vs %v", again.PrefillCost, est.PrefillCost)
		}
		if cpuEst := EstimateStrategyCost(&faster, model, s, cpu, ledger); cpuEst.PrefillCost == EstimateStrategyCost(caps, model, s, cpu, ledger).PrefillCost {
			t.Fatal("CPU execution ignores host bandwidth")
		}
	}
}

func TestHostExpertBelowThresholdOrUnknownKeepsCPUPath(t *testing.T) {
	caps, model, base, ledger := hostExpertFixture()
	cpu := Options{BackendTag: "llama", BackendHelp: mainlineHelp, HostWeightOffload: "off"}
	small := cloneStrategy(base)
	small.UBatchSize = 16
	short := EstimateStrategyCost(caps, model, small, Options{BackendTag: "llama", BackendHelp: mainlineHelp}, ledger)
	if short.HostExpert.PrefillStaged || short.PrefillCost != EstimateStrategyCost(caps, model, small, cpu, ledger).PrefillCost {
		t.Fatalf("a microbatch below the threshold was staged: %+v", short.HostExpert)
	}
	unknown := EstimateStrategyCost(caps, model, base, Options{BackendTag: "ik_llama", BackendHelp: mainlineHelp}, ledger)
	if unknown.HostExpert.Execution.Mode != HostExpertExecUnknown || unknown.HostExpert.PrefillStaged || unknown.Confidence != "low" {
		t.Fatalf("unreviewed backend acquired a policy: %+v", unknown)
	}
	// A staging device without a known link cannot be priced, and says so.
	caps.GPUs[0].BandwidthMBps = 0
	noLink := EstimateStrategyCost(caps, model, base, Options{BackendTag: "llama", BackendHelp: mainlineHelp}, ledger)
	if noLink.HostExpert.PrefillStaged || noLink.Confidence != "low" {
		t.Fatalf("unknown link priced as staged: %+v", noLink)
	}
}

// Performance evidence follows the execution policy; fit proof keys (built
// without a base) and bases without host experts do not.
func TestCalibrationScopeCarriesHostExecutionOnlyForHostExperts(t *testing.T) {
	caps, model, moe, _ := hostExpertFixture()
	on := Options{BackendTag: "llama", BackendHelp: mainlineHelp}
	off := on
	off.HostWeightOffload = "off"
	if k := NewCalibrationScopeKey(model, caps, on, nil); k.HostExecution != "" ||
		k.String() != NewCalibrationScopeKey(model, caps, off, nil).String() {
		t.Fatalf("verified-config scope depends on op offload: %+v", k)
	}
	if NewCalibrationScopeKey(model, caps, on, moe).String() == NewCalibrationScopeKey(model, caps, off, moe).String() {
		t.Fatal("decision measured with staging reused for CPU execution")
	}
	dense := &Strategy{Type: MultiGPUDense, TensorSplit: []float64{.5, .5}}
	if NewCalibrationScopeKey(model, caps, on, dense).String() != NewCalibrationScopeKey(model, caps, off, dense).String() {
		t.Fatal("op offload re-scoped a model without host experts")
	}
}

func TestStagedPrefillUBatchRaiseCandidatesRespectConstraints(t *testing.T) {
	caps, _, base, _ := hostExpertFixture()
	base.BatchSize = 2048
	staged := Options{BackendTag: "llama", BackendHelp: mainlineHelp}
	// Every power of two up to the logical batch: memory, not a fixed size,
	// decides how far a raise goes.
	if got := StagedPrefillUBatchRaiseCandidates(caps, staged, base); len(got) != 5 || got[0] != 2048 || got[4] != 128 {
		t.Fatalf("candidates %v, want [2048 1024 512 256 128]", got)
	}
	capped := cloneStrategy(base)
	capped.BatchSize = 256
	if got := StagedPrefillUBatchRaiseCandidates(caps, staged, capped); len(got) != 2 || got[0] != 256 {
		t.Fatalf("logical batch did not cap the raise: %v", got)
	}
	// Host layers of a dense model that does not fit are staged the same way.
	dense := &Strategy{Type: DenseCPUOffload, UBatchSize: 256, BatchSize: 1024, Parallel: 1}
	if got := StagedPrefillUBatchRaiseCandidates(caps, staged, dense); len(got) != 2 || got[0] != 1024 {
		t.Fatalf("dense host offload: %v", got)
	}
	fullyResident := &Strategy{Type: MultiGPUDense, UBatchSize: 256, BatchSize: 1024, Parallel: 1}
	if got := StagedPrefillUBatchRaiseCandidates(caps, staged, fullyResident); len(got) != 0 {
		t.Fatalf("fully resident plan proposed %v", got)
	}
	explicit := staged
	explicit.UBatchSizeExplicit = true
	tuned := cloneStrategy(base)
	tuned.BatchTuned = true
	resident := cloneStrategy(base)
	resident.NCPUMoE = 0
	slots := cloneStrategy(base)
	slots.Parallel = 2
	cases := map[string]struct {
		opts Options
		s    *Strategy
	}{
		"explicit ubatch": {explicit, base},
		"live-tuned":      {staged, tuned},
		"no host experts": {staged, resident},
		"parallel slots":  {staged, slots},
		"unknown backend": {Options{BackendTag: "ik_llama", BackendHelp: mainlineHelp}, base},
		"cpu execution":   {Options{BackendTag: "llama", BackendHelp: mainlineHelp, HostWeightOffload: "off"}, base},
	}
	for name, c := range cases {
		if got := StagedPrefillUBatchRaiseCandidates(caps, c.opts, c.s); len(got) != 0 {
			t.Errorf("%s: proposed %v", name, got)
		}
	}
}

func TestWithUBatchDropsMicrobatchScopedState(t *testing.T) {
	model := &ModelProfile{SlidingWindow: 128}
	s := &Strategy{UBatchSize: 64, BatchSize: 2048, ContextAllocationMB: 8477, ContextAllocationEvidence: "oracle",
		ResourceLedger: &ResourceLedger{Exact: true}, PlacementCachePath: "/cache/ub64.place", PlacementCacheHit: true,
		CheckpointMinStep: checkpointMinStep(model, 64)}
	next := WithUBatch(s, model, 512)
	if next.UBatchSize != 512 || next.ContextAllocationMB != 0 || next.ResourceLedger != nil ||
		next.PlacementCachePath != "" || next.PlacementCacheHit || next.CheckpointMinStep != checkpointMinStep(model, 512) {
		t.Fatalf("ubatch-scoped state survived: %+v", next)
	}
	if s.UBatchSize != 64 || s.PlacementCachePath == "" {
		t.Fatal("original strategy mutated")
	}
}

// The ladder trades microbatch for GPU expert layers only when that lowers the
// staged-prefill-plus-decode cost: two of fifty layers never pay for an 8x
// smaller staged batch, while moving most layers off the host does.
func TestHostExpertAgentCostRanksLadderTrades(t *testing.T) {
	caps, model, _, _ := hostExpertFixture()
	model.NumLayers, model.LeadingDense = 50, 0
	exec := ResolveHostExpertExecution(caps, Options{BackendTag: "llama", BackendHelp: mainlineHelp})
	cost := func(ub, cpu int) float64 {
		c, ok := hostExpertAgentCost(caps, model, exec, ub, cpu)
		if !ok {
			t.Fatal("cost unavailable")
		}
		return c
	}
	if cost(64, 48) <= cost(512, 50) {
		t.Fatal("two GPU expert layers bought an 8x smaller staged microbatch")
	}
	if cost(128, 10) >= cost(512, 50) {
		t.Fatal("moving 40 of 50 layers to GPU was not worth a 4x smaller microbatch")
	}
	caps.HostMemoryBandwidthMBps = 0
	if _, ok := hostExpertAgentCost(caps, model, exec, 512, 50); ok {
		t.Fatal("unknown host bandwidth priced")
	}
}
