package placement

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/benchmark"
	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestDiagnoseAgentBottleneckKeepsTightFitAsCapacityStage(t *testing.T) {
	strategy := &Strategy{
		Residency: ResidencyTight,
		ResourceLedger: &ResourceLedger{Devices: []DeviceResourceLedger{
			{GPU: 0, Active: true, SlackMB: 384},
			{GPU: 1, Active: true, SlackMB: 1024},
		}},
	}
	got := DiagnoseAgentBottleneck(nil, nil, strategy, &benchmark.Result{})
	if got.Primary != BottleneckCapacity || got.Confidence != "measured" ||
		!strings.Contains(got.Summary, "384 MiB") {
		t.Fatalf("capacity diagnosis=%+v", got)
	}
}

func TestDiagnoseAgentBottleneckFindsPhaseTopologyImbalance(t *testing.T) {
	strategy := &Strategy{
		Residency: ResidencyRoomy, Type: MultiGPUDense, TensorSplit: []float64{0.5, 0.5},
		ResourceLedger: &ResourceLedger{Devices: []DeviceResourceLedger{
			{GPU: 0, Active: true}, {GPU: 1, Active: true},
		}},
	}
	result := &benchmark.Result{PhaseUtilization: []benchmark.PhaseUtilization{{
		Phase: benchmark.AgentPhasePrefill, DurationS: 2, Observations: 4,
		GPUUtilization: []benchmark.GPUUtilization{
			{GPU: 0, SMPercent: 95}, {GPU: 1, SMPercent: 3},
		},
	}}}
	got := DiagnoseAgentBottleneck(nil, nil, strategy, result)
	if got.Primary != BottleneckGPUTopology || got.Confidence != "measured" {
		t.Fatalf("topology diagnosis=%+v", got)
	}
}

func TestDiagnoseAgentBottleneckKeepsTightFitAndDiagnosesMeasuredPhase(t *testing.T) {
	strategy := &Strategy{
		Residency: ResidencyTight, Type: MoEOffload, TensorSplit: []float64{0.5, 0.5},
		ResourceLedger: &ResourceLedger{Exact: true, Fits: true, Devices: []DeviceResourceLedger{
			{GPU: 0, Active: true, SlackMB: 384}, {GPU: 1, Active: true, SlackMB: 512},
		}},
	}
	result := &benchmark.Result{PhaseUtilization: []benchmark.PhaseUtilization{{
		Phase: benchmark.AgentPhasePrefill, DurationS: 2, Observations: 4,
		GPUUtilization: []benchmark.GPUUtilization{
			{GPU: 0, SMPercent: 91}, {GPU: 1, SMPercent: 4},
		},
	}}}
	got := DiagnoseAgentBottleneck(nil, nil, strategy, result)
	if got.Primary != BottleneckGPUTopology || strategy.Residency != ResidencyTight {
		t.Fatalf("fit/performance stages were conflated: diagnosis=%+v residency=%q", got, strategy.Residency)
	}
}

func TestDiagnoseAgentBottleneckKeepsCPUExpertPathComposite(t *testing.T) {
	strategy := &Strategy{
		Residency: ResidencyRoomy, Type: MoEOffload, IsMoE: true, NCPUMoE: 20,
		Threads: 14, ThreadsBatch: 14,
	}
	result := &benchmark.Result{PhaseUtilization: []benchmark.PhaseUtilization{{
		Phase: benchmark.AgentPhaseDecode, DurationS: 4, Observations: 8,
		ProcessCPUPercent: 500, ProcessObservations: 8,
		GPUUtilization: []benchmark.GPUUtilization{{GPU: 0, SMPercent: 25, MemPercent: 20}},
	}}}
	caps := &detect.Capabilities{HostMemoryBandwidthMBps: 42000}
	got := DiagnoseAgentBottleneck(caps, nil, strategy, result)
	if got.Primary != BottleneckHostExpertPath || got.Confidence != "inferred" {
		t.Fatalf("host expert diagnosis=%+v", got)
	}
	if !strings.Contains(strings.Join(got.Phases[0].Evidence, " "), "42000") {
		t.Fatalf("measured host bandwidth missing: %+v", got.Phases[0])
	}
}

func TestDiagnoseAgentBottleneckClaimsPCIeOnlyAgainstKnownCeiling(t *testing.T) {
	strategy := &Strategy{Residency: ResidencyRoomy, Type: MoEOffload, IsMoE: true, NCPUMoE: 20, Threads: 14, ThreadsBatch: 14}
	result := &benchmark.Result{PhaseUtilization: []benchmark.PhaseUtilization{{
		Phase: benchmark.AgentPhaseDecode, DurationS: 4, Observations: 8,
		ProcessCPUPercent: 500, ProcessObservations: 8,
		GPUUtilization: []benchmark.GPUUtilization{{GPU: 0, SMPercent: 25, PCIeRXMBps: 7000}},
	}}}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, BandwidthMBps: 8000}}}
	got := DiagnoseAgentBottleneck(caps, nil, strategy, result)
	if got.Primary != BottleneckHostExpertPath || got.Confidence != "measured" || !strings.Contains(got.Summary, "88%") {
		t.Fatalf("PCIe saturation diagnosis=%+v", got)
	}
	result.PhaseUtilization[0].GPUUtilization[0].PCIeRXMBps = 500
	got = DiagnoseAgentBottleneck(caps, nil, strategy, result)
	if got.Confidence != "inferred" || !strings.Contains(got.Summary, "not proven saturated") {
		t.Fatalf("low PCIe activity was overstated: %+v", got)
	}
}

func TestDiagnoseAgentBottleneckDetectsMixedSchedulerRegression(t *testing.T) {
	strategy := &Strategy{Residency: ResidencyRoomy, Threads: 8, ThreadsBatch: 8}
	result := &benchmark.Result{
		Parallel: 2, GenTPS: 20, MixedGenTPS: 4,
		PhaseUtilization: []benchmark.PhaseUtilization{{
			Phase: benchmark.AgentPhaseMixed, DurationS: 3, Observations: 6,
			GPUUtilization: []benchmark.GPUUtilization{{GPU: 0, SMPercent: 92}},
		}},
	}
	got := DiagnoseAgentBottleneck(nil, nil, strategy, result)
	if got.Primary != BottleneckScheduler || !strings.Contains(got.Summary, "40%") {
		t.Fatalf("scheduler diagnosis=%+v", got)
	}
}

func TestDiagnoseAgentBottleneckDistinguishesGPUMemoryAndUnderfill(t *testing.T) {
	strategy := &Strategy{Residency: ResidencyRoomy, Threads: 8, ThreadsBatch: 8}
	result := &benchmark.Result{PhaseUtilization: []benchmark.PhaseUtilization{
		{
			Phase: benchmark.AgentPhasePrefill, DurationS: 4, Observations: 8,
			GPUUtilization: []benchmark.GPUUtilization{{GPU: 0, SMPercent: 55, MemPercent: 91}},
		},
		{
			Phase: benchmark.AgentPhaseDecode, DurationS: 1, Observations: 4,
			GPUUtilization: []benchmark.GPUUtilization{{GPU: 0, SMPercent: 20, MemPercent: 15}},
		},
	}}
	got := DiagnoseAgentBottleneck(nil, nil, strategy, result)
	if got.Primary != BottleneckGPUMemory {
		t.Fatalf("primary diagnosis=%+v", got)
	}
	if len(got.Phases) != 2 || got.Phases[1].Kind != BottleneckPipeline {
		t.Fatalf("decode underfill was lost: %+v", got.Phases)
	}
}

func TestDiagnoseAgentBottleneckFailsClosedWithoutEvidence(t *testing.T) {
	got := DiagnoseAgentBottleneck(nil, nil, &Strategy{Residency: ResidencyRoomy}, &benchmark.Result{})
	if got.Primary != BottleneckUnknown || got.Confidence != "unknown" {
		t.Fatalf("missing evidence became a claim: %+v", got)
	}
}

func TestSelectBottleneckFinalistTargetsPrefillUnderfill(t *testing.T) {
	base := diagnosticCandidateStrategy()
	base.UBatchSize = 256
	base.BatchSize = 2048
	parallel := diagnosticCandidateStrategy()
	parallel.UBatchSize, parallel.BatchSize, parallel.Parallel = 256, 2048, 4
	largerUBatch := diagnosticCandidateStrategy()
	largerUBatch.UBatchSize, largerUBatch.BatchSize = 1024, 2048
	candidates := []CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true, AgentCost: 10}},
		{Name: "parallel-4", Strategy: parallel, Estimate: CandidateEstimate{Feasible: true, AgentCost: 4}},
		{Name: "ubatch-1024", Strategy: largerUBatch, Estimate: CandidateEstimate{Feasible: true, AgentCost: 6}},
	}
	got, ok := SelectBottleneckFinalist(candidates, WorkloadBottleneck{
		Primary: BottleneckPipeline, PrimaryPhase: benchmark.AgentPhasePrefill,
	})
	if !ok || got.Name != "ubatch-1024" {
		t.Fatalf("prefill-directed finalist=(%+v,%t)", got, ok)
	}
}

func TestSelectBottleneckFinalistTargetsMixedSchedulerQuantum(t *testing.T) {
	base := diagnosticCandidateStrategy()
	base.Parallel, base.BatchSize, base.UBatchSize = 2, 2048, 512
	shorter := diagnosticCandidateStrategy()
	shorter.Parallel, shorter.BatchSize, shorter.UBatchSize = 2, 512, 256
	wider := diagnosticCandidateStrategy()
	wider.Parallel, wider.BatchSize, wider.UBatchSize = 4, 512, 256
	candidates := []CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true, AgentCost: 10}},
		{Name: "parallel-4", Strategy: wider, Estimate: CandidateEstimate{Feasible: true, AgentCost: 3}},
		{Name: "batch-512-ubatch-256", Strategy: shorter, Estimate: CandidateEstimate{Feasible: true, AgentCost: 7}},
	}
	got, ok := SelectBottleneckFinalist(candidates, WorkloadBottleneck{
		Primary: BottleneckScheduler, PrimaryPhase: benchmark.AgentPhaseMixed,
	})
	if !ok || got.Name != "batch-512-ubatch-256" {
		t.Fatalf("scheduler-directed finalist=(%+v,%t)", got, ok)
	}
}

func TestSelectBottleneckFinalistTargetsMoreResidentExperts(t *testing.T) {
	base := diagnosticCandidateStrategy()
	base.IsMoE, base.Type, base.NCPUMoE = true, MoEOffload, 20
	denser := diagnosticCandidateStrategy()
	denser.IsMoE, denser.Type, denser.NCPUMoE = true, MoEOffload, 12
	unchanged := diagnosticCandidateStrategy()
	unchanged.IsMoE, unchanged.Type, unchanged.NCPUMoE = true, MoEOffload, 20
	candidates := []CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true, AgentCost: 10}},
		{Name: "topology-only", Strategy: unchanged, Estimate: CandidateEstimate{Feasible: true, AgentCost: 3}},
		{Name: "experts-denser", Strategy: denser, Estimate: CandidateEstimate{Feasible: true, AgentCost: 8}},
	}
	got, ok := SelectBottleneckFinalist(candidates, WorkloadBottleneck{Primary: BottleneckHostExpertPath})
	if !ok || got.Name != "experts-denser" {
		t.Fatalf("expert-directed finalist=(%+v,%t)", got, ok)
	}
}

func TestSelectBottleneckFinalistPrefersMeasuredHotCacheCoordinateForHostExpertPath(t *testing.T) {
	base := diagnosticCandidateStrategy()
	base.IsMoE, base.Type, base.NCPUMoE = true, MoEOffload, 20
	denser := diagnosticCandidateStrategy()
	denser.IsMoE, denser.Type, denser.NCPUMoE = true, MoEOffload, 12
	hot := diagnosticCandidateStrategy()
	hot.IsMoE, hot.Type, hot.NCPUMoE, hot.HotExpertCacheSlots = true, MoEOffload, 20, 8
	candidates := []CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true, AgentCost: 10}},
		{Name: "experts-denser", Strategy: denser, Estimate: CandidateEstimate{Feasible: true, AgentCost: 2}},
		{Name: "hot-experts-8", Strategy: hot, Estimate: CandidateEstimate{Feasible: true, AgentCost: 9}},
	}
	got, ok := SelectBottleneckFinalist(candidates, WorkloadBottleneck{Primary: BottleneckHostExpertPath})
	if !ok || got.Name != "hot-experts-8" {
		t.Fatalf("host-expert finalist=(%+v,%t)", got, ok)
	}
}

func TestSelectBottleneckFinalistDoesNotSpendTightCapacity(t *testing.T) {
	base := diagnosticCandidateStrategy()
	base.Residency = ResidencyTight
	challenger := diagnosticCandidateStrategy()
	challenger.UBatchSize = base.UBatchSize * 2
	_, ok := SelectBottleneckFinalist([]CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true}},
		{Name: "larger", Strategy: challenger, Estimate: CandidateEstimate{Feasible: true}},
	}, WorkloadBottleneck{Primary: BottleneckPipeline, PrimaryPhase: benchmark.AgentPhasePrefill})
	if ok {
		t.Fatal("tight capacity admitted a telemetry-directed performance reshuffle")
	}
}

func diagnosticCandidateStrategy() *Strategy {
	return &Strategy{
		Residency: ResidencyRoomy, Type: MultiGPUDense,
		ContextSize: 32768, BatchSize: 512, UBatchSize: 256, Parallel: 1,
		KVPlacement: "gpu", KVQuality: "mid", KVType: "q8_0",
		ResourceLedger: &ResourceLedger{Fits: true},
	}
}
