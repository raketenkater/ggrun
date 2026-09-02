package placement

import (
	"fmt"
	"math"
	"sort"

	"github.com/raketenkater/ggrun/pkg/benchmark"
	"github.com/raketenkater/ggrun/pkg/detect"
)

// BottleneckKind is deliberately coarser than a hardware counter. ggrun only
// names a resource it can distinguish with current evidence; ambiguous CPU-MoE
// stalls remain one host-expert-path class until bandwidth counters separate
// DRAM from PCIe and synchronization.
type BottleneckKind string

const (
	BottleneckUnknown        BottleneckKind = "unknown"
	BottleneckCapacity       BottleneckKind = "memory_capacity"
	BottleneckGPUCompute     BottleneckKind = "gpu_compute"
	BottleneckGPUMemory      BottleneckKind = "gpu_memory"
	BottleneckGPUTopology    BottleneckKind = "gpu_topology"
	BottleneckHostExecution  BottleneckKind = "host_execution"
	BottleneckHostExpertPath BottleneckKind = "host_expert_path"
	BottleneckStorage        BottleneckKind = "storage_or_page_fault"
	BottleneckScheduler      BottleneckKind = "scheduler"
	BottleneckPipeline       BottleneckKind = "pipeline_underfill"
)

// PhaseBottleneck records one phase-local diagnosis and the observations that
// justify it. Confidence is measured, inferred, or unknown.
type PhaseBottleneck struct {
	Phase      benchmark.AgentPhase `json:"phase"`
	Kind       BottleneckKind       `json:"kind"`
	Confidence string               `json:"confidence"`
	Summary    string               `json:"summary"`
	Evidence   []string             `json:"evidence,omitempty"`
	Levers     []string             `json:"levers,omitempty"`
}

// WorkloadBottleneck is the diagnosis for one complete agent screen. It is
// advisory evidence for finalist selection; only a live phase-safe comparison
// can promote a different launch configuration.
type WorkloadBottleneck struct {
	Primary      BottleneckKind       `json:"primary"`
	PrimaryPhase benchmark.AgentPhase `json:"primary_phase,omitempty"`
	Confidence   string               `json:"confidence"`
	Summary      string               `json:"summary"`
	Phases       []PhaseBottleneck    `json:"phases,omitempty"`
	Levers       []string             `json:"levers,omitempty"`
}

// DiagnoseAgentBottleneck separates the fit boundary from the performance
// diagnosis, then classifies each measured phase conservatively.
func DiagnoseAgentBottleneck(caps *detect.Capabilities, model *ModelProfile, strategy *Strategy, result *benchmark.Result) WorkloadBottleneck {
	_ = model // reserved for architecture-specific counters without guessing today
	if strategy == nil {
		return unknownWorkloadBottleneck("no resolved launch strategy")
	}
	// Fit classification and performance diagnosis are separate stages. A tight
	// allocation constrains which challenger may be admitted, but it must not
	// erase measured phase evidence: an exactly admitted MoE can still have a
	// serial GPU topology or host-expert bottleneck worth one contained A/B.
	// Preserve the capacity-only answer when no phase sampler produced evidence.
	if strategy.Residency != "" && strategy.Residency != ResidencyRoomy &&
		(result == nil || len(result.PhaseUtilization) == 0) {
		summary := fmt.Sprintf("%s launch is bounded by memory capacity", strategy.Residency)
		if slack, ok := minimumResourceSlack(strategy.ResourceLedger); ok {
			summary = fmt.Sprintf("%s launch is bounded by memory capacity (minimum device slack %d MiB)", strategy.Residency, slack)
		}
		return WorkloadBottleneck{
			Primary: BottleneckCapacity, Confidence: "measured", Summary: summary,
			Levers: bottleneckLevers(BottleneckCapacity),
		}
	}
	if result == nil {
		return unknownWorkloadBottleneck("no agent workload result")
	}

	phases := make([]PhaseBottleneck, 0, len(result.PhaseUtilization))
	for _, phase := range result.PhaseUtilization {
		phases = append(phases, diagnoseAgentPhase(caps, strategy, result, phase))
	}
	if len(phases) == 0 {
		// Mixed request timing can still expose scheduler starvation when platform
		// resource counters are unavailable.
		if schedulerRatio, starved := mixedSchedulerRatio(result); starved {
			summary := fmt.Sprintf("mixed decode is scheduler-limited (%.0f%% of isolated per-lane decode)", schedulerRatio*100)
			return WorkloadBottleneck{
				Primary: BottleneckScheduler, PrimaryPhase: benchmark.AgentPhaseMixed,
				Confidence: "measured", Summary: summary,
				Levers: bottleneckLevers(BottleneckScheduler),
			}
		}
		return unknownWorkloadBottleneck("phase resource evidence unavailable")
	}

	primary := selectPrimaryPhaseBottleneck(phases, result.PhaseUtilization)
	return WorkloadBottleneck{
		Primary: primary.Kind, PrimaryPhase: primary.Phase,
		Confidence: primary.Confidence, Summary: primary.Summary,
		Phases: phases, Levers: append([]string(nil), primary.Levers...),
	}
}

// SelectBottleneckFinalist maps a measured diagnosis onto an already-computed
// safe frontier. It does not promote anything and does not invent a coordinate:
// the returned complete strategy must still pass exact admission and the live
// prefill/decode/mixed regression gates.
func SelectBottleneckFinalist(candidates []CalibrationCandidate, diagnosis WorkloadBottleneck) (CalibrationCandidate, bool) {
	if len(candidates) < 2 || candidates[0].Strategy == nil ||
		candidates[0].Strategy.Residency != ResidencyRoomy {
		return CalibrationCandidate{}, false
	}
	base := candidates[0].Strategy
	bestIndex := -1
	bestScore := 0.0
	bestCost := math.Inf(1)
	for i := 1; i < len(candidates); i++ {
		candidate := candidates[i]
		if candidate.Strategy == nil || !candidate.Estimate.Feasible ||
			candidate.Strategy.ResourceLedger == nil || !candidate.Strategy.ResourceLedger.Fits ||
			candidate.Strategy.MMapRequired || !sameDiagnosticWorkload(base, candidate.Strategy) {
			continue
		}
		score := bottleneckFinalistScore(base, candidate.Strategy, diagnosis)
		if score <= 0 {
			continue
		}
		cost := candidate.Estimate.AgentCost
		if cost <= 0 || math.IsNaN(cost) || math.IsInf(cost, 0) {
			cost = math.Inf(1)
		}
		if bestIndex < 0 || score > bestScore+1e-9 ||
			(math.Abs(score-bestScore) <= 1e-9 && cost < bestCost) {
			bestIndex, bestScore, bestCost = i, score, cost
		}
	}
	if bestIndex < 0 {
		return CalibrationCandidate{}, false
	}
	return candidates[bestIndex], true
}

func bottleneckFinalistScore(base, candidate *Strategy, diagnosis WorkloadBottleneck) float64 {
	if base == nil || candidate == nil {
		return 0
	}
	switch diagnosis.Primary {
	case BottleneckHostExpertPath:
		if candidate.HotExpertCacheSlots > base.HotExpertCacheSlots {
			// This is the only bounded coordinate that attacks host expert
			// traffic without disturbing the proven tensor topology. Rank it
			// ahead of a whole-layer reshuffle; the live workload still rejects
			// low-locality or upload-bound caches.
			return 200 + float64(candidate.HotExpertCacheSlots-base.HotExpertCacheSlots)
		}
		if base.NCPUMoE <= 0 || candidate.NCPUMoE >= base.NCPUMoE {
			return 0
		}
		return 100 + float64(base.NCPUMoE-candidate.NCPUMoE)
	case BottleneckPipeline:
		switch diagnosis.PrimaryPhase {
		case benchmark.AgentPhasePrefill, benchmark.AgentPhaseAppend:
			if candidate.UBatchSize <= base.UBatchSize {
				return 0
			}
			return 10 * math.Log2(float64(candidate.UBatchSize)/float64(max(1, base.UBatchSize)))
		case benchmark.AgentPhaseDecode, benchmark.AgentPhaseMixed:
			if candidate.Parallel <= base.Parallel {
				return 0
			}
			return 10 * float64(candidate.Parallel-base.Parallel)
		}
	case BottleneckScheduler:
		// Mixed-phase starvation means a long scheduler quantum is delaying an
		// already-decoding foreground lane. Test a shorter batch/ubatch at the
		// same slot count so requested concurrency and context stay comparable.
		if candidate.Parallel != base.Parallel {
			return 0
		}
		score := 0.0
		if base.BatchSize > 0 && candidate.BatchSize < base.BatchSize {
			score += 10 * math.Log2(float64(base.BatchSize)/float64(max(1, candidate.BatchSize)))
		}
		if base.UBatchSize > 0 && candidate.UBatchSize < base.UBatchSize {
			score += 5 * math.Log2(float64(base.UBatchSize)/float64(max(1, candidate.UBatchSize)))
		}
		return score
	case BottleneckStorage:
		if base.MMap && !candidate.MMap {
			return 10
		}
	}
	return 0
}

func sameDiagnosticWorkload(base, candidate *Strategy) bool {
	if base == nil || candidate == nil {
		return false
	}
	return base.ContextSize == candidate.ContextSize &&
		base.KVPlacement == candidate.KVPlacement &&
		base.KVQuality == candidate.KVQuality &&
		base.KVType == candidate.KVType &&
		base.KVTypeV == candidate.KVTypeV &&
		base.SWAFull == candidate.SWAFull &&
		base.ReasoningOff == candidate.ReasoningOff
}

func diagnoseAgentPhase(caps *detect.Capabilities, strategy *Strategy, result *benchmark.Result, phase benchmark.PhaseUtilization) PhaseBottleneck {
	out := PhaseBottleneck{Phase: phase.Phase, Kind: BottleneckUnknown, Confidence: "unknown"}
	maxSM, maxMemory := phaseGPUCeilings(phase.GPUUtilization)
	maxPCIeRX, maxPCIeTX := phasePCIeCeilings(phase.GPUUtilization)
	cpuLoad := phaseCPUWorkerLoad(strategy, phase)
	readMBps := 0.0
	if phase.DurationS > 0 {
		readMBps = float64(phase.ProcessReadBytes) / (1024 * 1024) / phase.DurationS
	}
	out.Evidence = append(out.Evidence,
		fmt.Sprintf("gpu_sm_max=%d%%", maxSM),
		fmt.Sprintf("gpu_memory_max=%d%%", maxMemory),
	)
	if maxPCIeRX > 0 || maxPCIeTX > 0 {
		out.Evidence = append(out.Evidence,
			fmt.Sprintf("pcie_rx_max=%dMiB/s", maxPCIeRX),
			fmt.Sprintf("pcie_tx_max=%dMiB/s", maxPCIeTX))
	}
	if phase.ProcessObservations > 0 {
		out.Evidence = append(out.Evidence,
			fmt.Sprintf("process_cpu=%.0f%%", phase.ProcessCPUPercent),
			fmt.Sprintf("process_rss_peak=%dMiB", phase.ProcessRSSPeakMB),
			fmt.Sprintf("process_read=%.1fMiB/s", readMBps),
		)
	}

	if phase.Phase == benchmark.AgentPhaseMixed {
		if ratio, starved := mixedSchedulerRatio(result); starved {
			out.Kind = BottleneckScheduler
			out.Confidence = "measured"
			out.Summary = fmt.Sprintf("mixed decode is scheduler-limited (%.0f%% of isolated per-lane decode)", ratio*100)
			if phase.QueueDeferredMax > 0 {
				out.Evidence = append(out.Evidence, fmt.Sprintf("queue_deferred_max=%d", phase.QueueDeferredMax))
			}
			out.Levers = bottleneckLevers(out.Kind)
			return out
		}
	}

	signal := AnalyzeDeviceBalance(strategy, benchmarkGPUUtilSamples(phase.GPUUtilization))
	if signal.Imbalanced {
		out.Kind = BottleneckGPUTopology
		out.Confidence = "measured"
		out.Summary = fmt.Sprintf("%s is topology-limited: GPU %d is at %d%% SM while GPU %d is at %d%%",
			phase.Phase, signal.BusyGPU, signal.BusySM, signal.IdleGPU, signal.IdleSM)
		out.Levers = bottleneckLevers(out.Kind)
		return out
	}

	// Sustained storage reads plus low CPU/GPU occupancy indicate that file
	// faults or model streaming are on the critical path. The threshold avoids
	// classifying incidental log/config reads as inference I/O.
	if readMBps >= 64 && maxSM < 60 && cpuLoad < 0.75 {
		out.Kind = BottleneckStorage
		out.Confidence = "inferred"
		out.Summary = fmt.Sprintf("%s is waiting on storage/page faults (%.0f MiB/s with low compute occupancy)", phase.Phase, readMBps)
		out.Levers = bottleneckLevers(out.Kind)
		return out
	}
	if maxMemory >= 80 && maxSM < 80 {
		out.Kind = BottleneckGPUMemory
		out.Confidence = "measured"
		out.Summary = fmt.Sprintf("%s is GPU-memory limited (%d%% memory activity, %d%% SM)", phase.Phase, maxMemory, maxSM)
		out.Levers = bottleneckLevers(out.Kind)
		return out
	}
	if maxSM >= 85 {
		out.Kind = BottleneckGPUCompute
		out.Confidence = "measured"
		out.Summary = fmt.Sprintf("%s saturates GPU compute (%d%% SM)", phase.Phase, maxSM)
		out.Levers = bottleneckLevers(out.Kind)
		return out
	}
	if phase.ProcessObservations > 0 && cpuLoad >= 0.85 {
		out.Kind = BottleneckHostExecution
		out.Confidence = "measured"
		out.Summary = fmt.Sprintf("%s saturates configured host workers (%.0f%% of worker capacity)", phase.Phase, cpuLoad*100)
		out.Levers = bottleneckLevers(out.Kind)
		return out
	}
	if strategy.IsMoE && strategy.NCPUMoE > 0 && phase.ProcessObservations > 0 && maxSM < 70 && cpuLoad >= 0.20 {
		out.Kind = BottleneckHostExpertPath
		out.Confidence = "inferred"
		out.Summary = fmt.Sprintf("%s is limited in the CPU-expert path", phase.Phase)
		if pressure, known := phasePCIePressure(caps, phase.GPUUtilization); known && pressure >= 0.65 {
			out.Confidence = "measured"
			out.Summary = fmt.Sprintf("%s saturates the CPU-expert transfer path (PCIe %.0f%% of link capacity; RX/TX %d/%d MiB/s)", phase.Phase, pressure*100, maxPCIeRX, maxPCIeTX)
		} else if maxPCIeRX > 0 || maxPCIeTX > 0 {
			out.Summary += fmt.Sprintf("; PCIe is active at RX/TX %d/%d MiB/s but is not proven saturated, so DRAM and synchronization remain candidates", maxPCIeRX, maxPCIeTX)
		} else {
			out.Summary += "; transfer counters unavailable, so DRAM and synchronization remain unresolved"
		}
		if caps != nil && caps.HostMemoryBandwidthMBps > 0 {
			out.Evidence = append(out.Evidence, fmt.Sprintf("measured_host_bandwidth=%dMB/s", caps.HostMemoryBandwidthMBps))
		}
		out.Levers = bottleneckLevers(out.Kind)
		return out
	}
	if phase.Observations > 0 && maxSM > 0 && maxSM < 40 && cpuLoad < 0.50 && readMBps < 64 {
		out.Kind = BottleneckPipeline
		out.Confidence = "inferred"
		out.Summary = fmt.Sprintf("%s under-fills available compute (GPU <=%d%% SM, host workers %.0f%% occupied)", phase.Phase, maxSM, cpuLoad*100)
		out.Levers = bottleneckLevers(out.Kind)
		return out
	}

	out.Summary = fmt.Sprintf("%s has insufficient counters for a safe bottleneck claim", phase.Phase)
	return out
}

func selectPrimaryPhaseBottleneck(phases []PhaseBottleneck, utilization []benchmark.PhaseUtilization) PhaseBottleneck {
	durations := make(map[benchmark.AgentPhase]float64, len(utilization))
	maxDuration := 0.0
	for _, phase := range utilization {
		durations[phase.Phase] = phase.DurationS
		if phase.DurationS > maxDuration {
			maxDuration = phase.DurationS
		}
	}
	phaseWeight := map[benchmark.AgentPhase]float64{
		benchmark.AgentPhasePrefill: 1.15,
		benchmark.AgentPhaseAppend:  1.30,
		benchmark.AgentPhaseDecode:  1.15,
		benchmark.AgentPhaseMixed:   1.25,
	}
	confidenceWeight := map[string]float64{"measured": 3, "inferred": 2, "unknown": 0.25}
	type ranked struct {
		index int
		score float64
	}
	ranking := make([]ranked, 0, len(phases))
	for i, phase := range phases {
		score := confidenceWeight[phase.Confidence] * phaseWeight[phase.Phase]
		if maxDuration > 0 {
			score *= 1 + math.Min(1, durations[phase.Phase]/maxDuration)
		}
		ranking = append(ranking, ranked{index: i, score: score})
	}
	sort.SliceStable(ranking, func(i, j int) bool { return ranking[i].score > ranking[j].score })
	if len(ranking) == 0 {
		return PhaseBottleneck{Kind: BottleneckUnknown, Confidence: "unknown", Summary: "no phase evidence"}
	}
	return phases[ranking[0].index]
}

func mixedSchedulerRatio(result *benchmark.Result) (float64, bool) {
	if result == nil || result.GenTPS <= 0 || result.MixedGenTPS <= 0 {
		return 0, false
	}
	lanes := result.Parallel
	if lanes < 1 {
		lanes = 1
	}
	isolatedPerLane := result.GenTPS / float64(lanes)
	if isolatedPerLane <= 0 {
		return 0, false
	}
	ratio := result.MixedGenTPS / isolatedPerLane
	return ratio, ratio < 0.75
}

func phaseCPUWorkerLoad(strategy *Strategy, phase benchmark.PhaseUtilization) float64 {
	if phase.ProcessObservations <= 0 || strategy == nil {
		return 0
	}
	workers := strategy.Threads
	if phase.Phase == benchmark.AgentPhasePrefill || phase.Phase == benchmark.AgentPhaseAppend ||
		phase.Phase == benchmark.AgentPhaseMixed {
		if strategy.ThreadsBatch > 0 {
			workers = strategy.ThreadsBatch
		}
	}
	if workers < 1 {
		workers = 1
	}
	return phase.ProcessCPUPercent / (float64(workers) * 100)
}

func phaseGPUCeilings(samples []benchmark.GPUUtilization) (maxSM, maxMemory int) {
	for _, sample := range samples {
		if sample.SMPercent > maxSM {
			maxSM = sample.SMPercent
		}
		if sample.MemPercent > maxMemory {
			maxMemory = sample.MemPercent
		}
	}
	return maxSM, maxMemory
}

func phasePCIeCeilings(samples []benchmark.GPUUtilization) (maxRX, maxTX int) {
	for _, sample := range samples {
		if sample.PCIeRXMBps > maxRX {
			maxRX = sample.PCIeRXMBps
		}
		if sample.PCIeTXMBps > maxTX {
			maxTX = sample.PCIeTXMBps
		}
	}
	return maxRX, maxTX
}

func phasePCIePressure(caps *detect.Capabilities, samples []benchmark.GPUUtilization) (float64, bool) {
	if caps == nil {
		return 0, false
	}
	bandwidth := make(map[int]int, len(caps.GPUs))
	for _, gpu := range caps.GPUs {
		if gpu.BandwidthMBps > 0 {
			bandwidth[gpu.Index] = gpu.BandwidthMBps
		}
	}
	maximum := 0.0
	known := false
	for _, sample := range samples {
		ceiling := bandwidth[sample.GPU]
		if ceiling <= 0 {
			continue
		}
		known = true
		pressure := float64(max(sample.PCIeRXMBps, sample.PCIeTXMBps)) / float64(ceiling)
		if pressure > maximum {
			maximum = pressure
		}
	}
	return maximum, known
}

func benchmarkGPUUtilSamples(samples []benchmark.GPUUtilization) []GPUUtilSample {
	out := make([]GPUUtilSample, 0, len(samples))
	for _, sample := range samples {
		out = append(out, GPUUtilSample{GPU: sample.GPU, SMPercent: sample.SMPercent, MemPercent: sample.MemPercent})
	}
	return out
}

func minimumResourceSlack(ledger *ResourceLedger) (int, bool) {
	if ledger == nil || len(ledger.Devices) == 0 {
		return 0, false
	}
	minimum := math.MaxInt
	for _, device := range ledger.Devices {
		if !device.Active {
			continue
		}
		if device.SlackMB < minimum {
			minimum = device.SlackMB
		}
	}
	if minimum == math.MaxInt {
		return 0, false
	}
	return minimum, true
}

func bottleneckLevers(kind BottleneckKind) []string {
	switch kind {
	case BottleneckCapacity:
		return []string{"preserve proven fit boundary", "reduce KV/context before changing quality", "use mmap/offload only as contained fallback"}
	case BottleneckGPUCompute:
		return []string{"rebalance serial layer ownership", "test parallelism only against per-lane decode", "avoid larger batches that regress decode"}
	case BottleneckGPUMemory:
		return []string{"rebalance resident weight bytes by measured VRAM bandwidth", "test quantization or topology changes"}
	case BottleneckGPUTopology:
		return []string{"move serial layer work off the saturated GPU", "retain expert-storage roles unless routing proves them active"}
	case BottleneckHostExecution:
		return []string{"tune physical-core count and affinity", "separate batch and decode thread settings"}
	case BottleneckHostExpertPath:
		return []string{"move complete expert layers into available VRAM", "tune physical-core affinity", "measure DRAM and host-device transfer before choosing a cache"}
	case BottleneckStorage:
		return []string{"prefer resident or warmed pages when memory permits", "reserve mmap for capacity recovery"}
	case BottleneckScheduler:
		return []string{"test slots and parallelism at fixed requested concurrency", "preserve mixed foreground decode"}
	case BottleneckPipeline:
		return []string{"test larger ubatch for prefill", "test bounded request parallelism", "reduce transfer and synchronization boundaries"}
	default:
		return nil
	}
}

func unknownWorkloadBottleneck(summary string) WorkloadBottleneck {
	return WorkloadBottleneck{Primary: BottleneckUnknown, Confidence: "unknown", Summary: summary}
}
