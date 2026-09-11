package benchmark

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	agentBenchmarkMaxLanes  = 4
	agentBenchmarkGenTokens = 64
	// Roughly 8k tokens: long enough for batch/ubatch and scheduler differences
	// to dominate request overhead, while remaining portable at the 32k minimum
	// useful agent context. The same deterministic corpus is reused by every
	// candidate and contains no repository/user data.
	agentBenchmarkPromptLen = 32768
	agentBenchmarkProbeLen  = 1024
	agentBenchmarkMinLen    = 1024
	agentBenchmarkQuantum   = 256
	agentBenchmarkUBatchCap = 512
	agentBenchmarkTrials    = 2 // one noisy turn is never promotion evidence
	defaultGPUUtilInterval  = 250 * time.Millisecond
	activeGPUUtilThreshold  = 5
)

type timedChatResult struct {
	result  *chatResult
	elapsed time.Duration
	err     error
}

// PrefillProbe is a small measured sample used only to size a bounded automatic
// agent screen. PromptBytes and PromptTokens retain the tokenizer geometry so
// sizing does not assume four bytes per token on every public model.
type PrefillProbe struct {
	PromptBytes  int
	PromptTokens int
	PromptTPS    float64
	WallTimeS    float64
}

// ProbeAgentPrefill measures a short uncached prompt against the already-loaded
// baseline. It does not choose a winner and candidates do not rerun it: the one
// resulting prompt size is held identical for both sides of the comparison.
func (r *Runner) ProbeAgentPrefill() (PrefillProbe, error) {
	prompt := agentBenchmarkLongPromptSized(r.WorkloadID, -1, 0, agentBenchmarkProbeLen)
	started := time.Now()
	result, err := r.chatWithOptions(prompt, 1, 1, false, 73)
	elapsed := time.Since(started)
	if err != nil {
		return PrefillProbe{}, err
	}
	if result == nil {
		return PrefillProbe{}, fmt.Errorf("prefill probe returned no result")
	}
	tokens := result.PromptTokens
	if tokens <= 0 {
		tokens = estimateTokens(prompt)
	}
	tps := result.PromptTPS
	if tps <= 0 && elapsed > 0 {
		tps = float64(tokens) / elapsed.Seconds()
	}
	if tokens <= 0 || tps <= 0 {
		return PrefillProbe{}, fmt.Errorf("prefill probe returned incomplete timings")
	}
	return PrefillProbe{
		PromptBytes: len(prompt), PromptTokens: tokens,
		PromptTPS: tps, WallTimeS: elapsed.Seconds(),
	}, nil
}

// SuggestedAgentPromptBytes converts one measured baseline prefill into a
// per-lane prompt which keeps each active wave near target. The minimum still
// spans at least one useful microbatch (bounded at 512 tokens); the maximum is
// the full ~8k-token maintenance workload. Rounding makes the generated scope
// stable against tiny timing drift.
func SuggestedAgentPromptBytes(probe PrefillProbe, activeSlots, ubatch int, target time.Duration) int {
	if activeSlots < 1 {
		activeSlots = 1
	}
	if target <= 0 {
		target = time.Minute
	}
	bytesPerToken := 4.0
	if probe.PromptBytes > 0 && probe.PromptTokens > 0 {
		bytesPerToken = float64(probe.PromptBytes) / float64(probe.PromptTokens)
	}
	minTokens := ubatch
	if minTokens < agentBenchmarkMinLen/4 {
		minTokens = agentBenchmarkMinLen / 4
	}
	if minTokens > agentBenchmarkUBatchCap {
		minTokens = agentBenchmarkUBatchCap
	}
	minimum := int(float64(minTokens) * bytesPerToken)
	if minimum < agentBenchmarkMinLen {
		minimum = agentBenchmarkMinLen
	}
	wanted := minimum
	if probe.PromptTPS > 0 {
		wanted = int(probe.PromptTPS*target.Seconds()*bytesPerToken/float64(activeSlots) + 0.5)
	}
	if wanted < minimum {
		wanted = minimum
	}
	if wanted > agentBenchmarkPromptLen {
		wanted = agentBenchmarkPromptLen
	}
	wanted = ((wanted + agentBenchmarkQuantum - 1) / agentBenchmarkQuantum) * agentBenchmarkQuantum
	if wanted > agentBenchmarkPromptLen {
		wanted = agentBenchmarkPromptLen
	}
	return wanted
}

// RunAgentParallel screens the serving shape that matters for an agent
// workflow: concurrent cold ingestion, a cache-backed append turn, concurrent
// decode throughput, and a foreground decode while the other lanes ingest long
// prompts. Cold ingestion plus the cache-backed append is the primary objective;
// throughput metrics are diagnostics and tie-break evidence. Two distinct
// samples prevent one noisy request from becoming promotion evidence. This
// remains a bounded screen; the caller must still pass the normal functional,
// branch/replay, cache, gateway, and clean-relaunch gates before it can promote
// an automatic default.
func (r *Runner) RunAgentParallel(slots int) (*Result, error) {
	slots = max(1, min(slots, agentBenchmarkMaxLanes))
	return r.RunAgentWorkload(slots, slots)
}

// RunAgentWorkload sends the same requested work regardless of server capacity.
// A narrow server must actually queue the excess requests: multiplying a smaller
// sample cannot measure queue delays, cache eviction, or mixed-request contention.
// Reject unsupported demand instead of silently claiming an unmeasured workflow.
func (r *Runner) RunAgentWorkload(slots, lanes int) (*Result, error) {
	if slots < 1 || lanes < 1 || lanes > 8 {
		return nil, fmt.Errorf("agent workload requires positive slots and 1..8 requested lanes")
	}

	warmupBytes := agentBenchmarkProbeLen
	if r.AgentPromptBytes > 0 && r.AgentPromptBytes < warmupBytes {
		warmupBytes = r.AgentPromptBytes
	}
	warmup := agentBenchmarkLongPromptSized(r.WorkloadID, -2, 0, warmupBytes)
	if _, err := r.chatWithOptions(warmup, 8, 1, false, 1); err != nil {
		return nil, fmt.Errorf("agent warm-up: %w", err)
	}
	trials := make([]*Result, 0, agentBenchmarkTrials)
	for trial := 0; trial < agentBenchmarkTrials; trial++ {
		result, err := r.runAgentParallelTrial(lanes, trial)
		if err != nil {
			return nil, fmt.Errorf("agent sample %d: %w", trial+1, err)
		}
		trials = append(trials, result)
	}
	result := aggregateAgentTrials(trials)
	result.Parallel = slots
	result.AgentWorkloadLanes = lanes
	result.AgentWorkloadTimeS = result.AgentScenarioTimeS
	result.AgentWorkloadMaxS = result.AgentScenarioMaxS
	return result, nil
}

func (r *Runner) runAgentParallelTrial(lanes, trial int) (*Result, error) {
	prefillPrompts := make([]string, lanes)
	for lane := range prefillPrompts {
		prefillPrompts[lane] = r.agentBenchmarkLongPrompt(trial, lane)
	}
	phaseUtilization := make([]PhaseUtilization, 0, 4)
	stopPrefillObservation := startPhaseResourceObservation(r, AgentPhasePrefill)
	prefill, prefillWall, err := r.runParallelChats(prefillPrompts, 1, 1, true, 100+trial*1000)
	phaseUtilization = append(phaseUtilization, stopPrefillObservation())
	if err != nil {
		return nil, fmt.Errorf("parallel prefill: %w", err)
	}
	promptTokens := sumPromptTokens(prefill)
	if promptTokens <= 0 {
		for _, prompt := range prefillPrompts {
			promptTokens += estimateTokens(prompt)
		}
	}
	promptTPS := 0.0
	if prefillWall > 0 {
		promptTPS = float64(promptTokens) / prefillWall.Seconds()
	}

	// Measure the thing users wait for: a new agent turn extending a resident
	// conversation. Keeping the original text byte-for-byte at the front lets
	// the backend report exact cache_n/cached_tokens evidence for the reuse.
	appendPrompts := make([]string, lanes)
	for lane := range appendPrompts {
		appendPrompts[lane] = prefillPrompts[lane] + fmt.Sprintf("\nTool result for lane %d: tests passed. Produce the next compact engineering action list.", lane)
	}
	stopAppendObservation := startPhaseResourceObservation(r, AgentPhaseAppend)
	appendResults, appendWall, err := r.runParallelChats(appendPrompts, agentBenchmarkGenTokens, agentBenchmarkGenTokens, true, 200+trial*1000)
	phaseUtilization = append(phaseUtilization, stopAppendObservation())
	if err != nil {
		return nil, fmt.Errorf("cache-backed append turn: %w", err)
	}
	cachedTokens := minCachedTokens(appendResults)
	newPromptTokens := sumNewPromptTokens(appendResults)

	genPrompts := make([]string, lanes)
	for lane := range genPrompts {
		genPrompts[lane] = fmt.Sprintf("Lane %d: produce a compact deterministic engineering checklist. Continue until the token budget is exhausted.", lane)
	}
	stopDecodeObservation := startPhaseResourceObservation(r, AgentPhaseDecode)
	generation, genWall, err := r.runParallelChats(genPrompts, agentBenchmarkGenTokens, agentBenchmarkGenTokens, false, 300+trial*1000)
	phaseUtilization = append(phaseUtilization, stopDecodeObservation())
	if err != nil {
		return nil, fmt.Errorf("parallel generation: %w", err)
	}
	genTokens := sumCompletionTokens(generation)
	genTPS := 0.0
	if genWall > 0 {
		genTPS = float64(genTokens) / genWall.Seconds()
	}

	stopMixedObservation := startPhaseResourceObservation(r, AgentPhaseMixed)
	mixedTokens, mixedWall, mixedTPS, err := r.runMixedAgentPhase(lanes, trial)
	phaseUtilization = append(phaseUtilization, stopMixedObservation())
	if err != nil {
		return nil, fmt.Errorf("mixed agent phase: %w", err)
	}
	var gpuUtilization []GPUUtilization
	for _, phase := range phaseUtilization {
		gpuUtilization = mergeGPUUtilization(gpuUtilization, phase.GPUUtilization)
	}

	return &Result{
		Model:                r.Model,
		PromptTokens:         promptTokens,
		PromptTimeS:          prefillWall.Seconds(),
		PromptTPS:            promptTPS,
		GenTokens:            genTokens,
		GenTimeS:             genWall.Seconds(),
		GenTPS:               genTPS,
		Parallel:             lanes,
		MixedGenTokens:       mixedTokens,
		MixedTimeS:           mixedWall.Seconds(),
		MixedGenTPS:          mixedTPS,
		AgentTurnTimeS:       appendWall.Seconds(),
		AgentTurnMaxS:        appendWall.Seconds(),
		AgentScenarioTimeS:   prefillWall.Seconds() + appendWall.Seconds(),
		AgentScenarioMaxS:    prefillWall.Seconds() + appendWall.Seconds(),
		AgentPromptBytes:     len(prefillPrompts[0]),
		AgentSamples:         1,
		AgentCachedTokens:    cachedTokens,
		AgentNewPromptTokens: newPromptTokens,
		GPUUtilization:       gpuUtilization,
		PhaseUtilization:     phaseUtilization,
		Timestamp:            time.Now().Unix(),
	}, nil
}

func (r *Runner) runParallelChats(prompts []string, maxTokens, minTokens int, cachePrompt bool, seedBase int) ([]timedChatResult, time.Duration, error) {
	results := make([]timedChatResult, len(prompts))
	started := time.Now()
	var wg sync.WaitGroup
	for i, prompt := range prompts {
		wg.Add(1)
		go func(index int, text string) {
			defer wg.Done()
			callStarted := time.Now()
			result, err := r.chatWithOptions(text, maxTokens, minTokens, cachePrompt, seedBase+index)
			results[index] = timedChatResult{result: result, elapsed: time.Since(callStarted), err: err}
		}(i, prompt)
	}
	wg.Wait()
	wall := time.Since(started)
	for i, result := range results {
		if result.err != nil {
			return nil, wall, fmt.Errorf("lane %d: %w", i, result.err)
		}
		if result.result == nil {
			return nil, wall, fmt.Errorf("lane %d returned no result", i)
		}
	}
	return results, wall, nil
}

func (r *Runner) runMixedAgentPhase(lanes, trial int) (int, time.Duration, float64, error) {
	if lanes <= 1 {
		started := time.Now()
		result, err := r.chatWithOptions("Produce a deterministic engineering checklist.", agentBenchmarkGenTokens, agentBenchmarkGenTokens, false, 400+trial*1000)
		elapsed := time.Since(started)
		if err != nil {
			return 0, elapsed, 0, err
		}
		tokens := result.CompletionTokens
		if tokens <= 0 {
			tokens = estimateTokens(result.Content)
		}
		return tokens, elapsed, tokensPerSecond(tokens, elapsed), nil
	}

	decodeDone := make(chan timedChatResult, 1)
	go func() {
		started := time.Now()
		result, err := r.chatWithOptions("Foreground lane: produce a deterministic engineering checklist until the token budget is exhausted.", agentBenchmarkGenTokens, agentBenchmarkGenTokens, false, 400+trial*1000)
		decodeDone <- timedChatResult{result: result, elapsed: time.Since(started), err: err}
	}()

	// Give the short-prompt lane a small head start so it enters decode before
	// long-prompt workers arrive. This deliberately recreates the production
	// failure mode: an already-decoding main turn starved by workflow prefill.
	time.Sleep(75 * time.Millisecond)
	prompts := make([]string, lanes-1)
	for i := range prompts {
		prompts[i] = r.agentBenchmarkLongPrompt(trial, 1000+i)
	}
	if _, _, err := r.runParallelChats(prompts, 1, 1, false, 500+trial*1000); err != nil {
		return 0, 0, 0, err
	}
	decode := <-decodeDone
	if decode.err != nil {
		return 0, decode.elapsed, 0, decode.err
	}
	if decode.result == nil {
		return 0, decode.elapsed, 0, fmt.Errorf("foreground lane returned no result")
	}
	tokens := decode.result.CompletionTokens
	if tokens <= 0 {
		tokens = estimateTokens(decode.result.Content)
	}
	return tokens, decode.elapsed, tokensPerSecond(tokens, decode.elapsed), nil
}

func agentBenchmarkLongPrompt(workloadID string, trial, lane int) string {
	return agentBenchmarkLongPromptSized(workloadID, trial, lane, agentBenchmarkPromptLen)
}

func (r *Runner) agentBenchmarkLongPrompt(trial, lane int) string {
	length := agentBenchmarkPromptLen
	if r != nil && r.AgentPromptBytes > 0 {
		length = r.AgentPromptBytes
	}
	workloadID := ""
	if r != nil {
		workloadID = r.WorkloadID
	}
	return agentBenchmarkLongPromptSized(workloadID, trial, lane, length)
}

func agentBenchmarkLongPromptSized(workloadID string, trial, lane, length int) string {
	if length < 1 {
		length = agentBenchmarkPromptLen
	}
	var b strings.Builder
	b.Grow(length + 256)
	fmt.Fprintf(&b, "Workload %s sample %d lane %d repository context. Read every record and then answer with one token.\n", workloadID, trial, lane)
	for record := 0; b.Len() < length; record++ {
		fmt.Fprintf(&b, "record_%04d_%04d: inspect module state, dependency edge, test result, and rollback condition.\n", lane, record)
	}
	return b.String()
}

func minCachedTokens(results []timedChatResult) int {
	minimum := 0
	for _, result := range results {
		if result.result == nil || result.result.CachedTokens <= 0 {
			return 0
		}
		if minimum == 0 || result.result.CachedTokens < minimum {
			minimum = result.result.CachedTokens
		}
	}
	return minimum
}

func sumNewPromptTokens(results []timedChatResult) int {
	total := 0
	for _, result := range results {
		if result.result == nil {
			continue
		}
		fresh := result.result.PromptTokens - result.result.CachedTokens
		if fresh > 0 {
			total += fresh
		}
	}
	return total
}

func aggregateAgentTrials(trials []*Result) *Result {
	if len(trials) == 0 {
		return nil
	}
	out := &Result{
		Model: trials[0].Model, Parallel: trials[0].Parallel,
		AgentSamples: len(trials), AgentCachedTokens: trials[0].AgentCachedTokens,
		AgentPromptBytes: trials[0].AgentPromptBytes,
		Timestamp:        time.Now().Unix(),
	}
	for _, trial := range trials {
		out.PromptTokens += trial.PromptTokens
		out.PromptTimeS += trial.PromptTimeS
		out.PromptTPS += trial.PromptTPS
		out.GenTokens += trial.GenTokens
		out.GenTimeS += trial.GenTimeS
		out.GenTPS += trial.GenTPS
		out.MixedGenTokens += trial.MixedGenTokens
		out.MixedTimeS += trial.MixedTimeS
		out.MixedGenTPS += trial.MixedGenTPS
		out.AgentTurnTimeS += trial.AgentTurnTimeS
		out.AgentScenarioTimeS += trial.AgentScenarioTimeS
		out.AgentNewPromptTokens += trial.AgentNewPromptTokens
		if trial.AgentTurnTimeS > out.AgentTurnMaxS {
			out.AgentTurnMaxS = trial.AgentTurnTimeS
		}
		if trial.AgentScenarioTimeS > out.AgentScenarioMaxS {
			out.AgentScenarioMaxS = trial.AgentScenarioTimeS
		}
		if trial.AgentCachedTokens < out.AgentCachedTokens {
			out.AgentCachedTokens = trial.AgentCachedTokens
		}
		if len(trial.GPUUtilization) > 0 {
			out.GPUUtilization = mergeGPUUtilization(out.GPUUtilization, trial.GPUUtilization)
		}
		if len(trial.PhaseUtilization) > 0 {
			out.PhaseUtilization = mergePhaseUtilization(out.PhaseUtilization, trial.PhaseUtilization)
		}
	}
	n := float64(len(trials))
	out.PromptTokens /= len(trials)
	out.PromptTimeS /= n
	out.PromptTPS /= n
	out.GenTokens /= len(trials)
	out.GenTimeS /= n
	out.GenTPS /= n
	out.MixedGenTokens /= len(trials)
	out.MixedTimeS /= n
	out.MixedGenTPS /= n
	out.AgentTurnTimeS /= n
	out.AgentScenarioTimeS /= n
	out.AgentNewPromptTokens /= len(trials)
	averagePhaseUtilization(out.PhaseUtilization, len(trials))
	return out
}

func mergeGPUUtilization(dst, src []GPUUtilization) []GPUUtilization {
	type aggregate struct{ sm, memory, pcieRX, pcieTX, observations int }
	byGPU := make(map[int]aggregate, len(dst)+len(src))
	add := func(sample GPUUtilization) {
		observations := sample.Observations
		if observations <= 0 {
			observations = 1
		}
		agg := byGPU[sample.GPU]
		agg.sm += sample.SMPercent * observations
		agg.memory += sample.MemPercent * observations
		agg.pcieRX += sample.PCIeRXMBps * observations
		agg.pcieTX += sample.PCIeTXMBps * observations
		agg.observations += observations
		byGPU[sample.GPU] = agg
	}
	for _, sample := range dst {
		add(sample)
	}
	for _, sample := range src {
		add(sample)
	}
	out := make([]GPUUtilization, 0, len(byGPU))
	for gpu, agg := range byGPU {
		if agg.observations <= 0 {
			continue
		}
		out = append(out, GPUUtilization{
			GPU: gpu, SMPercent: agg.sm / agg.observations,
			MemPercent:   agg.memory / agg.observations,
			PCIeRXMBps:   agg.pcieRX / agg.observations,
			PCIeTXMBps:   agg.pcieTX / agg.observations,
			Observations: agg.observations,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].GPU < out[j].GPU })
	return out
}

func sumPromptTokens(results []timedChatResult) int {
	total := 0
	for _, result := range results {
		if result.result != nil {
			total += result.result.PromptTokens
		}
	}
	return total
}

func sumCompletionTokens(results []timedChatResult) int {
	total := 0
	for _, result := range results {
		if result.result == nil {
			continue
		}
		tokens := result.result.CompletionTokens
		if tokens <= 0 {
			tokens = estimateTokens(result.result.Content)
		}
		total += tokens
	}
	return total
}

func tokensPerSecond(tokens int, elapsed time.Duration) float64 {
	if tokens <= 0 || elapsed <= 0 {
		return 0
	}
	return float64(tokens) / elapsed.Seconds()
}
