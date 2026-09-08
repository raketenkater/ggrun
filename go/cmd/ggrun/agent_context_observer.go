package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
)

// The router already writes one JSON line per agent request with the usage
// block the upstream API returned. That is a complete record of how much
// context this deployment's agent traffic actually asks for, and until now
// nothing read it back: automatic context sizing started from the model's
// trained maximum and let the fit stage spend VRAM until something stopped it.
//
// This observer is deliberately a thin reader. The statistics and the policy
// live in placement (workload_context.go) next to the code that consumes them;
// everything here does is turn log lines into per-request context sizes.

// agentContextObserverMaxFiles bounds the scan so a long-lived deployment does
// not re-read years of logs on every launch. Files are taken newest-first, and
// the recorded maximum never shrinks, so an old peak stays honoured even once
// its log has aged out of this window.
const agentContextObserverMaxFiles = 200

// agentContextObserverMaxSamples bounds memory for the same reason.
const agentContextObserverMaxSamples = 200000

// observeAgentContextDemand reads the router's per-request metrics logs and
// returns the observed context distribution.
//
// A request's context is input + cache-read + cache-creation tokens: the cached
// prefix occupies KV exactly like freshly-sent tokens, so counting only
// input_tokens would under-measure a cache-backed agent turn by orders of
// magnitude -- which is the common case in agent work.
func observeAgentContextDemand(logDir string) (placement.AgentContextDemand, error) {
	if logDir == "" {
		return placement.AgentContextDemand{}, nil
	}
	paths, err := filepath.Glob(filepath.Join(logDir, "ggrun-claude-requests-*.jsonl"))
	if err != nil {
		return placement.AgentContextDemand{}, err
	}
	if len(paths) == 0 {
		return placement.AgentContextDemand{}, nil
	}
	type entry struct {
		path string
		mod  int64
	}
	entries := make([]entry, 0, len(paths))
	for _, p := range paths {
		fi, statErr := os.Stat(p)
		if statErr != nil {
			continue
		}
		entries = append(entries, entry{path: p, mod: fi.ModTime().Unix()})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].mod > entries[j].mod })
	if len(entries) > agentContextObserverMaxFiles {
		entries = entries[:agentContextObserverMaxFiles]
	}

	samples := make([]int, 0, 4096)
	for _, e := range entries {
		if len(samples) >= agentContextObserverMaxSamples {
			break
		}
		vals, readErr := readAgentContextSamples(e.path)
		if readErr != nil {
			continue // a truncated or in-flight log is not a launch failure
		}
		samples = append(samples, vals...)
	}
	return placement.AgentContextDemandFromSamples(samples), nil
}

// requestUsageLine is the subset of a router metrics line this needs.
type requestUsageLine struct {
	Usage struct {
		InputTokens         int `json:"input_tokens"`
		CacheReadTokens     int `json:"cache_read_input_tokens"`
		CacheCreationTokens int `json:"cache_creation_input_tokens"`
	} `json:"usage"`
}

func readAgentContextSamples(path string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []int
	sc := bufio.NewScanner(f)
	// Agent requests carry large prompts; the default 64 KiB token is not
	// enough for some router lines.
	sc.Buffer(make([]byte, 0, 1<<20), 8<<20)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var rec requestUsageLine
		if err := json.Unmarshal(line, &rec); err != nil {
			continue
		}
		ctx := rec.Usage.InputTokens + rec.Usage.CacheReadTokens + rec.Usage.CacheCreationTokens
		if ctx > 0 {
			out = append(out, ctx)
		}
	}
	// A scanner error mid-file still yields the samples read so far; a partially
	// read log is better evidence than none, and the caller never lowers a
	// previously recorded maximum from it.
	return out, sc.Err()
}

var agentContextObserveOnce sync.Once

// ensureAgentContextDemand refreshes the recorded demand at most once per
// process, before the first automatic context decision consumes it.
//
// Once per process rather than per call: the options builder this hangs off is
// invoked on every re-plan, and re-scanning the router logs each time would put
// a directory walk inside the placement retry loop for a figure that cannot
// change mid-launch.
func ensureAgentContextDemand(cacheDir string, parallel int) {
	agentContextObserveOnce.Do(func() {
		logDir := ""
		if cfg, err := config.Load(); err == nil && cfg != nil {
			logDir = cfg.LogDir
		}
		if logDir == "" {
			// Matches claudeRouterMetricsPath's own fallback, so the observer
			// looks where the router actually wrote.
			logDir = os.TempDir()
		}
		refreshAgentContextDemand(logDir, cacheDir, parallel)
	})
}

// refreshAgentContextDemand observes the router logs and records the result, so
// the next automatic context decision is sized from this deployment's own agent
// traffic. Best-effort throughout: a launch must never fail because the
// workload history could not be read.
func refreshAgentContextDemand(logDir, cacheDir string, parallel int) {
	demand, err := observeAgentContextDemand(logDir)
	if err != nil || demand.Samples <= 0 {
		return
	}
	if err := placement.RecordAgentContextDemand(cacheDir, demand); err != nil {
		fmt.Fprintf(os.Stderr, "[launch] warning: could not persist agent context demand: %v\n", err)
		return
	}
	stored := placement.MeasuredAgentContextDemand(cacheDir)
	fmt.Fprintf(os.Stderr, "[launch] %s\n", stored.Explain(parallel))
}

// recordServedThroughput stores what the just-finished serving run measured, so
// the next TUI screen can show it beside the model and an operator can compare
// configurations without re-running a benchmark.
//
// Best-effort and display-only: this figure never reaches placement, so a
// missing or stale value costs a banner line, not a memory decision.
func recordServedThroughput(cfg *config.Config, model *placement.ModelProfile,
	strategy *placement.Strategy, p *server.Process,
) {
	if cfg == nil || model == nil || strategy == nil || p == nil || p.LogBuf == nil {
		return
	}
	perf := placement.ParseServedThroughput(p.LogBuf.String())
	if !perf.Measured() {
		return
	}
	perf.ContextSize = strategy.ContextSize
	perf.UBatch = strategy.UBatchSize
	if moe := model.NumLayers - strategy.NCPUMoE; moe > 0 {
		perf.ExpertLayersOnGPU = moe
	}
	if err := placement.RecordObservedPerformance(cfg.CacheDir, model.Path, perf); err != nil {
		fmt.Fprintf(os.Stderr, "[launch] warning: could not persist observed throughput: %v\n", err)
		return
	}
	fmt.Fprintf(os.Stderr, "[launch] this run measured %s (ctx %d, ubatch %d, %d expert layer(s) on GPU)\n",
		perf.Summary(), perf.ContextSize, perf.UBatch, perf.ExpertLayersOnGPU)

	recordServedPhaseMix(cfg, model, p)
	recordServedDisplacementArm(cfg, model, strategy, perf)
}

// recordServedPhaseMix stores how this run's traffic actually split between
// prefill and decode.
//
// Recording only: the weight is read by candidate scoring, never by admission,
// so a missing or thin observation costs ranking precision rather than memory
// safety. It accumulates across launches because one session is not a workload
// model -- a single agent turn on 2026-09-08 read 80% prefill where the full
// twelve-turn session measured 42%.
func recordServedPhaseMix(cfg *config.Config, model *placement.ModelProfile, p *server.Process) {
	if cfg == nil || model == nil || p == nil || p.LogBuf == nil {
		return
	}
	mix := placement.ParseAgentPhaseTimings(p.LogBuf.String())
	if !mix.Measured() {
		return
	}
	if err := placement.RecordAgentPhaseTiming(cfg.CacheDir, model.Path, mix); err != nil {
		fmt.Fprintf(os.Stderr, "[launch] warning: could not persist agent phase mix: %v\n", err)
		return
	}
	if stored := placement.MeasuredAgentPhaseTiming(cfg.CacheDir, model.Path); stored.Measured() {
		fmt.Fprintf(os.Stderr, "[launch] agent phase mix: %s\n", stored.Summary())
	}
}

// recordServedDisplacementArm stores this run's decode rate as one arm of the
// hot-expert displacement comparison.
//
// Without this the gate cannot ever open: auto declines an unproven
// displacement, nothing records a proof, and auto declines forever. The two
// arms are collected across launches of the same placement -- a cache-free
// serve records the baseline, a later cache-on serve records the cache -- which
// is the same bootstrap shape ErrHotExpertBaselineUnmeasured already uses.
//
// The identity is always the packed cache-free placement's, so both arms key to
// the same row: a cache-on strategy carries that baseline, and a cache-free one
// is it.
func recordServedDisplacementArm(cfg *config.Config, model *placement.ModelProfile,
	strategy *placement.Strategy, perf placement.ObservedPerformance,
) {
	if cfg == nil || model == nil || strategy == nil || perf.DecodeTPS <= 0 {
		return
	}
	cacheOn := strategy.HotExpertCacheSlots > 0
	baseline := strategy
	demoted := 0
	if cacheOn {
		restored := placement.RestorePackedCacheFreeBaseline(strategy)
		if restored == nil {
			// A cache-on run whose pre-demotion topology was not captured cannot
			// be attributed to a baseline, and guessing one would compare two
			// different placements.
			return
		}
		baseline = restored
		// Demotion moves routed-expert layers to the host, so the difference in
		// NCPUMoE is what the cache cost in residency.
		if d := strategy.NCPUMoE - restored.NCPUMoE; d > 0 {
			demoted = d
		}
	}
	identity := placement.AllocationPlacementIdentity(baseline, model)
	if identity == "" {
		return
	}
	if err := placement.RecordHotExpertDisplacementArm(cfg.CacheDir, model.Path, identity,
		demoted, strategy.HotExpertCacheSlots, cacheOn, perf.DecodeTPS); err != nil {
		fmt.Fprintf(os.Stderr, "[launch] warning: could not persist hot-expert displacement arm: %v\n", err)
		return
	}
	proof, ok := placement.LoadHotExpertDisplacementProof(cfg.CacheDir, model.Path, identity)
	if !ok {
		return
	}
	if proof.Measured() {
		fmt.Fprintf(os.Stderr, "[launch] hot-expert displacement now measured on this placement: %s\n",
			proof.Summary())
		return
	}
	arm := "cache-free baseline"
	if cacheOn {
		arm = "cache-on"
	}
	fmt.Fprintf(os.Stderr, "[launch] recorded the %s arm (%.2f tok/s decode); the other arm completes the comparison\n",
		arm, perf.DecodeTPS)
}
