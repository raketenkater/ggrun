package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Automatic context sizing is capacity-driven: autoContextCap starts from the
// model's native trained context and the fit loop spends VRAM until something
// stops it. Nothing in that path asks how much context the workload actually
// uses.
//
// Measured 2026-09-03 over 11692 real agent requests across 55 sessions on this
// deployment:
//
//	p50   1002 tokens      p90  32484      p95  55040
//	p99  78958 tokens      max 143718
//	requests above 256K: 0        above 128K: 2 (0.02%)
//
// The server was serving --ctx-size 942080. The p99 request used 8.4% of it.
// That unused context is not free: at 942080 the backend reported 12391 MiB of
// compute buffers and 5060 MiB of KV across three devices -- 17.4 GB, which is
// why the same launch ran --n-cpu-moe 43 with every expert layer on the host.
//
// Contract invariant 5 is "the objective is real agent work". This file supplies
// the missing term: a context ceiling measured from the agent traffic this
// deployment actually serves, so the automatic path stops buying context nobody
// asks for and the freed VRAM can hold experts instead.
//
// Scope note: demand is a property of the workload, not of the model, so it is
// stored once per cache directory rather than per model. Tokenizers differ
// slightly between models, so a ceiling learned under one tokenizer is an
// approximation under another -- which is one reason the ceiling is deliberately
// generous (see AgentContextCeiling).

// AgentContextDemand is the observed per-request context distribution.
type AgentContextDemand struct {
	// Samples is how many requests contributed. A ceiling derived from a
	// handful of requests is not evidence; see AgentContextCeiling.
	Samples int
	// MaxTokens is the largest single-request context ever observed. It never
	// decreases across recordings, for the same reason RecordCompanionVRAM
	// keeps its largest sample: a ceiling that shrinks toward a quiet period
	// truncates the next long conversation.
	MaxTokens int
	// P99Tokens and P50Tokens are diagnostics: they explain the ceiling in a
	// launch banner without being load-bearing.
	P99Tokens  int
	P50Tokens  int
	ObservedAt time.Time
}

// minAgentContextSamples is the point below which an observed distribution is
// treated as anecdote rather than evidence. It is not a memory reserve and not
// a safety margin -- it gates whether the automatic path is allowed to *reduce*
// context at all, and below it ggrun keeps today's capacity-driven behaviour.
// MinAgentContextSamples is exported so diagnostics can explain the gate.
const MinAgentContextSamples = 200

const minAgentContextSamples = MinAgentContextSamples

// agentContextGrowthFactor is an explicit policy choice, not a measurement, and
// is named here rather than buried so it can be argued with.
//
// The ceiling must not truncate a conversation that merely grows past the
// longest one yet seen. Doubling the observed maximum covers that with room to
// spare while still being multiples below a model's native context. Erring high
// is the safe direction: a too-large ceiling costs VRAM and is caught by the
// ordinary fit stage, whereas a too-small one silently drops agent context.
//
// This is not the static fudge reserve invariant 8 forbids: that rule governs
// memory-safety reserves, which must be measured. This is a workload ceiling
// whose conservative direction is upward.
const agentContextGrowthFactor = 2

// AgentContextCeiling returns the total context the automatic path should aim
// for given this demand and slot count, or 0 when the demand is not yet
// evidence and the caller should keep its existing behaviour.
//
// Total, not per-slot: llama.cpp divides --ctx-size across --parallel slots, so
// a multi-agent shape needs the per-agent ceiling once per slot. This is what
// makes the ceiling correct for multi-agent workflows rather than only for the
// single-slot case it was measured in.
func AgentContextCeiling(d AgentContextDemand, parallel int) int {
	if d.Samples < minAgentContextSamples || d.MaxTokens <= 0 {
		return 0
	}
	slots := parallel
	if slots <= 0 {
		slots = 1
	}
	perAgent := d.MaxTokens * agentContextGrowthFactor
	if perAgent < contextMinimum {
		perAgent = contextMinimum
	}
	total := perAgent * slots
	// Round up to the granule the rest of the context search works in, so the
	// ceiling never sits mid-granule and get rounded back under demand.
	if contextGranularity > 0 {
		total = (total + contextGranularity - 1) / contextGranularity * contextGranularity
	}
	return total
}

// Explain renders the ceiling's basis for a launch banner.
func (d AgentContextDemand) Explain(parallel int) string {
	ceiling := AgentContextCeiling(d, parallel)
	if ceiling == 0 {
		return fmt.Sprintf("agent context demand not yet evidence (%d/%d samples)",
			d.Samples, minAgentContextSamples)
	}
	slots := parallel
	if slots <= 0 {
		slots = 1
	}
	return fmt.Sprintf(
		"agent context demand over %d requests: p50 %d, p99 %d, max %d tokens; ceiling %d (%dx max x %d slot(s))",
		d.Samples, d.P50Tokens, d.P99Tokens, d.MaxTokens, ceiling, agentContextGrowthFactor, slots)
}

func agentContextDemandPath(cacheDir string) string {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	return filepath.Join(cacheDir, "agent_context_demand.workload")
}

// AgentContextDemandFromSamples builds a demand record from raw per-request
// context sizes. Exported so the observer in cmd/ggrun can stay a thin log
// reader and the statistics live next to the policy that consumes them.
func AgentContextDemandFromSamples(samples []int) AgentContextDemand {
	clean := make([]int, 0, len(samples))
	for _, v := range samples {
		if v > 0 {
			clean = append(clean, v)
		}
	}
	if len(clean) == 0 {
		return AgentContextDemand{}
	}
	sort.Ints(clean)
	at := func(p int) int {
		idx := len(clean) * p / 100
		if idx >= len(clean) {
			idx = len(clean) - 1
		}
		return clean[idx]
	}
	return AgentContextDemand{
		Samples:   len(clean),
		MaxTokens: clean[len(clean)-1],
		P99Tokens: at(99),
		P50Tokens: at(50),
	}
}

// RecordAgentContextDemand stores an observed distribution, never lowering the
// recorded maximum or sample count.
func RecordAgentContextDemand(cacheDir string, d AgentContextDemand) error {
	path := agentContextDemandPath(cacheDir)
	if path == "" || d.Samples <= 0 || d.MaxTokens <= 0 {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	release, err := acquirePlacementLock(path+".lock", 5*time.Second)
	if err != nil {
		return err
	}
	defer release()
	if prev := MeasuredAgentContextDemand(cacheDir); prev.MaxTokens > 0 {
		if prev.MaxTokens > d.MaxTokens {
			d.MaxTokens = prev.MaxTokens
		}
		if prev.Samples > d.Samples {
			d.Samples = prev.Samples
			if d.P99Tokens == 0 {
				d.P99Tokens = prev.P99Tokens
			}
			if d.P50Tokens == 0 {
				d.P50Tokens = prev.P50Tokens
			}
		}
	}
	body := fmt.Sprintf(
		"# Observed agent context demand (workload-scoped, not model-scoped)\n"+
			"AGENT_CTX_SAMPLES=%d\nAGENT_CTX_MAX=%d\nAGENT_CTX_P99=%d\nAGENT_CTX_P50=%d\nAGENT_CTX_OBSERVED_AT=%d\n",
		d.Samples, d.MaxTokens, d.P99Tokens, d.P50Tokens, time.Now().Unix())
	return atomicWriteFile(path, []byte(body), 0o644)
}

// MeasuredAgentContextDemand returns the stored distribution, or a zero value
// when this deployment has never been observed serving agent traffic.
func MeasuredAgentContextDemand(cacheDir string) AgentContextDemand {
	data, err := os.ReadFile(agentContextDemandPath(cacheDir))
	if err != nil {
		return AgentContextDemand{}
	}
	var d AgentContextDemand
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		key, raw, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		v, convErr := strconv.Atoi(raw)
		if convErr != nil || v < 0 {
			continue
		}
		switch key {
		case "AGENT_CTX_SAMPLES":
			d.Samples = v
		case "AGENT_CTX_MAX":
			d.MaxTokens = v
		case "AGENT_CTX_P99":
			d.P99Tokens = v
		case "AGENT_CTX_P50":
			d.P50Tokens = v
		case "AGENT_CTX_OBSERVED_AT":
			d.ObservedAt = time.Unix(int64(v), 0)
		}
	}
	return d
}
