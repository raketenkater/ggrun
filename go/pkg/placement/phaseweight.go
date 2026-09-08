package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// Phase weight: how an agent workload actually spends its time.
//
// ggrun's objective is real agent work (invariant 5), which means prefill and
// decode both matter and neither may be optimised away for the other. Deciding
// how much each matters is not a judgement call, it is measurable -- and it has
// to be measured rather than reasoned about, because reasoning about it here
// produced the wrong answer twice before the numbers arrived.
//
// Request-level counts cannot answer it. Across 14,247 recorded agent requests
// prefill outnumbers decode 6.9:1 in tokens, but llama-server keeps a prefix
// cache, so an agent re-sending a stable prefix does not pay to recompute it.
// Measured over a real 12-turn agent session the *processed* ratio was 3.34:1 --
// half the request-level figure. This therefore parses the backend's own timing
// rows: what it reports it actually did, not what was sent to it.
//
// The same session on GLM 5.3 Flash measured:
//
//	prefill  10,321 tokens in 311.7 s = 33.11 tok/s   42.3% of processed time
//	decode    3,086 tokens in 425.9 s =  7.25 tok/s   57.7%
//
// Close to even, leaning decode. An estimate taken from the first turn alone
// said 80% prefill and was wrong twice over: that turn carried a 24:1 token
// ratio, and the prefill rate falls steeply as context grows (52.7 tok/s at
// 2906 tokens, 35.2 at 6096, then 10-15 for the rest of the session). Neither
// one turn nor request-level tokens is a workload model.
//
// The weight is per (rig, model, workload) and must stay measured. A rig whose
// experts are GPU-resident lands somewhere completely different, which is
// exactly why this cannot be a constant.

var phaseTimingPattern = regexp.MustCompile(
	`(prompt eval time|eval time)\s*=\s*([0-9.]+) ms /\s*([0-9]+) tokens`)

// AgentPhaseTiming is processed work split by phase, as the backend reported it.
type AgentPhaseTiming struct {
	PrefillMS     float64
	PrefillTokens int64
	DecodeMS      float64
	DecodeTokens  int64
	Samples       int
}

// Measured reports whether both phases were observed. A workload that only
// prefilled, or only decoded, cannot weigh one against the other.
func (t AgentPhaseTiming) Measured() bool {
	return t.PrefillMS > 0 && t.DecodeMS > 0 && t.Samples > 0
}

// PrefillTimeShare is the fraction of processed time spent in prefill, which is
// the weight a batch-size decision trades against. Zero when unmeasured.
func (t AgentPhaseTiming) PrefillTimeShare() float64 {
	if !t.Measured() {
		return 0
	}
	return t.PrefillMS / (t.PrefillMS + t.DecodeMS)
}

// PrefillTPS and DecodeTPS are the rates the backend actually achieved.
func (t AgentPhaseTiming) PrefillTPS() float64 { return tokensPerSecond(t.PrefillTokens, t.PrefillMS) }
func (t AgentPhaseTiming) DecodeTPS() float64  { return tokensPerSecond(t.DecodeTokens, t.DecodeMS) }

func tokensPerSecond(tokens int64, ms float64) float64 {
	if tokens <= 0 || ms <= 0 {
		return 0
	}
	return float64(tokens) / (ms / 1000)
}

// Summary renders the split for a launch record.
func (t AgentPhaseTiming) Summary() string {
	if !t.Measured() {
		return "agent phase mix unmeasured"
	}
	return fmt.Sprintf("prefill %.0f%% of processed time (%.2f tok/s over %d tokens), decode %.0f%% (%.2f tok/s over %d tokens), %d samples",
		t.PrefillTimeShare()*100, t.PrefillTPS(), t.PrefillTokens,
		(1-t.PrefillTimeShare())*100, t.DecodeTPS(), t.DecodeTokens, t.Samples)
}

// ParseAgentPhaseTimings reads the backend's per-request timing rows.
//
// Rows reporting a single token are discarded: llama-server emits a degenerate
// `eval time = 0.00 ms / 1 tokens` for a request that produced nothing, and
// counting those as decode samples drags the rate toward zero. An earlier
// hand-rolled version of this parser also swept `prompt eval time` rows into
// the decode set, which is what produced a spurious "cache degrades with use"
// reading on 2026-09-08; the phase is taken from the row label, never from
// position.
func ParseAgentPhaseTimings(logData string) AgentPhaseTiming {
	out := AgentPhaseTiming{}
	for _, m := range phaseTimingPattern.FindAllStringSubmatch(logData, -1) {
		ms, err1 := strconv.ParseFloat(m[2], 64)
		tokens, err2 := strconv.ParseInt(m[3], 10, 64)
		if err1 != nil || err2 != nil || ms <= 0 || tokens <= 1 {
			continue
		}
		if m[1] == "prompt eval time" {
			out.PrefillMS += ms
			out.PrefillTokens += tokens
		} else {
			out.DecodeMS += ms
			out.DecodeTokens += tokens
		}
		out.Samples++
	}
	return out
}

func phaseWeightPath(cacheDir, modelPath string) string {
	base := filepath.Base(strings.TrimSpace(modelPath))
	if base == "" || base == "." || base == string(filepath.Separator) {
		base = "model"
	}
	return filepath.Join(cacheDir, "agent-phase-"+base+".weight")
}

// RecordAgentPhaseTiming accumulates observed work for this model, so the
// weight strengthens across sessions instead of being re-derived from whatever
// one launch happened to serve. Only complete two-phase observations are kept.
func RecordAgentPhaseTiming(cacheDir, modelPath string, t AgentPhaseTiming) error {
	if cacheDir == "" || !t.Measured() {
		return nil
	}
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return err
	}
	if prev, ok := loadAgentPhaseTiming(cacheDir, modelPath); ok {
		t.PrefillMS += prev.PrefillMS
		t.PrefillTokens += prev.PrefillTokens
		t.DecodeMS += prev.DecodeMS
		t.DecodeTokens += prev.DecodeTokens
		t.Samples += prev.Samples
	}
	line := fmt.Sprintf("%.3f|%d|%.3f|%d|%d\n",
		t.PrefillMS, t.PrefillTokens, t.DecodeMS, t.DecodeTokens, t.Samples)
	return os.WriteFile(phaseWeightPath(cacheDir, modelPath), []byte(line), 0o644)
}

func loadAgentPhaseTiming(cacheDir, modelPath string) (AgentPhaseTiming, bool) {
	data, err := os.ReadFile(phaseWeightPath(cacheDir, modelPath))
	if err != nil {
		return AgentPhaseTiming{}, false
	}
	parts := strings.Split(strings.TrimSpace(string(data)), "|")
	if len(parts) != 5 {
		return AgentPhaseTiming{}, false
	}
	pMS, err1 := strconv.ParseFloat(parts[0], 64)
	pTok, err2 := strconv.ParseInt(parts[1], 10, 64)
	dMS, err3 := strconv.ParseFloat(parts[2], 64)
	dTok, err4 := strconv.ParseInt(parts[3], 10, 64)
	samples, err5 := strconv.Atoi(parts[4])
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil || err5 != nil {
		return AgentPhaseTiming{}, false
	}
	return AgentPhaseTiming{
		PrefillMS: pMS, PrefillTokens: pTok,
		DecodeMS: dMS, DecodeTokens: dTok, Samples: samples,
	}, true
}

// minAgentPhaseSamples is the point at which the split stops being one
// request's accident. It is a sample count, not a tuning constant: it says how
// much evidence is enough, never what the answer should be.
const minAgentPhaseSamples = 8

// MeasuredAgentPhaseTiming returns the accumulated phase mix for this model,
// or an unmeasured value when too little has been observed to weigh anything.
func MeasuredAgentPhaseTiming(cacheDir, modelPath string) AgentPhaseTiming {
	t, ok := loadAgentPhaseTiming(cacheDir, modelPath)
	if !ok || t.Samples < minAgentPhaseSamples {
		return AgentPhaseTiming{}
	}
	return t
}

// AgentTurnCostMS estimates one agent turn's processed time under a candidate's
// measured rates, weighted by the observed token mix.
//
// This is the comparison a batch-size decision needs: -b buys prefill and
// spends VRAM that would otherwise hold resident expert layers, which buys
// decode. Scoring either phase alone picks the wrong point, in whichever
// direction the measured mix happens to lean -- 42/58 prefill/decode here, but
// the ranking must hold for a rig that lands at 80/20 either way.
//
// Returns zero when either rate is unknown: an unscored candidate must not
// outrank a scored one (invariant 4).
func AgentTurnCostMS(mix AgentPhaseTiming, prefillTPS, decodeTPS float64) float64 {
	if !mix.Measured() || prefillTPS <= 0 || decodeTPS <= 0 {
		return 0
	}
	prefillTokens := float64(mix.PrefillTokens) / float64(mix.Samples)
	decodeTokens := float64(mix.DecodeTokens) / float64(mix.Samples)
	return (prefillTokens/prefillTPS + decodeTokens/decodeTPS) * 1000
}
