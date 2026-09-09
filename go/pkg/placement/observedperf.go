package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Throughput is recorded only by the calibration path: DefaultTPS and
// WinnerTPS on a CalibrationDecision are written from the A/B screen and
// nowhere else. A configuration that simply serves -- the normal case, and the
// only one that runs the user's real workload -- produces a full history of
// prompt-processing and eval timings in the backend log and ggrun keeps none of
// it. VerifiedConfig, the record actually reused on the next launch, carries no
// throughput at all.
//
// That is the same shape as the runtime-growth gap: a quantity learnable only
// from a special path, so ordinary successful operation teaches the system
// nothing. It leaves an operator unable to answer "what did this config
// actually do?" without re-running a benchmark, and leaves the TUI with nothing
// to show next to the model it is about to launch.
//
// This file records what a serving run measured, from the backend's own rows.
// It is a report, never an input to placement: nothing here is admission
// evidence, and a wrong or stale figure costs a misleading display line, not a
// bad memory decision.

// ObservedPerformance is what one serving run measured.
type ObservedPerformance struct {
	// DecodeTPS and PrefillTPS are medians, not maxima: a single fast turn on a
	// short cached prompt is not what the next run will feel.
	DecodeTPS  float64
	PrefillTPS float64
	// DecodeSamples and PrefillSamples say how much the medians rest on.
	DecodeSamples  int
	PrefillSamples int
	// ContextSize, UBatch and ExpertLayersOnGPU identify the configuration the
	// numbers belong to, so a display can say what produced them.
	ContextSize       int
	UBatch            int
	ExpertLayersOnGPU int
	ObservedAt        time.Time
}

// Measured reports whether this record carries anything worth showing.
func (p ObservedPerformance) Measured() bool {
	return p.DecodeSamples > 0 || p.PrefillSamples > 0
}

// Summary renders one line for a TUI or a status command.
func (p ObservedPerformance) Summary() string {
	if !p.Measured() {
		return ""
	}
	parts := make([]string, 0, 2)
	if p.DecodeSamples > 0 {
		parts = append(parts, fmt.Sprintf("decode %.1f tok/s", p.DecodeTPS))
	}
	if p.PrefillSamples > 0 {
		parts = append(parts, fmt.Sprintf("prefill %.1f tok/s", p.PrefillTPS))
	}
	return strings.Join(parts, ", ")
}

// The first version rejected here mixed prompt/decode timings and inferred a
// helper's identity from its speed. Those aggregates cannot be repaired.
const observedPerformanceSchema = 1

// Match the complete phase label in one pass: an unanchored decode-only
// regexp also matches the suffix of "prompt eval time". LogBuf belongs to one
// server.Process; model identity comes from that process, never its speed.
var servedTimingRe = regexp.MustCompile(`(?:^|[ \t])(?:(prompt) )?eval time = *[\d.]+ ms / *(\d+) tokens \( *[\d.]+ ms per token, *([\d.]+) tokens per second\)`)

// ParseServedThroughput extracts decode and prefill rates from a backend log.
//
// Decode samples below 20 generated tokens are dropped: at that length the
// per-token average is dominated by the first token's latency and does not
// describe steady-state generation.
func ParseServedThroughput(logData string) ObservedPerformance {
	var out ObservedPerformance
	var decode, prefill []float64
	for _, line := range strings.Split(logData, "\n") {
		m := servedTimingRe.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		n, err1 := strconv.Atoi(m[2])
		tps, err2 := strconv.ParseFloat(m[3], 64)
		if err1 != nil || err2 != nil || n <= 0 || tps <= 0 {
			continue
		}
		if m[1] == "prompt" {
			prefill = append(prefill, tps)
		} else if n >= 20 {
			decode = append(decode, tps)
		}
	}
	out.DecodeTPS, out.DecodeSamples = medianOf(decode)
	out.PrefillTPS, out.PrefillSamples = medianOf(prefill)
	return out
}

func medianOf(v []float64) (float64, int) {
	if len(v) == 0 {
		return 0, 0
	}
	sort.Float64s(v)
	mid := len(v) / 2
	if len(v)%2 == 0 {
		return v[mid-1]/2 + v[mid]/2, len(v)
	}
	return v[mid], len(v)
}

func observedPerfPath(cacheDir, modelBasename string) string {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	safe := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '.', r == '_':
			return r
		default:
			return '_'
		}
	}, modelBasename)
	return filepath.Join(cacheDir, "observed_perf_"+safe+".perf")
}

// RecordObservedPerformance stores what a serving run measured for this model.
//
// Latest-wins, unlike the measured-memory records which keep their largest
// sample. A reserve must cover the worst case ever seen; a performance display
// should describe the run the operator just did, including when a change made
// it slower. Keeping the best would turn the line into a high-water mark and
// hide exactly the regressions it should surface.
func RecordObservedPerformance(cacheDir, modelPath string, p ObservedPerformance) error {
	if !p.Measured() {
		return nil
	}
	path := observedPerfPath(cacheDir, CalibrationModelBasename(modelPath))
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	release, err := acquirePlacementLock(path+".lock", 5*time.Second)
	if err != nil {
		return err
	}
	defer release()
	body := fmt.Sprintf("# Observed serving throughput for %s\n"+
		"PERF_SCHEMA=%d\nPERF_DECODE_TPS=%.3f\nPERF_DECODE_SAMPLES=%d\n"+
		"PERF_PREFILL_TPS=%.3f\nPERF_PREFILL_SAMPLES=%d\n"+
		"PERF_CTX=%d\nPERF_UBATCH=%d\nPERF_GPU_EXPERT_LAYERS=%d\nPERF_AT=%d\n",
		CalibrationModelBasename(modelPath), observedPerformanceSchema,
		p.DecodeTPS, p.DecodeSamples, p.PrefillTPS, p.PrefillSamples,
		p.ContextSize, p.UBatch, p.ExpertLayersOnGPU, time.Now().Unix())
	return atomicWriteFile(path, []byte(body), 0o644)
}

// MeasuredObservedPerformance returns the last recorded serving throughput for
// this model, or a zero value when none exists.
func MeasuredObservedPerformance(cacheDir, modelPath string) ObservedPerformance {
	data, err := os.ReadFile(observedPerfPath(cacheDir, CalibrationModelBasename(modelPath)))
	if err != nil {
		return ObservedPerformance{}
	}
	var p ObservedPerformance
	schema := 0
	for _, line := range strings.Split(string(data), "\n") {
		key, raw, ok := strings.Cut(strings.TrimSpace(line), "=")
		if !ok {
			continue
		}
		switch key {
		case "PERF_SCHEMA":
			schema, _ = strconv.Atoi(raw)
		case "PERF_DECODE_TPS":
			p.DecodeTPS, _ = strconv.ParseFloat(raw, 64)
		case "PERF_DECODE_SAMPLES":
			p.DecodeSamples, _ = strconv.Atoi(raw)
		case "PERF_PREFILL_TPS":
			p.PrefillTPS, _ = strconv.ParseFloat(raw, 64)
		case "PERF_PREFILL_SAMPLES":
			p.PrefillSamples, _ = strconv.Atoi(raw)
		case "PERF_CTX":
			p.ContextSize, _ = strconv.Atoi(raw)
		case "PERF_UBATCH":
			p.UBatch, _ = strconv.Atoi(raw)
		case "PERF_GPU_EXPERT_LAYERS":
			p.ExpertLayersOnGPU, _ = strconv.Atoi(raw)
		case "PERF_AT":
			if v, err := strconv.ParseInt(raw, 10, 64); err == nil {
				p.ObservedAt = time.Unix(v, 0)
			}
		}
	}
	if schema != observedPerformanceSchema {
		return ObservedPerformance{}
	}
	return p
}
