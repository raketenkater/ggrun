package placement

import (
	"math"
	"os"
	"testing"
)

// A single backend process emits both prompt and decode timing rows. Progress
// rows are cumulative snapshots of one prompt, not independent samples.
const servedLogSample = `
slot print_timing: prompt processing, n_tokens = 6144, progress = 0.06, t = 202.36 s / 30.36
slot print_timing: prompt processing, n_tokens = 12288, progress = 0.12, t = 388.14 s / 31.66
slot print_timing: prompt processing, n_tokens = 16384, progress = 0.16, t = 488.72 s / 33.52
250.22.300.586 I slot print_timing: id  0 | task 28414 | prompt eval time = 20065.69 ms / 403 tokens ( 49.79 ms per token, 20.08 tokens per second)
250.22.300.589 I slot print_timing: id  0 | task 28414 |        eval time = 4745.27 ms / 31 tokens ( 158.18 ms per token, 6.32 tokens per second)
prompt eval time = 121459.56 ms / 4232 tokens ( 28.70 ms per token, 34.84 tokens per second)
eval time = 12830.28 ms / 96 tokens ( 135.06 ms per token, 7.40 tokens per second)
eval time = 12843.02 ms / 96 tokens ( 135.19 ms per token, 7.40 tokens per second)
eval time = 121459.56 ms / 4232 tokens ( 28.70 ms per token, 34.84 tokens per second)
eval time = 482.86 ms / 4 tokens ( 120.71 ms per token, 8.28 tokens per second)
`

func TestParseServedThroughputKeepsPhasesSeparate(t *testing.T) {
	p := ParseServedThroughput(servedLogSample)
	if p.DecodeSamples != 4 || p.DecodeTPS != 7.4 {
		t.Fatalf("all four eligible decode rows must count, regardless of speed: %+v", p)
	}
	if p.PrefillSamples != 2 || math.Abs(p.PrefillTPS-27.46) > 0.001 {
		t.Fatalf("only completed prompt rows count, with an even-sample median: %+v", p)
	}
}

func TestParseServedThroughputDoesNotInventModelIdentity(t *testing.T) {
	log := "slot print_timing: eval time = 1000 ms / 100 tokens ( 10 ms per token, 100 tokens per second)"
	p := ParseServedThroughput(log)
	if p.DecodeSamples != 1 || p.DecodeTPS != 100 {
		t.Fatalf("a fast model must retain its own decode evidence: %+v", p)
	}
}

func TestParseServedThroughputMissingPhaseStaysUnknown(t *testing.T) {
	log := "slot print_timing: prompt eval time = 1000 ms / 30 tokens ( 33.33 ms per token, 30 tokens per second)"
	p := ParseServedThroughput(log)
	if p.DecodeSamples != 0 || p.PrefillSamples != 1 || p.PrefillTPS != 30 {
		t.Fatalf("prompt timing must never become decode evidence: %+v", p)
	}
	progress := "slot print_timing: prompt processing, n_tokens = 100, t = 5.0 s / 20.0"
	if p := ParseServedThroughput(progress); p.Measured() {
		t.Fatalf("an unfinished prompt is not a completed timing sample: %+v", p)
	}
}

func TestObservedPerformanceRejectsLegacyPhaseMixing(t *testing.T) {
	dir := t.TempDir()
	path := observedPerfPath(dir, "m.gguf")
	if err := os.WriteFile(path, []byte("PERF_DECODE_TPS=12\nPERF_DECODE_SAMPLES=5\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if got := MeasuredObservedPerformance(dir, "/models/m.gguf"); got.Measured() {
		t.Fatalf("unversioned rates used ambiguous phase attribution: %+v", got)
	}
}

// TestParseServedThroughputDropsShortGenerations keeps first-token latency out
// of a steady-state number: a 4-token completion's per-token average is
// dominated by prefill hand-off and describes nothing about generation.
func TestParseServedThroughputDropsShortGenerations(t *testing.T) {
	short := `eval time = 482.86 ms / 4 tokens ( 120.71 ms per token, 8.28 tokens per second)`
	if p := ParseServedThroughput(short); p.DecodeSamples != 0 {
		t.Fatalf("a 4-token completion must not count as a decode sample, got %d", p.DecodeSamples)
	}
}

// TestObservedPerformanceRoundTrips pins the store, including the fields that
// say which configuration produced the numbers -- without them the display line
// cannot be attributed to anything.
func TestObservedPerformanceRoundTrips(t *testing.T) {
	dir := t.TempDir()
	in := ParseServedThroughput(servedLogSample)
	in.ContextSize, in.UBatch, in.ExpertLayersOnGPU = 287744, 256, 5
	if err := RecordObservedPerformance(dir, "/models/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf", in); err != nil {
		t.Fatalf("record: %v", err)
	}
	// Read back via the path a launch actually names. CalibrationModelBasename
	// canonicalises only the first shard, which is the one every launch points
	// at, so that is the identity this store inherits.
	got := MeasuredObservedPerformance(dir, "/models/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf")
	if !got.Measured() {
		t.Fatal("expected a stored measurement for the same model")
	}
	if math.Abs(got.DecodeTPS-in.DecodeTPS) > 0.01 || math.Abs(got.PrefillTPS-in.PrefillTPS) > 0.01 {
		t.Fatalf("rates did not round-trip: got %+v want %+v", got, in)
	}
	if got.ContextSize != 287744 || got.UBatch != 256 || got.ExpertLayersOnGPU != 5 {
		t.Fatalf("configuration identity lost: %+v", got)
	}
}

// TestObservedPerformanceKeepsTheLatestNotTheBest is the deliberate difference
// from every measured-memory record here. A reserve keeps its largest sample
// because it must cover the worst case; a performance line must describe the
// run just done, including when a change made things slower. Keeping the best
// would turn it into a high-water mark that hides regressions.
func TestObservedPerformanceKeepsTheLatestNotTheBest(t *testing.T) {
	dir := t.TempDir()
	model := "/models/m.gguf"
	fast := ObservedPerformance{DecodeTPS: 12, DecodeSamples: 5, PrefillTPS: 40, PrefillSamples: 5}
	slow := ObservedPerformance{DecodeTPS: 6, DecodeSamples: 5, PrefillTPS: 20, PrefillSamples: 5}
	if err := RecordObservedPerformance(dir, model, fast); err != nil {
		t.Fatalf("record fast: %v", err)
	}
	if err := RecordObservedPerformance(dir, model, slow); err != nil {
		t.Fatalf("record slow: %v", err)
	}
	got := MeasuredObservedPerformance(dir, model)
	if math.Abs(got.DecodeTPS-6) > 0.01 {
		t.Fatalf("a slower run must replace a faster one: got %.2f, want 6", got.DecodeTPS)
	}
}

// TestUnmeasuredPerformanceRendersNothing keeps the TUI from printing an empty
// or zeroed line on a model that has never been served.
func TestUnmeasuredPerformanceRendersNothing(t *testing.T) {
	var zero ObservedPerformance
	if zero.Measured() {
		t.Fatal("a zero record must not report itself as measured")
	}
	if zero.Summary() != "" {
		t.Fatalf("a zero record must render nothing, got %q", zero.Summary())
	}
	if got := MeasuredObservedPerformance(t.TempDir(), "/models/never-served.gguf"); got.Measured() {
		t.Fatal("an unserved model must have no measurement")
	}
}
