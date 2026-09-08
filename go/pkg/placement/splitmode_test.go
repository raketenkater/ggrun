package placement

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func splitModeRig() []detect.GPU {
	return []detect.GPU{
		{Index: 0, VRAMTotalMB: 12282},
		{Index: 1, VRAMTotalMB: 24564},
		{Index: 2, VRAMTotalMB: 12288},
	}
}

func fakeFitBin(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "llama-fit-params")
	if err := os.WriteFile(p, []byte("#!/bin/true\n"), 0o755); err != nil {
		t.Fatalf("write fake oracle: %v", err)
	}
	return p
}

func withProber(t *testing.T, fn func(context.Context, string, string, string, []string) (bool, string)) {
	t.Helper()
	prev := probeSplitMode
	probeSplitMode = fn
	t.Cleanup(func() { probeSplitMode = prev })
}

// TestSplitModeRecordsBackendRefusal reproduces the 2026-09-08 GLM finding: both
// alternatives abort at load, and each abort carries a distinct backend reason
// that the launch record must preserve rather than restate as our own guess.
func TestSplitModeRecordsBackendRefusal(t *testing.T) {
	cacheDir := t.TempDir()
	withProber(t, func(_ context.Context, _, _, mode string, _ []string) (bool, string) {
		switch mode {
		case "row":
			return false, "ggml_backend_split_buffer_type: device CUDA0 does not support split buffers"
		case "tensor":
			return false, "LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'glm5next'"
		}
		return true, ""
	})

	report := DetectSplitModeSupport(cacheDir, fakeFitBin(t), "m.gguf", "glm5next",
		splitModeRig(), nil, time.Second)

	got := map[string]SplitModeSupport{}
	for _, m := range report.Modes {
		got[m.Mode] = m
	}
	if !got["layer"].Eligible() {
		t.Error("layer must stay eligible; every backend implements it")
	}
	for _, mode := range []string{"row", "tensor"} {
		m := got[mode]
		if m.Eligible() {
			t.Errorf("%s must not be eligible after the backend refused it", mode)
		}
		if !m.Probed {
			t.Errorf("%s was answered by the backend, so it is probed", mode)
		}
		if m.Reason == "" {
			t.Errorf("%s must carry the backend's own reason", mode)
		}
	}
	if want := "does not support split buffers"; !strings.Contains(got["row"].Reason, want) {
		t.Errorf("row reason %q should carry %q", got["row"].Reason, want)
	}
	if modes := report.EligibleModes(); len(modes) != 1 || modes[0] != "layer" {
		t.Errorf("eligible modes: got %v, want [layer]", modes)
	}
}

// TestSplitModeSupportedModeBecomesEligible is the generalisation guard. The
// tensor denylist ends in `default: return true`, so an architecture outside it
// on the same cards must be offered the mode. Encoding this rig's answer as a
// universal default would deny those users a mode their model supports.
func TestSplitModeSupportedModeBecomesEligible(t *testing.T) {
	cacheDir := t.TempDir()
	withProber(t, func(_ context.Context, _, _, mode string, _ []string) (bool, string) {
		if mode == "tensor" {
			return true, ""
		}
		return false, "device CUDA0 does not support split buffers"
	})

	report := DetectSplitModeSupport(cacheDir, fakeFitBin(t), "m.gguf", "llama",
		splitModeRig(), nil, time.Second)

	modes := report.EligibleModes()
	if len(modes) != 2 || modes[0] != "layer" || modes[1] != "tensor" {
		t.Fatalf("eligible modes: got %v, want [layer tensor]", modes)
	}
}

// TestSplitModeTimeoutStaysUnknown separates "we could not ask" from "your
// hardware cannot do this". Only the second may persist as a refusal.
func TestSplitModeTimeoutStaysUnknown(t *testing.T) {
	cacheDir := t.TempDir()
	withProber(t, func(ctx context.Context, _, _, mode string, _ []string) (bool, string) {
		if mode == "row" {
			<-ctx.Done()
			return false, ""
		}
		return false, "not implemented for architecture 'x'"
	})

	report := DetectSplitModeSupport(cacheDir, fakeFitBin(t), "m.gguf", "x",
		splitModeRig(), nil, 30*time.Millisecond)

	for _, m := range report.Modes {
		if m.Mode != "row" {
			continue
		}
		if m.Probed {
			t.Error("a timed-out probe must not count as an answer")
		}
		if m.Supported {
			t.Error("a timed-out probe must not report support")
		}
	}
	// The unknown must not be persisted as a refusal for later launches.
	cached, _ := LoadSplitModeSupport(cacheDir, fakeFitBin(t), "x", splitModeRig())
	for _, m := range cached.Modes {
		if m.Mode == "row" {
			t.Error("an unprobed mode must not be written to the capability cache")
		}
	}
}

// TestSplitModeCacheAvoidsReprobe keeps launch latency off the ordinary path
// (invariant 10): one answer per backend/arch/topology, not one per launch.
func TestSplitModeCacheAvoidsReprobe(t *testing.T) {
	cacheDir := t.TempDir()
	bin := fakeFitBin(t)
	calls := 0
	withProber(t, func(_ context.Context, _, _, mode string, _ []string) (bool, string) {
		calls++
		return false, "refused"
	})

	DetectSplitModeSupport(cacheDir, bin, "m.gguf", "glm5next", splitModeRig(), nil, time.Second)
	first := calls
	DetectSplitModeSupport(cacheDir, bin, "m.gguf", "glm5next", splitModeRig(), nil, time.Second)

	if calls != first {
		t.Errorf("second call re-probed the backend: %d calls, want %d", calls, first)
	}
	if first == 0 {
		t.Fatal("first call probed nothing")
	}
}

// TestSplitModeKeyChangesWithBackendBuild guards invariant 9: a rebuilt backend
// must not inherit the previous build's capability claims.
func TestSplitModeKeyChangesWithBackendBuild(t *testing.T) {
	bin := fakeFitBin(t)
	before := SplitModeProbeKey(bin, "glm5next", splitModeRig())

	if err := os.WriteFile(bin, []byte("#!/bin/true\n# rebuilt with more bytes\n"), 0o755); err != nil {
		t.Fatalf("rewrite oracle: %v", err)
	}
	if after := SplitModeProbeKey(bin, "glm5next", splitModeRig()); after == before {
		t.Error("a rebuilt backend must produce a different capability key")
	}
	if other := SplitModeProbeKey(bin, "llama", splitModeRig()); other == SplitModeProbeKey(bin, "glm5next", splitModeRig()) {
		t.Error("a different architecture must produce a different capability key")
	}
	single := []detect.GPU{{Index: 0, VRAMTotalMB: 12282}}
	if other := SplitModeProbeKey(bin, "glm5next", single); other == SplitModeProbeKey(bin, "glm5next", splitModeRig()) {
		t.Error("a different topology must produce a different capability key")
	}
}

// TestSplitModeSingleGPUProbesNothing: splitting across one device is
// meaningless, so it must not cost a launch two oracle runs to discover.
func TestSplitModeSingleGPUProbesNothing(t *testing.T) {
	calls := 0
	withProber(t, func(_ context.Context, _, _, _ string, _ []string) (bool, string) {
		calls++
		return true, ""
	})

	report := DetectSplitModeSupport(t.TempDir(), fakeFitBin(t), "m.gguf", "llama",
		[]detect.GPU{{Index: 0, VRAMTotalMB: 12282}}, nil, time.Second)

	if calls != 0 {
		t.Errorf("probed %d time(s) on a single-GPU rig; want 0", calls)
	}
	if modes := report.EligibleModes(); len(modes) != 1 || modes[0] != "layer" {
		t.Errorf("eligible modes: got %v, want [layer]", modes)
	}
}

// TestFirstBackendErrorPrefersDiagnosis: the backend prints its diagnosis and
// then a generic failure. The diagnosis is the half a reader needs.
func TestFirstBackendErrorPrefersDiagnosis(t *testing.T) {
	out := `llama_model_load: error loading model: device CUDA0 does not support split buffers
llama_model_load_from_file: failed to load model
common_init_from_params: failed to load model 'x.gguf'`
	got := firstBackendError(out)
	if !strings.Contains(got, "does not support split buffers") {
		t.Errorf("got %q, want the split-buffer diagnosis", got)
	}

	// With no diagnosis available, the generic line is better than nothing.
	if got := firstBackendError("llama_model_load_from_file: failed to load model"); got == "" {
		t.Error("a generic failure line must still be reported")
	}
	if got := firstBackendError("everything is fine"); got != "" {
		t.Errorf("no failure line should yield no reason; got %q", got)
	}
}
