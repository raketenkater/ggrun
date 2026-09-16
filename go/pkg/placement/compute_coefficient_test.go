package placement

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// GLM-5.3-Flash as read from its GGUF: glm5next, 4096 hidden, 46 layers, 288
// experts with 8 used. It is an MoE that falls through to the dense coefficient.
func glmProfile() *ModelProfile {
	return &ModelProfile{
		Path: "/models/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf", ModelArch: "glm5next",
		HiddenSize: 4096, NumLayers: 46, ExpertUsedCount: 8,
	}
}

func writeComputeProbe(t *testing.T, dir, name string, ctx, ubatch int, bufs []int) {
	t.Helper()
	body := fmt.Sprintf("# Probe cache for %s\n# ctx=%d ubatch=%d kv_quality=q8_0 kv_placement=gpu\n",
		filepath.Base(glmProfile().Path), ctx, ubatch)
	for i, mb := range bufs {
		body += fmt.Sprintf("PROBED_COMPUTE_BUF_MB_CUDA%d=%d\n", i, mb)
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// The measured values from this machine on 2026-09-15. Two independent probes
// imply 369.6 and 363.2 bytes; the derivation must land between them.
func TestCoefficientDerivedFromMeasuredProbes(t *testing.T) {
	dir := t.TempDir()
	writeComputeProbe(t, dir, "a.probe", 529408, 128, []int{4501, 4630, 4624})
	writeComputeProbe(t, dir, "b.probe", 786432, 256, []int{6669, 6798, 6667})

	got := loadMeasuredComputeCoefficient(dir, glmProfile())
	if got < 350 || got > 380 {
		t.Fatalf("coefficient %.1f outside the measured range 363-370", got)
	}
}

// Without evidence nothing may change: a model with no probes must plan exactly
// as it does today.
func TestNoProbesLeavesTheBuiltInCoefficient(t *testing.T) {
	dir := t.TempDir()
	if got := loadMeasuredComputeCoefficient(dir, glmProfile()); got != 0 {
		t.Fatalf("empty cache produced coefficient %.1f, want 0", got)
	}
	model := glmProfile()
	withoutEvidence := firstLaunchComputeBufMBParallel(model, 256, 1)
	model.MeasuredComputeCoefficient = 0
	if again := firstLaunchComputeBufMBParallel(model, 256, 1); again != withoutEvidence {
		t.Fatalf("zero coefficient changed the estimate: %d vs %d", again, withoutEvidence)
	}
}

// A single probe is indistinguishable from a recording artefact.
func TestSingleProbeIsNotEnough(t *testing.T) {
	dir := t.TempDir()
	writeComputeProbe(t, dir, "only.probe", 529408, 128, []int{4501, 4630, 4624})
	if got := loadMeasuredComputeCoefficient(dir, glmProfile()); got != 0 {
		t.Fatalf("one probe produced coefficient %.1f, want 0", got)
	}
}

// Another model's probes must not be borrowed.
func TestOtherModelsProbesAreIgnored(t *testing.T) {
	dir := t.TempDir()
	writeComputeProbe(t, dir, "a.probe", 529408, 128, []int{4501, 4630, 4624})
	writeComputeProbe(t, dir, "b.probe", 786432, 256, []int{6669, 6798, 6667})

	other := glmProfile()
	other.Path = "/models/Some-Other-Model.gguf"
	if got := loadMeasuredComputeCoefficient(dir, other); got != 0 {
		t.Fatalf("coefficient %.1f derived from another model's probes", got)
	}
}

// The end-to-end point: with evidence, the estimate must match what the backend
// actually allocated. The dense coefficient predicts 2,026 MiB where 13,140 was
// measured.
func TestMeasuredCoefficientPredictsTheRealBuffer(t *testing.T) {
	model := glmProfile()
	dense := firstLaunchComputeBufMBForGPUParallelAtContext(model, 256, 1, 786432, 0, nil)
	if dense > 4000 {
		t.Fatalf("dense baseline unexpectedly high (%d MiB); test assumption broken", dense)
	}

	model.MeasuredComputeCoefficient = 365
	withEvidence := firstLaunchComputeBufMBForGPUParallelAtContext(model, 256, 1, 786432, 0, nil)

	const measured = 13140
	if err := math.Abs(float64(withEvidence-measured)) / measured; err > 0.10 {
		t.Fatalf("estimate %d MiB vs measured %d MiB: %.1f%% error", withEvidence, measured, err*100)
	}
	if withEvidence <= dense {
		t.Fatalf("measured evidence did not raise the estimate: %d vs dense %d", withEvidence, dense)
	}
}

// A measured coefficient is normalised to the reference context, so it has to
// scale down with the window. Charging it flat would bill every launch at 1M.
func TestMeasuredCoefficientScalesWithContext(t *testing.T) {
	model := glmProfile()
	model.MeasuredComputeCoefficient = 365
	small := firstLaunchComputeBufMBForGPUParallelAtContext(model, 128, 1, 262144, 0, nil)
	large := firstLaunchComputeBufMBForGPUParallelAtContext(model, 128, 1, 786432, 0, nil)
	if small >= large {
		t.Fatalf("estimate did not grow with context: %d at 262144 vs %d at 786432", small, large)
	}
}

// A device holding only a fragment must not drag the coefficient down for the
// devices carrying the real graph.
func TestFragmentDeviceDoesNotDepressTheCoefficient(t *testing.T) {
	dir := t.TempDir()
	writeComputeProbe(t, dir, "a.probe", 529408, 128, []int{4501, 4630, 97})
	writeComputeProbe(t, dir, "b.probe", 786432, 256, []int{6669, 6798, 155})

	got := loadMeasuredComputeCoefficient(dir, glmProfile())
	if got < 350 || got > 380 {
		t.Fatalf("fragment device skewed the coefficient to %.1f", got)
	}
}
