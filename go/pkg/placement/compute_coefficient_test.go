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

func glmCoefficientScope() computeCoefficientScope {
	return computeCoefficientScope{
		BackendTag: "llama@glm-build", GPUSignature: "4070sig", KVQuality: "q8_0", KVPlacement: "gpu",
	}
}

func writeComputeProbe(t *testing.T, dir, name string, ctx, ubatch int, bufs []int) {
	t.Helper()
	writeScopedComputeProbe(t, dir, name, glmCoefficientScope(), ctx, ubatch, bufs)
}

func writeScopedComputeProbe(t *testing.T, dir, name string, scope computeCoefficientScope, ctx, ubatch int, bufs []int) {
	t.Helper()
	body := fmt.Sprintf("# Probe cache for %s\n# ctx=%d ubatch=%d kv_quality=%s kv_placement=%s backend=%s gpu_sig=%s parallel=1\n",
		filepath.Base(glmProfile().Path), ctx, ubatch, scope.KVQuality, scope.KVPlacement, scope.BackendTag, scope.GPUSignature)
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

	got := loadMeasuredComputeCoefficient(dir, glmProfile(), glmCoefficientScope())
	if got < 350 || got > 380 {
		t.Fatalf("coefficient %.1f outside the measured range 363-370", got)
	}
}

// Without evidence nothing may change: a model with no probes must plan exactly
// as it does today.
func TestNoProbesLeavesTheBuiltInCoefficient(t *testing.T) {
	dir := t.TempDir()
	if got := loadMeasuredComputeCoefficient(dir, glmProfile(), glmCoefficientScope()); got != 0 {
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
	if got := loadMeasuredComputeCoefficient(dir, glmProfile(), glmCoefficientScope()); got != 0 {
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
	if got := loadMeasuredComputeCoefficient(dir, other, glmCoefficientScope()); got != 0 {
		t.Fatalf("coefficient %.1f derived from another model's probes", got)
	}
}

// A probe measured on another backend, another card, another KV type, or with
// full SWA (the feature lives in the backend tag) must not move this plan's
// coefficient. Two scopes must not be pooled into one median either.
func TestCoefficientDoesNotCrossBackendDeviceKVOrSWA(t *testing.T) {
	dir := t.TempDir()
	writeComputeProbe(t, dir, "a.probe", 529408, 128, []int{4501, 4630, 4624})
	writeComputeProbe(t, dir, "b.probe", 786432, 256, []int{6669, 6798, 6667})
	home := glmCoefficientScope()
	if got := loadMeasuredComputeCoefficient(dir, glmProfile(), home); got < 350 || got > 380 {
		t.Fatalf("matching scope coefficient %.1f", got)
	}

	otherBackend := home
	otherBackend.BackendTag = "ik_llama@other"
	if got := loadMeasuredComputeCoefficient(dir, glmProfile(), otherBackend); got != 0 {
		t.Fatalf("other backend inherited coefficient %.1f", got)
	}
	otherDevice := home
	otherDevice.GPUSignature = "3090sig"
	if got := loadMeasuredComputeCoefficient(dir, glmProfile(), otherDevice); got != 0 {
		t.Fatalf("other device inherited coefficient %.1f", got)
	}
	otherKV := home
	otherKV.KVQuality = "f16"
	if got := loadMeasuredComputeCoefficient(dir, glmProfile(), otherKV); got != 0 {
		t.Fatalf("other KV quality inherited coefficient %.1f", got)
	}
	swa := home
	swa.BackendTag = home.BackendTag + "|swa-full=true"
	if got := loadMeasuredComputeCoefficient(dir, glmProfile(), swa); got != 0 {
		t.Fatalf("full SWA inherited the windowed coefficient %.1f", got)
	}

	// A second scope's buffers must not be mixed into the matching median.
	foreign := home
	foreign.KVQuality = "f16"
	writeScopedComputeProbe(t, dir, "foreign-a.probe", foreign, 529408, 128, []int{100, 100, 100})
	writeScopedComputeProbe(t, dir, "foreign-b.probe", foreign, 786432, 256, []int{100, 100, 100})
	if got := loadMeasuredComputeCoefficient(dir, glmProfile(), home); got < 350 || got > 380 {
		t.Fatalf("foreign scope moved the coefficient to %.1f", got)
	}
	if got := loadMeasuredComputeCoefficient(dir, glmProfile(), foreign); got == 0 || got > 50 {
		t.Fatalf("foreign scope did not keep its own small coefficient: %.1f", got)
	}

	cpu := home
	cpu.KVPlacement = "cpu"
	writeScopedComputeProbe(t, dir, "cpu-a.probe", cpu, 529408, 128, []int{9000, 9000, 9000})
	writeScopedComputeProbe(t, dir, "cpu-b.probe", cpu, 786432, 256, []int{9000, 9000, 9000})
	auto := home
	auto.KVPlacement = "auto"
	if got := loadMeasuredComputeCoefficient(dir, glmProfile(), auto); got != 0 {
		t.Fatalf("auto placement pooled gpu and cpu probes into %.1f", got)
	}
	unscoped := filepath.Join(dir, "bare.probe")
	body := fmt.Sprintf("# Probe cache for %s\n# ctx=529408 ubatch=128\nPROBED_COMPUTE_BUF_MB_CUDA0=4501\n", filepath.Base(glmProfile().Path))
	if err := os.WriteFile(unscoped, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := loadMeasuredComputeCoefficient(dir, glmProfile(), home); got < 350 || got > 380 {
		t.Fatalf("a probe with no provenance changed the coefficient to %.1f", got)
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

	got := loadMeasuredComputeCoefficient(dir, glmProfile(), glmCoefficientScope())
	if got < 350 || got > 380 {
		t.Fatalf("fragment device skewed the coefficient to %.1f", got)
	}
}
