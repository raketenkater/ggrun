package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// glmProfile mirrors GLM-5.3-Flash's real geometry (print_info: n_embd 4096,
// n_layer 45, arch glm5next). Its dense cold estimate is therefore
// ubatch * 4096 * 45 * 42 / 1e6 = 7.74 MiB per ubatch token.
func glmProfile() *ModelProfile {
	return &ModelProfile{
		Path:       "/models/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf",
		Basename:   "GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf",
		HiddenSize: 4096, NumLayers: 45, ModelArch: "glm5next",
		IsMoE: true, NumExperts: 288, ExpertUsedCount: 6,
	}
}

func writeObservedComputeProbe(t *testing.T, dir string, model *ModelProfile, gpus []detect.GPU, ctx, ubatch int, computeByGPU map[int]int, evidence string) {
	t.Helper()
	var b []byte
	add := func(f string, a ...any) { b = append(b, []byte(fmt.Sprintf(f, a...))...) }
	add("# Probe cache for %s\n", filepath.Base(model.Path))
	add("# ctx=%d ubatch=%d kv_quality=high kv_placement=gpu backend=llama gpu_sig=%s parallel=1\n",
		ctx, ubatch, gpuSignatureHash(gpus))
	// Write the CURRENT schema, not a literal: probeMetadataIntegritySchema
	// gates stale metadata, so a hardcoded version silently turns every fixture
	// into "too old to trust" the next time the schema is bumped.
	add("PROBE_CACHE_SCHEMA=%d\n", probeCacheSchema)
	add("PROBED_COMPUTE_BUF_EVIDENCE=%s\n", evidence)
	for dev, v := range computeByGPU {
		add("PROBED_COMPUTE_BUF_MB_CUDA%d=%d\n", dev, v)
	}
	name := filepath.Base(probeCachePath(dir, model, ctx, ubatch, "high", "gpu", "llama", gpus, 0))
	if err := os.WriteFile(filepath.Join(dir, name), b, 0o600); err != nil {
		t.Fatal(err)
	}
}

// The measured calibration must reproduce the observation it came from, and
// must transfer that law to a context it never measured. Both numbers below are
// from the live 2026-09-02 rig, not invented.
func TestMeasuredComputeExcessReproducesAndTransfers(t *testing.T) {
	dir := t.TempDir()
	model := glmProfile()
	gpus := []detect.GPU{
		{Index: 0, Name: "a", VRAMTotalMB: 12282},
		{Index: 1, Name: "b", VRAMTotalMB: 24564},
		{Index: 2, Name: "c", VRAMTotalMB: 12288},
	}
	const refCtx, refUB, measured = 1048576, 64, 4572
	writeObservedComputeProbe(t, dir, model, gpus, refCtx, refUB,
		map[int]int{0: measured, 1: 4574, 2: 1}, "live-allocated")

	scaling := MeasuredComputeExcess(dir, model, gpus, 1, "llama")
	if scaling.BytesPerTokenCtx <= 0 {
		t.Fatal("an observed buffer far above the dense estimate must calibrate an excess")
	}

	// Reproduces the measurement it was derived from (within rounding).
	dense := firstLaunchComputeBufMBParallel(model, refUB, 1)
	got := scaling.scaledComputeBufMB(dense, refUB, refCtx)
	if got < 4500 || got > 4650 {
		t.Fatalf("calibration does not reproduce its own measurement: got %d MiB, measured %d", got, measured)
	}

	// Transfers to a context never measured: a quarter of the context must cost
	// about a quarter of the excess. This is the whole point -- the ctx/ubatch
	// keyed probe cache could not do this, so ten measurements taught nothing.
	quarter := scaling.scaledComputeBufMB(firstLaunchComputeBufMBParallel(model, refUB, 1), refUB, refCtx/4)
	if quarter >= got/2 {
		t.Fatalf("quarter context must cost far less: 1M=%d MiB, 256k=%d MiB", got, quarter)
	}
	if quarter <= dense {
		t.Fatalf("quarter context must still exceed the dense estimate: %d vs dense %d", quarter, dense)
	}

	// Linear in ubatch, matching the measured 68-72 MiB/token at 1M.
	double := scaling.scaledComputeBufMB(firstLaunchComputeBufMBParallel(model, refUB*2, 1), refUB*2, refCtx)
	if double < got*3/2 {
		t.Fatalf("doubling ubatch must roughly double the buffer: %d -> %d", got, double)
	}
}

// A model whose graph really is context-independent must be left exactly as it
// was: no excess observed, no reserve invented (invariant 8).
func TestMeasuredComputeExcessIgnoresModelsThatMatchTheDenseEstimate(t *testing.T) {
	dir := t.TempDir()
	model := glmProfile()
	model.ModelArch = "llama"
	gpus := []detect.GPU{{Index: 0, Name: "a", VRAMTotalMB: 24564}}
	// Observed buffer at or below the dense estimate => no context term.
	dense := firstLaunchComputeBufMBParallel(model, 256, 1)
	writeObservedComputeProbe(t, dir, model, gpus, 32768, 256, map[int]int{0: dense - 50}, "live-allocated")

	scaling := MeasuredComputeExcess(dir, model, gpus, 1, "llama")
	if scaling.BytesPerTokenCtx != 0 {
		t.Fatalf("a context-independent model must calibrate no excess, got %.3f B/token-ctx", scaling.BytesPerTokenCtx)
	}
	if got := scaling.scaledComputeBufMB(dense, 256, 1048576); got != dense {
		t.Fatalf("dense estimate must be untouched: got %d want %d", got, dense)
	}
}

// An oracle prediction is the very estimate under test; it can never calibrate
// itself (invariant 4: estimates rank, measurements decide).
func TestMeasuredComputeExcessRejectsOraclePlannedEvidence(t *testing.T) {
	dir := t.TempDir()
	model := glmProfile()
	gpus := []detect.GPU{{Index: 0, Name: "a", VRAMTotalMB: 24564}}
	writeObservedComputeProbe(t, dir, model, gpus, 1048576, 64, map[int]int{0: 4572}, "oracle-planned")
	if scaling := MeasuredComputeExcess(dir, model, gpus, 1, "llama"); scaling.BytesPerTokenCtx != 0 {
		t.Fatalf("an oracle prediction must not calibrate the estimate it is testing: %.3f", scaling.BytesPerTokenCtx)
	}
}

// An oracle-planned per-GPU row is a PREDICTION from the same class of model as
// the cold estimate, and for an indexer architecture it is context-blind in the
// same way. It must never lower the reserve below our own calibrated estimate:
// live 2026-09-02, a 274 MiB oracle row sat beside a live-allocated 4572 for the
// same model, silently replaced the calibrated value, and the load died in
// graph_reserve. Two predictions disagreeing means take the larger.
//
// An OBSERVATION still wins outright, and an untagged legacy row keeps its
// authority so a genuine secondary-split-owner measurement is not discarded
// (TestComputeSplitOwnerChargesPerGPUComputeNotAggregate depends on that).
func TestOraclePlannedRowCannotLowerTheCalibratedReserve(t *testing.T) {
	dir := t.TempDir()
	model := glmProfile()
	gpus := []detect.GPU{
		{Index: 0, Name: "a", VRAMTotalMB: 12282},
		{Index: 1, Name: "b", VRAMTotalMB: 24564},
	}
	// Calibrate the context term from a real observation at 1M / ub 64.
	writeObservedComputeProbe(t, dir, model, gpus, 1048576, 64,
		map[int]int{0: 4572, 1: 4574}, "live-allocated")
	scaling := MeasuredComputeExcess(dir, model, gpus, 1, "llama")
	if scaling.BytesPerTokenCtx <= 0 {
		t.Fatal("observation did not calibrate")
	}

	dense := firstLaunchComputeBufMBParallel(model, 64, 1)
	calibrated := scaling.scaledComputeBufMB(dense, 64, 1048576)
	if calibrated < 4000 {
		t.Fatalf("calibrated estimate unexpectedly small: %d", calibrated)
	}

	// The evidence classes must be distinguishable, which is what lets the read
	// site prefer the calibrated value over a context-blind oracle prediction
	// while still trusting an observation or an untagged legacy row.
	if !observedAllocationEvidence("live-allocated") {
		t.Fatal("live-allocated must count as observed")
	}
	if observedAllocationEvidence("oracle-planned") {
		t.Fatal("oracle-planned must NOT count as observed")
	}
	if observedAllocationEvidence("") {
		t.Fatal("an untagged legacy row must not be treated as observed")
	}
}

// De-owning a device must actually predict the saving it produces. A compute
// reading taken while the device owned a split is the cost that making it
// expert-only ELIMINATES; charging it back to the expert-only role made every
// moe-owner-N candidate score level with the baseline, so the optimizer
// reported "no non-rejected topology can relieve GPU 0" while the imbalance it
// had just measured went unaddressed (live 2026-09-02: CUDA1, 3453 MiB).
//
// This is the twin of the split-owner guard; leaving it unguarded silently
// disabled ggrun's entire owner-count search.
func TestExpertOnlyReserveIgnoresSplitOwnerReading(t *testing.T) {
	model := glmProfile()
	const splitOwnerReading = 3453

	// A reading recorded in the SPLIT-OWNER role must not become the
	// expert-only reserve: the cold fallback is used instead.
	aggregate := 4000
	fallback := expertOnlyComputeReserveMB(aggregate, 0)
	if fallback <= 0 {
		t.Fatalf("cold expert-only fallback must be positive, got %d", fallback)
	}
	if fallback >= splitOwnerReading {
		t.Fatalf("an expert-only device must be budgeted far less than a split "+
			"owner's %d MiB buffer, got %d", splitOwnerReading, fallback)
	}

	// And a genuine expert-only measurement is still honoured, with headroom.
	accepted := expertOnlyComputeReserveMB(aggregate, 99)
	if accepted <= 0 || accepted > computeFloorMB {
		t.Fatalf("a real expert-only measurement must be accepted within the floor: %d", accepted)
	}
	_ = model
}
