package placement

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// serveLogTwoRounds mirrors a real ggrun launch log: a first planning round's
// buffer rows followed by the round that actually served. Only the last round
// describes the running process.
const serveLogTwoRounds = `
load_tensors:        CUDA0 model buffer size =  9999.00 MiB
llama_kv_cache:      CUDA0 KV buffer size =  9999.00 MiB
sched_reserve:      CUDA0 compute buffer size =  9999.00 MiB
load_tensors:        CUDA0 model buffer size =  1500.00 MiB
load_tensors:        CUDA1 model buffer size =  6000.00 MiB
llama_kv_cache:      CUDA0 KV buffer size =   500.00 MiB
llama_kv_cache:      CUDA1 KV buffer size =  2000.00 MiB
sched_reserve:      CUDA0 compute buffer size =  1000.00 MiB
sched_reserve:      CUDA1 compute buffer size =  1200.00 MiB
`

// TestParseServeLedgersTakesTheServingRound guards the accounting against a
// re-planned launch: summing every round would inflate the accounted total and
// hide real growth, and taking the first round would invent growth that is not
// there.
func TestParseServeLedgersTakesTheServingRound(t *testing.T) {
	got := parseServeGrowthLedgers(serveLogTwoRounds)
	if got[0].ModelMB != 1500 || got[0].KVMB != 500 || got[0].ComputeMB != 1000 {
		t.Fatalf("CUDA0 must reflect the last round, got %+v", got[0])
	}
	if want := 3000; got[0].total() != want {
		t.Fatalf("CUDA0 accounted total = %d, want %d", got[0].total(), want)
	}
	if got[1].total() != 9200 {
		t.Fatalf("CUDA1 accounted total = %d, want 9200", got[1].total())
	}
}

// TestRecordGrowthFromServeLearnsWithoutAnOOM is the regression this file
// exists for. Before it, every production path to this evidence went through
// RecordRuntimeGraphGrowthFromOOM, so a rig that never crashed never learned
// its own graph growth and could never pack tighter.
func TestRecordGrowthFromServeLearnsWithoutAnOOM(t *testing.T) {
	dir := t.TempDir()
	logData := serveLogTwoRounds
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 12282}, {Index: 1, VRAMTotalMB: 24564}}
	model := &ModelProfile{Path: "m.gguf", NumLayers: 45}
	strategy := &Strategy{ContextSize: 287744, UBatchSize: 256, KVQuality: "high", KVPlacement: "gpu", Parallel: 1}

	// CUDA0 holds 4000 MiB live against 3000 accounted -> 1000 of growth.
	// CUDA1 holds 9500 live against 9200 accounted -> 300 of growth.
	restore := serveVRAMSampler
	serveVRAMSampler = func(idx int) int {
		switch idx {
		case 0:
			return 4000
		case 1:
			return 9500
		}
		return 0
	}
	defer func() { serveVRAMSampler = restore }()

	got := RecordRuntimeGraphGrowthFromServe(dir, model, strategy, "llama", gpus, logData, nil)
	if got[0] != 1000 {
		t.Fatalf("CUDA0 growth = %d, want 1000", got[0])
	}
	if got[1] != 300 {
		t.Fatalf("CUDA1 growth = %d, want 300", got[1])
	}
	// It must be readable back as EXACT evidence, or RelatedModelRuntimeGraphGrowth
	// refuses to carry it and the loop still cannot close.
	back := RuntimeGraphGrowthByGPU(dir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, "llama", gpus, strategy.Parallel)
	if back[0] != 1000 || back[1] != 300 {
		t.Fatalf("growth did not persist as exact evidence, got %v", back)
	}
	related := RelatedModelRuntimeGraphGrowth(dir, model, gpus, strategy.Parallel, "llama")
	if related[0] != 1000 {
		t.Fatalf("measured growth must be transferable, got %v", related)
	}
}

// TestRecordGrowthFromServeRefusesUnattributableDeltas keeps the recorder from
// manufacturing evidence. A device the backend never accounted for would have
// its entire footprint charged to growth, and a non-positive delta recorded as
// zero would tell a packer it may fill the device completely.
func TestRecordGrowthFromServeRefusesUnattributableDeltas(t *testing.T) {
	dir := t.TempDir()
	logData := serveLogTwoRounds
	// CUDA2 appears nowhere in the log.
	gpus := []detect.GPU{
		{Index: 0, VRAMTotalMB: 12282},
		{Index: 2, VRAMTotalMB: 12288},
	}
	model := &ModelProfile{Path: "m.gguf", NumLayers: 45}
	strategy := &Strategy{ContextSize: 287744, UBatchSize: 256, KVQuality: "high", KVPlacement: "gpu", Parallel: 1}

	restore := serveVRAMSampler
	serveVRAMSampler = func(idx int) int {
		switch idx {
		case 0:
			return 3000 // exactly accounted -> zero delta, must not be recorded
		case 2:
			return 8000 // unaccounted device entirely
		}
		return 0
	}
	defer func() { serveVRAMSampler = restore }()

	got := RecordRuntimeGraphGrowthFromServe(dir, model, strategy, "llama", gpus, logData, nil)
	if _, ok := got[2]; ok {
		t.Fatalf("a device with no backend accounting must be skipped, got %v", got)
	}
	if _, ok := got[0]; ok {
		t.Fatalf("a zero delta must not be recorded as measured growth, got %v", got)
	}
}

// TestRecordGrowthFromServeNetsOutCompanions keeps a GPU-seated reviewer from
// being charged to the main model's graph growth.
func TestRecordGrowthFromServeNetsOutCompanions(t *testing.T) {
	dir := t.TempDir()
	logData := serveLogTwoRounds
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 12282}}
	model := &ModelProfile{Path: "m.gguf", NumLayers: 45}
	strategy := &Strategy{ContextSize: 287744, UBatchSize: 256, KVQuality: "high", KVPlacement: "gpu", Parallel: 1}

	restore := serveVRAMSampler
	serveVRAMSampler = func(int) int { return 6000 } // 3000 accounted + 2000 companion + 1000 growth
	defer func() { serveVRAMSampler = restore }()

	got := RecordRuntimeGraphGrowthFromServe(dir, model, strategy, "llama", gpus, logData,
		map[int]int{0: 2000})
	if got[0] != 1000 {
		t.Fatalf("companion VRAM must not be charged to graph growth: got %d, want 1000", got[0])
	}
}
