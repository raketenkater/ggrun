package placement

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

type mimoGeometryFixture struct {
	Metadata struct {
		BlockCount int   `json:"mimo2.block_count"`
		NextN      int   `json:"mimo2.nextn_predict_layers"`
		HeadsKV    []int `json:"mimo2.attention.head_count_kv"`
		Pattern    []int `json:"mimo2.attention.sliding_window_pattern"`
		KeyLength  int   `json:"mimo2.attention.key_length"`
		ValueLen   int   `json:"mimo2.attention.value_length"`
		Window     int   `json:"mimo2.attention.sliding_window"`
	} `json:"metadata"`
	Assumptions struct {
		Context     int       `json:"context"`
		TensorSplit []float64 `json:"tensor_split"`
	} `json:"assumptions"`
	Cases []struct {
		UBatch   int     `json:"ubatch"`
		SWACells int     `json:"swa_cells"`
		TotalMiB float64 `json:"total_mib"`
		Devices  []struct {
			GPU          int   `json:"gpu"`
			GlobalBytes  int64 `json:"global_bytes"`
			SWABytes     int64 `json:"swa_bytes"`
			NormalLayers []int `json:"normal_layers"`
		} `json:"devices"`
	} `json:"cases"`
}

// loadMiMoFixture reads the geometry independently derived from the GGUF
// header and backend 6b790a9c2 (ubatch 64 matched to a live allocation). The
// model profile is built from its metadata only; nothing here is MiMo-specific.
func loadMiMoFixture(t *testing.T) (mimoGeometryFixture, *ModelProfile) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "mimo2-kv-geometry.json"))
	if err != nil {
		t.Fatalf("read verified fixture: %v", err)
	}
	var f mimoGeometryFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	m := f.Metadata
	model := &ModelProfile{
		ModelArch: "mimo2", NumLayers: m.BlockCount, NextNPredictLayers: m.NextN,
		HeadCountKVByLayer: m.HeadsKV, SWAPattern: m.Pattern,
		KeyLength: m.KeyLength, ValueLength: m.ValueLen, SlidingWindow: m.Window,
	}
	return f, model
}

func TestKVLayerGeometryMatchesVerifiedMixedHeadFixture(t *testing.T) {
	f, model := loadMiMoFixture(t)
	for _, c := range f.Cases {
		shape := kvCacheShape{Context: f.Assumptions.Context, UBatch: c.UBatch, Slots: 1, KVType: "q8_0"}
		layout, ok := modelKVLayerLayout(model)
		if !ok {
			t.Fatal("valid arrays rejected")
		}
		if _, swa, _ := layout.cells(shape); swa != int64(c.SWACells) {
			t.Fatalf("ubatch %d: SWA cells %d, want %d", c.UBatch, swa, c.SWACells)
		}
		layers, ok := kvLayerBytes(model, shape)
		if !ok {
			t.Fatal("per-layer bytes unavailable")
		}
		devices, outputDev := layerDeviceAssignments(f.Assumptions.TensorSplit, model.NumLayers)
		if outputDev != 2 {
			t.Fatalf("output slot owner %d; the backend distributes 52 slots", outputDev)
		}
		global := map[int]int64{}
		swa := map[int]int64{}
		normal := map[int][]int{}
		var total int64
		for il, b := range layers {
			if b == 0 {
				continue
			}
			total += b
			normal[devices[il]] = append(normal[devices[il]], il)
			if layout.Layers[il].SWA {
				swa[devices[il]] += b
			} else {
				global[devices[il]] += b
			}
		}
		if got := float64(total) / 1048576; got != c.TotalMiB {
			t.Fatalf("ubatch %d: total %.7f MiB, want %.7f", c.UBatch, got, c.TotalMiB)
		}
		for _, d := range c.Devices {
			if global[d.GPU] != d.GlobalBytes || swa[d.GPU] != d.SWABytes {
				t.Fatalf("ubatch %d gpu %d: full/swa %d/%d, want %d/%d",
					c.UBatch, d.GPU, global[d.GPU], swa[d.GPU], d.GlobalBytes, d.SWABytes)
			}
			if len(d.NormalLayers) > 0 && !reflect.DeepEqual(normal[d.GPU], d.NormalLayers) {
				t.Fatalf("gpu %d owns %v, want %v", d.GPU, normal[d.GPU], d.NormalLayers)
			}
			if len(d.NormalLayers) == 0 && len(normal[d.GPU]) != 0 {
				t.Fatalf("gpu %d charged KV for MTP/output-only slots: %v", d.GPU, normal[d.GPU])
			}
		}
	}
}

// Before per-layer geometry this model priced at zero KV and a uniform
// owned-layer share. The ledger must charge the owning devices, and the MTP
// blocks' device none.
func TestLedgerChargesKVToActualLayerOwners(t *testing.T) {
	f, model := loadMiMoFixture(t)
	model.Path = filepath.Join(t.TempDir(), "m.gguf")
	model.TotalSizeMB = 20000
	gpus := []detect.GPU{
		{Index: 0, VRAMTotalMB: 12282, VRAMUsedMB: 0, MemBandwidthMBps: 504000},
		{Index: 1, VRAMTotalMB: 24564, VRAMUsedMB: 0, MemBandwidthMBps: 1008000},
		{Index: 2, VRAMTotalMB: 12288, VRAMUsedMB: 0, MemBandwidthMBps: 360000},
	}
	s := &Strategy{Type: MultiGPUDense, TensorSplit: f.Assumptions.TensorSplit, SplitMode: "layer",
		ContextSize: f.Assumptions.Context, Parallel: 1, BatchSize: 2048, UBatchSize: 64,
		KVPlacement: "gpu", KVType: "q8_0", KVQuality: "q8_0"}
	shares := estimatedModelContextShares(model, s, gpus, computeKVTotalMBForStrategy(model, s))
	// 2816.953125+5.9765625 and 5633.90625+19.921875 MiB; apportioning a
	// whole-MiB total may round each share up by at most one more MiB.
	if shares[0] < 2823 || shares[0] > 2824 || shares[1] < 5654 || shares[1] > 5655 || shares[2] != 0 {
		t.Fatalf("KV owners %v, want ~[2823 5654 0]", shares)
	}
	// The split-fraction estimate this replaces put 6% of the cache on GPU 2.
	if old := estimatedContextShares(s, gpus, 8477); old[2] == 0 {
		t.Fatalf("fixture no longer exercises the owner fix: %v", old)
	}
	owned, _ := layerOwnership(s.TensorSplit, model.NumLayers)
	if got := kvLayerShareMB(model, kvShapeForStrategy(s, false), 8477, s.TensorSplit, owned, 2); got != 0 {
		t.Fatalf("MTP-only device charged %d MiB", got)
	}
	// Recovery sizing has no microbatch and prices the window at the backend
	// default 512: the fixture's ubatch-512 row puts 2834.88 MiB on GPU 0.
	if got := DeviceKVCacheMB(model, s.ContextSize, "q8_0", "gpu", false, s.TensorSplit, 0); got < 2835 || got > 2836 {
		t.Fatalf("recovery step credits GPU 0 with %d MiB", got)
	}
	if got := DeviceKVCacheMB(model, s.ContextSize, "q8_0", "gpu", false, s.TensorSplit, 2); got != 0 {
		t.Fatalf("recovery step credits the MTP-only GPU with %d MiB", got)
	}
}

// A model-wide measurement is the floor; only window growth that it could
// not have seen may raise it. The formula never lowers a measurement.
func TestStrategyKVRespectsMeasurementAndMicrobatch(t *testing.T) {
	f, model := loadMiMoFixture(t)
	ctx := f.Assumptions.Context
	s := &Strategy{ContextSize: ctx, UBatchSize: 64, Parallel: 1, KVType: "q8_0", KVPlacement: "gpu"}
	if got := computeKVTotalMBForStrategy(model, s); got != 8477 {
		t.Fatalf("cold ubatch 64 = %d, want 8477", got)
	}
	s.UBatchSize = 2048
	if got := computeKVTotalMBForStrategy(model, s); got != 8684 {
		t.Fatalf("cold ubatch 2048 = %d, want 8684", got)
	}
	// A rate recorded from the ubatch-64 launch.
	model.MeasuredKVBytesPerTok = map[string]float64{"q8_0": 8476.7578125 * 1048576 / float64(ctx)}
	if got := computeKVTotalMBForStrategy(model, s); got != 8684 {
		t.Fatalf("ubatch 2048 hid its window growth behind the ubatch-64 rate: %d", got)
	}
	s.UBatchSize = 64
	model.MeasuredKVBytesPerTok["q8_0"] *= 1.1
	if got := computeKVTotalMBForStrategy(model, s); got != 9324 {
		t.Fatalf("formula overrode a larger measurement: %d", got)
	}
	// Measured geometry keeps precedence at the model-wide level.
	model.MeasuredKVBytesPerTok = nil
	model.MeasuredKVGeometry = map[string]KVGeometry{"q8_0": {FullLayers: 48, BytesPerCellPerLayer: 1000}}
	if got := computeKVTotalMB(model, ctx, "q8_0", false); got != int(float64(48*ctx)*1000/1048576+0.5) {
		t.Fatalf("per-layer math overrode measured geometry: %d", got)
	}
}

func TestKVLayerShapeSlotsSWAFullAndTransposedV(t *testing.T) {
	_, model := loadMiMoFixture(t)
	layout, _ := modelKVLayerLayout(model)
	base := kvCacheShape{Context: 1000, UBatch: 64, KVType: "q8_0"}
	if full, swa, streams := layout.cells(base); full != 1024 || swa != 256 || streams != 1 {
		t.Fatalf("context not padded like the backend: %d/%d/%d", full, swa, streams)
	}
	split := base
	split.Context, split.Slots = 8192, 4
	if full, swa, streams := layout.cells(split); full != 2048 || swa != 256 || streams != 4 {
		t.Fatalf("per-slot streams: %d/%d/%d", full, swa, streams)
	}
	unified := split
	unified.Unified = true
	if full, swa, streams := layout.cells(unified); full != 8192 || swa != 768 || streams != 1 {
		t.Fatalf("unified window scales by slots: %d/%d/%d", full, swa, streams)
	}
	swaFull := base
	swaFull.SWAFull = true
	if full, swa, _ := layout.cells(swaFull); swa != full {
		t.Fatalf("--swa-full keeps a window: %d/%d", full, swa)
	}
	fa, _ := kvLayerBytes(model, base)
	trans := base
	trans.VTransposed = true
	noFA, _ := kvLayerBytes(model, trans)
	// A 4-head full layer pads its V rows to the 8-head width without FA.
	if noFA[0] <= fa[0] || noFA[1] != fa[1] {
		t.Fatalf("transposed V width: fa=%d/%d nofa=%d/%d", fa[0], fa[1], noFA[0], noFA[1])
	}
}

func TestKVLayerLayoutRejectsWhatItCannotDescribe(t *testing.T) {
	_, valid := loadMiMoFixture(t)
	clone := func(edit func(*ModelProfile)) *ModelProfile {
		m := *valid
		m.HeadCountKVByLayer = append([]int(nil), valid.HeadCountKVByLayer...)
		m.SWAPattern = append([]int(nil), valid.SWAPattern...)
		edit(&m)
		return &m
	}
	cases := map[string]*ModelProfile{
		"short heads":        clone(func(m *ModelProfile) { m.HeadCountKVByLayer = m.HeadCountKVByLayer[:48] }),
		"zero head":          clone(func(m *ModelProfile) { m.HeadCountKVByLayer[3] = 0 }),
		"pattern value":      clone(func(m *ModelProfile) { m.SWAPattern[1] = 2 }),
		"long pattern":       clone(func(m *ModelProfile) { m.SWAPattern = append(m.SWAPattern, 1) }),
		"window unstated":    clone(func(m *ModelProfile) { m.SlidingWindow = 0 }),
		"implicit pattern":   clone(func(m *ModelProfile) { m.SWAPattern = nil }),
		"mla":                clone(func(m *ModelProfile) { m.KVLoraRank = 512 }),
		"recurrent":          clone(func(m *ModelProfile) { m.HasSSM = 1 }),
		"looped":             clone(func(m *ModelProfile) { m.KVLoops = 2 }),
		"shared kv arch":     clone(func(m *ModelProfile) { m.ModelArch = "gemma4" }),
		"router arch":        clone(func(m *ModelProfile) { m.ModelArch = "graniteswitch" }),
		"no head width":      clone(func(m *ModelProfile) { m.KeyLength, m.ValueLength = 0, 0 }),
		"scalar-only models": {NumLayers: 32, HeadCountKV: 8, KeyLength: 128, ValueLength: 128},
	}
	for name, model := range cases {
		if _, ok := modelKVLayerLayout(model); ok {
			t.Errorf("%s: accepted", name)
		}
	}
	// A model made only of NextN blocks has no trunk; nothing is filtered.
	allMTP := clone(func(m *ModelProfile) { m.NextNPredictLayers = m.NumLayers })
	layout, ok := modelKVLayerLayout(allMTP)
	if !ok || !layout.Layers[50].Cached {
		t.Fatalf("trunkless NextN model lost its cache: ok=%v", ok)
	}
}

// Scalar models must price exactly as before: the arrays are an addition,
// not a replacement of the scalar contract.
func TestScalarModelsKeepScalarKVPath(t *testing.T) {
	dense := &ModelProfile{NumLayers: 32, HeadCountKV: 8, KeyLength: 128, ValueLength: 128}
	want := int(float64(32*32768*8*256) * 1.0625 / 1024 / 1024)
	if got := computeKVTotalMB(dense, 32768, "q8_0", false); got != want {
		t.Fatalf("scalar dense = %d, want %d", got, want)
	}
	s := &Strategy{ContextSize: 32768, UBatchSize: 2048, Parallel: 1, KVType: "q8_0"}
	if got := computeKVTotalMBForStrategy(dense, s); got != want {
		t.Fatalf("strategy scope changed a scalar model: %d", got)
	}
	owned, _ := layerOwnership([]float64{.5, .5}, 32)
	if got := kvLayerShareMB(dense, kvShapeForStrategy(s, false), 1000, []float64{.5, .5}, owned, 1); got != ownedShareMB(1000, owned, 32, 1) {
		t.Fatalf("scalar share changed: %d", got)
	}
	// A uniform explicit window pattern is priced exactly, with the backend's
	// microbatch slack and padding rather than the bare window.
	patterned := &ModelProfile{NumLayers: 4, HeadCountKV: 2, KeyLength: 64, ValueLength: 64,
		SlidingWindow: 100, SWAPattern: []int{1, 1, 1, 0}}
	mb, ok := kvLayerTotalMB(patterned, kvCacheShape{Context: 4096, UBatch: 512, KVType: "f16"})
	// 3 windowed layers x PAD(612,256)=768 cells + 1 full layer x 4096 cells,
	// each cell 2x128 f16 elements = 512 bytes.
	if wantMB := bytesToMiBCeil((3*768 + 4096) * 512); !ok || mb != wantMB {
		t.Fatalf("explicit pattern total %d (ok=%v), want %d", mb, ok, wantMB)
	}
}
