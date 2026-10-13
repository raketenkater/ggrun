package placement

import (
	"math"
	"reflect"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Qwen3.8-27B-UD-Q5_K_S.gguf (sha256 899e3838...), header fields as
// tools/gguf/parse_gguf.py reports them. The backend logs below loaded this
// file.
func qwen38Q5KSProfile() *ModelProfile {
	return &ModelProfile{
		Path: "Qwen3.8-27B-UD-Q5_K_S.gguf", ModelArch: "qwen35",
		SizeBytes: 18665753504, TotalSizeMB: 17801, CTXTrain: 262144, ContextSize: 262144,
		NumLayers: 65, NextNPredictLayers: 1, HasSSM: 1, FullAttnInterval: 4,
		HeadCount: 24, HeadCountKV: 4, KeyLength: 256, ValueLength: 256, RopeDim: 64,
		EmbeddingLength: 5120, HiddenSize: 5120, FeedForwardLength: 17408, VocabSize: 248320,
		SSMConvKernel: 4, SSMStateSize: 128, SSMGroupCount: 16, SSMInnerSize: 6144, SSMTimeStepRank: 48,
		NonExpertBytes: 18654756864, OutputBytes: 1042944000, TokenEmbdBytes: 715161600,
		NonExpertLayerBytes: []int64{
			227605376, 232479616, 261766016, 260179968, 256891776, 249928576, 274996096, 236955648,
			243374976, 236002176, 227646336, 216844288, 218307456, 205323136, 210443136, 204064768,
			231824256, 222731136, 215522176, 215943168, 252713856, 274996096, 261479296, 275908608,
			268032896, 268032896, 268032896, 272427008, 268032896, 264100736, 263854976, 264071168,
			252713856, 252713856, 249928576, 251660288, 263854976, 268032896, 257301376, 261531648,
			263854976, 263854976, 263854976, 262801408, 259922816, 258939776, 263854976, 251660288,
			263854976, 258939776, 268032896, 262801408, 279870336, 268032896, 275692416, 274638848,
			291707776, 303545216, 303545216, 299583488, 303545216, 303545216, 291707776, 299583488,
			351008768,
		},
		ByteWeightedLayerSplit: true,
	}
}

// ik_llama-server 1fddd12 run directly: --tensor-split 0.27,0.58,0.15
// --ctx-size 65536 -b 128 -ub 128 q8_0 flash attention, --parallel 1 and 4
// (acceptance evidence q38-split-probe-np1.log / -np4.log).
var qwen38ProbeSplit = []float64{0.27, 0.58, 0.15}

func qwen38ProbeShape(slots int) kvCacheShape {
	return kvCacheShape{Context: 65536, UBatch: 128, Slots: slots, KVType: "q8_0"}
}

// "Layer sizes" table, third column (weights + cache, MiB), blocks 0-64 and
// the output layer.
var qwen38LoggedLayerMiB = map[int][]float64{
	1: {220.18, 224.83, 252.76, 384.13, 248.11, 241.47, 265.37, 361.98, 235.22, 228.19, 220.22, 342.80, 211.31, 198.93, 203.81, 330.61, 224.20, 215.53, 208.66, 341.94, 244.12, 265.37, 252.48, 399.13, 258.73, 258.73, 258.73, 395.81, 258.73, 254.98, 254.75, 387.84, 244.12, 244.12, 241.47, 376.00, 254.75, 258.73, 248.50, 385.42, 254.75, 254.75, 254.75, 386.63, 251.00, 250.06, 254.75, 376.00, 254.75, 250.06, 258.73, 386.63, 270.02, 258.73, 266.04, 397.92, 281.31, 292.60, 292.60, 421.71, 292.60, 292.60, 281.31, 421.71, 136.00, 1091.38},
	4: {229.53, 234.18, 262.11, 384.13, 257.46, 250.82, 274.73, 361.98, 244.57, 237.54, 229.57, 342.80, 220.66, 208.28, 213.16, 330.61, 233.55, 224.88, 218.01, 341.94, 253.48, 274.73, 261.83, 399.13, 268.08, 268.08, 268.08, 395.81, 268.08, 264.33, 264.10, 387.84, 253.48, 253.48, 250.82, 376.00, 264.10, 268.08, 257.85, 385.42, 264.10, 264.10, 264.10, 386.63, 260.35, 259.41, 264.10, 376.00, 264.10, 259.41, 268.08, 386.63, 279.37, 268.08, 275.39, 397.92, 290.66, 301.95, 301.95, 421.71, 301.95, 301.95, 290.66, 421.71, 136.00, 1091.38},
}

const mib = 1048576.0

func TestIKLayerSizesMatchTheBackendTable(t *testing.T) {
	model := qwen38Q5KSProfile()
	for slots, logged := range qwen38LoggedLayerMiB {
		sizes, maxCompute, ok := ikLayerSizes(model, qwen38ProbeShape(slots))
		if !ok || len(sizes) != len(logged) {
			t.Fatalf("slots=%d: %d sizes (ok=%v), want %d", slots, len(sizes), ok, len(logged))
		}
		for il, want := range logged {
			if got := sizes[il] / mib; math.Abs(got-want) > 0.011 {
				t.Errorf("slots=%d block %d: %.3f MiB, backend %.2f", slots, il, got, want)
			}
		}
		// "24.50 MiB" per trunk block: (2*17408+5120)*128*4 + 2*5120*128*4.
		if maxCompute != 24.5*mib {
			t.Errorf("slots=%d: max_compute %.3f MiB, backend 24.50", slots, maxCompute/mib)
		}
	}
}

// The backend set blocks 0-19 on CUDA0, 20-58 on CUDA1, 59-65 on CUDA2
// ("Setting default device in layer N to D") at both slot counts; the slot
// rule gives 0-17 / 18-56 / 57-65.
func TestIKAssignmentReproducesTheLoggedSplit(t *testing.T) {
	model := qwen38Q5KSProfile()
	want := make([]int, 65)
	for il := range want {
		switch {
		case il <= 19:
			want[il] = 0
		case il <= 58:
			want[il] = 1
		default:
			want[il] = 2
		}
	}
	for _, slots := range []int{1, 4} {
		devices, output := backendLayerDeviceAssignments(model, qwen38ProbeShape(slots), qwen38ProbeSplit)
		if !reflect.DeepEqual(devices, want) || output != 2 {
			t.Errorf("slots=%d: assignment %v output %d, backend %v output 2", slots, devices, output, want)
		}
		owned, _ := backendLayerOwnership(model, qwen38ProbeShape(slots), qwen38ProbeSplit)
		if !reflect.DeepEqual(owned, []int{20, 39, 6}) {
			t.Errorf("slots=%d: owned %v, backend 20/39/6 blocks plus the output", slots, owned)
		}
	}
	// A mainline backend keeps the slot rule.
	model.ByteWeightedLayerSplit = false
	owned, output := backendLayerOwnership(model, qwen38ProbeShape(4), qwen38ProbeSplit)
	if !reflect.DeepEqual(owned, []int{18, 39, 8}) || output != 2 {
		t.Errorf("mainline: owned %v output %d, want the slot rule's 18/39/8", owned, output)
	}
}

// Per-device buffers of the same runs: "CUDAn buffer size" (weights) and
// "CUDAn KV buffer size" (attention KV plus delta-net state of the device's
// own blocks).
func TestIKPerDeviceBuffersMatchTheBackendLog(t *testing.T) {
	model := qwen38Q5KSProfile()
	weights := []float64{4433.47, 9917.10, 2423.22}
	kv := map[int][]float64{
		1: {726.76, 1317.52, 281.35},
		4: {867.03, 1598.07, 309.41},
	}
	for slots, wantKV := range kv {
		shape := qwen38ProbeShape(slots)
		sizes, maxCompute, _ := ikLayerSizes(model, shape)
		devices, output := ikLayerDeviceAssignments(sizes, maxCompute, qwen38ProbeSplit)
		for i, b := range ikDeviceWeightBytes(model, shape, devices, output, 3, true) {
			if got := float64(b) / mib; math.Abs(got-weights[i]) > 0.02 {
				t.Errorf("slots=%d CUDA%d weights %.2f MiB, backend %.2f", slots, i, got, weights[i])
			}
		}
		total := computeKVTotalMB(model, 65536, "q8_0", false)
		gotKV := ikDeviceKVMB(model, shape, total, devices, 3)
		s := &Strategy{ContextSize: 65536, UBatchSize: 128, Parallel: slots, KVType: "q8_0", KVPlacement: "gpu",
			TensorSplit: qwen38ProbeSplit}
		owned, _ := backendLayerOwnership(model, shape, qwen38ProbeSplit)
		for i, want := range wantKV {
			if float64(gotKV[i]) < want || float64(gotKV[i]) > want+2 {
				t.Errorf("slots=%d CUDA%d KV %d MiB, backend %.2f", slots, i, gotKV[i], want)
			}
			// The planner's share (MoE room, VRAM-delta probe) and the recovery
			// exchange rate price the same blocks.
			if share := kvLayerShareMB(model, shape, total, qwen38ProbeSplit, owned, i); share != gotKV[i] {
				t.Errorf("slots=%d CUDA%d planner share %d MiB, device KV %d", slots, i, share, gotKV[i])
			}
			if share := StrategyDeviceKVCacheMB(model, s, "q8_0", false, i); share != gotKV[i] {
				t.Errorf("slots=%d CUDA%d recovery share %d MiB, device KV %d", slots, i, share, gotKV[i])
			}
		}
	}
}

// The first --claude-code preflight (A2-q38-27b-claude-code-kvfix/serve.log):
// --ctx-size 661504 --parallel 4 -ub 128 --tensor-split 0.27,0.58,0.15 failed
// on "CUDA0 allocation 7051 MiB": five attention blocks and fifteen recurrent
// blocks of 0-19. The slot rule charged CUDA0 18 blocks, 5,666 MiB.
func TestIKPredictsTheFailedClaudeCodeAllocation(t *testing.T) {
	model := qwen38Q5KSProfile()
	shape := kvCacheShape{Context: 661504, UBatch: 128, Slots: 4, KVType: "q8_0"}
	s := &Strategy{ContextSize: 661504, UBatchSize: 128, Parallel: 4, KVType: "q8_0", KVPlacement: "gpu",
		TensorSplit: qwen38ProbeSplit}
	if got := StrategyDeviceKVCacheMB(model, s, "q8_0", false, 0); got < 7050 || got > 7052 {
		t.Fatalf("CUDA0 KV %d MiB, backend allocated 7051", got)
	}
	model.ByteWeightedLayerSplit = false
	owned, _ := backendLayerOwnership(model, shape, qwen38ProbeSplit)
	total := computeKVTotalMB(model, 661504, "q8_0", false)
	if got := kvLayerShareMB(model, shape, total, qwen38ProbeSplit, owned, 0); got >= 7050 {
		t.Fatalf("slot rule charged CUDA0 %d MiB; the test no longer separates the rules", got)
	}
}

// The default 1-slot launch of the same model (A2-q38-27b-first ggrun log):
// --ctx-size 262144 -ub 512 --tensor-split 0.13,0.87,0.00 loaded CUDA0
// 2365.29 / CUDA1 14408.50 MiB of weights and 1112.94 / 7740.69 MiB of KV.
// The slot rule gave CUDA0 9 blocks, 2,141 MiB of weights.
func TestIKPredictsTheServedControlSplit(t *testing.T) {
	model := qwen38Q5KSProfile()
	split := []float64{0.13, 0.87, 0}
	shape := kvCacheShape{Context: 262144, UBatch: 512, Slots: 1, KVType: "q8_0"}
	devices, output := backendLayerDeviceAssignments(model, shape, split)
	weights := ikDeviceWeightBytes(model, shape, devices, output, 3, true)
	total := computeKVTotalMB(model, 262144, "q8_0", false)
	kv := ikDeviceKVMB(model, shape, total, devices, 3)
	for i, want := range []float64{2365.29, 14408.50, 0} {
		if got := float64(weights[i]) / mib; math.Abs(got-want) > 0.02 {
			t.Errorf("CUDA%d weights %.2f MiB, backend %.2f", i, got, want)
		}
	}
	for i, want := range []float64{1112.94, 7740.69, 0} {
		if float64(kv[i]) < want || float64(kv[i]) > want+2 {
			t.Errorf("CUDA%d KV %d MiB, backend %.2f", i, kv[i], want)
		}
	}
}

func TestByteWeightedAssignmentNeedsTheBackendFactAndTensorBytes(t *testing.T) {
	splits := [][]float64{{0.27, 0.58, 0.15}, {0.5, 0.5}, {0.3, 0, 0.7}, {1}}
	for _, mutate := range []func(*ModelProfile){
		func(m *ModelProfile) { m.ByteWeightedLayerSplit = false },
		func(m *ModelProfile) { m.NonExpertLayerBytes = nil },
		func(m *ModelProfile) { m.VocabSize = 0 },
		func(m *ModelProfile) { m.ExpertBytes = 1 << 30 }, // MoE without per-block expert bytes
	} {
		model := qwen38Q5KSProfile()
		mutate(model)
		for _, split := range splits {
			gotDev, gotOut := backendLayerDeviceAssignments(model, qwen38ProbeShape(4), split)
			wantDev, wantOut := layerDeviceAssignments(split, model.NumLayers)
			if !reflect.DeepEqual(gotDev, wantDev) || gotOut != wantOut {
				t.Errorf("split %v: %v/%d, want the slot rule %v/%d", split, gotDev, gotOut, wantDev, wantOut)
			}
		}
	}
	// A single device is the whole model under either rule.
	model := qwen38Q5KSProfile()
	gotDev, gotOut := backendLayerDeviceAssignments(model, qwen38ProbeShape(4), []float64{1})
	wantDev, wantOut := layerDeviceAssignments([]float64{1}, model.NumLayers)
	if !reflect.DeepEqual(gotDev, wantDev) || gotOut != wantOut {
		t.Errorf("single device: %v/%d, want %v/%d", gotDev, gotOut, wantDev, wantOut)
	}
}

// The --claude-code box: RTX 4070, RTX 3090 Ti, and an RTX 3060 holding the
// 5,040 MiB reviewer (A2-q38-27b-claude-code-converge).
func claudeCodeBox() *detect.Capabilities {
	caps := benchBox()
	caps.GPUs[0].VRAMUsedMB = 690
	caps.GPUs[1].VRAMUsedMB = 1113
	caps.GPUs[2].VRAMUsedMB = 586 + 5040
	return caps
}

func claudeCodeOptions(t *testing.T, ctx int) Options {
	return Options{
		ContextSize: ctx, Parallel: 4, BatchSize: 128, UBatchSize: 128,
		BatchSizeExplicit: true, UBatchSizeExplicit: true, RequireMeasuredBuffers: true,
		KVPlacement: "gpu", KVQuality: "q8_0", CacheDir: t.TempDir(),
	}
}

// ikPlanOverfill is the most any device's backend-assigned weights and KV
// exceed its capacity under a plan's split.
func ikPlanOverfill(t *testing.T, caps *detect.Capabilities, model *ModelProfile, s *Strategy, opts Options) (int, int) {
	t.Helper()
	if len(s.TensorSplit) != len(caps.GPUs) {
		return math.MinInt, -1 // one device: no split to walk
	}
	ik := *model
	ik.ByteWeightedLayerSplit = true
	shape := kvShapeForStrategy(s, false)
	kvTotal := computeKVTotalMBForStrategy(&ik, s)
	modelMB, kvMB := ikDenseDeviceMB(&ik, shape, kvTotal, s.TensorSplit)
	cudaOverheadMB, computeBufMB := perGPUOverheadMB(caps, &ik, s, opts)
	worst, worstDev := math.MinInt, -1
	for i, g := range caps.GPUs {
		if s.TensorSplit[i] <= 0 && modelMB[i]+kvMB[i] == 0 {
			continue
		}
		if over := modelMB[i] + kvMB[i] - (g.VRAMFreeMB() - cudaOverheadMB - computeBufMB); over > worst {
			worst, worstDev = over, i
		}
	}
	return worst, worstDev
}

// The converging-preflight failure: every derate re-split the cards and the
// backend's byte walk moved the deficit to another device. Every context of
// the logged derate sequence fits the box's weights and KV, so each must plan
// a split whose backend assignment fits every card; and the slot-rule plan
// the launch used must show the overfill the backend reported. The recovery
// loop itself is driven in cmd/ggrun (TestIKKVDerateKeepsEveryDeviceWithinCapacity).
func TestIKDenseSplitKeepsEveryDeviceWithinCapacityAcrossDerates(t *testing.T) {
	caps := claudeCodeBox()
	countOverfilled := 0
	for _, ctx := range []int{661504, 593920, 518144, 483328, 476160, 262144, 131072} {
		opts := claudeCodeOptions(t, ctx)
		model := qwen38Q5KSProfile()
		s, err := Compute(caps, model, opts)
		if err != nil {
			t.Fatalf("ctx=%d: refused: %v", ctx, err)
		}
		if over, dev := ikPlanOverfill(t, caps, model, s, opts); over > 0 {
			t.Errorf("ctx=%d split %v: backend assignment overfills CUDA%d by %d MiB", ctx, s.TensorSplit, dev, over)
		}
		mainline := qwen38Q5KSProfile()
		mainline.ByteWeightedLayerSplit = false
		if old, err := Compute(caps, mainline, claudeCodeOptions(t, ctx)); err == nil {
			if over, dev := ikPlanOverfill(t, caps, mainline, old, opts); over > 0 {
				countOverfilled++
				t.Logf("ctx=%d slot-rule split %v overfills CUDA%d by %d MiB under the byte walk", ctx, old.TensorSplit, dev, over)
			}
		}
	}
	if countOverfilled == 0 {
		t.Fatal("the slot-rule plans never overfill under the byte walk; the test no longer reproduces the failure")
	}
}

// The automatic context fit lands on a context whose split fits every card
// under the backend's own assignment.
func TestIKAutoContextFitsEveryDevice(t *testing.T) {
	caps := claudeCodeBox()
	opts := claudeCodeOptions(t, 0)
	model := qwen38Q5KSProfile()
	s, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("auto context: %v", err)
	}
	if s.Type != MultiGPUDense || len(s.TensorSplit) != 3 {
		t.Fatalf("want a 3-GPU dense plan, got %v split %v", s.Type, s.TensorSplit)
	}
	if over, dev := ikPlanOverfill(t, caps, model, s, opts); over > 0 {
		t.Fatalf("ctx=%d split %v overfills CUDA%d by %d MiB", s.ContextSize, s.TensorSplit, dev, over)
	}
	t.Logf("auto ctx=%d split %v", s.ContextSize, s.TensorSplit)
}

// Nothing measured the compute buffer or runtime growth of the 4-slot
// --claude-code plan, so the automatic context must leave every used card its
// cold reserve. Fitting the search's candidates against measured buffers
// alone (0 here) grew the context to 622,592 and left CUDA1 636 MiB where the
// failed launch implied about 755 MiB was needed. An explicit context keeps
// that fallback: exact preflight decides it (TestIKDenseSplitRefusesOnlyOnFirmCapacity).
func TestIKAutoContextKeepsTheColdReserveOnEveryCard(t *testing.T) {
	caps := claudeCodeBox()
	opts := claudeCodeOptions(t, 0)
	model := qwen38Q5KSProfile()
	s, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("auto context: %v", err)
	}
	cold := opts
	cold.RequireMeasuredBuffers = false
	if _, computeBufMB := perGPUOverheadMB(caps, model, s, cold); computeBufMB < computeFloorMB {
		t.Fatalf("cold compute reserve %d MiB, want the %d MiB floor: the test needs an unmeasured cache", computeBufMB, computeFloorMB)
	}
	if over, dev := ikPlanOverfill(t, caps, model, s, cold); over > 0 {
		t.Fatalf("auto ctx=%d split %v: CUDA%d is %d MiB into its unmeasured cold reserve", s.ContextSize, s.TensorSplit, dev, over)
	}
	if s.ContextSize >= 622592 {
		t.Fatalf("auto ctx=%d, want below the 622,592 the measured-only fallback planned", s.ContextSize)
	}
}

// syntheticMoEProfile is an 80B-class MoE with per-block tensor bytes: three
// dense leading blocks, then 2 GiB of routed experts per block.
func syntheticMoEProfile() *ModelProfile {
	const n, dense = 40, 3
	m := &ModelProfile{
		Path: "synthetic-moe-80b.gguf", ModelArch: "qwen3moe", IsMoE: true, NumExperts: 128, ExpertUsedCount: 8,
		CTXTrain: 262144, ContextSize: 262144, NumLayers: n, LeadingDense: dense,
		HeadCount: 32, HeadCountKV: 4, KeyLength: 128, ValueLength: 128,
		EmbeddingLength: 4096, HiddenSize: 4096, FeedForwardLength: 12288, ExpertFF: 1536, ExpertSharedFF: 1536, HasShexp: 1,
		VocabSize: 151936, OutputBytes: 622329856, TokenEmbdBytes: 622329856,
	}
	for il := 0; il < n; il++ {
		nonExpert := int64(110 << 20)
		var routed, shexp, aux int64
		if il < dense {
			nonExpert = 400 << 20
		} else {
			routed, shexp, aux = 2<<30, 30<<20, 2<<20
		}
		m.NonExpertLayerBytes = append(m.NonExpertLayerBytes, nonExpert)
		m.RoutedExpertLayerBytes = append(m.RoutedExpertLayerBytes, routed)
		m.ShexpLayerBytes = append(m.ShexpLayerBytes, shexp)
		m.ExpertAuxLayerBytes = append(m.ExpertAuxLayerBytes, aux)
		m.ExpertLayerBytes = append(m.ExpertLayerBytes, routed+shexp)
		m.NonExpertBytes += nonExpert + aux
		m.ExpertBytes += routed + shexp
		m.ShexpBytes += shexp
		m.ExpertAuxBytes += aux
	}
	m.NonExpertBytes += m.OutputBytes + m.TokenEmbdBytes
	m.SizeBytes = m.NonExpertBytes + m.ExpertBytes
	m.TotalSizeMB = int(m.SizeBytes >> 20)
	return m
}

// Single-GPU and CPU-only plans have no split, so the backend fact changes
// nothing; mainline multi-GPU plans are the ones the slot rule produced
// before byte-weighted assignment existed (computed at 1a0ecb6).
func TestByteWeightedSplitLeavesOtherPlansUnchanged(t *testing.T) {
	args := func(caps *detect.Capabilities, model *ModelProfile, opts Options) []string {
		t.Helper()
		s, err := Compute(caps, model, opts)
		if err != nil {
			t.Fatalf("compute: %v", err)
		}
		return s.Args(model.Path, 8080)
	}
	single := claudeCodeBox()
	single.GPUs = single.GPUs[1:2]
	single.GPUs[0].Index = 0
	for name, tc := range map[string]struct {
		caps *detect.Capabilities
		opts Options
	}{
		"single-gpu": {single, Options{ContextSize: 65536, KVQuality: "q8_0", KVPlacement: "gpu", CacheDir: t.TempDir()}},
		"cpu-only":   {claudeCodeBox(), Options{ContextSize: 65536, KVQuality: "q8_0", CPUMode: true, CacheDir: t.TempDir()}},
	} {
		ik, mainline := qwen38Q5KSProfile(), qwen38Q5KSProfile()
		mainline.ByteWeightedLayerSplit = false
		if got, want := args(tc.caps, ik, tc.opts), args(tc.caps, mainline, tc.opts); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: byte-weighted backend changed the plan\n got %v\nwant %v", name, got, want)
		}
	}

	pin := func(layers, device string) string {
		return `blk\.(` + layers + `)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=` + device
	}
	claudeCode := claudeCodeOptions(t, 0)
	for _, tc := range []struct {
		name    string
		model   *ModelProfile
		opts    Options
		ctx     int
		split   []float64
		ncpumoe int
		ot      string
	}{
		{"dense explicit", qwen38Q5KSProfile(), claudeCodeOptions(t, 593920), 593920, []float64{0.31, 0.62, 0.07}, 0, ""},
		// The split the failing --claude-code launch emitted.
		{"dense auto", qwen38Q5KSProfile(), claudeCode, 622592, []float64{0.27, 0.58, 0.15}, 0, ""},
		{"moe explicit", syntheticMoEProfile(), Options{ContextSize: 131072, Parallel: 1, KVPlacement: "gpu", KVQuality: "q8_0", CacheDir: t.TempDir()},
			131072, []float64{0.28, 0.66, 0.06}, 28, pin("3|4|5|6|7|8", "CUDA1") + "," + pin("9|10", "CUDA0") + "," + pin("11", "CUDA2") + ",exps=CPU"},
		{"moe auto", syntheticMoEProfile(), Options{Parallel: 4, BatchSize: 128, UBatchSize: 128, BatchSizeExplicit: true, UBatchSizeExplicit: true,
			KVQuality: "q8_0", CacheDir: t.TempDir()}, 567296, []float64{0.29, 0.64, 0.07}, 36, pin("3", "CUDA2") + ",exps=CPU"},
	} {
		tc.model.ByteWeightedLayerSplit = false
		s, err := Compute(claudeCodeBox(), tc.model, tc.opts)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if s.ContextSize != tc.ctx || !reflect.DeepEqual(s.TensorSplit, tc.split) || s.NCPUMoE != tc.ncpumoe || s.OTString != tc.ot {
			t.Errorf("%s: mainline plan changed: ctx %d split %v n-cpu-moe %d ot %q\nwant ctx %d split %v n-cpu-moe %d ot %q",
				tc.name, s.ContextSize, s.TensorSplit, s.NCPUMoE, s.OTString, tc.ctx, tc.split, tc.ncpumoe, tc.ot)
		}
	}
}

// A card the plan does not need may hold less free VRAM than its CUDA context
// and compute reserve (a reviewer, a desktop, an OOM penalty). Its capacity is
// clamped at 0 and a device holding nothing is never over, so the ik plan uses
// the same cards and context as mainline.
func TestIKDenseSplitIgnoresAFullUnusedCard(t *testing.T) {
	for _, used := range []int{11700, 12200} {
		for name, opts := range map[string]Options{
			"explicit": {ContextSize: 262144, Parallel: 1, KVPlacement: "gpu", KVQuality: "q8_0", CacheDir: t.TempDir()},
			"auto":     {Parallel: 1, KVPlacement: "gpu", KVQuality: "q8_0", CacheDir: t.TempDir()},
		} {
			caps := claudeCodeBox()
			caps.GPUs[2].VRAMUsedMB = used
			mainline := qwen38Q5KSProfile()
			mainline.ByteWeightedLayerSplit = false
			want, err := Compute(caps, mainline, opts)
			if err != nil {
				t.Fatalf("used=%d %s: mainline: %v", used, name, err)
			}
			got, err := Compute(caps, qwen38Q5KSProfile(), opts)
			if err != nil {
				t.Fatalf("used=%d %s: ik refused the plan mainline makes (%v split %v): %v", used, name, want.Type, want.TensorSplit, err)
			}
			cards := func(s *Strategy) int {
				n := 0
				for _, v := range s.TensorSplit {
					if v > 0 {
						n++
					}
				}
				return n
			}
			if got.Type != want.Type || got.ContextSize != want.ContextSize || cards(got) != cards(want) {
				t.Errorf("used=%d %s: ik %v ctx %d split %v, mainline %v ctx %d split %v",
					used, name, got.Type, got.ContextSize, got.TensorSplit, want.Type, want.ContextSize, want.TensorSplit)
			}
			if got.Type == MultiGPUDense && got.TensorSplit[2] != 0 {
				t.Errorf("used=%d %s: split %v gives the full card a share", used, name, got.TensorSplit)
			}
		}
	}
}

// Block edges can leave a card short of the cold 1024 MiB compute floor. That
// estimate only ranks: the best split is emitted for exact preflight. Weights
// and KV alone over a card's capacity less measured reserves still refuse.
func TestIKDenseSplitRefusesOnlyOnFirmCapacity(t *testing.T) {
	model := qwen38Q5KSProfile()
	shape := kvCacheShape{Context: 622592, UBatch: 128, Slots: 4, KVType: "q8_0"}
	kvTotal := computeKVTotalMB(model, shape.Context, shape.KVType, false)
	intent := []float64{0.27, 0.58, 0.15}
	modelMB, kvMB := ikDenseDeviceMB(model, shape, kvTotal, intent)
	need := 0
	for i := range modelMB {
		need += modelMB[i] + kvMB[i]
	}
	// Total capacity one MiB short of the bytes: no split fits every card.
	devices := make([]ikDenseDevice, 3)
	left := need - 1
	for i := range devices {
		c := (modelMB[i] + kvMB[i]) * (need - 1) / need
		if i == len(devices)-1 {
			c = left
		}
		left -= c
		devices[i] = ikDenseDevice{FreeMB: c, CapacityMB: c, FirmMB: c}
	}
	if _, ok, err := ikDenseSplit(model, shape, kvTotal, intent, devices); !ok || err == nil {
		t.Fatalf("firm overfill not refused: ok=%v err=%v", ok, err)
	}
	// The same shortfall against a cold compute reserve emits a split: with
	// room for the largest block beside each card's share, block edges cannot
	// overfill the firm capacity.
	sizes, _, _ := ikLayerSizes(model, shape)
	largest := 0.0
	for _, size := range sizes {
		largest = max(largest, size)
	}
	for i := range devices {
		devices[i].FirmMB += int(largest/mib) + 1
	}
	split, ok, err := ikDenseSplit(model, shape, kvTotal, intent, devices)
	if !ok || err != nil {
		t.Fatalf("cold-reserve overfill refused: ok=%v err=%v", ok, err)
	}
	modelMB, kvMB = ikDenseDeviceMB(model, shape, kvTotal, split)
	for i, d := range devices {
		if mb := modelMB[i] + kvMB[i]; mb > d.FirmMB {
			t.Errorf("split %v: CUDA%d holds %d MiB over its firm %d", split, i, mb, d.FirmMB)
		}
	}

	// Explicit contexts planned with cold buffers on the --claude-code box,
	// which mainline plans (622,592 is the failing launch's split), are not
	// refused, and their weights and KV fit every card's free VRAM.
	caps := claudeCodeBox()
	for _, ctx := range []int{604160, 610304, 622592} {
		opts := claudeCodeOptions(t, ctx)
		opts.RequireMeasuredBuffers = false
		s, err := Compute(caps, qwen38Q5KSProfile(), opts)
		if err != nil {
			t.Fatalf("ctx=%d: cold ik plan refused: %v", ctx, err)
		}
		firm := opts
		firm.RequireMeasuredBuffers = true
		if over, dev := ikPlanOverfill(t, caps, qwen38Q5KSProfile(), s, firm); over > 0 {
			t.Errorf("ctx=%d split %v: CUDA%d weights and KV over free VRAM by %d MiB", ctx, s.TensorSplit, dev, over)
		}
	}
}

// MiniMax-M3-UD-IQ3_XXS (5 shards), header fields as ggrun's gguf parser
// reports them: three dense blocks, then MoE blocks of 2,585,788,416 routed
// expert bytes, 2,831,155,200 in blocks 52 and 58 and 3,359,637,504 in 59.
func minimaxM3Profile() *ModelProfile {
	m := &ModelProfile{
		Path: "MiniMax-M3-UD-IQ3_XXS-00001-of-00005.gguf", ModelArch: "minimax-m3",
		SizeBytes: 159407162624, TotalSizeMB: 152023, CTXTrain: 1048576, ContextSize: 1048576,
		NumLayers: 60, IsMoE: true, NumExperts: 128, ExpertUsedCount: 4, LeadingDense: 3,
		HeadCount: 64, HeadCountKV: 4, KeyLength: 128, ValueLength: 128, RopeDim: 64,
		EmbeddingLength: 6144, HiddenSize: 6144, FeedForwardLength: 12288, ExpertFF: 3072,
		ExpertSharedCount: 1, HasShexp: 1, VocabSize: 200064,
		ExpertBytes: 151315808256, NonExpertBytes: 8083051008, TokenEmbdBytes: 1008322560, OutputBytes: 1008322560,
		ShexpBytes: 2661285888, ExpertAuxBytes: 179306496,
		ByteWeightedLayerSplit: true,
	}
	for il := 0; il < m.NumLayers; il++ {
		nonExpert, routed, shexp, aux := int64(87787008), int64(2585788416), int64(46448640), int64(3145728)
		switch il {
		case 0:
			nonExpert, routed, shexp, aux = 273581056, 0, 0, 0
		case 1, 2:
			nonExpert, routed, shexp, aux = 291865600, 0, 0, 0
		case 52, 58:
			routed = 2831155200
		case 59:
			nonExpert, routed, shexp = 113690112, 3359637504, 60162048
		}
		m.NonExpertLayerBytes = append(m.NonExpertLayerBytes, nonExpert)
		m.RoutedExpertLayerBytes = append(m.RoutedExpertLayerBytes, routed)
		m.ShexpLayerBytes = append(m.ShexpLayerBytes, shexp)
		m.ExpertAuxLayerBytes = append(m.ExpertAuxLayerBytes, aux)
		m.ExpertLayerBytes = append(m.ExpertLayerBytes, routed+shexp)
	}
	return m
}

// ik_llama-server 1fddd12 under ggrun (matrix2/A1-minimax-learn-1 ggrun log):
// --ctx-size 436224 -b 2048 -ub 512 q8_0 --parallel 1 --tensor-split
// 0.27,0.59,0.14, routed experts on the CPU except blocks 3-4 (experts, shared
// experts and router pinned to CUDA2). The backend walk counts the CPU experts
// and gives 18/35/7 blocks; the slot rule gives 17/36/7.
func TestIKMiniMaxM3BuffersMatchTheBackendLog(t *testing.T) {
	model := minimaxM3Profile()
	var experts, shexp, aux int64
	for il := range model.ExpertLayerBytes {
		experts += model.ExpertLayerBytes[il]
		shexp += model.ShexpLayerBytes[il]
		aux += model.ExpertAuxLayerBytes[il]
	}
	if experts != model.ExpertBytes || shexp != model.ShexpBytes || aux != model.ExpertAuxBytes {
		t.Fatalf("per-block bytes %d/%d/%d do not sum to the header's %d/%d/%d", experts, shexp, aux,
			model.ExpertBytes, model.ShexpBytes, model.ExpertAuxBytes)
	}
	split := []float64{0.27, 0.59, 0.14}
	shape := kvCacheShape{Context: 436224, UBatch: 512, Slots: 1, KVType: "q8_0"}
	owned, output := backendLayerOwnership(model, shape, split)
	if !reflect.DeepEqual(owned, []int{18, 35, 7}) || output != 2 {
		t.Fatalf("owned %v output %d, backend 18/35/7 output 2", owned, output)
	}
	// "CUDAn KV buffer size": 8147.26 / 15841.89 / 3168.38 MiB.
	total := computeKVTotalMB(model, shape.Context, shape.KVType, false)
	for i, want := range []float64{8147.26, 15841.89, 3168.38} {
		if got := kvLayerShareMB(model, shape, total, split, owned, i); float64(got) < want || float64(got) > want+2 {
			t.Errorf("CUDA%d KV %d MiB, backend %.2f", i, got, want)
		}
	}
	// "CUDAn buffer size": 2688.26 / 4585.60 / 6943.13 MiB. The -ot pin moves
	// blocks 3-4's experts, shared experts and router from their owner to CUDA2.
	devices, outputDev := backendLayerDeviceAssignments(model, shape, split)
	weights := ikDeviceWeightBytes(model, shape, devices, outputDev, 3, false)
	for _, il := range []int{3, 4} {
		weights[devices[il]] -= model.ShexpLayerBytes[il] + model.ExpertAuxLayerBytes[il]
		weights[2] += model.ExpertLayerBytes[il] + model.ExpertAuxLayerBytes[il]
	}
	for i, want := range []float64{2688.26, 4585.60, 6943.13} {
		if got := float64(weights[i]) / mib; math.Abs(got-want) > 0.02 {
			t.Errorf("CUDA%d weights %.2f MiB, backend %.2f", i, got, want)
		}
	}
	model.ByteWeightedLayerSplit = false
	if owned, _ := backendLayerOwnership(model, shape, split); reflect.DeepEqual(owned, []int{18, 35, 7}) {
		t.Fatal("the slot rule also gives 18/35/7; the test no longer separates the rules")
	}
}

// The optimizer's resource ledger prices each device of an ik dense plan
// with the walk the planner refused overfills with: weights and KV together
// equal ikDenseDeviceMB, and a calibration split candidate that would overfill
// a card under the walk is not offered.
func TestIKResourceLedgerFollowsTheWalk(t *testing.T) {
	caps := claudeCodeBox()
	opts := claudeCodeOptions(t, 0)
	model := qwen38Q5KSProfile()
	s, err := Compute(caps, model, opts)
	if err != nil || s.Type != MultiGPUDense {
		t.Fatalf("auto plan: %v %v", s, err)
	}
	ledger := BuildResourceLedger(caps, model, s, opts)
	modelMB, kvMB := ikDenseDeviceMB(model, kvShapeForStrategy(s, false), computeKVTotalMBForStrategy(model, s), s.TensorSplit)
	for i, d := range ledger.Devices {
		if d.ModelMB+d.ContextMB != modelMB[i]+kvMB[i] {
			t.Errorf("CUDA%d ledger %d+%d MiB, walk %d+%d", i, d.ModelMB, d.ContextMB, modelMB[i], kvMB[i])
		}
	}
	for _, c := range CalibrationCandidates(caps, model, s, opts) {
		cs := c.Strategy
		if cs.Type != MultiGPUDense || len(cs.TensorSplit) != len(caps.GPUs) {
			continue
		}
		if over, dev := ikPlanOverfill(t, caps, model, cs, opts); over > 0 {
			t.Errorf("candidate %s split %v overfills CUDA%d by %d MiB under the walk", c.Name, cs.TensorSplit, dev, over)
		}
	}
	// Mainline keeps the split-fraction weights.
	mainline := qwen38Q5KSProfile()
	mainline.ByteWeightedLayerSplit = false
	shares := estimatedModelShares(mainline, s, caps.GPUs, mainline.TotalSizeMB)
	for i, v := range s.TensorSplit {
		if want := int(math.Ceil(float64(mainline.TotalSizeMB) * v)); shares[i] != want {
			t.Errorf("mainline CUDA%d weights %d MiB, want the split fraction's %d", i, shares[i], want)
		}
	}
}
