package placement

import (
	"math"
	"strings"
)

// Per-layer KV geometry.
//
// Some models state attention per block instead of with scalars. MiMo-V2
// (mainline llama.cpp 6b790a9c2, verified against its live allocation in
// docs/evidence/mimo-kv-geometry-20260929.json) has 51 stored blocks: 48 serve
// the normal context, 3 are MTP blocks with no normal cache; 9 full-attention
// blocks carry 4 KV heads and the 39 windowed ones carry 8. The scalar
// formula saw no head count at all and priced its KV at zero, and the
// layer-count share put 2 MTP blocks' worth of KV on a GPU that holds none.
//
// This file prices such a model from its arrays using the backend's own rules:
//
//   - n_embd_{k,v}_gqa(il) = head width(il) * n_head_kv(il), where windowed
//     layers use key_length_swa/value_length_swa when stated
//     (src/llama-hparams.cpp n_embd_head_k);
//   - the normal cache holds only blocks il < n_layer_all - nextn when the model
//     has a trunk (src/llama-model.cpp create_memory filter);
//   - the windowed cache has PAD(min(base, n_swa*(unified ? slots : 1) +
//     n_ubatch), 256) cells per stream, or base cells with --swa-full
//     (src/llama-kv-cache-iswa.cpp);
//   - base cells per stream are n_ctx (unified) or PAD(n_ctx/slots, 256), with
//     one stream per slot unless unified (src/llama-context.cpp);
//   - without flash attention V is transposed and every layer's V uses the
//     widest V row (src/llama-kv-cache.cpp TAG_V_CACHE_VARIABLE);
//   - each block's cache lives on that block's device.
//
// It is a formula, not a measurement: an exact scoped allocation, a measured
// geometry or a measured rate keeps precedence wherever computeKVTotalMB
// already prefers it. Layouts this file cannot describe (MLA, recurrent or
// hybrid state, looped blocks, shared/reused KV layers, router blocks, an SWA
// window without a stated pattern) return !ok and keep the existing path.

// kvLayer is one stored block's normal-context cache.
type kvLayer struct {
	Cached bool // false for MTP blocks outside the serving cache
	SWA    bool
	KWidth int // elements per cell
	VWidth int
}

type kvLayerLayout struct {
	Layers    []kvLayer
	Window    int
	VWidthMax int
}

// kvCacheShape is the launch geometry the backend sizes its caches from.
type kvCacheShape struct {
	Context     int
	UBatch      int // <= 0: the backend's default n_ubatch
	Slots       int // n_seq_max; <= 0 means 1
	Unified     bool
	SWAFull     bool
	KVType      string
	VTransposed bool // flash attention off
}

// backendDefaultUBatch is llama.cpp's n_ubatch when none is passed. Model-wide
// callers that have not resolved a microbatch yet price the window at it.
const backendDefaultUBatch = 512

// kvSpecialCacheArchs share or reuse KV between blocks, or repurpose the NextN
// count for a router block, so a per-block sum does not describe them.
var kvSpecialCacheArchs = map[string]bool{
	"gemma3n": true, "gemma4": true, "gemma4-assistant": true, "graniteswitch": true,
}

// modelKVLayerLayout validates the per-block arrays. It applies only when the
// model states at least one array; scalar-only models keep the scalar path.
func modelKVLayerLayout(model *ModelProfile) (kvLayerLayout, bool) {
	if model == nil || (len(model.HeadCountKVByLayer) == 0 && len(model.SWAPattern) == 0) {
		return kvLayerLayout{}, false
	}
	n := model.NumLayers
	if n <= 0 || model.KVLoraRank > 0 || hasSSMLayout(model) || model.KVLoops > 1 ||
		kvSpecialCacheArchs[strings.ToLower(model.ModelArch)] {
		return kvLayerLayout{}, false
	}
	if len(model.HeadCountKVByLayer) > 0 && len(model.HeadCountKVByLayer) != n {
		return kvLayerLayout{}, false
	}
	if len(model.SWAPattern) > 0 && len(model.SWAPattern) != n {
		return kvLayerLayout{}, false
	}
	if len(model.SWAPattern) == 0 && model.SlidingWindow > 0 {
		// The backend derives this model's windowed blocks from an
		// architecture rule that the metadata does not state.
		return kvLayerLayout{}, false
	}
	keyLen, valueLen := model.KeyLength, model.ValueLength
	if keyLen <= 0 && model.EmbeddingLength > 0 && model.HeadCount > 0 {
		keyLen = model.EmbeddingLength / model.HeadCount
	}
	if valueLen <= 0 {
		valueLen = keyLen
	}
	if keyLen <= 0 || valueLen <= 0 {
		return kvLayerLayout{}, false
	}
	keySWA, valueSWA := keyLen, valueLen
	if model.KeyLengthSWA > 0 {
		keySWA = model.KeyLengthSWA
	}
	if model.ValueLengthSWA > 0 {
		valueSWA = model.ValueLengthSWA
	}
	cached := n
	if nextn := model.NextNPredictLayers; nextn > 0 && nextn < n {
		cached = n - nextn
	}
	layout := kvLayerLayout{Layers: make([]kvLayer, n), Window: model.SlidingWindow}
	anySWA := false
	for il := 0; il < n; il++ {
		heads := model.HeadCountKV
		if len(model.HeadCountKVByLayer) > 0 {
			heads = model.HeadCountKVByLayer[il]
		}
		if heads <= 0 {
			return kvLayerLayout{}, false
		}
		swa := false
		if len(model.SWAPattern) > 0 {
			switch model.SWAPattern[il] {
			case 0:
			case 1:
				swa = true
			default:
				return kvLayerLayout{}, false
			}
		}
		layer := kvLayer{Cached: il < cached, SWA: swa, KWidth: heads * keyLen, VWidth: heads * valueLen}
		if swa {
			layer.KWidth, layer.VWidth = heads*keySWA, heads*valueSWA
		}
		if layer.Cached {
			anySWA = anySWA || swa
			layout.VWidthMax = max(layout.VWidthMax, layer.VWidth)
		}
		layout.Layers[il] = layer
	}
	if anySWA && model.SlidingWindow <= 0 {
		return kvLayerLayout{}, false
	}
	return layout, true
}

func padTo256(n int64) int64 {
	return (n + 255) / 256 * 256
}

// cells returns the per-stream depth of the full and windowed caches and the
// number of streams.
func (l kvLayerLayout) cells(shape kvCacheShape) (full, swa, streams int64) {
	slots := int64(max(1, shape.Slots))
	ctx := padTo256(int64(shape.Context))
	full, streams = ctx, 1
	if !shape.Unified && slots > 1 {
		full, streams = padTo256(ctx/slots), slots
	}
	if shape.SWAFull {
		return full, full, streams
	}
	ub := int64(shape.UBatch)
	if ub <= 0 {
		ub = backendDefaultUBatch
	}
	window := int64(l.Window)
	if shape.Unified {
		window *= slots
	}
	return full, padTo256(min(full, window+ub)), streams
}

// kvRowBytes is ggml_row_size for the cache types placement emits.
func kvRowBytes(kvType string, elems int) (int64, bool) {
	perElem, ok := kvTypeBytesPerElement(kvType)
	if !ok || elems <= 0 {
		return 0, false
	}
	if perElem >= 2 {
		return int64(elems) * int64(perElem), true
	}
	blockBytes := int64(math.Round(perElem * 32))
	return (int64(elems) + 31) / 32 * blockBytes, true
}

// kvLayerBytes prices every stored block at the given shape; uncached blocks
// cost zero.
func kvLayerBytes(model *ModelProfile, shape kvCacheShape) ([]int64, bool) {
	layout, ok := modelKVLayerLayout(model)
	if !ok || shape.Context <= 0 {
		return nil, false
	}
	full, swa, streams := layout.cells(shape)
	out := make([]int64, len(layout.Layers))
	for il, layer := range layout.Layers {
		if !layer.Cached {
			continue
		}
		vWidth := layer.VWidth
		if shape.VTransposed {
			vWidth = layout.VWidthMax
		}
		kRow, okK := kvRowBytes(shape.KVType, layer.KWidth)
		vRow, okV := kvRowBytes(shape.KVType, vWidth)
		if !okK || !okV {
			return nil, false
		}
		cells := full
		if layer.SWA {
			cells = swa
		}
		out[il] = (kRow + vRow) * cells * streams
	}
	return out, true
}

// kvLayerTotalMB is the per-layer formula's total in whole MiB (rounded up).
func kvLayerTotalMB(model *ModelProfile, shape kvCacheShape) (int, bool) {
	layers, ok := kvLayerBytes(model, shape)
	if !ok {
		return 0, false
	}
	var total int64
	for _, b := range layers {
		total += b
	}
	return bytesToMiBCeil(total), total > 0
}

// kvShapeForStrategy resolves the launch geometry of a complete strategy.
// Flash attention follows KV residency as in defaultFlashAttention. ggrun does
// not emit --kv-unified, so slots are separate streams.
func kvShapeForStrategy(s *Strategy, swaFull bool) kvCacheShape {
	if s == nil {
		return kvCacheShape{}
	}
	return kvCacheShape{
		Context: s.ContextSize, UBatch: s.UBatchSize, Slots: s.Parallel,
		SWAFull: swaFull || s.SWAFull, KVType: s.KVType,
		VTransposed: strings.EqualFold(s.KVPlacement, "cpu"),
	}
}

// computeKVTotalMBForStrategy is computeKVTotalMB with the strategy's own
// microbatch, slots and flash-attention state. A model-wide measurement stays
// the floor; the per-layer formula may only raise it, where the windowed
// cache grows with a microbatch or slot count the measurement never saw.
func computeKVTotalMBForStrategy(model *ModelProfile, s *Strategy) int {
	if s == nil {
		return 0
	}
	exact, ok := kvLayerTotalMB(model, kvShapeForStrategy(s, false))
	if !ok {
		return computeKVTotalMB(model, s.ContextSize, s.KVType, s.SWAFull)
	}
	if measured, ok := measuredKVTotalMB(model, s.ContextSize, s.KVType, s.SWAFull); ok && measured > exact {
		return measured
	}
	return exact
}

// kvLayerDeviceFractions is the share of the per-layer KV bytes owned by each
// device under a layer split: the same n_layer_all+1 slot assignment the
// backend uses for weights, applied to the blocks that actually hold cache.
func kvLayerDeviceFractions(model *ModelProfile, shape kvCacheShape, split []float64) ([]float64, bool) {
	layers, ok := kvLayerBytes(model, shape)
	if !ok || len(split) == 0 {
		return nil, false
	}
	devices, _ := layerDeviceAssignments(split, len(layers))
	out := make([]float64, len(split))
	var total int64
	for il, b := range layers {
		if dev := devices[il]; b > 0 && dev >= 0 && dev < len(out) {
			out[dev] += float64(b)
			total += b
		}
	}
	if total <= 0 {
		return nil, false
	}
	for i := range out {
		out[i] /= float64(total)
	}
	return out, true
}

// kvLayerShareMB charges device idx the KV its own layers hold. Without a
// validated per-layer layout it keeps the uniform owned-layer share.
func kvLayerShareMB(model *ModelProfile, shape kvCacheShape, totalMB int, split []float64, owned []int, idx int) int {
	if totalMB <= 0 || idx < 0 {
		return 0
	}
	if fractions, ok := kvLayerDeviceFractions(model, shape, split); ok {
		if idx >= len(fractions) || fractions[idx] <= 0 {
			return 0
		}
		return int(math.Ceil(float64(totalMB) * fractions[idx]))
	}
	numLayers := 0
	if model != nil {
		numLayers = model.NumLayers
	}
	return ownedShareMB(totalMB, owned, numLayers, idx)
}

// SWAFullKVMultiplier is how many times more KV per token --swa-full costs
// than the windowed cache at a reference context (131,072 or the trained
// context if smaller). ok is false when either size is unknown.
func SWAFullKVMultiplier(model *ModelProfile) (float64, bool) {
	if model == nil {
		return 0, false
	}
	ctx := 131072
	if model.CTXTrain > 0 && model.CTXTrain < ctx {
		ctx = model.CTXTrain
	}
	full, windowed := EstimateKVCacheMB(model, ctx, "q8_0", true), EstimateKVCacheMB(model, ctx, "q8_0", false)
	if full <= 0 || windowed <= 0 {
		return 0, false
	}
	return float64(full) / float64(windowed), true
}

// BackendSupportsCheckpointReuse reports whether the backend's help advertises
// context checkpoints with a spacing control, which is how prefix reuse works
// on sliding-window models without a full-size window cache.
func BackendSupportsCheckpointReuse(help string) bool {
	return backendHelpSupports(help, "--ctx-checkpoints") && backendCheckpointMinStepFlag(help, "") != "-"
}
