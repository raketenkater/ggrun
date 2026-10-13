package placement

import (
	"fmt"
	"math"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Byte-weighted layer assignment (ik_llama.cpp).
//
// Mainline llama.cpp gives each device a share of the n_layer+1 layer slots
// (layerDeviceAssignments). ik_llama.cpp since #1466 (0871ab29, 2026-03-20)
// fills devices by bytes, and --tensor-split is a byte fraction. At the
// shipped backend (1fddd12, src/llama.cpp):
//
//   - get_layer_sizes (3573-3799) sizes block il as every blk.il tensor,
//     routed experts included even when -ot keeps them on the CPU (NextN
//     blocks count 0 unless MTP decoding loads them), plus
//     llama_model::cache_size(il) (src/llama-model.cpp:2190): a recurrent
//     block's state times min(n_seq_max, n_ctx), otherwise K and V rows at
//     n_ctx for every block, windowed and NextN blocks included although a
//     NextN block's cache is not allocated. n_ctx, n_seq_max and n_ubatch are
//     the server's --ctx-size, --parallel and min(-b, -ub, n_ctx)
//     (common/common.cpp:4157, 4208-4210). The output slot is output.weight
//     (token_embd.weight when tied) plus n_vocab*n_ubatch*4 less max_compute.
//   - max_compute (3739-3784) is the largest per-block estimate of the FFN
//     activations plus the block input: max(routed*n_expert_used, shared,
//     dense) ne[1] sums times n_ubatch*4, plus n_embd*n_ubatch*4 (twice when
//     the block has a norm). An MLA block whose GGUF states attn_v_b adds
//     2*v_head*n_head*n_ctx*4 under the default -mla 3 and -amb 0.
//   - llm_load_tensors (3966-3990, 4057-4087) normalizes the split in float
//     and walks the blocks and, under -ngl > n_layer, the output slot in order
//     with one cumulative byte target per device, split * (sum +
//     max_compute*devices), charging max_compute once per device. A block that
//     crosses a target stays when it overshoots by less than the room it would
//     leave; the rest default to the last device.
//
// Qwen3.8-27B UD-Q5_K_S with --tensor-split 0.27,0.58,0.15 at 65,536 tokens
// (1 and 4 slots, -ub 128): the backend placed blocks 0-19/20-58/59-65 where
// the slot rule gives 0-17/18-56/57-65. An attention block holds 1.3 GiB of KV
// at 661k tokens, so one block off moved a --claude-code preflight's deficit
// between CUDA0 and CUDA1 on every re-split, and the NextN block's
// unallocated cache, counted in the backend's sum, shifts every target.
//
// Planner inputs the GGUF does not state are proxies, used only to place the
// boundaries; the bytes charged to each device keep the existing KV and weight
// pricing: an MLA GGUF that states value_length_mla is taken to carry split
// attn_k_b/attn_v_b (quantized, so the rebuilt attn_kv_b is q8_0), and a
// layout this file cannot describe spreads the model-wide KV evenly.

// ikLayerSizes mirrors get_layer_sizes: bytes per block, the output slot at
// index NumLayers, and max_compute. ok is false without per-block tensor
// bytes, a vocabulary or a context.
func ikLayerSizes(model *ModelProfile, shape kvCacheShape) ([]float64, float64, bool) {
	if model == nil || model.NumLayers <= 0 || shape.Context <= 0 || model.EmbeddingLength <= 0 ||
		model.VocabSize <= 0 || len(model.NonExpertLayerBytes) < model.NumLayers ||
		(model.ExpertBytes > 0 && len(model.ExpertLayerBytes) == 0) {
		return nil, 0, false
	}
	n := model.NumLayers
	cache := ikLayerCacheBytes(model, shape)
	ub := int64(shape.UBatch)
	if ub <= 0 {
		ub = backendDefaultUBatch
	}
	// get_batch_ubatch caps the logical batch, and so the microbatch, at n_ctx.
	ub = min(ub, int64(shape.Context))
	inp := float64(model.EmbeddingLength) * float64(ub) * 4
	nextn := 0
	if model.NextNPredictLayers > 0 && model.NextNPredictLayers < n {
		nextn = model.NextNPredictLayers
	}
	mlaWeights, mlaCompute := ikMLATerms(model, shape, ub)
	sizes := make([]float64, n+1)
	maxCompute := 0.0
	for il := 0; il < n; il++ {
		compute := inp
		if il < n-nextn || shape.MTP {
			sizes[il] = float64(ikBlockWeightBytes(model, il, true)) + mlaWeights
			compute = max(ikFFNComputeBytes(model, il, ub)+2*inp, mlaCompute)
		}
		maxCompute = max(maxCompute, compute)
		sizes[il] += float64(cache[il])
	}
	output := model.OutputBytes
	if output <= 0 {
		output = model.TokenEmbdBytes
	}
	sizes[n] = float64(output) + max(0, float64(model.VocabSize)*float64(ub)*4-maxCompute)
	return sizes, maxCompute, true
}

// ikBlockWeightBytes is every tensor of block il; routed experts only when
// requested.
func ikBlockWeightBytes(model *ModelProfile, il int, routed bool) int64 {
	b := layerByteValue(model.NonExpertLayerBytes, il) + layerByteValue(model.ExpertAuxLayerBytes, il)
	if routed {
		return b + layerByteValue(model.ExpertLayerBytes, il)
	}
	return b + layerByteValue(model.ShexpLayerBytes, il)
}

func layerByteValue(values []int64, il int) int64 {
	if il < 0 || il >= len(values) || values[il] < 0 {
		return 0
	}
	return values[il]
}

// ikLayerCacheBytes mirrors llama_model::cache_size for every block at n_ctx
// and n_seq_max.
func ikLayerCacheBytes(model *ModelProfile, shape kvCacheShape) []int64 {
	n := model.NumLayers
	ctx := int64(shape.Context)
	slots := min(int64(max(1, shape.Slots)), ctx)
	out := make([]int64, n)
	rows := func(kElems, vElems int) int64 {
		k, okK := kvRowBytes(shape.KVType, kElems)
		v, okV := kvRowBytes(shape.KVType, vElems)
		if !okK || !okV {
			return 0
		}
		return (k + v) * ctx
	}
	arch := strings.ToLower(model.ModelArch)
	if layout, ok := modelKVLayerLayout(model); ok {
		var state int64
		for _, layer := range layout.Layers {
			state = max(state, layer.RecurrentF32)
		}
		heads, keyLen, valueLen := model.HeadCountKV, model.KeyLength, model.ValueLength
		for il, layer := range layout.Layers {
			switch {
			case layer.RecurrentF32 > 0:
				out[il] = layer.RecurrentF32 * 4 * slots
			case layer.KWidth > 0:
				out[il] = rows(layer.KWidth, layer.VWidth)
			case state > 0 && arch == "qwen3next" && (il+1)%4 != 0:
				// qwen3next marks every block by the 4-block rule, NextN included.
				out[il] = state * 4 * slots
			case state > 0:
				// qwen35/qwen35moe NextN blocks use standard attention.
				out[il] = rows(heads*keyLen, heads*valueLen)
			}
		}
		return out
	}
	if model.KVLoraRank > 0 && (ikMLAArch(arch) || arch == "openpangu") {
		// MLA latent cache (flash attention, or -mla 3 without it): one row of
		// kv_lora_rank + n_rot per cell. OpenPangu's indexer keys are not modelled.
		row, ok := kvRowBytes(shape.KVType, model.KVLoraRank+model.RopeDim)
		if ok {
			for il := range out {
				out[il] = row * ctx
			}
		}
		return out
	}
	heads := model.HeadCountKV
	if heads <= 0 {
		heads = model.HeadCount
	}
	keyLen, valueLen := model.KeyLength, model.ValueLength
	if keyLen <= 0 && model.EmbeddingLength > 0 && model.HeadCount > 0 {
		keyLen = model.EmbeddingLength / model.HeadCount
	}
	if valueLen <= 0 {
		valueLen = keyLen
	}
	if hasSSMLayout(model) || heads <= 0 || keyLen <= 0 {
		// Recurrent or compressed layouts this file cannot describe: spread the
		// model-wide cache evenly over the blocks.
		per := int64(computeKVTotalMB(model, shape.Context, shape.KVType, true)) * 1048576 / int64(n)
		for il := range out {
			out[il] = per
		}
		return out
	}
	per := rows(heads*keyLen, heads*valueLen)
	for il := range out {
		out[il] = per
	}
	return out
}

func ikMLAArch(arch string) bool {
	return arch == "deepseek2" || arch == "glm-dsa" || arch == "mistral4"
}

// ikMLATerms is the per-block weight added for the attn_kv_b ik_llama
// rebuilds from attn_k_b/attn_v_b, and the -mla 3 attention compute; zero
// unless the GGUF states value_length_mla.
func ikMLATerms(model *ModelProfile, shape kvCacheShape, ub int64) (weights, compute float64) {
	if !ikMLAArch(strings.ToLower(model.ModelArch)) || model.KVLoraRank <= 0 || model.ValueLengthMLA <= 0 || model.HeadCount <= 0 {
		return 0, 0
	}
	row, ok := kvRowBytes("q8_0", model.KVLoraRank)
	if !ok {
		return 0, 0
	}
	heads := float64(model.HeadCount)
	weights = float64(row) * float64(model.ValueLengthMLA+model.KVLoraRank) * heads * 2
	compute = 2 * float64(model.ValueLengthMLA) * heads * float64(shape.Context) * 4
	return weights, compute
}

// ikFFNComputeBytes is max(routed, shared, dense) of the FFN ne[1] sums times
// n_ubatch*4: gate and up are ne[1] = ff, down is ne[1] = n_embd.
func ikFFNComputeBytes(model *ModelProfile, il int, ub int64) float64 {
	embd := float64(model.EmbeddingLength)
	perToken := float64(ub) * 4
	moe := layerByteValue(model.RoutedExpertLayerBytes, il) > 0
	if len(model.RoutedExpertLayerBytes) == 0 {
		moe = model.IsMoE && il >= model.LeadingDense
	}
	if !moe {
		if model.FeedForwardLength <= 0 {
			return 0
		}
		return (2*float64(model.FeedForwardLength) + embd) * perToken
	}
	ff := model.ExpertFF
	if ff <= 0 {
		ff = model.FeedForwardLength
	}
	routed := (2*float64(ff) + embd) * perToken * float64(max(1, model.ExpertUsedCount))
	shared := 0.0
	if model.ExpertSharedFF > 0 && (layerByteValue(model.ShexpLayerBytes, il) > 0 || model.HasShexp > 0) {
		shared = (2*float64(model.ExpertSharedFF) + embd) * perToken
	}
	return max(routed, shared)
}

// ikLayerDeviceAssignments mirrors the walk in llm_load_tensors, including
// its float arithmetic: --tensor-split values are parsed with stof and summed
// in float, the running total is a float, sizes and the overall sum are
// doubles. ggrun always emits -ngl 999 beside a split, so the walk starts at
// block 0 and includes the output slot.
func ikLayerDeviceAssignments(sizes []float64, maxCompute float64, split []float64) (layerDevices []int, outputDev int) {
	n := len(sizes) - 1
	if n <= 0 {
		return nil, -1
	}
	cum := make([]float32, len(split))
	var total float32
	for i, v := range split {
		total += float32(max(0, v))
		cum[i] = total
	}
	if total <= 0 {
		return layerDeviceAssignments(split, n)
	}
	for i := range cum {
		cum[i] /= total
	}
	assigned := make([]int, n+1)
	for i := range assigned {
		assigned[i] = len(split) - 1
	}
	if len(split) > 1 {
		sum := maxCompute * float64(len(split))
		for _, s := range sizes {
			sum += s
		}
		last := 0
		var loaded float32
		for id := range split {
			loaded = float32(float64(loaded) + maxCompute)
			target := float32(float64(cum[id]) * sum)
			il := last
			for ; il <= n; il++ {
				if float64(loaded)+sizes[il] <= float64(target) {
					assigned[il] = id
					loaded = float32(float64(loaded) + sizes[il])
					continue
				}
				if float64(loaded)+sizes[il]-float64(target) < float64(target-loaded) {
					assigned[il] = id
					loaded = float32(float64(loaded) + sizes[il])
					il++
				}
				break
			}
			last = il
		}
	}
	return assigned[:n], assigned[n]
}

// backendLayerDeviceAssignments is the device of every block and of the output
// slot under the selected backend's rule: ik_llama's byte walk when the model
// carries that backend fact and the tensor bytes it needs, otherwise the slot
// rule.
func backendLayerDeviceAssignments(model *ModelProfile, shape kvCacheShape, split []float64) ([]int, int) {
	numLayers := 0
	if model != nil {
		numLayers = model.NumLayers
		if model.ByteWeightedLayerSplit && len(split) > 1 {
			if sizes, maxCompute, ok := ikLayerSizes(model, shape); ok {
				return ikLayerDeviceAssignments(sizes, maxCompute, split)
			}
		}
	}
	return layerDeviceAssignments(split, numLayers)
}

// backendLayerOwnership is layerOwnership under the backend's rule.
func backendLayerOwnership(model *ModelProfile, shape kvCacheShape, split []float64) ([]int, int) {
	devices, outputDev := backendLayerDeviceAssignments(model, shape, split)
	owned := make([]int, len(split))
	for _, dev := range devices {
		if dev >= 0 && dev < len(owned) {
			owned[dev]++
		}
	}
	return owned, outputDev
}

// ikDeviceWeightBytes is the weight each device holds under an assignment:
// its blocks (NextN only under MTP decoding, which is when ik_llama loads
// them; routed experts only when they follow the block), and the output head
// with the remaining global tensors on the output device. Token embeddings
// stay in host memory; a tied head is duplicated onto the output device.
func ikDeviceWeightBytes(model *ModelProfile, shape kvCacheShape, devices []int, outputDev, numDevices int, routed bool) []int64 {
	out := make([]int64, numDevices)
	n := model.NumLayers
	nextn := 0
	if model.NextNPredictLayers > 0 && model.NextNPredictLayers < n {
		nextn = model.NextNPredictLayers
	}
	var layerBytes int64
	for il := 0; il < n; il++ {
		layerBytes += layerByteValue(model.NonExpertLayerBytes, il)
		if il >= n-nextn && !shape.MTP {
			continue
		}
		if il < len(devices) && devices[il] >= 0 && devices[il] < numDevices {
			out[devices[il]] += ikBlockWeightBytes(model, il, routed)
		}
	}
	if outputDev >= 0 && outputDev < numDevices {
		output := model.OutputBytes
		if output <= 0 {
			output = model.TokenEmbdBytes
		}
		global := model.NonExpertBytes - model.TokenEmbdBytes - model.OutputBytes - model.ExpertAuxBytes - layerBytes
		out[outputDev] += output + max(0, global)
	}
	return out
}

// ikDeviceKVMB charges each device the KV of its own blocks under an
// assignment: the per-block formula where the layout is known, otherwise an
// owned-block share of the model-wide total.
func ikDeviceKVMB(model *ModelProfile, shape kvCacheShape, totalMB int, devices []int, numDevices int) []int {
	out := make([]int, numDevices)
	if totalMB <= 0 {
		return out
	}
	if layers, ok := kvLayerBytes(model, shape); ok {
		if exact, ok := kvLayerTotalMB(model, shape); ok && exact > totalMB {
			totalMB = exact
		}
		if fractions, ok := kvLayerFractions(layers, devices, numDevices); ok {
			for i := range out {
				if fractions[i] > 0 {
					out[i] = int(math.Ceil(float64(totalMB) * fractions[i]))
				}
			}
			return out
		}
	}
	owned := make([]int, numDevices)
	for _, dev := range devices {
		if dev >= 0 && dev < numDevices {
			owned[dev]++
		}
	}
	for i := range out {
		out[i] = ownedShareMB(totalMB, owned, model.NumLayers, i)
	}
	return out
}

// ikDenseDevice is one device of a fully resident layer split.
type ikDenseDevice struct {
	FreeMB     int
	CapacityMB int // free less CUDA context, compute buffer and projector, at least 0
	// FirmMB is CapacityMB reserving only a measured compute buffer. A split
	// that overfills CapacityMB but not FirmMB is emitted for an explicit
	// context: the cold compute estimate only ranks, and exact preflight
	// decides. For an automatic-context candidate FirmMB equals CapacityMB.
	FirmMB int
}

// ikDenseDeviceMB is the weights plus KV each device holds under a split.
func ikDenseDeviceMB(model *ModelProfile, shape kvCacheShape, kvTotalMB int, split []float64) ([]int, []int) {
	sizes, maxCompute, _ := ikLayerSizes(model, shape)
	devices, outputDev := ikLayerDeviceAssignments(sizes, maxCompute, split)
	weights := ikDeviceWeightBytes(model, shape, devices, outputDev, len(split), true)
	modelMB := make([]int, len(split))
	for i, b := range weights {
		modelMB[i] = bytesToMiBCeil(b)
	}
	return modelMB, ikDeviceKVMB(model, shape, kvTotalMB, devices, len(split))
}

// ikDenseSplit chooses a fully resident split for a byte-weighted backend.
//
// The intended split (fill-fastest-first or balanced) states how much each
// device should hold. Its fractions cannot be emitted as they are: the backend
// takes them of its own byte sum (KV at n_ctx and n_seq_max, the unallocated
// NextN cache, the output buffer, one max_compute per device) and then rounds
// each boundary to the nearest block edge, so a card filled to capacity
// receives up to half a block more. Walking the devices in order, this picks
// the 0.01-grid fraction whose backend boundary brings the cumulative real
// bytes closest to the intended cumulative share without exceeding the
// device's capacity, centred among the grid points that give the same
// boundary so a small sizing error cannot move it. The last used device takes
// the rest. When that overfills a later device, the walk is repeated filling
// every earlier device as far as its capacity allows, then over every card
// with room. When no split fits, the same steps run against the firm
// capacity, which reserves only measured compute buffers. ok is false when the
// backend's sizes are unavailable; err reports a device whose weights and KV
// still exceed its firm capacity. A device that holds nothing is never over.
func ikDenseSplit(model *ModelProfile, shape kvCacheShape, kvTotalMB int, intent []float64, devices []ikDenseDevice) ([]float64, bool, error) {
	if _, _, ok := ikLayerSizes(model, shape); !ok || len(intent) < 2 || len(devices) != len(intent) {
		return intent, false, nil
	}
	active := make([]int, 0, len(intent))
	intentTotal := 0.0
	for i, v := range intent {
		if v > 0 {
			active = append(active, i)
			intentTotal += v
		}
	}
	if len(active) == 0 {
		return intent, false, nil
	}
	allMB := func(split []float64) []int {
		modelMB, kvMB := ikDenseDeviceMB(model, shape, kvTotalMB, split)
		for i := range modelMB {
			modelMB[i] += kvMB[i]
		}
		return modelMB
	}
	// The real bytes the plan places, distributed by the intended shares.
	realTotal := 0
	for _, mb := range allMB(intent) {
		realTotal += mb
	}

	capacity := func(dev int, firm bool) int {
		if firm {
			return devices[dev].FirmMB
		}
		return devices[dev].CapacityMB
	}
	choose := func(active []int, fill, firm bool) []float64 {
		split := make([]float64, len(intent))
		prev := 0 // cumulative grid points (hundredths) assigned so far
		cumIntent := 0.0
		for k, dev := range active[:len(active)-1] {
			cumIntent += intent[dev] / intentTotal
			target := cumIntent * float64(realTotal)
			candidate := func(points int) []float64 {
				c := append([]float64(nil), split...)
				c[dev] = float64(points-prev) / 100
				c[active[len(active)-1]] = float64(100-points) / 100
				return c
			}
			// Each grid point maps to one boundary; keep the points of the best
			// boundary and emit their middle.
			bestFirst, bestLast, bestErr := -1, -1, math.Inf(1)
			bestUsed := -1
			for points := prev; points <= 100; points++ {
				mb := allMB(candidate(points))
				if mb[dev] > capacity(dev, firm) {
					break
				}
				used := 0
				for _, d := range active[:k+1] {
					used += mb[d]
				}
				if used == bestUsed {
					bestLast = points
					continue
				}
				err := math.Abs(float64(used) - target)
				if fill {
					err = -float64(used)
				}
				if err < bestErr {
					bestFirst, bestLast, bestErr, bestUsed = points, points, err, used
				}
			}
			if bestFirst < 0 {
				bestFirst, bestLast = prev, prev
			}
			points := (bestFirst + bestLast + 1) / 2
			if bestFirst == prev {
				points = bestFirst + (bestLast-bestFirst)/2
			}
			split[dev] = float64(points-prev) / 100
			prev = points
		}
		split[active[len(active)-1]] = float64(100-prev) / 100
		for i := range split {
			split[i] = math.Round(split[i]*100) / 100
		}
		return split
	}
	over := func(split []float64, firm bool) (int, int, int) {
		mb := allMB(split)
		for i := range devices {
			if c := capacity(i, firm); mb[i] > 0 && mb[i] > c {
				return i, mb[i], c
			}
		}
		return -1, 0, 0
	}
	fit := func(firm bool) []float64 {
		split := choose(active, false, firm)
		if dev, _, _ := over(split, firm); dev >= 0 {
			split = choose(active, true, firm)
		}
		if dev, _, _ := over(split, firm); dev >= 0 {
			// Block edges can leave a card the intent did not need short of the
			// last block; fill every card that has room, in device order.
			var all []int
			for i := range devices {
				if capacity(i, firm) > 0 {
					all = append(all, i)
				}
			}
			if len(all) > len(active) {
				split = choose(all, true, firm)
			}
		}
		return split
	}
	split := fit(false)
	if dev, _, _ := over(split, false); dev >= 0 {
		// The cold compute estimate only ranks: when no split leaves it free,
		// fit the bytes against the measured reserves alone.
		split = fit(true)
	}
	if dev, mb, capacity := over(split, true); dev >= 0 {
		return split, true, fmt.Errorf("byte-weighted layer split leaves GPU ordinal %d %d MiB over its %d MiB capacity (%d MiB of weights and KV)",
			dev, mb-capacity, capacity, mb)
	}
	return split, true, nil
}

// ikDenseDevices prices the capacity of every card for ikDenseSplit the way
// the dense split functions do: free VRAM less the CUDA context, the device's
// compute buffer, and the projector on the first card of the fill order,
// clamped at 0 as denseFillFastestFirstSplit clamps it. measuredComputeForGPU
// is the compute buffer a probe measured for the card, 0 when none did.
func ikDenseDevices(s *Strategy, caps *detect.Capabilities, cudaOverheadMB int, computeBufForGPU, measuredComputeForGPU func(int) int) []ikDenseDevice {
	order := orderGPUsByBandwidth(caps.GPUs)
	if denseMemBandwidthKnown(caps.GPUs) {
		order = orderGPUsByMemoryBandwidth(caps.GPUs)
	}
	out := make([]ikDenseDevice, len(caps.GPUs))
	for i, g := range caps.GPUs {
		base := g.VRAMFreeMB() - cudaOverheadMB
		if len(order) > 0 && i == order[0] && s.MMProjSizeMB > 0 {
			base -= s.MMProjSizeMB
		}
		out[i].FreeMB = g.VRAMFreeMB()
		out[i].CapacityMB = max(0, base-computeBufForGPU(g.Index))
		out[i].FirmMB = max(0, base-measuredComputeForGPU(g.Index))
	}
	return out
}
