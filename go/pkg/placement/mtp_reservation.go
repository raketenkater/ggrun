package placement

import (
	"fmt"
	"strings"
)

// EmbeddedMTPExtra is the device memory an embedded MTP head adds to the
// target beyond its own KV context and graph, which the backend's no-alloc
// oracle cannot price: llama-fit-params has no --spec-type, runs without
// load_mtp and so skips the NextN blocks (llama.cpp common/fit.cpp), while
// llama-server loads them and widens the recurrent state for rollback. All
// values are MiB, rounded up.
//
// Measured on a 64+1-block delta-net hybrid at ctx 32,768, -b 8192, one slot,
// draft n_max 2 (CUDA, MTP on against off): model buffer +334.75 MiB (the
// NextN block), recurrent state 448.88 against 149.62 MiB (1 + 2 snapshots)
// and a 160.02 MiB h_nextn buffer. These terms reproduce those numbers from
// the GGUF and the plan.
type EmbeddedMTPExtra struct {
	// NextNWeightsMB is the tensor bytes of the last NextNPredictLayers
	// blocks.
	NextNWeightsMB int
	// RSSnapshotMB is n_rs_seq extra recurrent-state rows per slot: llama.cpp
	// sizes the recurrent memory as mem_size * (1 + n_rs_seq), n_rs_seq being
	// the draft n_max (common_params_speculative::need_n_rs_seq), and clamps it
	// to 0 for architectures without rollback.
	RSSnapshotMB int
	// HNextnMB is the f32 n_embd x (n_batch + n_seq_max) hand-off tensor the
	// MTP context keeps on the output device (src/llama-context.cpp).
	HNextnMB int
}

// rsRollbackArchs mirrors llm_arch_supports_rs_rollback (llama.cpp
// src/llama-arch.cpp): only these widen the recurrent state for speculation.
var rsRollbackArchs = map[string]bool{
	"kimi-k3": true, "kimi-k31": true, "qwen35": true, "qwen35moe": true, "clover_moe": true,
	"qwen4exp": true, "deepseek4": true, "deepseek41": true, "glm5-next": true,
	"nemotron_h": true, "nemotron_h_moe": true, "lfm2": true, "lfm2moe": true, "lfm3": true,
	"bailingmoe3": true,
}

// mainlineDraftNMax is llama.cpp's draft n_max when no --spec-draft-n-max is
// passed (common/common.h, common_params_speculative_draft).
const mainlineDraftNMax = 3

// mainlineBatch is llama.cpp's default n_batch.
const mainlineBatch = 2048

// EmbeddedMTPExtraReservation prices the three MTP terms the oracle misses.
// Any term it cannot derive from GGUF metadata is an error, never a guessed
// zero: the caller then treats embedded MTP as unproven.
func EmbeddedMTPExtraReservation(model *ModelProfile, s *Strategy) (EmbeddedMTPExtra, error) {
	var out EmbeddedMTPExtra
	if model == nil || s == nil || s.Draft == nil {
		return out, fmt.Errorf("embedded MTP terms need the model and its MTP plan")
	}
	n, nextn := model.NumLayers, model.NextNPredictLayers
	if nextn <= 0 || nextn >= n {
		return out, fmt.Errorf("the GGUF reports %d NextN blocks of %d", nextn, n)
	}
	if len(model.NonExpertLayerBytes) < n || (model.IsMoE && len(model.ExpertLayerBytes) < n) {
		return out, fmt.Errorf("the GGUF tensor table has no per-block bytes for the NextN blocks")
	}
	var weights int64
	for il := n - nextn; il < n; il++ {
		// Shared and routed expert bytes are subsets of ExpertLayerBytes;
		// routing tensors and the sparse lookup tier are kept apart.
		weights += layerByteValue(model.NonExpertLayerBytes, il) + layerByteValue(model.ExpertAuxLayerBytes, il) +
			layerByteValue(model.ExpertLayerBytes, il) + layerByteValue(model.SparseLookupLayerBytes, il)
	}
	if weights <= 0 {
		return out, fmt.Errorf("the GGUF tensor table has no bytes for the NextN blocks")
	}
	out.NextNWeightsMB = bytesToMiBCeil(weights)

	slots := max(1, s.Parallel)
	if rsRollbackArchs[strings.ToLower(model.ModelArch)] && (hasSSMLayout(model) || model.RecurrentState != 0) {
		perSlot := recurrentStateBytesPerSlot(model)
		if perSlot <= 0 {
			return out, fmt.Errorf("the recurrent state of %s cannot be sized from GGUF metadata", model.ModelArch)
		}
		nRS := s.Draft.DraftMax
		if nRS <= 0 {
			nRS = mainlineDraftNMax
		}
		out.RSSnapshotMB = bytesToMiBCeil(perSlot * int64(nRS) * int64(slots))
	}

	if model.EmbeddingLength <= 0 {
		return out, fmt.Errorf("the GGUF reports no embedding length")
	}
	batch := s.BatchSize
	if batch <= 0 {
		batch = mainlineBatch
	}
	if s.ContextSize > 0 && s.ContextSize < batch {
		batch = s.ContextSize // llama.cpp caps n_batch at n_ctx for causal models
	}
	out.HNextnMB = bytesToMiBCeil(4 * int64(model.EmbeddingLength) * int64(batch+slots))
	return out, nil
}

// recurrentStateBytesPerSlot is recurrentStateMiBPerSlot in bytes; 0 when
// unknown.
func recurrentStateBytesPerSlot(model *ModelProfile) int64 {
	layout, ok := deltaNetHybridLayout(model)
	if !ok {
		return 0
	}
	var f32 int64
	for _, layer := range layout.Layers {
		f32 += layer.RecurrentF32
	}
	return f32 * 4
}
