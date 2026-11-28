package placement

import "fmt"

// EmbeddedMTPExtra is the device memory an embedded MTP head adds to the
// target beyond its own KV context and graph, which the backend's no-alloc
// oracle cannot price: llama-fit-params has no --spec-type, runs without
// load_mtp and so skips the NextN blocks (llama.cpp common/fit.cpp), while
// llama-server loads them and widens the recurrent state for rollback. All
// values are MiB, rounded up.
//
// On a delta-net hybrid with one NextN block (ctx 32,768, -b 8192, one slot,
// draft n_max 2; MTP on against off) the backend logged exactly these three
// increases; the terms reproduce them from the GGUF and the plan.
type EmbeddedMTPExtra struct {
	// NextNWeightsMB is the tensor bytes of the last NextNPredictLayers
	// blocks.
	NextNWeightsMB int
	// RSSnapshotMB is n_rs_seq extra recurrent-state rows per slot: llama.cpp
	// sizes the recurrent memory as mem_size * (1 + n_rs_seq), n_rs_seq being
	// the draft n_max (common_params_speculative::need_n_rs_seq). The backend
	// clamps it to 0 for architectures without rollback, but which those are
	// depends on the build, so every model with recurrent state is charged:
	// an over-reserve on a build that clamps, never an under-reserve.
	RSSnapshotMB int
	// HNextnMB is the f32 n_embd x (n_batch + n_seq_max) hand-off tensor the
	// MTP context keeps on the output device (src/llama-context.cpp).
	HNextnMB int
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
	if hasSSMLayout(model) || model.RecurrentState != 0 {
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

	// llama.cpp sizes h_nextn by n_embd_out, which is n_embd unless the GGUF
	// sets a separate output width; ggrun reads only n_embd.
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
