package placement

import "fmt"

// Batch size as a placement variable.
//
// -b is treated as a throughput preset, but it is a placement decision: the
// backend's compute buffer scales with it, and that buffer competes for the same
// VRAM as resident expert layers. On the 2026-09-08 GLM 5.3 Flash serve the
// compute buffers were 14,825 MiB across three devices -- 30% of all VRAM on the
// rig, and 3.3 routed-expert layers' worth of space -- while the placement left
// 40 expert layers on the host. ggrun never considered that trade because it
// could not see the buffer at all (fixed: occupancy now counts it).
//
// The trade has a direction, and the direction is not obvious:
//
//	larger batch  -> faster prefill, fewer resident expert layers
//	smaller batch -> slower prefill, more resident expert layers -> faster decode
//
// Which side wins depends entirely on how the workload spends its time, which
// is why this consumes the measured phase mix rather than a preference. On this
// rig agent traffic is prefill-dominant (~80% of processed time), so the move
// that looks obvious -- shrink the batch, reclaim VRAM, seat more experts --
// sells the larger half to buy the smaller one. On a rig whose experts are
// already GPU-resident the same arithmetic points the other way.
//
// Following AnalyzeDeviceBalance: this derives a typed signal from measured
// evidence and nothing more. It does not prove a different batch is faster. It
// makes at most one batch challenger worth contained admission and a live
// workload comparison (invariant 7).

// BatchDirection is which way a challenger should move, if any.
type BatchDirection int

const (
	BatchHold BatchDirection = iota
	BatchRaise
	BatchLower
)

func (d BatchDirection) String() string {
	switch d {
	case BatchRaise:
		return "raise"
	case BatchLower:
		return "lower"
	default:
		return "hold"
	}
}

// BatchSizeSignal is measured evidence that the current batch point sits on the
// wrong side of the prefill/decode trade for this workload.
type BatchSizeSignal struct {
	Observed  bool
	Direction BatchDirection
	// PrefillShare is the measured fraction of processed time spent prefilling.
	PrefillShare float64
	// ComputeBufMB is what the current batch point costs across all devices.
	ComputeBufMB int
	// ExpertLayerMB is the unit that VRAM would be spent on instead.
	ExpertLayerMB int
	// LayersWorth is how many resident expert layers the compute buffer
	// currently displaces. This is the size of the trade, not a prediction.
	LayersWorth int
	Reason      string
}

// Summary renders the signal for a launch record.
func (s BatchSizeSignal) Summary() string {
	if !s.Observed {
		return "batch trade unmeasured"
	}
	return fmt.Sprintf("batch %s: prefill %.0f%% of processed time, compute buffer %d MiB (%d expert layer(s) worth); %s",
		s.Direction, s.PrefillShare*100, s.ComputeBufMB, s.LayersWorth, s.Reason)
}

// batchTradeMinShare is how lopsided the phase mix must be before moving the
// batch is worth a challenger at all. It is a confidence threshold on evidence,
// not a claim about hardware: near an even split the trade is close enough that
// a bounded challenger is unlikely to separate from noise, and the launch is
// better left alone.
const batchTradeMinShare = 0.65

// AnalyzeBatchSize derives the batch signal from the measured phase mix and the
// measured cost of the current batch point.
//
// Fails closed in every direction it cannot substantiate: an unmeasured mix, an
// unmeasured compute buffer, or an unknown expert-layer size all yield no
// signal. An even phase mix also yields no signal, because there is nothing to
// gain by trading one phase for the other.
func AnalyzeBatchSize(mix AgentPhaseTiming, seats ExpertSeatReport, computeByGPU map[int]int) BatchSizeSignal {
	signal := BatchSizeSignal{}
	if !mix.Measured() || seats.ExpertLayerMB <= 0 || len(computeByGPU) == 0 {
		return signal
	}
	computeMB := 0
	for _, v := range computeByGPU {
		if v > 0 {
			computeMB += v
		}
	}
	if computeMB <= 0 {
		return signal
	}
	signal.PrefillShare = mix.PrefillTimeShare()
	signal.ComputeBufMB = computeMB
	signal.ExpertLayerMB = seats.ExpertLayerMB
	signal.LayersWorth = computeMB / seats.ExpertLayerMB

	switch {
	case signal.PrefillShare >= batchTradeMinShare:
		// Prefill dominates. Reclaiming compute-buffer VRAM for expert layers
		// would buy decode at prefill's expense, which is the losing side here.
		// The challenger worth testing is the other one, and only where the
		// current buffer is not already displacing resident layers this
		// workload would rather have.
		signal.Observed = true
		signal.Direction = BatchRaise
		signal.Reason = "prefill dominates, so buffer VRAM spent on batch is better spent than on residency"
	case signal.PrefillShare <= 1-batchTradeMinShare:
		// Decode dominates and the buffer is displacing whole expert layers.
		// Only then is shrinking the batch a trade worth measuring.
		if signal.LayersWorth < 1 {
			signal.Reason = "decode dominates but the compute buffer is smaller than one expert layer; nothing to reclaim"
			return signal
		}
		signal.Observed = true
		signal.Direction = BatchLower
		signal.Reason = "decode dominates and the compute buffer displaces whole expert layers"
	default:
		signal.Reason = "phase mix too even to justify trading one phase for the other"
	}
	return signal
}

// BatchChallenger returns the single bounded batch point worth admitting, or 0
// when the signal does not support one.
//
// One challenger, one halving or doubling. A search over batch points would be
// a series of long reloads on an ordinary launch, which invariant 10 forbids,
// and invariant 7 allows prediction to select at most a finalist regardless.
func BatchChallenger(signal BatchSizeSignal, current int) int {
	if !signal.Observed || current <= 0 {
		return 0
	}
	switch signal.Direction {
	case BatchRaise:
		return current * 2
	case BatchLower:
		if next := current / 2; next >= 64 {
			// Below a floor the batch stops being a batch: prefill collapses
			// toward per-token cost and the reclaimed VRAM cannot pay for it.
			return next
		}
	}
	return 0
}
