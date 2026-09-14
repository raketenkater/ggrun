package main

import (
	"errors"
	"fmt"
	"testing"
)

// The failure budget exists to bound expensive work — a candidate that read the
// whole model and then died. Charging an argv-time refusal the same retires the
// entire search after a few cheap rejections inside one lever family, which is
// how Qwen3.8-Flash-Next spent its budget on three ubatch rungs (7026, 3608 and
// 1885 MiB deficits on CUDA0) with no model load between them, leaving expert
// packing, topology and slot count unmeasured.
func TestPreLoadRefusalsAreNotChargedAsReloadFailures(t *testing.T) {
	cheap := []exactAdmissionClass{
		exactAdmissionSpec,
		exactAdmissionCompat,
		exactAdmissionCompanion,
		exactAdmissionMemory,
		exactAdmissionMMap,
	}
	for _, class := range cheap {
		err := exactAdmissionError(class, " on CUDA0 (1885 MiB deficit)", nil)
		if exactAdmissionLoadedWeights(err) {
			t.Errorf("%s is refused before any weight is read, but was charged as a reload failure", class)
		}
		// It must still be usable as negative evidence; only the accounting changed.
		if !isStableExactAdmissionFailure(err) {
			t.Errorf("%s stopped being a stable admission failure", class)
		}
	}

	// A CUDA OOM surfaces while device allocations are being made, so it did
	// cost a load and must keep consuming the budget.
	oom := exactAdmissionError(exactAdmissionCUDAOOM, " on CUDA1", nil)
	if !exactAdmissionLoadedWeights(oom) {
		t.Error("a CUDA OOM was treated as costing no model load")
	}
}

// An untyped error is an ordinary start failure — a health timeout, an
// interrupted load, a transient backend fault. By then a load was usually under
// way, so the expensive classification is the safe default: mistaking a real
// reload for a cheap refusal is what would make the search unbounded.
func TestUnknownStartFailuresStayExpensive(t *testing.T) {
	for name, err := range map[string]error{
		"plain":   errors.New("health check timed out"),
		"wrapped": fmt.Errorf("starting candidate: %w", errors.New("backend exited")),
		"nil":     nil,
	} {
		if !exactAdmissionLoadedWeights(err) {
			t.Errorf("%s error was treated as costing no model load", name)
		}
	}
}

// The classification must survive wrapping, since admission errors travel up
// through several layers before the calibration loop reads them.
func TestAdmissionCostClassificationSurvivesWrapping(t *testing.T) {
	wrapped := fmt.Errorf("candidate ubatch-512: %w",
		exactAdmissionError(exactAdmissionMemory, " on CUDA0 (1885 MiB deficit)", nil))
	if exactAdmissionLoadedWeights(wrapped) {
		t.Error("a wrapped pre-load memory refusal was charged as a reload failure")
	}

	wrappedOOM := fmt.Errorf("candidate parallel-4: %w",
		exactAdmissionError(exactAdmissionCUDAOOM, " on CUDA2", nil))
	if !exactAdmissionLoadedWeights(wrappedOOM) {
		t.Error("a wrapped CUDA OOM stopped consuming the budget")
	}
}
