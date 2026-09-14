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

	// These are returned only after startLaunchProcess has succeeded, so the
	// weights were read and the budget must still be charged. A CUDA OOM
	// surfaces during device allocation; an mmap refusal comes from
	// validateObservedMMapPageability failing on a process that then gets
	// stopped. Calling either cheap would let real loads run unbudgeted.
	for _, class := range []exactAdmissionClass{exactAdmissionCUDAOOM, exactAdmissionMMap} {
		if !exactAdmissionLoadedWeights(exactAdmissionError(class, " on CUDA1", nil)) {
			t.Errorf("%s is returned after a model load, but was treated as costing none", class)
		}
	}
}

// Deny by default. A new failure class must be classified deliberately rather
// than inheriting "cheap" from being typed — which is how the mmap class, a
// post-load refusal, was first mis-filed as an argv-time one.
func TestEveryAdmissionClassIsClassified(t *testing.T) {
	// Every class declared in main.go, with where it is returned relative to
	// startLaunchProcess.
	declared := map[exactAdmissionClass]bool{ // true == returned before any start
		exactAdmissionSpec:      true,
		exactAdmissionCompat:    true,
		exactAdmissionCompanion: true,
		exactAdmissionMemory:    true,
		exactAdmissionMMap:      false,
		exactAdmissionCUDAOOM:   false,
	}
	for class, argvTime := range declared {
		if got := !exactAdmissionLoadedWeights(exactAdmissionError(class, "", nil)); got != argvTime {
			t.Errorf("%s: classified argv-time=%v, want %v", class, got, argvTime)
		}
	}
	for class := range argvTimeAdmissionClasses {
		if _, ok := declared[class]; !ok {
			t.Errorf("%s is treated as argv-time but is not recorded here; confirm it is returned before startLaunchProcess", class)
		}
	}
	// An unrecognised class must be expensive, not cheap.
	if !exactAdmissionLoadedWeights(exactAdmissionError(exactAdmissionClass("future-class"), "", nil)) {
		t.Error("an unclassified failure class defaulted to costing no model load")
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
