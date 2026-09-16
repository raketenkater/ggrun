package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

// Minimal fixtures: placementOptionsFromRequest reads req.CtxFlag and
// model.CTXTrain, so these only need to be non-nil and coherent. "fit" is the
// automatic context path, which is the one that failed to converge.
func probeTestRequest() *launchRequest { return &launchRequest{CtxFlag: "fit"} }

func probeTestModel() *placement.ModelProfile {
	return &placement.ModelProfile{NumLayers: 32, HeadCountKV: 8, KeyLength: 128, ValueLength: 128, CTXTrain: 1048576}
}

func probeTestBackend() *backendInfo { return &backendInfo{Tag: "llama"} }

// The memory-probe command previews the launch plan, so it has to converge the
// same way a launch does. It used to pass nil where the launch path passes its
// recovery ledger, which left every ratchet inert: observed on GLM-5.3-Flash
// 2026-09-15, `memory-probe -ctx fit` cycled n-cpu-moe 40 <-> 41 at a constant
// 500,736-token context and exhausted all six attempts on a configuration the
// launch path plans successfully.
//
// These tests pin the property that was missing rather than the numbers.

func TestNilLedgerBoundsNothing(t *testing.T) {
	// This is the defect, stated as a test: with no ledger there is nothing to
	// stop a recompute proposing a shape already disproved. It documents why the
	// probe loop may not pass nil.
	opts := placement.Options{AutoContextMax: 1_000_000, UBatchSize: 512}
	got := boundByProvenLimits(opts, nil)
	if got.AutoContextMax != 1_000_000 {
		t.Fatalf("nil ledger changed AutoContextMax: got %d, want %d", got.AutoContextMax, 1_000_000)
	}
	if got.UBatchSize != 512 {
		t.Fatalf("nil ledger changed UBatchSize: got %d, want %d", got.UBatchSize, 512)
	}
}

func TestProbeRecomputeIsBoundedByADisprovedContext(t *testing.T) {
	recovery := newLaunchMemoryRecovery()
	recovery.rejectContext(&placement.Strategy{ContextSize: 500736, ContextAuto: true}, 0)

	opts := memoryProbeRecomputeOptions(probeTestRequest(), probeTestModel(), probeTestBackend(), "", recovery)
	if opts.AutoContextMax <= 0 {
		t.Fatalf("probe recompute left the automatic context unbounded after a rejection: got %d", opts.AutoContextMax)
	}
	if opts.AutoContextMax >= 500736 {
		t.Fatalf("probe recompute may re-propose a disproved context: ceiling %d, rejected %d", opts.AutoContextMax, 500736)
	}
}

func TestProbeRecomputeCannotRaiseAProvenUBatch(t *testing.T) {
	recovery := newLaunchMemoryRecovery()
	// An exact preflight fitted at ubatch 128. The compute buffer at 256 is a
	// different, larger measurement, so recomputing back up is what makes the
	// loop cycle.
	recovery.acceptContext(&placement.Strategy{UBatchSize: 128, ContextSize: 262144, ContextAuto: true})

	opts := memoryProbeRecomputeOptions(probeTestRequest(), probeTestModel(), probeTestBackend(), "", recovery)
	if opts.UBatchSize != 128 {
		t.Fatalf("probe recompute did not pin the proven ubatch: got %d, want 128", opts.UBatchSize)
	}
}

func TestProbeRecomputeKeepsTheTighterOfTwoBounds(t *testing.T) {
	recovery := newLaunchMemoryRecovery()
	recovery.rejectContext(&placement.Strategy{ContextSize: 500736, ContextAuto: true}, 0)
	recovery.acceptContext(&placement.Strategy{UBatchSize: 128, ContextSize: 262144, ContextAuto: true})

	opts := memoryProbeRecomputeOptions(probeTestRequest(), probeTestModel(), probeTestBackend(), "", recovery)
	// The accepted context is the fixed point the loop is trying to reach, so it
	// wins over the rejected ceiling when it is lower.
	if opts.AutoContextMax != 262144 {
		t.Fatalf("ceiling did not take the accepted context: got %d, want 262144", opts.AutoContextMax)
	}
	if opts.UBatchSize != 128 {
		t.Fatalf("ubatch pin lost: got %d, want 128", opts.UBatchSize)
	}
}

func TestProbeRecomputeIsUnboundedBeforeAnyEvidence(t *testing.T) {
	// A fresh ledger must not invent bounds: the first attempt has to be free to
	// propose the user's requested shape, or the probe would preview something
	// the launch would never choose.
	recovery := newLaunchMemoryRecovery()
	opts := memoryProbeRecomputeOptions(probeTestRequest(), probeTestModel(), probeTestBackend(), "", recovery)
	if opts.AutoContextMax != 0 {
		t.Fatalf("fresh ledger bounded the context: got %d, want 0", opts.AutoContextMax)
	}
	if opts.UBatchSize != 0 {
		t.Fatalf("fresh ledger pinned a ubatch: got %d, want 0", opts.UBatchSize)
	}
}
