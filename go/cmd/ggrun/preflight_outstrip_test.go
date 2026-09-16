package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

func outstripModel() *placement.ModelProfile {
	return &placement.ModelProfile{NumLayers: 48, HeadCountKV: 8, KeyLength: 128, ValueLength: 128}
}

func outstripArgs() []string {
	return []string{"llama-server", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"}
}

func outstripStrategy(ctx int) *placement.Strategy {
	return &placement.Strategy{
		ContextSize: ctx, ContextAuto: true, KVType: "q8_0", KVPlacement: "gpu",
		TensorSplit: []float64{0.24, 0.59, 0.17},
	}
}

// The distinction the whole change rests on: a deficit larger than the device's
// KV share is PRICED (we know the geometry, the shortfall simply exceeds it) and
// must be separable from geometry we cannot price at all.
func TestOutstripIsDistinctFromUnknownGeometry(t *testing.T) {
	model, args := outstripModel(), outstripArgs()
	strategy := outstripStrategy(65536)

	if !contextDeficitOutstripsDevice(model, strategy, args, 100000, 0) {
		t.Fatal("a deficit far beyond the device KV share was not reported as outstripping")
	}
	// Same situation contextReclaimTokens reports as 0 — the two must not be
	// conflated, which is what the previous attempt got wrong.
	if got := contextReclaimTokens(model, strategy, args, 100000, 0); got != 0 {
		t.Fatalf("reclaim tokens should still bow out, got %d", got)
	}

	// Unknown KV geometry is NOT outstripping: we know nothing, so the
	// conservative one-granule step must be preserved.
	unknown := &placement.Strategy{ContextSize: 262144, ContextAuto: true}
	if contextDeficitOutstripsDevice(model, unknown, []string{"llama-server"}, 1500, 0) {
		t.Fatal("unknown geometry was misreported as outstripping")
	}

	// A coverable deficit is not outstripping either.
	if contextDeficitOutstripsDevice(model, strategy, args, 20, 0) {
		t.Fatal("a coverable deficit was reported as outstripping")
	}
}

// The defect: five rounds surrendering 1,024 tokens each against a multi-GiB
// deficit. The ceiling must now descend by a real fraction instead.
func TestOutstrippedRejectionDescendsInsteadOfCreeping(t *testing.T) {
	recovery := newLaunchMemoryRecovery()
	strategy := outstripStrategy(780288)
	recovery.rejectContext(strategy, 0)
	recovery.rejectContextOutstripped(strategy)

	ceiling := recovery.automaticContextCeiling()
	if ceiling >= strategy.ContextSize-1 {
		t.Fatalf("ceiling %d still creeps; want a bounded descent below %d", ceiling, strategy.ContextSize-1)
	}
	want := 780288 - 780288/contextRecoveryStepDivisor
	if ceiling != want {
		t.Fatalf("ceiling %d, want a %d-divisor step to %d", ceiling, contextRecoveryStepDivisor, want)
	}
}

// Proven context must never be spent chasing an unproven one.
func TestOutstrippedDescentNeverGoesBelowAnAcceptedContext(t *testing.T) {
	recovery := newLaunchMemoryRecovery()
	recovery.acceptContext(&placement.Strategy{ContextSize: 700000, ContextAuto: true, UBatchSize: 128})
	strategy := outstripStrategy(780288)
	recovery.rejectContext(strategy, 0)
	recovery.rejectContextOutstripped(strategy)

	if got := recovery.automaticContextCeiling(); got != 700000 {
		t.Fatalf("ceiling %d, want the accepted context 700000", got)
	}
}

// The usable-window floor still binds.
func TestOutstrippedDescentStopsAtTheFloor(t *testing.T) {
	recovery := newLaunchMemoryRecovery()
	strategy := outstripStrategy(contextRecoveryFloorTokens + 2048)
	recovery.rejectContext(strategy, 0)
	recovery.rejectContextOutstripped(strategy)

	if got := recovery.automaticContextCeiling(); got < contextRecoveryFloorTokens {
		t.Fatalf("ceiling %d fell below the usable-window floor %d", got, contextRecoveryFloorTokens)
	}
}

// The ceiling may only ratchet down, outstripped rejections included.
func TestOutstrippedDescentOnlyRatchetsDown(t *testing.T) {
	recovery := newLaunchMemoryRecovery()
	small := outstripStrategy(400000)
	recovery.rejectContext(small, 0)
	recovery.rejectContextOutstripped(small)
	first := recovery.automaticContextCeiling()

	large := outstripStrategy(900000)
	recovery.rejectContext(large, 0)
	recovery.rejectContextOutstripped(large)

	if got := recovery.automaticContextCeiling(); got > first {
		t.Fatalf("a later larger rejection raised the ceiling from %d to %d", first, got)
	}
}

// Without an outstripped rejection nothing changes: the conservative step stands.
func TestCeilingUnchangedWithoutAnOutstrippedRejection(t *testing.T) {
	recovery := newLaunchMemoryRecovery()
	strategy := outstripStrategy(262144)
	recovery.rejectContext(strategy, 0)

	if got := recovery.automaticContextCeiling(); got != 262143 {
		t.Fatalf("ceiling %d, want one token below 262144 when nothing outstripped", got)
	}
}

// Descent must converge in a handful of rounds, which is the point of the change.
func TestOutstrippedDescentConvergesQuickly(t *testing.T) {
	recovery := newLaunchMemoryRecovery()
	ctx := 780288
	rounds := 0
	for ctx > 200000 && rounds < 20 {
		s := outstripStrategy(ctx)
		recovery.rejectContext(s, 0)
		recovery.rejectContextOutstripped(s)
		next := recovery.automaticContextCeiling()
		if next >= ctx {
			t.Fatalf("round %d did not descend: %d -> %d", rounds, ctx, next)
		}
		ctx = next
		rounds++
	}
	if rounds > 8 {
		t.Fatalf("descent took %d rounds to pass 200000; the creep is not fixed", rounds)
	}
}
