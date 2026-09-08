package placement

import (
	"strings"
	"testing"
)

const proofModel = "GLM-5.3-Flash-UD-Q3_K_XL.gguf"

// The 2026-09-08 matched comparison: cache on 6.39 tok/s decode with 2 resident
// expert layers, cache off 7.26 with 5. Disjoint distributions, no warming.
func measuredLoss(identity string) HotExpertDisplacementProof {
	return HotExpertDisplacementProof{
		Identity: identity, Demoted: 3, Slots: 14,
		CacheDecodeTPS: 6.39, BaselineDecodeTPS: 7.26,
	}
}

// TestDisplacementFreeCacheNeedsNoProof: a cache funded from residual VRAM
// evicts nothing, so its cost is zero and there is nothing to justify.
func TestDisplacementFreeCacheNeedsNoProof(t *testing.T) {
	ok, why := hotExpertDisplacementAllowed(t.TempDir(), proofModel, "ident-a", 0, false)
	if !ok {
		t.Fatalf("a cache that displaces nothing must be allowed: %s", why)
	}
	if !strings.Contains(why, "displaces no resident expert layer") {
		t.Errorf("reason should say the cache costs nothing; got %q", why)
	}
}

// TestAutoDeclinesUnprovenDisplacement is the regression this gate exists for.
// Auto chose a 14-slot cache by evicting a resident expert layer and lost 12%
// decode. Without a measurement on this placement, auto must decline.
func TestAutoDeclinesUnprovenDisplacement(t *testing.T) {
	ok, why := hotExpertDisplacementAllowed(t.TempDir(), proofModel, "ident-a", 1, false)
	if ok {
		t.Fatal("auto must not evict resident expert layers on unproven benefit")
	}
	if !strings.Contains(why, "declined") || !strings.Contains(why, "evicts 1 resident expert layer") {
		t.Errorf("reason must name the cost; got %q", why)
	}
}

// TestExplicitRequestMayServeUnproven: refusing an explicit request for want of
// a measurement would make the measurement unobtainable, the same bootstrap
// deadlock ErrHotExpertBaselineUnmeasured already avoids. Invariant 3 also
// makes an explicit choice a constraint.
func TestExplicitRequestMayServeUnproven(t *testing.T) {
	ok, why := hotExpertDisplacementAllowed(t.TempDir(), proofModel, "ident-a", 2, true)
	if !ok {
		t.Fatalf("--hot-experts on must be able to serve once to gather evidence: %s", why)
	}
	if !strings.Contains(why, "unproven") {
		t.Errorf("reason must record that this serve is unproven; got %q", why)
	}
}

// TestMeasuredLossDeclinedForBothModes: once measured, a losing trade is
// refused even when explicitly requested -- the bootstrap allowance exists to
// obtain evidence, not to override it.
func TestMeasuredLossDeclinedForBothModes(t *testing.T) {
	cacheDir := t.TempDir()
	if err := RecordHotExpertDisplacementProof(cacheDir, proofModel, measuredLoss("ident-a")); err != nil {
		t.Fatalf("record proof: %v", err)
	}
	for _, required := range []bool{false, true} {
		ok, why := hotExpertDisplacementAllowed(cacheDir, proofModel, "ident-a", 3, required)
		if ok {
			t.Errorf("required=%v: a measured 6.39 vs 7.26 loss must not be funded", required)
		}
		if !strings.Contains(why, "lost") {
			t.Errorf("required=%v: reason should carry the verdict; got %q", required, why)
		}
	}
}

// TestMeasuredWinIsFunded: the gate is on evidence, not on the feature.
func TestMeasuredWinIsFunded(t *testing.T) {
	cacheDir := t.TempDir()
	win := HotExpertDisplacementProof{
		Identity: "ident-b", Demoted: 1, Slots: 48,
		CacheDecodeTPS: 9.10, BaselineDecodeTPS: 7.26,
	}
	if err := RecordHotExpertDisplacementProof(cacheDir, proofModel, win); err != nil {
		t.Fatalf("record proof: %v", err)
	}
	ok, why := hotExpertDisplacementAllowed(cacheDir, proofModel, "ident-b", 1, false)
	if !ok {
		t.Fatalf("a measured win must be funded: %s", why)
	}
	if !strings.Contains(why, "won") {
		t.Errorf("reason should carry the verdict; got %q", why)
	}
}

// TestTieGoesAgainstTheCache: equal throughput for fewer resident weights is
// the worse plan, because the evicted layers would otherwise absorb a longer
// context or a larger batch.
func TestTieGoesAgainstTheCache(t *testing.T) {
	cacheDir := t.TempDir()
	tie := HotExpertDisplacementProof{
		Identity: "ident-c", Demoted: 2, Slots: 20,
		CacheDecodeTPS: 7.26, BaselineDecodeTPS: 7.26,
	}
	if err := RecordHotExpertDisplacementProof(cacheDir, proofModel, tie); err != nil {
		t.Fatalf("record proof: %v", err)
	}
	if ok, _ := hotExpertDisplacementAllowed(cacheDir, proofModel, "ident-c", 2, false); ok {
		t.Error("a tie must not fund displacement")
	}
}

// TestOneArmedProofIsNotAComparison guards invariant 7: a baseline-only
// measurement is not a completed performance decision.
func TestOneArmedProofIsNotAComparison(t *testing.T) {
	cacheDir := t.TempDir()
	half := HotExpertDisplacementProof{
		Identity: "ident-d", Demoted: 1, Slots: 14, BaselineDecodeTPS: 7.26,
	}
	if err := RecordHotExpertDisplacementProof(cacheDir, proofModel, half); err != nil {
		t.Fatalf("record proof: %v", err)
	}
	if half.Measured() {
		t.Error("a proof with one arm must not read as measured")
	}
	// Auto still declines; an explicit request may still bootstrap.
	if ok, _ := hotExpertDisplacementAllowed(cacheDir, proofModel, "ident-d", 1, false); ok {
		t.Error("auto must decline on a one-armed proof")
	}
	if ok, _ := hotExpertDisplacementAllowed(cacheDir, proofModel, "ident-d", 1, true); !ok {
		t.Error("an explicit request must still be able to complete the comparison")
	}
}

// TestProofIsPerIdentity: a different placement is a different trade, so
// evidence must not leak between them.
func TestProofIsPerIdentity(t *testing.T) {
	cacheDir := t.TempDir()
	if err := RecordHotExpertDisplacementProof(cacheDir, proofModel, measuredLoss("ident-a")); err != nil {
		t.Fatalf("record proof: %v", err)
	}
	if _, ok := LoadHotExpertDisplacementProof(cacheDir, proofModel, "ident-other"); ok {
		t.Error("evidence recorded for one placement must not answer for another")
	}
	if ok, _ := hotExpertDisplacementAllowed(cacheDir, proofModel, "ident-other", 1, false); ok {
		t.Error("an unmeasured placement must not inherit another's verdict")
	}
}

// TestProofReplacedNotAppended: the freshest matched pair describes the machine
// as it is now; stale rows must not accumulate and win by ordering.
func TestProofReplacedNotAppended(t *testing.T) {
	cacheDir := t.TempDir()
	if err := RecordHotExpertDisplacementProof(cacheDir, proofModel, measuredLoss("ident-a")); err != nil {
		t.Fatalf("record loss: %v", err)
	}
	better := HotExpertDisplacementProof{
		Identity: "ident-a", Demoted: 1, Slots: 48,
		CacheDecodeTPS: 8.80, BaselineDecodeTPS: 7.26,
	}
	if err := RecordHotExpertDisplacementProof(cacheDir, proofModel, better); err != nil {
		t.Fatalf("record win: %v", err)
	}
	got, ok := LoadHotExpertDisplacementProof(cacheDir, proofModel, "ident-a")
	if !ok {
		t.Fatal("proof missing after replacement")
	}
	if got.Slots != 48 || got.CacheDecodeTPS < 8.7 {
		t.Errorf("later result must replace the earlier one; got %+v", got)
	}
	if !got.Earned() {
		t.Error("the replacing measurement should be the one consulted")
	}
}
