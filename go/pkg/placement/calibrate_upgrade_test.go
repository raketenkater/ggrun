package placement

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// writeDecision stores one decision at the path LoadCalibrationDecision reads.
func writeDecision(t *testing.T, dir, scope string, d CalibrationDecision) {
	t.Helper()
	path := CalibrationPath(dir, scope)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	data, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
}

// A cached decision recorded under an older admission policy must not be
// applied after the policy changes. Version 24 chose its winner in a search
// where the bounded ladder's fallback slots were unreachable — a pre-load
// refusal consumed the reload failure budget, and the slots that survived went
// to neighbouring rungs of the predicted finalist. A "default" winner from that
// search is not evidence that default wins the search version 25 can run.
//
// Moving cache files by hand, which is how the ladder was exercised during
// development, hides exactly this upgrade path.
func TestDecisionFromAnOlderAdmissionPolicyIsNotReused(t *testing.T) {
	dir := t.TempDir()
	const scope = "scope-upgrade"

	writeDecision(t, dir, scope, CalibrationDecision{
		SchemaVersion: CalibrationSchemaVersion - 1,
		ScopeKey:      scope,
		Winner:        "default",
		ModelBasename: "model.gguf",
	})

	if got, err := LoadCalibrationDecision(dir, scope); err == nil {
		t.Fatalf("a decision from the previous admission policy was reused: winner %q", got.Winner)
	}
}

// The bump must not make every launch re-search. A decision written by this
// build is reused on the next identical launch.
func TestFreshDecisionIsReusedOnAnIdenticalLaunch(t *testing.T) {
	dir := t.TempDir()
	const scope = "scope-fresh"

	writeDecision(t, dir, scope, CalibrationDecision{
		SchemaVersion: CalibrationSchemaVersion,
		ScopeKey:      scope,
		Winner:        "ubatch-512",
		ModelBasename: "model.gguf",
	})

	got, err := LoadCalibrationDecision(dir, scope)
	if err != nil {
		t.Fatalf("a decision written by this build was not reused: %v", err)
	}
	if got.Winner != "ubatch-512" {
		t.Errorf("reused winner = %q, want ubatch-512", got.Winner)
	}
}

// The version is what carries the policy change, so it may only move forward.
// Leaving it at 24 is the defect this pair of tests exists to prevent.
func TestAdmissionPolicyChangeAdvancedTheSchemaVersion(t *testing.T) {
	if CalibrationSchemaVersion < 25 {
		t.Fatalf("CalibrationSchemaVersion = %d; the ladder/budget admission change needs at least 25",
			CalibrationSchemaVersion)
	}
}
