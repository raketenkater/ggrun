package main

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

func ladderCandidates(names ...string) []placement.CalibrationCandidate {
	out := make([]placement.CalibrationCandidate, 0, len(names))
	for _, name := range names {
		out = append(out, placement.CalibrationCandidate{
			Name:     name,
			Strategy: &placement.Strategy{Parallel: 1},
		})
	}
	return out
}

func ladderNames(candidates []placement.CalibrationCandidate) string {
	names := make([]string, 0, len(candidates))
	for _, c := range candidates {
		names = append(names, c.Name)
	}
	return strings.Join(names, ",")
}

// The bounded ladder has three challenger slots. Spending all three on rungs of
// one coordinate measures one bottleneck three times: when a rung is refused,
// the neighbouring rung is refused for the same reason on the same device.
//
// Measured on Qwen3.8-Flash-Next: the frontier calculated five candidates
// including two topology shapes, and every challenger slot went to ubatch-2048,
// ubatch-1024 and ubatch-512 (deficits 7026, 3608 and 1885 MiB on CUDA0). No
// topology shape was ever admitted.
func TestAdmissionLadderSpreadsAcrossLevers(t *testing.T) {
	candidates := ladderCandidates("default", "ubatch-2048", "ubatch-1024", "ubatch-512", "topology-balanced-012", "moe-owner-1")
	plan := selectAutomaticCalibrationAdmissionPlan(candidates, nil, 4)

	if len(plan) != 4 {
		t.Fatalf("ladder kept %d entries, want 4: %s", len(plan), ladderNames(plan))
	}
	if plan[0].Name != "default" || plan[1].Name != "ubatch-2048" {
		t.Fatalf("the baseline and the predicted finalist must lead the ladder, got %s", ladderNames(plan))
	}
	families := map[string]int{}
	for _, c := range plan[1:] {
		for _, family := range calibrationLeverFamilies(c.Name) {
			families[family]++
		}
	}
	if families["ubatch"] > 1 {
		t.Errorf("ladder spent %d challenger slots on the ubatch family: %s", families["ubatch"], ladderNames(plan))
	}
	if len(families) < 3 {
		t.Errorf("ladder covered only %d lever families: %s", len(families), ladderNames(plan))
	}
}

// Spreading is a preference, not a quota. When one family is all the generator
// produced, the ladder must still fill its slots rather than return short —
// otherwise a model whose only legal alternative is a second ubatch rung would
// lose its bounded search entirely.
func TestAdmissionLadderStillFillsWhenOneFamilyIsAllThereIs(t *testing.T) {
	candidates := ladderCandidates("default", "ubatch-2048", "ubatch-1024", "ubatch-512")
	plan := selectAutomaticCalibrationAdmissionPlan(candidates, nil, 4)
	if len(plan) != 4 {
		t.Fatalf("ladder returned %d entries when only one family exists, want 4: %s", len(plan), ladderNames(plan))
	}
	if ladderNames(plan) != "default,ubatch-2048,ubatch-1024,ubatch-512" {
		t.Errorf("unexpected ladder for a single-family frontier: %s", ladderNames(plan))
	}
}

// The predicted finalist keeps its slot. The contract lets prediction choose at
// most a finalist, so spreading must never displace it.
func TestAdmissionLadderKeepsThePredictedFinalistFirst(t *testing.T) {
	candidates := ladderCandidates("default", "topology-balanced-012", "ubatch-1024", "moe-owner-1")
	plan := selectAutomaticCalibrationAdmissionPlan(candidates, nil, 3)
	if len(plan) < 2 || plan[1].Name != "topology-balanced-012" {
		t.Fatalf("the predicted finalist lost its slot: %s", ladderNames(plan))
	}
	if len(plan) != 3 {
		t.Fatalf("ladder kept %d entries, want 3: %s", len(plan), ladderNames(plan))
	}
}

// No candidate may appear twice, however the slots are filled.
func TestAdmissionLadderNeverRepeatsACandidate(t *testing.T) {
	candidates := ladderCandidates("default", "ubatch-2048", "parallel-2", "ubatch-1024", "parallel-4")
	plan := selectAutomaticCalibrationAdmissionPlan(candidates, nil, 5)
	seen := map[string]bool{}
	for _, c := range plan {
		if seen[c.Name] {
			t.Fatalf("candidate %s appears twice: %s", c.Name, ladderNames(plan))
		}
		seen[c.Name] = true
	}
}

func TestCalibrationLeverFamiliesReadsTheGeneratorNaming(t *testing.T) {
	for name, want := range map[string][]string{
		"ubatch-2048": {"ubatch"},
		"ubatch-512":  {"ubatch"},
		"parallel-4":  {"parallel"},
		// A compound moves two coordinates and must name both. Qwen3.8-27B's
		// frontier produces this form; Qwen3.8-Flash-Next's never did.
		"batch-1024-ubatch-512": {"batch", "ubatch"},
		"batch-8192-ubatch-32":  {"batch", "ubatch"},
		// Descriptive tails follow a key, not a value, so they are not keys.
		"topology-balanced-012": {"topology"},
		"moe-owner-1":           {"moe"},
		"moe-topology-a":        {"moe"},
		"single-gpu-2":          {"single"},
		"kv-alternate":          {"kv"},
		"split-inverted":        {"split"},
		"default":               {"default"},
		"-leading":              {"-leading"},
	} {
		got := calibrationLeverFamilies(name)
		if strings.Join(got, ",") != strings.Join(want, ",") {
			t.Errorf("calibrationLeverFamilies(%q) = %v, want %v", name, got, want)
		}
	}
}

// A compound sharing a coordinate with an already-selected candidate is not a
// new lever. Without this the ladder pairs "ubatch-512" with
// "batch-1024-ubatch-512", which carries the same ubatch that was just refused.
func TestAdmissionLadderTreatsCompoundNamesAsOverlapping(t *testing.T) {
	candidates := ladderCandidates("default", "ubatch-512", "batch-1024-ubatch-512", "topology-balanced-012", "parallel-2")
	plan := selectAutomaticCalibrationAdmissionPlan(candidates, nil, 4)
	if len(plan) != 4 {
		t.Fatalf("ladder kept %d entries, want 4: %s", len(plan), ladderNames(plan))
	}
	if plan[1].Name != "ubatch-512" {
		t.Fatalf("the predicted finalist lost its slot: %s", ladderNames(plan))
	}
	for _, c := range plan[2:] {
		for _, family := range calibrationLeverFamilies(c.Name) {
			if family == "ubatch" {
				t.Errorf("ladder followed ubatch-512 with %s, which moves ubatch too: %s", c.Name, ladderNames(plan))
			}
		}
	}
}
