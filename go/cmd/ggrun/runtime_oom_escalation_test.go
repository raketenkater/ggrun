package main

import (
	"errors"
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

// replanAfter returns the identical-argv refusal until `changesAt` reserves
// have been made, then a new plan.
func replanAfter(changesAt int, reserves *int) func() (*placement.Strategy, []string, error) {
	return func() (*placement.Strategy, []string, error) {
		if *reserves >= changesAt {
			return &placement.Strategy{}, []string{"-ot", "fewer-layers"}, nil
		}
		return nil, nil, errRuntimeOOMReplanIdentical
	}
}

func TestSizelessRuntimeOOMReservesMoreUntilThePlanChanges(t *testing.T) {
	reserves := 0
	_, args, err := escalateSizelessRuntimeOOM(true, replanAfter(2, &reserves), func() error { reserves++; return nil })
	if err != nil || len(args) == 0 {
		t.Fatalf("recovery must reach a different plan, got err=%v", err)
	}
	if reserves != 2 {
		t.Fatalf("reserved %d extra layers, want exactly the 2 that changed the plan", reserves)
	}
}

func TestSizelessRuntimeOOMEscalationIsBounded(t *testing.T) {
	reserves := 0
	_, _, err := escalateSizelessRuntimeOOM(true, replanAfter(1000, &reserves), func() error { reserves++; return nil })
	if !errors.Is(err, errRuntimeOOMReplanIdentical) {
		t.Fatalf("an unchangeable plan must still be refused, got %v", err)
	}
	if reserves != maxSizelessOOMEscalations {
		t.Fatalf("reserved %d times, want the bound %d", reserves, maxSizelessOOMEscalations)
	}
}

// A sized OOM already reserved what the backend asked for; guessing more is
// not justified by the evidence.
func TestSizedRuntimeOOMIsNotEscalated(t *testing.T) {
	reserves := 0
	_, _, err := escalateSizelessRuntimeOOM(false, replanAfter(1, &reserves), func() error { reserves++; return nil })
	if !errors.Is(err, errRuntimeOOMReplanIdentical) || reserves != 0 {
		t.Fatalf("sized OOM escalated (reserves=%d err=%v)", reserves, err)
	}
}

func TestRuntimeOOMEscalationPassesOtherFailuresThrough(t *testing.T) {
	boom := errors.New("placement failed")
	reserves := 0
	_, _, err := escalateSizelessRuntimeOOM(true, func() (*placement.Strategy, []string, error) { return nil, nil, boom },
		func() error { reserves++; return nil })
	if !errors.Is(err, boom) || reserves != 0 {
		t.Fatalf("a non-identical failure must not reserve more (reserves=%d err=%v)", reserves, err)
	}
	recordErr := errors.New("cache write failed")
	_, _, err = escalateSizelessRuntimeOOM(true, replanAfter(5, &reserves), func() error { return recordErr })
	if !errors.Is(err, recordErr) {
		t.Fatalf("a failed reserve must stop recovery, got %v", err)
	}
}
