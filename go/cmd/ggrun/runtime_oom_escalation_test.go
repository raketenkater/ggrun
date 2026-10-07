package main

import (
	"errors"
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
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

// The reserve is filed under the crashed plan's exact key. A re-plan that moves
// context or ubatch never sees it and refills the freed VRAM with KV.
func TestRuntimeOOMReplanKeepsTheCrashedShape(t *testing.T) {
	failed := &placement.Strategy{ContextSize: 405504, ContextAuto: true, UBatchSize: 64, BatchSize: 2048, Parallel: 1}
	opts := pinRuntimeOOMReplan(placement.Options{ContextSize: 0, AutoContextMax: 1 << 20, UBatchSize: 0, AutoParallel: true}, failed)
	if opts.ContextSize != 405504 || opts.AutoContextMax != 0 || opts.UBatchSize != 64 || opts.BatchSize != 2048 ||
		opts.Parallel != 1 || opts.AutoParallel {
		t.Fatalf("re-plan options not pinned to the crashed shape: %+v", opts)
	}
	base := placement.Options{ContextSize: 0, AutoContextMax: 123, UBatchSize: 512}
	if got := pinRuntimeOOMReplan(base, nil); got.AutoContextMax != 123 || got.UBatchSize != 512 {
		t.Fatalf("no crashed plan must leave the options alone: %+v", got)
	}

	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	req := &launchRequest{Parallel: 1}
	model := &placement.ModelProfile{SizeBytes: 1, NumLayers: 32, HeadCountKV: 8, KeyLength: 128, ValueLength: 128}
	be := &backendInfo{Tag: "llama", Identity: "build"}
	caps := &detect.Capabilities{CPU: detect.CPUInfo{Cores: 4}, RAM: detect.RAMInfo{TotalMB: 16384, FreeMB: 16384}}
	crashed := &placement.Strategy{ContextSize: 16384, ContextAuto: true, UBatchSize: 256, BatchSize: 1024, Parallel: 1}
	next, _, err := replanAfterRuntimeOOM(req, cfg, model, be, caps, []string{"crashed"}, crashed, newLaunchMemoryRecovery())
	if err != nil {
		t.Fatal(err)
	}
	if next.ContextSize != 16384 || next.UBatchSize != 256 || !next.ContextAuto {
		t.Fatalf("re-plan moved off the crashed shape: ctx=%d ub=%d auto=%v", next.ContextSize, next.UBatchSize, next.ContextAuto)
	}
}
