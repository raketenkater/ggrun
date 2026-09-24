package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

type admissionClock struct{ t time.Time }

func (c *admissionClock) now() time.Time          { return c.t }
func (c *admissionClock) advance(d time.Duration) { c.t = c.t.Add(d) }

// A nested probe used to take a fresh 30-minute timeout of its own. Under one
// admission every start draws from the same deadline and load allowance.
func TestNestedLoadsCannotResetTheAdmissionDeadline(t *testing.T) {
	clock := &admissionClock{t: time.Unix(1_700_000_000, 0)}
	work := newAdmissionWork("start", 50*time.Minute, 30*time.Minute, maxStartAdmissionLoads, "", clock.now)

	first, err := work.beginLoad(30 * time.Minute)
	if err != nil || first != 30*time.Minute {
		t.Fatalf("first probe = %s, %v; want its own 30m ceiling", first, err)
	}
	clock.advance(30 * time.Minute) // the probe ran to its timeout
	second, err := work.beginLoad(30 * time.Minute)
	if err != nil || second != 20*time.Minute {
		t.Fatalf("second load = %s, %v; want the 20m left in the window", second, err)
	}
	clock.advance(20*time.Minute - 500*time.Millisecond)
	if _, err := work.beginLoad(30 * time.Minute); !isAdmissionBudgetError(err) {
		t.Fatalf("expired window allowed another load: %v", err)
	}
	if work.loads != 2 || !work.loadedWeights() {
		t.Fatalf("loads = %d, loadedWeights = %v", work.loads, work.loadedWeights())
	}
}

func TestAdmissionLoadAllowanceIsFinite(t *testing.T) {
	clock := &admissionClock{t: time.Unix(1_700_000_000, 0)}
	work := newAdmissionWork("challenger", 10*time.Hour, time.Minute, maxBoundedAdmissionLoads, "", clock.now)
	for i := 0; i < maxBoundedAdmissionLoads; i++ {
		if _, err := work.beginLoad(time.Minute); err != nil {
			t.Fatalf("load %d refused early: %v", i+1, err)
		}
		clock.advance(time.Second)
	}
	_, err := work.beginLoad(time.Minute)
	var budget *admissionBudgetError
	if !errors.As(err, &budget) || budget.Loads != maxBoundedAdmissionLoads {
		t.Fatalf("third challenger load = %v", err)
	}
	// Budget exhaustion is not a memory verdict and must never be cached as one.
	if isStableExactAdmissionFailure(err) {
		t.Fatal("budget exhaustion classified as stable negative admission evidence")
	}
}

// A memory refusal decided after a contained full-load probe cost a load; the
// same refusal from the oracle or cached evidence did not.
func TestRefusalAfterContainedProbeIsExpensive(t *testing.T) {
	for _, class := range []exactAdmissionClass{exactAdmissionMemory, exactAdmissionCompat, exactAdmissionCompanion} {
		cheap := exactAdmissionError(class, " on CUDA0 (2701 MiB deficit)", nil)
		if exactAdmissionLoadedWeights(cheap) {
			t.Fatalf("%s from metadata was charged as a load", class)
		}
		probed := exactAdmissionError(class, " on CUDA0 (2701 MiB deficit)", nil)
		probed.(*exactAdmissionFailure).loadedWeights = true
		if !exactAdmissionLoadedWeights(fmt.Errorf("calibrate: %w", probed)) {
			t.Fatalf("%s after a contained probe was treated as free", class)
		}
		if !isStableExactAdmissionFailure(probed) {
			t.Fatalf("%s stopped being stable evidence", class)
		}
	}
}

// The observed lifecycle: an 11m28s baseline, no oracle, a 20-minute budget.
// Stopping the baseline for that finalist cannot be paid back.
func TestHealthyBaselineIsKeptWhenRestorationIsUnaffordable(t *testing.T) {
	cost := estimateChallengerCost(challengerCostInputs{
		BaselineLoad: 11*time.Minute + 28*time.Second, BaselineLoadObserved: true,
		BaselineWorkload: 3 * time.Minute, Ceiling: 30 * time.Minute,
		ProbeNeeded: true, BaselineRunning: true,
	})
	remaining := 20*time.Minute - 3*time.Minute
	if cost.Affordable(remaining) {
		t.Fatalf("MiMo finalist judged affordable: %s within %s", cost, remaining)
	}
	// Even with the oracle, candidate load + restoration exceed the budget.
	cost = estimateChallengerCost(challengerCostInputs{
		BaselineLoad: 11*time.Minute + 28*time.Second, BaselineLoadObserved: true,
		BaselineWorkload: 3 * time.Minute, Ceiling: 30 * time.Minute, BaselineRunning: true,
	})
	if cost.Affordable(remaining) {
		t.Fatalf("two 11.5-minute loads judged affordable in %s: %s", remaining, cost)
	}
	if cost.Restore < 11*time.Minute+28*time.Second {
		t.Fatalf("restoration reserve %s is below the observed load", cost.Restore)
	}
}

func TestSmallModelChallengerRemainsAffordable(t *testing.T) {
	cost := estimateChallengerCost(challengerCostInputs{
		BaselineLoad: 12 * time.Second, BaselineLoadObserved: true,
		BaselineWorkload: 70 * time.Second, Ceiling: 8 * time.Minute, BaselineRunning: true,
	})
	if !cost.Affordable(18 * time.Minute) {
		t.Fatalf("a 12-second model cannot afford one challenger: %s", cost)
	}
}

func TestUnobservedBaselineLoadIsPricedAtTheCeiling(t *testing.T) {
	cost := estimateChallengerCost(challengerCostInputs{Ceiling: 30 * time.Minute, BaselineWorkload: time.Minute})
	if cost.Load < 30*time.Minute || cost.Restore < 30*time.Minute {
		t.Fatalf("unknown load cost was assumed cheap: %s", cost)
	}
}

func readLaunchWork(t *testing.T, cacheDir string) []launchWorkRecord {
	t.Helper()
	f, err := os.Open(launchWorkLedgerPath(cacheDir))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out []launchWorkRecord
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var rec launchWorkRecord
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			t.Fatal(err)
		}
		out = append(out, rec)
	}
	return out
}

// End to end through the real start boundary: a loader that goes silent uses
// only its admission window, the attempt is in the ledger as a timeout with its
// silence, and the admission ends instead of starting another load.
func TestStalledProductionLoadIsBoundedAndRecorded(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX fake backend")
	}
	backend := writeFakeBackend(t, "llama-server", "echo 'load_tensors: loading' >&2\nexec sleep 30\n")
	cacheDir := t.TempDir()
	args := []string{backend, "--port", "59995"}
	started := time.Now()
	_, _, _, err := startLaunchWithCUDAOOMRecoveryState(
		&launchRequest{SpecMode: "off", Port: 59995}, &config.Config{CacheDir: cacheDir}, &placement.ModelProfile{Basename: "stall"},
		nil, &backendInfo{Path: backend, Identity: "fake"}, nil, args, 1500*time.Millisecond, newLaunchMemoryRecovery(),
	)
	if err == nil {
		t.Fatal("stalled load reported success")
	}
	if elapsed := time.Since(started); elapsed > 10*time.Second {
		t.Fatalf("admission outlived its window: %s", elapsed)
	}
	records := readLaunchWork(t, cacheDir)
	if len(records) != 1 {
		t.Fatalf("ledger = %#v, want exactly one production attempt", records)
	}
	rec := records[0]
	if rec.Kind != "production" || !rec.LoadedWeights || rec.Outcome != "timeout" || rec.AdmissionLoad != 1 {
		t.Fatalf("production record = %#v", rec)
	}
	if rec.LastOutputAge <= 0 || !strings.Contains(rec.Reason, "timeout waiting for server") {
		t.Fatalf("record lacks the terminal reason or silence: %#v", rec)
	}
	if filepath.Base(launchWorkLedgerPath(cacheDir)) != "launch-work.jsonl" {
		t.Fatal("ledger path changed")
	}
}

func fakeOracleBuild(t *testing.T, fitBody string) string {
	t.Helper()
	dir := t.TempDir()
	server := filepath.Join(dir, "llama-server")
	if err := os.WriteFile(server, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "llama-fit-params"), []byte("#!/bin/sh\n"+fitBody), 0o755); err != nil {
		t.Fatal(err)
	}
	return server
}

// Qwen3.5-4B, live: three challengers were refused by the oracle only after the
// healthy baseline had been stopped, which then cost a restoration load. The
// oracle can refuse them against the pre-launch resource state first.
func TestPrescreenRefusesOracleDeficitWithoutStoppingBaseline(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX fake oracle")
	}
	server := fakeOracleBuild(t, "echo 'CUDA0 2603 2141 990'\n")
	baseline := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "RTX 4070", VRAMTotalMB: 5729}}}
	cfg := &config.Config{CacheDir: t.TempDir()}
	strategy := &placement.Strategy{ContextSize: 125952, UBatchSize: 1024, Parallel: 1}
	refused, class, reason := prescreenCalibrationCandidate(&launchRequest{}, cfg, &placement.ModelProfile{Basename: "q"},
		&backendInfo{Path: server, Tag: "llama"}, baseline, strategy, []string{server, "-m", "q.gguf", "-ub", "1024"})
	if !refused || class != string(exactAdmissionMemory) || !strings.Contains(reason, "CUDA0 deficit") {
		t.Fatalf("prescreen = %v %q %q", refused, class, reason)
	}
}

func TestPrescreenNeverStartsAContainedProbe(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX fake oracle")
	}
	server := fakeOracleBuild(t, "echo 'boom' >&2; exit 1\n")
	baseline := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "RTX 4070", VRAMTotalMB: 12282}}}
	cacheDir := t.TempDir()
	refused, _, _ := prescreenCalibrationCandidate(&launchRequest{AllowLiveMemoryProbe: true}, &config.Config{CacheDir: cacheDir},
		&placement.ModelProfile{Basename: "q"}, &backendInfo{Path: server, Tag: "llama"}, baseline,
		&placement.Strategy{ContextSize: 4096, UBatchSize: 512, Parallel: 1}, []string{server, "-m", "q.gguf"})
	if refused {
		t.Fatal("an oracle failure was treated as a refusal")
	}
	for _, rec := range readLaunchWork(t, cacheDir) {
		if rec.LoadedWeights {
			t.Fatalf("prescreen started a weight-loading process: %#v", rec)
		}
	}
}
