package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
)

// Only a device planned against the guess (no measurement before the launch)
// and now measured below it can make a re-plan larger.
func TestOverheadReliefCountsOnlyNewlyMeasuredDevicesBelowTheReserve(t *testing.T) {
	gpus := []detect.GPU{{Index: 0}, {Index: 1}, {Index: 2}, {Index: 3}}
	before := map[int]int{1: 300}
	after := map[int]int{0: 190, 1: 250, 2: placement.UnmeasuredCUDAOverheadMB + 476}
	got := measuredOverheadRelief(before, after, gpus)
	if len(got) != 1 || got[0] != 190 {
		t.Fatalf("relief = %v, want only CUDA0=190", got)
	}
	if got := measuredOverheadRelief(nil, nil, gpus); got != nil {
		t.Fatalf("relief without any measurement = %v", got)
	}
	if got := measuredOverheadRelief(nil, map[int]int{0: 190}, nil); got != nil {
		t.Fatalf("CPU-only host reported relief: %v", got)
	}
	if got := measuredOverheadRelief(nil, map[int]int{0: placement.UnmeasuredCUDAOverheadMB}, gpus); got != nil {
		t.Fatalf("a measurement equal to the reserve is not relief: %v", got)
	}
}

// The gate must read the measured-only view the post-launch probe writes. A
// planning view would show the unmeasured reserve before the launch and hide
// that the device was never measured.
func TestOverheadReliefReadsThePostLaunchProbe(t *testing.T) {
	bin := t.TempDir()
	script := "#!/bin/sh\ncase \"$*\" in *memory.used*) echo 10039 ;; *) exit 1 ;; esac\n"
	if err := os.WriteFile(filepath.Join(bin, "nvidia-smi"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin)
	t.Setenv("HOME", t.TempDir())
	dir := t.TempDir()
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "gpu", VRAMTotalMB: 12282}}}
	before := measuredOverheadView(dir, caps)
	if len(before) != 0 {
		t.Fatalf("fresh cache measured overhead: %v", before)
	}
	log := "llama_model_load_from_file_impl: using device CUDA0\nload_tensors: CUDA0 model buffer size = 4002.00 MiB\n"
	placement.RunPostLaunchProbe(dir, caps.GPUs, log, 0, nil, map[int]int{0: 9428}, true)
	got := measuredOverheadRelief(before, measuredOverheadView(dir, caps), caps.GPUs)
	if got[0] != 611 {
		t.Fatalf("relief = %v, want CUDA0=611 from the probe", got)
	}
	// The next launch planned with the measurement: nothing left to relieve.
	if again := measuredOverheadRelief(measuredOverheadView(dir, caps), measuredOverheadView(dir, caps), caps.GPUs); again != nil {
		t.Fatalf("an already measured device reported relief again: %v", again)
	}
}

// A fresh ledger may only replace one whose every rejection was priced against
// the guessed reserve: a lifecycle with more than the serving weight load holds
// evidence (contained probe, real CUDA OOM) that must not be forgotten.
func TestOverheadReloadNeedsTheServingLoadToBeTheOnlyLoad(t *testing.T) {
	relief := map[int]int{0: 190}
	for _, c := range []struct {
		name   string
		relief map[int]int
		loads  int
		nilLed bool
		want   bool
		why    bool
	}{
		{"one load", relief, 1, false, true, false},
		{"no relief", nil, 1, false, false, false},
		{"probe or OOM reload", relief, 2, false, false, true},
		{"no load recorded", relief, 0, false, false, true},
		{"no ledger", relief, 1, true, false, true},
	} {
		recovery := newLaunchMemoryRecovery()
		recovery.weightLoads = c.loads
		if c.nilLed {
			recovery = nil
		}
		got, why := overheadReloadEligible(&launchRequest{}, overheadServedPlan(), c.relief, recovery)
		if got != c.want || (why != "") != c.why {
			t.Errorf("%s: eligible=%v why=%q", c.name, got, why)
		}
	}
}

func overheadServedPlan() *placement.Strategy {
	return &placement.Strategy{Type: placement.MoEOffload, ContextSize: 455680, UBatchSize: 64, Parallel: 4,
		KVType: "q8_0", KVTypeV: "q8_0", KVPlacement: "gpu"}
}

// The reload is a fit-lane correction, not an optimizer move: it may only
// restore context or ubatch the guess took, never trade one for the other or
// change slots, KV or expert residency, and never reload a non-resident launch.
func TestOverheadReloadUpgradeIsTheSameShapeAndStrictlyLarger(t *testing.T) {
	served := overheadServedPlan()
	with := func(edit func(*placement.Strategy)) *placement.Strategy {
		s := overheadServedPlan()
		edit(s)
		return s
	}
	for _, c := range []struct {
		name      string
		served    *placement.Strategy
		candidate *placement.Strategy
		want      bool
	}{
		{"full plan", served, with(func(s *placement.Strategy) { s.ContextSize, s.UBatchSize = 811008, 512 }), true},
		{"context only", served, with(func(s *placement.Strategy) { s.ContextSize = 811008 }), true},
		{"ubatch only", served, with(func(s *placement.Strategy) { s.UBatchSize = 128 }), true},
		{"lateral", served, with(func(s *placement.Strategy) { s.TensorSplit = []float64{0.3, 0.7} }), false},
		{"less ubatch", served, with(func(s *placement.Strategy) { s.ContextSize, s.UBatchSize = 811008, 32 }), false},
		{"less context", served, with(func(s *placement.Strategy) { s.ContextSize, s.UBatchSize = 400000, 512 }), false},
		{"slots", served, with(func(s *placement.Strategy) { s.ContextSize, s.Parallel = 811008, 2 }), false},
		{"kv type", served, with(func(s *placement.Strategy) { s.ContextSize, s.KVType = 811008, "q4_0" }), false},
		{"kv placement", served, with(func(s *placement.Strategy) { s.ContextSize, s.KVPlacement = 811008, "cpu" }), false},
		{"experts to cpu", served, with(func(s *placement.Strategy) { s.ContextSize, s.NCPUMoE = 811008, 2 }), false},
		{"mmap", served, with(func(s *placement.Strategy) { s.ContextSize, s.MMapRequired = 811008, true }), false},
		{"host spill", &placement.Strategy{Type: placement.SingleGPU, ContextSize: 32768, UBatchSize: 512},
			&placement.Strategy{Type: placement.DenseCPUOffload, ContextSize: 65536, UBatchSize: 512}, false},
		{"non-resident", &placement.Strategy{Type: placement.DenseCPUOffload, ContextSize: 32768, UBatchSize: 512},
			&placement.Strategy{Type: placement.DenseCPUOffload, ContextSize: 65536, UBatchSize: 512}, false},
		{"mmap served", with(func(s *placement.Strategy) { s.MMapRequired = true }),
			with(func(s *placement.Strategy) { s.ContextSize, s.MMapRequired = 811008, true }), false},
		{"second gpu", &placement.Strategy{Type: placement.SingleGPU, ContextSize: 32768, UBatchSize: 512},
			&placement.Strategy{Type: placement.MultiGPUDense, ContextSize: 65536, UBatchSize: 512}, true},
		{"nil", served, nil, false},
	} {
		if got := overheadReloadUpgrade(c.served, c.candidate); got != c.want {
			t.Errorf("%s: upgrade=%v, want %v", c.name, got, c.want)
		}
	}
}

type overheadReloadHarness struct {
	stderr      bytes.Buffer
	events      []string
	plans       int
	stopOK      bool
	running     bool
	killWorks   bool
	releaseErr  error
	started     *server.Process
	startResult *placement.Strategy
	startArgs   []string
	startErr    error
	startLedger *launchMemoryRecovery
	restored    *server.Process
	restoreErr  error
	restoreGot  []string
	restoreLed  *launchMemoryRecovery
	released    []*server.Process
}

func (h *overheadReloadHarness) fx() overheadReloadEffects {
	return overheadReloadEffects{
		plan: func(*launchRequest, *config.Config, *placement.ModelProfile, *backendInfo, *detect.Capabilities, *placement.Strategy, []string) (*placement.Strategy, []string, string) {
			h.plans++
			return nil, nil, "stub planner"
		},
		kill: func(*server.Process) {
			h.events = append(h.events, "kill")
			if h.killWorks {
				h.running = false
			}
		},
		stderr: &h.stderr,
		stopAndWait: func(*server.Process, string, *detect.Capabilities, time.Duration) bool {
			if !strings.Contains(h.stderr.String(), "reloading once") {
				h.events = append(h.events, "stop-before-warning")
			}
			h.events = append(h.events, "stop")
			return h.stopOK
		},
		running: func(*server.Process) bool { return h.running },
		release: func(p *server.Process, _ *detect.Capabilities, _ time.Duration) error {
			h.events = append(h.events, "release")
			h.released = append(h.released, p)
			return h.releaseErr
		},
		start: func(_ *launchRequest, _ *config.Config, _ *placement.ModelProfile, _ *placement.Strategy, _ *backendInfo, _ *detect.Capabilities,
			args []string, _ time.Duration, recovery *launchMemoryRecovery) (*server.Process, *placement.Strategy, []string, error) {
			h.events = append(h.events, "start")
			h.startLedger = recovery
			if recovery.isRejected(args) {
				h.events = append(h.events, "start-saw-stale-rejection")
			}
			return h.started, h.startResult, h.startArgs, h.startErr
		},
		restore: func(_ *launchRequest, _ *config.Config, _ *placement.ModelProfile, s *placement.Strategy, _ *backendInfo, _ *detect.Capabilities,
			args []string, _ time.Duration, recovery *launchMemoryRecovery) (*server.Process, *placement.Strategy, []string, error) {
			h.events = append(h.events, "restore")
			h.restoreGot, h.restoreLed = args, recovery
			return h.restored, s, args, h.restoreErr
		},
	}
}

func overheadReloadFixture() (overheadReload, *placement.Strategy, []string) {
	ledger := newLaunchMemoryRecovery()
	ledger.weightLoads = 1
	candidateArgs := []string{"llama-server", "-c", "811008", "-ub", "512"}
	// The first admission's oracle refused this argv against the guessed reserve.
	ledger.reject(candidateArgs)
	served := overheadReload{process: &server.Process{}, strategy: overheadServedPlan(),
		args: []string{"llama-server", "-c", "455680", "-ub", "64"}, recovery: ledger}
	candidate := overheadServedPlan()
	candidate.ContextSize, candidate.UBatchSize = 811008, 512
	return served, candidate, candidateArgs
}

func runOverheadReload(h *overheadReloadHarness, served overheadReload, candidate *placement.Strategy, candidateArgs []string) overheadReload {
	return reloadIntoMeasuredOverheadPlan(h.fx(), &launchRequest{}, &config.Config{}, &placement.ModelProfile{}, &backendInfo{},
		&detect.Capabilities{}, served, candidate, candidateArgs, map[int]int{0: 190, 2: 136}, nil)
}

func TestOverheadReloadWarnsThenServesTheMeasuredPlanWithAFreshLedger(t *testing.T) {
	served, candidate, candidateArgs := overheadReloadFixture()
	h := &overheadReloadHarness{stopOK: true, started: &server.Process{}, startResult: candidate, startArgs: candidateArgs}
	got := runOverheadReload(h, served, candidate, candidateArgs)
	if strings.Join(h.events, ",") != "stop,start" {
		t.Fatalf("events = %v, want a warning, one stop and one start", h.events)
	}
	if !strings.Contains(h.stderr.String(), "CUDA0=190 MiB, CUDA2=136 MiB") || !strings.Contains(h.stderr.String(), "ctx 455680 -> 811008") {
		t.Fatalf("warning does not name the measurement and the change:\n%s", h.stderr.String())
	}
	if got.process != h.started || got.strategy != candidate {
		t.Fatalf("serving %p/%v, want the started measured plan", got.process, got.strategy)
	}
	if got.recovery == served.recovery || got.recovery != h.startLedger {
		t.Fatal("the reload did not continue under the fresh ledger it was admitted with")
	}
}

func TestOverheadReloadRestoresTheServingPlanWhenAdmissionFails(t *testing.T) {
	served, candidate, candidateArgs := overheadReloadFixture()
	served.recovery = newLaunchMemoryRecovery() // the candidate starts unrejected here
	served.recovery.weightLoads = 1
	failed := &server.Process{}
	h := &overheadReloadHarness{stopOK: true, started: failed, startResult: candidate, startArgs: candidateArgs,
		startErr: errors.New("CUDA OOM"), restored: &server.Process{}}
	got := runOverheadReload(h, served, candidate, candidateArgs)
	if strings.Join(h.events, ",") != "stop,start,release,restore" {
		t.Fatalf("events = %v, want the failed start released before the restore", h.events)
	}
	if len(h.released) != 1 || h.released[0] != failed {
		t.Fatal("the failed candidate process was not released before the restore")
	}
	if got.process != h.restored || formatCommand(h.restoreGot) != formatCommand(served.args) || h.restoreLed != served.recovery {
		t.Fatalf("restored %v under %p, want the serving argv under its own ledger", h.restoreGot, h.restoreLed)
	}
	if !served.recovery.isRejected(candidateArgs) {
		t.Fatal("the refused candidate argv is not rejected for the rest of the launch")
	}
}

func TestOverheadReloadRestoresWhenTheAdmittedPlanIsSmaller(t *testing.T) {
	served, candidate, candidateArgs := overheadReloadFixture()
	smaller := overheadServedPlan()
	smaller.ContextSize = 400000
	h := &overheadReloadHarness{stopOK: true, started: &server.Process{}, startResult: smaller,
		startArgs: []string{"llama-server", "-c", "400000"}, restored: &server.Process{}}
	got := runOverheadReload(h, served, candidate, candidateArgs)
	if strings.Join(h.events, ",") != "stop,start,release,restore" || got.process != h.restored {
		t.Fatalf("events = %v; a smaller admitted plan must be stopped and the serving plan restored", h.events)
	}
}

func TestOverheadReloadKeepsOrRestoresWhenTheStopFails(t *testing.T) {
	served, candidate, candidateArgs := overheadReloadFixture()
	h := &overheadReloadHarness{stopOK: false, running: true}
	if got := runOverheadReload(h, served, candidate, candidateArgs); got.process != served.process || strings.Join(h.events, ",") != "stop" {
		t.Fatalf("events = %v; a server that did not stop must keep serving", h.events)
	}
	h = &overheadReloadHarness{stopOK: false, running: false, restored: &server.Process{}}
	if got := runOverheadReload(h, served, candidate, candidateArgs); got.process != h.restored || strings.Join(h.events, ",") != "stop,restore" {
		t.Fatalf("events = %v; unsettled memory must restore the serving plan, never start the new one", h.events)
	}
}

func TestOverheadReloadReportsNoServerWhenTheRestoreFails(t *testing.T) {
	served, candidate, candidateArgs := overheadReloadFixture()
	h := &overheadReloadHarness{stopOK: true, startErr: errors.New("preflight deficit"), restoreErr: errors.New("restore failed")}
	if got := runOverheadReload(h, served, candidate, candidateArgs); got.process != nil {
		t.Fatal("a failed restore reported a serving process")
	}
}

// CPU-only hosts and launches that were already measured never reload, and
// neither path plans anything.
func TestOverheadReloadIsSkippedWithoutRelief(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	served, _, _ := overheadReloadFixture()
	cfg := &config.Config{CacheDir: t.TempDir()}
	for name, caps := range map[string]*detect.Capabilities{
		"cpu-only":   {},
		"unmeasured": {GPUs: []detect.GPU{{Index: 0, Name: "gpu", VRAMTotalMB: 12282}}},
	} {
		got, attempted := replanAfterMeasuredOverhead(&launchRequest{}, cfg, &placement.ModelProfile{}, &backendInfo{},
			caps, caps, nil, served, nil)
		if attempted || got.process != served.process {
			t.Errorf("%s: attempted=%v", name, attempted)
		}
	}
}

// Every lab harness launches with --calibrate off and keys on the first ready
// line; a one-shot benchmark promises one load; a reused verified config is not
// a first launch. None of them re-plan, stop or reload, even with relief and a
// single weight load. Automatic and explicit calibration reach the planner.
func TestOverheadReloadHonoursCalibrateOffAndOneShotLaunches(t *testing.T) {
	bin := t.TempDir()
	script := "#!/bin/sh\ncase \"$*\" in *memory.used*) echo 10039 ;; *) exit 1 ;; esac\n"
	if err := os.WriteFile(filepath.Join(bin, "nvidia-smi"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin)
	t.Setenv("HOME", t.TempDir())
	cfg := &config.Config{CacheDir: t.TempDir()}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "gpu", VRAMTotalMB: 12282}}}
	before := measuredOverheadView(cfg.CacheDir, caps)
	log := "llama_model_load_from_file_impl: using device CUDA0\nload_tensors: CUDA0 model buffer size = 4002.00 MiB\n"
	placement.RunPostLaunchProbe(cfg.CacheDir, caps.GPUs, log, 0, nil, map[int]int{0: 9428}, true)
	if relief := measuredOverheadRelief(before, measuredOverheadView(cfg.CacheDir, caps), caps.GPUs); relief[0] != 611 {
		t.Fatalf("fixture has no relief: %v", relief)
	}
	saved := overheadReloadFx
	defer func() { overheadReloadFx = saved }()
	run := func(req *launchRequest, edit func(*placement.Strategy)) (*overheadReloadHarness, overheadReload, overheadReload, bool) {
		h := &overheadReloadHarness{stopOK: true}
		overheadReloadFx = h.fx()
		served, _, _ := overheadReloadFixture()
		if edit != nil {
			edit(served.strategy)
		}
		got, attempted := replanAfterMeasuredOverhead(req, cfg, &placement.ModelProfile{}, &backendInfo{}, caps, caps, before, served, nil)
		return h, served, got, attempted
	}
	for name, c := range map[string]struct {
		req  *launchRequest
		edit func(*placement.Strategy)
	}{
		"calibrate off":   {&launchRequest{Calibrate: calibrateOff}, nil},
		"benchmark":       {&launchRequest{Benchmark: true}, nil},
		"workerbench":     {&launchRequest{WorkerBenchmark: true}, nil},
		"verified config": {&launchRequest{}, func(s *placement.Strategy) { s.VerifiedConfigReused = true }},
	} {
		h, served, got, attempted := run(c.req, c.edit)
		if attempted || h.plans != 0 || len(h.events) != 0 || got.process != served.process || got.strategy != served.strategy {
			t.Errorf("%s: attempted=%v plans=%d events=%v", name, attempted, h.plans, h.events)
		}
	}
	for _, mode := range []string{"", calibrateAuto, calibrateOn} {
		h, _, _, _ := run(&launchRequest{Calibrate: mode}, nil)
		if h.plans != 1 {
			t.Errorf("calibrate %q: planner calls = %d, want 1", mode, h.plans)
		}
	}
}

// A process that will not stop is killed before the reload reports that nothing
// serves: the caller exits next and would leave it running.
func TestOverheadReloadNeverLeavesAStuckProcessBehind(t *testing.T) {
	served, candidate, candidateArgs := overheadReloadFixture()
	h := &overheadReloadHarness{stopOK: true, started: &server.Process{}, startErr: errors.New("CUDA OOM"),
		releaseErr: errors.New("did not return to baseline"), running: true, killWorks: false}
	if got := runOverheadReload(h, served, candidate, candidateArgs); got.process != nil ||
		strings.Join(h.events, ",") != "stop,start,release,kill" {
		t.Fatalf("events = %v; a failed candidate that survives a kill must stop the reload before any restore", h.events)
	}
	h = &overheadReloadHarness{stopOK: true, started: &server.Process{}, startErr: errors.New("CUDA OOM"),
		releaseErr: errors.New("did not return to baseline"), running: true, killWorks: true, restored: &server.Process{}}
	if got := runOverheadReload(h, served, candidate, candidateArgs); got.process != h.restored ||
		strings.Join(h.events, ",") != "stop,start,release,kill,restore" {
		t.Fatalf("events = %v; a killed candidate must be gone before the serving plan is restored", h.events)
	}
	h = &overheadReloadHarness{stopOK: true, startErr: errors.New("deficit"), restoreErr: errors.New("restore OOM"),
		restored: &server.Process{}, releaseErr: errors.New("did not return to baseline")}
	h.running = false
	got := runOverheadReload(h, served, candidate, candidateArgs)
	if got.process != nil {
		t.Fatal("a failed restore reported a serving process")
	}
	h = &overheadReloadHarness{stopOK: true, startErr: errors.New("deficit"), restoreErr: errors.New("restore OOM"),
		restored: &server.Process{}}
	h.events = nil
	// The restore's own failed process is still running when it returns.
	h.running, h.killWorks = false, true
	restoreRunning := h.fx()
	inner := restoreRunning.restore
	restoreRunning.restore = func(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, s *placement.Strategy, be *backendInfo, caps *detect.Capabilities,
		args []string, d time.Duration, r *launchMemoryRecovery) (*server.Process, *placement.Strategy, []string, error) {
		h.running, h.releaseErr = true, errors.New("did not return to baseline")
		return inner(req, cfg, model, s, be, caps, args, d, r)
	}
	got = reloadIntoMeasuredOverheadPlan(restoreRunning, &launchRequest{}, &config.Config{}, &placement.ModelProfile{}, &backendInfo{},
		&detect.Capabilities{}, served, candidate, candidateArgs, map[int]int{0: 190}, nil)
	if got.process != nil || !strings.HasSuffix(strings.Join(h.events, ","), "restore,release,kill") {
		t.Fatalf("events = %v; a running process from a failed restore must be killed", h.events)
	}
}
